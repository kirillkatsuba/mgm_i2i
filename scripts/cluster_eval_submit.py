"""Plan a shared BSD100 or DIV2K multicrop evaluation for frozen source checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from mgm_i2i.cluster.external_data import load_bank
from mgm_i2i.cluster.data import load_multicrop_evaluation
from cluster_submit import execute_plan


def make_plan(args):
    split = getattr(args, 'split', 'external')
    if split == 'external':
        _, data_hash = load_bank(args.data_root)
        dataset, description = 'BSD100', '100 native center128 HR crops'
    else:
        manifest, data_hash = load_multicrop_evaluation(args.data_root, split)
        dataset = 'DIV2K'
        description = f'50 photographs, {manifest["preparation"]["eval_crops_per_image"]} fixed random128 crops per photo'
    roots = [root.resolve() for root in args.source_runs]
    if len({root.name for root in roots}) != len(roots):
        raise ValueError('Source run directory names must be unique')
    plan = dict(version=4, separate_evaluation=True, project_dir=str(ROOT),
                evaluation_split=split, evaluation_dataset=dataset,
                protocol=f'{dataset}: {description}, average-pool x4; frozen checkpoint EMA; '
                         'RGB metrics, FP32 batch1; native and final-projected outputs',
                warmup=[], train=[], eval=[], source_plans=[],
                scheduler=dict(partition=args.partition, max_parallel=args.max_parallel,
                               constraint=args.constraint, account=args.account,
                               gpu_type=args.gpu_type))
    identities = set()
    for root in roots:
        path = root/'plan.json'
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if path.with_suffix('.sha256').read_text().strip() != checksum:
            raise ValueError(f'Source plan checksum mismatch: {path}')
        source = json.loads(raw)
        plan['source_plans'].append(dict(path=str(path), sha256=checksum))
        # Legacy plans evaluate in train jobs; new plans have explicit NFE entries.
        for original in source.get('eval', source['train']):
            cfg = original['config']
            nfe = original.get('nfe', 1)
            checkpoint = original.get('checkpoint', str(Path(original['output'])/'latest.pt'))
            identity = (checkpoint, nfe)
            if identity in identities:
                raise ValueError(f'Duplicate checkpoint/NFE: {identity}')
            identities.add(identity)
            label = root.name+'/'+original.get('label', cfg['variant'])
            output = args.run_root.resolve()/root.name/cfg['variant']/f'seed{cfg["seed"]}'/f'eval_nfe{nfe}'
            plan['eval'].append(dict(config=cfg, checkpoint=checkpoint, output=str(output),
                                     label=label, nfe=nfe, eval_batch_size=1,
                                     evaluation_data_root=str(args.data_root.resolve()), evaluation_data_hash=data_hash))
    if not plan['eval']:
        raise ValueError('No models in source plans')
    plan['budget'] = dict(training_gpu_hours=0., eval_limit_minutes=args.eval_minutes,
                          allocation_gpu_hours_upper_bound=len(plan['eval'])*args.eval_minutes/60)
    return plan


def verify_inputs(plan):
    for item in plan['eval']:
        if not Path(item['checkpoint']).is_file():
            raise FileNotFoundError('Trained checkpoint missing: '+item['checkpoint'])
        weights = Path(item['config']['weights_root'])/'hub/checkpoints/alexnet-owt-7be5be79.pth'
        if not weights.is_file():
            raise FileNotFoundError(f'Local LPIPS AlexNet weights missing: {weights}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-runs', nargs='+', required=True, type=Path)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/bsd100_center128'))
    parser.add_argument('--split', choices=['external','val','test'], default='external',
                        help='external=BSD100; val/test=shared DIV2K multicrop bank, including for older checkpoints')
    parser.add_argument('--max-parallel', type=int, default=4)
    parser.add_argument('--eval-minutes', type=int, default=30)
    parser.add_argument('--partition', default='rocky')
    parser.add_argument('--constraint', default='')
    parser.add_argument('--account', default='')
    parser.add_argument('--gpu-type', default='')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if min(args.max_parallel, args.eval_minutes) <= 0:
        parser.error('Budgets and parallelism must be positive')
    plan = make_plan(args)
    execute_plan(plan, args.run_root, args.submit, lambda: verify_inputs(plan))


if __name__ == '__main__':
    main()
