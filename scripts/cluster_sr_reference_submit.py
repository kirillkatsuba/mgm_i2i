"""Independent SinSR/VOSR probe → evaluation → collect chains, one GPU each."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from cluster_submit import execute_plan
from mgm_i2i.cluster.data import load_multicrop_evaluation
from mgm_i2i.cluster.sr_reference import PINS, digest, load_bundle, verify_sources


def make_plan(args, method):
    manifest, data_hash = load_multicrop_evaluation(args.data_root, args.split)
    if manifest['preparation']['eval_crops_per_image'] != 5:
        raise ValueError('Expected the same five fixed crops per photo as MGM')
    assets = args.weights_root/'sr_references'/('sinsr_v1' if method == 'sinsr' else 'vosr_05b_os')
    bundle = assets/'manifest.json'
    weights = load_bundle(bundle, method)
    sources = {name: str((args.third_party/PINS[name]['directory']).resolve())
               for name in (['sinsr'] if method == 'sinsr' else ['vosr', 'dino'])}
    verify_sources(sources, method)
    if not digest(args.weights_root/'hub/checkpoints/alexnet-owt-7be5be79.pth').startswith('7be5be79'):
        raise ValueError('Prepare the same LPIPS AlexNet weights used in MGM evaluation')
    root = args.run_root.resolve()/method
    cfg = dict(method='sr_reference', reference_method=method,
               variant='sinsr_official_v1' if method == 'sinsr' else 'vosr_official_05b_os',
               seed=0, architecture_version='author_pretrained', scale=4, crop=128, eval_crop=128,
               data_protocol='random128-multicrop-v4', eval_crops_per_image=5,
               data_root=str(args.data_root.resolve()), weights_root=str(args.weights_root.resolve()),
               cpus=args.cpus, workers=max(0, args.cpus-2), eval_batch_size=1, projection=False,
               reference_bundle=str(bundle.resolve()), reference_bundle_sha256=digest(bundle),
               reference_sources=sources)
    item = dict(config=cfg, checkpoint=str((assets/weights['checkpoint']).resolve()),
                output=str(root/'eval'), nfe=1, label=cfg['variant'], eval_batch_size=1,
                evaluation_data_root=str(args.data_root.resolve()), evaluation_data_hash=data_hash)
    probe = dict(item, output=str(root/'probe'))
    plan = dict(version=1, run_kind='sr-reference-evaluation', external_pretrained=True,
                separate_evaluation=True, project_dir=str(ROOT), evaluation_split=args.split,
                evaluation_dataset='DIV2K', phase_order=['probe', 'eval'],
                worker_script='slurm/sr_reference.sbatch', warmup=[], train=[],
                probe=[probe], eval=[item],
                protocol='Pinned author checkpoint; common float avgpool4 LR32 / HR128; '
                         'FP32 batch1, full pipeline, native + final-projected, two fixed draws; no training',
                scheduler=dict(partition=args.partition, max_parallel=1, account=args.account,
                               constraint=args.constraint, gpu_type=args.gpu_type,
                               job_name_prefix=method+'_ref'),
                budget=dict(training_gpu_hours=0, external_training_gpu_hours=None,
                            probe_limit_minutes=args.probe_minutes, eval_limit_minutes=args.eval_minutes,
                            allocation_gpu_hours_upper_bound=(args.probe_minutes+args.eval_minutes)/60,
                            note='Per-model cap, not runtime forecast. Probe blocks evaluation on invalid output or insufficient time.'))
    return plan, root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=['sinsr', 'vosr'], default=['sinsr', 'vosr'])
    p.add_argument('--run-root', type=Path, default=Path('cluster_runs/sr_refs_multi5_val_r1'))
    p.add_argument('--data-root', type=Path, default=Path('data/div2k_multicrop128'))
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--third-party', type=Path, default=Path('third_party'))
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--probe-minutes', type=int, default=20)
    p.add_argument('--eval-minutes', type=int, default=90)
    p.add_argument('--partition', default='rocky')
    p.add_argument('--account', default='')
    p.add_argument('--constraint', default='')
    p.add_argument('--gpu-type', default='')
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if min(args.cpus, args.probe_minutes, args.eval_minutes) < 1:
        p.error('CPU count and time limits must be positive')
    if args.split == 'test' and args.run_root.name == 'sr_refs_multi5_val_r1':
        p.error('Use a separate --run-root for the final test')
    # Validate both sources, assets and duplicate guards before submitting either chain.
    plans = [make_plan(args, model) for model in dict.fromkeys(args.models)]
    for _, root in plans:
        if args.submit and (root/'submission.json').exists():
            raise FileExistsError(f'{root}/submission.json exists; inspect jobs before retrying')
    for plan, root in plans:
        execute_plan(plan, root, args.submit)


if __name__ == '__main__':
    main()
