"""Plan the missing v1/v2 noise controls, reusing each architecture's MSE warmup.

One combined Slurm plan gives the whole sweep a single GPU concurrency limit.
GPU work is submitted only with --submit, from the cluster login node.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cluster_submit import execute_plan, make_plan, verify_data


SOURCES = {
    1: 'sr_v1_multi4_eval5_w50_f50_g80_b32_beta08',
    2: 'sr_v2_multi4_eval5_w50_f50_g80_b32_beta08',
}
CONTROLS = {1: 'mgm', 2: 'mgm_critic_noise'}
CORE = ['mgm_no_noise', 'mgm', 'mgm_critic_noise']
EXTRA = ['mgm_matched_noise_005', 'mgm_midpoint_critic_noise']
# Measured means in reports/mgm_multicrop_20261019/REPORT.md, V100 32GB.
MINUTES = {1: (32.70, 117.66), 2: (32.23, 118.61)}


def build_plan(args):
    variants = CORE + (EXTRA if args.suite == 'paper' else [])
    combined, source_records, matrix, references = None, [], [], []
    field_minutes = main_minutes = 0.
    for version in (1, 2):
        source_root = getattr(args, f'source_v{version}').resolve()
        child_args = SimpleNamespace(
            run_root=args.run_root/f'v{version}', data_root=args.data_root,
            weights_root=args.weights_root, architecture_version=version,
            reuse_warmup_from=source_root, variants=variants, seeds=args.seeds,
            data_protocol='random128-multicrop-v4', train_crops_per_image=4,
            eval_crops_per_image=5, width=120, batch_size=16, eval_batch_size=1,
            cpus=args.cpus, fp32=False, warmup_minutes=120., train_minutes=120.,
            warmup_epochs=50, field_warmup_epochs=50, train_epochs=80,
            field_warmup_minutes=0., field_initialization='warmup_features',
            mgm_optimizer='authors', mgm_beta1=.8, mgm_microbatch_size=32,
            grad_accumulation=1, warmup_time_limit_minutes=60,
            field_warmup_time_limit_minutes=args.field_minutes,
            train_time_limit_minutes=args.train_minutes, eval_minutes=args.eval_minutes,
            max_parallel=args.max_parallel, partition=args.partition,
            constraint=args.constraint, account=args.account, gpu_type=args.gpu_type)
        child = make_plan(child_args)  # Also checks source signatures and warmup configs.
        source_path = source_root/'plan.json'
        raw = source_path.read_bytes()
        source = json.loads(raw)
        source_records.append(dict(path=str(source_path), sha256=hashlib.sha256(raw).hexdigest()))
        if combined is None:
            combined = copy.deepcopy(child)
            for key in ('warmup', 'field_warmup', 'train', 'eval', 'reused_warmup'):
                combined[key] = []
        combined['reused_warmup'].extend(child['reused_warmup'])
        for item in child['train']:
            cfg = item['config']
            name, seed = cfg['variant'], cfg['seed']
            reuse_control = name == CONTROLS[version] and not args.retrain_controls
            field_item = next(entry for entry in child['field_warmup']
                              if entry['output'] == str(Path(item['field_init']).parent))
            evaluation = copy.deepcopy(next(entry for entry in child['eval']
                                           if entry['checkpoint'] == str(Path(item['output'])/'latest.pt')))
            evaluation['label'] = f'v{version}_{name}'
            if reuse_control:
                matches = [entry for entry in source['train']
                           if entry['config']['variant'] == name and entry['config']['seed'] == seed]
                if len(matches) != 1 or matches[0]['config'] != cfg:
                    raise ValueError(f'Existing v{version}/{name}/seed{seed} does not match the current recipe')
                checkpoint = str(Path(matches[0]['output'])/'latest.pt')
                evaluation.update(checkpoint=checkpoint,
                                  output=str(args.run_root.resolve()/f'references/v{version}/{name}/seed{seed}'))
                references.append(dict(architecture_version=version, variant=name, seed=seed,
                                       checkpoint=checkpoint, source_plan=str(source_path)))
            else:
                combined['field_warmup'].append(field_item)
                combined['train'].append(item)
                field_minutes += MINUTES[version][0]
                main_minutes += MINUTES[version][1]
            combined['eval'].append(evaluation)
            matrix.append(dict(architecture_version=version, variant=name, seed=seed,
                               action='reevaluate_checkpoint' if reuse_control else 'train_and_evaluate',
                               noise_f=cfg['noise_f'], noise_g=cfg['noise_g'], midpoint=cfg['midpoint']))
    combined.update(
        run_kind='mgm-noise-ablation', version=7, evaluation_split='val',
        protocol='DIV2K random128-multicrop-v4, pooling x4; v1/v2 noise controls; '
                 'shared architecture-specific MSE initializers, 50 field + 80 joint epochs; validation only',
        ablation=dict(suite=args.suite, retrain_controls=args.retrain_controls,
                      source_plans=source_records, reference_checkpoints=references, matrix=matrix,
                      note='Previous controls have matching saved configs; unavailable historical Git provenance '
                           'prevents a same-code claim. Use --retrain-controls for current-code controls.'))
    if getattr(args, 'with_test', False):
        combined['test'] = copy.deepcopy(combined['eval'])
        for item in combined['test']:
            # Per-crop PNG names are not split-prefixed: keep test samples separate.
            item['output'] = str(Path(item['output'])/'test')
        combined['phase_order'] = ['field_warmup', 'train', 'eval', 'test']
        combined['collect_splits'] = ['val', 'test']
        combined['protocol'] = combined['protocol'].replace('validation only', 'validation then held-out test')
        combined['budget']['test_limit_minutes'] = args.eval_minutes
    budget = combined['budget']
    budget['training_time_hint_gpu_hours'] = sum(
        entry['config']['seconds'] for phase in ('field_warmup', 'train')
        for entry in combined[phase])/3600
    budget['reused_warmup_training_gpu_hours'] = sum(
        entry['config']['seconds'] for entry in combined['reused_warmup'])/3600
    budget['reused_warmup_cost_note'] = 'Legacy configured minute hints, not measured cost of epoch-stopped warmups.'
    budget['estimated_training_gpu_hours_v100'] = round((field_minutes + main_minutes)/60, 2)
    budget['estimated_training_wall_hours_at_cap'] = round((field_minutes + main_minutes)/60/args.max_parallel, 2)
    budget['estimate_note'] = 'Extrapolation from previous V100 timings, excludes evaluation, startup and Slurm queues.'
    budget['allocation_gpu_hours_upper_bound'] = sum(
        len(combined.get(phase, []))*budget.get(phase+'_limit_minutes', 0)/60
        for phase in ('warmup', 'field_warmup', 'train', 'eval', 'test'))
    return combined


def verify_sources(plan):
    for source in plan['ablation']['source_plans']:
        raw = Path(source['path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source['sha256'] or Path(source['path']).with_suffix('.sha256').read_text().strip() != digest:
            raise ValueError('Source plan changed: '+source['path'])
    for reference in plan['ablation']['reference_checkpoints']:
        if not Path(reference['checkpoint']).is_file():
            raise FileNotFoundError('Existing control checkpoint missing: '+reference['checkpoint'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['core', 'paper'], default='core',
                        help='core: no noise/matched .01/critic .01; paper adds matched .05 and midpoint critic .01')
    parser.add_argument('--retrain-controls', action='store_true',
                        help='Train previous v1 matched/v2 critic controls again with current code; keep their MSE warmups')
    parser.add_argument('--with-test', action='store_true',
                        help='Evaluate all fixed checkpoints on held-out test after validation; collect both splits')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--source-v1', type=Path, default=Path('cluster_runs')/SOURCES[1])
    parser.add_argument('--source-v2', type=Path, default=Path('cluster_runs')/SOURCES[2])
    parser.add_argument('--data-root', type=Path, default=Path('data/div2k_multicrop128'))
    parser.add_argument('--weights-root', type=Path, default=Path('weights'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--max-parallel', type=int, default=2,
                        help='Maximum GPUs across BOTH architectures in this plan (one GPU per job)')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--field-minutes', type=int, default=90, help='Field-only Slurm limit, not training duration')
    parser.add_argument('--train-minutes', type=int, default=240, help='Joint MGM Slurm limit, not training duration')
    parser.add_argument('--eval-minutes', type=int, default=30)
    parser.add_argument('--partition', default='rocky')
    parser.add_argument('--constraint', default='')
    parser.add_argument('--account', default='')
    parser.add_argument('--gpu-type', default='')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if min(args.max_parallel, args.cpus, args.field_minutes, args.train_minutes, args.eval_minutes) < 1:
        parser.error('Parallelism, CPUs and time limits must be positive')
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    if args.run_root is None:
        suffix = '_fresh_controls' if args.retrain_controls else ''
        if args.with_test:
            suffix += '_val_test'
        args.run_root = Path(f'cluster_runs/mgm_noise_{args.suite}{suffix}_20261021')
    plan = build_plan(args)
    print(f'New field/main jobs: {len(plan["train"])} each; validation jobs: {len(plan["eval"])}; '
          f'test jobs: {len(plan.get("test", []))}; '
          f'reused control checkpoints: {len(plan["ablation"]["reference_checkpoints"])}.')

    def verify():
        verify_sources(plan)
        verify_data(args.data_root.resolve(), args.weights_root.resolve(), ['mgm'],
                    protocol='random128-multicrop-v4', eval_crops_per_image=5)

    execute_plan(plan, args.run_root, args.submit, verify)


if __name__ == '__main__':
    main()
