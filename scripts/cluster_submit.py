"""Write a reproducible plan; print sbatch commands unless --submit is specified."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_MINUTES = 120.
VARIANTS = {
    'mse': {'method': 'mse'},
    'gan': {'method': 'gan'},
    'mgm': {'method': 'mgm'},
    'mgm_midpoint': {'method': 'mgm', 'midpoint': True},
    'mgm_critic_noise': {'method': 'mgm', 'noise_g': 0.},
    'mgm_antithetic': {'method': 'mgm', 'antithetic': True},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_data(data_root, weights_root):
    manifest = json.loads((data_root/'manifest.json').read_text())
    if manifest['preparation']['bank_size'] != 256:
        raise ValueError('This protocol requires 256-pixel banks')
    for name, entry in manifest['banks'].items():
        if digest(data_root/name) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    for name in ('train.npy', 'val.npy', 'test.npy'):
        if name not in manifest['banks']:
            raise ValueError(f'Missing bank: {name}')
    if not digest(weights_root/'hub/checkpoints/alexnet-owt-7be5be79.pth').startswith('7be5be79'):
        raise ValueError('Missing or corrupt AlexNet weights')


def make_plan(args):
    base = dict(data_root=str(args.data_root.resolve()), weights_root=str(args.weights_root.resolve()),
                width=args.width, scale=4, crop=128, eval_crop=256, batch_size=args.batch_size,
                eval_batch_size=4, workers=max(0, args.cpus-2), cpus=args.cpus,
                amp=not args.fp32, projection=True, conditioned=True,
                noise_f=.01, noise_g=.01, midpoint=False, antithetic=False,
                lr_g=1e-4, lr_aux=2e-4, reconstruction_weight=0., ema_decay=.995,
                grad_clip=10., checkpoint_seconds=120., log_every=20)
    plan = dict(version=1, protocol='DIV2K average-pool x4; image-level val/test; shared MSE initialization',
                project_dir=str(ROOT), warmup=[], train=[],
                scheduler=dict(partition=args.partition, max_parallel=args.max_parallel,
                               constraint=args.constraint, account=args.account))
    for seed in args.seeds:
        warm_dir = args.run_root.resolve()/f'warmup/seed{seed}'
        warm = dict(base, method='mse', variant='warmup', seed=seed, seconds=args.warmup_minutes*60)
        plan['warmup'].append(dict(config=warm, output=str(warm_dir)))
        for name in args.variants:
            config = dict(base)
            config.update(VARIANTS[name])
            config.update(variant=name, seed=seed, seconds=args.train_minutes*60)
            plan['train'].append(dict(config=config, output=str(args.run_root.resolve()/f'{name}/seed{seed}'),
                                      init=str(warm_dir/'latest.pt')))
    plan['budget'] = dict(training_gpu_hours=(len(plan['warmup'])*args.warmup_minutes+
                                              len(plan['train'])*args.train_minutes)/60,
                          # USR1@120 can arrive 60 s early. Leave an additional
                          # two minutes for startup before the training timer.
                          warmup_limit_minutes=math.ceil(args.warmup_minutes+5),
                          train_limit_minutes=math.ceil(args.train_minutes+10))
    plan['budget']['allocation_gpu_hours_upper_bound'] = (
        len(plan['warmup'])*plan['budget']['warmup_limit_minutes']+
        len(plan['train'])*plan['budget']['train_limit_minutes'])/60
    return plan


def commands(plan, plan_path):
    scheduler, budget = plan['scheduler'], plan['budget']
    root = Path(plan['project_dir'])
    logs = plan_path.parent/'logs'
    common = ['sbatch', '--parsable', '--export=ALL', '--nodes=1', '--ntasks=1',
              '--partition='+scheduler['partition'], '--chdir='+str(root)]
    if scheduler['constraint']:
        common.append('--constraint='+scheduler['constraint'])
    if scheduler['account']:
        common.append('--account='+scheduler['account'])
    jobs = []
    for phase, limit in [('warmup', budget['warmup_limit_minutes']), ('train', budget['train_limit_minutes'])]:
        cmd = common + ['--gpus=1', '--cpus-per-task='+str(plan[phase][0]['config']['cpus']),
                        '--time='+str(limit), '--job-name=mgm_'+phase,
                        '--output='+str(logs/(phase+'-%A_%a.out')),
                        f'--array=0-{len(plan[phase])-1}%{scheduler["max_parallel"]}']
        if phase == 'train':
            cmd += ['--dependency=afterok:<warmup_job_id>', '--kill-on-invalid-dep=yes']
        cmd += [str(root/'slurm/cluster.sbatch'), str(plan_path), phase]
        jobs.append(cmd)
    jobs.append(common + ['--cpus-per-task=1', '--time=5', '--job-name=mgm_collect',
                          '--output='+str(logs/'collect-%j.out'),
                          '--dependency=afterany:<train_job_id>',
                          str(root/'slurm/collect.sbatch'), str(plan_path)])
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data/div2k'))
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument('--warmup-minutes', type=float, default=12.)
    p.add_argument('--train-minutes', type=float, default=DEFAULT_TRAIN_MINUTES,
                   help='Training budget per main task, excluding shared warmup (default: 120)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=48)
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--max-parallel', type=int, default=1)
    p.add_argument('--partition', default='rocky')
    p.add_argument('--constraint', default='')
    p.add_argument('--account', default='')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--pilot', action='store_true', help='1 seed; MGM; 1 min warmup + 2 min training')
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if args.pilot:
        args.seeds, args.variants, args.warmup_minutes, args.train_minutes = [0], ['mgm'], 1., 2.
    if min(args.batch_size, args.cpus, args.max_parallel, args.warmup_minutes, args.train_minutes) <= 0:
        p.error('batch size, CPUs, parallelism and budgets must be positive')
    if args.width < 8 or args.width % 8 or min(args.seeds) < 0:
        p.error('width must be divisible by 8; seeds must be nonnegative')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error('duplicate seeds or variants would collide in output paths')
    plan = make_plan(args)
    plan_path = args.run_root.resolve()/'plan.json'
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2)+'\n'
    if plan_path.exists():
        if plan_path.read_text() != encoded:
            raise ValueError('Existing plan differs. Use a fresh --run-root.')
        if plan_path.with_suffix('.sha256').read_text().strip() != hashlib.sha256(encoded.encode()).hexdigest():
            raise ValueError('Plan checksum mismatch. Use a fresh --run-root.')
    else:
        plan_path.write_text(encoded)
        plan_path.with_suffix('.sha256').write_text(hashlib.sha256(encoded.encode()).hexdigest()+'\n')
    (plan_path.parent/'logs').mkdir(exist_ok=True)
    print(json.dumps(plan['budget'], indent=2))
    jobs = commands(plan, plan_path)
    for cmd in jobs:
        print(shlex.join(cmd))
    if not args.submit:
        print('Plan written. No jobs submitted. Repeat identical arguments with --submit when ready.')
        return
    submission = plan_path.parent/'submission.json'
    if submission.exists():
        raise FileExistsError('Submission already attempted; inspect submission.json and resume individual jobs. No duplicate queues.')
    verify_data(args.data_root.resolve(), args.weights_root.resolve())
    record = dict(plan=str(plan_path), jobs={})
    submission.write_text(json.dumps(record, indent=2))
    for phase, cmd in zip(('warmup', 'train', 'collect'), jobs):
        cmd = [arg.replace('<warmup_job_id>', record['jobs'].get('warmup', ''))
                   .replace('<train_job_id>', record['jobs'].get('train', '')) for arg in cmd]
        response = subprocess.run(cmd, capture_output=True, text=True)
        if response.returncode:
            record['error'] = dict(phase=phase, stderr=response.stderr, command=cmd)
            submission.write_text(json.dumps(record, indent=2))
            raise RuntimeError(f'{phase} submission failed; earlier jobs retained. See {submission}')
        job_id = response.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError(f'Unrecognized sbatch response: {response.stdout!r}. Inspect squeue before retry.')
        record['jobs'][phase] = job_id
        submission.write_text(json.dumps(record, indent=2))
        print(phase, job_id, flush=True)


if __name__ == '__main__':
    main()
