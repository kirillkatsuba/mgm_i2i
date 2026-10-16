"""Write a reproducible plan; print sbatch commands unless --submit is specified."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WARMUP_MINUTES = 120.
DEFAULT_TRAIN_MINUTES = 120.
DEFAULT_VARIANTS = ['mse', 'mse_perceptual', 'esrgan', 'mgm', 'i2sb']
VARIANTS = {
    'mse': {'method': 'mse'},
    'mse_perceptual': {'method': 'mse_perceptual'},
    'esrgan': {'method': 'esrgan', 'projection': False, 'lr_g': 1e-4,
               'lr_aux': 1e-4, 'ema_decay': .999, 'pixel_weight': .01,
               'perceptual_weight': 1., 'gan_weight': .005},
    'mgm': {'method': 'mgm'},
    'i2sb': {'method': 'i2sb', 'projection': False, 'lr_g': 5e-5, 'ema_decay': .99},
    'gan': {'method': 'gan'},
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


def verify_data(data_root, weights_root, variants=()):
    manifest = json.loads((data_root/'manifest.json').read_text())
    prep = manifest['preparation']
    if (prep.get('protocol') != 'random128-v3' or prep.get('bank_size') != 128
            or prep.get('patches_per_image') != 1):
        raise ValueError('New plans require random128-v3 data; prepare fresh data/div2k_random128')
    for name, entry in manifest['banks'].items():
        if digest(data_root/name) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    for name in ('val.npy', 'test.npy'):
        if name not in manifest['banks']:
            raise ValueError(f'Missing bank: {name}')
    images = manifest.get('train_images', [])
    if len(images) != 800:
        raise ValueError('Expected 800 full DIV2K train images')
    for entry in images:
        if digest(data_root/entry['path']) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["path"]}')
    if not digest(weights_root/'hub/checkpoints/alexnet-owt-7be5be79.pth').startswith('7be5be79'):
        raise ValueError('Missing or corrupt AlexNet weights')

    if any(name in ('esrgan', 'mse_perceptual') for name in variants):
        if not digest(weights_root/'hub/checkpoints/vgg19-dcbb9e9d.pth').startswith('dcbb9e9d'):
            raise ValueError('Missing or corrupt VGG19 weights; run cluster_prepare.py without --skip-vgg')


def make_plan(args):
    base = dict(data_root=str(args.data_root.resolve()), weights_root=str(args.weights_root.resolve()),
                width=args.width, scale=4, crop=128, eval_crop=128, data_protocol='random128-v3',
                batch_size=args.batch_size, eval_batch_size=getattr(args, 'eval_batch_size', 1),
                workers=max(0, args.cpus-2), cpus=args.cpus,
                amp=not args.fp32, projection=True, conditioned=True, architecture_version=2,
                noise_f=.01, noise_g=.01, midpoint=False, antithetic=False,
                lr_g=1e-4, lr_aux=2e-4, reconstruction_weight=0., ema_decay=.995,
                perceptual_weight=getattr(args, 'mse_perceptual_weight', .01),
                i2sb_intervals=getattr(args, 'i2sb_intervals', 1000),
                i2sb_beta_max=getattr(args, 'i2sb_beta_max', .3),
                grad_clip=10., checkpoint_seconds=120., log_every=20)
    plan = dict(version=4, separate_evaluation=True,
                protocol='DIV2K random128-v3, online random train crops, center128 eval, average-pool x4; '
                         'architecture v2; native and final-projected outputs; independent image-level val/test',
                project_dir=str(ROOT), warmup=[], train=[], eval=[],
                scheduler=dict(partition=args.partition, max_parallel=args.max_parallel,
                               constraint=args.constraint, account=args.account,
                               gpu_type=getattr(args, 'gpu_type', ''), memory_gb=getattr(args, 'memory_gb', 32)))
    for seed in args.seeds:
        initializations = {}
        if any(VARIANTS[name]['method'] in ('mse', 'mse_perceptual', 'mgm', 'gan') for name in args.variants):
            warm_dir = args.run_root.resolve()/f'warmup/seed{seed}'
            cfg = dict(base, method='mse', variant='warmup', seed=seed, seconds=args.warmup_minutes*60)
            plan['warmup'].append(dict(config=cfg, output=str(warm_dir)))
            initializations['unet'] = str(warm_dir/'latest.pt')
        if 'esrgan' in args.variants:
            warm_dir = args.run_root.resolve()/f'rrdb_warmup/seed{seed}'
            cfg = dict(base, method='rrdb_l1', variant='rrdb_warmup', seed=seed,
                       projection=False, lr_g=1e-4, ema_decay=.999, seconds=args.warmup_minutes*60)
            plan['warmup'].append(dict(config=cfg, output=str(warm_dir)))
            initializations['rrdb'] = str(warm_dir/'latest.pt')
        for name in args.variants:
            config = dict(base, **VARIANTS[name])
            # I2SB has no reconstruction warmup: give it the same total training
            # time per method/seed as warmup + main for the other approaches.
            minutes = (getattr(args, 'i2sb_train_minutes', None) or
                       args.train_minutes + args.warmup_minutes) if name == 'i2sb' else args.train_minutes
            config.update(variant=name, seed=seed, seconds=minutes*60)
            run = args.run_root.resolve()/f'{name}/seed{seed}'
            family = 'rrdb' if name == 'esrgan' else 'unet'
            item = dict(config=config, output=str(run))
            if name != 'i2sb':
                item['init'] = initializations[family]
            plan['train'].append(item)
            nfes = getattr(args, 'i2sb_nfe', [1, 4, 16, 64]) if name == 'i2sb' else [1]
            for nfe in nfes:
                label = f'{name}_nfe{nfe}' if name == 'i2sb' else name
                plan['eval'].append(dict(config=config, checkpoint=str(run/'latest.pt'),
                                         output=str(run/f'eval_nfe{nfe}'), nfe=nfe, label=label))
    plan['budget'] = dict(
        training_gpu_hours=sum(item['config']['seconds'] for phase in ('warmup', 'train')
                               for item in plan[phase])/3600,
        warmup_limit_minutes=math.ceil(args.warmup_minutes+10),
        train_limit_minutes=math.ceil(max(item['config']['seconds'] for item in plan['train'])/60+10),
        eval_limit_minutes=getattr(args, 'eval_minutes', 30))
    plan['budget']['allocation_gpu_hours_upper_bound'] = sum(
        len(plan[phase])*plan['budget'][phase+'_limit_minutes']/60 for phase in ('warmup', 'train', 'eval'))
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
    if scheduler.get('memory_gb'):
        common.append('--mem='+str(scheduler['memory_gb'])+'G')
    gpu = scheduler.get('gpu_type', '')
    previous = None
    for phase in ('warmup', 'train', 'eval'):
        if not plan.get(phase):
            continue
        limit = budget[phase+'_limit_minutes']
        cmd = common + ['--gpus='+((gpu+':') if gpu else '')+'1',
                        '--cpus-per-task='+str(plan[phase][0]['config']['cpus']),
                        '--time='+str(limit), '--job-name=mgm_'+phase,
                        '--output='+str(logs/(phase+'-%A_%a.out')),
                        f'--array=0-{len(plan[phase])-1}%{scheduler["max_parallel"]}']
        if previous:
            cmd += [f'--dependency=afterok:<{previous}_job_id>', '--kill-on-invalid-dep=yes']
        cmd += [str(root/'slurm/cluster.sbatch'), str(plan_path), phase]
        jobs.append(cmd)
        previous = phase
    jobs.append(common + ['--cpus-per-task=1', '--time=5', '--job-name=mgm_collect',
                          '--output='+str(logs/'collect-%j.out'),
                          f'--dependency=afterany:<{previous}_job_id>',
                          str(root/'slurm/collect.sbatch'), str(plan_path)])
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data/div2k_random128'))
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    p.add_argument('--warmup-minutes', type=float, default=DEFAULT_WARMUP_MINUTES,
                   help='Training budget per shared MSE warmup (default: 120)')
    p.add_argument('--train-minutes', type=float, default=DEFAULT_TRAIN_MINUTES,
                   help='Training budget per main task, excluding shared warmup (default: 120)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--width', type=int, default=120,
                   help='Base U-Net width; stages w/2w/4w (default: 120/240/480)')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--max-parallel', type=int, default=4, help='Concurrent one-GPU jobs per phase (default: 4)')
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--eval-minutes', type=int, default=30, help='Separate evaluation job time limit')
    p.add_argument('--gpu-type', default='', help='Optional Slurm GRES type, e.g. v100; empty requests any GPU')
    p.add_argument('--memory-gb', type=int, default=32)
    p.add_argument('--mse-perceptual-weight', type=float, default=.01)
    p.add_argument('--i2sb-intervals', type=int, default=1000)
    p.add_argument('--i2sb-beta-max', type=float, default=.3)
    p.add_argument('--i2sb-nfe', type=int, nargs='+', default=[1,4,16,64])
    p.add_argument('--i2sb-train-minutes', type=float, default=None,
                   help='Default: warmup-minutes + train-minutes for equal total method budget')
    p.add_argument('--partition', default='rocky')
    p.add_argument('--constraint', default='')
    p.add_argument('--account', default='')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--pilot', action='store_true', help='All selected methods, one seed; 1 min warmup + 2 min main; I2SB 3 min; NFE 1/4')
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    if args.pilot:
        args.seeds, args.warmup_minutes, args.train_minutes = [0], 1., 2.
        args.i2sb_nfe = [1, 4]
    if min(args.batch_size, args.cpus, args.max_parallel, args.warmup_minutes, args.train_minutes,
           args.eval_batch_size, args.eval_minutes, args.memory_gb) <= 0:
        p.error('batch size, CPUs, parallelism and budgets must be positive')
    if args.width < 8 or args.width % 8 or min(args.seeds) < 0:
        p.error('width must be divisible by 8; seeds must be nonnegative')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error('duplicate seeds or variants would collide in output paths')
    if args.i2sb_intervals < 4 or args.i2sb_intervals % 2 or args.i2sb_beta_max <= 0:
        p.error('I2SB intervals must be even >=4, beta-max positive')
    if any(n < 1 or n >= args.i2sb_intervals for n in args.i2sb_nfe) or len(set(args.i2sb_nfe)) != len(args.i2sb_nfe):
        p.error('I2SB NFE must be unique and in [1, intervals-1]')
    if args.mse_perceptual_weight < 0 or (args.i2sb_train_minutes is not None and args.i2sb_train_minutes <= 0):
        p.error('Invalid loss weight or I2SB budget')
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
    verify_data(args.data_root.resolve(), args.weights_root.resolve(), args.variants)
    record = dict(plan=str(plan_path), jobs={})
    with submission.open('x') as stream:
        stream.write(json.dumps(record, indent=2))
    phases = [phase for phase in ('warmup', 'train', 'eval') if plan.get(phase)] + ['collect']
    for phase, cmd in zip(phases, jobs):
        for dependency, job_id in record['jobs'].items():
            cmd = [arg.replace(f'<{dependency}_job_id>', job_id) for arg in cmd]
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
