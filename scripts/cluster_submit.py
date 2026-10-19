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


def verify_data(data_root, weights_root, variants=(), *, protocol=None, eval_crops_per_image=None):
    manifest = json.loads((data_root/'manifest.json').read_text())
    prep = manifest['preparation']
    if (prep.get('protocol') not in ('random128-v3', 'random128-multicrop-v4') or prep.get('bank_size') != 128
            or prep.get('patches_per_image') != 1):
        raise ValueError('New plans require full-image random128 data; prepare a fresh data root')
    if protocol is not None and prep['protocol'] != protocol:
        raise ValueError('Plan data protocol differs from prepared data')
    if prep['protocol'] == 'random128-multicrop-v4':
        count = prep.get('eval_crops_per_image', 0)
        if (count < 1 or prep.get('eval_crop_policy') != 'fixed_random_without_replacement' or
                (eval_crops_per_image is not None and count != eval_crops_per_image)):
            raise ValueError('Evaluation crop policy differs from plan')
    for name, entry in manifest['banks'].items():
        if digest(data_root/name) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    for name in ('val.npy', 'test.npy'):
        if name not in manifest['banks']:
            raise ValueError(f'Missing bank: {name}')
        if prep['protocol'] == 'random128-multicrop-v4':
            rows = manifest['banks'][name]['crops']
            ids = {row['image'] for row in rows}
            if len(ids) != 50 or len(rows) != 50 * count or any(
                    len({(r['top'], r['left']) for r in rows if r['image'] == image}) != count for image in ids):
                raise ValueError('Expected distinct fixed evaluation crops for each of 50 photographs')
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
    architecture_version = getattr(args, 'architecture_version', 2)
    if architecture_version not in (1, 2):
        raise ValueError('Architecture version must be 1 or 2')
    if architecture_version == 1 and any(VARIANTS[name]['method'] not in
                                        ('mse', 'mse_perceptual', 'mgm', 'gan') for name in args.variants):
        raise ValueError('Architecture v1 applies only to our U-Net methods; select them explicitly')
    reuse_root = getattr(args, 'reuse_warmup_from', None)
    source_plan = None
    if reuse_root is not None:
        source_path = reuse_root.resolve()/'plan.json'
        raw = source_path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_path.with_suffix('.sha256').read_text().strip() != source_hash:
            raise ValueError('Source warmup plan checksum mismatch')
        source_plan = json.loads(raw)
    protocol = getattr(args, 'data_protocol', 'random128-v3')
    multicrop = protocol == 'random128-multicrop-v4'
    crops_per_image = getattr(args, 'train_crops_per_image', 4) if multicrop else 1
    eval_crops = getattr(args, 'eval_crops_per_image', 5) if multicrop else 1
    if min(crops_per_image, eval_crops) < 1:
        raise ValueError('Train/evaluation crop counts must be positive')
    base = dict(data_root=str(args.data_root.resolve()), weights_root=str(args.weights_root.resolve()),
                width=args.width, scale=4, crop=128, eval_crop=128, data_protocol=protocol,
                batch_size=args.batch_size, eval_batch_size=getattr(args, 'eval_batch_size', 1),
                workers=max(0, args.cpus-2), cpus=args.cpus,
                amp=not args.fp32, projection=True, conditioned=True, architecture_version=architecture_version,
                noise_f=.01, noise_g=.01, midpoint=False, antithetic=False,
                lr_g=1e-4, lr_aux=2e-4, reconstruction_weight=0., ema_decay=.995,
                perceptual_weight=getattr(args, 'mse_perceptual_weight', .01),
                i2sb_intervals=getattr(args, 'i2sb_intervals', 1000),
                i2sb_beta_max=getattr(args, 'i2sb_beta_max', .3),
                grad_clip=10., checkpoint_seconds=120., log_every=20)
    if multicrop:
        base.update(train_crops_per_image=crops_per_image, eval_crops_per_image=eval_crops)
    epoch_counts = {phase: getattr(args, phase+'_epochs', None)
                    for phase in ('warmup', 'train', 'field_warmup')}
    if any(value is not None and value < 1 for value in epoch_counts.values()):
        raise ValueError('Epoch counts must be positive integers')
    if getattr(args, 'pilot', False) and any(value is not None for value in epoch_counts.values()):
        raise ValueError('--pilot uses time budgets; use explicit small epoch counts for an epoch pilot')

    def with_epochs(config, phase):
        count = epoch_counts[phase]
        if count is not None:
            config.update(epochs=count, samples_per_epoch=800*crops_per_image)
            batch = config['batch_size'] * config.get('grad_accumulation', 1)
            if count * 800 * crops_per_image % batch:
                raise ValueError('Epoch target must be divisible by effective batch size')
        return config

    field_minutes = getattr(args, 'field_warmup_minutes', 0.)
    if epoch_counts['field_warmup'] is not None and not field_minutes:
        field_minutes = 15.  # Allocation estimate only when stopping by epochs.
    field_strategy = getattr(args, 'field_initialization', 'random')
    accumulation = getattr(args, 'grad_accumulation', 1)
    beta1 = getattr(args, 'mgm_beta1', None)
    if beta1 is not None and not 0 <= beta1 < 1:
        raise ValueError('MGM Adam beta1 must be in [0, 1)')
    if accumulation < 1 or field_minutes < 0:
        raise ValueError('Invalid accumulation factor or field warmup duration')
    if field_minutes and field_strategy != 'warmup_features':
        raise ValueError('Field warmup requires --field-initialization warmup_features')
    if (accumulation > 1 or field_minutes or field_strategy != 'random' or
            getattr(args, 'mgm_optimizer', 'legacy') != 'legacy' or
            beta1 is not None or
            getattr(args, 'mgm_microbatch_size', None) is not None):
        if any(VARIANTS[name]['method'] != 'mgm' for name in args.variants):
            raise ValueError('New MGM training options require MGM-only variants')
    sampling = (f'{crops_per_image} random train crops per photo/epoch, {eval_crops} fixed random eval crops per photo'
                if multicrop else 'online random train crops, center128 eval')
    plan = dict(version=6 if multicrop else 4, separate_evaluation=True,
                protocol=f'DIV2K {protocol}, {sampling}, average-pool x4; '
                         f'architecture v{architecture_version}; native and final-projected outputs; independent image-level val/test',
                project_dir=str(ROOT), warmup=[], train=[], eval=[],
                scheduler=dict(partition=args.partition, max_parallel=args.max_parallel,
                               constraint=args.constraint, account=args.account,
                               gpu_type=getattr(args, 'gpu_type', '')))
    if field_minutes:
        plan['field_warmup'] = []
    if source_plan is not None:
        plan['reused_warmup'] = []

    def initialization(config, output):
        if source_plan is None:
            plan['warmup'].append(dict(config=config, output=str(output)))
            return str(output/'latest.pt')
        matches = [item for item in source_plan['warmup']
                   if item['config']['method'] == config['method'] and
                   item['config']['seed'] == config['seed']]
        if len(matches) != 1:
            raise ValueError(f'Expected one source warmup for {config["method"]}/seed{config["seed"]}')
        source = matches[0]
        differences = sorted(key for key in config.keys() | source['config'].keys()
                             if config.get(key) != source['config'].get(key))
        if differences:
            raise ValueError('Source warmup config differs: '+', '.join(differences))
        checkpoint = str(Path(source['output'])/'latest.pt')
        plan['reused_warmup'].append(dict(config=source['config'], checkpoint=checkpoint,
                                         source_plan=str(source_path), source_plan_sha256=source_hash))
        return checkpoint

    for seed in args.seeds:
        initializations = {}
        if any(VARIANTS[name]['method'] in ('mse', 'mse_perceptual', 'mgm', 'gan') for name in args.variants):
            warm_dir = args.run_root.resolve()/f'warmup/seed{seed}'
            cfg = dict(base, method='mse', variant='warmup', seed=seed, seconds=args.warmup_minutes*60)
            with_epochs(cfg, 'warmup')
            initializations['unet'] = initialization(cfg, warm_dir)
        if 'esrgan' in args.variants:
            warm_dir = args.run_root.resolve()/f'rrdb_warmup/seed{seed}'
            cfg = dict(base, method='rrdb_l1', variant='rrdb_warmup', seed=seed,
                       projection=False, lr_g=1e-4, ema_decay=.999, seconds=args.warmup_minutes*60)
            with_epochs(cfg, 'warmup')
            initializations['rrdb'] = initialization(cfg, warm_dir)
        for name in args.variants:
            config = dict(base, **VARIANTS[name])
            if config['method'] == 'mgm':
                if getattr(args, 'mgm_optimizer', 'legacy') == 'authors':
                    config.update(lr_g=1e-5, lr_aux=1e-5, adam_betas=[0., .999], optimizer_profile='authors')
                if beta1 is not None:
                    config['adam_betas'] = [beta1, config.get('adam_betas', [.5, .99])[1]]
                    config['optimizer_profile'] = getattr(args, 'mgm_optimizer', 'legacy')+'_beta1_override'
                if getattr(args, 'mgm_microbatch_size', None) is not None:
                    config['batch_size'] = args.mgm_microbatch_size
                if accumulation > 1:
                    config['grad_accumulation'] = accumulation
                if field_strategy != 'random':
                    config['field_initialization'] = field_strategy
            # I2SB has no reconstruction warmup: give it the same total training
            # time per method/seed as warmup + main for the other approaches.
            minutes = (getattr(args, 'i2sb_train_minutes', None) or
                       args.train_minutes + args.warmup_minutes) if name == 'i2sb' else args.train_minutes
            config.update(variant=name, seed=seed, seconds=minutes*60)
            with_epochs(config, 'train')
            run = args.run_root.resolve()/f'{name}/seed{seed}'
            family = 'rrdb' if name == 'esrgan' else 'unet'
            item = dict(config=config, output=str(run))
            if name != 'i2sb':
                item['init'] = initializations[family]
            if config['method'] == 'mgm' and field_minutes:
                field_dir = args.run_root.resolve()/f'field_warmup/{name}/seed{seed}'
                field_cfg = dict(config, variant=name+'_field_warmup', field_only=True,
                                 seconds=field_minutes*60)
                field_cfg.pop('epochs', None)
                field_cfg.pop('samples_per_epoch', None)
                with_epochs(field_cfg, 'field_warmup')
                plan['field_warmup'].append(dict(config=field_cfg, output=str(field_dir), init=item['init']))
                item['field_init'] = str(field_dir/'latest.pt')
            plan['train'].append(item)
            nfes = getattr(args, 'i2sb_nfe', [1, 4, 16, 64]) if name == 'i2sb' else [1]
            for nfe in nfes:
                label = f'{name}_nfe{nfe}' if name == 'i2sb' else name
                plan['eval'].append(dict(config=config, checkpoint=str(run/'latest.pt'),
                                         output=str(run/f'eval_nfe{nfe}'), nfe=nfe, label=label))
    plan['budget'] = dict(
        training_gpu_hours=sum(item['config']['seconds'] for phase in ('warmup', 'field_warmup', 'train')
                               for item in plan.get(phase, []))/3600,
        warmup_limit_minutes=math.ceil(args.warmup_minutes+10),
        train_limit_minutes=math.ceil(max(item['config']['seconds'] for item in plan['train'])/60+10),
        eval_limit_minutes=getattr(args, 'eval_minutes', 30))
    if field_minutes:
        plan['budget']['field_warmup_limit_minutes'] = math.ceil(field_minutes+10)
    for phase in ('warmup', 'field_warmup', 'train'):
        limit = getattr(args, phase+'_time_limit_minutes', None)
        if limit is not None:
            if limit < 1:
                raise ValueError('Slurm time limits must be positive')
            plan['budget'][phase+'_limit_minutes'] = limit
    if any(item['config'].get('epochs') is not None
           for phase in ('warmup', 'field_warmup', 'train') for item in plan.get(phase, [])):
        plan['version'] = max(plan['version'], 5)
        plan['budget']['training_time_hint_gpu_hours'] = plan['budget']['training_gpu_hours']
        plan['budget']['training_gpu_hours'] = None
        plan['budget']['note'] = ('Epoch phases stop at sampled-patch targets. Minute budgets are runtime estimates only; '
                                  'Slurm limits apply per allocation. Interrupted checkpoints remain incomplete and can resume.')
    plan['budget']['allocation_gpu_hours_upper_bound'] = sum(
        len(plan.get(phase, []))*plan['budget'].get(phase+'_limit_minutes', 0)/60
        for phase in ('warmup', 'field_warmup', 'train', 'eval'))
    if source_plan is not None:
        if not plan['reused_warmup']:
            raise ValueError('--reuse-warmup-from requires a method with reconstruction warmup')
        plan['budget']['reused_warmup_training_gpu_hours'] = sum(
            item['config']['seconds'] for item in plan['reused_warmup'])/3600
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
    gpu = scheduler.get('gpu_type', '')
    previous = None
    for phase in plan.get('phase_order', ('warmup', 'field_warmup', 'train', 'eval')):
        if not plan.get(phase):
            continue
        limit = budget[phase+'_limit_minutes']
        cmd = common + ['--gpus='+((gpu+':') if gpu else '')+'1',
                        '--cpus-per-task='+str(plan[phase][0]['config']['cpus']),
                        '--time='+str(limit), '--job-name='+scheduler.get('job_name_prefix', 'mgm')+'_'+phase,
                        '--output='+str(logs/(phase+'-%A_%a.out')),
                        f'--array=0-{len(plan[phase])-1}%{scheduler["max_parallel"]}']
        if previous:
            cmd += [f'--dependency=afterok:<{previous}_job_id>', '--kill-on-invalid-dep=yes']
        cmd += [str(root/plan.get('worker_script', 'slurm/cluster.sbatch')), str(plan_path), phase]
        jobs.append(cmd)
        previous = phase
    jobs.append(common + ['--cpus-per-task=1', '--time=5', '--job-name='+scheduler.get('job_name_prefix', 'mgm')+'_collect',
                          '--output='+str(logs/'collect-%j.out'),
                          f'--dependency=afterany:<{previous}_job_id>',
                          str(root/'slurm/collect.sbatch'), str(plan_path)])
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('data/div2k_multicrop128'))
    p.add_argument('--data-protocol', choices=['random128-multicrop-v4', 'random128-v3'],
                   default='random128-multicrop-v4')
    p.add_argument('--train-crops-per-image', type=int, default=4,
                   help='Multicrop protocol: independently positioned crops per photograph per epoch')
    p.add_argument('--eval-crops-per-image', type=int, default=5,
                   help='Multicrop protocol: must match the fixed evaluation bank preparation')
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    p.add_argument('--warmup-minutes', type=float, default=DEFAULT_WARMUP_MINUTES,
                   help='Training budget per shared MSE warmup (default: 120)')
    p.add_argument('--reuse-warmup-from', type=Path,
                   help='Reuse matching warmup checkpoints from a signed run plan in this directory; '
                        'skip those warmup jobs. Configs must match exactly.')
    p.add_argument('--train-minutes', type=float, default=DEFAULT_TRAIN_MINUTES,
                   help='Training budget per main task, excluding shared warmup (default: 120)')
    for phase in ('warmup', 'train', 'field_warmup'):
        p.add_argument('--'+phase.replace('_', '-')+'-epochs', type=int, default=None,
                       help='Stop this phase after N epochs (800 x train crops per image patches), instead of minutes')
        p.add_argument('--'+phase.replace('_', '-')+'-time-limit-minutes', type=int, default=None,
                       help='Independent Slurm wall-time limit per allocation for this phase')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--mgm-microbatch-size', type=int, default=None,
                   help='Override physical MGM batch size without changing shared MSE warmup config')
    p.add_argument('--grad-accumulation', type=int, default=1,
                   help='MGM-only: effective batch = physical MGM batch x this factor')
    p.add_argument('--mgm-optimizer', choices=['legacy', 'authors'], default='legacy',
                   help='authors: MGM Adam lr_G=lr_field=1e-5, betas=(0,0.999); warmup stays unchanged')
    p.add_argument('--mgm-beta1', type=float, default=None,
                   help='Override Adam beta1 for both MGM optimizers, preserving preset learning rates and beta2')
    p.add_argument('--field-initialization', choices=['random', 'warmup_features'], default='random')
    p.add_argument('--field-warmup-minutes', type=float, default=0.,
                   help='Optional field-only regression before joint MGM, with G frozen')
    p.add_argument('--width', type=int, default=120,
                   help='Base U-Net width; stages w/2w/4w (default: 120/240/480)')
    p.add_argument('--architecture-version', type=int, choices=[1, 2], default=2,
                   help='Our U-Net architecture: 1=tanh residual/input time maps; 2=linear residual/block time embedding')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--max-parallel', type=int, default=4, help='Concurrent one-GPU jobs per phase (default: 4)')
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--eval-minutes', type=int, default=30, help='Separate evaluation job time limit')
    p.add_argument('--gpu-type', default='', help='Optional Slurm GRES type, e.g. v100; empty requests any GPU')
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
           args.eval_batch_size, args.eval_minutes) <= 0:
        p.error('batch size, CPUs, parallelism and budgets must be positive')
    if args.width < 8 or args.width % 8 or min(args.seeds) < 0:
        p.error('width must be divisible by 8; seeds must be nonnegative')
    if args.grad_accumulation < 1 or args.field_warmup_minutes < 0 or (
            args.mgm_microbatch_size is not None and args.mgm_microbatch_size < 1):
        p.error('Invalid MGM batch size, accumulation factor or field warmup duration')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error('duplicate seeds or variants would collide in output paths')
    if args.i2sb_intervals < 4 or args.i2sb_intervals % 2 or args.i2sb_beta_max <= 0:
        p.error('I2SB intervals must be even >=4, beta-max positive')
    if any(n < 1 or n >= args.i2sb_intervals for n in args.i2sb_nfe) or len(set(args.i2sb_nfe)) != len(args.i2sb_nfe):
        p.error('I2SB NFE must be unique and in [1, intervals-1]')
    if args.mse_perceptual_weight < 0 or (args.i2sb_train_minutes is not None and args.i2sb_train_minutes <= 0):
        p.error('Invalid loss weight or I2SB budget')
    plan = make_plan(args)
    execute_plan(plan, args.run_root, args.submit,
                 lambda: verify_data(args.data_root.resolve(), args.weights_root.resolve(), args.variants,
                                     protocol=args.data_protocol, eval_crops_per_image=args.eval_crops_per_image))


def execute_plan(plan, run_root, submit=False, verify=None):
    plan_path = run_root.resolve()/'plan.json'
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
    if not submit:
        print('Plan written. No jobs submitted. Repeat identical arguments with --submit when ready.')
        return
    submission = plan_path.parent/'submission.json'
    if submission.exists():
        raise FileExistsError('Submission already attempted; inspect submission.json and resume individual jobs. No duplicate queues.')
    for item in plan.get('reused_warmup', []):
        if not Path(item['checkpoint']).is_file():
            raise FileNotFoundError('Reused warmup checkpoint is missing: '+item['checkpoint'])
    if verify is not None:
        verify()
    record = dict(plan=str(plan_path), jobs={})
    with submission.open('x') as stream:
        stream.write(json.dumps(record, indent=2))
    phases = [phase for phase in plan.get('phase_order', ('warmup', 'field_warmup', 'train', 'eval')) if plan.get(phase)] + ['collect']
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
