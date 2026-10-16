"""Time-budgeted training. Run checkpoints are trusted local pickle files."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..losses import observation, variational_loss, field_loss, random_bridge
from .data import SampleRange, training_dataset, validate_data_protocol
from .models import Field, Discriminator, get_architecture_version
from .baselines import build_generator, VGGDiscriminator, VGGPerceptual, I2SBBridge, relativistic_loss
from .progress import BudgetProgress


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    def git(*args):
        # Git may only be installed on the login node. Optional metadata must
        # neither fail training/evaluation nor hang on a shared filesystem.
        try:
            p = subprocess.run(['git', *args], capture_output=True, text=True,
                               errors='replace', timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return 'unavailable'
        return p.stdout.strip() if p.returncode == 0 else 'unavailable'
    return dict(git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--porcelain'),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), job_id=os.getenv('SLURM_JOB_ID'),
                array_id=os.getenv('SLURM_ARRAY_TASK_ID'))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def frozen(module, value):
    module.requires_grad_(not value)
    module.train(not value)  # freeze spectral-norm buffers during G updates too


def mgm_objective(field, c, fake, real, cfg, for_generator):
    t, b, eps = random_bridge(fake, midpoint=cfg['midpoint'])
    strength = cfg['noise_g'] if for_generator else cfg['noise_f']
    delta = real - fake
    flips = (b, ~b) if cfg['antithetic'] else (b,)
    loss = fake.new_zeros(())
    for flip in flips:
        mix = observation(fake, real, t, flip, eps, strength)
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            prediction = field(c, t, mix)
        # Never detach delta or mix for G: both gradient paths implement Eq. (10).
        loss = loss + (variational_loss(prediction.float(), delta) if for_generator
                       else field_loss(prediction.float(), delta)) / len(flips)
    return loss


def sr_forward(model, c, z, config):
    if config['method'] in ('esrgan', 'rrdb_l1'):
        return 2 * model((c + 1) / 2).float() - 1
    return model(c, z).float()


def generator_objective(gen, aux, c, real, cfg, perceptual=None, bridge=None):
    """One method-specific loss; shared [-1,1] HR/LR interface for evaluation."""
    method = cfg['method']
    if method == 'i2sb':
        step = torch.randint(bridge.intervals, (len(real),), device=real.device)
        degraded = F.interpolate(c, scale_factor=cfg['scale'], mode='nearest')
        state, label = bridge.training_pair(real, degraded, step)
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            prediction = gen(state, bridge.time(step)).float()
        fake = state - bridge.std(step) * prediction
        return F.mse_loss(prediction, label), F.mse_loss(fake, real), fake
    deterministic = method in ('mse', 'mse_perceptual', 'rrdb_l1', 'esrgan')
    z = torch.zeros_like(real) if deterministic else torch.randn_like(real)
    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
        fake = sr_forward(gen, c, z, cfg)
    mse = F.mse_loss(fake, real)
    if method == 'mse':
        loss = mse
    elif method == 'mse_perceptual':
        loss = mse + cfg['perceptual_weight'] * perceptual((fake+1)/2, (real+1)/2)
    elif method == 'rrdb_l1':
        loss = F.l1_loss((fake+1)/2, (real+1)/2)
    elif method == 'esrgan':
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            scores_real = aux((real+1)/2).detach().float()
            scores_fake = aux((fake+1)/2).float()
        loss = (cfg['pixel_weight'] * F.l1_loss((fake+1)/2, (real+1)/2) +
                cfg['perceptual_weight'] * perceptual((fake+1)/2, (real+1)/2) +
                cfg['gan_weight'] * relativistic_loss(scores_real, scores_fake, generator=True))
    elif method == 'mgm':
        loss = mgm_objective(aux, c, fake, real, cfg, True)
    elif method == 'gan':
        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
            loss = -aux(c, fake).mean()
    else:
        raise ValueError(f'Unsupported method {method}')
    if method in ('gan', 'mgm'):
        loss = loss + cfg['reconstruction_weight'] * mse
    return loss, mse, fake


def train(cfg, output, init=None, resume=False):
    validate_data_protocol(cfg['data_root'], cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('Cluster training requires CUDA. Use scripts/train.py for the Mac prototype.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output/'latest.pt'
    if latest.exists() and not resume:
        raise FileExistsError(f'{latest}: use --resume or a new plan directory')
    if resume and not latest.exists():
        raise FileNotFoundError(f'Cannot resume: {latest}')
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpus'])
    device = torch.device('cuda')
    version = get_architecture_version(cfg)
    gen = build_generator(cfg).to(device)
    aux = (Field(cfg['width'], cfg['conditioned'], version) if cfg['method'] == 'mgm' else
           Discriminator(cfg['width']) if cfg['method'] == 'gan' else
           VGGDiscriminator() if cfg['method'] == 'esrgan' else None)
    if aux is not None:
        aux = aux.to(device)
    perceptual = (VGGPerceptual(Path(cfg['weights_root'])/'hub/checkpoints/vgg19-dcbb9e9d.pth').to(device)
                  if cfg['method'] in ('mse_perceptual', 'esrgan') else None)
    bridge = (I2SBBridge(cfg['i2sb_intervals'], cfg['i2sb_beta_max']).to(device)
              if cfg['method'] == 'i2sb' else None)
    data_hash = fingerprint(Path(cfg['data_root'])/'manifest.json')
    init_hash = None
    warmup_training_seconds = 0.
    if init and not resume:
        source = torch.load(init, map_location='cpu', weights_only=False)
        if not source['complete'] or source['data_hash'] != data_hash:
            raise ValueError('Warmup must be complete and use the same dataset manifest')
        if get_architecture_version(source['config']) != version:
            raise ValueError('Warmup architecture differs; create a new warmup for this architecture')
        for key in ('seed', 'width', 'scale', 'projection'):
            if source['config'][key] != cfg[key]:
                raise ValueError(f'Warmup mismatch: {key}')
        gen.load_state_dict(source['generator'])
        init_hash = fingerprint(init)
        warmup_training_seconds = source['elapsed_training_seconds']
    ema = copy.deepcopy(gen).eval().requires_grad_(False)
    betas = (.9, .99) if cfg['method'] in ('rrdb_l1', 'esrgan') else (.5, .99)
    opt_g = (torch.optim.AdamW(gen.parameters(), lr=cfg['lr_g'], weight_decay=0.)
             if bridge is not None else torch.optim.Adam(gen.parameters(), lr=cfg['lr_g'], betas=betas))
    opt_a = torch.optim.Adam(aux.parameters(), lr=cfg['lr_aux'], betas=betas) if aux is not None else None
    scheduler = (torch.optim.lr_scheduler.StepLR(opt_g, step_size=1000, gamma=.99)
                 if bridge is not None else None)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    step, consumed, elapsed, skipped_g, skipped_aux, successful_g = 0, 0, 0., 0, 0, 0
    history = []
    if resume:
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != cfg or state['data_hash'] != data_hash:
            raise ValueError('Resume config/data mismatch; use the original immutable plan')
        gen.load_state_dict(state['generator'])
        ema.load_state_dict(state['ema'])
        opt_g.load_state_dict(state['optimizer_g'])
        if aux is not None:
            aux.load_state_dict(state['auxiliary'])
            opt_a.load_state_dict(state['optimizer_aux'])
        scaler.load_state_dict(state['scaler'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        step, consumed, elapsed = state['step'], state['consumed'], state['elapsed_training_seconds']
        skipped_g, skipped_aux, successful_g = state['skipped_g'], state['skipped_aux'], state['successful_g']
        init_hash = state['init_hash']
        warmup_training_seconds = state.get('warmup_training_seconds', 0.)
        history = state['sessions']
        restore_rng(state['rng'])
        if state['complete']:
            print(f'Training already complete: {latest}; skipping updates.', flush=True)
            return state  # caller can retry evaluation without retraining
        # Discard log rows from updates lost after the last atomic checkpoint.
        log_path = output/'train.jsonl'
        if log_path.exists():
            retained = []
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row['step'] <= step:
                    retained.append(line)
            log_path.write_text(''.join(line+'\n' for line in retained))
    atomic_json(output/'config.json', cfg)
    session = provenance()
    session.update(generator_parameters=sum(p.numel() for p in gen.parameters()),
                   auxiliary_parameters=sum(p.numel() for p in aux.parameters()) if aux is not None else 0)
    history.append(session)
    atomic_json(output/'environment.json', history)
    dataset = training_dataset(cfg['data_root'], cfg)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'],
                        sampler=SampleRange(consumed, 10**12), num_workers=cfg['workers'],
                        pin_memory=True, drop_last=True, persistent_workers=cfg['workers'] > 0,
                        generator=torch.Generator().manual_seed(cfg['seed'] + 10000))
    interrupted = []
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        old_handlers[signum] = signal.signal(signum, lambda s, frame: interrupted.append(s))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    previous_checkpoint = start

    def save(complete):
        torch.cuda.synchronize()
        state = dict(config=cfg, generator=gen.state_dict(), ema=ema.state_dict(),
                     auxiliary=aux.state_dict() if aux is not None else None,
                     optimizer_g=opt_g.state_dict(), optimizer_aux=opt_a.state_dict() if opt_a else None,
                     scaler=scaler.state_dict(), rng=rng_state(), step=step, consumed=consumed,
                     scheduler=scheduler.state_dict() if scheduler is not None else None,
                     successful_g=successful_g, skipped_g=skipped_g, skipped_aux=skipped_aux,
                     elapsed_training_seconds=elapsed + time.monotonic()-start,
                     peak_allocated_bytes_this_session=torch.cuda.max_memory_allocated(),
                     data_hash=data_hash, init_hash=init_hash, sessions=history, complete=complete,
                     warmup_training_seconds=warmup_training_seconds,
                     received_signals=[signal.Signals(s).name for s in interrupted])
        temp = output/'latest.tmp.pt'
        torch.save(state, temp)
        temp.replace(latest)
        atomic_json(output/'status.json', {k: state[k] for k in (
            'step', 'consumed', 'successful_g', 'skipped_g', 'skipped_aux',
            'elapsed_training_seconds', 'peak_allocated_bytes_this_session', 'complete', 'data_hash', 'init_hash',
            'received_signals')})
        return state

    def update(loss, optimizer, parameters):
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite loss; last complete checkpoint retained')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'])
        if not cfg['amp'] and not torch.isfinite(norm).item():
            raise FloatingPointError('Non-finite FP32 gradient; last complete checkpoint retained')
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        return scaler.get_scale() >= scale

    progress = None
    try:
        progress = BudgetProgress(cfg['seconds'], f'{cfg["variant"]}/seed{cfg["seed"]}',
                                  elapsed=elapsed, step=step)
        # A valid initial checkpoint also permits recovery from an early OOM.
        if not latest.exists():
            save(False)
        with (output/'train.jsonl').open('a') as log:
            for batch in loader:
                if interrupted or elapsed + time.monotonic()-start >= cfg['seconds']:
                    break
                real = batch.to(device, non_blocking=True).float().div_(127.5).sub_(1)
                c = F.avg_pool2d(real, cfg['scale'])
                loss_a = None
                if aux is not None and cfg['method'] != 'esrgan':
                    frozen(aux, False)
                    opt_a.zero_grad(set_to_none=True)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                        detached = gen(c, torch.randn_like(real)).float()
                    if cfg['method'] == 'mgm':
                        loss_a = mgm_objective(aux, c, detached, real, cfg, False)
                    else:
                        with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                            real_score, fake_score = aux(c, real), aux(c, detached)
                        loss_a = F.relu(1-real_score).mean() + F.relu(1+fake_score).mean()
                    skipped_aux += not update(loss_a, opt_a, aux.parameters())
                    frozen(aux, True)
                opt_g.zero_grad(set_to_none=True)
                if cfg['method'] == 'esrgan':
                    aux.requires_grad_(False)  # Keep BN train mode as in BasicSR.
                loss_g, mse, fake = generator_objective(gen, aux, c, real, cfg, perceptual, bridge)
                success = update(loss_g, opt_g, gen.parameters())
                skipped_g += not success
                successful_g += success
                if success:
                    if scheduler is not None:
                        scheduler.step()
                    with torch.no_grad():
                        for target, current in zip(ema.parameters(), gen.parameters()):
                            target.lerp_(current, 1-cfg['ema_decay'])
                if cfg['method'] == 'esrgan':
                    aux.requires_grad_(True)
                    opt_a.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                        real_score = aux((real+1)/2).float()
                        fake_score = aux((fake.detach()+1)/2).float()
                    loss_a = relativistic_loss(real_score, fake_score, generator=False)
                    skipped_aux += not update(loss_a, opt_a, aux.parameters())
                step += 1
                consumed += len(batch)
                # Synchronization gives a real GPU wall-time budget, not enqueue time.
                torch.cuda.synchronize()
                if step % cfg['log_every'] == 0:
                    row = dict(step=step, seconds=elapsed+time.monotonic()-start,
                               g=loss_g.item(), mse=mse.item(),
                               aux=loss_a.item() if aux is not None else None,
                               skipped_g=skipped_g, skipped_aux=skipped_aux)
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    log.flush()
                    progress.update(row['seconds'], step, {key: row[key] for key in
                                    ('g', 'mse', 'aux', 'skipped_g', 'skipped_aux')})
                else:
                    progress.update(elapsed+time.monotonic()-start, step)
                if time.monotonic() - previous_checkpoint >= cfg['checkpoint_seconds']:
                    save(False)
                    previous_checkpoint = time.monotonic()
        complete = not interrupted and successful_g > 0 and elapsed+time.monotonic()-start >= cfg['seconds']
        result = save(complete)
        progress.update(result['elapsed_training_seconds'], step)
        progress.close()
        if interrupted:
            print(f'Training interrupted by {result["received_signals"]}; '
                  f'saved {latest} at {result["elapsed_training_seconds"]:.1f}/{cfg["seconds"]:.1f} '
                  'training seconds. Resume this checkpoint; exit code 75.', flush=True)
            raise SystemExit(75)
        if not complete or successful_g == 0:
            raise RuntimeError('Training did not complete any successful G updates')
        print(f'Training complete: {step} steps, {successful_g} successful G updates; '
              f'{result["elapsed_training_seconds"]:.1f} training seconds; checkpoint {latest}', flush=True)
        return result
    finally:
        if progress is not None:
            progress.update(elapsed+time.monotonic()-start, step)
            progress.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
