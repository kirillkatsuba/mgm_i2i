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
from .data import CropBank, SampleRange
from .models import Generator, Field, Discriminator
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


def train(cfg, output, init=None, resume=False):
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
    gen = Generator(cfg['width'], cfg['scale'], cfg['projection']).to(device)
    aux = (Field(cfg['width'], cfg['conditioned']) if cfg['method'] == 'mgm' else
           Discriminator(cfg['width']) if cfg['method'] == 'gan' else None)
    if aux is not None:
        aux = aux.to(device)
    data_hash = fingerprint(Path(cfg['data_root'])/'manifest.json')
    init_hash = None
    if init and not resume:
        source = torch.load(init, map_location='cpu', weights_only=False)
        if not source['complete'] or source['data_hash'] != data_hash:
            raise ValueError('Warmup must be complete and use the same dataset manifest')
        for key in ('seed', 'width', 'scale', 'projection'):
            if source['config'][key] != cfg[key]:
                raise ValueError(f'Warmup mismatch: {key}')
        gen.load_state_dict(source['generator'])
        init_hash = fingerprint(init)
    ema = copy.deepcopy(gen).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg['lr_g'], betas=(.5, .99))
    opt_a = torch.optim.Adam(aux.parameters(), lr=cfg['lr_aux'], betas=(.5, .99)) if aux is not None else None
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
        step, consumed, elapsed = state['step'], state['consumed'], state['elapsed_training_seconds']
        skipped_g, skipped_aux, successful_g = state['skipped_g'], state['skipped_aux'], state['successful_g']
        init_hash = state['init_hash']
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
    dataset = CropBank(Path(cfg['data_root'])/'train.npy', crop=cfg['crop'], seed=cfg['seed'])
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
                     successful_g=successful_g, skipped_g=skipped_g, skipped_aux=skipped_aux,
                     elapsed_training_seconds=elapsed + time.monotonic()-start,
                     peak_allocated_bytes_this_session=torch.cuda.max_memory_allocated(),
                     data_hash=data_hash, init_hash=init_hash, sessions=history, complete=complete,
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
                if aux is not None:
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
                z = torch.zeros_like(real) if cfg['method'] == 'mse' else torch.randn_like(real)
                with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                    fake = gen(c, z).float()
                mse = F.mse_loss(fake, real)
                if cfg['method'] == 'mse':
                    loss_g = mse
                elif cfg['method'] == 'mgm':
                    loss_g = mgm_objective(aux, c, fake, real, cfg, True)
                else:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['amp']):
                        loss_g = -aux(c, fake).mean()
                # Zero in the main MGM/GAN comparisons; optional reconstruction regularizer.
                if cfg['method'] != 'mse':
                    loss_g = loss_g + cfg['reconstruction_weight'] * mse
                success = update(loss_g, opt_g, gen.parameters())
                skipped_g += not success
                successful_g += success
                if success:
                    with torch.no_grad():
                        for target, current in zip(ema.parameters(), gen.parameters()):
                            target.lerp_(current, 1-cfg['ema_decay'])
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
