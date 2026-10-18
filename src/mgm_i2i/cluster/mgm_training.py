"""Warm-start the conditional field and accumulate MGM gradients without retaining graphs."""
import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def initialize_field(field, generator):
    if field.architecture_version != generator.architecture_version:
        raise ValueError('Field and generator architecture versions must match')
    source, target = generator.net.state_dict(), field.net.state_dict()
    copied, remapped, reset = [], [], []
    for name, tensor in target.items():
        if name.startswith('out.') or '.time_projection.' in name:
            tensor.zero_()
            reset.append(name)
        elif name == 'stem.weight':
            # Warmup's Z channels never received a signal at Z=0. Reuse the
            # trained RGB filters for BOTH LR and mixed-HR branches instead.
            rgb = source[name][:, :3]
            tensor.zero_()
            tensor[:, :3].copy_(.5 * rgb)
            tensor[:, 3:6].copy_(.5 * rgb)
            remapped.append(name)
        elif name in source and source[name].shape == tensor.shape:
            tensor.copy_(source[name])
            copied.append(name)
        else:
            raise ValueError(f'Unsupported field warm-start tensor: {name}')
    field.net.load_state_dict(target)
    return dict(strategy='warmup_features', copied=copied, remapped=remapped, reset=reset,
                note='Hidden conv/norm/skip weights transferred. Trained RGB stem filters split '
                     'between LR and mixed HR; latent filters not copied. New displacement head '
                     'and time injections zeroed. Time MLP keeps its independent initialization.')


def accumulated_update(chunks, optimizer, parameters, scaler, *, count, grad_clip, amp):
    """Each chunk yields a mean loss and detached scalar diagnostics.

    All chunks see fixed parameters; clip and Adam step occur once per effective
    batch. Graphs are released before computing the next microbatch.
    """
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    totals, seen = {}, 0
    for loss, diagnostics in chunks():
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite microbatch loss')
        scaler.scale(loss/count).backward()
        for key, value in dict(loss=loss, **diagnostics).items():
            totals[key] = totals.get(key, 0.) + value.detach()/count
        seen += 1
    if seen != count:
        raise ValueError(f'Expected {count} equal microbatches, got {seen}')
    scaler.unscale_(optimizer)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, grad_clip))
    finite = math.isfinite(norm)
    if not amp and not finite:
        raise FloatingPointError('Non-finite FP32 gradient; last checkpoint retained')
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return dict(success=scaler.get_scale() >= before, values=totals,
                grad_norm=norm if finite else None, clipped=bool(norm > grad_clip) if finite else None,
                nonfinite_gradient=not finite, amp_scale=scaler.get_scale())


def mgm_step(generator, field, real, config, optimizer_g, optimizer_f, scaler, objective):
    count = config.get('grad_accumulation', 1)
    size = config['batch_size']
    if len(real) != size * count:
        raise ValueError('Expected physical batch size times accumulation factor')
    batches = real.split(size)

    def field_chunks():
        for target in batches:
            c = F.avg_pool2d(target, config['scale'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                fake = generator(c, torch.randn_like(target)).float()
            loss = objective(field, c, fake, target, config, False)
            yield loss, {'zero_mse': F.mse_loss(fake, target)}

    field.requires_grad_(True).train()
    field_result = accumulated_update(field_chunks, optimizer_f, field.parameters(), scaler,
                                      count=count, grad_clip=config['grad_clip'], amp=config['amp'])
    field.requires_grad_(False).eval()
    if config.get('field_only', False):
        return dict(field=field_result, generator=None)

    def generator_chunks():
        for target in batches:
            c = F.avg_pool2d(target, config['scale'])
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                fake = generator(c, torch.randn_like(target)).float()
            mse = F.mse_loss(fake, target)
            loss = objective(field, c, fake, target, config, True)
            loss = loss + config['reconstruction_weight'] * mse
            yield loss, {'mse': mse}

    generator_result = accumulated_update(generator_chunks, optimizer_g, generator.parameters(), scaler,
                                          count=count, grad_clip=config['grad_clip'], amp=config['amp'])
    return dict(field=field_result, generator=generator_result)


def validate_field_checkpoint(state, config, data_hash, init_hash):
    source = state['config']
    if not state['complete'] or not source.get('field_only') or state['data_hash'] != data_hash:
        raise ValueError('Expected completed field-only warmup on the same training data')
    if init_hash is None or state['init_hash'] != init_hash:
        raise ValueError('Field warmup used different generator initialization')
    for key in ('seed', 'width', 'scale', 'projection', 'conditioned', 'architecture_version',
                'noise_f', 'batch_size', 'grad_accumulation', 'lr_aux', 'adam_betas', 'field_initialization'):
        if source.get(key) != config.get(key):
            raise ValueError(f'Field warmup mismatch: {key}')
    if state['step'] - state['skipped_aux'] <= 0:
        raise ValueError('Field warmup has no successful updates')
