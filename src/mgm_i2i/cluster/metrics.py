"""Fixed draws; aggregate crops within source images before averaging images."""
import json
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..evaluate import ssim
from .data import CropBank, validate_data_protocol, MULTICROP_PROTOCOL, load_multicrop_evaluation
from .engine import atomic_json, fingerprint, provenance, seed_all, sr_forward
from .models import project
from .baselines import build_generator, I2SBBridge


def fixed_latents(start, count, shape, device, draw):
    # CPU generator: independent of batch size and CUDA device/model RNG state.
    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(
        20260101 + (start+i)*2 + draw)) for i in range(count)]).to(device)


def aggregate_images(rows, keys):
    """Each source photograph contributes equally, regardless of crop count."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['image'], []).append(row)
    per_image = [dict(image=name, **{key: float(np.mean([r[key] for r in members])) for key in keys})
                 for name, members in sorted(grouped.items())]
    return per_image, {key: float(np.mean([row[key] for row in per_image])) for key in keys}


def predict_sr(model, c, z, cfg, *, bridge=None, nfe=1, offset=0, draw=0):
    if cfg['method'] not in ('i2sb', 'i2sb_official'):
        return sr_forward(model, c, z, cfg)
    # A separate RNG for each source image makes draws independent of batch size.
    generators = [torch.Generator(device=c.device).manual_seed(20260101 + (offset+i)*2 + draw)
                  for i in range(len(c))]
    degraded = F.interpolate(c, scale_factor=cfg['scale'], mode='nearest')
    return bridge.sample(model, degraded, nfe, generators=generators)


def evaluate(checkpoint, output, split='val', batch_size=4, nfe=1, *,
             evaluation_data_root=None, evaluation_data_hash=None, expected_config=None):
    # A complete training checkpoint already exists. Use the full remaining
    # allocation for evaluation instead of dying on the advance warning signal.
    previous = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        return _evaluate(checkpoint, output, split, batch_size, nfe,
                         evaluation_data_root=evaluation_data_root,
                         evaluation_data_hash=evaluation_data_hash, expected_config=expected_config)
    finally:
        signal.signal(signal.SIGUSR1, previous)


def evaluation_bank(cfg, training_data_hash, split, external_root=None, external_hash=None):
    """Keep checkpoint training provenance separate from a shared external bank."""
    if external_root is not None:
        from .external_data import load_bank
        if external_hash is None:
            raise ValueError('Shared evaluation requires a pinned manifest hash')
        root = Path(external_root)
        if split == 'external':
            manifest, data_hash = load_bank(root, external_hash)
            dataset = CropBank(root/'external.npy', crop=128, training=False, mode='center')
            return dataset, manifest['banks']['external.npy']['crops'], data_hash, 'BSD100', 128
        manifest, data_hash = load_multicrop_evaluation(root, split, external_hash)
        dataset = CropBank(root/f'{split}.npy', crop=128, training=False, mode='fixed')
        return dataset, manifest['banks'][f'{split}.npy']['crops'], data_hash, 'DIV2K', 128
    if split not in ('val', 'test'):
        raise ValueError('Native evaluation requires val or test split')
    root = Path(cfg['data_root'])
    validate_data_protocol(root, cfg)
    if fingerprint(root/'manifest.json') != training_data_hash:
        raise ValueError('Dataset manifest changed since training')
    manifest = json.loads((root/'manifest.json').read_text())
    dataset = CropBank(root/f'{split}.npy', crop=cfg['eval_crop'], training=False,
                       mode='fixed' if cfg.get('data_protocol') == MULTICROP_PROTOCOL else
                            'center' if cfg.get('data_protocol') else 'legacy')
    return dataset, manifest['banks'][f'{split}.npy']['crops'], training_data_hash, 'DIV2K', cfg['eval_crop']


@torch.no_grad()
def _evaluate(checkpoint, output, split, batch_size, nfe=1, *,
              evaluation_data_root=None, evaluation_data_hash=None, expected_config=None):
    started = time.monotonic()
    checkpoint, output = Path(checkpoint), Path(output)
    official = expected_config is not None and expected_config.get('method') == 'i2sb_official'
    official_metadata = {}
    if official:
        from .official_i2sb import load_for_evaluation
        state, gen, bridge, official_metadata = load_for_evaluation(expected_config, checkpoint)
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state['complete'] or state['config'].get('field_only', False):
        raise ValueError('Evaluate a completed run, or resume training first')
    cfg = state['config']
    if expected_config is not None and cfg != expected_config:
        raise ValueError('Checkpoint config differs from the source run plan')
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpus'])
    dataset, metadata, eval_hash, dataset_name, eval_crop = evaluation_bank(
        cfg, state['data_hash'], split, evaluation_data_root, evaluation_data_hash)
    if len(metadata) != len(dataset):
        raise ValueError('Crop metadata length mismatch')
    weights = Path(cfg['weights_root'])
    if not (weights/'hub/checkpoints/alexnet-owt-7be5be79.pth').is_file():
        raise FileNotFoundError('Prepare AlexNet weights before GPU evaluation; no auto-download allowed')
    torch.hub.set_dir(str(weights/'hub'))
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval()
    if not official:
        gen = build_generator(cfg).cuda().eval()
        gen.load_state_dict(state['ema'])
        bridge = (I2SBBridge(cfg['i2sb_intervals'], cfg['i2sb_beta_max']).cuda()
                  if cfg['method'] == 'i2sb' else None)
    if cfg['method'] not in ('i2sb', 'i2sb_official') and nfe != 1:
        raise ValueError('Only I2SB supports NFE > 1')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg['workers'], pin_memory=True)
    rows, pictures, offset, latency = [], [], 0, []
    pictured_images = set()
    individual_samples = evaluation_data_root is not None or cfg.get('data_protocol') == MULTICROP_PROTOCOL
    if individual_samples:
        (output/'samples').mkdir(parents=True, exist_ok=True)
    for batch in loader:
        real = batch.cuda(non_blocking=True).float().div_(127.5).sub_(1)
        c = F.avg_pool2d(real, cfg['scale'])
        z1 = fixed_latents(offset, len(real), real.shape[1:], real.device, 0)
        z2 = fixed_latents(offset, len(real), real.shape[1:], real.device, 1)
        if cfg['method'] in ('mse', 'mse_perceptual', 'esrgan', 'rrdb_l1'):
            z1.zero_()
            z2.zero_()
        # FP32 inference for every method. First batch excluded from throughput.
        torch.cuda.synchronize()
        tick = time.perf_counter()
        fake = predict_sr(gen, c, z1, cfg, bridge=bridge, nfe=nfe, offset=offset, draw=0)
        torch.cuda.synchronize()
        if offset:
            latency.append((time.perf_counter()-tick, len(real)))
        other = predict_sr(gen, c, z2, cfg, bridge=bridge, nfe=nfe, offset=offset, draw=1)
        bicubic = F.interpolate(c, size=real.shape[-2:], mode='bicubic', align_corners=False)
        projected = project(bicubic, c, cfg['scale'])
        outputs = {'model': fake, 'model_projected': project(fake, c, cfg['scale']),
                   'bicubic': bicubic, 'projected_bicubic': projected}
        values = {}
        for name, raw in outputs.items():
            clipped = raw.clamp(-1, 1)
            x, y = (clipped+1)/2, (real+1)/2
            values[name+'_psnr'] = -10*(x-y).square().flatten(1).mean(1).clamp_min(1e-12).log10()
            values[name+'_ssim'] = ssim(x, y)
            values[name+'_lpips'] = metric(clipped, real).flatten()
            values[name+'_lr_mse_raw'] = (F.avg_pool2d(raw, cfg['scale'])-c).square().flatten(1).mean(1)
            values[name+'_lr_mse_clipped'] = (F.avg_pool2d(clipped, cfg['scale'])-c).square().flatten(1).mean(1)
            values[name+'_out_of_range'] = (raw.abs()>1).float().flatten(1).mean(1)
        values['diversity_mse_raw'] = (fake-other).square().flatten(1).mean(1)
        values['diversity_lpips'] = metric(fake.clamp(-1, 1), other.clamp(-1, 1)).flatten()
        host = {key: tensor.cpu().tolist() for key, tensor in values.items()}
        for i in range(len(real)):
            crop_info = metadata[offset+i]
            coordinates = {key: crop_info[key] for key in ('crop_index', 'top', 'left') if key in crop_info}
            rows.append(dict(index=offset+i, image=crop_info['image'], **coordinates,
                             **{key: items[i] for key, items in host.items()}))
            preview = len(pictured_images) < 8 and crop_info['image'] not in pictured_images
            if preview or individual_samples:
                views = [F.interpolate(c, size=real.shape[-2:], mode='nearest')[i],
                         bicubic[i], projected[i], real[i], fake[i], other[i]]
                view_arrays = [((v.clamp(-1,1)+1)*127.5).round().byte()
                               .permute(1,2,0).cpu().numpy() for v in views]
                grid = np.concatenate(view_arrays, axis=1)
                if preview:
                    pictures.append(grid)
                    pictured_images.add(crop_info['image'])
                if individual_samples:
                    final_projected = ((outputs['model_projected'][i].clamp(-1,1)+1)*127.5).round().byte()
                    full_grid = np.concatenate([grid, final_projected.permute(1,2,0).cpu().numpy()], axis=1)
                    filename = f'{offset+i:03d}_{Path(metadata[offset+i]["image"]).stem}.png'
                    Image.fromarray(full_grid).save(output/'samples'/filename)
        offset += len(real)
        if official and (offset % 25 == 0 or offset == len(dataset)):
            print(f'Official I2SB NFE={nfe}: {offset}/{len(dataset)} crops; '
                  f'elapsed={(time.monotonic()-started)/60:.1f} min', flush=True)
    per_image, metrics = aggregate_images(rows, values.keys())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/f'{split}_patches.json', rows)
    atomic_json(output/f'{split}_images.json', per_image)
    Image.fromarray(np.concatenate(pictures, axis=0)).save(output/f'{split}_samples.png')
    result = dict(metrics=metrics, split=split, images=len(per_image), crops=len(rows),
                  seed=cfg['seed'], variant=cfg['variant'], nfe=nfe, checkpoint_sha256=fingerprint(checkpoint),
                  data_hash=state['data_hash'], step=state['step'], successful_g=state['successful_g'],
                  training_data_hash=state['data_hash'], evaluation_data_hash=eval_hash,
                  evaluation_dataset=dataset_name, evaluation_crop=eval_crop,
                  architecture_version=cfg.get('architecture_version', 1),
                  consumed=state['consumed'], training_seconds=state['elapsed_training_seconds'],
                  training_stop=state.get('training_stop', 'time'),
                  target_epochs=state.get('target_epochs'), epochs_completed=state.get('epochs_completed'),
                  warmup_training_seconds=state.get('warmup_training_seconds', 0.),
                  field_warmup_seconds=state.get('field_warmup_seconds', 0.),
                  field_init_hash=state.get('field_init_hash'),
                  evaluation_seconds=time.monotonic()-started,
                  inference_ms_per_image=1000*sum(t for t,n in latency)/sum(n for t,n in latency) if latency else None,
                  eval_batch_size=batch_size, precision='fp32',
                  environment=provenance(),
                  protocol='RGB clipped [0,1] PSNR/SSIM; LPIPS AlexNet v0.1 at native crop size; no border shave; mean of crop scores per source image, then mean images',
                  consistency_units='MSE on [-1,1] tensors; raw and clipped reported separately',
                  grid_columns='nearest LR / bicubic / projected bicubic / HR / draw 1 / draw 2',
                  individual_grid_columns=('nearest LR / bicubic / projected bicubic / HR / draw 1 / draw 2 / projected draw 1'
                                           if individual_samples else None),
                  throughput_protocol='Full SR inference in FP32, including all I2SB NFE/noise; GPU synchronization; excludes first batch, transfer, metrics and final projection; compare only equal batch sizes')
    if official:
        result.update(official_metadata)
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(output/f'{split}_metrics.json', result)
    return result
