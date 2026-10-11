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
from .data import CropBank
from .engine import atomic_json, fingerprint, provenance, seed_all
from .models import Generator, project


def fixed_latents(start, count, shape, device, draw):
    # CPU generator: independent of batch size and CUDA device/model RNG state.
    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(
        20260101 + (start+i)*2 + draw)) for i in range(count)]).to(device)


def evaluate(checkpoint, output, split='val', batch_size=4):
    # A complete training checkpoint already exists. Use the full remaining
    # allocation for evaluation instead of dying on the advance warning signal.
    previous = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        return _evaluate(checkpoint, output, split, batch_size)
    finally:
        signal.signal(signal.SIGUSR1, previous)


@torch.no_grad()
def _evaluate(checkpoint, output, split, batch_size):
    started = time.monotonic()
    checkpoint, output = Path(checkpoint), Path(output)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state['complete']:
        raise ValueError('Evaluate a completed run, or resume training first')
    cfg = state['config']
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpus'])
    root = Path(cfg['data_root'])
    if fingerprint(root/'manifest.json') != state['data_hash']:
        raise ValueError('Dataset manifest changed since training')
    weights = Path(cfg['weights_root'])
    if not (weights/'hub/checkpoints/alexnet-owt-7be5be79.pth').is_file():
        raise FileNotFoundError('Prepare AlexNet weights before GPU evaluation; no auto-download allowed')
    torch.hub.set_dir(str(weights/'hub'))
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval()
    gen = Generator(cfg['width'], cfg['scale'], cfg['projection']).cuda().eval()
    gen.load_state_dict(state['ema'])
    manifest = json.loads((root/'manifest.json').read_text())
    dataset = CropBank(root/f'{split}.npy', crop=cfg['eval_crop'], training=False)
    metadata = manifest['banks'][f'{split}.npy']['crops']
    if len(metadata) != len(dataset):
        raise ValueError('Crop metadata length mismatch')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg['workers'], pin_memory=True)
    rows, pictures, offset, latency = [], [], 0, []
    for batch in loader:
        real = batch.cuda(non_blocking=True).float().div_(127.5).sub_(1)
        c = F.avg_pool2d(real, cfg['scale'])
        z1 = fixed_latents(offset, len(real), real.shape[1:], real.device, 0)
        z2 = fixed_latents(offset, len(real), real.shape[1:], real.device, 1)
        if cfg['method'] == 'mse':
            z1.zero_()
            z2.zero_()
        # FP32 inference for every method. First batch excluded from throughput.
        torch.cuda.synchronize()
        tick = time.perf_counter()
        fake = gen(c, z1)
        torch.cuda.synchronize()
        if offset:
            latency.append((time.perf_counter()-tick, len(real)))
        other = gen(c, z2)
        bicubic = F.interpolate(c, size=real.shape[-2:], mode='bicubic', align_corners=False)
        projected = project(bicubic, c, cfg['scale'])
        outputs = {'model': fake, 'bicubic': bicubic, 'projected_bicubic': projected}
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
            rows.append(dict(index=offset+i, image=metadata[offset+i]['image'],
                             **{key: items[i] for key, items in host.items()}))
            if offset+i < 8:
                views = [F.interpolate(c, size=real.shape[-2:], mode='nearest')[i],
                         bicubic[i], projected[i], real[i], fake[i], other[i]]
                pictures.append(np.concatenate([((v.clamp(-1,1)+1)*127.5).round().byte()
                    .permute(1,2,0).cpu().numpy() for v in views], axis=1))
        offset += len(real)
    grouped = {}
    for row in rows:
        grouped.setdefault(row['image'], []).append(row)
    per_image = [dict(image=name, **{key: float(np.mean([r[key] for r in members]))
                                     for key in values}) for name, members in sorted(grouped.items())]
    metrics = {key: float(np.mean([r[key] for r in per_image])) for key in values}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/f'{split}_patches.json', rows)
    atomic_json(output/f'{split}_images.json', per_image)
    Image.fromarray(np.concatenate(pictures, axis=0)).save(output/f'{split}_samples.png')
    result = dict(metrics=metrics, split=split, images=len(per_image), crops=len(rows),
                  seed=cfg['seed'], variant=cfg['variant'], checkpoint_sha256=fingerprint(checkpoint),
                  data_hash=state['data_hash'], step=state['step'], successful_g=state['successful_g'],
                  consumed=state['consumed'], training_seconds=state['elapsed_training_seconds'],
                  evaluation_seconds=time.monotonic()-started,
                  inference_ms_per_image=1000*sum(t for t,n in latency)/sum(n for t,n in latency) if latency else None,
                  eval_batch_size=batch_size, precision='fp32',
                  environment=provenance(),
                  protocol='RGB clipped [0,1] PSNR/SSIM; LPIPS AlexNet v0.1 at native crop size; no border shave; mean of crop scores per source image, then mean images',
                  consistency_units='MSE on [-1,1] tensors; raw and clipped reported separately',
                  grid_columns='nearest LR / bicubic / projected bicubic / HR / draw 1 / draw 2',
                  throughput_protocol='G forward only; GPU synchronization; excludes first batch, transfer and latent sampling; not batch-1 latency')
    atomic_json(output/f'{split}_metrics.json', result)
    return result
