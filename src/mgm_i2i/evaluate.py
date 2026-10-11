import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import CIFARData, latent
from .models import upsample


def ssim(x, y):
    """RGB mean, 11x11 Gaussian window, valid convolution, data_range=1."""
    pos = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    weights = (-pos.square() / (2 * 1.5**2)).exp()
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :])[None, None].repeat(3, 1, 1, 1)
    conv = lambda a: F.conv2d(a, kernel, groups=3)
    mx, my = conv(x), conv(y)
    vx, vy = conv(x*x) - mx*mx, conv(y*y) - my*my
    xy = conv(x*y) - mx*my
    return (((2*mx*my + .01**2)*(2*xy + .03**2)) /
            ((mx*mx + my*my + .01**2)*(vx + vy + .03**2))).flatten(1).mean(1)


@torch.no_grad()
def evaluate_toy(generator, device, deterministic=False, output=None):
    generator.eval()
    c = torch.linspace(-.45, .45, 32, device=device).reshape(-1, 1)
    samples = 256
    cc = c.repeat_interleave(samples, 0)
    yy = generator(cc, latent(torch.zeros(len(cc), 2, device=device), deterministic))
    yy = yy.reshape(len(c), samples, 2)
    high = (yy[:, :, 0] - yy[:, :, 1]) / 2
    positive = (high > 0).float().mean(1)
    sorted_high = high.sort(1).values
    target = torch.cat([torch.full((samples//2,), -.35, device=device),
                        torch.full((samples//2,), .35, device=device)])
    metrics = {
        'conditional_w1': (sorted_high - target).abs().mean().item(),
        'mode_bias': (positive - .5).abs().mean().item(),
        'off_mode_distance': (high.abs() - .35).abs().mean().item(),
        'consistency_mse': (yy.mean(2) - c).square().mean().item(),
        'conditional_std': high.std(1, unbiased=False).mean().item(),
        'oracle_w1': 0., 'mean_predictor_w1': .35,
        'evaluation': '32 held-out condition values; 256 draws each; exact known conditional modes',
    }
    if output:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        axes[0].hist(high[16].cpu().numpy(), bins=40, range=(-.8, .8), density=True)
        for mode in [-.35, .35]:
            axes[0].axvline(mode, color='red', linestyle='--')
        axes[0].set(title='High-frequency residual at one condition', xlabel='(pixel 1 - pixel 2) / 2')
        axes[1].plot(c.flatten().cpu(), positive.cpu(), marker='.')
        axes[1].axhline(.5, color='red', linestyle='--')
        axes[1].set(title='Mode probability by input', xlabel='LR value c', ylabel='P(residual > 0)', ylim=(0, 1))
        fig.savefig(Path(output) / 'toy_distribution.png', dpi=160)
        plt.close(fig)
    return metrics


@torch.no_grad()
def evaluate_images(generator, cfg, device, output, use_lpips=False, split='val'):
    from PIL import Image
    generator.eval()
    dataset = CIFARData(device, root=cfg['data_root'], split=split,
                        limit=cfg['eval_limit'], seed=cfg['data_seed'])
    perceptual = None
    if use_lpips:
        # Check before LPIPS initialization so there is no implicit download.
        weight = Path(__file__).resolve().parents[2] / 'offline/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth'
        if not weight.exists():
            raise FileNotFoundError('Local AlexNet weights are missing; LPIPS will not download them.')
        import lpips
        perceptual = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    totals = dict(psnr=0., ssim=0., mse_raw=0., consistency_mse_raw=0.,
                  consistency_mse_clipped=0., diversity_mse=0., out_of_range_fraction=0.,
                  bicubic_psnr=0., bicubic_ssim=0.)
    if perceptual:
        totals['lpips'] = 0.
        totals['bicubic_lpips'] = 0.
    count = 0
    for start in range(0, len(dataset.indices), cfg['batch_size']):
        import numpy as np
        c, real = dataset.get(np.arange(start, min(start + cfg['batch_size'], len(dataset.indices))))
        deterministic = cfg['method'] == 'mse'
        fake = generator(c, latent(real, deterministic))
        other = generator(c, latent(real, deterministic))
        baseline = upsample(c, real.shape[-2:])
        clipped = fake.clamp(-1, 1)
        base_clipped = baseline.clamp(-1, 1)
        x, y = (clipped + 1) / 2, (real + 1) / 2
        bx = (base_clipped + 1) / 2
        mse = (x-y).square().flatten(1).mean(1).clamp_min(1e-12)
        bmse = (bx-y).square().flatten(1).mean(1).clamp_min(1e-12)
        n = len(real)
        totals['psnr'] += (-10 * mse.log10()).sum().item()
        totals['ssim'] += ssim(x, y).sum().item()
        totals['bicubic_psnr'] += (-10 * bmse.log10()).sum().item()
        totals['bicubic_ssim'] += ssim(bx, y).sum().item()
        totals['mse_raw'] += (fake-real).square().mean().item() * n
        totals['consistency_mse_raw'] += (F.avg_pool2d(fake, 2)-c).square().mean().item()*n
        totals['consistency_mse_clipped'] += (F.avg_pool2d(clipped, 2)-c).square().mean().item()*n
        totals['diversity_mse'] += (fake-other).square().mean().item()*n
        totals['out_of_range_fraction'] += (fake.abs() > 1).float().mean().item()*n
        if perceptual:
            # Nearest x2 resizing to 64x64 lets AlexNet operate on tiny CIFAR inputs.
            # This is a documented local protocol, not a benchmark-standard score.
            resize = lambda a: F.interpolate(a, size=(64,64), mode='nearest')
            totals['lpips'] += perceptual(resize(clipped), resize(real)).sum().item()
            totals['bicubic_lpips'] += perceptual(resize(base_clipped), resize(real)).sum().item()
        if start == 0:
            rows = []
            for i in range(min(8,n)):
                views = [F.interpolate(c, size=real.shape[-2:], mode='nearest')[i], baseline[i], real[i], fake[i], other[i]]
                tiles = [((a.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy() for a in views]
                rows.append(np.concatenate(tiles, axis=1))
            grid = Image.fromarray(np.concatenate(rows, axis=0))
            grid.resize((grid.width*3, grid.height*3), Image.Resampling.NEAREST).save(Path(output)/'samples.png')
        count += n
    results = {k: v/count for k,v in totals.items()}
    results.update({'count': count, 'split': split, 'psnr_ssim': 'clipped output; RGB [0,1]; mean per-image PSNR',
                    'lpips_protocol': 'AlexNet v0.1, nearest resize 32 to 64; lower is better' if perceptual else 'not computed',
                    'grid_columns': 'nearest LR / bicubic / target / sample 1 / sample 2'})
    return results
