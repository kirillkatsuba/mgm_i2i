"""Compare G(c, alpha*z) with shared validation inputs and exactly the same z at every alpha."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.baselines import build_generator
from mgm_i2i.cluster.engine import atomic_json, fingerprint, provenance, seed_all
from mgm_i2i.cluster.metrics import evaluation_bank, fixed_latents, aggregate_images
from mgm_i2i.evaluate import ssim


@torch.no_grad()
def latent_outputs(model, c, z, alphas):
    return [model(c, z*alpha).float() for alpha in alphas]


def rgb(image):
    return ((image.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()


def labeled_sample(name, c, real, outputs, alphas):
    views = [F.interpolate(c[None], size=real.shape[-2:], mode='nearest')[0], real] + outputs
    image = Image.new('RGB', (128*len(views), 156), 'white')
    draw = ImageDraw.Draw(image)
    labels = ['LR nearest', 'HR target'] + [f'alpha={a:g}' for a in alphas]
    for index, (view, label) in enumerate(zip(views, labels)):
        draw.text((index*128+3, 2), label, fill='black')
        if index == 0:
            draw.text((3, 14), name, fill='black')
        image.paste(Image.fromarray(rgb(view)), (index*128, 28))
    return image


@torch.no_grad()
def diagnose(checkpoint, output, *, alphas=(0., .1, .5, 1.), weights='generator'):
    checkpoint, output = Path(checkpoint), Path(output)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    if not state['complete'] or cfg['method'] not in ('mse','mgm'):
        raise ValueError('Expected a completed U-Net MSE warmup or MGM checkpoint')
    if 0. not in alphas or len(set(alphas)) != len(alphas) or any(a < 0 for a in alphas):
        raise ValueError('Unique nonnegative alphas including zero are required')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic in a Slurm GPU allocation')
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['cpus'])
    dataset, metadata, data_hash, _, crop = evaluation_bank(cfg, state['data_hash'], 'val')
    if crop != 128:
        raise ValueError('This controlled diagnostic requires the current native128 validation protocol')
    weight_root = Path(cfg['weights_root'])
    if not (weight_root/'hub/checkpoints/alexnet-owt-7be5be79.pth').is_file():
        raise FileNotFoundError('Prepare local AlexNet weights; diagnostic never downloads weights')
    torch.hub.set_dir(str(weight_root/'hub'))
    import lpips
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).cuda().eval()
    model = build_generator(cfg).cuda().eval()
    model.load_state_dict(state[weights])
    output.mkdir(parents=True, exist_ok=True)
    (output/'samples').mkdir(exist_ok=True)
    rows, pictures, pictured_images = [], [], set()
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for index, batch in enumerate(loader):
        real = batch.cuda().float().div(127.5).sub(1)
        c = F.avg_pool2d(real, cfg['scale'])
        z = fixed_latents(index, 1, real.shape[1:], real.device, draw=0)
        outputs = latent_outputs(model, c, z, alphas)
        zero = outputs[alphas.index(0.)]
        for alpha, raw in zip(alphas, outputs):
            clipped = raw.clamp(-1,1)
            error = ((clipped-real)/2).square().mean()
            coordinates = {key:metadata[index][key] for key in ('crop_index','top','left') if key in metadata[index]}
            rows.append(dict(image=metadata[index]['image'], alpha=alpha, **coordinates,
                             psnr=float(-10*error.clamp_min(1e-12).log10()),
                             ssim=float(ssim((clipped+1)/2,(real+1)/2)),
                             lpips=float(metric(clipped,real).flatten()[0]),
                             raw_mse=float((raw-real).square().mean()),
                             latent_delta_mse=float((raw-zero).square().mean()),
                             lr_mse_raw=float((F.avg_pool2d(raw,cfg['scale'])-c).square().mean()),
                             out_of_range=float((raw.abs()>1).float().mean())))
        picture = labeled_sample(metadata[index]['image'], c[0], real[0], [x[0] for x in outputs], alphas)
        picture.save(output/'samples'/f'{index:03d}_{Path(metadata[index]["image"]).stem}.png')
        if len(pictured_images) < 8 and metadata[index]['image'] not in pictured_images:
            pictures.append(np.asarray(picture))
            pictured_images.add(metadata[index]['image'])
    keys = [key for key in rows[0] if key not in ('image','alpha','crop_index','top','left')]
    per_image, summary = [], []
    for alpha in alphas:
        images, means = aggregate_images([row for row in rows if row['alpha']==alpha], keys)
        per_image.extend(dict(alpha=alpha, **row) for row in images)
        summary.append(dict(alpha=alpha, **means))
    with (output/'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    with (output/'per_crop.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Image.fromarray(np.concatenate(pictures, axis=0)).save(output/'alpha_samples.png')
    result = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=fingerprint(checkpoint),
                  weights=weights, seed=cfg['seed'], data_hash=data_hash, split='val',
                  images=len({row['image'] for row in rows}), crops=len(dataset),
                  summary=summary, environment=provenance(),
                  note='Frozen network, FP32; same fixed z and HR/LR per alpha. No test/tuning, no retraining. '
                       'generator weights are the exact warmup initialization used by the engine; EMA is optional.')
    atomic_json(output/'diagnostic.json', result)
    lines = ['# Warmup latent diagnostic', '',
             f'Weights: {weights}; seed {cfg["seed"]}; {result["images"]} validation images, {len(dataset)} crops.', '',
             '| alpha | PSNR | SSIM | LPIPS | Raw MSE | Change from alpha0, raw MSE |',
             '|---|---|---|---|---|---|']
    for row in summary:
        lines.append('| '+' | '.join(f'{row[k]:.6g}' for k in
                     ('alpha','psnr','ssim','lpips','raw_mse','latent_delta_mse'))+' |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.,.1,.5,1.])
    parser.add_argument('--weights', choices=['generator','ema'], default='generator')
    args = parser.parse_args()
    diagnose(args.checkpoint, args.output, alphas=args.alphas, weights=args.weights)


if __name__ == '__main__':
    main()
