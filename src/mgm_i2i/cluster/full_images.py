"""Full-frame DIV2K inference, kept separate from the fixed crop benchmark."""
import csv
import html
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..evaluate import ssim
from .engine import atomic_json, fingerprint, provenance
from .models import Generator, project


def prepare_frame(pixels, scale):
    """Preserve every source pixel; reflect-pad only the bottom/right LR boundary."""
    h, w = pixels.shape[:2]
    if min(h, w) < max(11, scale):
        raise ValueError('Image too small for the model and 11x11 SSIM')
    real = torch.from_numpy(np.array(pixels, copy=True)).permute(2, 0, 1)[None].float()/127.5 - 1
    bottom, right = (-h) % scale, (-w) % scale
    padded = F.pad(real, (0, right, 0, bottom), mode='reflect')
    return real, padded, F.avg_pool2d(padded, scale), dict(bottom=bottom, right=right)


def frame_latent(shape, image_id, draw, deterministic=False):
    """One spatial latent for the entire image, independent of model/seed/order."""
    if deterministic:
        return torch.zeros(shape)
    rng = torch.Generator().manual_seed(20261015 + int(image_id)*2 + draw)
    return torch.randn(shape, generator=rng)


def save_picture(path, tensor):
    pixels = ((tensor[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).numpy()
    picture = Image.fromarray(pixels)
    picture.save(path)
    preview = picture.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    preview.save(path.with_name(path.stem+'_preview.jpg'), quality=92)


def frame_scores(candidate, target):
    """Full-frame RGB clipped scores; CPU keeps the metric out of GPU memory."""
    if not torch.isfinite(candidate).all():
        raise FloatingPointError('Non-finite full-frame output')
    x, y = (candidate.clamp(-1, 1)+1)/2, (target+1)/2
    mse = (x-y).square().mean().clamp_min(1e-12)
    return dict(psnr=float(-10*mse.log10()), ssim=float(ssim(x, y)[0]),
                out_of_range=float((candidate.abs()>1).float().mean()))


def write_gallery(path, title, sections, notice):
    """Small previews link to the original-size PNGs; no server/network required."""
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>'+html.escape(title)+'</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}',
             '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}',
             'figure{margin:0;padding:12px;background:white;border:1px solid #ddd}',
             'img{display:block;width:100%;height:auto}figcaption{margin-bottom:8px}',
             'section{margin:32px 0}p{max-width:1000px}</style>',
             '<h1>'+html.escape(title)+'</h1><p>'+html.escape(notice)+'</p>']
    for label, cards in sections:
        parts.append('<section><h2>'+html.escape(label)+'</h2><div class="grid">')
        for caption, full_path in cards:
            relative = full_path.relative_to(path.parent).as_posix()
            preview = full_path.with_name(full_path.stem+'_preview.jpg').relative_to(path.parent).as_posix()
            parts.append('<figure><figcaption>'+html.escape(caption)+'</figcaption><a href="'+
                         html.escape(relative, quote=True)+'"><img loading="lazy" src="'+
                         html.escape(preview, quote=True)+'" alt="'+html.escape(caption, quote=True)+
                         '"></a></figure>')
        parts.append('</div></section>')
    parts.append('</html>')
    path.write_text('\n'.join(parts), encoding='utf-8')


@torch.inference_mode()
def infer_full(item, output_root, split='val', limit=8, image_ids=None, archive_path=None, precision='fp32'):
    if not torch.cuda.is_available():
        raise RuntimeError('Run full-frame inference in a GPU allocation, not on the login node')
    started = time.monotonic()
    checkpoint = Path(item['output'])/'latest.pt'
    checkpoint_hash = fingerprint(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not state['complete']:
        raise ValueError('Resume and complete training before full-frame inference')
    cfg = state['config']
    if cfg != item['config']:
        raise ValueError('Checkpoint configuration does not match this plan entry')
    torch.set_num_threads(cfg['cpus'])
    root = Path(cfg['data_root'])
    data_hash = fingerprint(root/'manifest.json')
    if data_hash != state['data_hash']:
        raise ValueError('Dataset manifest changed since training')
    manifest = json.loads((root/'manifest.json').read_text())
    members = manifest['banks'][f'{split}.npy']['images']
    if image_ids:
        wanted = {str(int(value)).zfill(4) for value in image_ids}
        available = {Path(member).stem for member in members}
        if not wanted <= available:
            raise ValueError(f'Images outside the {split} split: {sorted(wanted-available)}')
        members = [member for member in members if Path(member).stem in wanted]
    elif limit:
        members = members[:limit]
    if not members:
        raise ValueError('No images selected')
    source = manifest['sources']['valid']
    archive_path = Path(archive_path) if archive_path else Path(source['path'])
    if not archive_path.is_file():
        raise FileNotFoundError(f'{archive_path}: use --archive to locate the original validation ZIP')
    print('Verifying original DIV2K archive:', archive_path, flush=True)
    if fingerprint(archive_path) != source['sha256']:
        raise ValueError('Original DIV2K archive checksum differs from the prepared dataset')
    destination = output_root/cfg['variant']/f'seed{cfg["seed"]}'
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f'{destination} is not empty; choose a fresh --output-root for another attempt')
    request = dict(variant=cfg['variant'], seed=cfg['seed'], split=split, precision=precision,
                   images=members, checkpoint_sha256=checkpoint_hash, data_hash=data_hash,
                   protocol='Full-frame, single forward per draw; no tiling/resizing; reflect-pad to scale; '
                            'RGB clipped PSNR/SSIM on original image extent; no border shave; no LPIPS',
                   latent_seed='20261015 + 2*numeric_image_id + draw; MSE uses zeros',
                   lr_consistency_extent='Padded canvas before removing bottom/right padding',
                   environment=provenance())
    atomic_json(destination/'request.json', request)
    gen = Generator(cfg['width'], cfg['scale'], cfg['projection']).cuda().eval()
    gen.load_state_dict(state['ema'])
    del state
    torch.cuda.reset_peak_memory_stats()
    results, sections = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for member in tqdm(members, desc=f'full {cfg["variant"]}/seed{cfg["seed"]}', unit='image', ascii=True):
            image_id = Path(member).stem
            with archive.open(member) as stream, Image.open(stream) as image:
                pixels = np.asarray(image.convert('RGB'))
            real, padded, condition, padding = prepare_frame(pixels, cfg['scale'])
            h, w = real.shape[-2:]
            folder = destination/image_id
            folder.mkdir()
            save_picture(folder/'hr.png', real)
            save_picture(folder/'lr.png', condition)
            nearest = F.interpolate(condition, size=padded.shape[-2:], mode='nearest')
            bicubic = F.interpolate(condition, size=padded.shape[-2:], mode='bicubic', align_corners=False)
            projected = project(bicubic, condition, cfg['scale'])
            outputs = {'bicubic': bicubic[..., :h, :w], 'projected_bicubic': projected[..., :h, :w]}
            save_picture(folder/'nearest.png', nearest[..., :h, :w])
            scores = {}
            for name, output in outputs.items():
                save_picture(folder/f'{name}.png', output)
                scores[name] = frame_scores(output, real)
            del outputs, nearest, bicubic, projected, output
            c_gpu = condition.cuda()
            draws, seconds = [], []
            consistency = []
            for draw in range(2):
                z = frame_latent(padded.shape, image_id, draw, cfg['method']=='mse').cuda()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                try:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=precision=='fp16'):
                        generated = gen(c_gpu, z).float()
                except torch.cuda.OutOfMemoryError as error:
                    raise RuntimeError('Full-frame inference exceeded GPU memory. Use a fresh --output-root '
                                       'with --precision fp16, or a GPU with more memory. '
                                       'Tiling is not equivalent with GroupNorm and is not used automatically.') from error
                torch.cuda.synchronize()
                seconds.append(time.perf_counter()-tick)
                del z
                raw = generated.cpu()
                del generated
                consistency.append(dict(
                    raw=float((F.avg_pool2d(raw, cfg['scale'])-condition).square().mean()),
                    clipped=float((F.avg_pool2d(raw.clamp(-1, 1), cfg['scale'])-condition).square().mean())))
                output = raw[..., :h, :w].contiguous()
                name = f'draw{draw+1}'
                save_picture(folder/f'{name}.png', output)
                scores[name] = frame_scores(output, real)
                draws.append(output)
                del raw
            del c_gpu
            row = dict(image=image_id, width=w, height=h, padding=padding, metrics=scores,
                       lr_mse_on_padded_canvas=consistency, forward_seconds=seconds,
                       diversity_mse_raw=float((draws[0]-draws[1]).square().mean()))
            atomic_json(folder/'metrics.json', row)
            results.append(row)
            cards = [(label, folder/f'{name}.png') for label, name in [
                ('LR enlarged (nearest)', 'nearest'), ('Bicubic', 'bicubic'),
                ('Projected bicubic', 'projected_bicubic'), ('HR target', 'hr'),
                ('Draw 1', 'draw1'), ('Draw 2', 'draw2')]]
            sections.append((f'{image_id}: {w} x {h}', cards))
            print(f'{image_id}: draw1 PSNR={scores["draw1"]["psnr"]:.3f}, '
                  f'SSIM={scores["draw1"]["ssim"]:.5f}; saved {folder}', flush=True)
            del draws, real, padded, condition, output
    means = {name: {key: float(np.mean([row['metrics'][name][key] for row in results]))
                    for key in ('psnr', 'ssim', 'out_of_range')}
             for name in ('bicubic', 'projected_bicubic', 'draw1', 'draw2')}
    with (destination/'images.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'width', 'height', 'draw1_psnr', 'draw1_ssim',
                                                    'draw2_psnr', 'draw2_ssim', 'diversity_mse_raw'])
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in ('image', 'width', 'height', 'diversity_mse_raw')} |
                            {f'{name}_{key}': row['metrics'][name][key]
                             for name in ('draw1', 'draw2') for key in ('psnr', 'ssim')})
    write_gallery(destination/'index.html', f'Full {split}: {cfg["variant"]}, seed {cfg["seed"]}', sections,
                  'Click a preview for its full-size PNG. No image resizing is used by the generator. '
                  'Full-frame scores are a separate protocol from the crop benchmark; no LPIPS is computed.')
    result = dict(**request, complete=True, results=results, means=means,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started)
    atomic_json(destination/'result.json', result)
    print('Full-frame inference complete:', destination/'index.html', flush=True)
    return result


def comparison_gallery(plan, output_root):
    """Collect only finished runs; never read or change training outputs."""
    runs, missing = [], []
    for item in plan['train']:
        cfg = item['config']
        run = output_root/cfg['variant']/f'seed{cfg["seed"]}'
        if not (run/'result.json').is_file():
            missing.append(f'{cfg["variant"]}/seed{cfg["seed"]}')
            continue
        result = json.loads((run/'result.json').read_text())
        if (not result['complete'] or result['variant'] != cfg['variant'] or result['seed'] != cfg['seed'] or
                [Path(member).stem for member in result['images']] != [row['image'] for row in result['results']]):
            raise ValueError(f'Unexpected run metadata: {run}')
        runs.append((run, result))
    if not runs:
        raise ValueError('No completed full-frame runs in this output root')
    if len({(r['data_hash'], r['split'], r['precision'], r['protocol']) for _, r in runs}) != 1:
        raise ValueError('Cannot combine different data/splits/precision/protocols in one comparison')
    sections = []
    image_ids = sorted({row['image'] for _, result in runs for row in result['results']})
    for image_id in image_ids:
        matching = [(run, result) for run, result in runs if any(r['image']==image_id for r in result['results'])]
        first, _ = matching[0]
        cards = [(label, first/image_id/f'{name}.png') for label, name in [
            ('HR target', 'hr'), ('Bicubic', 'bicubic'), ('Projected bicubic', 'projected_bicubic')]]
        for run, result in matching:
            row = next(r for r in result['results'] if r['image']==image_id)
            for draw in ('draw1', 'draw2'):
                score = row['metrics'][draw]
                caption = (f'{result["variant"]}/seed{result["seed"]} {draw}: '
                           f'PSNR {score["psnr"]:.2f}, SSIM {score["ssim"]:.4f}')
                cards.append((caption, run/image_id/f'{draw}.png'))
        sections.append((image_id, cards))
    notice = (f'{len(runs)}/{len(plan["train"])} completed full-frame runs. '
              'Click previews for full-size PNGs. Compare the same image across methods and both draws. '
              'These are full-frame PSNR/SSIM, separate from crop validation. Missing runs: '+
              (', '.join(missing) if missing else 'none')+'.')
    write_gallery(output_root/'index.html', 'MGM I2I — full-image comparison', sections, notice)
    print('Comparison gallery:', output_root/'index.html', flush=True)
