"""Prepare full DIV2K train images for online random128 and center128 eval.

The explicit legacy mode preserves preparation of the original patch banks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from PIL import Image

SOURCES = {
    'train': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
    'valid': 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '30', '--retry', '5', '--retry-delay', '5',
                    '--continue-at', '-', '--output', str(partial), url], check=True)
    partial.replace(path)


def prepare(args):
    root = args.data_root.resolve()
    protocols = {'random': 'random128-v3', 'center': 'center128-v2',
                 'legacy': 'DIV2K HR average-pool synthetic SR; image-level split v1'}
    config = dict(bank_size=args.bank_size, patches_per_image=args.patches_per_image,
                  seed=args.seed, protocol=protocols[args.crop_mode])
    if args.crop_mode != 'legacy' and (args.bank_size != 128 or args.patches_per_image != 1):
        raise ValueError('Random/central protocols require bank-size=128 and patches-per-image=1')
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['preparation'] != config:
            raise SystemExit('Preparation settings changed. Use a fresh --data-root.')
        for name, info in existing['banks'].items():
            if digest(root/name) != info['sha256']:
                raise SystemExit(f'Bank checksum mismatch: {name}; use a fresh data root.')
        for entry in existing.get('train_images', []):
            if digest(root/entry['path']) != entry['sha256']:
                raise SystemExit(f'Train image checksum mismatch: {entry["path"]}; use a fresh data root.')
        print('Existing banks verified:', root)
    else:
        archives = {}
        names = {}
        for split, url in SOURCES.items():
            archive_root = args.archive_root if args.archive_root is not None else root/'downloads'
            archive = archive_root.resolve()/f'DIV2K_{split}_HR.zip'
            download(url, archive)
            with zipfile.ZipFile(archive) as z:
                candidates = sorted(n for n in z.namelist() if n.lower().endswith('.png'))
                expected = range(1, 801) if split == 'train' else range(801, 901)
                if [Path(n).stem for n in candidates] != [f'{i:04d}' for i in expected]:
                    raise ValueError(f'Unexpected official archive contents: {archive}')
                # Every member read below is CRC-checked by ZipFile. No extraction.
                names[split] = candidates
            archives[split] = dict(path=str(archive), url=url, sha256=digest(archive))
        order = np.random.default_rng(args.seed).permutation(100)
        selected = {'train': ('train', names['train']),
                    'val': ('valid', [names['valid'][i] for i in sorted(order[:50])]),
                    'test': ('valid', [names['valid'][i] for i in sorted(order[50:])])}
        banks, train_images = {}, []
        for split, (origin, members) in selected.items():
            if split == 'train' and args.crop_mode == 'random':
                (root/'train_images').mkdir(exist_ok=True)
                with zipfile.ZipFile(archives[origin]['path']) as z:
                    for i, member in enumerate(members):
                        relative = f'train_images/{Path(member).stem}.npy'
                        target = root/relative
                        temp = target.with_suffix('.partial.npy')
                        with z.open(member) as f, Image.open(f) as image:
                            pixels = np.asarray(image.convert('RGB'))
                            if min(pixels.shape[:2]) < 128:
                                raise ValueError(f'{member} smaller than train crop 128')
                            with temp.open('wb') as stream:
                                np.save(stream, pixels, allow_pickle=False)
                        temp.replace(target)
                        train_images.append(dict(image=Path(member).name, path=relative,
                                                 shape=list(pixels.shape), sha256=digest(target)))
                        if (i+1) % 50 == 0:
                            print(split, i+1, '/', len(members), flush=True)
                continue
            n_crops = 1 if args.crop_mode != 'legacy' else (args.patches_per_image if split == 'train' else 5)
            target = root/f'{split}.npy'
            temp = root/f'{split}.partial.npy'
            out = np.lib.format.open_memmap(temp, mode='w+', dtype='uint8',
                                           shape=(len(members)*n_crops, args.bank_size, args.bank_size, 3))
            rows = []
            with zipfile.ZipFile(archives[origin]['path']) as z:
                for i, member in enumerate(members):
                    with z.open(member) as f, Image.open(f) as image:
                        image = image.convert('RGB')
                        w, h = image.size
                        p = args.bank_size
                        if min(w, h) < p:
                            raise ValueError(f'{member} smaller than bank size {p}')
                        rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(Path(member).stem)]))
                        if args.crop_mode != 'legacy':
                            coords = [((h-p)//2, (w-p)//2)]
                        elif split == 'train':
                            coords = [(int(rng.integers(h-p+1)), int(rng.integers(w-p+1))) for _ in range(n_crops)]
                        else:
                            coords = [(0, 0), (0, w-p), (h-p, 0), (h-p, w-p), ((h-p)//2, (w-p)//2)]
                        for j, (top, left) in enumerate(coords):
                            out[i*n_crops+j] = np.asarray(image.crop((left, top, left+p, top+p)))
                            rows.append(dict(image=Path(member).name, top=top, left=left))
                    if (i+1) % 50 == 0:
                        print(split, i+1, '/', len(members), flush=True)
            out.flush()
            del out
            temp.replace(target)
            banks[target.name] = dict(sha256=digest(target), images=members, crops=rows)
        manifest = dict(preparation=config, sources=archives, banks=banks)
        if args.crop_mode == 'random':
            manifest['train_images'] = train_images
        manifest_path.write_text(json.dumps(manifest, indent=2))
    weights = args.weights_root.resolve()/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    download('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', weights)
    if not digest(weights).startswith('7be5be79'):
        raise ValueError('AlexNet SHA-256 mismatch. Remove the corrupt weights file and retry.')
    if not getattr(args, 'skip_vgg', False):
        vgg = args.weights_root.resolve()/'hub/checkpoints/vgg19-dcbb9e9d.pth'
        download('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', vgg)
        if not digest(vgg).startswith('dcbb9e9d'):
            raise ValueError('VGG19 SHA-256 mismatch. Remove the corrupt weights file and retry.')
    print('Ready. Training/evaluation requires no network. Banks:', root, 'weights:', args.weights_root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=None)
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--archive-root', type=Path, help='Reuse existing official ZIPs without copying or redownloading')
    p.add_argument('--crop-mode', choices=['random', 'center', 'legacy'], default='random')
    p.add_argument('--bank-size', type=int, default=None)
    p.add_argument('--patches-per-image', type=int, default=None)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--skip-vgg', action='store_true', help='Skip VGG19 (only for runs without perceptual losses)')
    args = p.parse_args()
    if args.data_root is None:
        args.data_root = Path({'random': 'data/div2k_random128', 'center': 'data/div2k_center128',
                               'legacy': 'data/div2k'}[args.crop_mode])
    if args.bank_size is None:
        args.bank_size = 256 if args.crop_mode == 'legacy' else 128
    if args.patches_per_image is None:
        args.patches_per_image = 8 if args.crop_mode == 'legacy' else 1
    if args.bank_size < 32 or args.bank_size % 4 or args.patches_per_image < 1:
        p.error('bank-size must be >=32 and divisible by 4; patches-per-image must be positive')
    if args.crop_mode != 'legacy' and (args.bank_size != 128 or args.patches_per_image != 1):
        p.error('random/center modes require bank-size=128 and patches-per-image=1')
    prepare(args)
