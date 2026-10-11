"""Download official DIV2K HR and prepare a reproducible crop bank on the cluster."""
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
    config = dict(bank_size=args.bank_size, patches_per_image=args.patches_per_image,
                  seed=args.seed, protocol='DIV2K HR average-pool synthetic SR; image-level split v1')
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['preparation'] != config:
            raise SystemExit('Preparation settings changed. Use a fresh --data-root.')
        for name, info in existing['banks'].items():
            if digest(root/name) != info['sha256']:
                raise SystemExit(f'Bank checksum mismatch: {name}; use a fresh data root.')
        print('Existing banks verified:', root)
    else:
        archives = {}
        names = {}
        for split, url in SOURCES.items():
            archive = root/'downloads'/f'DIV2K_{split}_HR.zip'
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
        banks = {}
        for split, (origin, members) in selected.items():
            n_crops = args.patches_per_image if split == 'train' else 5
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
                        coords = [(int(rng.integers(h-p+1)), int(rng.integers(w-p+1))) for _ in range(n_crops)] if split == 'train' else [
                            (0, 0), (0, w-p), (h-p, 0), (h-p, w-p), ((h-p)//2, (w-p)//2)]
                        for j, (top, left) in enumerate(coords):
                            out[i*n_crops+j] = np.asarray(image.crop((left, top, left+p, top+p)))
                            rows.append(dict(image=Path(member).name, top=top, left=left))
                    if (i+1) % 50 == 0:
                        print(split, i+1, '/', len(members), flush=True)
            out.flush()
            del out
            temp.replace(target)
            banks[target.name] = dict(sha256=digest(target), images=members, crops=rows)
        manifest_path.write_text(json.dumps(dict(preparation=config, sources=archives, banks=banks), indent=2))
    weights = args.weights_root.resolve()/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    download('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', weights)
    if not digest(weights).startswith('7be5be79'):
        raise ValueError('AlexNet SHA-256 mismatch. Remove the corrupt weights file and retry.')
    print('Ready. Training/evaluation requires no network. Banks:', root, 'weights:', args.weights_root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/div2k'))
    p.add_argument('--weights-root', type=Path, default=Path('weights'))
    p.add_argument('--bank-size', type=int, default=256)
    p.add_argument('--patches-per-image', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if args.bank_size < 32 or args.bank_size % 4 or args.patches_per_image < 1:
        p.error('bank-size must be >=32 and divisible by 4; patches-per-image must be positive')
    prepare(args)
