"""Prepare pinned SinSR/VOSR code and weights on a login node, never inside Slurm."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.sr_reference import ASSETS, PINS, HF_REVISION, digest, load_bundle, verify_sources


def download_file(path, spec, download):
    def valid():
        return (path.is_file() and path.stat().st_size == spec['size'] and
                (spec['sha256'] is None or digest(path) == spec['sha256']))
    if valid():
        return
    if not download:
        raise FileNotFoundError(f'Missing or corrupt {path}; use --download on the login node')
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.part')
    start = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={start}-'} if start else {}
    print(f'Downloading {path.name} ({spec["size"]/1e9:.2f} GB)', flush=True)
    with requests.get(spec['url'], headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        append = start and response.status_code == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
            raise ValueError('Unexpected download range')
        with partial.open('ab' if append else 'wb') as stream:
            for block in response.iter_content(1 << 20):
                stream.write(block)
    if partial.stat().st_size != spec['size'] or (spec['sha256'] and digest(partial) != spec['sha256']):
        raise ValueError(f'Download size/checksum mismatch: {partial}; remove it before retrying')
    partial.replace(path)


def prepare(method, third_party, weights_root, download=False):
    names = ['sinsr'] if method == 'sinsr' else ['vosr', 'dino']
    roots = {name: third_party/PINS[name]['directory'] for name in names}
    for name, root in roots.items():
        pin = PINS[name]
        if not root.exists():
            if not download:
                raise FileNotFoundError(f'Missing {root}; use --download')
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', pin['url'], str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'checkout', '--detach', pin['commit']], check=True)
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if head != pin['commit']:
            raise ValueError(f'{root} must be at {pin["commit"]}; existing source is never reset automatically')
    verify_sources(roots, method)
    folder = weights_root/('sinsr_v1' if method == 'sinsr' else 'vosr_05b_os')
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        load_bundle(manifest_path, method)
        print('Existing reference bundle verified:', manifest_path)
        return
    for name, spec in ASSETS[method].items():
        download_file(folder/name, spec, download)
    checkpoint = 'SinSR_v1.pth' if method == 'sinsr' else 'VOSR_0.5B_os/checkpoints/ema_model.safetensors'
    manifest = dict(version=1, method=method, hf_revision=HF_REVISION, checkpoint=checkpoint,
                    files={name: digest(folder/name) for name in ASSETS[method]},
                    source_pins={name: PINS[name] for name in names},
                    external_pretraining=('Author SinSR v1 / pretrained ResShift + VQ autoencoder'
                                          if method == 'sinsr' else
                                          'Author VOSR-0.5B-os / large web corpus + teacher distillation; pretrained SD2 VAE and DINOv2-B'),
                    checksum_note='VOSR uses publisher LFS SHA256; GitHub SinSR files are pinned by URL/size, '
                                  'then local SHA256 is recorded on first acquisition')
    folder.mkdir(parents=True, exist_ok=True)
    partial = manifest_path.with_suffix('.partial.json')
    partial.write_text(json.dumps(manifest, indent=2)+'\n')
    partial.replace(manifest_path)
    print('Prepared:', manifest_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', nargs='+', choices=['sinsr', 'vosr'], default=['sinsr', 'vosr'])
    p.add_argument('--third-party', type=Path, default=ROOT/'third_party')
    p.add_argument('--weights-root', type=Path, default=ROOT/'weights/sr_references')
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    for method in dict.fromkeys(args.models):
        prepare(method, args.third_party.resolve(), args.weights_root.resolve(), args.download)


if __name__ == '__main__':
    main()
