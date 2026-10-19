"""Download the pinned author sr4x-pool checkpoint and export its original EMA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.official_i2sb import (ARCHITECTURE, FILES, FOLDER_ID, PARAMETERS,
    SOURCE_HASHES, UPSTREAM_COMMIT, digest, ema_weights, load_bundle, read_options,
    reference_code, validate_options, verify_upstream)


def prepare(upstream, weights_root, download=False):
    upstream, weights_root = upstream.resolve(), weights_root.resolve()
    if not upstream.exists():
        if not download:
            raise FileNotFoundError('Missing author code; repeat with --download')
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/NVlabs/I2SB.git', str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
    verify_upstream(upstream)
    manifest_path = weights_root/'manifest.json'
    if manifest_path.exists():
        manifest, weights = load_bundle(manifest_path)
        print('Existing original EMA verified:', weights)
        return manifest
    raw = weights_root/'author_download'
    raw.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        path = raw/name
        if path.exists():
            continue
        if not download:
            raise FileNotFoundError(f'{path}; repeat with --download or place the official files there')
        try:
            import gdown
        except ImportError as error:
            raise RuntimeError('Install requirements-i2sb-official.txt on the login node') from error
        partial = path.with_suffix(path.suffix+'.part')
        result = gdown.download(id=file_id, output=str(partial), use_cookies=False, resume=True)
        if result is None:
            raise RuntimeError(f'Official Google Drive download failed: {name}')
        partial.replace(path)
    options = read_options(raw/'options.pkl')
    schedule = validate_options(options)
    cls, _, _ = reference_code(upstream)
    import torch
    torch.set_num_threads(4)
    with torch.device('meta'):
        net = cls(**ARCHITECTURE)
    if sum(p.numel() for p in net.parameters()) != PARAMETERS:
        raise ValueError('Author ADM parameter count changed')
    print('Reading original checkpoint (CPU mmap); exporting EMA only.', flush=True)
    checkpoint = torch.load(raw/'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    ema = ema_weights(net, checkpoint)
    path = weights_root/'official_ema.pt'
    partial = path.with_suffix('.partial.pt')
    torch.save(ema, partial)
    partial.replace(path)
    manifest = dict(version=1, checkpoint='sr4x-pool', upstream_commit=UPSTREAM_COMMIT,
                    source_folder='https://drive.google.com/drive/folders/'+FOLDER_ID,
                    source_file_ids=FILES, source_hashes=SOURCE_HASHES, parameters=PARAMETERS,
                    architecture=ARCHITECTURE, schedule=schedule, ema_file=path.name,
                    ema_sha256=digest(path), source_checkpoint_sha256=digest(raw/'latest.pt'),
                    source_options_sha256=digest(raw/'options.pkl'),
                    author_options={k: v if isinstance(v,(str,int,float,bool,type(None))) else str(v)
                                    for k,v in options.items()},
                    export='Original torch-ema shadow_params in original named_parameters order; no training')
    temp = manifest_path.with_suffix('.partial.json')
    temp.write_text(json.dumps(manifest,indent=2)+'\n')
    temp.replace(manifest_path)
    print('Prepared:',manifest_path,flush=True)
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=Path('third_party/I2SB-official'))
    parser.add_argument('--weights-root',type=Path,default=Path('weights/i2sb_official_sr4x_pool'))
    parser.add_argument('--download',action='store_true',help='Download public author code/checkpoint on the login node')
    args=parser.parse_args()
    prepare(args.upstream,args.weights_root,args.download)
