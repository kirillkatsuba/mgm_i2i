"""Pinned BSD100 HR files and a common center128 / average-pool x4 evaluation bank."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

PROTOCOL = 'bsd100-center128-pool4-v1'
CATALOG = Path(__file__).resolve().parents[3]/'research/bsd100_source.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(path, entry):
    raw = Path(path).read_bytes()
    blob_hash = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if len(raw) != entry['size'] or blob_hash != entry['git_blob_sha1']:
        raise ValueError(f'BSD100 source checksum mismatch: {path}')


def load_bank(root, expected_hash=None):
    root = Path(root)
    manifest_hash = digest(root/'manifest.json')
    if expected_hash is not None and manifest_hash != expected_hash:
        raise ValueError('External evaluation manifest changed after planning')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['preparation'] != dict(protocol=PROTOCOL, crop=128, scale=4):
        raise ValueError('Expected BSD100 center128 / average-pool x4 protocol')
    bank = manifest['banks']['external.npy']
    if digest(root/'external.npy') != bank['sha256']:
        raise ValueError('External evaluation bank checksum mismatch')
    array = np.load(root/'external.npy', mmap_mode='r', allow_pickle=False)
    rows = bank['crops']
    if array.dtype != np.uint8 or array.shape != (100, 128, 128, 3):
        raise ValueError('Expected exactly 100 RGB uint8 center128 images')
    if len(rows) != 100 or len({r['image'] for r in rows}) != 100:
        raise ValueError('Expected 100 unique source images')
    return manifest, manifest_hash


def prepare(root, workers=4):
    root = Path(root)
    catalog = json.loads(CATALOG.read_text())
    if len(catalog['files']) != 100:
        raise ValueError('Expected 100 pinned BSD100 source files')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest.json').exists():
        manifest, _ = load_bank(root)
        if manifest['source_catalog_sha256'] != digest(CATALOG):
            raise ValueError('BSD100 source catalog changed; use a fresh output directory')
        return manifest
    downloads = root/'downloads'
    downloads.mkdir(exist_ok=True)

    def fetch(entry):
        target = downloads/Path(entry['path']).name
        if not target.exists():
            url = f'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/{catalog["commit"]}/{entry["path"]}'
            partial = target.with_suffix('.part')
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--connect-timeout', '30', '--max-time', '180', '--retry', '3',
                            '--output', str(partial), url], check=True)
            verify_source(partial, entry)
            partial.replace(target)
        verify_source(target, entry)
        return target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sources = list(pool.map(fetch, catalog['files']))
    patches, rows = [], []
    for path, entry in zip(sources, catalog['files']):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        if min(height, width) < 128:
            raise ValueError(f'Image smaller than crop: {path}')
        top, left = (height-128)//2, (width-128)//2
        patches.append(rgb[top:top+128, left:left+128].copy())
        rows.append(dict(image=path.name, top=top, left=left, source_shape=list(rgb.shape),
                         source_sha256=digest(path), git_blob_sha1=entry['git_blob_sha1']))
    temporary = root/'external.partial.npy'
    np.save(temporary, np.stack(patches), allow_pickle=False)
    temporary.replace(root/'external.npy')
    manifest = dict(preparation=dict(protocol=PROTOCOL, crop=128, scale=4),
                    dataset='BSD100', source_repository=catalog['repository'],
                    source_commit=catalog['commit'], source_catalog_sha256=digest(CATALOG),
                    banks={'external.npy': dict(sha256=digest(root/'external.npy'), crops=rows)},
                    note='One native center128 crop per source HR PNG. RGB conversion, no resize. '
                         'LR is computed by exact average pooling x4 during inference. '
                         'This is not the standard full-frame bicubic BSD100 benchmark.')
    temporary = root/'manifest.partial.json'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(root/'manifest.json')
    load_bank(root)
    return manifest
