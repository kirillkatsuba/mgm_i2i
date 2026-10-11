"""Build a self-contained archive; the working virtualenv is deliberately excluded."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'MGM_I2I_OFFLINE_MAC.zip'
EXCLUDED = {'.venv', '.git', 'tmp', '__pycache__', '.pytest_cache', '.DS_Store', '.agents', '.codex', '.aws'}
COMPRESSED = {'.pdf', '.zip', '.gz', '.whl', '.png', '.jpg', '.woff', '.woff2'}

if __name__ == '__main__':
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)
                   and p.name != OUTPUT.name and not p.name.endswith('.part')
                   and p.name != 'package-verification.json')
    with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            compression = zipfile.ZIP_STORED if path.suffix in COMPRESSED else zipfile.ZIP_DEFLATED
            archive.write(path, 'mgm_i2i/'+str(path.relative_to(ROOT)), compress_type=compression)
    with zipfile.ZipFile(OUTPUT) as archive:
        bad = archive.testzip()
        assert bad is None, bad
        members = set(archive.namelist())
        for path in ['offline/guide.html', 'offline/wheels/torch-2.6.0-cp313-none-macosx_11_0_arm64.whl',
                     'offline/torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth',
                     'data/cifar10/train.npy', 'scripts/train.py', 'requirements.lock']:
            assert 'mgm_i2i/'+path in members, path
    sha = hashlib.sha256()
    with OUTPUT.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            sha.update(block)
    report = dict(status='passed', file=OUTPUT.name, bytes=OUTPUT.stat().st_size,
                  sha256=sha.hexdigest(), files=len(paths), integrity='all ZIP CRCs checked',
                  restore='extract archive; run sh scripts/setup_offline.sh using this Mac Python 3.13')
    (ROOT/'offline/package-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
