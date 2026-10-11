"""Convert the downloaded, checksum-verified official archive to offline arrays."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    archive = ROOT / 'data/cifar-10-python.tar.gz'
    if not archive.exists():
        raise SystemExit('CIFAR archive missing. Download before departure using scripts/download_materials.py.')
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != 'c58f30108f718f92721af3b95e74349a':
        raise SystemExit('Official CIFAR checksum mismatch; refusing to read pickle.')
    output = ROOT / 'data/cifar10'
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        def read(name):
            # No filesystem extraction or execution of third-party source code.
            member = tar.extractfile('cifar-10-batches-py/' + name)
            value = pickle.load(io.BytesIO(member.read()), encoding='bytes')
            return value[b'data'].reshape(-1, 3, 32, 32), np.asarray(value[b'labels'])
        batches = [read(f'data_batch_{i}') for i in range(1, 6)]
        test, labels = read('test_batch')
    np.save(output / 'train.npy', np.concatenate([b[0] for b in batches]))
    np.save(output / 'train_labels.npy', np.concatenate([b[1] for b in batches]))
    np.save(output / 'test.npy', test)
    np.save(output / 'test_labels.npy', labels)
    (output / 'provenance.json').write_text(json.dumps({
        'source': 'https://cave.cs.toronto.edu/kriz/cifar.html', 'archive_md5': digest,
        'train': 50000, 'test': 10000, 'shape': [3, 32, 32], 'dtype': 'uint8',
        'split_rule': 'NumPy default_rng(data_seed).permutation(50000): first 45000 train; last 5000 val',
        'degradation': 'nonoverlapping 2x2 average pooling',
    }, indent=2))
    print(f'Prepared 50,000 training and 10,000 test images in {output}')


if __name__ == '__main__':
    prepare()
