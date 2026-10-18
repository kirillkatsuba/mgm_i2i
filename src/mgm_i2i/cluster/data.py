"""Online random train crops, fixed evaluation crops, and legacy loaders."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .crop_sampling import random_crop_coordinates

DATA_PROTOCOL = 'random128-v3'
MULTICROP_PROTOCOL = 'random128-multicrop-v4'
CENTER_PROTOCOL = 'center128-v2'


def load_multicrop_evaluation(root, split, expected_hash=None):
    """Verify a shared DIV2K evaluation bank independently of checkpoint training provenance."""
    root = Path(root)
    raw = (root/'manifest.json').read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and checksum != expected_hash:
        raise ValueError('Evaluation manifest changed after planning')
    manifest = json.loads(raw)
    preparation = manifest['preparation']
    count = preparation.get('eval_crops_per_image', 0)
    if (split not in ('val', 'test') or preparation.get('protocol') != MULTICROP_PROTOCOL or
            preparation.get('bank_size') != 128 or count < 1 or
            preparation.get('eval_crop_policy') != 'fixed_random_without_replacement'):
        raise ValueError('Expected a fixed-random DIV2K multicrop evaluation bank')
    bank = manifest['banks'][f'{split}.npy']
    path = root/f'{split}.npy'
    if hashlib.sha256(path.read_bytes()).hexdigest() != bank['sha256']:
        raise ValueError('Evaluation bank checksum mismatch')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    rows = bank['crops']
    ids = {r['image'] for r in rows}
    train_ids = {r['image'] for r in manifest['train_images']}
    other = 'test' if split == 'val' else 'val'
    other_ids = {r['image'] for r in manifest['banks'][f'{other}.npy']['crops']}
    if (array.dtype != np.uint8 or array.shape != (50 * count, 128, 128, 3) or
            len(rows) != 50 * count or len(ids) != 50 or len(train_ids) != 800 or
            ids & (train_ids | other_ids) or train_ids & other_ids or len(other_ids) != 50):
        raise ValueError('Expected image-disjoint 800/50/50 DIV2K splits and matching crop bank shape')
    if any(len([r for r in rows if r['image'] == name]) != count or
           len({(r['top'],r['left']) for r in rows if r['image'] == name}) != count for name in ids):
        raise ValueError('Expected the same number of distinct crops per evaluation photograph')
    return manifest, checksum


def validate_data_protocol(root, config):
    """Reject legacy banks for new plans, without changing old run semantics."""
    protocol = config.get('data_protocol')
    if protocol is None:
        return
    if protocol not in (DATA_PROTOCOL, MULTICROP_PROTOCOL, CENTER_PROTOCOL):
        raise ValueError(f'Unsupported data protocol: {protocol}')
    manifest = json.loads((Path(root)/'manifest.json').read_text())
    preparation = manifest['preparation']
    if (preparation.get('protocol') != protocol or preparation.get('bank_size') != 128
            or preparation.get('patches_per_image') != 1
            or config.get('crop') != 128 or config.get('eval_crop') != 128):
        raise ValueError(f'{protocol} requires matching preparation and crop=eval_crop=128; prepare fresh data')
    if protocol in (DATA_PROTOCOL, MULTICROP_PROTOCOL) and not manifest.get('train_images'):
        raise ValueError('random128-v3 requires original full train images; prepare fresh data')
    if protocol == MULTICROP_PROTOCOL:
        count = config.get('eval_crops_per_image', 5)
        if (count < 1 or preparation.get('eval_crops_per_image') != count or
                preparation.get('eval_crop_policy') != 'fixed_random_without_replacement'):
            raise ValueError('Multicrop evaluation policy differs; prepare fresh data')


class RandomImageCrops(Dataset):
    """Legacy uniform draws or balanced multicrop epochs keyed by seed/index.

    Full RGB images are stored as individual mmap-able uint8 NPY files.
    Only the selected crop is copied into RAM; no per-step PNG decoding.
    Stateless sampling survives worker prefetch and checkpoint resume.
    """
    def __init__(self, root, crop=128, seed=0, crops_per_image=None):
        self.root, self.crop, self.seed = Path(root), crop, seed
        manifest = json.loads((self.root/'manifest.json').read_text())
        self.multicrop = manifest['preparation'].get('protocol') == MULTICROP_PROTOCOL
        if manifest['preparation'].get('protocol') not in (DATA_PROTOCOL, MULTICROP_PROTOCOL) or crop != 128:
            raise ValueError('RandomImageCrops requires random128-v3 and crop=128')
        self.crops_per_image = (4 if self.multicrop else 1) if crops_per_image is None else crops_per_image
        if (not isinstance(self.crops_per_image, int) or self.crops_per_image < 1 or
                (not self.multicrop and self.crops_per_image != 1)):
            raise ValueError('Multiple crops per image require the multicrop protocol and a positive integer')
        self._permutation_key, self._permutation = None, None
        self.images = manifest.get('train_images', [])
        if not self.images:
            raise ValueError('No original train images in manifest')

    def __len__(self):
        return len(self.images) * self.crops_per_image

    def sample_identity(self, index):
        """Each new-protocol epoch visits every photograph exactly K times."""
        if index < 0 or not self.multicrop:
            raise ValueError('Balanced sample identity requires multicrop protocol and nonnegative index')
        epoch, position = divmod(int(index), len(self))
        cycle, offset = divmod(position, len(self.images))
        key = (epoch, cycle)
        if self._permutation_key != key:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, cycle, 100]))
            self._permutation = rng.permutation(len(self.images))
            self._permutation_key = key
        return int(self._permutation[offset]), epoch, cycle

    def __getitem__(self, index):
        if self.multicrop:
            image_index, epoch, cycle = self.sample_identity(index)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
            image_index = int(rng.integers(len(self.images)))
        entry = self.images[image_index]
        image = np.load(self.root/entry['path'], mmap_mode='r')
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or min(image.shape[:2]) < self.crop):
            raise ValueError(f'Invalid full train image: {entry["path"]}')
        if self.multicrop:
            top, left = random_crop_coordinates(*image.shape[:2], self.crop, self.crops_per_image,
                                               [self.seed, epoch, image_index, 101])[cycle]
        else:
            top = int(rng.integers(image.shape[0] - self.crop + 1))
            left = int(rng.integers(image.shape[1] - self.crop + 1))
        patch = image[top:top+self.crop, left:left+self.crop]
        return torch.from_numpy(patch.transpose(2, 0, 1).copy())


def training_dataset(root, config):
    if config.get('data_protocol') in (DATA_PROTOCOL, MULTICROP_PROTOCOL):
        return RandomImageCrops(root, crop=config['crop'], seed=config['seed'],
                                crops_per_image=config.get('train_crops_per_image'))
    return CropBank(Path(root)/'train.npy', crop=config['crop'], seed=config['seed'],
                    mode='center' if config.get('data_protocol') == CENTER_PROTOCOL else 'legacy')


class CropBank(Dataset):
    def __init__(self, path, crop=128, seed=0, training=True, mode='legacy'):
        self.path = str(Path(path))
        if mode not in ('legacy', 'center', 'fixed'):
            raise ValueError(f'Unsupported crop mode: {mode}')
        self.crop, self.seed, self.training, self.mode = crop, seed, training, mode
        self.array = None
        a = np.load(self.path, mmap_mode='r')
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 or min(a.shape[1:3]) < crop:
            raise ValueError(f'Invalid crop bank: {self.path}, {a.shape}, {a.dtype}')
        if mode in ('center', 'fixed') and (crop != 128 or a.shape[1:3] != (128, 128)):
            raise ValueError('Central mode requires a bank of original central 128x128 crops')
        self.count = len(a)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r')
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
            image = self.array[rng.integers(self.count)]
            if self.mode == 'legacy':
                top = rng.integers(image.shape[0] - self.crop + 1)
                left = rng.integers(image.shape[1] - self.crop + 1)
                image = image[top:top+self.crop, left:left+self.crop]
                image = np.rot90(image, int(rng.integers(4)))
                if rng.integers(2):
                    image = image[:, ::-1]
        else:
            image = self.array[index]
            top, left = (image.shape[0]-self.crop)//2, (image.shape[1]-self.crop)//2
            image = image[top:top+self.crop, left:left+self.crop]
        return torch.from_numpy(np.array(image.transpose(2, 0, 1), copy=True))


class SampleRange(Sampler):
    def __init__(self, start, end):
        self.start, self.end = start, end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start
