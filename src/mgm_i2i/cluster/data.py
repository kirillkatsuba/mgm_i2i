"""Online random train crops, fixed evaluation crops, and legacy loaders."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DATA_PROTOCOL = 'random128-v3'
CENTER_PROTOCOL = 'center128-v2'


def validate_data_protocol(root, config):
    """Reject legacy banks for new plans, without changing old run semantics."""
    protocol = config.get('data_protocol')
    if protocol is None:
        return
    if protocol not in (DATA_PROTOCOL, CENTER_PROTOCOL):
        raise ValueError(f'Unsupported data protocol: {protocol}')
    manifest = json.loads((Path(root)/'manifest.json').read_text())
    preparation = manifest['preparation']
    if (preparation.get('protocol') != protocol or preparation.get('bank_size') != 128
            or preparation.get('patches_per_image') != 1
            or config.get('crop') != 128 or config.get('eval_crop') != 128):
        raise ValueError(f'{protocol} requires matching preparation and crop=eval_crop=128; prepare fresh data')
    if protocol == DATA_PROTOCOL and not manifest.get('train_images'):
        raise ValueError('random128-v3 requires original full train images; prepare fresh data')


class RandomImageCrops(Dataset):
    """Uniform image and crop selection keyed only by seed/sample index.

    Full RGB images are stored as individual mmap-able uint8 NPY files.
    Only the selected crop is copied into RAM; no per-step PNG decoding.
    Stateless sampling survives worker prefetch and checkpoint resume.
    """
    def __init__(self, root, crop=128, seed=0):
        self.root, self.crop, self.seed = Path(root), crop, seed
        manifest = json.loads((self.root/'manifest.json').read_text())
        if manifest['preparation'].get('protocol') != DATA_PROTOCOL or crop != 128:
            raise ValueError('RandomImageCrops requires random128-v3 and crop=128')
        self.images = manifest.get('train_images', [])
        if not self.images:
            raise ValueError('No original train images in manifest')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        entry = self.images[int(rng.integers(len(self.images)))]
        image = np.load(self.root/entry['path'], mmap_mode='r')
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or min(image.shape[:2]) < self.crop):
            raise ValueError(f'Invalid full train image: {entry["path"]}')
        top = int(rng.integers(image.shape[0] - self.crop + 1))
        left = int(rng.integers(image.shape[1] - self.crop + 1))
        patch = image[top:top+self.crop, left:left+self.crop]
        return torch.from_numpy(patch.transpose(2, 0, 1).copy())


def training_dataset(root, config):
    if config.get('data_protocol') == DATA_PROTOCOL:
        return RandomImageCrops(root, crop=config['crop'], seed=config['seed'])
    return CropBank(Path(root)/'train.npy', crop=config['crop'], seed=config['seed'],
                    mode='center' if config.get('data_protocol') == CENTER_PROTOCOL else 'legacy')


class CropBank(Dataset):
    def __init__(self, path, crop=128, seed=0, training=True, mode='legacy'):
        self.path = str(Path(path))
        if mode not in ('legacy', 'center'):
            raise ValueError(f'Unsupported crop mode: {mode}')
        self.crop, self.seed, self.training, self.mode = crop, seed, training, mode
        self.array = None
        a = np.load(self.path, mmap_mode='r')
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 or min(a.shape[1:3]) < crop:
            raise ValueError(f'Invalid crop bank: {self.path}, {a.shape}, {a.dtype}')
        if mode == 'center' and (crop != 128 or a.shape[1:3] != (128, 128)):
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
