from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class ToyData:
    """Two HR pixels share the same LR average, with a known two-mode target."""
    def __init__(self, device, seed=0, **unused):
        self.device = device
        self.rng = torch.Generator().manual_seed(seed)

    def batch(self, size):
        c = torch.rand(size, 1, generator=self.rng) - .5
        s = torch.randint(0, 2, (size, 1), generator=self.rng) * 2 - 1
        y = c + .35 * torch.cat([s, -s], 1)
        return c.to(self.device), y.to(self.device)


class CIFARData:
    """Reads prepared uint8 arrays; no download and no external model dependencies."""
    def __init__(self, device, root='data/cifar10', split='train', limit=0, seed=123):
        self.device = device
        root = Path(root)
        path = root / ('test.npy' if split == 'test' else 'train.npy')
        if not path.exists():
            raise FileNotFoundError(f'{path}: run .venv/bin/python scripts/prepare_data.py (offline)')
        self.images = np.load(path, mmap_mode='r')
        order = np.random.default_rng(seed).permutation(len(self.images))
        if split == 'train':
            order = order[:45000]
        elif split == 'val':
            order = order[45000:]
        elif split != 'test':
            raise ValueError(split)
        self.indices = order[:limit] if limit else order
        if len(self.indices) == 0:
            raise ValueError('Empty dataset split')
        self.rng = torch.Generator().manual_seed(seed + {'train': 0, 'val': 1, 'test': 2}[split])

    def batch(self, size):
        ids = torch.randint(len(self.indices), (size,), generator=self.rng).numpy()
        return self.get(ids)

    def get(self, ids):
        image = np.array(self.images[self.indices[ids]], copy=True)
        hr = torch.from_numpy(image).to(self.device, dtype=torch.float32) / 127.5 - 1
        # Exactly specified, reproducible synthetic degradation; not bicubic SR.
        lr = F.avg_pool2d(hr, 2)
        return lr, hr


def latent(real, deterministic=False):
    shape = (real.shape[0], 2) if real.ndim == 2 else (real.shape[0], 1, *real.shape[-2:])
    return torch.zeros(shape, device=real.device) if deterministic else torch.randn(shape, device=real.device)
