"""Prepared uint8 crop banks. Sample indices make worker prefetch resume-safe."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CropBank(Dataset):
    def __init__(self, path, crop=128, seed=0, training=True):
        self.path = str(Path(path))
        self.crop, self.seed, self.training = crop, seed, training
        self.array = None
        a = np.load(self.path, mmap_mode='r')
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3 or min(a.shape[1:3]) < crop:
            raise ValueError(f'Invalid crop bank: {self.path}, {a.shape}, {a.dtype}')
        self.count = len(a)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r')
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
            image = self.array[rng.integers(self.count)]
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
