"""Deterministic crop coordinates shared by offline evaluation and online training."""
import numpy as np


def random_crop_coordinates(height, width, size, count, seed_parts):
    rows, columns = height - size + 1, width - size + 1
    if rows < 1 or columns < 1 or count < 1 or count > rows * columns:
        raise ValueError('Image must contain enough distinct crop positions for the requested crop count')
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    positions = rng.choice(rows * columns, size=count, replace=False)
    return [(int(position // columns), int(position % columns)) for position in positions]
