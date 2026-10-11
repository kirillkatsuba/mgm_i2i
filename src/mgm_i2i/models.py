import math
import torch
from torch import nn
from torch.nn import functional as F


def upsample(condition, size):
    return F.interpolate(condition, size=size, mode='bicubic', align_corners=False)


def project_average(output, condition):
    """Exact consistency for our average-pool x2 degradation, before clipping."""
    error = condition - F.avg_pool2d(output, 2)
    return output + F.interpolate(error, scale_factor=2, mode='nearest')


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.layers(x)


class ImageGenerator(nn.Module):
    def __init__(self, width=32, blocks=3, project=False):
        super().__init__()
        self.project = project
        self.net = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(),
                                 *[ResidualBlock(width) for _ in range(blocks)],
                                 nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, c, z):
        base = upsample(c, z.shape[-2:])
        # A bounded residual keeps all required moments finite.
        y = base + 0.5 * self.net(torch.cat([base, z], dim=1)).tanh()
        return project_average(y, c) if self.project else y


class ImageField(nn.Module):
    def __init__(self, width=32, blocks=3, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.net = nn.Sequential(nn.Conv2d(10, width, 3, padding=1), nn.SiLU(),
                                 *[ResidualBlock(width) for _ in range(blocks)],
                                 nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, c, t, x):
        base = upsample(c, x.shape[-2:]) if self.conditioned else torch.zeros_like(x)
        times = torch.cat([t, t.square(), (2 * math.pi * t).sin(), (2 * math.pi * t).cos()], 1)
        times = times.expand(-1, -1, *x.shape[-2:])
        return self.net(torch.cat([base, x, times], dim=1))


class ImageDiscriminator(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv2d(width * 2, width * 2, 4, stride=2, padding=1), nn.LeakyReLU(.2),
                                 nn.Conv2d(width * 2, 1, 3, padding=1))

    def forward(self, c, y):
        return self.net(torch.cat([upsample(c, y.shape[-2:]), y], 1))


class ToyGenerator(nn.Module):
    def __init__(self, width=64, project=False, **unused):
        super().__init__()
        self.project = project
        self.net = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 2))

    def forward(self, c, z):
        residual = 0.7 * self.net(torch.cat([c, z], 1)).tanh()
        if self.project:
            residual = residual - residual.mean(1, keepdim=True)
        return c + residual


class ToyField(nn.Module):
    def __init__(self, width=64, conditioned=True, **unused):
        super().__init__()
        self.conditioned = conditioned
        self.net = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 2))

    def forward(self, c, t, x):
        return self.net(torch.cat([c if self.conditioned else torch.zeros_like(c), t, x], 1))


class ToyDiscriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.LeakyReLU(.2), nn.Linear(width, width),
                                 nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, c, y):
        return self.net(torch.cat([c, y], 1))
