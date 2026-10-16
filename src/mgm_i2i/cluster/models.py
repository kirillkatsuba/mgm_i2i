"""Fully convolutional, single-pass models; no pretrained generator required."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

ARCHITECTURE_VERSION = 2
DEFAULT_WIDTH = 120


def get_architecture_version(config):
    """Unversioned checkpoints used the original bounded-residual models."""
    version = config.get('architecture_version', 1)
    if version not in (1, ARCHITECTURE_VERSION):
        raise ValueError(f'Unsupported architecture version: {version}')
    return version


class TimeEmbedding(nn.Module):
    """Diffusion-style sinusoidal features followed by a learned MLP.

    MGM time remains in [0, .5]; scaling to [0, 500] exposes multiple
    frequencies without changing the interpolant or revealing its flip.
    """
    def __init__(self, width):
        super().__init__()
        frequencies = torch.exp(-math.log(10000.) * torch.arange(width // 2) / (width // 2))
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                 nn.Linear(4 * width, 4 * width))

    def forward(self, t):
        times = t.reshape(t.shape[0], -1)
        if times.shape[1] != 1:
            raise ValueError('Expected one MGM time per sample')
        phases = 1000. * times.float() * self.frequencies[None, :]
        return self.mlp(torch.cat([phases.cos(), phases.sin()], dim=1))


def project(y, c, scale):
    return y + F.interpolate(c - F.avg_pool2d(y, scale), size=y.shape[-2:], mode='nearest')


class Block(nn.Module):
    def __init__(self, cin, cout, time_dim=None):
        super().__init__()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.net = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.time_projection = nn.Linear(time_dim, cout) if time_dim is not None else None

    def forward(self, x, embedding=None):
        if self.time_projection is None:
            return self.skip(x) + .2 * self.net(x)
        if embedding is None:
            raise ValueError('Time-conditioned block requires an embedding')
        h = self.net[2](self.net[1](self.net[0](x)))
        h = h + self.time_projection(F.silu(embedding))[:, :, None, None]
        h = self.net[5](self.net[4](self.net[3](h)))
        return self.skip(x) + .2 * h


class UNet(nn.Module):
    def __init__(self, cin, width, time_dim=None):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('width must be a positive multiple of 8')
        self.stem = nn.Conv2d(cin, width, 3, padding=1)
        self.e0 = nn.Sequential(Block(width, width, time_dim), Block(width, width, time_dim))
        self.d0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.e1 = nn.Sequential(Block(width * 2, width * 2, time_dim), Block(width * 2, width * 2, time_dim))
        self.d1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(Block(width * 4, width * 4, time_dim), Block(width * 4, width * 4, time_dim))
        self.u1 = Block(width * 6, width * 2, time_dim)
        self.u0 = Block(width * 3, width, time_dim)
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def stage(blocks, x, embedding):
        for block in blocks:
            x = block(x, embedding)
        return x

    def forward(self, x, embedding=None):
        a = self.stage(self.e0, self.stem(x), embedding)
        b = self.stage(self.e1, self.d0(a), embedding)
        h = self.stage(self.mid, self.d1(b), embedding)
        h = self.u1(torch.cat([F.interpolate(h, size=b.shape[-2:], mode='nearest'), b], 1), embedding)
        h = self.u0(torch.cat([F.interpolate(h, size=a.shape[-2:], mode='nearest'), a], 1), embedding)
        return self.out(h)


class Generator(nn.Module):
    def __init__(self, width=DEFAULT_WIDTH, scale=4, consistency=True, architecture_version=ARCHITECTURE_VERSION):
        super().__init__()
        self.architecture_version = get_architecture_version({'architecture_version': architecture_version})
        self.scale, self.consistency = scale, consistency
        self.net = UNet(6, width)

    def forward(self, c, z):
        base = F.interpolate(c, size=z.shape[-2:], mode='bicubic', align_corners=False)
        residual = self.net(torch.cat([base, z], 1)).float()
        if self.architecture_version == 1:
            residual = residual.tanh()
        y = base + residual
        return project(y, c, self.scale) if self.consistency else y


class Field(nn.Module):
    def __init__(self, width=DEFAULT_WIDTH, conditioned=True, architecture_version=ARCHITECTURE_VERSION):
        super().__init__()
        self.architecture_version = get_architecture_version({'architecture_version': architecture_version})
        self.conditioned = conditioned
        if self.architecture_version == 1:
            self.net = UNet(10, width)
        else:
            self.net = UNet(6, width, time_dim=4 * width)
            self.time_embedding = TimeEmbedding(width)

    def forward(self, c, t, x):
        base = F.interpolate(c, size=x.shape[-2:], mode='bicubic', align_corners=False)
        if not self.conditioned:
            base = torch.zeros_like(base)
        if self.architecture_version == ARCHITECTURE_VERSION:
            return self.net(torch.cat([base, x], 1), self.time_embedding(t)).float()
        time = torch.cat([t, t*t, (2*torch.pi*t).sin(), (2*torch.pi*t).cos()], 1)
        return self.net(torch.cat([base, x, time.expand(-1, -1, *x.shape[-2:])], 1)).float()


class Discriminator(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        layers = []
        cin = 6
        for cout in [width, width*2, width*4]:
            layers.extend([spectral_norm(nn.Conv2d(cin, cout, 4, stride=2, padding=1)), nn.LeakyReLU(.2)])
            cin = cout
        layers.append(spectral_norm(nn.Conv2d(cin, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, c, y):
        base = F.interpolate(c, size=y.shape[-2:], mode='bicubic', align_corners=False)
        return self.net(torch.cat([base, y], 1)).float()
