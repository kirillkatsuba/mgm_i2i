"""Fully convolutional, single-pass models; no pretrained generator required."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def project(y, c, scale):
    return y + F.interpolate(c - F.avg_pool2d(y, scale), size=y.shape[-2:], mode='nearest')


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.net = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.GroupNorm(8, cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))

    def forward(self, x):
        return self.skip(x) + .2 * self.net(x)


class UNet(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('width must be a positive multiple of 8')
        self.stem = nn.Conv2d(cin, width, 3, padding=1)
        self.e0 = nn.Sequential(Block(width, width), Block(width, width))
        self.d0 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.e1 = nn.Sequential(Block(width * 2, width * 2), Block(width * 2, width * 2))
        self.d1 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(Block(width * 4, width * 4), Block(width * 4, width * 4))
        self.u1 = Block(width * 6, width * 2)
        self.u0 = Block(width * 3, width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        a = self.e0(self.stem(x))
        b = self.e1(self.d0(a))
        h = self.mid(self.d1(b))
        h = self.u1(torch.cat([F.interpolate(h, size=b.shape[-2:], mode='nearest'), b], 1))
        h = self.u0(torch.cat([F.interpolate(h, size=a.shape[-2:], mode='nearest'), a], 1))
        return self.out(h)


class Generator(nn.Module):
    def __init__(self, width=48, scale=4, consistency=True):
        super().__init__()
        self.scale, self.consistency = scale, consistency
        self.net = UNet(6, width)

    def forward(self, c, z):
        base = F.interpolate(c, size=z.shape[-2:], mode='bicubic', align_corners=False)
        y = base + self.net(torch.cat([base, z], 1)).float().tanh()
        return project(y, c, self.scale) if self.consistency else y


class Field(nn.Module):
    def __init__(self, width=48, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        self.net = UNet(10, width)

    def forward(self, c, t, x):
        base = F.interpolate(c, size=x.shape[-2:], mode='bicubic', align_corners=False)
        if not self.conditioned:
            base = torch.zeros_like(base)
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
