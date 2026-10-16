"""Local SR baselines, implemented from the documented method equations.

Architecture specifications: XPixelGroup/BasicSR rrdbnet_arch.py and
VGGStyleDiscriminator. Bridge equations/schedule: NVlabs/I2SB diffusion.py
and runner.py. Weights are never downloaded by these models.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .models import DEFAULT_WIDTH, TimeEmbedding, UNet, Generator, get_architecture_version


class DenseResidual(nn.Module):
    def __init__(self, features, growth):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(features + i * growth, growth, 3, padding=1)
                                    for i in range(4)] +
                                   [nn.Conv2d(features + 4 * growth, features, 3, padding=1)])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            with torch.no_grad():
                conv.weight.mul_(.1)
                conv.bias.zero_()

    def forward(self, x):
        parts = [x]
        for conv in self.convs[:-1]:
            parts.append(F.leaky_relu(conv(torch.cat(parts, 1)), .2))
        return x + .2 * self.convs[-1](torch.cat(parts, 1))


class RRDB(nn.Module):
    def __init__(self, features, growth):
        super().__init__()
        self.blocks = nn.Sequential(*(DenseResidual(features, growth) for _ in range(3)))

    def forward(self, x):
        return x + .2 * self.blocks(x)


class RRDBNet(nn.Module):
    """ESRGAN x4; receives and returns RGB [0,1], without final clipping."""
    def __init__(self, features=64, blocks=23, growth=32):
        super().__init__()
        self.first = nn.Conv2d(3, features, 3, padding=1)
        self.body = nn.Sequential(*(RRDB(features, growth) for _ in range(blocks)))
        self.body_out = nn.Conv2d(features, features, 3, padding=1)
        self.up = nn.ModuleList(nn.Conv2d(features, features, 3, padding=1) for _ in range(2))
        self.hr = nn.Conv2d(features, features, 3, padding=1)
        self.out = nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, lr):
        h = self.first(lr)
        h = h + self.body_out(self.body(h))
        for conv in self.up:
            h = F.leaky_relu(conv(F.interpolate(h, scale_factor=2, mode='nearest')), .2)
        return self.out(F.leaky_relu(self.hr(h), .2))


class VGGDiscriminator(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        layers = [nn.Conv2d(3, features, 3, padding=1), nn.LeakyReLU(.2)]
        cin = features
        for i in range(5):
            cout = features * min(2 ** i, 8)
            if i:
                layers.extend([nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                               nn.BatchNorm2d(cout), nn.LeakyReLU(.2)])
            layers.extend([nn.Conv2d(cout, cout, 4, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.LeakyReLU(.2)])
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(features * 8 * 4 * 4, 100), nn.LeakyReLU(.2), nn.Linear(100, 1))

    def forward(self, image):
        if image.shape[-2:] != (128, 128):
            raise ValueError('ESRGAN discriminator requires HR128 training crops')
        return self.head(self.features(image).flatten(1))


class VGGPerceptual(nn.Module):
    """Frozen VGG19 through pre-ReLU conv5_4, loaded only from local weights."""
    def __init__(self, weights_path):
        super().__init__()
        layers, cin = [], 3
        for stage, (cout, repeats) in enumerate([(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]):
            for index in range(repeats):
                layers.append(nn.Conv2d(cin, cout, 3, padding=1))
                cin = cout
                if not (stage == 4 and index == repeats - 1):
                    layers.append(nn.ReLU())
            if stage < 4:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f'Prepare VGG19 weights before training: {path}')
        state = torch.load(path, map_location='cpu', weights_only=True)
        prefix = 'features.'
        selected = {key[len(prefix):]: value for key, value in state.items()
                    if key.startswith(prefix) and key[len(prefix):] in self.features.state_dict()}
        self.features.load_state_dict(selected, strict=True)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, prediction, target):
        # Both inputs in [0,1]. Preserve gradients on the prediction branch.
        a = self.features((prediction.float() - self.mean) / self.std)
        with torch.no_grad():
            b = self.features((target.float() - self.mean) / self.std)
        return F.l1_loss(a, b)


class BridgeDenoiser(nn.Module):
    def __init__(self, width=DEFAULT_WIDTH):
        super().__init__()
        self.net = UNet(3, width, time_dim=4 * width)
        self.time_embedding = TimeEmbedding(width)

    def forward(self, state, time):
        return self.net(state, self.time_embedding(time)).float()


class I2SBBridge(nn.Module):
    """Discrete symmetric I2SB SR bridge, with positive endpoint variances.

    The final returned sample is at grid index 0, as in upstream; its
    posterior mean is not simply replaced by the last predicted clean HR.
    """
    def __init__(self, intervals=1000, beta_max=.3):
        super().__init__()
        if intervals < 4 or intervals % 2 or beta_max <= 0:
            raise ValueError('I2SB needs an even interval count >=4 and positive beta_max')
        ramp = torch.linspace(1e-2, (beta_max / intervals)**.5, intervals, dtype=torch.float64).square()
        beta = torch.cat([ramp[:intervals//2], ramp[:intervals//2].flip(0)])
        forward = beta.cumsum(0)
        backward = beta.flip(0).cumsum(0).flip(0)
        self.intervals = intervals
        self.register_buffer('variance_fwd', forward.float())
        self.register_buffer('clean_weight', (backward / (forward + backward)).float())
        self.register_buffer('std_bridge', (forward * backward / (forward + backward)).sqrt().float())

    def std(self, step):
        return self.variance_fwd[step].sqrt().reshape(-1, 1, 1, 1)

    def time(self, step):
        # Normalize to [1e-4,1], before the local embedding's x1000 scale.
        return 1e-4 + (1 - 1e-4) * step.float() / (self.intervals - 1)

    def training_pair(self, clean, degraded, step, noise=None):
        weight = self.clean_weight[step].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(clean) if noise is None else noise
        state = weight * clean + (1 - weight) * degraded + self.std_bridge[step].reshape(-1, 1, 1, 1) * noise
        return state.detach(), ((state - clean) / self.std(step)).detach()

    def posterior(self, current, clean_prediction, step, previous, noise=None):
        if not 0 <= previous < step < self.intervals:
            raise ValueError('Expected 0 <= previous < step < intervals')
        now, before = self.variance_fwd[step], self.variance_fwd[previous]
        ratio = before / now
        result = (1 - ratio) * clean_prediction + ratio * current
        if previous > 0:
            noise = torch.randn_like(current) if noise is None else noise
            result = result + (before * (1 - ratio)).sqrt() * noise
        return result

    @torch.no_grad()
    def sample(self, denoiser, degraded, nfe, generator=None, generators=None):
        if not 1 <= nfe < self.intervals:
            raise ValueError('I2SB NFE must be in [1, intervals-1]')
        if generators is not None and len(generators) != len(degraded):
            raise ValueError('Expected one noise generator per image')
        # Equal spacing with round, including both end indices (upstream convention).
        steps = [round(i * (self.intervals - 1) / nfe) for i in range(nfe + 1)]
        state = degraded.clone()
        for step, previous in zip(steps[:0:-1], steps[-2::-1]):
            indices = torch.full((len(state),), step, dtype=torch.long, device=state.device)
            clean = state - self.std(indices) * denoiser(state, self.time(indices))
            noise = None
            if previous > 0 and generators is not None:
                noise = torch.stack([torch.randn(state.shape[1:], device=state.device,
                                                 dtype=state.dtype, generator=rng) for rng in generators])
            elif previous > 0:
                noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            state = self.posterior(state, clean, step, previous, noise=noise)
        return state


def relativistic_loss(real_score, fake_score, *, generator):
    """Relativistic-average logistic loss, including the cross-sample mean path."""
    real_mean = real_score.mean() if generator else real_score.detach().mean()
    fake_mean = fake_score.mean() if generator else fake_score.detach().mean()
    return .5 * (F.binary_cross_entropy_with_logits(real_score - fake_mean,
                                                   torch.zeros_like(real_score) if generator else torch.ones_like(real_score)) +
                 F.binary_cross_entropy_with_logits(fake_score - real_mean,
                                                   torch.ones_like(fake_score) if generator else torch.zeros_like(fake_score)))


def build_generator(config):
    method = config['method']
    if method in ('esrgan', 'rrdb_l1'):
        return RRDBNet()
    if method == 'i2sb':
        return BridgeDenoiser(config['width'])
    if method not in ('mse', 'mse_perceptual', 'mgm', 'gan'):
        raise ValueError(f'Unknown method: {method}')
    return Generator(config['width'], config['scale'], config['projection'], get_architecture_version(config))
