"""Count selected architecture specifications; no training or weight downloads.

ESRGAN layer shapes follow BasicSR rrdbnet_arch.py and discriminator_arch.py:
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/discriminator_arch.py
Counts use the actual local baseline implementations; the legacy inventory helpers
remain as a shape reference.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mgm_i2i.cluster.models import DEFAULT_WIDTH, Generator, Field
from mgm_i2i.cluster.baselines import RRDBNet, VGGDiscriminator, BridgeDenoiser


def count(module):
    return sum(p.numel() for p in module.parameters())


def rrdb_inventory():
    # Five convolutions per dense block, three dense blocks per RRDB.
    blocks = []
    for _ in range(23):
        for _ in range(3):
            blocks.extend(nn.Conv2d(64 + j * 32, 32, 3, padding=1) for j in range(4))
            blocks.append(nn.Conv2d(64 + 4 * 32, 64, 3, padding=1))
    # first, body, two upsampling convolutions, HR, final RGB.
    blocks.append(nn.Conv2d(3, 64, 3, padding=1))
    blocks.extend(nn.Conv2d(64, 64, 3, padding=1) for _ in range(4))
    blocks.append(nn.Conv2d(64, 3, 3, padding=1))
    return nn.ModuleList(blocks)


def discriminator_inventory():
    # BasicSR VGGStyleDiscriminator(num_in_ch=3, num_feat=64, input_size=128).
    blocks = [nn.Conv2d(3, 64, 3, padding=1, bias=True)]
    specs = [(64, 64, 4), (64, 128, 3), (128, 128, 4),
             (128, 256, 3), (256, 256, 4), (256, 512, 3),
             (512, 512, 4), (512, 512, 3), (512, 512, 4)]
    for cin, cout, kernel in specs:
        blocks.extend([nn.Conv2d(cin, cout, kernel, stride=2 if kernel == 4 else 1,
                                 padding=1, bias=False), nn.BatchNorm2d(cout)])
    blocks.extend([nn.Linear(512 * 4 * 4, 100), nn.Linear(100, 1)])
    return nn.ModuleList(blocks)


def vgg19_feature_inventory():
    # VGG19 through conv5_4; no classifier, no weights downloaded.
    blocks, cin = [], 3
    for cout, repeats in [(64, 2), (128, 2), (256, 4), (512, 4), (512, 4)]:
        for _ in range(repeats):
            blocks.append(nn.Conv2d(cin, cout, 3, padding=1))
            cin = cout
    return nn.ModuleList(blocks)


def inventory(width=DEFAULT_WIDTH):
    with torch.device('meta'):
        generator = count(Generator(width=width, architecture_version=2))
        field_v1 = count(Field(width=width, architecture_version=1))
        field = count(Field(width=width, architecture_version=2))
        # I2SB-small: bridge state has 3 channels; time embedding is separate.
        bridge = count(BridgeDenoiser(width))
        esrgan = count(RRDBNet())
        discriminator = count(VGGDiscriminator())
        vgg = count(vgg19_feature_inventory())
    return {
        'unet_widths': [width, 2 * width, 4 * width],
        'counting': 'Parameters only; excludes buffers, EMA copies, optimizer states and evaluation networks.',
        'mse': {'inference': generator, 'trainable': generator},
        'mse_vgg': {'inference': generator, 'trainable': generator, 'frozen_loss_features': vgg},
        'mgm_v1': {'inference': generator, 'field': field_v1, 'trainable': generator + field_v1},
        'mgm_v2': {'inference': generator, 'field': field, 'trainable': generator + field},
        'esrgan_rrdb23': {'inference': esrgan, 'discriminator': discriminator,
                          'trainable': esrgan + discriminator, 'frozen_loss_features': vgg},
        'i2sb_small': {'inference': bridge, 'trainable': bridge,
                       'status': 'Local method adaptation with bridge training and sampling.'},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()
    if args.width < 8 or args.width % 8:
        parser.error('width must be a positive multiple of 8')
    print(json.dumps(inventory(args.width), indent=2))
