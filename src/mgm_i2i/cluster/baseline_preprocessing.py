"""Shared 128x128 SR pairs and reference native baseline transforms.

The main comparison uses shared_sr_pair on random train / center eval crops.
The native esrgan_pair/i2sb_transform functions remain reference utilities
for separate reproduction experiments, not the selected main protocol.
Baseline training integrations are not yet wired into the engine.
"""
import random

import numpy as np
import torch
from torchvision import transforms


def shared_sr_pair(hr, *, unit_range=False):
    """Canonical uint8 CHW/BCHW crop -> HR128/LR32, with pooling x4.

    ESRGAN may request [0,1]; other models use [-1,1]. Geometry and image
    identities are shared; random train cropping happens in the dataset.
    """
    if (hr.dtype != torch.uint8 or hr.ndim not in (3, 4)
            or hr.shape[-3:] != (3, 128, 128)):
        raise ValueError('Expected 128x128 RGB uint8 CHW or BCHW tensor')
    target = hr.float().div(255)
    if not unit_range:
        target = target.mul(2).sub(1)
    lr = torch.nn.functional.avg_pool2d(target, 4)
    return {'hr': target, 'lr': lr}


def esrgan_pair(hr, lr, *, training, rng=None):
    """Reference native PIL pair -> [0,1], crop and shared augmentation.

    Supply the matching x4 LR image (official DIV2K bicubic X4 for the
    native recipe). An injected random.Random allows reproducible workers.
    """
    rng = random if rng is None else rng
    gt = np.asarray(hr.convert('RGB'))
    lq = np.asarray(lr.convert('RGB'))
    h, w = lq.shape[:2]
    if training:
        if gt.shape[:2] != (h * 4, w * 4):
            raise ValueError('ESRGAN requires an aligned HR/LR x4 pair')
        if min(h, w) < 32:
            raise ValueError('ESRGAN LR must fit a 32x32 training crop')
        top, left = rng.randint(0, h - 32), rng.randint(0, w - 32)
        lq = lq[top:top + 32, left:left + 32]
        gt = gt[top * 4:top * 4 + 128, left * 4:left * 4 + 128]
        # BasicSR use_rot=True uses a vertical flip and transpose.
        hflip, vflip, transpose = (rng.random() < .5 for _ in range(3))
        pair = [gt, lq]
        for i, image in enumerate(pair):
            if hflip:
                image = image[:, ::-1]
            if vflip:
                image = image[::-1]
            if transpose:
                image = image.transpose(1, 0, 2)
            pair[i] = image
        gt, lq = pair
    else:
        # BasicSR trims unmatched GT borders for validation/testing.
        if gt.shape[0] < h * 4 or gt.shape[1] < w * 4:
            raise ValueError('ESRGAN HR is smaller than LR x4')
        gt = gt[:h * 4, :w * 4]

    def tensor(image):
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255)

    return {'hr': tensor(gt), 'lr': tensor(lq)}


def i2sb_transform(*, training):
    """Reference native PIL -> [-1,1] ImageNet-256 geometry/normalization.

    Dataset decoding must convert PIL images to RGB before this transform,
    as in the upstream loader. Global torch RNG controls the train flip.
    Degradation and bridge noise sampling are separate training operations.
    """
    steps = [transforms.Resize(256), transforms.CenterCrop(256)]
    if training:
        steps.append(transforms.RandomHorizontalFlip(p=.5))
    steps.extend([transforms.ToTensor(), transforms.Lambda(lambda t: t * 2 - 1)])
    return transforms.Compose(steps)
