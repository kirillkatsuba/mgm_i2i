"""Read-only shape/parameter audit of pinned official I2SB ADM; no weights/GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

COMMIT = '1ffdfaaf05495ef883ece2c1fe991b3049f814cc'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path('third_party/I2SB-official'))
    args = parser.parse_args()
    root = args.upstream.resolve()
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != COMMIT:
        raise ValueError(f'Expected upstream {COMMIT}, found {commit}')
    sys.path.insert(0, str(root))
    from guided_diffusion.unet import UNetModel

    # ckpt_util.create_argparser -> script_util.create_model; original ADM's
    # six-channel head is reduced to three channels by upstream itself.
    config = dict(image_size=256, in_channels=3, model_channels=256, out_channels=3,
                  num_res_blocks=2, attention_resolutions=(8, 16, 32), dropout=0.,
                  channel_mult=(1, 1, 2, 2, 4, 4), num_classes=None,
                  use_checkpoint=False, use_fp16=False, num_heads=4,
                  num_head_channels=64, num_heads_upsample=-1,
                  use_scale_shift_norm=True, resblock_updown=True,
                  use_new_attention_order=False)
    shapes = {}
    with torch.device('meta'):
        model = UNetModel(**config)
        count = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            for size in (128, 256):
                shapes[str(size)] = list(model(torch.empty(1, 3, size, size), torch.ones(1)).shape)
    assert count == 552807171
    print(json.dumps(dict(upstream_commit=commit, config=config, parameters=count,
                          fp32_weights_GiB=count*4/2**30, output_shapes=shapes,
                          verification='Meta shapes only; no pretrained weights, numerical inference, training or CUDA probe'),
                     indent=2))


if __name__ == '__main__':
    main()
