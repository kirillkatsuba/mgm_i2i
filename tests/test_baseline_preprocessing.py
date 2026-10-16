import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mgm_i2i.cluster.baseline_preprocessing import esrgan_pair, i2sb_transform, shared_sr_pair


@pytest.mark.parametrize('batched', [False, True])
def test_shared_pair_preserves_center_pixels_and_exact_block_means(batched):
    image = torch.arange(3 * 128 * 128).remainder(256).to(torch.uint8).reshape(3, 128, 128)
    if batched:
        image = image.unsqueeze(0).repeat(2, 1, 1, 1)
    pair = shared_sr_pair(image)
    esrgan = shared_sr_pair(image, unit_range=True)
    assert pair['hr'].shape == image.shape
    assert pair['lr'].shape == (*image.shape[:-2], 32, 32)
    torch.testing.assert_close((pair['hr'] + 1) / 2, esrgan['hr'])
    torch.testing.assert_close((pair['lr'] + 1) / 2, esrgan['lr'])
    torch.testing.assert_close(((pair['hr'] + 1) * 127.5).round().to(torch.uint8), image)
    expected_first = image[..., :4, :4].float().mean(dim=(-2, -1)) / 255
    torch.testing.assert_close(esrgan['lr'][..., 0, 0], expected_first)
    with pytest.raises(ValueError, match='128x128'):
        shared_sr_pair(image[..., :64, :64])
    with pytest.raises(ValueError, match='uint8'):
        shared_sr_pair(image.float())


def test_esrgan_augmentation_keeps_pairs_aligned_and_varies_crops():
    lr = np.random.default_rng(7).integers(0, 256, (45, 51, 3), dtype=np.uint8)
    # Block-constant fixture lets us detect spatial misalignment exactly.
    hr = lr.repeat(4, axis=0).repeat(4, axis=1)
    outputs = []
    for seed in range(8):
        pair = esrgan_pair(Image.fromarray(hr), Image.fromarray(lr),
                           training=True, rng=random.Random(seed))
        assert pair['hr'].shape == (3, 128, 128)
        assert pair['lr'].shape == (3, 32, 32)
        torch.testing.assert_close(F.avg_pool2d(pair['hr'], 4), pair['lr'])
        assert 0 <= pair['hr'].min() <= pair['hr'].max() <= 1
        outputs.append(pair['lr'])
    assert any(not torch.equal(outputs[0], image) for image in outputs[1:])
    # Validation retains the supplied full pair, without resize or augmentation.
    pair = esrgan_pair(Image.fromarray(hr), Image.fromarray(lr), training=False)
    assert pair['hr'].shape == (3, 180, 204)
    torch.testing.assert_close(pair['lr'], torch.from_numpy(lr.transpose(2, 0, 1).copy()).float()/255)
    with pytest.raises(ValueError, match='aligned'):
        esrgan_pair(Image.fromarray(hr[:-1]), Image.fromarray(lr), training=True)


def test_i2sb_native_resize_crop_rgb_range_and_train_flip():
    # Two halves and distinct channels expose geometry, flips, and normalization.
    pixels = np.zeros((300, 600, 3), dtype=np.uint8)
    pixels[:, :300, 0] = 255
    pixels[:, 300:, 1] = 255
    pixels[:, :, 2] = 128
    image = Image.fromarray(pixels)
    evaluate = i2sb_transform(training=False)
    expected = evaluate(image)
    assert expected.shape == (3, 256, 256)
    torch.testing.assert_close(expected, evaluate(image))
    assert expected[0, 128, 0] == 1 and expected[1, 128, -1] == 1
    torch.testing.assert_close(expected[2], torch.full((256, 256), 128/255*2-1))
    train = i2sb_transform(training=True)
    with torch.random.fork_rng():
        seen = set()
        for seed in range(8):
            torch.manual_seed(seed)
            result = train(image)
            if torch.equal(result, expected):
                seen.add('original')
            elif torch.equal(result, expected.flip(-1)):
                seen.add('flipped')
            else:
                pytest.fail('Unexpected train transform beyond horizontal flip')
    assert seen == {'original', 'flipped'}
