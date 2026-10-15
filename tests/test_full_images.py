"""Full-frame regression checks, authored but not run during code preparation."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mgm_i2i.cluster.full_images import prepare_frame, frame_latent, frame_scores, save_picture, write_gallery
from mgm_i2i.cluster.models import Generator


def test_full_frame_keeps_all_pixels_and_projects_on_padded_lr():
    pixels = np.random.default_rng(7).integers(0, 256, (35, 47, 3), dtype=np.uint8)
    real, padded, condition, padding = prepare_frame(pixels, 4)
    assert padding == dict(bottom=1, right=1)
    assert padded.shape == (1, 3, 36, 48)
    assert condition.shape == (1, 3, 9, 12)
    assert torch.equal(padded[..., :35, :47], real)
    recovered = ((real[0]+1)*127.5).round().byte().permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(recovered, pixels)
    gen = Generator(width=8).eval()
    torch.nn.init.normal_(gen.net.out.weight, std=.01)
    with torch.inference_mode():
        result = gen(condition, frame_latent(padded.shape, '0801', 0))
    assert result.shape == padded.shape
    assert result[..., :35, :47].shape == real.shape
    assert torch.allclose(F.avg_pool2d(result, 4), condition, atol=1e-6)


def test_full_latents_are_stable_by_image_and_draw():
    shape = (1, 3, 24, 28)
    original = frame_latent(shape, '0801', 0)
    torch.manual_seed(999)
    frame_latent(shape, '0802', 1)
    assert torch.equal(original, frame_latent(shape, '0801', 0))
    assert not torch.equal(original, frame_latent(shape, '0801', 1))
    assert not torch.equal(original, frame_latent(shape, '0802', 0))
    assert frame_latent(shape, '0801', 1, deterministic=True).count_nonzero() == 0


def test_full_metrics_and_gallery_preserve_original_resolution(tmp_path):
    from PIL import Image
    pixels = np.random.default_rng(11).integers(0, 256, (27, 31, 3), dtype=np.uint8)
    real, _, _, _ = prepare_frame(pixels, 4)
    scores = frame_scores(real, real)
    assert scores['psnr'] == pytest.approx(120.)
    assert scores['ssim'] == pytest.approx(1.)
    path = tmp_path/'draw1.png'
    save_picture(path, real)
    with Image.open(path) as picture:
        assert picture.size == (31, 27)
        np.testing.assert_array_equal(np.asarray(picture), pixels)
    write_gallery(tmp_path/'index.html', 'Test <title>', [('Image', [('Draw <1>', path)])], 'Preview')
    page = (tmp_path/'index.html').read_text()
    assert 'href="draw1.png"' in page and 'src="draw1_preview.jpg"' in page
    assert 'Draw &lt;1&gt;' in page and 'Test &lt;title&gt;' in page
