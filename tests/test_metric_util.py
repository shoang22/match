"""Masked image metrics: CPU tests on synthetic images."""

import math

import pytest
import torch

from match.utils import metric_util


H, W = 64, 64


def make_image(seed: int) -> torch.Tensor:
    """Smooth random RGB image in [0, 1], shape (3, H, W)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, H // 8, W // 8, generator=g)
    return torch.nn.functional.interpolate(
        low, size=(H, W), mode='bilinear', align_corners=False)[0]


def make_mask() -> torch.Tensor:
    """Centred square foreground, shape (H, W), bool."""
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[16:48, 16:48] = True
    return mask


METRICS = [metric_util.masked_psnr, metric_util.masked_ssim, metric_util.masked_lpips, metric_util.masked_l1,
           metric_util.masked_l2]


def test_identical_images_give_infinite_psnr():
    img = make_image(0)
    assert metric_util.masked_psnr(img, img.clone(), make_mask()) == math.inf


def test_identical_images_give_ssim_of_one():
    img = make_image(0)
    assert metric_util.masked_ssim(img, img.clone(), make_mask()) == pytest.approx(1.0, abs=1e-6)


def test_identical_images_give_lpips_of_zero():
    img = make_image(0)
    assert metric_util.masked_lpips(img, img.clone(), make_mask()) == pytest.approx(0.0, abs=1e-6)



def test_l1_and_l2_are_the_mean_absolute_and_squared_errors_over_foreground_pixels_and_channels():
    mask = make_mask()
    real = torch.full((3, H, W), 0.5)
    pred = real.clone()
    pred[0][mask] += 0.2  # one channel of every foreground pixel off by 0.2, the others exact
    pred[:, ~mask] = 1.0  # background errors don't count

    assert metric_util.masked_l1(pred, real, mask) == pytest.approx(0.2 / 3)
    assert metric_util.masked_l2(pred, real, mask) == pytest.approx(0.04 / 3)
    # PSNR is the same squared error in dB
    assert metric_util.masked_psnr(pred, real, mask) == pytest.approx(-10 * math.log10(0.04 / 3))

@pytest.mark.parametrize('metric', METRICS, ids=lambda f: f.__name__)
@pytest.mark.parametrize('which', ['pred', 'real'])
def test_pixels_outside_mask_do_not_change_score(metric, which):
    mask = make_mask()
    pred, real = make_image(0), make_image(1)
    baseline = metric(pred, real, mask)

    scribble = make_image(2)
    if which == 'pred':
        pred = torch.where(mask, pred, scribble)
    else:
        real = torch.where(mask, real, scribble)

    assert metric(pred, real, mask) == pytest.approx(baseline, rel=1e-6, abs=1e-7)


# Direction of improvement for each metric: +1 higher is better, -1 lower is better.
HIGHER_IS_BETTER = {'masked_psnr': 1, 'masked_ssim': 1, 'masked_lpips': -1, 'masked_l1': -1, 'masked_l2': -1}


@pytest.mark.parametrize('metric', METRICS, ids=lambda f: f.__name__)
def test_corrupting_pixels_inside_mask_worsens_score(metric):
    mask = make_mask()
    real = make_image(1)
    g = torch.Generator().manual_seed(3)
    mild = (real + 0.02 * torch.randn(real.shape, generator=g)).clamp(0, 1)
    corrupted = mild.clone()
    corrupted[:, 24:40, 24:40] = make_image(2)[:, 24:40, 24:40]  # inside the mask

    sign = HIGHER_IS_BETTER[metric.__name__]
    assert sign * metric(corrupted, real, mask) < sign * metric(mild, real, mask)
