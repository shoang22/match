"""Masked image metrics: CPU tests on synthetic images."""

import math

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


def test_identical_images_give_infinite_psnr():
  img = make_image(0)
  assert metric_util.masked_psnr(img, img.clone(), make_mask()) == math.inf
