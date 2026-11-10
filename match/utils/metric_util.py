"""Masked image-quality (PSNR, SSIM, LPIPS) and temporal-consistency metrics.

Plain functions with no dependence on the runner, dataset or model, so they run
on CPU in tests. Keep imports light here: torch, plus lpips and torchvision
imported lazily where needed.

Conventions for every image metric:
  pred, real: float tensors in [0, 1], shape (3, H, W) or (1, 3, H, W).
  mask: foreground mask, shape (H, W), (1, H, W) or (1, 1, H, W). Bool, or float
    where values > 0.5 count as foreground.
  Returns a Python float.
"""

import functools
import math
from typing import NamedTuple

import torch


def _prepare(pred: torch.Tensor, real: torch.Tensor, mask: torch.Tensor):
  """Return pred, real as (1, C, H, W) float and mask as (1, 1, H, W) bool."""
  if pred.shape != real.shape:
    raise ValueError(f'pred {tuple(pred.shape)} and real {tuple(real.shape)} differ')
  if pred.dim() == 3:
    pred, real = pred[None], real[None]
  if pred.dim() != 4 or pred.shape[0] != 1:
    raise ValueError(f'expected (C, H, W) or (1, C, H, W), got {tuple(pred.shape)}')
  h, w = pred.shape[-2:]
  if mask.dtype != torch.bool:
    mask = mask > 0.5
  mask = mask.reshape(1, 1, *mask.shape[-2:])
  if mask.shape[-2:] != (h, w):
    raise ValueError(f'mask {tuple(mask.shape[-2:])} does not match image {(h, w)}')
  if not mask.any():
    raise ValueError('mask has no foreground pixels')
  return pred.float(), real.float(), mask.to(pred.device)


def masked_psnr(pred: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> float:
  """PSNR in dB over the foreground pixels (all channels), peak value 1.

  Returns math.inf when pred and real are identical inside the mask (no cap).
  """
  pred, real, mask = _prepare(pred, real, mask)
  m = mask.expand_as(pred)
  mse = ((pred[m] - real[m]) ** 2).mean().item()
  if mse == 0:
    return math.inf
  return -10.0 * math.log10(mse)


def _fill_background(pred, real, mask, background: float):
  """Paint the same colour over the background of both images."""
  fill = torch.full_like(pred, background)
  return torch.where(mask, pred, fill), torch.where(mask, real, fill)


def _gaussian_window(size: int, sigma: float, channels: int) -> torch.Tensor:
  coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
  g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
  g = g / g.sum()
  window = g[:, None] * g[None, :]
  return window.expand(channels, 1, size, size).contiguous()


def masked_ssim(pred: torch.Tensor, real: torch.Tensor, mask: torch.Tensor,
                background: float = 0.0) -> float:
  """SSIM averaged over the foreground pixels.

  Both images get `background` painted over the background, then the standard
  SSIM map (11x11 Gaussian window, sigma 1.5, peak 1, reflect padding so the map
  matches the image size) is averaged over the mask. Windows near the mask edge
  see the shared fill colour, which is identical in both images.
  """
  pred, real, mask = _prepare(pred, real, mask)
  pred, real = _fill_background(pred, real, mask, background)
  c = pred.shape[1]
  size, sigma = 11, 1.5
  window = _gaussian_window(size, sigma, c).to(pred.device)

  def blur(x):
    x = torch.nn.functional.pad(x, [size // 2] * 4, mode='reflect')
    return torch.nn.functional.conv2d(x, window, groups=c)

  c1, c2 = 0.01 ** 2, 0.03 ** 2
  mu_p, mu_r = blur(pred), blur(real)
  var_p = blur(pred * pred) - mu_p ** 2
  var_r = blur(real * real) - mu_r ** 2
  cov = blur(pred * real) - mu_p * mu_r
  ssim_map = ((2 * mu_p * mu_r + c1) * (2 * cov + c2)) / (
      (mu_p ** 2 + mu_r ** 2 + c1) * (var_p + var_r + c2))
  return ssim_map[mask.expand_as(ssim_map)].mean().item()


@functools.lru_cache(maxsize=None)
def _lpips_model(net: str):
  import lpips  # pylint: disable=g-import-not-at-top
  return lpips.LPIPS(net=net, spatial=True, verbose=False).eval().requires_grad_(False)


def masked_lpips(pred: torch.Tensor, real: torch.Tensor, mask: torch.Tensor,
                 background: float = 0.0, net: str = 'vgg') -> float:
  """LPIPS averaged over the foreground pixels (lower is better, 0 = identical).

  Both images get `background` painted over the background, then the spatial
  LPIPS map is averaged over the mask. `net='vgg'` matches the runner's training
  loss (`MatchRunner.init_lpips`). The torchvision backbone weights are
  downloaded on first use if not cached; the linear-layer weights ship with the
  lpips package.
  """
  pred, real, mask = _prepare(pred, real, mask)
  pred, real = _fill_background(pred, real, mask, background)
  model = _lpips_model(net).to(pred.device)
  with torch.no_grad():
    lpips_map = model(pred, real, normalize=True)  # (1, 1, H, W)
  return lpips_map[mask].mean().item()


# --- Temporal consistency -----------------------------------------------------
#
# Conventions for every temporal metric:
#   frames: float tensor in [0, 1], shape (T, 3, H, W), T >= 2.
#   masks: foreground masks, shape (T, H, W) or (T, 1, H, W). Bool, or float
#     where values > 0.5 count as foreground.
#   Pair t compares frames t and t + 1.


class TemporalScores(NamedTuple):
  """per_pair[t] scores frames (t, t + 1); mean is over the finite values."""
  per_pair: list
  mean: float


def _prepare_sequence(frames: torch.Tensor, masks: torch.Tensor):
  """Return frames as (T, C, H, W) float and masks as (T, 1, H, W) bool."""
  if frames.dim() != 4 or frames.shape[0] < 2:
    raise ValueError(f'expected frames (T >= 2, C, H, W), got {tuple(frames.shape)}')
  t, _, h, w = frames.shape
  if masks.dtype != torch.bool:
    masks = masks > 0.5
  if masks.dim() == 3:
    masks = masks[:, None]
  if tuple(masks.shape) != (t, 1, h, w):
    raise ValueError(f'masks {tuple(masks.shape)} do not match frames {tuple(frames.shape)}')
  return frames.float(), masks.to(frames.device)


def _masked_mean_abs(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> float:
  """Mean |a - b| over the valid pixels (all channels); nan if none are valid."""
  m = valid.expand_as(a)
  if not m.any():
    return math.nan
  return (a[m] - b[m]).abs().mean().item()


def _scores(per_pair) -> TemporalScores:
  finite = [v for v in per_pair if math.isfinite(v)]
  return TemporalScores(per_pair, sum(finite) / len(finite) if finite else math.nan)


def raw_flicker(frames: torch.Tensor, masks: torch.Tensor) -> TemporalScores:
  """Mean absolute change between consecutive frames, inside the mask.

  Pair t averages |frame[t+1] - frame[t]| over all channels of the pixels that
  are foreground in both frames. A pair with no such pixels scores nan.
  """
  frames, masks = _prepare_sequence(frames, masks)
  return _scores([
      _masked_mean_abs(frames[i + 1], frames[i], masks[i] & masks[i + 1])
      for i in range(frames.shape[0] - 1)
  ])
