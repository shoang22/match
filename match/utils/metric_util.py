"""Masked image-quality metrics (PSNR, SSIM, LPIPS) on image tensors.

Plain functions with no dependence on the runner, dataset or model, so they run
on CPU in tests. Keep imports light here: torch and lpips only.

Conventions for every image metric:
  pred, real: float tensors in [0, 1], shape (3, H, W) or (1, 3, H, W).
  mask: foreground mask, shape (H, W), (1, H, W) or (1, 1, H, W). Bool, or float
    where values > 0.5 count as foreground.
  Returns a Python float.
"""

import functools
import math

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
