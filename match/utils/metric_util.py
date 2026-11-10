"""Masked image-quality metrics (PSNR, SSIM, LPIPS) on image tensors.

Plain functions with no dependence on the runner, dataset or model, so they run
on CPU in tests. Keep imports light here: torch and lpips only.

Conventions for every image metric:
  pred, real: float tensors in [0, 1], shape (3, H, W) or (1, 3, H, W).
  mask: foreground mask, shape (H, W), (1, H, W) or (1, 1, H, W). Bool, or float
    where values > 0.5 count as foreground.
  Returns a Python float.
"""

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
