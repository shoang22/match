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
    import lpips
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


@functools.lru_cache(maxsize=None)
def _raft_model(size: str):
    from torchvision.models import optical_flow
    if size == 'large':
        model = optical_flow.raft_large(weights=optical_flow.Raft_Large_Weights.DEFAULT)
    elif size == 'small':
        model = optical_flow.raft_small(weights=optical_flow.Raft_Small_Weights.DEFAULT)
    else:
        raise ValueError(f"size must be 'large' or 'small', got {size!r}")
    return model.eval().requires_grad_(False)


def _raft(img1: torch.Tensor, img2: torch.Tensor, size: str, batch_size: int) -> torch.Tensor:
    """RAFT flow from img1 to img2 (N, 3, H, W) in [0, 1]; returns (N, 2, H, W)."""
    model = _raft_model(size).to(img1.device)
    h, w = img1.shape[-2:]
    # RAFT needs H and W divisible by 8 and at least 128.
    pad = [0, max(128, w + (-w) % 8) - w, 0, max(128, h + (-h) % 8) - h]
    out = []
    with torch.no_grad():
        for i in range(0, img1.shape[0], batch_size):
            a, b = (torch.nn.functional.pad(x[i:i + batch_size] * 2 - 1, pad, mode='replicate')
                    for x in (img1, img2))
            out.append(model(a, b)[-1][..., :h, :w])
    return torch.cat(out)


def raft_flow(frames: torch.Tensor, size: str = 'large', batch_size: int = 8):
    """Estimate optical flow between consecutive frames with torchvision RAFT.

    Args:
        frames: (T, 3, H, W) float in [0, 1].
        size: 'large' or 'small' RAFT. Standard weights download on first use.
        batch_size: frame pairs per RAFT call (lower it if memory is tight).

    Returns:
        (forward, backward), each (T - 1, 2, H, W) in pixels, channels (dx, dy).
        forward[t] maps frame t to frame t + 1: pixel p of frame t moves to
        p + forward[t](p). backward[t] maps frame t + 1 back to frame t.
    """
    frames = frames.float()
    first, second = frames[:-1], frames[1:]
    flows = _raft(torch.cat([first, second]), torch.cat([second, first]), size, batch_size)
    return flows[:len(first)], flows[len(first):]


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample x (N, C, H, W) at p + flow(p) for each pixel p, bilinear, zeros outside."""
    _, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
    gx = (xs + flow[:, 0]) * (2 / max(w - 1, 1)) - 1
    gy = (ys + flow[:, 1]) * (2 / max(h - 1, 1)) - 1
    return torch.nn.functional.grid_sample(
        x, torch.stack([gx, gy], dim=-1), mode='bilinear', padding_mode='zeros',
        align_corners=True)


def _consistent(forward: torch.Tensor, backward: torch.Tensor,
                alpha1: float = 0.01, alpha2: float = 0.5) -> torch.Tensor:
    """Forward-backward check on the frame t + 1 grid (Sundaram et al. 2010).

    Pixel p of frame t + 1 lands at q = p + backward(p) in frame t. The flow is
    reliable there if following forward(q) brings it back near p:
        |forward(q) + backward(p)|^2 < alpha1 (|forward(q)|^2 + |backward(p)|^2) + alpha2
    Returns (N, 1, H, W) bool.
    """
    fwd_at_q = _warp(forward, backward)
    lhs = (fwd_at_q + backward).pow(2).sum(1, keepdim=True)
    rhs = alpha1 * (fwd_at_q.pow(2).sum(1, keepdim=True)
                    + backward.pow(2).sum(1, keepdim=True)) + alpha2
    return lhs < rhs


def flow_aligned_error(frames: torch.Tensor, masks: torch.Tensor, flow=None,
                       reference: torch.Tensor = None, raft_size: str = 'large',
                       raft_batch_size: int = 8) -> TemporalScores:
    """Mean absolute error left after warping each frame onto the next with flow.

    For pair t, frame t is backward-warped onto frame t + 1 and |frame[t+1] -
    warped frame[t]| is averaged over all channels of the valid pixels. A pixel p
    of frame t + 1 is valid when it is foreground in mask t + 1, every pixel of
    frame t that the bilinear warp reads from is foreground in mask t (and inside
    the image), and the flow passes the forward-backward consistency check. A
    pair with no valid pixels scores nan.

    The flow comes from, in order of precedence:
        flow: (forward, backward) as returned by `raft_flow`, e.g. estimated once on
            the real footage and reused for the prediction so both are aligned with
            the same motion.
        reference: a (T, 3, H, W) sequence to estimate flow on with RAFT.
        otherwise RAFT on `frames` itself.
    Pixels outside the mask never enter the error. They can only change the score
    through the flow, when the flow is estimated from these same frames.
    """
    frames, masks = _prepare_sequence(frames, masks)
    if flow is None:
        flow = raft_flow(frames if reference is None else reference.to(frames.device),
                         raft_size, raft_batch_size)
    forward, backward = (f.to(frames.device).float() for f in flow)
    n = frames.shape[0] - 1
    expected = (n, 2, *frames.shape[-2:])
    if tuple(forward.shape) != expected or tuple(backward.shape) != expected:
        raise ValueError(f'flow {tuple(forward.shape)}, {tuple(backward.shape)} '
                         f'does not match {expected}')

    warped = _warp(frames[:-1], backward)
    # A warped mask of 1 means every bilinear tap with non-zero weight was foreground.
    warped_mask = _warp(masks[:-1].float(), backward) > 0.999
    valid = masks[1:] & warped_mask & _consistent(forward, backward)
    return _scores([_masked_mean_abs(frames[i + 1], warped[i], valid[i]) for i in range(n)])
