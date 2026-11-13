"""Temporal texture filter: the per-frame MATCH baseline that smooths Gaussian textures over time.

Each texel's parameters are averaged over neighbouring frames with a centred Gaussian kernel, truncated at
`kernel_radius(sigma)` frames and renormalized where it runs past either end of the sequence, so it neither
lags motion nor fades the first and last frames.
"""
import math

import torch

CHANNELS = 2  # channel axis of a masked texture, (B,V,C,L,1)
# The predicted Gaussian parameters. Every other key is an input of the prediction and passes through from its
# own frame: 'uv' (the texel's fixed UV coordinate) and 'tempeh_xyz' (the TEMPEH surface point under the texel,
# #27). Keeping TEMPEH unfiltered scores the baseline like temporal MATCH: the TEMPEH jitter stays that of the
# input mesh, and the offset (position - tempeh_xyz) carries whatever the filter removed.
FILTERED_KEYS = ('xyz', 'rgb', 'scale', 'opacity', 'rotation')


def kernel_radius(sigma: float) -> int:
    """Frames on each side of the centre that the kernel reaches (4 sigma, as scipy.ndimage.gaussian_filter1d)."""
    return int(4 * sigma + 0.5)


def kernel(t: int, n: int, sigma: float) -> tuple[int, torch.Tensor]:
    """Weights of frame t's filter over an n-frame sequence: (first frame, weights summing to 1)."""
    r = kernel_radius(sigma)
    start, stop = max(0, t - r), min(n, t + r + 1)
    weights = torch.tensor([math.exp(-0.5 * ((s - t) / sigma) ** 2) for s in range(start, stop)], dtype=torch.float64)
    return start, weights / weights.sum()


def blend(textures: list[dict], weights: torch.Tensor, centre: int) -> dict:
    """Weighted average of the textures, key by key; textures[centre] is the frame being filtered.

    Each rotation quaternion is first flipped onto the centre frame's hemisphere (q and -q are the same
    rotation), then the average is renormalized to a unit quaternion. Keys other than FILTERED_KEYS are
    taken from the centre frame unchanged.
    """
    out = dict(textures[centre])
    if len(textures) == 1:
        return out
    for key in FILTERED_KEYS:
        reference = textures[centre][key].float()
        acc = torch.zeros_like(reference)
        for texture, w in zip(textures, weights.tolist()):
            value = texture[key].float()
            if key == 'rotation':
                value = torch.where((value * reference).sum(CHANNELS, keepdim=True) < 0, -value, value)
            acc += w * value
        out[key] = acc.to(textures[centre][key].dtype)
    out['rotation'] = torch.nn.functional.normalize(out['rotation'], p=2, dim=CHANNELS)
    return out
