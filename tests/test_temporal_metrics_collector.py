"""The sequence run's temporal-metrics collector: CPU tests on synthetic fixed-view sequences.

The collector sees one FrameContext per source frame and returns the 'temporal' section of metrics.json.
"""
import math

import pytest
import torch

from match.sequence_eval.context import FrameContext, HeldOutView
from match.sequence_eval.metrics import TemporalMetrics

H, W = 128, 128
CAMERA = '402040'
FRAMES = ['32885', '32897', '32909', '32921']
SOURCES = ['prediction', 'real']
SCORES = ['raw_flicker', 'flow_aligned_error']


def make_canvas(seed, size=192):
    """Smooth random RGB texture in [0, 1], shape (3, size, size)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, size // 8, size // 8, generator=g)
    return torch.nn.functional.interpolate(low, size=(size, size), mode='bicubic', align_corners=False)[0].clamp(0, 1)


def translating(seed, n=len(FRAMES), step=(3, 2)):
    """A window sliding across a texture by `step` (dx, dy) px per frame: (n, 3, H, W)."""
    canvas, (dx, dy) = make_canvas(seed), step
    return torch.stack([canvas[:, t * dy:t * dy + H, t * dx:t * dx + W] for t in range(n)])


def static(seed, n=len(FRAMES)):
    return make_canvas(seed)[:, :H, :W].expand(n, 3, H, W).clone()


def square_mask():
    mask = torch.zeros(1, H, W)
    mask[:, 32:96, 32:96] = 1.0
    return mask


def contexts(pred, real, frames=FRAMES, mask=None):
    """One FrameContext per frame whose only held-out camera (the fixed view) shows pred[i] and real[i]."""
    mask = square_mask() if mask is None else mask
    out = []
    for i, frame in enumerate(frames):
        view = HeldOutView(CAMERA, pred[i], real[i], mask, torch.eye(4), torch.tensor([1.0, 1.0, 0.5, 0.5]))
        out.append(FrameContext(runner=None, frame_index=i, frame_id=frame, batch=None, gaussians={},
                                heldout={CAMERA: view}, fixed_view_camera=CAMERA, output_frame_indices=[i],
                                n_output_frames=len(frames), fps=24))
    return out


def collect(ctxs, **kwargs):
    collector = TemporalMetrics(**kwargs)
    for ctx in ctxs:
        collector.add_frame(ctx)
    return collector.finish()


def test_temporal_section_has_both_scores_for_both_sources_per_consecutive_pair():
    sections = collect(contexts(translating(1), translating(0)))

    assert list(sections) == ['temporal']
    temporal = sections['temporal']
    assert temporal['camera'] == CAMERA
    # Each pair is keyed by its later source frame and names the frame before it.
    assert list(temporal['per_pair']) == FRAMES[1:]
    for previous, frame in zip(FRAMES, FRAMES[1:]):
        entry = temporal['per_pair'][frame]
        assert entry['previous_frame'] == previous
        for source in SOURCES:
            assert sorted(entry[source]) == sorted(SCORES)
            assert all(isinstance(v, float) and math.isfinite(v) for v in entry[source].values()), entry
