"""Metric collectors of the sequence evaluation run.

A collector sees every source frame (`add_frame(ctx)`, in frame order) and returns its sections of the
metrics JSON from `finish()`, as a dict of top-level keys. Sections of different collectors must not
share keys. To add metrics, write a collector here and add it to `build_metric_collectors` in
scripts/match/eval_sequence.py.
"""
import math
from typing import Callable, NamedTuple, Protocol

import torch

from match.sequence_eval.context import FrameContext
from match.utils import metric_util


class ImageMetric(NamedTuple):
    fn: Callable  # (pred, real, mask) -> float, see match/utils/metric_util.py
    higher_is_better: bool
    label: str  # plot title


class TemporalScore(NamedTuple):
    fn: Callable  # (frames (T,3,H,W), masks (T,1,H,W), flow from raft_flow) -> metric_util.TemporalScores
    label: str  # plot title; every temporal score is a change, lower is better


def _raw_flicker(frames, masks, flow):
    return metric_util.raw_flicker(frames, masks)


def _flow_aligned_error(frames, masks, flow):
    return metric_util.flow_aligned_error(frames, masks, flow=flow)


# Adding a metric here adds it to metrics.json, its summary and the plots.
IMAGE_METRICS = {
    'psnr': ImageMetric(metric_util.masked_psnr, True, 'Masked PSNR (dB, higher is better)'),
    'ssim': ImageMetric(metric_util.masked_ssim, True, 'Masked SSIM (higher is better)'),
    'lpips': ImageMetric(metric_util.masked_lpips, False, 'Masked LPIPS (lower is better)'),
}
# Temporal scores, computed on each fixed-view sequence in TEMPORAL_SOURCES.
TEMPORAL_SCORES = {
    'raw_flicker': TemporalScore(_raw_flicker, 'Raw flicker (lower is better)'),
    'flow_aligned_error': TemporalScore(_flow_aligned_error, 'Flow-aligned error (lower is better)'),
}
TEMPORAL_SOURCES = ('prediction', 'real')
N_WORST_FRAMES = 3


class MetricCollector(Protocol):
    """Sees every source frame, in frame order, and returns its top-level sections of metrics.json."""

    def add_frame(self, ctx: FrameContext) -> None: ...

    def finish(self) -> dict: ...


def summarize(values_by_frame, higher_is_better, n_worst=N_WORST_FRAMES):
    """Mean over the frames, and the n_worst frames (worst first) as [{'frame', 'value'}].

    Non-finite values (inf PSNR for identical images, nan for no valid pixels) are left out of both.
    """
    finite = {f: v for f, v in values_by_frame.items() if math.isfinite(v)}
    worst = sorted(finite.items(), key=lambda fv: fv[1], reverse=not higher_is_better)[:n_worst]
    return {
        'mean': sum(finite.values()) / len(finite) if finite else math.nan,
        'worst_frames': [{'frame': f, 'value': v} for f, v in worst],
    }


class ImageMetrics:
    """Masked PSNR, SSIM and LPIPS of the prediction against the real image, per frame and held-out camera.

    JSON sections:
      'per_frame': {frame_id: {camera_id: {'psnr', 'ssim', 'lpips'}}}
      'summary': {camera_id: {metric: {'mean', 'worst_frames'}}}
    """

    def __init__(self):
        self.per_frame = {}

    def add_frame(self, ctx):
        scores = {}
        for camera, view in ctx.heldout.items():
            scores[camera] = {name: m.fn(view.pred, view.real, view.mask) for name, m in IMAGE_METRICS.items()}
        self.per_frame[ctx.frame_id] = scores

    def finish(self):
        cameras = list(next(iter(self.per_frame.values())).keys()) if self.per_frame else []
        summary = {}
        for camera in cameras:
            summary[camera] = {
                name: summarize({f: s[camera][name] for f, s in self.per_frame.items()}, m.higher_is_better)
                for name, m in IMAGE_METRICS.items()
            }
        return {'per_frame': self.per_frame, 'summary': summary}


class TemporalMetrics:
    """Frame-to-frame change at the fixed novel viewpoint, for the MATCH prediction and the real footage.

    Both sequences use the real foreground masks. Optical flow is estimated once with RAFT on the real
    frames and reused for the prediction, so both are aligned with the same motion. Frames are kept on
    the CPU until finish(), where RAFT runs on `device`.

    JSON section 'temporal':
      'camera': the fixed-view camera ID
      'per_pair': {frame_id: {'previous_frame': frame_id,
                              'prediction': {'raw_flicker', 'flow_aligned_error'},
                              'real': {'raw_flicker', 'flow_aligned_error'}}}
        keyed by the later frame of each consecutive pair of source frames
      'summary': {'prediction'|'real': {score: {'mean', 'worst_frames'}}}, worst_frames keyed like per_pair,
        highest change first
    """

    def __init__(self, raft_size='large', raft_batch_size=8, device='cpu'):
        self.raft_size = raft_size
        self.raft_batch_size = raft_batch_size
        self.device = device
        self.fixed_view_camera = None
        self.frame_ids, self.pred, self.real, self.mask = [], [], [], []

    def add_frame(self, ctx):
        view = ctx.fixed_view
        self.fixed_view_camera = ctx.fixed_view_camera
        self.frame_ids.append(ctx.frame_id)
        self.pred.append(view.pred.detach().float().cpu())
        self.real.append(view.real.detach().float().cpu())
        self.mask.append(view.mask.detach().float().cpu())

    def _per_pair_scores(self):
        """{source: {score: [value per consecutive pair]}}."""
        pred, real, mask = (torch.stack(x).to(self.device) for x in (self.pred, self.real, self.mask))
        flow = metric_util.raft_flow(real, self.raft_size, self.raft_batch_size)
        frames = {'prediction': pred, 'real': real}
        return {
            source: {name: score.fn(frames[source], mask, flow).per_pair for name, score in TEMPORAL_SCORES.items()}
            for source in TEMPORAL_SOURCES
        }

    def finish(self):
        pair_frames = self.frame_ids[1:]
        if pair_frames:
            scores = self._per_pair_scores()
        else:
            scores = {source: {name: [] for name in TEMPORAL_SCORES} for source in TEMPORAL_SOURCES}
        per_pair = {}
        for t, (previous, frame) in enumerate(zip(self.frame_ids, pair_frames)):
            per_pair[frame] = {'previous_frame': previous}
            for source, by_score in scores.items():
                per_pair[frame][source] = {name: values[t] for name, values in by_score.items()}
        summary = {
            source: {name: summarize(dict(zip(pair_frames, values)), higher_is_better=False)
                     for name, values in by_score.items()}
            for source, by_score in scores.items()
        }
        return {'temporal': {'camera': self.fixed_view_camera, 'per_pair': per_pair, 'summary': summary}}
