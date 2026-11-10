"""Metric collectors of the sequence evaluation run.

A collector sees every source frame (`add_frame(ctx)`, in frame order) and returns its sections of the
metrics JSON from `finish()`, as a dict of top-level keys. Sections of different collectors must not
share keys. To add metrics, write a collector here and add it to `build_metric_collectors` in
scripts/match/eval_sequence.py.
"""
import math

import torch

from match.utils import metric_util

# For each image metric: the function and whether higher is better.
IMAGE_METRICS = {
    'psnr': (metric_util.masked_psnr, True),
    'ssim': (metric_util.masked_ssim, True),
    'lpips': (metric_util.masked_lpips, False),
}
N_WORST_FRAMES = 3


class MetricCollector:
    def add_frame(self, ctx):
        raise NotImplementedError

    def finish(self) -> dict:
        raise NotImplementedError


def summarize(values_by_frame, higher_is_better, n_worst=N_WORST_FRAMES):
    """Mean over the frames, and the n_worst frames (worst first) as [{'frame', 'value'}]."""
    finite = [v for v in values_by_frame.values() if math.isfinite(v)]
    worst = sorted(values_by_frame.items(), key=lambda fv: fv[1], reverse=not higher_is_better)[:n_worst]
    return {
        'mean': sum(finite) / len(finite) if finite else math.nan,
        'worst_frames': [{'frame': f, 'value': v} for f, v in worst],
    }


class ImageMetrics(MetricCollector):
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
            scores[camera] = {name: fn(view.pred, view.real, view.mask) for name, (fn, _) in IMAGE_METRICS.items()}
        self.per_frame[ctx.frame_id] = scores

    def finish(self):
        cameras = list(next(iter(self.per_frame.values())).keys()) if self.per_frame else []
        summary = {}
        for camera in cameras:
            summary[camera] = {
                name: summarize({f: s[camera][name] for f, s in self.per_frame.items()}, higher_is_better)
                for name, (_, higher_is_better) in IMAGE_METRICS.items()
            }
        return {'per_frame': self.per_frame, 'summary': summary}


class TemporalMetrics(MetricCollector):
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
        return {
            source: {
                'raw_flicker': metric_util.raw_flicker(frames, mask).per_pair,
                'flow_aligned_error': metric_util.flow_aligned_error(frames, mask, flow=flow).per_pair,
            }
            for source, frames in (('prediction', pred), ('real', real))
        }

    def finish(self):
        scores = self._per_pair_scores() if len(self.frame_ids) >= 2 else {}
        per_pair = {}
        for t, (previous, frame) in enumerate(zip(self.frame_ids, self.frame_ids[1:])):
            per_pair[frame] = {'previous_frame': previous}
            for source, by_score in scores.items():
                per_pair[frame][source] = {name: values[t] for name, values in by_score.items()}
        return {'temporal': {'camera': self.fixed_view_camera, 'per_pair': per_pair}}
