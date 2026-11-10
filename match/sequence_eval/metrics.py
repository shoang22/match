"""Metric collectors of the sequence evaluation run.

A collector sees every source frame (`add_frame(ctx)`, in frame order) and returns its sections of the
metrics JSON from `finish()`, as a dict of top-level keys. Sections of different collectors must not
share keys. To add metrics, write a collector here and add it to `build_metric_collectors` in
scripts/match/eval_sequence.py.
"""
import math

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
