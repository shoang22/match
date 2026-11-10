"""Per-frame metric plots of the sequence evaluation run, written next to the videos.

Each plot has the source frame ID on the x axis, so a spike can be matched to the moment in the video
(metrics.json 'output_frames' gives the source frame of every grid video frame).
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # offscreen; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

from match.sequence_eval.metrics import IMAGE_METRICS, TEMPORAL_SCORES, TEMPORAL_SOURCES  # noqa: E402

IMAGE_METRICS_PLOT = 'plot_image_metrics.png'
TEMPORAL_METRICS_PLOT = 'plot_temporal_metrics.png'
MAX_LABELLED_FRAMES = 40  # up to this many source frames, each one is a labelled x tick

IMAGE_METRIC_LABELS = {
    'psnr': 'Masked PSNR (dB, higher is better)',
    'ssim': 'Masked SSIM (higher is better)',
    'lpips': 'Masked LPIPS (lower is better)',
}
TEMPORAL_SCORE_LABELS = {
    'raw_flicker': 'Raw flicker (lower is better)',
    'flow_aligned_error': 'Flow-aligned error (lower is better)',
}


def _finite_or_nan(value):
    """Non-finite values (inf PSNR for identical images, nan for no valid pixels) are left as gaps."""
    value = float(value)
    return value if math.isfinite(value) else math.nan


def _finish_axes(axes, frame_ids, xlabel):
    """Shared x axis over all source frames of the run, so the plots line up with each other and the video."""
    frames = [int(f) for f in frame_ids]
    margin = max((frames[-1] - frames[0]) / 50, 1)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')
    bottom = axes[-1]
    bottom.set_xlim(frames[0] - margin, frames[-1] + margin)
    bottom.ticklabel_format(axis='x', style='plain', useOffset=False)  # whole frame IDs, not 3.29e4 + offset
    if len(frames) <= MAX_LABELLED_FRAMES:
        bottom.set_xticks(frames)  # every source frame gets its own tick
        bottom.tick_params(axis='x', labelrotation=90)
    bottom.set_xlabel(xlabel)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_image_metrics(path, metrics):
    """One subplot per image metric, one line per held-out camera."""
    frames, per_frame, cameras = metrics['frames'], metrics['per_frame'], metrics['heldout_cameras']
    x = [int(f) for f in frames]
    fig, axes = plt.subplots(len(IMAGE_METRICS), 1, figsize=(10, 3 * len(IMAGE_METRICS)), sharex=True)
    for ax, name in zip(axes, IMAGE_METRICS):
        for camera in cameras:
            ax.plot(x, [_finite_or_nan(per_frame[f][camera][name]) for f in frames], marker='o', markersize=3,
                    label=f'cam {camera}')
        ax.set_ylabel(name.upper())
        ax.set_title(IMAGE_METRIC_LABELS.get(name, name))
    _finish_axes(axes, frames, 'Source frame ID')
    fig.suptitle('Image quality at the held-out cameras, inside the foreground mask')
    _save(fig, path)


def plot_temporal_metrics(path, metrics):
    """One subplot per temporal score, prediction vs. real footage at the fixed view.

    A pair's value sits at its later frame (where the change appears), as in metrics.json.
    """
    temporal = metrics['temporal']
    pairs = list(temporal['per_pair'])
    x = [int(f) for f in pairs]
    fig, axes = plt.subplots(len(TEMPORAL_SCORES), 1, figsize=(10, 3 * len(TEMPORAL_SCORES)), sharex=True)
    for ax, score in zip(axes, TEMPORAL_SCORES):
        for source in TEMPORAL_SOURCES:
            ax.plot(x, [_finite_or_nan(temporal['per_pair'][f][source][score]) for f in pairs], marker='o',
                    markersize=3, label=source)
        ax.set_ylabel('mean |change|')
        ax.set_title(TEMPORAL_SCORE_LABELS.get(score, score))
    _finish_axes(axes, metrics['frames'], 'Source frame ID (change from the previous source frame)')
    fig.suptitle(f'Temporal consistency at the fixed view (cam {temporal["camera"]})')
    _save(fig, path)


def write_metric_plots(out_dir, metrics):
    """Writes the per-frame plots of `metrics` (the metrics.json dict) into out_dir. Returns their file names.

    Raises KeyError if the image metrics or the temporal section are missing.
    """
    for section in ('per_frame', 'temporal'):
        if section not in metrics:
            raise KeyError(f"metrics has no '{section}' section to plot")
    out_dir = Path(out_dir)
    plot_image_metrics(out_dir / IMAGE_METRICS_PLOT, metrics)
    plot_temporal_metrics(out_dir / TEMPORAL_METRICS_PLOT, metrics)
    return [IMAGE_METRICS_PLOT, TEMPORAL_METRICS_PLOT]
