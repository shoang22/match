"""Per-frame metric plots of the sequence evaluation run (#12): CPU tests on synthetic metrics.

The plots are written next to the video from the metrics dict that becomes metrics.json.
"""
import math

import pytest
from PIL import Image

from match.sequence_eval.plots import write_metric_plots

FRAMES = ['32885', '32897', '32909', '32921']
CAMERAS = ['402040', '401905', '402031']


def image_scores(seed):
    return {'psnr': 24.0 + seed, 'ssim': 0.75 + 0.01 * seed, 'lpips': 0.23 - 0.01 * seed}


def synthetic_metrics(frames=FRAMES, cameras=CAMERAS):
    per_pair = {
        frame: {'previous_frame': previous,
                'prediction': {'raw_flicker': 0.014, 'flow_aligned_error': 0.011},
                'real': {'raw_flicker': 0.015, 'flow_aligned_error': 0.010}}
        for previous, frame in zip(frames, frames[1:])
    }
    return {
        'frames': list(frames),
        'heldout_cameras': list(cameras),
        'per_frame': {f: {c: image_scores(i + j) for j, c in enumerate(cameras)} for i, f in enumerate(frames)},
        'temporal': {'camera': cameras[0], 'per_pair': per_pair, 'summary': {}},
    }


def assert_png(path):
    with Image.open(path) as image:
        assert image.format == 'PNG'
        assert min(image.size) > 200


def test_writes_an_image_metric_plot_and_a_temporal_plot_next_to_the_video(tmp_path):
    names = write_metric_plots(tmp_path, synthetic_metrics())
    assert names == ['plot_image_metrics.png', 'plot_temporal_metrics.png']
    for name in names:
        assert_png(tmp_path / name)


def test_non_finite_values_do_not_break_the_plots(tmp_path):
    metrics = synthetic_metrics()
    metrics['per_frame'][FRAMES[1]][CAMERAS[0]]['psnr'] = math.inf  # identical images inside the mask
    metrics['temporal']['per_pair'][FRAMES[2]]['real']['flow_aligned_error'] = math.nan  # no valid pixels
    for name in write_metric_plots(tmp_path, metrics):
        assert_png(tmp_path / name)


def test_a_single_frame_still_gets_both_plots(tmp_path):
    metrics = synthetic_metrics(frames=FRAMES[:1], cameras=CAMERAS[:1])
    assert metrics['temporal']['per_pair'] == {}
    for name in write_metric_plots(tmp_path, metrics):
        assert_png(tmp_path / name)


def test_missing_metric_sections_are_an_error(tmp_path):
    metrics = synthetic_metrics()
    del metrics['temporal']
    with pytest.raises(KeyError, match='temporal'):
        write_metric_plots(tmp_path, metrics)
