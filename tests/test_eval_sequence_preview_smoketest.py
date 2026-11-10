"""Preview video and metric plots of the sequence-evaluation smoke run (#12).

Skipped until the smoke run has been done (run command in configs/match/eval_sequence_smoketest.gin).
EVAL_SEQUENCE_SMOKE_DIR checks another output folder of the smoke config, e.g. one written with
  --gin_bindings "OUT_DIR = 'experiments/match/eval_sequence/<name>'"
"""
import json
import os
from pathlib import Path

import ffmpeg
import numpy as np
import pytest
from PIL import Image

OUT_DIR = Path(os.environ.get('EVAL_SEQUENCE_SMOKE_DIR', 'experiments/match/eval_sequence/smoketest'))
FRAMES = ['32885', '32897', '32909']
PREVIEW_FPS = 10  # each source frame shown for 0.1 s
PLOTS = ['plot_image_metrics.png', 'plot_temporal_metrics.png']

pytestmark = pytest.mark.skipif(
    not (OUT_DIR / 'metrics.json').exists(), reason=f'run the smoke config first (no {OUT_DIR}/metrics.json)')


@pytest.fixture(scope='module')
def metrics():
    metrics = json.loads((OUT_DIR / 'metrics.json').read_text())
    if 'preview' not in metrics:
        pytest.skip(f'{OUT_DIR}/metrics.json predates the preview video (#12); rerun the smoke config')
    return metrics


def probe_video(path):
    probe = ffmpeg.probe(str(path), count_frames=None)
    (stream,) = [s for s in probe['streams'] if s['codec_type'] == 'video']
    return stream, float(probe['format']['duration'])


def decode(path, width, height):
    out, _ = (ffmpeg.input(str(path)).output('pipe:', format='rawvideo', pix_fmt='rgb24')
              .run(capture_stdout=True, quiet=True))
    return np.frombuffer(out, np.uint8).reshape(-1, height, width, 3).astype(np.float32)


def test_preview_shows_each_source_frame_once_at_10_fps(metrics):
    stream, duration = probe_video(OUT_DIR / metrics['preview']['file'])
    assert stream['avg_frame_rate'] == f'{PREVIEW_FPS}/1'
    assert int(stream['nb_read_frames']) == len(FRAMES)
    assert duration == pytest.approx(len(FRAMES) / PREVIEW_FPS, abs=0.5 / PREVIEW_FPS)
    assert metrics['preview']['fps'] == PREVIEW_FPS
    assert metrics['preview']['output_frames'] == FRAMES


def test_preview_frames_are_grid_frames_not_blends(metrics):
    grid_stream, _ = probe_video(OUT_DIR / metrics['video']['file'])
    w, h = int(grid_stream['width']), int(grid_stream['height'])
    preview_stream, _ = probe_video(OUT_DIR / metrics['preview']['file'])
    assert (int(preview_stream['width']), int(preview_stream['height'])) == (w, h)

    grid = decode(OUT_DIR / metrics['video']['file'], w, h)
    preview = decode(OUT_DIR / metrics['preview']['file'], w, h)
    first_shown = [metrics['output_frames'].index(f) for f in FRAMES]  # grid frame where each source frame starts
    for i, k in enumerate(first_shown):
        same = np.abs(preview[i] - grid[k]).mean()
        others = [np.abs(preview[i] - grid[j]).mean() for j in first_shown if j != k]
        # Codec noise only: ~1.05 levels with all 4 panels (both videos are h264-encoded from the same image),
        # against ~6-8 between different source frames; a 50/50 blend of neighbours would sit near 3.
        assert same < 2.0, (i, same)
        assert same < 0.25 * min(others), (i, same, others)


def test_plots_of_every_per_frame_metric_are_written(metrics):
    assert metrics['plots'] == PLOTS
    for name in PLOTS:
        with Image.open(OUT_DIR / name) as image:
            assert image.format == 'PNG'
            assert min(image.size) > 200
