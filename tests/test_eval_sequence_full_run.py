"""Output contract of the full sequence evaluation (#13): all 40 frames, 3 held-out cameras.

Skipped until the full run has been done (about 25 min on one 24 GB GPU; rerun recipe in the header of
configs/match/eval_sequence_SEN_a_good_morrow.gin). EVAL_SEQUENCE_FULL_DIR checks another output folder
of the full config.
"""
import json
import math
import os
from pathlib import Path

import ffmpeg
import pytest
from PIL import Image

FULL_CONFIG = Path('configs/match/eval_sequence_SEN_a_good_morrow.gin')
OUT_DIR = Path(os.environ.get('EVAL_SEQUENCE_FULL_DIR', 'experiments/match/eval_sequence/SEN_a_good_morrow'))
FRAMES = [str(32885 + 12 * k) for k in range(40)]  # every frame of SEN_a_good_morrow_to_you_my_boy, 0.4 s apart
HELDOUT_CAMERAS = ['402040', '401905', '402031']
FIXED_VIEW_CAMERA = '402040'
INPUT_CAMERA_IDS = ['401875', '401919', '402020', '402012', '401920', '401965',
                    '401630', '401916', '401860', '402038', '401966', '401969']
VIDEO_W, VIDEO_H = 1024, 1344  # 2x2 grid of 512x640 panels under 32 px labels
PLOTS = ['plot_image_metrics.png', 'plot_temporal_metrics.png']

pytestmark = pytest.mark.skipif(
    not (OUT_DIR / 'metrics.json').exists(), reason=f'run the full config first: {FULL_CONFIG}')


@pytest.fixture(scope='module')
def metrics():
    return json.loads((OUT_DIR / 'metrics.json').read_text())


def probe_video(path):
    probe = ffmpeg.probe(str(path), count_frames=None)
    (stream,) = [s for s in probe['streams'] if s['codec_type'] == 'video']
    return stream, float(probe['format']['duration'])


def bindings(config_path):
    """The config's lines without comment lines."""
    return [line for line in config_path.read_text().splitlines() if not line.lstrip().startswith('#')]


def is_finite_float(value):
    return isinstance(value, float) and math.isfinite(value)


def test_grid_video_lasts_16_s_at_24_fps():
    stream, duration = probe_video(OUT_DIR / 'grid.mp4')
    assert stream['avg_frame_rate'] == '24/1'
    assert (int(stream['width']), int(stream['height'])) == (VIDEO_W, VIDEO_H)
    assert int(stream['nb_read_frames']) == 384  # 40 frames x 0.4 s x 24 fps
    assert abs(duration - 16.0) <= 1 / 24


def test_preview_shows_every_source_frame_once_at_10_fps(metrics):
    stream, duration = probe_video(OUT_DIR / 'preview.mp4')
    assert stream['avg_frame_rate'] == '10/1'
    assert (int(stream['width']), int(stream['height'])) == (VIDEO_W, VIDEO_H)
    assert int(stream['nb_read_frames']) == len(FRAMES)
    assert abs(duration - 4.0) <= 0.05
    assert metrics['preview']['output_frames'] == FRAMES


def test_grid_has_the_four_panels(metrics):
    assert metrics['video']['panels'] == [
        f'MATCH prediction (cam {FIXED_VIEW_CAMERA})', f'Real held-out (cam {FIXED_VIEW_CAMERA})',
        f'TEMPEH coarse mesh (cam {FIXED_VIEW_CAMERA})', 'Orbit']
    assert sorted(set(metrics['output_frames']), key=int) == FRAMES


def test_image_metrics_are_finite_for_every_frame_and_heldout_camera(metrics):
    assert metrics['frames'] == FRAMES
    assert metrics['heldout_cameras'] == HELDOUT_CAMERAS
    assert metrics['fixed_view_camera'] == FIXED_VIEW_CAMERA
    assert sorted(metrics['per_frame'], key=int) == FRAMES
    for frame in FRAMES:
        assert sorted(metrics['per_frame'][frame]) == sorted(HELDOUT_CAMERAS)
        for cam in HELDOUT_CAMERAS:
            scores = metrics['per_frame'][frame][cam]
            assert sorted(scores) == ['lpips', 'psnr', 'ssim']
            assert all(is_finite_float(v) for v in scores.values()), (frame, cam, scores)


def test_image_summary_has_mean_and_3_worst_frames_per_camera(metrics):
    for cam in HELDOUT_CAMERAS:
        for name in ['psnr', 'ssim', 'lpips']:
            summary = metrics['summary'][cam][name]
            assert is_finite_float(summary['mean'])
            assert len(summary['worst_frames']) == 3
            assert all(w['frame'] in FRAMES and is_finite_float(w['value']) for w in summary['worst_frames'])


def test_temporal_scores_are_finite_for_prediction_and_real_on_every_pair(metrics):
    temporal = metrics['temporal']
    assert temporal['camera'] == FIXED_VIEW_CAMERA
    assert list(temporal['per_pair']) == FRAMES[1:]
    for entry in temporal['per_pair'].values():
        for source in ['prediction', 'real']:
            assert sorted(entry[source]) == ['flow_aligned_error', 'raw_flicker']
            assert all(is_finite_float(v) and v >= 0 for v in entry[source].values()), entry
    for source in ['prediction', 'real']:
        for score in ['raw_flicker', 'flow_aligned_error']:
            summary = temporal['summary'][source][score]
            assert is_finite_float(summary['mean'])
            assert len(summary['worst_frames']) == 3


def test_pretrained_model_with_the_12_standard_inputs_on_every_frame(metrics):
    assert metrics['model'] == {'path': 'assets/pretrained_models/match/uv_res_1024', 'checkpoint': '860000'}
    assert metrics['input_cameras'] == {f: INPUT_CAMERA_IDS for f in FRAMES}


def test_plots_and_config_copies_are_written(metrics):
    assert metrics['plots'] == PLOTS
    for name in PLOTS:
        with Image.open(OUT_DIR / name) as image:
            assert image.format == 'PNG'
    # Same bindings as the committed config; its header comment may have been edited since the run.
    assert bindings(OUT_DIR / 'config_0.gin') == bindings(FULL_CONFIG)
    assert f"OUT_DIR = '{OUT_DIR}'" in (OUT_DIR / 'config_all.gin').read_text()
