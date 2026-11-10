"""Output contract of the sequence evaluation, checked on both the smoke run and the full run.

Smoke run: the first 3 frames of SEN_a_good_morrow_to_you_my_boy, held-out camera 402040.
Full run: all 40 frames, held-out cameras 402040, 401905, 402031.
Each run's tests skip until it has been done (see eval_sequence_runs.py). The smoke run's video content is
checked further in test_eval_sequence_smoketest.py.
"""
import math

import pytest
from PIL import Image

from eval_sequence_runs import (FPS, FULL, PANELS, PIX_FMT, PLOTS, PREVIEW_FPS, SMOKE, VIDEO_H, VIDEO_W,
                                config_bindings, load_metrics, probe_video)
from tce049 import FIXED_VIEW_CAMERA, INPUT_CAMERA_IDS, REPO_ROOT

# The spec's metrics and whether a higher value is better. Runs may record more.
IMAGE_METRICS = {'psnr': True, 'ssim': True, 'lpips': False}
TEMPORAL_SCORES = ['raw_flicker', 'flow_aligned_error']  # lower is better
TEMPORAL_SOURCES = ['prediction', 'real']
N_WORST_FRAMES = 3


@pytest.fixture(scope='module', params=[SMOKE, FULL], ids=lambda run: run.name)
def run(request):
    return request.param


@pytest.fixture(scope='module')
def metrics(run):
    return load_metrics(run)


def is_finite_float(value):
    return isinstance(value, float) and math.isfinite(value)


def assert_summary(summary, values, higher_is_better):
    """Mean over the frames, and the worst frames, worst first."""
    assert summary['mean'] == pytest.approx(sum(values.values()) / len(values))
    worst = sorted(values, key=values.get, reverse=not higher_is_better)[:N_WORST_FRAMES]
    assert summary['worst_frames'] == [{'frame': f, 'value': values[f]} for f in worst]


def test_grid_video_has_the_true_duration_at_24_fps(run, metrics):
    stream, duration = probe_video(run.out_dir / metrics['video']['file'])
    assert stream['avg_frame_rate'] == f'{FPS}/1'
    assert (int(stream['width']), int(stream['height'])) == (VIDEO_W, VIDEO_H)
    assert stream['pix_fmt'] == PIX_FMT
    assert int(stream['nb_read_frames']) == run.n_output_frames
    assert abs(duration - run.duration) <= 1 / FPS


def test_preview_shows_every_source_frame_once_at_10_fps(run, metrics):
    stream, duration = probe_video(run.out_dir / metrics['preview']['file'])
    assert stream['avg_frame_rate'] == f'{PREVIEW_FPS}/1'
    assert (int(stream['width']), int(stream['height'])) == (VIDEO_W, VIDEO_H)
    assert stream['pix_fmt'] == PIX_FMT
    assert int(stream['nb_read_frames']) == len(run.frames)
    assert duration == pytest.approx(len(run.frames) / PREVIEW_FPS, abs=0.5 / PREVIEW_FPS)
    assert metrics['preview']['fps'] == PREVIEW_FPS
    assert metrics['preview']['output_frames'] == run.frames


def test_every_grid_video_frame_records_its_source_frame(run, metrics):
    output_frames = metrics['output_frames']
    assert len(output_frames) == run.n_output_frames
    assert sorted(set(output_frames), key=int) == run.frames
    assert output_frames == sorted(output_frames, key=int)
    # each source frame is held for 0.4 s = 9.6 output frames, rounded to 9 or 10
    holds = [output_frames.count(f) for f in run.frames]
    assert all(h in (9, 10) for h in holds), holds


def test_grid_has_the_four_panels(metrics):
    assert metrics['video']['panels'] == PANELS


def test_image_metrics_are_finite_for_every_frame_and_heldout_camera(run, metrics):
    assert metrics['frames'] == run.frames
    assert metrics['heldout_cameras'] == run.heldout_cameras
    assert metrics['fixed_view_camera'] == FIXED_VIEW_CAMERA
    assert sorted(metrics['per_frame'], key=int) == run.frames
    for frame in run.frames:
        assert sorted(metrics['per_frame'][frame]) == sorted(run.heldout_cameras)
        for camera in run.heldout_cameras:
            scores = metrics['per_frame'][frame][camera]
            assert set(IMAGE_METRICS) <= set(scores)
            assert all(is_finite_float(v) for v in scores.values()), (frame, camera, scores)


def test_image_summary_has_the_mean_and_worst_frames_per_camera(run, metrics):
    for camera in run.heldout_cameras:
        for name, higher_is_better in IMAGE_METRICS.items():
            values = {f: metrics['per_frame'][f][camera][name] for f in run.frames}
            assert_summary(metrics['summary'][camera][name], values, higher_is_better)


def test_temporal_scores_are_finite_for_prediction_and_real_on_every_pair(run, metrics):
    temporal = metrics['temporal']
    assert temporal['camera'] == FIXED_VIEW_CAMERA
    # each pair is keyed by its later source frame
    assert list(temporal['per_pair']) == run.frames[1:]
    for previous, frame in zip(run.frames, run.frames[1:]):
        entry = temporal['per_pair'][frame]
        assert entry['previous_frame'] == previous
        for source in TEMPORAL_SOURCES:
            assert set(TEMPORAL_SCORES) <= set(entry[source])
            assert all(is_finite_float(v) and v >= 0 for v in entry[source].values()), entry


def test_temporal_summary_has_the_mean_and_worst_frames(run, metrics):
    temporal = metrics['temporal']
    for source in TEMPORAL_SOURCES:
        for score in TEMPORAL_SCORES:
            values = {f: temporal['per_pair'][f][source][score] for f in run.frames[1:]}
            assert_summary(temporal['summary'][source][score], values, higher_is_better=False)


def test_pretrained_model_with_the_12_standard_inputs_on_every_frame(run, metrics):
    assert metrics['model'] == {'path': 'assets/pretrained_models/match/uv_res_1024', 'checkpoint': '860000'}
    assert metrics['input_cameras'] == {f: INPUT_CAMERA_IDS for f in run.frames}


def test_plots_of_every_per_frame_metric_are_written(run, metrics):
    assert metrics['plots'] == PLOTS
    for name in PLOTS:
        with Image.open(run.out_dir / name) as image:
            assert image.format == 'PNG'
            assert min(image.size) > 200


def test_gin_configs_are_copied(run, metrics):
    # Same bindings as the committed config; its header comment may have been edited since the run.
    assert config_bindings(run.out_dir / 'config_0.gin') == config_bindings(REPO_ROOT / run.config)
    assert f"OUT_DIR = '{run.out_dir_setting}'" in (run.out_dir / 'config_all.gin').read_text()
