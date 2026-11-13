"""Output contract of the sequence evaluation, checked on the runs below.

Smoke run: the first 3 frames of SEN_a_good_morrow_to_you_my_boy, held-out camera 402040.
Full run: all 40 frames, held-out cameras 402040, 401905, 402031.
Flicker-only run: the smoke run's frames with no held-out cameras; input camera 401875 is the fixed view, and
there are temporal and jitter scores but no image-quality scores.
Temporal runs: the smoke run with temporal MATCH in sliding windows (3 frames, and 6 frames) and in block windows
(6 frames). They must write everything a per-frame run writes; their extras are checked in
test_eval_sequence_temporal_outputs.py.
`background 1` runs: flicker-only on all 30 frames of the `background 1` capture (input camera 0 as the fixed view),
with per-frame MATCH and with temporal MATCH in sliding windows.
Each run's tests skip until it has been done (see eval_sequence_runs.py). The video content of the smoke and
flicker-only runs is checked further in test_eval_sequence_smoketest.py.
"""
import math
import re

import pytest
from PIL import Image

from eval_sequence_runs import (BACKGROUND1_FLICKER, BACKGROUND1_TEMPORAL, FLICKER, FPS, FULL, IMAGE_METRICS_PLOT,
                                PIX_FMT, PREVIEW_FPS, SMOKE, TEMPORAL_BLOCK6, TEMPORAL_SLIDING6, TEMPORAL_SMOKE,
                                VIDEO_H, VIDEO_W, config_bindings, load_metrics, probe_video)
from tce049 import REPO_ROOT

# The spec's metrics and whether a higher value is better. Runs may record more.
IMAGE_METRICS = {'psnr': True, 'ssim': True, 'lpips': False}
TEMPORAL_SCORES = ['raw_flicker', 'flow_aligned_error']  # lower is better
TEMPORAL_SOURCES = ['prediction', 'real']
JITTER_SOURCES = ['tempeh', 'offset', 'final']  # final position = TEMPEH part + offset; lower is better
N_WORST_FRAMES = 3
FLICKER_ONLY_RUNS = [FLICKER, BACKGROUND1_FLICKER, BACKGROUND1_TEMPORAL]


@pytest.fixture(scope='module', params=[SMOKE, FULL, FLICKER, TEMPORAL_SMOKE, TEMPORAL_SLIDING6, TEMPORAL_BLOCK6,
                                         BACKGROUND1_FLICKER, BACKGROUND1_TEMPORAL],
                ids=lambda run: run.name)
def run(request):
    return request.param


@pytest.fixture(scope='module')
def metrics(run):
    return load_metrics(run)


@pytest.fixture(scope='module', params=[SMOKE, FULL], ids=lambda run: run.name)
def heldout_run(request):
    """The runs with held-out cameras, which score image quality there."""
    return request.param


@pytest.fixture(scope='module')
def heldout_metrics(heldout_run):
    return load_metrics(heldout_run)


def is_finite_float(value):
    return isinstance(value, float) and math.isfinite(value)


def assert_summary(summary, values, higher_is_better):
    """Mean over the frames, and the worst frames, worst first."""
    assert summary['mean'] == pytest.approx(sum(values.values()) / len(values))
    worst = sorted(values, key=values.get, reverse=not higher_is_better)[:N_WORST_FRAMES]
    assert summary['worst_frames'] == [{'frame': f, 'value': values[f]} for f in worst]


def test_grid_video_has_the_true_duration_at_24_fps(run, metrics):
    assert metrics['video']['capture_fps'] == run.capture_fps
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
    # each source frame is held for its spacing at 24 fps (Ava: 2/15 s = 3.2 output frames), rounded down or up
    holds = [output_frames.count(f) for f in run.frames]
    assert all(h in run.holds for h in holds), holds


def test_grid_has_the_four_panels(run, metrics):
    assert metrics['video']['panels'] == list(run.panels)


def test_the_fixed_view_camera_and_whether_it_is_held_out_or_an_input_are_recorded(run, metrics):
    assert metrics['frames'] == run.frames
    assert metrics['segments'] == list(run.segments)
    assert metrics['heldout_cameras'] == run.heldout_cameras
    assert metrics['fixed_view_camera'] == run.fixed_view_camera
    assert metrics['fixed_view_kind'] == run.fixed_view_kind


@pytest.mark.parametrize('flicker_run', FLICKER_ONLY_RUNS, ids=lambda run: run.name)
def test_a_flicker_only_run_has_no_image_quality_scores_and_its_fixed_view_is_an_input_camera(flicker_run):
    metrics = load_metrics(flicker_run)
    assert metrics['heldout_cameras'] == []
    assert 'per_frame' not in metrics and 'summary' not in metrics
    # ... and says so (story 46)
    assert metrics['quality_scores'].startswith('not computed'), metrics['quality_scores']
    assert f"(cam {flicker_run.fixed_view_camera}) is an input camera" in metrics['quality_scores']
    assert IMAGE_METRICS_PLOT not in metrics['plots']
    assert not (flicker_run.out_dir / IMAGE_METRICS_PLOT).exists()
    assert metrics['fixed_view_kind'] == 'input'
    assert all(flicker_run.fixed_view_camera in cameras for cameras in metrics['input_cameras'].values())


def test_image_metrics_are_finite_for_every_frame_and_heldout_camera(heldout_run, heldout_metrics):
    run, metrics = heldout_run, heldout_metrics
    assert sorted(metrics['per_frame'], key=int) == run.frames
    for frame in run.frames:
        assert sorted(metrics['per_frame'][frame]) == sorted(run.heldout_cameras)
        for camera in run.heldout_cameras:
            scores = metrics['per_frame'][frame][camera]
            assert set(IMAGE_METRICS) <= set(scores)
            assert all(is_finite_float(v) for v in scores.values()), (frame, camera, scores)


def test_image_summary_has_the_mean_and_worst_frames_per_camera(heldout_run, heldout_metrics):
    run, metrics = heldout_run, heldout_metrics
    for camera in run.heldout_cameras:
        for name, higher_is_better in IMAGE_METRICS.items():
            values = {f: metrics['per_frame'][f][camera][name] for f in run.frames}
            assert_summary(metrics['summary'][camera][name], values, higher_is_better)


def test_temporal_scores_are_finite_for_prediction_and_real_on_every_pair(run, metrics):
    temporal = metrics['temporal']
    assert temporal['camera'] == run.fixed_view_camera
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


def test_jitter_is_finite_for_every_source_on_every_triple_of_frames(run, metrics):
    jitter = metrics['jitter']
    assert jitter['units'] == 'mm'
    # each triple of consecutive source frames is keyed by its middle frame
    assert list(jitter['per_frame']) == run.frames[1:-1]
    for previous, frame, following in zip(run.frames, run.frames[1:], run.frames[2:]):
        entry = jitter['per_frame'][frame]
        assert (entry['previous_frame'], entry['next_frame']) == (previous, following)
        assert all(is_finite_float(entry[s]) and entry[s] >= 0 for s in JITTER_SOURCES), entry


def test_jitter_averages_over_the_face_skin_texels(metrics):
    texels = metrics['jitter'].get('texels')
    if texels is None:
        pytest.skip('the run predates the jitter texel region; rerun it')
    # The texels of uv_res_1024's grid whose triangle has every vertex in the fitting_region_skin vertex group
    # (assets/ava256/vertex_groups_with_neck_and_scalp.json), out of every texel the masked Gaussians hold.
    assert texels == {'region': 'fitting_region_skin', 'n_texels': 379_891, 'n_masked_texels': 1_048_528}


def test_jitter_summary_has_the_mean_and_worst_frames(run, metrics):
    jitter = metrics['jitter']
    for source in JITTER_SOURCES:
        values = {f: jitter['per_frame'][f][source] for f in run.frames[1:-1]}
        assert_summary(jitter['summary'][source], values, higher_is_better=False)


def test_final_jitter_is_at_most_the_tempeh_plus_the_offset_jitter(metrics):
    # Per texel |d2 final| <= |d2 tempeh| + |d2 offset|, so the means obey it too, if offset = final - tempeh.
    for frame, entry in metrics['jitter']['per_frame'].items():
        assert entry['final'] <= entry['tempeh'] + entry['offset'] + 1e-3, (frame, entry)


def test_tempeh_jitter_is_in_millimetres(run, metrics):
    low, high = run.tempeh_jitter_range_mm
    for frame, entry in metrics['jitter']['per_frame'].items():
        assert low <= entry['tempeh'] <= high, (frame, entry)


def test_pretrained_model_with_the_12_standard_inputs_on_every_frame(run, metrics):
    assert metrics['model'] == {'path': 'assets/pretrained_models/match/uv_res_1024', 'checkpoint': '860000'}
    assert metrics['input_cameras'] == {f: list(run.input_cameras) for f in run.frames}


def test_plots_of_every_per_frame_metric_are_written(run, metrics):
    assert metrics['plots'] == list(run.plots)
    for name in run.plots:
        with Image.open(run.out_dir / name) as image:
            assert image.format == 'PNG'
            assert min(image.size) > 200


def test_gin_configs_are_copied(run, metrics):
    # Same bindings as the committed config; its header comment may have been edited since the run.
    assert config_bindings(run.out_dir / 'config_0.gin') == config_bindings(REPO_ROOT / run.config)
    # The run wrote to the folder the tests read, however either path was spelled (relative to the repo root, or
    # absolute, e.g. through EVAL_SEQUENCE_*_DIR or a worktree's experiments/ symlink).
    (out_dir,) = re.findall(r"^OUT_DIR = '(.*)'$", (run.out_dir / 'config_all.gin').read_text(), flags=re.M)
    assert (REPO_ROOT / out_dir).resolve() == run.out_dir.resolve()
