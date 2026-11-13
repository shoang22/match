"""What a windowed (temporal) sequence-evaluation run adds to the output contract, and that temporal MATCH at
initialization scores like per-frame MATCH.

Runs checked (each test skips until its runs have been done, see eval_sequence_runs.py):
- the per-frame smoke run (configs/match/eval_sequence_smoketest.gin, EVAL_SEQUENCE_SMOKE_DIR),
- the temporal smoke run (configs/match/eval_sequence_smoketest_temporal.gin, EVAL_SEQUENCE_TEMPORAL_SMOKE_DIR):
  temporal MATCH at initialization, sliding windows; the 3 smoke frames are one window,
- the same on 6 frames in sliding and in block mode (EVAL_SEQUENCE_TEMPORAL_SLIDING6_DIR, ..._BLOCK6_DIR).
Everything a per-frame run writes is checked on the temporal runs in test_eval_sequence_outputs.py.
"""
import pytest

from eval_sequence_runs import SMOKE, TEMPORAL_BLOCK6, TEMPORAL_SLIDING6, TEMPORAL_SMOKE, load_metrics

# How far temporal MATCH at initialization may score from per-frame MATCH. At initialization the temporal sublayers
# are exact no-ops, so the only differences are GPU nondeterminism and fp16 numerics of running the 3 frames as one
# batch instead of one at a time. Each bound is ~3x the largest noise floor measured on the smoke frames (largest
# absolute difference over frames, cameras and sources) in two ways:
#                         run to run (per-frame twice)   batch size (per-frame MATCH, one frame vs a window of 3)
#   psnr (dB)             0.0076                         0.0070
#   ssim                  0.0032                         0.0028
#   lpips                 5.2e-4                         6.3e-4
#   raw flicker           6.4e-6                         4.1e-5
#   flow-aligned error    3.9e-6                         1.4e-5
#   jitter (mm)           8.5e-4                         1.8e-3
# Temporal MATCH at initialization vs per-frame MATCH measured 0.0045 dB, 0.0040, 4.0e-4, 8.1e-6, 9.8e-6, 1.6e-3 mm.
TOLERANCE = {
    'psnr': 0.025,  # dB
    'ssim': 0.01,
    'lpips': 2e-3,
    'raw_flicker': 1.5e-4,
    'flow_aligned_error': 1.5e-4,
    'jitter_mm': 0.005,
}


@pytest.fixture(scope='module')
def temporal():
    return load_metrics(TEMPORAL_SMOKE)


@pytest.fixture(scope='module')
def per_frame():
    return load_metrics(SMOKE)


def test_a_temporal_run_records_its_inference_mode_and_each_frames_window_start_and_slot(temporal):
    first, second, third = TEMPORAL_SMOKE.frames
    inference = temporal['inference']

    assert inference['mode'] == 'sliding'
    assert inference['clip_length'] == 4
    assert inference['window_length'] == 3  # 3 frames, fewer than the window: one clip of all 3
    assert inference['n_windows'] == 1
    assert inference['per_frame'] == {
        first: {'window_start': 0, 'window_start_frame': first, 'slot': 0},
        second: {'window_start': 0, 'window_start_frame': first, 'slot': 1},
        third: {'window_start': 0, 'window_start_frame': first, 'slot': 2}}


@pytest.mark.parametrize('run, n_windows, starts, slots', [
    # one frame before to two after, shifted inward at both ends: windows at frames 0, 1 and 2
    (TEMPORAL_SLIDING6, 3, [0, 0, 1, 2, 2, 2], [0, 1, 1, 1, 2, 3]),
    # frames 0-3, then the last block shifted inward to frames 2-5, which keeps only frames 4 and 5
    (TEMPORAL_BLOCK6, 2, [0, 0, 0, 0, 2, 2], [0, 1, 2, 3, 2, 3]),
], ids=lambda value: value.name if hasattr(value, 'name') else None)
def test_a_six_frame_run_records_the_window_and_slot_of_every_frame(run, n_windows, starts, slots):
    inference = load_metrics(run)['inference']
    frames = run.frames

    assert inference['mode'] == run.name.removeprefix('temporal_').removesuffix('6')
    assert (inference['window_length'], inference['n_windows']) == (4, n_windows)
    assert inference['per_frame'] == {
        frame: {'window_start': start, 'window_start_frame': frames[start], 'slot': slot}
        for frame, start, slot in zip(frames, starts, slots)}


def test_a_per_frame_run_has_no_inference_section(per_frame):
    assert 'inference' not in per_frame


def test_temporal_match_at_initialization_gets_per_frame_matchs_image_scores(temporal, per_frame):
    for frame, cameras in per_frame['per_frame'].items():
        for camera, scores in cameras.items():
            for name, value in scores.items():
                assert temporal['per_frame'][frame][camera][name] == pytest.approx(value, abs=TOLERANCE[name]), \
                    (frame, camera, name)


def test_temporal_match_at_initialization_gets_per_frame_matchs_temporal_scores(temporal, per_frame):
    for frame, entry in per_frame['temporal']['per_pair'].items():
        for source in ['prediction', 'real']:
            for score, value in entry[source].items():
                assert temporal['temporal']['per_pair'][frame][source][score] == pytest.approx(
                    value, abs=TOLERANCE[score]), (frame, source, score)


def test_temporal_match_at_initialization_gets_per_frame_matchs_jitter(temporal, per_frame):
    for frame, entry in per_frame['jitter']['per_frame'].items():
        for source in ['tempeh', 'offset', 'final']:
            assert temporal['jitter']['per_frame'][frame][source] == pytest.approx(
                entry[source], abs=TOLERANCE['jitter_mm']), (frame, source)
