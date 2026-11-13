"""What a windowed (temporal) sequence-evaluation run adds to the output contract, and that temporal MATCH at
initialization scores like per-frame MATCH.

Runs checked (each test skips until its runs have been done, see eval_sequence_runs.py):
- the per-frame smoke run (configs/match/eval_sequence_smoketest.gin, EVAL_SEQUENCE_SMOKE_DIR),
- the temporal smoke run (configs/match/eval_sequence_smoketest_temporal.gin, EVAL_SEQUENCE_TEMPORAL_SMOKE_DIR):
  temporal MATCH at initialization, sliding windows; the 3 smoke frames are one window,
- the same on 6 frames in sliding and in block mode (EVAL_SEQUENCE_TEMPORAL_SLIDING6_DIR, ..._BLOCK6_DIR),
- the `background 1` pilot (#36): flicker-only on all 30 frames of `background 1`, temporal MATCH at initialization in
  sliding windows (configs/match/eval_sequence_background1_temporal.gin, EVAL_SEQUENCE_BACKGROUND1_TEMPORAL_DIR),
  and the same with per-frame MATCH (configs/match/eval_sequence_background1_flicker.gin,
  EVAL_SEQUENCE_BACKGROUND1_FLICKER_DIR).
Everything a per-frame run writes is checked on the temporal runs in test_eval_sequence_outputs.py.
"""
import pytest

from eval_sequence_runs import (BACKGROUND1_FLICKER, BACKGROUND1_TEMPORAL, SMOKE, TEMPORAL_BLOCK6, TEMPORAL_SLIDING6,
                                TEMPORAL_SMOKE, TOLERANCE, load_metrics)

# (temporal MATCH at initialization, per-frame MATCH) on the same frames.
RUN_PAIRS = [(TEMPORAL_SMOKE, SMOKE), (BACKGROUND1_TEMPORAL, BACKGROUND1_FLICKER)]


@pytest.fixture(scope='module')
def temporal():
    return load_metrics(TEMPORAL_SMOKE)


@pytest.fixture(scope='module')
def per_frame():
    return load_metrics(SMOKE)


@pytest.fixture(scope='module', params=RUN_PAIRS, ids=lambda pair: pair[0].name)
def run_pair(request):
    """metrics.json of a temporal run at initialization and of the per-frame run on the same frames."""
    temporal_run, per_frame_run = request.param
    return load_metrics(temporal_run), load_metrics(per_frame_run)


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


def test_the_background1_pilot_records_the_sliding_window_and_slot_of_every_one_of_its_30_frames():
    inference = load_metrics(BACKGROUND1_TEMPORAL)['inference']
    frames = BACKGROUND1_TEMPORAL.frames
    # Windows of 4 at frames 0..26, one per frame from frame 1 to 27: frame t is slot 1 of the window starting at t-1
    # (one frame before, two after), except at the ends, where the window is shifted inward.
    starts = [0] + list(range(27)) + [26, 26]
    slots = [0] + [1] * 27 + [2, 3]

    assert inference['mode'] == 'sliding'
    assert (inference['clip_length'], inference['window_length'], inference['n_windows']) == (4, 4, 27)
    assert inference['per_frame'] == {
        frame: {'window_start': start, 'window_start_frame': frames[start], 'slot': slot}
        for frame, start, slot in zip(frames, starts, slots)}


@pytest.mark.parametrize('per_frame_run', [SMOKE, BACKGROUND1_FLICKER], ids=lambda run: run.name)
def test_a_per_frame_run_has_no_inference_section(per_frame_run):
    assert 'inference' not in load_metrics(per_frame_run)


def test_temporal_match_at_initialization_gets_per_frame_matchs_image_scores(temporal, per_frame):
    for frame, cameras in per_frame['per_frame'].items():
        for camera, scores in cameras.items():
            for name, value in scores.items():
                assert temporal['per_frame'][frame][camera][name] == pytest.approx(value, abs=TOLERANCE[name]), \
                    (frame, camera, name)


def test_temporal_match_at_initialization_gets_per_frame_matchs_temporal_scores(run_pair):
    temporal, per_frame = run_pair
    for frame, entry in per_frame['temporal']['per_pair'].items():
        for source in ['prediction', 'real']:
            for score, value in entry[source].items():
                assert temporal['temporal']['per_pair'][frame][source][score] == pytest.approx(
                    value, abs=TOLERANCE[score]), (frame, source, score)


def test_temporal_match_at_initialization_gets_per_frame_matchs_jitter(run_pair):
    temporal, per_frame = run_pair
    for frame, entry in per_frame['jitter']['per_frame'].items():
        for source in ['tempeh', 'offset', 'final']:
            assert temporal['jitter']['per_frame'][frame][source] == pytest.approx(
                entry[source], abs=TOLERANCE['jitter_mm']), (frame, source)
