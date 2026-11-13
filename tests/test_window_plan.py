"""Window plan of windowed sequence inference: which window and slot produce each output frame. Pure, CPU.

Run: conda run -n match python -m pytest -q tests/test_window_plan.py
"""
import pytest

from match.sequence_eval.windows import WindowSlot, window_plan


def starts_and_slots(plan):
    return [(w.start, w.slot) for w in plan]


def test_sliding_mode_keeps_each_interior_frame_from_the_window_one_frame_before_to_two_after():
    plan = window_plan(n_frames=10, clip_length=4, mode='sliding')

    assert plan[4] == WindowSlot(start=3, slot=1, length=4)
    assert plan[5] == WindowSlot(start=4, slot=1, length=4)


def test_sliding_windows_shift_inward_at_both_edges():
    plan = window_plan(n_frames=10, clip_length=4, mode='sliding')

    assert starts_and_slots(plan) == [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3)]


def test_block_mode_keeps_every_frame_of_non_overlapping_windows():
    plan = window_plan(n_frames=8, clip_length=4, mode='block')

    assert starts_and_slots(plan) == [(0, 0), (0, 1), (0, 2), (0, 3), (4, 0), (4, 1), (4, 2), (4, 3)]


def test_block_mode_shifts_the_last_block_inward_when_the_length_is_not_a_multiple_of_the_window():
    plan = window_plan(n_frames=10, clip_length=4, mode='block')

    # The last window covers frames 6-9; frames 6 and 7 still come from the block starting at 4.
    assert starts_and_slots(plan) == [(0, 0), (0, 1), (0, 2), (0, 3), (4, 0), (4, 1), (4, 2), (4, 3), (6, 2), (6, 3)]


@pytest.mark.parametrize('mode', ['sliding', 'block'])
def test_a_sequence_as_long_as_the_window_is_one_window(mode):
    plan = window_plan(n_frames=4, clip_length=4, mode=mode)

    assert starts_and_slots(plan) == [(0, 0), (0, 1), (0, 2), (0, 3)]


@pytest.mark.parametrize('mode', ['sliding', 'block'])
def test_a_sequence_shorter_than_the_window_is_one_shorter_window(mode):
    plan = window_plan(n_frames=3, clip_length=4, mode=mode)

    assert starts_and_slots(plan) == [(0, 0), (0, 1), (0, 2)]
    assert {w.length for w in plan} == {3}


@pytest.mark.parametrize('mode', ['sliding', 'block'])
@pytest.mark.parametrize('n_frames', [1, 2, 3, 4, 5, 7, 8, 9, 13, 40])
def test_every_frame_appears_exactly_once_in_the_output(mode, n_frames):
    plan = window_plan(n_frames=n_frames, clip_length=4, mode=mode)

    assert len(plan) == n_frames
    for t, w in enumerate(plan):
        assert w.length == min(4, n_frames)
        assert 0 <= w.start <= n_frames - w.length
        assert w.start + w.slot == t  # the kept slot is frame t itself
        assert 0 <= w.slot < w.length


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match='mode'):
        window_plan(n_frames=5, clip_length=4, mode='overlap')
