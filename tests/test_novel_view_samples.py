"""Sample plan of the single-frame novel-view benchmark: which test frames and target cameras it scores, and the
frames each test frame's window may use. Pure, CPU.

Run: conda run -n match python -m pytest -q tests/test_novel_view_samples.py
"""
import pytest

from match.novel_view.samples import FrameInputs, NoTargetCameraError, plan_samples

INPUTS = ('a', 'b')


def frames(sequence, ids, inputs=INPUTS, missing=()):
    """Frames of one sequence; the ids in `missing` have no complete inputs (e.g. no TEMPEH prediction)."""
    return [FrameInputs(frame=f, sequence=sequence, inputs=None if f in missing else inputs) for f in ids]


def all_targets(subject, frame):
    return ['x', 'y', 'z']


def test_every_frame_with_complete_inputs_is_a_test_frame_when_there_are_fewer_than_asked_for():
    plan = plan_samples({'S1': frames('seq', ['1', '2', '3', '4', '5'])}, all_targets, n_samples=100, seed=0,
                        context_length=4)

    assert [s.frame for s in plan.samples] == ['1', '2', '3', '4', '5']
    assert {s.subject for s in plan.samples} == {'S1'}


def test_a_frame_whose_neighbours_lack_inputs_gets_the_run_of_complete_frames_around_it_as_its_context():
    # 4 has no TEMPEH prediction, so 1-3 and 5-9 are separate runs; the run 1-3 is shorter than the window of 4
    plan = plan_samples({'S1': frames('seq', [str(i) for i in range(1, 10)], missing={'4'})}, all_targets,
                        n_samples=100, seed=0, context_length=4)

    assert [s.frame for s in plan.samples] == ['5', '6', '7', '8', '9']
    assert {s.run for s in plan.samples} == {('5', '6', '7', '8', '9')}
    assert [s.index for s in plan.samples] == [0, 1, 2, 3, 4]
    assert plan.n_without_inputs == 1
    assert plan.n_in_short_runs == 3


def test_runs_end_at_sequence_boundaries():
    plan = plan_samples({'S1': frames('A', ['1', '2', '3', '4']) + frames('B', ['5', '6', '7', '8'])}, all_targets,
                        n_samples=100, seed=0, context_length=4)

    assert [(s.sequence, s.run) for s in plan.samples] == [('A', ('1', '2', '3', '4'))] * 4 + [
        ('B', ('5', '6', '7', '8'))] * 4


def test_the_same_seed_gives_the_same_test_frames_and_targets_and_another_seed_others():
    captures = {'S1': frames('A', [str(i) for i in range(40)]), 'S2': frames('B', [str(i) for i in range(100, 140)])}

    def plan(seed):
        return [(s.subject, s.frame, s.target_camera)
                for s in plan_samples(captures, all_targets, n_samples=10, seed=seed, context_length=4).samples]

    assert plan(0) == plan(0)
    assert plan(0) != plan(1)
    assert len(plan(0)) == 10
    assert len(set(plan(0))) == 10


def test_samples_keep_frame_list_order():
    captures = {'S1': frames('A', [f'{i:02d}' for i in range(40)])}
    samples = plan_samples(captures, all_targets, n_samples=10, seed=3, context_length=4).samples

    assert [s.frame for s in samples] == sorted(s.frame for s in samples)


def test_the_target_camera_is_never_an_input_of_any_frame_of_the_test_frames_window():
    # Frame 3 uses x as an input. Sliding windows of 4 over frames 1-7: 1-4, 1-4, 2-5, 3-6, then 4-7 for frames 5-7.
    captures = {'S1': [FrameInputs(f, 'seq', ('a', 'x') if f == '3' else INPUTS) for f in '1234567']}

    def targets(subject, frame):
        return ['x', 'y']

    later_targets = set()
    for seed in range(20):
        by_frame = {s.frame: s.target_camera
                    for s in plan_samples(captures, targets, n_samples=7, seed=seed, context_length=4).samples}
        assert [by_frame[f] for f in '1234'] == ['y'] * 4
        later_targets |= {by_frame[f] for f in '567'}
    assert later_targets == {'x', 'y'}  # x is allowed once frame 3 is outside the window


def test_targets_come_only_from_the_cameras_available_for_the_test_frame():
    def targets(subject, frame):
        return ['y'] if frame == '2' else ['x', 'y', 'z']

    for seed in range(10):
        samples = plan_samples({'S1': frames('seq', '12345')}, targets, n_samples=5, seed=seed, context_length=4).samples
        assert {s.frame: s.target_camera for s in samples}['2'] == 'y'


def test_a_test_frame_with_no_possible_target_camera_raises():
    def targets(subject, frame):
        return ['a']  # an input camera

    with pytest.raises(NoTargetCameraError, match='frame 1'):
        plan_samples({'S1': frames('seq', '1234')}, targets, n_samples=4, seed=0, context_length=4)
