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


def all_targets(frame):
    return ['x', 'y', 'z']


def on_every_subject(targets, captures):
    """The target-camera callable of every subject of captures (plan_samples' target_cameras)."""
    return {subject: targets for subject in captures}


def test_every_frame_with_complete_inputs_is_a_test_frame_when_there_are_fewer_than_asked_for():
    plan = plan_samples({'S1': frames('seq', ['1', '2', '3', '4', '5'])}, {'S1': all_targets}, n_samples=100,
                        seed=0, context_length=4)

    assert [s.frame for s in plan.samples] == ['1', '2', '3', '4', '5']
    assert {s.subject for s in plan.samples} == {'S1'}


def test_a_frame_whose_neighbours_lack_inputs_gets_the_run_of_complete_frames_around_it_as_its_context():
    # 4 has no TEMPEH prediction, so 1-3 and 5-9 are separate runs; the run 1-3 is shorter than the window of 4
    plan = plan_samples({'S1': frames('seq', [str(i) for i in range(1, 10)], missing={'4'})}, {'S1': all_targets},
                        n_samples=100, seed=0, context_length=4)

    assert [s.frame for s in plan.samples] == ['5', '6', '7', '8', '9']
    assert {s.run for s in plan.samples} == {('5', '6', '7', '8', '9')}
    assert [s.index for s in plan.samples] == [0, 1, 2, 3, 4]
    assert plan.n_without_inputs == 1
    assert plan.n_in_short_runs == 3


def test_runs_end_at_sequence_boundaries():
    plan = plan_samples({'S1': frames('A', ['1', '2', '3', '4']) + frames('B', ['5', '6', '7', '8'])},
                        {'S1': all_targets},
                        n_samples=100, seed=0, context_length=4)

    assert [(s.sequence, s.run) for s in plan.samples] == [('A', ('1', '2', '3', '4'))] * 4 + [
        ('B', ('5', '6', '7', '8'))] * 4



def test_runs_end_where_the_input_cameras_change_so_every_window_has_one_camera_selection():
    # A temporal window aligns image tokens by input camera: from frame 5 on, camera 'b' is replaced by 'c'.
    sequence = frames('seq', ['1', '2', '3', '4']) + frames('seq', ['5', '6', '7', '8'], inputs=('a', 'c'))
    plan = plan_samples({'S1': sequence}, {'S1': all_targets}, n_samples=100, seed=0, context_length=4)

    assert [s.run for s in plan.samples] == [('1', '2', '3', '4')] * 4 + [('5', '6', '7', '8')] * 4

def test_the_same_seed_gives_the_same_test_frames_and_targets_and_another_seed_others():
    captures = {'S1': frames('A', [str(i) for i in range(40)]), 'S2': frames('B', [str(i) for i in range(100, 140)])}

    def plan(seed):
        return [(s.subject, s.frame, s.target_camera)
                for s in plan_samples(captures, on_every_subject(all_targets, captures), n_samples=10, seed=seed,
                                      context_length=4).samples]

    assert plan(0) == plan(0)
    assert plan(0) != plan(1)
    assert len(plan(0)) == 10
    assert len(set(plan(0))) == 10


def test_samples_keep_frame_list_order():
    captures = {'S1': frames('A', [f'{i:02d}' for i in range(40)])}
    samples = plan_samples(captures, on_every_subject(all_targets, captures), n_samples=10, seed=3,
                           context_length=4).samples

    assert [s.frame for s in samples] == sorted(s.frame for s in samples)


def test_the_target_camera_is_never_an_input_of_the_test_frames_window():
    # every frame of a run (so of a window) has the same inputs, here a and x
    captures = {'S1': frames('seq', '1234567', inputs=('a', 'x'))}

    def targets(frame):
        return ['x', 'y']

    for seed in range(20):
        samples = plan_samples(captures, {'S1': targets}, n_samples=7, seed=seed, context_length=4).samples
        assert [s.target_camera for s in samples] == ['y'] * 7


def test_targets_come_only_from_the_cameras_available_for_the_test_frame():
    def targets(frame):
        return ['y'] if frame == '2' else ['x', 'y', 'z']

    for seed in range(10):
        samples = plan_samples({'S1': frames('seq', '12345')}, {'S1': targets}, n_samples=5, seed=seed,
                               context_length=4).samples
        assert {s.frame: s.target_camera for s in samples}['2'] == 'y'


def test_a_test_frame_with_no_possible_target_camera_raises():
    def targets(frame):
        return ['a']  # an input camera

    with pytest.raises(NoTargetCameraError, match='frame 1'):
        plan_samples({'S1': frames('seq', '1234')}, {'S1': targets}, n_samples=4, seed=0, context_length=4)
