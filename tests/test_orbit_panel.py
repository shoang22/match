"""The orbit panel's camera path, independent of what it renders: it goes round once and closes. CPU, no model.

The video test of the orbit (test_eval_sequence_smoketest.py) also sees the head move between source frames;
this checks the path alone.
"""
import pytest

from match.sequence_eval.panels import OrbitPanel


@pytest.mark.parametrize('n_video_frames', [10, 72, 128])  # the TCE049 smoke run, `background 1`, the full segment
def test_the_step_from_the_last_orbit_pose_back_to_the_first_is_no_jump(n_video_frames):
    positions = OrbitPanel(n_video_frames).C2W[:, :3, 3]
    steps = (positions[1:] - positions[:-1]).norm(dim=-1)
    closing_step = (positions[0] - positions[-1]).norm()

    assert len(positions) == n_video_frames
    # The last pose is one step before the first, so the video loops without a jump or a repeated pose. (The sweep's
    # steps are not uniform: 0.45 to 0.77 apart on 10 poses, and the step back into the frontal pose is 0.32.)
    assert 0 < closing_step <= steps.max(), (closing_step, steps.max())
    assert closing_step > 0.25 * steps.median(), (closing_step, steps.median())
