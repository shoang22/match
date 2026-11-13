"""TEMPEH predictions for all 30 frames of `background 1` (#29, used by the #36 pilot), after the run of
configs/tempeh/save_predictions_background1.yml: a coarse mesh and one UV map per camera on every frame.

Skips until the run has been done (and without data/ in a worktree):
  conda run -n match python -m pytest tests/test_background1_tempeh_predictions.py
"""
import numpy as np
import pytest

from background1 import CAMERAS, FRAMES, TEMPEH_DIR

N_TEMPEH_VERTS = 5741

pytestmark = pytest.mark.skipif(
    not TEMPEH_DIR.is_dir(),
    reason='run the TEMPEH predictions first: condor/run.py configs/tempeh/save_predictions_background1.yml')


def test_outputs_exist_for_every_frame_of_the_capture():
    frames = sorted(p.name for p in TEMPEH_DIR.iterdir() if p.is_dir())
    assert frames == FRAMES


def test_each_frame_has_a_coarse_mesh_and_one_uv_map_per_camera():
    problems = {}
    for frame in FRAMES:
        frame_dir = TEMPEH_DIR / frame
        verts = np.load(frame_dir / 'verts.npy')
        uv_cameras = sorted(p.name[len('uv_cam'):-len('.png')] for p in frame_dir.glob('uv_cam*.png'))
        if verts.shape != (N_TEMPEH_VERTS, 3) or not np.isfinite(verts).all():
            problems[frame] = f'verts {verts.shape}, finite: {np.isfinite(verts).all()}'
        elif uv_cameras != sorted(CAMERAS):
            problems[frame] = f'UV maps for cameras {uv_cameras}'
    assert not problems, problems
