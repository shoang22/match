"""Clip mode of the Ava-256 and calibrated-capture datasets (issue #33).

A clip sample is `clip_length` frames, `clip_stride` frames apart, from one segment (Ava-256) or one capture,
stacked on a leading time axis, with the same cameras in every frame. The calibrated tests build the dataset
from config kwargs through MATCH's registry on a synthetic capture and use only `len()` and indexing.
"""
from pathlib import Path

import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

import synthetic_capture as synth  # noqa: E402
from match import data  # noqa: E402

pytestmark = pytest.mark.usefixtures('fail_instead_of_hanging')

HEIGHT, WIDTH = 128, 83  # the synthetic capture's 104x160 Ava-aspect crop, scaled by 0.8
CROP_HEIGHT, CROP_WIDTH = 96, 80
NUM_CAMERAS = 12


def build_calibrated(capture, **overrides):
    kwargs = dict(
        root_path=str(capture.root),
        mask_directory=str(capture.mask_dir),
        height=HEIGHT,
        width=WIDTH,
        camera_angles=synth.AVA_CAMERA_ANGLES,
        head_crop=True,
        head_crop_height=CROP_HEIGHT,
        head_crop_width=CROP_WIDTH,
        training=None,
        require_verts=False,
        require_segmentation=False,
    )
    kwargs.update(overrides)
    return data.get_dataset_cls('calibrated')(**kwargs)


def test_a_clip_stacks_frames_at_the_clip_stride_on_a_time_axis(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=5)
    dataset = build_calibrated(capture, clip_length=2, clip_stride=2)

    assert len(dataset) == 3  # clips start at frames 0, 1 and 2; one starting at 3 would need frame 5
    clip = dataset[1]
    assert clip['frame'] == ['000001', '000003']
    assert tuple(clip['image'].shape) == (2, NUM_CAMERAS, 3, CROP_HEIGHT, CROP_WIDTH)
