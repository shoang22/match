"""Frames of a flicker-only sequence evaluation (#28): a capture whose cameras are all inputs, no held-out cameras.

Builds the calibrated-capture dataset on a synthetic 12-camera capture (all 12 cameras are inputs), like
`background 1`. CPU, no model.
"""
import dataclasses
from pathlib import Path

import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

import synthetic_capture as synth  # noqa: E402
from match import data  # noqa: E402
from match.sequence_eval.frames import (FixedViewCameraError, SequenceFrames, check_fixed_view_input_camera,  # noqa: E402
                                        check_heldout_cameras, frame_ids, segment_ids, single_capture)

pytestmark = pytest.mark.usefixtures('fail_instead_of_hanging')

HEIGHT, WIDTH = 128, 83  # the synthetic capture's 104x160 Ava-aspect crop, scaled by 0.8
CROP_HEIGHT, CROP_WIDTH = 96, 80
FRONTAL_CAMERA = '0'  # rig camera 0 sits at az 1.0, el 0.2


def all_input_capture(tmp_path, nframes=3, **write_kwargs):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=nframes, **write_kwargs)
    dataset = data.get_dataset_cls('calibrated')(
        root_path=str(capture.root), mask_directory=str(capture.mask_dir), height=HEIGHT, width=WIDTH,
        camera_angles=synth.AVA_CAMERA_ANGLES, extra_camera_angles=None, head_crop=True,
        head_crop_height=CROP_HEIGHT, head_crop_width=CROP_WIDTH, training=None, require_verts=False,
        require_segmentation=False)
    return single_capture(dataset)


def test_every_frame_of_the_capture_is_evaluated_in_order(tmp_path):
    capture = all_input_capture(tmp_path)
    assert frame_ids(capture) == ['000000', '000001', '000002']
    assert segment_ids(capture) == ['capture']  # a calibrated capture is one sequence, named after its folder


def test_a_frame_has_the_twelve_input_views_including_the_fixed_view_and_no_extra_views(tmp_path):
    capture = all_input_capture(tmp_path)
    check_heldout_cameras(capture, [], frame_ids(capture))  # nothing to check, and no Ava files to read
    sequence = SequenceFrames(capture, heldout_camera_ids=[], fixed_view_input_camera=FRONTAL_CAMERA)

    sample = sequence[1]

    assert sample['frame'] == '000001'
    cameras = [str(c) for c in sample['camera'].tolist()]
    assert sorted(cameras, key=int) == [str(c) for c in range(12)]
    assert FRONTAL_CAMERA in cameras
    assert tuple(sample['image'].shape) == (12, 3, CROP_HEIGHT, CROP_WIDTH)
    assert not [k for k in sample if k.startswith('extra_')]


def test_a_fixed_view_camera_that_is_not_an_input_on_every_frame_fails_before_the_model_loads(tmp_path):
    # A 13th camera at camera 0's pose stands in for it on frame 2, where camera 0 has no image.
    cameras = synth.make_rig()
    cameras[12] = dataclasses.replace(cameras[0])
    capture = all_input_capture(tmp_path, cameras=cameras, missing_images=[(2, 0)])
    assert '0' not in capture.sample_cameras_from_camera_angles('000002')
    with pytest.raises(FixedViewCameraError, match="'0'.*000002"):
        check_fixed_view_input_camera(capture, FRONTAL_CAMERA, frame_ids(capture))


def test_a_fixed_view_camera_the_capture_does_not_have_is_an_error(tmp_path):
    capture = all_input_capture(tmp_path)
    with pytest.raises(FixedViewCameraError, match='401875'):
        check_fixed_view_input_camera(capture, '401875', frame_ids(capture))
