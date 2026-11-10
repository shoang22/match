"""Held-out cameras for the TCE049 sequence evaluation (#4).

Checks, through the real Ava-256 dataset on disk, that the three held-out
cameras are loaded by the extra-camera mechanism and never become inputs.
Needs the 4TB-tier TCE049 data under data/ava-256 (images and foreground
masks of the 12 input cameras plus the 3 held-out ones); skipped otherwise.

Run from the repo root:
  PYTHONPATH=$(pwd) conda run -n match python -m pytest tests/test_heldout_cameras.py
"""
import os

import numpy as np
import pytest

AVA_ROOT = 'data/ava-256'
DECODER_DIR = f'{AVA_ROOT}/20230807--0839--TCE049/decoder'
SEGMENT = 'SEN_a_good_morrow_to_you_my_boy'

# Input camera angles of the pretrained uv_res_1024 config, in order.
INPUT_ANGLES = [[0, 0], [0, -15],
                [20, 0], [20, 36], [20, -15],
                [-20, 0], [-20, 36], [-20, -15],
                [40, -5], [40, 25],
                [-40, -5], [-40, 25]]
# The camera picked for each input angle when only the 12 quickstart cameras
# were on disk (before #4's download).
EXPECTED_INPUT_IDS = ['401875', '401919', '402020', '402012', '401920', '401965',
                      '401630', '401916', '401860', '402038', '401966', '401969']

# Held-out cameras with their actual [az, el] (degrees) from camera_calibration.json.
HELDOUT = {
    '402040': [7.7, 24.0],    # interpolating, near [10, 10]; closest to frontal
    '401905': [-31.0, 4.7],   # interpolating, near [-30, 15]
                              # (401883 is closer but lacks 3 of the segment's 40 frames)
    '402031': [75.6, -1.5],   # extrapolating, near [60, 0]
}

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DECODER_DIR), reason='TCE049 Ava-256 data not on disk')


@pytest.fixture(scope='module')
def capture():
    from match.data.ava256_dataset import AvaMultiCaptureDataset
    dataset = AvaMultiCaptureDataset(
        root_path=AVA_ROOT, height=786, width=512,
        camera_angles=INPUT_ANGLES,
        extra_camera_angles=list(HELDOUT.values()),
        training=None, only_subjects=['TCE049'], frame_stride=1,
        head_crop=True, head_crop_height=640, head_crop_width=512, head_crop_offset_y=-64,
        require_verts=False, require_segmentation=False,
    )
    (capture,) = dataset.single_capture_datasets.values()
    return capture


@pytest.fixture(scope='module')
def segment_frames(capture):
    frames = capture.framelist[capture.framelist.seg_id == SEGMENT].frame_id.tolist()
    assert len(frames) == 40
    return frames


def test_heldout_images_and_masks_on_disk():
    import zipfile
    for cam in HELDOUT:
        for asset in ('image', 'foreground_masks'):
            path = f'{DECODER_DIR}/{asset}/cam{cam}.zip'
            assert os.path.isfile(path), path
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None, path


def test_inputs_unchanged_on_segment(capture, segment_frames):
    for frame in segment_frames:
        assert capture.sample_cameras_from_camera_angles(frame) == EXPECTED_INPUT_IDS, frame


def test_extra_cameras_are_exactly_the_heldout_ids(capture, segment_frames):
    for frame in segment_frames:
        inputs = capture.sample_cameras_from_camera_angles(frame)
        extras = capture.get_extra_camera_ids(frame, avoid_camera_ids=inputs)
        assert extras == list(HELDOUT), frame


def test_heldout_loaded_with_image_mask_and_calibration(capture, segment_frames):
    for frame in segment_frames:
        inputs = capture.sample_cameras_from_camera_angles(frame)
        extras = capture.get_extra_camera_ids(frame, avoid_camera_ids=inputs)
        sample = capture.fetch_data_from_disk(frame_id=frame, camera_ids=inputs, extra_camera_ids=extras)
        assert sample['frame'] == frame
        assert [str(c) for c in np.asarray(sample['extra_camera']).tolist()] == list(HELDOUT)
        assert not set(map(str, np.asarray(sample['camera']).tolist())) & set(HELDOUT)
        assert np.asarray(sample['extra_image']).shape == (3, 3, 640, 512)
        assert np.asarray(sample['extra_image']).max() > 0
        mask = np.asarray(sample['extra_mask'])
        assert mask.shape == (3, 1, 640, 512)
        assert all(m.max() > 0 for m in mask), frame
        assert np.isfinite(np.asarray(sample['extra_C2W'])).all()
        assert np.isfinite(np.asarray(sample['extra_fxfycxcy'])).all()
