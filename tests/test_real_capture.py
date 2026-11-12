"""Opt-in check of one sample from a real calibrated capture.

Set MATCH_REAL_CAPTURE to a capture folder (images/ and opencv_ba.txt) and MATCH_REAL_CAPTURE_MASKS to its mask
folder, e.g.

    MATCH_REAL_CAPTURE="/path/to/background 1" MATCH_REAL_CAPTURE_MASKS="/path/to/masks/background 1" \
      PYTHONPATH=$(pwd) python -m pytest tests/test_real_capture.py

The sample is built with the framing of configs/match/save_camera_sweeps_calibrated.gin. Skipped when the variables
are unset.
"""
import os
from pathlib import Path

import numpy as np
import pytest

CAPTURE = os.environ.get('MATCH_REAL_CAPTURE')
MASKS = os.environ.get('MATCH_REAL_CAPTURE_MASKS')

pytestmark = [pytest.mark.skipif(not (CAPTURE and MASKS),
                                 reason='set MATCH_REAL_CAPTURE and MATCH_REAL_CAPTURE_MASKS to run'),
              pytest.mark.usefixtures('fail_instead_of_hanging')]

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from match import data  # noqa: E402

# configs/match/save_camera_sweeps_calibrated.gin
HEIGHT, WIDTH = 786, 512  # Ava-256 size; the dataset cuts the images to Ava's aspect ratio
CROP_HEIGHT, CROP_WIDTH, CROP_OFFSET_Y = 640, 512, -128
AVA_CAMERA_ANGLES = [[0, 0], [0, -15], [20, 0], [20, 36], [20, -15], [-20, 0], [-20, 36], [-20, -15],
                     [40, -5], [40, 25], [-40, -5], [-40, 25]]
NUM_CAMERAS = 12


@pytest.fixture(scope='module')
def sample():
    dataset = data.get_dataset_cls('calibrated')(
        root_path=CAPTURE, mask_directory=MASKS, height=HEIGHT, width=WIDTH, camera_angles=AVA_CAMERA_ANGLES,
        head_crop=True, head_crop_height=CROP_HEIGHT, head_crop_width=CROP_WIDTH, head_crop_offset_y=CROP_OFFSET_Y,
        nframes=1, training=None, require_verts=False, require_segmentation=False)
    return dataset[0]


def test_all_twelve_cameras_are_present_with_head_crop_shapes(sample):
    assert sorted(int(c) for c in sample['camera'].tolist()) == list(range(NUM_CAMERAS))
    assert tuple(sample['image'].shape) == (NUM_CAMERAS, 3, CROP_HEIGHT, CROP_WIDTH)
    assert tuple(sample['mask'].shape[0:1] + sample['mask'].shape[-2:]) == (NUM_CAMERAS, CROP_HEIGHT, CROP_WIDTH)
    assert tuple(sample['C2W'].shape) == (NUM_CAMERAS, 4, 4)
    assert tuple(sample['fxfycxcy'].shape) == (NUM_CAMERAS, 4)


def project(sample, view, point):
    """Pixel coordinates with the top-left pixel's corner at (0, 0), as in MATCH."""
    _, _, H, W = sample['image'].shape
    w2c = np.linalg.inv(sample['C2W'][view].double().numpy())
    fx, fy, cx, cy = sample['fxfycxcy'][view].double().numpy() * [W, H, W, H]
    x, y, z = w2c[:3, :3] @ point + w2c[:3, 3]
    return fx * x / z + cx, fy * y / z + cy


def test_the_head_center_projects_onto_the_foreground_mask_in_every_view(sample):
    head = sample['headpose'][:3, 3].double().numpy()
    _, _, H, W = sample['image'].shape
    for view in range(NUM_CAMERAS):
        x, y = project(sample, view, head)
        assert 0 <= x < W and 0 <= y < H, (view, x, y)
        mask = sample['mask'][view].reshape(H, W).float().numpy()
        coverage = mask[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].mean()
        assert coverage > 0.9, (view, coverage)
