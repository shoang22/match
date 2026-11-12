"""Ava-256 keeps zero-filling UV maps that are missing from the configured `uv_directory`, now with a warning.

NeRSemble shares the same base-class path; no NeRSemble data is available locally to test it.
"""
import logging
from pathlib import Path

import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from match import data  # noqa: E402

AVA_ROOT = Path('data/ava-256')
AVA_CAMERA_ANGLES = [[0, 0], [0, -15], [20, 0], [20, 36], [20, -15], [-20, 0], [-20, 36],
                     [-20, -15], [40, -5], [40, 25], [-40, -5], [-40, 25]]


@pytest.mark.skipif(not AVA_ROOT.exists(), reason='needs data/ava-256')
def test_ava_zero_fills_uv_maps_missing_from_the_uv_directory_with_a_bounded_warning(tmp_path, caplog):
    dataset = data.get_dataset_cls('ava256')(
        root_path=str(AVA_ROOT), height=786, width=512, camera_angles=AVA_CAMERA_ANGLES, nframes=1,
        uv_directory=str(tmp_path / 'empty'), training=None, require_verts=False, require_segmentation=False)
    with caplog.at_level(logging.WARNING):
        sample = dataset[0]
    assert sample['uv'].shape == sample['image'].shape
    assert (sample['uv'] == 0).all()
    assert (sample['image'] > 0).any()
    # 12 input views zero-filled: one warning (the 1st), the 10th's, not one per view
    warnings = [r.getMessage() for r in caplog.records if 'zero-filled so far' in r.getMessage()]
    assert len(warnings) == 2, warnings
    assert '1 input view(s)' in warnings[0] and '10 input view(s)' in warnings[1]
