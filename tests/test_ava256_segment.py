"""Selecting one Ava-256 segment by name (issue #3).

Run from the repo root: `match.data.ava256_dataset` opens assets with cwd-relative paths at import time.
"""
from pathlib import Path

import pytest

from match.data.ava256_dataset import AvaSingleCaptureDataset

FRAME_LIST = """seg_id,frame_id
SEG_a,1
SEG_a,2
SEG_a,3
SEG_b,4
SEG_b,5
SEG_b,6
SEG_b,7
SEG_c,8
"""


@pytest.fixture
def capture_dir(tmp_path):
    (tmp_path / 'frame_list.csv').write_text(FRAME_LIST)
    return tmp_path


def frame_ids(framelist):
    return list(framelist['frame_id'])


def test_segment_selects_only_that_segment_in_order(capture_dir):
    framelist = AvaSingleCaptureDataset.get_framelist(capture_dir, segment='SEG_b')
    assert frame_ids(framelist) == ['4', '5', '6', '7']


def test_segment_is_applied_before_frame_stride(capture_dir):
    # Striding the whole list first would give rows 1, 3, 5, 7 -> SEG_b frames 5, 7.
    framelist = AvaSingleCaptureDataset.get_framelist(capture_dir, segment='SEG_b', frame_stride=2)
    assert frame_ids(framelist) == ['4', '6']


def test_unknown_segment_raises_naming_the_segment(capture_dir):
    with pytest.raises(ValueError, match='SEG_missing'):
        AvaSingleCaptureDataset.get_framelist(capture_dir, segment='SEG_missing')
