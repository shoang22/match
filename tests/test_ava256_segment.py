"""Selecting one Ava-256 segment by name (issue #3).

`match.data.ava256_dataset` opens assets with repo-relative paths at import time; conftest.py runs the tests
from the repo root.
"""
import pytest

from match.data.ava256_dataset import AvaMultiCaptureDataset, AvaSingleCaptureDataset
from tce049 import AVA_ROOT, SEGMENT, SEGMENT_FRAMES

needs_ava_data = pytest.mark.skipif(
    not AVA_ROOT.exists(), reason='needs data/ava-256 with capture TCE049 (symlink it into a worktree)'
)

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


def tce049_frame_ids(**kwargs):
    dataset = AvaMultiCaptureDataset(
        root_path=str(AVA_ROOT), height=786, width=512, training=None, only_subjects=['TCE049'],
        head_crop=False, require_verts=False, require_segmentation=False, **kwargs,
    )
    (capture_dataset,) = dataset.single_capture_datasets.values()
    return [capture_dataset.get_frameid(i) for i in range(len(capture_dataset))]


@needs_ava_data
def test_tce049_first_sentence_has_its_40_frames_in_order():
    assert tce049_frame_ids(segment=SEGMENT, frame_stride=1) == SEGMENT_FRAMES


@needs_ava_data
def test_tce049_without_segment_keeps_the_whole_capture():
    # Spec #1: the 4TB frame list for TCE049 has 5,024 frames, starting with the first sentence.
    frame_ids = tce049_frame_ids(frame_stride=1)
    assert len(frame_ids) == 5024
    assert frame_ids[0] == '32885'
