"""What the novel-view benchmark reads from an Ava-256 capture: which frames have complete inputs, which target
cameras have data for a frame, and a test frame's batch. TCE049 on disk, CPU.

Run: conda run -n match python -m pytest -q tests/test_novel_view_data.py
"""
import pytest

import synthetic_capture as synth
from tce049 import AVA_ROOT, HELDOUT_CAMERAS, INPUT_CAMERA_IDS, SEGMENT, SEGMENT_FRAMES

pytestmark = [
    pytest.mark.skipif(not AVA_ROOT.exists(), reason='needs data/ava-256 with capture TCE049 (symlink it into a worktree)'),
    pytest.mark.usefixtures('fail_instead_of_hanging'),
]

from match import data  # noqa: E402  (imports assets/ at import time)
from match.novel_view.data import FrameLoadError, frame_inputs, load_frame, target_cameras_on_disk  # noqa: E402

TEMPEH_DIR = 'data/tempeh_predictions/ava-256/seq_SEN_a_good_morrow/ava-256'  # TEMPEH for the 40 sentence frames only
# A camera with calibration but no image zip on disk
NOT_DOWNLOADED_CAMERA = '401883'


@pytest.fixture(scope='module')
def capture():
    dataset = data.get_dataset_cls('ava256')(
        root_path=str(AVA_ROOT), height=786, width=512, training=False, camera_angles=synth.AVA_CAMERA_ANGLES,
        uv_directory=TEMPEH_DIR, coarse_mesh_directory=TEMPEH_DIR, head_crop=True, head_crop_height=640,
        head_crop_width=512, head_crop_offset_y=-64, require_verts=False, require_segmentation=False)
    (capture,) = dataset.single_capture_datasets.values()
    return capture


def test_only_frames_with_a_tempeh_prediction_have_complete_inputs(capture):
    frames = frame_inputs(capture)
    complete = [f for f in frames if f.inputs is not None]

    assert len(frames) > 5000  # the whole capture's frame list
    assert [f.frame for f in complete] == SEGMENT_FRAMES
    assert {f.sequence for f in complete} == {SEGMENT}
    assert {f.inputs for f in complete} == {tuple(INPUT_CAMERA_IDS)}


def test_the_held_out_cameras_are_targets_for_every_sentence_frame_and_a_camera_without_images_never_is(capture):
    targets = target_cameras_on_disk(capture, list(HELDOUT_CAMERAS) + [NOT_DOWNLOADED_CAMERA])

    for frame in SEGMENT_FRAMES:
        assert targets(frame) == list(HELDOUT_CAMERAS)


def test_a_test_frame_loads_with_its_inputs_and_its_target_camera_as_the_extra_view(capture):
    batch = load_frame(capture, SEGMENT_FRAMES[5], '401905')

    assert batch['frame'] == [SEGMENT_FRAMES[5]]
    assert [str(c) for c in batch['camera'][0].tolist()] == INPUT_CAMERA_IDS
    assert [str(c) for c in batch['extra_camera'][0].tolist()] == ['401905']
    assert tuple(batch['image'].shape) == (1, 12, 3, 640, 512)
    assert tuple(batch['extra_image'].shape) == (1, 1, 3, 640, 512)


def test_a_window_frame_loads_without_a_target_view(capture):
    batch = load_frame(capture, SEGMENT_FRAMES[6], None)

    assert not any(k.startswith('extra_') for k in batch)


def test_a_target_camera_that_is_an_input_is_refused(capture):
    with pytest.raises(FrameLoadError, match='input'):
        load_frame(capture, SEGMENT_FRAMES[0], INPUT_CAMERA_IDS[0])


def test_a_frame_without_a_tempeh_prediction_is_refused(capture):
    with pytest.raises(FrameLoadError, match='TEMPEH'):
        load_frame(capture, '33365', None)
