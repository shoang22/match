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


# Six of the 12 target angles, so a frame that lacks one camera still has enough cameras. The rig's camera 0
# (az 1.0, el 0.2) is the one nearest to the first angle, [0, 0].
SIX_ANGLES = synth.AVA_CAMERA_ANGLES[:6]


def test_every_frame_of_a_clip_uses_the_same_cameras_even_when_one_frame_lacks_one(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=2, missing_images={(1, 0)})
    per_frame = build_calibrated(capture, camera_angles=SIX_ANGLES)
    # Single frames pick camera 0 where it exists and a substitute where it doesn't
    assert 0 in per_frame[0]['camera'].tolist()
    assert 0 not in per_frame[1]['camera'].tolist()

    clip = build_calibrated(capture, camera_angles=SIX_ANGLES, clip_length=2)[0]
    assert clip['frame'] == ['000000', '000001']
    assert clip['camera'].shape == (2, 6)
    assert clip['camera'][0].tolist() == clip['camera'][1].tolist()
    assert 0 not in clip['camera'][0].tolist()


def test_clips_never_cross_from_one_capture_into_the_next(tmp_path):
    root, masks = tmp_path / 'root', tmp_path / 'masks'
    synth.write_capture(root / 'take1', masks / 'take1', nframes=3)
    synth.write_capture(root / 'take2', masks / 'take2', nframes=2)
    dataset = build_calibrated(synth.SyntheticCapture(root=root, mask_dir=masks), clip_length=2)

    clips = [(dataset[i]['sequence'], dataset[i]['frame']) for i in range(len(dataset))]
    assert clips == [('take1', ['000000', '000001']), ('take1', ['000001', '000002']),
                     ('take2', ['000000', '000001'])]



def test_a_clip_that_cannot_load_moves_on_to_the_next_clip_like_a_single_frame_does(tmp_path):
    # Frame 2 lacks 7 of 12 cameras, too few for six angles: clips (1, 2) and (2, 3) can't load, and
    # indexing moves on to the next clip, wrapping around to clip (0, 1), as the dataset does for single frames
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=4,
                                  missing_images={(2, cam) for cam in range(7)})
    dataset = build_calibrated(capture, camera_angles=SIX_ANGLES, clip_length=2)
    assert len(dataset) == 3
    assert dataset[1]['frame'] == ['000000', '000001']
    assert dataset[2]['frame'] == ['000000', '000001']


@pytest.mark.parametrize('settings, message', [
    (dict(clip_length=0), 'clip_length'),
    (dict(clip_length=2, clip_stride=0), 'clip_stride'),
    # frame_stride strides the whole frame list, across segment boundaries; clips use clip_stride instead
    (dict(clip_length=2, frame_stride=2), 'frame_stride'),
])
def test_invalid_clip_settings_are_rejected_naming_the_setting(tmp_path, settings, message):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=3)
    with pytest.raises(ValueError, match=message):
        build_calibrated(capture, **settings)


from tce049 import AVA_ROOT, SEGMENT, SEGMENT_FRAMES  # noqa: E402

needs_ava_data = pytest.mark.skipif(
    not AVA_ROOT.exists(), reason='needs data/ava-256 with capture TCE049 (symlink it into a worktree)'
)
# The segment after SEN_a_good_morrow_to_you_my_boy in TCE049's frame list starts at frame 33365, 12 IDs after
# the sentence's last frame, 33353: frame IDs run on across the boundary, only seg_id changes.
NEXT_SEGMENT_FIRST_FRAME = '33365'


def tce049_capture(**kwargs):
    dataset = data.get_dataset_cls('ava256')(
        root_path=str(AVA_ROOT), height=786, width=512, training=None, only_subjects=['TCE049'],
        head_crop=False, require_verts=False, require_segmentation=False, **kwargs,
    )
    (capture_dataset,) = dataset.single_capture_datasets.values()
    return dataset, capture_dataset


@needs_ava_data
def test_ava_clips_of_one_segment_step_through_it_at_the_clip_stride():
    dataset, capture_dataset = tce049_capture(segment=SEGMENT, clip_length=4, clip_stride=2)
    assert len(dataset) == 40 - 6  # a clip spans 6 frame-list rows; the sentence has 40
    assert capture_dataset.get_clip(0) == SEGMENT_FRAMES[0:7:2]
    assert capture_dataset.get_clip(len(dataset) - 1) == SEGMENT_FRAMES[-7::2]


@needs_ava_data
def test_ava_clips_never_cross_into_the_next_segment():
    _, capture_dataset = tce049_capture(clip_length=4, clip_stride=2)
    clips = [capture_dataset.get_clip(i) for i in range(len(capture_dataset))]
    assert SEGMENT_FRAMES[-7::2] in clips  # the sentence's last clip ...
    assert not any(SEGMENT_FRAMES[-1] in clip and NEXT_SEGMENT_FIRST_FRAME in clip for clip in clips)
    assert [NEXT_SEGMENT_FIRST_FRAME, '33389', '33413', '33437'] in clips  # ... and the next segment's first


# In TCE049's frame list, input camera 402020 has frame 33437 but not the next one, 33449 (both in this sentence)
CAMERA_GAP_SEGMENT = 'SEN_all_your_wishful_thinking_wont_change_that'
CAMERA_GAP_FRAMES = ['33437', '33449']
CAMERA_GAP_CLIP = 6  # 33437 is the sentence's 7th frame: (33437 - 33365) / 12 = 6
CAMERA_GAP_CAMERA = '402020'


@needs_ava_data
def test_an_ava_clip_stacks_the_frames_of_a_segment_with_the_same_cameras_in_each():
    dataset, capture_dataset = tce049_capture(segment=CAMERA_GAP_SEGMENT, camera_angles=synth.AVA_CAMERA_ANGLES,
                                              clip_length=2)
    assert capture_dataset.get_clip(CAMERA_GAP_CLIP) == CAMERA_GAP_FRAMES
    clip = dataset[CAMERA_GAP_CLIP]

    assert clip['frame'] == CAMERA_GAP_FRAMES
    assert (clip['subject'], clip['sequence']) == ('TCE049', CAMERA_GAP_SEGMENT)
    assert tuple(clip['image'].shape) == (2, 12, 3, 786, 512)
    assert tuple(clip['C2W'].shape) == (2, 12, 4, 4)
    assert tuple(clip['headpose'].shape) == (2, 4, 4)
    assert tuple(clip['verts'].shape) == (2, 5741, 3)
    cameras = [[str(c) for c in frame_cameras] for frame_cameras in clip['camera'].tolist()]
    assert cameras[0] == cameras[1]
    assert CAMERA_GAP_CAMERA not in cameras[0]
