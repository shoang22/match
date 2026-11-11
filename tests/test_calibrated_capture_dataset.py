"""Sample-contract tests for the calibrated-capture dataset (`calibrated`).

Every test builds the dataset from config kwargs through MATCH's registry, then
uses only `len()` and indexing. CPU-only; the capture is synthesized on disk.
"""
import dataclasses
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from match import data  # noqa: E402

DATASET_NAME = 'calibrated'


def test_registries_resolve_the_new_name():
    dataset_cls = data.get_dataset_cls(DATASET_NAME)
    loader_cls = data.get_loader_cls(DATASET_NAME)
    assert loader_cls.DATASET_CLS is dataset_cls


import synthetic_capture as synth  # noqa: E402

HEIGHT, WIDTH = 128, 160          # native 160x200, scaled by 0.8
CROP_HEIGHT, CROP_WIDTH = 96, 80


def build_dataset(capture, **overrides):
    kwargs = dict(
        root_path=str(capture.root),
        mask_directory=str(capture.mask_dir),
        height=HEIGHT,
        width=WIDTH,
        camera_angles=synth.AVA_CAMERA_ANGLES,
        head_crop=True,
        head_crop_height=CROP_HEIGHT,
        head_crop_width=CROP_WIDTH,
        head_crop_offset_y=0,
        training=None,
        require_verts=False,
        require_segmentation=False,
    )
    kwargs.update(overrides)
    return data.get_dataset_cls(DATASET_NAME)(**kwargs)


@pytest.fixture
def capture(tmp_path):
    return synth.write_capture(tmp_path / 'capture 1', tmp_path / 'masks', nframes=3)


def test_the_dataset_names_its_prediction_output_folder_after_its_type(capture):
    # TEMPEH's save_predictions writes each sample under <out>/<DATASET_TYPE>/...
    assert build_dataset(capture).DATASET_TYPE == DATASET_NAME


def test_length_equals_number_of_frames(capture):
    assert len(build_dataset(capture)) == 3


def test_number_of_frames_and_frame_stride_are_configurable(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=5)
    dataset = build_dataset(capture, frame_stride=2, nframes=2)
    assert len(dataset) == 2
    assert [dataset[i]['frame'] for i in range(2)] == ['000000', '000002']


def test_frames_missing_an_image_or_a_mask_are_flagged_and_skipped(tmp_path, caplog):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=3,
                                  missing_images={(1, 4)}, missing_masks={(2, 7)})
    with caplog.at_level(logging.WARNING):
        dataset = build_dataset(capture)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(re.search(r'frame 000001.*missing image.*camera.*\b4\b', w) for w in warnings), warnings
    assert any(re.search(r'frame 000002.*missing mask.*camera.*\b7\b', w) for w in warnings), warnings

    assert len(dataset) == 3
    # 12 cameras for 12 angles: frames 1 and 2 cannot be used, so indexing moves on to frame 0
    assert dataset[1]['frame'] == '000000'
    assert dataset[2]['frame'] == '000000'


def test_indexing_raises_instead_of_retrying_forever_when_no_frame_is_loadable(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=2,
                                  missing_masks={(0, 0), (1, 0)})
    dataset = build_dataset(capture)
    with pytest.raises(RuntimeError, match='No loadable frame'):
        dataset[0]


SLOT_LINE = re.compile(r'camera (\d+) -> slot (\d+) \[az (-?[\d.]+), el (-?[\d.]+)\], angular error ([\d.]+) deg')


def test_each_cameras_angle_slot_and_error_are_logged_once_per_capture(capture, caplog):
    with caplog.at_level(logging.INFO):
        dataset = build_dataset(capture)
        dataset[0]
        dataset[1]
    assignments = [SLOT_LINE.search(line).groups()
                   for record in caplog.records for line in record.getMessage().splitlines()
                   if SLOT_LINE.search(line)]
    assert sorted(int(a[0]) for a in assignments) == list(range(12))
    by_camera = {int(a[0]): a for a in assignments}
    # The rig puts camera 0 at (az 1.0, el 0.2) and camera 1 at (az 19.7, el 0.5), next to the
    # [0, 0] and [20, 0] slots: about 1.02 and 0.58 degrees away.
    assert by_camera[0][2:4] == ('0', '0') and float(by_camera[0][4]) == pytest.approx(1.02, abs=0.05)
    assert by_camera[1][2:4] == ('20', '0') and float(by_camera[1][4]) == pytest.approx(0.58, abs=0.05)


def test_subfolders_with_a_calibration_file_are_captures_named_after_their_folder(tmp_path):
    root, masks = tmp_path / 'root', tmp_path / 'masks'
    synth.write_capture(root / 'background 1', masks / 'background 1', nframes=2)
    synth.write_capture(root / 'take2', masks / 'take2', nframes=1)
    (root / 'notes').mkdir()
    capture = synth.SyntheticCapture(root=root, mask_dir=masks)

    dataset = build_dataset(capture)
    assert len(dataset) == 3
    assert [(s['subject'], s['sequence']) for s in (dataset[0], dataset[2])] == [
        ('background 1', 'background 1'), ('take2', 'take2')]


def test_configured_ids_replace_the_folder_name_of_a_single_capture(capture):
    overridden = build_dataset(capture, subject_id='S01', sequence_id='outdoor')[0]
    assert (overridden['subject'], overridden['sequence']) == ('S01', 'outdoor')


@pytest.mark.parametrize('override', [dict(subject_id='S01'), dict(sequence_id='outdoor')])
def test_configured_ids_on_a_root_of_several_captures_raise_instead_of_colliding(tmp_path, override):
    root, masks = tmp_path / 'root', tmp_path / 'masks'
    synth.write_capture(root / 'take1', masks / 'take1', nframes=1)
    synth.write_capture(root / 'take2', masks / 'take2', nframes=1)
    capture = synth.SyntheticCapture(root=root, mask_dir=masks)
    with pytest.raises(ValueError, match=r'2 captures.*take1.*take2'):
        build_dataset(capture, **override)


def test_configured_ids_apply_to_a_root_holding_exactly_one_capture_subfolder(tmp_path):
    root, masks = tmp_path / 'root', tmp_path / 'masks'
    synth.write_capture(root / 'take1', masks / 'take1', nframes=1)
    capture = synth.SyntheticCapture(root=root, mask_dir=masks)
    sample = build_dataset(capture, subject_id='S01', sequence_id='outdoor')[0]
    assert (sample['subject'], sample['sequence']) == ('S01', 'outdoor')


def test_building_without_a_mask_folder_raises_a_clear_error(capture):
    with pytest.raises(ValueError, match='mask_directory'):
        build_dataset(capture, mask_directory=None)


AVA_ROOT = Path('data/ava-256')


@pytest.mark.skipif(not AVA_ROOT.exists(), reason='needs data/ava-256 for the reference sample')
def test_sample_keys_dtypes_and_per_view_shapes_match_ava(capture):
    common = dict(camera_angles=synth.AVA_CAMERA_ANGLES, head_crop=True, head_crop_height=CROP_HEIGHT,
                  head_crop_width=CROP_WIDTH, training=None, require_verts=False,
                  require_segmentation=False, nframes=1)
    expected = data.get_dataset_cls('ava256')(root_path=str(AVA_ROOT), height=786, width=512, **common)[0]
    sample = build_dataset(capture, **common)[0]

    assert set(sample) == set(expected)
    for key, ref in expected.items():
        value = sample[key]
        assert type(value) is type(ref), key
        if hasattr(ref, 'dtype'):
            assert value.dtype == ref.dtype, key
            assert value.shape == ref.shape, key


# --- geometry helpers: the sample contract, computed by hand ---

def to_sample_world(sample, points_mm):
    """Calibration units -> the sample's normalized world, using the sample's scene_* fields."""
    R = sample['scene_rotation'].double().numpy()
    s = float(sample['scene_scale'])
    c = sample['scene_center'].double().numpy()
    return s * (R @ np.asarray(points_mm, dtype=np.float64).T).T - c


def from_sample_world(sample, points):
    R = sample['scene_rotation'].double().numpy()
    s = float(sample['scene_scale'])
    c = sample['scene_center'].double().numpy()
    return (R.T @ (np.asarray(points, dtype=np.float64) + c).T).T / s


def project(sample, view, points):
    """Pixel coordinates with the top-left pixel's corner at (0, 0), as in MATCH."""
    _, _, H, W = sample['image'].shape
    w2c = np.linalg.inv(sample['C2W'][view].double().numpy())
    fx, fy, cx, cy = sample['fxfycxcy'][view].double().numpy() * [W, H, W, H]
    cam = (w2c[:3, :3] @ np.asarray(points).T).T + w2c[:3, 3]
    return np.stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy], axis=-1)


def centroid(channel):
    """Intensity-weighted centroid of a 2-D array, in the same pixel convention as `project`."""
    channel = np.asarray(channel, dtype=np.float64)
    ys, xs = np.mgrid[:channel.shape[0], :channel.shape[1]] + 0.5
    total = channel.sum()
    assert total > 1, 'marker not visible'
    return np.array([(channel * xs).sum() / total, (channel * ys).sum() / total])


@pytest.fixture
def sample(capture):
    return build_dataset(capture)[0]


def test_camera_centers_from_c2w_match_the_calibration(capture, sample):
    centers = from_sample_world(sample, sample['C2W'][:, :3, 3].double().numpy())
    for view, camera in enumerate(sample['camera'].tolist()):
        np.testing.assert_allclose(centers[view], capture.cameras[camera].center, atol=0.05)  # mm


MARKER_TOLERANCE_PX = 0.25  # well below the half-pixel convention error at this scale (0.4 px)


def test_image_markers_land_on_the_pinhole_projection_of_their_3d_points(sample):
    markers = to_sample_world(sample, synth.MARKERS_MM)
    for view in range(len(sample['camera'])):
        expected = project(sample, view, markers)
        for channel in range(3):
            measured = centroid(sample['image'][view, channel])
            np.testing.assert_allclose(measured, expected[channel], atol=MARKER_TOLERANCE_PX)


def test_head_center_defaults_to_the_optical_axes_intersection_and_lies_inside_every_crop(sample):
    head = sample['headpose'][:3, 3].double().numpy()
    np.testing.assert_allclose(head, 0, atol=1e-6)
    np.testing.assert_allclose(from_sample_world(sample, head[None])[0], synth.HEAD_CENTER_MM, atol=0.05)
    _, _, H, W = sample['image'].shape
    for view in range(len(sample['camera'])):
        x, y = project(sample, view, head[None])[0]
        assert 0 <= x <= W and 0 <= y <= H


def test_a_configured_scene_center_replaces_the_estimate(capture):
    override = [5.0, 20.0, 940.0]
    sample = build_dataset(capture, scene_center=override)[0]
    head = sample['headpose'][:3, 3].double().numpy()
    np.testing.assert_allclose(head, 0, atol=1e-6)
    np.testing.assert_allclose(from_sample_world(sample, head[None])[0], override, atol=1e-3)


def test_a_configured_scene_rotation_is_applied_and_carried_by_the_sample(capture):
    angle = np.deg2rad(10)
    rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    sample = build_dataset(capture, scene_rotation=rotation)[0]
    np.testing.assert_allclose(sample['scene_rotation'].numpy(), rotation, atol=1e-6)
    centers = from_sample_world(sample, sample['C2W'][:, :3, 3].double().numpy())
    for view, camera in enumerate(sample['camera'].tolist()):
        np.testing.assert_allclose(centers[view], capture.cameras[camera].center, atol=0.05)


def test_intrinsics_keep_the_calibrated_pixel_aspect_ratio(capture, sample):
    _, _, H, W = sample['image'].shape
    for view, camera in enumerate(sample['camera'].tolist()):
        fx, fy = sample['fxfycxcy'][view, :2].double().numpy() * [W, H]
        assert fy / fx == pytest.approx(capture.cameras[camera].aspect_ratio, rel=1e-6)


def test_a_resize_target_that_would_change_the_aspect_ratio_is_rejected(capture):
    with pytest.raises(ValueError, match='aspect'):
        build_dataset(capture, height=128, width=128)


def test_undistortion_keeps_image_size_and_intrinsics(tmp_path, capture, sample):
    pinhole_rig = {i: dataclasses.replace(cam, dist=np.zeros(5)) for i, cam in capture.cameras.items()}
    pinhole_capture = synth.write_capture(tmp_path / 'pinhole', tmp_path / 'pinhole masks', nframes=1,
                                          cameras=pinhole_rig)
    reference = build_dataset(pinhole_capture)[0]
    assert sample['camera'].tolist() == reference['camera'].tolist()
    for key in ('image', 'mask'):
        assert sample[key].shape == reference[key].shape, key
    for key in ('fxfycxcy', 'C2W'):
        np.testing.assert_array_equal(sample[key].numpy(), reference[key].numpy(), err_msg=key)


def test_mask_markers_coincide_with_image_markers(sample):
    # Masks left distorted would sit up to ~1.9 px off; undistorted with the images, within ~0.6 px
    # (the rest is the base class resizing masks with nearest-neighbor and images bilinearly).
    for view in range(len(sample['camera'])):
        np.testing.assert_allclose(centroid(sample['mask'][view, 0]), centroid(sample['image'][view, 0]), atol=1.0)


@pytest.mark.parametrize('mask_mode', ['L-soft', 'L-binary', 'RGB-soft', 'RGB-binary'])
def test_single_channel_and_rgb_masks_binary_or_soft_are_accepted(tmp_path, mask_mode):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=1, mask_mode=mask_mode)
    sample = build_dataset(capture)[0]
    mask = sample['mask'].numpy()
    assert mask.shape[1] == 1
    assert mask.min() >= 0 and mask.max() <= 1
    if mask_mode.endswith('binary'):
        # 0 and 255 come through as 0 and 1; undistortion's bilinear remap may soften the edges
        assert mask.min() == 0.0 and mask.max() == 1.0
    else:
        assert ((mask > 0.05) & (mask < 0.95)).any()
    for view in range(len(sample['camera'])):  # the first channel is the one used
        np.testing.assert_allclose(centroid(mask[view, 0]), centroid(sample['image'][view, 0]), atol=1.0)


def test_no_color_correction_is_applied_and_alpha_is_dropped(tmp_path):
    color = (37, 142, 201)
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=1, flat_color=color, alpha=90)
    image = build_dataset(capture)[0]['image'].numpy()
    assert image.shape[1] == 3
    for channel, value in enumerate(color):
        np.testing.assert_allclose(image[:, channel], value / 255, atol=1e-6)


def test_nonzero_skew_in_the_calibration_raises(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture', tmp_path / 'masks', nframes=1, skew_camera=3)
    with pytest.raises(ValueError, match='skew'):
        build_dataset(capture)


# --- TEMPEH handoff: coarse mesh and per-camera UV maps ---

UV_COLOR = (200, 40, 90)


def write_uv_maps(uv_dir, capture_name, frames, missing=()):
    """TEMPEH's layout: <uv_dir>/<subject>/<sequence>/<frame>/uv_cam<id>.png, flat-colored, native size."""
    uv = np.empty((synth.NATIVE_HEIGHT, synth.NATIVE_WIDTH, 3), np.uint8)
    uv[:] = UV_COLOR
    for frame in frames:
        folder = Path(uv_dir, capture_name, capture_name, f'{frame:06d}')
        folder.mkdir(parents=True, exist_ok=True)
        for camera in range(len(synth.RIG_ANGLES_DEG)):
            if (frame, camera) not in missing:
                Image.fromarray(uv).save(folder / f'uv_cam{camera}.png')


def test_uv_maps_from_the_configured_directory_reach_the_sample(tmp_path, capture):
    write_uv_maps(tmp_path / 'uv', capture.root.name, frames=[0])
    uv = build_dataset(capture, uv_directory=str(tmp_path / 'uv'), nframes=1)[0]['uv'].numpy()
    for channel, value in enumerate(UV_COLOR):
        np.testing.assert_allclose(uv[:, channel], value / 255, atol=1e-6)


def test_a_missing_uv_map_skips_the_frame_instead_of_zero_filling(tmp_path, capture, caplog):
    write_uv_maps(tmp_path / 'uv', capture.root.name, frames=[0, 1], missing={(0, 5)})
    dataset = build_dataset(capture, uv_directory=str(tmp_path / 'uv'), nframes=2)

    with caplog.at_level(logging.WARNING):
        sample = dataset[0]
    assert sample['frame'] == '000001'
    assert (sample['uv'] > 0).all()
    assert any('000000' in r.getMessage() and 'uv_cam5.png' in r.getMessage() for r in caplog.records)


def test_no_frame_is_returned_with_a_zero_filled_uv_map(tmp_path, capture):
    write_uv_maps(tmp_path / 'uv', capture.root.name, frames=[0], missing={(0, 5)})
    dataset = build_dataset(capture, uv_directory=str(tmp_path / 'uv'), nframes=1)
    with pytest.raises(RuntimeError, match='No loadable frame'):
        dataset[0]
