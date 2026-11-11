"""TEMPEH coarse-mesh stage: dataset registration by type and per-sample output routing.

Run from the repo root: `python -m pytest tests`.
"""
import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / 'assets').exists():
    pytest.skip('assets/ is missing (see CLAUDE.md, Worktrees)', allow_module_level=True)
os.chdir(REPO_ROOT)  # match.data reads assets/ relative to cwd at import time
for p in (str(REPO_ROOT / 'third_party' / 'TEMPEH'), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gtempeh_utils import dataset_registry  # noqa: E402

AVA_ANGLES = '[[0,0], [0, -15], [20,0], [20, 36], [20, -15], [-20,0], [-20, 36], [-20, -15], [40,-5], [40, 25], [-40,-5], [-40, 25]]'


def legacy_args(**overrides):
    """Options as TEMPEH's option parser leaves them for the Ava-256 quickstart config."""
    args = dict(
        datasets=None,
        dataset_directory='data/ava-256', subjects=None, camera_angles=AVA_ANGLES,
        height=256, width=167, frame_stride=10, max_captures=-1,
        nersemble_dataset_directory=None, nersemble_subjects=None, nersemble_camera_angles=None,
        nersemble_frame_stride=None, nersemble_height=None, nersemble_width=None, nersemble_max_captures=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_without_dataset_list_ava256_is_the_only_dataset():
    specs = dataset_registry.dataset_specs_from_args(legacy_args())

    assert len(specs) == 1
    assert specs[0]['type'] == 'ava256'
    assert specs[0]['root_path'] == 'data/ava-256'
    assert specs[0]['height'] == 256
    assert specs[0]['width'] == 167
    assert specs[0]['frame_stride'] == 10
    assert specs[0]['max_captures'] == -1
    assert specs[0]['camera_angles'][3] == [20, 36]
    assert specs[0]['only_subjects'] == []


def test_legacy_nersemble_options_add_a_nersemble_dataset_after_ava256():
    specs = dataset_registry.dataset_specs_from_args(legacy_args(
        nersemble_dataset_directory='data/nersemble', nersemble_subjects="['017']",
        nersemble_camera_angles='[[0, 0], [10, 5]]', nersemble_height=256, nersemble_width=176,
        nersemble_frame_stride=4, nersemble_max_captures=3,
    ))

    assert [s['type'] for s in specs] == ['ava256', 'nersemble']
    nersemble = specs[1]
    assert nersemble['root_path'] == 'data/nersemble'
    assert nersemble['camera_angles'] == [[0, 0], [10, 5]]
    assert nersemble['height'] == 256
    assert nersemble['width'] == 176
    assert nersemble['frame_stride'] == 4
    assert nersemble['max_captures'] == 3
    assert nersemble['only_subjects'] == ['017']
    assert nersemble['invalid_captures_path'] == 'assets/nersemble/invalid_captures.txt'


def test_configured_dataset_list_replaces_the_default_ava256_dataset():
    datasets = """[
        {'type': 'nersemble', 'root_path': 'data/nersemble', 'camera_angles': [[0, 0]], 'height': 256, 'width': 176,
         'frame_stride': 4},
        {'type': 'ava256', 'root_path': 'other/ava', 'camera_angles': [[0, 0], [20, 0]], 'height': 400, 'width': 400,
         'head_crop': True, 'head_crop_height': 256, 'head_crop_width': 167, 'head_crop_offset_y': -10,
         'frame_stride': 2},
    ]"""
    specs = dataset_registry.dataset_specs_from_args(legacy_args(datasets=datasets))

    assert [s['type'] for s in specs] == ['nersemble', 'ava256']
    assert specs[0]['root_path'] == 'data/nersemble'
    assert specs[0]['invalid_captures_path'] == 'assets/nersemble/invalid_captures.txt'
    assert specs[1] == {'type': 'ava256', 'root_path': 'other/ava', 'camera_angles': [[0, 0], [20, 0]],
                        'height': 400, 'width': 400, 'head_crop': True, 'head_crop_height': 256,
                        'head_crop_width': 167, 'head_crop_offset_y': -10, 'frame_stride': 2}


requires_ava = pytest.mark.skipif(not (REPO_ROOT / 'data' / 'ava-256').exists(), reason='data/ava-256 is missing')


def first_batch(datasets):
    """The first batch `save_predictions` sees: MultiDataset, batch size 1, no shuffling."""
    import torch
    from match.data import MultiDataset
    loader = torch.utils.data.DataLoader(MultiDataset(datasets), batch_size=1, shuffle=False, num_workers=0)
    return next(iter(loader))


@requires_ava
def test_ava256_quickstart_sample_is_routed_to_the_ava256_folder(tmp_path):
    datasets = dataset_registry.build_datasets(dataset_registry.dataset_specs_from_args(legacy_args()))
    batch = first_batch(datasets)

    out_dirs = dataset_registry.sample_output_dirs(batch, datasets, tmp_path)

    assert out_dirs == [tmp_path / 'ava-256' / 'TCE049' / 'SEN_a_good_morrow_to_you_my_boy' / '32885']


def test_output_folder_comes_from_the_producing_dataset_not_its_position(tmp_path):
    datasets = [SimpleNamespace(DATASET_TYPE='nersemble'), SimpleNamespace(DATASET_TYPE='ava-256'),
                SimpleNamespace(DATASET_TYPE='some-capture')]
    batch = dict(dataset_idx=np.array([2, 0, 1]), subject=['s2', 's0', 's1'], sequence=['q2', 'q0', 'q1'],
                 frame=['2', '0', '1'])

    out_dirs = dataset_registry.sample_output_dirs(batch, datasets, tmp_path)

    assert out_dirs == [tmp_path / 'some-capture/s2/q2/2', tmp_path / 'nersemble/s0/q0/0', tmp_path / 'ava-256/s1/q1/1']


@requires_ava
def test_each_dataset_spec_sets_its_own_size_and_head_crop():
    spec = dict(type='ava256', root_path='data/ava-256', camera_angles=ast.literal_eval(AVA_ANGLES),
                height=393, width=256, head_crop=True, head_crop_height=200, head_crop_width=150, frame_stride=10)
    datasets = dataset_registry.build_datasets([spec])

    batch = first_batch(datasets)

    assert tuple(batch['image'].shape[-2:]) == (200, 150)
    assert batch['image'].shape[1] == 12  # one view per camera angle


def test_predictions_return_to_the_frame_given_by_each_samples_scene_fields():
    import torch
    # sample 0: NeRSemble-style flip, unit scale, no centering
    # sample 1: a per-instance mm capture whose head center (0.01, -0.02, 0.9) m differs from any class constant
    batch = dict(
        scene_rotation=torch.tensor([[[1., 0, 0], [0, -1, 0], [0, 0, -1]], np.eye(3)], dtype=torch.float32),
        scene_center=torch.tensor([[0., 0, 0], [0.01, -0.02, 0.9]]),
        scene_scale=torch.tensor([1., 1. / 1000]),
    )
    tempeh_mm = np.array([[[100., -200., -300.]], [[0., 0., 100.]]], dtype=np.float32)

    points = dataset_registry.to_dataset_frame(tempeh_mm, batch)

    np.testing.assert_allclose(points[0], [[0.1, 0.2, 0.3]], atol=1e-6)
    np.testing.assert_allclose(points[1], [[10., -20., 1000.]], atol=1e-3)


# --- calibrated captures (`calibrated`): no Ava-256 root, framing from the spec ---

import synthetic_capture as synth  # noqa: E402


def calibrated_spec(capture, **overrides):
    """A TEMPEH `--datasets` entry for a calibrated capture, head-cropped like TEMPEH's Ava input."""
    spec = dict(type='calibrated', root_path=str(capture.root), mask_directory=str(capture.mask_dir),
                camera_angles=synth.AVA_CAMERA_ANGLES, height=128, width=160,
                head_crop=True, head_crop_height=96, head_crop_width=63, head_crop_offset_y=0, frame_stride=1)
    spec.update(overrides)
    return spec


@pytest.fixture
def capture(tmp_path):
    return synth.write_capture(tmp_path / 'capture 1', tmp_path / 'masks', nframes=1)


def test_a_calibrated_capture_alone_is_routed_to_the_calibrated_folder(capture, tmp_path):
    specs = dataset_registry.dataset_specs_from_args(legacy_args(
        dataset_directory='does/not/exist', datasets=repr([calibrated_spec(capture)])))
    datasets = dataset_registry.build_datasets(specs)
    batch = first_batch(datasets)

    out_dirs = dataset_registry.sample_output_dirs(batch, datasets, tmp_path / 'out')

    assert out_dirs == [tmp_path / 'out' / 'calibrated' / 'capture 1' / 'capture 1' / '000000']
    assert tuple(batch['image'].shape[-3:]) == (3, 96, 63)


def test_calibrated_predictions_return_to_calibration_units(capture):
    batch = first_batch(dataset_registry.build_datasets([calibrated_spec(capture)]))
    # TEMPEH's volume is centered at the normalized origin, which is the capture's head center
    tempeh_mm = np.array([[[0., 0., 0.], [100., 0., 0.]]], dtype=np.float32)

    points = dataset_registry.to_dataset_frame(tempeh_mm, batch)

    np.testing.assert_allclose(points[0, 0], synth.HEAD_CENTER_MM, atol=0.5)
    np.testing.assert_allclose(np.linalg.norm(points[0, 1] - points[0, 0]), 100., atol=1e-3)


def normalized_projection(c2w, fxfycxcy, points):
    """Projection of world points to image coordinates in [0, 1] (top-left image corner at 0)."""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    fx, fy, cx, cy = np.asarray(fxfycxcy, dtype=np.float64)
    cam = (w2c[:3, :3] @ np.asarray(points, dtype=np.float64).T).T + w2c[:3, 3]
    return np.stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy], axis=-1)


def test_uv_maps_of_head_cropped_samples_are_framed_like_the_full_image(capture):
    # MATCH resizes a loaded UV map to the full (uncropped) image before its own head crop, so the UV maps of a
    # head-cropped TEMPEH input must be rendered with the intrinsics of the full image.
    datasets = dataset_registry.build_datasets([calibrated_spec(capture)])
    batch = first_batch(datasets)

    fxfycxcy = dataset_registry.uv_frame_fxfycxcy(batch, datasets)

    markers = (synth.MARKERS_MM - synth.HEAD_CENTER_MM) / 1000.  # normalized world: m, head at the origin
    for view, camera in enumerate(batch['camera'][0].tolist()):
        expected = (capture.cameras[camera].project(synth.MARKERS_MM) + 0.5) / [synth.NATIVE_WIDTH, synth.NATIVE_HEIGHT]
        got = normalized_projection(batch['C2W'][0, view], fxfycxcy[0, view], markers)
        np.testing.assert_allclose(got, expected, atol=1e-3)  # 1e-3 of the image size


def test_uncropped_samples_keep_uv_maps_in_the_tempeh_input_frame(capture):
    datasets = dataset_registry.build_datasets([calibrated_spec(capture, head_crop=False)])
    batch = first_batch(datasets)

    assert dataset_registry.uv_frame_fxfycxcy(batch, datasets) is None


CALIBRATED_CONFIG = REPO_ROOT / 'configs' / 'tempeh' / 'save_predictions_calibrated.yml'


def parse_shipped_config(config_path, monkeypatch, **paths):
    """TEMPEH's options as `condor/run.py <config>` produces them, with the config's paths replaced."""
    import shlex
    from omegaconf import OmegaConf
    from option_handler.train_options_global import TrainOptions
    cfg = OmegaConf.load(config_path)
    for key, value in paths.items():
        assert key in cfg, key
        cfg[key] = value
    argv = shlex.split(cfg.CondorParameters.cmd)  # the command runs through bash
    monkeypatch.setattr(sys, 'argv', argv[1:])
    return argv, TrainOptions().parse()


def test_shipped_calibrated_config_runs_on_a_capture_alone_once_its_paths_are_edited(capture, tmp_path, monkeypatch):
    argv, args = parse_shipped_config(
        CALIBRATED_CONFIG, monkeypatch,
        data_dir=str(capture.root), mask_dir=str(capture.mask_dir), out_dir=str(tmp_path / 'out dir'))
    assert argv[1] == 'scripts/tempeh/save_predictions.py'
    assert args.outpath == str(tmp_path / 'out dir')

    specs = dataset_registry.dataset_specs_from_args(args)

    assert [s['type'] for s in specs] == ['calibrated']  # no Ava-256 dataset
    spec = specs[0]
    assert spec['root_path'] == str(capture.root)
    assert spec['mask_directory'] == str(capture.mask_dir)
    assert spec['camera_angles'] == synth.AVA_CAMERA_ANGLES
    # TEMPEH's input: a head crop with Ava's aspect ratio, at TEMPEH's 256x167 input size
    assert spec['head_crop'] is True
    assert (spec['head_crop_height'], spec['head_crop_width']) == (256, 167)
    assert spec['height'] == spec['width']  # the capture's native aspect ratio is 1:1
    # UV maps in the full image, at the capture's native aspect ratio
    assert args.uv_out_height == args.uv_out_width


def test_shipped_calibrated_config_frames_tempeh_input_at_avas_head_scale(monkeypatch):
    # A 200 mm head is ~121 px wide in Ava's 256x167 TEMPEH input; on the target capture (focal ~2400 px, head
    # ~1 m away, 1024 px images) it is ~480 px wide at native resolution.
    _, args = parse_shipped_config(CALIBRATED_CONFIG, monkeypatch)
    spec = dataset_registry.dataset_specs_from_args(args)[0]

    head_px = 480 * spec['width'] / 1024
    assert 105 <= head_px <= 135


def test_crop_boxes_place_the_tempeh_input_inside_the_full_image(capture):
    # e.g. to show UV maps rendered in the full image next to TEMPEH's head-cropped input
    datasets = dataset_registry.build_datasets([calibrated_spec(capture)])
    batch = first_batch(datasets)

    boxes = dataset_registry.crop_boxes_in_full_image(batch, datasets)

    markers = (synth.MARKERS_MM - synth.HEAD_CENTER_MM) / 1000.
    for view, camera in enumerate(batch['camera'][0].tolist()):
        expected = (capture.cameras[camera].project(synth.MARKERS_MM) + 0.5) / [synth.NATIVE_WIDTH, synth.NATIVE_HEIGHT]
        in_crop = normalized_projection(batch['C2W'][0, view], batch['fxfycxcy'][0, view], markers)
        left, top, right, bottom = boxes[0, view]
        got = [left, top] + in_crop * [right - left, bottom - top]
        np.testing.assert_allclose(got, expected, atol=1e-3)
