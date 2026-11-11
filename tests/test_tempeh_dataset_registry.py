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
