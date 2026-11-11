"""The shipped MATCH camera-sweeps config for calibrated captures builds a working validation loader
once its paths point at a capture, its masks and TEMPEH's output."""
from pathlib import Path

import pytest

CONFIG = Path('configs/match/save_camera_sweeps_calibrated.gin')
PRETRAINED_CONFIG = Path('assets/pretrained_models/match/uv_res_1024/config_0.gin')
if not PRETRAINED_CONFIG.exists():
    pytest.skip(f'{PRETRAINED_CONFIG} is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

import gin  # noqa: E402
import numpy as np  # noqa: E402

from match import data  # noqa: E402
from match.options import Options  # noqa: E402
from match.utils import gin_util  # noqa: E402,F401  registers join_strings, as save_camera_sweeps.py does

import synthetic_capture as synth  # noqa: E402
from test_calibrated_capture_dataset import UV_COLOR, write_uv_maps  # noqa: E402

PRETRAINED_CAMERA_ANGLES = synth.AVA_CAMERA_ANGLES


@pytest.fixture
def val_loader_params(tmp_path):
    """The config's val_loader_params with only the path macros (and, for the tiny synthetic
    capture, the resize target) rebound."""
    capture = synth.write_capture(tmp_path / 'capture 1', tmp_path / 'masks', nframes=2)
    write_uv_maps(tmp_path / 'tempeh' / 'calibrated', capture.root.name, frames=[0, 1])
    write_coarse_meshes(tmp_path / 'tempeh' / 'calibrated', capture.root.name, frames=[0, 1])
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        config_files=[str(CONFIG)],
        bindings=[f'CAPTURE_ROOT = "{capture.root}"',
                  f'CAPTURE_MASK_DIR = "{capture.mask_dir}"',
                  f'CAPTURE_TEMPEH_DIR = "{tmp_path / "tempeh" / "calibrated"}"',
                  f'CAPTURE_HEIGHT = {synth.NATIVE_HEIGHT * 4}',
                  f'CAPTURE_WIDTH = {synth.NATIVE_WIDTH * 4}'],
        skip_unknown=True)
    try:
        yield Options().val_loader_params
    finally:
        gin.clear_config()


def write_coarse_meshes(tempeh_dir, capture_name, frames):
    for frame in frames:
        folder = Path(tempeh_dir, capture_name, capture_name, f'{frame:06d}')
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'verts.npy', np.zeros((10, 3), np.float32))


def test_the_config_builds_a_calibrated_loader_over_the_pretrained_camera_angles(val_loader_params):
    [params] = val_loader_params
    assert params['cls'] == 'calibrated'
    dataset_kwargs = params['kwargs']['dataset_kwargs']
    assert dataset_kwargs['camera_angles'] == PRETRAINED_CAMERA_ANGLES

    loader = data.get_loader_cls(params['cls'])(**params['kwargs'])
    sample = loader.dataset[0]
    assert sample['image'].shape == (12, 3, 640, 512)
    assert sample['coarse_verts'].shape == (10, 3)
    for channel, value in enumerate(UV_COLOR):
        assert np.allclose(sample['uv'][:, channel].numpy(), value / 255, atol=1e-6)

