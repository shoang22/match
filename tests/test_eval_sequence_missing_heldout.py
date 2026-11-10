"""The sequence evaluation run fails clearly when a held-out camera is incomplete (#8).

Runs the real entry point (scripts/match/eval_sequence.py) with the smoke config on a copy of the
TCE049 capture folder in which one held-out camera's image zip, mask zip or calibration entry is
missing. The run must stop with an error naming the camera and the missing asset before it loads the
model or writes anything. CPU only; skipped without data/ava-256 and assets/.

Run from the repo root:
  conda run -n match python -m pytest tests/test_eval_sequence_missing_heldout.py
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from match.sequence_eval.frames import HeldOutCameraError

SMOKE_CONFIG = Path('configs/match/eval_sequence_smoketest.gin')
CAPTURE = '20230807--0839--TCE049'
DECODER_DIR = Path('data/ava-256') / CAPTURE / 'decoder'
HELDOUT_CAMERA = '402040'  # the smoke config's only held-out camera

pytestmark = pytest.mark.skipif(
    not (DECODER_DIR.is_dir() and Path('assets/pretrained_models/match/uv_res_1024').is_dir()),
    reason='needs data/ava-256 and assets/ (symlink them into a worktree)',
)


@pytest.fixture(scope='module')
def eval_sequence():
    spec = importlib.util.spec_from_file_location('eval_sequence_script', 'scripts/match/eval_sequence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mirror_capture(root, drop=None):
    """Mirrors the TCE049 decoder folder under root with symlinks, leaving out one asset.

    drop: 'image' or 'foreground_masks' to leave out the held-out camera's zip, 'calibration' to
    remove its entry from camera_calibration.json.
    """
    decoder = root / 'ava-256' / CAPTURE / 'decoder'
    decoder.mkdir(parents=True)
    for entry in DECODER_DIR.iterdir():
        target = decoder / entry.name
        if entry.name in ('image', 'foreground_masks'):
            target.mkdir()
            for zip_path in entry.iterdir():
                if entry.name == drop and zip_path.name == f'cam{HELDOUT_CAMERA}.zip':
                    continue
                (target / zip_path.name).symlink_to(zip_path.resolve())
        elif entry.name == 'camera_calibration.json' and drop == 'calibration':
            calibration = json.loads(entry.read_text())
            calibration['KRT'] = [c for c in calibration['KRT'] if c['cameraId'] != HELDOUT_CAMERA]
            target.write_text(json.dumps(calibration))
        else:
            target.symlink_to(entry.resolve())
    return root / 'ava-256'


def run_smoke_config(eval_sequence, tmp_path, ava_root):
    import gin
    out_dir = tmp_path / 'out'
    config = tmp_path / 'config.gin'
    config.write_text(
        f"include '{SMOKE_CONFIG.as_posix()}'\n"
        f"AVA_ROOT = '{ava_root}'\n"
        f"eval_sequence.out_dir = '{out_dir}'\n"
    )
    gin.clear_config()
    try:
        eval_sequence.main(['--gin_configs', str(config)])
    finally:
        gin.clear_config()
    return out_dir


@pytest.mark.parametrize('drop, asset', [
    ('image', 'image'),
    ('foreground_masks', 'mask'),
    ('calibration', 'calibration'),
])
def test_missing_heldout_asset_fails_clearly_before_writing(eval_sequence, tmp_path, monkeypatch, drop, asset):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    ava_root = mirror_capture(tmp_path, drop=drop)

    with pytest.raises(HeldOutCameraError) as error:
        run_smoke_config(eval_sequence, tmp_path, ava_root)

    message = str(error.value)
    assert HELDOUT_CAMERA in message and asset in message
    assert not (tmp_path / 'out').exists()
