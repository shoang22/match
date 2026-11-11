"""The sequence evaluation run fails clearly when a held-out camera is incomplete (#8).

Runs the real entry point (scripts/match/eval_sequence.py) in a subprocess, so its environment and gin
registrations stay out of the test session, with the smoke config on a copy of the TCE049 capture folder in
which one held-out camera's image zip, mask zip or calibration entry is missing. The run must stop with an
error naming the camera and the missing asset before it loads the model or writes anything. CPU only;
skipped without data/ava-256 and assets/.
"""
import json
import os
import subprocess
import sys

import pytest

from tce049 import ASSETS, DECODER_DIR, FIXED_VIEW_CAMERA, REPO_ROOT

SMOKE_CONFIG = 'configs/match/eval_sequence_smoketest.gin'
HELDOUT_CAMERA = FIXED_VIEW_CAMERA  # the smoke config's only held-out camera

pytestmark = pytest.mark.skipif(
    not (DECODER_DIR.is_dir() and (ASSETS / 'pretrained_models/match/uv_res_1024').is_dir()),
    reason='needs data/ava-256 and assets/ (symlink them into a worktree)',
)


def mirror_capture(root, drop):
    """Mirrors the TCE049 decoder folder under root with symlinks, leaving out one asset.

    drop: 'image' or 'foreground_masks' to leave out the held-out camera's zip, 'calibration' to
    remove its entry from camera_calibration.json.
    """
    decoder = root / 'ava-256' / DECODER_DIR.parent.name / 'decoder'
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


def run_smoke_config(tmp_path, ava_root, out_dir):
    """Runs the entry point on the CPU with the smoke config, reading ava_root. Returns the finished process."""
    config = tmp_path / 'config.gin'
    config.write_text(
        f"include '{SMOKE_CONFIG}'\n"
        f"AVA_ROOT = '{ava_root}'\n"
        f"eval_sequence.out_dir = '{out_dir}'\n"
    )
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(REPO_ROOT)}
    return subprocess.run([sys.executable, 'scripts/match/eval_sequence.py', '--gin_configs', str(config)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize('drop, asset', [
    ('image', 'image'),
    ('foreground_masks', 'mask'),
    ('calibration', 'calibration'),
])
def test_missing_heldout_asset_fails_clearly_before_writing(tmp_path, drop, asset):
    ava_root = mirror_capture(tmp_path, drop=drop)
    out_dir = tmp_path / 'out'

    result = run_smoke_config(tmp_path, ava_root, out_dir)

    assert result.returncode != 0
    error = result.stderr[result.stderr.rindex('HeldOutCameraError'):]
    assert HELDOUT_CAMERA in error and asset in error, error
    assert not out_dir.exists()
