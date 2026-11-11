"""The calibrated-capture debug CLI loads one sample and writes a visualization."""
from pathlib import Path

import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from PIL import Image  # noqa: E402

import synthetic_capture as synth  # noqa: E402
from match.data import calibrated_capture_dataset_cli  # noqa: E402


def test_cli_writes_a_visualization_of_one_sample(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture 1', tmp_path / 'masks', nframes=1)
    out = tmp_path / 'vis' / 'sample.png'
    calibrated_capture_dataset_cli.main([
        '--root', str(capture.root), '--mask-dir', str(capture.mask_dir),
        '--height', '128', '--width', '160', '--crop-height', '96', '--crop-width', '80',
        '--crop-offset-y', '0', '--out', str(out),
    ])
    with Image.open(out) as img:
        assert img.width > 0 and img.height > 0
