"""The calibrated-capture debug CLI loads one sample and writes a visualization."""
import re
from pathlib import Path

import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from PIL import Image  # noqa: E402

import synthetic_capture as synth  # noqa: E402
from match.data import calibrated_capture_dataset_cli  # noqa: E402

pytestmark = pytest.mark.usefixtures('fail_instead_of_hanging')


def test_cli_writes_a_visualization_of_one_sample(tmp_path):
    capture = synth.write_capture(tmp_path / 'capture 1', tmp_path / 'masks', nframes=1)
    out = tmp_path / 'vis' / 'sample.png'
    calibrated_capture_dataset_cli.main([
        '--root', str(capture.root), '--mask-dir', str(capture.mask_dir),
        '--height', '128', '--width', '83', '--crop-height', '96', '--crop-width', '80',
        '--crop-offset-y', '0', '--out', str(out),
    ])
    with Image.open(out) as img:
        assert img.width > 0 and img.height > 0


def gin_macros(path):
    return dict(re.findall(r'^(\w+) = (\S+)', Path(path).read_text(), re.MULTILINE))


def test_cli_defaults_to_the_framing_of_the_shipped_match_config():
    config = gin_macros('configs/match/save_camera_sweeps_calibrated.gin')
    pretrained = gin_macros('assets/pretrained_models/match/uv_res_1024/config_0.gin')
    assert (config['CAPTURE_HEIGHT'], config['CAPTURE_WIDTH']) == ('%AVA_HEIGHT', '%AVA_WIDTH')
    args = calibrated_capture_dataset_cli.parse_args(['--root', 'capture', '--mask-dir', 'masks'])
    assert (args.height, args.width) == (int(pretrained['AVA_HEIGHT']), int(pretrained['AVA_WIDTH']))
    assert (args.crop_height, args.crop_width) == (int(pretrained['HEAD_CROP_HEIGHT']), int(pretrained['HEAD_CROP_WIDTH']))
    assert args.crop_offset_y == int(config['CAPTURE_HEAD_CROP_OFFSET_Y'])
