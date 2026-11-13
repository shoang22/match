"""scripts/match/train_match.py's command line: the configs and the bindings that override them.

Imports the script, which pulls in the runner (~12 s). CPU, no model, no data; needs assets/ for the pretrained config.
Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_train_match_script.py
"""
import importlib.util
from pathlib import Path

import gin
import pytest

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

spec = importlib.util.spec_from_file_location('train_match', 'scripts/match/train_match.py')
train_match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_match)

CONFIG = 'configs/match/train_temporal_match_temporal_loss_smoketest.gin'


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def test_gin_bindings_override_the_configs_so_a_sweep_needs_no_new_config_file():
    opt = train_match.parse_options([
        '--gin_configs', CONFIG,
        '--gin_bindings', 'Options.temporal_loss_weight = 0.01',
        '--gin_bindings', "NAME = 'temporal_loss_weight_0.01_smoketest'",
    ])

    assert opt.temporal_loss_weight == 0.01
    assert opt.base_folder == 'experiments/match/temporal_loss_weight_0.01_smoketest'


def test_without_bindings_the_configs_decide():
    opt = train_match.parse_options(['--gin_configs', CONFIG])

    assert opt.temporal_loss_weight == 1e-3
    assert opt.base_folder == 'experiments/match/temporal_match_temporal_loss_smoketest'
