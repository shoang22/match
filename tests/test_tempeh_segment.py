"""The TEMPEH prediction script can be limited to one Ava-256 segment (issue #3).

Builds the dataset the way scripts/tempeh/save_predictions.py does (options parsed from the command line,
then Trainer.register_dataset), without loading the model.
"""
import sys

import pytest
import torch

from tce049 import ASSETS, AVA_ROOT, REPO_ROOT, SEGMENT

TEMPEH_CONFIG = ASSETS / 'pretrained_models/tempeh/config.json'
pytestmark = pytest.mark.skipif(
    not (AVA_ROOT.exists() and TEMPEH_CONFIG.exists()),
    reason='needs data/ava-256 and the pretrained TEMPEH config (symlink data/ and assets/ into a worktree)',
)


def tempeh_full_dataset(monkeypatch, *extra_args):
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'third_party/TEMPEH'))
    from option_handler.train_options_global import TrainOptions
    from trainer.global_trainer import Trainer

    argv = ['save_predictions.py', '--config-filename', str(TEMPEH_CONFIG), '--subjects', '["TCE049"]', *extra_args]
    monkeypatch.setattr(sys, 'argv', argv)
    args = TrainOptions().parse()
    trainer = Trainer(args, torch.device('cpu'))
    trainer.register_dataset(init_full_dataset=True)
    return trainer.dataset_full


def test_segment_option_limits_tempeh_dataset_to_that_segment(monkeypatch):
    dataset = tempeh_full_dataset(monkeypatch, '--frame-stride', '1', '--segment', SEGMENT)
    assert len(dataset) == 40


def test_without_segment_tempeh_dataset_covers_the_whole_capture(monkeypatch):
    # Quickstart settings: stride 10 over TCE049's 5,024 frames -> ceil(5024 / 10) = 503.
    dataset = tempeh_full_dataset(monkeypatch, '--frame-stride', '10')
    assert len(dataset) == 503
