"""Runs every test from the repo root, whatever directory pytest was started from.

The code under test and the configs use repo-relative paths ('assets/...', 'data/...', 'configs/...'), and
match.data.ava256_dataset opens files under assets/ when it is imported, so the working directory is set
before any test module is collected. Shared test data lives in tce049.py and eval_sequence_runs.py.
"""
import os

import pytest

from tce049 import REPO_ROOT


def pytest_configure(config):
    os.chdir(REPO_ROOT)


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT
