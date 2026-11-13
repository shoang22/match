"""Runs every test from the repo root, whatever directory pytest was started from.

The code under test and the configs use repo-relative paths ('assets/...', 'data/...', 'configs/...'), and
match.data.ava256_dataset opens files under assets/ when it is imported, so the working directory is set
before any test module is collected. Shared test data lives in tce049.py and eval_sequence_runs.py.
"""
import os
import signal

import pytest

from tce049 import REPO_ROOT

TEST_TIMEOUT_S = 60


def pytest_configure(config):
    os.chdir(REPO_ROOT)


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture
def fail_instead_of_hanging():
    """Dataset indexing retries on skippable errors; fail a test that runs longer than TEST_TIMEOUT_S.

    Opt in per module with `pytestmark = pytest.mark.usefixtures('fail_instead_of_hanging')`. It is not
    autouse because the sequence-evaluation smoke tests legitimately run longer.
    """
    def on_timeout(signum, frame):
        raise TimeoutError(f'test exceeded {TEST_TIMEOUT_S} s')
    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(TEST_TIMEOUT_S)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


@pytest.fixture(scope='module')
def tiny_frames():
    """Three consecutive synthetic frames for the tiny CPU model of tiny_match.py (needs assets/)."""
    from tiny_match import synthetic_frames
    return synthetic_frames(3)
