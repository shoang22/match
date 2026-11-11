"""Shared pytest setup.

`match` is installed in editable mode from the main checkout, and importing
`match.data` reads `assets/` relative to the working directory. Run the suite
from the repository root with `python -m pytest`.
"""
import os
import signal
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

TEST_TIMEOUT_S = 300


@pytest.fixture(autouse=True)
def _fail_instead_of_hanging():
    """Dataset indexing retries on skippable errors; never let a test hang on that."""
    def on_timeout(signum, frame):
        raise TimeoutError(f'test exceeded {TEST_TIMEOUT_S} s')
    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(TEST_TIMEOUT_S)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)
