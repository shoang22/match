"""Temporal scores in the sequence-evaluation smoke run's metrics.json (#11).

Skipped until the smoke run has been done (run command in configs/match/eval_sequence_smoketest.gin).
EVAL_SEQUENCE_SMOKE_DIR checks another output folder of the smoke config, e.g. one written with
  --gin_bindings "OUT_DIR = 'experiments/match/eval_sequence/<name>'"
"""
import json
import math
import os
from pathlib import Path

import pytest

OUT_DIR = Path(os.environ.get('EVAL_SEQUENCE_SMOKE_DIR', 'experiments/match/eval_sequence/smoketest'))
FRAMES = ['32885', '32897', '32909']
FIXED_VIEW_CAMERA = '402040'
SOURCES = ['prediction', 'real']
SCORES = ['flow_aligned_error', 'raw_flicker']

pytestmark = pytest.mark.skipif(
    not (OUT_DIR / 'metrics.json').exists(), reason=f'run the smoke config first (no {OUT_DIR}/metrics.json)')


@pytest.fixture(scope='module')
def temporal():
    metrics = json.loads((OUT_DIR / 'metrics.json').read_text())
    if 'temporal' not in metrics:
        pytest.skip(f'{OUT_DIR}/metrics.json predates the temporal metrics (#11); rerun the smoke config')
    return metrics['temporal']


def test_every_consecutive_pair_has_finite_temporal_scores_for_prediction_and_real(temporal):
    assert temporal['camera'] == FIXED_VIEW_CAMERA
    assert list(temporal['per_pair']) == FRAMES[1:]
    for previous, frame in zip(FRAMES, FRAMES[1:]):
        entry = temporal['per_pair'][frame]
        assert entry['previous_frame'] == previous
        for source in SOURCES:
            assert sorted(entry[source]) == SCORES
            assert all(isinstance(v, float) and math.isfinite(v) and v >= 0 for v in entry[source].values()), entry


def test_temporal_summary_has_mean_and_worst_frames(temporal):
    for source in SOURCES:
        for score in SCORES:
            values = {f: temporal['per_pair'][f][source][score] for f in FRAMES[1:]}
            summary = temporal['summary'][source][score]
            assert summary['mean'] == pytest.approx(sum(values.values()) / len(values))
            worst = max(values, key=values.get)
            assert summary['worst_frames'][0] == {'frame': worst, 'value': values[worst]}
