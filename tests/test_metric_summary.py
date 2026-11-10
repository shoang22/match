"""Per-metric summaries of the sequence run (mean and worst frames) with non-finite values.

masked_psnr returns inf for images identical inside the mask, and a temporal score is nan when a pair has no
valid pixels. Neither may hide a real worst frame or skew the mean.
"""
import math

import pytest

from match.sequence_eval.metrics import summarize


def test_nan_values_are_left_out_of_the_mean_and_the_worst_frames():
    values = {'1': 0.02, '2': math.nan, '3': 0.05, '4': 0.01, '5': 0.03}

    summary = summarize(values, higher_is_better=False, n_worst=3)

    assert summary['mean'] == pytest.approx(0.0275)
    assert summary['worst_frames'] == [{'frame': '3', 'value': 0.05}, {'frame': '5', 'value': 0.03},
                                       {'frame': '1', 'value': 0.02}]


def test_infinite_psnr_is_not_ranked_as_a_worst_frame():
    values = {'1': 25.0, '2': math.inf, '3': 24.0, '4': math.nan}

    summary = summarize(values, higher_is_better=True, n_worst=3)

    assert summary['mean'] == 24.5
    assert summary['worst_frames'] == [{'frame': '3', 'value': 24.0}, {'frame': '1', 'value': 25.0}]


def test_only_non_finite_values_give_a_nan_mean_and_no_worst_frames():
    summary = summarize({'1': math.nan, '2': math.inf}, higher_is_better=True)

    assert math.isnan(summary['mean'])
    assert summary['worst_frames'] == []
