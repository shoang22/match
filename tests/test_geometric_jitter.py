"""Geometric jitter of per-texel positions over a sequence: CPU tests on synthetic trajectories.

Jitter of frames (t-1, t, t+1) is the mean over valid texels of |x[t+1] - 2 x[t] + x[t-1]|.
"""
import math

import pytest
import torch

from match.utils import metric_util

T, N = 6, 500


def start_positions(seed=0):
    """(3, N) texel positions, ~10 cm around the origin like a head in metres."""
    return 0.1 * torch.randn(3, N, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def unit_directions(seed=1):
    d = torch.randn(3, N, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return d / d.norm(dim=0, keepdim=True)


def test_constant_positions_have_zero_jitter():
    positions = start_positions().expand(T, 3, N)
    assert metric_util.geometric_jitter(positions) == [0.0] * (T - 2)


def test_linear_motion_has_zero_jitter():
    velocity = 0.01 * unit_directions()  # 1 cm per frame, a different direction per texel
    positions = torch.stack([start_positions() + t * velocity for t in range(T)])
    assert metric_util.geometric_jitter(positions) == pytest.approx([0.0] * (T - 2), abs=1e-12)


def test_noise_in_one_frame_shows_in_the_three_triples_around_it():
    # Every texel of frame 3 moves 2 units off its path, so the triple centred on frame 3 sees -2 * 2 and the
    # triples centred on frames 2 and 4 see 2. Triples are keyed by their middle frame: 1, 2, 3, 4.
    velocity = 0.01 * unit_directions(seed=2)
    positions = torch.stack([start_positions() + t * velocity for t in range(T)])
    positions[3] += 2.0 * unit_directions(seed=3)
    assert metric_util.geometric_jitter(positions) == pytest.approx([0.0, 2.0, 4.0, 2.0], abs=1e-9)


def test_invalid_texels_are_ignored():
    positions = start_positions().expand(T, 3, N).clone()
    valid = torch.ones(N, dtype=torch.bool)
    valid[::3] = False
    positions[:, :, ~valid] = torch.randn(T, 3, int((~valid).sum()), dtype=torch.float64)
    assert metric_util.geometric_jitter(positions, valid) == [0.0] * (T - 2)


def test_jitter_is_averaged_over_the_valid_texels_only():
    # Half the valid texels jump by 1 in frame 2, so the triple centred on it scores 2 * 1 * 1/2.
    positions = start_positions().expand(T, 3, N).clone()
    valid = torch.zeros(N, dtype=torch.bool)
    valid[:100] = True
    positions[2, :, :50] += unit_directions()[:, :50]
    assert metric_util.geometric_jitter(positions, valid)[1] == pytest.approx(1.0)


def test_fewer_than_three_frames_have_no_triples():
    assert metric_util.geometric_jitter(start_positions().expand(2, 3, N)) == []


def test_no_valid_texels_score_nan():
    scores = metric_util.geometric_jitter(start_positions().expand(3, 3, N), torch.zeros(N, dtype=torch.bool))
    assert len(scores) == 1 and math.isnan(scores[0])
