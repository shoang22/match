"""The temporal trajectory loss (#38): the masked mean of the squared second difference over time of each
texel's position and colour, for the second stage of temporal MATCH's 2x2 experiment.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_temporal_loss.py
"""
import pytest
import torch

from match.models.match_model import temporal_trajectory_loss


def trajectories(values) -> torch.Tensor:
  """(B=1, T, L, C) from per-frame values of shape (L, C)."""
  return torch.tensor(values, dtype=torch.float32)[None]


@pytest.mark.parametrize('values', [
    [[[1., 2., 3.]]] * 4,  # constant
    [[[1. + t, 2. - 0.5 * t, 3.]] for t in range(4)],  # linear
], ids=['constant', 'linear'])
def test_a_constant_or_linear_trajectory_costs_nothing(values):
  torch.testing.assert_close(temporal_trajectory_loss(trajectories(values)), torch.zeros(1))


def test_jitter_costs_the_mean_squared_second_difference():
  # x = 0, 1, 0, 1: second differences -2 and 2, squared 4 and 4
  zigzag = trajectories([[[0., 0.]], [[1., 0.]], [[0., 0.]], [[1., 0.]]])

  torch.testing.assert_close(temporal_trajectory_loss(zigzag), torch.tensor([4.]))


@pytest.mark.parametrize('weights, expected', [
    ([0., 1.], 0.),  # only the still texel counts
    ([1., 0.], 4.),  # only the jittering texel counts
    ([1., 1.], 2.),  # the mean over both
], ids=['jitter-masked-out', 'still-masked-out', 'both'])
def test_the_loss_is_a_mean_over_the_texels_the_mask_keeps(weights, expected):
  # texel 0 zigzags 0, 1, 0, 1; texel 1 stands still
  clip = trajectories([[[0.], [5.]], [[1.], [5.]], [[0.], [5.]], [[1.], [5.]]])

  loss = temporal_trajectory_loss(clip, texel_weights=torch.tensor(weights))

  torch.testing.assert_close(loss, torch.tensor([expected]))


def test_a_clip_of_fewer_than_three_frames_costs_nothing():
  torch.testing.assert_close(temporal_trajectory_loss(trajectories([[[0.]], [[1.]]])), torch.zeros(1))
