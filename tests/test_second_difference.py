"""The second difference over time shared by the temporal trajectory loss and the geometric jitter. Pure, CPU."""
import torch

from match.utils.math_util import second_difference


def test_the_second_difference_along_time_is_zero_for_linear_motion_and_doubles_a_zigzag():
    # time along dim 1: texel 0 zigzags 0, 1, 0, 1, texel 1 moves linearly
    x = torch.tensor([[[0., 5.], [1., 6.], [0., 7.], [1., 8.]]])  # (B=1, T=4, L=2)

    torch.testing.assert_close(second_difference(x, dim=1), torch.tensor([[[-2., 0.], [2., 0.]]]))
    torch.testing.assert_close(second_difference(x[0], dim=0), torch.tensor([[-2., 0.], [2., 0.]]))


def test_fewer_than_three_frames_have_no_second_difference():
    for T in (1, 2):
        assert second_difference(torch.zeros(1, T, 5), dim=1).shape == (1, 0, 5)
