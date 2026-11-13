"""Temporal texture filter baseline: CPU tests on synthetic Gaussian textures, no model.

Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_texture_filter.py
"""
import math

import torch

from match.sequence_eval.texture_filter import filter_textures

L = 5  # texels


def texture(xyz=0.0, rgb=0.5, scale=0.01, opacity=0.8, rotation=(1.0, 0.0, 0.0, 0.0)):
    """A masked Gaussian texture, (1,1,C,L,1) per key, with every texel set to the same values."""
    def full(values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1).expand(1, 1, len(values), L, 1).clone()
    return {'xyz': full([xyz, 2 * xyz, -xyz]), 'rgb': full([rgb] * 3), 'scale': full([scale] * 3),
            'opacity': full([opacity]), 'rotation': full(list(rotation))}


def test_a_constant_sequence_is_unchanged():
    frames = [texture(xyz=0.3, rotation=(0.6, 0.0, 0.8, 0.0)) for _ in range(7)]

    filtered = filter_textures(frames, sigma=1.5)

    assert len(filtered) == 7
    for out in filtered:
        for key, value in frames[0].items():
            torch.testing.assert_close(out[key], value)


def test_linear_motion_passes_through_unchanged_in_the_interior():
    # sigma 1 reaches 4 frames each side, so frames 4..7 of 12 see the whole, symmetric kernel.
    frames = [texture(xyz=0.1 * t, rgb=0.2 + 0.05 * t, scale=0.01 + 0.001 * t, opacity=0.1 + 0.05 * t)
              for t in range(12)]

    filtered = filter_textures(frames, sigma=1.0)

    for t in range(4, 8):
        for key in ('xyz', 'rgb', 'scale', 'opacity'):
            torch.testing.assert_close(filtered[t][key], frames[t][key])


def test_the_first_and_last_frames_keep_their_intensity():
    # The kernel (sigma 3) is far wider than the 5 frames, so it is cut on both sides of every frame.
    frames = [texture(rgb=0.7, opacity=0.9) for _ in range(5)]

    filtered = filter_textures(frames, sigma=3.0)

    for t in (0, 4):
        torch.testing.assert_close(filtered[t]['rgb'], torch.full_like(frames[t]['rgb'], 0.7))
        torch.testing.assert_close(filtered[t]['opacity'], torch.full_like(frames[t]['opacity'], 0.9))


def test_filtered_rotations_have_unit_norm():
    angles = [0.0, 0.4, 0.9, 1.5, 2.2]
    frames = [texture(rotation=(math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)) for a in angles]

    filtered = filter_textures(frames, sigma=1.0)

    for out in filtered:
        torch.testing.assert_close(out['rotation'].norm(dim=2), torch.ones(1, 1, L, 1))


def test_a_rotation_predicted_with_either_quaternion_sign_is_filtered_as_that_rotation():
    # q and -q are the same rotation, and the model's normalized output may land on either sign from frame to
    # frame. A steady spin (0.3 rad per frame about x) keeps its angle in the interior, whatever the signs.
    angles = [0.3 * t for t in range(9)]
    signs = [1, -1, 1, 1, -1, -1, 1, -1, 1]
    frames = [texture(rotation=(s * math.cos(a / 2), s * math.sin(a / 2), 0.0, 0.0)) for a, s in zip(angles, signs)]

    filtered = filter_textures(frames, sigma=0.5)  # reaches 2 frames each side: frames 2..6 are interior

    for t in range(2, 7):
        expected = torch.tensor([math.cos(angles[t] / 2), math.sin(angles[t] / 2), 0.0, 0.0]).reshape(1, 1, 4, 1, 1)
        same_rotation = (filtered[t]['rotation'] * expected).sum(dim=2).abs()
        torch.testing.assert_close(same_rotation, torch.ones(1, 1, L, 1))


def test_sigma_near_zero_returns_the_input():
    torch.manual_seed(0)
    frames = [{key: torch.rand_like(value) for key, value in texture().items()} for _ in range(4)]
    for frame in frames:
        frame['rotation'] = torch.nn.functional.normalize(frame['rotation'], dim=2)

    filtered = filter_textures(frames, sigma=1e-3)

    for out, frame in zip(filtered, frames):
        for key, value in frame.items():
            torch.testing.assert_close(out[key], value, rtol=0, atol=0)


def test_inputs_that_are_not_predicted_parameters_pass_through_from_their_own_frame():
    # 'tempeh_xyz' (the TEMPEH surface point under each texel) and 'uv' are inputs of the prediction, not
    # parameters the filter smooths: each frame keeps its own, so the TEMPEH jitter score stays that of TEMPEH.
    frames = [{**texture(xyz=0.1 * t * t), 'tempeh_xyz': torch.full((1, 1, 3, L, 1), float(t * t)),
               'uv': torch.full((1, 1, 3, L, 1), 0.25)} for t in range(6)]

    filtered = filter_textures(frames, sigma=1.0)

    for t, out in enumerate(filtered):
        torch.testing.assert_close(out['tempeh_xyz'], frames[t]['tempeh_xyz'], rtol=0, atol=0)
        torch.testing.assert_close(out['uv'], frames[t]['uv'], rtol=0, atol=0)
    assert not torch.equal(filtered[2]['xyz'], frames[2]['xyz'])  # the position itself is filtered
