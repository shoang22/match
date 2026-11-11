"""Temporal-consistency metrics: CPU tests on synthetic frame sequences."""

import pytest
import torch

from match.utils import metric_util


T, H, W = 4, 128, 128


def make_canvas(seed: int, size: int = 192) -> torch.Tensor:
    """Smooth random RGB texture in [0, 1], shape (3, size, size)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, size // 8, size // 8, generator=g)
    return torch.nn.functional.interpolate(
        low, size=(size, size), mode='bicubic', align_corners=False)[0].clamp(0, 1)


def static_sequence(seed: int = 0) -> torch.Tensor:
    """The same frame T times, shape (T, 3, H, W)."""
    return make_canvas(seed)[:, :H, :W].expand(T, 3, H, W).clone()


def translating_sequence(seed: int = 0, step=(3, 2)) -> torch.Tensor:
    """A window sliding across a larger texture by `step` (dx, dy) pixels per frame."""
    canvas = make_canvas(seed)
    dx, dy = step
    return torch.stack([canvas[:, t * dy:t * dy + H, t * dx:t * dx + W] for t in range(T)])


def make_masks() -> torch.Tensor:
    """Centred square foreground in every frame, shape (T, H, W), bool."""
    masks = torch.zeros(T, H, W, dtype=torch.bool)
    masks[:, 32:96, 32:96] = True
    return masks


def test_static_sequence_has_zero_raw_flicker():
    scores = metric_util.raw_flicker(static_sequence(), make_masks())
    assert scores.per_pair == [0.0] * (T - 1)
    assert scores.mean == 0.0


def test_static_sequence_has_zero_flow_aligned_error():
    scores = metric_util.flow_aligned_error(static_sequence(), make_masks())
    assert len(scores.per_pair) == T - 1
    assert scores.per_pair == pytest.approx([0.0] * (T - 1), abs=1e-3)
    assert scores.mean == pytest.approx(0.0, abs=1e-3)


def test_translating_sequence_has_near_zero_flow_aligned_error_but_nonzero_raw_flicker():
    frames, masks = translating_sequence(), make_masks()
    flicker = metric_util.raw_flicker(frames, masks)
    aligned = metric_util.flow_aligned_error(frames, masks)

    assert all(v > 0.02 for v in flicker.per_pair)
    # RAFT is off by ~0.05 px here, which leaves a few % of the raw change.
    assert all(a < 0.05 * f for a, f in zip(aligned.per_pair, flicker.per_pair))


def test_given_exact_flow_aligns_a_translating_sequence():
    frames = translating_sequence(step=(3, 2))
    # Content moves by (-3, -2) px from each frame to the next.
    forward = torch.tensor([-3.0, -2.0]).view(1, 2, 1, 1).expand(T - 1, 2, H, W)
    scores = metric_util.flow_aligned_error(frames, make_masks(), flow=(forward, -forward))
    assert scores.per_pair == pytest.approx([0.0] * (T - 1), abs=1e-6)


@pytest.mark.parametrize('make_sequence', [static_sequence, translating_sequence],
                         ids=lambda f: f.__name__)
def test_per_frame_noise_increases_both_scores(make_sequence):
    clean, masks = make_sequence(), make_masks()
    g = torch.Generator().manual_seed(7)
    noisy = (clean + 0.05 * torch.randn(clean.shape, generator=g)).clamp(0, 1)
    # As in the evaluation run: flow from the clean footage, applied to both.
    flow = metric_util.raft_flow(clean)

    for score in (lambda x: metric_util.raw_flicker(x, masks),
                  lambda x: metric_util.flow_aligned_error(x, masks, flow=flow)):
        before, after = score(clean), score(noisy)
        assert all(a > b for a, b in zip(after.per_pair, before.per_pair))
        assert after.mean > before.mean


def test_pixels_outside_mask_do_not_change_either_score():
    frames = translating_sequence(step=(3, 2))
    # Foreground square that follows the content, so each frame's mask differs.
    masks = torch.zeros(T, H, W, dtype=torch.bool)
    for t in range(T):
        masks[t, 32 - 2 * t:96 - 2 * t, 32 - 3 * t:96 - 3 * t] = True
    scribbled = torch.stack([
        torch.where(masks[t], frames[t], make_canvas(100 + t)[:, :H, :W]) for t in range(T)])

    flicker = (metric_util.raw_flicker(frames, masks),
               metric_util.raw_flicker(scribbled, masks))
    # Flow estimated on the clean reference, so the background can't steer it.
    aligned = (metric_util.flow_aligned_error(frames, masks, reference=frames),
               metric_util.flow_aligned_error(scribbled, masks, reference=frames))

    for baseline, changed in (flicker, aligned):
        assert changed.per_pair == pytest.approx(baseline.per_pair, rel=1e-6, abs=1e-7)


def test_pixels_failing_forward_backward_check_are_ignored():
    frames = translating_sequence(step=(3, 2))
    forward = torch.tensor([-3.0, -2.0]).view(1, 2, 1, 1).expand(T - 1, 2, H, W)
    backward = (-forward).clone()
    backward[:, :, 50:70, 50:70] += 10.0  # wrong there; forward disagrees with it

    scores = metric_util.flow_aligned_error(frames, make_masks(), flow=(forward, backward))
    assert scores.per_pair == pytest.approx([0.0] * (T - 1), abs=1e-6)


def _lai_motion_boundary(flow: torch.Tensor) -> torch.Tensor:
    """Reference: the motion-boundary test of Lai et al. 2018's evaluation code
    (utils.compute_flow_gradients / detect_occlusion, fast_blind_video_consistency),
    ported per pixel for one (2, H, W) flow."""
    u, v = flow[0], flow[1]
    grads = []
    for c in (u, v):
        dx = torch.zeros_like(c)
        dy = torch.zeros_like(c)
        dx[:, :-1] = c[:, :-1] - c[:, 1:]
        dy[:-1, :] = c[:-1, :] - c[1:, :]
        grads += [dx, dy]
    grad_sq = sum(g ** 2 for g in grads)
    return grad_sq > 0.01 * (u ** 2 + v ** 2) + 0.002


def test_motion_boundary_matches_lai_reference_code():
    g = torch.Generator().manual_seed(3)
    low = 4 * torch.randn(2, 2, 8, 8, generator=g)
    flow = torch.nn.functional.interpolate(low, size=(H, W), mode='bilinear', align_corners=False)
    flow[:, :, 60:, :] += 1.5  # a sharp jump too
    got = metric_util._motion_boundary(flow)
    assert got.shape == (2, 1, H, W) and got.dtype == torch.bool
    for i in range(2):
        assert torch.equal(got[i, 0], _lai_motion_boundary(flow[i]))


def test_uniform_motion_has_no_motion_boundary():
    flow = torch.tensor([5.0, -3.0]).view(1, 2, 1, 1).expand(1, 2, H, W)
    assert not metric_util._motion_boundary(flow).any()


def test_motion_boundary_flags_the_pixel_before_a_jump():
    flow = torch.zeros(1, 2, H, W)
    flow[:, 1, 64:, :] = 3.0  # lower half moves down 3 px, upper half still
    boundary = metric_util._motion_boundary(flow)[0, 0]
    assert boundary[63].all()  # still side, just before the jump
    assert not boundary[:63].any() and not boundary[64:].any()


def test_motion_boundary_tolerance_grows_with_motion():
    # u ramps by 0.3 px per column: |grad|^2 = 0.09 everywhere but the last column.
    ramp = 0.3 * torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    slow = torch.zeros(1, 2, H, W)
    slow[0, 0] = ramp - ramp.mean()                # |w| small near the middle
    fast = slow.clone()
    fast[0, 1] = 20.0                              # same ramp, plus fast vertical motion
    mid = (slice(None), slice(W // 2 - 2, W // 2 + 2))
    assert metric_util._motion_boundary(slow)[0, 0][mid].all()
    assert not metric_util._motion_boundary(fast)[0, 0][mid].any()


def test_pixels_on_motion_boundaries_are_ignored():
    # Flat grey frames, so a wrong flow can't change the warp; frame t+1 is corrupted only
    # on row 63. The backward flow jumps by 0.5 px below row 63: small enough to pass the
    # forward-backward check, but a motion boundary on row 63.
    frames = torch.full((2, 3, H, W), 0.5)
    frames[1, :, 63, :] = 1.0
    masks = torch.zeros(2, H, W, dtype=torch.bool)
    masks[:, 32:96, 32:96] = True
    backward = torch.zeros(1, 2, H, W)
    backward[:, 1, 64:, :] = 0.5
    forward = -backward

    scores = metric_util.flow_aligned_error(frames, masks, flow=(forward, backward))
    assert scores.per_pair == pytest.approx([0.0], abs=1e-6)
