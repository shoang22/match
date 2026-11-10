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
