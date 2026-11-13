"""Clip training of temporal MATCH (#37): the loss on a clip, the optimizer's parameter groups, starting a
fine-tune from the pretrained weights, and the training configs.

The loss tests reuse the tiny CPU model of test_temporal_match_model.py (64x64 UV grid, 2 views of 32x24
pixels, no Sapiens). They need assets/ for the template mesh and the pretrained config.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_clip_training.py
"""
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from test_temporal_match_model import (  # noqa: E402
    as_batch, as_clip, build, synthetic_frame)

# One weight per dataset, as MatchRunner.init_loss_weights leaves them. The geometry, scale and opacity
# losses run on the CPU; the image losses (MSE, L1, SSIM) need the CUDA rasterizer. No LPIPS network.
GEOMETRY_LOSSES = dict(
    geom_only_steps=0, lpips_weight=[0.], render_weight=[0.], l1_weight=[0.], ssim_weight=[0.],
    xyz_weight=[1e-3], mesh_vert_weight=[1e-3], scale_weight=[1e-3], opacity_weight=[1e-3],
    num_render_views=2, output_bg='white', noloss_masks=[],
)
ALL_LOSSES = dict(GEOMETRY_LOSSES, render_weight=[0.5], l1_weight=[0.8], ssim_weight=[0.2])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='the Gaussian rasterizer needs CUDA')


@pytest.fixture(scope='module')
def frames():
  num_verts = build(temporal=False).num_verts
  return [synthetic_frame(num_verts, seed) for seed in range(3)]


def per_frame_twin(temporal_model):
  """Per-frame MATCH with the temporal model's spatial weights and options."""
  options = {k: getattr(temporal_model.opt, k) for k in ALL_LOSSES}
  model = build(temporal=False, **options).to(next(temporal_model.parameters()).device)
  missing, _ = model.load_state_dict(temporal_model.state_dict(), strict=False)
  assert not missing
  return model


def loss(model, batch) -> torch.Tensor:
  return model.compute_loss(batch, lpips_loss=None, step=1)['loss']


def to(batch: dict, device: str) -> dict:
  return {k: v.to(device) for k, v in batch.items()}


@pytest.mark.parametrize('losses, device', [
    (GEOMETRY_LOSSES, 'cpu'), pytest.param(ALL_LOSSES, 'cuda', marks=needs_cuda)], ids=['geometry', 'all'])
@torch.no_grad()
def test_the_loss_on_a_clip_is_the_sum_of_per_frame_matchs_losses_on_its_frames(frames, losses, device):
  temporal = build(temporal=True, clip_length=3, **losses).to(device)
  per_frame = per_frame_twin(temporal)

  expected = sum(loss(per_frame, to(as_batch(frame), device)) for frame in frames)

  torch.testing.assert_close(loss(temporal, to(as_clip(frames), device)), expected, atol=1e-5, rtol=1e-5)
