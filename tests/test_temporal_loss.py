"""The temporal trajectory loss (#38): the masked mean of the squared second difference over time of each
texel's position and colour, for the second stage of temporal MATCH's 2x2 experiment.

The loss tests on MatchModel reuse the tiny CPU model of tests/tiny_match.py and need assets/.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_temporal_loss.py
"""
from pathlib import Path

import einops
import pytest
import torch

from match.models.match_model import temporal_trajectory_loss
from tiny_match import (GEOMETRY_LOSSES, as_batch, as_clip, build, clip_loss_and_grads, loss,
                        per_frame_twin)
from training_configs import parsed_options as options


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


needs_assets = pytest.mark.skipif(not Path('assets').exists(), reason='assets/ is missing; see CLAUDE.md')


def clip_loss(frames, **options):
  model = build(clip_length=3, **dict(GEOMETRY_LOSSES, **options))
  return loss(model, as_clip(frames)), model


@needs_assets
@pytest.mark.parametrize('temporal', [True, False], ids=['temporal-attention', 'per-frame'])
@torch.no_grad()
def test_with_zero_weight_a_clip_loss_is_the_sum_of_per_frame_losses(tiny_frames, temporal):
  value, model = clip_loss(tiny_frames, temporal=temporal, temporal_loss_weight=[0.])

  per_frame = per_frame_twin(model)  # per-frame MATCH with the same spatial weights
  expected = sum(loss(per_frame, as_batch(frame)) for frame in tiny_frames)

  torch.testing.assert_close(value, expected, atol=1e-5, rtol=1e-5)


@needs_assets
@pytest.mark.parametrize('temporal', [True, False], ids=['temporal-attention', 'per-frame'])
@torch.no_grad()
def test_the_loss_adds_the_weighted_trajectory_loss_of_texel_positions_in_mm_and_colours(tiny_frames, temporal):
  without, _ = clip_loss(tiny_frames, temporal=temporal, temporal_loss_weight=[0.])
  with_loss, model = clip_loss(tiny_frames, temporal=temporal, temporal_loss_weight=[0.01], temporal_loss_rgb_weight=3.)

  _, masked = model.forward_gaussians(as_clip(tiny_frames), return_masked_gaussians=True)
  texels = {k: einops.rearrange(masked[k], 'b t 1 c l 1 -> b t l c') for k in ('xyz', 'rgb')}
  xyz_term = temporal_trajectory_loss(texels['xyz'] * 1000.)  # final positions (TEMPEH part + offset), mm
  rgb_term = temporal_trajectory_loss(texels['rgb'])
  assert xyz_term > 0 and rgb_term > 0

  torch.testing.assert_close(with_loss - without, (0.01 * (xyz_term + 3. * rgb_term))[0], rtol=1e-4, atol=1e-6)


@needs_assets
def test_gradient_checkpointing_does_not_change_the_loss_or_gradients_with_the_temporal_loss(tiny_frames):
  losses = dict(GEOMETRY_LOSSES, temporal_loss_weight=[0.01])
  expected_loss, expected_grads = clip_loss_and_grads(tiny_frames, False, losses, 'cpu', None)
  value, grads = clip_loss_and_grads(tiny_frames, True, losses, 'cpu', None)

  torch.testing.assert_close(value, expected_loss, atol=1e-6, rtol=1e-5)
  for n, g in expected_grads.items():
    torch.testing.assert_close(grads[n], g, atol=1e-6, rtol=1e-4, msg=n)


@needs_assets
def test_the_temporal_loss_alone_trains_the_temporal_sublayers(tiny_frames):
  """With every other loss off, the trajectory loss of the final positions reaches the temporal parameters."""
  off = dict(geom_only_steps=0, lpips_weight=[0.], render_weight=[0.], l1_weight=[0.], ssim_weight=[0.],
             xyz_weight=[0.], mesh_vert_weight=[0.], scale_weight=[0.], opacity_weight=[0.],
             num_render_views=2, output_bg='white', noloss_masks=[])
  model = build(temporal=True, clip_length=3, temporal_loss_weight=[0.01], **off).train()

  loss(model, as_clip(tiny_frames)).backward()

  # temporal attention's output projections start at zero, so at init only they get a gradient
  wo_grads = [p.grad for n, p in model.named_parameters() if n.endswith('temporal_uv.attn.wo.weight')]
  assert wo_grads and all(g is not None and g.abs().max() > 0 for g in wo_grads)


# --- Training configs ----------------------------------------------------------

STAGE1_CONFIG = 'configs/match/train_temporal_match.gin'
STAGE1_SMOKE_CONFIG = 'configs/match/train_temporal_match_smoketest.gin'
TEMPORAL_CONFIG = 'configs/match/train_temporal_match_temporal_loss.gin'
TEMPORAL_SMOKE_CONFIG = 'configs/match/train_temporal_match_temporal_loss_smoketest.gin'
PER_FRAME_CONFIG = 'configs/match/train_match_clips_temporal_loss.gin'
PER_FRAME_SMOKE_CONFIG = 'configs/match/train_match_clips_temporal_loss_smoketest.gin'


def test_stage_1_trains_without_the_temporal_loss():
  assert options(STAGE1_CONFIG).temporal_loss_weight == 0


@needs_assets
def test_the_temporal_loss_config_is_stage_1_temporal_match_plus_the_temporal_loss():
  stage1, opt = options(STAGE1_CONFIG), options(TEMPORAL_CONFIG)

  assert opt.temporal_loss_weight > 0
  assert opt.temporal and opt.temporal_freeze_spatial and opt.temporal_spatial_lr is None
  for k in ('clip_length', 'train_loader_params', 'val_loader_params', 'optimizer', 'lr_scheduler',
            'load_pretrained_model', 'load_pretrained_model_ckpt', 'grad_checkpoint', 'num_input_views',
            'num_render_views', 'geom_only_steps', 'max_train_steps'):
    assert getattr(opt, k) == getattr(stage1, k), k
  assert opt.base_folder != stage1.base_folder


@needs_assets
def test_the_temporal_loss_weighs_positions_and_colours_by_matchs_loss_convention():
  # MATCH weighs squared-mm geometry losses at 1e-3 (xyz_weight, mesh_vert_weight) and colour losses at ~1
  # (l1 0.8, lpips 1, ssim 0.2), so the trajectory loss's position term (mm^2) gets 1e-3 and its colour term 1.
  opt = options(TEMPORAL_CONFIG)
  assert opt.temporal_loss_weight == pytest.approx(1e-3)
  assert opt.temporal_loss_weight * opt.temporal_loss_rgb_weight == pytest.approx(1.)


@needs_assets
def test_the_per_frame_config_fine_tunes_only_the_offset_head_of_per_frame_match_on_clips_with_the_temporal_loss():
  temporal, opt = options(TEMPORAL_CONFIG), options(PER_FRAME_CONFIG)

  assert not opt.temporal  # no temporal attention
  # spatial weights stay frozen as in the temporal cells, except the layers that predict the per-texel offset
  assert opt.train_offset_head_only
  assert opt.fine_tune  # start at step 0 from the pretrained weights, not the pretrained run's state
  assert opt.temporal_loss_weight == temporal.temporal_loss_weight
  assert opt.temporal_loss_rgb_weight == temporal.temporal_loss_rgb_weight
  assert opt.train_loader_params['kwargs']['dataset_kwargs']['clip_length'] == 4
  for k in ('train_loader_params', 'val_loader_params', 'load_pretrained_model', 'grad_checkpoint',
            'num_input_views', 'num_render_views', 'max_train_steps'):
    assert getattr(opt, k) == getattr(temporal, k), k
  assert opt.base_folder not in (temporal.base_folder, options(STAGE1_CONFIG).base_folder)


@needs_assets
@pytest.mark.parametrize('config, smoke_config', [
    (TEMPORAL_CONFIG, TEMPORAL_SMOKE_CONFIG), (PER_FRAME_CONFIG, PER_FRAME_SMOKE_CONFIG)],
    ids=['temporal-attention', 'per-frame'])
def test_each_smoke_config_is_its_config_on_the_stage_1_smoke_setup(config, smoke_config):
  opt, smoke, stage1_smoke = options(config), options(smoke_config), options(STAGE1_SMOKE_CONFIG)

  for k in ('temporal', 'fine_tune', 'temporal_loss_weight', 'temporal_loss_rgb_weight', 'optimizer',
            'temporal_freeze_spatial', 'train_offset_head_only', 'load_pretrained_model', 'grad_checkpoint',
            'num_input_views', 'num_render_views'):
    assert getattr(smoke, k) == getattr(opt, k), k
  for k in ('train_loader_params', 'val_loader_params', 'max_train_steps', 'lr_scheduler', 'xyz_weight',
            'mesh_vert_weight'):
    assert getattr(smoke, k) == getattr(stage1_smoke, k), k
  assert smoke.base_folder not in (opt.base_folder, stage1_smoke.base_folder)
