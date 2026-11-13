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


def optimizer_of(model):
  """MatchRunner.init_optimizer on `model`, without the rest of the runner (display, accelerator, data)."""
  from match.runner import MatchRunner
  runner = MatchRunner.__new__(MatchRunner)
  runner.model, runner.opt = model, model.opt
  runner.init_optimizer()
  return runner


def names_of(model, params) -> set[str]:
  name_by_id = {id(p): n for n, p in model.named_parameters()}
  return {name_by_id[id(p)] for p in params}


def temporal_names(model) -> set[str]:
  per_frame_names = set(build(temporal=False).state_dict())
  return {n for n, _ in model.named_parameters()} - per_frame_names


def test_by_default_the_optimizer_holds_only_the_temporal_parameters_and_clips_their_gradients():
  model = build(temporal=True, clip_length=3)
  runner = optimizer_of(model)

  groups = runner.optimizer.param_groups
  assert len(groups) == 1
  assert names_of(model, groups[0]['params']) == temporal_names(model)
  assert groups[0]['lr'] == model.opt.optimizer['lr']
  # the parameters MatchRunner.train_step clips the gradient norm of
  assert names_of(model, runner.params_to_optimize) == temporal_names(model)


def test_the_spatial_weights_can_train_as_a_second_group_with_their_own_learning_rate():
  model = build(temporal=True, clip_length=3, temporal_freeze_spatial=False, temporal_spatial_lr=1e-6)
  runner = optimizer_of(model)

  temporal_group, spatial_group = runner.optimizer.param_groups
  everything = {n for n, _ in model.named_parameters()}
  assert names_of(model, temporal_group['params']) == temporal_names(model)
  assert names_of(model, spatial_group['params']) == everything - temporal_names(model)
  assert temporal_group['lr'] == model.opt.optimizer['lr']
  assert spatial_group['lr'] == 1e-6
  assert names_of(model, runner.params_to_optimize) == everything


@pytest.mark.parametrize('options', [
    dict(temporal=True, temporal_spatial_lr=1e-6),  # spatial weights frozen (the default)
    dict(temporal=False, temporal_spatial_lr=1e-6),  # no temporal parameters to set apart
], ids=['frozen', 'per-frame'])
def test_a_spatial_learning_rate_that_would_do_nothing_is_refused(options):
  with pytest.raises(ValueError, match='temporal_spatial_lr'):
    build(**options)


class RecordingAccelerator:
  """Stands in for accelerate.Accelerator in MatchRunner.resume_checkpoint: records which states it loads."""
  is_main_process = True

  def __init__(self):
    self.loaded = []

  def wait_for_everyone(self):
    pass

  def load_state(self, path, strict=True):
    self.loaded.append(path)


def resumed_runner(tmp_path, **overrides):
  """A MatchRunner ready for resume_checkpoint, for a run in tmp_path whose options load a pretrained model
  (an empty stand-in for assets/pretrained_models/match/uv_res_1024 and its training state at step 860000)."""
  import logging
  from match.runner import MatchRunner
  from match.utils import file_util
  pretrained = tmp_path / 'pretrained'
  (pretrained / 'checkpoints' / '860000').mkdir(parents=True)
  opt = build(temporal=True).opt
  opt.load_pretrained_model, opt.load_pretrained_model_ckpt = str(pretrained), 860_000
  for k, v in overrides.items():
    assert hasattr(opt, k), f'Options has no field {k}'
    setattr(opt, k, v)
  runner = MatchRunner.__new__(MatchRunner)
  runner.opt, runner.optimizer, runner.global_update_step = opt, None, 0
  runner.logger = logging.getLogger('test_clip_training')
  runner.accelerator = RecordingAccelerator()
  runner.ckpt_dir = file_util.Path(tmp_path / 'run' / 'checkpoints')
  runner.ckpt_dir.mkdir(parents=True)
  runner.local_tmp_dir = str(tmp_path / 'local')
  return runner


@pytest.mark.parametrize('overrides', [dict(), dict(temporal=False, fine_tune=True)], ids=['temporal', 'fine_tune'])
def test_a_fine_tune_starts_at_step_0_without_the_pretrained_runs_optimizer_and_schedule(tmp_path, overrides):
  runner = resumed_runner(tmp_path, **overrides)

  runner.resume_checkpoint()

  assert runner.accelerator.loaded == []
  assert runner.global_update_step == 0


def test_a_fine_tune_resumes_from_its_own_latest_checkpoint(tmp_path):
  runner = resumed_runner(tmp_path)
  (runner.ckpt_dir / '000020').mkdir()

  runner.resume_checkpoint()

  assert len(runner.accelerator.loaded) == 1
  assert runner.global_update_step == 20


def test_per_frame_training_still_resumes_the_pretrained_run(tmp_path):
  runner = resumed_runner(tmp_path, temporal=False)

  runner.resume_checkpoint()

  assert len(runner.accelerator.loaded) == 1
  assert runner.global_update_step == 860_000
