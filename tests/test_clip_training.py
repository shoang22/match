"""Clip training of temporal MATCH (#37): the loss on a clip, the optimizer's parameter groups, starting a
fine-tune from the pretrained weights, and the training configs.

The loss tests reuse the tiny CPU model of tests/tiny_match.py (64x64 UV grid, 2 views of 32x24
pixels, no Sapiens). They need assets/ for the template mesh and the pretrained config.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_clip_training.py
"""
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from tiny_match import (  # noqa: E402
    ALL_LOSSES, GEOMETRY_LOSSES, as_batch, as_clip, build, clip_loss_and_grads, loss, needs_cuda, per_frame_twin, to)
from training_configs import parsed_options  # noqa: E402


@pytest.mark.parametrize('losses, device', [
    (GEOMETRY_LOSSES, 'cpu'), pytest.param(ALL_LOSSES, 'cuda', marks=needs_cuda)], ids=['geometry', 'all'])
@torch.no_grad()
def test_the_loss_on_a_clip_is_the_sum_of_per_frame_matchs_losses_on_its_frames(tiny_frames, losses, device):
  temporal = build(temporal=True, clip_length=3, **losses).to(device)
  per_frame = per_frame_twin(temporal)

  expected = sum(loss(per_frame, to(as_batch(frame), device)) for frame in tiny_frames)

  torch.testing.assert_close(loss(temporal, to(as_clip(tiny_frames), device)), expected, atol=1e-5, rtol=1e-5)


@pytest.fixture(scope='module')
def lpips_vgg():
  import lpips
  return lpips.LPIPS(net='vgg', verbose=False).requires_grad_(False).eval()


@pytest.mark.parametrize('losses, device', [
    (GEOMETRY_LOSSES, 'cpu'), pytest.param(dict(ALL_LOSSES, lpips_weight=[1.]), 'cuda', marks=needs_cuda)],
    ids=['geometry', 'all'])
def test_gradient_checkpointing_does_not_change_a_clip_loss_or_its_gradients(tiny_frames, losses, device, request):
  lpips_loss = request.getfixturevalue('lpips_vgg').to(device) if device == 'cuda' else None
  expected_loss, expected_grads = clip_loss_and_grads(tiny_frames, False, losses, device, lpips_loss)
  value, grads = clip_loss_and_grads(tiny_frames, True, losses, device, lpips_loss)

  torch.testing.assert_close(value, expected_loss, atol=1e-6, rtol=1e-5)
  assert grads.keys() == expected_grads.keys()
  assert all(g.abs().max() > 0 for g in expected_grads.values() if g is not None)
  for n, g in expected_grads.items():
    torch.testing.assert_close(grads[n], g, atol=1e-6, rtol=1e-4, msg=n)


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


TRAIN_CONFIG = 'configs/match/train_temporal_match.gin'
SMOKE_CONFIG = 'configs/match/train_temporal_match_smoketest.gin'


def heldout_subjects() -> set[str]:
  import json
  with open('assets/ava256/validation_ids.json') as f:
    return set(json.load(f))


def subjects_kept(dataset_kwargs, subjects, root) -> set[str]:
  """The subjects among `subjects` that the Ava-256 dataset keeps with `dataset_kwargs`'s subject split."""
  from match.data.ava256_dataset import AvaMultiCaptureDataset
  for i, sid in enumerate(sorted(subjects)):
    (root / f'20230101--{i:04d}--{sid}' / 'decoder').mkdir(parents=True, exist_ok=True)
  captures, dirs = AvaMultiCaptureDataset.folder_parser(str(root), training=dataset_kwargs.get('training'))
  captures, _ = AvaMultiCaptureDataset.filter_captures(
      str(root), captures, dirs, exclude_subjects=dataset_kwargs.get('exclude_subjects', []),
      only_subjects=dataset_kwargs.get('only_subjects', []))
  return {c.sid for c in captures}


def test_the_default_training_config_fine_tunes_temporal_match_on_4_frame_clips_on_one_gpu():
  opt = parsed_options(TRAIN_CONFIG)
  dataset = opt.train_loader_params['kwargs']['dataset_kwargs']

  assert opt.temporal and opt.clip_length == 4
  assert dataset['clip_length'] == 4 and dataset['clip_stride'] in (1, 2) and dataset['frame_stride'] == 1
  assert opt.grad_checkpoint
  assert opt.temporal_freeze_spatial and opt.temporal_spatial_lr is None  # spatial weights frozen
  assert opt.load_pretrained_model == 'assets/pretrained_models/match/uv_res_1024'
  assert opt.load_pretrained_model_ckpt == 860_000
  # MATCH's 12 input and 12 rendered views, as the pretrained model was trained
  assert opt.num_input_views == opt.num_views == opt.num_render_views == 12
  # a fine-tune from step 0 must not redo the pretrained run's geometry-only phase and LPIPS warmup
  assert opt.geom_only_steps == 0 and opt.lpips_warmup_end == 0
  assert opt.train_loader_params['kwargs']['dataloader_kwargs']['batch_size'] == 1


def test_the_default_training_config_lists_the_heldout_subjects_and_never_trains_on_them(tmp_path):
  opt = parsed_options(TRAIN_CONFIG)
  dataset = opt.train_loader_params['kwargs']['dataset_kwargs']
  training_subjects = {'AAN112', 'ZZZ001'}
  everyone = heldout_subjects() | training_subjects

  assert set(dataset['exclude_subjects']) == heldout_subjects()
  assert subjects_kept(dataset, everyone, tmp_path) == training_subjects
  for val in opt.val_loader_params:
    assert 'clip_length' not in val['kwargs']['dataset_kwargs']  # validation scores single frames
    assert subjects_kept(val['kwargs']['dataset_kwargs'], everyone, tmp_path) <= heldout_subjects()


def test_the_smoke_config_trains_a_few_clip_steps_on_the_one_segment_with_tempeh_at_every_frame():
  default, smoke = parsed_options(TRAIN_CONFIG), parsed_options(SMOKE_CONFIG)
  dataset = smoke.train_loader_params['kwargs']['dataset_kwargs']

  assert smoke.max_train_steps <= 10
  assert dataset['only_subjects'] == ['TCE049'] and dataset['segment'] == 'SEN_a_good_morrow_to_you_my_boy'
  assert dataset['uv_directory'] == 'data/tempeh_predictions/ava-256/seq_SEN_a_good_morrow/ava-256'
  assert dataset['clip_length'] == 4 and dataset['frame_stride'] == 1
  for k in ('temporal', 'clip_length', 'grad_checkpoint', 'temporal_freeze_spatial', 'load_pretrained_model',
            'num_input_views', 'num_views', 'num_render_views', 'optimizer'):
    assert getattr(smoke, k) == getattr(default, k), k
  assert smoke.base_folder != default.base_folder
