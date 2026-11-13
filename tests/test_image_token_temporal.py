"""Pixel-aligned ablation (#39): a temporal sublayer over image tokens as well as UV tokens, switched on by
`Options.temporal_image_tokens`. One temporal sequence per (camera, image patch position).

Reuses the tiny CPU model of test_temporal_match_model.py (64x64 UV grid, one block pair, 2 views of 32x24
pixels, no Sapiens). Needs assets/ for the template mesh and the pretrained config.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_image_token_temporal.py
"""
import copy
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from test_temporal_match_model import (  # noqa: E402
    ATOL, as_batch, as_clip, build, per_frame_twin, synthetic_frame)

SWITCH_ON = dict(temporal=True, clip_length=3, temporal_image_tokens=True)


@pytest.fixture(scope='module')
def frames():
  num_verts = build(temporal=False).num_verts
  return [synthetic_frame(num_verts, seed) for seed in range(3)]


def names_with(model, part: str) -> set[str]:
  return {n for n, _ in model.named_parameters() if part in n.split('.')}


def test_only_the_switch_gives_the_temporal_model_image_token_temporal_parameters():
  assert names_with(build(temporal=True, clip_length=3), 'temporal_img') == set()

  with_images = build(temporal=True, clip_length=3, temporal_image_tokens=True)
  # One image-token sublayer per block (one CA and one per-view block here), next to the UV-token one.
  assert {n.split('.temporal_img.')[0] for n in names_with(with_images, 'temporal_img')} == {
      'transformer.blocks.0', 'transformer.blocks.1'}
  for block in ('transformer.blocks.0', 'transformer.blocks.1'):
    assert {n.split('.temporal_img.')[1] for n in names_with(with_images, 'temporal_img') if n.startswith(
        block)} == {n.split('.temporal_uv.')[1] for n in names_with(with_images, 'temporal_uv') if n.startswith(
            block)}


def test_the_image_token_sublayers_train_while_everything_else_is_frozen():
  model = build(temporal=True, clip_length=3, temporal_image_tokens=True)
  trainable = {n for n, p in model.named_parameters() if p.requires_grad}

  assert trainable == names_with(model, 'temporal_img') | names_with(model, 'temporal_uv') | {
      'transformer.temporal_frame_embedding'}


def test_the_image_token_switch_needs_the_temporal_model():
  with pytest.raises(ValueError, match='temporal_image_tokens'):
    build(temporal=False, temporal_image_tokens=True)


def per_frame_match_with(model):
  """Per-frame MATCH with `model`'s spatial weights."""
  twin = per_frame_twin(model)
  assert set(twin.state_dict()) == set(model.state_dict()) - names_with(model, 'temporal_img') - names_with(
      model, 'temporal_uv') - {'transformer.temporal_frame_embedding'}
  return twin


@torch.no_grad()
def test_with_the_switch_on_each_frame_of_a_clip_equals_per_frame_match_at_init(frames):
  model = build(**SWITCH_ON)
  per_frame = per_frame_match_with(model)

  clip_out = model.forward_gaussians(as_clip(frames), return_tempeh_xyz=True)

  for t, frame in enumerate(frames):
    for k, v in per_frame.forward_gaussians(as_batch(frame), return_tempeh_xyz=True).items():
      torch.testing.assert_close(clip_out[k][:, t], v, atol=ATOL, rtol=0, msg=f'{k} frame {t}')


def randomize(model, part: str, seed=1):
  """Nonzero weights for the parameters under `part` (e.g. 'temporal_img'), as if trained."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for n in sorted(names_with(model, part)):
      p = model.get_parameter(n)
      p.copy_(0.5 * torch.randn(p.shape, generator=g))


@torch.no_grad()
def test_the_image_token_sublayers_alone_carry_information_between_frames(frames):
  # UV-token sublayers stay zero (no-ops); only the image-token ones and the frame embedding are nonzero.
  model = build(**SWITCH_ON)
  randomize(model, 'temporal_img')

  before = model.forward_gaussians(as_clip(frames))
  changed = copy.deepcopy(frames)
  changed[0]['image'] = 1. - changed[0]['image']
  after = model.forward_gaussians(as_clip(changed))

  for t in (1, 2):
    for k in ('xyz', 'rgb', 'rotation'):
      assert (after[k][:, t] - before[k][:, t]).abs().max() > 1e-4, f'{k} frame {t} did not change'


def clip_grads(frames, grad_checkpoint):
  model = build(**SWITCH_ON, grad_checkpoint=grad_checkpoint)
  randomize(model, 'temporal_img')
  randomize(model, 'temporal_uv', seed=2)
  model.train()
  torch.manual_seed(0)  # the same input views in both runs
  out = model.forward_gaussians(as_clip(frames))
  value = out['xyz'].square().mean() + out['rgb'].square().mean()
  value.backward()
  return value.detach(), {n: p.grad for n, p in model.named_parameters() if p.requires_grad}


def test_gradient_checkpointing_keeps_the_image_token_sublayers_forward_and_gradients(frames):
  value, grads = clip_grads(frames, grad_checkpoint=False)
  value_ckpt, grads_ckpt = clip_grads(frames, grad_checkpoint=True)

  torch.testing.assert_close(value_ckpt, value, atol=1e-6, rtol=1e-5)
  assert any(g.abs().max() > 0 for n, g in grads.items() if 'temporal_img' in n)
  for n, g in grads.items():
    torch.testing.assert_close(grads_ckpt[n], g, atol=1e-6, rtol=1e-4, msg=n)


@pytest.mark.parametrize('grad_checkpoint', [False, True])
def test_running_the_temporal_sublayers_in_chunks_of_token_positions_changes_nothing(
    frames, grad_checkpoint, monkeypatch):
  # The full-size image-token sublayer has 61,440 token positions per frame; it runs a bounded number of
  # them at a time. Here 24 image and 16 UV positions per frame, so chunks of 5 split both.
  from match.models.networks import attention
  value, grads = clip_grads(frames, grad_checkpoint)
  monkeypatch.setattr(attention, 'TEMPORAL_CHUNK_POSITIONS', 5)
  value_chunked, grads_chunked = clip_grads(frames, grad_checkpoint)

  torch.testing.assert_close(value_chunked, value, atol=1e-6, rtol=1e-5)
  for n, g in grads.items():
    torch.testing.assert_close(grads_chunked[n], g, atol=1e-6, rtol=1e-4, msg=n)


def test_a_per_frame_checkpoint_loads_into_the_ablation_with_fresh_image_token_sublayers(tmp_path):
  from test_temporal_match_model import load_pretrained, save_checkpoint
  per_frame = build(seed=1, temporal=False)
  folder = save_checkpoint(per_frame.state_dict(), tmp_path / 'ckpt')
  model = build(seed=2, **SWITCH_ON)
  fresh = {k: v.clone() for k, v in model.state_dict().items() if k not in per_frame.state_dict()}
  assert names_with(model, 'temporal_img') <= set(fresh)

  load_pretrained(model, folder, strict=True)

  loaded = model.state_dict()
  for k, v in {**per_frame.state_dict(), **fresh}.items():
    torch.testing.assert_close(loaded[k], v, atol=0, rtol=0, msg=k)


@pytest.mark.parametrize('stage1_config, ablation_config, ablation_folder', [
    ('train_temporal_match.gin', 'train_temporal_match_image_tokens.gin', 'temporal_match_image_tokens'),
    ('train_temporal_match_smoketest.gin', 'train_temporal_match_image_tokens_smoketest.gin',
     'temporal_match_image_tokens_smoketest'),
], ids=['training', 'smoke'])
def test_the_ablation_config_differs_from_the_stage1_temporal_config_only_in_the_switch(
    stage1_config, ablation_config, ablation_folder):
  import dataclasses

  from test_clip_training import parsed_options
  stage1 = parsed_options(f'configs/match/{stage1_config}')
  ablation = parsed_options(f'configs/match/{ablation_config}')
  differing = {f.name for f in dataclasses.fields(stage1) if getattr(stage1, f.name) != getattr(ablation, f.name)}

  assert not stage1.temporal_image_tokens and ablation.temporal_image_tokens
  # base_folder: the run's own output folder (its NAME), so the two runs never resume each other's checkpoints
  assert differing == {'temporal_image_tokens', 'base_folder'}
  assert ablation.base_folder == f'experiments/match/{ablation_folder}'
