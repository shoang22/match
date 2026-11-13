"""Temporal MATCH: the model forward on a clip (a batch with a time axis), its trainable parameters and
loading the pretrained checkpoint into it.

The forward tests build a tiny model (64x64 UV grid, one block pair, 2 views of 32x24 pixels, no Sapiens) and
feed it synthetic inputs on CPU (tests/tiny_match.py). They need assets/ for the template mesh. The checkpoint test builds the
full-size pretrained configuration (~40 s on CPU) and loads assets/pretrained_models/match/uv_res_1024.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_temporal_match_model.py
"""
import copy
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from safetensors.torch import load_file  # noqa: E402

from match.models.match_model import MatchModel, checkpoint_allow_missing  # noqa: E402
from match.options import Options  # noqa: E402
from match.utils import general_util  # noqa: E402
from tiny_match import (  # noqa: E402
    ATOL, as_batch, as_clip, build, load_pretrained, per_frame_twin, randomize_temporal_parameters, save_checkpoint,
    temporal_parameter_names)


@torch.no_grad()
def test_at_init_each_frame_of_a_clip_equals_per_frame_match(tiny_frames):
  temporal = build(temporal=True, clip_length=3)
  per_frame = per_frame_twin(temporal)

  clip_out = temporal.forward_gaussians(as_clip(tiny_frames), return_tempeh_xyz=True)

  for t, frame in enumerate(tiny_frames):
    expected = per_frame.forward_gaussians(as_batch(frame), return_tempeh_xyz=True)
    assert 'tempeh_xyz' in expected
    for k, v in expected.items():
      assert clip_out[k].shape == (1, len(tiny_frames)) + v.shape[1:], k
      torch.testing.assert_close(clip_out[k][:, t], v, atol=ATOL, rtol=0, msg=f'{k} frame {t}')


@torch.no_grad()
def test_with_nonzero_temporal_weights_changing_one_frame_changes_the_others(tiny_frames):
  model = build(temporal=True, clip_length=3)
  randomize_temporal_parameters(model)

  before = model.forward_gaussians(as_clip(tiny_frames))
  changed = copy.deepcopy(tiny_frames)
  changed[0]['image'] = 1. - changed[0]['image']
  changed[0]['coarse_verts'] = changed[0]['coarse_verts'] * 1.1
  after = model.forward_gaussians(as_clip(changed))

  # The pretrained-style init zeroes the scale and opacity heads, so only these depend on the tokens here.
  for t in (1, 2):
    for k in ('xyz', 'rgb', 'rotation'):
      assert (after[k][:, t] - before[k][:, t]).abs().max() > 1e-4, f'{k} frame {t} did not change'


@torch.no_grad()
def test_a_clip_of_one_frame_gives_per_frame_match_even_with_trained_temporal_weights(tiny_frames):
  model = build(temporal=True, clip_length=3)
  randomize_temporal_parameters(model)
  expected = per_frame_twin(model).forward_gaussians(as_batch(tiny_frames[0]))

  clip_out = model.forward_gaussians(as_clip(tiny_frames[:1]))
  frame_out = model.forward_gaussians(as_batch(tiny_frames[0]))

  for k, v in expected.items():
    torch.testing.assert_close(clip_out[k][:, 0], v, atol=ATOL, rtol=0, msg=k)
    torch.testing.assert_close(frame_out[k], v, atol=ATOL, rtol=0, msg=k)


def test_only_the_temporal_parameters_train():
  model = build(temporal=True, clip_length=3)
  per_frame_names = set(build(temporal=False).state_dict())
  temporal_names = {n for n, _ in model.named_parameters()} - per_frame_names
  trainable = {n for n, p in model.named_parameters() if p.requires_grad}

  assert trainable == temporal_names
  # Per block (one CA and one per-view block here): a temporal sublayer of an RMSNorm (dim), q/k/v/o
  # projections (4 dim^2) and q/k LayerNorms (4 dim); plus one embedding per frame slot.
  dim, n_blocks, clip_len = 32, 2, 3
  assert sum(model.get_parameter(n).numel() for n in trainable) == (
      n_blocks * (dim + 4 * dim * dim + 4 * dim) + clip_len * dim)


@pytest.mark.parametrize('grad_checkpoint', [False, True])
def test_a_clip_loss_backpropagates_into_the_temporal_output_projections_only(tiny_frames, grad_checkpoint):
  model = build(temporal=True, clip_length=3, grad_checkpoint=grad_checkpoint).train()

  model.forward_gaussians(as_clip(tiny_frames))['xyz'].square().mean().backward()

  grads = {n: p.grad for n, p in model.named_parameters()}
  # At init the zero output projections are the only temporal weights with a gradient: they gate the rest.
  output_projections = {n for n in grads if n.endswith('temporal_uv.attn.wo.weight')}
  assert len(output_projections) == 2
  assert all(grads[n].abs().max() > 0 for n in output_projections)
  assert all(g is None for n, g in grads.items() if n not in temporal_parameter_names(model))


def test_per_frame_match_trains_every_parameter():
  model = build(temporal=False)
  assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('strict', [True, False])
def test_a_per_frame_checkpoint_loads_into_the_temporal_model_with_fresh_temporal_parameters(tmp_path, strict):
  per_frame = build(seed=1, temporal=False)
  folder = save_checkpoint(per_frame.state_dict(), tmp_path / 'ckpt')
  model = build(seed=2, temporal=True, clip_length=3)
  fresh = {k: v.clone() for k, v in model.state_dict().items() if k not in per_frame.state_dict()}

  load_pretrained(model, folder, strict)

  loaded = model.state_dict()
  for k, v in per_frame.state_dict().items():
    torch.testing.assert_close(loaded[k], v, atol=0, rtol=0, msg=k)
  for k, v in fresh.items():
    torch.testing.assert_close(loaded[k], v, atol=0, rtol=0, msg=k)


@pytest.mark.parametrize('strict', [True, False])
def test_an_unexpected_non_temporal_key_fails_the_temporal_load(tmp_path, strict):
  state = build(temporal=False).state_dict()
  state['transformer.blocks.0.attn.extra.weight'] = torch.zeros(3)
  folder = save_checkpoint(state, tmp_path / 'ckpt')

  with pytest.raises(Exception, match='attn.extra.weight'):
    load_pretrained(build(temporal=True, clip_length=3), folder, strict)


@pytest.mark.parametrize('strict', [True, False])
def test_a_missing_non_temporal_key_fails_the_temporal_load(tmp_path, strict):
  state = build(temporal=False).state_dict()
  del state['out_xyz.weight']
  folder = save_checkpoint(state, tmp_path / 'ckpt')

  with pytest.raises(Exception, match='out_xyz.weight'):
    load_pretrained(build(temporal=True, clip_length=3), folder, strict)


PRETRAINED = Path('assets/pretrained_models/match/uv_res_1024')


class SapiensStub:
  """Stands in for the 4.6 GB Sapiens torchscript, which is not an nn.Module and holds no checkpoint keys."""
  C = 1536

  def __init__(self, *args, **kwargs):
    pass

  def to(self, device):
    return self


@pytest.fixture
def pretrained_options(monkeypatch):
  import gin
  from match.models import match_model
  from match.utils import gin_util  # noqa: F401  registers join_strings for config_0.gin

  monkeypatch.setattr(match_model, 'SapiensEncoder', SapiensStub)
  gin.clear_config()
  gin.parse_config_files_and_bindings([str(PRETRAINED / 'config_0.gin')], None, skip_unknown=True)
  try:
    yield Options()
  finally:
    gin.clear_config()


@pytest.mark.skipif(not PRETRAINED.exists(), reason=f'needs {PRETRAINED}')
def test_the_pretrained_checkpoint_loads_into_temporal_match(pretrained_options):
  opt = pretrained_options
  assert opt.load_pretrained_model_strict is False  # the shipped config; the temporal load is strict anyway
  opt.temporal, opt.clip_length = True, 4
  model = MatchModel(opt)
  folder = general_util.get_ckpt_path(str(PRETRAINED / 'checkpoints'), -1)

  general_util.load_ckpt(folder, model, strict=opt.load_pretrained_model_strict,
                         allow_missing=checkpoint_allow_missing(opt))

  checkpoint = load_file(f'{folder}/model.safetensors')
  loaded = model.state_dict()
  assert len(checkpoint) == 184
  for k, v in checkpoint.items():
    # template_triangle_uvs_orig is float64 in the checkpoint and float32 in the model, as before
    torch.testing.assert_close(loaded[k], v, atol=0, rtol=0, check_dtype=False, msg=k)
  fresh = set(loaded) - set(checkpoint)
  assert fresh and all('temporal' in k for k in fresh)
  for block in range(2 * opt.num_blocks):
    assert (loaded[f'transformer.blocks.{block}.temporal_uv.attn.wo.weight'] == 0).all()
  assert (loaded['transformer.temporal_frame_embedding'] == 0).all()


@pytest.mark.skipif(not PRETRAINED.exists(), reason=f'needs {PRETRAINED}')
def test_with_the_switch_off_the_model_has_exactly_the_pretrained_checkpoints_keys(pretrained_options):
  opt = pretrained_options
  assert opt.temporal is False
  model = MatchModel(opt)
  folder = general_util.get_ckpt_path(str(PRETRAINED / 'checkpoints'), -1)

  general_util.load_ckpt(folder, model, strict=True, allow_missing=checkpoint_allow_missing(opt))

  assert set(model.state_dict()) == set(load_file(f'{folder}/model.safetensors'))
  assert all(p.requires_grad for p in model.parameters())


def test_folding_a_collated_clip_batch_makes_every_frame_a_sample():
  from torch.utils.data import default_collate
  from match.utils.data_util import MatchBatch

  def clip(subject, tiny_frames):  # what a dataset in clip mode returns
    return {'image': torch.stack([torch.full((2, 3, 4, 4), float(f)) for f in tiny_frames]),
            'scene_scale': torch.tensor([float(f) for f in tiny_frames]),
            'frame': [str(f) for f in tiny_frames], 'subject': subject}

  batch = MatchBatch(default_collate([clip('A', [1, 2, 3]), clip('B', [7, 8, 9])]))
  folded, clip_len = batch.fold_clip()

  assert clip_len == 3
  assert folded['image'].shape == (6, 2, 3, 4, 4)
  assert folded['image'][:, 0, 0, 0, 0].tolist() == [1, 2, 3, 7, 8, 9]
  assert folded['scene_scale'].tolist() == [1, 2, 3, 7, 8, 9]
  assert folded['frame'] == ['1', '2', '3', '7', '8', '9']
  assert folded['subject'] == ['A', 'A', 'A', 'B', 'B', 'B']
