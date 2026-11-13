"""Temporal MATCH: the model forward on a clip (a batch with a time axis), its trainable parameters and
loading the pretrained checkpoint into it.

The forward tests build a tiny model (64x64 UV grid, one block pair, 2 views of 32x24 pixels, no Sapiens) and
feed it synthetic inputs on CPU. They need assets/ for the template mesh. The checkpoint test builds the
full-size pretrained configuration (~40 s on CPU) and loads assets/pretrained_models/match/uv_res_1024.

    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_temporal_match_model.py
"""
import copy
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from safetensors.torch import load_file, save_file  # noqa: E402

from match.models.match_model import MatchModel, checkpoint_allow_missing  # noqa: E402
from match.options import Options  # noqa: E402
from match.utils import general_util  # noqa: E402

N_VIEWS = 2
HEIGHT, WIDTH = 32, 24
ATOL = 1e-5


def tiny_options(**overrides) -> Options:
  opt = Options(
      uv_res=64, uv_patch_size=16, img_patch_size=8, dim=32, num_heads=2, num_blocks=1,
      n_neighbors=4, num_input_views=N_VIEWS, input_res=HEIGHT, input_bg='white',
      use_sapiens_features=False, skip_connection=True, grad_checkpoint=False,
      segmentation_labels_path='assets/sapiens/segmentation_labels.json',
  )
  for k, v in overrides.items():
    assert hasattr(opt, k), f'Options has no field {k}'
    setattr(opt, k, v)
  return opt


def build(seed=0, **overrides) -> MatchModel:
  torch.manual_seed(seed)
  return MatchModel(tiny_options(**overrides)).eval()


def per_frame_twin(temporal_model: MatchModel) -> MatchModel:
  """Per-frame MATCH with the temporal model's spatial weights."""
  model = build(temporal=False)
  missing, unexpected = model.load_state_dict(temporal_model.state_dict(), strict=False)
  assert not missing
  return model


def look_at_origin(position):
  """Camera-to-world matrix (OpenCV axes: x right, y down, z forward) for a camera at `position`."""
  position = torch.tensor(position, dtype=torch.float32)
  forward = -position / position.norm()
  right = torch.linalg.cross(forward, torch.tensor([0., -1., 0.]))
  right = right / right.norm()
  down = torch.linalg.cross(forward, right)
  c2w = torch.eye(4)
  c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
  return c2w


def synthetic_frame(num_verts, seed) -> dict:
  """One frame of inputs for N_VIEWS cameras around the origin, unbatched (as a dataset returns it)."""
  g = torch.Generator().manual_seed(seed)
  uv = torch.rand(N_VIEWS, 3, HEIGHT, WIDTH, generator=g)
  uv[:, 2] = 1.  # every pixel shows the surface
  return {
      'image': torch.rand(N_VIEWS, 3, HEIGHT, WIDTH, generator=g),
      'mask': torch.ones(N_VIEWS, 1, HEIGHT, WIDTH),
      'uv': uv,
      'C2W': torch.stack([look_at_origin([0.3, 0., 1.]), look_at_origin([-0.3, 0., 1.])]),
      'fxfycxcy': torch.tensor([[30., 30., WIDTH / 2, HEIGHT / 2]] * N_VIEWS),
      'coarse_verts': 0.1 * torch.randn(num_verts, 3, generator=g),
      'verts': torch.zeros(num_verts, 3),
      'idindex': torch.tensor(seed),
  }


def as_batch(frame: dict) -> dict:
  return {k: v[None] for k, v in frame.items()}


def as_clip(frames: list[dict]) -> dict:
  """A batch of one clip: every tensor stacked on a time axis after the batch axis."""
  return {k: torch.stack([f[k] for f in frames])[None] for k in frames[0]}


@pytest.fixture(scope='module')
def frames():
  num_verts = build(temporal=False).num_verts
  return [synthetic_frame(num_verts, seed) for seed in range(3)]


@torch.no_grad()
def test_at_init_each_frame_of_a_clip_equals_per_frame_match(frames):
  temporal = build(temporal=True, clip_length=3)
  per_frame = per_frame_twin(temporal)

  clip_out = temporal.forward_gaussians(as_clip(frames))

  for t, frame in enumerate(frames):
    expected = per_frame.forward_gaussians(as_batch(frame))
    for k, v in expected.items():
      assert clip_out[k].shape == (1, len(frames)) + v.shape[1:], k
      torch.testing.assert_close(clip_out[k][:, t], v, atol=ATOL, rtol=0, msg=f'{k} frame {t}')


def temporal_parameter_names(temporal_model: MatchModel) -> set[str]:
  """Parameters the temporal model has and per-frame MATCH doesn't."""
  per_frame_names = set(build(temporal=False).state_dict())
  return {n for n, _ in temporal_model.named_parameters()} - per_frame_names


def randomize_temporal_parameters(model: MatchModel, seed=1):
  g = torch.Generator().manual_seed(seed)
  names = temporal_parameter_names(model)
  assert names, 'the temporal model has no temporal parameters'
  with torch.no_grad():
    for n, p in model.named_parameters():
      if n in names:
        p.copy_(0.5 * torch.randn(p.shape, generator=g))


@torch.no_grad()
def test_with_nonzero_temporal_weights_changing_one_frame_changes_the_others(frames):
  model = build(temporal=True, clip_length=3)
  randomize_temporal_parameters(model)

  before = model.forward_gaussians(as_clip(frames))
  changed = copy.deepcopy(frames)
  changed[0]['image'] = 1. - changed[0]['image']
  changed[0]['coarse_verts'] = changed[0]['coarse_verts'] * 1.1
  after = model.forward_gaussians(as_clip(changed))

  # The pretrained-style init zeroes the scale and opacity heads, so only these depend on the tokens here.
  for t in (1, 2):
    for k in ('xyz', 'rgb', 'rotation'):
      assert (after[k][:, t] - before[k][:, t]).abs().max() > 1e-4, f'{k} frame {t} did not change'


@torch.no_grad()
def test_a_clip_of_one_frame_gives_per_frame_match_even_with_trained_temporal_weights(frames):
  model = build(temporal=True, clip_length=3)
  randomize_temporal_parameters(model)
  expected = per_frame_twin(model).forward_gaussians(as_batch(frames[0]))

  clip_out = model.forward_gaussians(as_clip(frames[:1]))
  frame_out = model.forward_gaussians(as_batch(frames[0]))

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
def test_a_clip_loss_backpropagates_into_the_temporal_output_projections_only(frames, grad_checkpoint):
  model = build(temporal=True, clip_length=3, grad_checkpoint=grad_checkpoint).train()

  model.forward_gaussians(as_clip(frames))['xyz'].square().mean().backward()

  grads = {n: p.grad for n, p in model.named_parameters()}
  # At init the zero output projections are the only temporal weights with a gradient: they gate the rest.
  output_projections = {n for n in grads if n.endswith('temporal_uv.attn.wo.weight')}
  assert len(output_projections) == 2
  assert all(grads[n].abs().max() > 0 for n in output_projections)
  assert all(g is None for n, g in grads.items() if n not in temporal_parameter_names(model))


def test_per_frame_match_trains_every_parameter():
  model = build(temporal=False)
  assert all(p.requires_grad for p in model.parameters())


def save_checkpoint(state_dict: dict, folder: Path) -> Path:
  """A checkpoint folder as the runner saves it (model.safetensors)."""
  folder.mkdir(parents=True)
  save_file({k: v.contiguous() for k, v in state_dict.items()}, str(folder / 'model.safetensors'))
  return folder


def load_pretrained(model: MatchModel, folder: Path, strict: bool):
  """Loads a checkpoint the way MatchRunner.load_checkpoint does for this model."""
  return general_util.load_ckpt(str(folder), model, strict=strict,
                                allow_missing=checkpoint_allow_missing(model.opt))


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

  def clip(subject, frames):  # what a dataset in clip mode returns
    return {'image': torch.stack([torch.full((2, 3, 4, 4), float(f)) for f in frames]),
            'scene_scale': torch.tensor([float(f) for f in frames]),
            'frame': [str(f) for f in frames], 'subject': subject}

  batch = MatchBatch(default_collate([clip('A', [1, 2, 3]), clip('B', [7, 8, 9])]))
  folded, clip_len = batch.fold_clip()

  assert clip_len == 3
  assert folded['image'].shape == (6, 2, 3, 4, 4)
  assert folded['image'][:, 0, 0, 0, 0].tolist() == [1, 2, 3, 7, 8, 9]
  assert folded['scene_scale'].tolist() == [1, 2, 3, 7, 8, 9]
  assert folded['frame'] == ['1', '2', '3', '7', '8', '9']
  assert folded['subject'] == ['A', 'A', 'A', 'B', 'B', 'B']
