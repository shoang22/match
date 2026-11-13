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

from match.models.match_model import MatchModel  # noqa: E402
from match.options import Options  # noqa: E402

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
  temporal = build(temporal=True, clip_len=3)
  per_frame = per_frame_twin(temporal)

  clip_out = temporal.forward_gaussians(as_clip(frames))

  for t, frame in enumerate(frames):
    expected = per_frame.forward_gaussians(as_batch(frame))
    for k, v in expected.items():
      assert clip_out[k].shape == (1, len(frames)) + v.shape[1:], k
      torch.testing.assert_close(clip_out[k][:, t], v, atol=ATOL, rtol=0, msg=f'{k} frame {t}')
