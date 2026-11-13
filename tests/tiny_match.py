"""A tiny MatchModel on CPU and synthetic inputs, shared by the model, clip-training and temporal-loss tests.

The model: 64x64 UV grid, one block pair, 2 views of 32x24 pixels, no Sapiens. Building it needs assets/ (template
mesh, segmentation labels), so test modules that import this one skip first when assets/ is missing.
"""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from match.models.match_model import MatchModel, checkpoint_allow_missing
from match.options import Options
from match.utils import general_util

N_VIEWS = 2
HEIGHT, WIDTH = 32, 24
ATOL = 1e-5

# One weight per dataset, as MatchRunner.init_loss_weights leaves them. The geometry, scale and opacity
# losses run on the CPU; the image losses (MSE, L1, SSIM) need the CUDA rasterizer. No LPIPS network.
GEOMETRY_LOSSES = dict(
    geom_only_steps=0, lpips_weight=[0.], render_weight=[0.], l1_weight=[0.], ssim_weight=[0.],
    xyz_weight=[1e-3], mesh_vert_weight=[1e-3], scale_weight=[1e-3], opacity_weight=[1e-3],
    num_render_views=2, output_bg='white', noloss_masks=[],
)
ALL_LOSSES = dict(GEOMETRY_LOSSES, render_weight=[0.5], l1_weight=[0.8], ssim_weight=[0.2])
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='the Gaussian rasterizer needs CUDA')


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


def per_frame_twin(model: MatchModel) -> MatchModel:
  """Per-frame MATCH with `model`'s spatial weights and loss options, on its device."""
  options = {k: getattr(model.opt, k) for k in ALL_LOSSES}
  twin = build(temporal=False, **options).to(next(model.parameters()).device)
  missing, _ = twin.load_state_dict(model.state_dict(), strict=False)
  assert not missing
  return twin


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


def synthetic_frames(n=3) -> list[dict]:
  """n consecutive synthetic frames (seeds 0..n-1) for the tiny model's template."""
  num_verts = build(temporal=False).num_verts
  return [synthetic_frame(num_verts, seed) for seed in range(n)]


def as_batch(frame: dict) -> dict:
  return {k: v[None] for k, v in frame.items()}


def as_clip(frames: list[dict]) -> dict:
  """A batch of one clip: every tensor stacked on a time axis after the batch axis."""
  return {k: torch.stack([f[k] for f in frames])[None] for k in frames[0]}


def to(batch: dict, device: str) -> dict:
  return {k: v.to(device) for k, v in batch.items()}


def loss(model, batch, lpips_loss=None) -> torch.Tensor:
  return model.compute_loss(batch, lpips_loss=lpips_loss, step=1)['loss']


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


def clip_loss_and_grads(frames, grad_checkpoint, losses, device, lpips_loss):
  """The loss of the 3-frame clip on a temporal model with random temporal weights, and its gradients."""
  model = build(temporal=True, clip_length=3, grad_checkpoint=grad_checkpoint, **losses).to(device)
  randomize_temporal_parameters(model)
  model.train()
  torch.manual_seed(0)  # the same random input and render views and background colour in both runs
  value = loss(model, to(as_clip(frames), device), lpips_loss)
  value.backward()
  return value.detach(), {n: p.grad for n, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(state_dict: dict, folder: Path) -> Path:
  """A checkpoint folder as the runner saves it (model.safetensors)."""
  folder.mkdir(parents=True)
  save_file({k: v.contiguous() for k, v in state_dict.items()}, str(folder / 'model.safetensors'))
  return folder


def load_pretrained(model: MatchModel, folder: Path, strict: bool):
  """Loads a checkpoint the way MatchRunner.load_checkpoint does for this model."""
  return general_util.load_ckpt(str(folder), model, strict=strict,
                                allow_missing=checkpoint_allow_missing(model.opt))
