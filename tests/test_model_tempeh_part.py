"""MatchModel.forward_gaussians can return the TEMPEH part of every texel's position next to the Gaussians.

A texel's final position is the TEMPEH coarse surface at the texel plus the predicted offset. CPU tests on a
tiny untrained model (64x64 UV grid, one block pair) and a synthetic two-view batch; they need assets/ for
the template mesh.
"""
from pathlib import Path

import pytest
import torch

if not Path('assets').exists():
    pytest.skip('assets/ is missing; see the worktree section of CLAUDE.md', allow_module_level=True)

from match.models.match_model import MatchModel  # noqa: E402
from match.options import Options  # noqa: E402

GAUSSIAN_KEYS = ['xyz', 'rgb', 'scale', 'rotation', 'opacity', 'uv']
UV_RES = 64


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    opt = Options(uv_res=UV_RES, uv_patch_size=16, dim=32, num_blocks=1, num_heads=2, n_neighbors=4,
                  num_input_views=2, input_res=32, skip_connection=True, grad_checkpoint=False,
                  segmentation_labels_path='assets/sapiens/segmentation_labels.json')
    return MatchModel(opt).eval()


def head_mesh(model):
    """The template mesh as a TEMPEH mesh: metres, around the origin. (1, Nverts, 3)."""
    verts = model.template_vertices / 1000
    return (verts - verts.mean(dim=0))[None]


def make_batch(coarse_verts, seed=0):
    gen = torch.Generator().manual_seed(seed)
    b, v, h, w = 1, 2, 32, 32
    C2W = torch.eye(4).repeat(b, v, 1, 1)
    C2W[..., 2, 3] = -1.0
    return {'image': torch.rand(b, v, 3, h, w, generator=gen), 'mask': torch.ones(b, v, 1, h, w),
            'uv': torch.rand(b, v, 3, h, w, generator=gen), 'C2W': C2W,
            'fxfycxcy': torch.tensor([1.0, 1.0, 0.5, 0.5]).repeat(b, v, 1),
            'coarse_verts': coarse_verts, 'verts': coarse_verts, 'idindex': torch.zeros(b)}


def forward(model, batch, **kwargs):
    with torch.no_grad():
        return model.forward_gaussians(batch, **kwargs)


def valid_texels(model):
    return model.uv_grid_mask[..., 0]  # (H, W)


def test_the_gaussians_are_unchanged_by_default(model):
    assert list(forward(model, make_batch(head_mesh(model)))) == GAUSSIAN_KEYS


def test_asking_for_the_tempeh_part_adds_it_and_leaves_the_gaussians_as_they_were(model):
    batch = make_batch(head_mesh(model))
    plain = forward(model, batch)
    with_tempeh = forward(model, batch, return_tempeh_xyz=True)

    assert list(with_tempeh) == GAUSSIAN_KEYS + ['tempeh_xyz']
    assert with_tempeh['tempeh_xyz'].shape == (1, 1, 3, UV_RES, UV_RES)
    for key in GAUSSIAN_KEYS:
        torch.testing.assert_close(with_tempeh[key], plain[key], rtol=0, atol=0)


def test_a_mesh_collapsed_to_one_point_puts_every_valid_texel_there(model):
    point = torch.tensor([0.02, -0.05, 0.9])
    tempeh = forward(model, make_batch(point.expand(1, model.num_verts, 3).clone()),
                     return_tempeh_xyz=True)['tempeh_xyz'][0, 0]  # (3, H, W)

    valid = valid_texels(model)
    torch.testing.assert_close(tempeh[:, valid], point[:, None].expand(3, int(valid.sum())))


def test_the_tempeh_part_moves_with_the_mesh_and_the_offset_stays_in_range(model):
    shift = torch.tensor([0.01, 0.0, -0.03])
    mesh = head_mesh(model)
    base = forward(model, make_batch(mesh), return_tempeh_xyz=True)
    moved = forward(model, make_batch(mesh + shift), return_tempeh_xyz=True)

    valid = valid_texels(model)
    torch.testing.assert_close((moved['tempeh_xyz'] - base['tempeh_xyz'])[0, 0][:, valid],
                               shift[:, None].expand(3, int(valid.sum())), atol=1e-6, rtol=0)
    offset = (moved['xyz'] - moved['tempeh_xyz'])[0, 0][:, valid]
    assert offset.abs().max() < 1  # sigmoid * 2 - 1
