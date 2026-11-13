"""The image/UV patch correspondence search (uv_util.get_img_uv_patch_correspondences) gives the same matches
however many views it processes at once, and its outputs hold no more memory than their own size.

CPU only, synthetic UV renders:
    CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_uv_correspondences.py
"""
import pytest
import torch

from match.utils import uv_util

ARGS = dict(uv_res=64, uv_patch_size=16, img_patch_size=8, n_neighbors=10, downsample_for_computation=4)


def uv_renders(b=2, v=3, h=40, w=32, seed=0):
  """UV renders (B, V, 3, H, W): u, v in [0, 1] and a coverage channel with some uncovered pixels."""
  g = torch.Generator().manual_seed(seed)
  uv = torch.rand(b, v, 3, h, w, generator=g)
  uv[:, :, 2] = (torch.rand(b, v, h, w, generator=g) > 0.2).float()
  return uv


@pytest.mark.parametrize('views_per_chunk', [1, 2])
def test_processing_views_in_chunks_gives_the_same_matches(views_per_chunk):
  uv = uv_renders()
  expected_idcs, expected_scores = uv_util.get_img_uv_patch_correspondences(
      uv, return_uv_patch_mask=False, views_per_chunk=None, **ARGS)

  idcs, scores = uv_util.get_img_uv_patch_correspondences(
      uv, return_uv_patch_mask=False, views_per_chunk=views_per_chunk, **ARGS)

  assert torch.equal(idcs, expected_idcs)
  assert torch.equal(scores, expected_scores)


def test_the_matches_keep_the_patch_masks_when_asked():
  uv = uv_renders()
  expected = uv_util.get_img_uv_patch_correspondences(uv, return_uv_patch_mask=True, views_per_chunk=None, **ARGS)
  chunked = uv_util.get_img_uv_patch_correspondences(uv, return_uv_patch_mask=True, views_per_chunk=1, **ARGS)
  for e, c in zip(expected, chunked):
    assert torch.equal(c, e)


def test_the_matched_indices_do_not_keep_the_full_ranking_alive():
  idcs, scores = uv_util.get_img_uv_patch_correspondences(uv_renders(), return_uv_patch_mask=False, **ARGS)
  for t in (idcs, scores):
    assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
