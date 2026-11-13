"""Correspondences between UV patches and image patches (uv_util): checked against a loop over every pair of
patches written from the scores' definition. CPU, synthetic UV renders.

Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_uv_correspondences.py
"""
import torch

from match.utils.uv_util import bounding_box, get_img_uv_patch_correspondences


def test_bounding_box_of_each_mask_is_its_top_left_bottom_right_and_minus_one_when_empty():
    masks = torch.zeros(3, 5, 6, dtype=torch.bool)
    masks[0, 1:3, 2:5] = True
    masks[1, 4, 0] = True
    masks[1, 0, 5] = True

    assert bounding_box(masks).tolist() == [[1, 2, 2, 4], [0, 0, 4, 5], [-1, -1, -1, -1]]


UV_RES, UV_PATCH, IMG_PATCH, DOWNSAMPLE = 32, 8, 8, 2


def reference_scores(uv_renders):
    """Score of every (UV patch, image patch) pair, (B, nv_uv, nh_uv, V * nv_img * nh_img), one pair at a time.

    A pixel of the image (sampled every DOWNSAMPLE pixels) belongs to UV patch (r, c) if its render is valid and its
    (u, v) falls in the patch, with texel centres at k / (UV_RES - 1) and v pointing up. The score of an image patch
    is the larger of the fraction of its pixels in the UV patch, and 0.1 x the UV patch's pixel count in that view
    over the area of the bounding box around the image patch and the UV patch's pixels.
    """
    renders = uv_renders[..., ::DOWNSAMPLE, ::DOWNSAMPLE]
    b, v, _, h, w = renders.shape
    n_uv, ps = UV_RES // UV_PATCH, IMG_PATCH // DOWNSAMPLE
    px = 1 / (UV_RES - 1)
    edges = [-0.5 * px + px * UV_PATCH * k for k in range(n_uv + 1)]
    u, vv, valid = renders[:, :, 0], renders[:, :, 1], renders[:, :, 2] != 0
    scores = torch.zeros(b, n_uv, n_uv, v, h // ps, w // ps)
    for r in range(n_uv):
        for c in range(n_uv):
            inside = valid & (edges[c] <= u) & (u < edges[c + 1]) & (1 - edges[r + 1] <= vv) & (vv < 1 - edges[r])
            for bi in range(b):
                for vi in range(v):
                    m = inside[bi, vi]
                    rows, cols = torch.nonzero(m, as_tuple=True)
                    box = (rows.min(), cols.min(), rows.max(), cols.max()) if len(rows) else (-1, -1, -1, -1)
                    for i in range(h // ps):
                        for j in range(w // ps):
                            first = m[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps].float().mean()
                            t, l = min(i * ps, box[0]), min(j * ps, box[1])
                            bottom, right = max(i * ps + ps - 1, box[2]), max(j * ps + ps - 1, box[3])
                            union = (bottom - t + 1) * (right - l + 1)
                            scores[bi, r, c, vi, i, j] = max(first, 0.1 * m.sum() / union)
    return scores.reshape(b, n_uv, n_uv, -1)


def synthetic_uv_renders(seed):
    """Two views of a smooth UV gradient with an invalid band, so most UV patches cover a few image patches."""
    g = torch.Generator().manual_seed(seed)
    h, w = 32, 24
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
    views = []
    for _ in range(2):
        a, s = torch.rand(2, generator=g)
        u = (xs * (0.5 + s) + 0.3 * a * ys) % 1
        v = (1 - ys) * (0.6 + 0.4 * a)
        valid = (torch.rand(h, w, generator=g) > 0.1).float()
        valid[:, :3] = 0
        views.append(torch.stack([u, v, valid]))
    return torch.stack(views)[None]


def test_correspondences_are_each_uv_patchs_best_scoring_image_patches():
    uv_renders = synthetic_uv_renders(seed=0)
    n_neighbors = 10

    idcs, scores = get_img_uv_patch_correspondences(
        uv_renders, UV_RES, uv_patch_size=UV_PATCH, img_patch_size=IMG_PATCH, n_neighbors=n_neighbors,
        downsample_for_computation=DOWNSAMPLE, return_uv_patch_mask=False)

    expected = reference_scores(uv_renders)
    torch.testing.assert_close(scores, torch.gather(expected, -1, idcs))  # each returned score is its pair's score
    best = expected.sort(dim=-1, descending=True).values[..., :n_neighbors]
    torch.testing.assert_close(scores, best)  # and they are the best ones, best first
    assert (expected > 0).sum(-1).min() > 0  # every UV patch matches something, so the check isn't vacuous
