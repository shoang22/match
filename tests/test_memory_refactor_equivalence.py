"""The memory refactors of #35 and #37 compute what the code computed before them. CPU, fp32.

- The correspondence search (uv_util.get_img_uv_patch_correspondences) no longer materializes (B, H, W) coordinate
  grids or (b, v, nv_uv, nh_uv, nv_img, nh_img, 4) boxes. Checked bit for bit against the code before #35, copied
  below from 977fc06:match/utils/uv_util.py.
- CorrespondenceAwareAttention runs one sample at a time: checked against its computation on the whole batch.
- Sapiens runs one frame at a time when a frame's views fill whole tiles: checked with a stub encoder whose output
  depends on which images share a tile.
- The whole model, which now searches correspondences per sample, gives each sample the same Gaussians at batch
  size 2 as alone, including when its top-k correspondences end in ties.

#32 measured per-frame MATCH 0.49 mm apart at batch size 2 and 1 on the GPU, and #35 suspected the top-100 argsort
over tied zero scores. The tie test below shows the search before #35 already broke ties the same way in a batch
as alone; the same holds on CUDA at full size (uv_res 1024, 640x512 images, 4 views, top 100, 96% of the top-k
tied at zero: no index differs). The batch dependence is in Sapiens instead: the compiled bf16 encoder gives the
same 4 images features up to 0.34 apart (mean magnitude 0.22) when they run as half of a batch of 8, and is
deterministic for a fixed batch. #35 runs Sapiens one frame at a time, which is what removed the gap.

Run: CUDA_VISIBLE_DEVICES= conda run -n match python -m pytest -q tests/test_memory_refactor_equivalence.py
"""
from pathlib import Path

import einops
import pytest
import torch

from match.models.networks.attention import CorrespondenceAwareAttention
from match.utils import image_util
from match.utils.uv_util import get_img_uv_patch_correspondences

needs_assets = pytest.mark.skipif(not Path('assets').exists(), reason='assets/ is missing; see CLAUDE.md')


# --- The correspondence search before #35 (977fc06), verbatim but for formatting ----------------------------------

def bounding_box_before_35(mask):
    B, H, W = mask.shape
    rows = torch.arange(H, device=mask.device).view(1, H, 1)
    cols = torch.arange(W, device=mask.device).view(1, 1, W)
    masked_rows = torch.where(mask, rows, torch.full_like(rows, H))
    masked_cols = torch.where(mask, cols, torch.full_like(cols, W))
    t = masked_rows.amin(dim=(-2, -1))
    l = masked_cols.amin(dim=(-2, -1))  # noqa: E741
    masked_rows = torch.where(mask, rows, torch.full_like(rows, -1))
    masked_cols = torch.where(mask, cols, torch.full_like(cols, -1))
    b = masked_rows.amax(dim=(-2, -1))
    r = masked_cols.amax(dim=(-2, -1))
    boxes = torch.stack([t, l, b, r], dim=-1)
    has_true = mask.any(dim=(-2, -1))
    boxes[~has_true] = -1
    return boxes


@torch.no_grad()
def correspondences_before_35(uv_renders, uv_res, uv_patch_size, img_patch_size, n_neighbors,
                              downsample_for_computation=1, return_uv_patch_mask=True):
    device = uv_renders.device
    dtype = uv_renders.dtype
    uv_renders = image_util.pad_image_to_fit_patchification(uv_renders, patch_size=img_patch_size, constant_value=-1.0)
    b, v, _, h, w = uv_renders.shape
    h_downsampled = int(h / downsample_for_computation)
    w_downsampled = int(w / downsample_for_computation)
    downsampled_patch_size = int(img_patch_size / downsample_for_computation)
    if downsample_for_computation > 1:
        uv_renders = einops.rearrange(
            torch.nn.functional.interpolate(einops.rearrange(uv_renders, "b v c h w -> (b v) c h w"),
                                            scale_factor=1.0 / downsample_for_computation, mode="nearest"),
            "(b v) c h w -> b v c h w", b=b, v=v)
        uv_renders = uv_renders.to(device=device, dtype=dtype)
    n_imgpatches_v = h_downsampled // downsampled_patch_size
    n_imgpatches_h = w_downsampled // downsampled_patch_size

    uv_pixel_size = 1 / (uv_res - 1)
    sqrt_n_patches = uv_res // uv_patch_size
    patch_edges_u = (-0.5 * uv_pixel_size
                     + uv_pixel_size * uv_patch_size * torch.arange(sqrt_n_patches + 1, device=device, dtype=dtype))
    patch_edges_v = 1 - patch_edges_u
    patch_edges_upper = einops.rearrange(patch_edges_v[:-1], "n -> 1 n 1 1")
    patch_edges_lower = einops.rearrange(patch_edges_v[1:], "n -> 1 n 1 1")
    patch_edges_left = einops.rearrange(patch_edges_u[:-1], "n -> 1 n 1 1")
    patch_edges_right = einops.rearrange(patch_edges_u[1:], "n -> 1 n 1 1")
    uv_renders = einops.rearrange(uv_renders, "b v c h w -> (b v) 1 c h w")
    uv_patch_mask_u = torch.logical_and(patch_edges_left <= uv_renders[:, :, 0],
                                        uv_renders[:, :, 0] < patch_edges_right)
    uv_patch_mask_v = torch.logical_and(patch_edges_lower <= uv_renders[:, :, 1],
                                        uv_renders[:, :, 1] < patch_edges_upper)
    uv_patch_masks = torch.logical_and(einops.rearrange(uv_patch_mask_v, "bv n h w -> bv n 1 h w"),
                                       einops.rearrange(uv_patch_mask_u, "bv n h w -> bv 1 n h w"))
    uv_patch_masks = torch.logical_and(uv_patch_masks,
                                       einops.rearrange(uv_renders[:, :, -1].bool(), "bv n h w -> bv n 1 h w"))
    uv_patch_masks = einops.rearrange(uv_patch_masks, "(b v) nv nh h w -> b v nv nh h w", b=b, v=v)

    uv_img_patch_scores = einops.rearrange(
        uv_patch_masks,
        "b v nv_uv nh_uv (nv_img hpatch) (nh_img wpatch) -> b v nv_uv nh_uv nv_img nh_img (hpatch wpatch)",
        hpatch=downsampled_patch_size, wpatch=downsampled_patch_size).to(dtype).mean(dim=-1)
    uv_img_patch_scores = einops.rearrange(uv_img_patch_scores,
                                           "b v nv_uv nh_uv nv_img nh_img -> b nv_uv nh_uv (v nv_img nh_img)")

    uv_patch_areas = einops.rearrange(uv_patch_masks.to(dtype), 'b v nv nh h w -> b v nv nh (h w)').sum(dim=-1)
    uv_patch_bboxs = einops.rearrange(
        bounding_box_before_35(einops.rearrange(uv_patch_masks, 'b v nv nh h w -> (b v nv nh) h w')),
        '(b v nv nh) c -> b v nv nh c', b=b, v=v, nv=sqrt_n_patches, nh=sqrt_n_patches)
    uv_patch_bboxs = einops.repeat(uv_patch_bboxs, 'b v nv_uv nh_uv c -> b v nv_uv nh_uv nv_img nh_img c',
                                   nv_img=n_imgpatches_v, nh_img=n_imgpatches_h)
    img_patch_bbox_l = torch.arange(0, w_downsampled, downsampled_patch_size, device=device)
    img_patch_bbox_t = torch.arange(0, h_downsampled, downsampled_patch_size, device=device)
    img_patch_bbox_l = einops.repeat(img_patch_bbox_l, 'nh_img -> b v nv_uv nh_uv nv_img nh_img', b=b, v=v,
                                     nv_uv=sqrt_n_patches, nh_uv=sqrt_n_patches, nv_img=n_imgpatches_v)
    img_patch_bbox_t = einops.repeat(img_patch_bbox_t, 'nv_img -> b v nv_uv nh_uv nv_img nh_img', b=b, v=v,
                                     nv_uv=sqrt_n_patches, nh_uv=sqrt_n_patches, nh_img=n_imgpatches_h)
    img_patch_bboxs = torch.stack((img_patch_bbox_t, img_patch_bbox_l, img_patch_bbox_t + downsampled_patch_size - 1,
                                   img_patch_bbox_l + downsampled_patch_size - 1), dim=-1)
    union_bbxs = torch.stack((torch.minimum(img_patch_bboxs[..., 0], uv_patch_bboxs[..., 0]),
                              torch.minimum(img_patch_bboxs[..., 1], uv_patch_bboxs[..., 1]),
                              torch.maximum(img_patch_bboxs[..., 2], uv_patch_bboxs[..., 2]),
                              torch.maximum(img_patch_bboxs[..., 3], uv_patch_bboxs[..., 3])), dim=-1)
    union_areas = (union_bbxs[..., 2] - union_bbxs[..., 0] + 1) * (union_bbxs[..., 3] - union_bbxs[..., 1] + 1)
    secondary_scores = einops.rearrange(uv_patch_areas, 'b v nv_uv nh_uv -> b v nv_uv nh_uv 1 1') / union_areas
    secondary_scores = einops.rearrange(secondary_scores,
                                        'b v nv_uv nh_uv nv_img nh_img -> b nv_uv nh_uv (v nv_img nh_img)')
    uv_img_patch_scores = torch.maximum(uv_img_patch_scores, secondary_scores * .1)

    uv_img_patch_match_idcs = torch.argsort(uv_img_patch_scores, dim=-1, descending=True)[..., :n_neighbors]
    uv_img_patch_match_scores = torch.gather(input=uv_img_patch_scores, index=uv_img_patch_match_idcs, dim=-1)
    if return_uv_patch_mask:
        uv_patch_masks = torch.repeat_interleave(uv_patch_masks, downsample_for_computation, dim=-1)
        uv_patch_masks = torch.repeat_interleave(uv_patch_masks, downsample_for_computation, dim=-2)
        return uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores
    return uv_img_patch_match_idcs, uv_img_patch_match_scores


# --- Correspondence search ----------------------------------------------------------------------------------------

UV_RES, UV_PATCH, IMG_PATCH = 32, 8, 8


def uv_renders(seed, b=2, v=3, h=40, w=28):
    """Random UV renders (B, V, 3, H, W): u, v in [0, 1] plus a sprinkling of exact patch edges and out-of-range
    values, a random validity channel, one view with nothing valid and one view fully valid. H and W are not
    multiples of the image patch, so the padding counts too."""
    g = torch.Generator().manual_seed(seed)
    renders = torch.rand(b, v, 3, h, w, generator=g)
    px = 1 / (UV_RES - 1)
    edges = torch.tensor([-0.5 * px + px * UV_PATCH * k for k in range(UV_RES // UV_PATCH + 1)])
    on_edge = torch.rand(b, v, 2, h, w, generator=g) < 0.1
    renders[:, :, :2] = torch.where(on_edge, edges[torch.randint(len(edges), (b, v, 2, h, w), generator=g)],
                                    renders[:, :, :2])
    renders[:, :, :2] += (torch.rand(b, v, 2, h, w, generator=g) < 0.02) * 1.5  # outside the UV square
    renders[:, :, 2] = (torch.rand(b, v, h, w, generator=g) < 0.6).float()
    renders[0, 0, 2] = 0.  # an empty view: every UV patch mask empty in it
    renders[-1, -1, 2] = 1.
    return renders


@pytest.mark.parametrize('downsample', [1, 2, 4])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_correspondence_search_returns_what_it_returned_before_35(seed, downsample):
    renders = uv_renders(seed)
    kwargs = dict(uv_res=UV_RES, uv_patch_size=UV_PATCH, img_patch_size=IMG_PATCH, n_neighbors=7,
                  downsample_for_computation=downsample)

    masks, idcs, scores = get_img_uv_patch_correspondences(renders, **kwargs)
    old_masks, old_idcs, old_scores = correspondences_before_35(renders, **kwargs)

    assert torch.equal(masks, old_masks)
    assert torch.equal(scores, old_scores)
    assert torch.equal(idcs, old_idcs)


def test_the_correspondence_search_of_an_all_empty_render_returns_what_it_returned_before_35():
    renders = torch.zeros(1, 2, 3, 16, 16)  # nothing valid anywhere
    kwargs = dict(uv_res=UV_RES, uv_patch_size=UV_PATCH, img_patch_size=IMG_PATCH, n_neighbors=3,
                  return_uv_patch_mask=False)

    idcs, scores = get_img_uv_patch_correspondences(renders, **kwargs)
    old_idcs, old_scores = correspondences_before_35(renders, **kwargs)

    assert torch.equal(scores, old_scores) and torch.equal(idcs, old_idcs)
    assert (scores == 0).all()


# --- Correspondence-aware attention -------------------------------------------------------------------------------

def test_correspondence_aware_attention_one_sample_at_a_time_equals_the_whole_batch_at_once():
    torch.manual_seed(0)
    b, n_uv_side, n_img, d, k = 3, 4, 30, 32, 5
    attention = CorrespondenceAwareAttention(d, num_heads=2).eval()
    uv_tokens, img_tokens = torch.randn(b, n_uv_side ** 2, d), torch.randn(b, n_img, d)
    idcs = torch.randint(n_img, (b, n_uv_side, n_uv_side, k))  # repeated image tokens included
    scores = torch.rand(b, n_uv_side, n_uv_side, k) * (torch.rand(b, n_uv_side, n_uv_side, k) > 0.3)  # some zeros

    with torch.no_grad():
        per_sample = attention(uv_tokens, img_tokens, idcs, scores)
        whole_batch = attention._forward_sample(uv_tokens, img_tokens, idcs, scores)  # the computation before #35

    for got, expected in zip(per_sample, whole_batch):
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


# --- Sapiens chunking ---------------------------------------------------------------------------------------------

TILES_SQRT = 2  # Sapiens runs 2x2 images per tile


class TilingSapiensStub:
    """Stands in for SapiensEncoder: one feature per image patch, depending on the image and on the images it shares
    a tile with (consecutive groups of TILES_SQRT**2 in each call, as the real encoder tiles them). Records calls."""
    C = 6

    def __init__(self, *args, **kwargs):
        self.calls = []

    def to(self, *args, **kwargs):
        return self

    def __call__(self, imgs):
        n, _, h, w = imgs.shape
        tile = TILES_SQRT ** 2
        own = imgs.mean(dim=(1, 2, 3))
        groups = torch.nn.functional.pad(own, (0, -n % tile)).reshape(-1, tile).mean(dim=1).repeat_interleave(tile)[:n]
        features = (own + 10 * groups)[:, None, None, None].expand(n, self.C, h // 8, w // 8).clone()
        self.calls.append((imgs.clone(), features))
        return features


@needs_assets
@pytest.mark.parametrize('n_views, chunks', [(4, 'one call per frame'), (3, 'one call for the batch')])
def test_sapiens_one_frame_at_a_time_gives_the_features_of_one_batched_call(monkeypatch, n_views, chunks):
    from match.models import match_model
    from tiny_match import HEIGHT, WIDTH, build
    monkeypatch.setattr(match_model, 'SapiensEncoder', TilingSapiensStub)
    model = build(temporal=False, use_sapiens_features=True, sapiens_prediction_tiles_sqrt=TILES_SQRT,
                  num_input_views=None)
    n_frames = 3
    batch = {'image': torch.rand(n_frames, n_views, 3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(0)),
             'C2W': torch.eye(4).expand(n_frames, n_views, 4, 4),
             'fxfycxcy': torch.tensor([30., 30., WIDTH / 2, HEIGHT / 2]).expand(n_frames, n_views, 4)}

    with torch.no_grad():
        model._get_img_tokens(batch)

    calls = model.sapiens_encoder.calls
    expected_calls = [n_views] * n_frames if chunks == 'one call per frame' else [n_frames * n_views]
    assert [len(imgs) for imgs, _ in calls] == expected_calls
    chunked = torch.cat([features for _, features in calls])
    model.sapiens_encoder.calls = []
    batched = model.sapiens_encoder(torch.cat([imgs for imgs, _ in calls]))  # the call before #35
    assert torch.equal(chunked, batched)


# --- The whole model ----------------------------------------------------------------------------------------------

def frame_with_few_visible_patches(num_verts, seed):
    """A frame whose UV renders show only a small corner of the UV map, so most UV patches have fewer matching image
    patches than n_neighbors and their top-k ends in a tie of zero scores."""
    from tiny_match import synthetic_frame
    frame = synthetic_frame(num_verts, seed)
    frame['uv'][:, :2] *= 0.2
    frame['uv'][:, 2, :, 12:] = 0.
    return frame


@needs_assets
@pytest.mark.parametrize('tied', [False, True], ids=['random', 'tied-top-k'])
def test_each_sample_of_a_batch_of_two_gets_the_gaussians_it_gets_alone(tied):
    from tiny_match import as_batch, build, synthetic_frame
    model = build(temporal=False)
    make = frame_with_few_visible_patches if tied else synthetic_frame
    frames = [make(model.num_verts, seed) for seed in (0, 1)]
    if tied:
        uv = torch.stack([f['uv'] for f in frames]).half()
        search = dict(uv_res=model.opt.uv_res, uv_patch_size=model.opt.uv_patch_size,
                      img_patch_size=model.opt.img_patch_size, n_neighbors=model.opt.n_neighbors,
                      downsample_for_computation=model.opt.img_patch_size // 2, return_uv_patch_mask=False)
        idcs, scores = correspondences_before_35(uv, **search)
        assert (scores == 0).any(), 'the top-k should end in tied zero scores'
        # the search before #35 already broke the ties the same way in the batch as for each sample alone
        for i in range(2):
            alone_idcs, _ = correspondences_before_35(uv[i:i + 1], **search)
            assert torch.equal(idcs[i:i + 1], alone_idcs)

    with torch.no_grad():
        pair = model.forward_gaussians({k: torch.stack([f[k] for f in frames]) for k in frames[0]})
        alone = [model.forward_gaussians(as_batch(f)) for f in frames]

    for i, single in enumerate(alone):
        for k, v in single.items():
            torch.testing.assert_close(pair[k][i:i + 1], v, atol=1e-5, rtol=0, msg=f'{k} sample {i}')
