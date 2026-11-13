from typing import Optional, cast
import einops
import pudb
import torch
from torch import _tensor
from torch import nn
from torch import Tensor
import torch.nn.functional as tF


class RMSNorm(nn.Module):

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
  ):
    super().__init__()

    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x: Tensor):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


class Attention(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      qk_norm: bool = True,
      context_dim: Optional[int] = None,
  ):
    super().__init__()

    if context_dim is None:
      context_dim = dim

    self.num_heads = num_heads

    head_dim = dim // num_heads

    self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
    self.wk = nn.Linear(context_dim, num_heads * head_dim, bias=False)
    self.wv = nn.Linear(context_dim, num_heads * head_dim, bias=False)
    self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

    if qk_norm:
      self.q_norm = nn.LayerNorm(num_heads * head_dim)
      self.k_norm = nn.LayerNorm(num_heads * head_dim)
    else:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()

    # Initialize weights
    nn.init.xavier_uniform_(self.wq.weight)
    nn.init.xavier_uniform_(self.wk.weight)
    nn.init.xavier_uniform_(self.wv.weight)
    nn.init.xavier_uniform_(self.wo.weight)

  def forward(self, x: Tensor, context: Optional[Tensor] = None):
    if context is None:
      context = x

    q, k, v = self.wq(x), self.wk(context), self.wv(context)

    q = self.q_norm(q)
    k = self.k_norm(k)

    q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
    k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
    v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

    output = einops.rearrange(
        tF.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        ),
        "b h n d -> b n (h d)",
    )
    return self.wo(output)


class CorrespondenceAwareAttention(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      qk_norm: bool = True,
      context_dim: Optional[int] = None,
  ):
    super().__init__()

    if context_dim is None:
      context_dim = dim
    self.num_heads = num_heads

    head_dim = dim // num_heads

    self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
    self.wk = nn.Linear(context_dim, num_heads * head_dim, bias=False)
    self.wv = nn.Linear(context_dim, num_heads * head_dim, bias=False)
    self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

    if qk_norm:
      self.q_norm = nn.LayerNorm(num_heads * head_dim)
      self.k_norm = nn.LayerNorm(num_heads * head_dim)
    else:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()

    # Initialize weights
    nn.init.xavier_uniform_(self.wq.weight)
    nn.init.xavier_uniform_(self.wk.weight)
    nn.init.xavier_uniform_(self.wv.weight)
    nn.init.xavier_uniform_(self.wo.weight)

  def forward(
      self,
      uv_tokens: torch.Tensor,
      img_tokens: torch.Tensor,
      correspondence_idcs: torch.Tensor,
      correspondence_scores: torch.Tensor,
      context: Optional[Tensor] = None,
  ):
    """calculates correspondence-aware attention

    Args:
      uv_tokens: (B, nv_uv*nh_uv, D)
      img_tokens: (B, v * nv_img * nh_img, D)
      correspondence_idcs: indices of top n_neighbors matching image patches for
        each uv patch (B, nv_uv, nh_uv, n_neighbors)
      correspondence_scores: matching scores of top n_neighbors matching image
        patches for each uv patch (B, nv_uv, nh_uv, n_neighbors)
      context: (B, C, D) not used
    """
    # One sample at a time: every attention sequence (a uv token and its image neighbours) lies within one
    # sample, and its activations (~3.5 GB per full-size sample) would otherwise grow with the batch, which
    # holds T frames per clip.
    outputs = [
        self._forward_sample(
            uv_tokens[i:i + 1], img_tokens[i:i + 1], correspondence_idcs[i:i + 1],
            correspondence_scores[i:i + 1], context)
        for i in range(uv_tokens.shape[0])
    ]
    if len(outputs) == 1:
      return outputs[0]
    return tuple(torch.cat(parts) for parts in zip(*outputs))

  def _forward_sample(
      self,
      uv_tokens: torch.Tensor,
      img_tokens: torch.Tensor,
      correspondence_idcs: torch.Tensor,
      correspondence_scores: torch.Tensor,
      context: Optional[Tensor] = None,
  ):
    """`forward` for a batch of one sample."""
    d = uv_tokens.shape[-1]
    n_imgtokens = img_tokens.shape[1]
    b, nv_uv, nh_uv, n_neighbors = correspondence_idcs.shape
    # b ... batch_size
    # nv_uv ... number of uv patch rows
    # nh_uv ... number of uv patch columns
    # n_neighbors ... number of neighbors for each uv patch

    uv_tokens = einops.rearrange(uv_tokens, "b nv_nh d -> (b nv_nh) 1 d")
    correspondence_idcs = correspondence_idcs.unsqueeze(-1).expand(
        -1, -1, -1, -1, d
    )  # (B, nv_uv, nh_uv, n_neighbors, d)
    img_neighbor_tokens = torch.gather(
        input=img_tokens,
        index=einops.rearrange(
            correspondence_idcs,
            "b nv_uv nh_uv n_neighbors d -> b (nv_uv nh_uv n_neighbors) d",
        ),
        dim=1,
    )  # (B, nv_uv * nh_uv * n_neighbors, d)
    img_neighbor_tokens = einops.rearrange(
        img_neighbor_tokens,
        "b (nv_uv nh_uv n_neighbors) d -> (b nv_uv nh_uv) n_neighbors d",
        nv_uv=nv_uv,
        nh_uv=nh_uv,
        n_neighbors=n_neighbors,
    )

    x = torch.cat(
        [uv_tokens, img_neighbor_tokens], dim=1
    )  # (B * nv_uv * nh_uv, 1 + n_neighbors, d)

    assert context is None, NotImplementedError
    if context is None:
      context = x

    q, k, v = self.wq(x), self.wk(context), self.wv(context)

    q = self.q_norm(q)
    k = self.k_norm(k)

    q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
    k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
    v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

    attention_mask = None
    output = einops.rearrange(
        tF.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attention_mask,
        ),
        "b h n d -> b n (h d)",
    )

    output = self.wo(output)

    uv_tokens, img_neighbor_tokens = (
        output[:, :1],
        output[:, 1:],
    )  # (b * nv_uv * nh_uv, 1|n_neighbors, d)

    uv_tokens = einops.rearrange(uv_tokens, "(b nv_nh) 1 d -> b nv_nh d", b=b)

    # mute unused image neighbor tokens
    unused_neighbor_tokens_mask = (
        einops.rearrange(
            correspondence_scores,
            "b nv_uv nh_uv n_neighbors -> (b nv_uv nh_uv) n_neighbors",
        )
        == 0
    )
    img_neighbor_tokens[unused_neighbor_tokens_mask] = 0
    device = img_neighbor_tokens.device
    dtype = img_neighbor_tokens.dtype
    img_tokens = torch.scatter_reduce(
        input=torch.zeros((b, n_imgtokens, d), device=device, dtype=dtype),
        index=einops.rearrange(
            correspondence_idcs,
            "b nv_uv nh_uv n_neighbors d -> b (nv_uv nh_uv n_neighbors) d",
        ),
        src=einops.rearrange(
            img_neighbor_tokens,
            "(b nv_uv nh_uv) n_neighbors d -> b (nv_uv nh_uv n_neighbors) d",
            b=b,
            nv_uv=nv_uv,
            nh_uv=nh_uv,
        ),
        include_self=False,
        reduce="sum",
        dim=1,
    )  # (B, v * nv_img * nh_img, D)
    img_tokens_normalizer = torch.scatter_reduce(
        input=torch.zeros((b, n_imgtokens, 1), device=device, dtype=dtype),
        index=einops.rearrange(
            correspondence_idcs[..., :1],
            "b nv_uv nh_uv n_neighbors d -> b (nv_uv nh_uv n_neighbors) d",
        ),
        src=einops.rearrange(
            (~unused_neighbor_tokens_mask).to(dtype),
            "(b nv_uv nh_uv) n_neighbors -> b (nv_uv nh_uv n_neighbors) 1",
            b=b,
            nv_uv=nv_uv,
            nh_uv=nh_uv,
        ),
        include_self=False,
        reduce="sum",
        dim=1,
    )  # (B, v * nv_img * nh_img, D)
    img_tokens_normalizer = torch.clip(
        img_tokens_normalizer, min=1
    )  # avoid division by zero
    img_tokens = img_tokens / img_tokens_normalizer
    return uv_tokens, img_tokens


# The temporal sublayers run at most this many token positions (sequences of T frames) of a clip at a time.
# Every position is its own sequence, so the chunking changes nothing in the result. It bounds the memory of
# one attention call and its backward: the 61,440 image-token positions of a 12-view frame (the image-token
# ablation) as one call needed ~19 GiB in backward and failed in the kernel. 4096 = the UV-token positions of
# the pretrained model, which therefore stay one chunk.
TEMPORAL_CHUNK_POSITIONS = 4096


def _position_chunks(n: int):
  return [slice(s, s + TEMPORAL_CHUNK_POSITIONS) for s in range(0, n, TEMPORAL_CHUNK_POSITIONS)]


class TemporalAttention(nn.Module):
  """Temporal attention sublayer of divided space-time attention (TimeSformer).

  Every token position attends only to the same position in the other frames of its clip. For UV tokens a
  position is a UV patch, which is the same surface point in every frame. The sublayer has its own RMSNorm
  and attention weights. Its output projection is zero-initialized, so it is an exact no-op at
  initialization. The frame embedding is added to the normalized input only, never to the residual stream.
  """

  def __init__(
      self,
      dim: int,
      num_heads: int,
      qk_norm: bool = True,
      norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.norm = RMSNorm(dim, norm_eps)
    self.attn = Attention(dim, num_heads, qk_norm)
    nn.init.zeros_(self.attn.wo.weight)

  def forward(self, x: Tensor, clip_len: int, frame_embedding: Tensor):
    """Args:

    x: (B * T, N, D), frame t of clip b at sample b * T + t
    clip_len: T
    frame_embedding: (T, D), one vector per frame slot

    Returns:
      the residual (B * T, N, D)
    """
    h = einops.rearrange(self.norm(x), "(b t) n d -> b t n d", t=clip_len)
    h = h + frame_embedding[:, None].to(h)
    b, _, n, _ = h.shape
    return torch.cat([
        einops.rearrange(
            self.attn(einops.rearrange(h[:, :, chunk], "b t n d -> (b n) t d")),
            "(b n) t d -> (b t) n d", b=b,
        )
        for chunk in _position_chunks(n)
    ], dim=1)


def is_temporal_parameter(name: str) -> bool:
  """True for parameters (state-dict keys) of the temporal sublayers and the frame embedding."""
  return any(part.startswith("temporal_") for part in name.split("."))


def _temporal_sublayers(
    dim: int,
    num_heads: int,
    qk_norm: bool,
    norm_eps: float,
    temporal: bool,
    temporal_image_tokens: bool,
) -> tuple[Optional[TemporalAttention], Optional[TemporalAttention]]:
  """A block's temporal sublayers, run before its spatial attention: (over UV tokens, over image tokens).

  The UV-token sublayer (divided space-time attention) exists with `temporal`; the image-token one, one
  sequence per (camera, image patch position), only for the pixel-aligned ablation (`temporal_image_tokens`).
  None where off.
  """
  uv = TemporalAttention(dim, num_heads, qk_norm, norm_eps) if temporal else None
  img = (TemporalAttention(dim, num_heads, qk_norm, norm_eps)
         if temporal and temporal_image_tokens else None)
  return uv, img


def _add_temporal_residual(
    temporal_attn: Optional[TemporalAttention],
    x: Tensor,
    clip_len: int,
    frame_embedding: Optional[Tensor],
):
  """Applies a block's temporal sublayer with its residual. Clips of one frame skip it (per-frame path)."""
  if temporal_attn is None or clip_len == 1:
    return x
  return x + temporal_attn(x, clip_len, frame_embedding)


class FeedForward(nn.Module):

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,  # ensure `hidden_dim` is a multiple of this value
      ffn_dim_multiplier: Optional[
          float
      ] = None,  # custom mulitplier for `hidden_dim`
  ):
    super().__init__()

    hidden_dim = int(2 * hidden_dim / 3)
    # Custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    # Initialize weights
    nn.init.xavier_uniform_(self.w1.weight)
    nn.init.xavier_uniform_(self.w2.weight)
    nn.init.xavier_uniform_(self.w3.weight)

  def _forward_silu_gating(self, x1: Tensor, x3: Tensor):
    return tF.silu(x1) * x3

  def forward(self, x: Tensor):
    return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class LLaMaTransformerBlock(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
      qk_norm: bool = True,
      multiple_of: int = 256,
      ffn_dim_multiplier: Optional[float] = None,
      norm_eps: float = 1e-5,
  ):
    super().__init__()

    self.norm1 = RMSNorm(dim, norm_eps)
    self.attn = Attention(dim, num_heads, qk_norm)
    self.norm2 = RMSNorm(dim, norm_eps)
    self.mlp = FeedForward(dim, dim * 4, multiple_of, ffn_dim_multiplier)

    if use_cross_attention:
      self.norm3 = RMSNorm(dim, norm_eps)
      self.cross_attn = Attention(dim, num_heads, qk_norm, context_dim)

    self.use_cross_attention = use_cross_attention

  def forward(self, x: Tensor, context: Optional[Tensor] = None):
    x = x + self.attn(self.norm1(x))
    if context is not None:
      x = x + self.cross_attn(self.norm3(x), context)
    else:
      assert not self.use_cross_attention
    x = x + self.mlp(self.norm2(x))
    return x


class CorrespondenceAwareLLaMaTransformerBlock(nn.Module):
  is_correspondences_aware = True
  per_view_block = False

  def __init__(
      self,
      dim: int,
      num_heads: int,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
      qk_norm: bool = True,
      multiple_of: int = 256,
      ffn_dim_multiplier: Optional[float] = None,
      norm_eps: float = 1e-5,
      temporal: bool = False,
      temporal_image_tokens: bool = False,
  ):
    super().__init__()

    self.temporal_uv, self.temporal_img = _temporal_sublayers(
        dim, num_heads, qk_norm, norm_eps, temporal, temporal_image_tokens
    )
    self.norm1 = RMSNorm(dim, norm_eps)
    self.attn = CorrespondenceAwareAttention(
        dim, num_heads, qk_norm
    )
    self.norm2 = RMSNorm(dim, norm_eps)
    self.mlp = FeedForward(dim, dim * 4, multiple_of, ffn_dim_multiplier)

    if use_cross_attention:
      raise NotImplementedError
      self.norm3 = RMSNorm(dim, norm_eps)
      self.cross_attn = Attention(dim, num_heads, qk_norm, context_dim)

    self.use_cross_attention = use_cross_attention

  def forward(
      self,
      uv_tokens: Tensor,
      img_tokens: Tensor,
      correspondence_idcs: Tensor,
      correspondence_scores: Tensor,
      context: Optional[Tensor] = None,
      clip_len: int = 1,
      frame_embedding: Optional[Tensor] = None,
  ):
    """Args (besides the spatial inputs, with T folded into B):

    clip_len: T; consecutive groups of T samples are the frames of one clip
    frame_embedding: (T, D), added to the temporal sublayer's normalized input
    """
    uv_tokens = _add_temporal_residual(
        self.temporal_uv, uv_tokens, clip_len, frame_embedding
    )
    img_tokens = _add_temporal_residual(
        self.temporal_img, img_tokens, clip_len, frame_embedding
    )
    res_uv_tokens, res_img_tokens = self.attn(
        uv_tokens=self.norm1(uv_tokens),
        img_tokens=self.norm1(img_tokens),
        correspondence_idcs=correspondence_idcs,
        correspondence_scores=correspondence_scores,
    )
    uv_tokens = uv_tokens + res_uv_tokens
    img_tokens = img_tokens + res_img_tokens
    if context is not None:
      # x = x + self.cross_attn(self.norm3(x), context)
      raise NotImplementedError
    else:
      assert not self.use_cross_attention
    res_uv_tokens, res_img_tokens = self.mlp(self.norm2(uv_tokens)), self.mlp(
        self.norm2(img_tokens)
    )
    uv_tokens = uv_tokens + res_uv_tokens
    img_tokens = img_tokens + res_img_tokens
    return uv_tokens, img_tokens


class PerViewLLaMaTransformerBlock(nn.Module):
  is_correspondences_aware = False
  per_view_block = True

  def __init__(
      self,
      dim: int,
      num_heads: int,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
      qk_norm: bool = True,
      multiple_of: int = 256,
      ffn_dim_multiplier: Optional[float] = None,
      norm_eps: float = 1e-5,
      temporal: bool = False,
      temporal_image_tokens: bool = False,
  ):
    super().__init__()

    self.temporal_uv, self.temporal_img = _temporal_sublayers(
        dim, num_heads, qk_norm, norm_eps, temporal, temporal_image_tokens
    )
    self.norm1 = RMSNorm(dim, norm_eps)
    self.attn = Attention(dim, num_heads, qk_norm)
    self.norm2 = RMSNorm(dim, norm_eps)
    self.mlp = FeedForward(dim, dim * 4, multiple_of, ffn_dim_multiplier)

    if use_cross_attention:
      self.norm3 = RMSNorm(dim, norm_eps)
      self.cross_attn = Attention(dim, num_heads, qk_norm, context_dim)

    self.use_cross_attention = use_cross_attention

  def forward(
      self,
      uv_tokens: Tensor,
      img_tokens: Tensor,
      nviews: int,
      context: Optional[Tensor] = None,
      clip_len: int = 1,
      frame_embedding: Optional[Tensor] = None,
  ):
    """Args:

    uv_tokens: (B, nv_uv*nh_uv, D)
    img_tokens: (B, v * nv_img * nh_img, D)
    nviews: number of views
    clip_len: T; consecutive groups of T samples in B are the frames of one clip
    frame_embedding: (T, D), added to the temporal sublayer's normalized input

    Returns:
      uv_tokens: (B, nv_uv*nh_uv, D)
      img_tokens: (B, v * nv_img * nh_img, D)
    """
    uv_tokens = _add_temporal_residual(
        self.temporal_uv, uv_tokens, clip_len, frame_embedding
    )
    img_tokens = _add_temporal_residual(
        self.temporal_img, img_tokens, clip_len, frame_embedding
    )
    out_x = list()
    for x, v in zip([uv_tokens, img_tokens], [1, nviews]):
      x = einops.rearrange(x, "b (v n) d -> (b v) n d", v=v)
      x = x + self.attn(self.norm1(x))
      if context is not None:
        x = x + self.cross_attn(self.norm3(x), context)
      else:
        assert not self.use_cross_attention
      x = x + self.mlp(self.norm2(x))
      x = einops.rearrange(x, "(b v) n d -> b (v n) d", v=v)
      out_x.append(x)
    return out_x


class TransformerBlock(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
      **kwargs,  # for compatibility with `LLaMaTransformerBlock`
  ):
    super().__init__()

    self.norm1 = nn.LayerNorm(dim)
    self.attn = Attention(dim, num_heads, qk_norm=False)
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = nn.Sequential(
        nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
    )

    if use_cross_attention:
      self.norm3 = nn.LayerNorm(dim)
      self.cross_attn = Attention(
          dim, num_heads, qk_norm=False, context_dim=context_dim
      )

    self.use_cross_attention = use_cross_attention

  def forward(self, x: Tensor, context: Optional[Tensor] = None):
    x = x + self.attn(self.norm1(x))
    if context is not None:
      x = x + self.cross_attn(self.norm3(x), context)
    else:
      assert not self.use_cross_attention
    x = x + self.mlp(self.norm2(x))
    return x


class Transformer(nn.Module):

  def __init__(
      self,
      num_blocks: int = 12,
      dim: int = 512,
      num_heads: int = 8,
      llama_style: bool = True,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
  ):
    super().__init__()

    Block = LLaMaTransformerBlock if llama_style else TransformerBlock
    self.blocks = nn.ModuleList([
        Block(dim, num_heads, use_cross_attention, context_dim)
        for _ in range(num_blocks)
    ])

    self.grad_checkpointing = False

  def set_grad_checkpointing(self, flag=True):
    self.grad_checkpointing = flag

  def forward(self, x: Tensor, context: Optional[Tensor] = None):
    for block in self.blocks:
      if self.grad_checkpointing:
        x = torch.utils.checkpoint.checkpoint(block, x, context, use_reentrant=False)
      else:
        x = block(x, context)

    return x


def _checkpoint_clip_block(block: nn.Module, block_inputs: dict, context: Optional[Tensor]):
  """Runs one block on clip tokens (T folded into the batch) with gradient checkpointing, frame by frame.

  The temporal sublayers (UV tokens, and image tokens in the ablation) need every frame of a clip, so each
  is checkpointed on the whole batch, one chunk of token positions at a time (TEMPORAL_CHUNK_POSITIONS): the
  backward pass then recomputes one chunk's attention at a time. Everything after them (the spatial
  attention and MLP) treats each sample on its own, so it is checkpointed one sample at a time: the backward
  pass then recomputes one frame's spatial activations at a time instead of the whole clip's. The result is
  what the block computes on the whole batch.

  The spatial checkpoints keep the temporal sublayers' outputs until backward. For UV tokens that is small;
  for image tokens (the ablation) it is one more clip-sized image tensor per block (~240 MiB at 12 views,
  T=4). So with an image-token sublayer the whole block is checkpointed as well, which keeps only the block's
  input; the backward pass then runs the block's forward once more before recomputing as above.
  """
  if block.temporal_img is not None:
    return torch.utils.checkpoint.checkpoint(
        _run_clip_block_checkpointed, block, block_inputs, context, use_reentrant=False
    )
  return _run_clip_block_checkpointed(block, block_inputs, context)


def _run_clip_block_checkpointed(block: nn.Module, block_inputs: dict, context: Optional[Tensor]):
  inputs = dict(block_inputs)
  clip_len, frame_embedding = inputs.pop("clip_len"), inputs.pop("frame_embedding")
  for key, temporal_attn in (("uv_tokens", block.temporal_uv), ("img_tokens", block.temporal_img)):
    if temporal_attn is not None:
      tokens = inputs[key]
      inputs[key] = torch.cat([
          torch.utils.checkpoint.checkpoint(
              _add_temporal_residual, temporal_attn, tokens[:, chunk], clip_len, frame_embedding,
              use_reentrant=False,
          )
          for chunk in _position_chunks(tokens.shape[1])
      ], dim=1)
  # Positional, not keyword, arguments: a checkpoint keeps keyword arguments as a plain python object, out of
  # reach of an enclosing checkpoint, which then could not drop them.
  keys = list(inputs)

  def run_block_on_sample(*values):
    return block(**dict(zip(keys, values)), context=context, clip_len=1, frame_embedding=None)

  n_samples = inputs["uv_tokens"].shape[0]
  outputs = [
      torch.utils.checkpoint.checkpoint(
          run_block_on_sample,
          *(v[i:i + 1] if isinstance(v, Tensor) else v for v in inputs.values()),
          use_reentrant=False,
      )
      for i in range(n_samples)
  ]
  return [torch.cat(tokens) for tokens in zip(*outputs)]


class CorrespondenceAwareTransformer(nn.Module):

  def __init__(
      self,
      num_blocks: int = 12,
      dim: int = 512,
      num_heads: int = 8,
      llama_style: bool = True,
      use_cross_attention: bool = False,
      context_dim: Optional[int] = None,
      temporal: bool = False,
      clip_len: int = 1,
      temporal_image_tokens: bool = False,
  ):
    """With `temporal`, every block gets a temporal sublayer over UV tokens and the transformer gets one
    learned embedding per frame slot, `clip_len` slots, zero-initialized. With `temporal_image_tokens` as
    well (an ablation), every block also gets one over image tokens; both share the frame embedding."""
    super().__init__()

    if not llama_style:
      raise NotImplementedError
    CorrespondenceAwareBlock = CorrespondenceAwareLLaMaTransformerBlock
    PerViewBlock = PerViewLLaMaTransformerBlock
    module_list = list()
    for _ in range(num_blocks):
      module_list.append(
          CorrespondenceAwareBlock(
              dim,
              num_heads,
              use_cross_attention,
              context_dim,
              temporal=temporal,
              temporal_image_tokens=temporal_image_tokens,
          )
      )
      module_list.append(
          PerViewBlock(
              dim, num_heads, use_cross_attention, context_dim, temporal=temporal,
              temporal_image_tokens=temporal_image_tokens,
          )
      )
    self.blocks = nn.ModuleList(module_list)

    self.temporal_frame_embedding = (
        nn.Parameter(torch.zeros(clip_len, dim)) if temporal else None
    )

    self.grad_checkpointing = False

  def set_grad_checkpointing(self, flag=True):
    self.grad_checkpointing = flag

  def forward(
      self,
      uv_tokens: Tensor,
      img_tokens: Tensor,
      correspondence_idcs: Tensor,
      correspondence_scores: Tensor,
      nviews: int,
      context: Optional[Tensor] = None,
      clip_len: int = 1,
  ):
    """Spatial passes run with the clip's time axis folded into the batch: samples b * T ... b * T + T - 1
    are the frames of clip b, T = `clip_len`."""
    frame_embedding = None
    if self.temporal_frame_embedding is not None and clip_len > 1:
      if clip_len > len(self.temporal_frame_embedding):
        raise ValueError(
            f"clip of {clip_len} frames, but the model has"
            f" {len(self.temporal_frame_embedding)} frame slots (clip_len)"
        )
      frame_embedding = self.temporal_frame_embedding[:clip_len]
    for block in self.blocks:
      if block.is_correspondences_aware:
        block_inputs = dict(
            uv_tokens=uv_tokens,
            img_tokens=img_tokens,
            correspondence_idcs=correspondence_idcs,
            correspondence_scores=correspondence_scores,
        )
      elif block.per_view_block:
        block_inputs = dict(
            uv_tokens=uv_tokens,
            img_tokens=img_tokens,
            nviews=nviews,
        )
      else:
        raise ValueError("Unsupported block type.")
      block_inputs.update(clip_len=clip_len, frame_embedding=frame_embedding)

      if self.grad_checkpointing and clip_len > 1:
        uv_tokens, img_tokens = _checkpoint_clip_block(block, block_inputs, context)
      elif self.grad_checkpointing:
        uv_tokens, img_tokens = torch.utils.checkpoint.checkpoint(
            block, **block_inputs, context=context, use_reentrant=False
        )
      else:
        uv_tokens, img_tokens = block(**block_inputs, context=context)

    return uv_tokens, img_tokens
