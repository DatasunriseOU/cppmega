"""SparseMLA autograd.Function — ported from NVIDIA/Megatron-LM PR #3674.

Wraps TileLang fused sparse MLA forward + backward kernels into a single
autograd.Function that can be dropped into Megatron's DSA attention path
as a replacement for ``unfused_dsa_fn``.

The adapter function ``sparse_mla_as_unfused_dsa`` at the bottom of this
file bridges between the Megatron ``unfused_dsa_fn`` signature::

    unfused_dsa_fn(query, key, value, topk_indices, softmax_scale)
    # query:       [sq, b, np, hn]
    # key:         [sk, b, np, hn]
    # value:       [sk, b, np, hnv]
    # topk_indices:[b, sq, topk]
    # returns:     [sq, b, np * hnv]

and the TileLang SparseMLA signature::

    SparseMLA.apply(q, kv, indices, scaling)
    # q:       [batch, seq, heads, dim+tail_dim]   (packed QK dim)
    # kv:      [batch, seq_kv, kv_group, dim+tail_dim]  (packed K+V)
    # indices: [batch, seq, kv_group, topk]  (int32, -1 = invalid)
    # returns: (out [batch,seq,heads,dim], lse [batch,seq,heads])

Key interface translation:
- Q: permute [sq,b,np,hn] -> [b,sq,np,hn] (hn is already dim+tail_dim for MLA)
- K,V: for MLA, K and V share the same latent KV tensor. The TileLang kernel
  expects a single packed KV tensor [batch, seq_kv, kv_group, dim+tail_dim]
  where the first ``dim`` channels are used for V output and the full
  ``dim+tail_dim`` channels for the Q@K dot product.
  In Megatron DSA, K=[sk,b,np,hn] and V=[sk,b,np,hnv]. For MLA,
  hn = dim+tail_dim = 576 and hnv = dim = 512 (the V channels are the first
  512 of the 576). So we pack by just using K as the KV tensor (it contains
  all 576 channels; V is a prefix view).
- Indices: Megatron uses [b, sq, topk] (int64); TileLang uses
  [batch, seq, kv_group, topk] (int32, -1=invalid). We unsqueeze kv_group=1
  and convert dtype + sentinel.
- Output: TileLang returns [batch, seq, heads, dim]; Megatron expects
  [sq, b, np*hnv]. Permute + reshape.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger(__name__)


class SparseMLA(torch.autograd.Function):
    """Autograd wrapper around TileLang sparse-MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, scaling, d_v=None):
        """Forward pass.

        Args:
            q: Query tensor [batch, seq, heads, dim+tail_dim] or [seq, heads, dim+tail_dim]
            kv: KV tensor [batch, seq_kv, kv_group, dim+tail_dim] or [seq_kv, kv_group, dim+tail_dim]
            indices: Sparse indices [batch, seq, kv_group, topk] or [seq, kv_group, topk]
                     int32, -1 = invalid/masked
            scaling: float softmax scale
            d_v: int, value head dimension. If None, inferred from q's last dim.

        Returns:
            (out, lse) where out: [batch, seq, heads, dim], lse: [batch, seq, heads]
        """
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd import (
            sparse_mla_fwd_interface,
        )

        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling

        # Infer d_v from tensor shape if not provided.
        if d_v is None:
            d_v = q.shape[-1]

        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling, d_v=d_v)

        ctx.d_v = d_v
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)
        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """Backward pass.

        Returns:
            Gradients for (q, kv, indices=None, scaling=None).
        """
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd import sparse_mla_bwd

        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        tl_dq, tl_dkv = sparse_mla_bwd(
            q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling
        )

        return tl_dq, tl_dkv, None, None, None


class SparseMLA_FP8(torch.autograd.Function):
    """Autograd wrapper: FP8 forward + BF16 backward for sparse-MLA.

    The FP8 TileLang kernel handles forward only (Q@K in float8_e4m3fn with
    per-token scaling, S@V in BF16 after dequant). Backward reuses the
    standard BF16 TileLang backward kernel — this means we save the BF16
    copies of q/kv for backward even though forward ran in FP8. The memory
    cost is identical to the BF16 path but forward FLOPS benefit from FP8
    tensor-core throughput (2x on H100/H200).
    """

    @staticmethod
    def forward(ctx, q, kv, indices, scaling, d_v=None):
        """Forward pass using FP8 sparse MLA kernel.

        Args:
            q: Query tensor [batch, seq, heads, dim+tail_dim] BF16
            kv: KV tensor [batch, seq_kv, kv_group, dim+tail_dim] BF16
            indices: Sparse indices [batch, seq, kv_group, topk] int32
            scaling: float softmax scale
            d_v: int, value head dimension

        Returns:
            (out, lse) where out: [batch, seq, heads, dim], lse: [batch, seq, heads]
        """
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
            sparse_mla_fwd_fp8_interface,
        )

        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling

        if d_v is None:
            d_v = q.shape[-1]

        tl_out, tl_lse = sparse_mla_fwd_fp8_interface(
            q, kv, indices, sm_scale=scaling, d_v=d_v,
        )

        ctx.d_v = d_v
        # Save BF16 tensors for backward (BF16 backward kernel needs them).
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)
        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """Backward pass using BF16 sparse MLA kernel.

        Returns:
            Gradients for (q, kv, indices=None, scaling=None, d_v=None).
        """
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd import sparse_mla_bwd

        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        tl_dq, tl_dkv = sparse_mla_bwd(
            q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling
        )

        return tl_dq, tl_dkv, None, None, None


def sparse_mla_fp8_as_unfused_dsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for Megatron ``unfused_dsa_fn`` using FP8 SparseMLA.

    Same interface as ``sparse_mla_as_unfused_dsa`` but routes forward through
    the FP8 TileLang kernel for ~2x tensor-core throughput on H100/H200.

    Args:
        query: [sq, b, np, hn] — where hn = dim + tail_dim (576 for DeepSeek MLA)
        key:   [sk, b, np, hn] — full KV latent (576 channels)
        value: [sk, b, np, hnv] — value prefix (512 channels, subset of key)
        topk_indices: [b, sq, topk] int64 — KV positions to attend to
        softmax_scale: float
        **kwargs: mask, varlen_starts, varlen_ends, key_positions (ignored by
            TileLang kernel; causal masking is built into the kernel via the
            -1 sentinel convention on indices).

    Returns:
        output: [sq, b, np * hnv] — same shape as unfused_dsa_fn output
    """
    sq, b, np_, hn = query.shape
    sk = key.shape[0]
    hnv = value.shape[3]

    q = query.permute(1, 0, 2, 3).contiguous()
    kv = key.permute(1, 0, 2, 3)[:, :, 0:1, :].contiguous()

    indices_i32 = topk_indices.to(torch.int32)
    indices_i32 = indices_i32.unsqueeze(2)
    indices_i32 = torch.where(
        (indices_i32 >= 0) & (indices_i32 < sk),
        indices_i32,
        torch.tensor(-1, dtype=torch.int32, device=indices_i32.device),
    )
    indices_i32 = indices_i32.contiguous()

    out, _lse = SparseMLA_FP8.apply(q, kv, indices_i32, softmax_scale, hnv)

    output = out.permute(1, 0, 2, 3).contiguous().reshape(sq, b, np_ * hnv)
    return output


def sparse_mla_as_unfused_dsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for Megatron ``unfused_dsa_fn`` using TileLang SparseMLA.

    Translates between Megatron's unfused_dsa_fn signature and SparseMLA's
    expected tensor layouts. See module docstring for the full translation.

    Args:
        query: [sq, b, np, hn] — where hn = dim + tail_dim (576 for DeepSeek MLA)
        key:   [sk, b, np, hn] — full KV latent (576 channels)
        value: [sk, b, np, hnv] — value prefix (512 channels, subset of key)
        topk_indices: [b, sq, topk] int64 — KV positions to attend to
        softmax_scale: float
        **kwargs: mask, varlen_starts, varlen_ends, key_positions (ignored by
            TileLang kernel; causal masking is built into the kernel via the
            -1 sentinel convention on indices).

    Returns:
        output: [sq, b, np * hnv] — same shape as unfused_dsa_fn output
    """
    sq, b, np_, hn = query.shape
    sk = key.shape[0]
    hnv = value.shape[3]

    # ---- Q: [sq, b, np, hn] -> [b, sq, np, hn] ----
    q = query.permute(1, 0, 2, 3).contiguous()

    # ---- KV: use key as the packed KV tensor ----
    # For MLA, key has all dim+tail_dim channels; value is key[...,:dim].
    # TileLang expects KV [batch, seq_kv, kv_group, dim+tail_dim].
    # In the Megatron DSA path, kv_group=1 and heads dimension in key
    # corresponds to the full head count (GQA already expanded).
    # We need [b, sk, 1, hn] — treat all heads as one KV group.
    # But wait: in MLA, K/V are shared across heads (kv_group=1).
    # The key tensor [sk, b, np, hn] has np copies of the same latent.
    # We take [:, :, 0:1, :] as the single KV group.
    kv = key.permute(1, 0, 2, 3)[:, :, 0:1, :].contiguous()

    # ---- Indices: [b, sq, topk] -> [b, sq, 1, topk] int32 with -1 sentinel ----
    # Megatron indices are int64 positions into sk; TileLang uses int32 with -1=invalid.
    # The Megatron path may have indices that point beyond valid range for padding;
    # these are typically just not used, but we convert any >= sk to -1.
    indices_i32 = topk_indices.to(torch.int32)
    indices_i32 = indices_i32.unsqueeze(2)  # [b, sq, 1, topk]
    # Mask invalid positions
    indices_i32 = torch.where(
        (indices_i32 >= 0) & (indices_i32 < sk),
        indices_i32,
        torch.tensor(-1, dtype=torch.int32, device=indices_i32.device),
    )
    indices_i32 = indices_i32.contiguous()

    # ---- Forward ----
    out, _lse = SparseMLA.apply(q, kv, indices_i32, softmax_scale, hnv)
    # out: [b, sq, np, hnv] (dim = hnv = v_head_dim)

    # ---- Output: [b, sq, np, hnv] -> [sq, b, np * hnv] ----
    output = out.permute(1, 0, 2, 3).contiguous().reshape(sq, b, np_ * hnv)
    return output


def fused_sparse_mla_absorbed_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> Optional[torch.Tensor]:
    """Run fused SparseMLA FP8 kernel for absorbed-MLA path.

    Same interface as ``_fused_sparse_mla_absorbed`` in dsa.py but routes
    the forward pass through the FP8 TileLang kernel. Backward uses BF16.

    Inputs are expected in SBHD with MQA key heads (kv_group=1):
      query: [sq, b, np, d_total]
      key:   [skv, b, 1, d_total]
      topk:  [b, sq, topk]

    Returns:
      output: [sq, b, np, v_channels], or None if unsupported / unavailable.
    """
    if query.ndim != 4 or key.ndim != 4 or topk_indices.ndim != 3:
        return None
    if key.size(2) != 1:
        return None
    if query.size(1) != key.size(1) or topk_indices.size(0) != query.size(1):
        return None
    if topk_indices.size(1) != query.size(0):
        return None
    if query.size(-1) != key.size(-1):
        return None
    if topk_indices.size(-1) % 64 != 0:
        return None

    batch_output_list = []
    for bi in range(query.size(1)):
        q_t = query[:, bi].contiguous()
        kv_t = key[:, bi].contiguous()
        idx_t = topk_indices[bi].unsqueeze(1).to(torch.int32).contiguous()
        out, _ = SparseMLA_FP8.apply(q_t, kv_t, idx_t, softmax_scale, v_channels)
        if out.ndim != 3 or out.size(-1) != v_channels:
            return None
        batch_output_list.append(out)

    if not batch_output_list:
        return None
    return torch.stack(batch_output_list, dim=1).contiguous()
