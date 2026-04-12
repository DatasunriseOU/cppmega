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
    def forward(ctx, q, kv, indices, scaling):
        """Forward pass.

        Args:
            q: Query tensor [batch, seq, heads, dim+tail_dim] or [seq, heads, dim+tail_dim]
            kv: KV tensor [batch, seq_kv, kv_group, dim+tail_dim] or [seq_kv, kv_group, dim+tail_dim]
            indices: Sparse indices [batch, seq, kv_group, topk] or [seq, kv_group, topk]
                     int32, -1 = invalid/masked
            scaling: float softmax scale

        Returns:
            (out, lse) where out: [batch, seq, heads, dim], lse: [batch, seq, heads]
        """
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd import (
            sparse_mla_fwd_interface,
        )

        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling

        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

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

        return tl_dq, tl_dkv, None, None


def sparse_mla_as_unfused_dsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
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
    out, _lse = SparseMLA.apply(q, kv, indices_i32, softmax_scale)
    # out: [b, sq, np, hnv] (dim = hnv = 512)

    # ---- Output: [b, sq, np, hnv] -> [sq, b, np * hnv] ----
    output = out.permute(1, 0, 2, 3).contiguous().reshape(sq, b, np_ * hnv)
    return output
