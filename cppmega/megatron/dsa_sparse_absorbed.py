"""Sparse gather-scatter for absorbed-MLA DSA attention.

Replaces ``_unfused_absorbed_dsa_fn`` in PR #3674's DSA code which does a
full ``torch.matmul(q.float(), k.float())`` = 7 GiB at [b, np, sq, sk].

The absorbed path has MQA layout: key has 1 head (all query heads share it).
Sparse attention only needs to compute scores at topk positions per query:

  Dense: [b, np, sq, sk]  = 4×28×4096×4096 × 4 bytes = 7.0 GiB
  Sparse: [b, np, sq, topk] = 4×28×4096×256 × 4 bytes = 0.44 GiB

That's a 16× memory reduction.

Applied via monkey-patch in ``dsa_fp8_patch.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_absorbed_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
    mask=None,
    varlen_starts=None,
    varlen_ends=None,
    key_positions=None,
) -> torch.Tensor:
    """Sparse absorbed-MLA DSA attention.

    Args:
        query: [sq, b, np, hn] — multi-head query.
        key: [sk, b, 1, hn] — single-head absorbed key (MQA).
            Last ``v_channels`` dims contain the latent value.
        topk_indices: [b, sq, topk] int64 — which K positions to attend.
        softmax_scale: float.
        v_channels: int — number of value channels in key's last dim.

    Returns:
        output: [b, np, sq, v_channels].
    """
    sq, b, np_, hn = query.size()
    sk = key.size(0)
    topk = topk_indices.size(-1)

    # [sq, b, np, hn] → [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).contiguous()
    # [sk, b, 1, hn] → [b, sk, hn]  (squeeze the 1-head dim for simpler indexing)
    k_flat = key.squeeze(2).permute(1, 0, 2).contiguous()  # [b, sk, hn]

    # Gather K at topk positions using advanced indexing
    # topk_indices: [b, sq, topk]
    # k_flat: [b, sk, hn]
    # We want k_sparse: [b, sq, topk, hn]
    # Clamp indices to valid range (indexer may produce out-of-range values
    # for padded/masked positions).
    topk_indices = topk_indices.clamp(0, sk - 1)
    batch_idx = torch.arange(b, device=key.device).view(b, 1, 1).expand_as(topk_indices)
    k_sparse = k_flat[batch_idx, topk_indices]  # [b, sq, topk, hn]

    # Scores: q [b, np, sq, hn] vs k_sparse [b, sq, topk, hn]
    # Use matmul: q[:,:,i,:] @ k_sparse[:,i,:,:].T for each query position
    # Reshape for batched matmul:
    #   q: [b, np, sq, hn] → [b*np, sq, hn]
    #   k_sparse: [b, sq, topk, hn] → [b, sq, hn, topk] (transpose last 2)
    # But np != 1, so we need to broadcast k_sparse across heads.
    # k_sparse_T: [b, 1, sq, hn, topk] broadcast to [b, np, sq, hn, topk]
    # scores = q @ k_sparse_T = [b, np, sq, 1, hn] @ [b, 1, sq, hn, topk]
    #        = [b, np, sq, 1, topk] → [b, np, sq, topk]

    # Simpler: use einsum with explicit broadcast
    # q: [b, np, sq, hn], k_sparse: [b, sq, topk, hn]
    scores = torch.einsum("bnsd,bskd->bnsk", q.float(), k_sparse.float()) * softmax_scale
    # Wait, this doesn't work — s (sq) in q maps to s (sq) in k_sparse,
    # and k (topk) is contracted... no, d (hn) should be contracted.
    # Let me be more careful:
    # q[b,n,s,d] × k_sparse[b,s,k,d] → scores[b,n,s,k]
    # Contract over d, keep b,n,s,k
    # einsum: "bnsd,bskd->bnsk"  — yes, contract d, free b,n,s; k from k_sparse
    # But s appears in both — it's a batch dim (each query position attends to
    # its own set of topk keys). So this IS correct.

    # Causal mask: positions where topk_indices > query position are future
    sq_positions = torch.arange(sq, device=q.device, dtype=topk_indices.dtype)
    future_mask = topk_indices > sq_positions.view(1, -1, 1)  # [b, sq, topk]
    scores.masked_fill_(future_mask.unsqueeze(1), float("-inf"))

    # Softmax over topk dim
    attn_w = F.softmax(scores, dim=-1, dtype=torch.float32)

    # Gather V (first v_channels of key) at topk positions
    # k_flat: [b, sk, hn], take first v_channels
    v_flat = k_flat[..., :v_channels]  # [b, sk, v_channels]
    v_sparse = v_flat[batch_idx, topk_indices]  # [b, sq, topk, v_channels]

    # Weighted sum: attn_w[b,n,s,k] × v_sparse[b,s,k,v] → output[b,n,s,v]
    output = torch.einsum("bnsk,bskv->bnsv", attn_w.to(v_sparse.dtype), v_sparse)

    # _run_sparse_attention expects [sq, b, np, v] (same as _unfused_absorbed_dsa_fn)
    # then applies: output = einsum("sbhc,hdc->sbhd", output, up_v_weight)
    return output.permute(2, 0, 1, 3).contiguous()  # [b,n,s,v] → [s,b,n,v]
