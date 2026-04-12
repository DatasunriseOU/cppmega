"""Sparse gather-scatter DSA attention — replaces Megatron's unfused_dsa_fn.

Megatron's ``unfused_dsa_fn`` (dsa.py:920) materializes the FULL
``[b*np, sq, sk]`` FP32 attention scores tensor = 7.0 GiB at production
shape (b=4, np=28, sq=sk=4096), then masks non-topk to -inf before
softmax. This is O(n²) in memory despite the sparse attention pattern
only using top-K = 16 entries per query.

This module replaces it with a proper **sparse gather-scatter** approach:
1. Gather K at topk indices: ``[sq, b, np, hn]`` → ``[b, np, sq, topk, hn]``
2. Compute scores ONLY for topk entries: ``[b, np, sq, topk]`` = 28.7 MB
3. Apply causal mask on topk positions
4. Softmax over topk dimension
5. Gather V at topk indices, weighted sum

Memory: ~28.7 MB scores + ~1.8 GB gathered K + ~1.8 GB gathered V
= ~3.7 GiB at production shape vs 7.0 GiB = **~47% reduction per layer,
~250× reduction in attention scores tensor specifically**.

With 5 DSA layers per pipeline stage: ~18.5 GiB vs ~35 GiB = **~16.5 GiB saved**.

Applied via monkey-patch in ``dsa_fp8_patch.py`` on ``dsa.unfused_dsa_fn``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse gather-scatter DSA attention.

    Args:
        query: ``[sq, b, np, hn]`` — query tensor (bf16/fp16/fp32).
        key: ``[sk, b, np, hn]`` — key tensor.
        value: ``[sk, b, np, hnv]`` — value tensor.
        topk_indices: ``[b, sq, topk]`` int64 — which K/V positions to attend.
        softmax_scale: float — scaling factor for QK^T.

    Returns:
        output: ``[sq, b, np * hnv]`` — same shape as ``unfused_dsa_fn``.
    """
    sq, b, np_, hn = query.size()
    sk = key.size(0)
    hnv = value.size(3)
    topk = topk_indices.size(-1)

    # ===================================================================
    # 1) Permute Q/K/V to batch-first for efficient gathering
    # ===================================================================
    # [sq, b, np, hn] → [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).contiguous()
    # [sk, b, np, hn] → [b, np, sk, hn]
    k = key.permute(1, 2, 0, 3).contiguous()
    # [sk, b, np, hnv] → [b, np, sk, hnv]
    v = value.permute(1, 2, 0, 3).contiguous()

    # ===================================================================
    # 2) Gather topk K entries per query position
    #    topk_indices: [b, sq, topk] — same indices for all heads (GQA)
    #    Expand to [b, np, sq, topk] then gather from k along sk dim
    # ===================================================================
    # [b, sq, topk] → [b, 1, sq, topk, 1] → [b, np, sq, topk, hn]
    idx_k = topk_indices.unsqueeze(1).unsqueeze(-1).expand(b, np_, sq, topk, hn)
    # k_expanded: [b, np, sk, hn] → [b, np, 1, sk, hn] (broadcast sq dim)
    # Gather along dim=3 (sk): [b, np, sq, topk, hn]
    k_sparse = k.unsqueeze(2).expand(-1, -1, sq, -1, -1).gather(3, idx_k)

    # ===================================================================
    # 3) Compute sparse attention scores: Q @ K_sparse^T
    #    q: [b, np, sq, hn], k_sparse: [b, np, sq, topk, hn]
    #    → scores: [b, np, sq, topk] = 28.7 MB (not 7 GiB!)
    # ===================================================================
    scores = torch.einsum("bnqd,bnqkd->bnqk", q.float(), k_sparse.float()) * softmax_scale
    del k_sparse

    # ===================================================================
    # 4) Causal mask: set future positions to -inf
    #    topk_indices[b, sq, topk] > sq_position → future → mask
    # ===================================================================
    sq_positions = torch.arange(sq, device=scores.device, dtype=topk_indices.dtype)
    # [b, sq, topk]: True where topk_index is in the future
    future_mask = topk_indices > sq_positions.view(1, -1, 1)
    # Expand to [b, np, sq, topk]
    scores.masked_fill_(future_mask.unsqueeze(1), float("-inf"))
    del future_mask

    # ===================================================================
    # 5) Softmax over topk dimension (not full sk!)
    # ===================================================================
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    del scores

    # ===================================================================
    # 6) Gather topk V entries and weighted sum
    # ===================================================================
    idx_v = topk_indices.unsqueeze(1).unsqueeze(-1).expand(b, np_, sq, topk, hnv)
    v_sparse = v.unsqueeze(2).expand(-1, -1, sq, -1, -1).gather(3, idx_v)

    # [b, np, sq, topk] @ [b, np, sq, topk, hnv] → [b, np, sq, hnv]
    output = torch.einsum("bnqk,bnqkd->bnqd", attn_weights.to(v_sparse.dtype), v_sparse)
    del attn_weights, v_sparse

    # ===================================================================
    # 7) Reshape back to Megatron expected format: [sq, b, np * hnv]
    # ===================================================================
    output = output.permute(2, 0, 1, 3).contiguous().reshape(sq, b, np_ * hnv)
    return output
