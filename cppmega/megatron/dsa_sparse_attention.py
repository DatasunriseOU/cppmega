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
    # 2-6) Head-chunked sparse attention to avoid O(b*np*sq*topk*hn) peak
    #
    # With topk=256, the naive gather k_sparse[b,np,sq,topk,hn] is 42 GiB.
    # We chunk over heads to keep peak at O(b*chunk*sq*topk*hn).
    # ===================================================================
    # Precompute causal mask: topk_indices[b, sq, topk] > position → future
    sq_positions = torch.arange(sq, device=q.device, dtype=topk_indices.dtype)
    future_mask = topk_indices > sq_positions.view(1, -1, 1)  # [b, sq, topk]

    # Per-head index templates (same for all heads — GQA shared indices)
    # [b, sq, topk, 1]
    idx_k_base = topk_indices.unsqueeze(-1)
    idx_v_base = topk_indices.unsqueeze(-1)

    # Budget: K_gather(bf16) + V_gather(bf16) + scores(fp32) + attn_w(fp32) per chunk
    bytes_per_head = b * sq * topk * (hn * 2 + hnv * 2 + 4 + 4)
    free_bytes = torch.cuda.mem_get_info(q.device)[0]
    # Use at most 80% of free memory for the chunk to leave room for autograd
    head_chunk = min(np_, max(1, int(free_bytes * 0.8) // bytes_per_head))
    output = torch.zeros(b, np_, sq, hnv, device=q.device, dtype=q.dtype)

    for h0 in range(0, np_, head_chunk):
        h1 = min(h0 + head_chunk, np_)
        nc = h1 - h0

        # Gather K for this head chunk: [b, nc, sq, topk, hn]
        idx_k = idx_k_base.unsqueeze(1).expand(b, nc, sq, topk, hn)
        k_chunk = k[:, h0:h1].unsqueeze(2).expand(-1, -1, sq, -1, -1).gather(3, idx_k)

        # Scores in bf16 then upcast only for softmax
        scores = torch.einsum("bnqd,bnqkd->bnqk", q[:, h0:h1], k_chunk) * softmax_scale
        del k_chunk
        scores.masked_fill_(future_mask.unsqueeze(1), float("-inf"))
        attn_w = F.softmax(scores.float(), dim=-1, dtype=torch.float32)
        del scores

        # Gather V for this head chunk: [b, nc, sq, topk, hnv]
        idx_v = idx_v_base.unsqueeze(1).expand(b, nc, sq, topk, hnv)
        v_chunk = v[:, h0:h1].unsqueeze(2).expand(-1, -1, sq, -1, -1).gather(3, idx_v)

        output[:, h0:h1] = torch.einsum("bnqk,bnqkd->bnqd", attn_w.to(v_chunk.dtype), v_chunk)
        del attn_w, v_chunk

    del future_mask

    # ===================================================================
    # 7) Reshape back to Megatron expected format: [sq, b, np * hnv]
    # ===================================================================
    output = output.permute(2, 0, 1, 3).contiguous().reshape(sq, b, np_ * hnv)
    return output
