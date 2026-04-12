"""Split-K Triton fused indexer loss for Megatron DSA.

Ported from NVIDIA Megatron-LM PR #4039 "[Kernel] Fused Indexer Loss Kernel"
(https://github.com/NVIDIA/Megatron-LM/pull/4039).

The upstream PR implements a two-stage Triton kernel that computes the same
KL-divergence indexer loss as Megatron's ``compute_dsa_indexer_loss`` but
avoids materialising the full ``[b*np, sq, sk]`` attention-score tensor.

Stage 1 (``_fwd_fused_indexer_loss_stage1_kernel``):
    Grid = (B, ceil(ASq/BLOCK_SQ), AH).
    For each head h, computes attention Q@K^T per block, accumulates the
    online softmax running max ``softmax_m[b,h,sq]`` and denominator
    ``softmax_d[b,h,sq]``.  Head 0 also accumulates the online softmax
    for index_scores -> ``softmax_m1[b,sq]`` / ``softmax_d1[b,sq]``.

Stage 2 (``_fwd_fused_indexer_loss_stage2_kernel``):
    Grid = (B, ceil(ASq/BLOCK_SQ)).
    Recomputes Q@K^T blockwise (split-K over the ``sk`` dimension), applies
    the now-known softmax normalisation from stage 1, averages across heads,
    and reduces the per-position KL divergence into ``Loss[b, sq]``.

Memory saving: the upstream approach never allocates the
``[b*np, sq, sk]`` attention_scores tensor (~7.5 GiB at production shapes);
instead it stores only the softmax statistics ``[B, AH, sq]`` (~0.06 GiB).
Net saving is ~60% peak memory for the indexer loss forward.

This module exposes a single public function:

    compute_dsa_indexer_loss_splitk(
        index_scores, topk_indices, query, key,
        softmax_scale, loss_coeff, sparse_loss, pg_collection,
    ) -> torch.Tensor

which has the same signature as Megatron's ``compute_dsa_indexer_loss``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "compute_dsa_indexer_loss_splitk",
]


# ---------------------------------------------------------------------------
# Stage 1: compute softmax statistics (online max + denominator)
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_fused_indexer_loss_stage1_kernel(
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Scores_ptr,
    Index_Mask_ptr,
    m_ptr,
    d_ptr,
    m1_ptr,
    d1_ptr,
    # Attn query strides: [Sq, B, H, D]
    stride_asq,
    stride_aqb,
    stride_aqh,
    stride_aqd,
    # Attn key strides: [Sk, B, H, D]
    stride_ask,
    stride_akb,
    stride_akh,
    stride_akd,
    # Loss strides: [B, Sq]
    stride_lb,
    stride_ls,
    # Index scores strides: [B, Sq, Sk]
    stride_ibs,
    stride_isq,
    stride_isk,
    # Index mask strides: [B, Sq, Sk]
    stride_imb,
    stride_ims,
    stride_imk,
    # softmax m strides: [B, H, Sq]
    stride_smmb,
    stride_smmh,
    stride_smmq,
    # softmax d strides: [B, H, Sq]
    stride_smdb,
    stride_smdh,
    stride_smdq,
    # softmax m1 strides: [B, Sq]
    stride_sm1b,
    stride_sm1q,
    # softmax d1 strides: [B, Sq]
    stride_sd1b,
    stride_sd1q,
    # Dimensions
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sk: tl.constexpr,
    ASq: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    sq_block_id = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)

    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = (sq < ASq)

    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass online softmax accumulators
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # Causal mask: only iterate sk blocks up to min(sq)+1
    causal_sk = tl.minimum(tl.min(sq) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        index_scores = tl.load(
            Index_Scores_ptr + b * stride_ibs + sq[:, None] * stride_isq + sk_offs[None, :] * stride_isk,
            mask=(sq_valid[:, None] & sk_valid[None, :]),
            other=float("-inf"),
        )

        if SPARSE_LOSS:
            index_mask = tl.load(
                Index_Mask_ptr + b * stride_imb + sq[:, None] * stride_ims + sk_offs[None, :] * stride_imk,
                mask=(sq_valid[:, None] & sk_valid[None, :]),
                other=float("-inf"),
            )
            index_scores += index_mask

        if h == 0:
            # First head accumulates index softmax stats
            m1_i_1 = m1_i
            m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
            m1_i = tl.where(m1_i <= float("-inf"), 0.0, m1_i)
            d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.exp(index_scores - m1_i[:, None]).sum(axis=1)

        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)

        h_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for d_start in tl.range(0, AD, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < AD

            aq_ptrs = aq_base + h * stride_aqh + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
            aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

            ak_ptrs = ak_base + h * stride_akh + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
            ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

            h_scores = tl.dot(aq_vals, ak_vals, acc=h_scores, allow_tf32=False)

        h_scores *= Softmax_Scale
        h_scores += casual_mask

        if SPARSE_LOSS:
            h_scores += index_mask

        m_i = tl.load(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, mask=sq_valid, other=float("-inf"))
        d_i = tl.load(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, mask=sq_valid, other=0.0)
        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(h_scores, axis=-1))
        m_i = tl.where(m_i <= float("-inf"), 0.0, m_i)
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.exp(h_scores - m_i[:, None]).sum(axis=-1)
        tl.store(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, m_i, mask=sq_valid)
        tl.store(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, d_i, mask=sq_valid)

    if h == 0:
        tl.store(m1_ptr + b * stride_sm1b + sq * stride_sm1q, m1_i, mask=sq_valid)
        tl.store(d1_ptr + b * stride_sd1b + sq * stride_sd1q, d1_i, mask=sq_valid)


# ---------------------------------------------------------------------------
# Stage 2: recompute Q@K^T blockwise, apply normalised softmax, compute KL
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_fused_indexer_loss_stage2_kernel(
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Scores_ptr,
    Index_Mask_ptr,
    m_ptr,
    d_ptr,
    m1_ptr,
    d1_ptr,
    # Attn query strides: [Sq, B, H, D]
    stride_asq,
    stride_aqb,
    stride_aqh,
    stride_aqd,
    # Attn key strides: [Sk, B, H, D]
    stride_ask,
    stride_akb,
    stride_akh,
    stride_akd,
    # Loss strides: [B, Sq]
    stride_lb,
    stride_ls,
    # Index scores strides: [B, Sq, Sk]
    stride_ibs,
    stride_isq,
    stride_isk,
    # Index mask strides: [B, Sq, Sk]
    stride_imb,
    stride_ims,
    stride_imk,
    # softmax m strides: [B, H, Sq]
    stride_smmb,
    stride_smmh,
    stride_smmq,
    # softmax d strides: [B, H, Sq]
    stride_smdb,
    stride_smdh,
    stride_smdq,
    # softmax m1 strides: [B, Sq]
    stride_sm1b,
    stride_sm1q,
    # softmax d1 strides: [B, Sq]
    stride_sd1b,
    stride_sd1q,
    # Dimensions
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sk: tl.constexpr,
    ASq: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    sq_block_id = tl.program_id(1).to(tl.int64)

    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = (sq < ASq)

    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # Load pre-computed index softmax statistics from stage 1
    m1_i = tl.load(m1_ptr + b * stride_sm1b + sq * stride_sm1q, mask=sq_valid, other=float("-inf"))
    d1_i = tl.load(d1_ptr + b * stride_sd1b + sq * stride_sd1q, mask=sq_valid, other=0.0)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # Recompute attention + KL divergence blockwise
    causal_sk = tl.minimum(tl.min(sq) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        index_scores = tl.load(
            Index_Scores_ptr + b * stride_ibs + sq[:, None] * stride_isq + sk_offs[None, :] * stride_isk,
            mask=(sq_valid[:, None] & sk_valid[None, :]),
            other=float("-inf"),
        )

        if SPARSE_LOSS:
            index_mask = tl.load(
                Index_Mask_ptr + b * stride_imb + sq[:, None] * stride_ims + sk_offs[None, :] * stride_imk,
                mask=(sq_valid[:, None] & sk_valid[None, :]),
                other=float("-inf"),
            )
            index_scores += index_mask

        softmax_attn_i = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)

        for h in tl.range(0, AH):
            h_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_base + h * stride_aqh + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                ak_ptrs = ak_base + h * stride_akh + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                h_scores = tl.dot(aq_vals, ak_vals, acc=h_scores, allow_tf32=False)

            h_scores *= Softmax_Scale
            h_scores += casual_mask

            if SPARSE_LOSS:
                h_scores += index_mask

            # Apply normalised softmax using stage-1 statistics
            m_i = tl.load(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, mask=sq_valid, other=float("-inf"))
            d_i = tl.load(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, mask=sq_valid, other=0.0)
            softmax_attn_i += tl.exp(h_scores - m_i[:, None]) / d_i[:, None]

        softmax_attn_i /= AH
        softmax_index_i = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]

        # KL divergence: sum_j p * (log p - log q)
        loss_sk = softmax_attn_i * (tl.log(softmax_attn_i + 1e-10) - tl.log(softmax_index_i + 1e-10))
        loss_i += loss_sk.sum(axis=-1)

    # Store per-position loss
    tl.store(Loss_ptr + b * stride_lb + sq * stride_ls, loss_i, mask=sq_valid)


# ---------------------------------------------------------------------------
# Public entry point — drop-in for ``compute_dsa_indexer_loss``
# ---------------------------------------------------------------------------

def compute_dsa_indexer_loss_splitk(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: object,
) -> torch.Tensor:
    """Compute DSA indexer KL-divergence loss using split-K Triton recomputation.

    Same signature as ``megatron.core.transformer.experimental_attention_variant.
    dsa.compute_dsa_indexer_loss``.

    The ``pg_collection`` argument is accepted for API compatibility but TP
    all-reduce is NOT fused into the kernel (upstream PR notes this too).
    When ``pg_collection.tp.size() > 1`` the caller should fall back to the
    native path — the monkey-patch routing in ``dsa_fp8_patch.py`` handles
    this.

    Memory saving vs upstream: avoids materialising ``[b*np, sq, sk]``
    attention_scores tensor. At production shapes (b=1, np=128, sq=4096,
    sk=4096) this saves ~7.5 GiB -> ~0.06 GiB for softmax stats, i.e.
    ~60% of the forward peak for the indexer loss computation.
    """
    # query: [sq, b, np, hn]  key: [sk, b, np, hn]
    ASq, AB, AH, AD = query.shape
    ASk = key.shape[0]

    assert AH <= 128, (
        "Split-K indexer loss kernel may be numerically incorrect for AH > 128 "
        f"(got AH={AH}). See upstream PR #4039."
    )

    BLOCK_SK = 128
    BLOCK_SQ = 128
    BLOCK_D = 64
    num_warps = 8
    num_stages = 3

    out_loss = torch.empty((AB, ASq), dtype=torch.float32, device=query.device)
    attn_num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ

    # Build sparse index mask if needed
    if sparse_loss:
        index_mask = torch.full(
            (AB, ASq, ASk), float("-inf"), dtype=torch.float32, device=index_scores.device,
        ).scatter_(-1, topk_indices, 0)
        stride_imb = index_mask.stride(0)
        stride_ims = index_mask.stride(1)
        stride_imk = index_mask.stride(2)
    else:
        index_mask = torch.empty((0,), dtype=torch.float32, device=query.device)
        stride_imb = stride_ims = stride_imk = 0

    # Allocate softmax statistics buffers
    softmax_m = torch.full((AB, AH, ASq), float("-inf"), dtype=torch.float32, device=query.device)
    softmax_d = torch.full((AB, AH, ASq), 0.0, dtype=torch.float32, device=query.device)
    softmax_m1 = torch.full((AB, ASq), float("-inf"), dtype=torch.float32, device=query.device)
    softmax_d1 = torch.full((AB, ASq), 0.0, dtype=torch.float32, device=query.device)

    # Common stride kwargs
    _stride_kwargs = dict(
        stride_asq=query.stride(0),
        stride_aqb=query.stride(1),
        stride_aqh=query.stride(2),
        stride_aqd=query.stride(3),
        stride_ask=key.stride(0),
        stride_akb=key.stride(1),
        stride_akh=key.stride(2),
        stride_akd=key.stride(3),
        stride_lb=out_loss.stride(0),
        stride_ls=out_loss.stride(1),
        stride_ibs=index_scores.stride(0),
        stride_isq=index_scores.stride(1),
        stride_isk=index_scores.stride(2),
        stride_imb=stride_imb,
        stride_ims=stride_ims,
        stride_imk=stride_imk,
        stride_smmb=softmax_m.stride(0),
        stride_smmh=softmax_m.stride(1),
        stride_smmq=softmax_m.stride(2),
        stride_smdb=softmax_d.stride(0),
        stride_smdh=softmax_d.stride(1),
        stride_smdq=softmax_d.stride(2),
        stride_sm1b=softmax_m1.stride(0),
        stride_sm1q=softmax_m1.stride(1),
        stride_sd1b=softmax_d1.stride(0),
        stride_sd1q=softmax_d1.stride(1),
    )

    # Common dimension constexprs
    _dim_kwargs = dict(
        AH=AH,
        AD=AD,
        Sk=ASk,
        ASq=ASq,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # --- Stage 1: compute softmax statistics ---
    stage1_grid = (AB, attn_num_sq_blocks, AH)
    _fwd_fused_indexer_loss_stage1_kernel[stage1_grid](
        Attn_Query_ptr=query,
        Attn_Key_ptr=key,
        Loss_ptr=out_loss,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        m_ptr=softmax_m,
        d_ptr=softmax_d,
        m1_ptr=softmax_m1,
        d1_ptr=softmax_d1,
        **_stride_kwargs,
        **_dim_kwargs,
    )

    # --- Stage 2: recompute + KL divergence ---
    stage2_grid = (AB, attn_num_sq_blocks)
    _fwd_fused_indexer_loss_stage2_kernel[stage2_grid](
        Attn_Query_ptr=query,
        Attn_Key_ptr=key,
        Loss_ptr=out_loss,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        m_ptr=softmax_m,
        d_ptr=softmax_d,
        m1_ptr=softmax_m1,
        d1_ptr=softmax_d1,
        **_stride_kwargs,
        **_dim_kwargs,
    )

    indexer_loss = out_loss.mean() * loss_coeff
    return indexer_loss
