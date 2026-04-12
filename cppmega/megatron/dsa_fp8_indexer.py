"""FP8 port of DeepSeek V3.2 ``fp8_index`` for the Megatron DSA indexer path.

This module is a drop-in replacement for Megatron's BF16
``_compute_index_scores`` in
``megatron/core/transformer/experimental_attention_variant/dsa.py``.

Reference (TileLang implementation):
    https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L200-L275

Reference Megatron BF16 implementation (the function we are mirroring):
    megatron/core/transformer/experimental_attention_variant/dsa.py::_compute_index_scores

Design
------
The DeepSeek reference computes

    index_score = sum_h (relu(q @ k^T) * weights_h) * k_scale

with ``q`` and ``k`` quantised to ``float8_e4m3fn`` using rowwise per-group
scales (group_size = 128). For the Megatron DSA indexer ``index_head_dim`` is
always ``<= 128`` in our recipes, so per-row and per-128-group scaling
collapse into the same thing: one fp32 scale per row of ``q``/``k``.

Rather than carrying a TileLang kernel into Megatron-LM (which would add a
new build-time dependency and tilelang jit cache on every worker), we use
``torch._scaled_mm`` from torch 2.12 + cu132 on H200. ``_scaled_mm`` accepts
rowwise fp32 scales, emits fp32 accumulation, and lowers to WGMMA on
``sm_90a`` — so we get the same architectural behaviour as the TileLang
kernel (fp8 a @ fp8 b -> fp32) without a new kernel build.

Numerical contract vs BF16 path:

* Output shape: ``[batch, seqlen_q, seqlen_k]`` fp32 (identical).
* Precision: fp8_e4m3 with per-row scales -> expected max-rel-err ~5%,
  topk overlap (k=16 over seq 128) >=85% on gaussian inputs.
* Math equivalence: ``relu(q @ k^T) * weights_per_head`` summed over heads
  then transposed to ``[b, sq, sk]`` — identical op order to BF16 path.

The module exposes three public symbols:

* :func:`compute_index_scores_fp8` — drop-in replacement (takes ``q``,
  ``weights``, ``k``, returns ``[b, sq, sk]`` fp32).
* :func:`quantize_rowwise_fp8` — helper used by both forward and the
  monkey-patch hook below; also exposed for unit tests.
* :func:`patch_megatron_dsa` — imperative monkey-patch entry point that
  rebinds ``dsa._compute_index_scores`` when the config / environment asks
  for fp8. See :mod:`cppmega.megatron.dsa_fp8_patch`.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = [
    "FP8_E4M3_MAX",
    "compute_index_scores_fp8",
    "compute_index_scores_bf16_reference",
    "quantize_rowwise_fp8",
    "bwd_fused_indexer_loss_fp8",
    "bwd_fused_indexer_loss_bf16_reference",
]

# torch.float8_e4m3fn max representable magnitude.
FP8_E4M3_MAX: float = 448.0


def quantize_rowwise_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise absmax quantise ``x`` from bf16/fp16/fp32 to fp8_e4m3fn.

    The last dimension is treated as the rowwise axis (matching DeepSeek's
    ``act_quant(x, block_size=128)`` with ``x.size(-1) <= 128``).

    Args:
        x: tensor with shape ``[..., K]`` where ``K`` is the GEMM
            contraction dimension. Any leading shape is preserved in
            ``x_fp8`` and the per-row scales.

    Returns:
        Tuple ``(x_fp8, x_scale)`` where ``x_fp8`` has dtype
        ``torch.float8_e4m3fn`` (same shape as ``x``) and ``x_scale`` has
        shape ``x.shape[:-1]`` with dtype fp32. ``x ≈ x_fp8.float() *
        x_scale.unsqueeze(-1)``.
    """

    assert x.is_floating_point(), "quantize_rowwise_fp8 requires a float tensor"
    x_f32 = x.float()
    # amax along the contraction dim, shape [...,]
    amax = x_f32.abs().amax(dim=-1).clamp(min=1e-4)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    # scaled = x / scale, clamp to fp8 range, then cast
    scaled = (x_f32 / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale


def _scaled_mm_rowwise(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Call ``torch._scaled_mm`` with rowwise scales.

    ``a_fp8`` is ``[M, K]`` row-major, ``b_fp8`` is ``[N, K]`` row-major.
    Returns fp32 ``[M, N]`` where result[i, j] = sum_k a[i,k]*b[j,k] in fp8,
    scaled back by ``a_scale[i] * b_scale[j]``.
    """

    assert a_fp8.dim() == 2 and b_fp8.dim() == 2
    assert a_fp8.size(1) == b_fp8.size(1), (
        f"contraction mismatch: a_fp8={tuple(a_fp8.shape)}, b_fp8={tuple(b_fp8.shape)}"
    )
    M = a_fp8.size(0)
    N = b_fp8.size(0)
    # _scaled_mm expects b in "column-major" layout of shape [K, N]. We
    # achieve that by transposing a row-major [N, K] — the transpose flips
    # strides but keeps the storage, which the CUTLASS fp8 gemm treats as
    # column-major.
    b_colmajor = b_fp8.t()
    return torch._scaled_mm(
        a_fp8.contiguous(),
        b_colmajor,
        scale_a=a_scale.to(torch.float32).view(M, 1).contiguous(),
        scale_b=b_scale.to(torch.float32).view(1, N).contiguous(),
        out_dtype=torch.float32,
    )


def compute_index_scores_fp8(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """FP8 analogue of Megatron DSA's ``_compute_index_scores``.

    This mirrors the DeepSeek V3.2 ``fp8_index`` kernel's math using
    torch's ``_scaled_mm`` FP8 GEMM (lowered to WGMMA on H200 ``sm_90a``).

    Args:
        q: bf16 ``[seqlen_q, batch, index_n_heads, index_head_dim]`` query.
        weights: bf16 ``[seqlen_q, batch, index_n_heads]`` per-head weights.
        k: bf16 ``[seqlen_k, batch, index_head_dim]`` key.

    Returns:
        fp32 ``[batch, seqlen_q, seqlen_k]`` index scores, same shape and
        semantics as the BF16 path.
    """

    assert q.dim() == 4, f"q must be [sq,b,h,d], got shape {tuple(q.shape)}"
    assert k.dim() == 3, f"k must be [sk,b,d], got shape {tuple(k.shape)}"
    assert weights.dim() == 3, (
        f"weights must be [sq,b,h], got shape {tuple(weights.shape)}"
    )

    sq, b, h, d = q.shape
    sk, bk, dk = k.shape
    assert bk == b and dk == d, (
        f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}"
    )

    # ------------------------------------------------------------------
    # Rowwise FP8 quantisation
    # ------------------------------------------------------------------
    # q: [sq, b, h, d] -> q_fp8 [sq, b, h, d], q_scale [sq, b, h]
    # k: [sk, b, d]    -> k_fp8 [sk, b, d],    k_scale [sk, b]
    q_fp8, q_scale = quantize_rowwise_fp8(q)
    k_fp8, k_scale = quantize_rowwise_fp8(k)

    # ------------------------------------------------------------------
    # Per-batch GEMM with fused per-head reduction. ``_scaled_mm`` has no
    # bmm variant, so we loop over ``b`` AND over ``h``: each head
    # produces a ``[sq, sk]`` fp32 block which is relu'd, weighted, and
    # accumulated into the final ``[b, sq, sk]`` output in place. This is
    # critical for memory footprint --- the BF16 path allocated
    # ``index_scores [sq, b, h, sk] fp32`` live across ``sum(dim=2)``,
    # which is ``b*sq*h*sk*4`` bytes (~512 MB per DSA layer at
    # sq=sk=4096, b=2, h=8) and was the direct cause of the Stream D
    # OOM in docs/nam56r_grid_search_2026_04_12.md at the 9+4 DSA
    # layout. We now only keep ``[b, sq, sk]`` live.
    # ------------------------------------------------------------------
    index_scores = torch.zeros(
        (b, sq, sk),
        dtype=torch.float32,
        device=q.device,
    )
    for bi in range(b):
        # k is shared across heads in DSA's algorithm
        k_fp8_bi = k_fp8[:, bi, :].reshape(sk, d)
        k_s_bi = k_scale[:, bi].reshape(sk)
        for hi in range(h):
            a = q_fp8[:, bi, hi, :].reshape(sq, d)
            a_s = q_scale[:, bi, hi].reshape(sq)

            # fp8 gemm: [sq, d] @ [sk, d]^T -> [sq, sk] fp32
            logits = _scaled_mm_rowwise(a, a_s, k_fp8_bi, k_s_bi)  # [sq, sk]

            # Apply ReLU and weight by this head's per-token weight.
            # Fused into the accumulation; never allocates [sq, h, sk].
            w_h = weights[:, bi, hi].float().unsqueeze(-1)  # [sq, 1]
            index_scores[bi].add_(torch.relu(logits) * w_h)

    return index_scores


def compute_index_scores_bf16_reference(
    q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Local reference clone of Megatron's BF16 ``_compute_index_scores``.

    Kept here so unit tests can be run without a Megatron checkout. This
    function is byte-for-byte identical to the dsa.py implementation as of
    2026-04-12 (the "This is a BF16 implementation of the fp8_index logic"
    function).
    """

    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)
    return index_scores


# ---------------------------------------------------------------------------
# Stream G: FP8 backward port for DSA FusedDSAIndexerLoss.
# ---------------------------------------------------------------------------
#
# Upstream reference (bench3 megatron-lm, dsa.py::bwd_fused_indexer_loss_naive
# fetched 2026-04-12, lines ~346-500) does two distinct heavy GEMMs and a
# pair of einsums:
#
#   (1) Main attention ``Q@K^T`` for the KL target:
#         attention_scores [b*np, sq, sk] fp32 via torch.bmm(Q.float(), K.float())
#       This output is then reshaped to [b, np, sq, sk], masked, softmaxed,
#       summed over the head dim and L1 normalised. The softmax/L1 is
#       non-linear across ``sk`` and must happen BEFORE reducing heads ---
#       so we cannot stream one head at a time into a [b, sq, sk] accumulator
#       the way we can for the indexer side. We therefore LEAVE the main
#       attention bmm in BF16/FP32 exactly as upstream. Documented as a
#       deliberate trade-off in docs/nam56r_mtp_optimization_plan_...
#
#   (2) Indexer recompute ``torch.einsum('sbhd,tbd->sbht', q.float(), k.float())``
#       to build ``scores [sq, b, h, sk] fp32`` → ``relu_mask`` (bool)
#       + ``scores_after_relu``. Shape-wise this is ~3.2 GB at production.
#       This is a FP8 candidate AND the per-head fusion trick from forward
#       works here: relu_mask / scores_after_relu / grad_scores all live
#       only in a ``[sq, sk]`` working buffer per (b, h).
#
#   (3) Final ``grad_q = einsum('sbht,tbd->sbhd', grad_scores, k.float())``
#       and ``grad_k = einsum('sbht,sbhd->tbd', grad_scores, q.float())``.
#       Both can be fused into the same per-(b, h) loop that produces the
#       relu_mask --- we compute logits once, relu+apply grad_weighted, then
#       immediately do the two small FP8 GEMMs against k[bi] / q[bi, hi] to
#       accumulate into grad_q / grad_k.
#
# The FP8 backward therefore replaces (2)+(3) and leaves (1) alone. It
# mirrors the upstream control flow for masking and softmax so that the
# KL divergence path is numerically the same (aside from the indexer's FP8
# noise propagating into ``grad_index_scores_softmax``).


def _attention_target_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    topk_indices: torch.Tensor,
    sparse_loss: bool,
    pg_collection,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the KL target ``attention_scores_normalized`` via head-streaming.

    Original Megatron code materializes ``[b*np, sq, sk]`` in FP32 = 7.5 GiB
    at production shape (b=4, np=28, sq=sk=4096). This head-streaming variant
    loops over ``np`` heads, computing per-head softmax and accumulating into
    a ``[b, sq, sk]`` FP32 buffer. Peak live: ~0.8 GiB (3 × ``[b, sq, sk]``),
    a **~89% reduction**.

    Mathematically identical: softmax is per-head per-row (does not depend on
    other heads), and sum-over-heads is linear post-softmax. The only
    reduction across heads is the final sum which commutes with per-head
    softmax. L1 normalization happens AFTER summing — same in both variants.

    Pattern source: FlashAttention-2/3/4 online softmax (Milakov &
    Gimelshein 2018, arxiv 1805.02867) applied to the head axis instead of
    the KV-tile axis. DeepSeek V3.2 inference ``fp8_index_kernel`` uses the
    same principle: ``T.reduce_sum(logits, logits_sum, dim=1)`` sums over
    heads inside the kernel, never materializing ``[b, h, sq, sk]``.

    Returns:
        ``(attention_scores_normalized [b, sq, sk] fp32, index_mask [b, sq, sk] fp32)``
    """

    sq, b, np_, hn = query.size()
    sk = key.size(0)

    device = query.device

    # Masks: computed once, reused across all heads.
    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )
    # Validate topk_indices are in range [0, sk) before scatter.
    import sys as _sys
    _topk_max = topk_indices.max().item()
    _topk_min = topk_indices.min().item()
    if _topk_max >= sk or _topk_min < 0:
        print(
            f"[head_stream_debug] ERROR: topk_indices out of range! "
            f"min={_topk_min} max={_topk_max} sk={sk} "
            f"topk_indices.shape={tuple(topk_indices.shape)} "
            f"query.shape={tuple(query.shape)} key.shape={tuple(key.shape)} "
            f"topk_indices.dtype={topk_indices.dtype}",
            file=_sys.stderr, flush=True,
        )
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=device
    ).scatter_(-1, topk_indices.clamp(0, sk - 1), 0)

    # Accumulator for sum-over-heads of post-softmax distributions.
    acc = torch.zeros(b, sq, sk, dtype=torch.float32, device=device)

    for h in range(np_):
        # Per-head Q and K slices: [sq, b, hn] and [sk, b, hn].
        q_h = query[:, :, h, :].float().permute(1, 0, 2).contiguous()   # -> [b, sq, hn]
        k_h = key[:, :, h, :].float().permute(1, 2, 0).contiguous()     # -> [b, hn, sk]

        if h == 0:
            import sys
            _alloc = torch.cuda.memory_allocated(device) / (1024**3)
            _res = torch.cuda.memory_reserved(device) / (1024**3)
            print(
                f"[head_stream_debug] h=0 q_h={tuple(q_h.shape)} k_h={tuple(k_h.shape)} "
                f"q_h.is_contiguous={q_h.is_contiguous()} k_h.is_contiguous={k_h.is_contiguous()} "
                f"alloc={_alloc:.2f}GiB reserved={_res:.2f}GiB",
                file=sys.stderr, flush=True,
            )

        # Per-head scores: [b, sq, sk] = 268 MB at production shape.
        scores_h = torch.bmm(q_h, k_h) * softmax_scale
        del q_h, k_h

        # Apply masks in-place (no gradient through detached query/key).
        scores_h.add_(causal_mask)
        if sparse_loss:
            scores_h.add_(index_mask)

        # Per-head, per-row softmax — independent of other heads.
        softmax_h = torch.nn.functional.softmax(scores_h, dim=-1, dtype=torch.float32)
        del scores_h

        # Accumulate into head-sum buffer.
        acc.add_(softmax_h)
        del softmax_h

    del causal_mask

    # Optional TP all-reduce (sum local-TP heads across ranks).
    if pg_collection is not None and pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(acc.contiguous(), group=pg_collection.tp)

    # L1 normalize after summing (same math as original).
    attention_scores_normalized = acc / acc.sum(dim=-1, keepdim=True)
    del acc
    return attention_scores_normalized, index_mask


def _index_scores_softmax_fp32(
    index_scores_bsqsk: torch.Tensor,
    sq: int,
    sk: int,
    b: int,
    index_mask: torch.Tensor,
    sparse_loss: bool,
) -> torch.Tensor:
    """Apply causal + (optional) sparse mask to the pre-computed index
    scores and run FP32 softmax along ``sk``.

    ``index_scores_bsqsk`` is the FP8-recomputed ``[b, sq, sk] fp32`` output
    from ``compute_index_scores_fp8``. This helper only adds the masks and
    softmaxes --- no GEMMs. Needed so the caller can reuse the cheap FP8
    recompute from forward instead of doing a BF16 einsum over ``q @ k^T``
    all over again.
    """

    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=index_scores_bsqsk.device),
        diagonal=1,
    )
    index_scores = index_scores_bsqsk + causal_mask.unsqueeze(0)
    if sparse_loss:
        index_scores = index_scores + index_mask
    del causal_mask
    return torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)


def bwd_fused_indexer_loss_fp8(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    grad_loss: torch.Tensor,
    pg_collection,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 port of Megatron DSA ``bwd_fused_indexer_loss_naive``.

    Replaces two GEMM families with FP8 ``_scaled_mm`` + per-head fused
    accumulation:

    * Indexer score recompute (``einsum q @ k^T`` for relu_mask) --- fused
      with the ``grad_q / grad_k`` einsums so the ``[sq, b, h, sk] fp32``
      intermediate never materialises.
    * Main attention ``Q @ K^T`` for the KL target --- **kept BF16/FP32**
      because its output feeds a non-linear softmax → sum → L1 normalise
      that forces the full ``[b, np, sq, sk]`` tensor to live before
      reducing. See the design note in this module.

    The KL-divergence control flow (softmax, mean, masking, ``grad_loss``
    propagation) is identical to upstream so the autograd surface of
    ``FusedDSAIndexerLoss.backward`` stays the same.

    Args:
        q: bf16 ``[sq, b, h, d]`` indexer query.
        weights: bf16 ``[sq, b, h]`` per-head weights.
        k: bf16 ``[sk, b, d]`` indexer key.
        query: bf16 ``[sq, b, np, hn]`` main-attention query.
        key: bf16 ``[sk, b, np, hn]`` main-attention key.
        topk_indices: int64 ``[b, sq, topk]`` selected keys.
        softmax_scale: ``1/sqrt(hn)`` scale for the main attention.
        loss_coeff: scalar multiplier on the KL loss.
        sparse_loss: whether to restrict the KL target to topk positions.
        grad_loss: incoming scalar grad wrt the indexer loss.
        pg_collection: Megatron process group collection (may be ``None``
            in unit tests; TP all-reduce is then skipped).

    Returns:
        ``(grad_q, grad_weights, grad_k)`` in the same dtypes as the
        incoming tensors (bf16 by default).
    """

    assert q.dim() == 4 and k.dim() == 3 and weights.dim() == 3
    sq, b, h, d = q.shape
    sk = k.shape[0]

    # (A) Cheap recompute of the indexer output [b, sq, sk] via FP8 forward
    # path. This is Stream E code; reuses the same per-head fused loop.
    index_scores = compute_index_scores_fp8(q, weights, k)  # [b, sq, sk] fp32

    # (B) Compute the KL target (main-attention side, BF16/FP32).
    attention_scores_normalized, index_mask = _attention_target_fp32(
        query, key, softmax_scale, topk_indices, sparse_loss, pg_collection
    )

    # (C) Mask + softmax the indexer side to match upstream before taking
    # the KL backward.
    index_scores_softmax = _index_scores_softmax_fp32(
        index_scores, sq, sk, b, index_mask, sparse_loss
    )
    del index_scores

    # (D) Backward through loss = kl_div * loss_coeff.
    grad_kl_div = grad_loss * loss_coeff  # scalar tensor
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized
        / (index_scores_softmax + 1e-10)
        * grad_kl_per_element
    )
    del attention_scores_normalized

    # Backward through softmax.
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # (E) Zero out gradients for masked positions.
    causal_valid_mask = torch.tril(torch.ones((sq, sk), device=q.device, dtype=torch.bool))
    if sparse_loss:
        index_valid_mask = index_mask == 0
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask
        del index_valid_mask
    else:
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)
    del causal_valid_mask
    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Upstream then does grad_index_scores = grad_index_scores_logits.transpose(0, 1)
    # to get [sq, b, sk] and grad_weighted_scores = grad_index_scores.unsqueeze(2)
    # [sq, b, 1, sk]. We skip the explicit transpose/unsqueeze: we index
    # grad_index_scores_logits as [bi, si, :] inside the per-(b,h) loop.

    # (F) Allocate the three output grads. grad_q is [sq, b, h, d] fp32,
    # grad_k is [sk, b, d] fp32, grad_weights is [sq, b, h] fp32. We write
    # in fp32 first then cast down to match upstream's final `.to(dtype)`.
    grad_q_f32 = torch.zeros((sq, b, h, d), dtype=torch.float32, device=q.device)
    grad_k_f32 = torch.zeros((sk, b, d), dtype=torch.float32, device=k.device)
    grad_weights_f32 = torch.zeros((sq, b, h), dtype=torch.float32, device=weights.device)

    # ------------------------------------------------------------------
    # Per-(b, h) fused FP8 loop. For each (bi, hi) we do:
    #
    #   1. FP8 GEMM ``a_qh[sq, d] @ kb[sk, d]^T -> logits [sq, sk] fp32``
    #   2. relu_mask = logits > 0 ; scores_after_relu = relu(logits)
    #   3. grad_weights[si, bi, hi] = sum_sk(g_il[bi, si, :] * scores_after_relu[si, :])
    #   4. grad_scores[sq, sk] = g_il[bi, :, :] * weights[:, bi, hi].unsqueeze(-1) * relu_mask
    #   5. FP8 GEMM ``grad_scores [sq, sk] @ kb[sk, d] -> dq [sq, d]`` accumulate into grad_q
    #   6. FP8 GEMM ``grad_scores.t() [sk, sq] @ a_qh[sq, d] -> dk [sk, d]`` accumulate into grad_k
    #
    # Never materialises any [sq, h, sk] or [b, sq, h, sk] tensor.
    # ------------------------------------------------------------------
    q_fp8, q_scale = quantize_rowwise_fp8(q)  # [sq, b, h, d] e4m3fn + [sq, b, h] fp32
    k_fp8, k_scale = quantize_rowwise_fp8(k)  # [sk, b, d] e4m3fn + [sk, b] fp32

    for bi in range(b):
        kb_fp8 = k_fp8[:, bi, :].reshape(sk, d)
        kb_scale = k_scale[:, bi].reshape(sk)
        # Hoisted ``[d, sk]`` rowwise-fp8 view of ``k[:, bi, :]`` for the dq
        # GEMM below --- this tensor does not depend on ``hi`` so we quantise
        # once per batch and reuse across all heads. Same storage would be
        # possible via ``torch._scaled_mm``'s layout tricks but re-quantising
        # is simpler than strided-transpose gymnastics.
        kb_for_dq_fp8, kb_for_dq_scale = quantize_rowwise_fp8(
            k[:, bi, :].transpose(0, 1).contiguous()  # [d, sk]
        )
        g_il_bi = grad_index_scores_logits[bi]  # [sq, sk] fp32 (aliases bwd grad)
        for hi in range(h):
            # Step (1): FP8 forward GEMM (reproduces scores[:, bi, hi, :]).
            a_qh = q_fp8[:, bi, hi, :].reshape(sq, d)
            a_scale = q_scale[:, bi, hi].reshape(sq)
            logits = _scaled_mm_rowwise(a_qh, a_scale, kb_fp8, kb_scale)  # [sq, sk] fp32

            # Step (2): Produce relu_mask + relu(logits) on the fly.
            relu_mask = logits > 0
            scores_after_relu = torch.where(relu_mask, logits, torch.zeros_like(logits))
            del logits

            # Step (3): grad_weights[:, bi, hi] = sum_sk(g_il * scores_after_relu)
            grad_weights_f32[:, bi, hi] = (g_il_bi * scores_after_relu).sum(dim=-1)
            del scores_after_relu

            # Step (4): Per-head grad_scores in the working buffer.
            w_h = weights[:, bi, hi].float().unsqueeze(-1)  # [sq, 1]
            grad_scores = g_il_bi * w_h * relu_mask.float()  # [sq, sk] fp32
            del relu_mask, w_h

            # Step (5): dq[:, bi, hi, :] += grad_scores @ k[bi]
            # Quantise grad_scores rowwise (per sq-row) to fp8, call _scaled_mm.
            # k[bi] is still fp8 from the outer quantisation.
            gs_fp8, gs_scale = quantize_rowwise_fp8(grad_scores)
            # FP8 GEMM: [sq, sk] @ [sk, d] -> [sq, d] fp32. _scaled_mm_rowwise
            # expects b as [N, K] row-major which it transposes internally;
            # we need [d, sk] (column-major view of [sk, d]) to produce [sq, d].
            # Reuse kb_fp8 [sk, d] by calling _scaled_mm_rowwise(gs_fp8, kb_fp8)
            # with K = sk.
            # _scaled_mm_rowwise(a [M, K], b [N, K]) -> [M, N]. Here M=sq, K=sk,
            # N=d. ``kb_for_dq_fp8`` was hoisted outside the hi loop above with
            # rowwise fp8 scales per ``d`` row (not per ``sk`` row like the
            # forward gemm's ``kb_fp8``) --- we can't share scales between the
            # two directions because rowwise absmax is layout-specific.
            dq = _scaled_mm_rowwise(gs_fp8, gs_scale, kb_for_dq_fp8, kb_for_dq_scale)
            grad_q_f32[:, bi, hi, :].add_(dq)
            del dq

            # Step (6): dk[:, bi, :] += grad_scores.t() @ q[:, bi, hi, :]
            # Here M = sk, K = sq, N = d. We need a_fp8 [sk, sq] and b_fp8
            # [d, sq]. Quantise grad_scores.t() rowwise over sq (so each of
            # sk rows has its own scale). And quantise q[:, bi, hi, :].t()
            # rowwise over sq for the d-rows.
            gs_t_fp8, gs_t_scale = quantize_rowwise_fp8(grad_scores.t().contiguous())  # [sk, sq]
            q_for_dk_fp8, q_for_dk_scale = quantize_rowwise_fp8(
                q[:, bi, hi, :].transpose(0, 1).contiguous()  # [d, sq]
            )  # fp8 [d, sq], scale [d]
            dk = _scaled_mm_rowwise(gs_t_fp8, gs_t_scale, q_for_dk_fp8, q_for_dk_scale)  # [sk, d]
            grad_k_f32[:, bi, :].add_(dk)
            del dk, gs_fp8, gs_scale, gs_t_fp8, gs_t_scale, q_for_dk_fp8, q_for_dk_scale
            del grad_scores

    # Free the big grad tensor once we are done reading it.
    del grad_index_scores_logits

    return (
        grad_q_f32.to(q.dtype),
        grad_weights_f32.to(weights.dtype),
        grad_k_f32.to(k.dtype),
    )


def bwd_fused_indexer_loss_bf16_reference(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    grad_loss: torch.Tensor,
    pg_collection=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local BF16 reference clone of Megatron's ``bwd_fused_indexer_loss_naive``.

    Byte-for-byte copy of the upstream (dsa.py, 2026-04-12) implementation,
    with ``pg_collection`` allowed to be ``None`` for unit tests. Kept next
    to the FP8 path so the backward-parity test does not need a Megatron
    checkout.
    """

    index_scores = compute_index_scores_bf16_reference(q, weights, k)  # [B, Sq, Sk]

    sq, b, np_, hn = query.size()
    sk = key.size(0)

    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np_, sq, hn)
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np_, hn, sk)
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    del query_reshaped, key_reshaped

    attention_scores = attention_scores.reshape(b, np_, sq, sk)

    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    index_scores = index_scores + causal_mask.unsqueeze(0)
    del causal_mask

    if sparse_loss:
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        index_scores = index_scores + index_mask

    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    del attention_scores
    index_scores_softmax = torch.nn.functional.softmax(
        index_scores, dim=-1, dtype=torch.float32
    )
    del index_scores
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    del attention_scores_softmax
    if pg_collection is not None and pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    del attention_scores_sum

    grad_kl_div = grad_loss * loss_coeff
    grad_kl_per_row = grad_kl_div / (b * sq)
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    del attention_scores_normalized

    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    causal_valid_mask = torch.tril(torch.ones((sq, sk), device=q.device, dtype=torch.bool))
    if sparse_loss:
        index_valid_mask = index_mask == 0
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask
        del index_valid_mask
    else:
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())  # [sq, b, h, sk]
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    grad_scores = grad_scores_after_relu * relu_mask.float()
    del grad_scores_after_relu, relu_mask

    grad_q = torch.einsum("sbht,tbd->sbhd", grad_scores, k.float())  # [sq, b, h, d]
    grad_k = torch.einsum("sbht,sbhd->tbd", grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)
