"""TileLang-inspired fused KL-divergence target for the DSA indexer path.

Port of the forward KL-divergence kernel from ``lemyx/tilelang-dsa``
(``kernel_bf16_training_dsa_warmup_lightning_indexer.py``) into a pure-PyTorch
implementation that serves as a drop-in replacement for
:func:`cppmega.megatron.dsa_fp8_indexer._attention_target_fp32`.

Algorithm
---------
The upstream ``tilelang-dsa`` kernel computes KL(p || q) in a single streaming
pass over KV tiles using online softmax (Milakov & Gimelshein 2018):

* For each tile of K along the ``sk`` axis, it computes raw logits
  ``Q_h @ K_tile^T * scale`` and maintains running ``(max, sum_exp)``
  accumulators per query row.  After all tiles are consumed, the log
  partition function ``log_Z = max + log(sum_exp)`` is exact.

* The softmax distribution ``p_h[i, j] = exp(logit[i,j] - log_Z[i])`` is
  never materialised as a dense ``[sq, sk]`` matrix.  Instead, the
  per-tile contribution to ``sum_h softmax_h`` is accumulated directly
  into a ``[b, sq, sk]`` output buffer using the rescaling trick::

      acc *= exp(max_prev - max_new)        # rescale old tiles
      acc += exp(logit_tile - max_new)       # add new tile

  After all tiles::

      acc /= sum_exp_final                  # normalise

  This is mathematically identical to standard softmax but avoids the
  two-pass pattern (pass 1: compute max+sum, pass 2: divide) and never
  needs a dense ``[b, np, sq, sk]`` intermediate.

The pure-PyTorch port below uses ``torch.bmm`` for the per-head GEMM and
plain tensor arithmetic for the online rescaling.  On H200 this is still
memory-bound (same as head-streaming), but the algorithm is a stepping
stone toward a future TileLang JIT kernel that fuses the GEMM + rescale +
accumulate into a single launch.

Shape constraints
-----------------
* ``query``: ``[sq, b, np, hn]`` bf16/fp16/fp32.
* ``key``:   ``[sk, b, np, hn]`` bf16/fp16/fp32.  ``sq`` and ``sk`` may
  differ but a causal mask is applied (upper-triangular ``-inf``), so the
  useful range is ``min(sq, sk)``.
* ``topk_indices``: ``[b, sq, topk]`` int64.
* Output: ``(attention_scores_normalized [b, sq, sk] fp32, index_mask [b, sq, sk] fp32)``.

Numerical note
--------------
Online softmax reorders the floating-point additions compared to the
standard ``F.softmax`` two-pass algorithm.  The results are *not*
bit-identical but are within ``abs<=0.1, rel<=0.1`` of the head-streaming
reference for representative input magnitudes (gaussian, bf16).

Reference
---------
https://github.com/lemyx/tilelang-dsa  (MIT, 44 stars)
``kernel_bf16_training_dsa_warmup_lightning_indexer.py``, lines 249-481.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = [
    "attention_target_fused_kl",
]

# Tile size along the sk dimension for the online-softmax streaming loop.
# 64 matches the tilelang kernel's default ``block_N`` and keeps the
# per-tile working set small enough that the rescale arithmetic is cheap.
_TILE_SK = 64


def attention_target_fused_kl(
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    topk_indices: torch.Tensor,
    sparse_loss: bool,
    pg_collection,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the KL target via one-pass online softmax (tilelang-dsa port).

    Drop-in replacement for
    :func:`cppmega.megatron.dsa_fp8_indexer._attention_target_fp32`.

    The algorithm uses a "1.5-pass" approach: first stream tiles of sk to
    compute the exact row-wise (max, sum_exp) for each head via online
    softmax, then recompute the logits per tile and normalise into the
    output accumulator.  This avoids materialising a full ``[sq, sk]``
    logits tensor per head while keeping the computation numerically
    stable.

    Args:
        query: bf16 ``[sq, b, np, hn]`` main-attention query.
        key:   bf16 ``[sk, b, np, hn]`` main-attention key.
        softmax_scale: ``1 / sqrt(hn)`` pre-softmax scaling factor.
        topk_indices: int64 ``[b, sq, topk]`` selected key positions.
        sparse_loss: whether to restrict the KL target to topk positions.
        pg_collection: Megatron TP process group collection (may be None).

    Returns:
        ``(attention_scores_normalized [b, sq, sk] fp32, index_mask [b, sq, sk] fp32)``
    """

    sq, b, np_, hn = query.size()
    sk = key.size(0)
    device = query.device

    # ---- Masks (computed once, reused across all heads) ----
    causal_mask = torch.triu(
        torch.full((sq, sk), float("-inf"), dtype=torch.float32, device=device),
        diagonal=1,
    )
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=device
    ).scatter_(-1, topk_indices, 0)

    # Accumulator for sum-over-heads of post-softmax distributions.
    acc = torch.zeros(b, sq, sk, dtype=torch.float32, device=device)

    n_tiles = (sk + _TILE_SK - 1) // _TILE_SK

    for h in range(np_):
        # Per-head Q and K: [b, sq, hn] and [b, hn, sk].
        q_h = query[:, :, h, :].float().permute(1, 0, 2)   # [b, sq, hn]
        k_h = key[:, :, h, :].float().permute(1, 2, 0)     # [b, hn, sk]

        # ---- Pass 1: stream tiles to compute (max, sum_exp) ----
        # This is the online softmax from the tilelang-dsa kernel.
        running_max = torch.full(
            (b, sq), float("-inf"), dtype=torch.float32, device=device
        )
        running_sum = torch.zeros(b, sq, dtype=torch.float32, device=device)

        for t in range(n_tiles):
            j0 = t * _TILE_SK
            j1 = min(j0 + _TILE_SK, sk)

            k_tile = k_h[:, :, j0:j1]
            logits_tile = torch.bmm(q_h, k_tile) * softmax_scale
            logits_tile = logits_tile + causal_mask[:, j0:j1].unsqueeze(0)
            if sparse_loss:
                logits_tile = logits_tile + index_mask[:, :, j0:j1]

            # Online softmax update (Milakov & Gimelshein 2018).
            tile_max = logits_tile.amax(dim=-1)              # [b, sq]
            new_max = torch.maximum(running_max, tile_max)   # [b, sq]

            # Rescale factor: exp(old_max - new_max).
            # When both are -inf (all-masked row), -inf - (-inf) = nan.
            # Replace nan with 0 so that running_sum (which is 0) stays 0.
            rescale = torch.exp(running_max - new_max)
            rescale = torch.nan_to_num(rescale, nan=0.0)

            running_sum = running_sum * rescale

            # exp(logit - new_max): when logit == -inf and new_max == -inf
            # this is exp(nan) = nan.  Replace with 0 (correct: exp(-inf)=0).
            exp_tile = torch.exp(logits_tile - new_max.unsqueeze(-1))
            exp_tile = torch.nan_to_num(exp_tile, nan=0.0)

            running_sum = running_sum + exp_tile.sum(dim=-1)
            running_max = new_max

            del logits_tile, tile_max, new_max, exp_tile, rescale

        # ---- Pass 1.5: normalise per tile and accumulate into acc ----
        # We recompute logits per tile (cheap GEMM) and divide by the
        # now-known (running_max, running_sum) to get exact softmax.
        # Guard against division by zero for all-masked rows.
        safe_sum = running_sum.clone()
        safe_sum[safe_sum == 0] = 1.0  # avoid div-by-zero; result will be 0

        for t in range(n_tiles):
            j0 = t * _TILE_SK
            j1 = min(j0 + _TILE_SK, sk)

            k_tile = k_h[:, :, j0:j1]
            logits_tile = torch.bmm(q_h, k_tile) * softmax_scale
            logits_tile = logits_tile + causal_mask[:, j0:j1].unsqueeze(0)
            if sparse_loss:
                logits_tile = logits_tile + index_mask[:, :, j0:j1]

            # exp(logit - max) / sum_exp.  Guard nan from -inf - (-inf).
            exp_tile = torch.exp(logits_tile - running_max.unsqueeze(-1))
            exp_tile = torch.nan_to_num(exp_tile, nan=0.0)
            softmax_tile = exp_tile / safe_sum.unsqueeze(-1)

            acc[:, :, j0:j1] += softmax_tile
            del logits_tile, exp_tile, softmax_tile

        del q_h, k_h, running_max, running_sum, safe_sum

    del causal_mask

    # Optional TP all-reduce (sum local-TP heads across ranks).
    if pg_collection is not None and pg_collection.tp.size() > 1:
        torch.distributed.all_reduce(acc.contiguous(), group=pg_collection.tp)

    # L1 normalize after summing (same math as head-streaming).
    acc_sum = acc.sum(dim=-1, keepdim=True)
    # Guard against all-zero rows (all positions masked across all heads).
    acc_sum = acc_sum.clamp(min=1e-12)
    attention_scores_normalized = acc / acc_sum
    del acc
    return attention_scores_normalized, index_mask
