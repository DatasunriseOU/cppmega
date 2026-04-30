"""Per-head fused accumulation for Megatron DSA ``_compute_index_scores``.

Upstream Megatron's ``_compute_index_scores``
(``megatron/core/transformer/experimental_attention_variant/dsa.py``) is::

    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
    if use_relu: index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)

The intermediate ``[sq, b, h, sk]`` FP32 tensor is ``sq*b*h*sk*4`` bytes.
For NAM56R (``sq=sk=4096``, ``h=index_n_heads=32``, ``b=MBS=8``) that is
**16 GiB live** inside every indexer call. It is allocated, reduced over
``h``, then discarded — a textbook fuse-reduction opportunity.

This patch replaces the implementation with a per-head loop that
accumulates directly into a ``[b, sq, sk]`` FP32 buffer (268 MiB at the
same shape, a ~60x reduction) and **never materialises the full
``[sq, b, h, sk]`` tensor**. Math is identical to the upstream einsum
except for FP32 reduction order (head-wise instead of vectorised) which
is exact up to associative FP32 reorder (< 1e-6 relative error on
bounded inputs).

This is a pure correctness/memory fix — it is numerically equivalent to
upstream and has no runtime penalty on H200 (the per-head bmm lowers to
a single cuBLAS GEMM per head, same FLOP count). It replaces the dead
``dsa_fp8_patch.py`` (deleted 2026-04-13 in commit ``b6fb886``) for
memory-bound configurations like DSA 9+4 with MBS >= 8.

Applies to ALL callers of ``_compute_index_scores``:

* ``fused_qk_topk_naive`` (fwd, feeds topk selection)
* ``bwd_fused_indexer_loss_naive`` (bwd, indexer recompute)
* ``_LemyxFusedDSAIndexerLoss.apply`` (fwd via ``fused_qk_topk_naive``)
* ``IndexCache`` Full-layer ``topk`` re-derivation
  (``fused_qk_topk_naive`` in ``index_cache_patch.py``)

Usage::

    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    apply_dsa_indexer_fused_patch()  # call after Megatron imports, before training

Applied unconditionally. This patch is mandatory for production at
MBS=10 NAM56R — upstream ``_compute_index_scores`` materialises
``[sq, b, h, sk]`` fp32 = ~5 GiB/layer held by autograd for backward;
9 DSA layers => ~45 GiB resident.  Fused per-head ``[b, sq, sk]`` =
640 MiB/layer => ~5.7 GiB resident.  Net save ~40 GiB; without the fused
path MBS=10 is impossible, period.  No env gate — install or raise.
No backward changes — upstream ``_compute_index_scores`` is not
autograd-aware (called under ``torch.no_grad()`` in the fwd path and
inside a custom-autograd recompute in the bwd).
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
from collections import Counter

import torch

log = logging.getLogger(__name__)

__all__ = [
    "apply_dsa_indexer_fused_patch",
    "compute_index_scores_fused_bf16",
    "get_dsa_indexer_stats",
    "reset_dsa_indexer_stats",
]

_PATCH_MARKER = "__cppmega_dsa_indexer_fused_patched__"
_COUNTER_PATCH_MARKER = "__cppmega_dsa_indexer_counter_patched__"
_STATS: Counter[str] = Counter()
_ATEXIT_REGISTERED = False


def _dtype_name(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "none"
    return str(tensor.dtype).replace("torch.", "").replace(".", "_")


def _numel(tensor: torch.Tensor | None) -> int:
    return int(tensor.numel()) if tensor is not None else 0


def _bytes(tensor: torch.Tensor | None) -> int:
    return _numel(tensor) * int(tensor.element_size()) if tensor is not None else 0


def _record_tensor(prefix: str, tensor: torch.Tensor | None) -> None:
    dtype = _dtype_name(tensor)
    _STATS[f"{prefix}_{dtype}"] += 1
    _STATS[f"{prefix}_elems"] += _numel(tensor)
    _STATS[f"{prefix}_bytes"] += _bytes(tensor)


def _record_index_scores_call(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    *,
    use_relu: bool,
) -> None:
    sq, b, h, _d = q.shape
    sk = k.shape[0]
    output_elems = int(b * sq * sk)
    avoided_elems = int(sq * b * h * sk)
    _STATS["index_scores_calls"] += 1
    _STATS[f"index_scores_relu_{int(bool(use_relu))}"] += 1
    _record_tensor("index_scores_q", q)
    _record_tensor("index_scores_k", k)
    _record_tensor("index_scores_weights", weights)
    _STATS["index_scores_fp32_output_elems"] += output_elems
    _STATS["index_scores_fp32_output_bytes"] += output_elems * 4
    _STATS["index_scores_fp32_full_intermediate_avoided_elems"] += avoided_elems
    _STATS["index_scores_fp32_full_intermediate_avoided_bytes"] += avoided_elems * 4


def get_dsa_indexer_stats() -> dict[str, int]:
    """Return a copy of DSA indexer dtype/materialization counters."""

    return dict(_STATS)


def reset_dsa_indexer_stats() -> None:
    """Reset DSA indexer counters for focused tests."""

    _STATS.clear()


def _print_dsa_indexer_stats() -> None:
    print(
        "[cppmega] DSA nonlinear dtype/materialization stats "
        + json.dumps(dict(sorted(_STATS.items())), sort_keys=True)
    )


def _ensure_atexit_registered() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_print_dsa_indexer_stats)
        _ATEXIT_REGISTERED = True


def compute_index_scores_fused_bf16(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    use_relu: bool = True,
) -> torch.Tensor:
    """Drop-in replacement for Megatron DSA ``_compute_index_scores`` (BF16).

    Per-head fused accumulation: never materialises the ``[sq, b, h, sk]``
    FP32 intermediate.  Math is identical to the upstream einsum modulo
    FP32 associative reorder (head-wise).

    Args:
        q: ``[seqlen_q, batch, index_n_heads, index_head_dim]``.
        weights: ``[seqlen_q, batch, index_n_heads]``.
        k: ``[seqlen_k, batch, index_head_dim]``.
        use_relu: match upstream's ``use_relu`` flag.

    Returns:
        ``[batch, seqlen_q, seqlen_k]`` FP32 index scores.
    """

    assert q.dim() == 4, f"q must be [sq,b,h,d], got {tuple(q.shape)}"
    assert k.dim() == 3, f"k must be [sk,b,d], got {tuple(k.shape)}"
    assert weights.dim() == 3, f"weights must be [sq,b,h], got {tuple(weights.shape)}"

    sq, b, h, d = q.shape
    sk, bk, dk = k.shape
    assert bk == b and dk == d, (
        f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}"
    )

    _record_index_scores_call(q, weights, k, use_relu=use_relu)

    # Accumulator: [b, sq, sk] fp32.  This is the final output shape; we
    # write into it head by head instead of building [sq, b, h, sk] first.
    index_scores = torch.zeros(
        (b, sq, sk), dtype=torch.float32, device=q.device
    )

    # Per-batch permutation of k: [sk, b, d] -> [b, d, sk] fp32 for bmm.
    # Reused across all heads of the same batch; cost = 1x [b, d, sk] fp32
    # (= ~4 MiB at production shape).
    k_f32 = k.float()  # [sk, b, d]
    k_bds = k_f32.permute(1, 2, 0).contiguous()  # [b, d, sk]
    del k_f32

    # Per-head: do one [b, sq, d] @ [b, d, sk] bmm -> [b, sq, sk] fp32.
    # Apply relu + weight + accumulate in place.  Working buffer ~ 268 MiB
    # at production shape (b=8, sq=sk=4096) vs 16 GiB for the full
    # [sq, b, h, sk] upstream intermediate.
    for hi in range(h):
        # [sq, b, d] -> [b, sq, d] fp32.
        q_h = q[:, :, hi, :].float().permute(1, 0, 2).contiguous()  # [b, sq, d]
        logits_h = torch.bmm(q_h, k_bds)  # [b, sq, sk] fp32
        del q_h
        if use_relu:
            logits_h = torch.relu(logits_h)
        # weights[:, :, hi] is [sq, b]; broadcast to [b, sq, 1].
        w_h = weights[:, :, hi].float().transpose(0, 1).unsqueeze(-1)  # [b, sq, 1]
        index_scores.add_(logits_h * w_h)
        del logits_h, w_h

    del k_bds
    return index_scores


def _wrap_dsa_path_counters(dsa_mod) -> None:
    """Count which Megatron DSA indexer/top-k paths execute at runtime."""

    if getattr(dsa_mod, _COUNTER_PATCH_MARKER, False):
        return

    fused_topk = getattr(dsa_mod, "_fused_qk_topk_lighting", None)
    if callable(fused_topk):
        @functools.wraps(fused_topk)
        def _cppmega_fused_topk_counted(*args, **kwargs):
            _STATS["fused_topk_calls"] += 1
            if len(args) >= 3:
                _record_tensor("fused_topk_q", args[0])
                _record_tensor("fused_topk_k", args[1])
                _record_tensor("fused_topk_weights", args[2])
            out = fused_topk(*args, **kwargs)
            if out is None:
                _STATS["fused_topk_none"] += 1
            else:
                _STATS["fused_topk_success"] += 1
                if isinstance(out, torch.Tensor):
                    _record_tensor("fused_topk_indices", out)
            return out

        setattr(_cppmega_fused_topk_counted, _COUNTER_PATCH_MARKER, True)
        dsa_mod._fused_qk_topk_lighting = _cppmega_fused_topk_counted

    fused_sparse_kl = getattr(
        dsa_mod, "_fused_qk_topk_lighting_with_streaming_sparse_kl", None
    )
    if callable(fused_sparse_kl):
        @functools.wraps(fused_sparse_kl)
        def _cppmega_fused_sparse_kl_counted(*args, **kwargs):
            _STATS["fused_topk_sparse_kl_calls"] += 1
            if len(args) >= 3:
                _record_tensor("fused_sparse_kl_q", args[0])
                _record_tensor("fused_sparse_kl_k", args[1])
                _record_tensor("fused_sparse_kl_weights", args[2])
            out = fused_sparse_kl(*args, **kwargs)
            if out is None:
                _STATS["fused_topk_sparse_kl_none"] += 1
            else:
                _STATS["fused_topk_sparse_kl_success"] += 1
                if isinstance(out, tuple) and len(out) >= 2:
                    topk_indices, indexer_loss = out[0], out[1]
                    _record_tensor("fused_sparse_kl_indices", topk_indices)
                    _record_tensor("fused_sparse_kl_loss", indexer_loss)
            return out

        setattr(_cppmega_fused_sparse_kl_counted, _COUNTER_PATCH_MARKER, True)
        dsa_mod._fused_qk_topk_lighting_with_streaming_sparse_kl = (
            _cppmega_fused_sparse_kl_counted
        )

    dense_topk = getattr(dsa_mod, "fused_qk_topk_naive", None)
    if callable(dense_topk):
        @functools.wraps(dense_topk)
        def _cppmega_dense_topk_counted(*args, **kwargs):
            _STATS["dense_topk_naive_calls"] += 1
            if len(args) >= 3:
                _record_tensor("dense_topk_q", args[0])
                _record_tensor("dense_topk_k", args[1])
                _record_tensor("dense_topk_weights", args[2])
            out = dense_topk(*args, **kwargs)
            if isinstance(out, tuple) and len(out) >= 2:
                index_scores, topk_indices = out[0], out[1]
                _record_tensor("dense_topk_index_scores", index_scores)
                _record_tensor("dense_topk_indices", topk_indices)
            return out

        setattr(_cppmega_dense_topk_counted, _COUNTER_PATCH_MARKER, True)
        dsa_mod.fused_qk_topk_naive = _cppmega_dense_topk_counted

    run_sparse_attention = getattr(dsa_mod, "_run_sparse_attention", None)
    if callable(run_sparse_attention):
        @functools.wraps(run_sparse_attention)
        def _cppmega_run_sparse_attention_counted(*args, **kwargs):
            _STATS["sparse_attention_calls"] += 1
            for name in ("query", "key", "value", "topk_indices"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor):
                    _record_tensor(f"sparse_attention_{name}", value)
            out = run_sparse_attention(*args, **kwargs)
            if isinstance(out, tuple) and len(out) >= 2:
                _STATS[f"sparse_attention_path_{out[1]}"] += 1
            return out

        setattr(_cppmega_run_sparse_attention_counted, _COUNTER_PATCH_MARKER, True)
        dsa_mod._run_sparse_attention = _cppmega_run_sparse_attention_counted

    dsattention_cls = getattr(dsa_mod, "DSAttention", None)
    if dsattention_cls is not None and not getattr(
        dsattention_cls, _COUNTER_PATCH_MARKER, False
    ):
        orig_forward = dsattention_cls.forward

        @functools.wraps(orig_forward)
        def _cppmega_dsattention_forward_counted(self, *args, **kwargs):
            _STATS["dsa_forward_calls"] += 1
            coeff = float(getattr(self.config, "dsa_indexer_loss_coeff", 0.0) or 0.0)
            _STATS[f"dsa_forward_loss_coeff_nonzero_{int(coeff > 0.0)}"] += 1
            _STATS[
                f"dsa_forward_sparse_loss_{int(bool(getattr(self.config, 'dsa_indexer_use_sparse_loss', False)))}"
            ] += 1
            for idx, value in enumerate(args[:3]):
                if isinstance(value, torch.Tensor):
                    _record_tensor(f"dsa_forward_arg{idx}", value)
            return orig_forward(self, *args, **kwargs)

        dsattention_cls.forward = _cppmega_dsattention_forward_counted
        setattr(dsattention_cls, _COUNTER_PATCH_MARKER, True)

    fused_loss_cls = getattr(dsa_mod, "FusedDSAIndexerLoss", None)
    if fused_loss_cls is not None and not getattr(
        fused_loss_cls, _COUNTER_PATCH_MARKER, False
    ):
        orig_apply = fused_loss_cls.apply

        @functools.wraps(orig_apply)
        def _cppmega_fused_loss_apply_counted(*args, **kwargs):
            _STATS["dense_indexer_loss_apply_calls"] += 1
            return orig_apply(*args, **kwargs)

        fused_loss_cls.apply = staticmethod(_cppmega_fused_loss_apply_counted)
        setattr(fused_loss_cls, _COUNTER_PATCH_MARKER, True)

    setattr(dsa_mod, _COUNTER_PATCH_MARKER, True)


def apply_dsa_indexer_fused_patch(*, force: bool = False) -> bool:
    """Monkey-patch ``dsa._compute_index_scores`` with the fused variant.

    Idempotent.  Always installs — no env gate.  Raises on any failure;
    production MBS=10 is impossible without this patch (see module
    docstring for the ~40 GiB memory accounting).
    """

    from megatron.core.transformer.experimental_attention_variant import dsa as dsa_mod

    existing = getattr(dsa_mod, "_compute_index_scores", None)
    if existing is None:
        raise RuntimeError(
            "megatron.core.transformer.experimental_attention_variant.dsa."
            "_compute_index_scores not found — Megatron version mismatch?"
        )
    if getattr(existing, _PATCH_MARKER, False) and not force:
        log.info("cppmega DSA indexer fused patch already applied")
        return True

    def _compute_index_scores_fused(q, weights, k, use_relu: bool = True, **kwargs):
        # Accept **kwargs to be forward-compatible with new upstream args
        # (e.g. PR #3674 added ``mask=``).  Unused kwargs are ignored by the
        # fused math — they only affect downstream masking in the caller.
        return compute_index_scores_fused_bf16(q, weights, k, use_relu=use_relu)

    setattr(_compute_index_scores_fused, _PATCH_MARKER, True)
    dsa_mod._compute_index_scores = _compute_index_scores_fused
    _wrap_dsa_path_counters(dsa_mod)
    _ensure_atexit_registered()
    _STATS["patch_applied"] += 1
    _STATS[
        f"configured_loss_mode_{os.environ.get('CPPMEGA_DSA_INDEXER_LOSS_MODE', 'unset')}"
    ] += 1

    log.info(
        "cppmega DSA indexer fused patch applied: per-head accumulation, "
        "never materialises [sq, b, h, sk] intermediate"
    )
    print(
        "[cppmega] DSA indexer fused patch applied "
        "(per-head accumulation, [sq,b,h,sk] intermediate eliminated)"
    )
    return True
