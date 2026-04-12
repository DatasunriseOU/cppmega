"""Runtime monkey-patch for Megatron-LM DSA indexer to use the FP8 kernel.

The FP8 math lives in :mod:`cppmega.megatron.dsa_fp8_indexer`. This module
exposes an imperative ``apply_dsa_fp8_patch`` function that rebinds the
private ``_compute_index_scores`` in Megatron's
``experimental_attention_variant.dsa`` to the FP8 variant (only when
``CPPMEGA_DSA_INDEXER_DTYPE=fp8`` is set, or when ``config.dsa_indexer_dtype``
resolves to ``"fp8"`` on a live ``TransformerConfig``).

Call it from ``pretrain_mamba.py`` / ``pretrain_gpt.py`` launch wrappers
before the training loop starts — after the Megatron import side-effects
have already run. See ``scripts/remote_smoke_h200_dsa_fp8_indexer.sh`` for a
working example.

Design notes
------------

We deliberately avoid editing Megatron's ``dsa.py`` at file level for three
reasons:

1. The bench3 Megatron checkout is shared with Stream B / D runs; an
   in-place edit would break their BF16 baselines.
2. The patch must remain idempotent so repeated test invocations in the
   same Python process (e.g. pytest) do not double-wrap the function.
3. Monkey-patching at module level means the existing
   ``FusedDSAIndexerLoss`` autograd.Function picks up the new forward +
   backward path without any autograd changes — the recompute-based
   backward still calls ``_compute_index_scores`` exactly once.

The environment variable is the primary switch because bench3's
``arguments.py`` + ``transformer_config.py`` do **not** currently parse a
``--dsa-indexer-dtype`` flag. A companion helper
``add_dsa_indexer_dtype_arg`` patches the argparse group (same technique as
the existing DSA rope-fusion patch in
``scripts/remote_smoke_h200_dsa_full_nam56r.sh``) so the flag round-trips
from the cppmega launcher recipes into a real config field; the monkey
patch reads either source.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

__all__ = [
    "DSA_INDEXER_DTYPE_ENV",
    "DSA_KL_MODE_ENV",
    "DSA_SPARSE_MODE_ENV",
    "apply_dsa_fp8_patch",
    "apply_dsa_kl_mode_patch",
    "resolve_indexer_dtype",
    "resolve_kl_mode",
    "resolve_sparse_mode",
]

DSA_INDEXER_DTYPE_ENV = "CPPMEGA_DSA_INDEXER_DTYPE"
DSA_KL_MODE_ENV = "CPPMEGA_DSA_KL_MODE"
DSA_SPARSE_MODE_ENV = "CPPMEGA_DSA_SPARSE_MODE"

# Sentinel stored on the patched function so repeated calls are no-ops.
_PATCH_MARKER = "__cppmega_dsa_fp8_patched__"
# Separate sentinel for the backward patch so fwd + bwd patches remain
# independent (e.g. unit tests can flip one without the other).
_BWD_PATCH_MARKER = "__cppmega_dsa_fp8_bwd_patched__"


def resolve_indexer_dtype(config: Optional[object] = None) -> str:
    """Return the desired DSA indexer compute dtype.

    Resolution order (first hit wins):
    1. ``config.dsa_indexer_dtype`` if provided and non-None.
    2. ``CPPMEGA_DSA_INDEXER_DTYPE`` environment variable.
    3. ``"bf16"`` (current Megatron default, no behaviour change).
    """

    if config is not None:
        cfg_val = getattr(config, "dsa_indexer_dtype", None)
        if isinstance(cfg_val, str) and cfg_val:
            return cfg_val.lower()
    env_val = os.environ.get(DSA_INDEXER_DTYPE_ENV, "").strip().lower()
    if env_val:
        return env_val
    return "bf16"


def resolve_kl_mode() -> str:
    """Return the desired DSA KL-loss computation mode.

    Values:
    * ``"head_streaming"`` (default) — existing head-streaming path from
      Stream M tier 3 (commit 563fcb0).
    * ``"splitk"`` — Triton split-K recomputation ported from upstream
      PR #4039. Saves ~60% peak memory by avoiding the full
      ``[b*np, sq, sk]`` attention_scores materialisation.
    * ``"tilelang_fused"`` — one-pass online-softmax port of the
      lemyx/tilelang-dsa forward KL kernel. Uses tiled streaming along
      sk with running (max, sum_exp) accumulators. Pure PyTorch, no
      TileLang JIT dependency. Numerically close but not bit-identical
      to ``head_streaming`` due to reordered FP arithmetic.
    """
    val = os.environ.get(DSA_KL_MODE_ENV, "").strip().lower()
    if val in ("splitk", "split_k", "split-k"):
        return "splitk"
    if val in ("tilelang_fused", "tilelang-fused", "tilelang"):
        return "tilelang_fused"
    # default
    return "head_streaming"


def resolve_sparse_mode() -> str:
    """Return the desired DSA sparse attention mode.

    Values:
    * ``"tilelang"`` (default) — fused TileLang sparse MLA kernel ported
      from Megatron-LM PR #3674. Requires TileLang JIT (already installed
      via mamba_ssm). Uses online softmax in shared memory, avoids full
      attention score materialisation. Near-zero extra memory. Requires
      topk % 64 == 0 (production default topk=256 satisfies this).
    * ``"gather_scatter"`` — PyTorch gather-scatter fallback in
      ``dsa_sparse_attention.py``. No TileLang JIT dependency. Works on
      any GPU. Used automatically if TileLang import fails.
    """
    val = os.environ.get(DSA_SPARSE_MODE_ENV, "").strip().lower()
    if val in ("gather_scatter", "gather-scatter", "pytorch"):
        return "gather_scatter"
    # default: tilelang (falls back to gather_scatter on import error)
    return "tilelang"


# Sentinel for the tilelang_fused KL mode monkey-patch.
_KL_FUSED_PATCH_MARKER = "__cppmega_dsa_kl_tilelang_fused_patched__"


def apply_dsa_kl_mode_patch(*, force: bool = False) -> bool:
    """Monkey-patch ``_attention_target_fp32`` based on ``CPPMEGA_DSA_KL_MODE``.

    When ``resolve_kl_mode()`` returns ``"tilelang_fused"``, replaces
    :func:`cppmega.megatron.dsa_fp8_indexer._attention_target_fp32` with the
    one-pass online-softmax variant from
    :func:`cppmega.megatron.dsa_tilelang_fused_kl.attention_target_fused_kl`.

    Idempotent unless ``force`` is True. Returns True if the patch was
    applied (or already present), False if the mode is not tilelang_fused.
    """

    kl_mode = resolve_kl_mode()
    if kl_mode != "tilelang_fused":
        log.info("cppmega DSA KL mode patch skipped (mode=%s)", kl_mode)
        return False

    import cppmega.megatron.dsa_fp8_indexer as indexer_mod
    from cppmega.megatron.dsa_tilelang_fused_kl import attention_target_fused_kl

    existing = getattr(indexer_mod, "_attention_target_fp32", None)
    if existing is None:
        raise RuntimeError(
            "cppmega.megatron.dsa_fp8_indexer._attention_target_fp32 not found"
        )
    already = getattr(existing, _KL_FUSED_PATCH_MARKER, False)
    if already and not force:
        log.info("cppmega DSA KL tilelang_fused patch already applied")
        return True

    def _attention_target_tilelang_fused(
        query, key, softmax_scale, topk_indices, sparse_loss, pg_collection
    ):
        return attention_target_fused_kl(
            query, key, softmax_scale, topk_indices, sparse_loss, pg_collection
        )

    setattr(_attention_target_tilelang_fused, _KL_FUSED_PATCH_MARKER, True)
    indexer_mod._attention_target_fp32 = _attention_target_tilelang_fused
    log.info(
        "cppmega DSA KL tilelang_fused patch applied "
        "(online-softmax one-pass, ported from lemyx/tilelang-dsa)"
    )
    return True


def apply_dsa_fp8_patch(*, force: bool = False) -> bool:
    """Monkey-patch Megatron's DSA ``_compute_index_scores`` to FP8.

    Idempotent: calling multiple times only patches once unless ``force``
    is True. Returns ``True`` if the patch was applied (or already present),
    ``False`` if the requested dtype is bf16 and no patch is needed.

    Raises ``ValueError`` if an unknown dtype is requested.
    """

    dtype = resolve_indexer_dtype(None)
    if dtype == "bf16":
        log.info("cppmega DSA indexer patch skipped (dtype=bf16)")
        return False
    if dtype != "fp8":
        raise ValueError(
            f"Unknown CPPMEGA_DSA_INDEXER_DTYPE / dsa_indexer_dtype: {dtype!r} "
            "(expected 'bf16' or 'fp8')"
        )

    # Deferred import so module import doesn't pull in Megatron outside
    # the remote launcher context (keeps pytest importable on laptop).
    from megatron.core.transformer.experimental_attention_variant import dsa as dsa_mod

    from cppmega.megatron.dsa_fp8_indexer import (
        bwd_fused_indexer_loss_fp8,
        compute_index_scores_fp8,
    )

    existing = getattr(dsa_mod, "_compute_index_scores", None)
    if existing is None:
        raise RuntimeError(
            "megatron.core.transformer.experimental_attention_variant.dsa."
            "_compute_index_scores not found — Megatron version mismatch?"
        )
    fwd_already = getattr(existing, _PATCH_MARKER, False)

    if fwd_already and not force:
        log.info("cppmega DSA FP8 forward patch already applied")
    else:
        def _compute_index_scores_fp8(q, weights, k, **kwargs):
            """FP8 replacement for Megatron dsa._compute_index_scores."""
            return compute_index_scores_fp8(q, weights, k)

        setattr(_compute_index_scores_fp8, _PATCH_MARKER, True)
        dsa_mod._compute_index_scores = _compute_index_scores_fp8
        log.info("cppmega DSA FP8 forward patch applied (dtype=fp8)")

    # ------------------------------------------------------------------
    # Backward patch (Stream G, task #84). Replaces
    # ``bwd_fused_indexer_loss_naive`` with an FP8 variant that mirrors the
    # upstream math but routes the indexer recompute + grad_q / grad_k
    # einsums through ``torch._scaled_mm`` and per-head fused accumulation.
    # The main-attention Q @ K^T bmm is left BF16/FP32 because its output
    # is structurally required across np before softmax (see module
    # docstring of ``dsa_fp8_indexer``).
    #
    # CRITICAL: this must be applied BEFORE any training process imports
    # ``FusedDSAIndexerLoss`` because ``FusedDSAIndexerLoss.backward``
    # holds a direct reference to the function at class-definition time.
    # We therefore monkey-patch the module-level symbol AND also rebind
    # ``FusedDSAIndexerLoss.backward`` if the class has been imported.
    # ------------------------------------------------------------------
    existing_bwd = getattr(dsa_mod, "bwd_fused_indexer_loss_naive", None)
    if existing_bwd is None:
        raise RuntimeError(
            "megatron.core.transformer.experimental_attention_variant.dsa."
            "bwd_fused_indexer_loss_naive not found — Megatron version mismatch?"
        )
    bwd_already = getattr(existing_bwd, _BWD_PATCH_MARKER, False)
    if bwd_already and not force:
        log.info("cppmega DSA FP8 backward patch already applied")
        return True

    def _bwd_fused_indexer_loss_fp8(
        q, weights, k, query, key, topk_indices, softmax_scale,
        loss_coeff, sparse_loss, *rest_args, **rest_kwargs,
    ):
        """FP8 replacement for Megatron dsa.bwd_fused_indexer_loss_naive.

        PR #3674 added ``mask`` as a new positional arg before ``grad_loss``.
        We accept *rest_args to handle both old (grad_loss, pg_collection)
        and new (mask, grad_loss, pg_collection, ...) signatures.
        """
        # Extract grad_loss and pg_collection from the tail args.
        # Old: rest_args = (grad_loss, pg_collection)
        # New: rest_args = (mask, grad_loss, pg_collection, ...)
        if len(rest_args) >= 3:
            # New signature: mask, grad_loss, pg_collection, ...
            grad_loss = rest_args[1]
            pg_collection = rest_args[2]
        elif len(rest_args) == 2:
            # Old signature: grad_loss, pg_collection
            grad_loss = rest_args[0]
            pg_collection = rest_args[1]
        else:
            grad_loss = rest_kwargs.get("grad_loss", None)
            pg_collection = rest_kwargs.get("pg_collection", None)

        # If indexer loss was skipped in forward, return zero gradients
        if os.environ.get("CPPMEGA_DSA_SKIP_INDEXER_LOSS", "0") == "1":
            return (
                torch.zeros_like(q),
                torch.zeros_like(weights),
                torch.zeros_like(k),
            )

        return bwd_fused_indexer_loss_fp8(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss,
            grad_loss, pg_collection,
        )

    setattr(_bwd_fused_indexer_loss_fp8, _BWD_PATCH_MARKER, True)
    dsa_mod.bwd_fused_indexer_loss_naive = _bwd_fused_indexer_loss_fp8

    # The FusedDSAIndexerLoss.backward staticmethod holds a bound reference
    # via closure-less lookup: it calls ``bwd_fused_indexer_loss_naive(...)``
    # resolved at call time from the dsa module global, so rebinding the
    # module global is sufficient --- no class-level edit needed. We
    # verified this by reading dsa.py on bench3 (2026-04-12): the
    # staticmethod body does a plain ``bwd_fused_indexer_loss_naive(...)``
    # lookup, not a captured reference.
    log.info("cppmega DSA FP8 backward patch applied (dtype=fp8)")

    # ------------------------------------------------------------------
    # Stream M: three-tier memory optimization for DSA indexer forward
    #
    # Tier 1 (DSAttention.forward gate): when loss_coeff==0 during
    #   training, temporarily switch DSAttention to eval mode so the
    #   base forward takes the inference indexer path (no-grad topk
    #   only). Bypasses FusedDSAIndexerLoss.apply() entirely, saving
    #   183 MiB/layer of backward-saved tensors * 9 DSA layers = 1.6 GiB,
    #   plus the 63 GiB of KL loss computation, plus the backward
    #   recompute overhead. Critical for fitting DSA 9+4 in 140 GiB.
    #
    # Tier 2 (compute_dsa_indexer_loss gate): when loss_coeff==0, return
    #   torch.zeros(()) immediately. Redundant with tier 1 but kept as
    #   safety net in case tier 1 is ever bypassed.
    #
    # Tier 3 (head-streaming): when loss_coeff>0, _attention_target_fp32
    #   now loops over heads instead of materializing [b*np, sq, sk].
    #   Saves 7.5 GiB -> 0.8 GiB per DSA layer forward at production
    #   shape. Built into dsa_fp8_indexer.py; no additional monkey-patch
    #   needed.
    # ------------------------------------------------------------------
    import torch

    # -- Tier 1: DSAttention.forward gate --
    # When loss_coeff==0 during training, bypass FusedDSAIndexerLoss.apply()
    # entirely and use a no-grad topk-only path. This eliminates:
    # - FusedDSAIndexerLoss ctx.save_for_backward (183 MiB per DSA layer)
    # - compute_dsa_indexer_loss (7.5 GiB per DSA layer)
    # - bwd_fused_indexer_loss_naive backward computation + allocations
    # - DSAIndexerLossAutoScaler overhead
    # The index_scores tensor is freed immediately after topk selection.
    # Total saving with 9 DSA layers: ~1.6 GiB saved-tensors + 63 GiB loss.
    _DSA_FWD_GATE_MARKER = "__cppmega_dsattn_fwd_gate_patched__"
    _DSAttention = getattr(dsa_mod, "DSAttention", None)
    _unfused_dsa_fn = getattr(dsa_mod, "unfused_dsa_fn", None)
    if (
        _DSAttention is not None
        and _unfused_dsa_fn is not None
        and not getattr(_DSAttention, _DSA_FWD_GATE_MARKER, False)
    ):
        _orig_dsattn_forward = _DSAttention.forward

        def _gated_dsattn_forward(self, *args, **kwargs):
            # Peek at loss_coeff: if 0 during training, use inference-style
            # indexer path to avoid all indexer-loss autograd overhead.
            loss_coeff = getattr(self.config, "dsa_indexer_loss_coeff", 0.0)
            if not (self.training and torch.is_grad_enabled() and loss_coeff == 0.0):
                # Non-zero loss_coeff or not training: use original path.
                return _orig_dsattn_forward(self, *args, **kwargs)

            # -- loss_coeff==0 fast path --
            # Temporarily switch to eval mode so the base forward takes the
            # inference branch (no-grad topk + no FusedDSAIndexerLoss +
            # no DSAIndexerLossAutoScaler). Gradients still flow through
            # unfused_dsa_fn because torch.is_grad_enabled() is True; only
            # the indexer-loss autograd machinery is bypassed.
            # DSAttention.forward has no dropout/batchnorm, so eval mode
            # is safe here.
            self.eval()
            try:
                output = _orig_dsattn_forward(self, *args, **kwargs)
            finally:
                self.train()
            return output

        _DSAttention.forward = _gated_dsattn_forward
        setattr(_DSAttention, _DSA_FWD_GATE_MARKER, True)
        log.info(
            "cppmega DSAttention.forward gate applied "
            "(loss_coeff==0: eval indexer path, skips "
            "FusedDSAIndexerLoss + DSAIndexerLossAutoScaler entirely)"
        )

    # -- Tier 2: compute_dsa_indexer_loss gate (safety net) --
    # Also routes to split-K Triton kernel when CPPMEGA_DSA_KL_MODE=splitk.
    _LOSS_GATE_MARKER = "__cppmega_dsa_loss_gate_patched__"
    existing_loss = getattr(dsa_mod, "compute_dsa_indexer_loss", None)
    if existing_loss is not None and not getattr(existing_loss, _LOSS_GATE_MARKER, False):
        _orig_loss = existing_loss
        _kl_mode = resolve_kl_mode()

        if _kl_mode == "splitk":
            from cppmega.megatron.dsa_splitk_indexer_loss import (
                compute_dsa_indexer_loss_splitk,
            )

        def _gated_compute_dsa_indexer_loss(
            index_scores, topk_indices, query, key,
            softmax_scale, loss_coeff, sparse_loss=False, pg_collection=None,
            **kwargs,
        ):
            if loss_coeff == 0.0:
                return torch.zeros((), device=query.device, dtype=torch.float32)
            # Split-K mode: use Triton recomputation kernel (saves ~60%
            # peak memory by avoiding full [b*np, sq, sk] materialisation).
            # Falls back to head_streaming when TP > 1 because the Triton
            # kernel does not fuse the TP all-reduce.
            if _kl_mode == "splitk":
                tp_size = getattr(
                    getattr(pg_collection, "tp", None), "size", lambda: 1
                )()
                if tp_size <= 1:
                    return compute_dsa_indexer_loss_splitk(
                        index_scores, topk_indices, query, key,
                        softmax_scale, loss_coeff, sparse_loss, pg_collection,
                    )
                # TP > 1: fall through to head_streaming
            # loss_coeff > 0: upstream compute_dsa_indexer_loss does a full
            # torch.bmm(query.float(), key.float()) = 7 GiB at sq=sk=4096.
            # Force loss_coeff=0 gate when CPPMEGA_DSA_SKIP_INDEXER_LOSS=1
            # (temporary memory workaround — disables indexer loss training).
            if os.environ.get("CPPMEGA_DSA_SKIP_INDEXER_LOSS", "0") == "1":
                return torch.zeros((), device=query.device, dtype=torch.float32)
            return _orig_loss(
                index_scores, topk_indices, query, key,
                softmax_scale, loss_coeff, sparse_loss, pg_collection,
            )

        setattr(_gated_compute_dsa_indexer_loss, _LOSS_GATE_MARKER, True)
        dsa_mod.compute_dsa_indexer_loss = _gated_compute_dsa_indexer_loss
        if _kl_mode == "splitk":
            log.info(
                "cppmega DSA loss gate + split-K Triton applied "
                "(loss_coeff==0: skip; loss_coeff>0: split-K recompute, "
                "~60%% memory saving vs full materialisation)"
            )
        else:
            log.info(
                "cppmega DSA loss gate + head-streaming applied "
                "(loss_coeff==0: skip; loss_coeff>0: 0.8 GiB instead of 7.5 GiB per DSA layer)"
            )

    # ------------------------------------------------------------------
    # Tier 4: Sparse DSA attention
    #
    # unfused_dsa_fn (dsa.py:920) materializes FULL [b*np, sq, sk] FP32
    # attention scores = 7.0 GiB per DSA layer at production shape, then
    # masks non-topk to -inf before softmax.  This is the REAL memory
    # bottleneck for DSA 9+4 on H200 (5 layers × 7 GiB = 35 GiB extra).
    #
    # Two modes (controlled by CPPMEGA_DSA_SPARSE_MODE env var):
    #
    # "gather_scatter" (default): PyTorch gather-scatter implementation.
    #   sparse_dsa_fn gathers only topk K/V entries per query, computing
    #   [b, np, sq, topk=16] scores = 28.7 MB instead of 7 GiB.
    #   ~250× reduction in attention scores tensor.
    #
    # "tilelang": TileLang fused sparse MLA from Megatron-LM PR #3674.
    #   Uses online softmax in shared memory with LRU kernel caching.
    #   Requires TileLang JIT. Falls back to gather_scatter on import error.
    # ------------------------------------------------------------------
    _SPARSE_DSA_MARKER = "__cppmega_sparse_dsa_patched__"
    existing_unfused = getattr(dsa_mod, "unfused_dsa_fn", None)
    if existing_unfused is not None and not getattr(existing_unfused, _SPARSE_DSA_MARKER, False):
        sparse_mode = resolve_sparse_mode()

        if sparse_mode == "tilelang":
            try:
                from cppmega.megatron.sparse_mla_ops.sparse_mla import (
                    sparse_mla_as_unfused_dsa,
                )

                setattr(sparse_mla_as_unfused_dsa, _SPARSE_DSA_MARKER, True)
                dsa_mod.unfused_dsa_fn = sparse_mla_as_unfused_dsa
                log.info(
                    "cppmega TileLang SparseMLA applied (replaces unfused_dsa_fn: "
                    "fused online-softmax sparse attention from Megatron-LM PR #3674)"
                )
            except Exception as exc:
                log.warning(
                    "cppmega TileLang SparseMLA import failed (%s), "
                    "falling back to PyTorch gather-scatter sparse_dsa_fn",
                    exc,
                )
                sparse_mode = "gather_scatter"

        if sparse_mode == "gather_scatter":
            from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn

            setattr(sparse_dsa_fn, _SPARSE_DSA_MARKER, True)
            dsa_mod.unfused_dsa_fn = sparse_dsa_fn
            log.info(
                "cppmega sparse_dsa_fn applied (replaces unfused_dsa_fn: "
                "7.0 GiB → 28.7 MB attention scores per DSA layer, ~250× reduction)"
            )

    # ------------------------------------------------------------------
    # Tier 5: Sparse absorbed-MLA DSA attention
    #
    # PR #3674 adds _unfused_absorbed_dsa_fn which does full dense matmul
    # torch.matmul(q.float(), k.float()) = 7 GiB at [b, np, sq, sk].
    # Replace with sparse gather-scatter that only computes topk entries.
    # ------------------------------------------------------------------
    _unfused_absorbed = getattr(dsa_mod, "_unfused_absorbed_dsa_fn", None)
    if _unfused_absorbed is not None:
        from cppmega.megatron.dsa_sparse_absorbed import sparse_absorbed_dsa_fn

        dsa_mod._unfused_absorbed_dsa_fn = sparse_absorbed_dsa_fn
        log.info(
            "cppmega sparse_absorbed_dsa_fn applied (replaces _unfused_absorbed_dsa_fn: "
            "7.0 GiB → 0.44 GiB attention scores per absorbed DSA layer, ~16× reduction)"
        )

    # -- KL mode patch: tilelang_fused online-softmax --
    # When CPPMEGA_DSA_KL_MODE=tilelang_fused, replace _attention_target_fp32
    # in the dsa_fp8_indexer module with the one-pass online-softmax variant
    # ported from lemyx/tilelang-dsa. This is independent of the FP8 forward/
    # backward patches above and applies to the KL target computation only.
    apply_dsa_kl_mode_patch(force=force)

    return True


def add_dsa_indexer_dtype_arg(parser) -> None:
    """Register ``--dsa-indexer-dtype`` on an argparse parser (idempotent).

    Intended to be called from the ``_add_experimental_attention_variant_args``
    patch in the remote launcher scripts. Exposed here so it can also be
    unit-tested.
    """

    existing = {
        action.option_strings[0]
        for action in parser._actions
        if action.option_strings
    }
    if "--dsa-indexer-dtype" in existing:
        return
    group = None
    # Try to find the existing experimental_attention_variant group so the
    # flag lands in a sensible --help section.
    for g in getattr(parser, "_action_groups", []):
        if g.title == "experimental_attention_variant":
            group = g
            break
    if group is None:
        group = parser
    group.add_argument(
        "--dsa-indexer-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8"],
        help=(
            "Compute dtype for the DSA indexer's q@k^T path. "
            "'bf16' (default) matches upstream Megatron. 'fp8' routes "
            "through cppmega.megatron.dsa_fp8_indexer (FP8 _scaled_mm "
            "rowwise, ported from DeepSeek V3.2 inference kernel)."
        ),
    )
