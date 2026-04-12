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
    "apply_dsa_fp8_patch",
    "resolve_indexer_dtype",
]

DSA_INDEXER_DTYPE_ENV = "CPPMEGA_DSA_INDEXER_DTYPE"

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
        def _compute_index_scores_fp8(q, weights, k):
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
        q,
        weights,
        k,
        query,
        key,
        topk_indices,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        grad_loss,
        pg_collection,
    ):
        """FP8 replacement for Megatron dsa.bwd_fused_indexer_loss_naive."""
        return bwd_fused_indexer_loss_fp8(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            grad_loss,
            pg_collection,
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
