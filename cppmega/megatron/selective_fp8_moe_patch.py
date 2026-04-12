"""Monkey-patch Megatron's ``get_fp8_context`` for MoE-only FP8.

NAM56R has 52 layers with pattern AEMEAEMEAEMR (A=attention, E=MoE,
M=Mamba/Mamba3, R=M2RNN) tiled to depth 52.  FP8 on all layers is
37% slower because GEMMs are only 23.5% of total compute — Mamba scans
and attention are bandwidth-bound and pay FP8 casting overhead with no
upside.

MoE layers are pure expert FFN GEMMs where FP8 gives ~15% throughput
gain.  This patch makes ``get_fp8_context`` return ``nullcontext()``
for every non-MoE layer, so only MoE experts run in FP8.

Requirements:
    * ``--fp8-recipe tensorwise`` (NOT delayed — delayed shares a
      single context across all layers, defeating per-layer gating).
    * ``CPPMEGA_SELECTIVE_FP8_MOE=1`` environment variable.

Usage::

    from cppmega.megatron.selective_fp8_moe_patch import apply_selective_fp8_moe_patch
    apply_selective_fp8_moe_patch()  # call before MambaStack construction

Design follows the same monkey-patch idiom as ``dsa_fp8_patch.py``:
deferred imports, idempotent with sentinel marker, env-var gated.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Set

log = logging.getLogger(__name__)

__all__ = [
    "SELECTIVE_FP8_MOE_ENV",
    "apply_selective_fp8_moe_patch",
]

SELECTIVE_FP8_MOE_ENV = "CPPMEGA_SELECTIVE_FP8_MOE"

_PATCH_MARKER = "__cppmega_selective_fp8_moe_patched__"


def _compute_moe_layer_indices() -> Set[int]:
    """Return the set of 0-based layer indices that are MoE ('E') layers.

    Uses the same pattern/depth resolution as ``nam56r_layout.py`` so it
    respects ``CPPMEGA_NEM_PATTERN`` and ``CPPMEGA_LAYER_DEPTH`` overrides.
    """
    from cppmega.megatron.nam56r_layout import load_pattern
    from cppmega.recipes.nam56r_megatron import parse_nem_pattern

    pattern, depth = load_pattern()
    symbols = parse_nem_pattern(pattern, depth)
    return {i for i, sym in enumerate(symbols) if sym == "E"}


def apply_selective_fp8_moe_patch(*, force: bool = False) -> bool:
    """Monkey-patch ``megatron.core.fp8_utils.get_fp8_context`` for MoE-only FP8.

    Idempotent unless *force* is True.  Returns True if the patch was
    applied (or already present), False if the env var is not set.

    The patched function delegates to the original ``get_fp8_context`` for
    MoE layers and returns ``nullcontext()`` for everything else.  This
    means init-time contexts (``is_init=True``) for non-MoE layers also
    get ``nullcontext()``, so their parameters are allocated as BF16.
    """
    enabled = os.environ.get(SELECTIVE_FP8_MOE_ENV, "0").strip()
    if enabled != "1":
        log.info(
            "cppmega selective FP8 MoE patch skipped "
            "(set %s=1 to enable)", SELECTIVE_FP8_MOE_ENV,
        )
        return False

    import megatron.core.fp8_utils as fp8_mod

    original = getattr(fp8_mod, "get_fp8_context", None)
    if original is None:
        raise RuntimeError(
            "megatron.core.fp8_utils.get_fp8_context not found — "
            "Megatron version mismatch?"
        )

    if getattr(original, _PATCH_MARKER, False) and not force:
        log.info("cppmega selective FP8 MoE patch already applied")
        return True

    moe_indices = _compute_moe_layer_indices()

    # Build human-readable summary for the log.
    from cppmega.megatron.nam56r_layout import load_pattern
    from cppmega.recipes.nam56r_megatron import parse_nem_pattern

    pattern, depth = load_pattern()
    symbols = parse_nem_pattern(pattern, depth)
    sym_names = {"A": "attn", "M": "mamba", "E": "MoE", "R": "M2RNN", "D": "DSA", "G": "GDN"}

    fp8_layers = []
    bf16_layers = []
    for i, sym in enumerate(symbols):
        tag = f"L{i}({sym_names.get(sym, sym)})"
        if i in moe_indices:
            fp8_layers.append(tag)
        else:
            bf16_layers.append(tag)

    log.info(
        "cppmega selective FP8 MoE: %d/%d layers in FP8, %d in BF16",
        len(fp8_layers), len(symbols), len(bf16_layers),
    )
    log.info("  FP8  layers: %s", ", ".join(fp8_layers))
    log.info("  BF16 layers: %s", ", ".join(bf16_layers))

    def _selective_get_fp8_context(config, layer_no=-1, is_init=False):
        """Per-layer FP8 gate: FP8 for MoE, nullcontext for everything else."""
        if layer_no >= 0 and layer_no not in moe_indices:
            return nullcontext()
        return original(config, layer_no, is_init=is_init)

    setattr(_selective_get_fp8_context, _PATCH_MARKER, True)
    fp8_mod.get_fp8_context = _selective_get_fp8_context
    log.info("cppmega selective FP8 MoE patch applied (get_fp8_context monkey-patched)")
    return True
