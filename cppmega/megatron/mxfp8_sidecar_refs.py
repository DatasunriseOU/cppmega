"""MXFP8 transpose sidecar reference helpers.

These helpers deliberately avoid Transformer Engine and Megatron imports so
tests and memory probes can validate sidecar lifecycle behavior without loading
the full runtime shim and its machine-specific monkey patches.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

MXFP8_TN_SIDECAR_ATTR = "_cppmega_mxfp8_rowwise_transpose"
MXFP8_TN_SIDECAR_PERSISTENT_ATTR = "_cppmega_mxfp8_rowwise_transpose_persistent"
MXFP8_TN_SIDECAR_REF_ATTRS = (
    "_te_gemm_ready_rowwise_transpose_for_backward",
    "_te_rowwise_transpose_for_backward",
    "_te_rowwise_transpose_for_backward_unregister",
    MXFP8_TN_SIDECAR_ATTR,
    "_cppmega_mxfp8_rowwise_transpose_unregister",
    MXFP8_TN_SIDECAR_PERSISTENT_ATTR,
)

def clear_mxfp8_sidecar_refs(x: object) -> bool:
    """Drop producer-side references to a consumed MXFP8 transpose sidecar."""

    cleared = False
    for attr in MXFP8_TN_SIDECAR_REF_ATTRS:
        if not hasattr(x, attr):
            continue
        try:
            delattr(x, attr)
        except Exception:
            log.debug("delattr(%s) failed; trying setattr None", attr, exc_info=True)
            try:
                setattr(x, attr, None)
            except Exception:
                log.debug("setattr(%s, None) failed; leaving sidecar ref in place", attr, exc_info=True)
                continue
        cleared = True
    return cleared


# Counters that must all be zero under the compact_direct_v1 linear-kernel
# contract: no BF16 backward bridge, no TE-side transpose materialization, no
# live or lifetime sidecar-registry traffic. Mirrors the runtime stat keys used
# by tools/probes/gb10_accepted_path_validation_helpers.py.
MXFP8_COMPACT_DIRECT_ZERO_COUNTERS = (
    "bf16_fallback_dgrad",
    "bf16_fallback_wgrad",
    "mxfp8_tn_adapter_te_emit",
    "mxfp8_tn_adapter_te_emit_deferred",
    "mxfp8_tn_adapter_saved_transpose_operand",
    "mxfp8_tn_adapter_te_emit_swizzled",
    "mxfp8_tn_adapter_te_emit_swizzled_unavailable",
    "mxfp8_dense_grad_output_transpose_emit",
    "mxfp8_dense_grad_output_transpose_emit_failed",
    "mxfp8_tn_adapter_copy_transpose",
    "mxfp8_tn_adapter_missing_sidecar_copy",
    "mxfp8_tn_adapter_missing_sidecar_strict",
    "mxfp8_norm_quantize_sidecar_bridge",
    "mxfp8_tn_sidecar_registry_size",
    "mxfp8_tn_sidecar_registry_persistent",
    "mxfp8_tn_sidecar_registry_current_bytes",
    "mxfp8_tn_sidecar_tracked_attr_current_bytes",
    "mxfp8_tn_sidecar_registry_peak",
    "mxfp8_tn_sidecar_registry_peak_bytes",
    "mxfp8_tn_sidecar_tracked_attr_peak_bytes",
    "mxfp8_tn_sidecar_attr_attached",
    "mxfp8_tn_sidecar_attr_cleared",
    "mxfp8_tn_sidecar_consumed",
    "mxfp8_tn_sidecar_attr_attached_bytes",
)


def compact_direct_counter_violations(stats: dict) -> list[str]:
    """Return human-readable contract violations for compact_direct_v1.

    For every key in :data:`MXFP8_COMPACT_DIRECT_ZERO_COUNTERS` that is present
    in ``stats``, the value must be the integer ``0``. Non-integer values
    surface as separate violations so the caller can see the full picture in
    one pass.
    """

    violations: list[str] = []
    for key in MXFP8_COMPACT_DIRECT_ZERO_COUNTERS:
        if key not in stats:
            continue
        value = stats[key]
        if isinstance(value, bool) or not isinstance(value, int):
            violations.append(f"{key}={value!r}; expected integer 0")
            continue
        if value != 0:
            violations.append(f"{key}={value}; expected 0 for compact_direct_v1")
    return violations
