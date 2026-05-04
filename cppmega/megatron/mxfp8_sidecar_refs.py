"""MXFP8 transpose sidecar reference helpers.

These helpers deliberately avoid Transformer Engine and Megatron imports so
tests and memory probes can validate sidecar lifecycle behavior without loading
the full runtime shim and its machine-specific monkey patches.
"""

from __future__ import annotations

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
            try:
                setattr(x, attr, None)
            except Exception:
                continue
        cleared = True
    return cleared
