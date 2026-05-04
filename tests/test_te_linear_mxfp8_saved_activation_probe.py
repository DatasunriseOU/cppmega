from __future__ import annotations

from tools.probes.te_linear_mxfp8_saved_activation_probe import (
    _validate_compact_direct_contract,
)


def test_probe_contract_accepts_compact_direct_zero_sidecars():
    stats = {
        "bf16_fallback_dgrad": 0,
        "bf16_fallback_wgrad": 0,
        "mxfp8_tn_adapter_te_emit": 0,
        "mxfp8_tn_adapter_copy_transpose": 0,
        "mxfp8_tn_sidecar_attr_attached": 0,
        "mxfp8_tn_sidecar_registry_peak": 0,
        "mxfp8_tn_sidecar_registry_peak_bytes": 0,
        "mxfp8_tn_adapter_saved_transpose_operand": 0,
        "mxfp8_tn_adapter_te_emit_deferred": 0,
        "mxfp8_cutlass_native_dgrad": 1,
        "mxfp8_cutlass_native_wgrad": 1,
    }

    failures = _validate_compact_direct_contract(
        stats,
        saved_transpose_payload=[],
        cutlass_native=True,
    )

    assert failures == []


def test_probe_contract_rejects_missing_compact_columnwise_consumers():
    stats = {
        "bf16_fallback_dgrad": 0,
        "bf16_fallback_wgrad": 0,
        "mxfp8_tn_adapter_copy_transpose": 0,
        "mxfp8_tn_sidecar_registry_peak": 0,
        "mxfp8_tn_adapter_saved_transpose_operand": 0,
        "mxfp8_cutlass_native_dgrad": 0,
        "mxfp8_cutlass_native_wgrad": 0,
    }

    failures = _validate_compact_direct_contract(
        stats,
        saved_transpose_payload=[],
        cutlass_native=True,
    )

    joined = "\n".join(failures)
    assert "CUTLASS native backend did not handle dgrad" in joined
    assert "CUTLASS native backend did not handle wgrad" in joined


def test_probe_contract_rejects_copy_sidecar_and_bf16_counters():
    stats = {
        "bf16_fallback_dgrad": 1,
        "bf16_fallback_wgrad": 1,
        "mxfp8_tn_adapter_copy_transpose": 1,
        "mxfp8_tn_sidecar_registry_peak": 1,
        "mxfp8_tn_adapter_saved_transpose_operand": 1,
        "mxfp8_cutlass_native_dgrad": 1,
        "mxfp8_cutlass_native_wgrad": 1,
    }

    failures = _validate_compact_direct_contract(
        stats,
        saved_transpose_payload=[{"dtype": "torch.uint8"}],
        cutlass_native=True,
    )

    joined = "\n".join(failures)
    assert "bf16_fallback_dgrad=1" in joined
    assert "bf16_fallback_wgrad=1" in joined
    assert "mxfp8_tn_adapter_copy_transpose=1" in joined
    assert "mxfp8_tn_sidecar_registry_peak=1" in joined
    assert "mxfp8_tn_adapter_saved_transpose_operand=1" in joined
    assert "saved rowwise-transposed payload" in joined
