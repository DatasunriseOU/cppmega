from __future__ import annotations

import pytest
import torch

from cppmega.megatron.mxfp8_sidecar_refs import (
    MXFP8_COMPACT_DIRECT_ZERO_COUNTERS,
    compact_direct_counter_violations,
)


class _Dummy:
    pass


def test_clear_mxfp8_sidecar_refs_removes_all_producer_references(monkeypatch):
    for key in (
        "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER",
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)
    from cppmega.megatron.mxfp8_sidecar_refs import clear_mxfp8_sidecar_refs

    tensor = _Dummy()
    sidecar = _Dummy()

    tensor._te_gemm_ready_rowwise_transpose_for_backward = sidecar
    tensor._te_rowwise_transpose_for_backward = sidecar
    tensor._te_rowwise_transpose_for_backward_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose = sidecar
    tensor._cppmega_mxfp8_rowwise_transpose_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose_persistent = False

    assert clear_mxfp8_sidecar_refs(tensor)
    assert not hasattr(tensor, "_te_gemm_ready_rowwise_transpose_for_backward")
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward")
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_persistent")

    assert not clear_mxfp8_sidecar_refs(tensor)


def test_compact_direct_counter_contract_rejects_any_sidecar_or_bf16_bridge():
    stats = {key: 0 for key in MXFP8_COMPACT_DIRECT_ZERO_COUNTERS}
    assert compact_direct_counter_violations(stats) == []

    for key in MXFP8_COMPACT_DIRECT_ZERO_COUNTERS:
        violations = compact_direct_counter_violations({**stats, key: 1})
        assert violations == [f"{key}=1; expected 0 for compact_direct_v1"]


def test_compact_direct_counter_contract_rejects_non_integer_values():
    violations = compact_direct_counter_violations({"mxfp8_tn_sidecar_registry_peak": "live"})

    assert violations == ["mxfp8_tn_sidecar_registry_peak='live'; expected integer 0"]


def test_te_linear_consumed_sidecar_removes_producer_references(monkeypatch):
    for key in (
        "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER",
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "NVTE_BACKWARD_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)

    tex = pytest.importorskip("transformer_engine_torch")
    linear = pytest.importorskip("transformer_engine.pytorch.module.linear")
    storage_mod = pytest.importorskip(
        "transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage"
    )

    sidecar = storage_mod.MXFP8TensorStorage(
        rowwise_data=torch.empty((32, 32), dtype=torch.uint8),
        rowwise_scale_inv=torch.empty((128, 4), dtype=torch.uint8),
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=None,
        with_gemm_swizzled_scales=False,
        fake_dtype=torch.bfloat16,
    )
    tensor = _Dummy()
    unregister_calls = []

    def unregister(arg):
        unregister_calls.append(arg)

    tensor._te_rowwise_transpose_for_backward = sidecar
    tensor._te_rowwise_transpose_for_backward_unregister = unregister
    tensor._cppmega_mxfp8_rowwise_transpose = sidecar
    tensor._cppmega_mxfp8_rowwise_transpose_unregister = unregister
    tensor._cppmega_mxfp8_rowwise_transpose_persistent = False

    assert linear._get_rowwise_transpose_for_backward(tensor) is sidecar
    assert unregister_calls == [tensor]
    assert getattr(sidecar, "_te_rowwise_transpose_for_backward_operand", False)
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward")
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_persistent")
