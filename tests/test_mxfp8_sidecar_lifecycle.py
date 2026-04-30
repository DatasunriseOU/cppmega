from __future__ import annotations

import pytest
import torch


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
    from scripts.cppmega_fp8_shim import _cppmega_clear_mxfp8_sidecar_refs

    tensor = _Dummy()
    sidecar = _Dummy()

    tensor._te_rowwise_transpose_for_backward = sidecar
    tensor._te_rowwise_transpose_for_backward_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose = sidecar
    tensor._cppmega_mxfp8_rowwise_transpose_unregister = lambda _x: None
    tensor._cppmega_mxfp8_rowwise_transpose_persistent = False

    assert _cppmega_clear_mxfp8_sidecar_refs(tensor)
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward")
    assert not hasattr(tensor, "_te_rowwise_transpose_for_backward_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_unregister")
    assert not hasattr(tensor, "_cppmega_mxfp8_rowwise_transpose_persistent")

    assert not _cppmega_clear_mxfp8_sidecar_refs(tensor)


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
