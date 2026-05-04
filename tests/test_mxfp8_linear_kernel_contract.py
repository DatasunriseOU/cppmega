from __future__ import annotations

import atexit
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


class _FakeMXFP8Tensor:
    def __init__(self, name: str):
        self.name = name
        self._rowwise_data = object()
        self._rowwise_scale_inv = object()
        self._columnwise_data = object()
        self._columnwise_scale_inv = object()
        self._with_gemm_swizzled_scales = False


def _fresh_shim(monkeypatch, *, contract: str):
    for key in (
        "CPPMEGA_TE_MXFP8_DGRAD_BF16",
        "CPPMEGA_TE_MXFP8_WGRAD_BF16",
        "CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD",
        "CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT",
        "NVTE_BACKWARD_OVERRIDE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_BWD_BACKEND", "flashinfer_cutlass")
    monkeypatch.setenv("CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT", contract)
    monkeypatch.setenv("CPPMEGA_TE_VERSION_STRICT", "0")
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: func)
    monkeypatch.delitem(sys.modules, "scripts.cppmega_fp8_shim", raising=False)
    try:
        shim = importlib.import_module("scripts.cppmega_fp8_shim")
    except Exception as exc:  # pragma: no cover - host dependency guard
        pytest.skip(
            "cppmega_fp8_shim MXFP8 path unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    if not hasattr(shim, "_cppmega_wrap_general_gemm"):
        pytest.skip("cppmega_fp8_shim MXFP8 dense wrapper was not installed")
    return shim


def _wrap_fake_gemm(shim):
    calls = []

    def general_gemm(*args, **kwargs):
        calls.append((args, kwargs))
        return "te-result"

    module = SimpleNamespace(general_gemm=general_gemm)
    assert shim._cppmega_wrap_general_gemm(module)
    return module, calls


def test_dense_gemm_ready_v1_consumes_compact_direct_without_copy(monkeypatch):
    shim = _fresh_shim(monkeypatch, contract="gemm_ready_v1")
    module, calls = _wrap_fake_gemm(shim)

    flashinfer = ModuleType("cppmega.megatron.flashinfer_mxfp8_gemm")
    flash_calls = []

    class CompactColumnwiseUnsupportedError(RuntimeError):
        pass

    def normalize_gemm_kwargs(**kwargs):
        return kwargs

    def dgrad_nn_gemm(weight, dy, **kwargs):
        flash_calls.append(("dgrad", weight, dy, kwargs))
        return "direct-dgrad"

    flashinfer.CompactColumnwiseUnsupportedError = CompactColumnwiseUnsupportedError
    flashinfer.normalize_gemm_kwargs = normalize_gemm_kwargs
    flashinfer.dgrad_nn_gemm = dgrad_nn_gemm
    monkeypatch.setattr(shim, "_cppmega_flashinfer_mxfp8_module", [flashinfer])
    monkeypatch.setattr(
        shim,
        "_cppmega_mxfp8_colwise_as_rowwise_transpose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strict compact path used copy-transpose")
        ),
    )

    weight = _FakeMXFP8Tensor("weight")
    dy = _FakeMXFP8Tensor("dy")
    result = module.general_gemm(weight, dy, layout="NN", grad=True)

    assert result == ("direct-dgrad", None, None, None)
    assert calls == []
    assert len(flash_calls) == 1
    op_kind, called_weight, called_dy, kwargs = flash_calls[0]
    assert (op_kind, called_weight, called_dy) == ("dgrad", weight, dy)
    assert kwargs["bias"] is None
    assert kwargs["gelu"] is False
    assert kwargs["quantization_params"] is None
    assert kwargs["accumulate"] is False
    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_flashinfer_dgrad"] == 1
    assert stats["mxfp8_linear_contract_v1_dense_dgrad"] == 1
    assert stats["mxfp8_tn_adapter_copy_transpose"] == 0


def test_dense_gemm_ready_v1_rejects_compact_direct_miss(monkeypatch):
    shim = _fresh_shim(monkeypatch, contract="gemm_ready_v1")
    module, calls = _wrap_fake_gemm(shim)

    flashinfer = ModuleType("cppmega.megatron.flashinfer_mxfp8_gemm")

    class CompactColumnwiseUnsupportedError(RuntimeError):
        pass

    def normalize_gemm_kwargs(**kwargs):
        return kwargs

    def dgrad_nn_gemm(*_args, **_kwargs):
        raise CompactColumnwiseUnsupportedError("shape requires materialized transpose")

    flashinfer.CompactColumnwiseUnsupportedError = CompactColumnwiseUnsupportedError
    flashinfer.normalize_gemm_kwargs = normalize_gemm_kwargs
    flashinfer.dgrad_nn_gemm = dgrad_nn_gemm
    monkeypatch.setattr(shim, "_cppmega_flashinfer_mxfp8_module", [flashinfer])

    with pytest.raises(RuntimeError, match="gemm_ready_v1 forbids copy-transpose"):
        module.general_gemm(
            _FakeMXFP8Tensor("weight"),
            _FakeMXFP8Tensor("dy"),
            layout="NN",
            grad=True,
        )

    assert calls == []
    stats = shim.cppmega_te_mxfp8_bwd_stats
    assert stats["mxfp8_linear_contract_v1_dense_miss_dgrad"] == 1
    assert stats["mxfp8_tn_adapter_copy_transpose"] == 0
