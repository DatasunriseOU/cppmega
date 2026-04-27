"""Tests for FlashInfer MXFP8 GEMM wrapper validation."""

from __future__ import annotations

import pytest
import torch

from cppmega.megatron import flashinfer_mxfp8_gemm


def _normalize(**overrides):
    kwargs = {
        "out_dtype": torch.bfloat16,
        "out": None,
        "bias": None,
        "gelu": False,
        "gelu_in": None,
        "quantization_params": None,
        "accumulate": False,
        "alpha": 1.0,
        "beta": None,
    }
    kwargs.update(overrides)
    return flashinfer_mxfp8_gemm.normalize_gemm_kwargs(**kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_normalize_gemm_kwargs_accepts_supported_output_dtype(dtype: torch.dtype) -> None:
    out = torch.empty((4, 4), dtype=dtype)

    normalized = _normalize(out_dtype=dtype, out=out)

    assert normalized["out"] is out
    assert normalized["out_dtype"] == dtype


def test_normalize_gemm_kwargs_accepts_accumulate_beta_zero_as_overwrite() -> None:
    normalized = _normalize(accumulate=True, beta=0.0)

    assert normalized["out_dtype"] == torch.bfloat16


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"out_dtype": torch.float32}, "BF16/FP16"),
        ({"bias": torch.empty(4)}, "does not fuse bias"),
        ({"gelu": True}, "does not fuse GELU"),
        ({"gelu_in": torch.empty(4)}, "does not fuse GELU"),
        ({"quantization_params": object()}, "does not quantize"),
        ({"accumulate": True}, "does not implement accumulate"),
        ({"alpha": 0.5}, "requires alpha=1.0"),
        ({"beta": 1.0}, "requires beta unset/0"),
    ],
)
def test_normalize_gemm_kwargs_rejects_unsupported_epilogues(overrides, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _normalize(**overrides)


def test_epilogue_capability_report_names_unsupported_contracts() -> None:
    report = flashinfer_mxfp8_gemm.epilogue_capability_report()

    assert "out_dtype" in report["supported"]
    assert "bias" in report["unsupported"]
    assert "gelu" in report["unsupported"]
    assert "output_quantization" in report["unsupported"]
    assert "comm_overlap" in report["unsupported"]
