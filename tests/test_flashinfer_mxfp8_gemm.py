"""Tests for FlashInfer MXFP8 GEMM wrapper validation."""

from __future__ import annotations

from types import SimpleNamespace

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


# ---------------------------------------------------------------
# GEMM correctness tests (require CUDA + flashinfer + SM120/SM121)
# ---------------------------------------------------------------

_has_flashinfer_cutlass = False
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        try:
            import flashinfer

            _has_flashinfer_cutlass = all(
                hasattr(flashinfer, name)
                for name in ("mm_mxfp8", "mxfp8_quantize")
            )
        except ImportError:
            pass
except Exception:
    pass

requires_flashinfer_cutlass = pytest.mark.skipif(
    not _has_flashinfer_cutlass,
    reason="flashinfer with CUTLASS SM120/SM121 backend not available",
)


def _te_e4m3_dtype():
    try:
        import transformer_engine_torch as tex

        return tex.DType.kFloat8E4M3
    except ImportError:
        return "kFloat8E4M3"


def _make_rowwise_tensor(
    payload: torch.Tensor,
    compact_scale: torch.Tensor,
    rows: int,
    cols: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        _rowwise_data=payload.view(torch.uint8),
        _rowwise_scale_inv=compact_scale.view(rows, (cols + 31) // 32),
        _fp8_dtype=_te_e4m3_dtype(),
        _with_gemm_swizzled_scales=False,
    )


def _make_swizzled_scale(compact: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    k_blocks = (cols + 31) // 32
    padded_rows = ((rows + 127) // 128) * 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    num_tiles_x = (cols + 127) // 128
    result = torch.zeros((padded_rows * padded_k_blocks,), dtype=torch.uint8)
    compact_cpu = compact.cpu()
    for row in range(rows):
        for k_block in range(k_blocks):
            tile_idx_x = k_block // 4
            tile_idx_y = row // 128
            idx_in_tile_x = k_block % 4
            idx_in_tile_y = row % 128
            idx = (tile_idx_y * num_tiles_x + tile_idx_x) * 512
            idx += (
                (idx_in_tile_y % 32) * 16
                + (idx_in_tile_y // 32) * 4
                + idx_in_tile_x
            )
            result[idx] = compact_cpu[row, k_block]
    return result


@requires_flashinfer_cutlass
def test_fprop_tn_gemm_output_shape_and_finite() -> None:
    from flashinfer import mxfp8_quantize

    m, n, k = 64, 128, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    x_fp8, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=False, backend="cuda")
    w_fp8, w_scale = mxfp8_quantize(weight, is_sf_swizzled_layout=False, backend="cuda")
    x_tensor = _make_rowwise_tensor(x_fp8, x_scale, m, k)
    weight_tensor = _make_rowwise_tensor(w_fp8, w_scale, n, k)

    out = flashinfer_mxfp8_gemm.fprop_tn_gemm(weight_tensor, x_tensor)
    torch.cuda.synchronize()

    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@requires_flashinfer_cutlass
def test_swizzle_rowwise_scale_produces_correct_layout() -> None:
    rows, cols = 64, 128
    compact = torch.arange(rows * (cols // 32), device="cuda", dtype=torch.uint8).view(
        rows,
        cols // 32,
    )

    out = flashinfer_mxfp8_gemm.swizzle_rowwise_scale(compact, rows, cols)
    torch.cuda.synchronize()

    expected = _make_swizzled_scale(compact, rows, cols)
    torch.testing.assert_close(out.cpu(), expected)


def test_scale_for_rowwise_matrix_rejects_bad_pre_swizzled_shape() -> None:
    bad_scale = torch.empty(100, dtype=torch.uint8)

    with pytest.raises(ValueError, match="pre-swizzled scale has"):
        flashinfer_mxfp8_gemm._scale_for_rowwise_matrix(
            bad_scale,
            rows=64,
            cols=128,
            with_gemm_swizzled_scales=True,
        )
