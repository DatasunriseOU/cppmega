"""Tests for FlashInfer MXFP8 GEMM wrapper validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cppmega.megatron import flashinfer_mxfp8_gemm


def test_runner_config_defaults_to_flashinfer_api() -> None:
    config = flashinfer_mxfp8_gemm.runner_config()

    assert config.mode == "mm_mxfp8"
    assert config.tactic == 0


def test_runner_config_accepts_direct_alias() -> None:
    config = flashinfer_mxfp8_gemm.runner_config("direct-runner", "2")

    assert config.mode == "direct_tactic"
    assert config.tactic == 2


@pytest.mark.parametrize(
    ("mode", "tactic", "match"),
    [
        ("bad", 0, "must be one of"),
        ("direct_tactic", -1, "non-negative"),
        ("direct_tactic", "x", "must be an integer"),
    ],
)
def test_runner_config_rejects_invalid_values(mode, tactic, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        flashinfer_mxfp8_gemm.runner_config(mode, tactic)


def test_runner_config_from_env_is_legacy_profile_bridge(monkeypatch) -> None:
    monkeypatch.setenv("CPPMEGA_FLASHINFER_MXFP8_RUNNER", "direct_tactic")
    monkeypatch.setenv("CPPMEGA_FLASHINFER_MXFP8_TACTIC", "3")

    config = flashinfer_mxfp8_gemm.runner_config_from_env()

    assert config.mode == "direct_tactic"
    assert config.tactic == 3


def test_mm_mxfp8_dispatches_to_direct_runner_mode(monkeypatch) -> None:
    calls = []

    def fake_direct(a_data, b_data, a_scale, b_scale, *, out, out_dtype, config):
        calls.append((a_data, b_data, a_scale, b_scale, out, out_dtype, config))
        return out

    monkeypatch.setattr(flashinfer_mxfp8_gemm, "_mm_mxfp8_direct", fake_direct)
    config = flashinfer_mxfp8_gemm.FlashinferMxfp8RunnerConfig(
        mode="direct_tactic",
        tactic=2,
    )
    a = torch.empty((4, 4), dtype=torch.uint8)
    b = torch.empty((4, 4), dtype=torch.uint8)
    scale = torch.empty((4,), dtype=torch.uint8)
    out = torch.empty((4, 4), dtype=torch.bfloat16)

    result = flashinfer_mxfp8_gemm._mm_mxfp8(
        a,
        b,
        scale,
        scale,
        out=out,
        out_dtype=torch.bfloat16,
        config=config,
    )

    assert result is out
    assert calls and calls[0][-2] == torch.bfloat16
    assert calls[0][-1] is config


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
    assert "dense_compact_columnwise_dgrad_wgrad" in report["supported"]
    assert "bias" in report["unsupported"]
    assert "gelu" in report["unsupported"]
    assert "output_quantization" in report["unsupported"]
    assert "comm_overlap" in report["unsupported"]
    compact_report = report["dense_compact_columnwise"]
    assert compact_report["backend"] == "cutlass_native_compact_direct"
    assert compact_report["flashinfer_mm_mxfp8"] == {
        "accepted": False,
        "reason": "flashinfer_mm_mxfp8_no_compact_columnwise",
        "detail": (
            "FlashInfer mm_mxfp8 accepts rowwise operands plus layout_128x4 scales, "
            "not TE compact columnwise operands."
        ),
    }


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


def _make_compact_mxfp8_tensor(
    rows: int,
    cols: int,
    *,
    rowwise: bool = True,
    columnwise: bool = True,
    swizzled: bool = False,
) -> SimpleNamespace:
    tensor = SimpleNamespace(
        _fp8_dtype=_te_e4m3_dtype(),
        _with_gemm_swizzled_scales=swizzled,
    )
    if rowwise:
        tensor._rowwise_data = torch.empty((rows, cols), dtype=torch.uint8)
        tensor._rowwise_scale_inv = torch.empty((rows, cols // 32), dtype=torch.uint8)
    if columnwise:
        tensor._columnwise_data = torch.empty((rows, cols), dtype=torch.uint8)
        tensor._columnwise_scale_inv = torch.empty((rows // 32, cols), dtype=torch.uint8)
    return tensor


def test_dense_compact_columnwise_dgrad_status_accepts_metadata() -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, columnwise=False)
    weight = _make_compact_mxfp8_tensor(128, 256, rowwise=False)

    status = flashinfer_mxfp8_gemm.dense_compact_columnwise_dgrad_status(weight, dy)

    assert status.accepted
    assert status.reason == "accepted"
    assert status.backend == "cutlass_native_compact_direct"


def test_dense_compact_columnwise_wgrad_status_accepts_metadata() -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, rowwise=False)
    x = _make_compact_mxfp8_tensor(256, 384, rowwise=False)

    status = flashinfer_mxfp8_gemm.dense_compact_columnwise_wgrad_status(x, dy)

    assert status.accepted
    assert status.reason == "accepted"


def test_dgrad_compact_columnwise_dispatches_original_tensors(monkeypatch) -> None:
    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    dy = _make_compact_mxfp8_tensor(256, 128, columnwise=False)
    weight = _make_compact_mxfp8_tensor(128, 256, rowwise=False)
    sentinel = torch.empty((256, 256), dtype=torch.bfloat16)
    calls = []

    def fake_dgrad(dy_data, dy_scale, weight_data, weight_scale, **kwargs):
        calls.append((dy_data, dy_scale, weight_data, weight_scale, kwargs))
        return sentinel

    monkeypatch.setattr(cutlass, "dgrad_nn_gemm", fake_dgrad)

    result = flashinfer_mxfp8_gemm.dgrad_nn_gemm_compact_columnwise(weight, dy)

    assert result is sentinel
    assert calls
    assert calls[0][0] is dy._rowwise_data
    assert calls[0][1] is dy._rowwise_scale_inv
    assert calls[0][2] is weight._columnwise_data
    assert calls[0][3] is weight._columnwise_scale_inv
    assert calls[0][4]["out"] is None
    assert calls[0][4]["asymmetric"] is True


def test_wgrad_compact_columnwise_dispatches_original_tensors(monkeypatch) -> None:
    from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

    dy = _make_compact_mxfp8_tensor(256, 128, rowwise=False)
    x = _make_compact_mxfp8_tensor(256, 384, rowwise=False)
    sentinel = torch.empty((128, 384), dtype=torch.bfloat16)
    calls = []

    def fake_wgrad(dy_data, dy_scale, x_data, x_scale, **kwargs):
        calls.append((dy_data, dy_scale, x_data, x_scale, kwargs))
        return sentinel

    monkeypatch.setattr(cutlass, "wgrad_nt_gemm", fake_wgrad)

    result = flashinfer_mxfp8_gemm.wgrad_nt_gemm_compact_columnwise(x, dy)

    assert result is sentinel
    assert calls
    assert calls[0][0] is dy._columnwise_data
    assert calls[0][1] is dy._columnwise_scale_inv
    assert calls[0][2] is x._columnwise_data
    assert calls[0][3] is x._columnwise_scale_inv
    assert calls[0][4]["asymmetric"] is True


def test_legacy_dgrad_wrapper_prefers_compact_columnwise(monkeypatch) -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, columnwise=False)
    weight = _make_compact_mxfp8_tensor(128, 256, rowwise=False)
    sentinel = torch.empty((256, 256), dtype=torch.bfloat16)
    calls = []

    def fake_direct(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(flashinfer_mxfp8_gemm, "dgrad_nn_gemm_compact_columnwise", fake_direct)

    result = flashinfer_mxfp8_gemm.dgrad_nn_gemm(weight, dy)

    assert result is sentinel
    assert calls[0][0] == (weight, dy)


def test_legacy_wgrad_wrapper_prefers_compact_columnwise(monkeypatch) -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, rowwise=False)
    x = _make_compact_mxfp8_tensor(256, 384, rowwise=False)
    sentinel = torch.empty((128, 384), dtype=torch.bfloat16)
    calls = []

    def fake_direct(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(flashinfer_mxfp8_gemm, "wgrad_nt_gemm_compact_columnwise", fake_direct)

    result = flashinfer_mxfp8_gemm.wgrad_nt_gemm(x, dy)

    assert result is sentinel
    assert calls[0][0] == (x, dy)


def test_compact_columnwise_rejects_swizzled_scales_with_typed_reason() -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, columnwise=False)
    weight = _make_compact_mxfp8_tensor(128, 256, rowwise=False, swizzled=True)

    status = flashinfer_mxfp8_gemm.dense_compact_columnwise_dgrad_status(weight, dy)

    assert not status.accepted
    assert status.reason == "swizzled_scales"
    with pytest.raises(flashinfer_mxfp8_gemm.CompactColumnwiseUnsupportedError) as exc:
        flashinfer_mxfp8_gemm.dgrad_nn_gemm_compact_columnwise(weight, dy)
    assert exc.value.reason == "swizzled_scales"


def test_compact_columnwise_rejects_fp16_out_with_typed_reason() -> None:
    dy = _make_compact_mxfp8_tensor(256, 128, rowwise=False)
    x = _make_compact_mxfp8_tensor(256, 384, rowwise=False)

    status = flashinfer_mxfp8_gemm.dense_compact_columnwise_wgrad_status(
        x,
        dy,
        out_dtype=torch.float16,
    )

    assert not status.accepted
    assert status.reason == "unsupported_out_dtype"


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
