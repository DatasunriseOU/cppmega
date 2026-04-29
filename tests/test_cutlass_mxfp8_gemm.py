"""Tests for the CUTLASS MXFP8 GEMM backend on SM120 (GB10)."""

from __future__ import annotations

import pytest
import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor import MXFP8Quantizer


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUTLASS MXFP8 GEMM requires CUDA",
    ),
    pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning"),
]


def _make_mxfp8_operand(m: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a valid TE MXFP8 rowwise A[M,K] payload and E8M0 scale.

    Returns
    -------
    data : (M, K) uint8 tensor
    scale : (M, K//32) uint8 tensor
    """
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    source = (torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.01).contiguous()
    quantized = quantizer(source)
    return quantized._rowwise_data, quantized._rowwise_scale_inv


def _make_mxfp8_tensor(m: int, k: int, device: torch.device):
    """Create a TE MXFP8 tensor with compact rowwise and columnwise storage."""

    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    source = (torch.randn((m, k), dtype=torch.bfloat16, device=device) * 0.01).contiguous()
    return source, quantizer(source)


def _make_b_operand(n: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """B[N, K] is the second operand in TN layout (B is N-major)."""
    return _make_mxfp8_operand(n, k, device)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((a.float() - b.float()).reshape(-1))
    den = torch.linalg.vector_norm(b.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())


def _swizzle_rowwise_scale_cpu(rowwise_scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Small test-only implementation of CUTLASS/TE MXFP8 rowwise scale swizzle."""

    k_blocks = (cols + 31) // 32
    padded_rows = ((rows + 127) // 128) * 128
    padded_k_blocks = ((k_blocks + 3) // 4) * 4
    num_tiles_x = (cols + 127) // 128
    src = rowwise_scale.detach().cpu()
    out = torch.zeros((padded_rows * padded_k_blocks,), dtype=torch.uint8)
    for row in range(rows):
        for k_block in range(k_blocks):
            tile_idx_x = k_block // 4
            tile_idx_y = row // 128
            idx_in_tile_x = k_block % 4
            idx_in_tile_y = row % 128
            swizzled_idx = (tile_idx_y * num_tiles_x + tile_idx_x) * 512
            swizzled_idx += (idx_in_tile_y % 32) * 16 + (idx_in_tile_y // 32) * 4 + idx_in_tile_x
            out[swizzled_idx] = src[row, k_block]
    return out.reshape(padded_rows, padded_k_blocks).to(rowwise_scale.device)


class TestCutlassMxfp8Gemm:
    """Basic sanity checks for the CUTLASS MXFP8 TN GEMM entry point."""

    @pytest.fixture(autouse=True)
    def _import_ext(self) -> None:
        """Force-load the CUDA extension so import errors surface early."""
        from cppmega.megatron.cutlass_mxfp8_gemm import _load_cuda_ext

        self.ext = _load_cuda_ext()

    # ------------------------------------------------------------------
    # Known shapes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("m,n,k", [(128, 128, 128), (256, 256, 256)])
    def test_tn_gemm_known_shapes(self, m: int, n: int, k: int) -> None:
        """The GEMM completes without error for standard tile-aligned shapes."""
        device = torch.device("cuda")
        a_data, a_scale = _make_mxfp8_operand(m, k, device)
        b_data, b_scale = _make_b_operand(n, k, device)

        out = self.ext.tn_gemm_compact_scale(
            a_data, a_scale, b_data, b_scale,
            m, n, k,
            torch.empty(0, device=device, dtype=torch.bfloat16),
            False, False, 1.0, 0.0,
        )
        assert out is not None
        assert out.shape == (m, n)
        assert out.dtype == torch.bfloat16
        assert out.device.type == "cuda"

    # ------------------------------------------------------------------
    # Output is finite (no NaNs / infs)
    # ------------------------------------------------------------------

    def test_tn_gemm_output_finite(self) -> None:
        """GEMM output contains no NaN or inf values."""
        device = torch.device("cuda")
        m, n, k = 128, 128, 128
        a_data, a_scale = _make_mxfp8_operand(m, k, device)
        b_data, b_scale = _make_b_operand(n, k, device)

        out = self.ext.tn_gemm_compact_scale(
            a_data, a_scale, b_data, b_scale,
            m, n, k,
            torch.empty(0, device=device, dtype=torch.bfloat16),
            False, False, 1.0, 0.0,
        )
        assert torch.isfinite(out).all(), "Output contains NaN or inf"

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    def test_shape_validation_raises_on_zero(self) -> None:
        """M=0 raises a TORCH_CHECK error from the C++ extension."""
        device = torch.device("cuda")
        m, n, k = 0, 128, 128
        a_data, a_scale = _make_mxfp8_operand(max(m, 128), k, device)
        b_data, b_scale = _make_b_operand(n, k, device)

        with pytest.raises(RuntimeError, match="must be positive"):
            self.ext.tn_gemm_compact_scale(
                a_data, a_scale, b_data, b_scale,
                m, n, k,
                torch.empty(0, device=device, dtype=torch.bfloat16),
                False, False, 1.0, 0.0,
            )

    def test_shape_validation_raises_on_bad_alignment(self) -> None:
        """M non-multiple-of-128 raises."""
        device = torch.device("cuda")
        m, n, k = 129, 128, 128  # m is not a multiple of 128
        a_data, a_scale = _make_mxfp8_operand(256, k, device)
        b_data, b_scale = _make_b_operand(n, k, device)

        with pytest.raises(RuntimeError, match="multiples of 128"):
            self.ext.tn_gemm_compact_scale(
                a_data, a_scale, b_data, b_scale,
                m, n, k,
                torch.empty(0, device=device, dtype=torch.bfloat16),
                False, False, 1.0, 0.0,
            )

    def test_direct_backward_operands_match_materialized_tn_adapter(self) -> None:
        """Direct TE compact columnwise loads match the current TN sidecar adapter."""

        from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

        device = torch.device("cuda")
        m = n = k = 128
        x_ref, xq = _make_mxfp8_tensor(m, k, device)
        w_ref, wq = _make_mxfp8_tensor(n, k, device)
        dy_ref, dyq = _make_mxfp8_tensor(m, n, device)

        direct_dgrad = cutlass.dgrad_nn_gemm(
            dyq._rowwise_data,
            dyq._rowwise_scale_inv,
            wq._columnwise_data,
            wq._columnwise_scale_inv,
        )
        adapter_dgrad = cutlass.tn_gemm(
            dyq._rowwise_data,
            dyq._rowwise_scale_inv,
            wq._columnwise_data.t().contiguous(),
            wq._columnwise_scale_inv.t().contiguous(),
        )

        direct_wgrad = cutlass.wgrad_nt_gemm(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            xq._columnwise_data,
            xq._columnwise_scale_inv,
        )
        adapter_wgrad = cutlass.tn_gemm(
            dyq._columnwise_data.t().contiguous(),
            dyq._columnwise_scale_inv.t().contiguous(),
            xq._columnwise_data.t().contiguous(),
            xq._columnwise_scale_inv.t().contiguous(),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(direct_dgrad, adapter_dgrad, rtol=0, atol=0)
        torch.testing.assert_close(direct_wgrad, adapter_wgrad, rtol=0, atol=0)
        assert _rel_l2(direct_dgrad, dy_ref @ w_ref) < 0.15
        assert _rel_l2(direct_wgrad, dy_ref.t() @ x_ref) < 0.15

    def test_rowwise_swizzled_scale_entrypoint_matches_compact_direct(self) -> None:
        """Stock CUTLASS swizzled-scale probe matches compact-direct rowwise GEMM."""

        from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

        device = torch.device("cuda")
        m = n = k = 128
        x_ref, xq = _make_mxfp8_tensor(m, k, device)
        w_ref, wq = _make_mxfp8_tensor(n, k, device)
        a_scale = _swizzle_rowwise_scale_cpu(xq._rowwise_scale_inv, m, k)
        b_scale = _swizzle_rowwise_scale_cpu(wq._rowwise_scale_inv, n, k)

        swizzled = cutlass.tn_gemm_swizzled_scale(
            xq._rowwise_data,
            a_scale,
            wq._rowwise_data,
            b_scale,
        )
        compact = cutlass.tn_gemm_direct_rowwise(
            xq._rowwise_data,
            xq._rowwise_scale_inv,
            wq._rowwise_data,
            wq._rowwise_scale_inv,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(swizzled, compact, rtol=0, atol=0)
        assert _rel_l2(swizzled, x_ref @ w_ref.t()) < 0.15

    def test_mixed_wgrad_accepts_saved_x_rowwise_transpose(self) -> None:
        """wgrad direct path can consume saved ``x.T`` without a dense copy."""

        from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

        device = torch.device("cuda")
        m = n = k = 128
        x_ref, xq = _make_mxfp8_tensor(m, k, device)
        dy_ref, dyq = _make_mxfp8_tensor(m, n, device)

        x_t_data = xq._columnwise_data.t().contiguous()
        x_t_scale = xq._columnwise_scale_inv.t().contiguous()
        mixed_wgrad = cutlass.wgrad_nt_gemm_x_rowwise_transpose(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
        )
        early_wgrad = cutlass.wgrad_nt_gemm_x_rowwise_transpose(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
            b_tma_early=True,
        )
        legacy_wgrad = cutlass._tn_gemm_compact_direct(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
            m=n,
            n=k,
            k=m,
            a_source=cutlass._SOURCE_COLUMNWISE_TRANSPOSE,
            a_data_ld=int(dyq._columnwise_data.shape[1]),
            a_scale_ld=int(dyq._columnwise_scale_inv.shape[1]),
            b_source=cutlass._SOURCE_ROWWISE,
            b_data_ld=int(x_t_data.shape[1]),
            b_scale_ld=int(x_t_scale.shape[1]),
            asymmetric=True,
            a_columnwise_smem=False,
        )
        adapter_wgrad = cutlass.tn_gemm(
            dyq._columnwise_data.t().contiguous(),
            dyq._columnwise_scale_inv.t().contiguous(),
            x_t_data,
            x_t_scale,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(mixed_wgrad, legacy_wgrad, rtol=0, atol=0)
        torch.testing.assert_close(early_wgrad, mixed_wgrad, rtol=0, atol=0)
        torch.testing.assert_close(mixed_wgrad, adapter_wgrad, rtol=0, atol=0)
        assert _rel_l2(mixed_wgrad, dy_ref.t() @ x_ref) < 0.15

    def test_streaming_swizzled_stock_wgrad_matches_sidecar_stock(self) -> None:
        """Streaming stock probe avoids full dy.T sidecars and matches stock GEMM."""

        from cppmega.megatron import cutlass_mxfp8_gemm as cutlass

        device = torch.device("cuda")
        m = n = k = 128
        x_ref, xq = _make_mxfp8_tensor(m, k, device)
        dy_ref, dyq = _make_mxfp8_tensor(m, n, device)

        x_t_data = xq._columnwise_data.t().contiguous()
        x_t_scale = xq._columnwise_scale_inv.t().contiguous()
        streaming = cutlass.wgrad_nt_gemm_streaming_swizzled_stock(
            dyq._columnwise_data,
            dyq._columnwise_scale_inv,
            x_t_data,
            x_t_scale,
            tile_m=128,
            tile_n=128,
        )
        stock_sidecar = cutlass.tn_gemm_swizzled_scale(
            dyq._columnwise_data.t().contiguous(),
            _swizzle_rowwise_scale_cpu(dyq._columnwise_scale_inv.t().contiguous(), n, m),
            x_t_data,
            _swizzle_rowwise_scale_cpu(x_t_scale, k, m),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(streaming, stock_sidecar, rtol=0, atol=0)
        assert _rel_l2(streaming, dy_ref.t() @ x_ref) < 0.15
