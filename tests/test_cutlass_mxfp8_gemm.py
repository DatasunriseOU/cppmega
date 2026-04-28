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
