"""Numerical-parity tests for the TileLang Path C FP8 amax/quantize port.

The kernels under test live at
``cppmega_mlx/nn/_tilelang/fp8_amax.py`` and replace the CUDA-only Triton
``_amax_kernel`` / ``_quantize_kernel`` in
``cppmega/megatron/fp8_activations.py`` on hosts where TileLang is available
(both CUDA and Apple Metal SIMDgroup).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cppmega_mlx.nn._tilelang.fp8_amax")
from cppmega_mlx.nn._tilelang.fp8_amax import (  # noqa: E402
    _FP8_E4M3_MAX,
    fp8_amax_path_c_status,
    fp8_amax_tilelang,
    fp8_pack_tilelang,
    tilelang_supports,
)


_STATUS = fp8_amax_path_c_status()
_TILELANG_OK = _STATUS.available

_HAS_CUDA = torch.cuda.is_available()
_HAS_MPS = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())


def _pick_device() -> torch.device:
    if _HAS_CUDA and tilelang_supports(torch.device("cuda")):
        return torch.device("cuda")
    if _HAS_MPS and tilelang_supports(torch.device("mps")):
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_amax_matches_torch_reference():
    """TileLang amax must match ``tensor.abs().amax()`` within fp16 ULP."""

    torch.manual_seed(0xC0DE)
    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang amax requires a CUDA or Metal device")

    x = torch.randn(32, 4096, dtype=torch.float16, device=device)
    out = fp8_amax_tilelang(x)

    assert out.shape == (1,)
    assert out.dtype == torch.float32
    assert out.device == x.device

    ref = x.abs().amax().to(torch.float32)
    torch.testing.assert_close(out.squeeze(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton parity check requires CUDA")
def test_fp8_amax_matches_triton_reference():
    """On CUDA hosts, parity with the Triton ``_amax_kernel`` reference path."""

    if not _TILELANG_OK:
        pytest.skip(f"TileLang unavailable: {_STATUS.reason}")

    fp8 = pytest.importorskip("cppmega.megatron.fp8_activations")
    if not getattr(fp8, "_TRITON_AVAILABLE", False):
        pytest.skip("Triton not installed on this host")

    torch.manual_seed(0xBEEF)
    x = torch.randn(32, 4096, dtype=torch.float16, device="cuda").contiguous()
    flat = x.reshape(-1)

    BLOCK = 1024
    n_blocks = (flat.numel() + BLOCK - 1) // BLOCK
    triton_out = torch.zeros(1, dtype=torch.float32, device="cuda")
    fp8._amax_kernel[(n_blocks,)](flat, triton_out, flat.numel(), BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    tilelang_out = fp8_amax_tilelang(x)
    torch.testing.assert_close(tilelang_out, triton_out, rtol=0, atol=0)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_pack_tilelang_bf16_clamp_roundtrip():
    """Full pack(bf16, clamp=True) → unpack round-trip vs. torch reference."""

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available in this build")

    torch.manual_seed(0xFEED)
    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang pack requires a CUDA or Metal device")

    # Span values past the FP8 e4m3 max so clamping is non-trivial.
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device) * 1000.0
    fp8_out, scale, orig_dtype = fp8_pack_tilelang(x, clamp=True)

    assert fp8_out.dtype == torch.float8_e4m3fn
    assert fp8_out.shape == x.shape
    assert orig_dtype == torch.bfloat16
    assert scale.dtype == torch.float32

    # Reference: clamp -> compute amax -> scale -> cast -> dequantize.
    x_clamped = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    amax = x_clamped.abs().amax().to(torch.float32)
    if amax > 0:
        ref_scale = amax / _FP8_E4M3_MAX
    else:
        ref_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    torch.testing.assert_close(scale, ref_scale, rtol=1e-3, atol=1e-3)

    # Dequantize the kernel output and compare against the reference quant
    # path; FP8 e4m3 has ~3 mantissa bits so we expect ~10% relative error.
    deq = fp8_out.to(torch.float32) * scale
    ref_inv_scale = (1.0 / ref_scale.item()) if ref_scale.item() > 0 else 1.0
    ref_q = (x_clamped.to(torch.float32) * ref_inv_scale).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn).to(torch.float32) * ref_scale
    torch.testing.assert_close(deq, ref_q, rtol=0.15, atol=ref_scale.item() * 1.0)
