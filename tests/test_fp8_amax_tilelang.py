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
    _bucket_n,
    _pick_block_size,
    fp8_amax_path_c_status,
    fp8_amax_tilelang,
    fp8_pack_tilelang,
    make_fp8_amax_kernel,
    tilelang_supports,
    tilelang_supports_with_reason,
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


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_amax_partial_last_block_matches_torch_reference():
    """Partial-last-block correctness: a non-divisible N must NOT silently
    produce wrong output (either succeeds within the same tolerance as the
    aligned shape, or raises a precise diagnostic).

    Targets the BLOCK_SIZE-vs-threads concern from the wave-1 grok review:
    on a partial last block, the masked ``T.Parallel(BLOCK)`` strided loop
    must cover every in-range element exactly once.
    """

    torch.manual_seed(0xBAD)
    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang amax requires a CUDA or Metal device")

    # 100 is < BLOCK_SIZE (1024 cuda / 256 metal), exercising the
    # ``n_elements < block`` shrink path *and* a partial last block.
    # 4097 forces bucket_n = 8192 with a 4095-zero-padded tail.
    for shape in [(100,), (4097,), (1, 4097), (3, 1234)]:
        x = torch.randn(*shape, dtype=torch.float16, device=device) * 5.0
        out = fp8_amax_tilelang(x)
        ref = x.abs().amax().to(torch.float32)
        torch.testing.assert_close(
            out.squeeze(),
            ref,
            rtol=1e-3,
            atol=1e-3,
            msg=f"partial-block amax mismatch at shape={shape}",
        )


def test_block_size_threads_invariant_raises_on_violation():
    """Hand-rolled ``make_fp8_amax_kernel`` with a non-divisible (BLOCK, threads)
    must raise a precise ``RuntimeError`` rather than silently producing
    a kernel that under-covers the last block."""

    with pytest.raises(RuntimeError, match=r"BLOCK_SIZE=1000 not divisible by THREADS=128"):
        make_fp8_amax_kernel(n_elements=4096, block_size=1000, threads=128)


def test_pick_block_size_per_target_table():
    """The dynamic block-size picker honors the per-target table and the
    BLOCK % THREADS == 0 invariant."""

    cuda_block, cuda_threads = _pick_block_size("cuda", 1 << 20)
    assert cuda_block == 1024 and cuda_threads == 128
    assert cuda_block % cuda_threads == 0

    hip_block, hip_threads = _pick_block_size("hip", 1 << 20)
    assert hip_block == 1024 and hip_threads == 256
    assert hip_block % hip_threads == 0

    metal_block, metal_threads = _pick_block_size("metal -thread_warp_size=32", 1 << 20)
    assert metal_block == 256 and metal_threads == 64
    assert metal_block % metal_threads == 0

    # Tiny input: block shrinks to next pow2 >= threads
    tiny_block, tiny_threads = _pick_block_size("cuda", 50)
    assert tiny_block >= tiny_threads
    assert tiny_block % tiny_threads == 0


def test_bucket_n_collapses_close_shapes():
    """Bucket key collapses 5 close shapes (4097..8192) to a single key."""

    keys = {_bucket_n(n, block_size=1024) for n in [4097, 5000, 6000, 7777, 8192]}
    assert keys == {8192}, f"expected single 8192 bucket, got {keys}"

    # Exact pow2 stays at pow2; tiny values clamp to block_size.
    assert _bucket_n(8192, block_size=1024) == 8192
    assert _bucket_n(100, block_size=1024) == 1024


def test_tilelang_supports_with_reason_returns_2tuple():
    """Every return path of the diagnostic helper is a (bool, str) 2-tuple."""

    cases = [
        torch.device("cpu"),
        "cpu",
        torch.device("cuda"),
        "cuda",
        torch.device("mps") if hasattr(torch.backends, "mps") else None,
        None,
        "definitely-not-a-device",
    ]
    for case in cases:
        if case is None and not (
            hasattr(torch.backends, "mps") and case is None
        ):
            # the explicit None is a separate test-case; skip the conditional
            pass
        result = tilelang_supports_with_reason(case)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"non-tuple return for case={case!r}: {result!r}"
        )
        assert isinstance(result[0], bool), f"non-bool first: {result!r}"
        assert isinstance(result[1], str) and result[1], (
            f"empty/non-str reason for case={case!r}: {result!r}"
        )

    # The boolean wrapper must agree with the tuple's first element.
    for case in cases:
        assert tilelang_supports(case) == tilelang_supports_with_reason(case)[0]


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_pack_rejects_nonfinite_input():
    """fp8_pack_tilelang must raise FloatingPointError on NaN/Inf, not
    silently produce a degenerate (0 or NaN) scale that poisons downstream
    weights. Wave-3 self-audit: closes the silent-NaN-propagation hole
    in the host-side scale derivation.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang pack requires a CUDA or Metal device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available in this build")

    for poison in [float("nan"), float("inf"), float("-inf")]:
        x = torch.randn(32, 256, dtype=torch.float16, device=device)
        x[0, 0] = poison
        with pytest.raises(FloatingPointError, match=r"non-finite values"):
            fp8_pack_tilelang(x)


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_amax_padding_does_not_change_result():
    """The pow2 bucket pad-with-zeros must be a no-op for amax.

    Regression guard for the JIT bucket cache: if a future refactor of
    ``_bucket_n`` accidentally pads with non-zero (e.g. uninitialized
    ``empty``), this test catches it because the padded amax would diverge
    from the unpadded reference whenever the random tail contains values
    larger than the data.
    """

    torch.manual_seed(0xCAFE)
    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang amax requires a CUDA or Metal device")

    # Choose N that is *not* a power of two so the bucket cache pads.
    for n in [4097, 5000, 8193]:
        x = torch.randn(n, dtype=torch.float16, device=device) * 5.0
        out = fp8_amax_tilelang(x)
        ref = x.abs().amax().to(torch.float32)
        torch.testing.assert_close(
            out.squeeze(),
            ref,
            rtol=1e-3,
            atol=1e-3,
            msg=f"bucket-pad amax diverges from reference at n={n}",
        )


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
def test_fp8_amax_handles_signed_zero_only():
    """amax over only ±0.0 must be exactly 0.0 (not -0.0, not denormal).

    Wave-3 self-audit: closes a Metal-vs-CUDA divergence concern where
    atomic_max on fp32 0.0 vs -0.0 may differ between the CUDA atomicMax
    and the Metal CAS-loop emulation.
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang amax requires a CUDA or Metal device")

    # All-zero input.
    x = torch.zeros(1024, dtype=torch.float16, device=device)
    out = fp8_amax_tilelang(x)
    assert float(out.item()) == 0.0, f"amax(zeros) should be 0.0, got {out.item()!r}"

    # Mix of +0/-0 (negative-zero in fp16 still has |.| == 0).
    x = torch.zeros(1024, dtype=torch.float16, device=device)
    x[::2] = -0.0
    out = fp8_amax_tilelang(x)
    assert float(out.item()) == 0.0, f"amax(±0) should be 0.0, got {out.item()!r}"


@pytest.mark.skipif(not _TILELANG_OK, reason=f"TileLang unavailable: {_STATUS.reason}")
@pytest.mark.parametrize("n", [100, 4097, 8001, 33333])
def test_wave9_amax_non_pow2_no_pad(n):
    """Wave-9 #3 — fp8_amax skips pad+copy on non-pow2 shapes that would
    inflate kernel work by >=50%.

    The wrapper now branches on ``bucket_n * 2 >= 3 * n`` and dispatches
    to a per-shape kernel keyed on ``n`` directly. The per-block
    ``if gi < N`` predicate (wave-3 partial-block guard) handles the tail
    so the result still matches the bucket+pad path numerically.

    See grok rev_38ff59759f HIGH-severity perf finding: "For any activation
    shape not exactly a power-of-2 multiple of target BLOCK ... you pay full
    device ``mx.zeros + copy_`` *every forward/backward pass*. For N=4097
    → 8192 this is ~50% extra HBM traffic + allocation latency."
    """

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang amax requires a CUDA or Metal device")

    torch.manual_seed(0xFACE)
    x = torch.randn(n, dtype=torch.float16, device=device)
    out = fp8_amax_tilelang(x)
    ref = x.abs().amax().to(torch.float32)

    assert out.shape == (1,)
    torch.testing.assert_close(out.squeeze(), ref, rtol=1e-3, atol=1e-3)
