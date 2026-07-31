"""Optional parity tests for a separately supplied TileLang FP8 reference.

Set ``CPPMEGA_MLX_REFERENCE_ROOT`` to a cppmega.mlx checkout to enable these
tests. Production dispatch remains in ``cppmega.megatron`` and never imports
the reference package.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


_REFERENCE_ROOT_RAW = os.environ.get("CPPMEGA_MLX_REFERENCE_ROOT")
if not _REFERENCE_ROOT_RAW:
    pytest.skip(
        "set CPPMEGA_MLX_REFERENCE_ROOT to enable optional FP8 parity tests",
        allow_module_level=True,
    )
_REFERENCE_ROOT = Path(_REFERENCE_ROOT_RAW).expanduser().resolve()
if not (_REFERENCE_ROOT / "cppmega_mlx" / "__init__.py").is_file():
    pytest.skip(
        f"invalid CPPMEGA_MLX_REFERENCE_ROOT: {_REFERENCE_ROOT}",
        allow_module_level=True,
    )
sys.path.insert(0, str(_REFERENCE_ROOT))

_reference_module = pytest.importorskip("cppmega_mlx.nn._tilelang.fp8_amax")
assert Path(_reference_module.__file__).resolve().is_relative_to(_REFERENCE_ROOT)
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


def test_mxfp8_transpose_emit_flattens_real_3d_tensor() -> None:
    """The production shim normalizes TE's SBH activation to a 2D operand."""

    env = os.environ.copy()
    env.update(
        {
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "0",
            "CPPMEGA_TE_MXFP8_BWD_BACKEND": "te_tn_adapter",
            "CPPMEGA_TE_VERSION_STRICT": "0",
            "CPPMEGA_DSA_SPARSE_MODE": "gather_scatter",
            "CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW": "1",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    code = """
import torch
from scripts import cppmega_fp8_shim as shim

source = torch.arange(24).reshape(2, 3, 4)
flat = shim._cppmega_flatten_lastdim_2d(source)
assert flat.shape == (6, 4)
assert torch.equal(flat, source.reshape(6, 4))
matrix = torch.arange(24).reshape(6, 4)
assert shim._cppmega_flatten_lastdim_2d(matrix) is matrix
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    # torch MPS cannot cast to/from float8 on-device ("Undefined type
    # Float8_e4m3fn"), so the fp8 round-trip runs on CPU tensors; the kernel
    # output is produced on-device and only moved here for the comparison.
    deq = fp8_out.cpu().to(torch.float32) * scale.cpu()
    ref_inv_scale = (1.0 / ref_scale.item()) if ref_scale.item() > 0 else 1.0
    ref_q = (x_clamped.cpu().to(torch.float32) * ref_inv_scale).clamp(
        -_FP8_E4M3_MAX, _FP8_E4M3_MAX
    ).to(torch.float8_e4m3fn).to(torch.float32) * ref_scale.cpu()
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
    """fp8_pack_tilelang must raise FloatingPointError on Inf, not
    silently produce a degenerate (0 or NaN) scale that poisons downstream
    weights. Wave-3 self-audit: closes the silent-NaN-propagation hole
    in the host-side scale derivation.

    NaN is the documented wave-11 exception: the pre-filter substitutes the
    amax identity (0.0) for NaN before the cross-block reduction (CUDA
    atomicMax is UB on NaN; the Metal CAS loop would spin forever), so a
    NaN-poisoned input now yields the finite max over the real data instead
    of raising. The NaN case below locks that trade-off in so a filter
    regression (NaN leaking into the scale derivation) still fails loudly.
    """

    import math

    device = _pick_device()
    if device.type == "cpu":
        pytest.skip("TileLang pack requires a CUDA or Metal device")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available in this build")

    for poison in [float("inf"), float("-inf")]:
        x = torch.randn(32, 256, dtype=torch.float16, device=device)
        x[0, 0] = poison
        with pytest.raises(FloatingPointError, match=r"non-finite values"):
            fp8_pack_tilelang(x)

    # Wave-11 NaN filter semantics: finite scale derived from the real data,
    # no hang, no NaN scale.
    x = torch.randn(32, 256, dtype=torch.float16, device=device)
    x[0, 0] = float("nan")
    _fp8_out, scale, _orig_dtype = fp8_pack_tilelang(x)
    assert math.isfinite(scale.item()), (
        "wave-11 NaN filter regressed: NaN leaked into the scale derivation"
    )


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


def test_wave9_concurrent_amax_compile():
    """Wave-9 #6: ``_expose_to_globals`` mutates module ``__globals__`` in
    place, and ``functools.lru_cache`` is not thread-safe under concurrent
    miss + insert. Two threads compiling kernels for different
    ``(n, dtype)`` combos used to race the ``N`` / ``BLOCK`` / ``DTYPE``
    slots and could ship a corrupt PrimFunc to one of them. The
    ``_FP8_AMAX_LOCK`` guard added in this commit serialises both
    ``_amax_kernel_for`` and ``_quantize_kernel_for`` so the build path
    is single-threaded even when callers compile concurrently.

    The test does not require Metal / CUDA to be available because it
    exercises the *build* (PrimFunc construction + lower) path, not the
    runtime launch. Skips only when ``tilelang_supports`` reports the
    build path is unreachable for the host.
    """

    import threading

    device = _pick_device()
    if device.type == "cpu" or not tilelang_supports(device):
        pytest.skip("tilelang build path unreachable on this host")

    from cppmega_mlx.nn._tilelang.fp8_amax import (  # noqa: E402
        _amax_kernel_for,
        _quantize_kernel_for,
    )

    target = "metal"
    combos = [
        (256, "float16"),
        (1024, "float16"),
        (256, "bfloat16"),
        (4096, "float16"),
    ]
    results: dict[tuple[int, str], object] = {}
    errors: list[BaseException] = []

    def _build(combo: tuple[int, str]) -> None:
        try:
            results[combo] = _amax_kernel_for(combo[0], combo[1], target)
        except BaseException as exc:  # pragma: no cover - failure mode under test
            errors.append(exc)

    threads = [threading.Thread(target=_build, args=(c,)) for c in combos]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent _amax_kernel_for crashed: {errors}"
    assert len(results) == len(combos)
    # Distinct kernel objects per (n, dtype). Ensures the cache did not
    # alias a corrupt PrimFunc across concurrent misses.
    ids = {id(v) for v in results.values()}
    assert len(ids) == len(combos), (
        f"concurrent compiles aliased PrimFuncs (cache race): "
        f"{[(c, id(v)) for c, v in results.items()]}"
    )

    # Same exercise on the quantize side -- same lock so a regression in
    # _quantize_kernel_for would surface here.
    quant_results: dict[tuple[int, str], object] = {}

    def _build_q(combo: tuple[int, str]) -> None:
        try:
            quant_results[combo] = _quantize_kernel_for(combo[0], combo[1], target)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_build_q, args=(c,)) for c in combos]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent _quantize_kernel_for crashed: {errors}"
    assert len(quant_results) == len(combos)
    quant_ids = {id(v) for v in quant_results.values()}
    assert len(quant_ids) == len(combos), (
        f"concurrent quantize compiles aliased PrimFuncs: "
        f"{[(c, id(v)) for c, v in quant_results.items()]}"
    )


@pytest.mark.skipif(
    not _TILELANG_OK,
    reason=f"TileLang FP8 unavailable: {_STATUS.reason or 'unknown'}",
)
def test_wave11_amax_nan_input_does_not_hang() -> None:
    """Wave-11 #2: NaN-poisoned input must not hang the kernel.

    CUDA ``atomicMax`` on fp32 is undefined for NaN, and Metal has no
    native fp32 atomic_max — the lowering is a CAS loop. Without the
    wave-11 pre-filter (`amax_safe = if v == v else 0`), a single NaN
    in the input makes the CAS loop spin forever (NaN != NaN, so
    compare_exchange never succeeds). This test launches the kernel
    on a tensor that contains NaN and asserts it returns within a
    reasonable wall-clock budget.
    """
    import threading

    device = _pick_device()
    n = 4096
    x = torch.randn(n, device=device, dtype=torch.float32)
    # Poison a few elements with NaN.
    x[1] = float("nan")
    x[1234] = float("nan")
    x[-1] = float("nan")

    result_holder: list[object] = []

    def _run() -> None:
        try:
            result_holder.append(fp8_amax_tilelang(x))
        except BaseException as exc:  # pragma: no cover - propagated below
            result_holder.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30.0)  # generous: kernel build + launch + sync
    assert not t.is_alive(), (
        "fp8_amax_tilelang hung on NaN input -- wave-11 atomic_max NaN "
        "pre-filter regressed (Metal CAS loop spin)."
    )
    assert result_holder, "kernel thread produced no result"
    res = result_holder[0]
    if isinstance(res, BaseException) and not isinstance(res, FloatingPointError):
        raise res  # propagate unexpected error
    # Either a non-NaN amax (NaN was filtered to 0 → max over real data)
    # or a wave-3 FloatingPointError if all-NaN. Both are acceptable.



def test_wave11_amax_lock_is_global() -> None:
    """Wave-11 #3 (meta wave-10 review MED, NOT FIXED — design constraint).

    Meta flagged the global ``_FP8_AMAX_LOCK`` as a DoS amplifier and
    suggested per-signature locks. The fix was rejected because
    ``_expose_to_globals`` mutates *module-level* globals shared across
    every signature; per-signature locks would race on those globals.

    This test locks in the design choice: the lock IS still a single
    ``threading.Lock`` instance shared by both kernel factories. Future
    wave-12 work (per-thread globals via ``types.FunctionType`` rebuild
    or ``T.Var`` parameterisation) would flip this expectation; if
    you're updating it, please update the inline rationale comment in
    ``fp8_amax.py`` at the same time.
    """
    import threading
    from cppmega_mlx.nn._tilelang import fp8_amax as _fp8_amax

    assert isinstance(_fp8_amax._FP8_AMAX_LOCK, type(threading.Lock())), (
        "_FP8_AMAX_LOCK must be a threading.Lock instance — see "
        "wave-11 #3 rationale comment in fp8_amax.py"
    )


def test_wave12_resolve_in_dtype_rejects_non_float() -> None:
    """Wave-12 #4 (meta wave-11 MED): _resolve_in_dtype must reject any
    tensor whose dtype is not floating-point BEFORE the dict lookup, so
    a custom torch fork that registers a non-float dtype with a forged
    ``__torch_dtype__='float16'`` cannot pass through and corrupt the
    amax kernel's bit pattern.
    """
    from cppmega_mlx.nn._tilelang.fp8_amax import _resolve_in_dtype

    # Each integer dtype must raise TypeError, with the message naming the
    # offending dtype so downstream callers can diagnose without reflection.
    for int_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        x = torch.zeros(4, dtype=int_dtype)
        with pytest.raises(TypeError, match=r"floating-point"):
            _resolve_in_dtype(x)

    # Sanity: real floats still pass.
    for fp_dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.zeros(4, dtype=fp_dtype)
        assert _resolve_in_dtype(x) in {"float16", "bfloat16", "float32"}
