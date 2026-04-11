"""Correctness tests for CuTe DSL MIMO autograd.Function.

Tests:
1. Forward-only correctness vs TileLang reference
2. Forward + backward correctness (gradient comparison)
3. Timing comparison (fwd, bwd, total)

Target shapes: NAM56R training config
  B=2, S=256, H=8, N=64, P=64, R=4, chunk_size=16
"""

import os
import sys
import time
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F


def generate_test_inputs(
    B=2, S=256, H=8, G=1, N=64, P=64, R=4,
    chunk_size=16, rotary_dim_divisor=4,
    dtype=torch.bfloat16, device="cuda",
    requires_grad=True,
):
    """Generate random inputs matching NAM56R shapes.

    Scaling note: ADT must be small (goes into exp via cumsum over S tokens).
    Large ADT values cause exp() overflow and spurious mismatches between
    fp32-reference and bf16-kernel paths.
    """
    Q = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
    K = torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.1
    V = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1
    ADT = torch.randn(B, H, S, device=device, dtype=torch.float32) * 0.01
    DT = torch.randn(B, H, S, device=device, dtype=torch.float32).abs() * 0.1
    Trap = torch.randn(B, H, S, device=device, dtype=dtype) * 0.1
    Q_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    K_bias = torch.randn(H, R, N, device=device, dtype=torch.float32) * 0.01
    MIMO_V = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
    MIMO_Z = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
    MIMO_Out = torch.randn(H, R, P, device=device, dtype=torch.float32) * 0.1
    Angles = torch.randn(B, S, H, N // rotary_dim_divisor, device=device, dtype=torch.float32) * 0.5
    D = torch.randn(H, device=device, dtype=torch.float32) * 0.1
    Z = torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.1

    if requires_grad:
        for t in [Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Angles, D, Z]:
            t.requires_grad_(True)

    return {
        "Q": Q, "K": K, "V": V, "ADT": ADT, "DT": DT, "Trap": Trap,
        "Q_bias": Q_bias, "K_bias": K_bias,
        "MIMO_V": MIMO_V, "MIMO_Z": MIMO_Z, "MIMO_Out": MIMO_Out,
        "Angles": Angles, "D": D, "Z": Z,
        "chunk_size": chunk_size, "rotary_dim_divisor": rotary_dim_divisor,
        "dtype": dtype,
    }


def clone_inputs(inputs):
    """Deep-clone inputs for independent gradient computation."""
    cloned = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            c = v.detach().clone()
            if v.requires_grad:
                c.requires_grad_(True)
            cloned[k] = c
        else:
            cloned[k] = v
    return cloned


def run_mimo(fn, inputs_dict):
    """Call mimo function with standard arguments."""
    return fn(
        inputs_dict["Q"], inputs_dict["K"], inputs_dict["V"],
        inputs_dict["ADT"], inputs_dict["DT"], inputs_dict["Trap"],
        inputs_dict["Q_bias"], inputs_dict["K_bias"],
        inputs_dict["MIMO_V"], inputs_dict["MIMO_Z"], inputs_dict["MIMO_Out"],
        inputs_dict["Angles"], inputs_dict["D"], inputs_dict["Z"],
        inputs_dict["chunk_size"], inputs_dict["rotary_dim_divisor"],
        inputs_dict["dtype"],
    )


def test_forward_correctness():
    """Test 1: Forward output matches TileLang within tolerance."""
    print("=" * 70)
    print("TEST 1: Forward-only correctness (B=2, S=256, H=8, N=64, P=64, R=4)")
    print("=" * 70)

    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as tilelang_mimo
    from cppmega.megatron.cute_dsl_mimo.cute_dsl_mimo_autograd import cute_dsl_mimo_combined

    inputs = generate_test_inputs(requires_grad=False)
    inputs_clone = clone_inputs(inputs)

    # TileLang reference
    torch.cuda.synchronize()
    with torch.no_grad():
        out_tl = run_mimo(tilelang_mimo, inputs)
    torch.cuda.synchronize()

    # CuTe DSL
    with torch.no_grad():
        out_cd = run_mimo(cute_dsl_mimo_combined, inputs_clone)
    torch.cuda.synchronize()

    # Compare
    print(f"  TileLang output shape: {out_tl.shape}, dtype: {out_tl.dtype}")
    print(f"  CuTe DSL output shape: {out_cd.shape}, dtype: {out_cd.dtype}")

    diff = (out_tl.float() - out_cd.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_tl.float().abs() + 1e-8)).mean().item()

    print(f"  Max abs error:  {max_err:.6e}")
    print(f"  Mean abs error: {mean_err:.6e}")
    print(f"  Mean rel error: {rel_err:.6e}")

    rtol, atol = 1e-2, 1e-2
    passed = torch.allclose(out_tl.float(), out_cd.float(), rtol=rtol, atol=atol)
    print(f"  torch.allclose(rtol={rtol}, atol={atol}): {'PASS' if passed else 'FAIL'}")

    if not passed:
        worst_idx = diff.argmax()
        worst_flat = worst_idx.item()
        print(f"  Worst element index (flat): {worst_flat}")
        print(f"  TileLang value: {out_tl.float().flatten()[worst_flat]:.6e}")
        print(f"  CuTe DSL value: {out_cd.float().flatten()[worst_flat]:.6e}")

    print()
    return passed


def test_backward_correctness():
    """Test 2: Backward gradients match TileLang within tolerance."""
    print("=" * 70)
    print("TEST 2: Backward gradient correctness (B=1, S=64, H=4)")
    print("=" * 70)

    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as tilelang_mimo
    from cppmega.megatron.cute_dsl_mimo.cute_dsl_mimo_autograd import cute_dsl_mimo_combined

    # Smaller shapes for gradient test
    inputs_tl = generate_test_inputs(B=1, S=64, H=4, requires_grad=True)
    inputs_cd = clone_inputs(inputs_tl)

    # TileLang forward + backward
    out_tl = run_mimo(tilelang_mimo, inputs_tl)
    loss_tl = out_tl.float().sum()
    loss_tl.backward()
    torch.cuda.synchronize()

    # CuTe DSL forward + backward
    out_cd = run_mimo(cute_dsl_mimo_combined, inputs_cd)
    loss_cd = out_cd.float().sum()
    loss_cd.backward()
    torch.cuda.synchronize()

    # Compare output first
    out_diff = (out_tl.float() - out_cd.float()).abs().max().item()
    print(f"  Output max error: {out_diff:.6e}")

    rtol, atol = 1e-2, 1e-2
    all_passed = True
    grad_names = ["Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias",
                  "MIMO_V", "MIMO_Z", "MIMO_Out", "Angles", "D", "Z"]

    for name in grad_names:
        g_tl = inputs_tl[name].grad
        g_cd = inputs_cd[name].grad

        if g_tl is None and g_cd is None:
            print(f"  d{name:11s}: both None (OK)")
            continue
        if g_tl is None or g_cd is None:
            print(f"  d{name:11s}: MISMATCH (one is None)")
            all_passed = False
            continue

        diff = (g_tl.float() - g_cd.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        passed = torch.allclose(g_tl.float(), g_cd.float(), rtol=rtol, atol=atol)
        status = "PASS" if passed else "FAIL"
        print(f"  d{name:11s}: max_err={max_err:.4e} mean_err={mean_err:.4e} [{status}]")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed


def test_timing():
    """Test 3: Timing comparison between TileLang and CuTe DSL."""
    print("=" * 70)
    print("TEST 3: Timing comparison (B=2, S=256, H=8, N=64, P=64, R=4)")
    print("=" * 70)

    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as tilelang_mimo
    from cppmega.megatron.cute_dsl_mimo.cute_dsl_mimo_autograd import cute_dsl_mimo_combined

    def time_fn(fn, n_warmup=3, n_iter=10):
        """Time a function with CUDA events."""
        for _ in range(n_warmup):
            fn()
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / n_iter

    # --- Forward-only timing ---
    inputs_no_grad = generate_test_inputs(requires_grad=False)

    def tl_fwd():
        with torch.no_grad():
            return run_mimo(
                __import__('mamba_ssm.ops.tilelang.mamba3.mamba3_mimo', fromlist=['mamba3_mimo']).mamba3_mimo,
                inputs_no_grad
            )

    def cd_fwd():
        with torch.no_grad():
            return run_mimo(cute_dsl_mimo_combined, inputs_no_grad)

    tl_fwd_ms = time_fn(tl_fwd, n_warmup=5, n_iter=20)
    cd_fwd_ms = time_fn(cd_fwd, n_warmup=3, n_iter=5)

    print(f"  Forward only:")
    print(f"    TileLang:  {tl_fwd_ms:.3f} ms")
    print(f"    CuTe DSL:  {cd_fwd_ms:.3f} ms")
    if tl_fwd_ms > 0:
        print(f"    Ratio:     {cd_fwd_ms / tl_fwd_ms:.2f}x {'(slower)' if cd_fwd_ms > tl_fwd_ms else '(faster)'}")
    print()

    # --- Forward + backward timing ---
    inputs = generate_test_inputs(requires_grad=True)

    def tl_fwd_bwd():
        inp = clone_inputs(inputs)
        out = run_mimo(tilelang_mimo, inp)
        out.float().sum().backward()

    def cd_fwd_bwd():
        inp = clone_inputs(inputs)
        out = run_mimo(cute_dsl_mimo_combined, inp)
        out.float().sum().backward()

    tl_total_ms = time_fn(tl_fwd_bwd, n_warmup=2, n_iter=5)
    cd_total_ms = time_fn(cd_fwd_bwd, n_warmup=2, n_iter=5)

    print(f"  Forward + backward:")
    print(f"    TileLang:  {tl_total_ms:.3f} ms")
    print(f"    CuTe DSL:  {cd_total_ms:.3f} ms")
    if tl_total_ms > 0:
        print(f"    Ratio:     {cd_total_ms / tl_total_ms:.2f}x {'(slower)' if cd_total_ms > tl_total_ms else '(faster)'}")
    print()


def main():
    torch.manual_seed(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    fwd_ok = test_forward_correctness()
    if fwd_ok:
        bwd_ok = test_backward_correctness()
        test_timing()
    else:
        print("Forward correctness FAILED -- skipping backward and timing tests.")
        print("Fix the forward kernel first.")
        bwd_ok = False

    print("=" * 70)
    print(f"SUMMARY: fwd={'PASS' if fwd_ok else 'FAIL'}, bwd={'PASS' if bwd_ok else 'FAIL'}")
    print("=" * 70)

    sys.exit(0 if (fwd_ok and bwd_ok) else 1)


if __name__ == "__main__":
    main()
