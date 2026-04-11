"""
Test suite for FA4-pattern fused 3-GEMM kernels.

Tests correctness and timing of:
- FA4-pattern v2 (position-independent R2S, no intermediate output)
- Comparison against naive fused (smem->smem copy) and torch.mm

Run: CUDA_VISIBLE_DEVICES=0 python -m cppmega.megatron.cute_dsl_mimo.test_fa4_adapter
"""
import os
os.environ['CUTE_DSL_ARCH'] = 'sm_90a'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch
import time
import cuda.bindings.driver as cuda


def bench(label, fn, n=5000):
    """Benchmark a function, return microseconds per iteration."""
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000 / n
    print(f'  {label}: {us:.2f} us/iter')
    return us


def test_correctness():
    """Verify FA4-pattern v2 matches naive fused output."""
    from cppmega.megatron.cute_dsl_mimo.fa4_bwd_adapter_v2 import run_fa4_v2
    from cppmega.megatron.cute_dsl_mimo.fa4_bwd_adapter import run_fa4_fused as run_fused

    DIM = 64
    print(f'\n=== Correctness: {DIM}x{DIM} BF16 ===')

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Test multiple seeds
    all_pass = True
    for seed in [42, 123, 999]:
        torch.manual_seed(seed)
        K = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        Q = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        Dst = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        DPh = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        DstT = Dst.T.contiguous()
        DPhT = DPh.T.contiguous()

        # Naive reference
        DPs_n = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        LKQ_n = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        run_fused(DIM, K, Q, DstT, DPhT, DPs_n, LKQ_n, stream)

        # FA4 v2
        DPs_v2 = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
        run_fa4_v2(DIM, K, Q, DstT, DPhT, DPs_v2, stream)
        torch.cuda.synchronize()

        err = (DPs_v2.float() - DPs_n.float()).abs().max().item()
        ok = err < 2.0  # bf16 tolerance
        print(f'  seed={seed}: max err={err:.4f} {"PASS" if ok else "FAIL"}')
        all_pass = all_pass and ok

    print(f'Correctness: {"ALL PASS" if all_pass else "FAIL"}')
    return all_pass


def test_benchmark():
    """Compare FA4-pattern v2 vs naive fused vs torch."""
    from cppmega.megatron.cute_dsl_mimo.fa4_bwd_adapter_v2 import run_fa4_v2
    from cppmega.megatron.cute_dsl_mimo.fa4_bwd_adapter import run_fa4_fused as run_fused

    DIM = 64
    print(f'\n=== Benchmark: {DIM}x{DIM} BF16 on {torch.cuda.get_device_name(0)} ===')

    torch.manual_seed(42)
    K = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    Q = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    Dst = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DPh = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DstT = Dst.T.contiguous()
    DPhT = DPh.T.contiguous()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    DPs_n = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    LKQ_n = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DPs_v2 = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')

    # Compile
    print('Compiling...')
    run_fused(DIM, K, Q, DstT, DPhT, DPs_n, LKQ_n, stream)
    run_fa4_v2(DIM, K, Q, DstT, DPhT, DPs_v2, stream)
    torch.cuda.synchronize()

    print()
    results = {}

    def torch_3gemm():
        g1 = K @ Dst
        g2 = K @ Q.T
        g3 = g1 + g2 @ DPh
    results['torch_3gemm'] = bench('torch 3-GEMM (3 launches)', torch_3gemm)

    def naive_fused():
        run_fused(DIM, K, Q, DstT, DPhT, DPs_n, LKQ_n, stream)
    results['naive_fused'] = bench('Naive fused (smem->smem copy)', naive_fused)

    def fa4_v2():
        run_fa4_v2(DIM, K, Q, DstT, DPhT, DPs_v2, stream)
    results['fa4_v2'] = bench('FA4 v2 (position-indep R2S)', fa4_v2)

    print()
    print('=== Results ===')
    base = results['torch_3gemm']
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        print(f'  {name:40s} {t:8.2f} us   {base/t:.2f}x vs torch')

    naive_t = results['naive_fused']
    fa4_t = results['fa4_v2']
    print()
    print(f'FA4 v2 vs naive: {naive_t/fa4_t:.3f}x ({naive_t-fa4_t:.2f} us saved)')
    print(f'FA4 v2 vs torch: {base/fa4_t:.3f}x')
    return results


if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    ok = test_correctness()
    if not ok:
        print('CORRECTNESS FAILED')
        exit(1)

    test_benchmark()
