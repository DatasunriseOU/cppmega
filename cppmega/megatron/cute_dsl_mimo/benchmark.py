"""
Benchmark: CuTe DSL WGMMA vs torch on H200 sm_90a.
"""
import os
os.environ['CUTE_DSL_ARCH'] = 'sm_90a'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch
import time
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import make_fake_tensor, from_dlpack
import cutlass.cute as cute
from cutlass import BFloat16

from cppmega.megatron.cute_dsl_mimo.single_gemm_test import run_single_gemm
from cppmega.megatron.cute_dsl_mimo.bwd_bwd_wgmma import run_fused


def bench(label, fn, n_iter=1000):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000 / n_iter
    print(f'  {label}: {us:.2f} us/iter')
    return us


def main():
    DIM = 64
    n_iter = 1000
    print(f'Benchmark: {DIM}x{DIM} BF16 on {torch.cuda.get_device_name(0)}')
    print()

    K = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    Q = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    Dst = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DPh = torch.randn(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DstT = Dst.T.contiguous()
    DPhT = DPh.T.contiguous()
    C1 = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    C2 = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    DPs = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    LKQ = torch.zeros(DIM, DIM, dtype=torch.bfloat16, device='cuda')
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile kernels
    print('Compiling CuTe DSL kernels...')
    run_single_gemm(DIM, DIM, DIM, K, Q, C1, stream)
    run_fused(DIM, K, Q, DstT, DPhT, DPs, LKQ, stream)
    torch.cuda.synchronize()
    print('Done.')
    print()

    results = {}

    # 1. torch 3 matmuls
    def torch_3gemm():
        g1 = K @ Dst
        g2 = K @ Q.T
        g3 = g1 + g2 @ DPh
    results['torch_3gemm'] = bench('torch 3-GEMM (3 launches)', torch_3gemm, n_iter)

    # 2. CuTe DSL single GEMM
    def cute_single():
        run_single_gemm(DIM, DIM, DIM, K, Q, C1, stream)
    results['cute_single'] = bench('CuTe DSL single WGMMA', cute_single, n_iter)

    # 3. CuTe DSL 3 separate launches
    def cute_3sep():
        run_single_gemm(DIM, DIM, DIM, K, Q, C1, stream)
        run_single_gemm(DIM, DIM, DIM, K, Dst, C2, stream)
        run_single_gemm(DIM, DIM, DIM, C1, DPh, DPs, stream)
    results['cute_3separate'] = bench('CuTe DSL 3 separate WGMMA', cute_3sep, n_iter)

    # 4. CuTe DSL fused 3-GEMM
    def cute_fused():
        run_fused(DIM, K, Q, DstT, DPhT, DPs, LKQ, stream)
    results['cute_fused3'] = bench('CuTe DSL fused 3-GEMM', cute_fused, n_iter)

    print()
    print('=== Results ===')
    base = results['torch_3gemm']
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        speedup = base / t
        print(f'  {name:30s} {t:8.2f} us   {speedup:.2f}x vs torch')

    print()
    sep = results['cute_3separate']
    fused = results['cute_fused3']
    print(f'Fusion benefit: {sep/fused:.2f}x (3 launches {sep:.2f}us -> 1 launch {fused:.2f}us)')


if __name__ == '__main__':
    main()
