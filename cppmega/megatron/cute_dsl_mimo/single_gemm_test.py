"""
Phase 1: Single-GEMM CuTe DSL WGMMA kernel on sm_90a (H200).
C[M,N] = A[M,K] @ B[N,K]^T, BF16->F32 acc->BF16 out.
WGMMA (warpgroup.MmaF16BF16Op) with pointer-swizzled smem (PDSL).
Epilogue: acc -> smem via StMatrix -> gmem via vectorized copy.
"""
import os
os.environ.setdefault('CUTE_DSL_ARCH', 'sm_90a')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import torch
import time

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Boolean, const_expr
from cutlass.cute import arch
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.runtime import make_fake_tensor, from_dlpack
from cutlass.utils import LayoutEnum, SmemAllocator
import cutlass.utils.hopper_helpers as sm90_utils_basic

from quack import sm90_utils, copy_utils

import cuda.bindings.driver as cuda


class SingleGemmWGMMA:
    def __init__(self, M=64, N=64, K=64, dtype=BFloat16):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.num_threads = 128

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        copy_g2s: cute.TiledCopy,
        copy_s2g: cute.TiledCopy,
    ):
        tidx = arch.thread_idx()[0]

        smem = SmemAllocator()
        sA = smem.allocate_tensor(self.dtype, sA_layout.outer, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(self.dtype, sB_layout.outer, swizzle=sB_layout.inner)
        sC = smem.allocate_tensor(self.dtype, sC_layout)

        # GMEM -> SMEM
        thr_g2s = copy_g2s.get_slice(tidx)
        cute.copy(copy_g2s, thr_g2s.partition_S(gA), thr_g2s.partition_D(sA))
        cute.copy(copy_g2s, thr_g2s.partition_S(gB), thr_g2s.partition_D(sB))
        arch.sync_threads()

        # WGMMA
        thr_mma = tiled_mma.get_slice(tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self.M, self.N)), Float32
        )

        sm90_utils.gemm(tiled_mma, acc, tCrA, tCrB, zero_init=True, wg_wait=0)

        # Epilogue: acc (F32 regs) -> sC (BF16 smem) -> gC (BF16 gmem)
        # Use get_smem_store_C which handles WGMMA partition -> smem correctly
        copy_r2s_fn, _, _ = copy_utils.get_smem_store_C(
            tiled_mma, sC, tidx, arch=90
        )
        copy_r2s_fn(acc)
        arch.sync_threads()

        # sC -> gC with simple vectorized copy
        thr_s2g = copy_s2g.get_slice(tidx)
        cute.copy(copy_s2g, thr_s2g.partition_S(sC), thr_s2g.partition_D(gC))

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream,
    ):
        M, N, K = self.M, self.N, self.K

        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            a_dtype=self.dtype,
            b_dtype=self.dtype,
            a_leading_mode=warpgroup.OperandMajorMode.K,
            b_leading_mode=warpgroup.OperandMajorMode.K,
            acc_dtype=Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(M, N),
            a_source=warpgroup.OperandSource.SMEM,
        )

        sA_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (M, K)
        )
        sB_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (N, K)
        )
        sC_layout = cute.make_layout((M, N), stride=(N, 1))

        vec = 128 // self.dtype.width  # 8
        copy_atom = cute.make_copy_atom(
            CopyUniversalOp(), self.dtype, num_bits_per_copy=128
        )
        copy_g2s = cute.make_tiled_copy_tv(
            copy_atom, cute.make_layout(self.num_threads), cute.make_layout(vec)
        )
        copy_s2g = cute.make_tiled_copy_tv(
            copy_atom, cute.make_layout(self.num_threads), cute.make_layout(vec)
        )

        self.kernel(
            gA=mA, gB=mB, gC=mC,
            tiled_mma=tiled_mma,
            sA_layout=sA_layout,
            sB_layout=sB_layout,
            sC_layout=sC_layout,
            copy_g2s=copy_g2s,
            copy_s2g=copy_s2g,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


_CACHE = {}

def run_single_gemm(M, N, K, A, B, C_out, stream):
    key = (M, N, K)
    compiled = _CACHE.get(key)
    if compiled is None:
        kernel_obj = SingleGemmWGMMA(M, N, K, BFloat16)
        _mA = make_fake_tensor(BFloat16, (M, K), stride=(K, 1), assumed_align=16)
        _mB = make_fake_tensor(BFloat16, (N, K), stride=(K, 1), assumed_align=16)
        _mC = make_fake_tensor(BFloat16, (M, N), stride=(N, 1), assumed_align=16)
        compiled = cute.compile(kernel_obj, _mA, _mB, _mC, stream)
        _CACHE[key] = compiled
    mA = from_dlpack(A, assumed_align=16)
    mB = from_dlpack(B, assumed_align=16)
    mC = from_dlpack(C_out, assumed_align=16)
    compiled(mA, mB, mC, stream)


def run_phase1():
    M, N, K = 64, 64, 64
    print(f'Phase 1: Single WGMMA GEMM {M}x{N}x{K} BF16 on sm_90a')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    C_ref = (A.float() @ B.float().T).to(torch.bfloat16)
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    
    print(f'Reference C[0,:4]: {C_ref[0,:4]}')

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        print('Compiling WGMMA kernel...')
        t0 = time.time()
        run_single_gemm(M, N, K, A, B, C_out, stream)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f'First run (compile+launch): {t1-t0:.2f}s')
        
        max_err = (C_out.float() - C_ref.float()).abs().max().item()
        rel_err = max_err / C_ref.float().abs().max().item()
        print(f'Max absolute error: {max_err:.6f}')
        print(f'Max relative error: {rel_err:.6f}')
        print(f'C_out[0,:4]: {C_out[0,:4]}')
        print(f'C_ref[0,:4]: {C_ref[0,:4]}')
        
        passed = max_err < 1.0
        print(f'{PASS if passed else FAIL}: Correctness')
        
        if not passed:
            return False, None
        
        for _ in range(10):
            run_single_gemm(M, N, K, A, B, C_out, stream)
        torch.cuda.synchronize()
        
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        n_iter = 1000
        start_ev.record()
        for _ in range(n_iter):
            run_single_gemm(M, N, K, A, B, C_out, stream)
        end_ev.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_ev.elapsed_time(end_ev)
        per_iter_us = elapsed_ms * 1000 / n_iter
        tflops = (2 * M * N * K) / (per_iter_us * 1e-6) / 1e12
        print(f'Timing: {per_iter_us:.2f} us/iter ({n_iter} iters)')
        print(f'Throughput: {tflops:.4f} TFLOPS')
        
        return True, per_iter_us
        
    except Exception as e:
        print(f'WGMMA kernel failed: {e}')
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == '__main__':
    run_phase1()
