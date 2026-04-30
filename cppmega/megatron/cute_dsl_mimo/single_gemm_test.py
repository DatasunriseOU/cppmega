"""
Phase 1: Single-GEMM CuTe DSL WGMMA kernel on sm_90a (H200).
C[M,N] = A[M,K] @ B[N,K]^T, BF16->F32 acc->BF16 out.
WGMMA (warpgroup.MmaF16BF16Op) with pointer-swizzled smem (PDSL).
Epilogue: acc -> smem via StMatrix -> gmem via scalar copy.
"""
import os
os.environ.setdefault('CUTE_DSL_ARCH', 'sm_90a')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import math
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


def _make_row_major_tiled_copy(
    dtype, major_mode_size: int, num_threads: int, copy_bits: int = 128
):
    """Build a 2D row-major copy tiler matching CuTe DSL 4.4.x examples."""
    copy_elems = copy_bits // dtype.width
    if major_mode_size % copy_elems != 0:
        raise ValueError(
            f"major_mode_size={major_mode_size} must be divisible by copy_elems={copy_elems}"
        )
    loads_per_cache_line = 128 * 8 // copy_bits
    threads_per_row = major_mode_size // copy_elems
    if threads_per_row > loads_per_cache_line:
        threads_per_row = math.gcd(threads_per_row, loads_per_cache_line)
    if num_threads % threads_per_row != 0:
        raise ValueError(
            f"num_threads={num_threads} must be divisible by threads_per_row={threads_per_row}"
        )
    copy_atom = cute.make_copy_atom(
        CopyUniversalOp(), dtype, num_bits_per_copy=copy_bits
    )
    thread_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)


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
        sC_layout: cute.ComposedLayout,
        copy_g2s: cute.TiledCopy,
        copy_s2g: cute.TiledCopy,
    ):
        tidx = arch.thread_idx()[0]

        smem = SmemAllocator()
        sA = smem.allocate_tensor(self.dtype, sA_layout.outer, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(self.dtype, sB_layout.outer, swizzle=sB_layout.inner)
        # Keep C staging in the same pointer-swizzled form as A/B.  Passing the
        # composed layout directly builds a position-independent swizzle tensor,
        # which compiles but scrambles parts of the stmatrix -> gmem epilogue.
        sC = smem.allocate_tensor(self.dtype, sC_layout.outer, swizzle=sC_layout.inner)

        # GMEM -> SMEM
        thr_g2s = copy_g2s.get_slice(tidx)
        cute.copy(copy_g2s, thr_g2s.partition_S(gA), thr_g2s.partition_D(sA))
        cute.copy(copy_g2s, thr_g2s.partition_S(gB), thr_g2s.partition_D(sB))
        arch.sync_threads()

        # WGMMA
        thr_mma = tiled_mma.get_slice(tidx)
        acc, tCrA, tCrB = sm90_utils.partition_fragment_ABC(
            thr_mma, (self.M, self.N, self.K), sA, sB
        )

        sm90_utils.gemm(tiled_mma, acc, tCrA, tCrB, zero_init=True, wg_wait=0)

        # Epilogue: acc (F32 regs) -> sC (BF16 smem) -> gC (BF16 gmem).
        # Match quack's SM90 epilogue: retile the accumulator into the exact
        # R2S register layout before the stmatrix store.
        tiled_copy_C_atom = copy_utils.epilog_smem_copy_atom(
            tiled_mma, (self.M, self.N)
        )
        copy_atom_r2s = sm90_utils_basic.sm90_get_smem_store_op(
            LayoutEnum.ROW_MAJOR, elem_ty_d=self.dtype, elem_ty_acc=Float32
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC_shape = thr_copy_r2s.partition_S(
            cute.make_identity_tensor(sC.shape[:2])
        ).shape
        tRS_rC = cute.make_rmem_tensor(tRS_rC_shape, Float32)
        tRS_rAcc = cute.flat_divide(acc, tRS_rC.layout)
        cute.autovec_copy(tRS_rAcc[None, None, None, 0], tRS_rC)
        copy_utils.cvt_copy(tiled_copy_r2s, tRS_rC, tRS_sC)
        arch.sync_threads()

        # sC -> gC.  Use scalar copies for correctness with swizzled smem.
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
        sC_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (M, N)
        )

        copy_g2s = _make_row_major_tiled_copy(
            self.dtype, K, self.num_threads, copy_bits=self.dtype.width
        )
        copy_s2g = _make_row_major_tiled_copy(
            self.dtype, N, self.num_threads, copy_bits=self.dtype.width
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


def _make_deterministic_inputs(M, N, K):
    row_m = torch.arange(M, dtype=torch.float32, device='cuda')[:, None]
    row_n = torch.arange(N, dtype=torch.float32, device='cuda')[:, None]
    col_k = torch.arange(K, dtype=torch.float32, device='cuda')[None, :]

    identity_A = torch.eye(M, K, dtype=torch.float32, device='cuda')
    identity_B = (((row_n * 7 + col_k * 3) % 23) - 11) / 7.0

    structured_A = (((row_m * 3 + col_k * 5) % 17) - 8) / 8.0
    structured_B = (((row_n * 7 - col_k * 2) % 19) - 9) / 9.0

    torch.manual_seed(42)
    random_A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').float()
    random_B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').float()

    return [
        ('identity_transpose', identity_A.to(torch.bfloat16), identity_B.to(torch.bfloat16)),
        ('structured_mod', structured_A.to(torch.bfloat16), structured_B.to(torch.bfloat16)),
        ('random_seed_42', random_A.to(torch.bfloat16), random_B.to(torch.bfloat16)),
    ]


def _print_diff_diagnostics(case_name, C_out, C_ref):
    diff = (C_out.float() - C_ref.float()).abs()
    flat = diff.flatten()
    topk = min(8, flat.numel())
    vals, idxs = torch.topk(flat, topk)
    row_max = diff.max(dim=1).values
    col_max = diff.max(dim=0).values
    worst_row = int(row_max.argmax().item())
    worst_col = int(col_max.argmax().item())

    print(f'  Diagnostics for {case_name}:')
    for rank in range(topk):
        flat_idx = int(idxs[rank].item())
        i = flat_idx // C_ref.shape[1]
        j = flat_idx % C_ref.shape[1]
        print(
            '    '
            f'#{rank + 1}: ({i},{j}) '
            f'out={C_out[i, j].float().item():.6f} '
            f'ref={C_ref[i, j].float().item():.6f} '
            f'abs={vals[rank].item():.6f}'
        )
    print(
        f'    worst_row={worst_row} row_max={row_max[worst_row].item():.6f} '
        f'worst_col={worst_col} col_max={col_max[worst_col].item():.6f}'
    )
    print(f'    C_out[0:4,0:8]:\n{C_out[:4, :8]}')
    print(f'    C_ref[0:4,0:8]:\n{C_ref[:4, :8]}')
    r0 = max(0, worst_row - 1)
    r1 = min(C_ref.shape[0], worst_row + 2)
    c0 = max(0, worst_col - 4)
    c1 = min(C_ref.shape[1], worst_col + 5)
    print(f'    C_out[{r0}:{r1},{c0}:{c1}]:\n{C_out[r0:r1, c0:c1]}')
    print(f'    C_ref[{r0}:{r1},{c0}:{c1}]:\n{C_ref[r0:r1, c0:c1]}')


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

    cases = _make_deterministic_inputs(M, N, K)
    C_out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        print('Compiling WGMMA kernel...')
        t0 = time.time()
        first_A, first_B = cases[0][1], cases[0][2]
        run_single_gemm(M, N, K, first_A, first_B, C_out, stream)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f'First run (compile+launch): {t1-t0:.2f}s')

        max_err = 0.0
        rel_err = 0.0
        for case_name, A, B in cases:
            C_out.zero_()
            run_single_gemm(M, N, K, A, B, C_out, stream)
            torch.cuda.synchronize()

            C_ref = (A.float() @ B.float().T).to(torch.bfloat16)
            case_max_err = (C_out.float() - C_ref.float()).abs().max().item()
            case_rel_err = case_max_err / max(C_ref.float().abs().max().item(), 1.0)
            max_err = max(max_err, case_max_err)
            rel_err = max(rel_err, case_rel_err)
            print(
                f'Case {case_name}: max_abs={case_max_err:.6f} '
                f'max_rel={case_rel_err:.6f}'
            )
            print(f'  C_out[0,:4]: {C_out[0,:4]}')
            print(f'  C_ref[0,:4]: {C_ref[0,:4]}')
            if case_max_err >= 1.0:
                _print_diff_diagnostics(case_name, C_out, C_ref)
                print('FAIL: Correctness')
                return False, None

        print(f'Max absolute error across cases: {max_err:.6f}')
        print(f'Max relative error across cases: {rel_err:.6f}')
        print('PASS: Correctness')

        A, B = cases[-1][1], cases[-1][2]
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
