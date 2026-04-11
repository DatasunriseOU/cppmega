"""
FA4-pattern fused 3-GEMM kernel v2: no intermediate LKQ output, minimal smem.

Key optimizations over v1:
1. Remove LKQ gmem output (not needed for mamba3 bwd_bwd chain)
2. Single swizzled smem buffer for lkq intermediate (reuse input buffer)
3. Remove redundant sLKQout buffer and gmem store
4. fence_view_async_shared only where needed

GEMMs:
  GEMM1: dPsiV  = K @ Dstates
  GEMM2: lkq    = K @ Q^T
  GEMM3: dPsiV += lkq @ DPhiO
"""
import os
os.environ.setdefault('CUTE_DSL_ARCH', 'sm_90a')

import torch
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32
from cutlass.cute import arch
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.runtime import make_fake_tensor, from_dlpack
from cutlass.utils import LayoutEnum, SmemAllocator
import cutlass.utils.hopper_helpers as sm90_utils_basic

from quack import sm90_utils, copy_utils

import cuda.bindings.driver as cuda


class FA4PatternFused3GemmV2:
    """Fused 3-GEMM v2: minimal overhead, no intermediate output."""

    def __init__(self, DIM=64, dtype=BFloat16):
        self.DIM = DIM
        self.dtype = dtype
        self.num_threads = 128

    @cute.kernel
    def kernel(
        self,
        gK, gQ, gDstT, gDPhT, gDPs,
        tiled_mma,
        sLayout,
        copy_2d,
    ):
        tidx = arch.thread_idx()[0]
        DIM = self.DIM
        smem = SmemAllocator()

        # Input buffers
        sK    = smem.allocate_tensor(self.dtype, sLayout.outer, swizzle=sLayout.inner)
        sQ    = smem.allocate_tensor(self.dtype, sLayout.outer, swizzle=sLayout.inner)
        sDstT = smem.allocate_tensor(self.dtype, sLayout.outer, swizzle=sLayout.inner)
        sDPhT = smem.allocate_tensor(self.dtype, sLayout.outer, swizzle=sLayout.inner)
        # Reuse the Q buffer for lkq after GEMM2. Q is dead once acc_lkq is computed,
        # so keeping a separate swizzled intermediate only inflates SMEM footprint.
        sLKQ = sQ
        # Output buffer
        sOut  = smem.allocate_tensor(self.dtype, sLayout.outer, swizzle=sLayout.inner)

        # Load inputs
        thr = copy_2d.get_slice(tidx)
        cute.copy(copy_2d, thr.partition_S(gK), thr.partition_D(sK))
        cute.copy(copy_2d, thr.partition_S(gQ), thr.partition_D(sQ))
        cute.copy(copy_2d, thr.partition_S(gDstT), thr.partition_D(sDstT))
        cute.copy(copy_2d, thr.partition_S(gDPhT), thr.partition_D(sDPhT))
        arch.sync_threads()

        # GEMM1: dPsiV = K @ (DstT)^T
        thr1 = tiled_mma.get_slice(tidx)
        tA1 = thr1.make_fragment_A(thr1.partition_A(sK))
        tB1 = thr1.make_fragment_B(thr1.partition_B(sDstT))
        acc = cute.make_rmem_tensor(thr1.partition_shape_C((DIM, DIM)), Float32)
        sm90_utils.gemm(tiled_mma, acc, tA1, tB1, zero_init=True, wg_wait=0)

        # GEMM2: lkq = K @ Q^T (K reused in smem, Q dies after this GEMM)
        thr2 = tiled_mma.get_slice(tidx)
        tA2 = thr2.make_fragment_A(thr2.partition_A(sK))
        tB2 = thr2.make_fragment_B(thr2.partition_B(sQ))
        acc_lkq = cute.make_rmem_tensor(thr2.partition_shape_C((DIM, DIM)), Float32)
        sm90_utils.gemm(tiled_mma, acc_lkq, tA2, tB2, zero_init=True, wg_wait=0)

        # FA4 R2S: lkq regs -> recycled sQ buffer (swizzled, position-independent)
        copy_lkq_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma, sLKQ, tidx, arch=90,
            position_independent=True,
        )
        copy_lkq_r2s(acc_lkq)
        arch.fence_view_async_shared()
        arch.sync_threads()

        # GEMM3: dPsiV += lkq @ (DPhT)^T
        thr3 = tiled_mma.get_slice(tidx)
        tA3 = thr3.make_fragment_A(thr3.partition_A(sLKQ))
        tB3 = thr3.make_fragment_B(thr3.partition_B(sDPhT))
        sm90_utils.gemm(tiled_mma, acc, tA3, tB3, zero_init=False, wg_wait=0)

        # Store result
        copy_out_fn, _, _ = copy_utils.get_smem_store_C(
            tiled_mma, sOut, tidx, arch=90,
            position_independent=True,
        )
        copy_out_fn(acc)
        arch.fence_view_async_shared()
        arch.sync_threads()
        thr_o = copy_2d.get_slice(tidx)
        cute.copy(copy_2d, thr_o.partition_S(sOut), thr_o.partition_D(gDPs))

    @cute.jit
    def __call__(self, mK, mQ, mDstT, mDPhT, mDPs, stream):
        DIM = self.DIM
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, (1,1,1), (DIM, DIM), warpgroup.OperandSource.SMEM)

        sLayout = sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, (DIM, DIM))

        vec = 128 // self.dtype.width
        tpr = DIM // vec
        ca = cute.make_copy_atom(CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
        c2d = cute.make_tiled_copy_tv(ca,
            cute.make_ordered_layout((self.num_threads//tpr, tpr), order=(1,0)),
            cute.make_layout((1, vec)))

        self.kernel(mK, mQ, mDstT, mDPhT, mDPs,
                    tiled_mma, sLayout, c2d,
        ).launch(grid=(1,1,1), block=(self.num_threads,1,1), stream=stream)


_V2_CACHE = {}

def run_fa4_v2(DIM, K, Q, DstT, DPhT, DPs, stream):
    key = (DIM, 'fa4_v2')
    compiled = _V2_CACHE.get(key)
    if compiled is None:
        obj = FA4PatternFused3GemmV2(DIM, BFloat16)
        mk = lambda s: make_fake_tensor(BFloat16, s, stride=(s[1],1), assumed_align=16)
        compiled = cute.compile(obj, mk((DIM,DIM)), mk((DIM,DIM)), mk((DIM,DIM)),
                                mk((DIM,DIM)), mk((DIM,DIM)), stream)
        _V2_CACHE[key] = compiled
    dl = lambda t: from_dlpack(t, assumed_align=16)
    compiled(dl(K), dl(Q), dl(DstT), dl(DPhT), dl(DPs), stream)
