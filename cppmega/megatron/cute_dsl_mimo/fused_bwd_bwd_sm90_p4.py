"""Phase 4: Full bwd_bwd kernel with ALL outputs computed inside CuTe DSL.

WGMMA (K-major both) computes A @ B^T.
TileLang GEMMs that use A @ B (no transpose_B) get B^T pre-transposed on host.
TileLang GEMM9 (transpose_A) handled by in-kernel dki transpose via smem.

GEMM mapping (TileLang -> CuTe DSL):
  1. K @ dstates      (no tr_B)   -> sK @ sDstT    where DstT = dstates^T from host
  2. K @ Q^T           (tr_B)     -> sK @ sQ        (native)
  3. lkq @ DPhiO       (no tr_B)  -> sLKQ @ sDPhT   where DPhT = DPhiO^T from host
  4. DPhiO @ PsiV^T    (tr_B)     -> sDPh @ sPsi    (native)
  5. PsiV @ dstates^T  (tr_B)     -> sPsi @ sDst    where Dst = dstates from host (native)
  6. PsiV @ DPhiO^T    (tr_B)     -> sPsi @ sDPh    (native)
  7. dki @ Q            (no tr_B)  -> sDKI @ sQT     where QT = Q^T from host
  8. DPhiO @ states^T  (tr_B)     -> sDPh @ sSts    (native)
  9. dki^T @ K          (tr_A)    -> sDKI_T @ sKT   where DKI_T = in-kernel transpose, KT = K^T from host
"""
import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import torch
import math
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass import BFloat16, Float32, Int32, Boolean, const_expr
from cutlass.cute import arch
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.runtime import make_fake_tensor, from_dlpack
from cutlass.utils import LayoutEnum, SmemAllocator
import cutlass.utils.hopper_helpers as sm90_utils_basic

from quack import sm90_utils, copy_utils, layout_utils
try:
    from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
except Exception:
    def assume_tensor_aligned(x):
        return x

import cuda.bindings.driver as cuda


def _warp_id() -> int:
    return arch.thread_idx()[0] // 32


def _is_producer_warp() -> bool:
    return _warp_id() == 0


def _set_warpgroup_reg_budget(is_producer: bool):
    if not hasattr(warpgroup, "setmaxregister"):
        return
    try:
        if is_producer:
            warpgroup.setmaxregister(40, increase=False)
        else:
            warpgroup.setmaxregister(232, increase=True)
    except TypeError:
        try:
            if is_producer:
                warpgroup.setmaxregister(40)
            else:
                warpgroup.setmaxregister(232)
        except Exception:
            pass
    except Exception:
        pass


def _has_async_pipeline_surface() -> bool:
    arch_has_barrier = (
        hasattr(arch, "mbarrier_init")
        and hasattr(arch, "mbarrier_arrive")
        and hasattr(arch, "mbarrier_try_wait")
    )
    arch_has_bulk = (
        hasattr(arch, "cp_async_bulk_tensor_2d_global_to_shared")
        or hasattr(arch, "cp_async_bulk")
    )
    pipeline_has_helpers = (
        hasattr(cutlass_pipeline, "Pipeline")
        or hasattr(cutlass_pipeline, "MbarrierArray")
        or hasattr(cutlass_pipeline, "make_pipeline")
    )
    return arch_has_barrier and arch_has_bulk and pipeline_has_helpers


class FusedBwdBwdP4:
    """Full bwd_bwd kernel with correct GEMM transposes and all outputs."""

    def __init__(self, N=64, P=64, R=4, chunk_size=16, dtype=BFloat16):
        self.N = N
        self.P = P
        self.R = R
        self.chunk_size = chunk_size
        self.fcs = chunk_size * R
        self.dtype = dtype
        self.num_threads = 128

    @cute.kernel
    def kernel(
        self,
        mK, mQ, mDst, mDPh, mPsi, mSts,
        mDstT, mDPhT, mQT, mKT,  # Pre-transposed inputs
        mDPsiV, mDK, mDQ, mDqkd, mDstatesOut,
        H,
        tiled_mma,
        sLayout_64x64,
        copy_g2s, copy_s2g,
    ):
        tidx = arch.thread_idx()[0]
        is_producer = _is_producer_warp()
        _set_warpgroup_reg_budget(is_producer)
        bidx_h = arch.block_idx()[0]
        bidx_b = arch.block_idx()[1]
        N = self.N
        P = self.P
        fcs = self.fcs

        hb = bidx_b * H + bidx_h

        gK_all = mK[hb, None, None, None]
        gQ_all = mQ[hb, None, None, None]
        gDst_all = mDst[hb, None, None, None]
        gDPh_all = mDPh[hb, None, None, None]
        gPsi_all = mPsi[hb, None, None, None]
        gSts_all = mSts[hb, None, None, None]
        gDstT_all = mDstT[hb, None, None, None]
        gDPhT_all = mDPhT[hb, None, None, None]
        gQT_all = mQT[hb, None, None, None]
        gKT_all = mKT[hb, None, None, None]

        gDPs_all = mDPsiV[hb, None, None, None]
        gDK_all = mDK[hb, None, None, None]
        gDQ_all = mDQ[hb, None, None, None]
        gDqd_all = mDqkd[hb, None, None, None]

        nchunks = cute.size(gK_all.shape[0])

        smem = SmemAllocator()
        sK0   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sK1   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQ0   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQ1   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDst0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDst1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPh0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPh1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sPsi0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sPsi1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sSts0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sSts1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDstT0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDstT1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPhT0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPhT1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQT0   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQT1   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sKT0   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sKT1   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sLKQ0  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sLKQ1  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI0  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI1  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sOutL = cute.make_layout((fcs, P), stride=(P, 1))
        sOut0 = smem.allocate_tensor(self.dtype, sOutL)
        sOut1 = smem.allocate_tensor(self.dtype, sOutL)

        thr_g2s = copy_g2s.get_slice(tidx)
        thr_s2g = copy_s2g.get_slice(tidx)
        wg_mma = tiled_mma.get_slice(tidx)
        shape_mnk = (fcs, N, N)

        if nchunks > 0:
            first_chunk_idx = nchunks - 1
            if is_producer:
                cute.copy(copy_g2s, thr_g2s.partition_S(gK_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sK0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gQ_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sQ0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gDst_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sDst0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gDPh_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sDPh0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gPsi_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sPsi0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gSts_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sSts0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gDstT_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sDstT0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sDPhT0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gQT_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sQT0))
                cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[first_chunk_idx, None, None]),
                          thr_g2s.partition_D(sKT0))
        arch.sync_threads()

        for chunk_rev in cutlass.range(nchunks, unroll=1):
            chunk_idx = nchunks - 1 - chunk_rev
            next_chunk_idx = chunk_idx - 1

            if (chunk_rev & 1) == 0:
                # GEMM1: K @ dstates (no tr_B) -> use sDstT (dstates^T)
                # WGMMA: K(fcs,N) @ sDstT(P,N)^T = K @ dstates
                _, tA1, tB1 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK0, sDstT0)
                acc_dPsiV = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dPsiV, tA1, tB1, zero_init=True, wg_wait=0)

                # GEMM2: K @ Q^T (tr_B) -> native
                _, tA2, tB2 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK0, sQ0)
                acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_lkq, tA2, tB2, zero_init=True, wg_wait=0)

                # GEMM4: DPhiO @ PsiV^T (tr_B) -> native
                _, tA4, tB4 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh0, sPsi0)
                acc_dqkd = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dqkd, tA4, tB4, zero_init=True, wg_wait=0)

                # GEMM5: PsiV @ dstates^T (tr_B) -> native, use sDst
                _, tA5, tB5 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi0, sDst0)
                acc_dk = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dk, tA5, tB5, zero_init=True, wg_wait=0)

                # GEMM6: PsiV @ DPhiO^T (tr_B) -> native
                _, tA6, tB6 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi0, sDPh0)
                acc_dki = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dki, tA6, tB6, zero_init=True, wg_wait=0)

                # GEMM8: DPhiO @ states^T (tr_B) -> native
                _, tA8, tB8 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh0, sSts0)
                acc_dq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dq, tA8, tB8, zero_init=True, wg_wait=0)

                if next_chunk_idx >= 0:
                    if is_producer:
                        cute.copy(copy_g2s, thr_g2s.partition_S(gK_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sK1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gQ_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sQ1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDst_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDst1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDPh_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDPh1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gPsi_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sPsi1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gSts_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sSts1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDstT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDstT1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDPhT1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gQT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sQT1))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sKT1))

                copy_r2s_lkq, _, _ = copy_utils.get_smem_store_C(tiled_mma, sLKQ0, tidx, arch=90, position_independent=True)
                lkq_frg = layout_utils.reshape_acc_to_frgA(acc_lkq)
                lkq_cvt = cute.make_rmem_tensor_like(lkq_frg, self.dtype)
                lkq_cvt.store(lkq_frg.load().to(self.dtype))
                copy_r2s_lkq(lkq_cvt)

                copy_r2s_dki, _, _ = copy_utils.get_smem_store_C(tiled_mma, sDKI0, tidx, arch=90, position_independent=True)
                dki_frg = layout_utils.reshape_acc_to_frgA(acc_dki)
                dki_cvt = cute.make_rmem_tensor_like(dki_frg, self.dtype)
                dki_cvt.store(dki_frg.load().to(self.dtype))
                copy_r2s_dki(dki_cvt)

                arch.fence_view_async_shared()
                arch.sync_threads()

                _, tA3, tB3 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ0, sDPhT0)
                sm90_utils.gemm(tiled_mma, acc_dPsiV, tA3, tB3, zero_init=False, wg_wait=0)

                _, tA7, tB7 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDKI0, sQT0)
                sm90_utils.gemm(tiled_mma, acc_dk, tA7, tB7, zero_init=False, wg_wait=0)

                copy_r2s_dqkd, _, _ = copy_utils.get_smem_store_C(tiled_mma, sLKQ0, tidx, arch=90, position_independent=True)
                dqkd_frg = layout_utils.reshape_acc_to_frgA(acc_dqkd)
                dqkd_cvt = cute.make_rmem_tensor_like(dqkd_frg, self.dtype)
                dqkd_cvt.store(dqkd_frg.load().to(self.dtype))
                copy_r2s_dqkd(dqkd_cvt)

                arch.fence_view_async_shared()
                arch.sync_threads()

                _, tA9, tB9 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ0, sKT0)
                sm90_utils.gemm(tiled_mma, acc_dq, tA9, tB9, zero_init=False, wg_wait=0)

                gDPs_c = gDPs_all[chunk_idx, None, None]
                gDK_c = gDK_all[chunk_idx, None, None]
                gDQ_c = gDQ_all[chunk_idx, None, None]
                gDqd_c = gDqd_all[chunk_idx, None, None]

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut0, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dPsiV)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut0), thr_s2g.partition_D(gDPs_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut0, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dk)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut0), thr_s2g.partition_D(gDK_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut0, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dq)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut0), thr_s2g.partition_D(gDQ_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut0, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dqkd)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut0), thr_s2g.partition_D(gDqd_c))
                arch.sync_threads()
            else:
                _, tA1, tB1 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK1, sDstT1)
                acc_dPsiV = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dPsiV, tA1, tB1, zero_init=True, wg_wait=0)

                _, tA2, tB2 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK1, sQ1)
                acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_lkq, tA2, tB2, zero_init=True, wg_wait=0)

                _, tA4, tB4 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh1, sPsi1)
                acc_dqkd = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dqkd, tA4, tB4, zero_init=True, wg_wait=0)

                _, tA5, tB5 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi1, sDst1)
                acc_dk = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dk, tA5, tB5, zero_init=True, wg_wait=0)

                _, tA6, tB6 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi1, sDPh1)
                acc_dki = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dki, tA6, tB6, zero_init=True, wg_wait=0)

                _, tA8, tB8 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh1, sSts1)
                acc_dq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
                sm90_utils.gemm(tiled_mma, acc_dq, tA8, tB8, zero_init=True, wg_wait=0)

                if next_chunk_idx >= 0:
                    if is_producer:
                        cute.copy(copy_g2s, thr_g2s.partition_S(gK_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sK0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gQ_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sQ0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDst_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDst0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDPh_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDPh0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gPsi_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sPsi0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gSts_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sSts0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDstT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDstT0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sDPhT0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gQT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sQT0))
                        cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sKT0))

                copy_r2s_lkq, _, _ = copy_utils.get_smem_store_C(tiled_mma, sLKQ1, tidx, arch=90, position_independent=True)
                lkq_frg = layout_utils.reshape_acc_to_frgA(acc_lkq)
                lkq_cvt = cute.make_rmem_tensor_like(lkq_frg, self.dtype)
                lkq_cvt.store(lkq_frg.load().to(self.dtype))
                copy_r2s_lkq(lkq_cvt)

                copy_r2s_dki, _, _ = copy_utils.get_smem_store_C(tiled_mma, sDKI1, tidx, arch=90, position_independent=True)
                dki_frg = layout_utils.reshape_acc_to_frgA(acc_dki)
                dki_cvt = cute.make_rmem_tensor_like(dki_frg, self.dtype)
                dki_cvt.store(dki_frg.load().to(self.dtype))
                copy_r2s_dki(dki_cvt)

                arch.fence_view_async_shared()
                arch.sync_threads()

                _, tA3, tB3 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ1, sDPhT1)
                sm90_utils.gemm(tiled_mma, acc_dPsiV, tA3, tB3, zero_init=False, wg_wait=0)

                _, tA7, tB7 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDKI1, sQT1)
                sm90_utils.gemm(tiled_mma, acc_dk, tA7, tB7, zero_init=False, wg_wait=0)

                copy_r2s_dqkd, _, _ = copy_utils.get_smem_store_C(tiled_mma, sLKQ1, tidx, arch=90, position_independent=True)
                dqkd_frg = layout_utils.reshape_acc_to_frgA(acc_dqkd)
                dqkd_cvt = cute.make_rmem_tensor_like(dqkd_frg, self.dtype)
                dqkd_cvt.store(dqkd_frg.load().to(self.dtype))
                copy_r2s_dqkd(dqkd_cvt)

                arch.fence_view_async_shared()
                arch.sync_threads()

                _, tA9, tB9 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ1, sKT1)
                sm90_utils.gemm(tiled_mma, acc_dq, tA9, tB9, zero_init=False, wg_wait=0)

                gDPs_c = gDPs_all[chunk_idx, None, None]
                gDK_c = gDK_all[chunk_idx, None, None]
                gDQ_c = gDQ_all[chunk_idx, None, None]
                gDqd_c = gDqd_all[chunk_idx, None, None]

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut1, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dPsiV)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut1), thr_s2g.partition_D(gDPs_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut1, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dk)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut1), thr_s2g.partition_D(gDK_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut1, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dq)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut1), thr_s2g.partition_D(gDQ_c))
                arch.sync_threads()

                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut1, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc_dqkd)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut1), thr_s2g.partition_D(gDqd_c))
                arch.sync_threads()

    @cute.jit
    def __call__(self, mK, mQ, mDst, mDPh, mPsi, mSts,
                 mDstT, mDPhT, mQT, mKT,
                 mDPsiV, mDK, mDQ, mDqkd, mDstatesOut, H, B, stream):
        fcs = self.fcs
        N = self.N

        mK = assume_tensor_aligned(mK)
        mQ = assume_tensor_aligned(mQ)
        mDst = assume_tensor_aligned(mDst)
        mDPh = assume_tensor_aligned(mDPh)
        mPsi = assume_tensor_aligned(mPsi)
        mSts = assume_tensor_aligned(mSts)
        mDstT = assume_tensor_aligned(mDstT)
        mDPhT = assume_tensor_aligned(mDPhT)
        mQT = assume_tensor_aligned(mQT)
        mKT = assume_tensor_aligned(mKT)
        mDPsiV = assume_tensor_aligned(mDPsiV)
        mDK = assume_tensor_aligned(mDK)
        mDQ = assume_tensor_aligned(mDQ)
        mDqkd = assume_tensor_aligned(mDqkd)
        mDstatesOut = assume_tensor_aligned(mDstatesOut)

        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, (1, 1, 1), (fcs, fcs),
            warpgroup.OperandSource.SMEM,
        )

        sLayout_64x64 = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (fcs, N)
        )

        vec = 128 // self.dtype.width
        tpr = N // vec
        ca = cute.make_copy_atom(CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
        copy_g2s = cute.make_tiled_copy_tv(
            ca,
            cute.make_ordered_layout((self.num_threads // tpr, tpr), order=(1, 0)),
            cute.make_layout((1, vec)),
        )
        copy_s2g = cute.make_tiled_copy_tv(
            ca,
            cute.make_ordered_layout((self.num_threads // tpr, tpr), order=(1, 0)),
            cute.make_layout((1, vec)),
        )

        self.kernel(
            mK, mQ, mDst, mDPh, mPsi, mSts,
            mDstT, mDPhT, mQT, mKT,
            mDPsiV, mDK, mDQ, mDqkd, mDstatesOut, H,
            tiled_mma, sLayout_64x64, copy_g2s, copy_s2g,
        ).launch(
            grid=(H, B, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


_CACHE_P4 = {}

def _get_compiled(N, P, R, chunk_size, nchunks, H, B):
    key = (N, P, R, chunk_size, nchunks, H, B)
    if key not in _CACHE_P4:
        fcs = chunk_size * R
        obj = FusedBwdBwdP4(N, P, R, chunk_size, BFloat16)
        total = B * H

        def mk4d(rows, cols):
            shape = (total, nchunks, rows, cols)
            stride = (nchunks * rows * cols, rows * cols, cols, 1)
            return make_fake_tensor(BFloat16, shape, stride=stride, assumed_align=16)

        stream_arg = cuda.CUstream(0)
        compiled = cute.compile(
            obj,
            mk4d(fcs, N),   # K
            mk4d(fcs, N),   # Q
            mk4d(N, P),     # Dst (dstates)
            mk4d(fcs, P),   # DPh (DPhiO)
            mk4d(fcs, P),   # Psi (PsiV)
            mk4d(N, P),     # Sts (states)
            mk4d(P, N),     # DstT (dstates^T)
            mk4d(P, fcs),   # DPhT (DPhiO^T)
            mk4d(N, fcs),   # QT (Q^T)
            mk4d(N, fcs),   # KT (K^T)
            mk4d(fcs, P),   # DPsiV out
            mk4d(fcs, N),   # DK out
            mk4d(fcs, N),   # DQ out
            mk4d(fcs, fcs), # Dqkd out
            mk4d(N, P),     # Dstates out
            H, B, stream_arg,
        )
        _CACHE_P4[key] = compiled
    return _CACHE_P4[key]


def run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
           DPsiV, DK, DQ, Dqkd, DstatesOut, nchunks, H, B, stream):
    """Run P4 kernel. Auto-creates transposed inputs."""
    DstT = Dst.transpose(-1, -2).contiguous()
    DPhT = DPh.transpose(-1, -2).contiguous()
    QT = Q.transpose(-1, -2).contiguous()
    KT = K.transpose(-1, -2).contiguous()

    compiled = _get_compiled(N, P, R, chunk_size, nchunks, H, B)
    dl = lambda t: from_dlpack(t, assumed_align=16)
    compiled(dl(K), dl(Q), dl(Dst), dl(DPh), dl(Psi), dl(Sts),
             dl(DstT), dl(DPhT), dl(QT), dl(KT),
             dl(DPsiV), dl(DK), dl(DQ), dl(Dqkd), dl(DstatesOut),
             H, B, stream)


def test_p4():
    """Test Phase 4 against TileLang-correct references."""
    N, P, R, chunk_size = 64, 64, 4, 16
    B, S, H = 1, 256, 4
    nchunks = S // chunk_size
    fcs = chunk_size * R
    total = B * H

    torch.manual_seed(0)
    dev = "cuda"
    # K (fcs, N), Q (fcs, N), dstates (N, P), DPhiO (fcs, P), PsiV (fcs, P), states (N, P)
    K = torch.randn(total, nchunks, fcs, N, dtype=torch.bfloat16, device=dev)
    Q = torch.randn(total, nchunks, fcs, N, dtype=torch.bfloat16, device=dev)
    Dst = torch.randn(total, nchunks, N, P, dtype=torch.bfloat16, device=dev)
    DPh = torch.randn(total, nchunks, fcs, P, dtype=torch.bfloat16, device=dev)
    Psi = torch.randn(total, nchunks, fcs, P, dtype=torch.bfloat16, device=dev)
    Sts = torch.randn(total, nchunks, N, P, dtype=torch.bfloat16, device=dev)
    DPsiV = torch.zeros(total, nchunks, fcs, P, dtype=torch.bfloat16, device=dev)
    DK = torch.zeros(total, nchunks, fcs, N, dtype=torch.bfloat16, device=dev)
    DQ = torch.zeros(total, nchunks, fcs, N, dtype=torch.bfloat16, device=dev)
    Dqkd = torch.zeros(total, nchunks, fcs, fcs, dtype=torch.bfloat16, device=dev)
    DstO = torch.zeros(total, nchunks, N, P, dtype=torch.bfloat16, device=dev)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    import time
    t0 = time.time()
    run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
           DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
    torch.cuda.synchronize()
    print(f"Compile+launch: {time.time()-t0:.2f}s")

    # Reference: TileLang formulas
    errs = {}
    for hb in range(min(total, 2)):
        for c in range(min(nchunks, 4)):
            k = K[hb, c].float()    # (fcs, N)
            q = Q[hb, c].float()    # (fcs, N)
            dst = Dst[hb, c].float()  # dstates (N, P)
            dph = DPh[hb, c].float()  # DPhiO (fcs, P)
            psi = Psi[hb, c].float()  # PsiV (fcs, P)
            sts = Sts[hb, c].float()  # states (N, P)

            # GEMM1: K @ dstates -> (fcs, P)
            g1 = k @ dst

            # GEMM2: K @ Q^T -> (fcs, fcs)
            g2 = k @ q.T
            g2_bf = g2.bfloat16().float()

            # GEMM3: lkq @ DPhiO -> (fcs, P)
            g3 = g2_bf @ dph
            dpsiV_ref = g1 + g3

            # GEMM4: DPhiO @ PsiV^T -> (fcs, fcs)
            dqkd_ref = dph @ psi.T

            # GEMM5: PsiV @ dstates^T -> (fcs, N)
            g5 = psi @ dst.T

            # GEMM6: PsiV @ DPhiO^T -> (fcs, fcs)
            g6 = psi @ dph.T
            g6_bf = g6.bfloat16().float()

            # GEMM7: dki @ Q -> (fcs, N)
            dk_ref = g5 + g6_bf @ q

            # GEMM8: DPhiO @ states^T -> (fcs, N)
            dq_ref = dph @ sts.T

            # GEMM9: dki^T @ K -> (fcs, N)
            # dki^T = dqkd = DPhiO @ PsiV^T, bf16-rounded during R2S
            dqkd_bf = dqkd_ref.bfloat16().float()
            dq_ref = dq_ref + dqkd_bf @ k

            def relcheck(got, ref, tag):
                diff = (got.float() - ref).abs()
                me = diff.max().item()
                mr = ref.abs().max().item()
                rel = me / max(mr, 1e-8)
                errs.setdefault(tag, []).append(rel)

            relcheck(DPsiV[hb, c], dpsiV_ref, "dPsiV(G1+G3)")
            relcheck(Dqkd[hb, c], dqkd_ref, "dqkd(G4)")
            relcheck(DK[hb, c], dk_ref, "dk(G5+G7)")
            relcheck(DQ[hb, c], dq_ref, "dq(G8+G9)")

    print("=== Phase 4 correctness (TileLang-correct refs) ===")
    for tag, rels in errs.items():
        mx = max(rels)
        avg = sum(rels) / len(rels)
        print(f"  {tag:20}: max rel={mx:.4f} avg={avg:.4f} {'PASS' if mx < 0.05 else 'FAIL'}")

    overall = max(max(rels) for rels in errs.values())
    status = "PASS" if overall < 0.05 else "FAIL"
    print(f"Overall: max rel = {overall:.4f} -> {status}")

    # Benchmark
    if status == "PASS":
        for _ in range(10):
            run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
                   DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        iters = 200
        s.record()
        for _ in range(iters):
            run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
                   DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000 / iters
        print(f"Benchmark: {us:.1f} us (smoke B={B} S={S} H={H})")

    return overall < 0.05


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    test_p4()
