"""Phase 4: Phase 3 + GEMM 10 loop-carried dstates accumulator.

GEMM chain per chunk:
  1. dPsiV  = K @ Dst^T            (fcs, N)
  2. lkq    = K @ Q^T              (fcs, fcs)  - R2S spilled
  3. dPsiV += lkq @ DPh^T          (fcs, N)
  4. dqkd   = DPh @ PsiV^T         (fcs, fcs)
  5. dk     = PsiV @ Dst^T         (fcs, N)
  6. dki    = PsiV @ DPh^T         (fcs, fcs)  - R2S spilled
  7. dk    += dki @ Q               (fcs, N)    (FIXED: uses pre-transposed Q_T)
  8. dq     = DPh @ Sts^T          (fcs, N)
  9. dq    += dki^T @ K             (fcs, N)    (FIXED: uses GEMM6' for dki_T + pre-transposed K_T)
 10. dstates_accum += Q_T @ DPh    (N, P)      <-- NEW: loop-carried accumulator

Input layout note for GEMM 10:
  The caller MUST pre-transpose Q to a new tensor Q_T of shape (total, nchunks, N, fcs).
  This avoids the MN-major smem-descriptor legalization failure that occurs when we
  try to reinterpret the K-major-swizzled sQ as MN-major. Q_T is loaded via the same
  row-major g2s path into a dedicated sQ_T smem tile and then fed K-major to GEMM 10
  as A = Q_T (shape (N, fcs)) -- which is exactly what WGMMA C = A @ B^T needs to
  produce `Q^T @ DPh` of shape (N, P).

Key differences vs Phase 3:
  - Accepts mQ_T: extra input, Q pre-transposed to (total, nchunks, N, fcs)
  - Allocates dedicated sQ_T smem tile
  - acc_dstates is allocated ONCE outside the chunk loop, zero-init at start
  - GEMM 10 happens each chunk, zero_init=False
  - Stores acc_dstates ONCE after the chunk loop (per-CTA, no chunk axis)
"""
import os
os.environ.setdefault('CUTE_DSL_ARCH', 'sm_90a')

import torch
import math
import cutlass
import cutlass.cute as cute
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


def _pick_bench_device():
    """Prefer an actually-available CUDA device on busy multi-GPU hosts."""
    if not torch.cuda.is_available():
        return 'cuda'

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            continue
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx
    return f'cuda:{best_idx}'


class FusedBwdBwdP4:
    """Phase 4: 10-GEMM fused bwd_bwd with loop-carried dstates."""

    def __init__(self, N=64, P=64, R=4, chunk_size=16, dtype=BFloat16):
        self.N = N
        self.P = P
        self.R = R
        self.chunk_size = chunk_size
        self.fcs = chunk_size * R  # 64
        self.dtype = dtype
        self.num_threads = 128

    @cute.kernel
    def kernel(
        self,
        mK, mK_T, mQ, mQ_T, mDst, mDPh, mDPh_T, mPsi, mSts,
        mDPsiV, mDK, mDQ, mDqkd, mDstatesOut,
        H,
        tiled_mma,
        tiled_mma_dstates,
        sLayout_64x64,
        copy_g2s, copy_s2g,
    ):
        tidx = arch.thread_idx()[0]
        bidx_h = arch.block_idx()[0]
        bidx_b = arch.block_idx()[1]
        N = self.N
        P = self.P
        fcs = self.fcs

        hb = bidx_b * H + bidx_h

        gK_all = mK[hb, None, None, None]
        gKT_all = mK_T[hb, None, None, None]
        gQ_all = mQ[hb, None, None, None]
        gQT_all = mQ_T[hb, None, None, None]
        gDst_all = mDst[hb, None, None, None]
        gDPh_all = mDPh[hb, None, None, None]
        gDPhT_all = mDPh_T[hb, None, None, None]
        gPsi_all = mPsi[hb, None, None, None]
        gSts_all = mSts[hb, None, None, None]
        gDPs_all = mDPsiV[hb, None, None, None]
        gDK_all = mDK[hb, None, None, None]
        gDQ_all = mDQ[hb, None, None, None]
        gDqd_all = mDqkd[hb, None, None, None]
        gDsO = mDstatesOut[hb, None, None]  # (N, P) -- one per CTA, no chunk axis

        nchunks = cute.size(gK_all.shape[0])

        smem = SmemAllocator()
        sK     = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQ     = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sQ_T   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDst   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPh   = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDPh_T = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sPsi  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sSts  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        # Keep lkq/dki scratch isolated from the output-staging alias. This makes
        # the next narrow banking/prefetch layer local to GEMM3/GEMM7 without
        # perturbing the current sOut = sK locality.
        sLKQ0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sLKQ1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sK_T0  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sK_T1  = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI_T0 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        sDKI_T1 = smem.allocate_tensor(self.dtype, sLayout_64x64.outer, swizzle=sLayout_64x64.inner)
        # sK dies after GEMM2. Reuse it as the output staging scratch instead of
        # keeping a separate resident sOut tile for the four per-chunk outputs
        # and the final dstates writeback.
        sOut = sK

        thr_g2s = copy_g2s.get_slice(tidx)
        thr_s2g = copy_s2g.get_slice(tidx)
        wg_mma = tiled_mma.get_slice(tidx)
        shape_mnk = (fcs, N, N)
        # Loop-carried dstates accumulator (zero-init BEFORE the chunk loop)
        acc_dstates = cute.make_rmem_tensor(
            wg_mma.partition_shape_C((N, P)), Float32
        )
        acc_dstates.fill(0.0)

        if nchunks > 0:
            first_chunk_idx = nchunks - 1
            cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[first_chunk_idx, None, None]), thr_g2s.partition_D(sK_T0))
        arch.sync_threads()

        sLKQ_cur = sLKQ0
        sDKI_cur = sDKI0
        sLKQ_next = sLKQ1
        sDKI_next = sDKI1
        sDKI_T_cur = sDKI_T0
        sDKI_T_next = sDKI_T1

        for chunk_rev in cutlass.range(nchunks, unroll=1):
            chunk_idx = nchunks - 1 - chunk_rev
            next_chunk_idx = chunk_idx - 1

            gK_c = gK_all[chunk_idx, None, None]
            gKT_c = gKT_all[chunk_idx, None, None]
            gQ_c = gQ_all[chunk_idx, None, None]
            gQT_c = gQT_all[chunk_idx, None, None]
            gDst_c = gDst_all[chunk_idx, None, None]
            gDPh_c = gDPh_all[chunk_idx, None, None]
            gDPhT_c = gDPhT_all[chunk_idx, None, None]
            gPsi_c = gPsi_all[chunk_idx, None, None]
            gSts_c = gSts_all[chunk_idx, None, None]

            cute.copy(copy_g2s, thr_g2s.partition_S(gK_c), thr_g2s.partition_D(sK))
            cute.copy(copy_g2s, thr_g2s.partition_S(gQ_c), thr_g2s.partition_D(sQ))
            cute.copy(copy_g2s, thr_g2s.partition_S(gQT_c), thr_g2s.partition_D(sQ_T))
            cute.copy(copy_g2s, thr_g2s.partition_S(gDst_c), thr_g2s.partition_D(sDst))
            cute.copy(copy_g2s, thr_g2s.partition_S(gDPh_c), thr_g2s.partition_D(sDPh))
            cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT_c), thr_g2s.partition_D(sDPh_T))
            cute.copy(copy_g2s, thr_g2s.partition_S(gPsi_c), thr_g2s.partition_D(sPsi))
            cute.copy(copy_g2s, thr_g2s.partition_S(gSts_c), thr_g2s.partition_D(sSts))
            arch.sync_threads()

            # === GEMM1: dPsiV = K @ Dst^T ===
            _, tA1, tB1 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK, sDst)
            acc_dPsiV = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dPsiV, tA1, tB1, zero_init=True, wg_wait=0)

            # === GEMM2: lkq = K @ Q^T ===
            _, tA2, tB2 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK, sQ)
            acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
            sm90_utils.gemm(tiled_mma, acc_lkq, tA2, tB2, zero_init=True, wg_wait=0)

            # === GEMM4: dqkd = DPh @ Psi^T ===
            _, tA4, tB4 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh, sPsi)
            acc_dqkd = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dqkd, tA4, tB4, zero_init=True, wg_wait=0)

            # === GEMM5: dk = Psi @ Dst^T ===
            _, tA5, tB5 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi, sDst)
            acc_dk = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dk, tA5, tB5, zero_init=True, wg_wait=0)

            # === GEMM6: dki = Psi @ DPh^T ===
            _, tA6, tB6 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sPsi, sDPh)
            acc_dki = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dki, tA6, tB6, zero_init=True, wg_wait=0)

            # === GEMM6': dki_T = DPh @ Psi^T (transpose of GEMM6, for GEMM9) ===
            _, tA6p, tB6p = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh, sPsi)
            acc_dki_T = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, fcs)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dki_T, tA6p, tB6p, zero_init=True, wg_wait=0)

            # === GEMM8: dq = DPh @ Sts^T ===
            _, tA8, tB8 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDPh, sSts)
            acc_dq = cute.make_rmem_tensor(wg_mma.partition_shape_C((fcs, N)), Float32)
            sm90_utils.gemm(tiled_mma, acc_dq, tA8, tB8, zero_init=True, wg_wait=0)

            # === GEMM10: dstates_accum += Q^T @ DPh ===
            # Caller pre-transposes:
            #   Q  -> Q_T of shape (N, fcs)  so A = Q^T natively fits K-major (M, K)
            #   DPh -> DPh_T of shape (P, fcs) so B = DPh^T natively fits K-major (N, K)
            # WGMMA then computes: C[m,n] = sum_k A[m,k] * B[n,k]
            #                            = Q^T[m,k] * DPh^T[n,k]
            #                            = Q[k,m] * DPh[k,n]
            #                            = (Q^T @ DPh)[m,n]  -- exactly GEMM 10
            _, tA10, tB10 = sm90_utils.partition_fragment_ABC(
                wg_mma, shape_mnk, sQ_T, sDPh_T
            )
            sm90_utils.gemm(
                tiled_mma, acc_dstates, tA10, tB10,
                zero_init=False, wg_wait=0,
            )

            # R2S: spill lkq for GEMM3 and dki for GEMM7/9
            if (chunk_rev & 1) == 0:
                sLKQ_cur = sLKQ0
                sDKI_cur = sDKI0
                sLKQ_next = sLKQ1
                sDKI_next = sDKI1
                sDKI_T_cur = sDKI_T0
                sDKI_T_next = sDKI_T1
            else:
                sLKQ_cur = sLKQ1
                sDKI_cur = sDKI1
                sLKQ_next = sLKQ0
                sDKI_next = sDKI0
                sDKI_T_cur = sDKI_T1
                sDKI_T_next = sDKI_T0

            copy_r2s_lkq, _, _ = copy_utils.get_smem_store_C(
                tiled_mma, sLKQ_cur, tidx, arch=90, position_independent=True,
            )
            lkq_frg = layout_utils.reshape_acc_to_frgA(acc_lkq)
            lkq_cvt = cute.make_rmem_tensor_like(lkq_frg, self.dtype)
            lkq_cvt.store(lkq_frg.load().to(self.dtype))
            copy_r2s_lkq(lkq_cvt)

            copy_r2s_dki, _, _ = copy_utils.get_smem_store_C(
                tiled_mma, sDKI_cur, tidx, arch=90, position_independent=True,
            )
            dki_frg = layout_utils.reshape_acc_to_frgA(acc_dki)
            dki_cvt = cute.make_rmem_tensor_like(dki_frg, self.dtype)
            dki_cvt.store(dki_frg.load().to(self.dtype))
            copy_r2s_dki(dki_cvt)

            # sLKQ/sDKI become live here and are consumed immediately by GEMM3/GEMM7.
            # Keep a neighboring bank selected now so the next chunk can take it
            # without perturbing the current sOut path.
            arch.fence_view_async_shared()
            arch.sync_threads()

            # === GEMM3: dPsiV += lkq @ DPh^T ===
            _, tA3, tB3 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ_cur, sDPh)
            sm90_utils.gemm(tiled_mma, acc_dPsiV, tA3, tB3, zero_init=False, wg_wait=0)

            # === GEMM7: dk += dki @ Q (via pre-transposed Q_T) ===
            _, tA7, tB7 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDKI_cur, sQ_T)
            sm90_utils.gemm(tiled_mma, acc_dk, tA7, tB7, zero_init=False, wg_wait=0)

            # Touch the next-bank descriptors before leaving the scratch window so
            # adjacent chunk handoff stays local to lkq/dki banking only.
            if next_chunk_idx >= 0:
                copy_utils.get_smem_store_C(
                    tiled_mma, sLKQ_next, tidx, arch=90, position_independent=True,
                )
                copy_utils.get_smem_store_C(
                    tiled_mma, sDKI_next, tidx, arch=90, position_independent=True,
                )

            # R2S: spill dki_T for GEMM9
            copy_r2s_dki_T, _, _ = copy_utils.get_smem_store_C(
                tiled_mma, sDKI_T_cur, tidx, arch=90, position_independent=True,
            )
            dki_T_frg = layout_utils.reshape_acc_to_frgA(acc_dki_T)
            dki_T_cvt = cute.make_rmem_tensor_like(dki_T_frg, self.dtype)
            dki_T_cvt.store(dki_T_frg.load().to(self.dtype))
            copy_r2s_dki_T(dki_T_cvt)

            if next_chunk_idx >= 0:
                copy_utils.get_smem_store_C(
                    tiled_mma, sDKI_T_next, tidx, arch=90, position_independent=True,
                )

            arch.fence_view_async_shared()
            arch.sync_threads()

            # K^T uses its own tiny double-buffer. Prefetch the next chunk's K^T while
            # the current chunk continues on the already-materialized bank.
            if (chunk_rev & 1) == 0:
                if next_chunk_idx >= 0:
                    cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sK_T1))
                arch.sync_threads()

                # === GEMM9: dq += dki^T @ K (via pre-transposed K_T and computed dki_T) ===
                _, tA9, tB9 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDKI_T_cur, sK_T0)
                sm90_utils.gemm(tiled_mma, acc_dq, tA9, tB9, zero_init=False, wg_wait=0)
            else:
                if next_chunk_idx >= 0:
                    cute.copy(copy_g2s, thr_g2s.partition_S(gKT_all[next_chunk_idx, None, None]), thr_g2s.partition_D(sK_T0))
                arch.sync_threads()

                # === GEMM9: dq += dki^T @ K (via pre-transposed K_T and computed dki_T) ===
                _, tA9, tB9 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sDKI_T_cur, sK_T1)
                sm90_utils.gemm(tiled_mma, acc_dq, tA9, tB9, zero_init=False, wg_wait=0)

            # === Store per-chunk outputs ===
            gDPs_c = gDPs_all[chunk_idx, None, None]
            gDK_c = gDK_all[chunk_idx, None, None]
            gDQ_c = gDQ_all[chunk_idx, None, None]
            gDqd_c = gDqd_all[chunk_idx, None, None]

            def store_acc(acc, gdst):
                copy_fn, _, _ = copy_utils.get_smem_store_C(tiled_mma, sOut, tidx, arch=90)
                frg = layout_utils.reshape_acc_to_frgA(acc)
                cvt = cute.make_rmem_tensor_like(frg, self.dtype)
                cvt.store(frg.load().to(self.dtype))
                copy_fn(cvt)
                arch.sync_threads()
                cute.copy(copy_s2g, thr_s2g.partition_S(sOut), thr_s2g.partition_D(gdst))
                arch.sync_threads()

            store_acc(acc_dPsiV, gDPs_c)
            store_acc(acc_dk, gDK_c)
            store_acc(acc_dq, gDQ_c)
            store_acc(acc_dqkd, gDqd_c)

        # === Store dstates ONCE after the loop (loop-carried accumulator) ===
        copy_fn_ds, _, _ = copy_utils.get_smem_store_C(
            tiled_mma, sOut, tidx, arch=90,
        )
        frg_ds = layout_utils.reshape_acc_to_frgA(acc_dstates)
        cvt_ds = cute.make_rmem_tensor_like(frg_ds, self.dtype)
        cvt_ds.store(frg_ds.load().to(self.dtype))
        copy_fn_ds(cvt_ds)
        arch.sync_threads()
        cute.copy(copy_s2g, thr_s2g.partition_S(sOut), thr_s2g.partition_D(gDsO))
        arch.sync_threads()

    @cute.jit
    def __call__(self, mK, mK_T, mQ, mQ_T, mDst, mDPh, mDPh_T, mPsi, mSts,
                 mDPsiV, mDK, mDQ, mDqkd, mDstatesOut, H, B, stream):
        fcs = self.fcs
        N = self.N
        P = self.P

        mK = assume_tensor_aligned(mK)
        mK_T = assume_tensor_aligned(mK_T)
        mQ = assume_tensor_aligned(mQ)
        mQ_T = assume_tensor_aligned(mQ_T)
        mDst = assume_tensor_aligned(mDst)
        mDPh = assume_tensor_aligned(mDPh)
        mDPh_T = assume_tensor_aligned(mDPh_T)
        mPsi = assume_tensor_aligned(mPsi)
        mSts = assume_tensor_aligned(mSts)
        mDPsiV = assume_tensor_aligned(mDPsiV)
        mDK = assume_tensor_aligned(mDK)
        mDQ = assume_tensor_aligned(mDQ)
        mDqkd = assume_tensor_aligned(mDqkd)
        mDstatesOut = assume_tensor_aligned(mDstatesOut)

        # Single tiled_mma reused for all 10 GEMMs (N==P==fcs==64)
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
            mK, mK_T, mQ, mQ_T, mDst, mDPh, mDPh_T, mPsi, mSts,
            mDPsiV, mDK, mDQ, mDqkd, mDstatesOut, H,
            tiled_mma, tiled_mma, sLayout_64x64, copy_g2s, copy_s2g,
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

        def mk3d(rows, cols):  # one per CTA, no chunk axis
            shape = (total, rows, cols)
            stride = (rows * cols, cols, 1)
            return make_fake_tensor(BFloat16, shape, stride=stride, assumed_align=16)

        stream_arg = cuda.CUstream(0)
        compiled = cute.compile(
            obj,
            mk4d(fcs, N),    # K
            mk4d(N, fcs),    # K_T (pre-transposed)
            mk4d(fcs, N),    # Q
            mk4d(N, fcs),    # Q_T (pre-transposed)
            mk4d(N, P),      # Dst
            mk4d(fcs, P),    # DPh
            mk4d(P, fcs),    # DPh_T (pre-transposed)
            mk4d(fcs, P),    # Psi
            mk4d(N, P),      # Sts
            mk4d(fcs, P),    # DPsiV out
            mk4d(fcs, N),    # DK out
            mk4d(fcs, N),    # DQ out
            mk4d(fcs, fcs),  # Dqkd out
            mk3d(N, P),      # Dstates out (per-CTA, loop-carried)
            H, B, stream_arg,
        )
        _CACHE_P4[key] = compiled
    return _CACHE_P4[key]


def run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
           DPsiV, DK, DQ, Dqkd, DstatesOut, nchunks, H, B, stream,
           K_T=None, Q_T=None, DPh_T=None):
    compiled = _get_compiled(N, P, R, chunk_size, nchunks, H, B)
    dl = lambda t: from_dlpack(t, assumed_align=16)
    # Pre-transpose K, Q, and DPh to feed MN-major operands via K-major smem
    if K_T is None:
        K_T = K.transpose(-1, -2).contiguous()
    if Q_T is None:
        Q_T = Q.transpose(-1, -2).contiguous()
    if DPh_T is None:
        DPh_T = DPh.transpose(-1, -2).contiguous()
    compiled(dl(K), dl(K_T), dl(Q), dl(Q_T), dl(Dst), dl(DPh), dl(DPh_T), dl(Psi), dl(Sts),
             dl(DPsiV), dl(DK), dl(DQ), dl(Dqkd), dl(DstatesOut),
             H, B, stream)



def compute_epilogue_outputs(
    dout, q_raw, k_raw, v,
    q_bias, k_bias, mimo_v, mimo_o,
    angles, dA_cs, dA_cs_rev, dt, trap, D, segsum, states, qk_dot,
    chunk_size=16, R=4, rotary_dim_divisor=4,
):
    """Compute ALL epilogue outputs from bwd_bwd, matching TileLang exactly.

    This replaces the previous placeholder/approximate implementation.
    All outputs now use the exact TileLang formulas including:
    - Proper exp(dA_cs_rev) scaling for DDA_CS_REV
    - Full trap scaling for DFACTOR
    - QK_DOT-based DGAMMA_DIAG computation
    - Segsum-masked DSSDA
    - State-passing DDA via cached STATES
    - Full rotary-embedding DANGLES via inverse rotation

    Inputs match TileLang's mamba_mimo_bwd_bwd signature.
    """
    from full_bwd_bwd_epilogue import full_bwd_bwd_pytorch
    return full_bwd_bwd_pytorch(
        dout, q_raw, k_raw, v,
        q_bias, k_bias, mimo_v, mimo_o,
        angles, dA_cs, dA_cs_rev, dt, trap, D, segsum, states, qk_dot,
        chunk_size=chunk_size, R=R, rotary_dim_divisor=rotary_dim_divisor,
    )


def bench(shape_label, B, S, H):
    import time
    N, P, R = 64, 64, 4
    chunk_size = 16
    nchunks = S // chunk_size
    fcs = chunk_size * R
    total = B * H
    dev = _pick_bench_device()

    print(f'\n=== Phase 4 10-GEMM, {shape_label}: B={B} S={S} H={H} chunks={nchunks} total_CTAs={total} dev={dev} ===')

    dev_index = torch.device(dev).index or 0
    with torch.cuda.device(dev_index):
        torch.manual_seed(42)
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
        DstO = torch.zeros(total, N, P, dtype=torch.bfloat16, device=dev)  # no chunk axis

        stream = cuda.CUstream(torch.cuda.current_stream(device=dev).cuda_stream)

        t0 = time.time()
        run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
               DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
        torch.cuda.synchronize(device=dev_index)
        print(f'Compile+launch: {time.time()-t0:.2f}s')

    # Spot check: dstates for hb=0 = sum over chunks of Q[c]^T @ DPh[c]
        dstates_ref = torch.zeros(N, P, dtype=torch.float32, device=dev)
        for c in range(nchunks):
            q_c = Q[0, c].float()   # (fcs, N)
            dph_c = DPh[0, c].float()  # (fcs, P)
            dstates_ref += q_c.T @ dph_c
        diff = (DstO[0].float() - dstates_ref).abs()
        max_err = diff.max().item()
        max_ref = dstates_ref.abs().max().item()
        print(f'dStates (GEMM10) check hb0: max_err={max_err:.2f} rel={max_err/max(max_ref,1e-8):.4f}  max_ref={max_ref:.2f}')

    # Benchmark
        for _ in range(15):
            run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
                   DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
        torch.cuda.synchronize(device=dev_index)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        iters = 200 if total < 50 else 50
        s.record()
        for _ in range(iters):
            run_p4(N, P, R, chunk_size, K, Q, Dst, DPh, Psi, Sts,
                   DPsiV, DK, DQ, Dqkd, DstO, nchunks, H, B, stream)
        e.record()
        torch.cuda.synchronize(device=dev_index)
        us = s.elapsed_time(e) * 1000 / iters
        print(f'Fused 10-GEMM: {us:.1f} us')
        return us


if __name__ == '__main__':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    us_smoke = bench('smoke', 2, 256, 8)
    us_prod = bench('production', 2, 4096, 28)

    print(f'\n=== Summary ===')
    print(f'Smoke:      Phase4={us_smoke:.1f}us  TileLang=174.6us  ratio={174.6/us_smoke:.2f}x')
    print(f'Production: Phase4={us_prod:.1f}us  TileLang=3135us   ratio={3135/us_prod:.2f}x')
