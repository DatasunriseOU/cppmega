"""Fused CuTe tiles for state/apply plus DV/DMIMO_V consumers.

The one-chunk tile is the next bounded scan-owner probe after
``masked_lkq_apply``.  It keeps the three 64x64 BF16 WGMMA products in one CTA:

  1. state = K @ DStates
  2. lkq   = future_mask(K @ Q.T)
  3. apply = lkq @ dPhi

``state`` and ``apply`` are rounded to BF16 in shared memory, matching the Wave
5/6 tile-chain semantics, but neither tile is written to global memory.  The
kernel then computes the scalar DV and DMIMO_V consumers in-kernel and writes
only those final FP32 outputs.

The multi-chunk tile extends that path into a reverse scan owner.  It carries
``DStates.T`` in registers as ``carry_t`` so it can be spilled directly as the
WGMMA B operand for the next chunk's ``K @ DStates`` product.  It also includes
the same-time qk diagonal contribution to the scalar ``dpsi`` consumers:

  carry_t = exp(dA_cs_last) * carry_t + (dPhi * exp(dA_cs)).T @ Q
"""

from __future__ import annotations

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32
from cutlass.cute import arch
from cutlass.cute.nvgpu import warpgroup
from cutlass.cute.runtime import from_dlpack, make_fake_tensor
from cutlass.utils import LayoutEnum, SmemAllocator
import cutlass.utils.hopper_helpers as sm90_utils_basic
from quack import copy_utils, layout_utils, sm90_utils

from cppmega.megatron.cute_dsl_mimo.single_gemm_test import _make_row_major_tiled_copy


class StateApplyConsumersWGMMA:
    """One-CTA fused state/apply/consumer tile for the fixed 64x64 probe."""

    def __init__(self, dim: int = 64, rank: int = 4, chunk_size: int = 16, dtype=BFloat16):
        self.dim = dim
        self.rank = rank
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.num_threads = 128

    @cute.jit
    def _apply_future_mask(self, acc_lkq: cute.Tensor, coord_lkq: cute.Tensor) -> None:
        for i in cutlass.range(cute.size(coord_lkq), unroll_full=True):
            row_t = coord_lkq[i][0] // self.rank
            col_t = coord_lkq[i][1] // self.rank
            if not (row_t < col_t):
                acc_lkq[i] = 0.0

    @cute.jit
    def _spill_acc_bf16(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        smem_tensor: cute.Tensor,
        tidx: cutlass.Int32,
        position_independent: cutlass.Constexpr[bool],
    ) -> None:
        copy_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma,
            smem_tensor,
            tidx,
            arch=90,
            position_independent=position_independent,
        )
        frg = layout_utils.reshape_acc_to_frgA(acc)
        bf16_frg = cute.make_rmem_tensor_like(frg, self.dtype)
        bf16_frg.store(frg.load().to(self.dtype))
        copy_r2s(bf16_frg)

    @cute.kernel
    def kernel(
        self,
        gK: cute.Tensor,
        gQ: cute.Tensor,
        gDstT: cute.Tensor,
        gDPhT: cute.Tensor,
        gV: cute.Tensor,
        gMimoV: cute.Tensor,
        gDV: cute.Tensor,
        gDMimoV: cute.Tensor,
        tiled_mma: cute.TiledMma,
        s_layout: cute.ComposedLayout,
        copy_g2s: cute.TiledCopy,
    ) -> None:
        tidx = arch.thread_idx()[0]
        dim = self.dim
        rank = self.rank
        chunk_size = self.chunk_size

        smem = SmemAllocator()
        sK = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sQ = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sDstT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sDPhT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)

        # Lifetime aliases after the producer operands are dead:
        #   K    -> BF16 state tile
        #   Q    -> BF16 masked LKQ tile
        #   DPhT -> BF16 apply tile
        sState = sK
        sLKQ = sQ
        sApply = sDPhT

        thr_g2s = copy_g2s.get_slice(tidx)
        cute.copy(copy_g2s, thr_g2s.partition_S(gK), thr_g2s.partition_D(sK))
        cute.copy(copy_g2s, thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ))
        cute.copy(copy_g2s, thr_g2s.partition_S(gDstT), thr_g2s.partition_D(sDstT))
        cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT), thr_g2s.partition_D(sDPhT))
        arch.sync_threads()

        wg_mma = tiled_mma.get_slice(tidx)
        shape_mnk = (dim, dim, dim)

        # GEMM1: state = K @ DStates == K @ DstT.T
        _, tA_state, tB_state = sm90_utils.partition_fragment_ABC(
            wg_mma, shape_mnk, sK, sDstT
        )
        acc_state = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
        sm90_utils.gemm(tiled_mma, acc_state, tA_state, tB_state, zero_init=True, wg_wait=0)

        # GEMM2: LKQ = K @ Q.T.  K/Q can be recycled after this point.
        _, tA_lkq, tB_lkq = sm90_utils.partition_fragment_ABC(
            wg_mma, shape_mnk, sK, sQ
        )
        acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
        sm90_utils.gemm(tiled_mma, acc_lkq, tA_lkq, tB_lkq, zero_init=True, wg_wait=0)

        coord_lkq = wg_mma.partition_C(cute.make_identity_tensor((dim, dim)))
        self._apply_future_mask(acc_lkq, coord_lkq)

        self._spill_acc_bf16(tiled_mma, acc_state, sState, tidx, True)
        self._spill_acc_bf16(tiled_mma, acc_lkq, sLKQ, tidx, True)
        arch.fence_view_async_shared()
        arch.sync_threads()

        # GEMM3: apply = masked(LKQ) @ dPhi == masked(LKQ) @ DPhT.T
        _, tA_apply, tB_apply = sm90_utils.partition_fragment_ABC(
            wg_mma, shape_mnk, sLKQ, sDPhT
        )
        acc_apply = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
        sm90_utils.gemm(tiled_mma, acc_apply, tA_apply, tB_apply, zero_init=True, wg_wait=0)

        self._spill_acc_bf16(tiled_mma, acc_apply, sApply, tidx, True)
        arch.fence_view_async_shared()
        arch.sync_threads()

        # Scalar consumers.  This intentionally mirrors the existing torch-side
        # probe: dpsi is state.float() + apply.float(), with state/apply already
        # BF16-rounded through shared memory.
        for tile in cutlass.range_constexpr(8):
            linear = tile * self.num_threads + tidx
            t = linear // dim
            p = linear - t * dim
            acc = Float32(0.0)
            for r in cutlass.range_constexpr(4):
                f = t * rank + r
                dpsi = Float32(sState[f, p]) + Float32(sApply[f, p])
                acc += dpsi * Float32(gMimoV[r, p])
            gDV[t, p] = acc

        for tile in cutlass.range_constexpr(2):
            linear = tile * self.num_threads + tidx
            r = linear // dim
            p = linear - r * dim
            acc = Float32(0.0)
            for t in cutlass.range_constexpr(16):
                f = t * rank + r
                dpsi = Float32(sState[f, p]) + Float32(sApply[f, p])
                acc += dpsi * Float32(gV[t, p])
            gDMimoV[r, p] = acc

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mQ: cute.Tensor,
        mDstT: cute.Tensor,
        mDPhT: cute.Tensor,
        mV: cute.Tensor,
        mMimoV: cute.Tensor,
        mDV: cute.Tensor,
        mDMimoV: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        dim = self.dim
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (dim, dim),
            warpgroup.OperandSource.SMEM,
        )
        s_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (dim, dim),
        )
        copy_g2s = _make_row_major_tiled_copy(
            self.dtype,
            dim,
            self.num_threads,
            copy_bits=self.dtype.width,
        )

        self.kernel(
            mK,
            mQ,
            mDstT,
            mDPhT,
            mV,
            mMimoV,
            mDV,
            mDMimoV,
            tiled_mma,
            s_layout,
            copy_g2s,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


_CACHE: dict[tuple[int, int, int], object] = {}
_CACHE_MULTI: dict[tuple[int, int, int, int], object] = {}


def run_state_apply_consumers(
    dim: int,
    rank: int,
    chunk_size: int,
    k: object,
    q: object,
    dstates_t: object,
    dphi_t: object,
    v: object,
    mimo_v: object,
    dv: object,
    dmimo_v: object,
    stream: cuda.CUstream,
) -> None:
    if (dim, rank, chunk_size) != (64, 4, 16):
        raise ValueError(
            "StateApplyConsumersWGMMA currently specializes dim=64, rank=4, chunk_size=16"
        )
    key = (dim, rank, chunk_size)
    compiled = _CACHE.get(key)
    if compiled is None:
        obj = StateApplyConsumersWGMMA(dim, rank, chunk_size, BFloat16)

        def fake_bf16(shape: tuple[int, int]) -> cute.Tensor:
            return make_fake_tensor(
                BFloat16,
                shape,
                stride=(shape[1], 1),
                assumed_align=16,
            )

        def fake_f32(shape: tuple[int, int]) -> cute.Tensor:
            return make_fake_tensor(
                Float32,
                shape,
                stride=(shape[1], 1),
                assumed_align=16,
            )

        compiled = cute.compile(
            obj,
            fake_bf16((dim, dim)),
            fake_bf16((dim, dim)),
            fake_bf16((dim, dim)),
            fake_bf16((dim, dim)),
            fake_bf16((chunk_size, dim)),
            fake_bf16((rank, dim)),
            fake_f32((chunk_size, dim)),
            fake_f32((rank, dim)),
            stream,
        )
        _CACHE[key] = compiled

    dl = lambda tensor: from_dlpack(tensor, assumed_align=16)
    compiled(
        dl(k),
        dl(q),
        dl(dstates_t),
        dl(dphi_t),
        dl(v),
        dl(mimo_v),
        dl(dv),
        dl(dmimo_v),
        stream,
    )


class MultiChunkStateApplyConsumersWGMMA(StateApplyConsumersWGMMA):
    """One-CTA reverse scan-owner prototype for a fixed small chunk count."""

    @cute.jit
    def _apply_future_mask_scaled(
        self,
        acc_lkq: cute.Tensor,
        coord_lkq: cute.Tensor,
        gSegsum: cute.Tensor,
    ) -> None:
        for i in cutlass.range(cute.size(coord_lkq), unroll_full=True):
            row_t = coord_lkq[i][0] // self.rank
            col_t = coord_lkq[i][1] // self.rank
            if row_t < col_t:
                acc_lkq[i] = acc_lkq[i] * cute.exp(Float32(gSegsum[col_t, row_t]))
            else:
                acc_lkq[i] = 0.0

    @cute.jit
    def _scale_acc_rows_by_exp(
        self,
        acc: cute.Tensor,
        coord: cute.Tensor,
        gScale: cute.Tensor,
    ) -> None:
        for i in cutlass.range(cute.size(coord), unroll_full=True):
            row_t = coord[i][0] // self.rank
            acc[i] = acc[i] * cute.exp(Float32(gScale[row_t]))

    @cute.jit
    def _scale_acc_by_scalar(self, acc: cute.Tensor, scale: Float32) -> None:
        for i in cutlass.range(cute.size(acc), unroll_full=True):
            acc[i] = acc[i] * scale

    @cute.jit
    def _fill_scaled_dphi_t(
        self,
        gDPhT: cute.Tensor,
        gDACS: cute.Tensor,
        sDPhScaledT: cute.Tensor,
        tidx: cutlass.Int32,
    ) -> None:
        dim = self.dim
        rank = self.rank
        for tile in cutlass.range_constexpr(32):
            linear = tile * self.num_threads + tidx
            p = linear // dim
            f = linear - p * dim
            t = f // rank
            scale = cute.exp(Float32(gDACS[t]))
            sDPhScaledT[p, f] = sDPhScaledT.element_type(
                Float32(gDPhT[p, f]) * scale
            )

    @cute.kernel
    def kernel(
        self,
        gK: cute.Tensor,
        gQ: cute.Tensor,
        gQT: cute.Tensor,
        gDPhT: cute.Tensor,
        gDACS: cute.Tensor,
        gDACSRev: cute.Tensor,
        gSegsum: cute.Tensor,
        gQKDot: cute.Tensor,
        gGamma: cute.Tensor,
        gV: cute.Tensor,
        gMimoV: cute.Tensor,
        gDV: cute.Tensor,
        gDMimoV: cute.Tensor,
        tiled_mma: cute.TiledMma,
        s_layout: cute.ComposedLayout,
        copy_g2s: cute.TiledCopy,
    ) -> None:
        tidx = arch.thread_idx()[0]
        dim = self.dim
        rank = self.rank
        chunk_size = self.chunk_size

        gK_all = gK[None, None, None]
        gQ_all = gQ[None, None, None]
        gQT_all = gQT[None, None, None]
        gDPhT_all = gDPhT[None, None, None]
        gDACS_all = gDACS[None, None]
        gDACSRev_all = gDACSRev[None, None]
        gSegsum_all = gSegsum[None, None, None]
        gQKDot_all = gQKDot[None, None, None, None]
        gGamma_all = gGamma[None, None]
        gV_all = gV[None, None, None]
        gDV_all = gDV[None, None, None]
        nchunks = cute.size(gK_all.shape[0])

        smem = SmemAllocator()
        sK = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sQ = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sQT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sDPhT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sDPhScaledT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sCarryT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)

        # After producer operands are consumed:
        #   sK becomes BF16 state
        #   sQ first holds BF16 masked LKQ, then BF16 apply
        sState = sK
        sLKQ = sQ
        sApply = sQ

        thr_g2s = copy_g2s.get_slice(tidx)
        wg_mma = tiled_mma.get_slice(tidx)
        shape_mnk = (dim, dim, dim)

        # carry_t stores DStates.T as (P, N), matching the WGMMA B operand for
        # state = K @ DStates.  It starts at zero for the bounded production
        # scan prototype and is updated after each reverse chunk.
        acc_carry_t = cute.make_rmem_tensor(
            wg_mma.partition_shape_C((dim, dim)), Float32
        )
        acc_carry_t.fill(0.0)

        for tile in cutlass.range_constexpr(2):
            linear = tile * self.num_threads + tidx
            r = linear // dim
            p = linear - r * dim
            gDMimoV[r, p] = Float32(0.0)
        arch.sync_threads()

        for chunk_rev in cutlass.range(nchunks, unroll=1):
            chunk_idx = nchunks - 1 - chunk_rev

            gK_c = gK_all[chunk_idx, None, None]
            gQ_c = gQ_all[chunk_idx, None, None]
            gQT_c = gQT_all[chunk_idx, None, None]
            gDPhT_c = gDPhT_all[chunk_idx, None, None]
            gDACS_c = gDACS_all[chunk_idx, None]
            gDACSRev_c = gDACSRev_all[chunk_idx, None]
            gSegsum_c = gSegsum_all[chunk_idx, None, None]
            gQKDot_c = gQKDot_all[chunk_idx, None, None, None]
            gGamma_c = gGamma_all[chunk_idx, None]

            cute.copy(copy_g2s, thr_g2s.partition_S(gK_c), thr_g2s.partition_D(sK))
            cute.copy(copy_g2s, thr_g2s.partition_S(gQ_c), thr_g2s.partition_D(sQ))
            cute.copy(copy_g2s, thr_g2s.partition_S(gQT_c), thr_g2s.partition_D(sQT))
            cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT_c), thr_g2s.partition_D(sDPhT))
            arch.sync_threads()

            self._fill_scaled_dphi_t(gDPhT_c, gDACS_c, sDPhScaledT, tidx)
            arch.sync_threads()

            self._spill_acc_bf16(tiled_mma, acc_carry_t, sCarryT, tidx, True)
            arch.fence_view_async_shared()
            arch.sync_threads()

            # GEMM1: state = K @ carry, where sCarryT holds carry.T.
            _, tA_state, tB_state = sm90_utils.partition_fragment_ABC(
                wg_mma, shape_mnk, sK, sCarryT
            )
            acc_state = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
            sm90_utils.gemm(tiled_mma, acc_state, tA_state, tB_state, zero_init=True, wg_wait=0)
            coord_state = wg_mma.partition_C(cute.make_identity_tensor((dim, dim)))
            self._scale_acc_rows_by_exp(acc_state, coord_state, gDACSRev_c)

            # GEMM2: LKQ = K @ Q.T, masked before BF16 shared-memory spill.
            _, tA_lkq, tB_lkq = sm90_utils.partition_fragment_ABC(
                wg_mma, shape_mnk, sK, sQ
            )
            acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
            sm90_utils.gemm(tiled_mma, acc_lkq, tA_lkq, tB_lkq, zero_init=True, wg_wait=0)

            coord_lkq = wg_mma.partition_C(cute.make_identity_tensor((dim, dim)))
            self._apply_future_mask_scaled(acc_lkq, coord_lkq, gSegsum_c)

            self._spill_acc_bf16(tiled_mma, acc_state, sState, tidx, True)
            self._spill_acc_bf16(tiled_mma, acc_lkq, sLKQ, tidx, True)
            arch.fence_view_async_shared()
            arch.sync_threads()

            # GEMM3: apply = masked(LKQ) @ dPhi == masked(LKQ) @ DPhT.T.
            _, tA_apply, tB_apply = sm90_utils.partition_fragment_ABC(
                wg_mma, shape_mnk, sLKQ, sDPhT
            )
            acc_apply = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
            sm90_utils.gemm(tiled_mma, acc_apply, tA_apply, tB_apply, zero_init=True, wg_wait=0)

            self._spill_acc_bf16(tiled_mma, acc_apply, sApply, tidx, True)
            arch.fence_view_async_shared()
            arch.sync_threads()

            gDV_c = gDV_all[chunk_idx, None, None]
            gV_c = gV_all[chunk_idx, None, None]

            for tile in cutlass.range_constexpr(8):
                linear = tile * self.num_threads + tidx
                t = linear // dim
                p = linear - t * dim
                acc = Float32(0.0)
                for r in cutlass.range_constexpr(4):
                    f = t * rank + r
                    dpsi = Float32(sState[f, p]) + Float32(sApply[f, p])
                    gamma = Float32(gGamma_c[t])
                    for r_out in cutlass.range_constexpr(4):
                        f_out = t * rank + r_out
                        dpsi += (
                            gamma
                            * Float32(gQKDot_c[t, r_out, r])
                            * Float32(sDPhT[p, f_out])
                        )
                    acc += dpsi * Float32(gMimoV[r, p])
                gDV_c[t, p] = acc

            for tile in cutlass.range_constexpr(2):
                linear = tile * self.num_threads + tidx
                r = linear // dim
                p = linear - r * dim
                acc = Float32(0.0)
                for t in cutlass.range_constexpr(16):
                    f = t * rank + r
                    dpsi = Float32(sState[f, p]) + Float32(sApply[f, p])
                    gamma = Float32(gGamma_c[t])
                    for r_out in cutlass.range_constexpr(4):
                        f_out = t * rank + r_out
                        dpsi += (
                            gamma
                            * Float32(gQKDot_c[t, r_out, r])
                            * Float32(sDPhT[p, f_out])
                        )
                    acc += dpsi * Float32(gV_c[t, p])
                gDMimoV[r, p] = Float32(gDMimoV[r, p]) + acc

            arch.sync_threads()

            # Loop-carried update for the next older chunk:
            # carry_t = exp(dA_cs_last) * carry_t + (dPhi * exp(dA_cs)).T @ Q.
            # Q_T is a harness-side transpose used to keep the B operand K-major
            # and avoid the MN-major smem descriptor issue already hit in P4.
            carry_decay = cute.exp(Float32(gDACS_c[chunk_size - 1]))
            self._scale_acc_by_scalar(acc_carry_t, carry_decay)
            _, tA_carry, tB_carry = sm90_utils.partition_fragment_ABC(
                wg_mma, shape_mnk, sDPhScaledT, sQT
            )
            sm90_utils.gemm(
                tiled_mma,
                acc_carry_t,
                tA_carry,
                tB_carry,
                zero_init=False,
                wg_wait=0,
            )
            arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mQ: cute.Tensor,
        mQT: cute.Tensor,
        mDPhT: cute.Tensor,
        mDACS: cute.Tensor,
        mDACSRev: cute.Tensor,
        mSegsum: cute.Tensor,
        mQKDot: cute.Tensor,
        mGamma: cute.Tensor,
        mV: cute.Tensor,
        mMimoV: cute.Tensor,
        mDV: cute.Tensor,
        mDMimoV: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        dim = self.dim
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (dim, dim),
            warpgroup.OperandSource.SMEM,
        )
        s_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (dim, dim),
        )
        copy_g2s = _make_row_major_tiled_copy(
            self.dtype,
            dim,
            self.num_threads,
            copy_bits=self.dtype.width,
        )

        self.kernel(
            mK,
            mQ,
            mQT,
            mDPhT,
            mDACS,
            mDACSRev,
            mSegsum,
            mQKDot,
            mGamma,
            mV,
            mMimoV,
            mDV,
            mDMimoV,
            tiled_mma,
            s_layout,
            copy_g2s,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


def run_multi_chunk_state_apply_consumers(
    dim: int,
    rank: int,
    chunk_size: int,
    k: object,
    q: object,
    q_t: object,
    dphi_t: object,
    dA_cs: object,
    dA_cs_rev: object,
    segsum: object,
    qk_dot: object,
    gamma: object,
    v: object,
    mimo_v: object,
    dv: object,
    dmimo_v: object,
    stream: cuda.CUstream,
) -> None:
    nchunks = int(k.shape[0])
    if (dim, rank, chunk_size) != (64, 4, 16):
        raise ValueError(
            "MultiChunkStateApplyConsumersWGMMA currently specializes dim=64, rank=4, chunk_size=16"
        )
    if nchunks not in (2, 4, 8):
        raise ValueError("multi-chunk prototype currently supports nchunks in {2, 4, 8}")

    key = (dim, rank, chunk_size, nchunks)
    compiled = _CACHE_MULTI.get(key)
    if compiled is None:
        obj = MultiChunkStateApplyConsumersWGMMA(dim, rank, chunk_size, BFloat16)

        def fake_bf16_2d(shape: tuple[int, int]) -> cute.Tensor:
            return make_fake_tensor(
                BFloat16,
                shape,
                stride=(shape[1], 1),
                assumed_align=16,
            )

        def fake_bf16_3d(shape: tuple[int, int, int]) -> cute.Tensor:
            return make_fake_tensor(
                BFloat16,
                shape,
                stride=(shape[1] * shape[2], shape[2], 1),
                assumed_align=16,
            )

        def fake_bf16_4d(shape: tuple[int, int, int, int]) -> cute.Tensor:
            return make_fake_tensor(
                BFloat16,
                shape,
                stride=(
                    shape[1] * shape[2] * shape[3],
                    shape[2] * shape[3],
                    shape[3],
                    1,
                ),
                assumed_align=16,
            )

        def fake_f32_2d(shape: tuple[int, int]) -> cute.Tensor:
            return make_fake_tensor(
                Float32,
                shape,
                stride=(shape[1], 1),
                assumed_align=16,
            )

        def fake_f32_3d(shape: tuple[int, int, int]) -> cute.Tensor:
            return make_fake_tensor(
                Float32,
                shape,
                stride=(shape[1] * shape[2], shape[2], 1),
                assumed_align=16,
            )

        compiled = cute.compile(
            obj,
            fake_bf16_3d((nchunks, dim, dim)),
            fake_bf16_3d((nchunks, dim, dim)),
            fake_bf16_3d((nchunks, dim, dim)),
            fake_bf16_3d((nchunks, dim, dim)),
            fake_f32_2d((nchunks, chunk_size)),
            fake_f32_2d((nchunks, chunk_size)),
            fake_f32_3d((nchunks, chunk_size, chunk_size)),
            fake_bf16_4d((nchunks, chunk_size, rank, rank)),
            fake_f32_2d((nchunks, chunk_size)),
            fake_bf16_3d((nchunks, chunk_size, dim)),
            fake_bf16_2d((rank, dim)),
            fake_f32_3d((nchunks, chunk_size, dim)),
            fake_f32_2d((rank, dim)),
            stream,
        )
        _CACHE_MULTI[key] = compiled

    dl = lambda tensor: from_dlpack(tensor, assumed_align=16)
    compiled(
        dl(k),
        dl(q),
        dl(q_t),
        dl(dphi_t),
        dl(dA_cs),
        dl(dA_cs_rev),
        dl(segsum),
        dl(qk_dot),
        dl(gamma),
        dl(v),
        dl(mimo_v),
        dl(dv),
        dl(dmimo_v),
        stream,
    )
