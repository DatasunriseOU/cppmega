"""Fused CuTe tile for state/apply plus DV/DMIMO_V consumers.

This is the next bounded scan-owner probe after ``masked_lkq_apply``.  It keeps
the three 64x64 BF16 WGMMA products in one CTA:

  1. state = K @ DStates
  2. lkq   = future_mask(K @ Q.T)
  3. apply = lkq @ dPhi

``state`` and ``apply`` are rounded to BF16 in shared memory, matching the Wave
5/6 tile-chain semantics, but neither tile is written to global memory.  The
kernel then computes the scalar DV and DMIMO_V consumers in-kernel and writes
only those final FP32 outputs.
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
