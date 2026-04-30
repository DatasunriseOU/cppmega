"""Minimal fused CuTe tile for ``future_mask(K @ Q.T) @ dPhi``.

This keeps the Wave 5 scalar-copy GEMM as the correctness mode, but moves the
LKQ mask and apply GEMM into one CuTe kernel so LKQ is never written to global
memory on the fused path.  The tile is intentionally fixed at 64x64x64 BF16
for the current scan-owner probe.
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


class MaskedLKQApplyWGMMA:
    """One-CTA fused masked-apply tile.

    Inputs:
      K      [DIM, DIM]
      Q      [DIM, DIM]
      DPhT   [DIM, DIM] == dPhi.T
    Output:
      Apply  [DIM, DIM] == future_mask(K @ Q.T) @ dPhi
    """

    def __init__(self, dim: int = 64, rank: int = 4, dtype=BFloat16):
        self.dim = dim
        self.rank = rank
        self.dtype = dtype
        self.num_threads = 128

    @cute.jit
    def _apply_future_mask(self, acc_lkq: cute.Tensor, coord_lkq: cute.Tensor) -> None:
        for i in cutlass.range(cute.size(coord_lkq), unroll_full=True):
            row_t = coord_lkq[i][0] // self.rank
            col_t = coord_lkq[i][1] // self.rank
            if not (row_t < col_t):
                acc_lkq[i] = 0.0

    @cute.kernel
    def kernel(
        self,
        gK: cute.Tensor,
        gQ: cute.Tensor,
        gDPhT: cute.Tensor,
        gApply: cute.Tensor,
        tiled_mma: cute.TiledMma,
        s_layout: cute.ComposedLayout,
        copy_g2s: cute.TiledCopy,
        copy_s2g: cute.TiledCopy,
    ) -> None:
        tidx = arch.thread_idx()[0]
        dim = self.dim

        smem = SmemAllocator()
        sK = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sQ = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sDPhT = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)
        sApply = smem.allocate_tensor(self.dtype, s_layout.outer, swizzle=s_layout.inner)

        # Q is dead after the first WGMMA, so recycle its swizzled storage for
        # masked LKQ.  This is the only LKQ materialization in the fused path.
        sLKQ = sQ

        thr_g2s = copy_g2s.get_slice(tidx)
        cute.copy(copy_g2s, thr_g2s.partition_S(gK), thr_g2s.partition_D(sK))
        cute.copy(copy_g2s, thr_g2s.partition_S(gQ), thr_g2s.partition_D(sQ))
        cute.copy(copy_g2s, thr_g2s.partition_S(gDPhT), thr_g2s.partition_D(sDPhT))
        arch.sync_threads()

        wg_mma = tiled_mma.get_slice(tidx)
        shape_mnk = (dim, dim, dim)

        # GEMM1: LKQ = K @ Q.T
        _, tA1, tB1 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sK, sQ)
        acc_lkq = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
        sm90_utils.gemm(tiled_mma, acc_lkq, tA1, tB1, zero_init=True, wg_wait=0)

        # Mask in the accumulator before the BF16 R2S spill.  This is the
        # bounded chunk future mask: keep rows whose time index is before cols.
        coord_lkq = wg_mma.partition_C(cute.make_identity_tensor((dim, dim)))
        self._apply_future_mask(acc_lkq, coord_lkq)

        # Spill masked LKQ to swizzled smem as a valid WGMMA A operand.
        copy_lkq_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma,
            sLKQ,
            tidx,
            arch=90,
            position_independent=True,
        )
        lkq_frg = layout_utils.reshape_acc_to_frgA(acc_lkq)
        lkq_bf16 = cute.make_rmem_tensor_like(lkq_frg, self.dtype)
        lkq_bf16.store(lkq_frg.load().to(self.dtype))
        copy_lkq_r2s(lkq_bf16)
        arch.fence_view_async_shared()
        arch.sync_threads()

        # GEMM2: Apply = masked(LKQ) @ dPhi == masked(LKQ) @ DPhT.T
        _, tA2, tB2 = sm90_utils.partition_fragment_ABC(wg_mma, shape_mnk, sLKQ, sDPhT)
        acc_apply = cute.make_rmem_tensor(wg_mma.partition_shape_C((dim, dim)), Float32)
        sm90_utils.gemm(tiled_mma, acc_apply, tA2, tB2, zero_init=True, wg_wait=0)

        copy_apply_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma,
            sApply,
            tidx,
            arch=90,
        )
        apply_frg = layout_utils.reshape_acc_to_frgA(acc_apply)
        apply_bf16 = cute.make_rmem_tensor_like(apply_frg, self.dtype)
        apply_bf16.store(apply_frg.load().to(self.dtype))
        copy_apply_r2s(apply_bf16)
        arch.sync_threads()

        thr_s2g = copy_s2g.get_slice(tidx)
        cute.copy(copy_s2g, thr_s2g.partition_S(sApply), thr_s2g.partition_D(gApply))

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mQ: cute.Tensor,
        mDPhT: cute.Tensor,
        mApply: cute.Tensor,
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
        copy_s2g = _make_row_major_tiled_copy(
            self.dtype,
            dim,
            self.num_threads,
            copy_bits=self.dtype.width,
        )

        self.kernel(
            mK,
            mQ,
            mDPhT,
            mApply,
            tiled_mma,
            s_layout,
            copy_g2s,
            copy_s2g,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )


_CACHE: dict[tuple[int, int], object] = {}


def run_masked_lkq_apply(
    dim: int,
    rank: int,
    k: object,
    q: object,
    dphi_t: object,
    apply: object,
    stream: cuda.CUstream,
) -> None:
    key = (dim, rank)
    compiled = _CACHE.get(key)
    if compiled is None:
        obj = MaskedLKQApplyWGMMA(dim, rank, BFloat16)
        fake = lambda: make_fake_tensor(
            BFloat16,
            (dim, dim),
            stride=(dim, 1),
            assumed_align=16,
        )
        compiled = cute.compile(obj, fake(), fake(), fake(), fake(), stream)
        _CACHE[key] = compiled

    dl = lambda tensor: from_dlpack(tensor, assumed_align=16)
    compiled(dl(k), dl(q), dl(dphi_t), dl(apply), stream)
