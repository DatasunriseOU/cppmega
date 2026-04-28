#!/usr/bin/env python3
"""Probe TileLang fragment-layout limits for M2RNN summary VxV updates."""

from __future__ import annotations

import sys
import traceback

import torch


def main() -> int:
    import tilelang
    from tilelang import language as T

    V = 16
    be = T.dynamic("be")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        mat_layout = T.Fragment(
            [V, V],
            forward_thread_fn=lambda i, j: (i * V + j) % 128,
            forward_index_fn=lambda i, j: (i * V + j) // 128,
        )

        @T.prim_func
        def main_kernel(
            W: T.Tensor([be, V, V], T.float32),
            Out: T.Tensor([be, V, V], T.float32),
        ):
            with T.Kernel(be, threads=128) as be_i:
                P = T.alloc_fragment([V, V], T.float32)
                P_next = T.alloc_fragment([V, V], T.float32)

                for vi in T.serial(V):
                    for vj in T.serial(V):
                        P[vi, vj] = T.if_then_else(vi == vj, 1.0, 0.0)

                for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                    P_next[vi, vj] = P[vi, vj]
                    for vk in T.unroll(V):
                        P_next[vi, vj] += W[be_i, vk, vi] * P[vk, vj]

                for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                    Out[be_i, vi, vj] = P_next[vi, vj]

        return main_kernel

    print(f"tilelang={getattr(tilelang, '__version__', 'unknown')} path={tilelang.__file__}")
    kernel = kernel_builder()
    W = torch.randn(1, V, V, device="cuda", dtype=torch.float32)
    out = torch.empty_like(W)
    kernel(W, out)
    torch.cuda.synchronize()
    print("fragment_parallel_probe=ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
