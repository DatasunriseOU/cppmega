#!/usr/bin/env python3
"""Probe TileLang fragment-layout limits for M2RNN summary VxV updates.

The default variant keeps the known-failing mixed access pattern for regression
diagnostics.  ``--variant coeff`` tests a two-fragment rewrite that reads the
old prefix matrix only through ``P_old[vk, vj]`` inside the parallel VxV update.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["mixed", "coeff", "coeff-auto", "shared-old"],
        default="mixed",
        help="mixed reproduces the blocker; coeff removes the [vi,vj] read; coeff-auto also lets TileLang infer the parallel layout; shared-old keeps the old prefix in shared memory.",
    )
    args = parser.parse_args()

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
                P_shared = T.alloc_shared([V, V], T.float32)

                for vi in T.serial(V):
                    for vj in T.serial(V):
                        P[vi, vj] = T.if_then_else(vi == vj, 1.0, 0.0)
                        P_shared[vi, vj] = T.if_then_else(vi == vj, 1.0, 0.0)

                if args.variant == "shared-old":
                    for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                        P_next[vi, vj] = 0.0
                        for vk in T.unroll(V):
                            P_next[vi, vj] += (
                                W[be_i, vk, vi]
                                + T.if_then_else(vk == vi, 1.0, 0.0)
                            ) * P_shared[vk, vj]
                elif args.variant == "coeff-auto":
                    for vi, vj in T.Parallel(V, V):
                        P_next[vi, vj] = 0.0
                        for vk in T.unroll(V):
                            P_next[vi, vj] += (
                                W[be_i, vk, vi]
                                + T.if_then_else(vk == vi, 1.0, 0.0)
                            ) * P[vk, vj]
                else:
                    for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                        if args.variant == "mixed":
                            P_next[vi, vj] = P[vi, vj]
                        else:
                            P_next[vi, vj] = 0.0
                        for vk in T.unroll(V):
                            P_next[vi, vj] += (
                                W[be_i, vk, vi]
                                + T.if_then_else(
                                    args.variant == "coeff" and vk == vi,
                                    1.0,
                                    0.0,
                                )
                            ) * P[vk, vj]

                if args.variant == "coeff-auto":
                    for vi, vj in T.Parallel(V, V):
                        Out[be_i, vi, vj] = P_next[vi, vj]
                else:
                    for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                        Out[be_i, vi, vj] = P_next[vi, vj]

        return main_kernel

    print(f"tilelang={getattr(tilelang, '__version__', 'unknown')} path={tilelang.__file__}")
    kernel = kernel_builder()
    W = torch.randn(1, V, V, device="cuda", dtype=torch.float32)
    out = torch.empty_like(W)
    kernel(W, out)
    torch.cuda.synchronize()
    print(f"fragment_parallel_probe=ok variant={args.variant}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
