# TileLang → cuTile Python (CUDA Tile) port: canonical GEMM

Author: research spike, 2026-04-10

## Scope and clarification

The user requested a "1:1 port to NVIDIA cuTile Python (part of
nvidia-cutlass-dsl / CUTLASS 4.x)". In reality there are **two
separate** Python DSLs from NVIDIA and they are frequently conflated:

| DSL               | Package                   | Level | Abstractions                                                                                                                            |
| ----------------- | ------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **CuTe DSL**      | `nvidia-cutlass-dsl`      | low   | `cute.make_layout`, `SmemAllocator`, `cute.copy`, `TiledMma`, `TiledCopy`, explicit pipeline stages. Essentially CUTLASS 3.x in Python. |
| **cuTile Python** | `cuda-tile` (`cuda.tile`) | high  | `ct.load`, `ct.store`, `ct.mma`, `ct.matmul`, `ct.full`, `ct.bid`, `ct.launch`. Built on Numba + Tile IR, announced at GTC 2025.        |

cuTile is the *tile-based* model ("think in tiles, not threads"). It is
**not** part of `nvidia-cutlass-dsl` — it lives in the `cuda.tile`
namespace and is distributed separately. The two projects do share
heritage (tile abstractions, Tile IR) but have different APIs and
different target audiences.

Because TileLang is semantically closest to cuTile (both are
tile-at-a-time DSLs that hide per-thread indexing), the port below
targets **cuTile Python** (`import cuda.tile as ct`). A sketch for the
CuTe DSL flavour is included at the end for completeness.

## Source: canonical TileLang GEMM

```python
import tilelang as tl
import tilelang.language as T

@tl.jit(out_idx=-1)
def matmul(M, N, K, block_M, block_N, block_K,
           dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N),
                      T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main
```

## Port: equivalent cuTile Python kernel

```python
# cutile_matmul.py
from math import ceil

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


@ct.kernel
def matmul_kernel(
    A,                       # (M, K) fp16
    B,                       # (K, N) fp16
    C,                       # (M, N) fp16
    tm: ConstInt,            # == block_M
    tn: ConstInt,            # == block_N
    tk: ConstInt,            # == block_K
):
    # 2D block id. cuTile launches a 1D grid by convention so we use
    # ct.bid(0) + shape of C to recover (by, bx). Many examples use a
    # 2D grid directly via ct.bid(0) / ct.bid(1); either is fine.
    by = ct.bid(1)           # row-tile index  (TileLang: by)
    bx = ct.bid(0)           # col-tile index  (TileLang: bx)

    # T.alloc_fragment((block_M, block_N), "float") + T.clear(C_local)
    acc = ct.full((tm, tn), 0, dtype=ct.float32)

    # T.ceildiv(K, block_K); cuTile exposes this as num_tiles along an axis.
    num_k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # T.Pipelined(..., num_stages=3) → plain Python for-loop.
    # cuTile schedules multi-stage async copies automatically; the user
    # cannot request "num_stages=3" explicitly from the kernel body.
    for k in range(num_k_tiles):
        # T.copy(A[by*block_M, k*block_K], A_shared)
        a_tile = ct.load(A, index=(by, k), shape=(tm, tk))
        # T.copy(B[k*block_K, bx*block_N], B_shared)
        b_tile = ct.load(B, index=(k, bx), shape=(tk, tn))
        # T.gemm(A_shared, B_shared, C_local)  (fused multiply-accumulate)
        acc = ct.mma(a_tile, b_tile, acc)

    # T.copy(C_local, C[by*block_M, bx*block_N])
    # Cast back to the storage dtype of C if needed.
    ct.store(C, index=(by, bx), tile=acc.astype(ct.float16))


def matmul(A: torch.Tensor, B: torch.Tensor,
           block_M: int = 128, block_N: int = 256, block_K: int = 64) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (ceil(N / block_N), ceil(M / block_M), 1)   # (bx, by, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        matmul_kernel,
        (A, B, C, block_M, block_N, block_K),
    )
    return C
```

## Side-by-side primitive mapping

| TileLang primitive                                  | cuTile Python equivalent                                                       | Notes                                                                                                                                                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@tl.jit(out_idx=-1)`                               | `@ct.kernel` (+ plain Python host wrapper that allocates `C`)                  | cuTile does not auto-allocate the output tensor; the caller does.                                                                                                                               |
| `@T.prim_func` / `T.Tensor((M,K), dtype)`           | Plain kernel function; tensors are passed as `torch.Tensor` / CuPy             | No explicit shape/dtype annotation on kernel params.                                                                                                                                            |
| `with T.Kernel(gx, gy, threads=128) as (bx, by)`    | `ct.bid(0)`, `ct.bid(1)` inside the kernel; `ct.launch(..., grid)` on the host | **`threads=128` has no counterpart.** cuTile hides per-thread scheduling entirely — the compiler picks the warp/CTA layout.                                                                     |
| `A_shared = T.alloc_shared((bM, bK), dtype)`        | *implicit* — created by `ct.load(A, index=..., shape=(tm, tk))`                | Shared memory is not a first-class user object.                                                                                                                                                 |
| `B_shared = T.alloc_shared((bK, bN), dtype)`        | *implicit* — `ct.load(B, ...)`                                                 | Same.                                                                                                                                                                                           |
| `C_local = T.alloc_fragment((bM, bN), accum_dtype)` | `acc = ct.full((tm, tn), 0, dtype=ct.float32)`                                 | Register-resident accumulator tile.                                                                                                                                                             |
| `T.clear(C_local)`                                  | Fused into `ct.full((...), 0, ...)`                                            | One call instead of two.                                                                                                                                                                        |
| `for k in T.Pipelined(n_k, num_stages=3):`          | `for k in range(num_k_tiles):`                                                 | **No explicit `num_stages` knob.** cuTile picks the pipeline depth automatically. Compile-time loops can be annotated with `ct.static_iter` but that is an unrolling hint, not a pipeline hint. |
| `T.copy(A[by*bM, k*bK], A_shared)`                  | `a_tile = ct.load(A, index=(by, k), shape=(tm, tk))`                           | Index is in *tiles*, not elements.                                                                                                                                                              |
| `T.copy(B[k*bK, bx*bN], B_shared)`                  | `b_tile = ct.load(B, index=(k, bx), shape=(tk, tn))`                           | Same.                                                                                                                                                                                           |
| `T.gemm(A_shared, B_shared, C_local)`               | `acc = ct.mma(a_tile, b_tile, acc)`                                            | `ct.mma(x, y, acc)` is fused multiply-accumulate and returns the new accumulator value. `ct.matmul(x, y)` is the non-accumulating variant.                                                      |
| `T.copy(C_local, C[by*bM, bx*bN])`                  | `ct.store(C, index=(by, bx), tile=acc.astype(ct.float16))`                     | Need explicit dtype cast from fp32 accumulator.                                                                                                                                                 |
| Grid arg `(T.ceildiv(N, bN), T.ceildiv(M, bM))`     | `grid = (ceil(N/bN), ceil(M/bM), 1)` passed to `ct.launch`                     | `ct.launch(stream, grid, kernel, args)`.                                                                                                                                                        |

## Primitives with no direct cuTile equivalent

1. **`threads=128`** — TileLang lets you choose the CTA width. cuTile
   does not: the compiler picks threads/warps from tile shape and the
   target architecture. You lose this tuning knob entirely.
2. **`num_stages=3`** in `T.Pipelined`** — There is no user-visible
   pipeline depth argument in cuTile as of the 13.x release. The
   compiler schedules multi-stage `cp.async` / TMA automatically. Users
   cannot force 2-stage vs 3-stage vs 4-stage. Observed consequence:
   you cannot manually trade SMEM pressure for latency hiding.
3. **`T.alloc_shared` / `T.alloc_fragment`** — No explicit shared or
   fragment allocation. Everything that looks like SMEM is an
   implicit side-effect of `ct.load(global_tensor, shape=...)`. This
   blocks common TileLang tricks:
   - reusing a single SMEM buffer across two operators (e.g. GEMM +
     softmax epilogue that needs SMEM scratch),
   - explicit ping-pong buffers,
   - cross-operator SMEM aliasing in fused kernels.
4. **Fragment-level primitives** — TileLang exposes
   `T.copy(fragment, fragment)`, `T.atomic_add`, warp reductions over
   fragments, etc. cuTile only models *tiles*; the fragment/register
   layer is compiler-owned.
5. **`T.copy` with arbitrary slices** (`A[by*bM, k*bK]`) — TileLang
   accepts element-space slicing. cuTile requires tile-space indices
   and the tile shape passed to `ct.load`. You can still load
   rectangular tiles but the index arithmetic semantics differ.
6. **Per-thread / warp-local control** — TileLang allows dropping down
   to warp-level primitives (`T.block_T.tma_store`,
   `T.block_T.wait_group`, etc.). cuTile has no escape hatch: if the
   compiler doesn't emit what you need, you can't force it.
7. **Custom swizzles / layouts** — TileLang allows explicit shared-mem
   swizzles; cuTile picks them automatically. (Note: CuTe DSL *does*
   expose layouts, but that is a different port target.)
8. **Autotuner integration** — TileLang has `@tl.autotune`. cuTile has
   no equivalent today (you must autotune by looping over `tm, tn, tk`
   from Python yourself).

## Feasibility

**Verdict: feasible for plain GEMM. Partially feasible for a full MIMO
(multi-input multi-output, e.g. fused-QKV / FlashAttention / structure-
aware attention) kernel.**

The canonical GEMM above ports cleanly — the code is roughly the same
length and every TileLang line has a corresponding cuTile line. The
kernel went from **18 lines** (TileLang prim_func body) to **19 lines**
(cuTile body), so LOC is essentially a wash. The *host* wrapper gains
~8 lines because cuTile does not auto-allocate outputs.

## Estimated LOC change for a real MIMO kernel

Calibrating against `cppmega/megatron/structure_batch.py` +
`custom_embedding.py` style kernels in this repo (which are the closest
MIMO analogue we already ship — sparse structure-aware embedding +
reduction + scatter):

| Component                                                                                             | TileLang LOC | cuTile LOC | Δ        |
| ----------------------------------------------------------------------------------------------------- | ------------ | ---------- | -------- |
| GEMM body (1 op, the example above)                                                                   | 18           | 19         | +1       |
| Fused QKV projection (3 outputs)                                                                      | ~45          | ~55        | +10      |
| FlashAttention v2 fwd (softmax, masking, scaling, online-renorm)                                      | ~180         | ~260       | +80      |
| FlashAttention v2 bwd (dQ/dK/dV, recompute)                                                           | ~320         | ~480       | +160     |
| Structure-aware attention (our MIMO case, with ngram-hash ingest + gather + masked softmax + scatter) | ~420         | ~600–650   | +180–230 |

**Why the blow-up on anything beyond a plain GEMM?**

1. **Lost `num_stages` control.** FlashAttention's fwd kernel depends
   on a 2-stage K/V pipeline to hide HBM latency while softmax online-
   renormalisation runs. With cuTile you can't ask for it; if the
   compiler chooses wrong, throughput drops and you have *no recourse
   from inside the kernel*. Workaround: split the K-loop into two
   manually staggered half-loops, which adds ~40 lines and makes the
   online-renorm math ugly.
2. **No explicit SMEM scratch.** FlashAttention's softmax needs a
   block-local scratch row for `m_i` / `l_i` running statistics. In
   TileLang this is one `T.alloc_shared((block_M,), "float")`. In
   cuTile you have to keep it in registers (so it has to live inside
   the tile abstraction), which forces you to broadcast/reduce via
   `ct.sum` / `ct.max` and re-materialise rather than in-place update.
3. **No fragment-level scatter.** Our structure-aware path does
   `T.atomic_add` on fragment tiles into a shared hash bucket. cuTile
   exposes atomics only on *global-memory tiles* (`ct.atomic_*` in the
   ops list), so the reduction has to go through global memory and
   back, or through a separate launch.
4. **Epilogue fusion is harder.** TileLang lets you write the epilogue
   (layernorm scale, activation, residual add) in the same prim_func
   as the matmul, sharing `C_local`. cuTile requires the epilogue to
   consume `acc` (a register tile) before the `ct.store`, so
   multi-output epilogues (e.g. attention + routing decisions) need
   either multiple stores or a second kernel.
5. **No autotuner.** You have to hand-roll a Python-level sweep over
   `(tm, tn, tk)`. For our workloads (NAM56R, structure attention)
   that is an extra ~60–100 lines of scaffolding.

So: net **~+40% LOC** plus the sweep scaffolding plus an almost-certain
loss of the final 10–25% of peak performance on anything that relied
on the `num_stages` / SMEM-scratch tricks.

## Expected performance difference

On plain square fp16 GEMM (the example above), cuTile is reported by
NVIDIA to match CuTe DSL / hand-written CUTLASS, which on H100/H200
means effectively SOL (within 2–3% of cuBLAS at the usual shapes).
TileLang also hits SOL on the same shapes. **Expect parity (±3%) for
the canonical GEMM.**

For the MIMO kernels we actually run in this repo (NAM56R mixed-A
attention, structure-aware embedding with scatter), based on the above
losses I expect:

| Workload                              | TileLang tok/s        | cuTile projected         | Δ               |
| ------------------------------------- | --------------------- | ------------------------ | --------------- |
| Plain GEMM (reference)                | SOL                   | SOL                      | ~0%             |
| Fused QKV                             | SOL                   | SOL − 2%                 | −2%             |
| FlashAttention fwd (H200)             | SOL − 3%              | SOL − 8 to −12%          | −5 to −9%       |
| FlashAttention bwd (H200)             | SOL − 5%              | SOL − 12 to −18%         | −7 to −13%      |
| NAM56R structure attention (our MIMO) | 183k tok/s (measured) | 150–165k tok/s projected | **−10 to −18%** |

The biggest risks are (a) the missing `num_stages` knob eating latency-
hiding on the K/V loop and (b) the forced global-memory round-trip for
scatter reductions on the structure path.

## Recommendation

- **For prototyping:** cuTile Python is *much* easier to write. If the
  target is a plain GEMM or a simple fused epilogue, port it — the
  code is shorter and the cognitive load is lower.
- **For our NAM56R / structure-aware production kernels:** do **not**
  port yet. The missing `num_stages` and `alloc_shared` escape hatches
  will cost us the 10–18% we've been clawing back since February, and
  cuTile has no counterpart to the `T.atomic_add(fragment, smem_hash)`
  trick we use for structure bucketing. Revisit when cuTile exposes
  (a) explicit pipeline-depth control, (b) an explicit SMEM-scratch
  API, and (c) fragment-level atomics.
- **Alternative target:** if a port *is* mandated, target **CuTe DSL**
  (`nvidia-cutlass-dsl`, `import cutlass.cute as cute`) instead of
  cuTile. CuTe DSL gives us layouts, `SmemAllocator`, `cute.copy`,
  `TiledMma`, and explicit pipeline staging — essentially a 1:1 map of
  TileLang's surface area at ~1.2× the LOC but no feature loss.

## Appendix: CuTe DSL sketch (for comparison)

For reference, the same kernel in **CuTe DSL** (which *is* part of
`nvidia-cutlass-dsl`) looks structurally much closer to TileLang:

```python
import cutlass
import cutlass.cute as cute
from cutlass.utils import SmemAllocator

@cute.kernel
def matmul_kernel(mA, mB, mC, sA_layout, sB_layout, tiled_mma,
                  tile_m: cutlass.Constexpr, tile_n: cutlass.Constexpr,
                  tile_k: cutlass.Constexpr, num_stages: cutlass.Constexpr):
    bidx, bidy, _ = cute.arch.block_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout)     # alloc_shared A
    sB = smem.allocate_tensor(mB.element_type, sB_layout)     # alloc_shared B

    thr_mma = tiled_mma.get_slice(cute.arch.thread_idx()[0])
    tCrA = thr_mma.partition_A(sA)
    tCrB = thr_mma.partition_B(sB)
    tCrC = cute.make_fragment_like(thr_mma.partition_C(mC))   # alloc_fragment
    tCrC.fill(0.0)                                            # T.clear

    k_tile_count = cute.size(mA, mode=[1]) // tile_k
    for k_tile in cutlass.range(k_tile_count, unroll=num_stages):  # T.Pipelined
        cute.copy(...sA...)                                   # T.copy to A_shared
        cute.copy(...sB...)                                   # T.copy to B_shared
        cute.arch.cp_async_wait()                             # explicit pipeline
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)          # T.gemm

    cute.copy(tCrC, thr_mma.partition_C(mC)[bidx, bidy])      # T.copy to C
```

CuTe DSL preserves every TileLang primitive (including explicit SMEM
allocation, explicit pipeline depth via `cutlass.range(unroll=...)` +
`cp_async_wait`, and fragment-level partitioning), at the cost of being
significantly more verbose than either TileLang or cuTile.

## Sources

- NVIDIA Developer Blog — "Simplify GPU Programming with NVIDIA CUDA
  Tile in Python" (cuTile overview + vector add example).
- NVIDIA Developer Blog — "How to Write High-Performance Matrix
  Multiply in NVIDIA CUDA Tile" (canonical cuTile matmul kernel).
- NVIDIA docs: `docs.nvidia.com/cuda/cutile-python/` — operation
  reference (`cuda.tile.mma`, `cuda.tile.matmul`, `cuda.tile.load`,
  `cuda.tile.store`, `cuda.tile.full`, `cuda.tile.bid`,
  `cuda.tile.num_tiles`, `cuda.tile.launch`, `cuda.tile.Constant`).
- NVIDIA CUTLASS 4.x docs: `docs.nvidia.com/cutlass/latest/` — CuTe
  DSL (`nvidia-cutlass-dsl`) reference and `cutlass/examples/python/
  CuTeDSL/ampere/tensorop_gemm.py` (for the CuTe DSL appendix).
