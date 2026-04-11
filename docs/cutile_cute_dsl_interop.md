# cuTile Python ↔ CuTe DSL interop

How the two NVIDIA tile-DSL packages we use — **cuTile Python** (`cuda.tile`, compiler-managed tiles, high-level) and **CuTe DSL** (`cutlass.cute`, explicit memory management, low-level) — can be combined in the same cppmega pipeline. Short answer: **host-level orchestration works trivially, inline embedding does not, binary-level interop is technically possible but impractical.**

## The three interop levels

### 1. Inline embedding (CuTe DSL code inside `@ct.kernel` body) — NOT POSSIBLE

- cuTile `_cext.pyi` exports only `launch`, `TileDispatcher`, `TileContext` — no FFI, no `foreign_call`, no `extern` hook
- cuTile AST parser (`cuda/tile/_passes/ast2hir.py`) accepts only cuTile primitives; CuTe DSL op names are not recognized
- `@ct.function` decorator only accepts pure-cuTile tile functions
- Would need a cuTile upstream feature request

### 2. Loading cuTile cubin into CuTe DSL via `ExternalBinaryModule` — NOT POSSIBLE out of the box

- `cutlass.cute.runtime.ExternalBinaryModule(file_path, enable_tvm_ffi=False)` exists at `nvidia_cutlass_dsl/python_packages/cutlass/base_dsl/export/external_binary_module.py`
- Accepts `.so` shared libs OR object files
- BUT the `__getattr__` path calls `decode_metadata_from_execution_engine(function_prefix, ...)` which expects **CuTe DSL-specific metadata** baked into the binary via `cute.compile(..., export=...)`. Raw cuTile cubins don't have this metadata → lookup fails
- Could theoretically be made to work via a custom adapter that hand-crafts CuTe-compatible metadata, but the effort vastly exceeds the benefit
- cuTile's `TileLibrary` class exposes zero public methods (`dir()` returns only dunders), so extracting the cubin path from a compiled kernel at runtime is also not officially supported

### 3. Host-level orchestration — WORKS TRIVIALLY

Both packages launch CUDA kernels via standard CUDA streams and both accept torch tensors through DLPack. A single Python function can interleave cuTile and CuTe DSL launches on the same stream with zero-copy tensor sharing.

**cuTile side:**
```python
import cuda.tile as ct
import torch

# ct.Array wraps a torch CUDA tensor (DLPack implicit)
q_arr = ct.Array(q_torch)                                 # shape/dtype from torch
dq_arr = ct.Array(dq_torch)
ct.launch(stream.cuda_stream, grid, cutile_kernel, (q_arr, dq_arr, ...))
```

**CuTe DSL side:**
```python
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute

# Verified on GB10: from_dlpack(torch.randn(64, 64, device='cuda'))
# returns Tensor<0x...@gmem o (64,64):(64,1)>
q_cute = from_dlpack(q_torch)
dq_cute = from_dlpack(dq_torch)
cute_kernel(q_cute, dq_cute, stream=stream.cuda_stream)  # pre-compiled via cute.compile()
```

Both share the same torch CUDA tensor memory with zero copies. Both accept `stream.cuda_stream` as the CUDA stream handle.

**Hybrid recipe:**
```python
def bwd_bwd_hybrid(q, k, v, ...):
    stream = torch.cuda.current_stream().cuda_stream
    # Phase 1 — simple pointwise/reduction in cuTile (brevity, compiler-managed)
    ct.launch(stream, grid1, cutile_phase1, (q_arr, dq_arr, ...))
    # Phase 2 — hot GEMM with smem/TMA/WGMMA control in CuTe DSL (compiled once, cached)
    cute_phase2_kernel(from_dlpack(q), from_dlpack(dq), stream=stream)
    # Phase 3 — cleanup in cuTile
    ct.launch(stream, grid3, cutile_phase3, (dk_arr, ...))
```

## When hybrid is worth it

| Target | Hybrid valuable? | Why |
|---|---|---|
| **H200 / sm_90a (Hopper)** | **YES** | CuTe DSL gives WGMMA + TMA + swizzled smem + warp specialization; cuTile leaves performance on the table on hot GEMMs. Hybrid CuTe-hot + cuTile-simple wins |
| **B200 / sm_100a (datacenter Blackwell)** | **YES** | CuTe DSL gives tcgen05 + TMEM + TMA + 2-SM UMMA; cuTile does not expose tcgen05. Hybrid wins |
| **GB10 / sm_121a (consumer Blackwell)** | **YES, for BF16/FP16** (newly verified 2026-04-11) | `cute.nvgpu.warp.MmaF16BF16Op` + `cute.nvgpu.cpasync.CopyBulkTensorTile*` + persistent scheduler all work out of the box. `blackwell_geforce/dense_gemm.py` proves it end-to-end. tcgen05 / FP4 paths are blocked (see `docs/gb10_sm121_hardware.md`) but BF16 is open |
| **RTX 5090 / sm_120a** | YES, for BF16/FP16 (same as sm_121a) | Same software situation |

## cuTile-specific rule: write the algorithm, not the memory placement

cuTile Python has **no explicit shared memory / register control** — the compiler handles all placement. There is no `alloc_shared`, no thread count control, no swizzle atom selection. When a cuTile kernel spills (high local memory traffic, compiler-chosen 384 threads × 168 regs/thread), the root cause is ALWAYS that the algorithm holds too many cold tiles in the same scope.

**Fix = restructure the algorithm** so each tile is produced, consumed, and drops out of scope minimally (SSA-style). **Not:** add pseudo-smem staging, not: try to pin registers, not: fight the compiler's thread count choice. See NVIDIA's own cuTile Python docs: *"The contents of a tile do not necessarily have a physical representation in memory"* and *"Threads cannot be explicitly identified or manipulated in tile programs."*

This rule applies ONLY to **cuTile Python**. **CuTe DSL is the opposite** — it gives you full control over memory placement (`cute.arch.alloc_smem`, swizzled smem atoms, TMA producer/consumer patterns, explicit warp specialization via `setmaxnreg`, etc.) and you SHOULD manage placement there.

## cuTile Python API recap

- `cuda.tile` package at version 1.2.0 (2026-03-05) on bench3/GB10
- Tile primitives: `Array`, `Tile`, `Scalar`, `Constant`
- Load/store: `load`, `store`, `gather`, `scatter`
- Factory: `zeros`, `ones`, `full`, `arange`
- GEMM: `mma`, `matmul`
- Reductions: `sum`, `max`, `min`, `prod`, `argmax`, `argmin`, `cumsum`, `cumprod`, `scan`, `reduce` (custom reducer, added 1.1.0)
- Shape: `cat`, `extract`, `broadcast_to`, `expand_dims`, `permute`, `transpose`, `reshape`
- Metaprogramming: `static_iter` (added 1.2.0), `static_eval`, `assert_`, `static_assert`
- Kernel decorator: `@ct.kernel(num_ctas=..., occupancy=..., opt_level=...)` — ONLY these three options
- Function decorator: `@ct.function` — nested functions and lambdas added 1.1.0
- Launch: `ct.launch(stream, grid, kernel, args)`
- **Memory model**: `MemoryScope` and `MemoryOrder` are for **atomic operation ordering only**, NOT for allocation. Do not confuse `MemoryScope.BLOCK` with shared memory.

## cuTile 1.1.0+ features relevant for optimization

- **`ct.reduce(tile, axis, combine_fn, identity)`** — custom reduction across axis. Useful for non-standard reducers (log-sum-exp, unique dA combiners in mamba scan bwd)
- **`Array.slice(axis, start, stop)`** — zero-copy view of an Array sliced along a single axis. Shares memory with the original. Useful for per-chunk dataflow factorization so each phase sees only its own data
- **Nested functions and lambdas** — factor kernel phases into separate `@ct.function`-decorated tile functions; scope boundaries naturally give the compiler cleaner lifetime information
- **`ct.static_iter(N, unroll=...)`** (1.2.0) — compile-time `for` loop; use for fixed-chunk unrolling when beneficial

## References

- NVIDIA cuTile Python docs: https://docs.nvidia.com/cuda/cutile-python/
- NVIDIA cuTile release notes: https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html
- NVIDIA cuTile operations reference: https://docs.nvidia.com/cuda/cutile-python/operations.html
- NVIDIA blog post: [Focus on Your Algorithm — NVIDIA CUDA Tile Handles the Hardware](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/)
- NVIDIA CUTLASS `cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` — the canonical BF16 persistent GEMM that works on sm_121a
- NVIDIA CUTLASS `cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm.py` — minimal warp MMA reference
- Local pivot test artifacts on GB10 (`/tmp/cute_gb10_test/`): `test_arch_override.py`, `test5_atom_probe.py`, `test6_monkeypatch.py`, `test_smem.py`
