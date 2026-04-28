# M2RNN ParaRNN Tiled CUDA Status - 2026-04-28

Worktree: `/home/dave/source/cppmega/.claude/worktrees/m2rnn-tiled-cuda`  
Branch: `worker/m2rnn-tiled-cuda`  
Base commit for this continuation: `c26f237 feat(m2rnn): add tiled CUDA ParaRNN prototype`

## Scope

Implemented the first CUDA extension probe for a true tiled/streaming
ParaRNN Newton scan.  The kernels use one CUDA block per
`(B, H, K, tile)` chain segment and cooperatively assemble the dense
`V x V` affine map for `V <= 16` in shared/register state.

This is not the Apple one-thread-per-equation dense-Jacobian port:

- The kernel does not materialize full per-token Jacobian
  `A[B,S,H,K,V,V]`.
- The production summary kernel writes only tile summary `(A_tile,b_tile)`.
- The tile summary scan over `(A_tile,b_tile)` now runs in the CUDA
  extension; the previous Python `for tile: einsum(...)` path remains only as
  a test/debug reference.
- The production apply kernel receives scanned tile carries, recomputes the
  per-token local prefix inside the tile, and writes `delta[Be,S,V]`.
- `local_prefix[Be,S,V,V]` is no longer allocated in the production forward
  path.  It remains only in the debug entrypoint used by the local affine
  unit test.

## Files

- `cppmega/megatron/cuda_ext/m2rnn_tiled_affine_scan.cu`
- `cppmega/megatron/m2rnn_pararnn_tiled_cuda.py`
- `tests/test_m2rnn_pararnn_tiled_cuda.py`
- `tools/probes/m2rnn_pararnn_tiled_cuda_probe.py`
- `docs/status/m2rnn_tiled_cuda_2026_04_28.md`

## Verification

Device: NVIDIA GB10, CUDA capability `(12, 1)`, PyTorch
`2.13.0.dev20260417+cu132`.

Commands run after the GPU summary-scan patch:

```bash
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_cuda.py tests/test_m2rnn_pararnn_tiled_cuda.py tools/probes/m2rnn_pararnn_tiled_cuda_probe.py
CPPMEGA_VERBOSE_EXT_BUILD=1 pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s
CPPMEGA_VERBOSE_EXT_BUILD=1 python tools/probes/m2rnn_pararnn_tiled_cuda_probe.py --B 1 --S 33 --H 2 --K 4 --V 16 --tile-size 8 --max-its 6
BENCH_B=1 BENCH_S=1024 BENCH_H=4 BENCH_K=32 BENCH_V=16 BENCH_WARMUP=1 BENCH_ITERS=3 BENCH_TORCH=0 python scripts/bench_m2rnn.py
```

Results:

- CUDA test run: `7 passed, 19 warnings`.
- Probe exit code: pass.
- Triton comparison bench for `B=1,S=1024,H=4,K=32,V=16,bf16`:
  `triton fwd = 0.52 ms/iter`, `triton fwd+bwd = 2.62 ms/iter`.

Probe parity for `B=1,S=33,H=2,K=4,V=16,tile=8,max_its=6`:

- tiled CUDA vs sequential output max abs: `5.781650543212891e-06`
- tiled CUDA vs sequential h_final max abs: `1.7434358596801758e-06`
- PyTorch ParaRNN vs sequential output max abs: `1.4901161193847656e-07`
- PyTorch ParaRNN vs sequential h_final max abs: `2.2351741790771484e-08`
- prototype wall time, including CUDA summary scan and CUDA recompute apply:
  `239.91 ms` on the small probe shape

GB10 bf16 timing sweep for `B=1,S=1024,H=4,K=32,V=16,max_its=3`:

| tile_size | before Python summary scan | after CUDA summary scan |
| ---: | ---: | ---: |
| 8 | 14.34 ms | 13.41 ms |
| 16 | 12.46 ms | 11.66 ms |
| 32 | 11.59 ms | 11.12 ms |
| 64 | 11.36 ms | 11.21 ms |
| 128 | 11.54 ms | 11.45 ms |

## ptxas / Resources

Verbose extension build emitted:

```text
ptxas info    : Compiling entry function ... m2rnn_tile_summary_kernel ... for 'sm_121'
ptxas info    : Function properties ...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 94 registers, used 1 barriers, 3456 bytes smem
ptxas info    : Compiling entry function ... m2rnn_apply_tile_prefix_kernel ... for 'sm_121'
ptxas info    : Function properties ...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 3520 bytes smem
ptxas info    : Compiling entry function ... m2rnn_scan_tile_summaries_kernel ... for 'sm_121'
ptxas info    : Function properties ...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 1 barriers, 128 bytes smem
ptxas info    : Compiling entry function ... m2rnn_local_tile_scan_debug_kernel ... for 'sm_121'
ptxas info    : Function properties ...
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, used 1 barriers, 3456 bytes smem
```

Resource summary:

- summary kernel: `94` registers/thread, `3456` bytes smem, `0` spills
- apply kernel: `90` registers/thread, `3520` bytes smem, `0` spills
- tile summary scan kernel: `26` registers/thread, `128` bytes smem,
  `0` spills
- debug kernel: `96` registers/thread, `3456` bytes smem, `0` spills
- launch bounds: `256` threads/block, min `2` blocks/SM for all four kernels

## Memory Accounting

For probe shape `B=1,S=33,H=2,K=4,V=16,tile=8`, `Be=B*H*K=8`:

| Tensor | Bytes |
| --- | ---: |
| forbidden full Jacobian `Be*S*V*V*f32` | 270,336 |
| `h_trajectory` | 16,896 |
| `delta` | 16,896 |
| `tile_A` | 40,960 |
| `tile_b` | 2,560 |
| `tile_inputs` | 2,560 |
| debug-only `local_delta` | 16,896 |
| debug-only `local_prefix` | 270,336 |

The forbidden full Jacobian is not allocated.  Production forward also no
longer allocates `local_prefix[Be,S,V,V]`; the equally sized tensor is only
available through the explicit `local_tile_scan_debug()` test/probe API.

Production accounting now keeps only:

- `h_trajectory` or a recompute/checkpointed equivalent,
- `delta` for the current Newton update,
- `tile_A/tile_b` summaries,
- scanned `tile_inputs[Be,n_tiles,V]`, now produced by CUDA,
- no production `local_prefix`.

## Input Dtypes

The CUDA extension kernels take fp32 tensors and keep fp32 solve accumulators.
The Python forward accepts fp32 or bf16 inputs; bf16 inputs are converted to
fp32 before the Newton solve and the output/h_final tensors are fp32.  The
test suite includes bf16 input coverage against a fp32 reference built from
the quantized bf16 inputs.

## Next Production Step

Remaining work:

1. Fuse the Newton update (`h += omega * delta`) into the apply kernel or a
   small CUDA update kernel.
2. Replace the sequential per-chain summary scan kernel with a true parallel
   tile-prefix composition if `n_tiles` becomes the dominant cost.
3. Remove or gate the debug local-prefix entrypoint once local affine tests no
   longer need it.

The largest previous production allocation target is removed; the remaining
prototype compromise is duplicate summary/apply recomputation, not global
per-token prefix storage or Python-side tile-summary scan.

## Optimization Cycle 3 - 2026-04-28

Continuation base: `16d625d perf(m2rnn): scan CUDA tile summaries on GPU`.

### Search / References

Local context reviewed:

- This status file, especially the previous ptxas and GB10 tile sweep.
- `cppmega/megatron/m2rnn_pararnn_tiled_cuda.py`
- `cppmega/megatron/cuda_ext/m2rnn_tiled_affine_scan.cu`
- `tools/probes/m2rnn_pararnn_tiled_cuda_probe.py`
- `scripts/bench_m2rnn.py`

External docs searched before profiling/patching:

- NVIDIA CUDA Programming Guide, CUDA Graphs: graphs reduce repeated stream
  launch setup and are useful when short kernels become CPU-launch limited.
  https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- NVIDIA CUDA Graph Best Practice for PyTorch: launch overhead includes Python,
  framework C++ dispatch, runtime, and driver layers; CUDA Graph capture can
  amortize those costs for static workflows.
  https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/introduction.html
- PyTorch CUDA Graphs blog: replay skips Python/C++/driver dispatch and submits
  the captured work with one `cudaGraphLaunch`.
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyTorch `torch.utils.cpp_extension` docs: JIT extension build/load mechanics
  and architecture targeting.
  https://docs.pytorch.org/docs/stable/cpp_extension.html
- NVIDIA CUDA Occupancy Calculator docs: register use constrains active blocks
  and occupancy.
  https://docs.nvidia.com/cuda/archive/11.7.1/cuda-occupancy-calculator/index.html
- NVIDIA CCCL/CUB docs: CUB provides warp/block/device scan primitives, useful
  if tile summary scan becomes important.
  https://nvidia.github.io/cccl/cub/index.html
- NVIDIA Nsight Systems User Guide: CUDA API tracing is the right tool for
  separating CUDA API launch gaps from GPU kernel time.
  https://docs.nvidia.com/nsight-systems/UserGuide/

### Stage Profiling Probe

Added `tools/probes/m2rnn_tiled_cuda_stage_profile.py`.  It reports:

- cold/cached extension load wall time,
- dtype conversion and contiguous prep,
- one-time `h` and workspace allocation,
- per-Newton `summary`, `scan`, `apply`, and ATen update timing,
- final layout/einsum timing,
- whole forward wall time.

Timing uses CUDA events plus wall time around each stage.  The per-stage wall
and CUDA-event times are effectively identical for the heavy stages, so Python
extension dispatch is not the dominant cost on this shape.

### Baseline, Before Patch

Command:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=1 python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 1 --iters 3 --dtype bf16 --json /tmp/m2rnn_stage_baseline_t32.json
```

GB10 bf16 `B=1,S=1024,H=4,K=32,V=16,max_its=3,tile=32`:

| Stage | Mean ms |
| --- | ---: |
| prep/cast/contiguous | 0.062 |
| h alloc | 0.025 |
| summary, per Newton | 1.685 |
| scan, per Newton | 0.023 |
| apply, per Newton | 1.877 |
| ATen update, per Newton | 0.122 |
| final layout | 0.071 |
| final einsum | 0.106 |
| whole forward wall | 11.318 |

Tile 64 baseline whole forward wall: `11.232 ms`; per-Newton summary
`1.710 ms`, scan `0.018 ms`, apply `1.881 ms`, update `0.123 ms`.

Diagnosis from baseline: scan and Python launch overhead are tiny.  The
runtime is dominated by the two recomputing CUDA kernels: summary plus apply
are about `3.56 ms` per Newton iteration, or roughly `10.7 ms` of the
`11.3 ms` forward.

### Patch

Implemented preallocated extension outputs:

- `tile_summaries_out(q, k, v, W, xf, h, h0, tile_A, tile_b, tile_size)`
- `scan_tile_summaries_out(tile_A, tile_b, tile_inputs)`
- `apply_tile_prefixes_out(..., tile_inputs, delta, tile_size)`

The production Python forward now allocates `tile_A`, `tile_b`, `tile_inputs`,
and `delta` once per forward and reuses them across Newton iterations.  The old
allocating APIs remain for compatibility/debug tests.

I also tried fusing apply plus Newton update into the apply kernel.  It was
correct but slower on GB10 (`whole_forward_wall_ms` reruns in the `26-45 ms`
range, with apply-update kernel time above the old apply+update pair), so that
variant was discarded and not kept.

### After Patch

Command:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=1 python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 1 --iters 3 --dtype bf16 --json /tmp/m2rnn_stage_after_out_t32.json
```

GB10 bf16 `B=1,S=1024,H=4,K=32,V=16,max_its=3,tile=32`:

| Stage | Mean ms |
| --- | ---: |
| prep/cast/contiguous | 0.062 |
| h alloc | 0.024 |
| workspace alloc | 0.013 |
| summary, per Newton | 1.679 |
| scan, per Newton | 0.021 |
| apply, per Newton | 1.872 |
| ATen update, per Newton | 0.115 |
| final layout | 0.067 |
| final einsum | 0.100 |
| whole forward wall | 11.197 |

Tile 64 after patch whole forward wall: `11.258 ms`; tile 32 remains the best
measured setting in this cycle.

The measurable improvement is small but positive for tile 32:

- `11.318 ms -> 11.197 ms`, about `1.1%` faster.
- summary/apply kernel resources unchanged.
- The allocator/API overhead is not the reason for the Triton gap.

### Correctness / Tests

Commands:

```bash
python -m py_compile \
  cppmega/megatron/m2rnn_pararnn_tiled_cuda.py \
  tests/test_m2rnn_pararnn_tiled_cuda.py \
  tools/probes/m2rnn_pararnn_tiled_cuda_probe.py \
  tools/probes/m2rnn_tiled_cuda_stage_profile.py

CPPMEGA_VERBOSE_EXT_BUILD=0 pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s

CPPMEGA_VERBOSE_EXT_BUILD=0 python tools/probes/m2rnn_pararnn_tiled_cuda_probe.py \
  --B 1 --S 33 --H 2 --K 4 --V 16 --tile-size 8 --max-its 6 \
  --json /tmp/m2rnn_probe_after_out.json

BENCH_B=1 BENCH_S=1024 BENCH_H=4 BENCH_K=32 BENCH_V=16 \
  BENCH_WARMUP=1 BENCH_ITERS=3 BENCH_TORCH=0 python scripts/bench_m2rnn.py
```

Results:

- `pytest`: `7 passed, 19 warnings`.
- Probe parity, tiled CUDA vs sequential: output max abs
  `5.781650543212891e-06`, h_final max abs
  `1.7434358596801758e-06`.
- PyTorch ParaRNN vs sequential: output max abs `1.4901161193847656e-07`,
  h_final max abs `2.2351741790771484e-08`.
- Small probe CUDA wall: `251.93 ms`.
- Triton comparison on the target shape: fwd `0.50 ms/iter`,
  fwd+bwd `4.25 ms/iter`.

### ptxas / Resources After Patch

Verbose extension build emitted:

```text
ptxas info    : Compiling entry function ... m2rnn_local_tile_scan_debug_kernel ... for 'sm_121'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, used 1 barriers, 3456 bytes smem
ptxas info    : Compiling entry function ... m2rnn_scan_tile_summaries_kernel ... for 'sm_121'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 26 registers, used 1 barriers, 128 bytes smem
ptxas info    : Compiling entry function ... m2rnn_apply_tile_prefix_kernel ... for 'sm_121'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 3520 bytes smem
ptxas info    : Compiling entry function ... m2rnn_tile_summary_kernel ... for 'sm_121'
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 94 registers, used 1 barriers, 3456 bytes smem
```

### Bottleneck Diagnosis

The production CUDA path is not currently CPU-launch dominated on the target
shape.  A CUDA Graph or batched per-Newton host API may still help integration
overheads in a larger model, but the standalone forward profile shows the
dominant cost is device work inside `m2rnn_tile_summary_kernel` and
`m2rnn_apply_tile_prefix_kernel`.

The core issue is algorithmic recomputation: summary and apply both walk every
token in every Newton iteration and both rebuild dense `V x V` transition
state.  With `max_its=3`, this means six full recurrent tile passes.  The scan
kernel is already negligible at this sequence length and tile count.

Most promising next work:

1. Specialize the hot `V=16,tile=32` path to reduce register pressure and
   shared-memory round trips in summary/apply.
2. Revisit apply+update fusion only if it can avoid the extra old-`h` load and
   avoid cross-tile races; the naive one-kernel write-next-h variant was worse.
3. Explore reducing the duplicate summary/apply recomputation rather than
   optimizing scan or Python dispatch.

## Optimization Cycle 4 - 2026-04-28

Continuation base requested by user: `fd5742e`.  No git commit was created.

### Search / References

Local context reviewed:

- `cppmega/megatron/cuda_ext/m2rnn_tiled_affine_scan.cu`
- `cppmega/megatron/m2rnn_pararnn_tiled_cuda.py`
- `tools/probes/m2rnn_tiled_cuda_stage_profile.py`
- `tools/probes/m2rnn_pararnn_tiled_cuda_probe.py`
- prior sections in this status file

External CUDA docs searched before patching:

- CUDA Cooperative Groups: `tiled_partition` creates fixed-size subgroups; group
  creation and collectives are collective operations all participating threads
  must reach.  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- CUDA C++ Programming Guide, Cooperative Groups API: `thread_block_tile` exposes
  `shfl`, `shfl_down`, `shfl_xor`, `sync`, and notes that templated tile sizes
  enable better compile-time optimization.  https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html
- CUDA C++ Programming Guide, Occupancy Calculator: `cudaOccupancy*` APIs and
  Nsight Compute occupancy reporting convert active blocks to active warps and
  occupancy.  https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html
- CUB/CCCL docs: CUB provides warp-wide and block-wide collectives such as
  `WarpScan` and `BlockScan`, but the measured bottleneck here is summary/apply
  math, not tile scan.  https://nvidia.github.io/cccl/unstable/cub/index.html

### Baseline Profile

Command:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=0 python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 1 --iters 3 --dtype bf16 --json /tmp/m2rnn_cycle4_baseline_stage.json
```

GB10 bf16 target shape, default kernels:

| Stage | Mean ms |
| --- | ---: |
| summary, per Newton | 1.702 |
| scan, per Newton | 0.022 |
| apply, per Newton | 1.874 |
| update, per Newton | 0.134 |
| whole forward wall | 11.215 |

Nsight Compute command:

```bash
ncu --target-processes all \
  --kernel-name 'regex:m2rnn_.*(tile_summary|apply_tile_prefix).*' \
  --launch-count 6 --section LaunchStats --section Occupancy \
  python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 \
  --max-its 3 --warmup 0 --iters 1 --dtype bf16
```

Nsight result for both default hot kernels:

| Kernel | Threads/block | Registers/thread | Static smem | Theoretical occupancy | Achieved occupancy | Limiter |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `m2rnn_tile_summary_kernel` | 256 | 94 | 3.46 KiB | 33.33% | ~33.08% | registers |
| `m2rnn_apply_tile_prefix_kernel` | 256 | 90 | 3.52 KiB | 33.33% | ~33.09% | registers |

Diagnosis: the one-block-per-chain/tile strategy launches enough blocks
(`4096` blocks, `42.67` waves/SM), but each block is register-limited to two
active blocks/SM.  Within each token step, only `V=16` threads do the vector
matvec/rhs/d update work, while the `VxV` transition update uses 256 threads
with a serial length-16 reduction per output element and full block barriers
between every phase.  Scan is still negligible; Python is not the bottleneck.

### Patch: V=16 Warp-Per-Row Variant

Added opt-in experimental kernels:

- `m2rnn_tile_summary_v16_warprow_kernel`
- `m2rnn_apply_tile_prefix_v16_warprow_kernel`

They map one warp to each matrix/vector row for `V=16`, use warp shuffle
reductions for the vector matvec, `d_next`, and carry prefix dot products, and
keep the existing fp32 accumulator behavior.  The variant is disabled by
default and selected with:

```bash
CPPMEGA_M2RNN_WARPROW_V16=1
```

`tools/probes/m2rnn_tiled_cuda_stage_profile.py` now reports
`kernel_variant` and profiles the selected extension entrypoints.

### ptxas / Resources

Verbose extension build emitted:

```text
m2rnn_apply_tile_prefix_v16_warprow_kernel:
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  Used 72 registers, used 1 barriers, 3520 bytes smem
m2rnn_tile_summary_v16_warprow_kernel:
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  Used 74 registers, used 1 barriers, 3456 bytes smem
m2rnn_apply_tile_prefix_kernel:
  Used 90 registers, 0 spills
m2rnn_tile_summary_kernel:
  Used 94 registers, 0 spills
```

The new variant reduced registers but not occupancy.

Nsight Compute for the opt-in variant:

```bash
CPPMEGA_M2RNN_WARPROW_V16=1 ncu --target-processes all \
  --kernel-name 'regex:m2rnn_.*v16_warprow.*' --launch-count 6 \
  --section LaunchStats --section Occupancy \
  python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 \
  --max-its 3 --warmup 0 --iters 1 --dtype bf16
```

| Kernel | Threads/block | Registers/thread | Static smem | Theoretical occupancy | Achieved occupancy | Limiter |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `m2rnn_tile_summary_v16_warprow_kernel` | 512 | 74 | 3.46 KiB | 33.33% | ~33.31% | registers + smem |
| `m2rnn_apply_tile_prefix_v16_warprow_kernel` | 512 | 72 | 3.52 KiB | 33.33% | ~33.33% | registers + smem |

### Timings

Opt-in variant command:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=0 CPPMEGA_M2RNN_WARPROW_V16=1 \
  python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 \
  --max-its 3 --warmup 1 --iters 3 --dtype bf16 \
  --json /tmp/m2rnn_cycle4_warprow_stage.json
```

| Stage | Default mean ms | `v16_warprow` mean ms |
| --- | ---: | ---: |
| summary, per Newton | 1.702 | 2.633 |
| scan, per Newton | 0.022 | 0.021 |
| apply, per Newton | 1.874 | 3.056 |
| update, per Newton | 0.134 | 0.126 |
| whole forward wall | 11.215 | 17.490 |

The new strategy is slower on GB10.  It does parallelize the small row
reductions, but the dominant `M_next = P @ M` work still has one lane per
output element doing a length-16 serial reduction.  The larger 512-thread block
does not increase active warps/SM, and it doubles launched thread count for the
same 256 active matrix elements per token.  Therefore the variant does not beat
the old path and remains default-off.

### Correctness / Tests

Commands:

```bash
python -m py_compile \
  cppmega/megatron/m2rnn_pararnn_tiled_cuda.py \
  tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  tests/test_m2rnn_pararnn_tiled_cuda.py

CPPMEGA_VERBOSE_EXT_BUILD=0 pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s

CPPMEGA_VERBOSE_EXT_BUILD=0 CPPMEGA_M2RNN_WARPROW_V16=1 \
  pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s

CPPMEGA_VERBOSE_EXT_BUILD=0 python tools/probes/m2rnn_pararnn_tiled_cuda_probe.py \
  --B 1 --S 33 --H 2 --K 4 --V 16 --tile-size 8 --max-its 6 \
  --json /tmp/m2rnn_cycle4_default_probe.json

CPPMEGA_VERBOSE_EXT_BUILD=1 CPPMEGA_M2RNN_WARPROW_V16=1 \
  python tools/probes/m2rnn_pararnn_tiled_cuda_probe.py \
  --B 1 --S 17 --H 2 --K 4 --V 16 --tile-size 8 --max-its 4 \
  --json /tmp/m2rnn_cycle4_warprow_probe.json

BENCH_B=1 BENCH_S=1024 BENCH_H=4 BENCH_K=32 BENCH_V=16 \
  BENCH_WARMUP=1 BENCH_ITERS=3 BENCH_TORCH=0 python scripts/bench_m2rnn.py
```

Results:

- default pytest: `7 passed, 19 warnings`
- opt-in warprow pytest: `7 passed, 19 warnings`
- default probe parity vs sequential: output max abs
  `5.781650543212891e-06`, h_final max abs
  `1.7434358596801758e-06`
- opt-in probe parity vs sequential: output max abs
  `5.334615707397461e-06`, h_final max abs
  `3.516674041748047e-06`
- Triton comparison bench on target shape: fwd `0.51 ms/iter`,
  fwd+bwd `2.06 ms/iter`

### Conclusion

The current CUDA path is slow because the hot work is still the duplicated
summary/apply recurrent tile math, not scan, allocation, or Python dispatch.
The block strategy is poor for the small vector phases, but simply mapping one
warp per row does not improve active occupancy or the `P @ M` inner loop, so it
regresses total time.  The next promising strategy should attack the matrix
composition itself: split the length-16 `P @ M` reductions across lanes or
avoid recomputing the full dense prefix in both summary and apply.

## Optimization Cycle 5 - 2026-04-28

Continuation base requested by user: `c3a22a4`.  No git commit was created.

### Search / References

Local context reviewed:

- `cppmega/megatron/m2rnn_pararnn_tiled_cuda.py`
- `cppmega/megatron/cuda_ext/m2rnn_tiled_affine_scan.cu`
- `tools/probes/m2rnn_tiled_cuda_stage_profile.py`
- `tools/probes/m2rnn_pararnn_tiled_cuda_probe.py`
- `tests/test_m2rnn_pararnn_tiled_cuda.py`
- prior sections in this status file

External CUDA/CUB references checked before patching:

- NVIDIA CUDA Occupancy Calculator: occupancy is active warps divided by
  maximum warps, and registers/shared memory constrain active blocks per SM.
  https://docs.nvidia.com/cuda/archive/11.7.1/cuda-occupancy-calculator/index.html
- CUDA C++ Programming Guide: occupancy APIs such as
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` predict active blocks from
  block size and shared memory, then derive active warps/occupancy.
  https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html
- CUDA C++ Best Practices Guide: register allocation and shared-memory
  partitioning make occupancy/resource tradeoffs architecture dependent.
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CCCL/CUB `WarpScan`: useful for warp-wide collectives, but not directly
  decisive here because the measured scan stage remains negligible.
  https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html

### Decision Probe / Patch

Added `tools/probes/m2rnn_cuda_variant_decision.py`.  It compares, on the same
input tensors and shape:

- default tiled CUDA,
- opt-in `CPPMEGA_M2RNN_WARPROW_V16=1`,
- Triton reference forward.

The probe emits JSON and a Markdown decision doc:

- `docs/status/m2rnn_cuda_variant_decision_2026_04_28.md`

It encodes the decision rule for this branch: a CUDA candidate must beat the
default CUDA path by more than `20%` on CUDA-event timing before it becomes a
production follow-up candidate.  Otherwise the CUDA branch stays
resource/diagnostic and Triton remains the active path.

The probe now refuses to run by default if `nvidia-smi` reports unrelated CUDA
compute processes.  This was added after later retry runs were polluted by
concurrent `pretrain_mamba.py`, kernel bench, and pytest processes on the same
GB10.

Added a unit test that proves the slow warprow path stays opt-in:

- no env var: disabled,
- `CPPMEGA_M2RNN_WARPROW_V16=0`: disabled,
- `CPPMEGA_M2RNN_WARPROW_V16=1` with `V=16`: enabled,
- `V!=16`: disabled even with the env var.

### Timings

Uncontended Cycle 5 decision probe:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=0 python tools/probes/m2rnn_cuda_variant_decision.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 2 --iters 5 \
  --json /tmp/m2rnn_cycle5_decision.json \
  --markdown docs/status/m2rnn_cuda_variant_decision_2026_04_28.md
```

| Variant | CUDA event ms/iter | Wall ms/iter | Speedup vs default event | Decision |
| --- | ---: | ---: | ---: | --- |
| default CUDA | 11.256 | 11.257 | 1.00x | default |
| `v16_warprow` opt-in | 17.425 | 17.425 | 0.65x | diagnostic only |
| Triton reference | 0.500 | 0.501 | 22.50x | active reference |

Stage profiler, same target shape, default CUDA:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=0 python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 2 --iters 3 --dtype bf16 \
  --json /tmp/m2rnn_cycle5_default_stage.json
```

Clean run before later GPU contention: summary `1.824 ms`, scan `0.035 ms`,
apply `2.072 ms`, update `0.132 ms` per Newton iteration; whole forward
`12.51 ms` in that stage-profiler run.  The aggregate decision probe above is
the canonical same-input CUDA/Triton comparison.

Stage profiler for opt-in warprow:

```bash
CPPMEGA_VERBOSE_EXT_BUILD=0 CPPMEGA_M2RNN_WARPROW_V16=1 \
  python tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  --B 1 --S 1024 --H 4 --K 32 --V 16 --tile-size 32 --max-its 3 \
  --warmup 2 --iters 3 --dtype bf16 \
  --json /tmp/m2rnn_cycle5_warprow_stage.json
```

Warprow stage profile: summary `2.632 ms`, scan `0.023 ms`, apply
`3.068 ms`, update `0.121 ms` per Newton iteration; whole forward `17.50 ms`.

### ptxas / Resources

Fresh extension build command:

```bash
rm -rf /tmp/cppmega_m2rnn_cycle5_ext
TORCH_EXTENSIONS_DIR=/tmp/cppmega_m2rnn_cycle5_ext CPPMEGA_VERBOSE_EXT_BUILD=1 \
  python tools/probes/m2rnn_pararnn_tiled_cuda_probe.py \
  --B 1 --S 17 --H 2 --K 4 --V 16 --tile-size 8 --max-its 4 \
  --json /tmp/m2rnn_cycle5_ptxas_probe.json
```

ptxas emitted:

| Kernel | Registers/thread | Static smem | Spills |
| --- | ---: | ---: | ---: |
| `m2rnn_tile_summary_kernel` | 94 | 3456 B | 0 |
| `m2rnn_apply_tile_prefix_kernel` | 90 | 3520 B | 0 |
| `m2rnn_scan_tile_summaries_kernel` | 26 | 128 B | 0 |
| `m2rnn_tile_summary_v16_warprow_kernel` | 74 | 3456 B | 0 |
| `m2rnn_apply_tile_prefix_v16_warprow_kernel` | 72 | 3520 B | 0 |
| `m2rnn_local_tile_scan_debug_kernel` | 96 | 3456 B | 0 |

The resource picture did not change from Cycle 4: warprow reduces registers
but increases launched threads/block and remains slower.

### Row-Block Prototype Decision

The proposed one-block-per-`(Be,tile,row)` summary-row prototype was not kept.
It is not a safe small patch for this algorithm because each row update at
token `s+1` depends on the full previous `d[16]` and every row of `M[16,16]`.
Separate CUDA blocks cannot exchange shared tile state or synchronize inside
the token loop.  A correct row-block split would need extra global intermediate
state or one launch per token step, which reintroduces the large prefix storage
or fine-grained launch overhead this branch is trying to avoid.

### Correctness / Tests

Commands:

```bash
python -m py_compile \
  cppmega/megatron/m2rnn_pararnn_tiled_cuda.py \
  tests/test_m2rnn_pararnn_tiled_cuda.py \
  tools/probes/m2rnn_tiled_cuda_stage_profile.py \
  tools/probes/m2rnn_cuda_variant_decision.py

CPPMEGA_VERBOSE_EXT_BUILD=0 pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s

CPPMEGA_VERBOSE_EXT_BUILD=0 CPPMEGA_M2RNN_WARPROW_V16=1 \
  pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s
```

Results:

- `py_compile`: pass
- default pytest: `8 passed, 19 warnings`
- opt-in warprow pytest: `8 passed, 19 warnings`

### Recommendation

Pause CUDA production optimization for this branch.  Keep it as a
resource/diagnostic implementation and do not add more slow CUDA variants
unless a new strategy first clears the `>20%` improvement threshold over
default CUDA on an idle GPU.  Triton remains the active path for target
execution.
