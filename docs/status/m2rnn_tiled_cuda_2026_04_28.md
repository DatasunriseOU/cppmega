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
