# M2RNN Tiled Triton ParaRNN Prototype - 2026-04-28

Worker: T / Triton

Worktree: `/home/dave/source/cppmega/.claude/worktrees/m2rnn-tiled-triton`

Branch: `worker/m2rnn-tiled-triton`

Base: `0b7acbc5d18dead10ad206ee5c111e2cb08ab1ef`

## Fifth Optimization Cycle After `2ca204b`

### Search Pass

Local context reviewed:

- This document's prior `2ca204b` notes: summary scan is only about 4-7% for
  short one-tile profiles, while local/apply replay and fixed launch overhead
  dominate.
- `cppmega/megatron/m2rnn_pararnn_tiled_triton.py`: Triton path still used
  local summary, summary scan, and apply kernels even when `num_tiles == 1`.
- `scripts/bench_m2rnn_tiled_triton.py`: existing CUDA-event stage profiler and
  fp32/bf16 sweep are sufficient for a narrow measured patch.

Web/docs reviewed:

- Triton language API: <https://triton-lang.org/main/python-api/triton.language.html>
- Triton fused softmax tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Triton persistent matmul tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
- PyTorch persistent grouped GEMM blog:
  <https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/>

Takeaway: the small-S case is launch/intermediate-buffer dominated. For
`num_tiles == 1`, the tile summary and carry scan are mathematically redundant:
the incoming carry is zero, so apply replay can run directly in one kernel.

### Baseline Profiling

Environment:

- GPU: NVIDIA GB10, compute capability 12.1
- PyTorch: `2.13.0.dev20260417+cu132`
- Triton: `3.7.0`

Commands:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 16 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 10 --repeat 100 --dtype fp32 --stage-profile
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 16 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 10 --repeat 100 --dtype bf16 --stage-profile
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 32 --H 4 --K 16 --V 16 --tiles 32,64 --iters 1 --warmup 10 --repeat 100 --dtype fp32 --stage-profile
```

Captured rows before patch:

| dtype | S | tile | num_tiles | full forward | local | scan | apply | scan pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 16 | 16 | 1 | 0.070 ms | 0.005150 ms | 0.000758 ms | 0.004998 ms | 6.95% |
| fp32 | 16 | 32 | 1 | 0.074 ms | 0.006879 ms | 0.000767 ms | 0.006388 ms | 5.47% |
| fp32 | 32 | 32 | 1 | 0.081 ms | 0.011432 ms | 0.000786 ms | 0.010264 ms | 3.50% |
| bf16 | 16 | 16 | 1 | 0.111 ms | 0.005141 ms | 0.000785 ms | 0.004985 ms | 7.19% |
| bf16 | 16 | 32 | 1 | 0.118 ms | 0.006925 ms | 0.000790 ms | 0.006413 ms | 5.59% |

The `S=32,tile=64` stage-profile compile was stopped after roughly a minute;
large `tl.static_range(64)` variants are not useful for this short interactive
cycle.

### Patch

Changed `cppmega/megatron/m2rnn_pararnn_tiled_triton.py`:

- Added `_one_tile_update_kernel`.
- `_tiled_newton_delta` now dispatches to this kernel when the Triton path is
  active and `num_tiles == 1`.
- The fast path starts with zero carry, recomputes the local `M_t,b_t`, applies
  the Newton update, and writes `h_next` in one launch.
- Multi-tile fallback is unchanged.
- Memory accounting now reports zero summary/carry bytes for one-tile solves.

Added `tests/test_m2rnn_pararnn_tiled_triton.py::test_triton_one_tile_fast_path_matches_torch_streaming_cuda`.

### Measured Result

Direct kernel-sequence A/B, old three launches versus new one launch:

| dtype | S | tile | old solve update | new solve update | max diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 16 | 16 | 0.019681 ms | 0.006467 ms | 0 |
| bf16 | 16 | 16 | 0.019540 ms | 0.006457 ms | 0 |

Full forward A/B using an in-script old path:

| dtype | S | tile | iters | old full | new full | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 16 | 16 | 1 | 0.084575 ms | 0.073594 ms | -13.0% |
| fp32 | 16 | 16 | 3 | 0.133507 ms | 0.099195 ms | -25.7% |
| fp32 | 32 | 32 | 3 | 0.132006 ms | 0.099179 ms | -24.9% |
| bf16 | 16 | 16 | 1 | 0.103116 ms | 0.094852 ms | -8.0% |
| bf16 | 16 | 16 | 3 | 0.156046 ms | 0.120632 ms | -22.7% |
| bf16 | 32 | 32 | 3 | 0.154540 ms | 0.120697 ms | -21.9% |

Regular benchmark after patch:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 16 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 10 --repeat 100 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 16 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 10 --repeat 100 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 32 --H 4 --K 16 --V 16 --tiles 32 --iters 1 --warmup 10 --repeat 100 --dtype fp32
```

| dtype | S | tile | latency | peak alloc |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 16 | 16 | 0.075 ms | 32.28 MiB |
| fp32 | 16 | 32 | 0.076 ms | 32.28 MiB |
| fp32 | 32 | 32 | 0.076 ms | 32.48 MiB |
| bf16 | 16 | 16 | 0.097 ms | 32.27 MiB |
| bf16 | 16 | 32 | 0.099 ms | 32.27 MiB |

The regular benchmark still includes input preparation and final output
projection, so the kernel win is partly hidden at one Newton iteration. It
becomes visible in the direct A/B and in `iters=3`.

### Validation

```bash
pytest -q tests/test_m2rnn_pararnn_tiled_triton.py
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype fp32 --B 1 --S 16 --H 1 --K 2 --V 4 --tile 16 --iters 3
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype bf16 --B 1 --S 16 --H 1 --K 2 --V 4 --tile 16 --iters 3
```

Results:

- `7 passed`.
- fp32 probe parity: `max_out=1.788139e-06`, `max_h=1.594424e-06`.
- bf16 probe parity: `max_out=0.000000e+00`, `max_h=0.000000e+00`.

### No-Go Items For This Cycle

- Hierarchical summary scan remains deprioritized: measured scan cost is only
  about 3.5-7.2% for the relevant short profiles.
- Reusable output buffer/prealloc API was not pursued because direct one-tile
  fusion produced a measured solve-update win first; allocator savings are
  small in the current regular benchmark peak numbers.
- Dense ParaRNN versus tiled Triton decision-table integration is still useful,
  but it is a benchmark-product task rather than the smallest measured kernel
  patch for this cycle.

## Continuation After `e4f24be`

### Triton Docs / Examples Reviewed

Primary/near-primary sources checked before changing code:

- Triton `tl.associative_scan` API:
  <https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html>
- Triton language index, scan/reduction APIs:
  <https://triton-lang.org/main/python-api/triton.language.html>
- Triton fused softmax tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Triton persistent matmul tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
- Triton discussion/example using `tl.associative_scan` for roll/shift:
  <https://github.com/triton-lang/triton/discussions/4472>

Concrete takeaways:

1. `tl.associative_scan` supports tuple inputs and custom `@triton.jit`
   combine functions, so an affine summary operator `(M2 @ M1, M2 @ b1 + b2)`
   is expressible inside one program.
2. `tl.associative_scan` scans along an in-block tensor axis; it does not by
   itself create a multi-program/global prefix over arbitrary tile summaries.
   A hierarchical scan would still need an additional inter-block level.
3. The fused softmax tutorial's central lesson applies here: for bandwidth and
   launch-bound chains, avoid writing intermediate tensors that are consumed
   immediately by the next operation.
4. The persistent matmul tutorial demonstrates reducing launch/work scheduling
   overhead by keeping programs resident and iterating over tiles.  That is a
   longer-term direction for local tile/apply; it is larger than today's patch.
5. The current bottleneck is not the existing sequential summary scan at
   `gb10-small`/tile 16; local tile assembly and apply replay dominate the
   kernel split.

### GB10 Baseline / Bottleneck

Environment:

- GPU: NVIDIA GB10, compute capability 12.1
- PyTorch: `2.13.0.dev20260417+cu132`
- Triton: `3.7.0`

First attempted command:

```bash
python scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32,64,128 --iters 1 --warmup 10 --repeat 50 --dtype fp32 --check-dense
```

This was stopped because dense parity at `S=512` spent more than a minute in
the dense reference before producing useful kernel timings.  Parity was checked
separately on smaller shapes.

Useful before benchmark:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32,64,128 --iters 1 --warmup 2 --repeat 5 --dtype fp32
```

The tile 64/128 part was stopped after tile 32 because large unrolled
`tl.static_range(TILE)` variants were spending too long in compile/progress for
this interactive pass.  Before rows captured:

| tile | num_tiles | before latency | before peak alloc |
| ---: | ---: | ---: | ---: |
| 16 | 32 | 0.154 ms | 46.58 MiB |
| 32 | 16 | 0.147 ms | 45.58 MiB |

One-off CUDA event stage split, tile 16, fp32, after JIT warmup:

| stage | before |
| --- | ---: |
| local tile kernel | 0.041788 ms |
| summary scan kernel | 0.007659 ms |
| apply-carry kernel | 0.026049 ms |
| separate PyTorch update | 0.003962 ms |

Interpretation: summary scan is small for this profile. The real easy waste was
an unused `local_delta` write/allocation plus a separate `h + omega * delta`
update after the apply kernel.

### Optimization Patch

Changed `cppmega/megatron/m2rnn_pararnn_tiled_triton.py`:

- Removed unused `local_delta` allocation and stores from the Triton local tile
  kernel and PyTorch fallback.
- Fused Newton update into the apply-carry kernel/fallback:
  `h_next = h + omega_sor * delta`.
- The public bf16 contract is unchanged: bf16 inputs are promoted to fp32 for
  solve accumulation and outputs are cast back to the input dtype.
- `local_delta_bytes` in memory accounting is now zero because the buffer is no
  longer allocated.

After stage split, tile 16, fp32:

| stage | before | after |
| --- | ---: | ---: |
| local tile kernel | 0.041788 ms | 0.038375 ms |
| summary scan kernel | 0.007659 ms | 0.007672 ms |
| apply/update kernel | 0.026049 ms + 0.003962 ms | 0.026059 ms |

After benchmark:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 2 --repeat 20 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 2 --repeat 20 --dtype bf16
```

| dtype | tile | num_tiles | after latency | after peak alloc |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 16 | 32 | 0.109 ms | 40.83 MiB |
| fp32 | 32 | 16 | 0.109 ms | 39.70 MiB |
| bf16 | 16 | 32 | 0.127 ms | 40.76 MiB |
| bf16 | 32 | 16 | 0.121 ms | 39.63 MiB |

Measured against the captured fp32 before rows:

| tile | before | after | delta |
| ---: | ---: | ---: | ---: |
| 16 | 0.154 ms | 0.109 ms | -29.2% |
| 32 | 0.147 ms | 0.109 ms | -25.9% |

Parity after patch:

```bash
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype fp32 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 1
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype bf16 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 2
```

- fp32: `max_out=2.801418e-06`, `max_h=5.736947e-07`
- bf16: `max_out=1.907349e-06`, `max_h=0.000000e+00`

Tests:

```bash
pytest -q tests/test_m2rnn_pararnn_tiled_triton.py
```

Result: `6 passed`.

## What Changed

Added an isolated tiled/streaming Newton linear solve prototype in
`cppmega/megatron/m2rnn_pararnn_tiled_triton.py`.

The solve does not materialize the full Jacobian tensor
`A[B, S, H, K, V, V]`.

Implemented pipeline:

1. Triton local tile pass for CUDA fp32 small-V shapes:
   - one program per `(B * H * K, sequence_tile)`;
   - assembles `M_t = -A_t` and `b_t = -F_t` inside the tile;
   - computes inclusive affine prefix under zero boundary;
   - writes tile summary `(M_tile, b_tile)`.
2. Triton tile-summary scan across sequence tiles on CUDA:
   - one program per `(B * H * K)` chain;
   - sequential scan over `num_tiles` stays on GPU;
   - removes the previous Python/PyTorch scan loop from the Triton path.
3. Triton apply-carry pass:
   - replays the tile;
   - recomputes local `M_t, b_t`;
   - starts from the scanned incoming carry;
   - writes the updated hidden state `h_next = h + omega_sor * delta`.
4. CPU / missing-Triton fallback uses the same streaming algorithm in PyTorch.
5. bf16 input/output path:
   - q/k/v/W/xf may be bf16;
   - Newton solve and Triton kernels run fp32 accumulators;
   - public outputs cast back to the input dtype.

Added `scripts/bench_m2rnn_tiled_triton.py` for tile-size latency/memory sweeps
with `gb10-small` and `h200-small` profiles. This is a probe only; nothing is
wired into training defaults.

The module is not wired into training defaults.

## Validation

Commands run:

```bash
pytest -q tests/test_m2rnn_pararnn_tiled_triton.py
pytest -q tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_triton.py
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cpu --dtype fp64 --B 1 --S 32 --H 2 --K 4 --V 4 --tile 8 --iters 3 --force-torch
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype fp32 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 1
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype bf16 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 2
python scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32,64 --iters 1 --warmup 2 --repeat 5 --dtype fp32
python scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32,64 --iters 1 --warmup 2 --repeat 5 --dtype bf16
```

Results:

- `tests/test_m2rnn_pararnn_tiled_triton.py`: 6 passed.
- `tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_triton.py`: 19 passed.
- CPU probe parity vs dense ParaRNN: `max_out=0.0`, `max_h=0.0`.
- CUDA/Triton probe parity vs dense ParaRNN: `max_out=2.801418e-06`, `max_h=5.736947e-07`.
- CUDA/Triton bf16 probe parity vs dense ParaRNN: `max_out=1.907349e-06`, `max_h=0.0`.

Direct CUDA test coverage now includes Triton summary scan vs PyTorch summary
scan and bf16 input/output parity vs dense ParaRNN.

## Memory Accounting

For NAM56R-like solve shape `B=2, S=4096, H=8, K=64, V=16, tile=64`, fp32:

- Avoided full `A`: `2 * 8 * 64 * 4096 * 16 * 16 * 4 = 4.00 GiB`.
- Peak tile-local `A/M`: `2 * 8 * 64 * 64 * 16 * 16 * 4 = 64.00 MiB`.
- Tile summaries `(M_tile, b_tile)`: `68.00 MiB`.
- Full-A / tile-local ratio: `64x`.

The Triton kernels hold `M_t` and prefix matrices in registers for a single
tile program. The PyTorch fallback materializes only `[Be, tile, V, V]`, never
`[Be, S, V, V]`.

## GB10 Tile Timing

Hardware observed locally: NVIDIA GB10.

Command:

```bash
python scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32,64 --iters 1 --warmup 2 --repeat 5 --dtype fp32
```

Shape: `B=1, S=512, H=4, K=16, V=16`, fp32 compute, one Newton iteration.

| tile | num_tiles | latency | peak alloc | full_A | peak_tile_A | summary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 0.149 ms | 46.58 MiB | 32.00 MiB | 1.00 MiB | 2.12 MiB |
| 32 | 16 | 0.154 ms | 45.58 MiB | 32.00 MiB | 2.00 MiB | 1.06 MiB |
| 64 | 8 | 0.169 ms | 45.08 MiB | 32.00 MiB | 4.00 MiB | 544.00 KiB |

bf16 inputs on the same shape:

| tile | num_tiles | latency | peak alloc | full_A | peak_tile_A | summary |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 0.154 ms | 46.51 MiB | 32.00 MiB | 1.00 MiB | 2.12 MiB |
| 32 | 16 | 0.157 ms | 45.51 MiB | 32.00 MiB | 2.00 MiB | 1.06 MiB |
| 64 | 8 | 0.153 ms | 45.01 MiB | 32.00 MiB | 4.00 MiB | 544.00 KiB |

The timing is a warm-cache microbenchmark after Triton compilation. It is meant
to compare tile sizes, not to claim end-to-end training throughput.

## Current Limits

- Summary scan on CUDA is now Triton, but still sequential over `num_tiles`
  inside one program per `Be` chain. There is no parallel prefix over tile
  summaries yet.
- Apply-carry pass recomputes tile-local affine coefficients instead of storing
  per-token prefix matrices. This is intentional for memory, but doubles local
  tile assembly work.
- Triton path is forward-only prototype for CUDA, `V <= 32`; fp32 and bf16
  inputs are covered, with fp32 solve accumulation.
- No custom autograd or training integration.
- No performance tuning yet: one program per `(Be, tile)`, `num_warps=1`, and
  tile size is static.
- H200 profile is present in the benchmark script but was not run on H200 in
  this worktree.

## Production Next Steps

1. Add a Triton or CUB-style parallel scan for tile summaries.
2. Fuse output projection or write trajectory in production layout to avoid
   extra reshape/einsum overhead.
3. Add custom autograd or derive backward through the tiled solve.
4. Benchmark tile sizes and register pressure on full H200/GB10 production shapes.

## Third Optimization Cycle After `6ee0687`

### Docs / MCP / Web Search

Checked before code changes:

- Triton `tl.associative_scan` API:
  <https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html>
- Triton `tl.static_range` API:
  <https://triton-lang.org/main/python-api/generated/triton.language.static_range.html>
- Triton `tl.range` API and `loop_unroll_factor` knob:
  <https://triton-lang.org/main/python-api/generated/triton.language.range.html>
- Triton persistent matmul tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
- Triton MLIR `tt.scan` op reference:
  <https://triton-lang.org/main/dialects/TritonOps.html#tt-scan-triton-scanop>
- GitHub `tl.associative_scan` multi-input PR:
  <https://github.com/openai/triton/pull/2947>
- GitHub `tl.range` loop unroll PR:
  <https://github.com/triton-lang/triton/pull/4662>

Brave MCP search returned exhausted API credits during this pass, so Exa MCP
and web search were used for the GitHub/docs references above. Local docs/code
search confirmed the current split: local tile and apply dominate `gb10-small`,
while summary scan remains small.

### Baseline Profiling

Commands:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 50 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 50 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 30 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 30 --dtype bf16
```

| shape | dtype | tile 16 | tile 32 |
| --- | --- | ---: | ---: |
| `B=1,S=512,H=4,K=16,V=16` | fp32 | 0.167 ms | 0.120 ms |
| `B=1,S=512,H=4,K=16,V=16` | bf16 | 0.174 ms | 0.119 ms |
| `B=1,S=4096,H=4,K=16,V=16` | fp32 | 1.232 ms | 1.152 ms |
| `B=1,S=4096,H=4,K=16,V=16` | bf16 | 1.240 ms | 1.186 ms |

### Recompute Experiments

Two approaches were tested and intentionally not shipped:

1. Cache per-token vector coefficients `b` and `alpha=(1-f)*sech2` as two
   `[Be,S,V]` fp32 tensors. This keeps the memory contract well below the
   forbidden `[Be,S,V,V]` local prefix/full-A form, but GB10 runtime got worse:
   small tile32 fp32 rose to 0.308 ms and `S=4096` tile32 fp32 rose to
   3.314 ms because the extra global stores/loads dominated.
2. Recompute reduction in apply as `f*d + alpha*(Wt@d) + b` without materialized
   `M`. This was also slower in the measured kernels, so it was reverted.

### Shipped Patch

The only code change shipped in this cycle is conservative: the
`TiledTritonConfig` default `tile_size` is now 32 instead of 64. This avoids the
large default `tl.static_range(64)` specialization for callers that do not pass
an explicit tile. Explicit tile sweeps remain available.

Large `tile64` first-use checks on the `S=4096` shape spent over a minute in
compile/progress before producing timings. The explicit tile16/32 runs complete
quickly, and tile32 was consistently the better explicit choice on the larger
shape in this pass.

### After Profiling

Same commands as baseline, after reverting the losing recompute experiments and
keeping only the default tile change:

| shape | dtype | tile 16 | tile 32 |
| --- | --- | ---: | ---: |
| `B=1,S=512,H=4,K=16,V=16` | fp32 | 0.104 ms | 0.105 ms |
| `B=1,S=512,H=4,K=16,V=16` | bf16 | 0.117 ms | 0.118 ms |
| `B=1,S=4096,H=4,K=16,V=16` | fp32 | 1.190 ms | 1.125 ms |
| `B=1,S=4096,H=4,K=16,V=16` | bf16 | 1.243 ms | 1.154 ms |

The after rows are warm-cache microbenchmarks; the main functional change is
the default tile selection, not an explicit-tile kernel rewrite.

### Parity / Tests

```bash
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_triton.py scripts/bench_m2rnn_tiled_triton.py tools/probes/m2rnn_pararnn_tiled_triton_probe.py
pytest -q tests/test_m2rnn_pararnn_tiled_triton.py
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype fp32 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 1
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype bf16 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 2
```

Results:

- `py_compile`: passed.
- `tests/test_m2rnn_pararnn_tiled_triton.py`: 6 passed.
- CUDA fp32 probe: `max_out=2.801418e-06`, `max_h=5.736947e-07`.
- CUDA bf16 probe: `max_out=1.907349e-06`, `max_h=0.000000e+00`.

### Remaining Blockers

- The best way to remove double recompute is not obvious on GB10: vector cache
  and direct apply algebra both lost. A fused local-summary/apply path may still
  make sense for `num_tiles == 1`, but it does not address `S=512/4096`.
- Hierarchical/global summary scan remains unimplemented. For the measured
  GB10 shapes, tile32 still spends more time in local/apply than in the scan, so
  this was not the highest-return patch for this cycle.
- `tl.static_range(TILE)` still hurts compile for 64/128. The next safe step is
  a separate kernel variant using `tl.range(..., loop_unroll_factor=...)` for
  large tiles, with explicit correctness/perf gating.

## Fourth Optimization Cycle After `0494da7`

### Docs / MCP / Web / Local Search

Local search covered the tiled Triton implementation, benchmark, tests, and this
status doc (`m2rnn_pararnn_tiled_triton.py`,
`scripts/bench_m2rnn_tiled_triton.py`, and
`tests/test_m2rnn_pararnn_tiled_triton.py`).
Brave MCP search was attempted for Triton loop/autotune docs, but the provider
returned exhausted credits, so the external references below use direct web
search/opened official Triton docs.

External sources checked before changing code:

- Triton `tl.range` API:
  <https://triton-lang.org/main/python-api/generated/triton.language.range.html>
  - relevant because `loop_unroll_factor`, `num_stages`, `flatten`, and LICM
    controls are only available on `tl.range`.
- Triton `tl.static_range` API:
  <https://triton-lang.org/main/python-api/generated/triton.language.static_range.html>
  - documents that it guides the compiler to aggressively unroll loops.
- Triton `autotune` API:
  <https://triton-lang.org/main/python-api/generated/triton.autotune.html>
  - important caveat: autotune evaluates configs by running kernels multiple
    times, which is awkward for update/apply kernels unless reset/restore hooks
    are carefully used.
- Triton `heuristics` API:
  <https://triton-lang.org/main/python-api/generated/triton.heuristics.html>
  - useful for meta-parameter selection when autotune is too expensive or not
    applicable.
- Triton fused softmax tutorial:
  <https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
  - used for the occupancy/num_warps tuning reminder and the same "avoid
    unnecessary DRAM traffic" principle already applied in this path.

### Baseline / Profiling

Environment:

- GPU: NVIDIA GB10, compute capability 12.1
- PyTorch: `2.13.0.dev20260417+cu132`
- Triton: `3.7.0`

Full forward baseline commands:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 60 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 60 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 40 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 40 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 8192 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 25 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 8192 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 25 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 8 --K 32 --V 16 --tiles 16,32 --iters 1 --warmup 3 --repeat 20 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 8 --K 32 --V 16 --tiles 16,32 --iters 1 --warmup 3 --repeat 20 --dtype bf16
```

| shape | dtype | tile16 | tile32 |
| --- | --- | ---: | ---: |
| `B=1,S=512,H=4,K=16,V=16` | fp32 | 0.105 ms | 0.105 ms |
| `B=1,S=512,H=4,K=16,V=16` | bf16 | 0.117 ms | 0.117 ms |
| `B=1,S=4096,H=4,K=16,V=16` | fp32 | 1.205 ms | 1.132 ms |
| `B=1,S=4096,H=4,K=16,V=16` | bf16 | 1.257 ms | 1.177 ms |
| `B=1,S=8192,H=4,K=16,V=16` | fp32 | 6.004 ms | 5.473 ms |
| `B=1,S=8192,H=4,K=16,V=16` | bf16 | 6.035 ms | 5.771 ms |
| `B=1,S=4096,H=8,K=32,V=16` | fp32 | 5.048 ms | 4.913 ms |
| `B=1,S=4096,H=8,K=32,V=16` | bf16 | 5.113 ms | 4.946 ms |

Interpretation: tile32 remains the robust default among tile16/32. Tile64 can
win slightly after compilation on `S=512/4096`, but it is not robust enough to
make the default: `S=8192,tile64` spent more than a minute without returning a
timing and was killed.

### Hierarchical Scan Check

Added a benchmark-script stage profiler so the local/scan/apply split can be
reproduced without ad-hoc snippets:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 32 --iters 1 --warmup 10 --repeat 80 --dtype fp32 --stage-profile
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 8192 --H 4 --K 16 --V 16 --tiles 32 --iters 1 --warmup 10 --repeat 50 --dtype fp32 --stage-profile
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 8 --K 32 --V 16 --tiles 32 --iters 1 --warmup 8 --repeat 40 --dtype fp32 --stage-profile
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 8192 --H 8 --K 32 --V 16 --tiles 32 --iters 1 --warmup 5 --repeat 25 --dtype fp32 --stage-profile
```

| shape | local | scan | apply | stage sum | scan share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S=4096,H=4,K=16,Be=64` | 0.376894 ms | 0.034900 ms | 0.346736 ms | 0.758530 ms | 4.60% |
| `S=8192,H=4,K=16,Be=64` | 0.759608 ms | 0.064470 ms | 0.690934 ms | 1.515012 ms | 4.26% |
| `S=4096,H=8,K=32,Be=256` | 1.241892 ms | 0.168243 ms | 1.206270 ms | 2.616406 ms | 6.43% |
| `S=8192,H=8,K=32,Be=256` | 5.431620 ms | 0.698976 ms | 5.222426 ms | 11.353021 ms | 6.16% |

Conclusion: sequential per-chain summary scan is not the primary bottleneck on
GB10 for these shapes. A perfect hierarchical scan would only recover the scan
share above, while local tile assembly and apply replay dominate. No complex
hierarchical scan patch was made in this cycle.

### Tuning Attempts Not Shipped

- Replacing local/apply `tl.static_range(0, TILE)` with
  `tl.range(0, TILE, loop_unroll_factor=1)` reduced unroll/compile pressure but
  made `S=512,H=4,K=16,V=16` tile32 fp32 slower: `0.105 ms -> 0.241 ms`.
- `loop_unroll_factor=4` also lost: tile32 fp32 `0.146 ms`.
- `loop_unroll_factor=8` was mixed: tile16 fp32 `0.117 ms`, but tile32 fp32
  `0.250 ms`.
- Summary scan `num_warps=2` was not robust. Same-process A/B showed small wins
  on some rows but a loss on `S=4096,H=8,K=32`, so the kernel remains
  `num_warps=1`.

### Shipped Patch

Changed `scripts/bench_m2rnn_tiled_triton.py` only:

- Added `--stage-profile`.
- It reports per-tile `local_ms`, `scan_ms`, `apply_ms`, summed kernel-stage
  time, and scan percentage.
- This is intentionally a measurement patch, not a production-kernel change:
  profiling showed hierarchical scan is not high-return enough yet, and the
  tested runtime/compile tweaks were not robust.

### After / Validation

After full-forward commands:

```bash
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 60 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --profile gb10-small --tiles 16,32 --iters 1 --warmup 5 --repeat 60 --dtype bf16
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 40 --dtype fp32
python -u scripts/bench_m2rnn_tiled_triton.py --B 1 --S 4096 --H 4 --K 16 --V 16 --tiles 16,32 --iters 1 --warmup 5 --repeat 40 --dtype bf16
```

| shape | dtype | tile16 | tile32 |
| --- | --- | ---: | ---: |
| `B=1,S=512,H=4,K=16,V=16` | fp32 | 0.103 ms | 0.104 ms |
| `B=1,S=512,H=4,K=16,V=16` | bf16 | 0.117 ms | 0.117 ms |
| `B=1,S=4096,H=4,K=16,V=16` | fp32 | 1.251 ms | 1.154 ms |
| `B=1,S=4096,H=4,K=16,V=16` | bf16 | 1.258 ms | 1.152 ms |

Correctness / tests:

```bash
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_triton.py scripts/bench_m2rnn_tiled_triton.py tools/probes/m2rnn_pararnn_tiled_triton_probe.py
pytest -q tests/test_m2rnn_pararnn_tiled_triton.py
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype fp32 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 1
python tools/probes/m2rnn_pararnn_tiled_triton_probe.py --device cuda --dtype bf16 --B 1 --S 32 --H 1 --K 2 --V 4 --tile 8 --iters 2
```

Results:

- `py_compile`: passed.
- `tests/test_m2rnn_pararnn_tiled_triton.py`: 6 passed.
- CUDA fp32 probe: `max_out=2.801418e-06`, `max_h=5.736947e-07`.
- CUDA bf16 probe: `max_out=1.907349e-06`, `max_h=0.000000e+00`.

### Current Blockers

- Hierarchical scan is measurable but not dominant on GB10; local/apply replay
  remains the larger target.
- `tl.range` unroll controls are not a drop-in replacement for local/apply
  `static_range` on tile32; runtime losses were too large.
- Tile64 runtime can be good after compilation, but first-use/progress behavior
  at `S=8192` is bad enough that tile32 remains the safer default.
