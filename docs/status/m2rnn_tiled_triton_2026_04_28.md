# M2RNN Tiled Triton ParaRNN Prototype - 2026-04-28

Worker: T / Triton

Worktree: `/home/dave/source/cppmega/.claude/worktrees/m2rnn-tiled-triton`

Branch: `worker/m2rnn-tiled-triton`

Base: `0b7acbc5d18dead10ad206ee5c111e2cb08ab1ef`

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
   - writes local deltas and tile summary `(M_tile, b_tile)`.
2. Triton tile-summary scan across sequence tiles on CUDA:
   - one program per `(B * H * K)` chain;
   - sequential scan over `num_tiles` stays on GPU;
   - removes the previous Python/PyTorch scan loop from the Triton path.
3. Triton apply-carry pass:
   - replays the tile;
   - recomputes local `M_t, b_t`;
   - starts from the scanned incoming carry;
   - writes full Newton `delta`.
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
