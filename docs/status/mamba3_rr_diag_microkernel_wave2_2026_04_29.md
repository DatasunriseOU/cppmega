# Mamba3 R x R Diagonal Microkernel Wave 2 - 2026-04-29

Branch: `worker/mamba3-rr-diag-microkernel`

Base: wave1 commit `d62b96585e235ca5296ee7688256a8b07c86b02a`.

## Goal

Test a coarser split-kernel diagonal reconstruction path for Mamba3 MIMO
`bwd_bwd`: one Triton program per `(B, H, chunk)` reconstructing all 16
same-time `R x R` blocks in that chunk, instead of wave1's one program per
`(B, H, timestep)`.

## Implemented Variant

Added `stage2_rr_diag_triton_chunk` to
`scripts/modal_mamba3_stage2_force_nontma_benchmark.py`.

Mechanics:

- keep the stage2 `mamba3_bwd_stage2_force_nontma.patch`;
- keep the wave1 `mamba3_bwd_stage2_rr_diag_skip.patch`;
- launch a module-level Triton post-kernel over `B * H * ceil(S / chunk)`;
- each Triton program uses `tl.static_range(0, chunk)` to reconstruct
  `DGAMMA_DIAG`, `DK`, and `DQ` diagonal deltas for all 16 timesteps owned by
  the chunk.

The older wave1 `stage2_rr_diag_triton` one-program-per-timestep split remains
in the benchmark for direct comparison.

## Local Checks

```text
python -m py_compile scripts/modal_mamba3_stage2_force_nontma_benchmark.py
```

Passed.

## H200 Full-Chain Smoke

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_chunk_smoke_20260429_1 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_bf1_bb0,stage2_rr_diag_triton,stage2_rr_diag_triton_chunk \
  --warmup 1 \
  --iters 5
```

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_smoke_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_smoke_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_smoke_20260429_1/summary.json`

Shape: `smoke` (`B=1,S=256,H=4,G=1,N=64,P=64,R=4`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad abs diff vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.07976 | 0.16518 | 0.22595 | reference |
| stage2_bf1_bb0 | 0.07946 | 0.16319 | 0.22756 | 0.0 |
| stage2_rr_diag_triton | 0.08559 | 0.16572 | 0.22977 | 7.276e-12 |
| stage2_rr_diag_triton_chunk | 0.07949 | 0.16479 | 0.22887 | 7.276e-12 |

Read: the chunk kernel is correct and reduces the smoke `bwd_bwd` penalty
relative to the wave1 timestep split, but remains slower than `stage2_bf1_bb0`.

## H200 Full-Chain Representative

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_chunk_representative_20260429_1 \
  --shape-csv representative \
  --variant-csv baseline,stage2_bf1_bb0,stage2_rr_diag_triton,stage2_rr_diag_triton_chunk \
  --warmup 1 \
  --iters 3
```

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_representative_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_representative_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_chunk_representative_20260429_1/summary.json`

Shape: `representative` (`B=2,S=1024,H=16,G=1,N=64,P=64,R=4`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad abs diff vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.27801 | 0.66496 | 0.93479 | reference |
| stage2_bf1_bb0 | 0.28538 | 0.66191 | 0.93606 | 0.0 |
| stage2_rr_diag_triton | 0.27915 | 1.06541 | 1.35898 | 7.276e-12 |
| stage2_rr_diag_triton_chunk | 0.27760 | 1.06689 | 1.35685 | 7.276e-12 |

Read: representative is a clear negative result. Chunk ownership removes
program-level parallelism and lands at the same full-chain regression class as
wave1. Compared with `stage2_bf1_bb0`, the chunk variant is `0.6204x` on
`bwd_bwd` and `0.6899x` on chain.

## Source Markers

The source markers still show the intended stage2 compilation state:

| variant | shape | bwd_fwd WS | bwd_fwd TMA loads | bwd_bwd WS | bwd_bwd TMA loads |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | smoke | false | 0 | false | 0 |
| stage2_bf1_bb0 | smoke | true | 4 | false | 0 |
| stage2_rr_diag_triton | smoke | true | 4 | false | 0 |
| stage2_rr_diag_triton_chunk | smoke | true | 4 | false | 0 |
| baseline | representative | false | 0 | false | 0 |
| stage2_bf1_bb0 | representative | true | 4 | false | 0 |
| stage2_rr_diag_triton | representative | true | 4 | false | 0 |
| stage2_rr_diag_triton_chunk | representative | true | 4 | false | 0 |

## TileLang In-Launch Check

I rechecked the existing TileLang R x R patch:
`mamba3_bwd_stage2_rr_diag.patch` still computes the diagonal products with
`T.serial(P)`, which wave1 already ruled out.

A parallel per-`cs` form does not look viable in current TileLang without
reintroducing most of the lost work:

- padded per-`cs` `T.gemm` would use `16 x P @ P x 16` for each of 16
  timesteps, matching the full `64 x P @ P x 64` tensor-core work while adding
  loop overhead;
- a product-then-`T.reduce_sum` formulation would need a large
  `[chunk, R, R, P]` accumulator footprint inside the already heavy bwd_bwd
  body.

No TileLang candidate was promoted to an H200 run in wave2 because both options
violate the "avoid `T.serial(P)` and reduce work" premise.

## Modal Cleanup

Wave2 Modal apps:

- `ap-KJnl4K1NjMcygrUQjtVzd2` smoke run, stopped, tasks=0.
- `ap-AbQrQ5ZwvMJfFK3E968ps4` representative run, stopped, tasks=0.
- `ap-y4qGZnHG7MQuJ2DgLQBHU7` fresh ephemeral app from cleanup/listing,
  explicitly stopped, tasks=0.

The pre-existing deployed `cppmega-prebuilt` app was not created by this wave
and was left alone.

## Conclusion

Wave2 is negative. The coarser chunk-level Triton split is correct, but it is
not faster. On representative it performs essentially like the wave1 split,
because one program per chunk under-parallelizes the diagonal reconstruction.

Recommendation: pivot Lane A away from split Triton post-kernels and away from
TileLang R x R reconstruction inside the current bwd_bwd body. Continue only if
wave3 changes the ownership boundary more substantially, for example by fusing
diagonal reconstruction into a custom CUDA/CuTe bwd_bwd kernel that keeps enough
CTA parallelism while avoiding an extra post-kernel launch.
