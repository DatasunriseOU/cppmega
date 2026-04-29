# Mamba3 R x R Diagonal Microkernel Wave 1 - 2026-04-29

Branch: `worker/mamba3-rr-diag-microkernel`

Base note: this wave starts from the prior serial-P TileLang inline R x R patch
state, where full-kernel smoke was correct but regressed `bwd_bwd` from
`0.1635 ms` to `0.5519 ms`.

## Goal

Find a faster full `mamba_mimo_bwd_fwd + mamba_mimo_bwd_bwd` variant for the
R x R diagonal path without a serial `P` loop in the TileLang bwd_bwd body.

## Implemented Variant

I added `stage2_rr_diag_triton` to
`scripts/modal_mamba3_stage2_force_nontma_benchmark.py`.

Mechanics:

- apply existing `mamba3_bwd_stage2_force_nontma.patch`;
- apply new `mamba3_bwd_stage2_rr_diag_skip.patch`;
- the TileLang bwd_bwd variant skips the full
  `[chunk_size * R, chunk_size * R]` `dqk_from_diag` GEMM and omits its DK/DQ
  consumers;
- a module-level Triton post-kernel reconstructs only the same-time R x R
  diagonal terms for `DGAMMA_DIAG`, `DK`, and `DQ`.

This was the quickest full-kernel test of the "separate Triton microkernel"
route.  It is not production-ready: the Triton kernel is intentionally simple
and uses one program per `(B, H, S)` timestep.

## Local Checks

```text
python -m py_compile scripts/modal_mamba3_stage2_force_nontma_benchmark.py
patch -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_force_nontma.patch
patch -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_stage2_rr_diag_skip.patch
```

Both checks passed.

## H200 Full-Kernel Smoke

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_triton_smoke_cached_20260429_1 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_force_nontma,stage2_rr_diag_triton \
  --warmup 1 \
  --iters 5
```

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_smoke_cached_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_smoke_cached_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_smoke_cached_20260429_1/summary.json`

Modal app:

- `ap-1CEjmHvwrpFXJU0POXbiwC`, stopped, tasks=0.

Shape: `smoke` (`B=1,S=256,H=4,G=1,N=64,P=64,R=4`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad abs diff vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.08125 | 0.16433 | 0.22872 | reference |
| stage2_force_nontma | 0.08002 | 0.16318 | 0.22653 | 0.0 |
| stage2_rr_diag_triton | 0.08274 | 0.16890 | 0.23285 | 7.276e-12 |

Read: correctness is good, but the Triton split is slower than stage2 on smoke.
Compared with stage2, `bwd_bwd` is `0.97298x` and chain is `0.98224x`.

Earlier in the same wave, I accidentally measured a version that re-created the
Triton JIT function inside each timed call.  That produced a false
`bwd_bwd=11.27 ms` smoke result.  The result above is the corrected module-level
JIT measurement.

## H200 Representative

The user cutoff arrived while this run was active; I let it finish and launched
no further experiments.

Command:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run --timestamps scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_rr_diag_triton_representative_20260429_1 \
  --shape-csv representative \
  --variant-csv baseline,stage2_force_nontma,stage2_rr_diag_triton \
  --warmup 1 \
  --iters 3
```

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_representative_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_representative_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_rr_diag_triton_representative_20260429_1/summary.json`

Modal app:

- `ap-91pR7EhbpX5LM1J4ig08EQ`, stopped, tasks=0.

Shape: `representative` (`B=2,S=1024,H=16,G=1,N=64,P=64,R=4`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | max main grad abs diff vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.27961 | 0.66103 | 0.93230 | reference |
| stage2_force_nontma | 0.27762 | 0.65991 | 0.92418 | 0.0 |
| stage2_rr_diag_triton | 0.27503 | 1.06177 | 1.35033 | 7.276e-12 |

Read: correctness remains good, but the split is a clear full-kernel regression.
Compared with stage2, `bwd_bwd` is `0.6217x` and chain is `0.6844x`.

## Productionish

Not run.  Smoke did not meet the "faster/equal" gate, representative confirmed
the regression, and the user cutoff stopped new launches.

## Profiler / Search Notes

No torch profiler run was launched before cutoff.  Source markers from the
benchmark still show the intended kernel states:

- stage2 bwd_fwd uses WS/TMA (`bwd_fwd_ws=true`, `bwd_fwd_tma_loads=4`);
- stage2 and `stage2_rr_diag_triton` bwd_bwd stay non-WS/non-TMA
  (`bwd_bwd_ws=false`, `bwd_bwd_tma_loads=0`).

The split-kernel path loses because the post-kernel launch and per-timestep
program granularity cost more than the removed full diagonal GEMM inside the
already-fused TileLang bwd_bwd.  The slowdown grows on representative despite
good isolated Triton companion results, so isolated subgraph speed does not
transfer to the full chain in this form.

## Modal Cleanup

`modal app list` after the cutoff showed all wave-owned `cppmega-mamba3-*` apps
as `stopped` with `Tasks=0`.  The pre-existing deployed `cppmega-prebuilt` app
was not created by this wave and was left alone.

## Conclusion

Wave 1 result is negative.  The separate Triton post-kernel is correct but not
faster for the full bwd_bwd chain.

Next wave recommendation:

1. Do not continue the one-program-per-timestep Triton post-kernel as a
   production candidate.
2. If staying in Triton/CUDA, fuse diagonal reconstruction with the TileLang
   bwd_bwd work boundary more tightly: one CTA per chunk/head with multiple
   timesteps per program, or a custom CUDA kernel that consumes chunk-local
   `dPhiO`, `PsiV`, pre-rot Q/K and writes DK/DQ without a separate tiny launch
   per timestep shape.
3. Revisit TileLang per-`cs` R x R through `T.gemm` or another parallel
   reduction form, but only if it avoids the serial-P loop and does not add a
   separate launch.
