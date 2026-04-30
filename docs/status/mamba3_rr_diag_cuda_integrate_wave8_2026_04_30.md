# Mamba3 R x R Diagonal CUDA Integration Wave 8 - 2026-04-30

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Add the next missing production consumer to the wave7 one-launch CUDA owner
while preserving the chunk-warp model and avoiding per-chunk global partials
for `DMIMO_V`.

Wave7 productionish baseline:

| path | mean ms |
| --- | ---: |
| TileLang `stage2_bf1_bb0` `bwd_bwd` | 3.70674 |
| wave7 diag-only refreshed in harness | 1.76130 |
| wave7 qk/dV only | 0.35417 |
| wave7 diag + qk/dV one launch | 1.91459 |

## Inspected

- `rr_diag_wave7_chunk_owner_cuda.py` and `rr_diag_cuda_kernel.cu`: current
  one-warp-per-timestep chunk owner for `DGAMMA_DIAG`, diagonal `DK/DQ`, and
  `qk_dot -> dPsiV -> DV`.
- `mamba3_bwd_stage2_force_nontma.patch`: remaining TileLang `bwd_bwd`
  consumers after `dPsiV_combined_shared`, especially `DV`, `DMIMO_V`,
  `DDA_CS_REV`, `DFACTOR`, `DDA`, `DDA_CS`, `DSSDA`, `DANGLES`, and
  non-diagonal `DK/DQ`.
- `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py`: algebra map for
  full `bwd_bwd`, including `DPsi_acc += dPsiV_combined * V` for `DMIMO_V`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py`
- `scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py`
- `docs/status/mamba3_rr_diag_cuda_integrate_wave8_2026_04_30.md`

New CUDA entry points:

1. `stage2_qk_dmimo_v_sequence_owner_kernel`
   - one CTA owns one final `DMIMO_V[b, h, r, :]` row;
   - loops over all timesteps and reduces directly into `[B, H, R, P]`;
   - writes unique output rows, so there are no atomics and no global partial
     tensor.

2. `stage2_rr_diag_qk_dv_dmimo_v_owner_kernel`
   - one CUDA launch;
   - first `B*H*nchunks` CTAs run the existing wave7 chunk-warp owner body;
   - trailing `B*H*R` CTAs run the sequence-owner `DMIMO_V` reduction.

This covers the same-time `qk_dot -> dPsiV -> DMIMO_V` contribution. It does
not yet cover `DMIMO_V` contributions from state passing, LKQ, or optional `D`.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py \
  scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_chunk_owner_cuda.py \
  --shape smoke --device cpu --iters 1 --warmup 0

git diff --check
```

All passed.

## H200 Run

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave8_chunk_owner_cuda.py \
  --shape-csv smoke,productionish \
  --warmup 3 \
  --iters 10
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

The Modal app completed and stopped normally.

## Correctness

Diagonal outputs compare against the existing wave5 production-layout CUDA
timestep post reference. `DV` compares against the existing wave7 torch
reference. New `DMIMO_V` compares against independent torch algebra for the
same-time `qk_dot -> dPsiV -> DMIMO_V` contribution.

| shape | check | max abs diff |
| --- | --- | ---: |
| smoke | wave8 combined `DGAMMA_DIAG` vs wave5 CUDA | 1.776e-15 |
| smoke | wave8 combined `DK` vs wave5 CUDA | 2.842e-14 |
| smoke | wave8 combined `DQ` vs wave5 CUDA | 5.684e-14 |
| smoke | wave8 combined `DV` vs torch qk/dV reference | 2.274e-13 |
| smoke | wave8 combined `DMIMO_V` vs torch qk reference | 5.551e-15 |
| productionish | wave8 combined `DGAMMA_DIAG` vs wave5 CUDA | 7.105e-15 |
| productionish | wave8 combined `DK` vs wave5 CUDA | 9.095e-13 |
| productionish | wave8 combined `DQ` vs wave5 CUDA | 4.547e-13 |
| productionish | wave8 combined `DV` vs torch qk/dV reference | 1.455e-11 |
| productionish | wave8 combined `DMIMO_V` vs torch qk reference | 8.882e-14 |

The isolated qk/`DMIMO_V` component produced the same `DMIMO_V` diffs as the
combined kernel on both shapes.

## Performance

| shape | component | mean ms | notes |
| --- | --- | ---: | --- |
| smoke | wave6 chunk-warp diag | 0.02499 | noisy underfilled shape |
| smoke | wave7 qk/dV only | 0.01262 | isolated component |
| smoke | wave7 diag + qk/dV total | 0.02522 | refreshed in wave8 harness |
| smoke | wave8 qk/DMIMO_V only | 0.08418 | sequence-owner reduction |
| smoke | wave8 diag + qk/dV + qk/DMIMO_V total | 0.04500 | one launch |
| productionish | wave6 chunk-warp diag | 1.77041 | refreshed in wave8 harness |
| productionish | wave7 qk/dV only | 0.35446 | isolated component |
| productionish | wave7 diag + qk/dV total | 1.92669 | refreshed in wave8 harness |
| productionish | wave8 qk/DMIMO_V only | 2.52371 | no atomics, full sequence reduction |
| productionish | wave8 diag + qk/dV + qk/DMIMO_V total | 3.00059 | one launch |

Productionish combined read:

- incremental cost over refreshed wave7 combined: `+1.07389 ms`;
- incremental cost over refreshed diag-only: `+1.23018 ms`;
- isolated component sum: `4.64858 ms`; combined is `1.55x` lower than the
  component sum because the sequence-owner CTAs overlap with the chunk-owner
  CTAs in the same launch envelope;
- combined total is `80.95%` of the wave6 full TileLang
  `stage2_bf1_bb0 bwd_bwd` time (`3.00059 / 3.70674`);
- combined total remains below the old wave5 timestep-post diagonal slice
  alone (`3.16204 ms`).

## Resource Metadata

| kernel | regs/thread | local bytes | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| wave6 chunk-warp diag | 88 | 64 | 5 | 31.25% |
| wave7 qk/dV only | 48 | 0 | 10 | 62.5% |
| wave7 diag + qk/dV | 80 | 64 | 6 | 37.5% |
| wave8 qk/DMIMO_V only | 48 | 0 | 10 | 62.5% |
| wave8 diag + qk/dV + qk/DMIMO_V | 80 | 64 | 6 | 37.5% |

The wave8 combined kernel kept the wave7 register/occupancy profile because
the new sequence-owner branch is lighter than the chunk-warp body.

## Read

The path still survives, but the margin is now much thinner.

`DMIMO_V` can be owned without atomics by reducing one final `[B,H,R,P]` row per
CTA. That avoids a global partial disaster and gives exact qk-path correctness,
but it serializes over `S` inside each owner and costs real time. Productionish
total moved from refreshed wave7 `1.92669 ms` to wave8 `3.00059 ms`, still under
TileLang `3.70674 ms`.

The next frontier should not blindly add more sequence reductions. The remaining
large pieces are the state/LKQ intra-chunk work and non-diagonal `DK/DQ`; those
need a fuller chunk-local matrix skeleton or a deliberate reduction plan that
does not strand long sequence loops in a small set of owner CTAs.
