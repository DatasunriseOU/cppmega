# Mamba3 R x R Diagonal CUDA Integration Wave 7 - 2026-04-30

Branch: `worker/mamba3-rr-diag-cuda-integrate`

## Goal

Expand the wave6 chunk-warp owner CUDA slice toward a fuller custom
`bwd_bwd` rewrite by adding a major production consumer without returning to
the dead split/post-kernel integration path.

Prior wave6 H200 productionish signals:

| path | mean ms |
| --- | ---: |
| full TileLang `stage2_bf1_bb0` `bwd_bwd` | 3.70674 |
| wave6 chunk-warp diagonal slice | 1.77566 |
| wave5 timestep post diagonal slice | 3.16204 |

## Inspected

- `rr_diag_wave6_inlaunch_cuda.py` and `rr_diag_cuda_kernel.cu`: wave6
  chunk-owner and chunk-warp-owner diagonal kernels.
- `mamba3_bwd_bwd_rr_diag_tilelang.patch`: diagonal-only TileLang algebra and
  the three existing consumers of `dqk_from_diag`: `DGAMMA_DIAG`, `DK`, `DQ`.
- `mamba3_bwd_stage2_force_nontma.patch` / `mamba3_bwd_layout_fix.patch`:
  production stage2 `bwd_bwd` path around the next consumer:
  `qk_dot -> dPsiV_D_fused -> dV/dPsi`.
- Wave6 generated-source notes: productionish `stage2_bf1_bb0`
  `bwd_bwd_kernel_source.cu` was 83,354 chars, sha256
  `63da45df...79618bab`, launch bounds `(256, 1)`, no TMA loads/stores in
  `bwd_bwd`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave7_chunk_owner_cuda.py`
- `scripts/modal_mamba3_rr_diag_wave7_chunk_owner_cuda.py`

New CUDA entry points:

1. `stage2_qk_dv_chunk_warp_owner_kernel`
   - one CTA owns one `(B, H, chunk)` tile;
   - one warp owns one timestep at a time;
   - computes the production same-time `qk_dot` contribution into `dPsiV`,
     then the direct `DV` contribution;
   - writes `DV` uniquely by timestep, so no global partials or reductions.

2. `stage2_rr_diag_qk_dv_chunk_warp_owner_kernel`
   - same ownership model;
   - computes wave6 `DGAMMA_DIAG`, `DK`, `DQ` diagonal outputs plus the new
     `qk_dot -> dPsiV -> DV` consumer in one CUDA launch.

`DMIMO_V` is intentionally not included yet. With `(B, H, chunk)` ownership it
would require cross-chunk accumulation into `[B, H, R, P]`, which is exactly
the global-partial shape this wave is avoiding.

## Local Checks

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave7_chunk_owner_cuda.py \
  scripts/modal_mamba3_rr_diag_wave7_chunk_owner_cuda.py

git diff --check

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave7_chunk_owner_cuda.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

All passed.

## H200 Run

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_rr_diag_wave7_chunk_owner_cuda.py \
  --shape-csv smoke,productionish \
  --warmup 3 \
  --iters 10
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

Diagonal outputs compare against the existing wave5 production-layout CUDA
timestep post reference. New `DV` compares against independent torch algebra
for the production `qk_dot -> dPsiV -> dV` consumer.

| shape | check | max abs diff |
| --- | --- | ---: |
| smoke | combined `DGAMMA_DIAG` vs wave5 CUDA | 1.776e-15 |
| smoke | combined `DK` vs wave5 CUDA | 2.842e-14 |
| smoke | combined `DQ` vs wave5 CUDA | 5.684e-14 |
| smoke | combined `DV` vs torch qk/dV reference | 2.274e-13 |
| productionish | combined `DGAMMA_DIAG` vs wave5 CUDA | 7.105e-15 |
| productionish | combined `DK` vs wave5 CUDA | 9.095e-13 |
| productionish | combined `DQ` vs wave5 CUDA | 4.547e-13 |
| productionish | combined `DV` vs torch qk/dV reference | 1.455e-11 |

The isolated qk/dV component produced the same `DV` diffs as the combined
kernel on both shapes.

## Performance

| shape | component | mean ms | notes |
| --- | --- | ---: | --- |
| smoke | wave6 chunk-warp diag | 0.02423 | noisy underfilled shape |
| smoke | wave7 qk/dV only | 0.01260 | isolated component |
| smoke | wave7 diag + qk/dV total | 0.02392 | one launch; noise hides increment |
| productionish | wave6 chunk-warp diag | 1.76130 | refreshed in wave7 harness |
| productionish | wave7 qk/dV only | 0.35417 | 20.1% of diag-only time |
| productionish | wave7 diag + qk/dV total | 1.91459 | one-launch prototype total |

Productionish combined read:

- incremental cost over refreshed wave6 diag: `+0.15329 ms`;
- isolated component sum: `2.11548 ms`; combined is `1.105x` lower than the
  component sum because the qk/dV path reuses the same per-timestep ownership
  and launch envelope;
- combined total is `51.65%` of the wave6 full TileLang
  `stage2_bf1_bb0 bwd_bwd` time (`1.91459 / 3.70674`);
- combined total remains `1.65x` faster than the old wave5 timestep post
  diagonal slice alone (`3.16204 / 1.91459`).

## Resource Metadata

| kernel | regs/thread | local bytes | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| wave6 chunk-warp diag | 88 | 64 | 5 | 31.25% |
| wave7 qk/dV only | 48 | 0 | 10 | 62.5% |
| wave7 diag + qk/dV | 80 | 64 | 6 | 37.5% |

The combined kernel compiled with fewer registers than the wave6 diag-only
kernel and slightly higher theoretical occupancy. That is a useful signal, but
not a guarantee that the remaining production paths will fit without staging.

## Read

The warp-owner path still survives.

Adding the `qk_dot -> dPsiV -> DV` production consumer did not break the
economics: productionish total moved from `1.76130 ms` for the refreshed
diag-only slice to `1.91459 ms` for diag plus qk/dV in the same launch. This is
still far below the full TileLang `bwd_bwd` time, but the prototype now covers
one more real consumer and one more real output path.

The next hard boundary is no longer same-time diagonal math. It is the
off-time intra-chunk/state work plus outputs that need cross-chunk reductions
(`DMIMO_V`, possibly angle/state scalars). Those require either:

1. a fuller chunk CTA that carries the local triangular/off-diagonal work with
   careful register staging; or
2. an explicit reduction design for the few outputs whose natural owner is not
   `(B, H, chunk)`.

Do not revive split/post integration. Continue with a custom CUDA/CuTe
`bwd_bwd` chunk skeleton around the chunk-warp owner.
