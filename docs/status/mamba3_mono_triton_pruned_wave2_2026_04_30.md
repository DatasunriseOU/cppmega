# Mamba3 Monolithic Triton Pruned Wave2 - 2026-04-30

Status: evidence / negative result
Canonical: no
Branch: `worker/mamba3-mono-triton-model`

## Scope

Wave1 showed that monolithic reuse alone is not enough:

- full-mask monolithic Triton checksum lower bound: `4.53881 ms`
- full-mask monolithic FMA: `114.63B`
- ideal reuse plus full triangular causal pruning estimate: `96.38B`
- TileLang full `stage2_bf1_bb0` `bwd_bwd`: `3.70674 ms`

Wave2 asked whether a triangular/causal-pruned Triton owner body can make that
estimate real enough to pull the lower bound below TileLang.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave2_mono_triton_pruned_model.py`
- `scripts/modal_mamba3_mono_triton_pruned_wave2.py`

The Wave2 Triton checksum kernel still owns one full `(B,H,chunk)` body and
stores one checksum per owner.  It changes the masked products from Wave1:

1. Split the `64x64` fused chunk matrix into `16x16` FCS tiles, equal to four
   chunk timesteps by four chunk timesteps.
2. Compute all `LKQ` and `dk_intra` tiles because `DSSDA` needs the full
   unmasked products.
3. Apply causal consumers only for tiles on/above the causal frontier:
   `LKQ -> dPsiV`, `dk_intra -> DK`, and `dk_intra.T -> DQ`.
4. Skip below-frontier apply tiles entirely.
5. Keep diagonal `16x16` tiles internally masked for correctness.  This means
   the measured tile-pruned FMA is higher than a fully split `4x4` triangular
   implementation, but it is a real tile-level pruning model rather than the
   Wave1 full-mask model.

I also added a batched torch checksum reference for full productionish
correctness.  The materialized production torch reference was stopped by Modal
before returning an exception; the batched reference matches the materialized
reference exactly on smoke (`max_abs_delta=0.0`) and avoids large temporary
materialization.

## H200 Runs

Smoke correctness/timing:

```text
modal run --timestamps scripts/modal_mamba3_mono_triton_pruned_wave2.py \
  --shape-csv smoke --num-warps-csv 4 \
  --iters 3 --warmup 1 --check-torch-shapes smoke
```

Productionish timing sweep:

```text
modal run --timestamps scripts/modal_mamba3_mono_triton_pruned_wave2.py \
  --shape-csv productionish --num-warps-csv 4,8 \
  --iters 5 --warmup 2 --check-torch-shapes ''
```

Productionish correctness with batched torch reference:

```text
modal run --timestamps scripts/modal_mamba3_mono_triton_pruned_wave2.py \
  --shape-csv productionish --num-warps-csv 4 \
  --iters 3 --warmup 1 --check-torch-shapes productionish \
  --torch-reference-batch 512
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

Checksum comparison against the Wave1 torch algebra:

| shape | torch reference | max abs delta | mean abs delta | max ref abs |
| --- | --- | ---: | ---: | ---: |
| smoke | materialized | `6.083846e-04` | `1.390079e-04` | `2.242615` |
| productionish | batched, `512` chunk owners/batch | `7.650852e-04` | `1.364793e-04` | `3.575903` |

This remains checksum correctness only.  The kernel is a lower-bound algebra
probe and does not write full `DV`, `DK`, `DQ`, `DMIMO_V`, or scalar outputs.

## FMA Model

Productionish shape: `B=4, S=4096, H=32, N=64, P=128, R=4, chunk=16`.

| model | FMA | read |
| --- | ---: | --- |
| separate recompute | `125.37B` | previous slice-style model |
| Wave1 monolithic full-mask | `114.63B` | full `64x64` masked applies |
| Wave2 measured tile-pruned | `101.75B` | `16x16` tiles, diagonal tiles internally masked |
| ideal fully split triangular | `96.38B` | no invalid causal apply products |

Wave2 removes `12.88B` FMA versus Wave1 full-mask (`11.24%`) but still carries
`5.37B` FMA over the ideal triangular model because each diagonal `16x16` tile
uses a mask instead of six `4x4` causal subtiles.

## Timings

Triton checksum lower bound, no full output stores:

| shape | warps | mean ms | min ms | tile-pruned TFMA/s | ideal triangular ms at same rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 4 | `0.07589` | `0.07206` | `2.62` | `0.07189` |
| productionish | 4 | `8.79331` | `8.78688` | `11.57` | `8.32932` |
| productionish | 8 | `13.35442` | `13.34934` | `7.62` | `12.64976` |

The production correctness run also measured a 3-sample 4-warp mean of
`8.64545 ms`; it does not change the conclusion.

## Read

Tile-level pruning reduced modeled FMA but badly hurt execution rate:

- Wave1 full-mask productionish: `114.63B / 4.53881 ms = 25.26 TFMA/s`
- Wave2 tile-pruned productionish: `101.75B / 8.79331 ms = 11.57 TFMA/s`

The split-tile schedule more than halves effective throughput.  Even if the
diagonal tiles were fully split to reach the ideal `96.38B` model, using the
measured Wave2 rate projects `~8.33 ms`, far above TileLang `3.70674 ms`.

Even the optimistic Wave1-throughput projection is weak:

```text
96.38B / 25.26 TFMA/s ~= 3.82 ms
```

That is still slower than TileLang before adding the real output stores and
`DMIMO_V` reduction traffic.

## Conclusion

Pruning does not make this Triton owner design plausible.  The only remaining
CUDA/CuTe path would need to prune triangular work while preserving at least
full-tile Wave1 throughput, and realistically more than `26 TFMA/s` after
budgeting `~778 MiB` of required output writes plus `DMIMO_V` reduction
traffic.  Wave2 shows the straightforward triangular split loses too much
scheduler/tensor-core efficiency, so the current Triton/CUDA monolithic route
should not be treated as a path to beating TileLang.
