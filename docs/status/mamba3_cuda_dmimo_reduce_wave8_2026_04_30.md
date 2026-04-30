# Mamba3 CUDA DMIMO_V Reduction Wave 8 - 2026-04-30

Branch: `worker/mamba3-cuda-dmimo-reduce`

Base: `d399cf2` wave7 chunk-owner qk/dV CUDA prototype.

## Goal

Explore the `DMIMO_V` ownership problem that wave7 intentionally deferred.
With wave7's `(B, H, chunk)` owner, `DMIMO_V[b,h,r,p]` needs a reduction over
all chunks/timesteps, unlike `DV`, `DK`, `DQ`, and `DGAMMA_DIAG` which have
local output ownership.

Scope of this experiment: the same-time `qk_dot -> dPsiV` contribution already
covered by wave7's qk/dV path. This isolates the cross-chunk reduction
strategy; the full future `bwd_bwd` still needs the state/intra-chunk `dPsiV`
terms integrated.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/dmimo_reduce_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/dmimo_reduce_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_wave8_dmimo_reduce_cuda.py`
- `scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py`

CUDA variants:

1. `atomic_chunk`
   - one CTA owns `(B, H, chunk, R)`;
   - reduces inside the chunk, then `atomicAdd`s one fp32 partial per
     `[B,H,R,P]` element.

2. `two_pass`
   - pass 1 writes fp32 partials `[B,H,nchunks,R,P]`;
   - pass 2 reduces partials into `[B,H,R,P]`.

3. `output_owner`
   - one CTA owns `(B, H, R, P-tile)`;
   - loops over all `S` and writes output uniquely.

4. `output_owner_rvec`
   - one CTA owns `(B, H, P-tile)` and computes all four `R` outputs;
   - reuses each loaded timestep/base value across all `R` lanes;
   - no atomics and no partial tensor.

## H200 Run

Command:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_dmimo_reduce_wave8_cuda.py \
  --shape-csv smoke,productionish \
  --warmup 3 \
  --iters 10
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

## Correctness

All CUDA variants compare against a torch reference for the qk-dot
`DMIMO_V` contribution.

| shape | check | max abs diff |
| --- | --- | ---: |
| smoke | atomic vs torch | 1.776e-15 |
| smoke | two-pass vs torch | 1.360e-15 |
| smoke | output-owner vs torch | 2.665e-15 |
| smoke | output-owner all-R vs torch | 2.665e-15 |
| productionish | atomic vs torch | 2.487e-14 |
| productionish | two-pass vs torch | 2.132e-14 |
| productionish | output-owner vs torch | 1.066e-13 |
| productionish | output-owner all-R vs torch | 1.066e-13 |
| productionish | output-owner vs all-R output-owner | 0.000e+00 |

## Performance

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| component | mean ms | memory/temp | projected total with wave7 1.91459ms | ratio vs TileLang 3.70674ms |
| --- | ---: | --- | ---: | ---: |
| refreshed wave7 diag+qk/dV | 1.92434 | existing outputs | - | - |
| `DMIMO_V` atomic chunk | 1.95194 | no temp, 16.78M atomics | 3.86653 | 1.043 |
| `DMIMO_V` two-pass partial writer | 1.97627 | writes 64.0 MiB partials | - | - |
| `DMIMO_V` two-pass final reduce | 0.03641 | reads 64.0 MiB partials | - | - |
| `DMIMO_V` two-pass total | 2.00769 | 64.0 MiB temp, 128.25 MiB extra global R/W | 3.92228 | 1.058 |
| `DMIMO_V` output-owner `(B,H,R,Ptile)` | 0.92715 | no temp | 2.84174 | 0.767 |
| `DMIMO_V` output-owner `(B,H,Ptile)` all-R | 0.53634 | no temp | 2.45093 | 0.661 |

Smoke shape is underfilled and not decision quality, but all paths were
correct. In smoke, the tiny two-pass path is fastest due launch/noise effects;
productionish reverses that.

## Resource Metadata

| kernel | regs/thread | static smem | active blocks/SM | occupancy |
| --- | ---: | ---: | ---: | ---: |
| atomic chunk | 120 | 0 B | 4 | 25.0% |
| partial writer | 127 | 0 B | 4 | 25.0% |
| final reducer | 32 | 0 B | 16 | 100.0% |
| output-owner `(B,H,R,Ptile)` | 48 | 512 B | 10 | 62.5% |
| output-owner all-R `(B,H,Ptile)` | 40 | 2048 B | 12 | 75.0% |
| wave7 diag+qk/dV | 80 | 0 B | 6 | 37.5% |

## Read

Do not use atomics for `DMIMO_V`: chunk-level aggregation still leaves
16.78M contended atomics at productionish shape and lands slower than current
TileLang when added to wave7.

Do not use a standalone two-pass partial writer for this qk-dot slice: the
final reduction itself is cheap (`0.036 ms`), but producing the partial tensor
costs about `1.98 ms` and requires a 64 MiB temporary. Two-pass only remains
interesting if a future full chunk kernel can piggyback partial accumulation
while `dPsiV` is already live.

Recommended ownership for `DMIMO_V`: remap to an output tile,
`(B, H, Ptile)` with all `R` lanes computed in one CTA. This is the best
measured productionish path (`0.536 ms`), uses no extra temporary memory, and
keeps the projected wave7+DMIMO_V total at `2.451 ms`, about `66%` of the
current TileLang `bwd_bwd` time.
