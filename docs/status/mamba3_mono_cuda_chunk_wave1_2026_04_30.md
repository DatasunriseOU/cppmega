# Mamba3 Monolithic CUDA Chunk Wave 1 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Start Lane A for a hand-written CUDA monolithic chunk kernel.  The target is the
missing state/LKQ/D side of `bwd_bwd`: keep chunk-local `LKQ` and state-derived
intermediates live and feed multiple output families from the same CTA/launch.

The prior best covered-subset context remains:

- wave10 two-launch wave7 + qk/`DMIMO_V` output-owner: `2.09673 ms`
- wave10 one-launch combined: `2.31212 ms`
- TileLang `stage2_bf1_bb0` `bwd_bwd`: `3.70674 ms`

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave1.py`
- `scripts/modal_mamba3_mono_cuda_chunk_wave1.py`
- `docs/status/mamba3_mono_cuda_chunk_wave1_2026_04_30.md`

Changes:

- Added compile-time `RR_DIAG_MONO_P_TILE` and threaded it into the extension
  cache key and NVCC defines.
- Added `stage2_mono_state_lkq_d_chunk_owner` CUDA entry points, including an
  allocating wrapper, out wrapper, and metadata query.
- Added a chunk-owner kernel with one CTA per `(B, H, chunk)`.
- Inside each CTA, materialized `LKQ = K @ Q.T` once into shared memory.
- Built a p-tiled state/LKQ/D `dPsiV` shared-memory tile and reused it for:
  - `DV`
  - per-chunk `DMIMO_V`
- Reused the same live `LKQ` tile for a scalar-family `DSSDA` consumer.
- Added a PyTorch reference for this subset and a Modal runner for H200/H100.

This wave is still a subset/skeleton.  It uses preprocessed `q_flat/k_flat` plus
bias as the Q/K contract.  Rotary/trap preprocessing is not included.  The
`DMIMO_V` result is a per-chunk partial, so a final reduction owner is still
needed before this can replace the full stage.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave1.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave1.py

git diff --check
```

Both passed.

CPU smoke:

```text
python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave1.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

Reference self-check passed.  CPU reference timing was `2.5392 ms`.

Local GB10 CUDA smoke:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave1 \
  RR_DIAG_MONO_P_TILE=32 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave1.py \
  --shape smoke --device cuda --iters 1 --warmup 0
```

GB10 smoke results:

| metric | value |
| --- | ---: |
| `DV` max abs diff | `1.19209e-07` |
| per-chunk `DMIMO_V` max abs diff | `5.82077e-11` |
| `DSSDA` max abs diff | `4.44089e-16` |
| timing mean | `0.539936 ms` |
| registers/thread | `88` |
| dynamic smem | `25600 B` |
| active blocks/SM | `2` |
| theoretical occupancy | `33.3%` |

## H200 Smoke

Command:

```text
timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave1.py \
  --shape-csv smoke \
  --warmup 2 \
  --iters 5
```

Device/runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Smoke shape:
`B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`.

Correctness:

| output | max abs diff |
| --- | ---: |
| `DV` | `7.450580596923828e-09` |
| per-chunk `DMIMO_V` | `5.820766091346741e-11` |
| `DSSDA` | `4.440892098500626e-16` |

Timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave1 monolithic state/LKQ/D chunk-owner slice | `0.3085055947303772` | `0.0018416476610451065` | `0.30646398663520813` | `0.30825600028038025` | `0.3113279938697815` |

Metadata:

| metric | value |
| --- | ---: |
| threads/block | `256` |
| registers/thread | `128` |
| dynamic smem | `25600 B` |
| active blocks/SM | `2` |
| theoretical occupancy | `25.0%` |

## H200 Productionish

Command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave1.py \
  --shape-csv productionish \
  --warmup 2 \
  --iters 5
```

Productionish shape:
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

Correctness:

| output | max abs diff |
| --- | ---: |
| `DV` | `4.76837158203125e-07` |
| per-chunk `DMIMO_V` | `1.3096723705530167e-10` |
| `DSSDA` | `1.3322676295501878e-15` |

Timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave1 monolithic state/LKQ/D chunk-owner slice | `89.02105560302735` | `0.013613886470832191` | `89.00796508789062` | `89.01606750488281` | `89.04617309570312` |

Samples:
`[89.04617309570312, 89.02365112304688, 89.00796508789062, 89.01606750488281, 89.01142120361328]`

Comparison:

| comparison | value |
| --- | ---: |
| wave1 slice / TileLang `bwd_bwd` | `24.01599669872377x` |
| projected wave10 two-launch + wave1 slice | `91.11778560302734 ms` |
| projected combined / TileLang `bwd_bwd` | `24.58165007608501x` |
| projected margin vs TileLang | `-87.41104560302735 ms` |

Metadata:

| metric | value |
| --- | ---: |
| threads/block | `256` |
| registers/thread | `128` |
| dynamic smem | `25600 B` |
| active blocks/SM | `2` |
| theoretical occupancy | `25.0%` |

## H100 Smoke

Command:

```text
env CPPMEGA_MODAL_GPU=H100 \
  timeout 1200s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave1.py \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3
```

Device/runtime:

- GPU: `NVIDIA H100 80GB HBM3`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Smoke shape:
`B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`.

Correctness:

| output | max abs diff |
| --- | ---: |
| `DV` | `7.450580596923828e-09` |
| per-chunk `DMIMO_V` | `5.820766091346741e-11` |
| `DSSDA` | `4.440892098500626e-16` |

Timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wave1 monolithic state/LKQ/D chunk-owner slice | `0.3237653374671936` | `0.01530694313624637` | `0.310016006231308` | `0.3161599934101105` | `0.34512001276016235` |

Metadata matched H200 for this kernel family: `128` registers/thread,
`25600 B` dynamic shared memory, `2` active blocks/SM, `25.0%` theoretical
occupancy.

## Modal Build Note

Initial H200 productionish attempts were slowed by remote extension builds for
multiple visible architectures (`sm_90`, `sm_100`, `sm_121`).  The Modal runner
now sets `TORCH_CUDA_ARCH_LIST=9.0` inside `run_remote`, which keeps the H200
and H100 builds scoped to `sm_90`.

`modal app list` after cleanup showed all wave1 apps stopped with `0` tasks.

## Blockers

- The CTA model is correct for reuse but the math implementation is still scalar
  FMA work inside CUDA loops.  It is a compile-checked monolithic skeleton, not a
  viable production kernel.
- The productionish kernel launches only `B * H * nchunks = 32768` CTAs, and
  each CTA carries high scalar work with `128` registers/thread and only
  `25%` theoretical occupancy.
- `LKQ`, state `K @ dstates`, `LKQ @ dPhi`, and `DSSDA` all need tensor-core
  paths.  Continuing with handwritten scalar loops will not close the `89 ms`
  gap.
- This slice has not integrated rotary/trap preprocessing.
- Per-chunk `DMIMO_V` partials still need a reduction strategy.

## Read

The monolithic direction is viable only if the next wave is a real tensorized
CuTe/WGMMA implementation.  The reuse pattern is the right one: keep `LKQ`
chunk-local, feed both `DV` and `DMIMO_V` from a live `dPsiV` tile, and attach
scalar-family consumers in the same owner.  The current scalar FMA body proves
the ownership and reference contract but is not performance-viable.

Next wave recommendation:

1. Stop expanding this scalar CUDA body.
2. Build a minimal CuTe/WGMMA chunk-owner microkernel for one reused subpath:
   `LKQ = K @ Q.T` followed by `dPsiV += LKQ @ dPhi`, with the existing
   `DV`/per-chunk `DMIMO_V` epilogue.
3. Keep `DSSDA` behind a compile flag until the tensorized `LKQ` path is under
   `1 ms` productionish.
4. Decide whether `DMIMO_V` reduction is chunk-owned plus second launch, or a
   wider output-owner schedule that consumes the same live state tile.
