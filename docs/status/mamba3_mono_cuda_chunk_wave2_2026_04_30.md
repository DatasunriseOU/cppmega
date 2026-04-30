# Mamba3 Monolithic CUDA Chunk Wave 2 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Replace the Wave 1 scalar LKQ/state chunk-owner body with tensor-core CUDA
pieces.  Keep `LKQ` resident and feed at least one downstream tensorized apply
with output reuse.

Baseline context:

- Wave 1 scalar monolithic state/LKQ/D slice: `89.02105560302735 ms` on H200
  productionish.
- TileLang `stage2_bf1_bb0` full `bwd_bwd`: `3.70674 ms`.
- Wave10 two-launch covered subset: `2.09673 ms`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave2.py`
- `scripts/modal_mamba3_mono_cuda_chunk_wave2.py`
- `docs/status/mamba3_mono_cuda_chunk_wave2_2026_04_30.md`

Kernel shape:

- One CTA owns one `(B, H, chunk)`.
- Fixed stage2 specialization remains `chunk=16`, `R=4`, `N=64`,
  `P % 16 == 0`, bf16 inputs.
- Stages Q/K plus bias, `dPhi = dout * mimo_o`, and `PsiV = v * mimo_v` into
  bf16 shared memory.
- Uses CUDA WMMA bf16 tensor-core tiles for:
  - `LKQ = K @ Q.T` in the existing stage2 consumer orientation.  This is the
    transpose of the common `Q @ K.T` spelling.
  - `dki = PsiV @ dPhi.T` for `DSSDA`.
  - `state = K @ dstates`.
  - `dPsi += masked(LKQ) @ dPhi`.
- Reuses the live `dPsi` tile for both `DV` and per-chunk `DMIMO_V`.
- Keeps the Wave 1 output contract: `DV`, per-chunk `DMIMO_V`, and `DSSDA`.

The torch reference mirrors the bf16 staging contract so the WMMA path is not
judged against a stricter fp32-preprocessed reference by default.  A secondary
comparison against the Wave 1 fp32 reference is still reported.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_extension.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave2.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave2.py

git diff --check

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave2.py \
  --shape smoke --device cpu --iters 1 --warmup 0
```

All passed.  CPU reference self-check was exact.

Local GB10 CUDA smoke:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave2_wmma \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave2.py \
  --shape smoke --device cuda --iters 1 --warmup 0
```

GB10 smoke:

| metric | value |
| --- | ---: |
| `DV` max diff vs bf16-staged ref | `2.9802322387695312e-08` |
| per-chunk `DMIMO_V` max diff vs bf16-staged ref | `7.275957614183426e-11` |
| `DSSDA` max diff vs bf16-staged ref | `6.661338147750939e-16` |
| mean timing | `0.07683199644088745 ms` |
| registers/thread | `96` |
| dynamic smem | `73732 B` |

## H200 Smoke

Command:

```text
timeout 1800s modal run --detach --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave2.py \
  --shape-csv smoke,productionish \
  --warmup 2 \
  --iters 5
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Smoke shape:
`B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`.

Correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `7.275957614183426e-11` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `3.893546818289906e-07` | `9.892642260922457e-12` |

Timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 2 WMMA chunk owner | `0.04206080064177513` | `0.03964800015091896` | `0.040832001715898514` | `0.0461760014295578` |

Metadata:

| metric | value |
| --- | ---: |
| threads/block | `256` |
| registers/thread | `72` |
| dynamic smem | `73732 B` |
| active blocks/SM | `3` |
| theoretical occupancy | `37.5%` |

## H200 Productionish

Productionish shape:
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

Correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `4.76837158203125e-07` | `3.069544618483633e-10` | `3.1086244689504383e-15` |
| vs Wave 1 fp32 ref | `9.5367431640625e-07` | `6.133341230452061e-07` | `3.331246389848275e-11` |

Timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wave 2 WMMA state/LKQ/D chunk-owner slice | `8.919168281555176` | `0.003510120921150315` | `8.91427230834961` | `8.918944358825684` | `8.92518424987793` |

Samples:
`[8.91801643371582, 8.919424057006836, 8.918944358825684, 8.91427230834961, 8.92518424987793]`

Comparison:

| comparison | value |
| --- | ---: |
| speedup vs Wave 1 scalar monolithic slice | `9.980869604974572x` |
| Wave 2 slice / TileLang full `bwd_bwd` | `2.406202830939094x` |
| projected Wave10 two-launch + Wave 2 slice | `11.015898281555177 ms` |
| projected combined / TileLang full `bwd_bwd` | `2.9718562083003333x` |
| projected margin vs TileLang | `-7.309158281555177 ms` |

Metadata:

| metric | value |
| --- | ---: |
| threads/block | `256` |
| registers/thread | `72` |
| dynamic smem | `106500 B` |
| active blocks/SM | `2` |
| theoretical occupancy | `25.0%` |

## H100 Smoke

Command:

```text
env CPPMEGA_MODAL_GPU=H100 \
  timeout 1200s modal run --detach --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave2.py \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3
```

Runtime:

- GPU: `NVIDIA H100 80GB HBM3`
- Torch: `2.13.0.dev20260426+cu132`

Correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `7.275957614183426e-11` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `3.893546818289906e-07` | `9.892642260922457e-12` |

Timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 2 WMMA chunk owner | `0.04554666578769684` | `0.04134399816393852` | `0.042047999799251556` | `0.053247999399900436` |

## Read

Wave 2 proves the scalar LKQ/state loop was the wrong direction: replacing it
with WMMA gives a real `~10x` H200 productionish speedup while keeping the Wave
1 outputs correct.

It is still not budget-plausible.  The straightforward WMMA owner executes many
small `64x64x{64,128}` products per chunk, stores/reloads accumulator tiles
through shared memory, and runs at only `8.919 ms` for this slice.  That is
already `2.41x` slower than the entire TileLang `bwd_bwd` and projects to
`11.02 ms` with the existing Wave10 covered subset.

Conclusion: tensor-core CUDA is moving in the right direction versus the dead
scalar kernel, but this naive WMMA chunk-owner is not moving toward the TileLang
budget.  A viable CUDA lane needs a deeper schedule change: prune triangular
work, reduce shared-memory round trips, and likely use a CuTe/CUTLASS WGMMA
mainloop/epilogue rather than warp-level WMMA tiles.
