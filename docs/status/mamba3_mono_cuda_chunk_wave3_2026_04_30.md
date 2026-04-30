# Mamba3 Monolithic CUDA Chunk Wave 3 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Optimize the Wave 2 CUDA WMMA chunk-owner schedule without scalar expansion.

Baseline context:

- Wave 1 scalar monolithic state/LKQ/D slice: `89.02105560302735 ms` on H200
  productionish.
- Wave 2 WMMA chunk-owner slice: `8.919168281555176 ms` on H200
  productionish.
- TileLang `stage2_bf1_bb0` full `bwd_bwd`: `3.70674 ms`.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_diag_cuda_kernel.cu`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave3.py`
- `docs/status/mamba3_mono_cuda_chunk_wave3_2026_04_30.md`

Schedule change:

- Reuse the dead `Q^T` bf16 shared-memory tile as masked `LKQ` storage after
  `LKQ = K @ Q.T` has completed.
- Change `dPsi += masked(LKQ) @ dPhi` to a triangular-pruned WMMA apply:
  row tile `m` starts its tile-k loop at `m`, skipping tile-k blocks that are
  fully below the causal triangle.
- Keep LKQ, dki, state, and masked-LKQ apply on CUDA WMMA tensor cores.
  No scalar matrix-loop fallback was introduced.

The historical shared CUDA extension was updated with the same schedule change.
Wave 3 also adds a minimal one-kernel extension so Modal compiles only the
WMMA chunk-owner path instead of all prior wave kernels.

## Resource Delta

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 2 | Wave 3 |
| --- | ---: | ---: |
| registers/thread | `72` | `72` |
| dynamic smem | `106500 B` | `98308 B` |
| active blocks/SM | `2` | `2` |
| theoretical occupancy | `25.0%` | `25.0%` |
| `masked(LKQ) @ dPhi` WMMA ops/CTA | `128` | `80` |
| skipped WMMA ops/CTA | `0` | `48` |

Smoke `P=64` dynamic smem drops from `73732 B` to `65540 B`.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave3.py

git diff --check

env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave3_minimal \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave3.py \
  --shape smoke --device cuda --iters 5 --warmup 2
```

GB10 smoke correctness vs bf16-staged torch reference:

| output | max diff |
| --- | ---: |
| `DV` | `2.9802322387695312e-08` |
| per-chunk `DMIMO_V` | `7.275957614183426e-11` |
| `DSSDA` | `6.661338147750939e-16` |

GB10 cached smoke timing: mean `0.09640959799289703 ms`, p50
`0.07068800181150436 ms`; first sample was an outlier.

## H200 Smoke And Productionish

Command:

```text
modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave3.py::run_remote \
  --shape-csv smoke,productionish \
  --warmup 2 \
  --iters 5
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Modal app: `ap-klSVGlEe73hWykSGFiAXxT`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

Smoke correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `7.275957614183426e-11` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `3.893546818289906e-07` | `9.892642260922457e-12` |

Smoke timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 3 WMMA triangular chunk owner | `0.04418560042977333` | `0.04163200035691261` | `0.04419200122356415` | `0.04809600114822388` |

Productionish correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `4.76837158203125e-07` | `3.069544618483633e-10` | `3.1086244689504383e-15` |
| vs Wave 1 fp32 ref | `9.5367431640625e-07` | `6.133341230452061e-07` | `3.331246389848275e-11` |

Productionish timing:

| path | mean ms | std ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wave 3 WMMA triangular chunk-owner slice | `8.467136001586914` | `0.00339283266551086` | `8.463007926940918` | `8.465727806091309` | `8.4716157913208` |

Samples:
`[8.465727806091309, 8.4716157913208, 8.464672088623047, 8.463007926940918, 8.470656394958496]`

Comparison:

| comparison | value |
| --- | ---: |
| speedup vs Wave 1 scalar monolithic slice | `10.513715096384773x` |
| speedup vs Wave 2 WMMA slice | `1.05338667996871x` |
| delta vs Wave 2 WMMA slice | `-0.4520322799682628 ms` |
| Wave 3 slice / TileLang full `bwd_bwd` | `2.2842540889263647x` |
| margin vs TileLang full `bwd_bwd` | `-4.760396001586914 ms` |
| projected Wave10 two-launch + Wave 3 slice | `10.563866001586913 ms` |
| projected combined / TileLang full `bwd_bwd` | `2.849907466287604x` |

## H100 Smoke

Command:

```text
env CPPMEGA_MODAL_GPU=H100 \
  modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave3.py::run_remote \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3
```

Runtime:

- GPU: `NVIDIA H100 80GB HBM3`
- Torch: `2.13.0.dev20260426+cu132`
- Modal app: `ap-5lTLATxRlFA8ZaKlIoVSUO`

Correctness:

| comparison | `DV` | per-chunk `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `7.275957614183426e-11` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `3.893546818289906e-07` | `9.892642260922457e-12` |

Timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 3 WMMA triangular chunk owner | `0.042304000506798424` | `0.03936000168323517` | `0.04195199906826019` | `0.04560000076889992` |

## Read

Wave 3 is a real schedule win but still not enough.  The triangular WMMA skip
and shared-memory reuse cut `0.452 ms` from the H200 productionish slice and
reduce this slice from `2.41x` to `2.28x` TileLang, but the slice alone remains
slower than the full TileLang `bwd_bwd`.

The remaining budget issue is not correctness or scalar fallback; it is the
CTA-local WMMA schedule.  Even after pruning the causal tile-k work, the kernel
still launches many small warp-level WMMA products and keeps the chunk-owner
state/DSSDA/DV/DMIMO_V contract inside one CTA.  A larger next step likely needs
WGMMA/CuTe-style mainloops or a different producer/consumer split, not another
small in-place WMMA cleanup.
