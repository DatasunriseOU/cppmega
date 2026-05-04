# Mamba3 Monolithic CUDA Chunk Wave 6 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Prototype split scan ownership for the Wave 5 monolithic CUDA chunk path so the
productionish H200 shape is no longer limited to only `B*H = 128` CTAs.

Wave 6 uses a chunk-group owner:

- one CTA owns `(B, H, chunk_group)`;
- the CTA reverse-loops over only that group's chunks;
- each chunk still computes `LKQ = K @ Q.T` once and reuses it across all P64
  panels in the group;
- each CTA writes a `DMIMO_V[B,H,chunk_group,R,P]` partial;
- a second small CUDA reduction kernel sums chunk-group partials into final
  `DMIMO_V[B,H,R,P]`.

DK/DQ remain out of scope for this wave.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave6.py`
- `docs/status/mamba3_mono_cuda_chunk_wave6_2026_04_30.md`

The kernel keeps the Wave 2 bf16-staged WMMA math contract and compares against
the same staged torch reference used by Wave 5.  The timed `*_out` path
preallocates the `DMIMO_V` group scratch tensor so timings cover the two CUDA
kernels, not allocator overhead.

## Resource Shape

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 5 scan owner | Wave 6 group8 | Wave 6 group16 | Wave 6 group32 |
| --- | ---: | ---: | ---: | ---: |
| owner CTAs | `128` | `4096` | `2048` | `1024` |
| CTAs per H200 SM | `0.97` | `31.03` | `15.52` | `7.76` |
| chunks per CTA | `256` | `8` | `16` | `32` |
| logical chunk visits | `32768` | `32768` | `32768` | `32768` |
| reduction CTAs | n/a | `512` | `512` | `512` |
| dynamic smem | `68612 B` | `68612 B` | `68612 B` | `68612 B` |
| H200 regs/thread | `190` | `168` | `168` | `168` |
| active blocks/SM | `1` | `1` | `1` | `1` |
| theoretical occupancy | `12.5%` | `12.5%` | `12.5%` | `12.5%` |
| `DMIMO_V` scratch | n/a | `8 MiB` | `4 MiB` | `2 MiB` |

Wave 6 successfully recovers CTA count while preserving chunk-local LKQ reuse,
but it pays extra global scratch writes and a reduction launch.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave6.py
```

GB10 split smoke:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave6_group4 \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6.py \
  --shape smoke --device cuda --chunk-group-size 4 --iters 1 --warmup 0
```

GB10 smoke result:

- vs bf16-staged torch reference: `DV=2.9802322387695312e-08`,
  `DMIMO_V=1.4551915228366852e-10`, `DSSDA=6.661338147750939e-16`.
- vs Wave 1 fp32 reference: `DV=4.76837158203125e-07`,
  `DMIMO_V=9.119758033193648e-07`, `DSSDA=9.892642260922457e-12`.
- main kernel ptxas: `89` regs/thread, no spills.

GB10 P128 two-panel check:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave6_group8_p128 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave6.py \
  --B 1 --S 256 --H 4 --G 1 --N 64 --P 128 --R 4 --chunk 16 \
  --device cuda --chunk-group-size 8 --iters 3 --warmup 1
```

GB10 P128 result:

- vs bf16-staged torch reference: `DV=2.384185791015625e-07`,
  `DMIMO_V=1.7462298274040222e-10`, `DSSDA=1.3322676295501878e-15`.
- mean timing: `0.30136533578236896 ms`.

## H200 Smoke And Productionish

H200 smoke command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave6.py::run_remote \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3 \
  --chunk-group-size 8
```

H200 smoke:

- Modal app: `ap-3pTDxHhgcGW7GWRjIjEjGo`
- App state after run: stopped, `0` tasks.
- Correctness vs bf16-staged torch reference: `DV=1.862645149230957e-09`,
  `DMIMO_V=1.4551915228366852e-10`, `DSSDA=6.661338147750939e-16`.
- Mean timing: `0.2341759999593099 ms`.
- H200 metadata: `168` regs/thread, `67588 B` smem, `1` active block/SM.

Productionish commands:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave6.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1 \
  --chunk-group-size 8

timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave6.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1 \
  --chunk-group-size 16

timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave6.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1 \
  --chunk-group-size 32
```

Productionish correctness for all three split sizes matched the staged
reference within the same tolerance band:

| split | `DV` | final `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| group8 | `4.76837158203125e-07` | `9.313225746154785e-10` | `2.6645352591003757e-15` |
| group16 | `4.76837158203125e-07` | `9.313225746154785e-10` | `2.6645352591003757e-15` |
| group32 | `4.76837158203125e-07` | `9.313225746154785e-10` | `2.6645352591003757e-15` |

Productionish one-sample timing:

| path | owner CTAs | mean ms | delta vs Wave 5 | speedup vs Wave 5 | ratio vs TileLang full `bwd_bwd` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wave 5 scan owner | `128` | `14.08131217956543` | n/a | n/a | `3.79884x` |
| Wave 6 group8 | `4096` | `14.515359878540039` | `+0.4340476989746094` | `0.9701x` | `3.91594x` |
| Wave 6 group16 | `2048` | `14.648032188415527` | `+0.5667200088500977` | `0.9613x` | `3.95173x` |
| Wave 6 group32 | `1024` | `14.580384254455566` | `+0.4990720748901367` | `0.9658x` | `3.93348x` |

Recent Wave6 Modal apps checked with:

```text
modal app list --json
```

All Wave6 apps were stopped with `0` tasks:

- `ap-3pTDxHhgcGW7GWRjIjEjGo` smoke
- `ap-uAqTaINYjApZpKFrzYD45M` productionish group8
- `ap-0DeAVmjaFDuhvLfQe45BdZ` productionish group16
- `ap-BupRhqQemndj4mvZFFeOoP` productionish group32

## Verdict

Chunk-group scan ownership is structurally correct, but this direct split does
not beat Wave 5.

The split fixes the visible CTA-count problem (`128` CTAs -> up to `4096` CTAs)
without breaking LKQ reuse inside each chunk.  However, the main kernel still
runs at one active block/SM with large dynamic smem, and the split adds `DMIMO_V`
scratch traffic plus a reduction launch.  The result is a small regression
(`+0.43` to `+0.57 ms`) rather than a speedup.

The next useful direction is not a pure chunk-group split of the same heavy CTA.
It needs either smaller per-owner smem/register footprint, a P-panel split with
a cheaper `DSSDA`/`DMIMO_V` reduction plan, or a separate owner for the
state/LKQ-heavy path that avoids writing and reducing full `DMIMO_V` partials.
