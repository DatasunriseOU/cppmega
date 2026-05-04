# Mamba3 Monolithic CUDA Chunk Wave 8 - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: Tensor-core reduced-residency probe for the Mamba3 monolithic chunk path.

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Wave 7 proved that full `64x64` LKQ residency is not required to reach more
than one active block/SM, but its scalar row-stream schedule was too slow:
`179.76535034179688 ms` on H200 productionish.  Wave 8 keeps tensor-core work
while reducing LKQ/accumulator residency.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave8.py`
- `docs/status/mamba3_mono_cuda_chunk_wave8_2026_04_30.md`

Schedule:

- one CTA owns `(B, H, chunk, P64-panel)`;
- K/Q, dPhi, and PsiV are staged in bf16 shared memory;
- `DKI = PsiV @ dPhi.T` is tensorized with WMMA;
- LKQ is streamed as one `16x16` WMMA tile, consumed immediately for DSSDA;
- the same `64x64` float workspace is reused for `state/dPsi`;
- LKQ is streamed a second time for `masked(LKQ tile) @ dPhi` WMMA;
- `DV` and `DSSDA` are emitted by the panel-owner kernel;
- per-chunk `DMIMO_V` partials are reduced to final `DMIMO_V`.

## Resource Shape

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 4 P64 WMMA | Wave 7 row-stream | Wave 8 tile-stream WMMA |
| --- | ---: | ---: | ---: |
| owner | `(B,H,chunk,P64)` | `(B,H,chunk)` | `(B,H,chunk,P64)` |
| owner CTAs | `65536` | `32768` | `65536` |
| final `DMIMO_V` | no | yes | yes |
| dynamic smem | `65540 B` | `42244 B` | `50692 B` |
| H200 regs/thread | `40` | `125` | `72` |
| active blocks/SM | `3` | `2` | `3` |
| theoretical occupancy | `37.5%` | `25.0%` | `37.5%` |
| full LKQ live elements | `4096` | `0` | `0` |
| streamed LKQ live elements | `0` | `64` | `256` |
| DKI tensor-core | yes | no | yes |

## Local Checks

Syntax:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave8.py

git diff --check
```

GB10 smoke command:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave8_tile_stream_t256_dki_wmma \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
  --shape smoke --device cuda --iters 1 --warmup 0
```

GB10 smoke result:

- vs bf16-staged torch reference: `DV=2.9802322387695312e-08`,
  final `DMIMO_V=1.4551915228366852e-10`,
  `DSSDA=6.661338147750939e-16`.
- sm121 ptxas main kernel: `56` regs/thread, `0` spills.
- sm121 ptxas reduction kernel: `37` regs/thread, `0` spills.
- metadata: `50692 B` dynamic smem, `1` active block/SM.
- timing: `0.08326400071382523 ms`.

GB10 P128 command:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave8_tile_stream_t256_p128_dki_wmma \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
  --B 1 --S 256 --H 4 --G 1 --N 64 --P 128 --R 4 --chunk 16 \
  --device cuda --iters 3 --warmup 1
```

GB10 P128 result:

- vs bf16-staged torch reference: `DV=2.384185791015625e-07`,
  final `DMIMO_V=1.7462298274040222e-10`,
  `DSSDA=1.3322676295501878e-15`.
- mean timing: `0.10633600254853566 ms`.

## H200 Smoke And Productionish

Verbose smoke command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave8.py::run_remote \
  --shape-csv smoke \
  --warmup 0 \
  --iters 1 \
  --threads 256 \
  --verbose-build
```

H200 smoke:

- Modal app: `ap-jxl2mIyaIIouSoXwypOHvD`.
- vs bf16-staged torch reference: `DV=1.862645149230957e-09`,
  final `DMIMO_V=1.4551915228366852e-10`,
  `DSSDA=6.661338147750939e-16`.
- sm90 ptxas main kernel: `72` regs/thread, `0` spill stores,
  `0` spill loads, `1` barrier.
- sm90 ptxas reduction kernel: `32` regs/thread, `0` spill stores,
  `0` spill loads.
- metadata: `50692 B` dynamic smem, `3` active blocks/SM,
  `37.5%` theoretical occupancy.
- timing: `0.07558400183916092 ms`.

Productionish command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave8.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1 \
  --threads 256
```

Productionish result:

- Modal app: `ap-pZJvjK6R1XW9kOFTP5rfMp`.
- vs bf16-staged torch reference: `DV=4.76837158203125e-07`,
  final `DMIMO_V=1.979060471057892e-09`,
  `DSSDA=2.6645352591003757e-15`.
- vs Wave 1 fp32 reference: `DV=9.5367431640625e-07`,
  final `DMIMO_V=5.432288162410259e-06`,
  `DSSDA=3.33120198092729e-11`.
- timing: `11.180607795715332 ms`.
- ratio vs TileLang full `bwd_bwd`: `3.0162913491950696x`.
- speedup vs Wave 5 scan owner: `1.259440670565487x`.
- speedup vs Wave 7 row-stream: `16.07831645885004x`.
- speedup vs Wave 4 P64 WMMA slice: `0.7856774946034606x`.

Modal apps checked with:

```text
modal app list --json
```

All Wave8 apps were stopped with `0` tasks:

- `ap-ImgbFfE2522QCWtMwRohnS` first scalar-DKI verbose smoke
- `ap-Gp0VMchL9AnfxP5d5WaZb1` first scalar-DKI productionish
- `ap-jxl2mIyaIIouSoXwypOHvD` final tensorized-DKI verbose smoke
- `ap-pZJvjK6R1XW9kOFTP5rfMp` final tensorized-DKI productionish

## Verdict

Wave 8 is useful evidence but not a production CUDA path.

The reduced-residency tensor-core variant improves on Wave 5 and fixes the
Wave 7 scalar cliff, while preserving correctness for `DV`, final `DMIMO_V`,
and `DSSDA`.  The cost is still too high: `11.18 ms` for this subset is about
`3.0x` the full TileLang `bwd_bwd`, and the earlier Wave 4 P64 WMMA slice is
still faster even without final `DMIMO_V` reduction.

CUDA should stop here except as a reference/profiling harness.  Further
production work should move to the TileLang/CuTe/CUTLASS class of kernels
rather than more monolithic CUDA WMMA scheduling.
