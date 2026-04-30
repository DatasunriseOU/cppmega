# Mamba3 Monolithic CUDA Chunk Wave 7 - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: Low-live-set CUDA row-stream probe for the Mamba3 monolithic chunk path.

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Wave 5 and Wave 6 proved the scan-owner and chunk-group-owner contracts, but
both stayed at one active block/SM on H200:

- Wave 5 productionish: `14.08131217956543 ms`, `190` regs, `68612 B` smem,
  `1` block/SM, only `128` owner CTAs.
- Wave 6 group8 productionish: `14.515359878540039 ms`, `168` regs,
  `68612 B` smem, `1` block/SM, `4096` owner CTAs plus `DMIMO_V` reduction.

Wave 7 tests whether a materially smaller CTA live set can reach more than one
block/SM while preserving the `DV`, final `DMIMO_V`, and `DSSDA` correctness
subset.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave7.py`
- `docs/status/mamba3_mono_cuda_chunk_wave7_2026_04_30.md`

Schedule:

- one CTA owns `(B, H, chunk)`;
- K and Q are staged once per chunk;
- dPhi and dPsi live for one P64 panel;
- raw LKQ and masked LKQ are not materialized as full `64x64` tiles;
- one `64`-element LKQ row is streamed through shared memory;
- the chunk kernel emits `DV`, `DSSDA`, and per-chunk `DMIMO_V` partials;
- a small reduction kernel produces final `DMIMO_V[B,H,R,P]`.

This keeps the Wave 2 bf16-staged math contract: K/Q, dPhi, PsiV, and masked
LKQ are rounded through bf16 at the same semantic boundaries before comparing
against the staged torch reference.

## Resource Shape

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 6 group8 | Wave 7 row-stream |
| --- | ---: | ---: |
| owner | `(B,H,chunk_group)` | `(B,H,chunk)` |
| owner CTAs | `4096` | `32768` |
| owner CTAs per 132 H200 SMs | `31.03` | `248.24` |
| reduction CTAs | `512` | `512` |
| dynamic smem | `68612 B` | `42244 B` |
| H200 regs/thread | `168` | `125` |
| active blocks/SM | `1` | `2` |
| theoretical occupancy | `12.5%` | `25.0%` |
| full LKQ live elements | `4096` | `0` |
| streamed LKQ row elements | `0` | `64` |
| `DMIMO_V` scratch | `8 MiB` for group8 | `64 MiB` per-chunk |

Wave 7 answers the resource question positively: more than one H200 block/SM is
attainable by shrinking LKQ/accumulator residency.

## Local Checks

Syntax:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave7.py
```

GB10 smoke command:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave7_t256 \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7.py \
  --shape smoke --device cuda --iters 1 --warmup 0
```

GB10 smoke result:

- vs bf16-staged torch reference: `DV=3.637978807091713e-12`,
  final `DMIMO_V=1.1641532182693481e-10`,
  `DSSDA=4.440892098500626e-16`.
- vs Wave 1 fp32 reference: `DV=4.76837158203125e-07`,
  final `DMIMO_V=9.119539754465222e-07`,
  `DSSDA=9.892420216317532e-12`.
- sm121 ptxas main kernel: `108` regs/thread, `0` spills.
- sm121 ptxas reduction kernel: `37` regs/thread, `0` spills.
- metadata: `42244 B` dynamic smem, `2` active blocks/SM.
- timing: `0.24217599630355835 ms`.

GB10 P128 command:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave7_t256_p128 \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave7.py \
  --B 1 --S 256 --H 4 --G 1 --N 64 --P 128 --R 4 --chunk 16 \
  --device cuda --iters 3 --warmup 1
```

GB10 P128 result:

- vs bf16-staged torch reference: `DV=5.960464477539063e-08`,
  final `DMIMO_V=1.1641532182693481e-10`,
  `DSSDA=1.3322676295501878e-15`.
- mean timing: `0.4479253391424815 ms`.
- metadata: `42244 B` dynamic smem, `108` regs/thread, `2` blocks/SM.

GB10 128-thread smoke did not improve block residency:

- sm121 ptxas main kernel: `108` regs/thread, `0` spills.
- metadata: `42244 B` dynamic smem, `2` blocks/SM.

## H200 Smoke And Productionish

Smoke command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave7.py::run_remote \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3 \
  --threads 256
```

Smoke result:

- Modal app: `ap-7ZbpQdpx4rORGO6W0uBMKa`
- GPU: `NVIDIA H200`
- vs bf16-staged torch reference: `DV=0.0`,
  final `DMIMO_V=1.1641532182693481e-10`,
  `DSSDA=4.440892098500626e-16`.
- vs Wave 1 fp32 reference: `DV=4.76837158203125e-07`,
  final `DMIMO_V=9.54110873863101e-07`,
  `DSSDA=9.892420216317532e-12`.
- mean timing: `0.44710399707158405 ms`.
- H200 metadata: `125` regs/thread, `42244 B` dynamic smem,
  `2` active blocks/SM, `25.0%` theoretical occupancy.

Productionish command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave7.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1 \
  --threads 256
```

Productionish result:

- Modal app: `ap-8voLXcCUJDZ50heMw5kzaR`
- GPU: `NVIDIA H200`
- vs bf16-staged torch reference: `DV=4.76837158203125e-07`,
  final `DMIMO_V=1.862645149230957e-09`,
  `DSSDA=2.6645352591003757e-15`.
- vs Wave 1 fp32 reference: `DV=9.5367431640625e-07`,
  final `DMIMO_V=5.432055331766605e-06`,
  `DSSDA=3.33111316308532e-11`.
- timing: `179.76535034179688 ms`.
- ratio vs TileLang full `bwd_bwd`: `48.496886844450074x`.
- speedup vs Wave 5 scan owner: `0.07833162593787915x`.
- speedup vs Wave 6 group8: `0.08074614963863312x`.

Verbose H200 smoke command for ptxas:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave7.py::run_remote \
  --shape-csv smoke \
  --warmup 0 \
  --iters 1 \
  --threads 256 \
  --verbose-build
```

Verbose H200 smoke:

- Modal app: `ap-mr53XCj8cEiJcqXH3p62RX`
- sm90 ptxas main kernel: `125` regs/thread, `0` spill stores,
  `0` spill loads, `1` barrier.
- sm90 ptxas reduction kernel: `32` regs/thread, `0` spill stores,
  `0` spill loads.
- metadata matched the previous H200 smoke: `42244 B` dynamic smem,
  `2` active blocks/SM.

Recent Wave7 Modal apps checked with:

```text
modal app list --json
```

All Wave7 apps were stopped with `0` tasks:

- `ap-7ZbpQdpx4rORGO6W0uBMKa` smoke
- `ap-8voLXcCUJDZ50heMw5kzaR` productionish
- `ap-mr53XCj8cEiJcqXH3p62RX` verbose smoke

## Verdict

CUDA should not continue down this row-stream/scalar-recompute path as a
production candidate.

The positive result is resource-related: full LKQ residency is not required to
reach `>1` block/SM.  Wave 7 cuts H200 dynamic smem from `68612 B` to
`42244 B`, cuts registers from Wave 6's `168` to `125`, and reaches `2`
active blocks/SM.

The negative result is decisive for runtime: recomputing/streaming LKQ rows with
scalar consumers destroys throughput.  Productionish time regressed to
`179.76535034179688 ms`, far behind Wave 5/6 and TileLang.  A viable CUDA path
would need to keep tensor-core work and reduce live state at the same time,
for example a WMMA P64/smaller-LKQ owner with explicit reductions, or a
consumer split that leaves the state/LKQ-heavy work tensorized.  The current
monolithic CUDA path should not continue unless that new tensorized low-live-set
schedule is the next experiment.
