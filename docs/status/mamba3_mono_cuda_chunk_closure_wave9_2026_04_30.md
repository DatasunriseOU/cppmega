# Mamba3 Monolithic CUDA Chunk Closure Wave 9 - 2026-04-30

Status: evidence
Canonical: none
Date: 2026-04-30
Scope: Close `worker/mamba3-mono-cuda-chunk` as a reference/profiling artifact.

Branch: `worker/mamba3-mono-cuda-chunk`

## Closure Decision

Do not merge any monolithic CUDA chunk code as a production path.  The branch is
useful as reference code, correctness scaffolding, and profiling evidence for
why this ownership class does not beat the existing TileLang full `bwd_bwd`
baseline.

The best CUDA slice in this branch remains Wave 4, which does not compute final
`DMIMO_V`; the best final-`DMIMO_V` CUDA slice is Wave 8, but it is still about
`3.0x` slower than the TileLang full backward.

## H200 Timing Matrix

Productionish shape:
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

Commands:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave8.py::run_remote \
  --shape-csv productionish \
  --warmup 1 \
  --iters 3 \
  --threads 256

timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave5.py::run_remote \
  --shape-csv productionish \
  --warmup 1 \
  --iters 3

timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave4.py::run_remote \
  --shape-csv productionish \
  --warmup 1 \
  --iters 3
```

Results on `NVIDIA H200`, Torch `2.13.0.dev20260426+cu132`, image
`ghcr.io/jewelmusicee/cppmega:785c3fd`:

| path | output contract | mean ms | samples ms | ratio vs TileLang full `bwd_bwd` |
| --- | --- | ---: | --- | ---: |
| Wave 4 P64 WMMA chunk owner | `DV`, per-chunk `DMIMO_V`, `DSSDA` | `8.835327784220377` | `[8.837247848510742, 8.833919525146484, 8.834815979003906]` | `2.383584439216232x` |
| Wave 5 scan owner | `DV`, final `DMIMO_V`, `DSSDA` | `14.08907699584961` | `[14.145471572875977, 14.061727523803711, 14.06003189086914]` | `3.8009347825446644x` |
| Wave 8 tile-stream WMMA | `DV`, final `DMIMO_V`, `DSSDA` | `11.1550079981486` | `[11.156895637512207, 11.154144287109375, 11.153984069824219]` | `3.009385065623324x` |
| TileLang `stage2_bf1_bb0` full `bwd_bwd` | full TileLang backward | `3.70674` | n/a | `1.0x` |

Fresh-run comparisons:

- Wave 8 is `1.2630270635563845x` faster than Wave 5.
- Wave 8 is `2.3196802139282227 ms` slower than Wave 4 despite reducing LKQ
  residency and still lacks Wave 4's timing advantage.
- Wave 4 is still `5.128587784220377 ms` slower than TileLang while covering
  less of the final output contract.
- Wave 8 is `7.4482679981486 ms` slower than TileLang even though TileLang
  computes the full `bwd_bwd`.

## Resource Evidence

| path | owner | owner CTAs | regs/thread | dynamic smem | active blocks/SM | occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wave 4 | `(B,H,chunk,P64-panel)` | `65536` | `40` | `65540 B` | `3` | `37.5%` |
| Wave 5 | `(B,H)` scan | `128` | `190` | `68612 B` | `1` | `12.5%` |
| Wave 8 | `(B,H,chunk,P64-panel)` | `65536` | `72` | `50692 B` | `3` | `37.5%` |

Existing verbose-build evidence for Wave 8 H200 smoke showed ptxas at `72`
regs/thread for the main kernel and `32` regs/thread for the reduction kernel,
with no spill loads/stores.  The Wave 4 and Wave 5 reruns above used runtime
CUDA function metadata for resource accounting.

Local `ncu` and `nsys` exist on the GB10 workstation, but no bounded H200 Modal
runner in this branch exposes an `ncu`/`nsys` capture.  I did not add a profiler
harness or new algorithm for branch closure; the H200 evidence here is CUDA
event timing plus resource metadata and the existing ptxas smoke record.

## Bottleneck Diagnosis

Wave 5 demonstrates the scan-local reuse tradeoff but underfills H200: the
productionish shape launches only `B*H = 128` scan-owner CTAs, or
`0.9696969696969697` owner CTAs per 132-SM H200, with `190` regs/thread and one
active block/SM.  It reuses LKQ but loses too much parallelism.

Wave 4 recovers occupancy with P64 panel CTAs and low registers, but the panel
split duplicates Q/K staging and `LKQ = K @ Q.T` for each P panel.  For P128,
that doubles LKQ work from `64` to `128` WMMA ops per logical chunk and adds
cross-panel `DSSDA` zero/atomic behavior.  It is the fastest CUDA slice but
still slower than full TileLang and does not reduce final `DMIMO_V`.

Wave 8 removes full LKQ residency and keeps tensor-core DKI/state/LKQ consumers,
but the extra tile streams plus final `DMIMO_V` reduction still leave it at
`11.155 ms`.  Its resource shape is acceptable (`72` regs, `50692 B` smem, three
blocks/SM), so the remaining gap is structural work duplication and memory/
launch overhead inside this monolithic WMMA owner class, not a simple occupancy
or spill problem.

## Modal Cleanup

Checked with:

```text
modal app list --json
```

The new matrix apps were stopped with `0` tasks:

- `ap-jfRXx8My5VElde7fapADjn` - Wave 8 H200 matrix
- `ap-fpFvoDBGPoxNliWrJEpwsN` - Wave 5 H200 matrix
- `ap-WeoRq2jiOEuVNNTO98GZjT` - Wave 4 H200 matrix

The same listing showed prior Mamba3 monolithic CUDA apps stopped with `0`
tasks.

## Recommendation

Merge no CUDA implementation from this branch into `main` as production code.
If anything is kept, keep only docs and reference/profiling artifacts under the
existing example/status locations.  Future production work should stay in the
TileLang/CuTe/CUTLASS class of implementations, not further monolithic CUDA
WMMA scheduling on this branch.
