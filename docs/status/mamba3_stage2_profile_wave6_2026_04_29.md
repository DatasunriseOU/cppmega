# Mamba3 Stage2 Profile Wave6 - 2026-04-29

Branch: `worker/mamba3-stage2-cuda-ab`

Scope: profiling/resource follow-up for stage2 `(bf=1,bb=0)` after wave5.
Runs crossed UTC midnight; the requested status date/file name remains
`2026_04_29`.

## Harness

Added:

- `scripts/modal_mamba3_stage2_profile_wave6.py`

The harness reuses the wave5 stage2 A/B helper for CUDA-event timings,
correctness, torch profiler traces, and generated kernel source artifacts. It
then tries focused NCU captures per variant/phase with:

- `--target-processes all`
- `--profile-from-start off`
- CUDA profiler API range around one requested launch
- LD_LIBRARY_PATH repair paths:
  `/usr/local/nvidia/lib64`, `/usr/local/nvidia/lib`,
  `/usr/local/cuda/lib64`, `/usr/local/cuda/compat`,
  `/usr/local/cuda/extras/CUPTI/lib64`,
  `/usr/local/cuda/targets/x86_64-linux/lib`,
  `/usr/lib/x86_64-linux-gnu`, plus Nsight Compute host/target lib dirs if
  present
- NCU attempts:
  `LaunchStats` with LD fix, `LaunchStats` default env,
  `LaunchStats+Occupancy+MemoryWorkloadAnalysis`, `--set basic`, and explicit
  launch metrics for registers/static-smem/dynamic-smem
- optional nsys fallback, plus torch profiler/generated source fallback
- partial `partial_report.json` commits after timing and after each profiler
  phase, so profiler crashes preserve already-collected artifacts

## Commands

Local compile:

```text
python -m py_compile scripts/modal_mamba3_stage2_profile_wave6.py
```

Smoke with nsys fallback enabled:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_stage2_profile_wave6.py \
  --run-id mamba3_stage2_profile_wave6_smoke_20260429_1 \
  --shape-csv smoke \
  --variant-csv baseline,stage2_force_nontma \
  --phase-csv bwd_bwd \
  --warmup 0 \
  --iters 1 \
  --profiler-warmup 0 \
  --profiler-launches 1 \
  --ncu-timeout-sec 180 \
  --nsys-timeout-sec 240 \
  --no-strace-on-ncu-failure
```

Productionish `bwd_bwd`, 8-iteration timing, NCU attempts, nsys disabled after
the smoke showed nsys crashes:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 2400s \
modal run scripts/modal_mamba3_stage2_profile_wave6.py \
  --run-id mamba3_stage2_profile_wave6_h200_prod_bwd_bwd_20260429_3 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma \
  --phase-csv bwd_bwd \
  --warmup 2 \
  --iters 8 \
  --profiler-warmup 1 \
  --profiler-launches 1 \
  --ncu-timeout-sec 480 \
  --nsys-timeout-sec 600 \
  --no-nsys-fallback \
  --no-strace-on-ncu-failure
```

Productionish `bwd_fwd` NCU failure check, short timing:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
modal run scripts/modal_mamba3_stage2_profile_wave6.py \
  --run-id mamba3_stage2_profile_wave6_h200_prod_bwd_fwd_20260429_4 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma \
  --phase-csv bwd_fwd \
  --warmup 0 \
  --iters 1 \
  --profiler-warmup 1 \
  --profiler-launches 1 \
  --ncu-timeout-sec 480 \
  --nsys-timeout-sec 600 \
  --no-nsys-fallback \
  --no-strace-on-ncu-failure
```

## Artifacts

Modal Volume: `cppmega-mamba3-benchmarks`

- `/benchmarks/mamba3_stage2_profile_wave6/mamba3_stage2_profile_wave6_smoke_20260429_1/summary.json`
- `/benchmarks/mamba3_stage2_profile_wave6/mamba3_stage2_profile_wave6_h200_prod_bwd_bwd_20260429_3/summary.json`
- `/benchmarks/mamba3_stage2_profile_wave6/mamba3_stage2_profile_wave6_h200_prod_bwd_bwd_20260429_3/report.json`
- `/benchmarks/mamba3_stage2_profile_wave6/mamba3_stage2_profile_wave6_h200_prod_bwd_fwd_20260429_4/summary.json`
- per-variant torch tables/traces and generated source under each run's
  `timing/productionish/{baseline,stage2_force_nontma}/`
- NCU stdout/stderr/command files under
  `productionish/{variant}/{phase}/ncu/`

## Device and Tooling

- GPU: `NVIDIA H200`, 2 visible devices, capability `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA runtime in torch: `13.2`
- `nvidia-smi`: driver `580.95.05`, CUDA version `13.2`
- NCU: `/usr/local/cuda/bin/ncu`,
  `NVIDIA Nsight Compute 2026.1.1.0 (build 37634170)`
- nsys: `NVIDIA Nsight Systems 2026.1.1.0`
- TileLang: `0.1.8+cu132.gitf309d814`

## Productionish Timing

Shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16, bf16`

Authoritative wave6 timing is from
`mamba3_stage2_profile_wave6_h200_prod_bwd_bwd_20260429_3`
(`warmup=2`, `iters=8`):

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA | speedup vs baseline |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| baseline | 1.8733 | 3.6912 | 5.5411 | no / 0 | no / 0 | 1.0000x |
| stage2 `(bf=1,bb=0)` | 1.7843 | 3.6881 | 5.4424 | yes / 5 | no / 0 | 1.0181x |

Correctness vs baseline:

- `max_main_grad_abs_diff=0.0`
- all tracked outputs/grads, including `qk_dot` and `states`, had
  `max_abs=0.0`

Read: this independently reproduces wave5's conclusion. Stage2's win is still
from `bwd_fwd`; `bwd_bwd` remains flat.

Torch profiler table from the earlier productionish attempt
`mamba3_stage2_profile_wave6_h200_prod_20260429_1` agrees:

| variant | kernel | calls | CUDA total | CUDA avg |
| --- | --- | ---: | ---: | ---: |
| baseline | `mamba_mimo_bwd_fwd_kernel_kernel` | 3 | 5.554 ms | 1.851 ms |
| baseline | `mamba_mimo_bwd_bwd_kernel_kernel` | 3 | 11.142 ms | 3.714 ms |
| stage2 | `mamba_mimo_bwd_fwd_kernel_kernel` | 3 | 5.338 ms | 1.779 ms |
| stage2 | `mamba_mimo_bwd_bwd_kernel_kernel` | 3 | 11.103 ms | 3.701 ms |

## Source and Resource Metadata

Real runtime register/smem launch attributes were not captured because both NCU
and nsys failed. Generated source metadata was captured:

| kernel source | launch bounds | TMA loads/stores | WS producer guard | source-derived dyn-smem lower bound | local-array pressure proxy |
| --- | --- | ---: | --- | ---: | ---: |
| baseline `bwd_fwd` | `(128,1)` | 0 / 0 | no | ~90,114 bytes | float[361] + bf16[326] |
| stage2 `bwd_fwd` | `(256,1)` | 5 / 3 | `128 <= threadIdx.x` | ~86,018 bytes | float[363] + bf16[324] |
| baseline `bwd_bwd` | `(256,1)` | 0 / 0 | no | ~190,082 bytes | float[500] + bf16[340] |
| stage2 `bwd_bwd` | `(256,1)` | 0 / 0 | no | ~190,082 bytes | float[500] + bf16[332] |

The smem numbers above are a source-scrape lower bound from max visible
`buf_dyn_shmem` offsets, not CUDA launch attributes. Treat them as directionally
useful only. Register counts still require working NCU/nsys or a compiler
resource dump path.

## NCU Result

For productionish `bwd_bwd`, both baseline and stage2 tried:

| attempt | LD fix | result |
| --- | --- | --- |
| `LaunchStats` | yes | failed, rc 9 |
| `LaunchStats` | no | failed, rc 9 |
| `LaunchStats+Occupancy+MemoryWorkloadAnalysis` | yes | failed, rc 9 |
| `--set basic` | yes | failed, rc 9 |
| explicit launch regs/static-smem/dynamic-smem metrics | yes | failed, rc 9 |

The productionish `bwd_fwd` run failed the same way for both variants.

Representative NCU output:

```text
==PROF== Connected to process ...
==ERROR== An error was reported by the counter measurement library:
==ERROR== Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
==PROF== Trying to shutdown target application
==ERROR== The application returned an error code (9).
```

LD_LIBRARY_PATH fixes did not change the failure, which makes a missing
container-side `libcuda.so` path unlikely.

## nsys Fallback Result

The smoke run tried both nsys `--capture-range=cudaProfilerApi` and full capture.
Both ran the workload but produced no usable report, ending with:

```text
FATAL ERROR: .../CUDA13.2/QuadD/Target/Daemon/TimeConversion.cpp(531)
Dynamic exception type: boost::wrapexcept<QuadDCommon::InternalErrorException>
std::exception::what: InternalErrorException
```

The first productionish combined run also died while entering the nsys fallback
after baseline `bwd_bwd` NCU attempts. Productionish `_3` and `_4` therefore
disabled nsys fallback and used torch profiler/generated source fallback.

## Root Cause

Most likely root cause: the Modal H200 host is on R580 (`580.95.05`) while the
container carries CUDA 13.2 developer tools (`ncu`/`nsys` 2026.1.1). NVIDIA's
Nsight Compute 2026.1 page lists Linux driver `595.58.03` or newer as the
recommended driver for that release, while Nsight Compute 2025.3 lists Linux
driver `580.82.07` or newer and CUDA Toolkit 13.0 support. References:

- https://developer.nvidia.com/tools-overview/nsight-compute/get-started
- https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3
- https://developer.nvidia.com/blog/using-nsight-compute-in-containers/

This matches the observed behavior: the profiler tools start, see the target,
then fail inside NVIDIA's counter/QuadD path rather than at Python, TileLang, or
CUDA workload initialization.

## Wave7 Guidance

1. Fix profiler tooling before chasing roofline counters:
   - preferred on current Modal R580 hosts: install/use Nsight Compute 2025.3.x
     and matching Nsight Systems 2025.3.x in the Modal image, or point
     `CPPMEGA_NCU_BIN` at that install;
   - alternative: move profiling to a host/image pair with driver
     `>=595.58.03` for CUDA 13.2/Nsight 2026.1.
2. Once NCU works, collect only `bwd_bwd` first with:
   `LaunchStats`, `Occupancy`, `MemoryWorkloadAnalysis`, `SpeedOfLight`, and
   `SpeedOfLight_RooflineChart`.
3. Optimization target remains `bwd_bwd`, not stage2 TMA plumbing:
   - stage2 `bwd_fwd` already improves about 4.6-5.0%;
   - stage2 `bwd_bwd` is unchanged (`~3.69 ms`) and consumes about two-thirds
     of the stage2 chain;
   - source proxy shows `bwd_bwd` has high local-array pressure and ~190 KiB
     dynamic-smem lower bound, so wave7 should prioritize reducing `bwd_bwd`
     register/smem pressure or replacing the diagonal/GEMM subpath inside the
     existing launch boundary.
4. Keep `(bf=1,bb=0)` as the only mergeable candidate. Do not enable
   `bb_num_stages > 0` without fresh productionish timing and counters.
