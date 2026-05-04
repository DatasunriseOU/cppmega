# Mamba3 Stage2 Profile Wave7 - 2026-04-30

Branch: `worker/mamba3-stage2-cuda-ab`

Scope: make Nsight Compute usable for Mamba3 stage2 `bwd_bwd` profiling on
Modal H200, or prove the exact blocker after the Wave6 CUDA 13.2 / Nsight
2026.1 failure.

## Official NVIDIA Findings

- Nsight Compute 2025.3 is the CUDA 13.0-era profiler. NVIDIA's 2025.3 page
  lists the Linux Desktop download, CUDA Toolkit 13.0 support, and recommended
  Linux driver `580.82.07` or newer.
- NVIDIA's public devtools repo exposes the exact Linux package used here:
  `nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb`.
- CUDA 13.0 Update 1 release notes list `Nsight Compute 2025.3.1.4` and
  `NVIDIA Linux Driver 580.82.07`.
- The current Nsight Compute 2026.1 page recommends Linux driver `595.58.03`
  or newer, and CUDA 13.2 Update 1 release notes list `Nsight Compute
  2026.1.1.2` with `NVIDIA Linux Driver 595.58.03`.

Sources:

- https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3
- https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/
- https://docs.nvidia.com/cuda/archive/13.0.1/pdf/CUDA_Toolkit_Release_Notes.pdf
- https://developer.nvidia.com/tools-overview/nsight-compute/get-started
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/nsight-compute/2025.3/ReleaseNotes/index.html

## Harness

Added:

- `scripts/modal_mamba3_stage2_profile_wave7.py`

The harness is based on Wave6 but changes profiler handling:

- downloads NVIDIA's public `nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb`;
- extracts it into `/opt/nvidia/nsight-compute-2025.3.1-deb` to avoid dpkg
  conflicts with the CUDA 13.2 image's installed 2026.1 package;
- sets `CPPMEGA_NCU_BIN` to
  `/opt/nvidia/nsight-compute-2025.3.1-deb/opt/nvidia/nsight-compute/2025.3.1/ncu`;
- records all NCU candidates and selects the first runnable `2025.3` binary;
- attempts `LaunchStats+Occupancy+MemoryWorkloadAnalysis` first;
- adds a clean host-driver `LD_LIBRARY_PATH` mode that excludes
  `/usr/local/cuda/compat` and excludes 2026.1 Nsight library directories;
- no longer auto-inserts `baseline` when `--variant-csv stage2_force_nontma`
  is requested.

Local compile:

```text
python -m py_compile scripts/modal_mamba3_stage2_profile_wave7.py
```

## Runs

Smoke with baseline and stage2, before the clean host-driver LD attempt was
added:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
modal run scripts/modal_mamba3_stage2_profile_wave7.py \
  --run-id mamba3_stage2_profile_wave7_smoke_20260430_2 \
  --shape-csv smoke \
  --variant-csv stage2_force_nontma \
  --phase-csv bwd_bwd \
  --warmup 0 \
  --iters 1 \
  --profiler-warmup 0 \
  --profiler-launches 1 \
  --ncu-timeout-sec 240 \
  --nsys-timeout-sec 240 \
  --no-nsys-fallback \
  --no-strace-on-ncu-failure
```

Final smoke with clean host-driver LD attempt:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1200s \
modal run scripts/modal_mamba3_stage2_profile_wave7.py \
  --run-id mamba3_stage2_profile_wave7_smoke_cleanhostld_20260430_1 \
  --shape-csv smoke \
  --variant-csv stage2_force_nontma \
  --phase-csv bwd_bwd \
  --warmup 0 \
  --iters 1 \
  --profiler-warmup 0 \
  --profiler-launches 1 \
  --ncu-timeout-sec 240 \
  --nsys-timeout-sec 240 \
  --no-nsys-fallback \
  --no-strace-on-ncu-failure
```

Final productionish `stage2_force_nontma bwd_bwd` with clean host-driver LD
attempt:

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
modal run scripts/modal_mamba3_stage2_profile_wave7.py \
  --run-id mamba3_stage2_profile_wave7_prod_stage2_bwd_bwd_hostld_20260430_2 \
  --shape-csv productionish \
  --variant-csv stage2_force_nontma \
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

## Artifacts

Modal Volume: `cppmega-mamba3-benchmarks`

- `/benchmarks/mamba3_stage2_profile_wave7/mamba3_stage2_profile_wave7_smoke_20260430_2/summary.json`
- `/benchmarks/mamba3_stage2_profile_wave7/mamba3_stage2_profile_wave7_smoke_cleanhostld_20260430_1/report.json`
- `/benchmarks/mamba3_stage2_profile_wave7/mamba3_stage2_profile_wave7_smoke_cleanhostld_20260430_1/summary.json`
- `/benchmarks/mamba3_stage2_profile_wave7/mamba3_stage2_profile_wave7_prod_stage2_bwd_bwd_hostld_20260430_2/report.json`
- `/benchmarks/mamba3_stage2_profile_wave7/mamba3_stage2_profile_wave7_prod_stage2_bwd_bwd_hostld_20260430_2/summary.json`
- NCU stdout/stderr/command artifacts under
  `productionish/stage2_force_nontma/bwd_bwd/ncu/`
- Torch profiler trace/table under
  `timing/productionish/stage2_force_nontma/`

## Environment

Final productionish run:

- GPU: `NVIDIA H200`, 2 visible devices, capability `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA runtime in torch: `13.2`
- TileLang: `0.1.8+cu132.gitf309d814`
- driver: `580.95.05`
- selected NCU:
  `/opt/nvidia/nsight-compute-2025.3.1-deb/opt/nvidia/nsight-compute/2025.3.1/ncu`
- selected NCU version:
  `Version 2025.3.1.0 (build 36398880) (public-release)`
- installed image NCU still present:
  `/usr/local/cuda/bin/ncu` and `nsight-compute-2026.1.1 2026.1.1.2-1`
- `NVIDIA_DRIVER_CAPABILITIES=all`
- `/proc/driver/nvidia/params`: `RmProfilingAdminOnly: 0`

Clean host-driver LD mode used for the first NCU attempt:

```text
/usr/lib/x86_64-linux-gnu:
/usr/local/cuda/lib64:
/usr/local/cuda/targets/x86_64-linux/lib:
/opt/nvidia/nsight-compute-2025.3.1-deb/opt/nvidia/nsight-compute/2025.3.1/host/linux-desktop-glibc_2_11_3-x64:
/opt/nvidia/nsight-compute-2025.3.1-deb/opt/nvidia/nsight-compute/2025.3.1/target/linux-desktop-glibc_2_11_3-x64:
/usr/local/lib/python3.13/dist-packages/z3/lib
```

That path intentionally excludes `/usr/local/cuda/compat` and 2026.1 Nsight
directories.

## Timing Data

Smoke A/B from `mamba3_stage2_profile_wave7_smoke_20260430_2`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA |
| --- | --- | ---: | ---: | ---: | --- | --- |
| smoke | baseline | 0.0892 | 0.2063 | 0.2321 | no / 0 | no / 0 |
| smoke | stage2 `(bf=1,bb=0)` | 0.0912 | 0.1986 | 0.2346 | yes / 4 | no / 0 |

Smoke correctness vs baseline:

- `max_main_grad_abs_diff=0.0`
- all tracked outputs/grads, including `qk_dot` and `states`, had
  `max_abs=0.0`

Final productionish stage2-only timing from
`mamba3_stage2_profile_wave7_prod_stage2_bwd_bwd_hostld_20260430_2`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA |
| --- | --- | ---: | ---: | ---: | --- | --- |
| productionish | stage2 `(bf=1,bb=0)` | 1.8107 | 3.7190 | 5.5032 | yes / 5 | no / 0 |

## NCU Result

NCU did not work. No `LaunchStats`, `Occupancy`, or `MemoryWorkloadAnalysis`
counter rows were captured.

Final productionish attempts:

| attempt | LD mode | result |
| --- | --- | --- |
| `launch_occupancy_memory_host_driver_ld` | clean host-driver, no CUDA compat | failed, rc 9 |
| `launch_occupancy_memory_ldfix` | compatibility LD fix | failed, rc 9 |
| `launchstats_ldfix` | compatibility LD fix | failed, rc 9 |
| `launchstats_default_env` | default env | failed, rc 9 |
| `basic_ldfix` | compatibility LD fix | failed, rc 9 |
| `speed_of_light_ldfix` | compatibility LD fix | failed, rc 9 |
| `launch_metrics_ldfix` | compatibility LD fix | failed, rc 9 |

Representative first-attempt output:

```text
==WARNING== Note: Running with unmodified GPU clocks. If not controlled otherwise, profiling results may be inconsistent.
==PROF== Connected to process 162 (/usr/bin/python3.13)

==ERROR== An error was reported by the counter measurement library:
==ERROR== Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
==PROF== Trying to shutdown target application
==ERROR== The application returned an error code (9).
```

## Conclusion

Wave7 solved the binary-selection part: Modal can download and run NVIDIA
Nsight Compute 2025.3.1, and the harness uses it instead of the CUDA 13.2
bundled 2026.1 binary.

It did not make counter profiling work. The exact blocker is now narrower:
`LibraryNotLoaded` is returned by NVIDIA's counter measurement library even with
Nsight Compute `2025.3.1.0`, Modal H200 driver `580.95.05`, profiler access
apparently enabled (`RmProfilingAdminOnly: 0`), `NVIDIA_DRIVER_CAPABILITIES=all`,
and a clean host-driver LD path that excludes CUDA 13.2 compatibility stubs.

This is therefore not just the original 2026.1/R580 version mismatch. The likely
remaining blocker is Modal's host/container driver-library exposure for NCU
counter collection on H200, or a provider-side counter profiling restriction
that surfaces as `LibraryNotLoaded`.

## Next Steps

1. For Waves 8-10, do not spend kernel time waiting for Modal NCU counters on
   this image/host pair. Use torch profiler/CUDA events on Modal, or move NCU
   counter collection to a non-Modal H200/H100 host.
2. Ask Modal for an H200 image/host combination where Nsight Compute counter
   collection is supported, explicitly citing:
   - NCU `2025.3.1.0`
   - driver `580.95.05`
   - `RmProfilingAdminOnly: 0`
   - `NVIDIA_DRIVER_CAPABILITIES=all`
   - clean host-driver LD path
   - repeated `LibraryNotLoaded`
3. Alternative provider path: use a host with driver `>=595.58.03` and the
   existing CUDA 13.2 / Nsight Compute 2026.1 stack.
4. Keep this Wave7 harness for future checks; if Modal changes host driver or
   container injection, rerun the final smoke command first and require actual
   `LaunchStats+Occupancy+MemoryWorkloadAnalysis` rows before spending a
   productionish run.

## Modal App State

After the final run, `modal app list --json` showed all Wave7 apps stopped with
`Tasks=0`. An older unrelated deployed `cppmega-prebuilt-smoke` app still had
`Tasks=0`; it was not modified.
