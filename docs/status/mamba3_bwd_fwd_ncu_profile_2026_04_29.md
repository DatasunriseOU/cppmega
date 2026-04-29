# Mamba3 bwd_fwd NCU Profile Attempt - 2026-04-29

Branch: `worker/mamba3-ncu-bwd-fwd`

Base: `972608d perf(mamba3): keep stage2 ws to bwd fwd`

Scope: bounded Modal Nsight Compute profiling for Mamba3 MIMO `bwd_fwd`
baseline vs stage2 default `(bf_num_stages=1, bb_num_stages=0)` on H200:2.
No kernel patch was changed.

## Harness

Script:

- `scripts/modal_mamba3_bwd_fwd_ncu_profile.py`

The harness reuses the existing stage2 benchmark helpers, compiles only the
selected variant's `bwd_fwd` kernel, warms it once, then enables a CUDA profiler
range around one `bwd_fwd` launch. It runs `ncu` with:

```text
--target-processes all
--profile-from-start off
--launch-count 1
--section SpeedOfLight
--section Occupancy
--section MemoryWorkloadAnalysis
```

Productionish shape:

```text
B=4, S=4096, H=32, G=1, N=64, P=128, R=4, dtype=bf16
```

## Runs

### Image default NCU 2026.1.1

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
modal run scripts/modal_mamba3_bwd_fwd_ncu_profile.py \
  --run-id mamba3_bwd_fwd_ncu_h200_20260429_1 \
  --shape productionish \
  --variant-csv baseline,stage2_force_nontma \
  --launches 1 \
  --ncu-timeout-sec 720
```

Modal app: `ap-IEaj0Zh4bQxsaPoygYdLGR`

Artifacts:

- `/benchmarks/mamba3_bwd_fwd_ncu_profile/mamba3_bwd_fwd_ncu_h200_20260429_1/report.json`
- per-variant `ncu_stdout.txt`, `ncu_stderr.txt`, `ncu_command.txt`

Result:

- `/usr/local/cuda/bin/ncu --version`: Nsight Compute `2026.1.1.0`
- GPU: `NVIDIA H200`, driver `580.95.05`, CUDA `13.2`
- both variants returned `9`
- NCU stdout:

```text
Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
```

### Image default NCU 2026.1.1 with diagnostics

```text
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 1800s \
modal run scripts/modal_mamba3_bwd_fwd_ncu_profile.py \
  --run-id mamba3_bwd_fwd_ncu_h200_20260429_2 \
  --shape productionish \
  --variant-csv baseline,stage2_force_nontma \
  --launches 1 \
  --ncu-timeout-sec 720
```

Modal app: `ap-ecgnjD6gqGnLShl1m9RbIX`

Artifacts:

- `/benchmarks/mamba3_bwd_fwd_ncu_profile/mamba3_bwd_fwd_ncu_h200_20260429_2/report.json`
- per-variant `nsight-compute-*.log`

Result:

- explicit `LD_LIBRARY_PATH` included `/usr/local/cuda/compat` and
  `/usr/lib/x86_64-linux-gnu`
- both variants returned `9`
- internal Nsight log:

```text
Failed to initialize LOP (error = 1)
Failed to initialize LOP target
Profiler failed to initialize (error 15)
```

NVIDIA's public Nsight Compute 2026.1 system requirements list recommended
Linux driver `595.58.03` or newer; Modal H200 exposed `580.95.05`.

### Installed NCU 2025.3.1

To separate a 2026.1 vs driver-version mismatch from a Modal profiler access
issue, the harness can install a specific NCU package into the image:

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
CPPMEGA_NCU_DEB_URL=https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb \
CPPMEGA_NCU_BIN=/opt/nvidia/nsight-compute/2025.3.1/ncu \
timeout 2400s \
modal run scripts/modal_mamba3_bwd_fwd_ncu_profile.py \
  --run-id mamba3_bwd_fwd_ncu_h200_20260429_3_ncu2025 \
  --shape productionish \
  --variant-csv baseline,stage2_force_nontma \
  --launches 1 \
  --ncu-timeout-sec 900
```

Modal app: `ap-goSt3pwQ3L32vjhaQGYkLe`

Artifacts:

- `/benchmarks/mamba3_bwd_fwd_ncu_profile/mamba3_bwd_fwd_ncu_h200_20260429_3_ncu2025/report.json`
- per-variant `ncu_stdout.txt`, `ncu_stderr.txt`, `ncu_command.txt`

Result:

- `/opt/nvidia/nsight-compute/2025.3.1/ncu --version`: Nsight Compute
  `2025.3.1.0`
- NVIDIA's public Nsight Compute 2025.3 system requirements list recommended
  Linux driver `580.82.07` or newer; Modal H200 has `580.95.05`
- both variants still returned `9`
- stdout still reported `LibraryNotLoaded`

This makes a pure NCU 2026.1 driver-version mismatch unlikely to be the only
blocker. The Modal H200 container can run CUDA workloads, but NCU counter
collection cannot initialize its profiler target in this environment.

## Requested Metrics

No NCU hardware-counter metrics were emitted for either variant. The requested
SpeedOfLight, Occupancy, MemoryWorkloadAnalysis, TMA/global-copy, register, and
shared-memory counters are therefore unavailable from this Modal environment.

Available non-NCU facts from the prior accepted benchmark on the same base:

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA |
| --- | ---: | ---: | ---: | --- |
| baseline | 1.8718 | 3.7084 | 5.5628 | no / 0 TMA loads in source |
| stage2 default `(1,0)` | 1.7886 | 3.6940 | 5.4567 | yes / 5 TMA loads in source |

`bwd_fwd` speedup from that run: `1.8718 / 1.7886 = 1.0465x`
(`4.65%` faster).

## Conclusion

NCU does not currently support or refute the `bwd_fwd` hypothesis on Modal H200,
because counter collection fails before the profiled launch. The timing/source
evidence still supports the hypothesis:

- `stage2_force_nontma` keeps WS/TMA only for `bwd_fwd`;
- `bwd_fwd` is about `4.65%` faster in the accepted H200 timing run;
- `bwd_bwd` stays on the non-WS path and avoids the previously observed WS/TMA
  regression.

The next actionable profiling step is to run this harness in an environment
where Nsight Compute can initialize PerfWorks/LOP counters for H200, or ask
Modal for a profiling-enabled image/driver combination.

## Modal App State

Checked after the runs with:

```text
modal app list --json
```

The apps created by this task were stopped with `Tasks=0`:

- `ap-IEaj0Zh4bQxsaPoygYdLGR`
- `ap-ecgnjD6gqGnLShl1m9RbIX`
- `ap-goSt3pwQ3L32vjhaQGYkLe`

Other live apps were present and were not touched because they belong to other
parallel work:

- `ap-2cPV42MwXGtToPGeY6IUJ2` (`cppmega-mamba3-bwd-bwd-pressure-reduce-benchmark`), `Tasks=1`
- `ap-gAj8TOU7vvvs02fETlmd0z` (`cppmega-mamba3-psiv-hoist-probe`), `Tasks=0`
- `ap-KxVIsieQYhqIll6K2zl5v3` (`cppmega-prebuilt-smoke`), deployed, `Tasks=0`

References:

- NVIDIA Nsight Compute 2026.1 getting started/system requirements:
  https://developer.nvidia.com/tools-overview/nsight-compute/get-started
- NVIDIA Nsight Compute 2025.3 getting started/system requirements:
  https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3
