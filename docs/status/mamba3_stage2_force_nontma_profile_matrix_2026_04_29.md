# Mamba3 Stage2 Force Non-TMA Profile Matrix - 2026-04-29

Branch: `worker/mamba3-stage2-force-nontma`

Goal: answer whether the smem-safe `(1,1)` stage2 force-nonTMA path is actually
useful, and identify which phase benefits from WS/TMA.

## Run

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
timeout 1200s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_force_nontma_h200_profile_matrix_20260429_1 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_bf1_bb0,stage2_bf0_bb1,stage2_force_nontma \
  --torch-profile \
  --warmup 1 \
  --iters 4
```

Modal app:

- `ap-jZNFhIcoPJPwValLet94Ua`
- state after run: stopped, `Tasks=0`

Artifacts in Modal Volume `cppmega-mamba3-benchmarks`:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_profile_matrix_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_profile_matrix_20260429_1/summary.csv`
- per-variant torch profiler tables/traces under the `productionish/<variant>/`
  directories.

Device:

- GPU: `NVIDIA H200`
- device count: `2`
- capability: `(9, 0)`
- image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

## CUDA Event Matrix

Shape: `productionish` (`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, bf16`)

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | bwd_fwd WS/TMA | bwd_bwd WS/TMA | chain speedup |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| baseline | 1.8740 | 3.7103 | 5.5525 | no / 0 | no / 0 | 1.0000x |
| `stage2_bf1_bb0` | 1.8063 | 3.7097 | 5.4667 | yes / 5 | no / 0 | 1.0157x |
| `stage2_bf0_bb1` | 1.9665 | 3.9734 | 5.9092 | no / 0 | yes / 7 | 0.9396x |
| `stage2_force_nontma` old `(1,1)` | 1.7919 | 3.9727 | 5.7331 | yes / 5 | yes / 7 | 0.9685x |

Correctness:

- `stage2_bf1_bb0` and `stage2_bf0_bb1`: `max_main_grad_abs_diff=0.0`
- old `(1,1)`: `max_main_grad_abs_diff=9.747900264756026e-10`
- `qk_dot` and `states` max diff: `0.0` for all variants

## Torch Profiler Check

The torch profiler tables agree with CUDA-event timing.

Per-call CUDA averages from 3 profiler steps:

| variant | bwd_fwd profiler ms | bwd_bwd profiler ms | read |
| --- | ---: | ---: | --- |
| baseline | 1.841 | 3.685 | reference |
| `stage2_bf1_bb0` | 1.767 | 3.667 | best |
| `stage2_bf0_bb1` | 1.945 | 3.949 | bwd_bwd WS/TMA hurts |
| old `(1,1)` | 1.794 | 4.001 | bwd_fwd win is eaten by bwd_bwd loss |

## Read

The useful optimization is asymmetric:

- keep `bwd_fwd` on the flattened Q/K + TMA/WS path (`bf_num_stages=1`);
- keep `bwd_bwd` on the non-WS path (`bb_num_stages=0`).

`bwd_bwd` still dominates chain time, but enabling WS/TMA there is a regression
on H200 productionish. The production candidate default was changed to
`bf_num_stages=1`, `bb_num_stages=0` after this run.

Next useful work:

1. Run a longer H200 sample for `baseline` vs the new default `(1,0)` only.
2. Run Nsight Compute on `bwd_fwd` baseline vs `(1,0)` to verify whether the win
   is memory-transfer/TMA efficiency or scheduler/occupancy.
3. For `bwd_bwd`, stop pursuing WS/TMA until register/shared-memory pressure is
   reduced. The likely next kernel work is smaller live-set / PsiV hoist, not
   another `num_stages` sweep.

## Default Confirmation

After changing `stage2_force_nontma` to mean `(bf_num_stages=1,
bb_num_stages=0)`, a longer non-profiler H200 run confirmed the result:

```text
GHCR_TAG=785c3fd \
CPPMEGA_MODAL_GPU=H200:2 \
timeout 900s \
modal run scripts/modal_mamba3_stage2_force_nontma_benchmark.py \
  --run-id stage2_force_nontma_h200_default_bf1bb0_20260429_1 \
  --shape-csv productionish \
  --variant-csv baseline,stage2_force_nontma \
  --warmup 2 \
  --iters 12
```

App:

- `ap-15k9yZgZOOZzUzaz0eXy9F`
- state after run: stopped

Artifacts:

- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_default_bf1bb0_20260429_1/report.json`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_default_bf1bb0_20260429_1/summary.csv`
- `/benchmarks/mamba3_stage2_force_nontma_benchmark/stage2_force_nontma_h200_default_bf1bb0_20260429_1/summary.json`

| variant | bwd_fwd ms | bwd_bwd ms | chain ms | WS/TMA | speedup |
| --- | ---: | ---: | ---: | --- | ---: |
| baseline | 1.8718 | 3.7084 | 5.5628 | no/no | 1.0000x |
| new default `(1,0)` | 1.7886 | 3.6940 | 5.4567 | bwd_fwd only | 1.0194x |

Correctness:

- `max_main_grad_abs_diff=0.0`
- all tracked diffs, including `qk_dot` and `states`, had `max_abs=0.0`
