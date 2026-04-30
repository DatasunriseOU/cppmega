# Mamba3 Wave30 Lane C H100 - 2026-04-30

Branch: `worker/mamba3-wave29-tilelang-prod-tune`

Scope: production-path instrumentation only. No TileLang production patch
semantics changed; the guarded candidate remains `bf_num_stages=1,
bb_num_stages=0`.

## Changes

- Extended `scripts/modal_mamba3_wave29_lane_c_h100.py` with targeted
  split-kernel profiling controls for stage2 `bwd_fwd` and `bwd_bwd`.
- NVTX labels now include variant, shape, stage, phase, and iteration, e.g.
  `mamba3_stage2:stage2_current:representative:bwd_bwd:bench:0`.
- Added optional CUDA profiler API windows around each targeted stage bench
  loop for NCU/profile-from-start workflows.
- Profile targeting defaults to `stage2_current`; baseline timings remain
  uninstrumented unless `--profile-target '*'` is requested.

## Commands

```text
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py \
  scripts/modal_mamba3_wave29_lane_c_h100.py \
  tests/test_mamba3_stage2_force_nontma_applier.py

pytest -q tests/test_mamba3_stage2_force_nontma_applier.py

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave30_lane_c_h100_20260430_hooks_1

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave30_lane_c_h100_20260430_hooks_profile_1 \
  --profile-nvtx --cuda-profile --profile-target stage2_current
```

H100 only. No H200 or H200-mini runs were used.

Modal app name: `cppmega-wave29-lane-c-h100`

Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`

GPU: `NVIDIA H100 80GB HBM3`, capability `(9, 0)`.

Artifacts:

- `/benchmarks/mamba3_wave29_lane_c_h100/wave30_lane_c_h100_20260430_hooks_1/report.json`
- `/benchmarks/mamba3_wave29_lane_c_h100/wave30_lane_c_h100_20260430_hooks_profile_1/report.json`

## H100 Correctness/Perf/Memory

Run `wave30_lane_c_h100_20260430_hooks_1`:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak alloc MiB delta | peak reserved MiB delta | max diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0368 | 0.0824 | 0.6342 | 1.21 | 0.00 | 0.0 |
| smoke B1 S128 H4 | stage2 current | 0.0377 | 0.0819 | 0.5359 | 1.21 | 0.00 | 0.0 |
| rep B2 S512 H8 | baseline | 0.1335 | 0.3173 | 0.6971 | 33.04 | 38.00 | 0.0 |
| rep B2 S512 H8 | stage2 current | 0.1372 | 0.3155 | 0.6933 | 33.04 | 42.00 | 0.0 |

Correctness:

- `stage2_current_vs_baseline`: every non-`None` output had `max_abs=0.0`.

Memory:

- Peak allocated memory was unchanged by stage2 current on both shapes.
- Representative peak reserved memory stayed 4 MiB higher for stage2 current,
  matching Wave29 behavior and reflecting allocator reservation rather than
  live allocated tensor memory.

Profiler hook verification:

- `wave30_lane_c_h100_20260430_hooks_profile_1` reported
  `nvtx_enabled=true` and `cuda_profile_enabled=true` only for
  `stage2_current`.
- CUDA profiler start/stop events succeeded for both `bwd_fwd` and `bwd_bwd`
  on smoke and representative shapes.
- Baseline remained `profile_targeted=false`, `nvtx_enabled=false`,
  `cuda_profile_enabled=false`.

## Judgment

Safe for main:

- Yes: targeted H100 harness profiling hooks and this status note.
- No: production TileLang patch semantic changes. H100 evidence still does not
  show a clear kernel win; `bwd_bwd` is flat/slightly faster in this repeat,
  `bwd_fwd` is slower on the representative shape, and chain timing remains
  noisy.
