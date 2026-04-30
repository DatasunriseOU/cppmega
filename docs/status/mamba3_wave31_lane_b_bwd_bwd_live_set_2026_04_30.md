# Mamba3 Wave31 Lane B bwd_bwd live-set - 2026-04-30

Branch: `worker/mamba3-wave31-bwd-bwd-live-set`

Scope: production TileLang `mamba_mimo_bwd_bwd` redesign probe, H100 component
only. No H200 and no GCloud.

## Candidate

The candidate replaces the long-lived full
`dqk_from_diag_shared [fused_chunk_size, fused_chunk_size]` tile with a compact
same-step diagonal cache:

- old: `[chunk_size * R, chunk_size * R]` fp32 shared, 64 x 64 at R=4
- new: `[chunk_size, R, R]` fp32 shared, 16 x 4 x 4 at R=4

It computes only the per-step R x R qk-dot diagonal consumer used by
`DGAMMA_DIAG`, DK, and DQ. This avoids materializing cross-step blocks that this
consumer path does not use.

Earlier local-fragment variants were rejected by TileLang layout inference:

- full local recompute: `Cannot prove divisible for 8 and 16`
- fused-index local recompute: `Cannot prove divisible for 4 and 8`
- serial-cs local fragment: `Loop layout is not injective`
- 3D local fragment: `no available layout found`

## Commands

```text
python -m py_compile \
  cppmega/megatron/upstream_patches/apply_mamba3_bwd_bwd_live_set_patches.py \
  scripts/modal_mamba3_wave29_lane_c_h100.py \
  tests/test_mamba3_stage2_force_nontma_applier.py

pytest -q tests/test_mamba3_stage2_force_nontma_applier.py

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave31_lane_b_h100_20260430_shared_diag_1

modal run scripts/modal_mamba3_wave29_lane_c_h100.py \
  --run-id wave31_lane_b_h100_20260430_shared_diag_profile_1 \
  --profile-nvtx --cuda-profile --profile-target wave31_late_dqk_recompute
```

## H100 Results

GPU: `NVIDIA H100 80GB HBM3`, capability `(9, 0)`.

Artifacts:

- `/benchmarks/mamba3_wave31_lane_b_h100/wave31_lane_b_h100_20260430_shared_diag_1/report.json`
- `/benchmarks/mamba3_wave31_lane_b_h100/wave31_lane_b_h100_20260430_shared_diag_profile_1/report.json`

Normal run:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | peak alloc MiB | max abs vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke B1 S128 H4 | baseline | 0.0365 | 0.0805 | 0.4762 | 1.21 | - |
| smoke B1 S128 H4 | stage2 current | 0.0373 | 0.0806 | 0.4634 | 1.21 | 0.0 |
| smoke B1 S128 H4 | wave31 shared diag | 0.0371 | 0.0952 | 0.4697 | 1.21 | 1.82e-6 |
| rep B2 S512 H8 | baseline | 0.1308 | 0.3094 | 0.6222 | 33.04 | - |
| rep B2 S512 H8 | stage2 current | 0.1336 | 0.3061 | 1.4480 | 33.04 | 0.0 |
| rep B2 S512 H8 | wave31 shared diag | 0.1335 | 0.3706 | 1.6590 | 33.04 | 2.57e-6 |

Profiler-hook run:

| shape | variant | bwd_fwd ms | bwd_bwd ms | chain ms | NVTX/CUDA profiler |
| --- | --- | ---: | ---: | ---: | --- |
| smoke B1 S128 H4 | baseline | 0.0369 | 0.0809 | 0.5020 | off |
| smoke B1 S128 H4 | stage2 current | 0.0376 | 0.0814 | 0.4746 | off |
| smoke B1 S128 H4 | wave31 shared diag | 0.0380 | 0.0963 | 0.4779 | on, start/stop OK |
| rep B2 S512 H8 | baseline | 0.1339 | 0.3138 | 0.6211 | off |
| rep B2 S512 H8 | stage2 current | 0.1345 | 0.3094 | 0.7131 | off |
| rep B2 S512 H8 | wave31 shared diag | 0.1349 | 0.3737 | 0.7010 | on, start/stop OK |

Correctness:

- Stage2 remains bitwise equal to baseline on all non-`None` outputs.
- Wave31 shared-diag max abs diff vs stage2/baseline is `2.57e-6` on the
  representative shape, from changed accumulation order in the diagonal
  microkernel.

Memory:

- Peak allocated memory unchanged on both shapes.
- Representative peak reserved memory matches stage2 (`42 MiB`) and remains
  higher than baseline (`38 MiB`) due allocator reservation.

## Judgment

Safe for main:

- No production kernel change. The redesign compiles and is correct, but
  representative `bwd_bwd` regresses from `0.306 ms` to `0.371 ms`
  (`+21%`) versus stage2 current.
- The only reusable pieces are the default-off applier and harness/doc evidence.
  They should stay on this worker branch unless we want a catalog of rejected
  candidates in main.

Conclusion: full-tile `dqk_from_diag_shared` is wasteful in theory, but the
compact scalar R x R diagonal microkernel loses to the original full GEMM/shared
path on H100. Do not ship this candidate.
