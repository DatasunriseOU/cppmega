# Session And Probe Notes

This index keeps dated docs discoverable without moving the existing top-level
files. Treat these notes as evidence from a specific date or worktree; cite the
canonical status doc for current behavior when one exists.

See [../status/README.md](../status/README.md) for status labels and retention
rules.

## Current Working Set

| Date | Note | Status | Canonical / follow-up |
| --- | --- | --- | --- |
| 2026-04-25 | [../deprecated_path_gates_2026_04_25.md](../deprecated_path_gates_2026_04_25.md) | active | Gate/deprecation behavior; keep near current code changes. |
| 2026-04-25 | [../gb10_dense_mxfp8_status_2026_04_25.md](../gb10_dense_mxfp8_status_2026_04_25.md) | canonical | Current dense GB10 MXFP8/NVFP4 status for this narrow topic. |
| 2026-04-25 | [../sparse_mla_blockscaled_gemm_plan_2026_04_25.md](../sparse_mla_blockscaled_gemm_plan_2026_04_25.md) | active | SparseMLA block-scaled plan; resolve into a status doc when settled. |
| 2026-04-25 | [../sparse_mla_blockscaled_qk_runtime_2026_04_25.md](../sparse_mla_blockscaled_qk_runtime_2026_04_25.md) | evidence | QK-only runtime probe for the SparseMLA plan. |
| 2026-04-25 | [../sparse_mla_blockscaled_fused_backend_2026_04_25.md](../sparse_mla_blockscaled_fused_backend_2026_04_25.md) | active | Fused backend prototype; unsafe backward remains blocked. |
| 2026-04-25 | [../te_mxfp8_backward_gb10_plan_2026_04_25.md](../te_mxfp8_backward_gb10_plan_2026_04_25.md) | active | TE MXFP8 backward plan. |
| 2026-04-25 | [../dense_blockscaled_weight_cache_probe_2026_04_25.md](../dense_blockscaled_weight_cache_probe_2026_04_25.md) | evidence | Supports dense GB10 MXFP8/NVFP4 status. |
| 2026-04-25 | [../deepgemm_gb10_check_2026_04_25.md](../deepgemm_gb10_check_2026_04_25.md) | evidence | External DeepGEMM not a GB10 drop-in path. |
| 2026-04-25 | [../gb10_local_memory_perf_2026_04_25.md](../gb10_local_memory_perf_2026_04_25.md) | evidence | Local GB10 memory/perf session. |
| 2026-04-25 | [../hf_kernel_replacements_2026_04_25.md](../hf_kernel_replacements_2026_04_25.md) | evidence | Kernel replacement scan. |
| 2026-04-25 | [../lion8bit_ab_2026_04_25.md](../lion8bit_ab_2026_04_25.md) | evidence | Lion8bit A/B result. |
| 2026-04-25 | [../memory_dtype_audit_2026_04_25.md](../memory_dtype_audit_2026_04_25.md) | evidence | Memory dtype audit. |

## Earlier Session Notes

| Date | Notes | Status |
| --- | --- | --- |
| 2026-04-15 | [GB10 regression](../gb10_regression_investigation_2026_04_15.md), [grad NaN investigation](../grad_nan_investigation_2026_04_15.md), [grad NaN bisect](../grad_nan_bisect_2026_04_15.md) | evidence |
| 2026-04-14 | [session findings](../findings_2026_04_14_session.md), [FP8 research](../fp8_research_session_2026_04_14.md), [Mamba fork canonical](../mamba_fork_canonical_2026_04_14.md), [session closeout](../session_3_closeout_2026_04_14.md), [gap audit](../session_3_gap_audit.md) | evidence |
| 2026-04-13 | [FP8 optimization session](../fp8_optimization_session_2026_04_13.md), [optimization session](../optimization_session_2026_04_13.md), [optimization session RU](../optimization_session_2026_04_13_ru.md) | evidence |
| 2026-04-12 | [Blackwell feature sweep](../blackwell_feature_sweep_2026_04_12.md), [DSA EP=2 sweep](../dsa_ep2_tilelang_sweep_2026_04_12.md), [NAM56R grid search](../nam56r_grid_search_2026_04_12.md) | evidence |
| 2026-04-11 | [GB10 bwd_bwd conclusion](../gb10_bwd_bwd_optimization_conclusion.md), [Modal B200 sweep](../modal_b200_cutile_variant_sweep_2026_04_11.md), [NAM56R baseline](../nam56r_mimo7_baseline_2026_04_11.md), [nsys profile](../nam56r_mimo7_nsys_profile_2026_04_11.md), [reproducibility](../nam56r_mimo7_reproducibility_2026_04_11.md), [VPP result](../nam56r_mimo7_vpp_112k_2026_04_11.md), [MTP plan](../nam56r_mtp_optimization_plan_2026_04_11.md), [MTP plan RU](../nam56r_mtp_optimization_plan_2026_04_11_ru.md), [session summary RU](../session_2026_04_11_summary_ru.md) | evidence |
| 2026-04-10 | [Modal B200 cuTile status](../modal_b200_cutile_status.md) | evidence |

## Adding A New Note

Prefer `docs/sessions/YYYY-MM-DD-<topic>.md` for new session or probe notes.
If a same-day topic note already exists, append to it instead of creating a
near-duplicate. When the note changes current guidance, update the relevant
canonical status doc and link back here.
