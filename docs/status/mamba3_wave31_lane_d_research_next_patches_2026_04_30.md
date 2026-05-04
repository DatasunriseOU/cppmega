# Mamba3 Wave31 Lane D: research and next-patch plan

Scope: research TileLang/Mamba3/TE blockers after Wave30, implement only
low-risk harness or applier guard helpers, and leave risky kernel rewrites to
the dedicated A/B/C lanes.

## Implemented

- Added `cppmega.megatron.upstream_patches.apply_mamba3_gqa_bwd_patches`.
  It is default-off and mutates installed `mamba_ssm` only with both:
  `CPPMEGA_MAMBA3_GQA_BWD=1` and
  `MAMBA3_GQA_BWD_ALLOW_FILE_MUTATION=1`.
- Added rollback via `CPPMEGA_MAMBA3_GQA_BWD_ROLLBACK=1`.
- Added tests for patched, absent, rollback, and partial-marker states.
- Updated `scripts/modal_mamba3_wave30_h200_attn_debug.py` to record
  `gqa_bwd` in kernel status and apply this compatibility patch before the
  baseline/stage2 H200 debug sweep. This applies equally to both variants and
  is not a performance candidate.

## Local and external evidence

- Upstream `state-spaces/mamba` main still raises for intermediate G:
  `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py:1299-1314`.
  Source:
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
- Our local upstream patch pack already documents and implements the missing
  `1 < G < H` branch:
  `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.patch` and
  `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.md`.
- Wave30 H200 attention debug reached Mamba backward after TE fallback and
  then failed on `G value of 8 is not currently supported!`:
  `docs/status/mamba3_wave30_modal_h200_attn_debug_2026_04_30.md`.
- Transformer Engine docs list the backend controls used in the Wave30
  fallback matrix: `NVTE_FLASH_ATTN`, `NVTE_FUSED_ATTN`,
  `NVTE_UNFUSED_ATTN`, and `NVTE_FUSED_ATTN_BACKEND`.
  Source:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/envvars.html
- Transformer Engine attention docs describe backend selection by model shape,
  software versions, architecture, mask/bias/layout, and performance, so the
  `attention_backend=auto`/FusedAttention fallback is the correct debug path
  for MLA when FlashAttention is rejected.
  Source:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
- TileLang docs confirm `T.copy(..., disable_tma=False)` can lower to TMA or
  other copy mechanisms while keeping synchronous semantics, and
  `T.async_copy` is strict and fails instead of silently falling back.
  Source:
  https://tilelang.com/programming_guides/instructions.html
- TileLang transform docs expose `WarpSpecializedPipeline`,
  `InjectTmaBarrier`, `FlattenBuffer`, `MergeSharedMemoryAllocations`, and
  `AnnotateWarpGroupRegAlloc`; this matches the observed bwd blocker class:
  layout/liveness/register pressure, not a simple `num_stages` knob.
  Source:
  https://tilelang.com/autoapi/tilelang/transform/index.html
- TileLang builtins include TMA descriptors, TMA load/store, max-register
  annotations, and explicit ldg/stg intrinsics. These are useful only if a
  lane leaves high-level TileLang and owns the schedule/layout explicitly.
  Source:
  https://tilelang.com/autoapi/tilelang/language/builtin/index.html
- PyTorch/Triton warp-specialization notes frame WS as compiler partitioning
  of data/compute/epilogue paths plus memory planning; it pays off when there
  is enough loop body work to overlap and when live ranges are controlled.
  Source:
  https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- The open Mamba issue about Blackwell Mamba3 backward shows the same class
  of failure mode: compiler/resource pressure can eliminate good configs or
  cause extreme spilling, so B200/B300 work should start from register/TMEM
  diagnostics before porting H200 choices blindly.
  Source:
  https://github.com/state-spaces/mamba/issues/904

## Ranked next-patch ideas

1. Lane A, immediate unblock: use the new GQA applier in the Modal H200
   fallback case, rerun `fallback_auto_full` for one step, then the 20-step
   baseline-vs-stage2 gate only after step 1 completes. Expected payoff: high,
   because current H200 gate is blocked before any tok/sec comparison.

2. Lane C, H100/local only: keep `bf=1,bb=0` as the only stage2 candidate and
   use the Wave30 NVTX/CUDA profiler windows on `stage2_current:bwd_fwd` and
   `stage2_current:bwd_bwd`. Expected payoff: medium; it can tell whether the
   `bwd_fwd` TMA path is real on H100/H200 without mutating `bwd_bwd`.

3. Lane B, R&D: port the GQA branch contract into the CuTe prototype before
   deeper fusion. Expected payoff: medium-low for production today, but it
   prevents future R&D kernels from hiding the same grouped-head reduction
   mismatch.

4. Larger redesign lane: focus on bwd_bwd live-set ownership, not another
   global PsiV helper. Based on Wave29/Wave30 data, helper boundaries lose
   unless they remove more than the write/read cost. The credible design is a
   monolithic or split-with-on-chip-reuse kernel that consumes `QK_DOT`,
   `STATES`, `Psi/Phi`, D/DDA consumers, and DK/DQ reductions under one
   ownership model.

5. B200/B300 lane: treat Blackwell separately. First collect ptxas/TMEM/spill
   diagnostics on the smallest representative Mamba3 backward kernels, because
   upstream SISO reports show Blackwell can pick radically worse configs even
   when H200 is healthy.

## Safe-for-main judgment

Safe for main: yes, for the default-off applier, tests, docs, and Wave30 debug
harness instrumentation. Not safe for default-on production: any new
stage2/TMA semantic kernel change or bwd_bwd redesign still needs H100 smoke,
H200 full-boundary 20-step tok/sec/memory, and gradient parity.
