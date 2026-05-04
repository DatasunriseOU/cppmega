# Mamba3 Wave32 Lane D: Gate Stability and bwd_bwd Redesign

Date: 2026-04-30

Branch: `worker/mamba3-wave32-research-gate-stability`

Scope: web/code research plus low-risk harness changes only. No H200 jobs were
run. No production kernel semantics were changed.

## Implemented

1. `scripts/modal_mamba3_wave31_g8_reachability.py`
   - Added `mla_auto_full` as the default full-boundary gate case. It keeps
     `--use-flash-attn` present but requests `--attention-backend auto`, letting
     TE select the only viable MLA backend instead of forcing flash.
   - Reclassified `te_flash_full` as a negative control, not a production
     throughput case.
   - Added parser fields for TE available/selected backend and whether
     FlashAttention was rejected because of MLA.
   - Added requested/selected attention backend columns to `summary.md`.

2. `tests/test_mamba3_wave31_gate_stability.py`
   - Covers that the full gate defaults to `mla_auto_full`.
   - Covers parsing of TE backend debug lines.

## Findings

### TE MLA backend policy

The Wave30/Wave31 logs and TE docs point to the same rule: for MLA on Hopper,
forcing `attention_backend=flash` is not a stable production gate. TE debug
reported FlashAttention 2 rejected MLA, while `auto` selected FusedAttention
sub-backend 1.

Local anchors:

- `scripts/modal_mamba3_wave31_g8_reachability.py:46` defines the stable
  `mla_auto_full` case and keeps forced flash as a negative control.
- `scripts/modal_mamba3_wave31_g8_reachability.py:574`,
  `scripts/modal_mamba3_wave31_g8_reachability.py:917`, and
  `scripts/modal_mamba3_wave31_g8_reachability.py:1130` now default to
  `mla_auto_full`.
- `docs/status/mamba3_wave30_modal_h200_attn_debug_2026_04_30.md:73`
  records the earlier forced-flash failure mode.
- `docs/status/mamba3_wave31_g8_reachability_2026_04_30.md:73` records the
  successful `auto -> FusedAttention` reachability path.

### "GQA" naming vs MLA

The grouped-head patch is not replacing MLA attention. It is the Mamba3 MIMO
backward reduction for `1 < G < H` after the TileLang kernels emit
`dq/dk [B,S,R,H,N]` and Python reduces them back to grouped Q/K shapes.

Local anchors:

- `cppmega/megatron/upstream_patches/apply_mamba3_grouped_head_bwd_patches.py:1`
  describes the patch as grouped-head Mamba3 backward support.
- `cppmega/megatron/upstream_patches/apply_mamba3_grouped_head_bwd_patches.py:54`
  inserts the `H % G == 0` reduction path.
- `cppmega/megatron/upstream_patches/apply_mamba3_grouped_head_bwd_patches.py:124`
  validates that both `dq` and `dk` reductions exist.

### TileLang WS/TMA for bwd_bwd

Current stage2 is correctly asymmetric: keep the useful 2D flattening/TMA
eligibility where it helps, keep `bwd_bwd` at `bb_num_stages=0`, and do not
turn every tiny float32 vector slice into a TMA candidate.

Local anchors:

- `cppmega/megatron/upstream_patches/apply_mamba3_stage2_force_nontma_patches.py:19`
  documents the `bf=1,bb=0` production-control default.
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch:7`
  removes global TMA-disable and enables WS in the patch.
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch:52`
  force-disables TMA for small scalar-vector slices.
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_stage2_force_nontma.patch:153`
  flattens `QK_DOT` for the `bwd_bwd` path.
- `scripts/modal_mamba3_wave29_lane_c_h100.py:227` still instantiates
  `bwd_bwd` with `num_stages=0`.
- `scripts/modal_mamba3_wave29_lane_c_h100.py:281` exposes targeted NVTX/CUDA
  profiler windows per split kernel.

## Ranked Patch List

1. Run the real 20-step H200 gate only through `mla_auto_full`.
   - Command target: `scripts/modal_mamba3_wave31_g8_reachability.py::launch_gate`.
   - Required case: `--case-label mla_auto_full`.
   - Required checks: real dataset present, `grad norm` finite, `lm loss` finite,
     baseline and stage2 both complete 20 steps, peak memory parsed, selected
     backend captured as FusedAttention or another TE-supported MLA backend.

2. Keep grouped-head backward enabled only as a guarded applier until a real-data
   20-step gate passes.
   - This removes the source blocker for NAM56R `G=8`, but the synthetic Wave31
     run had `grad norm: nan`, so it is reachability evidence only.
   - Rename user-facing docs from `GQA` to `grouped-head Mamba3 backward` where
     possible to avoid confusing it with MLA attention.

3. Do not push more WS/TMA into `bwd_bwd` small slices.
   - Use the H100 profiler hooks in `scripts/modal_mamba3_wave29_lane_c_h100.py`
     to compare `bf=1,bb=0` against baseline, then use H200 only for the
     full-boundary gate.
   - Treat `WS skipped: no TMA copies in pipeline loop` as an expected signal
     when the loop has no large producer copy to specialize.

4. Larger `bwd_bwd` redesign should target ownership/layout, not local scalar
   cleanup.
   - Fuse the consumers around `dPsiV_D`, `dqk_from_diag`, `dk_intrachunk`,
     `dk_frag`, `dq_frag`, `DDA_CS`, `DDA_CS_REV`, `DSSDA`, `DGAMMA_DIAG`, and
     `DANGLES`.
   - Reference math anchors:
     `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py:178`,
     `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py:210`,
     `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py:223`,
     `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py:247`, and
     `cppmega/megatron/cute_dsl_mimo/full_bwd_bwd_epilogue.py:336`.
   - Gate for accepting a redesign: it must beat the current `bwd_bwd` by more
     than the extra helper/global roundtrip cost and preserve gradient parity.

5. Modal artifacts/logging.
   - Continue writing logs/results to the mounted volume after each variant and
     calling `results_vol.commit()`.
   - Add `modal app logs --tail ...` capture only in the local launcher layer,
     not inside the GPU function, so the container does not keep running just to
     collect logs.
   - After every Modal run, stop the app explicitly and verify `modal app list
     --json` shows no live app for the lane.

## Sources

- NVIDIA Transformer Engine environment variables:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/envvars.html
- NVIDIA Transformer Engine attention backend support matrix:
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html
- TileLang `T.copy(..., disable_tma=...)` data movement docs:
  https://tilelang.com/programming_guides/instructions.html
- TileLang pass config docs for WS/TMA flags and ptxas controls:
  https://tilelang.com/autoapi/tilelang/transform/pass_config/index.html
- TileLang builtins for TMA/register control:
  https://tilelang.com/autoapi/tilelang/language/builtin/index.html
- PyTorch/Triton warp specialization design notes:
  https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
- Modal Volumes docs:
  https://modal.com/docs/guide/volumes
- Modal app logs/stop docs:
  https://modal.com/docs/reference/cli/app
- Modal FunctionCall docs:
  https://modal.com/docs/reference/modal.FunctionCall
- Upstream Mamba3 TileLang source:
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py

## Safe-for-main Judgment

Safe for main: yes, for the harness default, summary parser, tests, and this
research note.

Not safe for default production: any new `bwd_bwd` kernel redesign or making the
grouped-head source mutation unconditional. Those still require H100 smoke plus
real-data H200 full-boundary 20-step memory/tok-sec/grad-parity evidence.
