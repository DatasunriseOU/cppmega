# Upstream PR Submission Checklist

**Prepared: 2026-04-14.** All upstream states verified via `gh api` (not MCP).
**Filing gate:** the user must give explicit "post it" approval per PR. This
document is prep only — no PR has been filed yet. Prefer grouped submission
(below) to reduce reviewer fatigue.

Legend:
- `ready = Y` — template + reproducer + explanation_ru present, upstream
  state understood, reproducer has been exercised on a listed host.
- `ready = N` — a blocker exists (reproducer not validated on a CC=9 host,
  upstream duplicate, etc.).
- `ready` is a filing-prep flag only; it does **not** mean a retained on-disk
  receipt exists. Check `upstream_prs/examples/validation_manifest.yaml`
  before describing any pack as receipted or fully validated.

---

## Group A — Safe first wave (solid, no ambiguity)

### PR 01 — DSA CUDA graph safety
- **Target repo:** NVIDIA/Megatron-LM
- **Upstream state:** `new_report` (no matching issue/PR found via `gh api
  search/issues repo:NVIDIA/Megatron-LM dsa cuda graph`).
- **Template:** `upstream_prs/01_dsa_cuda_graph_safety.md`
- **Reproducer:** `upstream_prs/examples/01_dsa_cuda_graph_safety/reproducer.py`
  (last verified bench3, exit 0, BUG_REPRODUCED + FIX_VALIDATED).
- **Explanation (RU):** `upstream_prs/examples/01_dsa_cuda_graph_safety/explanation_ru.md`
- **Ready:** Y
- **File as:** ISSUE first (report + patch link), then PR if maintainer asks.
- **Command:**
  ```
  gh issue create --repo NVIDIA/Megatron-LM \
    --title "DSA: CUDA graph capture sees stale pointer through unfused indexer path" \
    --body-file upstream_prs/01_dsa_cuda_graph_safety.md
  ```
- **Risk — upstream engagement:** medium. `mcore-oncall` is active; DSA
  (PR #3674 open, #4039 open) is clearly a current focus area.
- **Risk — us:** none. Purely defensive change, no API churn.

### PR 12 — DSA `_compute_index_scores` memory
- **Target repo:** NVIDIA/Megatron-LM
- **Upstream state:** `new_report`, but **OVERLAP with PR #4039**
  (`[Kernel] Fused Indexer Loss Kernel` by laixinn, open, updated
  2026-03-27). #4039 solves the same 16 GiB materialisation problem via
  split-K Triton (60% mem save, 32% perf hit; TP deferred). Our approach:
  per-head bf16 streaming accumulation (89% mem save, no perf hit).
- **Template:** `upstream_prs/12_megatron_dsa_compute_index_scores_memory.md`
- **Reproducer:** `upstream_prs/examples/12_dsa_indexer_memory/reproducer.py`
  (last verified GB10, exit 0, MEMORY_SAVE_VERIFIED, plus `run_gb10.log`).
- **Explanation (RU):** `upstream_prs/examples/12_dsa_indexer_memory/explanation_ru.md`
- **Ready:** Y
- **File as:** COMMENT on PR #4039 first, offering pack 12 as a
  complementary path (no perf hit, no TP deferral). Do NOT open a
  competing PR — that will just stall both.
- **Command:**
  ```
  gh pr comment 4039 --repo NVIDIA/Megatron-LM \
    --body-file upstream_prs/12_megatron_dsa_compute_index_scores_memory.md
  ```
- **Risk — upstream engagement:** medium-high (PR #4039 already has
  maintainer attention, and #4039 explicitly notes TP support deferred,
  which our streaming approach sidesteps).
- **Risk — us:** medium. If #4039 lands first, our pack becomes
  "alternative" rather than "the fix". Offer early.

### PR 13 — TileLang FloorMod LayoutInference divide-by-zero
- **Target repo:** tile-ai/tilelang
- **Upstream state:** `new_report`. Adjacent open issue #904 (our own
  GB200 sm100 autotuner issue) in the same repo.
- **Template:** `upstream_prs/13_tilelang_floormod_layout_inference_dbz.md`
- **Reproducer:** `upstream_prs/examples/13_tilelang_floormod_dbz/reproducer.py`
  (last verified bench3, exit 0, BUG_REPRODUCED). Patch file
  `mamba3_bwd_layout_fix.patch` is present (do **not** include this as
  a fix — the memory `tma-layout-fix-3d-to-2d` note says it's broken on
  H200; file the **bug report**, not the fix).
- **Explanation (RU):** `upstream_prs/examples/13_tilelang_floormod_dbz/explanation_ru.md`
- **Ready:** Y (bug report only).
- **File as:** ISSUE.
- **Command:**
  ```
  gh issue create --repo tile-ai/tilelang \
    --title "[Bug] LayoutInference divide-by-zero on FloorMod(expr, R) inside T.Parallel" \
    --body-file upstream_prs/13_tilelang_floormod_layout_inference_dbz.md
  ```
- **Risk — upstream engagement:** high (TileLang maintainers are very
  active — PRs #746/#761/#2005/#904 all closed recently).
- **Risk — us:** low; filing as a bug does not commit us to a fix.
  **Important:** do NOT attach `tma-layout-fix-3d-to-2d` as a PR — see
  `reference_tma_layout_fix_broken_h200.md`.

---

## Group B — Needs maintainer attention first (piggyback on open PRs)

### PR 09 — Liger FLCE reduction="none" backward
- **Target repo:** linkedin/Liger-Kernel
- **Upstream state:** **Issue #968 CLOSED** 2026-03-03 (updated field matches
  `2026-03-03T18:48:36Z`). Draft PR **#1126 is OPEN draft**, but it only
  *prevents* `reduction='none'` backward (errors out), does not fix the
  corruption. Our pack 09 provides the working mean-broadcast fix.
- **Template:** `upstream_prs/09_liger_flce_reduction_none_backward.md`
- **Reproducer:** `upstream_prs/examples/09_liger_flce_reduction_none/reproducer.py`
  (bench3, exit 0, BUG_REPRODUCED + FIX_VALIDATED).
- **Explanation (RU):** `upstream_prs/examples/09_liger_flce_reduction_none/explanation_ru.md`
- **Ready:** Y
- **File as:** COMMENT on draft PR #1126 showing that a *working* fix
  exists (do not reopen #968 — closed by maintainer). Offer our patch.
- **Command:**
  ```
  gh pr comment 1126 --repo linkedin/Liger-Kernel \
    --body-file upstream_prs/09_liger_flce_reduction_none_backward.md
  ```
- **Risk — upstream engagement:** medium. Maintainers chose "prevent"
  route; our patch flips that policy decision, so may be declined.
- **Risk — us:** low. Patch lives in our tree already
  (`apply_linear_ce_patch.py` path #3).

### PR 10 — Megatron Hopper kernels for fused linear CE
- **Target repo:** NVIDIA/Megatron-LM
- **Upstream state:** PR **#3345 OPEN**, non-draft, updated 2026-03-23.
- **Template:** `upstream_prs/10_megatron_fused_linear_ce_hopper_support.md`
- **Reproducer:** `upstream_prs/examples/10_megatron_flce_hopper/reproducer.py`
  + `mtp_nan_reproducer.py`. Last verified on mac-local = **SKIPPED** (no
  CC=9). Needs H200 rerun before filing.
- **Explanation (RU):** `upstream_prs/examples/10_megatron_flce_hopper/explanation_ru.md`
- **Ready:** **N** — blocked on H200 reproducer run. Do a bench3/europe
  run first, capture retained log, and flip `exit_code` from null to 0.
- **File as:** COMMENT on PR #3345 with H200-validated reproducer.
- **Receipt gate:** pack 10 still lacks a retained H200 filing receipt; do
  not describe it as receipted, H200-validated, or ready-to-file until that
  rerun artifact exists on disk.
- **Command (after H200 rerun):**
  ```
  gh pr comment 3345 --repo NVIDIA/Megatron-LM \
    --body-file upstream_prs/10_megatron_fused_linear_ce_hopper_support.md
  ```
- **Risk — upstream engagement:** medium; PR hasn't moved in 3 weeks.
- **Risk — us:** low once H200-validated.

### PR 11 — MambaModel missing LinearCrossEntropyModule
- **Target repo:** NVIDIA/Megatron-LM
- **Upstream state:** `new_report`. Related: PR #3226 (closed+merged,
  reapply Linear CE fusion) and #3207 (closed+merged, reapply MTP for
  hybrid models). Our pack reports that MambaModel didn't get the same
  LinearCrossEntropyModule class-swap that GPTModel did.
- **Template:** `upstream_prs/11_megatron_mamba_linear_ce_module.md`
- **Reproducer:** `upstream_prs/examples/11_mamba_linear_ce/reproducer.py`
  (bench3, exit 0, FIX_VALIDATED).
- **Explanation (RU):** `upstream_prs/examples/11_mamba_linear_ce/explanation_ru.md`
- **Ready:** Y
- **File as:** ISSUE, cross-referencing #3226 and #3207 as the merges
  that left MambaModel behind.
- **Command:**
  ```
  gh issue create --repo NVIDIA/Megatron-LM \
    --title "MambaModel.output_layer never swapped to LinearCrossEntropyModule (regression vs GPTModel #3226/#3207)" \
    --body-file upstream_prs/11_megatron_mamba_linear_ce_module.md
  ```
- **Risk — upstream engagement:** medium-high; hybrid-model work is
  active (PRs #3207/#3226 were both reapplies, so this is current).
- **Risk — us:** low.

---

## Group C — Regression tests / already fixed (internal only)

### PR 08 — TileLang LowerBulkCopy 3D smem regression test
- **Target repo:** tile-ai/tilelang
- **Upstream state:** **MERGED** (#746 closed+merged 2025-08-22; follow-up
  #761 closed+merged 2025-08-28; #2005 regression test closed+merged
  2026-04-14). All verified via `gh api 2026-04-14`.
- **Template:** `upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md`
- **Reproducer:** `upstream_prs/examples/08_tilelang_tma_bulk_copy_3d_smem/reproducer.py`
  (REGRESSION_TEST_PASSED on bench3 + gb10).
- **Ready:** N/A — DO NOT file.
- **Action:** Keep as internal regression test. No upstream submission.

---

## Group D — Mamba3 bundle (coordinate as one exchange)

All three target `state-spaces/mamba`. Maintainer velocity on that repo
is low — bundle them to avoid three separate notifications.

### PR 04 — Mamba3 SISO bwd redundant `v_dot`
- **Upstream state:** `new_report`. Adjacent closed issue #868
  (`mamba3_siso_bwd_kernel_dqkv fails with bf16 inputs due to dtype
  mismatch in tl.dot`) — same kernel, already closed.
- **Template:** `upstream_prs/04_mamba3_siso_bwd_eliminate_redundant_vdot.md`
- **Reproducer:** `upstream_prs/examples/04_mamba3_siso_bwd_vdot/reproducer.py`
  (bench3, exit 0, BITWISE_IDENTICAL).
- **Explanation (RU):** `upstream_prs/examples/04_mamba3_siso_bwd_vdot/explanation_ru.md`
- **Ready:** Y
- **File as:** ISSUE with patch link (it's a code-clarity refactor,
  not a perf or correctness fix — frame as cleanup).

### PR 05 — Mamba3 dt fp32 GQA backward
- **Upstream state:** `new_report`. Related: OPEN issue #886
  (`Mamba3 backward pass crashes with misaligned address when
  nheads % 4 != 0 and seqlen % 4 != 0`).
- **Template:** `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.md` (+ `.patch`)
- **Reproducer:** `upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd/reproducer.py`
  (bench3, exit 0, BUG_REPRODUCED + FIX_VALIDATED + GB10_BLOCKED).
- **Shared-evidence note:** the same reproducer also covers PR 16's
  Megatron-side Float16Module cast bug. For filing, split by stage:
  `bf16/fp32` -> PR 16, `gqa_unpatched/gqa_patched` -> PR 05.
- **Ready:** Y (H200-only validation; GB10 blocked by pack 13).
- **File as:** PR with the Mamba-side GQA diff only; do not present the
  bundled local convenience patch as if every hunk belongs in
  `state-spaces/mamba`. Cross-reference #886 as related.

### PR 16 — Megatron Float16Module silently casts Mamba3 fp32-contract params
- **Target repo:** NVIDIA/Megatron-LM
- **Upstream state:** `new_report`.
- **Template:** `upstream_prs/16_megatron_float16module_mamba3_cast.md`
- **Reproducer:** shared with PR 05 at
  `upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd/reproducer.py`
  (bench3, exit 0; `bf16` reproduces the cast symptom, `fp32` validates the
  fp32-contract path).
- **Ready:** Y (shared H200 transcript-backed evidence only; no retained log yet).
- **File as:** ISSUE first against Megatron-LM. Frame it as a generic
  Float16Module contract bug, not as a Mamba-only local workaround request.

### PR 07 — Mamba3 MIMO 3D→2D smem refactor
- **Upstream state:** `new_report`. **Target resolved:**
  `state-spaces/mamba` (template explicitly names the repo;
  `mamba3_mimo_bwd.py` lives there). Manifest previously said
  `tile-ai/tilelang` — that was wrong and has been corrected in
  `examples/validation_manifest.yaml`. The TileLang-side blocker for
  bwd_bwd is pack 13, not pack 07.
- **Template:** `upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md`
- **Reproducer:** `upstream_prs/examples/07_mamba3_mimo_3d_to_2d_smem/reproducer.py`
  (GB10, exit 0, LEGALITY_PROVED).
- **Ready:** Y (legality-proof refactor; enables TileLang TMA lowering).
- **File as:** PR on `state-spaces/mamba` with the refactor patch.

### Bundle command (Mamba3 group, one submission wave):
```
# Confirm order with user; suggested order 04 -> 05 -> 07
gh issue create --repo state-spaces/mamba --title ... --body-file upstream_prs/04_...md
gh pr create    --repo state-spaces/mamba --title ... --body-file upstream_prs/05_...md
gh issue create --repo state-spaces/mamba --title ... --body-file upstream_prs/07_...md
```

- **Risk — upstream engagement:** LOW. state-spaces/mamba has low
  maintainer velocity; expect 2-6 week response time.
- **Risk — us:** low; bundle keeps reviewer fatigue down.

---

## Group E — TileLang SparseMLA (same repo, coordinate)

Both target `tile-ai/tilelang` under `examples/deepseek_v32`. Bundle
together so the maintainer sees one coordinated contribution.

### PR 02 — SparseMLA dimension generalisation
- **Upstream state:** `new_report`.
- **Template:** `upstream_prs/02_sparse_mla_generalize_dimensions.md`
- **Reproducer:** `upstream_prs/examples/02_sparse_mla_dimensions/reproducer.py`
  (bench3, exit 0).
- **Ready:** Y

### PR 03 — SparseMLA FP8 dispatch
- **Upstream state:** `new_report`. **Target repo resolved:**
  `NVIDIA/TransformerEngine` (confirmed by reading template body: the
  complaint is about TE Float8Tensor wrapper behavior — `.dtype` lies,
  `.data_ptr()` returns NULL, `.to()`/`.contiguous()` don't unwrap. The
  user's original mapping to `tile-ai/tilelang examples/deepseek_v32`
  is incorrect; the issue is a TE dispatch hazard, not a TileLang kernel
  bug). **Hedge:** TE ≥ 2.13 silently auto-dequantizes, masking the
  crash but still wasting FP8 bandwidth; reproducer documents this.
- **Template:** `upstream_prs/03_sparse_mla_fp8_dispatch.md`
- **Reproducer:** `upstream_prs/examples/03_sparse_mla_fp8_dispatch/reproducer.py`
  (bench3, exit 0, HAZARDS_SHOWN).
- **Ready:** Y (file to NVIDIA/TransformerEngine, not tile-ai/tilelang).

### Bundle commands (PR 02 and PR 03 go to DIFFERENT repos):
```
# 02 -> tile-ai/tilelang examples/deepseek_v32
gh issue create --repo tile-ai/tilelang --title ... --body-file upstream_prs/02_sparse_mla_generalize_dimensions.md

# 03 -> NVIDIA/TransformerEngine (dispatch hazard, not a TileLang bug)
gh issue create --repo NVIDIA/TransformerEngine --title ... --body-file upstream_prs/03_sparse_mla_fp8_dispatch.md
```

- **Risk — upstream engagement:** TileLang high, TE medium.
- **Risk — us:** low now that repos are disambiguated. PR 03 is a
  hazard report with TE ≥ 2.13 auto-dequant hedge noted in body.

---

## Pre-flight checklist (do before any "post it")

1. [ ] Reproducer passes on the listed host on the day of filing
   (not trust the stale manifest date).
2. [ ] Template markdown renders cleanly on github.com preview (paste
   into a draft, check).
3. [ ] Attach reproducer as a file or a gist link — not inline
   hundreds of lines.
4. [ ] RU explanation stays INTERNAL — do **not** include in upstream
   body. English only on public PRs.
5. [ ] For Megatron PRs: run `tools/autoformat.sh` if patching, per the
   PR template checklist.
6. [ ] No secrets / gcloud URLs / internal host names in body.
7. [ ] User explicit "post it" for THIS pack.

## Submission order (recommended)

Wave 1 (Group A, today if approved): **01, 13, 12**
 — PR 12 filed as a comment on #4039, not a competing PR.

Wave 2 (Group B, after H200 rerun for pack 10): **11, 09, 10**
 — Wait 2-3 days between comments to avoid spam.

Wave 3 (Group E, after disambiguating 03): **02, 03**

Wave 4 (Group D, bundle): **04, 05, 07**

Skip: **08** (already merged upstream via #746/#761/#2005).
