# Session 3 — Gap Audit + Doc Corrections

Date: 2026-04-14 (session 3, post PR-pack validation)
Scope: numbering gaps, missing PR templates, backup manifest accuracy, doc drift.

---

## Finding 1 — PR 06 is a genuine missing slot (leave gap)

`upstream_prs/` contains template folders `01…05, 07…13` — **no `06_*`** folder, file, or `.patch`. Same gap appears in `upstream_prs/examples/` and in `upstream_prs/examples/validation_manifest.yaml`.

Git archaeology (all branches):

```
git log --all --oneline -- upstream_prs/06_* → no matches
git log --all -S "PR 06"                     → no matches
git log --all -S "06_"                       → no matches in upstream_prs history
git log --all --diff-filter=D --summary      → no deleted 06_* filenames
```

Conclusion: **PR 06 was never written**. The numbering was assigned consecutively as templates were drafted, and slot 06 was skipped (most likely because an early draft was folded into PR 05 — `05_mamba3_dt_fp32_gqa_bwd` covers TWO bugs: dt fp32 cast + GQA B/C layout — originally these could have been split as 05 and 06, but were merged into one pack and the number was not reclaimed).

Decision: **leave the gap**. Re-numbering 07-13 → 06-12 would invalidate every reference to these PRs in:
- `plan.md`, `docs/findings_2026_04_14_session.md`, memory notes
- commit messages: `b5a13af`, `c1c4719`, `a9ebb78`, `b8d6245`
- `validation_manifest.yaml` keys

Stable numbering is more valuable than a clean sequence. Future PRs should continue from 14.

Action: none (gap is intentional, documented here).

---

## Finding 2 — Patches without PR templates

Audited all three patch files in `cppmega/megatron/upstream_patches/` + the top-level `apply_linear_ce_patch.py` against existing PR templates 01-13.

| Patch file | Covered by PR template(s)? | Status |
|---|---|---|
| `apply_dsa_cg_patches.py` patches 1-9 + 9b | PR 01 (CG safety), PR 02 (dims), PR 03 (FP8 dispatch) | COVERED |
| `apply_linear_ce_patch.py` | PR 09 (Liger FLCE bug), PR 10 (Megatron Hopper CE), PR 11 (Mamba LinearCE class-swap) | COVERED |
| `apply_mamba3_mimo_p1_patches.py` | **NOT YET PR'd** — TMA + warpspec enable on Mamba3 MIMO fwd | **GAP** |
| `apply_dualpipev_patch.py` | N/A — DualPipeV is experimental, EP>1 incompatible, phantom per memory | NOT A BUG FIX — skip |
| `dsa_indexer_fused_patch.py` | PR 12 (DSA indexer memory) | COVERED |
| `index_cache_patch.py` | Not a bug fix — cppmega-specific layer pattern (3 Full + 6 Shared), not upstreamable | INTERNAL |
| `lemyx_dsa_warmup.py` | External dependency wrapping (lemyx/tilelang-dsa) | INTERNAL |
| `mamba_recompute_patch.py` | Not a bug fix — opt-in activation checkpointing | INTERNAL |
| `mamba3_compile_patch.py` | Not a bug fix — torch.compile regional wrapping | INTERNAL |
| `mtp_liger_ce.py` | Subsumed by PR 09 / 11 logic | COVERED |
| `mtp_native_hopper_ce.py` | Status: infrastructure only, grad_norm=NaN — NOT ready to upstream | BLOCKED |
| `selective_fp8_moe_patch.py` | FP8 MoE gate — internal config, no upstream target | INTERNAL |

### Gap — Candidate PR 14: Mamba3 MIMO P1 (TMA + warpspec enable on fwd kernels)

Target repo: `state-spaces/mamba`
Scope: flip `TL_DISABLE_TMA_LOWER=True` + `TL_DISABLE_WARP_SPECIALIZED=True` to `False` in `mamba_mimo_fwd` sites of `mamba3_mimo_fwd.py` and `mamba3_mimo_fwd_varlen.py`. Fwd-only because bwd kernels have 3D smem descriptors TileLang's TMA lower cannot handle (see PR 08/13).

This is a legitimate upstream contribution: turns on a correctness-verified perf path that was left disabled for reasons that no longer apply post-TileLang PR #746. Draft for session 4.

---

## Finding 3 — Backup manifest corrections

### Bench3 manifest — megatron-core 0.18.0rc0, not 0.16.0rc0

Raw evidence in `.tmp/backup_bench3_2026_04_14/venvs/`:

- `02_lib_versions.txt` line 12: `megatron.core: ver=0.18.0rc0 path=/mnt/data/venv/lib/python3.13/site-packages/megatron/core/__init__.py`
- `pip_venv.txt` line 76: `megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@980211ae6308dd541ec24bfe5af664ef31215256`

Cross-checked upstream via `gh api`:
- Commit `980211ae6308dd541ec24bfe5af664ef31215256` → "Miscellaneous MTP inference fixes (#4191)", 2026-04-09
- `package_info.py` at that ref: `MAJOR=0, MINOR=18, PATCH=0, PRE_RELEASE='rc0'` → `0.18.0rc0` ✓

So the **installed** bench3 megatron-core is unambiguously 0.18.0rc0.

The earlier (session-2) conclusion that "both machines are on 0.16.0rc0" was based on reading `package_info.py` inside the *bench3 tarball snapshot* (`/mnt/data/cppmega-root/megatron-lm/`) — which is a **separate source tree** from the pip-installed one. The tarball's `package_info.py` does report 0.16, but that tarball is NOT the source the venv resolved. The venv's `import megatron.core` points at site-packages/0.18.0rc0.

Europe remains 0.16.0rc0 correctly (editable install from `/home/dave/cppmega-root/megatron-lm/` on `dev_latest` branch 2 ahead of `origin/dev`).

### Other bench3 manifest version claims — all spot-checked accurate

- torch 2.12.0.dev20260410+cu132 ✓ (matches pip freeze)
- transformer_engine 2.13.0 ✓
- tilelang 0.1.8+cuda.gitf309d814 ✓
- mamba_ssm 2.3.1 ✓
- triton 3.7.0 ✓
- flash_attn 2.8.3 ✓
- fast_hadamard_transform 1.1.0 ✓
- tvm_ffi 0.1.9 ✓ (within MUST-BE-<0.1.10 constraint)

Only megatron-core row was wrong. Fixed.

### Europe manifest — no corrections needed

Spot-checked all version rows vs `07_lib_versions.txt` — all match. Europe manifest correctly flags the README drift (before fix) as "README says 0.18 — YES, older on europe (0.16.0rc0)".

### Files modified (backup manifest fixes)

- `.tmp/backup_bench3_2026_04_14/MANIFEST.md` — corrected the megatron-core row (table), the "CRITICAL UNIQUE ARTIFACTS" section 1 narrative, and the RESTORABILITY step 3.

---

## Finding 4 — README / plan / findings consistency

### Doc drift BEFORE this audit

`README.md` had:
- line 37 (Quick Start): "cherry-picks the currently-open upstream PRs #3674 and #4268 on top of 0.16.0rc0"
- lines 201, 205, 233: "both bench3 and europe are on 0.16.0rc0"

`plan.md` had:
- line 133: "version is `0.16.0rc0` on both machines"
- line 149: "memory note claimed bench3 '0.18' — false. Both are 0.16.0rc0"

`docs/findings_2026_04_14_session.md` had:
- line 408: "both bench3 and europe are on `0.16.0rc0`, NOT `0.18` as previously documented"
- line 444 (RU): same claim

These four assertions are **all wrong about bench3**. They were based on reading the bench3 source-tarball `package_info.py` without cross-checking the installed site-packages. The raw data captured in the backup (both `02_lib_versions.txt` and `pip_venv.txt`) directly contradicts.

### Fixes applied this audit

- `README.md` — corrected 4 locations (Quick Start prereq comment, Megatron Version table, "both hosts on 0.16.0rc0" paragraph, Software Stack row). Now states bench3=0.18.0rc0 (pip-git @980211ae), europe=0.16.0rc0 (editable dev_latest).

### Fixes NOT applied (flagged for session 4)

- `plan.md` lines 130-153 — contain the wrong claim + related wrong-reasoning block. Left in place because they are a "Drift discoveries" note from session 2 and rewriting them would require a session-4 correction block anyway. **Session 4 action**: add a correction note "SEE session_3_gap_audit.md — bench3 is actually 0.18.0rc0".
- `docs/findings_2026_04_14_session.md` lines 408, 444 — same situation. Session-specific document, better to append a correction than edit in place (preserves historical reading).
- Memory note `reference_megatron_version.md` — caller flagged it claims `core_v0.15.0rc7 dev_latest`, which itself contradicts both 0.16 (europe) and 0.18 (bench3). That memory is stale on TWO axes and should be rewritten by the user; not auto-editable from here.

### Production-status consistency check

`README.md` production config (line 7-9, 178-185): europe 289 TFLOP/s BF16 PP=1 EP=4 MBS=8, bench3 253 TFLOP/s same topology.

`plan.md` production config (line 7-10): europe 289 TFLOP/s BF16 (same), bench3 **~268-269 TFLOP/s FP8 MBS=10 EP=8 v3** (NOT the 253 figure).

**These two are inconsistent**. Reason: `plan.md` reflects bench3's newer record per `reference_fp8_mbs10_bench3_wins.md` and `reference_main_head_liger_ce_gap.md` (268 and 269.4 respectively, FP8 MBS=10 EP=8 + Liger main-head). README has NOT been updated to reflect the bench3 record — it still cites 253 (BF16 MBS=8 EP=4 at PP=1, which is the *topology-matched-to-europe* baseline, not bench3's best).

This is **not wrong, but stale**: README documents the cross-machine-comparable topology, plan.md documents per-machine bests. Session 4 should either:
- Add a "bench3 best record" row to the README throughput table (269.4 TFLOP/s FP8 MBS=10 EP=8 + Liger main-head), OR
- Add an explicit note that README "bench3 253" is topology-matched-for-comparison only, not bench3's record.

Flagged, not fixed in this audit.

---

## Finding 5 — README quick-start dry-run

Read README quick-start (lines 31-100) as if a fresh user. Verdict: **would succeed on bench3 + europe with caveats**.

### What works

- All file paths cited exist:
  - `scripts/remote_smoke_h200_dsa_9_4_m.sh` ✓
  - `scripts/install_tilelang_wheel.sh` ✓
  - `cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py` ✓
  - All 10 patch files cited in "Env-Gated" and "Always On" tables ✓
  - Project-structure tree (lines 256-283) — all files present, with ONE exception (below)
- Environment variables documented match actual checks in `scripts/remote_smoke_h200_dsa_9_4_m.sh` (spot-checked CPPMEGA_INDEX_CACHE, CPPMEGA_LEMYX_DSA, CPPMEGA_LIGER_CE, EP_SIZE_OVERRIDE, CG_FLAGS, PP_SIZE, VPP_SIZE, MBS)
- `dualpipe` pip-git URL is correct (deepseek-ai/DualPipe)

### What's subtly off

- **Line 274** cites `apply_mamba3_mimo_tma_layout_fix.py (branch: tma-layout-fix-3d-to-2d)` — this file exists only on the `tma-layout-fix-3d-to-2d` branch, NOT on main. README is explicit about that, but the reader might be confused because it's in the same "upstream_patches/" section as files that ARE on main. Per memory `reference_tma_layout_fix_broken_h200.md`, that branch **is broken on H200 — DO NOT MERGE**. Adding a `(DO NOT USE — broken on H200)` caveat would be prudent. Flagged for session 4.
- **Line 39 `cd /mnt/data/cppmega-root/megatron-lm`** — this is correct for bench3 (tarball path) but the comment under line 40 says `cd /home/dave/cppmega-root/megatron-lm` for europe. Per europe backup manifest, europe's megatron-lm lives at `/home/dave/cppmega-root/megatron-lm` (confirmed). ✓ matches.
- **Line 49 `pip install apex # NVIDIA apex from source with --cpp_ext --cuda_ext`** — per bench3 backup, apex is in site-packages. Per europe backup, apex is MISSING. If europe's production doesn't need apex (MFU works at 289), the Quick Start may be over-specifying a prerequisite. Flagged for session 4.

### Missing prerequisites that Quick Start implies but doesn't state

- Active venv required. README does NOT mention `source /mnt/data/venv/bin/activate` (bench3) or `source /mnt/data/cppmega-root/cppmega-venv/bin/activate` (europe). A fresh user hitting `bash scripts/remote_smoke_h200_dsa_9_4_m.sh` without an active venv will import-error. Flagged for session 4.
- `pip install -e .` for the local cppmega repo? The script imports `cppmega.megatron.*` — if cppmega itself isn't editable-installed, the imports will fail. Not stated in Quick Start. Flagged for session 4.

### Verdict

README Quick Start is **~90% correct**. A fresh user would succeed if they're already on bench3/europe with the venv active and cppmega editable-installed. On a fresh GCE rebuild, the venv activation + cppmega editable-install steps are missing and would need to be inferred from the backup restoration recipe.

---

## Files modified in this audit

| File | Change |
|---|---|
| `README.md` | 4 edits — megatron-core bench3/europe version, Quick Start comment, Megatron Version table row, Software Stack row |
| `.tmp/backup_bench3_2026_04_14/MANIFEST.md` | 3 edits — active-libs-row, CRITICAL UNIQUE ARTIFACTS section 1, RESTORABILITY step 3 |
| `docs/session_3_gap_audit.md` | NEW (this file) |

## Items flagged for session 4 (not fixed this audit)

1. Draft PR 14 — `apply_mamba3_mimo_p1_patches.py` → state-spaces/mamba
2. Add correction note to `plan.md` line 130-153 re: bench3 megatron version
3. Add correction note to `docs/findings_2026_04_14_session.md` lines 408 + 444
4. Update memory note `reference_megatron_version.md` (user-facing; claims 0.15.0rc7 which is also wrong)
5. Reconcile README throughput table vs plan.md production table (bench3 253 vs 269.4)
6. Add "broken on H200" caveat to README line 274 `apply_mamba3_mimo_tma_layout_fix.py` citation
7. Add venv-activation + `pip install -e .` prereqs to README Quick Start
8. Drop or clarify `pip install apex` (europe production runs without it)

## PR 06 slot decision

**Leave gap**. Stable numbering > clean sequence. Session-4 PRs continue from **14**.
