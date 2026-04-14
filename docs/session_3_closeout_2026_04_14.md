# Session 3 Close-out — 2026-04-14

Scope: consolidate what shipped in commit range `c1c4719..HEAD`, what was empirically validated vs assumed, outstanding blockers, and prioritised session-4 work.

---

## 1. Shipped in session 3

Commit range `c1c4719..HEAD` (8 commits, `4ddff4f` → `d04225d`):

| Commit | Summary |
|---|---|
| `4ddff4f` | Mamba fork canonical reconciliation (`docs/mamba_fork_canonical_2026_04_14.md`), Megatron-LM restoration recipe, fast-hadamard wheel install script |
| `a9ebb78` | Session-3 code audit fixes, backup manifest unification (bench3 + europe), PR pack validation (`upstream_prs/examples/validation_manifest.yaml`), `conftest.py` hardening |
| `95e4ebf` | Upstream PR submission checklist + verified `upstream_status.md` table |
| `bf52831` | Session 3 gap audit (`docs/session_3_gap_audit.md`) + README megatron-core version correction (bench3 actually 0.18.0rc0, not 0.16.0rc0) |
| `36ce8cd` | Test hygiene — removed phantom DSA references and obsolete FP8 indexer shim scripts from test tree |
| `9129103` | Mamba3 P2 PsiV cache design + prototype scaffold (NOT wired into active path) |
| `6dfb8ba` | Tightened PR 01 (CUDA-graph-capture guards), PR 02 (dim generalisation), PR 10 (Hopper CE precision split) drafts |
| `d04225d` | Single production-status doc (`docs/production_status.md`), corrected stale `LOCATION_3` defaults, cleaned DSA FP8 sweep script |

### Key deliverables

- **Upstream PR pack** 01–13 (with gap at 06) with validation manifest and submission checklist — all PRs have reproducers and verified status tables.
- **Full backup manifests** for bench3 (`.tmp/backup_bench3_2026_04_14/`) and europe (`.tmp/backup_europe_2026_04_14/`) with sha256 integrity and GS-bucket mirror.
- **Gap audit** (`docs/session_3_gap_audit.md`) correcting doc drift on bench3 megatron-core version (0.18 not 0.16) and flagging PR 14 candidate (Mamba3 MIMO P1 TMA + warpspec).
- **Mamba fork canonicalisation** — both machines on HEAD `31f3d7baba`; only `mamba3_siso_combined.py` drifts (bench3 has PR #909 checkpoint-compat cache).
- **Test tree conftest + `_megatron_stub.py`** — tests that touch Megatron imports don't crash on env-less collection.

---

## 2. Validated empirically

| Claim | Validation |
|---|---|
| Bench3 megatron-core = 0.18.0rc0 | `gh api` lookup of installed git sha `980211ae`, plus `venvs/02_lib_versions.txt` line 12 |
| Europe megatron-core = 0.16.0rc0 | `07_lib_versions.txt` line reading `megatron.core 0.16.0rc0` via editable install |
| Mamba_ssm HEAD = `31f3d7baba` on both | Direct git-state capture in both backup bundles; md5-diff of working-tree files |
| Only `mamba3_siso_combined.py` diverges bench3↔europe | Four-file md5 comparison in `docs/mamba_fork_canonical_2026_04_14.md` |
| PR 14 (Mamba3 P1 TMA+warpspec fwd) is a real gap | Walked all `upstream_patches/*.py` against PR templates; `apply_mamba3_mimo_p1_patches.py` has no upstream target |

## 3. Claimed but NOT re-tested this session

- **Bench3 golden config 269.4 TFLOP/s** (`reference_main_head_liger_ce_gap.md`, `reference_golden_config_2026_04_13.md`) — pending a remote rerun after session-3 code changes landed.
- **Europe baseline 289 TFLOP/s** (BF16 PP=1 EP=4 MBS=8) — pending a remote rerun. Test conftest changes alter no training-path code, but throughput deserves verification before the next claim.
- **PR 10 Hopper CE reproducer** — authored but needs an H200 (`cc=9`) execution to confirm the precision-split claim.
- **MTP NaN root cause** — Suspect #1 (MTP shape guard) was refuted by `253f740`; Suspect #2 (Liger FLCE `reduction="none"` gradient path) is opened but not closed. **Do not ship MTP native path until closed.**

---

## 3. Open questions (ship nothing that depends on these)

1. **MTP NaN root cause** — `reduction="none"` Liger FLCE gradient path silently corrupts gradients per upstream issue referenced in `upstream_prs/09_liger_flce_reduction_bug.md`; only the non-fused path is fixed by Liger PR #680. The Megatron-Hopper native CE patch (`mtp_native_hopper_ce.py`) produced `grad_norm=NaN` end-to-end in session-3 dry runs. Needs targeted repro on H200 before any claim.
2. **Bench3 golden config post-session-3** — conftest + patch-file reorg could theoretically alter import ordering for the patches applied in `apply_dsa_cg_patches.py` / `apply_linear_ce_patch.py`. No change to patch content, but verify on remote once available.
3. **Europe baseline 289 post-session-3** — same reasoning; no training-code change, but verify.
4. **PR 10 Hopper CE numeric claim** — split-precision path shows 2–3× over naive only in bench-math simulation; actual kernel run on `cc=9` needed.
5. **Why bench3 and europe diverged on megatron-core** — bench3 pip-installs from pinned upstream sha `980211ae` (0.18.0rc0), europe runs an editable clone on `dev_latest` 2 ahead of `origin/dev` (0.16.0rc0). Decide whether to sync (and which direction) in session 4.

---

## 4. Risks / gotchas for next session

- **Bench3 and europe are on DIFFERENT Megatron-core versions** (0.18.0rc0 vs 0.16.0rc0). Any "same behaviour on both" claim must be verified before citing. See updated `reference_megatron_version.md`.
- **Liger FLCE `reduction="none"` silently corrupts gradients.** Only the non-fused path was fixed by upstream PR #680. Fused path remains broken; the workaround is to use `reduction="mean"` and gate it with a correctness test. Document this next to `apply_linear_ce_patch.py`.
- **TileLang FloorMod bug still blocks Mamba3 bwd_bwd on GB10** (see `reference_p1_blocked_tilelang_tma_layout.md` and `reference_tma_layout_fix_broken_h200.md`). The tma-layout-fix-3d-to-2d branch is BROKEN on H200 — do not merge. Upstream issue is still open.
- **GB10 sometimes unreachable** (hostname resolution intermittently fails); local caching of `/Volumes/external/sources/cppmega/.tmp/artifacts/` is the fallback for anything we need off that box.
- **Fresh venv + editable install is NOT in the Quick Start.** Finding 5 in `docs/session_3_gap_audit.md` flagged this — a cold restore from backup needs manual activation steps the README doesn't list.

---

## 5. Session 4 TODO (prioritised)

1. **Draft PR 14** — `apply_mamba3_mimo_p1_patches.py` → `state-spaces/mamba`. Scope: flip `TL_DISABLE_TMA_LOWER` + `TL_DISABLE_WARP_SPECIALIZED` from `True` → `False` in `mamba_mimo_fwd` sites. Fwd-only (bwd blocked on 3D smem descriptor TMA lower; see TileLang upstream issue).
2. **Apply Liger `reduction="mean"` workaround + correctness test gated** — build a test-only alt path that verifies loss/grad parity vs the `reduction="none"` bug path. Keep the test **gated off** in prod (the bug doesn't fire at `reduction="mean"`).
3. **Close MTP Suspect #2** — author targeted reproducer for Liger FLCE `reduction="none"` grad corruption on real Megatron shapes. Do NOT ship `mtp_native_hopper_ce.py` until closed.
4. **Reconcile bench3 0.18 vs europe 0.16 megatron-core drift** — either:
   - Bring europe up to `980211ae` (`0.18.0rc0`), re-run europe baseline, confirm 289 still holds, OR
   - Document explicitly why the divergence stays (e.g. europe's PR #3674 + #4268 don't port cleanly to 0.18).
5. **Remote rerun of golden configs** — bench3 269.4 TFLOP/s + europe 289 TFLOP/s, verify no regression from session-3 conftest/test-hygiene changes.
6. **Add venv-activation + `pip install -e .` to README Quick Start** (Finding 5 carry-over).
7. **Drop or caveat `pip install apex`** in README Quick Start (europe production runs without it).
8. **Add `(DO NOT USE — broken on H200)` caveat** to README citation of `apply_mamba3_mimo_tma_layout_fix.py` branch.

---

## 6. Files NOT committed / TBD

Working tree at close of session 3 has these uncommitted items — intentionally left out because they're state files, not code:

- `.omx/metrics.json`, `.omx/state/*.json` — harness state (routinely dirty)
- `.tmp/fht_work/fast-hadamard-transform` — nested git clone (build artifact)
- `README.md`, `plan.md` — have in-flight edits; hold for session 4 cleanup
- `tests/test_{fastmtp_layer,mamba3_te_mixer,nam56r_noconv_spec,noconv_mamba_mixer}.py` — test hygiene edits in progress
- `tests/_megatron_stub.py` (untracked, new) — conftest helper
- `docs/production_status.md` (untracked, new) — may already be committed via `d04225d`; recheck before session 4
- `upstream_prs/01_dsa_cuda_graph_safety.md`, `upstream_prs/02_sparse_mla_generalize_dimensions.md` — re-tightened by `6dfb8ba`, may have residual edits
- `upstream_prs/14_sparse_mla_precision.md` (untracked, new) — draft shell for PR 14 (next session)

All the above are either harness state, in-flight work for session 4, or build artefacts. None block.

---

## 7. Cross-references

- `docs/session_3_gap_audit.md` — parent audit with PR numbering, backup manifest corrections, README consistency check
- `docs/mamba_fork_canonical_2026_04_14.md` — mamba_ssm fork reconciliation
- `docs/megatron_restoration_recipe.md` — how to rebuild bench3 / europe venvs from backup
- `upstream_prs/examples/validation_manifest.yaml` — PR reproducer status
- `.tmp/backup_bench3_2026_04_14/MANIFEST.md`, `.tmp/backup_europe_2026_04_14/MANIFEST.md` — bench snapshots (GS mirrored)

## 8. Memory updates in this closeout

Updated memory notes (point-in-time corrections at `/Users/dave/.claude/projects/-Volumes-external-sources-cppmega/memory/`):

- `reference_megatron_version.md` — removed stale `0.15.0rc7 dev_latest` claim; split per-machine (bench3 0.18.0rc0 pip, europe 0.16.0rc0 editable); listed applied PR cherry-picks per machine.
- `reference_env_drift_bench3_europe.md` — marked the GQA md5 mismatch as CLOSED (both machines converged on `mamba3_mimo_bwd.py`); only `mamba3_siso_combined.py` still differs via bench3's PR #909 cache tweak.
- `reference_bench3_h200_stack.md` — refreshed pre-installed-packages block against 2026-04-14 backup (tilelang now present, cppmega editable install, megatron-core git sha call-out).
- `reference_stack_bench.md` — re-verified europe versions vs `07_lib_versions.txt`; noted megatron-core divergence from bench3.
