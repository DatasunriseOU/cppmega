# MXFP8 Worktree Audit - 2026-05-01

Base: `main` at `4fbafe9004fb859a79075afe79e5a25dbe182eaa`.
Audit worktree: `/home/dave/source/cppmega-mxfp8-worktree-audit-20260501`.
Main worktree was left untouched and had pre-existing uncommitted edits.

## Requested Worktrees

- Present: `/home/dave/source/cppmega/.claude/worktrees/mxfp8-perf-fixes`, branch `mxfp8-perf-fixes`, HEAD `9ac8f1d`.
- Present under different path: `/home/dave/source/cppmega/.claude/worktrees/mxfp8-v2`, branch `fix/mxfp8-v2`, HEAD `846952d`.
- Missing path: `/home/dave/source/cppmega/.claude/worktrees/mxfp8-fixes`; no matching local branch found.
- Missing path: `/home/dave/source/cppmega/.claude/worktrees/fix/mxfp8-v2`; the branch exists in the path above.
- Missing path: `/home/dave/source/cppmega/.claude/worktrees/fix/mxfp8-probe-review`; branch `fix/mxfp8-probe-review` exists, HEAD `5aa7ee5`.

## Recommendation

Do not merge any requested branch wholesale. The named branches have no merge base with current `main`, and `fix/mxfp8-v2` / `fix/mxfp8-probe-review` are older trees that would remove large current surfaces such as grouped MXFP8, MTP CE, profile probes, and tests.

Cherry-pick nothing immediately from `mxfp8-perf-fixes`: its four tip commits contain useful work, but current `main` already has the important content in divergent later form:

- `79bbac9` CUTLASS MXFP8 beta/ptr_C/workspace fix: already represented by `_resolve_beta`, `ptr_C = nullptr` when `beta == 0`, and no bespoke workspace cache.
- `95f97ee` FP8 shim TE guardrails: current shim is substantially newer; do not replay this old patch. If TE fall-through behavior needs more audit, port individual checks manually.
- `d191860` QMuon performance/finite fixes: already represented by 128-thread blocks, warp-shuffle reductions, `isfinite` scale guard, and absmax epsilon handling.
- `9ac8f1d` SM120 auxiliary-load enablement: already represented by `UseAuxiliaryLoad` dispatch policy and direct compact-scale backend using the auxiliary load split.

Discard as branch merges:

- `fix/mxfp8-v2` (`846952d`): includes a gitlink at `.claude/worktrees/mxfp8-fixes`, broad stale probe/script/test changes, and would roll back or delete many files now on `main`.
- `fix/mxfp8-probe-review` (`5aa7ee5`): final auxiliary-load idea is already present; branch tree is stale and would delete current grouped MXFP8/profiling surfaces.
- Missing `mxfp8-fixes` path/branch: nothing to merge from the requested name.

## Validation

Focused pytest command attempted:

```bash
python -m pytest -q tests/test_cutlass_mxfp8_gemm.py tests/test_cutlass_mxfp8_gemm_config.py tests/test_cutlass_mxfp8_shim_routing.py tests/test_quantized_muon_momentum.py tests/test_mxfp8_probe_helpers.py tests/test_run_profiles.py
```

Result: blocked during root `conftest.py` import because the checkout resolves a source TransformerEngine tree without its compiled core shared library, raising `FileNotFoundError: Could not find shared object file for Transformer Engine core lib.` This is an environment/import blocker, not a targeted test failure.
