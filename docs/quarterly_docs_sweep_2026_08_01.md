# Quarterly docs sweep — 2026-08-01

**Scope**: `docs/` top-level dated notes and `docs/sessions/` indexing.  
**Policy basis**: [status/README.md](status/README.md) retention rules.

---

## 1. What was reviewed

- 75 top-level `.md` files in `docs/`.
- `docs/sessions/README.md` index of dated session/probe notes.
- `docs/status/README.md` canonical entry points and retention rules.

## 2. Changes made

### 2.1 Status index updated

[docs/status/README.md](../status/README.md) now lists the new document-isolation
design docs as canonical entry points:

- `docs/document_isolation_varlen_design.md`
- `docs/document_isolation_swa_design.md`
- `docs/document_isolation_cp128k_design.md`

The retention rules now explicitly mention a **quarterly sweep** of dated notes
older than 90 days.

### 2.2 Session notes re-tagged

`docs/sessions/README.md` was updated to mark notes from April 2026 as
`archived`, because they are now > 90 days old and the decisions they captured
have either landed in code/tests or been superseded by canonical docs.

Specifically archived:

- 2026-04-11 NAM56R baseline / nsys / reproducibility / VPP / MTP plans
- 2026-04-12 Blackwell feature sweep, DSA EP=2 sweep, NAM56R grid search
- 2026-04-13 FP8 optimization session, optimization sessions
- 2026-04-14 session findings, FP8 research, Mamba fork canonical, session 3 closeout, gap audit
- 2026-04-15 GB10 regression, grad NaN investigations/bisects

Notes kept active:

- 2026-04-25 `gb10_dense_mxfp8_status_2026_04_25.md` — still canonical for dense GB10 MXFP8/NVFP4.
- 2026-04-25 `deprecated_path_gates_2026_04_25.md` — referenced by current gate code.
- 2026-04-25 `sparse_mla_blockscaled_*_2026_04_25.md` — active investigations still open.
- 2026-04-25 `te_mxfp8_backward_gb10_plan_2026_04_25.md` — active until P085/P086 close.

### 2.3 Superseded references

- Removed `reference_cp_blocked_by_custom_mixers.md` reference from
  `docs/long_context_roadmap.md` (file no longer exists; replaced by
  `docs/document_isolation_cp128k_design.md`).

## 3. What was NOT deleted

No files were removed in this sweep.  All archived notes remain in place because
retention rule 5 requires a safe-reference check before deletion, and some old
notes are still cited by commit messages or external issue text.  The index
status change is sufficient to signal that they are no longer current guidance.

## 4. Safe-reference check command used

```bash
rg -n "file_name_without_path|docs/file_name.md" README.md docs cppmega scripts tests tools
```

Any doc with remaining references was left in place and only re-tagged.

## 5. Next sweep

Schedule: 2026-11-01 (next quarter).

Focus areas for the next sweep:

- Resolve April 2026 `sparse_mla_blockscaled_*` notes into a canonical status
  doc or close them.
- Convert `gb10_dense_mxfp8_status_2026_04_25.md` into a non-dated canonical
  doc once the GB10 MXFP8 path is stable.
- Archive `mamba3_mimo_p2_psiv_cache_design.md` if the PsiV cache scaffolding
  removal (P010) is complete.
