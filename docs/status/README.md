# Documentation Status And Retention Policy

This repo keeps two kinds of docs:

- **Canonical status docs**: maintained in place and safe to cite for current
  behavior.
- **Dated session/probe notes**: append-only evidence from a specific work
  session, run, machine, or branch. These are indexed instead of repeatedly
  moved or renamed.

## Status Labels

Use one of these labels in new docs and in index rows:

| Label | Meaning | Expected action |
| --- | --- | --- |
| `canonical` | Current source of truth for a topic. | Update in place when facts change. |
| `active` | Current investigation or implementation note that may still affect work. | Keep indexed until resolved into a canonical doc or closed. |
| `evidence` | Historical measurement, probe, or audit that supports a decision. | Keep, but cite through the canonical doc when possible. |
| `superseded` | Replaced by a newer canonical doc or measurement. | Do not delete unless all references are updated. |
| `archived` | Useful paper trail, not current guidance. | Leave in place unless a safe archival move is needed. |

## Current Canonical Entry Points

| Topic | Canonical doc | Notes |
| --- | --- | --- |
| Production configs and numbers | [../production_status.md](../production_status.md) | Current throughput, launch env, deprecated measurements. |
| Cppmega architecture/status | [cppmega_architecture_status.md](cppmega_architecture_status.md) | Current precision paths, GB10/H200 caveats, and rationale. |
| Typed run profiles and token flow | [cppmega_run_profiles_and_token_flow.md](cppmega_run_profiles_and_token_flow.md) | Dataclass run-profile contract, local GB10 token flow, dimensions, and precision boundaries. |
| Reproducible launch commands | [../reproducible_runs.md](../reproducible_runs.md) | Single-command wrappers for validated configs. |
| Porting rules | [../porting_policy.md](../porting_policy.md) | What to port, what to reuse upstream, validation order. |
| FP8 path status | [../fp8_path_status.md](../fp8_path_status.md) | Current per-path FP8 support where still relevant. |
| GB10 hardware facts | [../gb10_sm121_hardware.md](../gb10_sm121_hardware.md) | Hardware capability reference. |
| GB10 software stack | [../gb10_software_stack.md](../gb10_software_stack.md) | Library and toolchain compatibility. |
| Dense GB10 MXFP8/NVFP4 | [../gb10_dense_mxfp8_status_2026_04_25.md](../gb10_dense_mxfp8_status_2026_04_25.md) | Dated file, but currently canonical for this narrow topic. |
| Upstream bugs | [../upstream_bugs.md](../upstream_bugs.md) | Known upstream issues and local workarounds. |
| Data preparation | [../data_preparation.md](../data_preparation.md) | Dataset/tokenizer pipeline. |
| Long-context roadmap | [../long_context_roadmap.md](../long_context_roadmap.md) | Context-length thresholds and deferred work. |

## Retention Rules

1. Do not create a new dated doc for a topic that already has an active dated
   note from the same day and same workstream. Append to the existing note or
   add a clearly scoped subsection.
2. New dated notes should start with a small metadata block:

   ```text
   Status: active | evidence | superseded | archived
   Canonical: docs/<path>.md or none
   Date: YYYY-MM-DD
   Scope: one sentence
   ```

3. Within seven days of a session note that changes project direction, update
   the canonical status doc and mark the session note as `evidence` or
   `superseded` in [../sessions/README.md](../sessions/README.md).
4. Keep dated notes in place while they are referenced by code, tests, README
   sections, other docs, commit messages, or external issue/PR text.
5. Only move or delete a dated doc after a safe-reference check:

   ```bash
   rg -n "file_name_without_path|docs/file_name.md" README.md docs cppmega scripts tests tools
   ```

   If references exist, update them in the same change. Prefer an index status
   update over moving files.
6. If many notes accumulate for one topic, create or update a canonical status
   doc under `docs/status/` and link the individual dated notes as evidence.
   Avoid mass moves unless the reference check is clean.

## Suggested Paths For New Docs

- Current state: `docs/status/<topic>.md`
- Session/probe notes: `docs/sessions/YYYY-MM-DD-<topic>.md`
- Long-lived operational guides: `docs/<topic>.md`

Existing dated docs at the top level are intentionally left in place. The
policy applies to new docs and to future cleanup passes.
