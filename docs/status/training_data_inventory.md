# Training-Data Inventory

Status: canonical

Release gate: [data release checklist](../data_release_checklist.md)

Live artifact:
`/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.json`

Human rendering:
`/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.md`

Append-only changelog:
`/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/changelog.jsonl`

This is the canonical entry point for corpus counts, versions, sidecars,
physical Parquet paths, and `bs=192` capacity. Do not reconstruct totals from
directory names or add snapshots by hand.

## Refresh

```bash
/Volumes/external/sources/.venvs/cppmega.source/bin/python \
  scripts/report_training_data_status.py \
  --config configs/training_data_status.json \
  --jobs 4
```

For a continuously refreshed local ledger:

```bash
/Volumes/external/sources/.venvs/cppmega.source/bin/python \
  scripts/report_training_data_status.py \
  --config configs/training_data_status.json \
  --jobs 4 \
  --watch-seconds 300
```

The output is atomic and durable on `/Volumes/external`; nothing is written to
`/tmp`. A changelog row is appended only when the semantic status SHA changes.
The checked-in
`configs/launchd/ai.cppmega.training-data-status.plist` runs this command every
five minutes as a macOS LaunchAgent and restarts it after an unexpected exit.

## Counting Rules

- `valid_tokens`: non-padding tokens physically present in packed rows.
- `trained_tokens`: tokens for which `loss_mask == 1`.
- `full_batches`: `floor(rows / 192)` independently inside each length bucket.
- A sealed snapshot and a newer live snapshot are never added without an
  explicit canonical-union receipt.
- CI store-local exact-dedup counters are an upper bound until cross-store
  canonical union/global dedup completes.
- A frozen threshold snapshot is reported separately from its overlapping live
  source store; the two counters are never added.
- After CASE5 export, only `ci-case5-train-*` audit counts are ready tokens;
  validation and test splits remain visible but excluded from that total.
- SQLite/CAS records are staged data, not training-ready tokens.
- Quarantine, smoke, validation, and legacy Parquet are reported but never
  included in production-ready totals.

## Current Baseline

The first canonical publication was generated on 2026-07-31:

| dataset | state | valid tokens | trained tokens | production-ready |
|---|---|---:|---:|---|
| live source Parquet | packed, unsealed | 3,525,307,879 | 3,515,655,055 | no |
| sealed Megatron bundle `macro_routes_v1_20260713` | `.bin/.idx`, audited | 4,132,747,235 | 4,123,963,840 | yes |
| verified PR/MR SQLite | staged | 0 | 0 | no |
| current CI CAS | staged | 0 | 0 | no |
| legacy 1,855-job CI Parquet | legacy sample | 37,571,933 | 37,568,685 | no |

The two source rows overlap and have no valid combined total. The live values
change as the conveyor promotes new repositories; cite `current.json`, not the
baseline table, for present-tense counts.

The frozen CI threshold snapshot contains exactly 61,311,228,208 store-local
unique payload tokens. They remain staged, with zero ready tokens, until its
final `case5/export_receipt.json` passes the existing CASE5 receipt validator.

## Current Blockers

- The live source completion receipt double-counts the case-fold collision
  `DirectXTK::code` / `directxtk::code`; physical Parquet is lower by 453,368
  valid tokens and 215 rows.
- The source conveyor is incomplete and still has failed units.
- Python auxiliary documents are token-conserving but still share physical
  packed rows with the C/C++/SQL/build/test primary stream.
- The PR store is verified, but primary five-bucket materialization was
  cancelled before producing eligible Parquet.
- CI acquisition is still non-exhaustive; canonical union/global dedup,
  primary-scope routing, five-bucket ZSTD Parquet export, and audit have not
  completed.

## Sidecar Contracts

The live source Parquet currently has one uniform 82-field schema:

- tensors and masks: `input_ids`, `target_ids`, `loss_mask`, `doc_ids`;
- source-document identity and routing: stable repo/file/doc IDs, token
  lengths, platform IDs, doc/header/build kinds, PR/discussion fields, IFIM,
  commit/pre/post/diff token IDs;
- pack provenance: repo, filepath, commit/parents/timestamps, merge and PR
  identity, local commit index, reconstruction/rename ambiguity;
- dense token routes: platform, structure, dependency, AST, domain, role,
  entity, scope, source identity, confidence, symbol, call/type/def-use, change,
  hunk, and edit-op IDs;
- ragged graph routes: chunk boundaries/kinds/dependency and call, type,
  domain, build, shell, diagnostic, and cross-domain edges;
- `source_identity_registry`, changed chunk IDs, and changed spans.

Its metadata binds `case5_domain_routes_v1`,
`full_macro_concept_routes_v1`, symbol identity schema 3, tokenizer contract
SHA-256 `77e7c934...bc38b8`, and domain schema SHA-256
`522bf7d6...42a7a2a`. Every observed column chunk is ZSTD.

The sealed Megatron bundle exposes dense sidecars as aligned `.bin` files,
ragged graph sidecars as offsets/data pairs, and source-platform offsets/IDs
under `cppmega_source_platform_v1`. Its manifest and source Parquet audit are
linked from `current.json`.

CI CAS binds the exact tokenizer contract and each store's actual
`sidecar_set_sha256`. Its sidecars cover repository/run/workflow/job/step,
actors and runners, language/platform/toolchain/build classification,
commands/build actions, tests, diagnostics, entities/edges, chunk boundaries,
and conservation. Those are not yet exported into eligible Parquet.

## Pinned Inputs

`configs/training_data_status.json` is the reviewed catalog of authoritative
paths. Update that config when a new source generation, sealed bundle, PR scan,
or CI interval supersedes the current one. A missing or unexpected configured
artifact fails the refresh; the reporter does not guess a replacement.
