# CI Output Pipeline — Diagnosis Report

**Status:** Pipeline does NOT run end-to-end. CI data is fetched and enriched
but never reaches the reindexed parquet training set.

**Date:** 2026-07-23
**Scope:** `cppmega` (scripts) + `cppmega.mlx` (outputs / reindexed parquet)

---

## 1. What Exists (the working half)

### Stage A — `scripts/fetch_ci_diagnostics.py`
- Pulls completed GitHub Actions runs for a hardcoded list of 15 repos via `gh api`.
- Extracts compiler/linker/CMake diagnostics from job logs + check-run annotations.
- Writes per-repo JSONL to `outputs/ci_diagnostics/<repo>.jsonl`.
- **Actual output:** `cppmega.mlx/outputs/ci_diagnostics/` has ~170 repo JSONL files
  (≈ 748 unique diagnostics across 35 repos per `_receipt.json`) plus
  `ci_diagnostics_all.parquet` (21 KB — metadata only, NOT tokenized).

### Stage B — `scripts/fetch_ci_logs.py`
- Reads `outputs/ci_diagnostics/*.jsonl`, downloads full job logs via `gh api`,
  filters to relevant build output (errors/warnings ± 3 lines of context).
- Emits enriched documents with `doc_type="diagnostic"`, `domain_kind=0`,
  empty `symbol_identities=[]`, empty `domain_sidecars={}`.
- **Actual output:** `cppmega.mlx/outputs/ci_enriched/ci_logs_enriched.jsonl`
  (1805 lines).

### Stage C — `scripts/join_ci_with_diffs.py`
- Joins CI logs with commit diffs from `extract_cache_case5_shared/<repo>_commits.jsonl`.
- Produces paired "diff + CI output" documents, classifies into a `domain_kind`
  (40–48: COMPILER_DIAGNOSTIC … SANITIZER_OUTPUT).
- **Actual output:** `cppmega.mlx/outputs/ci_enriched/ci_paired_enriched.jsonl`
  (30 lines — extremely small; most CI commits had no matching diff in the cache).

All three stages are **standalone CLI scripts**. They run by hand and stop there.

---

## 2. The Breakpoint — Where the Pipeline Stops

**There is no Stage D.** Nothing consumes `outputs/ci_enriched/*.jsonl`.

The canonical training-data pipeline is `scripts/streaming_conveyor.py`, which
orchestrates exactly TWO streams per repo:

```
.git-preserving tarball
  ├─ CODE stream:    index_project.py --enriched
  │                    → clang_enriched_to_parquet.py (tokenize @65536, materialize)
  │                    → route_by_fit → pack_enriched_rows
  │                    → outputs/reindexed/{1024,2048,4096}/<repo>.parquet
  └─ COMMITS stream: extract_git_history.py → process_commits.py
                       → materialize → route_by_fit → pack
                       → outputs/reindexed_commits/<L>/<repo>_r<start>.parquet
```

The CI stream is **not a third conveyor stream**. No conveyor stage, no
reindexer entry point, and no config (yaml/json/toml) anywhere in `cppmega`
or `cppmega.mlx` references `ci_enriched`, `ci_logs_enriched`,
`ci_paired_enriched`, `fetch_ci_logs`, or `join_ci_with_diffs`. A repo-wide
grep confirms the only references are the scripts themselves.

The reindexed parquet tree (`cppmega.mlx/outputs/reindexed_case5_v7_*_code/`,
2294 parquet files across 5 length buckets) contains zero CI rows.

---

## 3. Why It Cannot Just Be Wired In — Schema Mismatch

Even if a conveyor stage pointed at `ci_enriched/*.jsonl`, the documents would
fail downstream. The materializer
(`cppmega/data/nanochat_pipeline/tokenized_enriched.py`) and the parquet
converter (`scripts/nanochat_data/clang_enriched_to_parquet.py`) expect the
full enriched-document contract that only `tools/clang_indexer/index_project.py`
produces.

### 3.1 Hard blocker — `domain_kind=0` is rejected
`ci_logs_enriched.jsonl` rows carry `domain_kind: 0`. The materializer raises:

```python
# tokenized_enriched.py:618
if domain == DomainKind.UNKNOWN:
    raise ValueError("unknown domain_kind 0 disables delimiter insertion")
```

`fetch_ci_logs.py:223` hardcodes `"domain_kind": 0`. Only the paired joiner
sets a real value (40–48), which is why even the 30 paired docs are the only
ones that could in principle proceed.

### 3.2 Missing enriched-schema fields
A `ci_logs_enriched.jsonl` row has only:
`text, doc_type, repo, commit_hash, filepath, source_doc_id, ci_metadata,
symbol_identities, domain_kind, domain_sidecars`.

Required by the converter / materializer and **MISSING**:

| Field | Status |
|---|---|
| `structure_ids` | MISSING |
| `chunk_boundaries` | MISSING |
| `call_edges`, `type_edges` | MISSING |
| `source_identity_registry`, `source_identity_id` | MISSING |
| `symbol_identity_schema_version` | MISSING |
| `language_info` | MISSING |
| `embedded_domain_spans` | MISSING |
| `ast_depth`, `ast_node_type`, `sibling_index` | MISSING |
| `symbol_ids`, `call_targets`, `type_refs`, `def_use` | MISSING |
| `domain_ids`, `domain_role_ids`, `domain_*_ids` | MISSING |

The converter uses `.get(field, [])` for many of these (so it would not crash
on every column), but `require_project_identity(repo)` plus the
`source_identity_registry` validation, the symbol-identity registry
(`row_registry.require_ids`), and the domain-delimiter contract all assume the
index_project output shape. CI docs have no AST, no graph routes, no symbol
identities — they are plain text with a header comment.

### 3.3 No sidecar pipeline for CI
The sidecar enrichers (AST, call graph, domain routes, prompt-graph indices)
key off `index_project`'s chunk boundaries and symbol IDs. CI docs have none
of that, so even a permissive converter pass would produce rows the
domain-routed dataloader / DSA indexer cannot route
(`cppmega/megatron/domain_route_contract.py` expects `token_diagnostic_edges`,
which only the materializer emits from real `diagnostic_edges`).

---

## 4. Secondary Failure Modes (confirmed)

| Mode | Status |
|---|---|
| Missing GitHub credentials | Not the blocker. `gh api` worked: 1805 logs fetched, 170 repo JSONLs written. Some logs return 410 (expired >90 days) — counted as `expired` in the fetcher, not fatal. |
| Missing conveyor stage definition for CI | **CONFIRMED — primary blocker.** No stage, no config, no entrypoint. |
| Schema mismatch (jsonl → parquet columns) | **CONFIRMED.** `domain_kind=0` raises; ~12 required enriched columns absent. |
| CI not going through sidecar pipeline | **CONFIRMED.** No AST / graph routes / symbol identities; `domain_sidecars={}`. |
| Volume too small to matter | **CONFIRMED — amplifier.** 1805 + 30 docs ≈ a few hundred K tokens. The fetcher's own estimate prints "~N steps (bs=192 seq=1024)"; at this size it is rounding error vs the 2294-shard code corpus. Even fully wired, CI would be <0.1% of training data without a much larger fetch. |
| Join sparsity | `ci_paired_enriched.jsonl` is only 30 lines because CI commit SHAs rarely match the `extract_cache_case5_shared` diff cache (different repo sets / time windows). |

---

## 5. What Is Needed to Fix It

In rough order of effort:

1. **Fix `domain_kind` in `fetch_ci_logs.py`** (trivial, ~10 min).
   Replace the hardcoded `0` with a real `DomainKind` (reuse
   `join_ci_with_diffs.classify_ci_output`, default
   `DomainKind.BUILD_DIAGNOSTIC = 41`). Without this, every row raises.

2. **Write a CI ingestion stage** (medium, ~1–2 days).
   A new script (e.g. `scripts/nanochat_data/ci_enriched_to_parquet.py` or a
   third stream in `streaming_conveyor.py`) that:
   - reads `outputs/ci_enriched/*.jsonl`,
   - synthesizes the missing enriched contract for plain-text diagnostic docs
     (single whole-doc `chunk_boundaries` span with a DIAGNOSTIC `kind`,
     flat `structure_ids`, empty edge tables, a valid
     `source_identity_registry` built from `repo` + `source_doc_id`,
     `symbol_identity_schema_version`, `language_info`),
   - tokenizes via the same `clang_enriched_to_parquet` machinery,
   - routes/packs into `outputs/reindexed_diagnostics/<L>/ci_*.parquet`.
   The closest existing template is the `build` doc path in
   `index_project.py` (single BUILD_KIND span, no call/type graph) — diagnostic
   docs need the same "whole-doc one span" treatment with a diagnostic kind.

3. **Decide the training contract for diagnostic docs** (design, ~0.5 day).
   Either (a) emit them as first-class `doc_type="diagnostic"` rows with a
   real `token_diagnostic_edges` sidecar so the domain-routed dataloader can
   consume them, or (b) downscope to plain text-in / text-out and bypass the
   graph-route requirements. (a) is consistent with the existing
   `DomainKind.COMPILER_DIAGNOSTIC..SANITIZER_OUTPUT` enum and the
   `diagnostic_edges` family already wired through the megatron patches.

4. **Scale the fetch** (operational, ~0.5 day + API budget).
   1805 docs is too small to move training. Expand the repo list (the
   diagnostics dir already has ~170 repos scraped — only a fraction had
   fetchable, non-expired logs), page beyond `per_page=10` runs, and backfill
   before logs hit the 90-day GitHub expiry. Re-run the join against a
   matching diff cache to grow the paired set beyond 30.

5. **Add a manifest / config entry** (small).
   Register the CI stream in whatever drives the conveyor runs (currently the
   conveyor is invoked directly via CLI; there is no yaml stage list — the
   "config" is the conveyor's argparse defaults). Document the new stage in
   `docs/data_preparation.md`.

---

## 6. Estimated Effort

| Item | Effort |
|---|---|
| `domain_kind` fix in fetcher | ~10 min |
| CI ingestion stage (jsonl → enriched parquet) | 1–2 days |
| Diagnostic training-contract decision + sidecar wiring | 0.5–1 day |
| Fetch scale-up + join backfill | 0.5 day + API/rate-limit wall time |
| Config/docs + conveyor integration | 0.5 day |
| **Total to first CI rows in reindexed parquet** | **~3–4 engineer-days** |
| **Total to a meaningfully-sized CI stream** | **+ fetch wall-time / API budget** |

---

## 7. One-Line Summary

The fetchers work; the breakpoint is that **no conveyor/reindexer stage exists
for CI data**, and the enriched CI documents are **schema-incompatible** with
the existing tokenizer/materializer (`domain_kind=0` raises; ~12 enriched
columns absent; no AST/graph sidecars) — and at 1835 total docs the volume is
too small to justify a quick hack, so the data has simply never been bridged
into `outputs/reindexed*/`.
