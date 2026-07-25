---
phase: lane-2-landed-parquet-truth
reviewed: 2026-07-13T13:28:23Z
depth: deep
files_reviewed: 11790
files_reviewed_list:
  - "/Volumes/external/sources/cppmega.mlx/outputs/reindexed_macro_routes_v1_20260710_135335_code/{1024,2048,4096,8192,16384}/*.parquet (1,441 manifest-complete files)"
  - "/Volumes/external/sources/cppmega.mlx/outputs/reindexed_macro_routes_v1_20260710_135335_commits/{1024,2048,4096,8192,16384}/*.parquet (10,330 manifest-complete files)"
  - /Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json
  - /Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_reservations.json
  - /Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/code.restart.log
  - /Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/commits.restart_parallel.log
  - /Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py
  - /Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/nanochat_pipeline/packed_rows_schema.py
  - /Volumes/external/sources/cppmega.mlx/scripts/audit_sidecar_parquet.py
  - /Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py
  - /Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py
  - /Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py
  - /Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py
  - /Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py
  - /Volumes/external/sources/cppmega/cppmega/megatron/structure_dataset_patch.py
  - /Volumes/external/sources/cppmega/scripts/data/verify_tokenizer_contract.py
  - /Volumes/external/sources/cppmega.mlx/scripts/_validate_reindexed.py
  - /Volumes/external/sources/cppmega/scripts/data/verify_side_channel_shapes.py
findings:
  critical: 8
  warning: 5
  info: 0
  total: 13
status: issues_found
---

# Lane 2: Landed Parquet Truth — Deep Review

**Reviewed:** 2026-07-13T13:28:23Z  
**Depth:** deep  
**Status:** issues_found — **do not start Megatron conversion**  
**Code snapshots:** `cppmega@8eb95bd344dadabc2b0c677b83bc7008f2e8da18` and `cppmega.mlx@dd2b703bd69d197569f2faa41b51cbd5c83f0829`, both with concurrent uncommitted changes that were not made or altered by this review.

## Confirmed bugs and operational blockers

### CR-01 — Critical / BLOCKER: 810,921 landed rows train across real document boundaries

**Files:**

- `/Volumes/external/sources/cppmega.mlx/outputs/reindexed_macro_routes_v1_20260710_135335_code/1024/4.4bsd-lite2.parquet:row 38`
- `/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1277-1284`
- `/Volumes/external/sources/cppmega.mlx/scripts/audit_sidecar_parquet.py:381-448`
- `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:251-270,890-903`

**Failure mode:** the landed producer used a provenance-stable file/document ID as the packed `doc_ids` boundary channel. Distinct logical documents from the same file therefore shared an ID. `loss_mask` was derived from that collapsed channel, so the final token of the first logical document remained trainable against the first token of the next document. Megatron conversion only checks side-channel length and writes the bad mask unchanged.

**Evidence:** exhaustive scalar checks over all 2,631,463 manifest-complete rows found 810,921 violations of the independent invariant `trained_token_count == valid_token_count - num_docs`. Every violation is in `code/1024`; 290 of 293 files are affected. The affected rows contain 792,987,820 valid tokens and expose 3,719,382 excess trainable targets. The cited row has `valid=1024`, `num_docs=2`, `trained=1023`, but the correct trained count is 1022. Delta histogram: `{1: 95999, 2: 66682, 3: 76430, 4: 95977, 5: 140295, 6: 181087, 7: 146784, 8: 7667}`.

The current uncommitted packer now assigns row-local IDs at lines 1277-1284 and the current uncommitted audit derives boundaries independently at lines 381-448. Those edits document the original defect but do not repair already-landed parquet. All manifest-complete files were produced with the old semantics. Physical, unmanifested files written after the packer changed already use the new semantics: `paddle_r13020.parquet` and `valgrind_r5773.parquet` have zero sampled boundary errors, while manifest-complete `aria2_r0.parquet` has 404/406 rows with old `doc_ids` semantics. The shared output root is therefore also mixed-version despite retaining one schema metadata value.

**Focused fix/test:** quarantine and repack all 290 affected `code/1024` files, or repack the entire frozen manifest generation for one producer-version contract. Before promotion, exhaustively derive expected `doc_ids` and `loss_mask` from `source_doc_token_lengths`; require exact array equality plus `trained == valid - num_docs` for every row. Add a regression fixture with two functions from the same filepath/stable provenance ID and assert the boundary token has `loss_mask=0`.

### CR-02 — Critical / BLOCKER: conversion globs failed and interrupted files as if they were landed

**Files:**

- `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:139-164`
- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:471-478,566-569`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151678-151679`

**Failure mode:** commit bucket files are copied into final bucket directories one length at a time, before the unit's dedup stage is promoted and before `_done.json` marks the unit complete. The converter selects every `*.parquet` in a bucket and has no manifest allowlist. A later bucket failure or process interruption therefore leaves valid-looking final files that conversion silently treats as ready.

**Evidence:** the final directories contain 11,810 parquet files, but only 11,771 belong to manifest-complete units. The 39 extras are:

- 35 unmanifested files left by the interrupted commit run: five buckets each for `paddle_r11020`, `r11520`, `r12020`, `r12520`, `r13020`, `valgrind_r5773`, and `r5983`;
- four failed residuals for `radare2_r0` at 1024/2048/4096/8192. Its 16384 pack failed, so the unit was never complete.

No filename marked `.tmp`, `.partial`, or `.incomplete` exists in the final roots, and none of these files is zero-byte. Filename/Parquet validity therefore cannot distinguish them from landed data.

**Focused fix/test:** convert from an immutable manifest-generated allowlist, not a directory glob. Land each unit under a staging generation and atomically promote all of its buckets only after all packs, recompression, dedup promotion, and manifest write succeed. Inject a 16384 pack failure in a test and assert that zero files from the unit are selectable by conversion.

### HI-01 — High / BLOCKER: required token source/platform channels are universally zero

**Files:**

- `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/nanochat_pipeline/packed_rows_schema.py:184-200`
- `/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1308-1314`
- `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:64-85`
- `/Volumes/external/sources/cppmega/cppmega/megatron/structure_dataset_patch.py:28-47,71-89`

**Failure mode:** absent token-aligned `token_platform_ids` and `token_source_doc_ids` are silently filled with zero. The converter emits both by default, and the training bridge consumes them as real channels, so training receives a syntactically valid but semantically empty source/platform signal.

**Evidence:** footer statistics over all 11,771 ready files show min=max=0 for both columns across all 5,351,270,400 capacity slots. In the same rows, 8,783,395 source documents carry nonzero source-level platform provenance covering all 4,132,747,235 valid tokens; 8,740,456 source docs have multiple platform IDs. This is not an empty-provenance corpus.

**Focused fix/test:** define the token-level mapping explicitly. For source identity, assign each token its row-local source-document ID. For platform, define how a token maps when a source doc has multiple platform IDs (single canonical ID, multihot sidecar, or no platform embedding). Fail packing/conversion if a required channel is all fallback while corresponding source provenance is populated. Test a two-source packed row with distinct and multi-platform provenance end-to-end through the Megatron batch bridge.

### HI-02 — High / BLOCKER: SHA-based PR joins retain discussion text but lose the PR number

**Files:**

- `/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:135-163`
- `/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:1938-1946`
- `/Volumes/external/sources/cppmega.mlx/outputs/reindexed_macro_routes_v1_20260710_135335_commits/1024/aria2_r0.parquet:row 52`

**Failure mode:** `PRStoreLookup.attach()` can resolve a PR by commit SHA, but on a hit it writes only `record["pr_discussion"]`. It does not copy the found PR number/title into the record. Later serialization parses `pr_number` from the unchanged source record, yielding null while discussion flags and text are populated.

**Evidence:** all-row commit scan found 38,418 source docs with PR discussion. Only 24,377 carry a PR number; 14,041 discussion-bearing source docs across 14,038 rows have null PR numbers. Those missing-number discussions contain 13,602,524 characters and 312,353 lines. `aria2_r0.parquet:row 52` has discussion text (474 chars, 9 lines) and null PR number; the live PR store maps commit `80df...` to `aria2/aria2#1467`.

**Focused fix/test:** on lookup hit, copy the canonical `pr_number` and title from `rec` before rendering discussion. Add a SHA-only lookup test that begins without `record["pr_number"]` and asserts number, title, discussion, and packed `source_pr_numbers` all survive materialization.

### HI-03 — High / BLOCKER: retry paths bypass the shared global dedup contract

**Files:**

- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1526-1527,1552-1568`
- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:2323-2339`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/code.restart.log:5,30,1657,1666`

**Failure mode:** adaptive and resume retries default to `dedup_db=None` and `dedup_near=False`. Successful retry output is landed without checking or claiming hashes in the shared corpus dedup database.

**Evidence:** `Xbox Live Source::code` and `mesa::code` were explicitly retried with isolated dedup and then marked done. Together they contribute 160,908,019 ready valid tokens (98,659,692 + 62,248,327) outside the global claim path. The global DB contains 4,015,842 exact hashes and 8,656,992 chunk claims, but cannot prove those two successful units were compared against them. This review did not infer duplicate content from repeated shard-local source IDs; the confirmed defect is that global duplicate exclusion was bypassed, so corpus-wide uniqueness is unproven.

**Focused fix/test:** preserve a per-repo staging DB for the low-worker retry, then promote against the same global dedup DB transactionally. Seed the global DB with a duplicate from a retry fixture and assert the retry rejects it and records no unclaimed output.

### HI-04 — High / BLOCKER: code-stream overlong drops are not durable in the manifest

**Files:**

- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:351-420`
- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:1077-1095`
- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:550-599`

**Failure mode:** shared routing writes `dropped_overlong.json` under an ephemeral temp directory. The commit caller lifts it into the manifest; the code caller ignores it and returns only per-length counts. Once temp is removed, the code exclusion is recoverable only by scraping logs.

**Evidence:** logs for all 295 manifest-complete code repos report 7,224 dropped docs / 200,412,883 tokens, but none of the code manifest entries carries `dropped_overlong`. Commit entries durably record 13,250 docs / 346,087,005 tokens. Total accepted-after-dedup but excluded-before-fixed-bucket data is 20,474 docs / 546,499,888 tokens. Ready + dropped accepted tokens are 2,682,292,863 for code and 1,996,954,260 for commits (4,679,247,123 total), while the manifest alone reports only ready tokens.

**Focused fix/test:** make `route_by_fit` return paths plus drop statistics and require both callers to persist them. Reconcile `accepted_tokens == ready_valid_tokens + dropped_overlong_tokens` per unit and corpus. Test both code and commit callers with one >16,384-token document.

### HI-05 — High / BLOCKER: platform cardinality can fail only the last bucket after earlier buckets land

**Files:**

- `/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:542-556`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151678-151679`

**Failure mode:** packing chooses documents without constraining the union of platform IDs, then `_merged_platform_ids_for_docs()` hard-fails when the completed row has more than `MAX_PLATFORM_IDS=20`. Because buckets are appended sequentially, the failure can occur at 16384 after four shorter files have already reached final directories.

**Evidence:** `radare2::r0` failed `pack_16384` with 22 unique IDs. Its 1024/2048/4096/8192 files remain as the four failed residuals counted in CR-02.

**Focused fix/test:** include platform-union cardinality in bin feasibility or use a versioned variable-length/multihot representation. Test a best-fit pack whose documents individually fit but whose union has 21 IDs; it must split deterministically before any final file is promoted.

### HI-06 — High / BLOCKER: the run is live, incomplete, and still mutating the conversion input roots

**Files:**

- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151602-151681`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_reservations.json:2-107`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/code.restart.log:2428,2754-2755`
- `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/commits.restart_parallel.log:15123,15198-15199`

**Failure mode:** the previous code conveyor checkpointed after SIGINT at 13:16:31Z and the previous commit conveyor was force-stopped on a second signal at 13:17:06Z. Another actor restarted both streams at local 15:24 while this review was being finalized. The converter input directories are therefore live mutable state, `_done.json` still contains 16 failures, and conversion cannot be tied to a stable corpus generation.

**Evidence:** at 13:28:23Z, active conveyor PIDs were 77460 (code) and 78298 (commits). The manifest remained at 2,401 done / 16 failed, while `_reservations.json` had two newly active code units, `dealii::code` and `dragonflybsd::code`; the commit process was still preparing work and had not yet acquired a range reservation. The directories still contained 11,810 physical parquet files, including the 39 unmanifested residuals from CR-02. The active packer/audit source had also changed at local 14:52, so resumed subprocesses can emit new row-local boundary semantics beside the old landed semantics without changing the Parquet schema-version metadata.

**Focused fix/test:** let the authorized run controller stop or finish the active processes; do not clear or promote reservations by hand. Freeze producer commits before resume, reconcile every interrupted unit to staged dedup state and outputs, then require zero running producers, zero reservations, zero failures, and an unchanged manifest/output hash across the full preflight before conversion.

## Design and robustness gaps

### ME-01 — Medium / WARNING: conversion writes directly to final prefixes and leaves partial products on failure

**File:** `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:617-636,698-704,711-728,834-845,923-932`

**Issue:** both writer backends open final `.bin` and sidecar paths before validating all rows. Exception handling closes graph/side writers but does not unlink or invalidate the already-written final prefix. A bad late row can leave a plausible `.bin` and sidecars without a valid complete manifest.

**Focused fix/test:** write every output under a unique temporary prefix, fsync/close/validate all file sizes and the JSON receipt, then atomically publish the generation. Inject a failure in the final shard and assert no final-prefix file exists.

### ME-02 — Medium / WARNING: converter validation does not enforce the full graph contract

**File:** `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:460-521`

**Issue:** token-coordinate edge triples and chunk token offsets receive upper-bound checks, but `edge_pairs` are written at lines 472-480 without verifying endpoints against the row's chunk count. The converter also does not require equal lengths for starts/ends/kinds/dep-level arrays or enforce `0 <= start < end <= valid_token_count`. A future malformed shard can become a Megatron graph sidecar and fail later in training.

**Evidence:** this is a design gap, not a current landed-data defect. The all-row audit found the active 11,771-file snapshot clean for these invariants: 13,594,174 chunks; 644,450 call edges; 452,370 type edges; and 17,949,252 token-coordinate edges (15,889,000 code domain + 1,488,023 commit domain + 572,229 code build); every checked endpoint and span was in bounds.

**Focused fix/test:** validate all four chunk arrays as one contract, compute `nchunks`, and reject call/type endpoints outside `[0,nchunks)`. Add malformed fixtures for mismatched arrays, `start == end`, and endpoint `== nchunks`.

### ME-03 — Medium / WARNING: code units are marked done before deferred recompression succeeds

**File:** `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1401-1414,2375-2378,3191-3206`

**Issue:** with background recompression enabled, `run_code_half` submits jobs and returns; the caller marks the unit done immediately. Recompression failures surface only during global shutdown. A consumer watching `_done.json` can ingest a file still being rewritten or a done unit whose recompression later fails.

**Evidence:** no current ready-file codec defect was found: every one of 2,511,544 Parquet column chunks in the 11,771-file audit is ZSTD. This remains a publication-order gap.

**Focused fix/test:** include recompression futures in the unit transaction and mark done only after all complete. Block a recompressor in a test and assert the unit remains non-done until release; inject a failure and assert no done entry is written.

### ME-04 — Medium / WARNING: standalone stream drivers return success when failures remain

**Files:**

- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:1424-1432`
- `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:936-947`

**Issue:** each driver returns zero whenever it processed at least one unit/range, even if `manifest.failed` is nonempty. Automation can treat a partial corpus as successful. The current conveyor source has corrected semantics at `streaming_conveyor.py:3254-3260`, but the two direct entry points still do not.

**Focused fix/test:** return nonzero whenever unresolved failures remain. Test a run with one success and one failure and assert nonzero exit plus an explicit partial-run summary.

## Stale documentation/tooling

### LO-01 — Low / WARNING: legacy validators do not point at or understand the active corpus

**Files:**

- `/Volumes/external/sources/cppmega.mlx/scripts/_validate_reindexed.py:2-6,15-16`
- `/Volumes/external/sources/cppmega/scripts/data/verify_side_channel_shapes.py:2-9,32-51`

**Issue:** `_validate_reindexed.py` hardcodes the old `outputs/reindexed` root and only 1024/2048/4096. `verify_side_channel_shapes.py` expects legacy `token_ids`, omits current `input_ids`, `loss_mask`, `doc_ids`, source-ID, role/domain/entity/scope/confidence channels, and samples rather than freezing a manifest generation. Running either as documented can provide false reassurance for this corpus.

**Focused fix/test:** retire them in favor of the manifest-aware current audit, or update them to consume the packed schema registry and all five bucket lengths. Add a smoke test against one current macro-routes shard and reject raw-glob scope.

## Exact frozen inventory

The exhaustive integrity scan covers the 11,771 files selected by the `_done.json` generation at local `2026-07-13T15:16:30+0200`, SHA-256 `319f2f6ffb416cb14d716346fb2bef47a1248510a1a8d4ea3f911279767e5199`. Live state was refreshed at 13:28:23Z after both streams restarted: `_done.json` was unchanged, and the active-reservations SHA-256 was `bbab6ee4c0eff4e6568d3623dc8ae2a09c3adc2b3c7cf52ba25bd4c3476c5b2b`.

### Manifest-complete, conversion-eligible files only

| Source | Bucket | Files | Rows | Valid tokens | Pad tokens | Capacity tokens |
|---|---:|---:|---:|---:|---:|---:|
| code | 1024 | 293 | 1,509,110 | 1,495,779,442 | 49,549,198 | 1,545,328,640 |
| code | 2048 | 293 | 222,445 | 315,334,535 | 140,232,825 | 455,567,360 |
| code | 4096 | 289 | 100,355 | 282,151,520 | 128,902,560 | 411,054,080 |
| code | 8192 | 289 | 39,880 | 223,059,383 | 103,637,577 | 326,696,960 |
| code | 16384 | 277 | 14,863 | 165,555,100 | 77,960,292 | 243,515,392 |
| **code subtotal** | | **1,441** | **1,886,653** | **2,481,879,980** | **500,282,452** | **2,982,162,432** |
| commits | 1024 | 2,073 | 294,972 | 216,519,785 | 85,531,543 | 302,051,328 |
| commits | 2048 | 2,074 | 218,695 | 314,170,230 | 133,717,130 | 447,887,360 |
| commits | 4096 | 2,074 | 129,649 | 371,189,937 | 159,852,367 | 531,042,304 |
| commits | 8192 | 2,069 | 70,160 | 398,656,066 | 176,094,654 | 574,750,720 |
| commits | 16384 | 2,040 | 31,334 | 350,331,237 | 163,045,019 | 513,376,256 |
| **commits subtotal** | | **10,330** | **744,810** | **1,650,867,255** | **718,240,713** | **2,369,107,968** |
| **grand total** | | **11,771** | **2,631,463** | **4,132,747,235** | **1,218,523,165** | **5,351,270,400** |

Manifest units: code has 295 units with files and 3 explicit no-trainable-source skips; commits has 2,075 units with files and 28 done-without-output units (empty-after-dedup or no-git). Total done keys: 2,401.

Valid-token accounting is internally additive (`valid + pad == capacity`) for every bucket, but trained-token accounting is not semantically correct because of CR-01:

| Source | Expected trained (`valid - source docs`) | Manifest/parquet trained | Excess |
|---|---:|---:|---:|
| code | 2,473,847,685 | 2,477,567,067 | 3,719,382 |
| commits | 1,650,116,155 | 1,650,116,155 | 0 |
| **total** | **4,123,963,840** | **4,127,683,222** | **3,719,382** |

### Not conversion-eligible

| State | Units/files | Detail |
|---|---:|---|
| Failed manifest units | 16 units | Includes `radare2_r0` partial pack, two interrupted Open Watcom code aliases, `paddle_r13520`, and 12 prior failures. |
| Active reservations | 2 units | `dealii::code` and `dragonflybsd::code`; both newly acquired by the restarted code conveyor and had no final files at snapshot. |
| Unmanifested residual physical files | 39 files | 35 from the interrupted commit run plus failed `radare2_r0` at 1024/2048/4096/8192. |
| Physical parquet total | 11,810 files | Exactly 11,771 ready + 39 non-ready. |

## Exhaustive landed-shard checks that passed

- Read all 11,771 manifest-complete Parquet footers: 5,891,666,763 bytes, no unreadable file, no zero-byte final, one exact schema, and metadata `cppmega.macro_routes_version=full_macro_concept_routes_v1` everywhere.
- All 2,511,544 column chunks use ZSTD. No final-root or conveyor-temp filename matched `.tmp`, `.partial`, or `.incomplete` at freeze time.
- Every token-aligned list leaf has exactly `rows * bucket_length` values. `input_ids` range is 0..65,532, inside the 65,536 tokenizer contract.
- Current tokenizer verification passes both tokenizer artifacts: vocab 65,536, max added ID 7,199, all special/reserved IDs exact, and 23 START/END delimiter-role pairs exact.
- Scanned every ready row for source-array cardinality, `sum(source_doc_token_lengths)==valid_token_count`, PR aggregate flags/counts, stable-ID rollups, parent/merge fields, chunk-array alignment, strict chunk bounds, call/type endpoints `< nchunks`, token-edge endpoints `< valid`, and change-ID/span alignment. Apart from the findings above, these checks passed.
- Graph totals scanned: code 9,466,871 chunks / 515,216 call / 386,469 type / 15,889,000 domain / 572,229 build edges; commits 4,127,303 chunks / 129,234 call / 65,901 type / 1,488,023 domain edges. Shell, diagnostic, and cross-domain edges are zero in this snapshot. Commit changed chunks total 1,045,679; code changed chunks are zero.
- Repeated `source_doc_id` values across bucket files were not classified as duplicate content because the IDs restart in independently packed shards. No content duplicate was proven by that signal. Corpus-wide duplicate exclusion remains unproven specifically because of HI-03.

## Conversion gate

Megatron conversion is blocked until all of the following are true:

1. Repack/quarantine the 290 corrupt `code/1024` shards and pass the independent exhaustive boundary oracle.
2. Freeze one producer version and one immutable manifest allowlist; exclude all 39 non-ready physical files.
3. Resolve and populate, or explicitly remove/version, the universally zero token source/platform channels.
4. Repair/rebuild PR number linkage if PR provenance is a required training contract.
5. Reconcile isolated-dedup units and durable overlong accounting.
6. Finish or stop the active runs through their controller, resolve all 16 failures and every reservation through normal resume/recovery, then freeze at zero failed/reserved units and zero producer processes.
7. Make conversion publication atomic and run a full manifest-aware preflight before publishing any Megatron prefix.

---

_Reviewed: 2026-07-13T13:28:23Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
