**Findings**
Critical: none confirmed.

High:
1. **[Confirmed bug] Code outputs can be marked done before background recompression is proven.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1388](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1388), [:1404](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1404), [:2359](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:2359), [:3176](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:3176).  
Failure mode: dedup is promoted, recompress is submitted, then the code unit is checkpointed. If background recompress later fails, resume skips the already-done code unit, leaving an output that did not satisfy the zstd-max contract. Sync recompress failure has the same ordering problem after dedup promotion.  
Evidence: live run used `--background-code-recompress` in [/Volumes/external/sources/cppmega.mlx/outputs/conveyor/progress_macro_full_expansion_regen_20260710_053845_code_v4.jsonl:1](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/progress_macro_full_expansion_regen_20260710_053845_code_v4.jsonl:1).  
Focused test/fix: inject a failing `recompress_zstd_max`; assert no `repo::code` done mark and no promoted dedup/output, or record recompress as its own manifest gate before code completion.

2. **[Confirmed bug] Commit record failures are swallowed and ranges can complete with skipped data.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2289](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2289), [:2299](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2299), [:2549](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2549), [:2589](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2589).  
Failure mode: any `process_record` exception increments `parse_errors` and continues; missing input files only warn; CLI returns `0`. A range with one good record and many failed records is treated as successful upstream.  
Evidence: `streaming_conveyor.py` marks deferred ranges done after promotion at [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:2087](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:2087), so partial commit output becomes durable.  
Focused test/fix: fixture with one valid record plus one record that raises in `process_record`; require nonzero exit unless `--allow-parse-errors` is explicit and manifest records the loss.

Medium:
3. **[Design gap] Batched deferred commit dedup has a cross-range duplicate leak window.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1933](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1933), [:1980](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1980), [/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/dedup_store.py:646](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/dedup_store.py:646), [:700](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/dedup_store.py:700), [:991](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/dedup_store.py:991).  
Failure mode: concurrent staged ranges only see committed global rows and their own stage. Sibling stages in the same promotion batch cannot suppress each other, so duplicate exact/chunk/near docs can already be written before promotion merges counts.  
Focused test/fix: two deferred ranges emit identical chunk tokens/minhashes; assert second output is suppressed or promotion fails and removes output before `mark_done`.

4. **[Design gap] Code retry disables global dedup entirely.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1557](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1557), [/Volumes/external/sources/cppmega.mlx/tests/test_streaming_conveyor_progress.py:407](/Volumes/external/sources/cppmega.mlx/tests/test_streaming_conveyor_progress.py:407).  
Failure mode: after retryable index failure, the single-worker retry runs with `dedup_db=None` and `dedup_near=False`; duplicates can leak across code/commit streams or prior repos. Current tests lock this in.  
Focused test/fix: seed global dedup with a known exact/chunk claim, force retry, assert duplicate is not emitted. Keep exact/chunk dedup on retry; isolate only the memory-heavy near path if needed.

5. **[Design gap] PR enrichment is documented as read-only but opens a writable SQLite connection.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:89](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:89), [:104](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:104), [/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/pr_store.py:109](/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/pr_store.py:109), [:117](/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/pr_store.py:117), [:119](/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/pr_store.py:119).  
Failure mode: enrichment workers can create WAL/shm state, run schema DDL, contend with ingest/export, or fail on a truly read-only PR store.  
Focused test/fix: add `readonly=True` URI `mode=ro` connection path that skips WAL/schema writes; assert no `-wal`/`-shm` files are created.

6. **[Design gap] PID-only reservations can strand restart work after forced exit or PID reuse.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1040](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1040), [:1108](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1108), [/Volumes/external/sources/cppmega.mlx/outputs/conveyor/_reservations.json:3](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/_reservations.json:3), [:39](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/_reservations.json:39).  
Failure mode: current live artifact still has `dealii::code` active from `2026-07-10T15:01:44`; progress shows it started at [/Volumes/external/sources/cppmega.mlx/outputs/conveyor/progress_macro_full_expansion_regen_20260710_053845_code_v4.jsonl:212](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/progress_macro_full_expansion_regen_20260710_053845_code_v4.jsonl:212). I could not verify PID liveness because `ps` is sandbox-blocked. If the PID was reused, acquire treats an unrelated process as holder and skips the unit.  
Focused test/fix: persist process start time/heartbeat, validate command identity, and expire old claims safely on startup.

Low:
7. **[Design gap] Published parquet copies are not atomic.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:990](/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:990), [/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:476](/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:476), [/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1793](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1793).  
Failure mode: a kill during copy/write can leave a partial final parquet. Manifest usually prevents marking it done, but external readers/audits can see torn files.  
Focused test/fix: copy/write to sibling temp, stats-read temp, then `replace()`; clean `*.tmp.parquet` on startup.

8. **[Stale doc] Signal checkpoint text claims zero work lost, but default cleanup removes interrupted work root.**  
Path/line: [/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:3184](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:3184), [:3237](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:3237), [/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_full_expansion_regen_20260710_053845_code_v4.log:1082](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_full_expansion_regen_20260710_053845_code_v4.log:1082).  
Failure mode: operators may expect exact zero-rework resume; actual default removed `dotnet-runtime` partial work and retained only extract cache.  
Focused test/fix: update log/docs or make default signal behavior match the message. Current signal test uses `--retain-partial-work` at [/Volumes/external/sources/cppmega.mlx/tests/test_conveyor_signal_checkpoint_resume.py:117](/Volumes/external/sources/cppmega.mlx/tests/test_conveyor_signal_checkpoint_resume.py:117), so it misses the default mismatch.

**Regression-Test Integrity**
Existing loss-mask/doc boundary coverage is good: [/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1220](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1220) zeros cross-doc/pad targets, and [/Volumes/external/sources/cppmega.mlx/tests/test_pack_enriched_rows.py:172](/Volumes/external/sources/cppmega.mlx/tests/test_pack_enriched_rows.py:172) asserts `[1, 0, 1, 0, 0]`. Remaining gap: direct packer can create overlong rows via `row_length = max(...)` at [/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1375](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1375); conveyor route-by-fit drops overlong rows in the live log at [/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_full_expansion_regen_20260710_053845_code_v4.log:1040](/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_full_expansion_regen_20260710_053845_code_v4.log:1040).

Recent 7-day reviewed changes in `cppmega.mlx`: `dd2b703` and `1dfbbd9` touched this lane. I found no material Lane 1 pipeline changes in `/Volumes/external/sources/cppmega`.

Verification was read-only source review plus small live artifact reads (`_done.json`, `_reservations.json`, progress/log tails). I did not run tests or alter files because the workspace is read-only and you asked not to disturb data processes. Changed files: none. Remaining risks are skipped commit data, dedup leaks, non-atomic parquet publication, PR-store lock contention, and restart ambiguity around the current `dealii::code` reservation.

