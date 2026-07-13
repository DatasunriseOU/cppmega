**Findings**
Critical - confirmed bug: adaptive commit resume can falsely treat unfinished repos as complete. `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:631` accepts any last range shorter than `range_size` as proof of EOF, and `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1288` uses that to skip commit staging. That is invalid because `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:691` plans adaptive byte-bounded ranges. Evidence: `paddle::r10552` is done at `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:93478`, but later ranges are active/started at `progress.jsonl:5025` and `_reservations.json:99`; same pattern for `valgrind::r5499` at `_done.json:151127` followed by `progress.jsonl:5042` and `_reservations.json:375`. Focused fix/test: persist explicit commit-count/EOF or planned-range-count proof, and add a regression where a non-final adaptive short range must return `None`.

Critical - confirmed bug: failed manifests can still produce success exit status. `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:3243` returns success when `processed_repos > 0` even if `manifest.failed` is non-empty; `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:1432` has the same pattern. Evidence: current manifest has failures starting at `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151602`. Failure mode: a wrapper can proceed to Megatron conversion from an incomplete run. Focused fix/test: return non-zero whenever `manifest.failed` is non-empty; cover `processed > 0` plus one failed unit.

High - confirmed bug: platform provenance cap aborts a valid commit pack. `/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:551` rejects merged `platform_ids` above `MAX_PLATFORM_IDS`; `/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:552` raises. Evidence: `radare2::r0` failed at `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151663` with `22 unique IDs; MAX_PLATFORM_IDS=20`. Focused fix/test: construct a 22-platform packed row and either support it or enforce an explicit overflow encoding before packing.

High - design gap: conversion currently trusts directory contents, not manifest truth. `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:147` globs all `*.parquet`; `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:164` returns all of them for `split=all`. Evidence: commit buckets have 24 manifest-orphan parquet files, e.g. `paddle_r11020.parquet` and `radare2_r0.parquet`, adding 6,018,443 physical valid tokens beyond manifest-backed counts. Focused fix/test: conversion should require a manifest allowlist or a staged clean directory; add a preflight that fails on extra/missing files.

Medium - confirmed bug: `dedup_exhausted` is classified by reindex but not skipped by conveyor. `/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:146` returns `dedup_exhausted`, while `/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1445` only skips `no_trainable_source`. Evidence: `gsl::code` is failed at `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_done.json:151623`; existing skip test only covers no-trainable-source at `/Volumes/external/sources/cppmega.mlx/tests/test_streaming_conveyor_progress.py:509`. Focused fix/test: add a `dedup_exhausted` skip policy test or fail-loud with a distinct exclusion manifest.

Medium - design gap: current artifact tree is not quiesced. Active reservations begin at `/Volumes/external/sources/cppmega.mlx/outputs/conveyor/macro_routes_v1_20260710_135335/_reservations.json:2`, with active examples at lines `3`, `329`, and `421`; `progress.jsonl:5054` shows `paddle::r13520` started after several orphan files already exist. Failure mode: conversion can race a still-mutating output tree. Focused fix/test: require empty reservations plus a final green marker before conversion.

Medium - design gap: standalone PR bucket is empty by policy. `/Volumes/external/sources/cppmega.mlx/tests/test_verified_sidecar_manifest_selection.py:41` asserts default standalone PR exclusion, and line `393` records `outputs/reindexed_pr` as excluded. Evidence: live `outputs/reindexed_pr` has 0 parquet files; integrated commit PR fields exist and physical commit rows include PR metadata, but standalone PR ready count is zero. Focused fix/test: decide whether Megatron conversion needs standalone PR; if yes, make it explicit opt-in and gate on nonzero PR buckets.

Low - stale docs: converter comments still describe legacy shard naming. `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:142` describes `shard_00000.parquet`/`val_shard.parquet`, not macro-route repo/range buckets; `/Volumes/external/sources/cppmega/scripts/data/prepare_format_megacpp.py:178` says “max observed 65529”, while sampled live max was 65475. Fix: update docs to say macro-route conversion must use manifest-backed bucket directories.

**Ready Counts**
Manifest-backed ready counts, not physical directory counts:

```text
source   bucket  ready_files  rows       valid_tokens
code     1024    293          1,509,110  1,495,779,442
code     2048    293          222,445    315,334,535
code     4096    289          100,355    282,151,520
code     8192    289          39,880     223,059,383
code     16384   277          14,863     165,555,100
commits  1024    2,073        294,972    216,519,785
commits  2048    2,074        218,695    314,170,230
commits  4096    2,074        129,649    371,189,937
commits  8192    2,069        70,160     398,656,066
commits  16384   2,040        31,334     350,331,237
pr       all     0            0          0
```

Totals: code `1,441` files / `2,481,879,980` valid tokens; commits `10,330` files / `1,650,867,255` valid tokens; PR `0`. Grand manifest-backed total: `11,771` files, `2,631,463` rows, `4,132,747,235` valid tokens.

**Validation Notes**
All 11,795 physical parquet files were readable, ZSTD-compressed, and had one schema hash (`2c2fe5ea3a244ea9`). No final-root partial/temp files were found. Row-level sampling covered 54 files and 269 rows, including orphan paddle/radare2 files; token lengths, `loss_mask`, `doc_ids`, source doc lengths, graph/chunk bounds, PR columns, and tokenizer IDs passed in that sample. Max sampled token ID was `65475`.

Changed files: none. I did not commit or edit anything. I also did not inspect/kill processes; `ps` was denied by sandbox, so live-process evidence is from reservations/progress/log state only.

