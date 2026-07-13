**Confirmed Bugs**
Critical: current parquet cannot be converted with defaults.
File: [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:617), [prepare_format_megacpp.py](/Volumes/external/sources/cppmega/scripts/data/prepare_format_megacpp.py:142), [build_dataset_manifest.py](/Volumes/external/sources/cppmega/scripts/data/build_dataset_manifest.py:35), [verify_side_channel_shapes.py](/Volumes/external/sources/cppmega/scripts/data/verify_side_channel_shapes.py:32).
Failure: all these default/require `token_ids`, but live Lane 8 parquet buckets use `input_ids`. The converter will fail at `table.column(token_column)`, and the manifest/side-channel gates will reject current buckets before conversion.
Evidence: PyArrow metadata showed `input_ids` present and `token_ids` missing for local `outputs/reindexed{,_commits}` and macro-route code/commit buckets 1024/2048/4096/8192/16384.
Focused test/fix: add an `input_ids`-only packed parquet fixture; require `prepare_format_megacpp.py` and both verifiers to consume the same configurable token column, defaulting to current packed schema or auto-detecting `input_ids`/`token_ids` fail-closed on ambiguity.

Critical: Stage 3 wrapper/artifact drops sidecars required by structure training.
File: [prepare_format_megacpp.py](/Volumes/external/sources/cppmega/scripts/data/prepare_format_megacpp.py:47), [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:59), [structure_dataset_patch.py](/Volumes/external/sources/cppmega/cppmega/megatron/structure_dataset_patch.py:608), [h200_fp8_recompute_debug_1024_train.json](/Volumes/external/sources/cppmega/outputs/megatron_ready/h200_fp8_recompute_debug_1024_train.json:7).
Failure: wrapper default has only 12 token sidecars, omitting `loss_mask`, `doc_ids`, domain/role/entity/scope/source/confidence. Runtime requires `loss_mask` and all canonical token sidecars, and graph mode requires full graph routes.
Evidence: local Megatron manifest has only 12 `side_channel_paths` and graph sidecars only for call/type/chunks; runtime raises on missing loss mask at `structure_dataset_patch.py:615` and missing graph columns at `structure_dataset_patch.py:255`.
Focused test/fix: make wrapper default reuse `DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS`; add regression that wrapper defaults equal/superset converter defaults and a tiny end-to-end manifest loads with `CPPMEGA_STRUCTURE_ENABLED=1` and graph routes.

Critical: upload can publish files that the receipt did not certify.
File: [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:125), [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:184), [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:214).
Failure: token total is computed from audit receipt, but `_existing_sources` only checks directory existence, then `aws s3 sync` uploads whatever is currently in that directory.
Evidence: current local dirs vs final receipt are mismatched: `code/1024` local 118 files vs receipt 104; `commits/1024` local 57 vs receipt 592; all PR bucket dirs are absent while receipt expects 129/130 files.
Focused test/fix: pass receipt `by_kind_bucket` into source validation and assert exact file count before upload; then add per-file name/size/sha256 manifest checks.

High: 16k code exists locally but is not selected/certified.
File: [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:30), [download_verified_sidecar_from_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/download_verified_sidecar_from_nebius_s3.py:139).
Failure: code selections include 1024/2048/4096/8192 only; no `parquet/code/16384`. Local `outputs/reindexed/16384` has 111 parquet files and macro-route code/16384 has 277.
Evidence: final/current audit keys also omit `code/16384`.
Focused test/fix: either add `code/16384` to upload/download/audit profile, or explicitly list it in manifest `excluded` with reason and a failing test that all intended buckets 1024..16384 are accounted for.

High: split heuristic is unsafe for bucketed parquet.
File: [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:146).
Failure: without `val_shard.parquet`, `train` silently excludes lexicographic last shard and `val` uses only that shard. Bucketed `outputs/reindexed/*` are not nanochat train/val shard sets, so default `train,val` creates arbitrary splits and drops one file from train.
Evidence: current bucket dirs are repo/bucket file collections, not explicit split dirs.
Focused test/fix: require explicit split manifest or `--split all` for bucketed inputs; block heuristic unless shard naming matches known nanochat convention.

**Design Gaps**
Medium: manifests do not carry object hashes or object lists.
File: [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:268), [download_verified_sidecar_from_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/download_verified_sidecar_from_nebius_s3.py:259).
Failure: same file count with different contents can pass consume-side verification; S3 multipart ETags are not a robust dataset hash contract.
Fix: include relative path, size, sha256, receipt hash, and audit command in manifest; verify local and S3 `ContentLength`/checksum before marking upload complete.

Medium: upload is not atomic at prefix level.
File: [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:330), [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:354), [upload_verified_sidecar_to_nebius_s3.py](/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py:371).
Failure: data syncs directly into final prefix, then manifest uploads last. Failed/retried sync can leave partial or stale objects in final prefix.
Fix: upload to staging prefix keyed by run id, verify object manifest, then publish a small final pointer/manifest.

Medium: Megatron conversion output is not atomic/resumable.
File: [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:661), [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:725), [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:755).
Failure: `.bin`, sidecars, `.idx`, and JSON are written directly to final names. An interrupt can leave a prefix that looks present but is incomplete.
Fix: write under `.tmp-$pid`, validate counts/sizes, then `os.replace` manifest last; add resume that skips only fully validated prefixes.

**Stale Docs/Artifacts**
Low: converter docs still describe `token_ids` parquet.
File: [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:4).
Evidence: live packed Lane 8 buckets are `input_ids`-based; audit code requires `input_ids` at [audit_sidecar_parquet.py](/Volumes/external/sources/cppmega.mlx/scripts/audit_sidecar_parquet.py:723).
Fix: update docs/examples after token-column fix.

Low: local upload manifests are historical and conflict with current script/defaults.
File: `/Volumes/external/sources/cppmega.mlx/outputs/sidecar_upload_manifest.json:1`.
Evidence: old `verified-20260627` excludes some buckets and lacks current `profile`; `valid-all-20260627` includes PR buckets but current `outputs/reindexed_pr` is empty.
Fix: regenerate manifests only after local receipt/source/S3 counts reconcile.

**Safe Command Sequence**
I did not edit or commit files. I did not run pytest because this was a read-only audit and pytest/cache/temp writes are likely.

Safe verification commands:
```bash
cd /Volumes/external/sources/cppmega.mlx
export PYTHONDONTWRITEBYTECODE=1 AWS_MAX_ATTEMPTS=1 AWS_RETRY_MODE=standard

jq '{bucket,prefix,verified_valid_tokens,profile,selections:[.selections[]|.remote],excluded}' \
  outputs/sidecar_upload_manifest*.json

. ./.env
export AWS_ACCESS_KEY_ID="$NEBIUS_S3_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$NEBIUS_S3_SECRET_ACCESS_KEY"
aws s3 cp s3://cppmega-sidecar-20260627/cppmega-sidecar/valid-all-20260627/manifest.json - \
  --endpoint-url "${NEBIUS_S3_ENDPOINT_URL:-https://storage.eu-north1.nebius.cloud}" --no-progress
```

Expected artifacts: no new local files; stdout JSON for local manifests and S3 manifest. Blocker here: S3 read failed from this sandbox with `Could not connect to the endpoint URL`, so live S3 object counts remain unproven. Local blockers are current source/receipt count mismatch, empty PR dirs, missing code/16384 selection, and `token_ids` defaults against `input_ids` parquet.

Changed files: none. Simplifications made: none, read-only review only.

