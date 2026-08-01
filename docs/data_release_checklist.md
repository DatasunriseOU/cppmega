# Data Release Checklist

This checklist evaluates the canonical live ledger. It never infers readiness
from directory names or adds overlapping datasets.

Set the paths once:

```bash
CPPMEGA_REPO=/Volumes/external/sources/cppmega
CPPMEGA_STATUS=/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.json
```

The LaunchAgent refreshes the ledger every five minutes. To force one atomic
refresh:

```bash
/Volumes/external/sources/.venvs/cppmega.source/bin/python \
  "$CPPMEGA_REPO/scripts/report_training_data_status.py" \
  --config "$CPPMEGA_REPO/configs/training_data_status.json" \
  --jobs 4
```

Every command below must exit zero. A non-zero result means that the release is
not ready.

## 1. Physical source Parquet matches its receipt

```bash
jq -e '
  (.freshness.stale == []) and
  (.datasets.live_source.casefold_collisions == []) and
  ([.datasets.live_source.receipt_minus_physical[]] | all(. == 0))
' "$CPPMEGA_STATUS"
```

This catches stale inputs, case-fold collisions, and any row/token difference
between the conveyor receipt and the published Parquet files.

## 2. Source conveyor is terminal

```bash
jq -e '
  (.datasets.live_source.conveyor.failed == 0) and
  (.datasets.live_source.conveyor.not_terminal == 0)
' "$CPPMEGA_STATUS"
```

## 3. Auxiliary languages are physically isolated

```bash
jq -e '
  .datasets.live_source.strict_primary_is_separate_parquet == true
' "$CPPMEGA_STATUS"
```

The primary C/C++/SQL/build/test stream must not share physical rows with the
Python auxiliary stream.

## 4. PR/MR data is materialized

```bash
jq -e '
  (.datasets.pr_mr.release_ready == true) and
  (.datasets.pr_mr.training_readable == true) and
  (.datasets.pr_mr.blockers == []) and
  (.datasets.pr_mr.eligible_parquet.files > 0)
' "$CPPMEGA_STATUS"
```

## 5. CI data is globally deduplicated and exported

```bash
jq -e '
  (.datasets.ci.release_ready == true) and
  (.datasets.ci.training_readable == true) and
  (.datasets.ci.blockers == []) and
  (.datasets.ci.token_accounting.ready_trained_tokens > 0)
' "$CPPMEGA_STATUS"
```

Store-local CAS token counters are not accepted by this gate.

## Final sealed-bundle gate

```bash
jq -e '
  (.datasets.sealed_megatron.release_ready == true) and
  (.datasets.sealed_megatron.training_readable == true) and
  (.datasets.sealed_megatron.blockers == [])
' "$CPPMEGA_STATUS"
CPPMEGA_MANIFEST=$(jq -er '.datasets.sealed_megatron.manifest' "$CPPMEGA_STATUS")
test -s "$CPPMEGA_MANIFEST"
```

## Combined gate

```bash
jq -e '
  (.freshness.stale == []) and
  (.datasets.live_source.casefold_collisions == []) and
  ([.datasets.live_source.receipt_minus_physical[]] | all(. == 0)) and
  (.datasets.live_source.conveyor.failed == 0) and
  (.datasets.live_source.conveyor.not_terminal == 0) and
  (.datasets.live_source.strict_primary_is_separate_parquet == true) and
  (.datasets.pr_mr.release_ready == true) and
  (.datasets.pr_mr.training_readable == true) and
  (.datasets.pr_mr.blockers == []) and
  (.datasets.pr_mr.eligible_parquet.files > 0) and
  (.datasets.ci.release_ready == true) and
  (.datasets.ci.training_readable == true) and
  (.datasets.ci.blockers == []) and
  (.datasets.ci.token_accounting.ready_trained_tokens > 0) and
  (.datasets.sealed_megatron.release_ready == true) and
  (.datasets.sealed_megatron.training_readable == true) and
  (.datasets.sealed_megatron.blockers == [])
' "$CPPMEGA_STATUS"
CPPMEGA_MANIFEST=$(jq -er '.datasets.sealed_megatron.manifest' "$CPPMEGA_STATUS")
test -s "$CPPMEGA_MANIFEST"
```

At the time this checklist was introduced, gates 1-5 correctly failed and the
existing sealed-bundle gate passed. Re-run the commands for current truth.
