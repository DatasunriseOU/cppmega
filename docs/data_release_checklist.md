# Data Release Checklist

Reproducible status gate for the live training-data release. Each item maps to
one of the five blockers listed in
[`docs/status/training_data_inventory.md`](status/training_data_inventory.md).

Run the commands in order; any non-green result means the release is **not**
ready.

## 1. DirectXTK case-fold collision is resolved

**Blocker:** live source receipt double-counts `DirectXTK::code` /
`directxtk::code` (453,368 valid tokens / 215 rows).

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python scripts/report_training_data_status.py \
  --config configs/training_data_status.json --jobs 4
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
assert d.get('casefold_collisions') == [], d['casefold_collisions']
print('casefold_collisions: ok')
PY
```

**Pass criterion:** `casefold_collisions` is empty and the live source valid
 token count matches the physical Parquet sum.

## 2. Source conveyor has no unhandled failed units

**Blocker:** source conveyor is incomplete and still has failed units.

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
failed = d['datasets']['live_source'].get('failed_units', [])
assert failed == [], f"failed_units: {len(failed)}"
print('live_source failed_units: 0')
PY
```

**Pass criterion:** `failed_units` is empty, or every remaining unit has a
quarantine receipt with a reason.

## 3. Python auxiliary documents are separated from the primary stream

**Blocker:** Python auxiliary documents share physical packed rows with the
C/C++/SQL/build/test primary stream.

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
blockers = d['datasets']['live_source'].get('blockers', [])
assert "Python auxiliary documents are still mixed into main rows" not in blockers
print('python auxiliary stream: ok')
PY
```

**Pass criterion:** the mixed-row blocker is gone from `live_source.blockers`.

## 4. PR/MR store is materialized to eligible Parquet

**Blocker:** PR store verified but primary five-bucket materialization was
cancelled before producing eligible Parquet.

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
pr = d['datasets'].get('pr_mr', {})
eligible = pr.get('eligible', False)
print(f'pr_mr eligible: {eligible}')
assert eligible, "PR/MR dataset is not eligible"
PY
```

**Pass criterion:** `datasets.pr_mr.eligible` is `true` and the configured
Parquet path exists.

## 5. CI acquisition, dedup, and five-bucket export are complete

**Blocker:** CI acquisition is non-exhaustive; canonical union/global dedup,
primary-scope routing, five-bucket ZSTD Parquet export, and audit have not
completed.

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
ci = d['datasets'].get('ci', {})
blockers = ci.get('blockers', [])
assert blockers == [], f"ci blockers: {blockers}"
assert ci.get('eligible') is True, "CI dataset is not eligible"
print('ci dataset: ok')
PY
```

**Pass criterion:** `datasets.ci.blockers` is empty and `datasets.ci.eligible`
is `true`.

## 6. Sealed bundle v2 is pinned (final gate)

**Blocker:** live source Parquet is unsealed; no reproducible sealed bundle v2
exists.

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python - <<'PY'
import json
d = json.load(open('outputs/training_data_status/current.json'))
sealed = d['datasets'].get('sealed_megatron', {})
assert sealed.get('eligible') is True, sealed
print(f'sealed bundle: {sealed.get("manifest")}')
PY
```

**Pass criterion:** `datasets.sealed_megatron.eligible` is `true` and the
manifest path points to a valid `.bin/.idx` bundle audited for sidecar
contract.

## Quick combined command

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python scripts/report_training_data_status.py \
  --config configs/training_data_status.json --jobs 4 && \
.venv/bin/python - <<'PY'
import json, sys
d = json.load(open('outputs/training_data_status/current.json'))
errors = []
if d.get('casefold_collisions'):
    errors.append(f"casefold_collisions: {d['casefold_collisions']}")
ls = d['datasets']['live_source']
if ls.get('failed_units'):
    errors.append(f"live_source failed_units: {len(ls['failed_units'])}")
if "Python auxiliary documents are still mixed into main rows" in ls.get('blockers', []):
    errors.append("python aux still mixed")
pr = d['datasets'].get('pr_mr', {})
if not pr.get('eligible'):
    errors.append(f"pr_mr not eligible: {pr.get('blockers')}")
ci = d['datasets'].get('ci', {})
if ci.get('blockers') or not ci.get('eligible'):
    errors.append(f"ci not ready: {ci.get('blockers')}")
sealed = d['datasets'].get('sealed_megatron', {})
if not sealed.get('eligible'):
    errors.append(f"sealed bundle not ready")
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print('release-ready checks passed')
PY
```
