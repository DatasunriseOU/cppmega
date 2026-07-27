#!/usr/bin/env bash
# GH Archive FALLBACK runner (only used when GraphQL hits a wall).
#
# HYBRID strategy: GraphQL is PRIMARY. This script is the documented fallback
# path that graphql_pr_stream.TokenExhausted points at. It runs the BigQuery
# extraction in gharchive_query.sql for the resolved repo list, then optionally
# loads the resulting PR/Review/Comment events into the unified pr_store when
# PR_STORE_DB is set. The SQL emits both raw and projected fields so old root
# receipts and the normalized production store remain compatible.
#
# RULE #1: no silent success. If bq is missing or the query fails, we exit
# non-zero and print why -- we do NOT pretend data was fetched.
set -euo pipefail

PROJECT="${BQ_PROJECT:-natural-bison-491019-t9}"
TABLE_GLOB="${TABLE_GLOB:-githubarchive.month.20*}"
REPO_LIST="${REPO_LIST:-outputs/pr_ingest/repo_list.json}"
OUT="${OUT:-outputs/pr_ingest/gharchive_events.json}"
PR_STORE_DB="${PR_STORE_DB:-}"

if ! command -v bq >/dev/null 2>&1; then
  echo "FATAL: bq (BigQuery CLI) not found; cannot run GH Archive fallback." >&2
  exit 2
fi
if [[ ! -f "$REPO_LIST" ]]; then
  echo "FATAL: repo list not found at $REPO_LIST (run repo_list_from_tarball.py)" >&2
  exit 2
fi

# Build the IN-list from GitHub-only repo_names. Mixed-forge identities stay in
# repo_list.json but are intentionally excluded from this GitHub-only query.
IN_LIST=$(python3 -c 'import json, sys; d=json.load(open(sys.argv[1])); raw=d.get("repo_names") or [r["owner_repo"] for r in d.get("repos", []) if r.get("owner_repo")]; names=list(dict.fromkeys(raw)); quote=chr(39); print(", ".join(quote + name.replace(quote, quote * 2) + quote for name in names))' "$REPO_LIST")
if [[ -z "$IN_LIST" ]]; then
  echo "FATAL: repo list resolved to zero repos; refusing to query." >&2
  exit 2
fi

SQL=$(sed -e "s|{table_glob}|$TABLE_GLOB|g" -e "s|{repo_in_list}|$IN_LIST|g" \
  "$(dirname "$0")/gharchive_query.sql")

REPO_COUNT=$(python3 -c 'import json, sys; d=json.load(open(sys.argv[1])); raw=d.get("repo_names") or [r["owner_repo"] for r in d.get("repos", []) if r.get("owner_repo")]; print(len(dict.fromkeys(raw)))' "$REPO_LIST")
echo "[gharchive] project=$PROJECT table=$TABLE_GLOB repos=$REPO_COUNT" >&2
echo "[gharchive] dry-run cost gate" >&2
bq --project_id="$PROJECT" query --use_legacy_sql=false --dry_run "$SQL"
if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "[gharchive] DRY_RUN_ONLY=1; stopping before real query." >&2
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
bq --project_id="$PROJECT" query --use_legacy_sql=false --format=prettyjson "$SQL" > "$OUT"
echo "[gharchive] wrote $OUT" >&2
if [[ -n "$PR_STORE_DB" ]]; then
  python3 "$(dirname "$0")/pr_store.py" ingest-gharchive \
    --store "$PR_STORE_DB" --input "$OUT"
else
  echo "[gharchive] PR_STORE_DB not set; raw events only, pr_store load skipped." >&2
fi
