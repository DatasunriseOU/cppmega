#!/usr/bin/env bash
# GH Archive FALLBACK runner (only used when GraphQL hits a wall).
#
# HYBRID strategy: GraphQL is PRIMARY. This script is the documented fallback
# hook that graphql_pr_stream.TokenExhausted points at. It runs the BigQuery
# extraction in gharchive_query.sql for the resolved repo list, then the caller
# loads the resulting PR/Review/Comment events into pr_store.
#
# RULE #1: no silent success. If bq is missing or the query fails, we exit
# non-zero and print why -- we do NOT pretend data was fetched.
set -euo pipefail

PROJECT="${BQ_PROJECT:-natural-bison-491019-t9}"
TABLE_GLOB="${TABLE_GLOB:-githubarchive.month.20*}"
REPO_LIST="${REPO_LIST:-outputs/pr_ingest/repo_list.json}"
OUT="${OUT:-outputs/pr_ingest/gharchive_events.json}"

if ! command -v bq >/dev/null 2>&1; then
  echo "FATAL: bq (BigQuery CLI) not found; cannot run GH Archive fallback." >&2
  exit 2
fi
if [[ ! -f "$REPO_LIST" ]]; then
  echo "FATAL: repo list not found at $REPO_LIST (run repo_list_from_tarball.py)" >&2
  exit 2
fi

# Build the IN-list from the resolved repo_list.json (real owner/repo entries).
IN_LIST=$(python3 -c "import json,sys; d=json.load(open('$REPO_LIST')); print(', '.join(\"'\"+r['owner_repo'].replace(\"'\",\"''\")+\"'\" for r in d['repos']))")
if [[ -z "$IN_LIST" ]]; then
  echo "FATAL: repo list resolved to zero repos; refusing to query." >&2
  exit 2
fi

SQL=$(sed -e "s|{table_glob}|$TABLE_GLOB|g" -e "s|{repo_in_list}|$IN_LIST|g" \
  "$(dirname "$0")/gharchive_query.sql")

echo "[gharchive] project=$PROJECT table=$TABLE_GLOB repos=$(python3 -c "import json;print(len(json.load(open('$REPO_LIST'))['repos']))")" >&2
bq --project_id="$PROJECT" query --use_legacy_sql=false --format=prettyjson "$SQL" > "$OUT"
echo "[gharchive] wrote $OUT" >&2
