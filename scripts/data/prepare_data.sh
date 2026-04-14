#!/bin/bash
# megacpp master data prep dispatcher.
#
# Runs download -> tokenize -> format -> cache -> verify to produce the
# Megatron-format ``clang_semantic_4k_v10`` dataset used by NAM56R training
# (see scripts/remote_smoke_h200_dsa_9_4_m.sh).
#
# See docs/data_preparation.md for full details.
#
# Usage:
#   bash prepare_data.sh                # run all stages
#   bash prepare_data.sh download       # just clone source repos
#   bash prepare_data.sh tokenize       # requires MEGACPP_NANOCHAT_ROOT
#   bash prepare_data.sh format         # parquet -> Megatron .bin/.idx
#   bash prepare_data.sh cache          # warm/validate index cache
#   bash prepare_data.sh verify         # sanity-check final dataset
#
# Env knobs (all optional):
#   MEGACPP_DATA_ROOT    default: /home/dave/cppmega-root/data
#   MEGACPP_DATASET_NAME default: clang_semantic_4k_v10
#   MEGACPP_NANOCHAT_ROOT  (required for 'tokenize' stage)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

STAGE="${1:-all}"

run_download() {
    bash "$HERE/prepare_download_megacpp.sh"
}
run_tokenize() {
    "$PY" "$HERE/prepare_tokenize_megacpp.py"
}
run_format() {
    "$PY" "$HERE/prepare_format_megacpp.py"
}
run_cache() {
    "$PY" "$HERE/prepare_cache_megacpp.py"
}
run_verify() {
    "$PY" "$HERE/verify_dataset_megacpp.py"
}

case "$STAGE" in
    download) run_download ;;
    tokenize) run_tokenize ;;
    format)   run_format ;;
    cache)    run_cache ;;
    verify)   run_verify ;;
    all)
        run_download
        run_tokenize
        run_format
        run_cache
        run_verify
        ;;
    *)
        echo "usage: $0 {download|tokenize|format|cache|verify|all}" >&2
        exit 2
        ;;
esac
