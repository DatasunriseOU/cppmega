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
#   bash prepare_data.sh audit          # fail-closed parquet/tokenizer gates
#   bash prepare_data.sh format         # parquet -> Megatron .bin/.idx
#   bash prepare_data.sh cache          # warm/validate index cache
#   bash prepare_data.sh verify         # sanity-check final dataset
#
# Env knobs (all optional):
#   MEGACPP_DATA_ROOT    default: /home/dave/cppmega-root/data
#   MEGACPP_DATASET_NAME default: clang_semantic_4k_v10
#   MEGACPP_NANOCHAT_ROOT  (required for 'tokenize' stage)
#   MEGACPP_DATASET_KIND default: static_code (use commits for commit shards)
#   MEGACPP_TOKENIZER_CONTRACT default: ../cppmega.mlx/cppmega_mlx/tokenizer/tokenizer_contract_v1.json
#   MEGACPP_TOKENIZER_JSON default: ../cppmega.mlx/cppmega_mlx/tokenizer/tokenizer.json

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SOURCE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PY="${PYTHON:-python3}"
DATA_ROOT="${MEGACPP_DATA_ROOT:-/home/dave/cppmega-root/data}"
DATASET_NAME="${MEGACPP_DATASET_NAME:-clang_semantic_4k_v10}"
DATASET_KIND="${MEGACPP_DATASET_KIND:-static_code}"
DATASET_DIR="${DATA_ROOT}/parquet/${DATASET_NAME}"
CONTRACT="${MEGACPP_TOKENIZER_CONTRACT:-${SOURCE_ROOT}/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer_contract_v1.json}"
TOKENIZER_JSON="${MEGACPP_TOKENIZER_JSON:-${SOURCE_ROOT}/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer.json}"
MANIFEST_OUT="${MEGACPP_MANIFEST_OUT:-${DATA_ROOT}/manifests/${DATASET_NAME}.json}"

STAGE="${1:-all}"

run_download() {
    bash "$HERE/prepare_download_megacpp.sh"
}
run_tokenize() {
    "$PY" "$HERE/prepare_tokenize_megacpp.py"
}
run_audit() {
    "$PY" "$HERE/verify_tokenizer_contract.py" \
        --root "$SOURCE_ROOT" \
        --contract "$CONTRACT"
    "$PY" "$HERE/verify_provenance.py" \
        --dataset-dir "$DATASET_DIR" \
        --kind "$DATASET_KIND"
    "$PY" "$HERE/verify_side_channel_shapes.py" \
        --dataset-dir "$DATASET_DIR" \
        --require-full-sidecars
    "$PY" "$HERE/audit_megacpp_4k.py" \
        --dataset-dir "$DATASET_DIR" \
        --kind "$DATASET_KIND" \
        --vocab-size 65536 \
        --seq-len 4096 \
        --graph
    "$PY" "$HERE/build_dataset_manifest.py" \
        --dataset-dir "$DATASET_DIR" \
        --seq-len 4096 \
        --contract "$CONTRACT" \
        --tokenizer "$TOKENIZER_JSON" \
        --out "$MANIFEST_OUT"
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
    audit)    run_audit ;;
    format)   run_format ;;
    cache)    run_cache ;;
    verify)   run_verify ;;
    all)
        run_download
        run_tokenize
        run_audit
        run_format
        run_cache
        run_verify
        ;;
    *)
        echo "usage: $0 {download|tokenize|audit|format|cache|verify|all}" >&2
        exit 2
        ;;
esac
