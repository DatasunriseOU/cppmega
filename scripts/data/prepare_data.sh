#!/usr/bin/env bash
# Root data preparation dispatcher.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"

DATA_ROOT="${MEGACPP_DATA_ROOT:-${REPO_ROOT}/outputs/data}"
DATASET_NAME="${MEGACPP_DATASET_NAME:-clang_semantic_4k_v10}"
DATASET_KIND="${MEGACPP_DATASET_KIND:-static_code}"
SOURCE_ROOT="${MEGACPP_SOURCE_ROOT:-}"
OUTPUT_ROOT="${MEGACPP_OUTPUT_ROOT:-}"
TOKENIZER="${MEGACPP_TOKENIZER_JSON:-${REPO_ROOT}/data/tokenizer_v2/tokenizer.json}"
TOKENIZER_CONTRACT="${MEGACPP_TOKENIZER_CONTRACT:-}"
DOMAIN_SCHEMA="${MEGACPP_DOMAIN_SCHEMA:-${REPO_ROOT}/data/domain_schema_v1.json}"
TARGET_LENGTHS="${MEGACPP_TARGET_LENGTHS:-1024,2048,4096}"
MANIFEST_OUT_OVERRIDE="${MEGACPP_MANIFEST_OUT:-}"
DRY_RUN=0
STAGE="all"

usage() {
    cat <<EOF
usage: $0 [download|tokenize|audit|format|cache|verify|all] [options]

Options:
  --data-root PATH            Megatron/manifests root (default: $DATA_ROOT)
  --dataset-name NAME         Dataset prefix (default: $DATASET_NAME)
  --source-root PATH          Extracted source repositories
  --output-root PATH          Packed parquet root; buckets live at PATH/LENGTH
  --tokenizer PATH            tokenizer.json artifact (default: $TOKENIZER)
  --tokenizer-contract PATH   Tokenizer contract (default: beside --tokenizer)
  --domain-schema PATH        Domain schema (default: $DOMAIN_SCHEMA)
  --target-lengths CSV        Packed lengths (default: $TARGET_LENGTHS)
  --dry-run                   Print commands without executing them
  -h, --help                  Show this help
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
    fi
}

if [[ $# -gt 0 && "$1" != -* ]]; then
    STAGE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            require_value "$@"
            DATA_ROOT="$2"
            shift 2
            ;;
        --dataset-name)
            require_value "$@"
            DATASET_NAME="$2"
            shift 2
            ;;
        --source-root)
            require_value "$@"
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --output-root)
            require_value "$@"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --tokenizer)
            require_value "$@"
            TOKENIZER="$2"
            shift 2
            ;;
        --tokenizer-contract)
            require_value "$@"
            TOKENIZER_CONTRACT="$2"
            shift 2
            ;;
        --domain-schema)
            require_value "$@"
            DOMAIN_SCHEMA="$2"
            shift 2
            ;;
        --target-lengths)
            require_value "$@"
            TARGET_LENGTHS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SOURCE_ROOT="${SOURCE_ROOT:-${DATA_ROOT}/cpp_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/parquet/${DATASET_NAME}}"
TOKENIZER_CONTRACT="${TOKENIZER_CONTRACT:-$(dirname "$TOKENIZER")/tokenizer_contract_v1.json}"

TARGET_LENGTH_VALUES=()
IFS=',' read -r -a RAW_TARGET_LENGTHS <<< "$TARGET_LENGTHS"
for raw_length in "${RAW_TARGET_LENGTHS[@]}"; do
    length="${raw_length//[[:space:]]/}"
    [[ -z "$length" ]] && continue
    if [[ ! "$length" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid target length: $raw_length" >&2
        exit 2
    fi
    duplicate=0
    for existing in "${TARGET_LENGTH_VALUES[@]:-}"; do
        if [[ "$existing" == "$length" ]]; then
            duplicate=1
            break
        fi
    done
    if [[ "$duplicate" -eq 0 ]]; then
        TARGET_LENGTH_VALUES+=("$length")
    fi
done
if [[ ${#TARGET_LENGTH_VALUES[@]} -eq 0 ]]; then
    echo "ERROR: --target-lengths produced no lengths" >&2
    exit 2
fi

SORTED_TARGET_LENGTHS=()
while IFS= read -r length; do
    SORTED_TARGET_LENGTHS+=("$length")
done < <(printf '%s\n' "${TARGET_LENGTH_VALUES[@]}" | sort -n)
TARGET_LENGTH_VALUES=("${SORTED_TARGET_LENGTHS[@]}")
TARGET_LENGTHS="$(IFS=,; printf '%s' "${TARGET_LENGTH_VALUES[*]}")"

case "$STAGE" in
    download|tokenize|audit|format|cache|verify|all) ;;
    *)
        echo "ERROR: unknown stage: $STAGE" >&2
        usage >&2
        exit 2
        ;;
esac

print_command() {
    printf 'DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
}

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        print_command "$@"
    else
        "$@"
    fi
}

ensure_dir() {
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$1"
    fi
}

manifest_path() {
    local length="$1"
    if [[ -z "$MANIFEST_OUT_OVERRIDE" ]]; then
        printf '%s/manifests/%s_%s.json' "$DATA_ROOT" "$DATASET_NAME" "$length"
    elif [[ ${#TARGET_LENGTH_VALUES[@]} -eq 1 ]]; then
        printf '%s' "$MANIFEST_OUT_OVERRIDE"
    elif [[ "$MANIFEST_OUT_OVERRIDE" == *.json ]]; then
        printf '%s_%s.json' "${MANIFEST_OUT_OVERRIDE%.json}" "$length"
    else
        printf '%s_%s' "$MANIFEST_OUT_OVERRIDE" "$length"
    fi
}

run_download() {
    run_cmd bash "$HERE/prepare_download_megacpp.sh" "$SOURCE_ROOT"
}

run_tokenize() {
    run_cmd "$PY" "$HERE/prepare_tokenize_megacpp.py" \
        --source-root "$SOURCE_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --tokenizer "$TOKENIZER" \
        --target-lengths "$TARGET_LENGTHS"
}

run_audit() {
    run_cmd "$PY" "$HERE/verify_tokenizer_contract.py" \
        --contract "$TOKENIZER_CONTRACT" \
        --tokenizer "$TOKENIZER" \
        --domain-schema "$DOMAIN_SCHEMA"

    local length dataset_dir manifest_out
    for length in "${TARGET_LENGTH_VALUES[@]}"; do
        dataset_dir="$OUTPUT_ROOT/$length"
        manifest_out="$(manifest_path "$length")"
        ensure_dir "$(dirname "$manifest_out")"
        run_cmd "$PY" "$HERE/verify_provenance.py" \
            --dataset-dir "$dataset_dir" \
            --kind "$DATASET_KIND"
        run_cmd "$PY" "$HERE/verify_side_channel_shapes.py" \
            --dataset-dir "$dataset_dir" \
            --require-full-sidecars
        run_cmd "$PY" "$HERE/audit_megacpp_4k.py" \
            --dataset-dir "$dataset_dir" \
            --kind "$DATASET_KIND" \
            --vocab-size 65536 \
            --seq-len "$length" \
            --graph
        run_cmd "$PY" "$HERE/build_dataset_manifest.py" \
            --dataset-dir "$dataset_dir" \
            --seq-len "$length" \
            --contract "$TOKENIZER_CONTRACT" \
            --tokenizer "$TOKENIZER" \
            --out "$manifest_out"
    done
}

run_format() {
    local length dataset_dir output_prefix
    ensure_dir "$DATA_ROOT/megatron"
    for length in "${TARGET_LENGTH_VALUES[@]}"; do
        dataset_dir="$OUTPUT_ROOT/$length"
        output_prefix="$DATA_ROOT/megatron/${DATASET_NAME}_${length}_train"
        run_cmd "$PY" "$REPO_ROOT/scripts/data_prep_parquet_to_megatron.py" \
            --input-dir "$dataset_dir" \
            --output-prefix "$output_prefix" \
            --split all \
            --dtype int32
    done
}

run_cache() {
    local length
    for length in "${TARGET_LENGTH_VALUES[@]}"; do
        run_cmd "$PY" "$HERE/prepare_cache_megacpp.py" \
            --data-root "$DATA_ROOT" \
            --dataset-name "${DATASET_NAME}_${length}" \
            --seq-length "$length"
    done
}

run_verify() {
    local length
    for length in "${TARGET_LENGTH_VALUES[@]}"; do
        run_cmd "$PY" "$HERE/verify_dataset_megacpp.py" \
            --data-root "$DATA_ROOT" \
            --dataset-name "${DATASET_NAME}_${length}" \
            --splits train
    done
}

case "$STAGE" in
    download) run_download ;;
    tokenize) run_tokenize ;;
    audit) run_audit ;;
    format) run_format ;;
    cache) run_cache ;;
    verify) run_verify ;;
    all)
        run_download
        run_tokenize
        run_audit
        run_format
        run_cache
        run_verify
        ;;
esac
