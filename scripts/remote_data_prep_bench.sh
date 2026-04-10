#!/usr/bin/env bash
# Download clang parquet data from GCS and convert to Megatron binary format
# on the h200_1 machine.
#
# Datasets converted:
#   - clang_semantic_4k_v10 (code, 5.96 GiB, 65 shards)
#   - clang_commits_4k_v1 (commits, 17.69 GiB, 103 shards)
#
# Output: Megatron-ready .bin/.idx files at $REMOTE_ROOT/data/megatron/
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dave/cppmega-root}"
REMOTE_VENV="${REMOTE_VENV:-${REMOTE_ROOT}/cppmega-venv}"
GCS_DATA_BUCKET="${GCS_DATA_BUCKET:-sftp://BUCKET_TRAINING_DATA/data/parquet}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-data-prep.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-data-prep.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_ROOT}/cppmega:${REMOTE_ROOT}/megatron-lm:${PYTHONPATH:-}"

PARQUET_DIR="${REMOTE_ROOT}/data/parquet"
MEGATRON_DIR="${REMOTE_ROOT}/data/megatron"
mkdir -p "${PARQUET_DIR}" "${MEGATRON_DIR}"

# ---- Download from GCS if not present ----
for dataset in clang_semantic_4k_v10 clang_commits_4k_v1; do
  dest="${PARQUET_DIR}/${dataset}"
  if [ -d "${dest}" ] && [ "$(ls "${dest}"/*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "${dataset}: already downloaded ($(ls "${dest}"/*.parquet | wc -l) shards)"
  else
    echo "${dataset}: downloading from GCS..."
    mkdir -p "${dest}"
    gcloud storage cp -r "${GCS_DATA_BUCKET}/${dataset}/*" "${dest}/"
    echo "${dataset}: downloaded ($(ls "${dest}"/*.parquet | wc -l) shards)"
  fi
done

echo ""
echo "=== Data downloaded ==="
du -sh "${PARQUET_DIR}"/*/ 2>/dev/null || true
echo ""

# ---- Convert to Megatron binary format ----
PREP_SCRIPT="${REMOTE_ROOT}/cppmega/scripts/data_prep_parquet_to_megatron.py"

for dataset in clang_semantic_4k_v10 clang_commits_4k_v1; do
  for split in train val; do
    prefix="${MEGATRON_DIR}/${dataset}_${split}"
    if [ -f "${prefix}.bin" ] && [ -f "${prefix}.idx" ]; then
      echo "${dataset} ${split}: already converted"
      continue
    fi
    echo "${dataset} ${split}: converting to Megatron format..."
    python "${PREP_SCRIPT}" \
      --input-dir "${PARQUET_DIR}/${dataset}" \
      --output-prefix "${prefix}" \
      --split "${split}" \
      --token-column token_ids \
      --dtype uint16
  done
done

echo ""
echo "=== Megatron data ready ==="
ls -lh "${MEGATRON_DIR}"/*.{bin,idx} 2>/dev/null || echo "no megatron data files found"

# ---- Print data blend path for training script ----
echo ""
echo "=== Use these data paths in training ==="
echo "Code + commits blend (50/50):"
echo "  CPPMEGA_DATA_PATH='1.0 ${MEGATRON_DIR}/clang_semantic_4k_v10_train 1.0 ${MEGATRON_DIR}/clang_commits_4k_v1_train'"
echo ""
echo "Code only:"
echo "  CPPMEGA_DATA_PATH='${MEGATRON_DIR}/clang_semantic_4k_v10_train'"
echo ""
echo "Commits only:"
echo "  CPPMEGA_DATA_PATH='${MEGATRON_DIR}/clang_commits_4k_v1_train'"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" \
  --zone "${REMOTE_ZONE}" \
  --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_VENV='${REMOTE_VENV}' GCS_DATA_BUCKET='${GCS_DATA_BUCKET}' bash '${REMOTE_TMP_SCRIPT}'" \
  || { echo "data prep failed"; exit 1; }
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"

echo "data prep complete on ${REMOTE_HOST}"
