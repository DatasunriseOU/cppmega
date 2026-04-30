#!/usr/bin/env bash
# Full-size NAM56R production A/B gate for the guarded Mamba3 stage2
# force-nonTMA candidate.
#
# Run this from an H200 host checkout. It wraps the existing production v1
# runner twice:
#   1. baseline, after a best-effort rollback of any installed stage2 patch
#   2. candidate, after applying the guarded bf=1,bb=0 stage2 patch
#
# The script writes per-run logs plus summary.{csv,json,md}. It refuses
# non-H200 GPUs unless CPPMEGA_ALLOW_NON_H200_FULL_GATE=1 is set. Do not use
# that override for mini/component/prototype tests.
set -euo pipefail

if [[ "${CPPMEGA_ALLOW_NON_H200_FULL_GATE:-0}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found; this full-boundary gate requires H200." >&2
    exit 2
  fi
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H200"; then
    echo "ERROR: this full-boundary gate requires H200 by default." >&2
    echo "Set CPPMEGA_ALLOW_NON_H200_FULL_GATE=1 only for explicitly approved full-size reruns." >&2
    exit 2
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PROD_RUNNER="${CPPMEGA_PROD_RUNNER:-scripts/remote_production_h200_nam56r_v1.sh}"
if [[ ! -x "${PROD_RUNNER}" && ! -f "${PROD_RUNNER}" ]]; then
  echo "ERROR: production runner not found: ${PROD_RUNNER}" >&2
  exit 2
fi

TRAIN_ITERS="${TRAIN_ITERS:-${CPPMEGA_TRAIN_ITERS:-20}}"
if (( TRAIN_ITERS < 20 )); then
  echo "ERROR: TRAIN_ITERS=${TRAIN_ITERS}; production A/B gate requires at least 20." >&2
  exit 2
fi

STAMP="${CPPMEGA_GATE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GATE_ID="${CPPMEGA_GATE_ID:-wave28_lane_a_h200_stage2_prod_ab_${STAMP}}"
ARTIFACT_ROOT="${CPPMEGA_GATE_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/mamba3_stage2_prod_gate/${GATE_ID}}"
mkdir -p "${ARTIFACT_ROOT}"

BASELINE_LOG="${ARTIFACT_ROOT}/baseline.log"
CANDIDATE_LOG="${ARTIFACT_ROOT}/candidate_stage2_force_nontma.log"
SUMMARY_CSV="${ARTIFACT_ROOT}/summary.csv"
SUMMARY_JSON="${ARTIFACT_ROOT}/summary.json"
SUMMARY_MD="${ARTIFACT_ROOT}/summary.md"

run_applier_default_off_noop() {
  echo "=== stage2 applier default-off no-op check ==="
  env -u CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA \
      -u MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION \
      -u CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK \
      PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
      python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
}

rollback_stage2_patch() {
  echo "=== stage2 rollback guard ==="
  if CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1 \
      PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
      python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches; then
    echo "rollback/status: restored-or-clean"
  else
    echo "rollback/status: no backup or reverse patch unavailable; verifying clean state before continuing" >&2
  fi
}

apply_stage2_patch() {
  echo "=== stage2 apply guard ==="
  CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1 \
  MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1 \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches
}

verify_stage2_clean() {
  echo "=== verify stage2 patch is inactive ==="
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python - <<'PY'
from cppmega.megatron.upstream_patches import apply_mamba3_stage2_force_nontma_patches as p

path = p._find_mamba3_bwd_file()
text = path.read_text()
if p._is_patched(text) or p._has_partial_stage2_markers(text):
    raise SystemExit(f"{path}: stage2 patch markers still present; refusing baseline")
print(f"clean: {path}")
PY
}

verify_stage2_patched() {
  echo "=== verify stage2 patch is active bf=1,bb=0 ==="
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python - <<'PY'
from cppmega.megatron.upstream_patches import apply_mamba3_stage2_force_nontma_patches as p

path = p._find_mamba3_bwd_file()
p._validate_patched(path)
text = path.read_text()
if "bf_num_stages=1" not in text or "bb_num_stages=0" not in text:
    raise SystemExit(f"{path}: expected bf_num_stages=1 and bb_num_stages=0")
print(f"patched: {path}")
PY
}

PATCH_ACTIVE=0
cleanup_stage2_patch() {
  if [[ "${PATCH_ACTIVE}" == "1" ]]; then
    echo "=== cleanup: rolling back active stage2 patch ===" >&2
    rollback_stage2_patch >&2 || true
    verify_stage2_clean >&2 || true
  fi
}
trap cleanup_stage2_patch EXIT

run_variant() {
  local variant="$1"
  local log="$2"
  local run_id="${GATE_ID}_${variant}"

  echo "=== running ${variant} ==="
  echo "log: ${log}"
  RUN_ID="${run_id}" \
  LOG="${log}" \
  TRAIN_ITERS="${TRAIN_ITERS}" \
  bash "${PROD_RUNNER}"
}

summarize() {
  python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${SUMMARY_MD}" "${BASELINE_LOG}" "${CANDIDATE_LOG}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

summary_csv, summary_json, summary_md, baseline_log, candidate_log = map(Path, sys.argv[1:])


def last_float(patterns, text):
    values = []
    for pattern in patterns:
        values.extend(float(m.group(1)) for m in re.finditer(pattern, text, re.I))
    return values[-1] if values else None


def parse_log(label, path):
    text = path.read_text(errors="replace") if path.exists() else ""
    tok_sec = last_float(
        [
            r"(?:tokens/sec|tok/sec|tokens per second|tokens/s)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/sec|tok/sec|tokens per second|tokens/s)",
        ],
        text,
    )
    elapsed_ms = last_float(
        [r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"],
        text,
    )
    peak_alloc = [
        float(m.group(1))
        for m in re.finditer(r"peak_alloc_gib=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    peak_reserved = [
        float(m.group(1))
        for m in re.finditer(r"peak_reserved_gib=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    return {
        "variant": label,
        "log": str(path),
        "tok_sec": tok_sec,
        "elapsed_ms": elapsed_ms,
        "peak_alloc_gib": max(peak_alloc) if peak_alloc else None,
        "peak_reserved_gib": max(peak_reserved) if peak_reserved else None,
        "exit_ok_marker": "=== Exit code: 0 ===" in text or "Training complete" in text,
    }


rows = [
    parse_log("baseline", baseline_log),
    parse_log("stage2_force_nontma_bf1_bb0", candidate_log),
]

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary_json.write_text(json.dumps(rows, indent=2) + "\n")

lines = [
    "| variant | tok/sec | elapsed ms/iter | peak alloc GiB | peak reserved GiB | log |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{value:.3f}"

    lines.append(
        "| {variant} | {tok_sec} | {elapsed_ms} | {peak_alloc_gib} | "
        "{peak_reserved_gib} | {log} |".format(
            variant=row["variant"],
            tok_sec=fmt(row["tok_sec"]),
            elapsed_ms=fmt(row["elapsed_ms"]),
            peak_alloc_gib=fmt(row["peak_alloc_gib"]),
            peak_reserved_gib=fmt(row["peak_reserved_gib"]),
            log=row["log"],
        )
    )
summary_md.write_text("\n".join(lines) + "\n")
print(summary_md.read_text())
PY
}

echo "gate_id=${GATE_ID}"
echo "artifact_root=${ARTIFACT_ROOT}"
echo "gpu_names:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_applier_default_off_noop | tee "${ARTIFACT_ROOT}/default_off_noop.txt"
rollback_stage2_patch | tee "${ARTIFACT_ROOT}/pre_baseline_rollback.txt"
verify_stage2_clean | tee "${ARTIFACT_ROOT}/pre_baseline_clean_check.txt"

unset CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA
unset MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION
unset CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK
run_variant "baseline" "${BASELINE_LOG}"

apply_stage2_patch | tee "${ARTIFACT_ROOT}/candidate_apply.txt"
verify_stage2_patched | tee "${ARTIFACT_ROOT}/candidate_patched_check.txt"
PATCH_ACTIVE=1
run_variant "stage2_force_nontma_bf1_bb0" "${CANDIDATE_LOG}"

rollback_stage2_patch | tee "${ARTIFACT_ROOT}/post_candidate_rollback.txt"
verify_stage2_clean | tee "${ARTIFACT_ROOT}/post_candidate_clean_check.txt"
PATCH_ACTIVE=0
summarize

echo "summary_csv=${SUMMARY_CSV}"
echo "summary_json=${SUMMARY_JSON}"
echo "summary_md=${SUMMARY_MD}"
