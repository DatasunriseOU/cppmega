#!/usr/bin/env bash
# Driver: sequential DSA 9+4 production parameter sweep on bench3
#
# Runs configs B1-B8 one at a time. For each:
#   1. Kills any previous nam56r_sweep_* tmux session
#   2. Launches in tmux nam56r_sweep_bN
#   3. Waits for completion (iter 50 or OOM)
#   4. Extracts metrics from log
#   5. Records to results table
#   6. Moves to next config
#
# Designed to run directly on bench3 (h200_1).
# Expected total runtime: ~80 min (8 configs x ~10 min each).
#
# Usage:
#   bash scripts/drive_sweep_dsa_production.sh
#   # Or start from a specific config:
#   START_FROM=B3 bash scripts/drive_sweep_dsa_production.sh
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
SWEEP_DIR="${REMOTE_ROOT}/cppmega-root/cppmega/nam56r_dsa_sweep"
RESULTS_CSV="${SWEEP_DIR}/sweep_results.csv"
SCRIPT_DIR="${REMOTE_ROOT}/cppmega-root/cppmega/scripts"
SWEEP_SCRIPT="${SCRIPT_DIR}/remote_sweep_h200_dsa_production.sh"
START_FROM="${START_FROM:-B1}"

mkdir -p "${SWEEP_DIR}"

# Write CSV header
if [ ! -f "${RESULTS_CSV}" ]; then
  echo "config,pp,vpp,mbs,gbs,mtp,recompute,cuda_graphs,ep,tok_sec,iter_ms,peak_gb,loss_at_50,status" > "${RESULTS_CSV}"
fi

CONFIGS="B1 B2 B3 B4 B5 B6 B7 B8"
STARTED=0

# Config metadata for CSV
declare -A CFG_PP=([B1]=2 [B2]=2 [B3]=2 [B4]=2 [B5]=2 [B6]=2 [B7]=2 [B8]=2)
declare -A CFG_VPP=([B1]=2 [B2]=2 [B3]=2 [B4]=2 [B5]=2 [B6]=2 [B7]=2 [B8]=1)
declare -A CFG_MBS=([B1]=4 [B2]=4 [B3]=8 [B4]=2 [B5]=4 [B6]=4 [B7]=4 [B8]=4)
declare -A CFG_GBS=([B1]=64 [B2]=64 [B3]=64 [B4]=64 [B5]=128 [B6]=64 [B7]=64 [B8]=64)
declare -A CFG_MTP=([B1]=2 [B2]=2 [B3]=2 [B4]=2 [B5]=2 [B6]=1 [B7]=2 [B8]=2)
declare -A CFG_RECOMPUTE=([B1]="sel_moe_act" [B2]="sel_moe_act" [B3]="sel_moe_act" [B4]="sel_moe_act" [B5]="sel_moe_act" [B6]="sel_moe_act" [B7]="sel_full" [B8]="sel_moe_act")
declare -A CFG_CG=([B1]="per_module" [B2]="OFF" [B3]="per_module" [B4]="per_module" [B5]="per_module" [B6]="per_module" [B7]="per_module" [B8]="per_module")
declare -A CFG_EP=([B1]=1 [B2]=1 [B3]=1 [B4]=1 [B5]=1 [B6]=1 [B7]=1 [B8]=1)

extract_metrics() {
  local log_file="$1"
  local config="$2"
  local gbs="$3"
  local seq_len=4096

  if [ ! -f "${log_file}" ]; then
    echo "${config},${CFG_PP[$config]},${CFG_VPP[$config]},${CFG_MBS[$config]},${CFG_GBS[$config]},${CFG_MTP[$config]},${CFG_RECOMPUTE[$config]},${CFG_CG[$config]},${CFG_EP[$config]},,,,,NO_LOG"
    return
  fi

  # Check for OOM
  if grep -q "OutOfMemoryError\|CUDA out of memory" "${log_file}"; then
    # Try to get peak memory from nvidia-smi or atexit reporter
    local peak_gb
    peak_gb=$(grep -o "peak_alloc_gib=[0-9.]*" "${log_file}" | head -1 | cut -d= -f2)
    [ -z "${peak_gb}" ] && peak_gb="OOM"
    echo "${config},${CFG_PP[$config]},${CFG_VPP[$config]},${CFG_MBS[$config]},${CFG_GBS[$config]},${CFG_MTP[$config]},${CFG_RECOMPUTE[$config]},${CFG_CG[$config]},${CFG_EP[$config]},OOM,OOM,${peak_gb},,OOM"
    return
  fi

  # Check for other crash
  if grep -q "SIGABRT\|SIGSEGV\|RuntimeError\|AssertionError" "${log_file}"; then
    echo "${config},${CFG_PP[$config]},${CFG_VPP[$config]},${CFG_MBS[$config]},${CFG_GBS[$config]},${CFG_MTP[$config]},${CFG_RECOMPUTE[$config]},${CFG_CG[$config]},${CFG_EP[$config]},,,,, CRASH"
    return
  fi

  # Extract iter times for iter 30-50
  # Megatron log format: " iteration       30/      50 | ... | elapsed time per iteration (ms): XXXX.X |"
  local iter_times
  iter_times=$(grep -E "iteration\s+[3-4][0-9]/|iteration\s+50/" "${log_file}" | \
    grep -oP "elapsed time per iteration \(ms\): \K[0-9.]+" || true)

  if [ -z "${iter_times}" ]; then
    # Try alternate format: " iteration       30/ ... elapsed (ms) per iteration: XXXX.X"
    iter_times=$(grep -E "iteration\s+[3-4][0-9]/|iteration\s+50/" "${log_file}" | \
      grep -oP "per iteration:\s*\K[0-9.]+" || true)
  fi

  if [ -z "${iter_times}" ]; then
    echo "${config},${CFG_PP[$config]},${CFG_VPP[$config]},${CFG_MBS[$config]},${CFG_GBS[$config]},${CFG_MTP[$config]},${CFG_RECOMPUTE[$config]},${CFG_CG[$config]},${CFG_EP[$config]},,,,,NO_ITER_DATA"
    return
  fi

  # Compute mean iter_ms
  local mean_iter_ms
  mean_iter_ms=$(echo "${iter_times}" | awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')

  # Compute tok/sec = GBS * seq_len * 1000 / mean_iter_ms
  local tok_sec
  tok_sec=$(echo "${mean_iter_ms}" | awk -v gbs="${gbs}" -v seq="${seq_len}" '{if($1>0) printf "%.0f", gbs*seq*1000/$1; else print "0"}')

  # Extract loss at last iteration
  local loss_at_end
  loss_at_end=$(grep -E "iteration\s+50/" "${log_file}" | grep -oP "lm loss:\s*\K[0-9.]+E[+-][0-9]+" | tail -1 || true)
  if [ -z "${loss_at_end}" ]; then
    loss_at_end=$(grep -E "iteration\s+50/" "${log_file}" | grep -oP "lm loss:\s*\K[0-9.]+" | tail -1 || true)
  fi
  [ -z "${loss_at_end}" ] && loss_at_end="N/A"

  # Extract peak memory from atexit reporter
  local peak_gb
  peak_gb=$(grep -o "peak_alloc_gib=[0-9.]*" "${log_file}" | sort -t= -k2 -nr | head -1 | cut -d= -f2)
  [ -z "${peak_gb}" ] && peak_gb="N/A"

  echo "${config},${CFG_PP[$config]},${CFG_VPP[$config]},${CFG_MBS[$config]},${CFG_GBS[$config]},${CFG_MTP[$config]},${CFG_RECOMPUTE[$config]},${CFG_CG[$config]},${CFG_EP[$config]},${tok_sec},${mean_iter_ms},${peak_gb},${loss_at_end},OK"
}

for cfg in ${CONFIGS}; do
  if [ "${STARTED}" = "0" ] && [ "${cfg}" != "${START_FROM}" ]; then
    echo "[sweep] Skipping ${cfg} (START_FROM=${START_FROM})"
    continue
  fi
  STARTED=1

  SESSION_NAME="nam56r_sweep_${cfg,,}"
  LOG_FILE="${SWEEP_DIR}/nam56r_dsa_sweep_${cfg}.log"

  echo ""
  echo "=================================================================="
  echo "[sweep] Starting config ${cfg} at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=================================================================="

  # Kill any existing sweep session
  tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

  # Also kill any leftover torch processes
  pkill -f "pretrain_mamba" 2>/dev/null || true
  sleep 1

  # Clear GPU memory
  nvidia-smi --gpu-reset 2>/dev/null || true
  sleep 1

  # Launch in tmux
  tmux new-session -d -s "${SESSION_NAME}" \
    "CONFIG=${cfg} LOG=${LOG_FILE} bash -l ${SWEEP_SCRIPT} ; echo ${cfg}_DONE_\$? > ${SWEEP_DIR}/${cfg}_status.txt"

  echo "[sweep] Launched ${cfg} in tmux session '${SESSION_NAME}'"
  echo "[sweep] Log: ${LOG_FILE}"

  # Wait for completion: poll for the status file or OOM in log
  MAX_WAIT=1200   # 20 min max per config
  ELAPSED=0
  POLL_INTERVAL=15

  while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    # Check if tmux session still alive
    if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
      echo "[sweep] ${cfg}: tmux session ended at $(date '+%H:%M:%S') (elapsed ${ELAPSED}s)"
      break
    fi

    # Check for OOM in log
    if [ -f "${LOG_FILE}" ] && grep -q "OutOfMemoryError\|CUDA out of memory" "${LOG_FILE}"; then
      echo "[sweep] ${cfg}: OOM detected at $(date '+%H:%M:%S') (elapsed ${ELAPSED}s)"
      tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
      sleep 2
      break
    fi

    # Check for completion (iter 50)
    if [ -f "${LOG_FILE}" ] && grep -q "iteration\s*50/" "${LOG_FILE}"; then
      echo "[sweep] ${cfg}: iter 50 reached at $(date '+%H:%M:%S') (elapsed ${ELAPSED}s)"
      # Give a moment for clean shutdown
      sleep 5
      if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        break
      fi
      # If still running after iter 50, wait a bit more then kill
      sleep 10
      tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
      break
    fi

    sleep ${POLL_INTERVAL}
    ELAPSED=$((ELAPSED + POLL_INTERVAL))

    # Progress indicator every 60s
    if [ $((ELAPSED % 60)) -eq 0 ]; then
      local_iter=$(grep -c "iteration" "${LOG_FILE}" 2>/dev/null || echo "0")
      echo "[sweep] ${cfg}: waiting... elapsed=${ELAPSED}s, iterations_seen=${local_iter}"
    fi
  done

  if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "[sweep] ${cfg}: TIMEOUT after ${MAX_WAIT}s"
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    sleep 2
  fi

  # Kill any leftover processes
  pkill -f "pretrain_mamba" 2>/dev/null || true
  sleep 2

  # Extract metrics
  RESULT=$(extract_metrics "${LOG_FILE}" "${cfg}" "${CFG_GBS[$cfg]}")
  echo "${RESULT}" >> "${RESULTS_CSV}"
  echo "[sweep] ${cfg} result: ${RESULT}"

  # nvidia-smi snapshot between runs
  echo "[sweep] GPU state after ${cfg}:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo ""
done

echo ""
echo "=================================================================="
echo "[sweep] All configs complete at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo ""
echo "Results CSV: ${RESULTS_CSV}"
cat "${RESULTS_CSV}"
