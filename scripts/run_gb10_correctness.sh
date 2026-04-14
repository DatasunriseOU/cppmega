#!/usr/bin/env bash
# ============================================================================
# GB10 correctness — single-GPU TileLang kernel sanity test on sm_121
# ============================================================================
#
# Model:         NAM56R (single-GPU, no MoE since EP=1 = only 1 expert/rank)
# Hardware:      1x NVIDIA GB10 (Grace+Blackwell, sm_121, 128 GB unified)
# Precision:     BF16 (FP8 dead path on GB10, see reference_fp8_mamba_ssm_dead_path.md)
# Topology:      TP=1 PP=1 EP=1 DP=1, MBS=1 GBS=1, seq_len=2048, MTP=0
# Goal:          Verify TileLang kernels compile + produce finite grads on
#                sm_121 (no tcgen05, no WGMMA, 99 KiB smem cap).
# Iterations:    5 (override with TRAIN_ITERS env)
#
# This is a CORRECTNESS test, not a throughput run. Expected throughput is
# ~single-digit TFLOP/s on sm_121a (no tcgen05/TMEM/WGMMA, use -arch=sm_120f
# for the 9x-better perf path). The goal is: did the kernels compile, did
# the preflight smem check pass, are grads finite at iter 5?
#
# Prerequisites:
#   - GB10 working stack per reference_gb10_working_stack_2026_04.md
#   - TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True already applied to the
#     installed mamba_ssm mamba3_mimo_* pass_configs (see
#     scripts/remote_train_gb10_nam56r_single.sh preamble)
#   - REMOTE_ROOT defaults to /home/dave on GB10
#
# Environment knobs:
#   CPPMEGA_SMEM_CHECK=1          — enable preflight smem static audit
#                                   (hard-fails if any kernel breaches the
#                                   99 KiB sm_121 cap)
#   CPPMEGA_SMEM_CHECK_STRICT=1   — promote warnings to errors
#
# Logs:
#   ${REMOTE_ROOT}/cppmega/cppmega_nam56r_gb10_correctness.log
#
# See docs/reproducible_runs.md for interpretation guidance.
# ============================================================================
set -euo pipefail

# Force single-GPU + Megatron single-rank defaults. No MoE (EP=1 collapses
# the flex dispatcher to alltoall; combined with DP=1 means 1 expert/rank).
exec env \
  VARIANT=v0 \
  TP_SIZE=1 \
  PP_SIZE=1 \
  VPP_SIZE=1 \
  EP_SIZE=1 \
  DP_SIZE=1 \
  MBS=1 \
  GBS=1 \
  SEQ_LEN=2048 \
  TRAIN_ITERS="${TRAIN_ITERS:-5}" \
  MTP_DEPTHS=0 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  CG_FLAGS=NONE \
  CPPMEGA_SMEM_CHECK="${CPPMEGA_SMEM_CHECK:-1}" \
  CPPMEGA_INDEX_CACHE=0 \
  CPPMEGA_LEMYX_DSA=0 \
  CPPMEGA_LINEAR_CE_KERNEL=auto \
  bash "$(dirname "$0")/remote_smoke_h200_dsa_9_4_m.sh"
