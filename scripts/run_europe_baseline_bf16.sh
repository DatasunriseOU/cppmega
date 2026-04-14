#!/usr/bin/env bash
# ============================================================================
# europe baseline BF16 — NAM56R on 8xH200, PP=1 EP=4 MBS=8 BF16
# ============================================================================
#
# Model:         NAM56R (52 hybrid Mamba3-MIMO + 9 DSA + 4 full-attn layers)
# Hardware:      8x H200 SXM, LOCATION_2 (h200_1)
# Precision:     BF16 (every FP8 variant regresses on this fabric, see
#                reference_europe_fp8_all_paths_regress.md)
# Topology:      TP=1 PP=1 EP=4 DP=2, MBS=8 GBS=64, seq_len=4096, MTP=2
# Target:        ~289 TFLOP/s steady-state (29.2% MFU), europe gold record
# Peak memory:   ~127 GiB / 141 GiB per rank (MBS=10 OOMs on this machine)
# Iterations:    15 (override with TRAIN_ITERS env)
#
# Prerequisites:
#   - europe stack: ${REMOTE_ROOT:-/home/dave/cppmega-root}/cppmega-venv,
#     torch 2.12+cu132, mamba_ssm 2.3.1, TE 2.13, Megatron 0.18
#   - commit a9ebb78+ (post-dd4da34 unconditional patches)
#   - Dataset prepared via scripts/data_prep_parquet_to_megatron.py
#
# Always-on patches: identical to bench3; see README.md "Always On" section.
#
# Logs:
#   ${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v1.log
#
# See docs/production_status.md for the canonical measurement context.
# See docs/reproducible_runs.md for launch walkthrough.
# ============================================================================
set -euo pipefail

exec env \
  VARIANT=v1 \
  PP_SIZE=1 \
  VPP_SIZE=1 \
  EP_SIZE=4 \
  TRAIN_ITERS="${TRAIN_ITERS:-15}" \
  MBS=8 \
  GBS=64 \
  CG_FLAGS=NONE \
  CPPMEGA_INDEX_CACHE=1 \
  CPPMEGA_LEMYX_DSA=1 \
  CPPMEGA_LINEAR_CE_KERNEL=liger \
  EXTRA_FLAGS="--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear" \
  bash "$(dirname "$0")/remote_smoke_h200_dsa_9_4_m.sh"
