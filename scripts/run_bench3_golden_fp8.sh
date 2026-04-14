#!/usr/bin/env bash
# ============================================================================
# bench3 golden FP8 — NAM56R on 8xH200, PP=1 EP=8 MBS=10 FP8 tensorwise
# ============================================================================
#
# Model:         NAM56R (52 hybrid Mamba3-MIMO + 9 DSA + 4 full-attn layers)
# Hardware:      8x H200 SXM, LOCATION_1 (h200_1)
# Precision:     FP8 tensorwise (global --fp8-format hybrid)
# Topology:      TP=1 PP=1 EP=8 DP=1, MBS=10 GBS=80, seq_len=4096, MTP=2
# Target:        ~268 TFLOP/s steady-state (27.1% MFU), σ < 0.5
# Peak memory:   ~115 GiB / 141 GiB per rank (CG_FLAGS=NONE required)
# Iterations:    15 (override with TRAIN_ITERS env)
#
# Prerequisites:
#   - bench3 stack: torch 2.12+cu132, mamba_ssm 2.3.1, TE 2.13, Megatron 0.18
#   - ${REMOTE_ROOT:-/mnt/data/cppmega-root}/cppmega checked out at commit
#     a9ebb78+ (post-dd4da34 unconditional patches)
#   - Dataset prepared via scripts/data_prep_parquet_to_megatron.py
#
# Always-on patches (no env gate, unconditional since dd4da34):
#   - MTP Liger fused CE (reduction="mean" + broadcast, Liger #968 workaround)
#   - Mamba LinearCE class-swap on output_layer (PR #3226->#3207 regression fix)
#   - DSA indexer fused per-head bmm (saves ~40 GiB at MBS=10)
#   - Mamba3 regional torch.compile (5.93x data-dep-A fusion)
#   See README.md "Always On" section for the full list.
#
# Logs:
#   ${REMOTE_ROOT}/cppmega/cppmega_nam56r_dsa_9_4_fp8_m_v3.log
#
# See docs/production_status.md for the canonical measurement context.
# See docs/reproducible_runs.md for launch walkthrough.
# ============================================================================
set -euo pipefail

exec env \
  VARIANT=v3 \
  PP_SIZE=1 \
  VPP_SIZE=1 \
  EP_SIZE=8 \
  DP_SIZE=1 \
  TRAIN_ITERS="${TRAIN_ITERS:-15}" \
  MBS=10 \
  GBS=80 \
  CG_FLAGS=NONE \
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" \
  CPPMEGA_INDEX_CACHE=1 \
  CPPMEGA_LEMYX_DSA=1 \
  CPPMEGA_LINEAR_CE_KERNEL=liger \
  CPPMEGA_DSA_SKIP_INDEXER_LOSS=0 \
  CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001 \
  EXTRA_FLAGS="--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear --recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj m2rnn --mla-down-proj-fusion --clip-grad 1.0" \
  bash "$(dirname "$0")/remote_smoke_h200_dsa_9_4_m.sh"
