#!/bin/bash
# Bench3 FP8 MBS=8 EP=8 v3 — same as golden bench3 but MBS=8 instead of 10
# to give ~10 GiB memory margin. MBS=10 OOMs at 134.65 GiB / 139.80 cap on
# bench3 H200 with current always-on patches (2026-04-15 verified).
#
# Target throughput: ~80% of golden 268 = ~210-220 TFLOP/s with reduced
# batch. Acceptable as proof-of-life run.
set -e
exec env \
  VARIANT=v3 PP_SIZE=1 VPP_SIZE=1 EP_SIZE=8 DP_SIZE=1 \
  TRAIN_ITERS="${TRAIN_ITERS:-7}" MBS=8 GBS=64 \
  CG_FLAGS=NONE \
  FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max" \
  CPPMEGA_INDEX_CACHE=1 CPPMEGA_LEMYX_DSA=1 \
  CPPMEGA_LINEAR_CE_KERNEL=liger \
  CPPMEGA_DSA_SKIP_INDEXER_LOSS=0 \
  CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001 \
  EXTRA_FLAGS="--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear --recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj m2rnn --mla-down-proj-fusion --clip-grad 1.0" \
  bash "$(dirname "$0")/remote_smoke_h200_dsa_9_4_m.sh"
