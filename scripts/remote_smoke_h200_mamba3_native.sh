#!/usr/bin/env bash
# Smoke test: CppMegaMamba3Mixer (native SSD kernel + Mamba3 features)
# Compares vanilla nemo_native vs mamba3_native on mock data.
#
# Usage: ssh h200_1 bash -s < scripts/remote_smoke_h200_mamba3_native.sh

set -euo pipefail
MEGATRON_DIR="${MEGATRON_DIR:-/home/dave/cppmega-root/megatron-lm}"
CPPMEGA_DIR="${CPPMEGA_DIR:-/home/dave/cppmega-root/cppmega}"
RESULTS_DIR="${RESULTS_DIR:-/home/dave/cppmega-root/benchmarks}"

mkdir -p "$RESULTS_DIR"
export PYTHONPATH="$MEGATRON_DIR:$CPPMEGA_DIR:${PYTHONPATH:-}"

# Common args for both runs
COMMON_ARGS=(
    --num-layers 52
    --hidden-size 3584
    --ffn-hidden-size 18944
    --num-attention-heads 56
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --seq-length 4096
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 65536
    --micro-batch-size 4
    --global-batch-size 32
    --train-iters 20
    --lr 3e-4
    --min-lr 3e-5
    --lr-warmup-iters 5
    --lr-decay-iters 100
    --lr-decay-style cosine
    --bf16
    --is-hybrid-model
    --hybrid-override-pattern "*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME*"
    --mamba-state-dim 64
    --mamba-head-dim 64
    --mamba-num-groups 8
    --num-experts 16
    --moe-router-topk 4
    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --no-gradient-accumulation-fusion
    --cross-entropy-loss-fusion
    --first-last-layers-bf16
    --attention-backend auto
    --mock-data
    --eval-iters 0
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope attn mamba moe_router moe_preprocess
)

echo "================================================================"
echo " Benchmark 1: nemo_native (vanilla Mamba-2 SSD, baseline)"
echo "================================================================"
NEMO_ENV=(
    CUDA_DEVICE_MAX_CONNECTIONS=1
    NVTE_FWD_LAYERNORM_SM_MARGIN=16
    NVTE_BWD_LAYERNORM_SM_MARGIN=16
    NVTE_NORM_FWD_USE_CUDNN=1
    NCCL_AVOID_RECORD_STREAMS=1
    NCCL_GRAPH_REGISTER=0
)

env "${NEMO_ENV[@]}" \
torchrun --nproc-per-node=8 "$MEGATRON_DIR/pretrain_mamba.py" \
    "${COMMON_ARGS[@]}" \
    --mamba-num-heads 56 \
    --eval-interval 100 \
  --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    2>&1 | tee "$RESULTS_DIR/nemo_native_baseline.log"

echo ""
echo "================================================================"
echo " Benchmark 2: mamba3_native (CppMegaMamba3Mixer, QK-Norm + bias)"
echo "================================================================"
MAMBA3_ENV=(
    "${NEMO_ENV[@]}"
    CPPMEGA_MAMBA3_QKNORM=1
    CPPMEGA_MAMBA3_BIAS=1
    CPPMEGA_MAMBA3_DATA_DEP_A=0
)

env "${MAMBA3_ENV[@]}" \
torchrun --nproc-per-node=8 "$MEGATRON_DIR/pretrain_mamba.py" \
    "${COMMON_ARGS[@]}" \
    --mamba-num-heads 56 \
    --eval-interval 100 \
  --spec cppmega.megatron.mamba3_te_stack_spec cppmega_mamba3_te_stack_spec \
    2>&1 | tee "$RESULTS_DIR/mamba3_native_qknorm_bias.log"

echo ""
echo "================================================================"
echo " Benchmark 3: mamba3_native + data-dependent A"
echo "================================================================"
MAMBA3_DDA_ENV=(
    "${NEMO_ENV[@]}"
    CPPMEGA_MAMBA3_QKNORM=1
    CPPMEGA_MAMBA3_BIAS=1
    CPPMEGA_MAMBA3_DATA_DEP_A=1
)

env "${MAMBA3_DDA_ENV[@]}" \
torchrun --nproc-per-node=8 "$MEGATRON_DIR/pretrain_mamba.py" \
    "${COMMON_ARGS[@]}" \
    --mamba-num-heads 56 \
    --eval-interval 100 \
  --spec cppmega.megatron.mamba3_te_stack_spec cppmega_mamba3_te_stack_spec \
    2>&1 | tee "$RESULTS_DIR/mamba3_native_qknorm_bias_dda.log"

echo ""
echo "================================================================"
echo " Results Summary"
echo "================================================================"
for log in "$RESULTS_DIR"/nemo_native_baseline.log \
           "$RESULTS_DIR"/mamba3_native_qknorm_bias.log \
           "$RESULTS_DIR"/mamba3_native_qknorm_bias_dda.log; do
    name=$(basename "$log" .log)
    iter_time=$(grep -oP 'elapsed time per iteration \(ms\): \K[\d.]+' "$log" | tail -5 | awk '{s+=$1;n++}END{if(n>0)printf "%.0f",s/n}')
    if [ -n "$iter_time" ]; then
        tokens_per_iter=$((32 * 4096))
        tok_sec=$(echo "$tokens_per_iter / ($iter_time / 1000)" | bc)
        echo "$name: ${iter_time}ms/iter → ${tok_sec} tok/sec"
    else
        echo "$name: no timing data found"
    fi
done
