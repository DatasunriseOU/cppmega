set -euo pipefail
cp -a /overlay/. /opt/cppmega/
export PYTHONPATH="/opt/cppmega:/opt/megatron-lm:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="/data/.triton-cache"
mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_results

python - <<'PY'
import importlib
import json
import torch

modules = [
    "torch",
    "transformer_engine",
    "transformer_engine.pytorch",
    "flash_attn",
    "flash_attn_3",
    "flash_attn.cute",
    "cutlass",
    "quack",
    "mamba_ssm",
    "megatron.core",
    "cppmega",
]
report = {}
for name in modules:
    mod = importlib.import_module(name)
    report[name] = {
        "file": getattr(mod, "__file__", None),
        "version": getattr(mod, "__version__", None),
    }
import megatron.core.utils as core_utils
report["megatron.core.utils.get_batch_on_this_tp_rank"] = hasattr(
    core_utils, "get_batch_on_this_tp_rank"
)
import cppmega.megatron.structure_dataset_patch
report["cuda"] = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
}
print("CPPMEGA_STACK_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
assert report["megatron.core.utils.get_batch_on_this_tp_rank"], report
PY

for BS in 256 512 1024; do
  LOG="/data/cppmega_h200_results/bs_${BS}.log"
  NVSMI="/data/cppmega_h200_results/bs_${BS}.nvsmi.csv"
  echo "[container] starting batch=${BS}" | tee "$LOG"
  (
    while true; do
      ts="$(date '+%Y-%m-%dT%H:%M:%S')"
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu                 --format=csv,noheader,nounits |
        while IFS=, read -r mu mt ug tg; do
          echo "${ts},${mu},${mt},${ug},${tg}"
        done
      sleep 1
    done
  ) > "$NVSMI" 2>&1 &
  NVSMI_PID=$!

  set +e
  bash -lc "
    set -euo pipefail
    WORKDIR=\$(mktemp -d /tmp/cppmega-h200-world.XXXXXX)
    trap 'rm -rf \"\$WORKDIR\"' EXIT
    cp /opt/megatron-lm/pretrain_mamba.py \"\$WORKDIR/pretrain_mamba_inner.py\"
    cat >\"\$WORKDIR/pretrain_mamba.py\" <<'PYWRAP'
from __future__ import annotations
import atexit
import os
import runpy
import sys

if os.environ.get('CPPMEGA_STRUCTURE_ENABLED', '0') == '1':
    import cppmega.megatron.structure_dataset_patch  # noqa: F401

@atexit.register
def _cppmega_peak_memory_report():
    try:
        import torch
        if torch.cuda.is_available():
            print(
                'CPPMEGA_CUDA_PEAK allocated_gib='
                f'{torch.cuda.max_memory_allocated() / 1024**3:.3f} '
                'reserved_gib='
                f'{torch.cuda.max_memory_reserved() / 1024**3:.3f}',
                flush=True,
            )
    except Exception as exc:
        print(f'CPPMEGA_CUDA_PEAK_ERROR {exc}', flush=True)

_inner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrain_mamba_inner.py')
sys.path.insert(0, os.path.dirname(_inner))
sys.argv[0] = _inner
runpy.run_path(_inner, run_name='__main__')
PYWRAP
    cat >\"\$WORKDIR/mamba_builders.py\" <<'PY'
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
PY
    cat >\"\$WORKDIR/model_provider.py\" <<'PY'
from megatron.training import get_args

def model_provider(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    args = get_args()
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
PY
    eval \"\$(python -m cppmega.recipes.run_profiles shell h200_cpp_world_mini \
      --seq-length 1024 \
      --micro-batch-size ${BS} \
      --global-batch-size ${BS} \
      --train-iters 3 \
      --fp8-recipe off)\"

    DATA_ARGS=(--data-path 1.0 /data/cppmega_sidecar/cppmega_1024_smoke_mix_train)
    OPTIMIZER_ARGS=(--optimizer \"\$CPPMEGA_OPTIMIZER\")
    if [[ \"\$CPPMEGA_OPTIMIZER\" == muon || \"\$CPPMEGA_OPTIMIZER\" == dist_muon || \"\$CPPMEGA_OPTIMIZER\" == adaptive_muon ]]; then
      OPTIMIZER_ARGS+=(--muon-momentum \"\$CPPMEGA_MUON_MOMENTUM\" --muon-scale-mode \"\$CPPMEGA_MUON_SCALE_MODE\" --muon-num-ns-steps \"\$CPPMEGA_MUON_NUM_NS_STEPS\" --muon-tp-mode \"\$CPPMEGA_MUON_TP_MODE\" --muon-scalar-optimizer \"\$CPPMEGA_MUON_SCALAR_OPTIMIZER\")
      if [[ \"\$CPPMEGA_MUON_QUANTIZED_MOMENTUM\" == 1 ]]; then
        OPTIMIZER_ARGS+=(--muon-quantized-momentum --muon-quantized-momentum-dtype \"\$CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE\" --muon-quantized-momentum-block-size \"\$CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE\")
      fi
    fi
    if [[ \"\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER\" == 1 ]]; then OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-optimizer); fi
    if [[ \"\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER\" == 1 ]]; then OPTIMIZER_ARGS+=(--use-bf16-no-master-emerging-fallback-optimizer); fi
    if [[ \"\$CPPMEGA_GRAD_REDUCE_IN_BF16\" == 1 || \"\$CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER\" == 1 ]]; then OPTIMIZER_ARGS+=(--grad-reduce-in-bf16); fi
    if [[ \"\$CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER\" == 1 ]]; then OPTIMIZER_ARGS+=(--local-ddp-disable-contiguous-grad-buffer); fi

    GQA_ARGS=(--group-query-attention --num-query-groups \"\$CPPMEGA_NUM_QUERY_GROUPS\" --kv-channels \"\$CPPMEGA_KV_CHANNELS\" --swiglu --rotary-base 10000)

    python -m torch.distributed.run --nproc_per_node=1 \"\$WORKDIR/pretrain_mamba.py\" \
      \"\${DATA_ARGS[@]}\" \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model /data/cpp_tokenizer_hf \
      --vocab-size 65536 \
      --make-vocab-size-divisible-by 128 \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --context-parallel-size 1 \
      --no-gradient-accumulation-fusion \
      --no-persist-layer-norm \
      --no-masked-softmax-fusion \
      --hybrid-layer-pattern \"\$HYBRID_LAYER_PATTERN\" \
      --hidden-size \"\$CPPMEGA_HIDDEN_SIZE\" \
      --ffn-hidden-size \"\$CPPMEGA_FFN_HIDDEN_SIZE\" \
      --num-attention-heads \"\$CPPMEGA_NUM_ATTN_HEADS\" \
      \"\${GQA_ARGS[@]}\" \
      --seq-length 1024 \
      --max-position-embeddings 1024 \
      --micro-batch-size ${BS} \
      --global-batch-size ${BS} \
      --train-iters 3 \
      --eval-interval 50000000 \
      --eval-iters 0 \
      --lr \"\$CPPMEGA_LR\" \
      --min-lr \"\$CPPMEGA_MIN_LR\" \
      --lr-decay-style constant \
      --position-embedding-type rope \
      --no-rope-fusion \
      --normalization RMSNorm \
      --disable-bias-linear \
      --bf16 \
      --use-mcore-models \
      --transformer-impl transformer_engine \
      --use-flash-attn \
      --attention-backend flash \
      --spec cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec \
      --cross-entropy-loss-fusion \
      --cross-entropy-fusion-impl linear \
      --recompute-granularity selective \
      --recompute-modules mlp \
      --clip-grad 1.0 \
      \"\${OPTIMIZER_ARGS[@]}\" \
      --no-check-for-nan-in-loss-and-grad \
      --rerun-mode disabled \
      --save-interval 50000000 \
      --log-interval 1
  " >>"$LOG" 2>&1
  status=$?
  kill "$NVSMI_PID" 2>/dev/null || true
  wait "$NVSMI_PID" 2>/dev/null || true
  set -e
  peak="$(awk -F, '{ if ($2+0 > peak) peak=$2+0 } END { print peak+0 }' "$NVSMI")"
  echo "CPPMEGA_NVIDIA_SMI_PEAK batch=${BS} peak_used_mib=${peak}" | tee -a "$LOG"
  if [[ "$status" != 0 ]]; then
    echo "CPPMEGA_BATCH_RESULT batch=${BS} status=FAIL exit=${status}" | tee -a "$LOG"
    if grep -qiE 'out of memory|cuda error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$LOG"; then
      echo "CPPMEGA_BATCH_OOM batch=${BS}" | tee -a "$LOG"
      exit 0
    fi
    exit "$status"
  fi
  echo "CPPMEGA_BATCH_RESULT batch=${BS} status=OK" | tee -a "$LOG"
done
