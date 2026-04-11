#!/usr/bin/env bash
# FP8 smoke matrix for the 4 cppmega Mamba3 paths on bench3 H200x8.
#
# This is a one-shot diagnostic to empirically verify whether FP8 works for:
#   A -> cppmega.megatron.mamba3_te_stack_spec     (CppMegaMamba3Mixer)
#   B -> cppmega.megatron.mamba3_author_spec       (Author SISO)
#   C -> cppmega.megatron.mamba3_author_spec + MIMO patch (rank=4, chunk=16)
#   D -> cppmega.megatron.nam56r_noconv_spec       (NoConvMamba3BCMixer)
#
# The entry point in scripts/remote_train_h200_nam56r_noconv.sh hardcoded the
# noconv spec and had no FP8 toggle.  Rather than cloning that script 4x, this
# one accepts env vars CPPMEGA_SPEC_MODULE / CPPMEGA_SPEC_NAME + CPPMEGA_FP8 +
# CPPMEGA_MAMBA3_MIMO and routes accordingly.
#
# Usage (per path):
#
#     REMOTE_HOST=h200_1 \
#     REMOTE_ZONE=LOCATION_1 \
#     CPPMEGA_SPEC_MODULE=cppmega.megatron.mamba3_te_stack_spec \
#     CPPMEGA_SPEC_NAME=cppmega_mamba3_te_stack_spec \
#     CPPMEGA_FP8=1 \
#     CPPMEGA_CUDA_GRAPH=none \
#     CPPMEGA_RUN_ID=cppmega_fp8_path_a_nograph \
#     bash scripts/remote_smoke_h200_fp8_mamba3_matrix.sh
#
# For Path C (MIMO) also set CPPMEGA_MAMBA3_MIMO=1.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h200_1}"
REMOTE_ZONE="${REMOTE_ZONE:-LOCATION_1}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data}"
REMOTE_CPPMEGA_ROOT="${REMOTE_CPPMEGA_ROOT:-/mnt/data/cppmega-root/cppmega}"
REMOTE_MEGATRON_ROOT="${REMOTE_MEGATRON_ROOT:-/mnt/data/cppmega-root/megatron-lm}"
REMOTE_VENV="${REMOTE_VENV:-/mnt/data/venv}"
CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-cppmega_fp8_matrix_run}"
REMOTE_LOG="${REMOTE_LOG:-/home/dave/cppmega/${CPPMEGA_RUN_ID}.log}"
REMOTE_TMP_SCRIPT="${REMOTE_TMP_SCRIPT:-/tmp/cppmega-remote-fp8-matrix.sh}"
LOCAL_TMP_SCRIPT="$(mktemp -t cppmega-remote-fp8-matrix.XXXXXX.sh)"

trap 'rm -f "${LOCAL_TMP_SCRIPT}"' EXIT

# Defaults (keep small for fast iteration -- 5 iters).
: "${CPPMEGA_TRAIN_ITERS:=5}"
: "${CPPMEGA_EVAL_ITERS:=1}"
: "${CPPMEGA_PP_SIZE:=4}"
: "${CPPMEGA_MICRO_BATCH_SIZE:=2}"
: "${CPPMEGA_GLOBAL_BATCH_SIZE:=16}"
: "${CPPMEGA_SAVE_INTERVAL:=100}"
: "${CPPMEGA_ATTN_BACKEND:=auto}"
: "${CPPMEGA_CUDA_GRAPH:=none}"
: "${CPPMEGA_FP8:=0}"
: "${CPPMEGA_MAMBA3_MIMO:=0}"
: "${CPPMEGA_MAMBA_NUM_GROUPS:=}"
: "${CPPMEGA_SPEC_MODULE:=cppmega.megatron.nam56r_noconv_spec}"
: "${CPPMEGA_SPEC_NAME:=build_cppmega_nam56r_noconv_stack_spec}"
: "${CPPMEGA_DATA_PATH:=1.0 /home/dave/cppmega-root/data/megatron/clang_commits_4k_v1_train 1.0 /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train}"
: "${CPPMEGA_TOKENIZER_TYPE:=HuggingFaceTokenizer}"
: "${CPPMEGA_TOKENIZER_MODEL:=/home/dave/cppmega-root/cpp_tokenizer_hf}"
: "${CPPMEGA_SEQ_LENGTH:=4096}"

cat > "${LOCAL_TMP_SCRIPT}" <<'INNER'
set -euo pipefail
source "${REMOTE_VENV}/bin/activate"
export PYTHONPATH="${REMOTE_CPPMEGA_ROOT}:${REMOTE_MEGATRON_ROOT}:${PYTHONPATH:-}"

_VENV_CUDNN_LIB="${REMOTE_VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib"
if [ -d "${_VENV_CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${_VENV_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
mkdir -p "$(dirname "${REMOTE_LOG}")" "/mnt/data/.triton-cache"
REMOTE_CKPT_DIR="/home/dave/cppmega/${CPPMEGA_RUN_ID}_ckpt"
REMOTE_WORKDIR="$(mktemp -d /tmp/cppmega-fp8-matrix.XXXXXX)"
trap 'rm -rf "${REMOTE_WORKDIR}"' EXIT

# --- Monkey patch / MIMO shim -------------------------------------------------
#
# Paths B and C import cppmega.megatron.mamba3_te_mixer which does:
#   from megatron.core.inference.contexts.static_context import deprecate_inference_params
# but in the installed megatron-core 0.18rc0 that function lives in
# megatron.core.utils.  Inject a compatibility alias before pretrain_mamba runs.
#
# Path C additionally needs TransformerConfig.cppmega_mamba3_is_mimo=True set at
# __init__ time via a __post_init__ hook.

cat > "${REMOTE_WORKDIR}/cppmega_fp8_shim.py" <<'PY'
"""Shim: install deprecate_inference_params alias and optional MIMO patch."""
from __future__ import annotations

import os

# (1) deprecate_inference_params compatibility shim.
try:
    from megatron.core.inference.contexts import static_context as _sc
    if not hasattr(_sc, "deprecate_inference_params"):
        try:
            from megatron.core.utils import deprecate_inference_params as _dip
        except ImportError:
            def _dip(inference_context, inference_params):
                if inference_context is None and inference_params is not None:
                    return inference_params
                return inference_context
        _sc.deprecate_inference_params = _dip
except Exception as _exc:  # pragma: no cover
    import sys
    print(f"[cppmega_fp8_shim] static_context alias skipped: {_exc}", file=sys.stderr)

# (2) Optional MIMO patch for Path C: add default fields to TransformerConfig
# so the author Mamba3 mixer picks up is_mimo=True / rank=4 / chunk=16.
# Also optional mamba_num_groups override via CPPMEGA_MAMBA_NUM_GROUPS.
_mimo_on = os.environ.get("CPPMEGA_MAMBA3_MIMO", "0") == "1"
_num_groups_override = os.environ.get("CPPMEGA_MAMBA_NUM_GROUPS", "")
if _mimo_on or _num_groups_override:
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        _orig_post_init = TransformerConfig.__post_init__

        def _cppmega_mimo_post_init(self):
            if _num_groups_override:
                try:
                    _ng = int(_num_groups_override)
                    object.__setattr__(self, "mamba_num_groups", _ng)
                    print(f"[cppmega_fp8_shim] mamba_num_groups override -> {_ng}")
                except Exception as _e:
                    import sys
                    print(f"[cppmega_fp8_shim] mamba_num_groups override failed: {_e}", file=sys.stderr)
            _orig_post_init(self)
            if _mimo_on:
                if not getattr(self, "cppmega_mamba3_is_mimo", False):
                    object.__setattr__(self, "cppmega_mamba3_is_mimo", True)
                if not getattr(self, "cppmega_mamba3_mimo_rank", None):
                    object.__setattr__(self, "cppmega_mamba3_mimo_rank", 4)
                if not getattr(self, "cppmega_mamba3_chunk_size", None):
                    object.__setattr__(self, "cppmega_mamba3_chunk_size", 16)

        TransformerConfig.__post_init__ = _cppmega_mimo_post_init
        if _mimo_on:
            print("[cppmega_fp8_shim] MIMO patch installed (rank=4, chunk=16)")
        if _num_groups_override:
            print(f"[cppmega_fp8_shim] mamba_num_groups override arm set -> {_num_groups_override}")
    except Exception as _exc:  # pragma: no cover
        import sys
        print(f"[cppmega_fp8_shim] MIMO patch failed: {_exc}", file=sys.stderr)

# (3) Always allow dynamic mamba3 attr lookups on the dataclass.
# IMPORTANT: we must RAISE AttributeError for unset cppmega_mamba3_* attrs so
# that ``getattr(config, "cppmega_mamba3_rope_fraction", 0.5)`` falls back to
# the supplied default 0.5 (returning None would propagate through and trip
# asserts like ``rope_fraction in (0.5, 1.0)`` inside CppMegaMamba3TE).
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    _TC_BASE_GETATTR = getattr(TransformerConfig, "__getattr__", None)

    def _cppmega_getattr(self, name):
        if name.startswith("cppmega_mamba3_"):
            raise AttributeError(name)
        if _TC_BASE_GETATTR is not None:
            return _TC_BASE_GETATTR(self, name)
        raise AttributeError(name)

    if not hasattr(TransformerConfig, "_cppmega_mamba3_attr_patched"):
        TransformerConfig.__getattr__ = _cppmega_getattr
        TransformerConfig._cppmega_mamba3_attr_patched = True
except Exception:
    pass
PY

# --- Quick import smoke -------------------------------------------------------
PYTHONPATH="${REMOTE_WORKDIR}:${PYTHONPATH:-}" python - <<PY
import cppmega_fp8_shim  # noqa: F401 -- installs compat aliases
import cppmega
import megatron
import transformer_engine
print("import smoke ok", cppmega.__version__, "TE", transformer_engine.__version__)
import importlib
mod = importlib.import_module("${CPPMEGA_SPEC_MODULE}")
obj = getattr(mod, "${CPPMEGA_SPEC_NAME}")
print("spec module ok:", "${CPPMEGA_SPEC_MODULE}.${CPPMEGA_SPEC_NAME}", type(obj).__name__)
PY

# --- Copy pretrain_mamba and wrap so the shim loads first ---------------------
cp "${REMOTE_MEGATRON_ROOT}/pretrain_mamba.py" "${REMOTE_WORKDIR}/pretrain_mamba_original.py"

cat > "${REMOTE_WORKDIR}/pretrain_mamba.py" <<'PY'
import os, sys, runpy
# Ensure the workdir is on sys.path so cppmega_fp8_shim is importable.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import cppmega_fp8_shim  # noqa: F401  -- install compat aliases

# Delegate to the original Megatron pretrain_mamba (which calls pretrain()).
runpy.run_path(
    os.path.join(_here, "pretrain_mamba_original.py"),
    run_name="__main__",
)
PY

cat > "${REMOTE_WORKDIR}/mamba_builders.py" <<'PY'
"""Workdir mamba_builders shim that wraps cppmega_mamba_builder.

The upstream cppmega_mamba_builder has a legacy callable-check that treats a
ModuleSpec instance (which implements __call__) as a factory, then invokes it
with just ``config`` which fails the MambaStack constructor's ``pg_collection``
assertion.  Here we intercept that case: if the imported spec object is a
ModuleSpec we pass it through untouched; only plain factory functions get
called with ``config``.
"""
from megatron.core.transformer.spec_utils import import_module, ModuleSpec
from cppmega.megatron.mamba_builder import cppmega_mamba_builder as _impl


def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    # Resolve the spec here so we can special-case ModuleSpec BEFORE the
    # upstream (buggy) callable-check inside _impl fires.
    if args.spec is None:
        raise ValueError("mamba_builders shim requires --spec")
    spec_obj = import_module(args.spec)
    if isinstance(spec_obj, ModuleSpec):
        # Monkeypatch the args.spec temporarily to hand back the resolved
        # ModuleSpec via a trivial lambda that _impl will not call (lambdas
        # are callable but _impl's branch still invokes with config).  Easier:
        # directly construct via _impl's inner flow using a dummy wrapper.
        # Implementation: we replicate cppmega_mamba_builder behaviour but
        # pass the ModuleSpec straight through.
        from cppmega.megatron.custom_mamba_model import CppMegaMambaModel
        from megatron.core.transformer import TransformerConfig
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.training import print_rank_0
        if config is None:
            config = core_transformer_config_from_args(args, TransformerConfig)
        print_rank_0("building cppmega MAMBA model (workdir shim, ModuleSpec path)...")
        return CppMegaMambaModel(
            config=config,
            mamba_stack_spec=spec_obj,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            hybrid_layer_pattern=args.hybrid_layer_pattern,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    # Factory function path -- delegate to the existing cppmega builder.
    return _impl(args, pre_process, post_process, vp_stage=vp_stage, config=config, pg_collection=pg_collection)
PY

cat > "${REMOTE_WORKDIR}/model_provider.py" <<'PY'
from megatron.training import get_args


def model_provider(
    model_builder,
    pre_process=True,
    post_process=True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    return model_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
PY

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="/mnt/data/.triton-cache"
export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"
export CPPMEGA_LAYER_DEPTH=52
export CPPMEGA_R_LAYER_INDICES="12,24,36,48"
export CPPMEGA_NGRAM_HASH_ENABLED="${CPPMEGA_NGRAM_HASH_ENABLED:-1}"
export CPPMEGA_NGRAM_HASH_ORDERS="${CPPMEGA_NGRAM_HASH_ORDERS:-2,3}"
export CPPMEGA_NGRAM_HASH_HEADS="${CPPMEGA_NGRAM_HASH_HEADS:-8}"
export CPPMEGA_NGRAM_HASH_TABLE_SIZE="${CPPMEGA_NGRAM_HASH_TABLE_SIZE:-500000}"
export CPPMEGA_NGRAM_HASH_EMBED_DIM="${CPPMEGA_NGRAM_HASH_EMBED_DIM:-16}"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
export CPPMEGA_STRUCTURE_COMPONENTS="${CPPMEGA_STRUCTURE_COMPONENTS:-core}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

HYBRID_LAYER_PATTERN="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
print(build_default_hybrid_layer_pattern(mtp_depths=1))
PY
)"

NATIVE_ARGS="$(${REMOTE_VENV}/bin/python - <<'PY'
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

plan = build_nam56r_feature_plan(pattern='AEMEAEMEAEMR', depth=52, mtp_depths=1)
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=True,
    mtp_mode='hybrid',
    enable_moe=True,
)
print(bundle.to_shell_fragment())
PY
)"

DATA_ARGS=(--data-path ${CPPMEGA_DATA_PATH})
DATA_ARGS+=(--split "${CPPMEGA_DATA_SPLIT:-98,1,1}")
DATA_ARGS+=(
  --tokenizer-type "${CPPMEGA_TOKENIZER_TYPE}"
  --tokenizer-model "${CPPMEGA_TOKENIZER_MODEL}"
)

CUDA_GRAPH_ARGS=()
case "${CPPMEGA_CUDA_GRAPH}" in
  te_attn)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-scope attn
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
    )
    ;;
  te_full)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl transformer_engine
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
    )
    ;;
  local_full)
    CUDA_GRAPH_ARGS+=(
      --cuda-graph-impl local
      --cuda-graph-scope full_iteration
      --cuda-graph-warmup-steps "${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}"
      --no-check-for-nan-in-loss-and-grad
    )
    ;;
  none)
    CUDA_GRAPH_ARGS+=(--cuda-graph-impl none)
    ;;
  *)
    echo "Unknown CPPMEGA_CUDA_GRAPH=${CPPMEGA_CUDA_GRAPH}" >&2
    exit 2
    ;;
esac

FP8_ARGS=()
if [ "${CPPMEGA_FP8}" = "1" ]; then
  FP8_ARGS+=(
    --fp8-format hybrid
    --fp8-amax-history-len 16
    --fp8-amax-compute-algo max
  )
fi

echo "=== FP8 matrix run ==="
echo "SPEC: ${CPPMEGA_SPEC_MODULE} ${CPPMEGA_SPEC_NAME}"
echo "FP8: ${CPPMEGA_FP8}  CUDA_GRAPH: ${CPPMEGA_CUDA_GRAPH}  MIMO: ${CPPMEGA_MAMBA3_MIMO}"
echo "Iters: ${CPPMEGA_TRAIN_ITERS}  MBS: ${CPPMEGA_MICRO_BATCH_SIZE}  GBS: ${CPPMEGA_GLOBAL_BATCH_SIZE}"

python -m torch.distributed.run --nproc_per_node=8 "${REMOTE_WORKDIR}/pretrain_mamba.py" \
  "${DATA_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size "${CPPMEGA_PP_SIZE}" \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "${HYBRID_LAYER_PATTERN}" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --seq-length "${CPPMEGA_SEQ_LENGTH}" \
  --max-position-embeddings "${CPPMEGA_SEQ_LENGTH}" \
  --micro-batch-size "${CPPMEGA_MICRO_BATCH_SIZE}" \
  --global-batch-size "${CPPMEGA_GLOBAL_BATCH_SIZE}" \
  --train-iters "${CPPMEGA_TRAIN_ITERS}" \
  --eval-interval 50000000 \
  --eval-iters "${CPPMEGA_EVAL_ITERS}" \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --attention-backend "${CPPMEGA_ATTN_BACKEND}" \
  --spec "${CPPMEGA_SPEC_MODULE}" "${CPPMEGA_SPEC_NAME}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${FP8_ARGS[@]}" \
  ${NATIVE_ARGS} \
  --save "${REMOTE_CKPT_DIR}" \
  --save-interval "${CPPMEGA_SAVE_INTERVAL}" \
  --log-interval 1 \
  > "${REMOTE_LOG}" 2>&1

tail -n 120 "${REMOTE_LOG}"
INNER

gcloud compute scp --zone "${REMOTE_ZONE}" "${LOCAL_TMP_SCRIPT}" "${REMOTE_HOST}:${REMOTE_TMP_SCRIPT}"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_CPPMEGA_ROOT='${REMOTE_CPPMEGA_ROOT}' REMOTE_MEGATRON_ROOT='${REMOTE_MEGATRON_ROOT}' REMOTE_VENV='${REMOTE_VENV}' REMOTE_LOG='${REMOTE_LOG}' CPPMEGA_RUN_ID='${CPPMEGA_RUN_ID}' CPPMEGA_TRAIN_ITERS='${CPPMEGA_TRAIN_ITERS}' CPPMEGA_PP_SIZE='${CPPMEGA_PP_SIZE}' CPPMEGA_SEQ_LENGTH='${CPPMEGA_SEQ_LENGTH}' CPPMEGA_MICRO_BATCH_SIZE='${CPPMEGA_MICRO_BATCH_SIZE}' CPPMEGA_GLOBAL_BATCH_SIZE='${CPPMEGA_GLOBAL_BATCH_SIZE}' CPPMEGA_DATA_PATH='${CPPMEGA_DATA_PATH}' CPPMEGA_DATA_SPLIT='${CPPMEGA_DATA_SPLIT:-98,1,1}' CPPMEGA_TOKENIZER_TYPE='${CPPMEGA_TOKENIZER_TYPE}' CPPMEGA_TOKENIZER_MODEL='${CPPMEGA_TOKENIZER_MODEL}' CPPMEGA_EVAL_ITERS='${CPPMEGA_EVAL_ITERS}' CPPMEGA_SAVE_INTERVAL='${CPPMEGA_SAVE_INTERVAL}' CPPMEGA_ATTN_BACKEND='${CPPMEGA_ATTN_BACKEND}' CPPMEGA_CUDA_GRAPH='${CPPMEGA_CUDA_GRAPH}' CPPMEGA_CUDA_GRAPH_WARMUP_STEPS='${CPPMEGA_CUDA_GRAPH_WARMUP_STEPS:-3}' CPPMEGA_FP8='${CPPMEGA_FP8}' CPPMEGA_MAMBA3_MIMO='${CPPMEGA_MAMBA3_MIMO}' CPPMEGA_MAMBA_NUM_GROUPS='${CPPMEGA_MAMBA_NUM_GROUPS}' CPPMEGA_SPEC_MODULE='${CPPMEGA_SPEC_MODULE}' CPPMEGA_SPEC_NAME='${CPPMEGA_SPEC_NAME}' bash '${REMOTE_TMP_SCRIPT}' || (status=\$?; echo 'remote fp8 matrix run failed; tail follows:'; tail -n 200 '${REMOTE_LOG}' || true; exit \$status)"
gcloud compute ssh "${REMOTE_HOST}" --zone "${REMOTE_ZONE}" --command "rm -f '${REMOTE_TMP_SCRIPT}'"
