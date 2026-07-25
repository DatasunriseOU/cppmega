"""Modal H200: Megatron-backed CppMega curriculum training (same stack as Nebius).

Curriculum: seq 1024 → 2048 → 4096 → 8192, model-only warm start between stages.
Each stage converts its parquet, trains, saves checkpoint.

Usage:
    modal run --detach scripts/modal_megatron_train.py
    modal run --detach scripts/modal_megatron_train.py --stages 1024,2048
    modal run --detach scripts/modal_megatron_train.py --skip-convert
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, cast

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_DIGEST = os.environ.get(
    "GHCR_DIGEST",
    "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983",
)
# FA4 beta23 image digest (flash-attn-4 4.0.0b23 + apache-tvm-ffi >=0.1.12).
# Placeholder until docker/Dockerfile.beta23 is built and pushed to GHCR; see
# docs/fa4_beta23_upgrade_plan.md. Replace with the real digest from:
#   docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/datasunriseou/cppmega:beta23
GHCR_DIGEST_BETA23 = os.environ.get(
    "GHCR_DIGEST_BETA23",
    "sha256:PLACEHOLDER_BETA23_DIGEST_NOT_YET_PUSHED",
)
# Opt into the beta23 image with CPPMEGA_BETA23=1 or a --beta23 flag on the
# modal run command line. Image selection happens at import time because the
# @app.function decorators build the image when the module loads.
USE_BETA23 = os.environ.get("CPPMEGA_BETA23", "0") == "1" or "--beta23" in sys.argv
GHCR_REF = f"{GHCR_REPO}@{GHCR_DIGEST_BETA23 if USE_BETA23 else GHCR_DIGEST}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")
MEGATRON_ROOT = "/opt/megatron-lm"

app = modal.App("cppmega-megatron-train")

parquet_vol = modal.Volume.from_name("nanochat-training-data", create_if_missing=True)
megatron_data_vol = modal.Volume.from_name("cppmega-megatron-data", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("cppmega-megatron-checkpoints", create_if_missing=True)

MEGATRON_DATA_DIR = "/data/megatron"
CHECKPOINT_DIR = "/checkpoints"
TOKENIZER_DIR = "/data/tokenizer"


@dataclass(frozen=True)
class Stage:
    seq: int
    batch: int
    micro_batch: int
    code_dir: str
    commits_dir: str


STAGES = [
    Stage(seq=1024, batch=192, micro_batch=192,
          code_dir="/nanochat_data/parquet/cppmega_code_1k",
          commits_dir="/nanochat_data/parquet/cppmega_commits_1k"),
    Stage(seq=2048, batch=96, micro_batch=96,
          code_dir="/nanochat_data/parquet/cppmega_code_2k",
          commits_dir="/nanochat_data/parquet/cppmega_commits_2k"),
    Stage(seq=4096, batch=48, micro_batch=48,
          code_dir="/nanochat_data/parquet/cppmega_code_4k",
          commits_dir="/nanochat_data/parquet/cppmega_commits_4k"),
    Stage(seq=8192, batch=24, micro_batch=4,
          code_dir="/nanochat_data/parquet/cppmega_code_8k",
          commits_dir="/nanochat_data/parquet/cppmega_commits_8k"),
]


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
            "MEGATRON_LM_REPO": MEGATRON_ROOT,
            "CPPMEGA_MEGATRON_COMMIT": os.environ.get(
                "CPPMEGA_MEGATRON_COMMIT", "980211ae"
            ),
        }
    )
    img = img.pip_install("pyarrow==24.0.0")
    img = (
        img.add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True)
        .add_local_dir(str(_REPO_ROOT / "scripts"), remote_path="/opt/cppmega/scripts", copy=True)
        .add_local_dir(str(_REPO_ROOT / "data"), remote_path="/opt/cppmega/data", copy=True)
        .add_local_dir(str(_REPO_ROOT / "configs"), remote_path="/opt/cppmega/configs", copy=True)
        .add_local_dir(str(_REPO_ROOT / "data" / "tokenizer_v2"), remote_path="/data/tokenizer", copy=True)
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
        .add_local_file(str(_REPO_ROOT / "conftest.py"), remote_path="/opt/cppmega/conftest.py")
    )
    return img


@app.function(
    image=_image(),
    gpu=None,
    timeout=86400,
    volumes={
        "/nanochat_data": parquet_vol,
        "/data/megatron": megatron_data_vol,
    },
)
def convert_all_stages(seq_list: list[int]) -> dict[str, Any]:
    """Convert packed parquet → Megatron indexed binary for all requested stages."""
    import sys as _sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/cppmega:/opt/megatron-lm"

    subprocess.run(
        [_sys.executable, "-m", "pip", "install", "-e", "/opt/cppmega", "--no-deps", "-q"],
        env=env, capture_output=True, check=True,
    )

    results = {}
    for stage in STAGES:
        if stage.seq not in seq_list:
            continue
        for kind, parquet_dir in [("code", stage.code_dir), ("commits", stage.commits_dir)]:
            output_prefix = f"{MEGATRON_DATA_DIR}/cppmega_{kind}_{stage.seq}_train"
            print(f"\n{'='*60}")
            print(f"Converting {kind} seq={stage.seq}: {parquet_dir} -> {output_prefix}")
            print(f"{'='*60}", flush=True)

            cmd = [
                _sys.executable, "-u",
                "/opt/cppmega/scripts/data_prep_parquet_to_megatron.py",
                "--input-dir", parquet_dir,
                "--output-prefix", output_prefix,
                "--split", "all",
                "--token-column", "auto",
                "--length-column", "auto",
                "--dtype", "uint16",
                "--vocab-size", "65536",
            ]
            start = time.time()
            proc = subprocess.run(cmd, env=env, cwd="/opt/cppmega", text=True, timeout=86400)
            elapsed = time.time() - start
            key = f"{kind}_{stage.seq}"
            results[key] = {"exit_code": proc.returncode, "elapsed_s": elapsed}
            print(f"[{key}] exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Conversion failed for {key} with exit {proc.returncode}")

    megatron_data_vol.commit()
    print("\nAll conversions complete, volume committed.", flush=True)
    return results


@app.function(
    image=_image(),
    gpu=GPU_SPEC,
    timeout=86400,
    volumes={
        "/data/megatron": megatron_data_vol,
        "/checkpoints": checkpoint_vol,
        "/nanochat_data": parquet_vol,
    },
)
def train_curriculum(
    seq_list: list[int],
    code_weight: float = 0.7,
    commits_weight: float = 0.3,
    save_interval: int = 500,
    eval_interval: int = 500,
    fa4_score_mod: bool = False,
) -> dict[str, Any]:
    """Run Megatron curriculum training: sequential stages with model-only warm start."""
    import sys as _sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/cppmega:/opt/megatron-lm"
    env["MEGATRON_LM_REPO"] = MEGATRON_ROOT
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["NVTE_DEBUG"] = "1"
    env["NVTE_DEBUG_LEVEL"] = "2"

    subprocess.run(
        [_sys.executable, "-m", "pip", "install", "-e", "/opt/cppmega", "--no-deps", "-q"],
        env=env, capture_output=True, check=True,
    )

    megatron_head = subprocess.run(
        ["git", "-C", MEGATRON_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    env["CPPMEGA_MEGATRON_COMMIT"] = megatron_head or env.get("CPPMEGA_MEGATRON_COMMIT", "")

    workdir = tempfile.mkdtemp(prefix="cppmega-megatron-train.")

    # pretrain_mamba.py wrapper with cppmega patches (identical to Nebius)
    pretrain_wrapper = f"""\
from __future__ import annotations
import atexit
import os
import runpy
import sys

from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

apply_te_checkpoint_kwarg_patch()
if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1':
    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    apply_dsa_indexer_fused_patch()
apply_graph_route_attention_bias_patch()

if os.environ.get('CPPMEGA_STRUCTURE_ENABLED', '0') == '1':
    import cppmega.megatron.structure_dataset_patch  # noqa: F401

@atexit.register
def _cppmega_distributed_shutdown():
    try:
        import torch
        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f'CPPMEGA_DISTRIBUTED_SHUTDOWN_ERROR {{exc}}', flush=True)

@atexit.register
def _cppmega_peak_memory_report():
    try:
        import torch
        if torch.cuda.is_available():
            print(
                'CPPMEGA_CUDA_PEAK allocated_gib='
                f'{{torch.cuda.max_memory_allocated() / 1024**3:.3f}} '
                'reserved_gib='
                f'{{torch.cuda.max_memory_reserved() / 1024**3:.3f}}',
                flush=True,
            )
    except Exception as exc:
        print(f'CPPMEGA_CUDA_PEAK_ERROR {{exc}}', flush=True)

_workdir = os.path.dirname(os.path.abspath(__file__))
_inner = '{MEGATRON_ROOT}/pretrain_mamba.py'
sys.path.insert(0, _workdir)
sys.path.insert(1, os.path.dirname(_inner))
sys.argv[0] = _inner
runpy.run_path(_inner, run_name='__main__')
"""
    with open(f"{workdir}/pretrain_mamba.py", "w") as f:
        f.write(pretrain_wrapper)
    with open(f"{workdir}/mamba_builders.py", "w") as f:
        f.write("from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder\n")
    with open(f"{workdir}/hybrid_builders.py", "w") as f:
        f.write("from cppmega.megatron.mamba_builder import cppmega_mamba_builder as hybrid_builder\n")

    # Get profile shell assignments (base config)
    # Match Nebius curriculum: dense=True (GQA), nam56r_noconv_spec
    # FP8_RECIPE=off is intentional: the FA4 score_mod path does not support
    # FP8 yet. Nebius sidecar run used FP8 with TE dense bias; the FA4 lane
    # must stay BF16-only until FA4+FP8 is validated.
    profile_cmd = [
        _sys.executable, "-c",
        "import sys; sys.path.insert(0, '/opt/cppmega'); "
        "from cppmega.recipes.run_profiles import get_run_profile, profile_shell_assignments; "
        "p = get_run_profile('h200_cpp_world_mini'); "
        "p.training.seq_length = 1024; "
        "p.training.micro_batch_size = 192; "
        "p.training.global_batch_size = 192; "
        "p.training.train_iters = 100; "
        "p.precision.fp8_recipe = 'off'; "
        "assignments = profile_shell_assignments(p); "
        "[print(f'{k}={v}') for k, v in sorted(assignments.items())]; "
        "print(f'NATIVE_ARGS={p.native_args_fragment()}'); "
        "print(f'SPEC_MODULE={p.spec_module}'); "
        "print(f'SPEC_FUNCTION={p.spec_function}'); "
    ]
    profile_result = subprocess.run(
        profile_cmd, env=env, cwd="/opt/cppmega",
        capture_output=True, text=True, check=True,
    )
    native_args_str = ""
    spec_module = "cppmega.megatron.nam56r_noconv_spec"
    spec_function = "build_cppmega_nam56r_noconv_stack_spec"
    for line in profile_result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("NATIVE_ARGS="):
            native_args_str = line[len("NATIVE_ARGS="):]
        elif line.startswith("SPEC_MODULE="):
            spec_module = line[len("SPEC_MODULE="):]
        elif line.startswith("SPEC_FUNCTION="):
            spec_function = line[len("SPEC_FUNCTION="):]
        elif "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            key = key.replace("export ", "").strip()
            val = val.strip().strip('"').strip("'")
            env[key] = val

    # Profile sets /data/cpp_tokenizer_hf (Nebius path); Modal image has /data/tokenizer
    env["CPPMEGA_TOKENIZER_MODEL"] = TOKENIZER_DIR

    # FORCE graph routes + structure + ngram AFTER profile parsing.
    # Dense GQA model: graph attention BIAS enabled (additive bias in attention),
    # but DSA indexer auxiliary objective DISABLED (no DSA layers in dense model).
    # Graph attention bias != DSA indexer loss. Bias goes through attention_bias
    # mechanism; DSA indexer loss requires actual DSA layers which don't exist here.
    env["CPPMEGA_DSA_PATCH_ENABLED"] = "0"
    env["CPPMEGA_DSA_GRAPH_AUX_ENABLED"] = "0"
    env["CPPMEGA_NGRAM_HASH_ENABLED"] = "1"
    env["CPPMEGA_STRUCTURE_ENABLED"] = "1"
    env["CPPMEGA_GRAPH_ROUTES_ENABLED"] = "1"
    env["CPPMEGA_GRAPH_MAX_EDGES"] = "256"
    env["CPPMEGA_GRAPH_MAX_CHUNKS"] = "256"

    # Opt-in FA4 score_mod (default off; profile exports CPPMEGA_FA4_SCORE_MOD=0).
    # Enable only for explicit FA4 testing -- the GHCR image has b19 and the
    # production adapter hasn't been validated end-to-end yet.
    if fa4_score_mod:
        env["CPPMEGA_FA4_SCORE_MOD"] = "1"

    # Parse DSA native args into list
    import shlex
    dsa_native_args = shlex.split(native_args_str) if native_args_str else []

    # Filter out MLA-related flags (conflict with --group-query-attention) and
    # flags we pass explicitly per-stage.  dense=True profiles should not emit
    # these, but filter defensively in case the profile surface changes.
    _MLA_FLAGS_WITH_VALUES = frozenset({
        "--experimental-attention-variant",
        "--q-lora-rank", "--kv-lora-rank",
        "--qk-head-dim", "--qk-pos-emb-head-dim", "--v-head-dim",
        "--dsa-indexer-n-heads", "--dsa-indexer-head-dim",
        "--dsa-indexer-topk",
    })
    _MLA_BOOL_FLAGS = frozenset({
        "--multi-latent-attention", "--multi-head-latent-attention",
    })
    _CONFLICTING_FLAGS_WITH_VALUES = frozenset({
        "--num-attention-heads", "--hidden-size", "--ffn-hidden-size",
        "--seq-length", "--micro-batch-size", "--global-batch-size",
        "--train-iters",
    })
    _filtered: list[str] = []
    _skip_next = False
    for _arg in dsa_native_args:
        if _skip_next:
            _skip_next = False
            continue
        if _arg in _MLA_BOOL_FLAGS:
            continue
        if _arg in _MLA_FLAGS_WITH_VALUES or _arg in _CONFLICTING_FLAGS_WITH_VALUES:
            _skip_next = True  # skip the value token that follows
            continue
        _filtered.append(_arg)
    dsa_native_args = _filtered

    # Derive --num-layers from the hybrid-layer-pattern (main block before any
    # "/" MTP separator, ignoring "|" pipeline-parallel chunk separators).
    _hybrid_pattern = env.get("HYBRID_LAYER_PATTERN", "AF" * 24)
    _main_pattern = _hybrid_pattern.split("/")[0].replace("|", "")
    num_layers = len(_main_pattern)

    stage_results = {}
    prev_ckpt_dir = None

    for stage in STAGES:
        if stage.seq not in seq_list:
            continue

        # Calculate train_iters from data size (1 epoch)
        # tokens_per_step = batch * seq
        # Estimated tokens from parquet row counts
        token_estimates = {
            1024: {"code": 2157_000_000, "commits": 245_000_000},
            2048: {"code": 629_000_000, "commits": 407_000_000},
            4096: {"code": 588_000_000, "commits": 511_000_000},
            8192: {"code": 492_000_000, "commits": 604_000_000},
        }
        est = token_estimates[stage.seq]
        total_tokens = est["code"] + est["commits"]
        tokens_per_step = stage.batch * stage.seq
        train_iters = int(total_tokens / tokens_per_step)

        ckpt_dir = f"{CHECKPOINT_DIR}/stage_seq{stage.seq}_bs{stage.batch}"

        # Data path — Megatron expects blend as separate tokens: weight path weight path
        code_prefix = f"{MEGATRON_DATA_DIR}/cppmega_code_{stage.seq}_train"
        commits_prefix = f"{MEGATRON_DATA_DIR}/cppmega_commits_{stage.seq}_train"
        data_path_args = [str(code_weight), code_prefix, str(commits_weight), commits_prefix]
        data_path = f"{code_weight} {code_prefix} {commits_weight} {commits_prefix}"

        # Checkpoint: warm-start from previous stage (model weights only)
        checkpoint_args = ["--save", ckpt_dir, "--save-interval", str(save_interval)]
        if prev_ckpt_dir:
            checkpoint_args += [
                "--load", prev_ckpt_dir,
                "--no-load-optim", "--no-load-rng",
                "--finetune", "--override-opt-param-scheduler",
            ]
        elif os.path.isdir(ckpt_dir):
            checkpoint_args += ["--load", ckpt_dir]

        cmd = [
            _sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node=1",
            f"{workdir}/pretrain_mamba.py",
            "--data-path", *data_path_args,
            "--tokenizer-type", "HuggingFaceTokenizer",
            "--tokenizer-model", env.get("CPPMEGA_TOKENIZER_MODEL", TOKENIZER_DIR),
            "--vocab-size", "65536",
            "--make-vocab-size-divisible-by", "128",
            "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1",
            "--hybrid-layer-pattern", env.get("HYBRID_LAYER_PATTERN", "AF" * 24),
            "--num-layers", str(num_layers),
            "--hidden-size", env.get("CPPMEGA_HIDDEN_SIZE", "1280"),
            "--ffn-hidden-size", env.get("CPPMEGA_FFN_HIDDEN_SIZE", "3456"),
            "--num-attention-heads", env.get("CPPMEGA_NUM_ATTN_HEADS", "20"),
            "--group-query-attention",
            "--num-query-groups", env.get("CPPMEGA_NUM_QUERY_GROUPS", "4"),
            "--kv-channels", env.get("CPPMEGA_KV_CHANNELS", "64"),
            "--seq-length", str(stage.seq),
            "--max-position-embeddings", str(stage.seq),
            "--micro-batch-size", str(stage.micro_batch),
            "--global-batch-size", str(stage.batch),
            "--train-iters", str(train_iters),
            "--eval-interval", str(eval_interval),
            "--eval-iters", "1",
            "--optimizer", env.get("CPPMEGA_OPTIMIZER", "adam"),
            "--lr", env.get("CPPMEGA_LR", "3e-4"),
            "--min-lr", env.get("CPPMEGA_MIN_LR", "3e-5"),
            "--lr-decay-style", "constant",
            "--position-embedding-type", "rope",
            "--no-rope-fusion",
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--bf16",
            "--use-mcore-models",
            "--transformer-impl", "transformer_engine",
            "--attention-backend", "auto",
            # Nebius baseline used dropout=0.1 with cuDNN.  When FA4 score_mod
            # is active, --attention-dropout 0 is appended below (hard kernel
            # constraint).  Otherwise the profile default (0.1) applies.
            "--no-gradient-accumulation-fusion",
            "--no-persist-layer-norm",
            "--no-masked-softmax-fusion",
            "--cross-entropy-loss-fusion",
            "--cross-entropy-fusion-impl", "te",
            "--recompute-granularity", "selective",
            "--recompute-modules", "mlp",
            "--clip-grad", "1.0",
            "--rerun-mode", "disabled",
            "--log-interval", "1",
            # Spec + DSA native args (matches Nebius sweep)
            "--spec", spec_module, spec_function,
            *dsa_native_args,
            *checkpoint_args,
        ]

        # FA4 beta23 does not support attention dropout; setting to 0 is
        # required.  This is a hard constraint of the FA4 kernel, not a
        # feature toggle -- the score_mod path has no dropout support.
        if fa4_score_mod:
            cmd.extend(["--attention-dropout", "0"])

        print(f"\n{'='*70}")
        print(f"STAGE seq={stage.seq} bs={stage.batch} micro_bs={stage.micro_batch} iters={train_iters}")
        print(f"  Data: {data_path}")
        print(f"  Checkpoint: {ckpt_dir}")
        if prev_ckpt_dir:
            print(f"  Warm-start from: {prev_ckpt_dir} (model weights only)")
        print(f"{'='*70}\n", flush=True)

        start = time.time()
        proc = subprocess.Popen(
            cmd, cwd=workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.wait()
        elapsed = time.time() - start

        stage_results[f"seq_{stage.seq}"] = {
            "exit_code": proc.returncode,
            "elapsed_hours": elapsed / 3600,
            "train_iters": train_iters,
            "checkpoint_dir": ckpt_dir,
        }
        print(f"\n[seq={stage.seq}] exit={proc.returncode} elapsed={elapsed/3600:.2f}h", flush=True)

        if proc.returncode != 0:
            print(f"WARNING: stage seq={stage.seq} failed, NOT committing volume", flush=True)
            break

        checkpoint_vol.commit()
        prev_ckpt_dir = ckpt_dir

    return {
        "megatron_head": megatron_head,
        "stages": stage_results,
    }


@app.local_entrypoint()
def main(
    stages: str = "1024,2048,4096,8192",
    skip_convert: bool = False,
    code_weight: float = 0.7,
    commits_weight: float = 0.3,
    save_interval: int = 500,
    eval_interval: int = 500,
    fa4_score_mod: bool = False,
    beta23: bool = False,
) -> None:
    """Launch CppMega Megatron curriculum on Modal H200.

    Pass --beta23 to target the FA4 beta23 GHCR image (requires the beta23
    image to have been built and pushed; see docs/fa4_beta23_upgrade_plan.md).
    """
    seq_list = [int(s) for s in stages.split(",")]
    print("CppMega Megatron Curriculum Training")
    print(f"  GPU: {GPU_SPEC}")
    print(f"  Image: {GHCR_REF}")
    print(f"  Image variant: {'beta23' if (beta23 or USE_BETA23) else 'default (b19)'}")
    print(f"  Stages: {seq_list}")
    print(f"  Mix: code={code_weight}, commits={commits_weight}")

    if not skip_convert:
        print("\n--- Phase 1: Parquet → Megatron indexed conversion ---")
        conv_result = convert_all_stages.remote(seq_list=seq_list)
        print(f"Conversion: {json.dumps(conv_result, indent=2)}")
    else:
        print("\n--- Skipping conversion (--skip-convert) ---")

    print("\n--- Phase 2: Curriculum training ---")
    result = train_curriculum.remote(
        seq_list=seq_list,
        code_weight=code_weight,
        commits_weight=commits_weight,
        save_interval=save_interval,
        eval_interval=eval_interval,
        fa4_score_mod=fa4_score_mod,
    )
    print(f"\nCurriculum result: {json.dumps(result, indent=2)}")
