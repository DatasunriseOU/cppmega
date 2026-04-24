"""Modal app: NAM56R-quarter NaN reproduction sweep on H100:2.

Purpose
-------
bench3 H200 EP=8 was deleted before we could isolate the iter-1 `grad norm
NaN`. GB10 sm_121a (single-GPU) ran 22 finite configs: ALL cppmega commits,
FP8 × BF16, MBS 1-10, 13L and 52L, full feature stack. NaN did not
reproduce on single-GPU. Residual suspects: EP>1 DeepEP/flex A2A backward
OR H200/Hopper TE FP8 kernel specifics.

Modal H100:2 is the shortest path to EP=2 + DeepEP + TE FP8 on real Hopper
hardware. 20 configurations probe the search space:

  1-3:  baselines (BF16 EP=1, BF16 EP=2, FP8 EP=1)
  4-6:  FP8 × EP=2 × MBS ladder (4/8/10, alltoall)
  7-9:  FP8 × flex dispatcher (DeepEP) × EP=2 × MBS ladder  <- prime suspect
 10-11: 52L scale-up (matches bench3 depth exactly, FP8 EP=2 flex)
 12-16: Feature flip at worst-case config (minus SparseMLA, minus
        dsa_indexer_fused, minus Liger, minus structure, minus IndexCache)
 17:    FP8 blockwise recipe (not tensorwise)
 18:    FP8 current_scaling recipe
 19:    60-iter warmup run (amax history saturation test)
 20:    TP=2 instead of EP=2 (distributed-in-general vs EP-specific)

Each config runs as a subprocess inside one H100:2 container, per-config
timeout 600 s, on timeout/crash we capture the log and move on. Logs +
metrics go to `modal.Volume("cppmega-nan-sweep")`; central results.jsonl
at /vol/logs/<run_id>/results.jsonl.

Usage
-----
    # One-time: build image (heavy, ~30 min GPU time for source builds)
    modal run --detach scripts/modal_nan_sweep_h100.py::build_image

    # Run sweep (sequential, ~2 h wall, single H100:2 container)
    modal run --detach scripts/modal_nan_sweep_h100.py::sweep

    # Tail logs while running
    modal app logs cppmega-nan-sweep -f

    # After completion, download artifacts
    modal volume get cppmega-nan-sweep /logs ./modal_nan_sweep_logs

Cost estimate: H100:2 = ~$7.90/hr on Modal. First build ~30 min ($4). Sweep
~2 h ($16). Total < $25 for 20 data points.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

import modal

# Base image: heavy stack (torch 2.12 cu132 / TE 2.13 / mamba_ssm / TileLang
# dev / flash-attn / Megatron dev). Inlined for Modal packaging; identical
# definition to scripts/modal_cppmega_base.py — same content hash → layers
# share Modal's cache across both apps.
_REPO_ROOT = pathlib.Path(__file__).parent.parent
CUDA_BASE = "nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04"
PYTHON_VERSION = "3.13"
TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"
MEGATRON_COMMIT = "980211ae"
_WHEELS_DIR = "/tmp/cppmega_wheels"
_WHEEL_FILES = [
    "transformer_engine_torch-2.13.0-cp313-cp313-linux_x86_64.whl",
    "mamba_ssm-2.3.1+pristine-cp313-cp313-linux_x86_64.whl",
    "causal_conv1d-1.6.1-cp313-cp313-linux_x86_64.whl",
    "flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl",
    "qoptim_cuda-0.0.0-cp313-cp313-linux_x86_64.whl",
    "tilelang-0.1.8+cuda.gitf309d814-cp38-abi3-linux_x86_64.whl",
    "fast_hadamard_transform-1.1.0-cp313-cp313-linux_x86_64.whl",
]


def cppmega_base_image() -> modal.Image:
    """MUST be byte-identical (same arg strings, same order) to
    scripts/modal_cppmega_base.py::cppmega_base_image so Modal's content-
    hashed layer cache hits across both apps."""
    from typing import Any
    base: Any = modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
    img = (
        base.apt_install(
            "git", "build-essential", "curl", "ca-certificates",
            "pkg-config", "libnuma-dev",
        )
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/sbin",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.13/site-packages/z3/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "PYTHONPATH": "/opt/megatron-lm",
        })
        .pip_install(
            "torch==2.12.*",
            "numpy>=1.26", "packaging", "wheel", "setuptools", "ninja",
            "einops", "pybind11", "pyyaml", "regex",
            "sentencepiece", "tiktoken", "six", "scipy",
            extra_index_url=TORCH_NIGHTLY_INDEX,
            pre=True,
        )
        .pip_install(
            "apache-tvm-ffi==0.1.9",
            "transformer-engine-cu13==2.13.0",
            "transformer-engine==2.13.0",
            "nvidia-nccl-cu13",
            "onnxscript", "onnx",
            "pydantic", "nvdlfw-inspect",
            "z3-solver==4.15.*",
            "cloudpickle", "psutil", "pynvml", "typing-extensions",
            "huggingface_hub", "transformers", "tokenizers",
            "datasets", "accelerate", "tensorboard",
            "wandb", "tqdm", "pytest", "filelock",
            "liger-kernel",
        )
    )
    wheels_path = pathlib.Path(_WHEELS_DIR)
    for whl in _WHEEL_FILES:
        p = wheels_path / whl
        if p.exists():
            img = img.add_local_file(str(p), f"/wheels/{whl}", copy=True)
    img = img.run_commands(
        "Z3LIB=/usr/local/lib/python3.13/site-packages/z3/lib && "
        "if [ ! -f $Z3LIB/libz3.so.4.15 ]; then "
        "  ln -sf $Z3LIB/libz3.so $Z3LIB/libz3.so.4.15; fi && ls $Z3LIB/libz3*",
    ).run_commands(
        "pip install --no-deps /wheels/*.whl && "
        "python -c 'import transformer_engine.pytorch as te; print(\"TE Linear ok:\", te.Linear)' && "
        "python -c 'import mamba_ssm, flash_attn, tilelang; "
        "print(\"mamba_ssm\", mamba_ssm.__version__, "
        "\"flash_attn\", flash_attn.__version__, "
        "\"tilelang\", tilelang.__version__)' && "
        "python -c 'from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo; print(\"mamba3_mimo ok\")'",
    ).run_commands(
        f"cd /opt && git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm && "
        f"cd megatron-lm && git checkout -q {MEGATRON_COMMIT} && "
        "pip install --no-deps -e .",
        "python -c 'import megatron.core; print(\"megatron-core\", megatron.core.__version__)'",
    )
    return img

app = modal.App("cppmega-nan-sweep")
logs_vol = modal.Volume.from_name("cppmega-nan-sweep", create_if_missing=True)
cache_vol = modal.Volume.from_name("cppmega-nan-sweep-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _image() -> modal.Image:
    """Thin delta on top of the shared `cppmega-base` image.

    The base (torch nightly + TileLang + mamba_ssm + TE + flash-attn +
    Megatron + training deps) lives in `modal_cppmega_base.py` and is
    built once per account. Here we only add the sweep-specific bits:
    cppmega local source (mutable, reloads on every run) + PYTHONPATH.
    """
    return (
        cppmega_base_image()
        .env({"PYTHONPATH": "/opt/cppmega:/opt/megatron-lm"})
        .add_local_dir(
            str(_REPO_ROOT / "cppmega"),
            remote_path="/opt/cppmega/cppmega",
        )
        .add_local_file(
            str(_REPO_ROOT / "pyproject.toml"),
            remote_path="/opt/cppmega/pyproject.toml",
        )
    )


image = _image()


# ---------------------------------------------------------------------------
# Config matrix (20 variants)
# ---------------------------------------------------------------------------

@dataclass
class SweepCfg:
    name: str
    description: str
    # Environment overrides on top of BASE_ENV.
    env: dict[str, str] = field(default_factory=dict)
    # Megatron CLI overrides: {"flag": "value"} — adds/replaces --flag value.
    cli_override: dict[str, str] = field(default_factory=dict)
    # Per-rank MBS.
    mbs: int = 8
    # GBS = mbs * dp (dp = nproc // ep).
    gbs: int | None = None
    # nproc_per_node.
    nproc: int = 2
    ep: int = 1
    tp: int = 1
    pp: int = 1
    # Layer depth (13L = quarter, 52L = full).
    depth: int = 13
    # --fp8-format (empty = BF16).
    fp8_format: str = ""
    fp8_recipe: str = "tensorwise"
    # Dispatcher: alltoall | flex.
    dispatcher: str = "alltoall"
    # Iterations.
    iters: int = 10
    # Per-config timeout seconds.
    timeout_s: int = 900


# Feature stack knobs (match bench3 remote_smoke_h200_dsa_9_4_m.sh).
_FULL_FEATURES = {
    # Skip Liger CE by default — requires megatron.core.transformer.linear_cross_entropy
    # which may not be in the pinned commit on some checkouts. Flip to 0 once verified.
    "CPPMEGA_SKIP_LIGER_CE": "1",
    "CPPMEGA_MAMBA3_MIMO": "1",
    "CPPMEGA_MAMBA_NUM_GROUPS": "8",
    "CPPMEGA_NGRAM_HASH_ENABLED": "1",
    "CPPMEGA_NGRAM_HASH_ORDERS": "2,3",
    "CPPMEGA_NGRAM_HASH_HEADS": "8",
    "CPPMEGA_NGRAM_HASH_TABLE_SIZE": "500000",
    "CPPMEGA_NGRAM_HASH_EMBED_DIM": "16",
    "CPPMEGA_STRUCTURE_ENABLED": "1",
    "CPPMEGA_STRUCTURE_COMPONENTS": "core",
    "CPPMEGA_DSA_SPARSE_MODE": "tilelang",
    "CPPMEGA_MAMBA_RECOMPUTE": "1",
    "CPPMEGA_DSA_SKIP_INDEXER_LOSS": "1",
    "CPPMEGA_DSA_INDEXER_LOSS_COEFF": "0",
    "CPPMEGA_INDEX_CACHE": "1",
    "CPPMEGA_LEMYX_DSA": "0",  # LEMYX repo not in Modal image
    "CPPMEGA_MAMBA3_P1": "0",
}


def _build_configs() -> list[SweepCfg]:
    """20-variant matrix. Ordering: baselines → suspect axis → features flip."""
    c: list[SweepCfg] = []

    # 1-3 baselines
    c.append(SweepCfg("01_bf16_ep1_mbs8", "BF16 EP=1 MBS=8 alltoall 13L", mbs=8, ep=1, nproc=2, dispatcher="alltoall"))
    c.append(SweepCfg("02_bf16_ep2_mbs4", "BF16 EP=2 MBS=4 alltoall 13L — first real EP>1", mbs=4, ep=2, nproc=2, dispatcher="alltoall"))
    c.append(SweepCfg("03_fp8_ep1_mbs8", "FP8tw EP=1 MBS=8 alltoall 13L — FP8 baseline", mbs=8, ep=1, nproc=2, dispatcher="alltoall", fp8_format="hybrid"))

    # 4-6 FP8 × EP=2 × MBS ladder
    c.append(SweepCfg("04_fp8_ep2_mbs4", "FP8tw EP=2 MBS=4 alltoall 13L", mbs=4, ep=2, nproc=2, dispatcher="alltoall", fp8_format="hybrid"))
    c.append(SweepCfg("05_fp8_ep2_mbs8", "FP8tw EP=2 MBS=8 alltoall 13L", mbs=8, ep=2, nproc=2, dispatcher="alltoall", fp8_format="hybrid"))
    c.append(SweepCfg("06_fp8_ep2_mbs10", "FP8tw EP=2 MBS=10 alltoall 13L — bench3 MBS", mbs=10, ep=2, nproc=2, dispatcher="alltoall", fp8_format="hybrid"))

    # 7-9 FP8 × flex dispatcher (prime suspect — DeepEP A2A)
    c.append(SweepCfg("07_bf16_ep2_mbs4_flex", "BF16 EP=2 MBS=4 FLEX 13L", mbs=4, ep=2, nproc=2, dispatcher="flex"))
    c.append(SweepCfg("08_fp8_ep2_mbs4_flex", "FP8tw EP=2 MBS=4 FLEX 13L", mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid"))
    c.append(SweepCfg("09_fp8_ep2_mbs10_flex", "FP8tw EP=2 MBS=10 FLEX 13L — CLOSEST TO BENCH3", mbs=10, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid"))

    # 10-11 52L scale-up
    c.append(SweepCfg("10_fp8_ep2_mbs4_flex_52L", "FP8tw EP=2 MBS=4 FLEX 52L — bench3 depth", mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid", depth=52, timeout_s=1800))
    c.append(SweepCfg("11_fp8_ep2_mbs8_flex_52L", "FP8tw EP=2 MBS=8 FLEX 52L — heavier scale", mbs=8, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid", depth=52, timeout_s=1800))

    # 12-16 feature flip at 13L FP8 EP=2 flex MBS=4 (worst-case short run)
    def _worst_case(**env_overrides: str) -> dict[str, str]:
        return {**_FULL_FEATURES, **env_overrides}

    c.append(SweepCfg("12_flip_minus_sparsemla", "Worst-case MINUS SparseMLA (native DSA)",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid",
                      env=_worst_case(CPPMEGA_DSA_SPARSE_MODE="native")))
    c.append(SweepCfg("13_flip_minus_dsa_fused", "Worst-case MINUS dsa_indexer_fused patch",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid",
                      env=_worst_case(CPPMEGA_SKIP_DSA_INDEXER_FUSED="1")))
    c.append(SweepCfg("14_flip_minus_liger_ce", "Worst-case MINUS Liger CE (MTP+main)",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid",
                      env=_worst_case(CPPMEGA_SKIP_LIGER_CE="1")))
    c.append(SweepCfg("15_flip_minus_struct_ngram", "Worst-case MINUS structure+ngram_hash",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid",
                      env=_worst_case(CPPMEGA_NGRAM_HASH_ENABLED="0", CPPMEGA_STRUCTURE_ENABLED="0")))
    c.append(SweepCfg("16_flip_minus_indexcache", "Worst-case MINUS IndexCache",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid",
                      env=_worst_case(CPPMEGA_INDEX_CACHE="0")))

    # 17-19 FP8 recipe + long run
    c.append(SweepCfg("17_fp8_delayed", "FP8 delayed recipe EP=2 flex MBS=4", mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid", fp8_recipe="delayed"))
    c.append(SweepCfg("18_fp8_param_gather", "FP8 delayed + --fp8-param-gather EP=2 flex MBS=4",
                      mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid", fp8_recipe="delayed",
                      env={"CPPMEGA_FP8_PARAM_GATHER": "1"}))
    c.append(SweepCfg("19_fp8_60iters", "FP8tw EP=2 flex MBS=4 60 iters — amax warmup test", mbs=4, ep=2, nproc=2, dispatcher="flex", fp8_format="hybrid", iters=60, timeout_s=1800))

    # 20 TP=2 instead of EP=2
    c.append(SweepCfg("20_fp8_tp2", "FP8tw TP=2 EP=1 MBS=4 alltoall — distributed-in-general vs EP-specific",
                      mbs=4, ep=1, tp=2, nproc=2, dispatcher="alltoall", fp8_format="hybrid"))

    return c


# ---------------------------------------------------------------------------
# Megatron command builder (runs inside container)
# ---------------------------------------------------------------------------

_NAM56R_PATTERN = "AEMEAEMEAEMR"  # same 12-char seed for 13L and 52L; depth tiles it


_SHIM = r"""
# Runtime shim: apply all cppmega monkey-patches before Megatron initializes.
import os, sys
print(f"[shim] LOCAL_RANK={os.environ.get('LOCAL_RANK','?')} "
      f"RANK={os.environ.get('RANK','?')} WORLD={os.environ.get('WORLD_SIZE','?')}",
      file=sys.stderr)

if os.environ.get("CPPMEGA_SKIP_LIGER_CE", "0") != "1":
    from cppmega.megatron.mtp_liger_ce import patch_mtp_loss_with_liger
    patch_mtp_loss_with_liger()
    from cppmega.megatron.apply_linear_ce_patch import patch_mamba_output_layer_with_linear_ce
    patch_mamba_output_layer_with_linear_ce()
    print("[shim] Liger CE installed", file=sys.stderr)

if os.environ.get("CPPMEGA_SKIP_DSA_INDEXER_FUSED", "0") != "1":
    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    apply_dsa_indexer_fused_patch()
    print("[shim] DSA indexer fused installed", file=sys.stderr)

_sparse_mode = os.environ.get("CPPMEGA_DSA_SPARSE_MODE", "tilelang").strip().lower()
if _sparse_mode in ("tilelang", "gather_scatter", "pytorch"):
    from megatron.core.transformer.experimental_attention_variant import dsa as _dsa_mod
    if _sparse_mode == "tilelang":
        from cppmega.megatron.sparse_mla_ops.sparse_mla import sparse_mla_as_unfused_dsa as _fn
        setattr(_fn, "__cppmega_sparse_dsa_patched__", True)
        _dsa_mod.unfused_dsa_fn = _fn
        print("[shim] TileLang SparseMLA applied", file=sys.stderr)
    elif _sparse_mode != "native":
        from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn as _fn
        setattr(_fn, "__cppmega_sparse_dsa_patched__", True)
        _dsa_mod.unfused_dsa_fn = _fn
        print("[shim] gather_scatter sparse_dsa_fn applied", file=sys.stderr)
print("[shim] ready", file=sys.stderr)
"""


def _write_workdir(workdir: pathlib.Path, megatron_root: str) -> None:
    """Write pretrain_mamba.py + shim + builders into workdir."""
    (workdir / "cppmega_shim.py").write_text(_SHIM)
    src = pathlib.Path(megatron_root) / "pretrain_mamba.py"
    dst = workdir / "pretrain_mamba.py"
    dst.write_text("import cppmega_shim  # NaN-sweep\n" + src.read_text())
    (workdir / "mamba_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder\n"
    )
    (workdir / "model_provider.py").write_text(
        "from megatron.training import get_args\n"
        "def model_provider(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):\n"
        "    args = get_args()\n"
        "    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)\n"
    )


def _build_cmd(cfg: SweepCfg, workdir: pathlib.Path, data_path: str, tokenizer: str) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) for this config."""
    env = {**os.environ}
    # Base feature stack.
    env.update(_FULL_FEATURES)
    # Per-depth pattern.
    env["CPPMEGA_NEM_PATTERN"] = _NAM56R_PATTERN
    env["CPPMEGA_LAYER_DEPTH"] = str(cfg.depth)
    if cfg.depth == 13:
        env["CPPMEGA_DSA_A_LAYER_RANKS"] = "1,2,3"
    else:
        env["CPPMEGA_DSA_A_LAYER_RANKS"] = "1,2,3,5,6,7,9,10,11"
    # Per-cfg overrides win.
    env.update(cfg.env)
    # Always on.
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Megatron: DP = nproc / (TP*PP*CP). EP does NOT reduce DP (EP is
    # orthogonal to DP, not a divider). GBS must be divisible by MBS*DP.
    dp = max(cfg.nproc // max(cfg.tp * cfg.pp, 1), 1)
    gbs = cfg.gbs or (cfg.mbs * dp)
    gbs = max(gbs, cfg.mbs * dp)

    # Build hybrid-layer-pattern + native args via helper python (same as GB10).
    helper = workdir / "_helper.py"
    helper.write_text(
        "import os\n"
        "from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern\n"
        "from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args\n"
        "from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan\n"
        "plan = build_nam56r_feature_plan(\n"
        "    pattern=os.environ['CPPMEGA_NEM_PATTERN'],\n"
        "    depth=int(os.environ['CPPMEGA_LAYER_DEPTH']),\n"
        "    mtp_depths=1,\n"
        ")\n"
        "bundle = build_nam56r_megatron_native_args(\n"
        "    plan=plan, enable_mla=True, enable_mtp=True,\n"
        "    mtp_mode='hybrid', enable_moe=True,\n"
        ")\n"
        "pattern = build_default_hybrid_layer_pattern(mtp_depths=1)\n"
        "print('PATTERN', pattern)\n"
        "print('NATIVE_ARGS', bundle.to_shell_fragment())\n"
    )
    out = subprocess.check_output([sys.executable, str(helper)], env=env, text=True)
    pattern = ""
    native_args = ""
    for line in out.splitlines():
        if line.startswith("PATTERN "):
            pattern = line[len("PATTERN "):]
        elif line.startswith("NATIVE_ARGS "):
            native_args = line[len("NATIVE_ARGS "):]

    argv = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={cfg.nproc}",
        "--master_port=29500",
        str(workdir / "pretrain_mamba.py"),
        "--data-path", "1.0", data_path,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer,
        "--vocab-size", "65536",
        "--make-vocab-size-divisible-by", "128",
        "--tensor-model-parallel-size", str(cfg.tp),
        "--pipeline-model-parallel-size", str(cfg.pp),
        "--context-parallel-size", "1",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--hybrid-layer-pattern", pattern,
        "--hidden-size", "2048",
        "--ffn-hidden-size", "5632",
        "--num-attention-heads", "16",
        "--seq-length", "2048",
        "--max-position-embeddings", "2048",
        "--micro-batch-size", str(cfg.mbs),
        "--global-batch-size", str(gbs),
        "--train-iters", str(cfg.iters),
        "--eval-interval", "50000000",
        "--eval-iters", "1",
        "--lr", "1e-4",
        "--min-lr", "1e-5",
        "--lr-decay-style", "constant",
        "--position-embedding-type", "rope",
        "--normalization", "RMSNorm",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--use-flash-attn",
        "--attention-backend", "flash",
        "--spec", "cppmega.megatron.nam56r_noconv_spec", "build_cppmega_nam56r_noconv_stack_spec",
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl", "te",
        "--recompute-granularity", "selective",
        "--recompute-modules", "moe_act", "mlp", "mla_up_proj",
        "--mla-down-proj-fusion",
        "--clip-grad", "1.0",
        "--save-interval", "50000000",
        "--log-interval", "1",
    ]
    if cfg.fp8_format:
        argv += [
            "--fp8-format", cfg.fp8_format,
            "--fp8-recipe", cfg.fp8_recipe,
            "--fp8-amax-history-len", "1024",
            "--fp8-amax-compute-algo", "max",
        ]
        if env.get("CPPMEGA_FP8_PARAM_GATHER") == "1":
            argv += ["--fp8-param-gather"]
    # Merge NATIVE_ARGS from helper (MoE / MTP / MLA bundle) FIRST,
    # then our per-config overrides so argparse's last-occurrence-wins
    # lets us force dispatcher/EP for each variant.
    if native_args:
        argv += native_args.split()
    # Per-config overrides: these MUST come after NATIVE_ARGS to win.
    argv += ["--moe-token-dispatcher-type", cfg.dispatcher]
    argv += ["--expert-model-parallel-size", str(cfg.ep)]
    return argv, env


# ---------------------------------------------------------------------------
# Sweep function (runs on H100:2)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="H100:2",
    cpu=16,
    memory=128 * 1024,
    volumes={"/vol": logs_vol, "/cache": cache_vol},
    timeout=4 * 3600,  # 4h total budget
    retries=0,
)
def sweep(run_id: str = "") -> dict:
    import datetime
    run_id = run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = pathlib.Path(f"/vol/logs/{run_id}")
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"

    # Data + tokenizer — synthesize on Volume if absent. Mock data is fine
    # for iter-1 NaN investigation: any input triggers the suspect path.
    data_prefix = "/vol/mock_data"
    data_path = data_prefix + "_text_document"  # Megatron preprocess_data suffix
    tokenizer = "/vol/mock_tokenizer"
    if not pathlib.Path(data_path + ".idx").exists():
        print("[sweep] synthesizing mock Megatron dataset...", flush=True)
        _synth_mock_data(data_prefix, tokenizer)
        logs_vol.commit()

    configs = _build_configs()
    summary = {"run_id": run_id, "total": len(configs), "finite": 0, "nan": 0, "crash": 0, "timeout": 0}

    for idx, cfg in enumerate(configs):
        cfg_dir = out_root / cfg.name
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))

        workdir = pathlib.Path(f"/tmp/sweep_workdir_{cfg.name}")
        workdir.mkdir(parents=True, exist_ok=True)
        _write_workdir(workdir, "/opt/megatron-lm")

        try:
            argv, env = _build_cmd(cfg, workdir, data_path, tokenizer)
        except Exception as e:
            result = {"cfg": cfg.name, "status": "setup_fail", "error": str(e)}
            results_path.open("a").write(json.dumps(result) + "\n")
            summary["crash"] += 1
            print(f"[{idx+1}/{len(configs)}] {cfg.name}: SETUP FAIL {e}", flush=True)
            continue

        log_file = cfg_dir / "stdout.log"
        print(f"\n=== [{idx+1}/{len(configs)}] {cfg.name}: {cfg.description} ===", flush=True)
        print(f"    timeout={cfg.timeout_s}s log={log_file}", flush=True)

        t0 = time.time()
        status, last_grad, nan_iter = _run_with_timeout(argv, env, log_file, cfg.timeout_s)
        elapsed = time.time() - t0

        result = {
            "cfg": cfg.name,
            "desc": cfg.description,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "last_grad_norm": last_grad,
            "nan_iter": nan_iter,
            "mbs": cfg.mbs, "ep": cfg.ep, "tp": cfg.tp, "depth": cfg.depth,
            "fp8_format": cfg.fp8_format, "fp8_recipe": cfg.fp8_recipe,
            "dispatcher": cfg.dispatcher, "iters": cfg.iters,
        }
        (cfg_dir / "result.json").write_text(json.dumps(result, indent=2))
        results_path.open("a").write(json.dumps(result) + "\n")
        logs_vol.commit()

        summary[{"finite": "finite", "nan": "nan", "timeout": "timeout"}.get(status, "crash")] += 1
        print(f"    -> {status} ({elapsed:.1f}s) last_grad={last_grad} nan_iter={nan_iter}", flush=True)

    # Write summary.
    summary_md = out_root / "summary.md"
    with summary_md.open("w") as f:
        f.write(f"# NaN sweep {run_id}\n\n")
        f.write(f"Total: {summary['total']}  finite: {summary['finite']}  "
                f"NaN: {summary['nan']}  timeout: {summary['timeout']}  crash: {summary['crash']}\n\n")
        f.write("| # | cfg | status | elapsed | last_grad | nan_iter |\n|---|---|---|---|---|---|\n")
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            f.write(f"| | {r['cfg']} | {r['status']} | {r.get('elapsed_s','')} | {r.get('last_grad_norm','')} | {r.get('nan_iter','')} |\n")
    logs_vol.commit()
    print(f"\n=== SWEEP DONE: {summary} ===", flush=True)
    return summary


def _synth_mock_data(data_path: str, tokenizer_dir: str) -> None:
    """Create minimal Megatron dataset + HF tokenizer for sweep input."""
    import shutil
    # Minimal HF tokenizer — use a small existing model that fits.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    pathlib.Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tokenizer_dir)

    # Build 100k-token mock dataset via Megatron's tools/preprocess_data.py.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        import random
        random.seed(0)
        vocab = tok.get_vocab()
        tokens = list(vocab.keys())[:1024]
        for _ in range(2000):
            line = " ".join(random.choices(tokens, k=256))
            f.write(json.dumps({"text": line}) + "\n")
        raw = f.name
    subprocess.run([
        sys.executable, "/opt/megatron-lm/tools/preprocess_data.py",
        "--input", raw,
        "--output-prefix", data_path,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        "--workers", "2",
        "--append-eod",
    ], check=True)


def _run_with_timeout(argv: list[str], env: dict[str, str], log_file: pathlib.Path,
                      timeout_s: int) -> tuple[str, float | str | None, int | None]:
    """Run subprocess, stream stdout to log_file and parent stdout.

    Returns (status, last_grad_norm, nan_iter) where status is one of
    finite | nan | timeout | crash.
    """
    import re
    import signal
    grad_re = re.compile(r"grad norm:\s*([\d.eE+\-]+|nan|NaN|inf)")
    iter_re = re.compile(r"iteration\s+(\d+)/")
    last_grad: float | str | None = None
    nan_iter: int | None = None
    cur_iter: int | None = None
    status = "crash"

    with log_file.open("wb") as lf:
        proc = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout
        t0 = time.time()
        try:
            for line in iter(proc.stdout.readline, b""):
                lf.write(line); lf.flush()
                sys.stdout.buffer.write(line); sys.stdout.flush()
                s = line.decode("utf-8", errors="replace")
                m = iter_re.search(s)
                if m:
                    cur_iter = int(m.group(1))
                mg = grad_re.search(s)
                if mg:
                    g = mg.group(1)
                    if g.lower() in ("nan", "inf"):
                        nan_iter = cur_iter
                        last_grad = g
                    else:
                        try:
                            last_grad = float(g)
                        except ValueError:
                            last_grad = g
                if time.time() - t0 > timeout_s:
                    raise TimeoutError()
            rc = proc.wait(timeout=5)
            if rc == 0:
                status = "nan" if nan_iter is not None else "finite"
            else:
                status = "nan" if nan_iter is not None else "crash"
        except TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(5)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            status = "timeout"
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            status = "crash"
    return status, last_grad, nan_iter


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=3600)
def build_image() -> str:
    """Force image build by running a trivial check — layers cache across calls."""
    import torch
    import transformer_engine.pytorch as te  # noqa
    import megatron.core  # noqa
    return f"torch={torch.__version__} cuda={torch.version.cuda} sm_90_ok"


@app.local_entrypoint()
def launch(run_id: str = ""):
    """Fire-and-forget launch: spawn sweep, print FunctionCall id, exit.

        modal deploy scripts/modal_nan_sweep_h100.py      # register functions
        modal run scripts/modal_nan_sweep_h100.py::launch # spawn → returns now

    After launch, Modal keeps the sweep running server-side. Local shell
    can exit freely. Monitor via:
        modal app logs cppmega-nan-sweep -f
        modal volume ls cppmega-nan-sweep /logs
    """
    call = sweep.spawn(run_id)
    print(f"spawned: function_call_id={call.object_id}")
    print(f"logs:    modal app logs cppmega-nan-sweep -f")
    print(f"result:  modal call-graph {call.object_id}   (after completion)")


@app.local_entrypoint()
def main(run_id: str = ""):
    """Blocking run (stays tied to local shell). Use `launch` for detached."""
    summary = sweep.remote(run_id)
    print(json.dumps(summary, indent=2))
