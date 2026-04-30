"""Modal H200:2 full-boundary gate for Mamba3 stage2 force-nonTMA.

This app compares the installed baseline against the guarded
stage2_force_nontma_bf1_bb0 candidate on the NAM56R full training path. It
uses the GHCR prebuilt cppmega image, overlays the current checkout's cppmega
and upstream_prs trees, and writes logs/results to a Modal volume.

Real data is used only if it is already present on the volume. Otherwise the
app creates an indexed Megatron dataset under /vol/mock_data and labels the
run as synthetic_full_shape_mock_data in every result artifact.
"""
# ruff: noqa: E402

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).parent.parent

APP_NAME = "cppmega-wave29-modal-h200-profiler"
RESULTS_VOL = "cppmega-mamba3-benchmarks"
BENCH_DIR = "/benchmarks/mamba3_wave29_modal_h200_profiler"
GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = "H200:2"


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    return (
        img.env(
            {
                "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
                "TILELANG_EXECUTION_BACKEND": "cython",
            }
        )
        .add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega")
        .add_local_dir(str(_REPO_ROOT / "upstream_prs"), remote_path="/opt/cppmega/upstream_prs")
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )


app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
cache_vol = modal.Volume.from_name("cppmega-modal-cache", create_if_missing=True)
image = _image()


_SHIM = r'''
"""Runtime shim for the Modal H200 NAM56R full-boundary gate."""
from __future__ import annotations
import atexit
import os
import sys

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
except Exception as _exc:
    print(f"[wave29_shim] static_context alias skipped: {_exc}", file=sys.stderr)

if os.environ.get("CPPMEGA_MAMBA3_MIMO", "0") == "1":
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
        _orig_post_init = TransformerConfig.__post_init__
        def _cppmega_mimo_post_init(self):
            _orig_post_init(self)
            if not getattr(self, "cppmega_mamba3_is_mimo", False):
                object.__setattr__(self, "cppmega_mamba3_is_mimo", True)
            if not getattr(self, "cppmega_mamba3_mimo_rank", None):
                object.__setattr__(self, "cppmega_mamba3_mimo_rank", 4)
            if not getattr(self, "cppmega_mamba3_chunk_size", None):
                object.__setattr__(self, "cppmega_mamba3_chunk_size", 16)
        TransformerConfig.__post_init__ = _cppmega_mimo_post_init
        print("[wave29_shim] MIMO config hook installed")
    except Exception as _exc:
        print(f"[wave29_shim] MIMO config hook failed: {_exc}", file=sys.stderr)

try:
    from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
    import torch as _torch
    _FP32_NAMES = ("B_bias", "C_bias", "D", "dt_bias", "mimo_x", "mimo_z", "mimo_o")
    def _restore_bias_fp32(module, _inputs):
        for _name in _FP32_NAMES:
            _p = getattr(module, _name, None)
            if _p is not None and _p.dtype != _torch.float32:
                _p.data = _p.data.float()
    if not getattr(_Mamba3, "_cppmega_fp32_bias_hook", False):
        _Mamba3._cppmega_fp32_bias_hook = True
        _orig_init = _Mamba3.__init__
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.register_forward_pre_hook(_restore_bias_fp32)
        _Mamba3.__init__ = _patched_init
        print("[wave29_shim] Mamba3 fp32-bias hook installed")
except Exception as _exc:
    print(f"[wave29_shim] Mamba3 fp32-bias hook failed: {_exc}", file=sys.stderr)

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

print(f"[wave29_shim] DSA indexer dtype={os.environ.get('CPPMEGA_DSA_INDEXER_DTYPE', 'bf16')}")
print(f"[wave29_shim] DSA sparse mode={os.environ.get('CPPMEGA_DSA_SPARSE_MODE', 'tilelang')}")
print(f"[wave29_shim] DSA KL mode={os.environ.get('CPPMEGA_DSA_KL_MODE', 'head_streaming')}")

def _peak_mem_report():
    try:
        import torch
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        rank = int(os.environ.get("RANK", "0"))
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
        print(
            f"[production_peak_mem] rank={rank} device={dev} "
            f"peak_alloc_gib={peak_alloc:.3f} peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    except Exception as _exc:
        print(f"[production_peak_mem] report failed: {_exc}", file=sys.stderr)

atexit.register(_peak_mem_report)
'''


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "TILELANG_EXECUTION_BACKEND": "cython",
            "CPPMEGA_MAMBA3_MIMO": "1",
            "CPPMEGA_MAMBA_NUM_GROUPS": "8",
            "CPPMEGA_NEM_PATTERN": "AEMEAEMEAEMR",
            "CPPMEGA_LAYER_DEPTH": "52",
            "CPPMEGA_R_LAYER_INDICES": "12,24,36,48",
            "CPPMEGA_NGRAM_HASH_ENABLED": "1",
            "CPPMEGA_NGRAM_HASH_ORDERS": "2,3",
            "CPPMEGA_NGRAM_HASH_HEADS": "8",
            "CPPMEGA_NGRAM_HASH_TABLE_SIZE": "500000",
            "CPPMEGA_NGRAM_HASH_EMBED_DIM": "16",
            "CPPMEGA_STRUCTURE_ENABLED": "1",
            "CPPMEGA_STRUCTURE_COMPONENTS": "core",
            "CPPMEGA_DSA_A_LAYER_RANKS": "1,2,3,5,6,7,9,10,11",
            "CPPMEGA_DSA_INDEXER_DTYPE": "bf16",
            "CPPMEGA_DSA_SPARSE_MODE": "tilelang",
            "CPPMEGA_DSA_KL_MODE": "head_streaming",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_GRAPH_REGISTER": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_IB_SL": "1",
            "NCCL_NET_PLUGIN": "none",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "TRITON_CACHE_DIR": "/cache/triton",
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface",
        }
    )
    for bad in ("NCCL_NET",):
        env.pop(bad, None)
    return env


def _run_capture(
    argv: list[str],
    env: dict[str, str],
    cwd: pathlib.Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=False)


def _run_streaming(
    argv: list[str],
    env: dict[str, str],
    log_file: pathlib.Path,
    timeout_s: int,
    cwd: pathlib.Path | None = None,
) -> dict[str, Any]:
    status = "crash"
    t0 = time.time()
    timed_out = False
    with log_file.open("wb") as lf:
        proc = subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            while True:
                line = proc.stdout.readline()
                if line:
                    lf.write(line)
                    lf.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()
                elif proc.poll() is not None:
                    break
                if time.time() - t0 > timeout_s:
                    timed_out = True
                    raise TimeoutError()
            rc = proc.wait(timeout=10)
            status = "ok" if rc == 0 else "crash"
        except TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(5)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait(timeout=10)
            status = "timeout"
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait(timeout=10)
            status = "crash"
    return {
        "status": status,
        "returncode": rc,
        "elapsed_s": round(time.time() - t0, 3),
        "timed_out": timed_out,
        "log": str(log_file),
    }


def _write_workdir(workdir: pathlib.Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "cppmega_mimo_shim.py").write_text(_SHIM)
    (workdir / "pretrain_mamba.py").write_text(
        textwrap.dedent(
            """
            import os
            import runpy
            import sys

            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in sys.path:
                sys.path.insert(0, _here)
            import cppmega_mimo_shim  # noqa: F401

            runpy.run_path(
                os.path.join("/opt/megatron-lm", "pretrain_mamba.py"),
                run_name="__main__",
            )
            """
        ).lstrip()
    )
    (workdir / "mamba_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder\n"
    )
    (workdir / "model_provider.py").write_text(
        "from megatron.training import get_args\n"
        "def model_provider(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):\n"
        "    args = get_args()\n"
        "    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)\n"
    )


def _kernel_status(env: dict[str, str]) -> dict[str, Any]:
    code = r"""
import importlib.util
import json
import pathlib

spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
path = pathlib.Path(next(iter(spec.submodule_search_locations))) / "mamba3_mimo_bwd.py"
text = path.read_text()
print(json.dumps({
    "path": str(path),
    "flat_q": "Q: T.Tensor([B, S * R, G, N], dtype)" in text,
    "flat_qk": "QK_DOT: T.Tensor([B, H, S, R * R], dtype)" in text,
    "bf_num_stages_1": "bf_num_stages=1" in text,
    "bb_num_stages_0": "bb_num_stages=0" in text,
    "bb_num_stages_1": "bb_num_stages=1" in text,
    "disable_tma_count": text.count("disable_tma=True"),
    "alloc_shared_count": text.count("T.alloc_shared"),
    "copy_count": text.count("T.copy("),
}))
"""
    proc = _run_capture([sys.executable, "-c", code], env)
    if proc.returncode != 0:
        return {"error": proc.stdout + proc.stderr, "returncode": proc.returncode}
    return json.loads(proc.stdout)


def _applier(action: str, env: dict[str, str]) -> dict[str, Any]:
    applier_env = env.copy()
    if action == "noop":
        applier_env.pop("CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA", None)
        applier_env.pop("MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION", None)
        applier_env.pop("CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK", None)
    elif action == "apply":
        applier_env["CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA"] = "1"
        applier_env["MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION"] = "1"
        applier_env.pop("CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK", None)
    elif action == "rollback":
        applier_env["CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK"] = "1"
        applier_env.pop("CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA", None)
    else:
        raise ValueError(action)
    proc = _run_capture(
        [sys.executable, "-m", "cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches"],
        applier_env,
    )
    return {
        "action": action,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "kernel": _kernel_status(env),
    }


def _synth_mock_data(data_prefix: str, tokenizer_dir: str) -> None:
    from transformers import AutoTokenizer

    idx_path = pathlib.Path(data_prefix + "_text_document.idx")
    if idx_path.exists() and pathlib.Path(tokenizer_dir, "tokenizer.json").exists():
        return

    pathlib.Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tok.save_pretrained(tokenizer_dir)

    raw_path = pathlib.Path(tempfile.mkstemp(prefix="wave29_mock_", suffix=".jsonl")[1])
    vocab = list(tok.get_vocab().keys())[:2048]
    import random

    random.seed(20260430)
    with raw_path.open("w") as f:
        for _ in range(12000):
            f.write(json.dumps({"text": " ".join(random.choices(vocab, k=256))}) + "\n")

    pathlib.Path(data_prefix).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "/opt/megatron-lm/tools/preprocess_data.py",
            "--input",
            str(raw_path),
            "--output-prefix",
            data_prefix,
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            tokenizer_dir,
            "--workers",
            "2",
            "--append-eod",
        ],
        check=True,
    )


def _resolve_dataset() -> dict[str, str]:
    candidates = [
        ("/vol/data/megatron/clang_semantic_4k_v10_train_text_document", "/vol/data/tokenizer"),
        ("/vol/data/megatron/clang_semantic_4k_v10_train_text_document", "/vol/tokenizer"),
        ("/vol/real_data/clang_semantic_4k_v10_train_text_document", "/vol/tokenizer"),
    ]
    for data_path, tokenizer in candidates:
        if pathlib.Path(data_path + ".idx").exists() and pathlib.Path(data_path + ".bin").exists():
            if pathlib.Path(tokenizer, "tokenizer.json").exists():
                return {
                    "kind": "production_real_data",
                    "data_path": data_path,
                    "tokenizer": tokenizer,
                    "note": "real Megatron indexed dataset and HF tokenizer found on Modal volume",
                }

    data_prefix = "/vol/mock_data/clang_semantic_4k_v10_train"
    tokenizer = "/vol/mock_tokenizer"
    _synth_mock_data(data_prefix, tokenizer)
    return {
        "kind": "synthetic_full_shape_mock_data",
        "data_path": data_prefix + "_text_document",
        "tokenizer": tokenizer,
        "note": "real dataset not present on Modal volume; generated indexed mock text and kept NAM56R full model/sequence boundary",
    }


def _build_pattern_and_native_args(
    env: dict[str, str],
    pp_size: int,
    vpp_size: int,
    mtp_depths: int,
    dsa_loss_coeff: float,
) -> tuple[str, list[str]]:
    code = f"""
from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

mtp_depths = {mtp_depths}
pp = {pp_size}
vpp = {vpp_size}
p = build_default_hybrid_layer_pattern(mtp_depths=max(mtp_depths, 0))
if "/" in p:
    main, mtp_part = p.split("/", 1)
else:
    main, mtp_part = p, ""
if mtp_depths == 0:
    mtp_part = ""
n_chunks = pp * max(vpp, 1)
if n_chunks > 1:
    total = len(main)
    per = total // n_chunks
    assert total % n_chunks == 0, f"cannot split {{total}}-layer main into {{n_chunks}} equal chunks"
    main = "|".join(main[i * per:(i + 1) * per] for i in range(n_chunks))
pattern = main + (("/" + mtp_part) if mtp_part else "")

plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=max(mtp_depths, 1))
bundle = build_nam56r_megatron_native_args(
    plan=plan,
    enable_mla=True,
    enable_mtp=mtp_depths > 0,
    mtp_mode="hybrid",
    mtp_num_predictors=mtp_depths,
    enable_moe=True,
    enable_dsa=True,
    dsa_indexer_loss_coeff={dsa_loss_coeff},
)
print("PATTERN", pattern)
print("NATIVE_ARGS", bundle.to_shell_fragment())
"""
    proc = _run_capture([sys.executable, "-c", code], env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    pattern = ""
    native = ""
    for line in proc.stdout.splitlines():
        if line.startswith("PATTERN "):
            pattern = line[len("PATTERN ") :]
        elif line.startswith("NATIVE_ARGS "):
            native = line[len("NATIVE_ARGS ") :]
    if not pattern:
        raise RuntimeError(f"helper did not emit pattern:\n{proc.stdout}")
    return pattern, shlex.split(native)


def _build_train_cmd(
    workdir: pathlib.Path,
    dataset: dict[str, str],
    train_iters: int,
    profile: bool,
) -> tuple[list[str], dict[str, Any]]:
    env = _base_env()
    nproc = 2
    tp_size = 1
    pp_size = 2
    vpp_size = 2
    ep_size = 1
    mbs = int(os.environ.get("CPPMEGA_WAVE29_MBS", "1"))
    gbs = int(os.environ.get("CPPMEGA_WAVE29_GBS", "8"))
    seq_len = int(os.environ.get("CPPMEGA_WAVE29_SEQ_LEN", "4096"))
    mtp_depths = int(os.environ.get("CPPMEGA_WAVE29_MTP_DEPTHS", "2"))
    dsa_loss_coeff = float(os.environ.get("CPPMEGA_WAVE29_DSA_LOSS_COEFF", "0.001"))

    pattern, native_args = _build_pattern_and_native_args(
        env,
        pp_size,
        vpp_size,
        mtp_depths,
        dsa_loss_coeff,
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port=29529",
        str(workdir / "pretrain_mamba.py"),
        "--data-path",
        "1.0",
        dataset["data_path"],
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        dataset["tokenizer"],
        "--split",
        "98,1,1",
        "--vocab-size",
        "131072",
        "--make-vocab-size-divisible-by",
        "128",
        "--tensor-model-parallel-size",
        str(tp_size),
        "--pipeline-model-parallel-size",
        str(pp_size),
        "--context-parallel-size",
        "1",
        "--sequence-parallel",
        "--use-distributed-optimizer",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--no-rope-fusion",
        "--hybrid-layer-pattern",
        pattern,
        "--hidden-size",
        "3584",
        "--ffn-hidden-size",
        "18944",
        "--num-attention-heads",
        "28",
        "--seq-length",
        str(seq_len),
        "--max-position-embeddings",
        str(seq_len),
        "--micro-batch-size",
        str(mbs),
        "--global-batch-size",
        str(gbs),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        "50000000",
        "--eval-iters",
        "1",
        "--lr",
        "1e-4",
        "--min-lr",
        "1e-5",
        "--lr-decay-style",
        "constant",
        "--position-embedding-type",
        "rope",
        "--normalization",
        "RMSNorm",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl",
        "transformer_engine",
        "--use-flash-attn",
        "--attention-backend",
        "flash",
        "--spec",
        "cppmega.megatron.nam56r_full_spec",
        "build_cppmega_nam56r_full_stack_spec",
        "--cuda-graph-impl",
        "transformer_engine",
        "--cuda-graph-scope",
        "attn",
        "mamba",
        "moe_router",
        "moe_preprocess",
        "--cuda-graph-warmup-steps",
        "3",
        "--moe-pad-expert-input-to-capacity",
        "--moe-expert-capacity-factor",
        "1.0",
        "--no-check-for-nan-in-loss-and-grad",
        *native_args,
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "moe_act",
        "--save",
        str(workdir / "ckpt"),
        "--load",
        str(workdir / "ckpt"),
        "--save-interval",
        "1000000",
        "--log-interval",
        "1",
        "--log-throughput",
    ]
    attention_backend = os.environ.get("CPPMEGA_WAVE29_ATTN_BACKEND", "flash")
    meta = {
        "nproc_per_node": nproc,
        "gpu_spec": GPU_SPEC,
        "tp": tp_size,
        "pp": pp_size,
        "vpp": vpp_size,
        "ep": ep_size,
        "mbs": mbs,
        "gbs": gbs,
        "seq_len": seq_len,
        "mtp_depths": mtp_depths,
        "dsa_loss_coeff": dsa_loss_coeff,
        "hybrid_layer_pattern": pattern,
        "tokens_per_iter": gbs * seq_len,
        "profile_mode": profile,
        "attention_backend": attention_backend,
    }
    try:
        idx = cmd.index("--attention-backend")
        cmd[idx + 1] = attention_backend
    except ValueError:
        raise RuntimeError("internal command construction error: missing --attention-backend")
    return cmd, meta


def _parse_log(log_path: pathlib.Path, tokens_per_iter: int) -> dict[str, Any]:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    elapsed_ms = [
        float(m.group(1))
        for m in re.finditer(r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    ]
    tok_sec_values = []
    for pattern in (
        r"(?:tokens/sec|tok/sec|tokens per second|tokens/s)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens/sec|tok/sec|tokens per second|tokens/s)",
    ):
        tok_sec_values.extend(float(m.group(1)) for m in re.finditer(pattern, text, re.I))
    computed_tok_sec = None
    if elapsed_ms:
        computed_tok_sec = tokens_per_iter / (elapsed_ms[-1] / 1000.0)
    peak_alloc = [
        float(m.group(1))
        for m in re.finditer(r"peak_alloc_gib=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    peak_reserved = [
        float(m.group(1))
        for m in re.finditer(r"peak_reserved_gib=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    iterations = [int(m.group(1)) for m in re.finditer(r"iteration\s+([0-9]+)\s*/", text)]
    return {
        "iterations_seen": max(iterations) if iterations else 0,
        "step_elapsed_ms": elapsed_ms,
        "last_step_elapsed_ms": elapsed_ms[-1] if elapsed_ms else None,
        "avg_step_elapsed_ms": (sum(elapsed_ms) / len(elapsed_ms)) if elapsed_ms else None,
        "tok_sec_reported_last": tok_sec_values[-1] if tok_sec_values else None,
        "tok_sec_from_last_step": computed_tok_sec,
        "peak_alloc_gib": max(peak_alloc) if peak_alloc else None,
        "peak_reserved_gib": max(peak_reserved) if peak_reserved else None,
        "last_120_lines": "\n".join(text.splitlines()[-120:]),
    }


def _write_summary(out_dir: pathlib.Path, result: dict[str, Any]) -> None:
    rows = result.get("variants", [])
    lines = [
        "| variant | status | steps seen | tok/sec | avg step ms | peak alloc GiB | peak reserved GiB | log |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        tok = metrics.get("tok_sec_reported_last") or metrics.get("tok_sec_from_last_step")
        lines.append(
            "| {variant} | {status} | {steps} | {tok} | {avg_ms} | {peak_alloc} | {peak_reserved} | {log} |".format(
                variant=row.get("variant"),
                status=row.get("run", {}).get("status"),
                steps=metrics.get("iterations_seen", 0),
                tok="" if tok is None else f"{tok:.3f}",
                avg_ms="" if metrics.get("avg_step_elapsed_ms") is None else f"{metrics['avg_step_elapsed_ms']:.3f}",
                peak_alloc="" if metrics.get("peak_alloc_gib") is None else f"{metrics['peak_alloc_gib']:.3f}",
                peak_reserved="" if metrics.get("peak_reserved_gib") is None else f"{metrics['peak_reserved_gib']:.3f}",
                log=row.get("run", {}).get("log"),
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=16,
    memory=128 * 1024,
    timeout=1800,
    volumes={"/vol": results_vol, "/cache": cache_vol},
    retries=0,
)
def preflight(run_id: str = "") -> dict[str, Any]:
    import torch
    import transformer_engine
    import transformer_engine.pytorch  # noqa: F401
    import flash_attn  # noqa: F401
    import mamba_ssm  # noqa: F401
    import tilelang

    run_id = run_id or f"preflight_{_utc_stamp()}"
    out_dir = pathlib.Path("/vol") / BENCH_DIR.lstrip("/") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    noop = _applier("noop", env)
    rollback = _applier("rollback", env)
    apply = _applier("apply", env)
    final_rollback = _applier("rollback", env)
    result = {
        "run_id": run_id,
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
        "device_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine": transformer_engine.__version__,
        "tilelang": getattr(tilelang, "__version__", None),
        "tilelang_file": getattr(tilelang, "__file__", None),
        "applier_noop": noop,
        "applier_pre_rollback": rollback,
        "applier_apply": apply,
        "applier_final_rollback": final_rollback,
        "final_kernel": _kernel_status(env),
    }
    (out_dir / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results_vol.commit()
    return result


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=24,
    memory=192 * 1024,
    timeout=8 * 3600,
    volumes={"/vol": results_vol, "/cache": cache_vol},
    retries=0,
)
def gate(
    run_id: str = "",
    train_iters: int = 20,
    profile: bool = False,
    timeout_per_variant_s: int = 10800,
) -> dict[str, Any]:
    if train_iters < 20:
        raise ValueError(f"train_iters={train_iters}; full gate requires at least 20")

    run_id = run_id or f"gate_{_utc_stamp()}"
    out_dir = pathlib.Path("/vol") / BENCH_DIR.lstrip("/") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env()
    if profile:
        env.setdefault("NVTE_DEBUG", "1")
        env.setdefault("NVTE_DEBUG_LEVEL", "2")
    dataset = _resolve_dataset()
    workdir = pathlib.Path(f"/tmp/{run_id}")
    _write_workdir(workdir)

    cmd, meta = _build_train_cmd(workdir, dataset, train_iters, profile=profile)
    (out_dir / "command.json").write_text(json.dumps({"argv": cmd, "meta": meta, "dataset": dataset}, indent=2) + "\n")

    result: dict[str, Any] = {
        "run_id": run_id,
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
        "requested_train_iters": train_iters,
        "dataset": dataset,
        "command_meta": meta,
        "variants": [],
        "preflight_kernel": _kernel_status(env),
        "profile_mode": profile,
    }

    noop = _applier("noop", env)
    pre_rollback = _applier("rollback", env)
    result["applier_noop"] = noop
    result["applier_pre_rollback"] = pre_rollback

    try:
        for variant in ("baseline", "stage2_force_nontma_bf1_bb0"):
            if variant == "stage2_force_nontma_bf1_bb0":
                result["applier_apply"] = _applier("apply", env)
            kernel = _kernel_status(env)
            log_file = out_dir / f"{variant}.log"
            run = _run_streaming(cmd, env, log_file, timeout_per_variant_s, cwd=workdir)
            metrics = _parse_log(log_file, meta["tokens_per_iter"])
            row = {
                "variant": variant,
                "kernel": kernel,
                "run": run,
                "metrics": metrics,
            }
            result["variants"].append(row)
            _write_summary(out_dir, result)
            results_vol.commit()
            if run["status"] != "ok":
                result["blocker"] = {
                    "variant": variant,
                    "status": run["status"],
                    "returncode": run["returncode"],
                    "log": run["log"],
                    "tail": metrics.get("last_120_lines", ""),
                }
                break
    finally:
        result["applier_final_rollback"] = _applier("rollback", env)
        result["final_kernel"] = _kernel_status(env)
        _write_summary(out_dir, result)
        results_vol.commit()
    return result


@app.local_entrypoint()
def main(
    run_id: str = "",
    train_iters: int = 20,
    profile: bool = False,
    preflight_only: bool = False,
    timeout_per_variant_s: int = 10800,
) -> None:
    if preflight_only:
        result = preflight.remote(run_id or f"preflight_{_utc_stamp()}")
    else:
        result = gate.remote(run_id or f"gate_{_utc_stamp()}", train_iters, profile, timeout_per_variant_s)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def launch_preflight(run_id: str = "") -> None:
    result = preflight.remote(run_id or f"preflight_{_utc_stamp()}")
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def launch_gate(
    run_id: str = "",
    train_iters: int = 20,
    profile: bool = False,
    timeout_per_variant_s: int = 10800,
) -> None:
    result = gate.remote(run_id or f"gate_{_utc_stamp()}", train_iters, profile, timeout_per_variant_s)
    print(json.dumps(result, indent=2, sort_keys=True))
