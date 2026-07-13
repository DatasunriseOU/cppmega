#!/usr/bin/env python3
"""Run the fail-closed cppmega production Megatron H200 preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega.recipes.run_profiles import get_run_profile, profile_shell_assignments
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _validate_prefix_manifest_contract,
    _validate_tokenizer_directory,
)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_wrappers(workdir: Path) -> Path:
    wrapper = workdir / "pretrain_mamba.py"
    wrapper.write_text(
        """from __future__ import annotations
import atexit
import os
import runpy
import sys

from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

apply_te_checkpoint_kwarg_patch()
apply_dsa_indexer_fused_patch()
apply_graph_route_attention_bias_patch()
import cppmega.megatron.structure_dataset_patch  # noqa: F401

@atexit.register
def _cppmega_distributed_shutdown():
    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@atexit.register
def _cppmega_peak_memory_report():
    import torch
    if torch.cuda.is_available():
        print(
            'CPPMEGA_CUDA_PEAK allocated_gib='
            f'{torch.cuda.max_memory_allocated() / 1024**3:.3f} '
            'reserved_gib='
            f'{torch.cuda.max_memory_reserved() / 1024**3:.3f}',
            flush=True,
        )

_inner = '/opt/megatron-lm/pretrain_mamba.py'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(_inner))
sys.argv[0] = _inner
runpy.run_path(_inner, run_name='__main__')
""",
        encoding="utf-8",
    )
    (workdir / "mamba_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder\n",
        encoding="utf-8",
    )
    (workdir / "hybrid_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as hybrid_builder\n",
        encoding="utf-8",
    )
    return wrapper


def _profile_environment(
    *, sequence_length: int, micro_batch_size: int, fp8_recipe: str
) -> dict[str, str]:
    profile = get_run_profile("h200_cpp_world_mini")
    profile.training.seq_length = sequence_length
    profile.training.micro_batch_size = micro_batch_size
    profile.training.global_batch_size = micro_batch_size
    profile.precision.fp8_recipe = fp8_recipe
    environment = os.environ.copy()
    environment.update(profile_shell_assignments(profile))
    environment.update(
        {
            "CPPMEGA_STRUCTURE_ENABLED": "1",
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "1",
            "CPPMEGA_GRAPH_MAX_EDGES": environment.get(
                "CPPMEGA_GRAPH_MAX_EDGES", "256"
            ),
            "CPPMEGA_GRAPH_MAX_CHUNKS": environment.get(
                "CPPMEGA_GRAPH_MAX_CHUNKS", "256"
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_GRAPH_REGISTER": "0",
        }
    )
    return environment


def build_megatron_command(
    *,
    wrapper: Path,
    data_prefix: Path,
    tokenizer_model: Path,
    checkpoint_root: Path,
    sequence_length: int,
    micro_batch_size: int,
    train_iters: int,
    environment: dict[str, str],
    load_checkpoint: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        str(wrapper),
        "--data-path",
        "1.0",
        str(data_prefix),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_model),
        "--vocab-size",
        "65536",
        "--make-vocab-size-divisible-by",
        "128",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--hybrid-layer-pattern",
        environment["HYBRID_LAYER_PATTERN"],
        "--hidden-size",
        environment["CPPMEGA_HIDDEN_SIZE"],
        "--ffn-hidden-size",
        environment["CPPMEGA_FFN_HIDDEN_SIZE"],
        "--num-attention-heads",
        environment["CPPMEGA_NUM_ATTN_HEADS"],
        "--group-query-attention",
        "--num-query-groups",
        environment["CPPMEGA_NUM_QUERY_GROUPS"],
        "--kv-channels",
        environment["CPPMEGA_KV_CHANNELS"],
        "--swiglu",
        "--rotary-base",
        "10000",
        "--seq-length",
        str(sequence_length),
        "--max-position-embeddings",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(micro_batch_size),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        "50000000",
        "--eval-iters",
        "1",
        "--lr",
        environment["CPPMEGA_LR"],
        "--min-lr",
        environment["CPPMEGA_MIN_LR"],
        "--lr-decay-style",
        "constant",
        "--position-embedding-type",
        "rope",
        "--no-rope-fusion",
        "--normalization",
        "RMSNorm",
        "--disable-bias-linear",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        environment["CPPMEGA_ATTN_BACKEND"],
        "--spec",
        "cppmega.megatron.nam56r_noconv_spec",
        "build_cppmega_nam56r_noconv_stack_spec",
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl",
        "te",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "mlp",
        "--clip-grad",
        "1.0",
        "--optimizer",
        environment["CPPMEGA_OPTIMIZER"],
        "--no-check-for-nan-in-loss-and-grad",
        "--rerun-mode",
        "disabled",
        "--save",
        str(checkpoint_root),
        "--save-interval",
        "1",
        "--log-interval",
        "1",
    ]
    if environment.get("CPPMEGA_USE_FLASH_ATTN") == "1":
        command.append("--use-flash-attn")
    if environment.get("CPPMEGA_FP8_RECIPE") == "tensorwise":
        command.extend(
            [
                "--fp8-format",
                environment["CPPMEGA_FP8_FORMAT"],
                "--fp8-recipe",
                "tensorwise",
                "--fp8-amax-history-len",
                "16",
                "--fp8-amax-compute-algo",
                "max",
            ]
        )
    if load_checkpoint:
        command.extend(["--load", str(checkpoint_root)])
    return command


def _stack_report() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("H200 preflight requires CUDA")
    device = torch.cuda.get_device_properties(0)
    if "H200" not in device.name or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError(
            f"H200 preflight requires an H200 SM90 GPU, got {device.name!r} "
            f"capability={torch.cuda.get_device_capability(0)!r}"
        )
    modules = {}
    for name in (
        "torch",
        "transformer_engine",
        "transformer_engine.pytorch",
        "flash_attn",
        "megatron.core",
        "cppmega",
    ):
        module = importlib.import_module(name)
        modules[name] = {
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": sys.version,
        "modules": modules,
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": device.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(device.total_memory),
        },
        "nvidia_smi": nvidia_smi,
    }


def _iteration_evidence(text: str, *, expected_iteration: int) -> dict[str, object]:
    if not re.search(
        rf"iteration\s+{expected_iteration}/\s*{expected_iteration}", text
    ):
        raise RuntimeError(
            f"H200 preflight log lacks iteration {expected_iteration} completion"
        )

    def last_float(label: str, pattern: str) -> float:
        values = re.findall(pattern, text, flags=re.IGNORECASE)
        if not values:
            raise RuntimeError(f"H200 preflight log lacks {label}")
        try:
            return float(values[-1])
        except ValueError as error:
            raise RuntimeError(f"H200 preflight log has invalid {label}") from error

    loss = last_float("LM loss", r"\blm loss:\s*([^\s|]+)")
    grad_norm = last_float("grad norm", r"\bgrad norm:\s*([^\s|]+)")
    if not math.isfinite(loss) or loss <= 0:
        raise RuntimeError(
            f"H200 preflight requires finite positive LM loss, got {loss}"
        )
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(
            f"H200 preflight requires finite positive grad norm, got {grad_norm}"
        )
    skipped_values = re.findall(
        r"number of skipped iterations:\s*(\d+)", text, flags=re.IGNORECASE
    )
    nan_values = re.findall(
        r"number of nan iterations:\s*(\d+)", text, flags=re.IGNORECASE
    )
    if not skipped_values or int(skipped_values[-1]) != 0:
        raise RuntimeError("H200 preflight log reports skipped iterations")
    if not nan_values or int(nan_values[-1]) != 0:
        raise RuntimeError("H200 preflight log reports NaN iterations")
    return {
        "iteration": expected_iteration,
        "lm_loss": loss,
        "grad_norm": grad_norm,
        "skipped_iterations": 0,
        "nan_iterations": 0,
    }


def _run_phase(
    *,
    name: str,
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    batch_receipt: Path,
    expected_iteration: int,
    checkpoint_root: Path,
) -> dict[str, object]:
    batch_receipt.unlink(missing_ok=True)
    phase_environment = dict(environment)
    phase_environment["CPPMEGA_H200_BATCH_RECEIPT"] = str(batch_receipt)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            env=phase_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"H200 preflight {name} phase failed with exit {result.returncode}"
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iteration_evidence = _iteration_evidence(
        text, expected_iteration=expected_iteration
    )
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != str(
        expected_iteration
    ):
        raise RuntimeError(
            f"H200 preflight {name} checkpoint did not reach {expected_iteration}"
        )
    if not batch_receipt.is_file():
        raise RuntimeError(f"H200 preflight {name} did not record a production batch")
    batch = json.loads(batch_receipt.read_text(encoding="utf-8"))
    if batch.get("status") != "verified":
        raise RuntimeError(f"H200 preflight {name} production batch is not verified")
    peak = re.findall(
        r"CPPMEGA_CUDA_PEAK allocated_gib=([0-9.]+) reserved_gib=([0-9.]+)", text
    )
    if not peak:
        raise RuntimeError(f"H200 preflight {name} log lacks CUDA peak memory")
    return {
        "status": "passed",
        "command": command,
        "command_shell": shlex.join(command),
        "log": str(log_path),
        "batch_receipt": str(batch_receipt),
        "completed_iteration": expected_iteration,
        "iteration_evidence": iteration_evidence,
        "cuda_peak_allocated_gib": float(peak[-1][0]),
        "cuda_peak_reserved_gib": float(peak[-1][1]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-prefix", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise"), default="off")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/data/cppmega_h200_preflight_checkpoint"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.sequence_length <= 0 or args.micro_batch_size <= 0:
        raise ValueError("sequence length and micro batch size must be positive")
    prefix_manifest, _referenced = _validate_prefix_manifest_contract(args.data_prefix)
    _validate_tokenizer_directory(args.tokenizer_model)
    environment = _profile_environment(
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        fp8_recipe=args.fp8_recipe,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing stale H200 preflight receipt: {output}")
    if args.checkpoint_root.exists():
        raise RuntimeError(
            f"refusing stale H200 preflight checkpoint root: {args.checkpoint_root}"
        )

    with tempfile.TemporaryDirectory(prefix="cppmega-h200-preflight-") as raw_workdir:
        workdir = Path(raw_workdir)
        wrapper = _write_wrappers(workdir)
        save_command = build_megatron_command(
            wrapper=wrapper,
            data_prefix=args.data_prefix,
            tokenizer_model=args.tokenizer_model,
            checkpoint_root=args.checkpoint_root,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            train_iters=1,
            environment=environment,
            load_checkpoint=False,
        )
        restore_command = build_megatron_command(
            wrapper=wrapper,
            data_prefix=args.data_prefix,
            tokenizer_model=args.tokenizer_model,
            checkpoint_root=args.checkpoint_root,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            train_iters=2,
            environment=environment,
            load_checkpoint=True,
        )
        base_receipt: dict[str, object] = {
            "schema": "cppmega_h200_megatron_preflight_v1",
            "status": "dry_run" if args.dry_run else "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data": {
                "prefix": str(args.data_prefix.resolve()),
                "manifest": prefix_manifest,
                "sequence_length": args.sequence_length,
                "micro_batch_size": args.micro_batch_size,
            },
            "checkpoint": {
                "root": str(args.checkpoint_root),
                "save_iteration": 1,
                "restored_from_iteration": 1,
                "post_restore_iteration": 2,
                "full_optimizer_and_rng_state": True,
            },
            "commands": {
                "save": save_command,
                "restore": restore_command,
            },
        }
        if args.dry_run:
            _write_json_atomic(output, base_receipt)
            print(json.dumps(base_receipt, indent=2, sort_keys=True))
            return 0

        stack = _stack_report()
        try:
            save = _run_phase(
                name="save",
                command=save_command,
                environment=environment,
                log_path=output.parent / "h200_preflight_save.log",
                batch_receipt=output.parent / "h200_preflight_save_batch.json",
                expected_iteration=1,
                checkpoint_root=args.checkpoint_root,
            )
            restore = _run_phase(
                name="restore",
                command=restore_command,
                environment=environment,
                log_path=output.parent / "h200_preflight_restore.log",
                batch_receipt=output.parent / "h200_preflight_restore_batch.json",
                expected_iteration=2,
                checkpoint_root=args.checkpoint_root,
            )
        except Exception as error:
            _write_json_atomic(
                output,
                {
                    **base_receipt,
                    "status": "failed",
                    "stack": stack,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        receipt = {
            **base_receipt,
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "stack": stack,
            "phases": {"save": save, "restore": restore},
        }
        _write_json_atomic(output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
