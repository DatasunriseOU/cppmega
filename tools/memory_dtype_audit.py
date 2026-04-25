"""Runtime dtype/storage audit hook for the local NAM56R GB10 run.

Enable without editing launch scripts by putting this directory on PYTHONPATH
and setting CPPMEGA_MEMORY_DTYPE_AUDIT=1. The companion tools/sitecustomize.py
imports this module and calls install().
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - hook is inert without torch
    torch = None  # type: ignore[assignment]


_INSTALLED = False
_CAPTURED_MODELS: Any = None
_CAPTURED_OPTIMIZER: Any = None
_STEP_COUNT = 0


def _rank0() -> bool:
    if torch is None:
        return True
    try:
        return (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    except Exception:
        return True


def _tensor_storage_id(tensor: torch.Tensor) -> tuple[str, int]:
    try:
        storage = tensor.untyped_storage()
        return str(tensor.device), int(storage.data_ptr())
    except Exception:
        return str(tensor.device), int(tensor.data_ptr())


def _tensor_nbytes(tensor: torch.Tensor, seen: set[tuple[str, int]] | None = None) -> int:
    if seen is not None:
        key = _tensor_storage_id(tensor)
        if key in seen:
            return 0
        seen.add(key)
        try:
            return int(tensor.untyped_storage().nbytes())
        except Exception:
            pass
    return int(tensor.numel() * tensor.element_size())


def _gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024**3)


def _unwrap_models(models: Any) -> list[Any]:
    if models is None:
        return []
    if isinstance(models, (list, tuple)):
        return list(models)
    return [models]


def _param_category(name: str, param: torch.Tensor) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {
        "A_log",
        "dt_bias",
        "D",
        "B_bias",
        "C_bias",
        "B_norm_weight",
        "C_norm_weight",
        "mimo_x",
        "mimo_z",
        "mimo_o",
    }:
        return "mamba_scalar_bc"
    if getattr(param, "is_embedding_or_output_parameter", False):
        return "scalar_fallback_embedding_or_output"
    if getattr(param, "is_emerging_optimizer_fallback_parameter", False):
        return "scalar_fallback_tagged"
    if len(getattr(param, "shape", ())) != 2:
        return "scalar_fallback_non_2d"
    return "muon_matrix"


def _new_bucket() -> dict[str, Any]:
    return {"count": 0, "numel": 0, "bytes": 0}


def _add_bucket(
    buckets: dict[str, dict[str, Any]],
    key_parts: tuple[Any, ...],
    *,
    numel: int,
    nbytes: int,
) -> None:
    key = "|".join(str(part) for part in key_parts)
    bucket = buckets.setdefault(key, _new_bucket())
    bucket["count"] += 1
    bucket["numel"] += int(numel)
    bucket["bytes"] += int(nbytes)


def _collect_params(models: Any) -> tuple[dict[str, Any], dict[int, str], dict[int, str]]:
    seen: set[tuple[str, int]] = set()
    by_dtype: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    main_params: dict[str, dict[str, Any]] = {}
    grads: dict[str, dict[str, Any]] = {}
    main_grads: dict[str, dict[str, Any]] = {}
    mamba_params: list[dict[str, Any]] = []
    param_names: dict[int, str] = {}
    param_categories: dict[int, str] = {}
    model_list = _unwrap_models(models)
    single_model = len(model_list) == 1

    for model_idx, model in enumerate(model_list):
        for name, param in model.named_parameters():
            full_name = name if single_model else f"model{model_idx}.{name}"
            param_names[id(param)] = full_name
            category = _param_category(full_name, param)
            param_categories[id(param)] = category
            nbytes = _tensor_nbytes(param, seen)
            _add_bucket(
                by_dtype,
                (param.dtype, param.device),
                numel=param.numel(),
                nbytes=nbytes,
            )
            _add_bucket(
                by_category,
                (category, param.dtype, param.device),
                numel=param.numel(),
                nbytes=nbytes,
            )
            leaf = full_name.rsplit(".", 1)[-1]
            if category == "mamba_scalar_bc":
                mamba_params.append(
                    {
                        "name": full_name,
                        "leaf": leaf,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "device": str(param.device),
                        "numel": int(param.numel()),
                        "bytes": int(nbytes),
                    }
                )

            main_param = getattr(param, "main_param", None)
            if torch.is_tensor(main_param):
                _add_bucket(
                    main_params,
                    (category, main_param.dtype, main_param.device),
                    numel=main_param.numel(),
                    nbytes=_tensor_nbytes(main_param, seen),
                )

            grad = getattr(param, "grad", None)
            if torch.is_tensor(grad):
                _add_bucket(
                    grads,
                    (category, grad.dtype, grad.device),
                    numel=grad.numel(),
                    nbytes=_tensor_nbytes(grad, seen),
                )

            main_grad = getattr(param, "main_grad", None)
            if torch.is_tensor(main_grad):
                _add_bucket(
                    main_grads,
                    (category, main_grad.dtype, main_grad.device),
                    numel=main_grad.numel(),
                    nbytes=_tensor_nbytes(main_grad, seen),
                )

    return (
        {
            "model_params_by_dtype": by_dtype,
            "model_params_by_category": by_category,
            "optimizer_main_params": main_params,
            "param_grad": grads,
            "param_main_grad": main_grads,
            "mamba_scalar_bc_params": sorted(mamba_params, key=lambda row: row["name"]),
        },
        param_names,
        param_categories,
    )


def _walk_optimizer_wrappers(optimizer: Any) -> list[Any]:
    seen: set[int] = set()
    stack = [optimizer]
    found: list[Any] = []
    while stack:
        opt = stack.pop()
        if opt is None or id(opt) in seen:
            continue
        seen.add(id(opt))
        found.append(opt)
        for attr in ("chained_optimizers", "optimizers"):
            children = getattr(opt, attr, None)
            if isinstance(children, (list, tuple)):
                stack.extend(children)
        inner = getattr(opt, "optimizer", None)
        if inner is not None and inner is not opt:
            stack.append(inner)
        inner = getattr(opt, "_inner", None)
        if inner is not None and inner is not opt:
            stack.append(inner)
    return found


def _iter_state_tensors(prefix: str, value: Any):
    if torch is not None and torch.is_tensor(value):
        yield prefix, value
        return
    if is_dataclass(value):
        for key, item in vars(value).items():
            yield from _iter_state_tensors(f"{prefix}.{key}", item)
        return
    if hasattr(value, "data") and torch is not None and torch.is_tensor(value.data):
        yield f"{prefix}.data", value.data
    if hasattr(value, "absmax") and torch is not None and torch.is_tensor(value.absmax):
        yield f"{prefix}.absmax", value.absmax
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_state_tensors(f"{prefix}.{key}", item)


def _collect_optimizer(
    optimizer: Any,
    param_names: dict[int, str],
    param_categories: dict[int, str],
) -> dict[str, Any]:
    seen_storage: set[tuple[str, int]] = set()
    state_buckets: dict[str, dict[str, Any]] = {}
    wrappers: list[dict[str, Any]] = []

    for opt in _walk_optimizer_wrappers(optimizer):
        wrappers.append(
            {
                "class": type(opt).__name__,
                "base_class": type(getattr(opt, "optimizer", None)).__name__
                if getattr(opt, "optimizer", None) is not None
                else None,
                "param_groups": len(getattr(opt, "param_groups", []) or []),
                "state_entries": len(getattr(opt, "state", {}) or {}),
            }
        )
        if getattr(opt, "chained_optimizers", None):
            continue
        state = getattr(opt, "state", None)
        if not isinstance(state, dict) and not hasattr(state, "items"):
            continue
        try:
            items = list(state.items())
        except Exception:
            continue
        for key, value in items:
            param = key
            if isinstance(key, tuple) and key:
                param = key[-1]
            pcat = param_categories.get(id(param), "<unknown-category>")
            if isinstance(value, dict):
                state_dict = value
            else:
                state_dict = {"value": value}
            for state_key, state_value in state_dict.items():
                for path, tensor in _iter_state_tensors(str(state_key), state_value):
                    nbytes = _tensor_nbytes(tensor, seen_storage)
                    _add_bucket(
                        state_buckets,
                        (pcat, path, tensor.dtype, tensor.device),
                        numel=tensor.numel(),
                        nbytes=nbytes,
                    )

    return {
        "optimizer_wrappers": wrappers,
        "optimizer_state_by_category_key_dtype": state_buckets,
    }


def _collect_ddp_buffers(models: Any) -> dict[str, Any]:
    seen: set[tuple[str, int]] = set()
    buckets: dict[str, dict[str, Any]] = {}
    for model in _unwrap_models(models):
        stack = [model]
        visited: set[int] = set()
        while stack:
            module = stack.pop()
            if id(module) in visited:
                continue
            visited.add(id(module))
            raw_buffers = getattr(module, "__dict__", {}).get("buffers", None)
            if isinstance(raw_buffers, (list, tuple)):
                for idx, buf in enumerate(raw_buffers):
                    for attr in ("param_data", "grad_data", "shared_buffer"):
                        tensor = getattr(buf, attr, None)
                        if torch is not None and torch.is_tensor(tensor):
                            _add_bucket(
                                buckets,
                                (type(buf).__name__, idx, attr, tensor.dtype, tensor.device),
                                numel=tensor.numel(),
                                nbytes=_tensor_nbytes(tensor, seen),
                            )
            try:
                stack.extend(list(module.children()))
            except Exception:
                pass
    return buckets


def _snapshot(tag: str) -> dict[str, Any]:
    params, param_names, param_categories = _collect_params(_CAPTURED_MODELS)
    optimizer = _collect_optimizer(_CAPTURED_OPTIMIZER, param_names, param_categories)
    ddp_buffers = _collect_ddp_buffers(_CAPTURED_MODELS)
    cuda = {}
    if torch is not None and torch.cuda.is_available():
        device = torch.cuda.current_device()
        cuda = {
            "memory_allocated": int(torch.cuda.memory_allocated(device)),
            "memory_reserved": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    return {
        "tag": tag,
        "time_unix": time.time(),
        "pid": os.getpid(),
        "cuda": cuda,
        **params,
        **optimizer,
        "ddp_param_and_grad_buffers": ddp_buffers,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"[dtype_audit] tag={report['tag']}", flush=True)
    for section in (
        "model_params_by_dtype",
        "optimizer_main_params",
        "param_grad",
        "param_main_grad",
        "optimizer_state_by_category_key_dtype",
        "ddp_param_and_grad_buffers",
    ):
        rows = report.get(section, {})
        print(f"[dtype_audit] {section}", flush=True)
        if not rows:
            print("[dtype_audit]   <empty>", flush=True)
            continue
        for key, value in sorted(rows.items(), key=lambda item: (-item[1]["bytes"], item[0]))[:80]:
            print(
                f"[dtype_audit]   {key} count={value['count']} "
                f"numel={value['numel']:,} bytes={value['bytes']:,} "
                f"gib={_gib(value['bytes']):.6f}",
                flush=True,
            )


def _write_report(report: dict[str, Any]) -> None:
    out = os.environ.get("CPPMEGA_MEMORY_DTYPE_AUDIT_OUT")
    if not out:
        run_id = os.environ.get("RUN_ID", f"dtype_audit_{int(time.time())}")
        out = f"/home/dave/logs/{run_id}_dtype_audit.json"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(report)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[dtype_audit] wrote {path}", flush=True)


def capture(tag: str) -> None:
    if torch is None or not _rank0():
        return
    try:
        report = _snapshot(tag)
        _print_summary(report)
        _write_report(report)
    except Exception as exc:  # pragma: no cover - best-effort runtime hook
        print(f"[dtype_audit] capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def install() -> None:
    global _INSTALLED
    if _INSTALLED or os.environ.get("CPPMEGA_MEMORY_DTYPE_AUDIT", "0") != "1":
        return
    _INSTALLED = True
    try:
        import functools
        import megatron.training.training as training
    except Exception as exc:  # pragma: no cover
        print(f"[dtype_audit] install skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return

    original_setup = training.setup_model_and_optimizer

    @functools.wraps(original_setup)
    def wrapped_setup(*args, **kwargs):
        global _CAPTURED_MODELS, _CAPTURED_OPTIMIZER
        result = original_setup(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            _CAPTURED_MODELS = result[0]
            _CAPTURED_OPTIMIZER = result[1]
            capture("after_setup_model_and_optimizer")
        return result

    training.setup_model_and_optimizer = wrapped_setup

    max_steps = int(os.environ.get("CPPMEGA_MEMORY_DTYPE_AUDIT_STEPS", "1"))
    original_train_step = training.train_step

    @functools.wraps(original_train_step)
    def wrapped_train_step(*args, **kwargs):
        global _STEP_COUNT
        result = original_train_step(*args, **kwargs)
        _STEP_COUNT += 1
        if _STEP_COUNT <= max_steps:
            capture(f"after_train_step_{_STEP_COUNT}")
        return result

    training.train_step = wrapped_train_step
    print("[dtype_audit] hooks installed", flush=True)


if __name__ == "__main__":
    print(
        "Set CPPMEGA_MEMORY_DTYPE_AUDIT=1 and put this tools directory on "
        "PYTHONPATH before launching training.",
        file=sys.stderr,
    )
