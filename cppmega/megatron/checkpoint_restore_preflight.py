"""Runtime proof that Megatron restored model, optimizer, and RNG state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import struct
from typing import Any

import numpy as np

from cppmega.receipt_binding import validate_binding_shape

_PATCH_MARKER = "__cppmega_checkpoint_restore_preflight_patched__"


def _update_fingerprint(digest: Any, value: object) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _update_fingerprint(digest, key)
            _update_fingerprint(digest, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        digest.update(b"sequence\0")
        digest.update(struct.pack("<Q", len(value)))
        for item in value:
            _update_fingerprint(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + bytes([value]))
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"int\0" + struct.pack("<Q", len(encoded)) + encoded)
    elif isinstance(value, float):
        if math.isnan(value):
            digest.update(b"float\0nan")
        else:
            digest.update(b"float\0" + struct.pack("<d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str\0" + struct.pack("<Q", len(encoded)) + encoded)
    elif isinstance(value, (bytes, bytearray)):
        encoded = bytes(value)
        digest.update(b"bytes\0" + struct.pack("<Q", len(encoded)) + encoded)
    else:
        raise TypeError(
            "checkpoint fingerprint encountered unsupported state value "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )


def state_fingerprint(value: object) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, value)
    return digest.hexdigest()


def _model_state(model: object) -> object:
    models = (
        list(model)
        if isinstance(model, Sequence) and not isinstance(model, (str, bytes))
        else [model]
    )
    states = []
    for index, item in enumerate(models):
        state_dict = getattr(item, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError(f"checkpoint model[{index}] has no state_dict")
        states.append(state_dict())
    return states


def _optimizer_state(optimizer: object) -> object:
    state_dict = getattr(optimizer, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("checkpoint optimizer has no state_dict")
    return state_dict()


def _rng_state() -> dict[str, object]:
    import torch

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        tracker = get_cuda_rng_tracker()
        get_states = getattr(tracker, "get_states", None)
        if callable(get_states):
            state["megatron_cuda_rng_tracker"] = get_states()
    except (ImportError, AttributeError):
        pass
    return state


def runtime_state_fingerprints(model: object, optimizer: object) -> dict[str, str]:
    return {
        "model": state_fingerprint(_model_state(model)),
        "optimizer": state_fingerprint(_optimizer_state(optimizer)),
        "rng": state_fingerprint(_rng_state()),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _argument(
    function: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
) -> object:
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    if name not in bound.arguments:
        raise RuntimeError(f"Megatron checkpoint call did not provide {name}")
    return bound.arguments[name]


def _receipt_path() -> Path:
    raw = os.environ.get("CPPMEGA_H200_CHECKPOINT_STATE_RECEIPT")
    if not raw:
        raise RuntimeError("CPPMEGA_H200_CHECKPOINT_STATE_RECEIPT is required")
    return Path(raw)


def _proof_mode() -> str:
    mode = os.environ.get("CPPMEGA_H200_CHECKPOINT_PROOF_MODE", "")
    if mode not in {"save", "restore"}:
        raise RuntimeError(
            "CPPMEGA_H200_CHECKPOINT_PROOF_MODE must be 'save' or 'restore'"
        )
    return mode


def _receipt_binding() -> dict[str, object]:
    raw = os.environ.get("CPPMEGA_H200_RECEIPT_BINDING")
    if not raw:
        raise RuntimeError("CPPMEGA_H200_RECEIPT_BINDING is required")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid H200 receipt binding JSON") from error
    return validate_binding_shape(value, where="checkpoint state")


def _write_state_receipt(
    *,
    mode: str,
    iteration: int,
    model: object,
    optimizer: object,
) -> None:
    if iteration != 1:
        raise RuntimeError(
            f"H200 checkpoint {mode} proof requires iteration 1, got {iteration}"
        )
    if optimizer is None:
        raise RuntimeError("H200 checkpoint proof requires optimizer state")
    receipt = {
        "schema": "cppmega_h200_checkpoint_state_v1",
        "status": "verified",
        "mode": mode,
        "iteration": iteration,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "rank": int(os.environ.get("RANK", "0")),
        "fingerprints": runtime_state_fingerprints(model, optimizer),
        "binding": _receipt_binding(),
    }
    if receipt["rank"] == 0:
        _write_json_atomic(_receipt_path(), receipt)


def _loaded_iteration(result: object) -> int:
    value = result[0] if isinstance(result, tuple) else result
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"Megatron load_checkpoint returned invalid iteration {value!r}"
        )
    return value


def install_checkpoint_restore_preflight() -> None:
    """Patch the exact save/load functions imported by Megatron training."""

    from megatron.training import checkpointing
    from megatron.training import training

    original_save = checkpointing.save_checkpoint
    if not getattr(original_save, _PATCH_MARKER, False):

        @functools.wraps(original_save)
        def save_checkpoint_with_proof(*args, **kwargs):
            iteration = int(_argument(original_save, args, kwargs, "iteration"))
            if _proof_mode() != "save":
                return original_save(*args, **kwargs)
            model = _argument(original_save, args, kwargs, "model")
            optimizer = _argument(original_save, args, kwargs, "optimizer")
            if iteration != 1:
                return original_save(*args, **kwargs)
            fingerprints = runtime_state_fingerprints(model, optimizer)
            result = original_save(*args, **kwargs)
            receipt = {
                "schema": "cppmega_h200_checkpoint_state_v1",
                "status": "verified",
                "mode": "save",
                "iteration": iteration,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "rank": int(os.environ.get("RANK", "0")),
                "fingerprints": fingerprints,
                "binding": _receipt_binding(),
            }
            if receipt["rank"] == 0:
                _write_json_atomic(_receipt_path(), receipt)
            return result

        setattr(save_checkpoint_with_proof, _PATCH_MARKER, True)
        checkpointing.save_checkpoint = save_checkpoint_with_proof
        if getattr(training, "save_checkpoint", None) is original_save:
            training.save_checkpoint = save_checkpoint_with_proof

    original_load = checkpointing.load_checkpoint
    if not getattr(original_load, _PATCH_MARKER, False):

        @functools.wraps(original_load)
        def load_checkpoint_with_proof(*args, **kwargs):
            model = _argument(original_load, args, kwargs, "model")
            optimizer = _argument(original_load, args, kwargs, "optimizer")
            result = original_load(*args, **kwargs)
            if _proof_mode() != "restore":
                raise RuntimeError(
                    "Megatron loaded a checkpoint outside the restore proof phase"
                )
            iteration = _loaded_iteration(result)
            expected = int(
                os.environ.get("CPPMEGA_H200_EXPECTED_LOAD_ITERATION", "1")
            )
            if iteration != expected:
                raise RuntimeError(
                    "Megatron checkpoint load did not restore the required "
                    f"iteration: {iteration} != {expected}"
                )
            _write_state_receipt(
                mode="load",
                iteration=iteration,
                model=model,
                optimizer=optimizer,
            )
            return result

        setattr(load_checkpoint_with_proof, _PATCH_MARKER, True)
        checkpointing.load_checkpoint = load_checkpoint_with_proof
        if getattr(training, "load_checkpoint", None) is original_load:
            training.load_checkpoint = load_checkpoint_with_proof


__all__ = [
    "install_checkpoint_restore_preflight",
    "runtime_state_fingerprints",
    "state_fingerprint",
]
