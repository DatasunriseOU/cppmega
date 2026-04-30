"""Runtime counters for Muon q8 momentum and Newton-Schulz dtype paths."""

from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import dataclass
import functools
import sys
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class MuonDtypeAuditConfig:
    """Typed configuration for Muon dtype-path audit hooks."""

    enabled: bool = True
    log_at_exit: bool = True


_COUNTERS: Counter[str] = Counter()
_INSTALLED = False
_ATEXIT_REGISTERED = False


def reset_muon_dtype_audit() -> None:
    """Reset audit counters. Intended for focused tests."""

    _COUNTERS.clear()


def get_muon_dtype_audit_snapshot() -> dict[str, int]:
    """Return a stable copy of Muon dtype audit counters."""

    snapshot = {key: int(value) for key, value in sorted(_COUNTERS.items())}
    snapshot["bf16_owned_path_observed"] = int(
        snapshot.get("qmuon_grad_dtype_bfloat16_tensors", 0) > 0
        or snapshot.get("ns_step_dtype_bfloat16_tensors", 0) > 0
        or snapshot.get("ns_lowmem_output_dtype_bfloat16_tensors", 0) > 0
    )
    return snapshot


def format_muon_dtype_audit_snapshot(snapshot: dict[str, int] | None = None) -> str:
    """Format counters as one machine-greppable key=value line."""

    if snapshot is None:
        snapshot = get_muon_dtype_audit_snapshot()
    fields = " ".join(f"{key}={value}" for key, value in sorted(snapshot.items()))
    return f"[cppmega_muon_dtype_audit] {fields}"


def print_muon_dtype_audit_snapshot() -> None:
    """Print the current audit snapshot to stderr."""

    print(format_muon_dtype_audit_snapshot(), file=sys.stderr, flush=True)


def install_muon_dtype_audit(config: MuonDtypeAuditConfig | None = None) -> bool:
    """Install runtime wrappers around cppmega qMuon and Megatron NS helpers."""

    global _INSTALLED, _ATEXIT_REGISTERED
    if config is None:
        config = MuonDtypeAuditConfig()
    if not config.enabled:
        return False
    if _INSTALLED:
        return True

    from cppmega.megatron import quantized_muon_momentum as qmuon

    _patch_attr(
        qmuon,
        "quantized_muon_momentum_update_multi_and_normalize_groups_",
        _wrap_qmuon_group_update,
    )
    _patch_attr(
        qmuon,
        "quantized_muon_momentum_update_multi_with_group_sumsq_",
        _wrap_qmuon_group_update,
    )
    _patch_attr(
        qmuon,
        "quantized_muon_momentum_update_multi_and_normalize_",
        _wrap_qmuon_update,
    )
    _patch_attr(qmuon, "quantized_muon_momentum_update_multi_", _wrap_qmuon_update)
    _patch_attr(
        qmuon,
        "quantized_muon_momentum_update_mxfp8_carrier_",
        _wrap_qmuon_mxfp8_carrier_update,
    )

    try:
        import megatron.core.optimizer.emerging_optimizers as eopt
    except Exception:
        eopt = None

    if eopt is not None:
        for name in (
            "quantized_muon_momentum_update_multi_and_normalize_groups_",
            "quantized_muon_momentum_update_multi_with_group_sumsq_",
        ):
            if hasattr(eopt, name):
                _patch_attr(eopt, name, _wrap_qmuon_group_update)
        _patch_attr(eopt, "_newton_schulz_lowmem", _wrap_ns_lowmem)
        _patch_attr(eopt, "newton_schulz_step", _wrap_ns_step)

    _INSTALLED = True
    if config.log_at_exit and not _ATEXIT_REGISTERED:
        atexit.register(print_muon_dtype_audit_snapshot)
        _ATEXIT_REGISTERED = True
    return True


def _patch_attr(module: Any, name: str, wrapper_factory: Callable[[Callable], Callable]) -> None:
    if not hasattr(module, name):
        return
    fn = getattr(module, name)
    if getattr(fn, "_cppmega_muon_dtype_audit_wrapped", False):
        return
    wrapped = wrapper_factory(fn)
    setattr(wrapped, "_cppmega_muon_dtype_audit_wrapped", True)
    setattr(wrapped, "_cppmega_muon_dtype_audit_original", fn)
    setattr(module, name, wrapped)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _tensor_numel(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel())
    except Exception:
        return 0


def _record_tensor(prefix: str, tensor: torch.Tensor) -> None:
    dtype = _dtype_name(tensor.dtype)
    _COUNTERS[f"{prefix}_dtype_{dtype}_tensors"] += 1
    _COUNTERS[f"{prefix}_dtype_{dtype}_elems"] += _tensor_numel(tensor)


def _record_tensors(prefix: str, tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            _record_tensor(prefix, tensor)


def _record_qmuon_update(states: Iterable[Any], grads: Iterable[torch.Tensor], *, grouped: bool) -> None:
    grads = list(grads)
    states = list(states)
    _COUNTERS["qmuon_group_update_calls" if grouped else "qmuon_update_calls"] += 1
    _COUNTERS["qmuon_grad_tensors"] += len(grads)
    _record_tensors("qmuon_grad", grads)
    for state in states:
        _record_qmuon_state(state)


def _record_qmuon_state(state: Any) -> None:
    data = getattr(state, "data", None)
    absmax = getattr(state, "absmax", None)
    if isinstance(data, torch.Tensor):
        _record_tensor("qmuon_state", data)
    if isinstance(absmax, torch.Tensor):
        _record_tensor("qmuon_absmax", absmax)


def _record_mxfp8_carrier(carrier: Any) -> None:
    rowwise_data = getattr(carrier, "rowwise_data", None)
    rowwise_scale_inv = getattr(carrier, "rowwise_scale_inv", None)
    if isinstance(rowwise_data, torch.Tensor):
        _record_tensor("qmuon_carrier_rowwise_data", rowwise_data)
    if isinstance(rowwise_scale_inv, torch.Tensor):
        _record_tensor("qmuon_carrier_rowwise_scale", rowwise_scale_inv)


def _wrap_qmuon_group_update(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(states, grads, *args, **kwargs):
        _record_qmuon_update(states, grads, grouped=True)
        return fn(states, grads, *args, **kwargs)

    return wrapped


def _wrap_qmuon_update(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(states, grads, *args, **kwargs):
        _record_qmuon_update(states, grads, grouped=False)
        return fn(states, grads, *args, **kwargs)

    return wrapped


def _wrap_qmuon_mxfp8_carrier_update(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(state, grad, carrier, *args, **kwargs):
        _COUNTERS["qmuon_mxfp8_carrier_update_calls"] += 1
        _COUNTERS["qmuon_grad_tensors"] += 1
        if isinstance(grad, torch.Tensor):
            _record_tensor("qmuon_grad", grad)
        _record_qmuon_state(state)
        _record_mxfp8_carrier(carrier)
        result = fn(state, grad, carrier, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            _record_tensor("qmuon_carrier_inv_norm", result)
        return result

    return wrapped


def _wrap_ns_lowmem(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(x, steps, coefficient_type, *args, **kwargs):
        _COUNTERS["ns_lowmem_calls"] += 1
        if isinstance(x, torch.Tensor):
            _record_tensor("ns_lowmem_input", x)
            if kwargs.get("already_normalized", False):
                _record_tensor("ns_lowmem_already_normalized_input", x)
        result = fn(x, steps, coefficient_type, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            _record_tensor("ns_lowmem_output", result)
        return result

    return wrapped


def _wrap_ns_step(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(X, *args, **kwargs):
        _COUNTERS["ns_step_calls"] += 1
        if isinstance(X, torch.Tensor):
            _record_tensor("ns_step", X)
        return fn(X, *args, **kwargs)

    return wrapped
