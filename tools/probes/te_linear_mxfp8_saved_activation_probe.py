#!/usr/bin/env python3
"""Check cppmega MXFP8 TE Linear saves MXFP8 backward operands, not BF16 inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cppmega_fp8_shim() -> Any:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    shim_path = repo_root / "scripts" / "cppmega_fp8_shim.py"
    spec = importlib.util.spec_from_file_location(
        "cppmega_fp8_shim_saved_activation_probe",
        shim_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_mxfp8_profile_env(backend: str | None) -> str:
    os.environ.setdefault("CPPMEGA_ALLOW_TE_MXFP8_SM12", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    if backend is not None:
        os.environ["CPPMEGA_TE_MXFP8_BWD_BACKEND"] = backend
    backend = os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_BACKEND", "te_tn_adapter")
    no_sidecar = backend == "cutlass_native"
    os.environ.setdefault(
        "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND",
        "off" if no_sidecar else "te",
    )
    os.environ.setdefault(
        "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED",
        "0" if no_sidecar else "1",
    )
    os.environ.setdefault(
        "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT",
        "0" if no_sidecar else "1",
    )
    os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK", "0")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_DGRAD_BF16", "0")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_WGRAD_BF16", "0")
    return backend


def _saved_tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "is_cuda": bool(tensor.is_cuda),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "data_ptr": int(tensor.data_ptr()) if tensor.is_cuda else 0,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.m % 32 or args.n % 32 or args.k % 32:
        raise SystemExit("--m, --n, and --k must be multiples of 32 for MXFP8")

    backend = _set_mxfp8_profile_env(args.backend)
    shim_module = _load_cppmega_fp8_shim()

    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    torch.manual_seed(args.seed)
    linear = te.Linear(
        args.k,
        args.n,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    inp = torch.randn(
        args.m,
        args.k,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    saved_tensors: list[dict[str, Any]] = []

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        saved_tensors.append(_saved_tensor_record(tensor))
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.MXFP8BlockScaling()):
            out = linear(inp)
            loss = out.float().square().mean()

    loss.backward()
    torch.cuda.synchronize()

    input_shape = [args.m, args.k]
    weight_shape = [args.n, args.k]
    weight_data_ptr = int(linear.weight.data_ptr())
    transpose_payload_shape = [args.k, args.m]
    saved_bf16_input = [
        rec
        for rec in saved_tensors
        if (
            rec["dtype"] == "torch.bfloat16"
            and rec["shape"] == input_shape
            and rec["data_ptr"] != weight_data_ptr
        )
    ]
    saved_bf16_weight = [
        rec
        for rec in saved_tensors
        if (
            rec["dtype"] == "torch.bfloat16"
            and rec["shape"] == weight_shape
            and rec["data_ptr"] == weight_data_ptr
        )
    ]
    saved_transpose_payload = [
        rec
        for rec in saved_tensors
        if rec["dtype"] == "torch.uint8" and rec["shape"] == transpose_payload_shape
    ]
    saved_input_columnwise_payload = [
        rec
        for rec in saved_tensors
        if rec["dtype"] == "torch.uint8" and rec["shape"] == input_shape
    ]

    stats = (
        shim_module.cppmega_te_mxfp8_bwd_stats_snapshot()
        if hasattr(shim_module, "cppmega_te_mxfp8_bwd_stats_snapshot")
        else {}
    )
    finite_input_grad = bool(torch.isfinite(inp.grad).all().item())
    finite_weight_grad = bool(torch.isfinite(linear.weight.grad).all().item())

    failures: list[str] = []
    if saved_bf16_input:
        failures.append("BF16 input-shaped activation was saved for Linear backward")
    if not finite_input_grad:
        failures.append("input gradient is not finite")
    if not finite_weight_grad:
        failures.append("weight gradient is not finite")
    if (
        int(stats.get("bf16_fallback_dgrad", 0)) != 0
        or int(stats.get("bf16_fallback_wgrad", 0)) != 0
    ):
        failures.append("MXFP8 backward used BF16 fallback")

    transpose_emit_backend = os.environ.get("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND", "")
    direct_no_sidecar = backend == "cutlass_native" and transpose_emit_backend == "off"
    if direct_no_sidecar:
        if int(stats.get("mxfp8_cutlass_native_dgrad", 0)) <= 0:
            failures.append("CUTLASS native backend did not handle dgrad")
        if int(stats.get("mxfp8_cutlass_native_wgrad", 0)) <= 0:
            failures.append("CUTLASS native backend did not handle wgrad")
        if int(stats.get("mxfp8_tn_adapter_te_emit", 0)) != 0:
            failures.append("no-sidecar backend emitted TE transpose operands")
        if int(stats.get("mxfp8_tn_sidecar_attr_attached", 0)) != 0:
            failures.append("no-sidecar backend attached MXFP8 transpose sidecars")
    else:
        if not saved_transpose_payload:
            failures.append(
                "MXFP8 rowwise-transposed payload was not saved for Linear backward"
            )
        if int(stats.get("mxfp8_tn_adapter_saved_transpose_operand", 0)) <= 0:
            failures.append("TN adapter did not consume a saved transpose operand")
        if int(stats.get("mxfp8_tn_adapter_te_emit_deferred", 0)) <= 0:
            failures.append("TE Linear did not defer eager sidecar emission")
        for key in (
            "mxfp8_tn_adapter_te_emit",
            "mxfp8_tn_sidecar_attr_attached",
            "mxfp8_tn_sidecar_registry_peak",
            "mxfp8_tn_sidecar_registry_peak_bytes",
        ):
            if int(stats.get(key, 0)) != 0:
                failures.append(f"{key}={stats.get(key)}; expected 0 for TE Linear deferred path")

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "backend": backend,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "saved_bf16_input_count": len(saved_bf16_input),
        "saved_bf16_weight_count": len(saved_bf16_weight),
        "saved_transpose_payload_count": len(saved_transpose_payload),
        "saved_input_columnwise_payload_count": len(saved_input_columnwise_payload),
        "finite_input_grad": finite_input_grad,
        "finite_weight_grad": finite_weight_grad,
        "saved_tensors": saved_tensors,
        "shim_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--backend",
        choices=("te_tn_adapter", "flashinfer_cutlass", "cutlass_native"),
        default=None,
        help="Override CPPMEGA_TE_MXFP8_BWD_BACKEND before loading the shim.",
    )
    args = parser.parse_args()

    report = _run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
