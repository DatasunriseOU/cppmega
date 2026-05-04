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

from cppmega.recipes.run_profiles import RunProfile, profile_shell_assignments


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


def _set_mxfp8_profile_env(
    backend: str | None,
    linear_kernel_contract: str,
) -> tuple[str, str]:
    profile = RunProfile(
        name="te_linear_mxfp8_saved_activation_probe",
        description="Focused TE Linear MXFP8 saved-operand contract probe.",
    )
    precision = profile.precision
    precision.fp8_recipe = "mxfp8"
    precision.allow_te_mxfp8_sm12 = True
    precision.mxfp8_te_cast_only_fast_path = True
    precision.mxfp8_bwd_tn_adapter = True
    precision.mxfp8_bwd_backend = backend or "te_tn_adapter"
    precision.mxfp8_transpose_emit_backend = (
        "off" if precision.mxfp8_bwd_backend == "cutlass_native" else "te"
    )
    precision.mxfp8_transpose_emit_swizzled = precision.mxfp8_bwd_backend != "cutlass_native"
    precision.mxfp8_transpose_emit_strict = precision.mxfp8_bwd_backend != "cutlass_native"
    precision.mxfp8_bwd_allow_bf16_fallback = False
    precision.mxfp8_dgrad_bf16 = False
    precision.mxfp8_wgrad_bf16 = False
    precision.mxfp8_dense_saved_operands = True
    precision.mxfp8_linear_kernel_contract = linear_kernel_contract  # type: ignore[assignment]
    if linear_kernel_contract == "compact_direct_v1":
        if backend is not None and backend != "cutlass_native":
            raise SystemExit(
                "--linear-kernel-contract compact_direct_v1 requires "
                "--backend cutlass_native"
            )
        precision.mxfp8_bwd_backend = "cutlass_native"
        precision.mxfp8_compact_columnwise_backward = True
        precision.mxfp8_dense_saved_operands = False
        precision.mxfp8_transpose_emit_backend = "off"
        precision.mxfp8_transpose_emit_swizzled = False
        precision.mxfp8_transpose_emit_strict = False

    rendered = profile_shell_assignments(profile)
    for key, value in rendered.items():
        os.environ.setdefault(key, value)
    if backend is not None and linear_kernel_contract != "compact_direct_v1":
        os.environ["CPPMEGA_TE_MXFP8_BWD_BACKEND"] = backend
    return os.environ["CPPMEGA_TE_MXFP8_BWD_BACKEND"], linear_kernel_contract


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

    backend, linear_kernel_contract = _set_mxfp8_profile_env(
        args.backend,
        args.linear_kernel_contract,
    )
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
    direct_no_sidecar = (
        linear_kernel_contract == "compact_direct_v1"
        and backend == "cutlass_native"
        and transpose_emit_backend == "off"
    )
    flashinfer_compact_direct = (
        backend == "flashinfer_cutlass"
        and int(stats.get("mxfp8_flashinfer_dgrad", 0)) > 0
        and int(stats.get("mxfp8_flashinfer_wgrad", 0)) > 0
        and int(stats.get("mxfp8_tn_adapter_copy_transpose", 0)) == 0
    )
    if direct_no_sidecar or flashinfer_compact_direct:
        if direct_no_sidecar:
            if int(stats.get("mxfp8_cutlass_native_dgrad", 0)) <= 0:
                failures.append("CUTLASS native backend did not handle dgrad")
            if int(stats.get("mxfp8_cutlass_native_wgrad", 0)) <= 0:
                failures.append("CUTLASS native backend did not handle wgrad")
        else:
            if int(stats.get("mxfp8_flashinfer_dgrad", 0)) <= 0:
                failures.append("FlashInfer/CUTLASS compact-direct backend did not handle dgrad")
            if int(stats.get("mxfp8_flashinfer_wgrad", 0)) <= 0:
                failures.append("FlashInfer/CUTLASS compact-direct backend did not handle wgrad")
        if int(stats.get("mxfp8_tn_adapter_te_emit", 0)) != 0:
            failures.append("compact-direct backend emitted TE transpose operands")
        if int(stats.get("mxfp8_tn_sidecar_attr_attached", 0)) != 0:
            failures.append("compact-direct backend attached MXFP8 transpose sidecars")
        if int(stats.get("mxfp8_tn_sidecar_registry_peak", 0)) != 0:
            failures.append("compact-direct backend used the sidecar registry")
        for key in (
            "bf16_fallback_dgrad",
            "bf16_fallback_wgrad",
            "mxfp8_tn_adapter_copy_transpose",
            "mxfp8_tn_adapter_missing_sidecar_copy",
            "mxfp8_tn_adapter_saved_transpose_operand",
            "mxfp8_tn_adapter_te_emit_deferred",
            "mxfp8_tn_sidecar_attr_attached",
            "mxfp8_tn_sidecar_registry_peak_bytes",
        ):
            if int(stats.get(key, 0)) != 0:
                failures.append(f"{key}={stats.get(key)}; expected 0 for compact-direct lane")
    else:
        if not saved_transpose_payload:
            failures.append(
                "MXFP8 rowwise-transposed payload was not saved for Linear backward"
            )
        saved_operand_count = int(
            stats.get("mxfp8_tn_adapter_saved_transpose_operand", 0)
        ) + int(stats.get("mxfp8_tn_adapter_direct_saved_operand", 0))
        if saved_operand_count <= 0:
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
        "linear_kernel_contract": linear_kernel_contract,
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
    parser.add_argument(
        "--linear-kernel-contract",
        choices=(
            "legacy",
            "gemm_ready_v1",
            "gemm_ready_v1_dense_only",
            "compact_direct_v1",
        ),
        default="gemm_ready_v1",
        help="Typed MXFP8 Linear saved-operand contract rendered by run profiles.",
    )
    args = parser.parse_args()

    report = _run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
