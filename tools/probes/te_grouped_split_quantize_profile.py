#!/usr/bin/env python3
"""Profile grouped MXFP8 split_quantize producer modes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Literal

import torch


LinearContract = Literal["legacy", "gemm_ready_v1", "gemm_ready_v1_dense_only"]


@dataclass(frozen=True)
class GroupedSplitQuantizeProfile:
    contract: LinearContract
    groups: int = 16
    m_per_group: int = 256
    k: int = 3584
    warmup: int = 5
    iters: int = 10
    seed: int = 1234
    transformer_engine_source: str | None = (
        "/home/dave/TransformerEngine-wave27A-te-dual-output-quantize"
    )
    profile_dir: str | None = None
    disable_bulk_allocation: bool = False
    enable_cast_only_fast_path: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cppmega_fp8_shim(repo_root: Path) -> Any:
    sys.path.insert(0, str(repo_root))
    shim_path = repo_root / "scripts" / "cppmega_fp8_shim.py"
    spec = importlib.util.spec_from_file_location(
        "cppmega_fp8_shim_grouped_split_quantize_profile",
        shim_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shim from {shim_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_profile_env(profile: GroupedSplitQuantizeProfile) -> None:
    os.environ.setdefault("CPPMEGA_ALLOW_TE_MXFP8_SM12", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_BACKEND", "te_tn_adapter")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND", "te")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_DENSE_SAVED_OPERANDS", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_GROUPED_GEMM_READY_BACKWARD", "1")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_GROUPED_DIRECT_BACKWARD", "0")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK", "0")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_DGRAD_BF16", "0")
    os.environ.setdefault("CPPMEGA_TE_MXFP8_WGRAD_BF16", "0")
    os.environ["CPPMEGA_TE_MXFP8_LINEAR_KERNEL_CONTRACT"] = profile.contract
    os.environ["ENABLE_CAST_ONLY"] = "1" if profile.enable_cast_only_fast_path else "0"


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return 0


def _mxfp8_payload_bytes(value: Any) -> int:
    total = 0
    for attr in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        total += _tensor_nbytes(getattr(value, attr, None))
    return total


def _attached_sidecar(value: Any) -> Any | None:
    for attr in (
        "_te_gemm_ready_rowwise_transpose_for_backward",
        "_cppmega_mxfp8_gemm_ready_rowwise_transpose",
        "_te_rowwise_transpose_for_backward",
        "_cppmega_mxfp8_rowwise_transpose",
    ):
        sidecar = getattr(value, attr, None)
        if sidecar is not None:
            return sidecar
    return None


def _summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
    output_payload_bytes = sum(_mxfp8_payload_bytes(out) for out in outputs)
    sidecars = [_attached_sidecar(out) for out in outputs]
    sidecars = [sidecar for sidecar in sidecars if sidecar is not None]
    return {
        "outputs": len(outputs),
        "output_payload_bytes": output_payload_bytes,
        "output_payload_mib": output_payload_bytes / (1024 * 1024),
        "sidecars": len(sidecars),
        "sidecar_payload_bytes": sum(_mxfp8_payload_bytes(sidecar) for sidecar in sidecars),
        "sidecar_payload_mib": sum(_mxfp8_payload_bytes(sidecar) for sidecar in sidecars)
        / (1024 * 1024),
        "sidecar_swizzled": sum(
            1 for sidecar in sidecars if bool(getattr(sidecar, "_with_gemm_swizzled_scales", False))
        ),
    }


def _profiler_rows(prof: torch.profiler.profile) -> list[dict[str, Any]]:
    rows = []
    interesting = (
        "mxfp8",
        "quantize",
        "split",
        "cudaLaunchKernel",
        "cudaMalloc",
        "Memcpy",
        "memcpy",
    )
    for event in prof.key_averages():
        key = event.key
        if not any(token in key for token in interesting):
            continue
        self_cuda_us = float(
            getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0))
        )
        cuda_total_us = float(
            getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0))
        )
        rows.append(
            {
                "name": key,
                "calls": int(event.count),
                "self_cuda_us": self_cuda_us,
                "cuda_total_us": cuda_total_us,
                "self_cpu_us": float(event.self_cpu_time_total),
                "cpu_total_us": float(event.cpu_time_total),
                "cuda_memory_bytes": int(
                    getattr(event, "cuda_memory_usage", getattr(event, "device_memory_usage", 0))
                ),
                "self_cuda_memory_bytes": int(
                    getattr(
                        event,
                        "self_cuda_memory_usage",
                        getattr(event, "self_device_memory_usage", 0),
                    )
                ),
            }
        )
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    return rows


def _producer_summary(rows: list[dict[str, Any]], *, outputs: int, iters: int) -> dict[str, Any]:
    producer_rows = [
        row
        for row in rows
        if "mxfp8" in row["name"] or "quantize" in row["name"]
    ]
    producer_calls = sum(row["calls"] for row in producer_rows)
    producer_cuda_us = sum(row["self_cuda_us"] for row in producer_rows)
    total_outputs = outputs * iters
    return {
        "producer_kernel_rows": len(producer_rows),
        "producer_calls": producer_calls,
        "producer_cuda_us": producer_cuda_us,
        "producer_ms_per_iter": producer_cuda_us / max(iters, 1) / 1000.0,
        "producer_us_per_output": producer_cuda_us / max(total_outputs, 1),
        "producer_calls_per_iter": producer_calls / max(iters, 1),
        "producer_calls_per_output": producer_calls / max(total_outputs, 1),
    }


def _run(profile: GroupedSplitQuantizeProfile) -> dict[str, Any]:
    if profile.groups <= 0:
        raise SystemExit("--groups must be positive")
    if profile.m_per_group <= 0:
        raise SystemExit("--m-per-group must be positive")
    if profile.k <= 0:
        raise SystemExit("--k must be positive")
    if profile.m_per_group % 32 or profile.k % 32:
        raise SystemExit("--m-per-group and --k must be multiples of 32")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    if profile.transformer_engine_source:
        sys.path.insert(0, profile.transformer_engine_source)
    _set_profile_env(profile)
    shim_module = _load_cppmega_fp8_shim(_repo_root())

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    torch.manual_seed(profile.seed)
    total_m = profile.groups * profile.m_per_group
    source = torch.randn(total_m, profile.k, device="cuda", dtype=torch.bfloat16)
    split_sections = [profile.m_per_group] * profile.groups
    quantizers = [
        MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        for _ in range(profile.groups)
    ]
    for quantizer in quantizers:
        quantizer.internal = True
        quantizer.optimize_for_gemm = False
        quantizer.set_usage(rowwise=True, columnwise=True)

    def split_once() -> list[Any]:
        return tex.split_quantize(
            source,
            split_sections,
            quantizers,
            disable_bulk_allocation=profile.disable_bulk_allocation,
        )

    for _ in range(profile.warmup):
        outputs = split_once()
        del outputs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        for _ in range(profile.iters):
            outputs = split_once()
            torch.cuda.synchronize()
            del outputs

    torch.cuda.synchronize()
    retained_outputs = split_once()
    torch.cuda.synchronize()

    if profile.profile_dir:
        profile_dir = Path(profile.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(profile_dir / f"{profile.contract}_trace.json"))
        (profile_dir / f"{profile.contract}_cuda_table.txt").write_text(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=80,
            ),
            encoding="utf-8",
        )

    stats = (
        shim_module.cppmega_te_mxfp8_bwd_stats_snapshot()
        if hasattr(shim_module, "cppmega_te_mxfp8_bwd_stats_snapshot")
        else {}
    )
    profiler_rows = _profiler_rows(prof)
    output_summary = _summarize_outputs(retained_outputs)
    return {
        "profile": asdict(profile),
        "shape": {
            "groups": profile.groups,
            "m_total": total_m,
            "m_per_group": profile.m_per_group,
            "k": profile.k,
        },
        "allocator": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "max_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "max_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        },
        "outputs": output_summary,
        "producer_summary": _producer_summary(
            profiler_rows,
            outputs=output_summary["outputs"],
            iters=profile.iters,
        ),
        "shim_stats": stats,
        "profiler_rows": profiler_rows,
    }


def _parse_args() -> GroupedSplitQuantizeProfile:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        choices=("legacy", "gemm_ready_v1", "gemm_ready_v1_dense_only"),
        required=True,
    )
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--m-per-group", type=int, default=256)
    parser.add_argument("--k", type=int, default=3584)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--transformer-engine-source",
        default="/home/dave/TransformerEngine-wave27A-te-dual-output-quantize",
    )
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument(
        "--disable-bulk-allocation",
        action="store_true",
        help="Forward disable_bulk_allocation=True to tex.split_quantize.",
    )
    parser.add_argument(
        "--enable-cast-only-fast-path",
        action="store_true",
        help="Enable TE's specialized MXFP8 cast-only quantize kernels.",
    )
    args = parser.parse_args()
    return GroupedSplitQuantizeProfile(
        contract=args.contract,
        groups=args.groups,
        m_per_group=args.m_per_group,
        k=args.k,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        transformer_engine_source=args.transformer_engine_source,
        profile_dir=args.profile_dir,
        disable_bulk_allocation=args.disable_bulk_allocation,
        enable_cast_only_fast_path=args.enable_cast_only_fast_path,
    )


def main() -> None:
    report = _run(_parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
