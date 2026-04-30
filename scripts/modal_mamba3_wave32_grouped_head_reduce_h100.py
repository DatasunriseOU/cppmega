"""Modal H100 benchmark for Mamba3 grouped-head backward reduction.

This is a component-only probe. It measures the tail reduction currently done
after TileLang returns expanded ``dq`` / ``dk`` tensors:

    [B, S, R, H, N] -> [B, S, R, G, N]

No H200, no GCloud, and no production source mutation.
"""
# ruff: noqa: E402

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import time
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).parent.parent

APP_NAME = "cppmega-wave32-grouped-head-reduce"
RESULTS_VOL = "cppmega-mamba3-benchmarks"
BENCH_DIR = "/benchmarks/mamba3_wave32_grouped_head_reduce_h100"
GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"


def _image() -> modal.Image:
    return (
        modal.Image.from_registry(
            GHCR_REF,
            secret=modal.Secret.from_name("ghcr-pull"),
            add_python=None,
        )
        .env({"PYTHONPATH": "/opt/cppmega:/opt/megatron-lm"})
        .add_local_dir(str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega")
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )


app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
image = _image()


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


@app.function(
    image=image,
    gpu="H100",
    timeout=2400,
    volumes={"/vol": results_vol},
)
def run_bench(
    run_id: str,
    warmup: int = 20,
    iters: int = 100,
    include_full: bool = True,
) -> dict[str, Any]:
    import torch

    from cppmega.megatron.mamba3_grouped_head_reduce import (
        reduce_grouped_heads_torch,
        reduce_grouped_heads_triton,
    )

    out_dir = pathlib.Path("/vol") / BENCH_DIR.lstrip("/") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.txt"

    def mark(step: str) -> None:
        with progress_file.open("a") as fh:
            fh.write(f"{time.time():.3f} {step}\n")
        results_vol.commit()

    def make_inputs(shape: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(20260430 + shape["seed"])
        device = torch.device("cuda")
        dtype = torch.bfloat16
        dq = torch.randn(
            shape["B"],
            shape["S"],
            shape["R"],
            shape["H"],
            shape["N"],
            device=device,
            dtype=dtype,
        )
        dk = torch.randn_like(dq)
        return dq.contiguous(), dk.contiguous()

    def tensor_mib(t: torch.Tensor) -> float:
        return t.numel() * t.element_size() / (1024 * 1024)

    def summarize_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
        diff = (a.float() - b.float()).abs()
        return {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }

    def bench_one(
        label: str,
        fn,
        dq: torch.Tensor,
        dk: torch.Tensor,
        groups: int,
    ) -> dict[str, Any]:
        for _ in range(warmup):
            out = fn(dq, dk, groups)
        torch.cuda.synchronize()
        del out
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        out = None
        for _ in range(iters):
            out = fn(dq, dk, groups)
        end.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter()
        assert out is not None
        after_alloc = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        out_bytes = out[0].numel() * out[0].element_size() + out[1].numel() * out[1].element_size()
        input_bytes = dq.numel() * dq.element_size() + dk.numel() * dk.element_size()
        elapsed_ms = start.elapsed_time(end) / iters
        logical_gib = (input_bytes + out_bytes) / (1024**3)
        return {
            "label": label,
            "cuda_ms": elapsed_ms,
            "wall_ms": (wall_end - wall_start) * 1000.0 / iters,
            "logical_gib_per_iter": logical_gib,
            "effective_gib_s": logical_gib / (elapsed_ms / 1000.0),
            "peak_alloc_mib_delta": (peak_alloc - before_alloc) / (1024 * 1024),
            "peak_reserved_mib_delta": (peak_reserved - before_reserved) / (1024 * 1024),
            "end_alloc_mib_delta": (after_alloc - before_alloc) / (1024 * 1024),
            "end_reserved_mib_delta": (after_reserved - before_reserved) / (1024 * 1024),
            "out0_mib": out[0].numel() * out[0].element_size() / (1024 * 1024),
            "out1_mib": out[1].numel() * out[1].element_size() / (1024 * 1024),
        }

    shapes = [
        {"name": "smoke_hpg2", "B": 1, "S": 512, "R": 4, "H": 16, "G": 8, "N": 64, "seed": 1},
        {"name": "half_seq_hpg16", "B": 2, "S": 2048, "R": 4, "H": 128, "G": 8, "N": 64, "seed": 2},
    ]
    if include_full:
        shapes.append(
            {"name": "fullish_seq4096_hpg16", "B": 2, "S": 4096, "R": 4, "H": 128, "G": 8, "N": 64, "seed": 3}
        )

    report: dict[str, Any] = {
        "run_id": run_id,
        "utc": _utc_stamp(),
        "gpu_name": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup": warmup,
        "iters": iters,
        "shapes": [],
    }

    mark("start")
    for shape in shapes:
        mark(f"shape_start {shape['name']}")
        dq, dk = make_inputs(shape)
        groups = shape["G"]
        ref_dq, ref_dk = reduce_grouped_heads_torch(dq, dk, groups)
        tri_dq, tri_dk = reduce_grouped_heads_triton(dq, dk, groups)
        torch.cuda.synchronize()
        correctness = {
            "dq": summarize_diff(ref_dq, tri_dq),
            "dk": summarize_diff(ref_dk, tri_dk),
        }
        del ref_dq, ref_dk, tri_dq, tri_dk
        torch.cuda.empty_cache()
        torch_result = bench_one("torch_view_sum_pair", reduce_grouped_heads_torch, dq, dk, groups)
        triton_result = bench_one("triton_fused_pair", reduce_grouped_heads_triton, dq, dk, groups)
        input_mib = tensor_mib(dq) + tensor_mib(dk)
        output_mib = (
            shape["B"] * shape["S"] * shape["R"] * shape["G"] * shape["N"] * dq.element_size() * 2
        ) / (1024 * 1024)
        report["shapes"].append(
            {
                "shape": shape,
                "heads_per_group": shape["H"] // shape["G"],
                "input_mib_pair": input_mib,
                "output_mib_pair": output_mib,
                "correctness_vs_torch": correctness,
                "bench": {
                    "torch": torch_result,
                    "triton": triton_result,
                    "triton_minus_torch_ms": triton_result["cuda_ms"] - torch_result["cuda_ms"],
                    "triton_speedup_vs_torch": torch_result["cuda_ms"] / triton_result["cuda_ms"],
                    "triton_peak_alloc_mib_delta_minus_torch": (
                        triton_result["peak_alloc_mib_delta"] - torch_result["peak_alloc_mib_delta"]
                    ),
                },
            }
        )
        del dq, dk
        torch.cuda.empty_cache()
        mark(f"shape_done {shape['name']}")

    report_file = out_dir / "report.json"
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / "summary.md").write_text(_summary_markdown(report))
    results_vol.commit()
    mark("done")
    return report


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wave32 Grouped-Head Reduce H100",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- gpu: `{report['gpu_name']}`",
        f"- warmup/iters: `{report['warmup']}/{report['iters']}`",
        "",
        "| shape | torch ms | triton ms | speedup | peak torch MiB | peak triton MiB | max_abs dq/dk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["shapes"]:
        bench = item["bench"]
        corr = item["correctness_vs_torch"]
        lines.append(
            "| {name} | {torch_ms:.6f} | {triton_ms:.6f} | {speedup:.3f} | "
            "{torch_peak:.2f} | {triton_peak:.2f} | {dq_abs:.6g}/{dk_abs:.6g} |".format(
                name=item["shape"]["name"],
                torch_ms=bench["torch"]["cuda_ms"],
                triton_ms=bench["triton"]["cuda_ms"],
                speedup=bench["triton_speedup_vs_torch"],
                torch_peak=bench["torch"]["peak_alloc_mib_delta"],
                triton_peak=bench["triton"]["peak_alloc_mib_delta"],
                dq_abs=corr["dq"]["max_abs"],
                dk_abs=corr["dk"]["max_abs"],
            )
        )
    lines.append("")
    return "\n".join(lines)


@app.local_entrypoint()
def main(
    run_id: str = "wave32_grouped_head_reduce_h100",
    warmup: int = 20,
    iters: int = 100,
    include_full: bool = True,
) -> None:
    result = run_bench.remote(run_id, warmup, iters, include_full)
    print(json.dumps(result, indent=2, sort_keys=True))
