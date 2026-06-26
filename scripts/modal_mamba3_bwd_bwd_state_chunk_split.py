"""Modal probe for the Mamba3 bwd_bwd state/chunk split prototype.

This is intentionally not a production patch. It validates the correctness
handoff used by a feasible split:

  pass1: compute fp32 dstates_before_chunk [B, H, nchunks, N, P]
  pass2: consume that tensor and compute the normal stitched bwd_bwd outputs

The H200 timing in this harness measures the unavoidable gmem round-trip for the
handoff tensor. That is a lower bound: a TileLang pass1 would also recompute
Q/dPhiO and the dstates recurrence.

Run:

    GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200:2 timeout 600s \
      modal run scripts/modal_mamba3_bwd_bwd_state_chunk_split.py \
      --shape-csv smoke,productionish --warmup 5 --iters 20
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")
BENCH_VOLUME_NAME = os.environ.get("CPPMEGA_MODAL_BENCH_VOLUME", "cppmega-mamba3-benchmarks")

APP_NAME = "cppmega-mamba3-bwd-bwd-state-chunk-split"
CPPMEGA_ROOT = "/opt/cppmega"
BENCH_ROOT = "/benchmarks"
BENCH_PREFIX = "mamba3_bwd_bwd_state_chunk_split"

bench_volume = modal.Volume.from_name(BENCH_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class Shape:
    name: str
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int
    chunk: int = 16
    rotary_dim_divisor: int = 4


SHAPES: dict[str, Shape] = {
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    img = img.env({"CPPMEGA_IMAGE_REF": GHCR_REF})
    img = img.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    return img


app = modal.App(APP_NAME)


def _install_source_paths() -> None:
    import sys

    if CPPMEGA_ROOT not in sys.path:
        sys.path.insert(0, CPPMEGA_ROOT)


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _selected_shapes(shape_csv: str) -> list[Shape]:
    shapes = []
    for name in [part.strip() for part in shape_csv.split(",")]:
        if not name:
            continue
        if name not in SHAPES:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(SHAPES)}")
        shapes.append(SHAPES[name])
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean_ms": mean,
        "min_ms": ordered[0],
        "p50_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
        "samples_ms": values,
    }


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _dstates_bytes(shape: Shape) -> int:
    nchunks = math.ceil(shape.S / shape.chunk)
    return shape.B * shape.H * nchunks * shape.N * shape.P * 4


def _bandwidth_probe(shape: Shape, *, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    nchunks = math.ceil(shape.S / shape.chunk)
    src = torch.empty(shape.B, shape.H, nchunks, shape.N, shape.P, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    def copy_roundtrip() -> None:
        dst.copy_(src)

    stats = _stats(_time_cuda_events(copy_roundtrip, warmup=warmup, iters=iters))
    tensor_bytes = _dstates_bytes(shape)
    traffic_bytes = tensor_bytes * 2
    stats["dstates_tensor_bytes"] = tensor_bytes
    stats["roundtrip_traffic_bytes"] = traffic_bytes
    stats["mean_effective_gib_s"] = (traffic_bytes / (1024**3)) / (stats["mean_ms"] / 1000.0)
    return stats


def _correctness_smoke() -> dict[str, Any]:
    import torch

    from cppmega.megatron.cute_dsl_mimo.full_bwd_bwd_epilogue import (
        compute_dstates_before_chunks_pytorch,
        full_bwd_bwd_pytorch,
        full_bwd_bwd_pytorch_state_chunk_split,
    )

    torch.manual_seed(20260429)
    B, S, H, G, N, P, R = 1, 32, 2, 1, 8, 8, 2
    chunk = 16
    rdim = N // 4
    nchunks = S // chunk
    dtype = torch.float32
    device = "cuda"
    inputs = {
        "dout": torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01,
        "q_raw": torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01,
        "k_raw": torch.randn(B, S, R, G, N, device=device, dtype=dtype) * 0.01,
        "v": torch.randn(B, S, H, P, device=device, dtype=dtype) * 0.01,
        "q_bias": torch.randn(H, R, N, device=device, dtype=dtype) * 0.01,
        "k_bias": torch.randn(H, R, N, device=device, dtype=dtype) * 0.01,
        "mimo_v": torch.randn(H, R, P, device=device, dtype=dtype) * 0.01,
        "mimo_o": torch.randn(H, R, P, device=device, dtype=dtype) * 0.01,
        "angles": torch.randn(B, S, H, rdim, device=device, dtype=dtype) * 0.01,
        "dA_cs": -torch.rand(B, H, S, device=device, dtype=dtype) * 0.01,
        "dA_cs_rev": -torch.rand(B, H, S, device=device, dtype=dtype) * 0.01,
        "dt": torch.randn(B, H, S, device=device, dtype=dtype) * 0.01,
        "trap": torch.randn(B, H, S, device=device, dtype=dtype) * 0.01,
        "D": torch.randn(H, device=device, dtype=dtype) * 0.01,
        "segsum": torch.randn(B, H, nchunks, chunk, chunk, device=device, dtype=dtype) * 0.01,
        "states": torch.randn(B, H, nchunks, N, P, device=device, dtype=dtype) * 0.01,
        "qk_dot": torch.randn(B, H, S, R, R, device=device, dtype=dtype) * 0.01,
        "chunk_size": chunk,
        "R": R,
        "rotary_dim_divisor": 4,
    }
    monolithic = full_bwd_bwd_pytorch(**inputs)
    split = full_bwd_bwd_pytorch_state_chunk_split(**inputs)
    pass1 = compute_dstates_before_chunks_pytorch(
        inputs["dout"],
        inputs["q_raw"],
        inputs["q_bias"],
        inputs["mimo_o"],
        inputs["angles"],
        inputs["dA_cs"],
        chunk_size=chunk,
        rotary_dim_divisor=4,
    )
    captured = full_bwd_bwd_pytorch(**inputs, return_dstates_before=True)["DSTATES_BEFORE_CHUNKS"]

    diffs = {}
    for name, expected in monolithic.items():
        actual = split[name]
        diffs[name] = float((expected - actual).abs().max().item())
    diffs["DSTATES_BEFORE_CHUNKS"] = float((pass1 - captured).abs().max().item())
    return {"status": "ok", "max_abs_by_output": diffs, "max_abs": max(diffs.values())}


@app.function(image=_image(), gpu=GPU_SPEC, timeout=600, volumes={BENCH_ROOT: bench_volume})
def run_probe(requested_gpu: str, run_id: str | None, shape_csv: str, warmup: int, iters: int) -> dict[str, Any]:
    import time

    _install_source_paths()
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_rel = f"{BENCH_PREFIX}/{run_id}"
    run_dir = os.path.join(BENCH_ROOT, run_rel)
    os.makedirs(run_dir, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "volume": BENCH_VOLUME_NAME,
        "volume_relpath": f"/{run_rel}",
        "device": _device_report(requested_gpu),
        "settings": {"shape_csv": shape_csv, "warmup": warmup, "iters": iters},
        "correctness_smoke": _correctness_smoke(),
        "bandwidth": [],
    }
    for shape in _selected_shapes(shape_csv):
        report["bandwidth"].append({"shape": asdict(shape), "dstates_roundtrip": _bandwidth_probe(shape, warmup=warmup, iters=iters)})

    artifact = os.path.join(run_dir, "summary.json")
    with open(artifact, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    report["artifact"] = artifact
    bench_volume.commit()
    return report


@app.local_entrypoint()
def main(run_id: str | None = None, shape_csv: str = "smoke,productionish", warmup: int = 5, iters: int = 20) -> None:
    result = run_probe.remote(GPU_SPEC, run_id, shape_csv, warmup, iters)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
