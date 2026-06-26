"""Modal H200 microbench for Mamba3 bwd_bwd owner/layout rewrites.

This is not a full Mamba3 replacement kernel.  It isolates the bwd_bwd
cross-P reduction shape that makes P-tiled ownership hard:

    DK-like: [F, P] @ [P, N] -> [F, N]
    DQ-like: [F, P] @ [P, N] -> [F, N]

The candidate decomposition gives each program ownership of
``(B, H, chunk, p_tile)`` and reduces directly into the final fp32 output with
``tl.atomic_add``.  That avoids a large fp32 partial-output handoff and keeps
the per-program live-set at ``P_TILE`` instead of full ``P``.

Run:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave1.py \
        --shape-csv smoke,productionish --p-tile-csv 64 --warmup 2 --iters 8
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-bwd-bwd-owner-rewrite-wave1"


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

    @property
    def chunks(self) -> int:
        return (self.S + self.chunk - 1) // self.chunk

    @property
    def fused_chunk(self) -> int:
        return self.chunk * self.R

    @property
    def owners(self) -> int:
        return self.B * self.H * self.chunks


SHAPES: dict[str, Shape] = {
    "tiny": Shape("tiny", B=1, S=64, H=2, G=1, N=64, P=64, R=4),
    "smoke": Shape("smoke", B=1, S=256, H=4, G=1, N=64, P=64, R=4),
    "smoke_p128": Shape("smoke_p128", B=1, S=256, H=4, G=1, N=64, P=128, R=4),
    "representative": Shape("representative", B=2, S=1024, H=8, G=1, N=64, P=128, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    return img.env({"CPPMEGA_IMAGE_REF": GHCR_REF})


app = modal.App(APP_NAME)


def _define_kernels() -> tuple[Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _fullp_dqdk_kernel(
        psi,
        dphi,
        state,
        dstate,
        dk_out,
        dq_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :], mask=p_mask[None, :], other=0.0)
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :], mask=p_mask[None, :], other=0.0)
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P, mask=p_mask[:, None], other=0.0)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P, mask=p_mask[:, None], other=0.0)

        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)

    @triton.jit
    def _ptile_atomic_dqdk_kernel(
        psi,
        dphi,
        state,
        dstate,
        dk_out,
        dq_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        P_TILE: tl.constexpr,
    ):
        owner = tl.program_id(0)
        p_block = tl.program_id(1)
        p_start = p_block * P_TILE
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = p_start + tl.arange(0, P_TILE)
        p_mask = offs_p < P

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :], mask=p_mask[None, :], other=0.0)
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :], mask=p_mask[None, :], other=0.0)
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P, mask=p_mask[:, None], other=0.0)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P, mask=p_mask[:, None], other=0.0)

        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)

        out_base = owner * F * N
        tl.atomic_add(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk, sem="relaxed")
        tl.atomic_add(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq, sem="relaxed")

    return _fullp_dqdk_kernel, _ptile_atomic_dqdk_kernel


def _device_report(requested_gpu: str) -> dict[str, Any]:
    import torch
    import triton

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "requested_gpu_spec": requested_gpu,
        "image_ref": os.environ.get("CPPMEGA_IMAGE_REF", GHCR_REF),
    }


def _time_cuda_events(fn: Any, *, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples) if samples else None,
        "min_ms": min(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
        "samples_ms": samples,
    }


def _make_inputs(shape: Shape) -> dict[str, Any]:
    import torch

    torch.manual_seed(20260429)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = 0.01
    return {
        "psi": torch.randn(shape.owners, shape.fused_chunk, shape.P, device=device, dtype=dtype) * scale,
        "dphi": torch.randn(shape.owners, shape.fused_chunk, shape.P, device=device, dtype=dtype) * scale,
        "state": torch.randn(shape.owners, shape.N, shape.P, device=device, dtype=dtype) * scale,
        "dstate": torch.randn(shape.owners, shape.N, shape.P, device=device, dtype=dtype) * scale,
    }


def _scratch_model(shape: Shape, p_tile: int) -> dict[str, Any]:
    n_p_tiles = (shape.P + p_tile - 1) // p_tile
    f = shape.fused_chunk
    fp32 = 4
    bf16 = 2
    dqdk_partial_bytes = shape.owners * n_p_tiles * 2 * f * shape.N * fp32
    dqdk_final_bytes = shape.owners * 2 * f * shape.N * fp32
    dqdk_live_input_bytes_fullp = (2 * f * shape.P + 2 * shape.N * shape.P) * bf16
    dqdk_live_input_bytes_ptile = (2 * f * p_tile + 2 * shape.N * p_tile) * bf16
    dstates_before_chunks_fp32_bytes = shape.B * shape.H * shape.chunks * shape.N * shape.P * fp32
    return {
        "n_p_tiles": n_p_tiles,
        "dqdk_partial_fp32_handoff_bytes_if_not_atomic": dqdk_partial_bytes,
        "dqdk_final_fp32_bytes": dqdk_final_bytes,
        "dstates_before_chunks_fp32_bytes_discarded_path": dstates_before_chunks_fp32_bytes,
        "atomic_extra_handoff_bytes": 0,
        "per_program_live_input_bytes_fullp": dqdk_live_input_bytes_fullp,
        "per_program_live_input_bytes_ptile": dqdk_live_input_bytes_ptile,
        "per_program_live_input_reduction_ratio": (
            dqdk_live_input_bytes_fullp / dqdk_live_input_bytes_ptile
            if dqdk_live_input_bytes_ptile
            else None
        ),
    }


def _run_shape(shape: Shape, p_tile: int, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    import torch

    fullp_kernel, ptile_kernel = _define_kernels()
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "p_tile": p_tile,
        "memory_model": _scratch_model(shape, p_tile),
    }
    if shape.fused_chunk != 64 or shape.N != 64:
        result.update({"status": "skipped", "reason": "kernel prototype specializes F=N=64"})
        return result
    if p_tile > shape.P:
        result.update({"status": "skipped", "reason": "p_tile > P"})
        return result
    if p_tile not in (16, 32, 64, 128):
        result.update({"status": "skipped", "reason": "p_tile must be one of 16,32,64,128 for this prototype"})
        return result

    try:
        tensors = _make_inputs(shape)
        dk_ref = torch.empty(shape.owners, shape.fused_chunk, shape.N, device="cuda", dtype=torch.float32)
        dq_ref = torch.empty_like(dk_ref)
        dk_atomic = torch.empty_like(dk_ref)
        dq_atomic = torch.empty_like(dk_ref)
        grid_full = (shape.owners,)
        grid_ptile = (shape.owners, (shape.P + p_tile - 1) // p_tile)

        def run_fullp() -> None:
            fullp_kernel[grid_full](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["dstate"],
                dk_ref,
                dq_ref,
                shape.fused_chunk,
                shape.N,
                shape.P,
                triton_next_power_of_2(shape.P),
                num_warps=8,
            )

        def run_ptile_nozero() -> None:
            ptile_kernel[grid_ptile](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["dstate"],
                dk_atomic,
                dq_atomic,
                shape.fused_chunk,
                shape.N,
                shape.P,
                p_tile,
                num_warps=4,
            )

        def run_ptile_with_zero() -> None:
            dk_atomic.zero_()
            dq_atomic.zero_()
            run_ptile_nozero()

        run_fullp()
        run_ptile_with_zero()
        torch.cuda.synchronize()

        dk_diff = (dk_ref - dk_atomic).abs()
        dq_diff = (dq_ref - dq_atomic).abs()
        result.update(
            {
                "status": "ok",
                "correctness": {
                    "dk_max_abs": float(dk_diff.max().item()),
                    "dq_max_abs": float(dq_diff.max().item()),
                    "dk_allclose_1e_2": bool(torch.allclose(dk_ref, dk_atomic, rtol=1.0e-2, atol=1.0e-2)),
                    "dq_allclose_1e_2": bool(torch.allclose(dq_ref, dq_atomic, rtol=1.0e-2, atol=1.0e-2)),
                },
                "elapsed": {
                    "fullp_ms": _time_cuda_events(run_fullp, warmup=warmup, iters=iters),
                    "ptile_atomic_compute_only_ms": _time_cuda_events(run_ptile_nozero, warmup=warmup, iters=iters),
                    "ptile_atomic_with_zero_ms": _time_cuda_events(run_ptile_with_zero, warmup=warmup, iters=iters),
                },
            }
        )
        full_ms = result["elapsed"]["fullp_ms"]["mean_ms"]
        atomic_ms = result["elapsed"]["ptile_atomic_with_zero_ms"]["mean_ms"]
        if full_ms and atomic_ms:
            result["speed"] = {"ptile_atomic_with_zero_vs_fullp": full_ms / atomic_ms}
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-6000:],
            }
        )
    return result


def triton_next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _selected_shapes(shape_csv: str) -> list[Shape]:
    selected: list[Shape] = []
    for raw in shape_csv.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in SHAPES:
            raise ValueError(f"unknown shape {name!r}; choose one of {sorted(SHAPES)}")
        selected.append(SHAPES[name])
    if not selected:
        raise ValueError("at least one shape required")
    return selected


def _selected_p_tiles(p_tile_csv: str) -> list[int]:
    selected = [int(raw.strip()) for raw in p_tile_csv.split(",") if raw.strip()]
    if not selected:
        raise ValueError("at least one p_tile required")
    return selected


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800)
def run_probe(requested_gpu: str, shape_csv: str, p_tile_csv: str, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    try:
        device = _device_report(requested_gpu)
        results = []
        for shape in _selected_shapes(shape_csv):
            for p_tile in _selected_p_tiles(p_tile_csv):
                print(f"[owner-wave1] shape={shape.name} p_tile={p_tile}", flush=True)
                results.append(_run_shape(shape, p_tile, warmup, iters))
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "p_tile_csv": p_tile_csv,
                "warmup": warmup,
                "iters": iters,
            },
            "results": results,
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "app_name": APP_NAME,
            "top_level_status": "crashed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
        }


@app.local_entrypoint()
def main(shape_csv: str = "smoke_p128", p_tile_csv: str = "64", warmup: int = 1, iters: int = 4) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, p_tile_csv, warmup, iters)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
