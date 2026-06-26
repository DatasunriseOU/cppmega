"""Modal H200 microbench for Mamba3 bwd_bwd owner/layout rewrite wave 2.

This isolates the P=128 DQ/DK/diag-like cross-P reductions and compares:

* fullp: one owner program computes full P=128 reductions.
* ptile_atomic: wave1-style two P_TILE=64 owner programs with atomic adds.
* serial2: one owner program runs two serial P_TILE=64 passes and keeps the
  DQ/DK/diag accumulators on-chip before final stores.

Run:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_owner_rewrite_wave2.py \
        --shape-csv smoke_p128,productionish --warmup 2 --iters 8
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

APP_NAME = "cppmega-mamba3-bwd-bwd-owner-rewrite-wave2"
P_TILE = 64


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


def _define_kernels() -> tuple[Any, Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _fullp_kernel(
        psi,
        dphi,
        state,
        dstate,
        qk_dot,
        dk_out,
        dq_out,
        diag_qk_out,
        diag_intra_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, BLOCK_P)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)

        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)
        dqk = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)
        dki = tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * dqk, axis=1)
        diag_intra = tl.sum(qk * dki, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)
        tl.store(diag_qk_out + owner * F + offs_f, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_f, diag_intra)

    @triton.jit
    def _ptile_atomic_kernel(
        psi,
        dphi,
        state,
        dstate,
        qk_dot,
        dk_out,
        dq_out,
        diag_qk_out,
        diag_intra_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        P_TILE_C: tl.constexpr,
    ):
        owner = tl.program_id(0)
        p_block = tl.program_id(1)
        p_start = p_block * P_TILE_C
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = p_start + tl.arange(0, P_TILE_C)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)

        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)
        dqk = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)
        dki = tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * dqk, axis=1)
        diag_intra = tl.sum(qk * dki, axis=1)

        out_base = owner * F * N
        tl.atomic_add(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk, sem="relaxed")
        tl.atomic_add(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq, sem="relaxed")
        tl.atomic_add(diag_qk_out + owner * F + offs_f, diag_qk, sem="relaxed")
        tl.atomic_add(diag_intra_out + owner * F + offs_f, diag_intra, sem="relaxed")

    @triton.jit
    def _serial2_kernel(
        psi,
        dphi,
        state,
        dstate,
        qk_dot,
        dk_out,
        dq_out,
        diag_qk_out,
        diag_intra_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        P_TILE_C: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_pt = tl.arange(0, P_TILE_C)

        dk_acc = tl.zeros((F, N), tl.float32)
        dq_acc = tl.zeros((F, N), tl.float32)
        dqk_acc = tl.zeros((F, F), tl.float32)
        dki_acc = tl.zeros((F, F), tl.float32)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        for p_pass in tl.static_range(0, 2):
            offs_p = p_pass * P_TILE_C + offs_pt
            psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
            dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
            state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
            dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)

            dk_acc += tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
            dq_acc += tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)
            dqk_acc += tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)
            dki_acc += tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * dqk_acc, axis=1)
        diag_intra = tl.sum(qk * dki_acc, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk_acc)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq_acc)
        tl.store(diag_qk_out + owner * F + offs_f, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_f, diag_intra)

    return _fullp_kernel, _ptile_atomic_kernel, _serial2_kernel


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
        "qk_dot": torch.randn(shape.owners, shape.fused_chunk, shape.fused_chunk, device=device, dtype=dtype) * scale,
    }


def _empty_outputs(shape: Shape) -> dict[str, Any]:
    import torch

    dk = torch.empty(shape.owners, shape.fused_chunk, shape.N, device="cuda", dtype=torch.float32)
    return {
        "dk": dk,
        "dq": torch.empty_like(dk),
        "diag_qk": torch.empty(shape.owners, shape.fused_chunk, device="cuda", dtype=torch.float32),
        "diag_intra": torch.empty(shape.owners, shape.fused_chunk, device="cuda", dtype=torch.float32),
    }


def _zero_outputs(outputs: dict[str, Any]) -> None:
    for tensor in outputs.values():
        tensor.zero_()


def _memory_model(shape: Shape) -> dict[str, Any]:
    f = shape.fused_chunk
    n_tiles = shape.P // P_TILE
    bf16 = 2
    fp32 = 4
    dqdk_outputs = 2 * f * shape.N * fp32
    diag_outputs = 2 * f * fp32
    final_output = dqdk_outputs + diag_outputs
    qk_input = f * f * bf16
    fullp_inputs = (2 * f * shape.P + 2 * shape.N * shape.P) * bf16 + qk_input
    ptile_inputs = (2 * f * P_TILE + 2 * shape.N * P_TILE) * bf16 + qk_input
    accumulators = (2 * f * shape.N + 2 * f * f + 2 * f) * fp32
    fullp_programs = shape.owners
    atomic_programs = shape.owners * n_tiles
    return {
        "p_tile": P_TILE,
        "n_p_tiles": n_tiles,
        "program_count": {
            "fullp": fullp_programs,
            "serial2": fullp_programs,
            "ptile_atomic": atomic_programs,
        },
        "per_program_live_input_bytes": {
            "fullp": fullp_inputs,
            "serial2_peak_per_pass": ptile_inputs,
            "ptile_atomic": ptile_inputs,
        },
        "per_program_accumulator_bytes": {
            "fullp": accumulators,
            "serial2": accumulators,
            "ptile_atomic": accumulators,
        },
        "per_program_estimated_peak_live_bytes": {
            "fullp": fullp_inputs + accumulators,
            "serial2": ptile_inputs + accumulators,
            "ptile_atomic": ptile_inputs + accumulators,
        },
        "per_program_live_input_reduction_serial2_vs_fullp": fullp_inputs / ptile_inputs,
        "global_memory_traffic_bytes_estimate": {
            "fullp": shape.owners * (fullp_inputs + final_output),
            "serial2": shape.owners * (fullp_inputs + final_output),
            "ptile_atomic_with_zero": shape.owners
            * (fullp_inputs + final_output + n_tiles * 2 * final_output),
        },
        "final_output_bytes": shape.owners * final_output,
        "fp32_partial_handoff_bytes_if_serial2_wrote_partials": shape.owners * n_tiles * final_output,
    }


def _compare(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for name in ("dk", "dq", "diag_qk", "diag_intra"):
        diff = (ref[name] - got[name]).abs()
        out[name] = {
            "max_abs": float(diff.max().item()),
            "allclose_1e_2": bool(torch.allclose(ref[name], got[name], rtol=1.0e-2, atol=1.0e-2)),
        }
    out["allclose_count"] = sum(1 for name in ("dk", "dq", "diag_qk", "diag_intra") if out[name]["allclose_1e_2"])
    out["allclose_total"] = 4
    return out


def _run_shape(shape: Shape, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    import torch

    fullp_kernel, atomic_kernel, serial2_kernel = _define_kernels()
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "memory_model": _memory_model(shape),
    }
    if shape.P != 128 or shape.fused_chunk != 64 or shape.N != 64:
        result.update({"status": "skipped", "reason": "wave2 prototype specializes P=128,F=64,N=64"})
        return result

    try:
        tensors = _make_inputs(shape)
        ref = _empty_outputs(shape)
        atomic = _empty_outputs(shape)
        serial2 = _empty_outputs(shape)
        grid_owner = (shape.owners,)
        grid_atomic = (shape.owners, 2)

        def run_fullp() -> None:
            fullp_kernel[grid_owner](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["dstate"],
                tensors["qk_dot"],
                ref["dk"],
                ref["dq"],
                ref["diag_qk"],
                ref["diag_intra"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                shape.P,
                num_warps=8,
            )

        def run_atomic_nozero() -> None:
            atomic_kernel[grid_atomic](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["dstate"],
                tensors["qk_dot"],
                atomic["dk"],
                atomic["dq"],
                atomic["diag_qk"],
                atomic["diag_intra"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                P_TILE,
                num_warps=4,
            )

        def run_atomic_with_zero() -> None:
            _zero_outputs(atomic)
            run_atomic_nozero()

        def run_serial2() -> None:
            serial2_kernel[grid_owner](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["dstate"],
                tensors["qk_dot"],
                serial2["dk"],
                serial2["dq"],
                serial2["diag_qk"],
                serial2["diag_intra"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                P_TILE,
                num_warps=8,
            )

        run_fullp()
        run_atomic_with_zero()
        run_serial2()
        torch.cuda.synchronize()

        result.update(
            {
                "status": "ok",
                "correctness": {
                    "atomic_vs_fullp": _compare(ref, atomic),
                    "serial2_vs_fullp": _compare(ref, serial2),
                },
                "elapsed": {
                    "fullp_ms": _time_cuda_events(run_fullp, warmup=warmup, iters=iters),
                    "ptile_atomic_compute_only_ms": _time_cuda_events(run_atomic_nozero, warmup=warmup, iters=iters),
                    "ptile_atomic_with_zero_ms": _time_cuda_events(run_atomic_with_zero, warmup=warmup, iters=iters),
                    "serial2_ms": _time_cuda_events(run_serial2, warmup=warmup, iters=iters),
                },
            }
        )
        full_ms = result["elapsed"]["fullp_ms"]["mean_ms"]
        atomic_ms = result["elapsed"]["ptile_atomic_with_zero_ms"]["mean_ms"]
        serial_ms = result["elapsed"]["serial2_ms"]["mean_ms"]
        if full_ms and atomic_ms and serial_ms:
            result["speed"] = {
                "serial2_vs_fullp": full_ms / serial_ms,
                "serial2_vs_ptile_atomic_with_zero": atomic_ms / serial_ms,
                "ptile_atomic_with_zero_vs_fullp": full_ms / atomic_ms,
            }
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "crashed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback_tail": traceback.format_exc()[-8000:],
            }
        )
    return result


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


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800)
def run_probe(requested_gpu: str, shape_csv: str, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    try:
        device = _device_report(requested_gpu)
        results = []
        for shape in _selected_shapes(shape_csv):
            print(f"[owner-wave2] shape={shape.name} p_tile={P_TILE}", flush=True)
            results.append(_run_shape(shape, warmup, iters))
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "p_tile": P_TILE,
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
def main(shape_csv: str = "smoke_p128", warmup: int = 1, iters: int = 4) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, warmup, iters)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
