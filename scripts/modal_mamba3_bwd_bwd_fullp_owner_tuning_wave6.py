"""Modal H200 microbench for Mamba3 bwd_bwd full-P owner tuning wave 6.

This returns to the no-atomic full-P DQ/DK/diag subproblem from wave2 and
sweeps local ownership/compiler choices without global partial reductions.

The wave2 reference computes:

    DK        = Psi  @ dState
    DQ        = dPhi @ State
    G         = dPhi @ Psi.T
    M         = Psi  @ dPhi.T
    diag_qk   = row_sum(qk * G)
    diag_intr = row_sum(qk * M)

The one-diag-dot candidate keeps identical semantics but uses M = G.T.

Run:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_fullp_owner_tuning_wave6.py \
        --shape-csv smoke_p128,productionish --warmup 2 --iters 8
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:2")

APP_NAME = "cppmega-mamba3-bwd-bwd-fullp-owner-tuning-wave6"


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


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    diag_num_warps: int | None = None
    diag_num_stages: int | None = None


SHAPES: dict[str, Shape] = {
    "smoke_p128": Shape("smoke_p128", B=1, S=256, H=4, G=1, N=64, P=128, R=4),
    "representative": Shape("representative", B=2, S=1024, H=8, G=1, N=64, P=128, R=4),
    "productionish": Shape("productionish", B=4, S=4096, H=32, G=1, N=64, P=128, R=4),
}


VARIANTS: dict[str, Variant] = {
    # Exact wave2 semantics and ownership. The w8s3 entry is the reference.
    "full64_two_w8s3": Variant("full64_two_w8s3", "full_two", 64, 64, 8, 3),
    "full64_two_w4s3": Variant("full64_two_w4s3", "full_two", 64, 64, 4, 3),
    "full64_two_w8s2": Variant("full64_two_w8s2", "full_two", 64, 64, 8, 2),
    "full64_two_w8s4": Variant("full64_two_w8s4", "full_two", 64, 64, 8, 4),
    # Same outputs, but computes only G=dPhi@Psi.T and uses G.T for diag_intr.
    "full64_one_w8s3": Variant("full64_one_w8s3", "full_one", 64, 64, 8, 3),
    "full64_one_w4s3": Variant("full64_one_w4s3", "full_one", 64, 64, 4, 3),
    "full64_one_w8s2": Variant("full64_one_w8s2", "full_one", 64, 64, 8, 2),
    "full64_one_w8s4": Variant("full64_one_w8s4", "full_one", 64, 64, 8, 4),
    "full64_one_w4s4": Variant("full64_one_w4s4", "full_one", 64, 64, 4, 4),
    # Unique row-tile owners. They keep P=128 and read full M columns for diag.
    "row32_two_w4s3": Variant("row32_two_w4s3", "row_two", 32, 64, 4, 3),
    "row32_two_w8s3": Variant("row32_two_w8s3", "row_two", 32, 64, 8, 3),
    "row32_two_w4s2": Variant("row32_two_w4s2", "row_two", 32, 64, 4, 2),
    "row16_two_w4s3": Variant("row16_two_w4s3", "row_two", 16, 64, 4, 3),
    "row16_two_w8s3": Variant("row16_two_w8s3", "row_two", 16, 64, 8, 3),
    # No partials: a unique DQ/DK tile kernel plus a unique full-owner diag kernel.
    "split64x32_diag1_w4s3": Variant("split64x32_diag1_w4s3", "split_one", 64, 32, 4, 3, 4, 3),
    "split64x32_diag1_w8s3": Variant("split64x32_diag1_w8s3", "split_one", 64, 32, 8, 3, 4, 3),
    "split32x32_diag1_w4s3": Variant("split32x32_diag1_w4s3", "split_one", 32, 32, 4, 3, 4, 3),
    "split32x64_diag1_w4s3": Variant("split32x64_diag1_w4s3", "split_one", 32, 64, 4, 3, 4, 3),
}

DEFAULT_VARIANT_CSV = ",".join(VARIANTS)


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    return img.env({"CPPMEGA_IMAGE_REF": GHCR_REF})


app = modal.App(APP_NAME)


def _define_kernels() -> tuple[Any, Any, Any, Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _full_two_diag_kernel(
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
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

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
        g = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)
        m = tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * g, axis=1)
        diag_intra = tl.sum(qk * m, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)
        tl.store(diag_qk_out + owner * F + offs_f, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_f, diag_intra)

    @triton.jit
    def _full_one_diag_kernel(
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
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

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
        g = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * g, axis=1)
        diag_intra = tl.sum(qk * tl.trans(g), axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)
        tl.store(diag_qk_out + owner * F + offs_f, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_f, diag_intra)

    @triton.jit
    def _row_two_diag_kernel(
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
        BM: tl.constexpr,
    ):
        owner = tl.program_id(0)
        m_block = tl.program_id(1)
        offs_m = m_block * BM + tl.arange(0, BM)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_m = tl.load(psi_base + offs_m[:, None] * P + offs_p[None, :])
        dphi_m = tl.load(dphi_base + offs_m[:, None] * P + offs_p[None, :])
        psi_all = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_all = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)

        dk = tl.dot(psi_m, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_m, state_tile, out_dtype=tl.float32)
        g_rows = tl.dot(dphi_m, tl.trans(psi_all), out_dtype=tl.float32)
        m_rows = tl.dot(psi_m, tl.trans(dphi_all), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_m[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * g_rows, axis=1)
        diag_intra = tl.sum(qk * m_rows, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_m[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_m[:, None] * N + offs_n[None, :], dq)
        tl.store(diag_qk_out + owner * F + offs_m, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_m, diag_intra)

    @triton.jit
    def _dqdk_tile_kernel(
        psi,
        dphi,
        state,
        dstate,
        dk_out,
        dq_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        owner = tl.program_id(0)
        m_block = tl.program_id(1)
        n_block = tl.program_id(2)
        offs_m = m_block * BM + tl.arange(0, BM)
        offs_n = n_block * BN + tl.arange(0, BN)
        offs_p = tl.arange(0, P)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P

        psi_m = tl.load(psi_base + offs_m[:, None] * P + offs_p[None, :])
        dphi_m = tl.load(dphi_base + offs_m[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)

        dk = tl.dot(psi_m, dstate_tile, out_dtype=tl.float32)
        dq = tl.dot(dphi_m, state_tile, out_dtype=tl.float32)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_m[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_m[:, None] * N + offs_n[None, :], dq)

    @triton.jit
    def _diag_one_only_kernel(
        psi,
        dphi,
        qk_dot,
        diag_qk_out,
        diag_intra_out,
        F: tl.constexpr,
        P: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_p = tl.arange(0, P)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        g = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)

        qk = tl.load(qk_dot + owner * F * F + offs_f[:, None] * F + offs_f[None, :])
        diag_qk = tl.sum(qk * g, axis=1)
        diag_intra = tl.sum(qk * tl.trans(g), axis=1)

        tl.store(diag_qk_out + owner * F + offs_f, diag_qk)
        tl.store(diag_intra_out + owner * F + offs_f, diag_intra)

    return (
        _full_two_diag_kernel,
        _full_one_diag_kernel,
        _row_two_diag_kernel,
        _dqdk_tile_kernel,
        _diag_one_only_kernel,
    )


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
        "qk_dot": torch.randn(shape.owners, shape.fused_chunk, shape.fused_chunk, device=device, dtype=dtype)
        * scale,
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


def _compare(ref: dict[str, Any], got: dict[str, Any], check_owners: int) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for name in ("dk", "dq", "diag_qk", "diag_intra"):
        diff = (ref[name][:check_owners] - got[name][:check_owners]).abs()
        out[name] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose_1e_2": bool(
                torch.allclose(ref[name][:check_owners], got[name][:check_owners], rtol=1.0e-2, atol=1.0e-2)
            ),
        }
    out["allclose_count"] = sum(1 for name in ("dk", "dq", "diag_qk", "diag_intra") if out[name]["allclose_1e_2"])
    out["allclose_total"] = 4
    return out


def _memory_model(shape: Shape, variant: Variant) -> dict[str, Any]:
    f = shape.fused_chunk
    bf16 = 2
    fp32 = 4
    full_inputs = (2 * f * shape.P + 2 * shape.N * shape.P + f * f) * bf16
    final_outputs = (2 * f * shape.N + 2 * f) * fp32
    if variant.mode == "full_two":
        accum = (2 * f * shape.N + 2 * f * f + 2 * f) * fp32
        return {
            "program_count": {"kernels": 1, "programs": shape.owners},
            "block_m": 64,
            "block_n": 64,
            "block_p": shape.P,
            "diag_surface_dots": 2,
            "estimated_peak_live_bytes_per_program": full_inputs + accum,
            "global_bytes_estimate": shape.owners * (full_inputs + final_outputs),
        }
    if variant.mode == "full_one":
        accum = (2 * f * shape.N + f * f + 2 * f) * fp32
        return {
            "program_count": {"kernels": 1, "programs": shape.owners},
            "block_m": 64,
            "block_n": 64,
            "block_p": shape.P,
            "diag_surface_dots": 1,
            "estimated_peak_live_bytes_per_program": full_inputs + accum,
            "global_bytes_estimate": shape.owners * (full_inputs + final_outputs),
        }
    if variant.mode == "row_two":
        bm = variant.block_m
        m_blocks = f // bm
        row_inputs = (2 * bm * shape.P + 2 * f * shape.P + 2 * shape.N * shape.P + bm * f) * bf16
        row_accum = (2 * bm * shape.N + 2 * bm * f + 2 * bm) * fp32
        row_outputs = (2 * bm * shape.N + 2 * bm) * fp32
        return {
            "program_count": {"kernels": 1, "programs": shape.owners * m_blocks},
            "block_m": bm,
            "block_n": 64,
            "block_p": shape.P,
            "diag_surface_dots": 2,
            "estimated_peak_live_bytes_per_program": row_inputs + row_accum,
            "global_bytes_estimate": shape.owners * m_blocks * row_inputs + shape.owners * final_outputs,
        }
    bm = variant.block_m
    bn = variant.block_n
    m_blocks = f // bm
    n_blocks = shape.N // bn
    dqdk_inputs = (2 * bm * shape.P + 2 * bn * shape.P) * bf16
    dqdk_accum = (2 * bm * bn) * fp32
    diag_inputs = (2 * f * shape.P + f * f) * bf16
    diag_accum = (f * f + 2 * f) * fp32
    return {
        "program_count": {
            "kernels": 2,
            "dqdk_programs": shape.owners * m_blocks * n_blocks,
            "diag_programs": shape.owners,
        },
        "block_m": bm,
        "block_n": bn,
        "block_p": shape.P,
        "diag_surface_dots": 1,
        "estimated_peak_live_bytes_per_program": {
            "dqdk": dqdk_inputs + dqdk_accum,
            "diag": diag_inputs + diag_accum,
        },
        "global_bytes_estimate": shape.owners
        * (m_blocks * n_blocks * dqdk_inputs + diag_inputs + final_outputs),
    }


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


def _selected_variants(variant_csv: str) -> list[Variant]:
    selected: list[Variant] = []
    for raw in variant_csv.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; choose one of {sorted(VARIANTS)}")
        selected.append(VARIANTS[name])
    if not selected:
        raise ValueError("at least one variant required")
    return selected


def _run_shape(shape: Shape, variants: list[Variant], warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    import torch

    full_two_kernel, full_one_kernel, row_two_kernel, dqdk_tile_kernel, diag_one_only_kernel = _define_kernels()
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "baselines_ms": {
            "wave2_fullp_dq_dk_diag_productionish": 1.0865,
            "stage2_bwd_bwd_productionish_default_bf1bb0": 3.6940,
        },
    }
    if shape.P != 128 or shape.fused_chunk != 64 or shape.N != 64:
        result.update({"status": "skipped", "reason": "wave6 prototype specializes P=128,F=64,N=64"})
        return result

    try:
        tensors = _make_inputs(shape)
        ref = _empty_outputs(shape)
        check_owners = shape.owners if shape.name == "smoke_p128" else min(shape.owners, 64)
        grid_owner = (shape.owners,)

        def run_ref() -> None:
            full_two_kernel[grid_owner](
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
                num_warps=8,
                num_stages=3,
            )

        run_ref()
        torch.cuda.synchronize()

        variant_results = []
        for variant in variants:
            out = _empty_outputs(shape)
            variant_result: dict[str, Any] = {
                "name": variant.name,
                "variant": asdict(variant),
                "memory_model": _memory_model(shape, variant),
            }

            if shape.fused_chunk % variant.block_m != 0:
                variant_result.update({"status": "skipped", "reason": "block_m must divide fused chunk"})
                variant_results.append(variant_result)
                continue
            if shape.N % variant.block_n != 0:
                variant_result.update({"status": "skipped", "reason": "block_n must divide N"})
                variant_results.append(variant_result)
                continue

            if variant.mode == "full_two":

                def run_variant() -> None:
                    full_two_kernel[grid_owner](
                        tensors["psi"],
                        tensors["dphi"],
                        tensors["state"],
                        tensors["dstate"],
                        tensors["qk_dot"],
                        out["dk"],
                        out["dq"],
                        out["diag_qk"],
                        out["diag_intra"],
                        shape.fused_chunk,
                        shape.N,
                        shape.P,
                        num_warps=variant.num_warps,
                        num_stages=variant.num_stages,
                    )

                elapsed = {"total_ms": _time_cuda_events(run_variant, warmup=warmup, iters=iters)}
            elif variant.mode == "full_one":

                def run_variant() -> None:
                    full_one_kernel[grid_owner](
                        tensors["psi"],
                        tensors["dphi"],
                        tensors["state"],
                        tensors["dstate"],
                        tensors["qk_dot"],
                        out["dk"],
                        out["dq"],
                        out["diag_qk"],
                        out["diag_intra"],
                        shape.fused_chunk,
                        shape.N,
                        shape.P,
                        num_warps=variant.num_warps,
                        num_stages=variant.num_stages,
                    )

                elapsed = {"total_ms": _time_cuda_events(run_variant, warmup=warmup, iters=iters)}
            elif variant.mode == "row_two":
                grid_row = (shape.owners, shape.fused_chunk // variant.block_m)

                def run_variant() -> None:
                    row_two_kernel[grid_row](
                        tensors["psi"],
                        tensors["dphi"],
                        tensors["state"],
                        tensors["dstate"],
                        tensors["qk_dot"],
                        out["dk"],
                        out["dq"],
                        out["diag_qk"],
                        out["diag_intra"],
                        shape.fused_chunk,
                        shape.N,
                        shape.P,
                        variant.block_m,
                        num_warps=variant.num_warps,
                        num_stages=variant.num_stages,
                    )

                elapsed = {"total_ms": _time_cuda_events(run_variant, warmup=warmup, iters=iters)}
            elif variant.mode == "split_one":
                grid_dqdk = (
                    shape.owners,
                    shape.fused_chunk // variant.block_m,
                    shape.N // variant.block_n,
                )

                def run_dqdk() -> None:
                    dqdk_tile_kernel[grid_dqdk](
                        tensors["psi"],
                        tensors["dphi"],
                        tensors["state"],
                        tensors["dstate"],
                        out["dk"],
                        out["dq"],
                        shape.fused_chunk,
                        shape.N,
                        shape.P,
                        variant.block_m,
                        variant.block_n,
                        num_warps=variant.num_warps,
                        num_stages=variant.num_stages,
                    )

                def run_diag() -> None:
                    diag_one_only_kernel[grid_owner](
                        tensors["psi"],
                        tensors["dphi"],
                        tensors["qk_dot"],
                        out["diag_qk"],
                        out["diag_intra"],
                        shape.fused_chunk,
                        shape.P,
                        num_warps=variant.diag_num_warps or 4,
                        num_stages=variant.diag_num_stages or 3,
                    )

                def run_variant() -> None:
                    run_dqdk()
                    run_diag()

                elapsed = {
                    "dqdk_ms": _time_cuda_events(run_dqdk, warmup=warmup, iters=iters),
                    "diag_kernel_ms": _time_cuda_events(run_diag, warmup=warmup, iters=iters),
                    "total_ms": _time_cuda_events(run_variant, warmup=warmup, iters=iters),
                }
            else:
                variant_result.update({"status": "skipped", "reason": f"unknown mode {variant.mode}"})
                variant_results.append(variant_result)
                continue

            run_variant()
            torch.cuda.synchronize()
            total_ms = elapsed["total_ms"]["mean_ms"]
            variant_result.update(
                {
                    "status": "ok",
                    "correctness": _compare(ref, out, check_owners),
                    "elapsed": elapsed,
                    "speed": {
                        "vs_wave2_1p0865": total_ms / 1.0865 if total_ms else None,
                        "vs_stage2_3p6940": total_ms / 3.6940 if total_ms else None,
                    },
                }
            )
            variant_results.append(variant_result)

        result.update(
            {
                "status": "ok",
                "correctness_checked_owners": check_owners,
                "variants": variant_results,
            }
        )
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


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800)
def run_probe(requested_gpu: str, shape_csv: str, variant_csv: str, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    try:
        device = _device_report(requested_gpu)
        variants = _selected_variants(variant_csv)
        results = []
        for shape in _selected_shapes(shape_csv):
            print(
                f"[fullp-owner-wave6] shape={shape.name} variants={','.join(v.name for v in variants)}",
                flush=True,
            )
            results.append(_run_shape(shape, variants, warmup, iters))
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "variant_csv": variant_csv,
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
def main(
    shape_csv: str = "smoke_p128",
    variant_csv: str = DEFAULT_VARIANT_CSV,
    warmup: int = 1,
    iters: int = 4,
) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, variant_csv, warmup, iters)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
