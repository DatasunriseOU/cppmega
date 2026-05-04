"""Modal H200 microbench for Mamba3 bwd_bwd accumulator-surface reduction wave 4.

This isolates the bwd_bwd stage-2 F-by-F accumulator surface at the production
shape used by the owner-rewrite probes:

* F = chunk * R = 64
* N = 64
* P = 128

The full-surface reference computes two full [F,F] products:

    G = dphi @ psi.T          # dqk_from_diag surface
    M = psi @ dphi.T          # dk_intrachunk surface

The streaming candidate avoids materializing either full [F,F] accumulator:

* M @ q is streamed over j blocks as sum_j (psi_i dot dphi_j) * q_j.
* M.T @ k is streamed over i blocks as sum_i (psi_i dot dphi_j) * k_i.
* DSSDA-like row reductions stream the same M tiles.
* dqk_from_diag only needs same-token R x R blocks, so a small block kernel
  computes G_s = dphi_s @ psi_s.T for each token s.

Run:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_surface_reduction_wave4.py \
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

APP_NAME = "cppmega-mamba3-bwd-bwd-surface-reduction-wave4"
BLOCK_J = 16


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


def _define_kernels() -> tuple[Any, Any, Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def _full_surface_kernel(
        psi,
        dphi,
        state,
        dstate,
        q,
        k,
        weight,
        qk_dot,
        dk_out,
        dq_out,
        dgamma_out,
        dssda_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        dstate_base = dstate + owner * N * P
        q_base = q + owner * F * N
        k_base = k + owner * F * N
        weight_base = weight + owner * F * F
        qk_base = qk_dot + owner * F * F

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)
        q_tile = tl.load(q_base + offs_f[:, None] * N + offs_n[None, :])
        k_tile = tl.load(k_base + offs_f[:, None] * N + offs_n[None, :])
        weight_tile = tl.load(weight_base + offs_f[:, None] * F + offs_f[None, :])
        qk_tile = tl.load(qk_base + offs_f[:, None] * F + offs_f[None, :])

        g = tl.dot(dphi_tile, tl.trans(psi_tile), out_dtype=tl.float32)
        m = tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)

        row_t = offs_f[:, None] // R
        col_t = offs_f[None, :] // R
        causal = row_t >= col_t
        same_token = row_t == col_t
        mw = tl.where(causal, m * weight_tile, 0.0)
        gbd = tl.where(same_token, g, 0.0)

        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dk += tl.dot(mw.to(tl.bfloat16), q_tile, out_dtype=tl.float32)
        dk += tl.dot(tl.trans(gbd.to(tl.bfloat16)), q_tile, out_dtype=tl.float32)

        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)
        dq += tl.dot(tl.trans(mw.to(tl.bfloat16)), k_tile, out_dtype=tl.float32)
        dq += tl.dot(gbd.to(tl.bfloat16), k_tile, out_dtype=tl.float32)

        dgamma = tl.sum(qk_tile * gbd, axis=1)
        dssda = tl.sum(qk_tile * m, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)
        tl.store(dgamma_out + owner * F + offs_f, dgamma)
        tl.store(dssda_out + owner * F + offs_f, dssda)

    @triton.jit
    def _stream_dk_kernel(
        psi,
        dphi,
        dstate,
        q,
        weight,
        qk_dot,
        dk_out,
        dgamma_out,
        dssda_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
        BJ: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)
        offs_bj = tl.arange(0, BJ)
        offs_r = tl.arange(0, R)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        dstate_base = dstate + owner * N * P
        q_base = q + owner * F * N
        weight_base = weight + owner * F * F
        qk_base = qk_dot + owner * F * F

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dstate_tile = tl.load(dstate_base + offs_p[:, None] + offs_n[None, :] * P)
        dk = tl.dot(psi_tile, dstate_tile, out_dtype=tl.float32)
        dssda = tl.zeros((F,), tl.float32)
        dgamma = tl.zeros((F,), tl.float32)

        for jb in tl.static_range(0, 4):
            offs_j = jb * BJ + offs_bj
            dphi_j = tl.load(dphi_base + offs_j[:, None] * P + offs_p[None, :])
            q_j = tl.load(q_base + offs_j[:, None] * N + offs_n[None, :])
            w_j = tl.load(weight_base + offs_f[:, None] * F + offs_j[None, :])
            qk_j = tl.load(qk_base + offs_f[:, None] * F + offs_j[None, :])
            m = tl.dot(psi_tile, tl.trans(dphi_j), out_dtype=tl.float32)
            causal = (offs_f[:, None] // R) >= (offs_j[None, :] // R)
            mw = tl.where(causal, m * w_j, 0.0)
            dk += tl.dot(mw.to(tl.bfloat16), q_j, out_dtype=tl.float32)
            dssda += tl.sum(qk_j * m, axis=1)

        for tok in tl.static_range(0, 16):
            tok_f = tok * R + offs_r
            psi_r = tl.load(psi_base + tok_f[:, None] * P + offs_p[None, :])
            dphi_r = tl.load(dphi_base + tok_f[:, None] * P + offs_p[None, :])
            qk_r = tl.load(qk_base + tok_f[:, None] * F + tok_f[None, :])
            g = tl.dot(dphi_r, tl.trans(psi_r), out_dtype=tl.float32)
            g_for_dot = g.to(tl.bfloat16).to(tl.float32)
            dgamma_r = tl.sum(qk_r * g, axis=1)
            for dst in tl.static_range(0, 4):
                dgamma_dst = tl.sum(tl.where(offs_r == dst, dgamma_r, 0.0), axis=0)
                dgamma += tl.where(offs_f == tok * R + dst, dgamma_dst, 0.0)
                dk_row = tl.zeros((N,), tl.float32)
                for src in tl.static_range(0, 4):
                    q_src = tl.load(q_base + (tok * R + src) * N + offs_n)
                    g_col = tl.sum(
                        tl.where(
                            (offs_r[:, None] == src) & (offs_r[None, :] == dst),
                            g_for_dot,
                            0.0,
                        )
                    )
                    dk_row += g_col * q_src
                dk += tl.where(offs_f[:, None] == tok * R + dst, dk_row[None, :], 0.0)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dssda_out + owner * F + offs_f, dssda)
        tl.store(dgamma_out + owner * F + offs_f, dgamma)

    @triton.jit
    def _stream_dq_kernel(
        psi,
        dphi,
        state,
        k,
        weight,
        dq_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
        BI: tl.constexpr,
    ):
        owner = tl.program_id(0)
        offs_f = tl.arange(0, F)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)
        offs_bi = tl.arange(0, BI)
        offs_r = tl.arange(0, R)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        state_base = state + owner * N * P
        k_base = k + owner * F * N
        weight_base = weight + owner * F * F

        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        state_tile = tl.load(state_base + offs_p[:, None] + offs_n[None, :] * P)
        dq = tl.dot(dphi_tile, state_tile, out_dtype=tl.float32)

        for ib in tl.static_range(0, 4):
            offs_i = ib * BI + offs_bi
            psi_i = tl.load(psi_base + offs_i[:, None] * P + offs_p[None, :])
            k_i = tl.load(k_base + offs_i[:, None] * N + offs_n[None, :])
            w_i = tl.load(weight_base + offs_i[:, None] * F + offs_f[None, :])
            m = tl.dot(psi_i, tl.trans(dphi_tile), out_dtype=tl.float32)
            causal = (offs_i[:, None] // R) >= (offs_f[None, :] // R)
            mw = tl.where(causal, m * w_i, 0.0)
            dq += tl.dot(tl.trans(mw.to(tl.bfloat16)), k_i, out_dtype=tl.float32)

        for tok in tl.static_range(0, 16):
            tok_f = tok * R + offs_r
            psi_r = tl.load(psi_base + tok_f[:, None] * P + offs_p[None, :])
            dphi_r = tl.load(dphi_base + tok_f[:, None] * P + offs_p[None, :])
            g = tl.dot(dphi_r, tl.trans(psi_r), out_dtype=tl.float32)
            g_for_dot = g.to(tl.bfloat16).to(tl.float32)
            for dst in tl.static_range(0, 4):
                dq_row = tl.zeros((N,), tl.float32)
                for src in tl.static_range(0, 4):
                    k_src = tl.load(k_base + (tok * R + src) * N + offs_n)
                    g_col = tl.sum(
                        tl.where(
                            (offs_r[:, None] == dst) & (offs_r[None, :] == src),
                            g_for_dot,
                            0.0,
                        )
                    )
                    dq_row += g_col * k_src
                dq += tl.where(offs_f[:, None] == tok * R + dst, dq_row[None, :], 0.0)

        out_base = owner * F * N
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)

    @triton.jit
    def _stream_gbd_kernel(
        psi,
        dphi,
        q,
        k,
        qk_dot,
        dk_out,
        dq_out,
        dgamma_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
    ):
        owner = tl.program_id(0)
        tok = tl.program_id(1)
        offs_r = tl.arange(0, R)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)
        offs_f = tok * R + offs_r

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        q_base = q + owner * F * N
        k_base = k + owner * F * N
        qk_base = qk_dot + owner * F * F

        psi_r = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_r = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        q_r = tl.load(q_base + offs_f[:, None] * N + offs_n[None, :])
        k_r = tl.load(k_base + offs_f[:, None] * N + offs_n[None, :])
        qk_r = tl.load(qk_base + offs_f[:, None] * F + offs_f[None, :])

        g = tl.dot(dphi_r, tl.trans(psi_r), out_dtype=tl.float32)
        g_for_dot = g.to(tl.bfloat16).to(tl.float32)
        dk_add = tl.zeros((R, N), tl.float32)
        dq_add = tl.zeros((R, N), tl.float32)
        for rr in tl.static_range(0, 4):
            q_rr = tl.load(q_base + (tok * R + rr) * N + offs_n)
            k_rr = tl.load(k_base + (tok * R + rr) * N + offs_n)
            g_t_col = tl.sum(tl.where(offs_r[None, :] == rr, tl.trans(g_for_dot), 0.0), axis=1)
            g_col = tl.sum(tl.where(offs_r[None, :] == rr, g_for_dot, 0.0), axis=1)
            dk_add += g_t_col[:, None] * q_rr[None, :]
            dq_add += g_col[:, None] * k_rr[None, :]
        dgamma = tl.sum(qk_r * g, axis=1)

        out_base = owner * F * N
        tl.atomic_add(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk_add, sem="relaxed")
        tl.atomic_add(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq_add, sem="relaxed")
        tl.store(dgamma_out + owner * F + offs_f, dgamma)

    return _full_surface_kernel, _stream_dk_kernel, _stream_dq_kernel, _stream_gbd_kernel


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
        "q": torch.randn(shape.owners, shape.fused_chunk, shape.N, device=device, dtype=dtype) * scale,
        "k": torch.randn(shape.owners, shape.fused_chunk, shape.N, device=device, dtype=dtype) * scale,
        "weight": torch.randn(shape.owners, shape.fused_chunk, shape.fused_chunk, device=device, dtype=dtype) * scale,
        "qk_dot": torch.randn(shape.owners, shape.fused_chunk, shape.fused_chunk, device=device, dtype=dtype) * scale,
    }


def _empty_outputs(shape: Shape) -> dict[str, Any]:
    import torch

    dk = torch.empty(shape.owners, shape.fused_chunk, shape.N, device="cuda", dtype=torch.float32)
    return {
        "dk": dk,
        "dq": torch.empty_like(dk),
        "dgamma": torch.empty(shape.owners, shape.fused_chunk, device="cuda", dtype=torch.float32),
        "dssda": torch.empty(shape.owners, shape.fused_chunk, device="cuda", dtype=torch.float32),
    }


def _compare(ref: dict[str, Any], got: dict[str, Any]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for name in ("dk", "dq", "dgamma", "dssda"):
        diff = (ref[name] - got[name]).abs()
        out[name] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose_1e_2": bool(torch.allclose(ref[name], got[name], rtol=1.0e-2, atol=1.0e-2)),
        }
    out["allclose_count"] = sum(1 for name in ("dk", "dq", "dgamma", "dssda") if out[name]["allclose_1e_2"])
    out["allclose_total"] = 4
    return out


def _torch_full_reference(tensors: dict[str, Any], shape: Shape, check_owners: int) -> dict[str, Any]:
    import torch

    f = shape.fused_chunk
    n = shape.N
    r = shape.R
    sl = slice(0, check_owners)
    psi = tensors["psi"][sl]
    dphi = tensors["dphi"][sl]
    state = tensors["state"][sl]
    dstate = tensors["dstate"][sl]
    q = tensors["q"][sl]
    k = tensors["k"][sl]
    weight = tensors["weight"][sl]
    qk_dot = tensors["qk_dot"][sl]

    idx = torch.arange(f, device=psi.device)
    causal = (idx[:, None] // r) >= (idx[None, :] // r)
    same_token = (idx[:, None] // r) == (idx[None, :] // r)

    psi_f = psi.float()
    dphi_f = dphi.float()
    g = torch.matmul(dphi_f, psi_f.transpose(-1, -2))
    m = torch.matmul(psi_f, dphi_f.transpose(-1, -2))
    mw = (m * weight.float() * causal.unsqueeze(0)).to(torch.bfloat16).float()
    gbd = torch.where(same_token.unsqueeze(0), g, torch.zeros_like(g))
    gbd_for_dot = gbd.to(torch.bfloat16).float()

    dk = torch.matmul(psi_f, dstate.float().transpose(-1, -2))
    dk = dk + torch.matmul(mw, q.float())
    dk = dk + torch.matmul(gbd_for_dot.transpose(-1, -2), q.float())

    dq = torch.matmul(dphi_f, state.float().transpose(-1, -2))
    dq = dq + torch.matmul(mw.transpose(-1, -2), k.float())
    dq = dq + torch.matmul(gbd_for_dot, k.float())

    dgamma = (qk_dot.float() * gbd).sum(dim=-1)
    dssda = (qk_dot.float() * m).sum(dim=-1)

    return {
        "dk": dk.reshape(check_owners, f, n),
        "dq": dq.reshape(check_owners, f, n),
        "dgamma": dgamma.reshape(check_owners, f),
        "dssda": dssda.reshape(check_owners, f),
    }


def _memory_model(shape: Shape) -> dict[str, Any]:
    f = shape.fused_chunk
    n = shape.N
    p = shape.P
    bj = BLOCK_J
    fp32 = 4
    return {
        "fused_chunk": f,
        "block_j": bj,
        "full_surface_accumulator_bytes": {
            "dk_dq_g_m": (2 * f * n + 2 * f * f) * fp32,
            "g_or_m_single": f * f * fp32,
        },
        "stream_surface_accumulator_bytes": {
            "stream_dk": (f * n + f * bj + 2 * f) * fp32,
            "stream_dq": (f * n + bj * f) * fp32,
            "folded_gbd_per_token": (shape.R * shape.R) * fp32,
        },
        "algebraic_intermediate_if_associative_no_mask_bytes": {
            "dphi_T_q_or_psi_T_k": p * n * fp32,
        },
        "program_count": {
            "full_surface": shape.owners,
            "stream_dk": shape.owners,
            "stream_dq": shape.owners,
            "stream_gbd": "folded_into_stream_dk_and_stream_dq",
        },
    }


def _run_shape(shape: Shape, warmup: int, iters: int) -> dict[str, Any]:
    import traceback

    import torch

    _, stream_dk_kernel, stream_dq_kernel, _ = _define_kernels()
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "memory_model": _memory_model(shape),
        "wave2_baseline_ms": {
            "fullp_dq_dk_diag_productionish": 1.0865,
            "serial2_productionish": 1.6625,
            "ptile_atomic_with_zero_productionish": 2.1496,
        },
    }
    if shape.P != 128 or shape.fused_chunk != 64 or shape.N != 64 or shape.R != 4:
        result.update({"status": "skipped", "reason": "wave4 prototype specializes P=128,F=64,N=64,R=4"})
        return result

    try:
        tensors = _make_inputs(shape)
        stream = _empty_outputs(shape)
        grid_owner = (shape.owners,)

        def run_stream_dk() -> None:
            stream_dk_kernel[grid_owner](
                tensors["psi"],
                tensors["dphi"],
                tensors["dstate"],
                tensors["q"],
                tensors["weight"],
                tensors["qk_dot"],
                stream["dk"],
                stream["dgamma"],
                stream["dssda"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                shape.R,
                BLOCK_J,
                num_warps=8,
            )

        def run_stream_dq() -> None:
            stream_dq_kernel[grid_owner](
                tensors["psi"],
                tensors["dphi"],
                tensors["state"],
                tensors["k"],
                tensors["weight"],
                stream["dq"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                shape.R,
                BLOCK_J,
                num_warps=8,
            )

        def run_stream_all() -> None:
            run_stream_dk()
            run_stream_dq()

        run_stream_all()
        torch.cuda.synchronize()
        check_owners = shape.owners if shape.name == "smoke_p128" else min(shape.owners, 64)
        ref = _torch_full_reference(tensors, shape, check_owners)
        stream_check = {name: tensor[:check_owners] for name, tensor in stream.items()}

        result.update(
            {
                "status": "ok",
                "correctness_checked_owners": check_owners,
                "correctness": {
                    "stream_vs_torch_full_surface": _compare(ref, stream_check),
                },
                "elapsed": {
                    "stream_all_ms": _time_cuda_events(run_stream_all, warmup=warmup, iters=iters),
                    "stream_dk_ms": _time_cuda_events(run_stream_dk, warmup=warmup, iters=iters),
                    "stream_dq_ms": _time_cuda_events(run_stream_dq, warmup=warmup, iters=iters),
                },
            }
        )
        stream_ms = result["elapsed"]["stream_all_ms"]["mean_ms"]
        if stream_ms:
            result["speed"] = {
                "stream_ms_over_wave2_fullp_productionish": stream_ms / 1.0865,
                "stream_ms_over_wave2_serial2_productionish": stream_ms / 1.6625,
                "wave2_fullp_ms_over_stream": 1.0865 / stream_ms,
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
            print(f"[surface-wave4] shape={shape.name} block_j={BLOCK_J}", flush=True)
            results.append(_run_shape(shape, warmup, iters))
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "block_j": BLOCK_J,
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
