"""Modal H200 microbench for Mamba3 bwd_bwd fused M-tile ownership wave 5.

This probes a one-live-M tile design for the weighted intrachunk surface:

    M = Psi @ dPhi.T
    DK += (M * W) @ Q + blockdiag(M) @ Q
    DQ += (M * W).T @ K + blockdiag(M).T @ K
    DGAMMA = row_sum(qk * blockdiag(M).T)
    DSSDA = row_sum(qk * M)

Ownership tested here:

* full_m_surface: one owner program materializes the full F x F M surface.
* row_owner: one program owns an I tile of DK/DSSDA/DGAMMA, loops over J
  tiles, forms exactly one live M_{I,J} tile, consumes it for DK/DQ/DSSDA
  while live, and writes DQ partials without atomics.
* reduce_dq: one program owns a final DQ J tile and reduces DQ partials across
  I tile owners.

Run:

    python -m py_compile scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py
    CPPMEGA_MODAL_GPU=H200:2 timeout 20m modal run \
        scripts/modal_mamba3_bwd_bwd_fused_m_tile_owner_wave5.py \
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

APP_NAME = "cppmega-mamba3-bwd-bwd-fused-m-tile-owner-wave5"
DEFAULT_BLOCK_I = 16
DEFAULT_BLOCK_J = 16


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
    def _full_m_surface_kernel(
        psi,
        dphi,
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
        q_base = q + owner * F * N
        k_base = k + owner * F * N
        weight_base = weight + owner * F * F
        qk_base = qk_dot + owner * F * F

        psi_tile = tl.load(psi_base + offs_f[:, None] * P + offs_p[None, :])
        dphi_tile = tl.load(dphi_base + offs_f[:, None] * P + offs_p[None, :])
        q_tile = tl.load(q_base + offs_f[:, None] * N + offs_n[None, :])
        k_tile = tl.load(k_base + offs_f[:, None] * N + offs_n[None, :])
        weight_tile = tl.load(weight_base + offs_f[:, None] * F + offs_f[None, :])
        qk_tile = tl.load(qk_base + offs_f[:, None] * F + offs_f[None, :])

        m = tl.dot(psi_tile, tl.trans(dphi_tile), out_dtype=tl.float32)
        row_tok = offs_f[:, None] // R
        col_tok = offs_f[None, :] // R
        causal = row_tok >= col_tok
        same_token = row_tok == col_tok

        mw = tl.where(causal, m * weight_tile, 0.0)
        block_m = tl.where(same_token, m, 0.0)
        block_g = tl.where(same_token, tl.trans(m), 0.0)

        dk = tl.dot(mw.to(tl.bfloat16), q_tile, out_dtype=tl.float32)
        dk += tl.dot(block_m.to(tl.bfloat16), q_tile, out_dtype=tl.float32)
        dq = tl.dot(tl.trans(mw.to(tl.bfloat16)), k_tile, out_dtype=tl.float32)
        dq += tl.dot(tl.trans(block_m.to(tl.bfloat16)), k_tile, out_dtype=tl.float32)
        dssda = tl.sum(qk_tile * m, axis=1)
        dgamma = tl.sum(qk_tile * block_g, axis=1)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_f[:, None] * N + offs_n[None, :], dk)
        tl.store(dq_out + out_base + offs_f[:, None] * N + offs_n[None, :], dq)
        tl.store(dgamma_out + owner * F + offs_f, dgamma)
        tl.store(dssda_out + owner * F + offs_f, dssda)

    @triton.jit
    def _row_owner_kernel(
        psi,
        dphi,
        q,
        k,
        weight,
        qk_dot,
        dq_partials,
        dk_out,
        dgamma_out,
        dssda_out,
        F: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
        BI: tl.constexpr,
        BJ: tl.constexpr,
        I_BLOCKS: tl.constexpr,
        J_BLOCKS: tl.constexpr,
    ):
        owner = tl.program_id(0)
        i_block = tl.program_id(1)
        offs_i = i_block * BI + tl.arange(0, BI)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)
        offs_bj = tl.arange(0, BJ)

        psi_base = psi + owner * F * P
        dphi_base = dphi + owner * F * P
        q_base = q + owner * F * N
        k_base = k + owner * F * N
        weight_base = weight + owner * F * F
        qk_base = qk_dot + owner * F * F
        partial_base = dq_partials + (owner * I_BLOCKS + i_block) * F * N

        psi_i = tl.load(psi_base + offs_i[:, None] * P + offs_p[None, :])
        k_i = tl.load(k_base + offs_i[:, None] * N + offs_n[None, :])
        dk_acc = tl.zeros((BI, N), tl.float32)
        dssda_acc = tl.zeros((BI,), tl.float32)
        dgamma_acc = tl.zeros((BI,), tl.float32)

        for j_block in tl.static_range(0, J_BLOCKS):
            offs_j = j_block * BJ + offs_bj
            dphi_j = tl.load(dphi_base + offs_j[:, None] * P + offs_p[None, :])
            q_j = tl.load(q_base + offs_j[:, None] * N + offs_n[None, :])
            w_ij = tl.load(weight_base + offs_i[:, None] * F + offs_j[None, :])
            qk_ij = tl.load(qk_base + offs_i[:, None] * F + offs_j[None, :])

            m = tl.dot(psi_i, tl.trans(dphi_j), out_dtype=tl.float32)
            row_tok = offs_i[:, None] // R
            col_tok = offs_j[None, :] // R
            causal = row_tok >= col_tok
            same_token = row_tok == col_tok
            mw = tl.where(causal, m * w_ij, 0.0)
            block_m = tl.where(same_token, m, 0.0)
            block_g = tl.where(same_token, tl.trans(m), 0.0)

            mw_bf = mw.to(tl.bfloat16)
            block_bf = block_m.to(tl.bfloat16)
            dk_acc += tl.dot(mw_bf, q_j, out_dtype=tl.float32)
            dk_acc += tl.dot(block_bf, q_j, out_dtype=tl.float32)

            dq_part = tl.dot(tl.trans(mw_bf), k_i, out_dtype=tl.float32)
            dq_part += tl.dot(tl.trans(block_bf), k_i, out_dtype=tl.float32)
            dssda_acc += tl.sum(qk_ij * m, axis=1)
            dgamma_acc += tl.sum(qk_ij * block_g, axis=1)

            tl.store(partial_base + offs_j[:, None] * N + offs_n[None, :], dq_part)

        out_base = owner * F * N
        tl.store(dk_out + out_base + offs_i[:, None] * N + offs_n[None, :], dk_acc)
        tl.store(dgamma_out + owner * F + offs_i, dgamma_acc)
        tl.store(dssda_out + owner * F + offs_i, dssda_acc)

    @triton.jit
    def _reduce_dq_kernel(
        dq_partials,
        dq_out,
        F: tl.constexpr,
        N: tl.constexpr,
        BJ: tl.constexpr,
        I_BLOCKS: tl.constexpr,
    ):
        owner = tl.program_id(0)
        j_block = tl.program_id(1)
        offs_j = j_block * BJ + tl.arange(0, BJ)
        offs_n = tl.arange(0, N)

        dq = tl.zeros((BJ, N), tl.float32)
        for i_block in tl.static_range(0, I_BLOCKS):
            partial_base = dq_partials + (owner * I_BLOCKS + i_block) * F * N
            dq += tl.load(partial_base + offs_j[:, None] * N + offs_n[None, :])

        out_base = owner * F * N
        tl.store(dq_out + out_base + offs_j[:, None] * N + offs_n[None, :], dq)

    return _full_m_surface_kernel, _row_owner_kernel, _reduce_dq_kernel


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


def _empty_partials(shape: Shape, block_i: int) -> Any:
    import torch

    i_blocks = shape.fused_chunk // block_i
    return torch.empty(shape.owners, i_blocks, shape.fused_chunk, shape.N, device="cuda", dtype=torch.float32)


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


def _torch_m_surface_reference(tensors: dict[str, Any], shape: Shape, check_owners: int) -> dict[str, Any]:
    import torch

    f = shape.fused_chunk
    n = shape.N
    r = shape.R
    sl = slice(0, check_owners)
    psi = tensors["psi"][sl].float()
    dphi = tensors["dphi"][sl].float()
    q = tensors["q"][sl].float()
    k = tensors["k"][sl].float()
    weight = tensors["weight"][sl].float()
    qk_dot = tensors["qk_dot"][sl].float()

    idx = torch.arange(f, device=psi.device)
    causal = (idx[:, None] // r) >= (idx[None, :] // r)
    same_token = (idx[:, None] // r) == (idx[None, :] // r)

    m = torch.matmul(psi, dphi.transpose(-1, -2))
    mw = (m * weight * causal.unsqueeze(0)).to(torch.bfloat16).float()
    block_m = torch.where(same_token.unsqueeze(0), m, torch.zeros_like(m))
    block_dot = block_m.to(torch.bfloat16).float()
    dk = torch.matmul(mw, q) + torch.matmul(block_dot, q)
    dq = torch.matmul(mw.transpose(-1, -2), k) + torch.matmul(block_dot.transpose(-1, -2), k)
    dssda = (qk_dot * m).sum(dim=-1)
    dgamma = (qk_dot * block_m.transpose(-1, -2)).sum(dim=-1)

    return {
        "dk": dk.reshape(check_owners, f, n),
        "dq": dq.reshape(check_owners, f, n),
        "dgamma": dgamma.reshape(check_owners, f),
        "dssda": dssda.reshape(check_owners, f),
    }


def _memory_model(shape: Shape, block_i: int, block_j: int) -> dict[str, Any]:
    f = shape.fused_chunk
    n = shape.N
    p = shape.P
    i_blocks = f // block_i
    j_blocks = f // block_j
    bf16 = 2
    fp32 = 4
    final_outputs = (2 * f * n + 2 * f) * fp32
    dq_final = f * n * fp32
    dq_partials = i_blocks * f * n * fp32
    full_inputs = (2 * f * p + 2 * f * n + 2 * f * f) * bf16
    row_live_inputs = (block_i * p + block_i * n + block_j * p + block_j * n + 2 * block_i * block_j) * bf16
    row_accumulators = (block_i * n + block_j * n + block_i * block_j + 2 * block_i) * fp32
    return {
        "block_i": block_i,
        "block_j": block_j,
        "i_blocks": i_blocks,
        "j_blocks": j_blocks,
        "program_count": {
            "full_m_surface": shape.owners,
            "row_owner": shape.owners * i_blocks,
            "reduce_dq": shape.owners * j_blocks,
        },
        "per_owner_bytes": {
            "full_live_inputs": full_inputs,
            "final_outputs": final_outputs,
            "dq_final": dq_final,
            "dq_partials": dq_partials,
            "dq_partial_multiplier_vs_final_dq": dq_partials / dq_final,
        },
        "per_row_owner_peak_live_bytes_estimate": row_live_inputs + row_accumulators,
        "global_bytes_estimate": {
            "full_m_surface_inputs_plus_outputs": shape.owners * (full_inputs + final_outputs),
            "row_owner_dq_partial_write": shape.owners * dq_partials,
            "reduce_dq_partial_read": shape.owners * dq_partials,
            "row_owner_final_writes_excluding_dq": shape.owners * ((f * n + 2 * f) * fp32),
            "reduce_dq_final_write": shape.owners * dq_final,
        },
    }


def _run_shape(shape: Shape, warmup: int, iters: int, block_i: int, block_j: int) -> dict[str, Any]:
    import traceback

    import torch

    full_kernel, row_kernel, reduce_kernel = _define_kernels()
    result: dict[str, Any] = {
        "shape": asdict(shape),
        "memory_model": _memory_model(shape, block_i, block_j),
        "baselines_ms": {
            "wave2_fullp_dq_dk_diag_productionish": 1.0865,
            "stage2_bwd_bwd_productionish_default_bf1bb0": 3.6940,
        },
    }
    if shape.P != 128 or shape.fused_chunk != 64 or shape.N != 64 or shape.R != 4:
        result.update({"status": "skipped", "reason": "wave5 prototype specializes P=128,F=64,N=64,R=4"})
        return result
    if block_i != block_j:
        result.update({"status": "skipped", "reason": "wave5 prototype currently requires square M tiles"})
        return result
    if shape.fused_chunk % block_i != 0 or shape.fused_chunk % block_j != 0:
        result.update({"status": "skipped", "reason": "block sizes must divide fused chunk"})
        return result

    try:
        tensors = _make_inputs(shape)
        ref = _empty_outputs(shape)
        row = _empty_outputs(shape)
        dq_partials = _empty_partials(shape, block_i)
        i_blocks = shape.fused_chunk // block_i
        j_blocks = shape.fused_chunk // block_j
        grid_owner = (shape.owners,)
        grid_row = (shape.owners, i_blocks)
        grid_reduce = (shape.owners, j_blocks)

        def run_full() -> None:
            full_kernel[grid_owner](
                tensors["psi"],
                tensors["dphi"],
                tensors["q"],
                tensors["k"],
                tensors["weight"],
                tensors["qk_dot"],
                ref["dk"],
                ref["dq"],
                ref["dgamma"],
                ref["dssda"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                shape.R,
                num_warps=8,
            )

        def run_row_compute() -> None:
            row_kernel[grid_row](
                tensors["psi"],
                tensors["dphi"],
                tensors["q"],
                tensors["k"],
                tensors["weight"],
                tensors["qk_dot"],
                dq_partials,
                row["dk"],
                row["dgamma"],
                row["dssda"],
                shape.fused_chunk,
                shape.N,
                shape.P,
                shape.R,
                block_i,
                block_j,
                i_blocks,
                j_blocks,
                num_warps=4,
            )

        def run_reduce_dq() -> None:
            reduce_kernel[grid_reduce](
                dq_partials,
                row["dq"],
                shape.fused_chunk,
                shape.N,
                block_j,
                i_blocks,
                num_warps=4,
            )

        def run_row_all() -> None:
            run_row_compute()
            run_reduce_dq()

        run_full()
        run_row_all()
        torch.cuda.synchronize()

        check_owners = shape.owners if shape.name == "smoke_p128" else min(shape.owners, 64)
        ref_check = {name: tensor[:check_owners] for name, tensor in ref.items()}
        row_check = {name: tensor[:check_owners] for name, tensor in row.items()}
        torch_ref = _torch_m_surface_reference(tensors, shape, check_owners)

        result.update(
            {
                "status": "ok",
                "correctness_checked_owners": check_owners,
                "correctness": {
                    "full_m_surface_vs_torch": _compare(torch_ref, ref_check),
                    "row_owner_vs_full_m_surface": _compare(ref_check, row_check),
                },
                "elapsed": {
                    "full_m_surface_ms": _time_cuda_events(run_full, warmup=warmup, iters=iters),
                    "row_owner_compute_only_ms": _time_cuda_events(run_row_compute, warmup=warmup, iters=iters),
                    "reduce_dq_only_ms": _time_cuda_events(run_reduce_dq, warmup=warmup, iters=iters),
                    "row_owner_total_ms": _time_cuda_events(run_row_all, warmup=warmup, iters=iters),
                },
            }
        )
        full_ms = result["elapsed"]["full_m_surface_ms"]["mean_ms"]
        row_ms = result["elapsed"]["row_owner_total_ms"]["mean_ms"]
        compute_ms = result["elapsed"]["row_owner_compute_only_ms"]["mean_ms"]
        if full_ms and row_ms and compute_ms:
            result["speed"] = {
                "row_total_vs_full_m_surface": row_ms / full_ms,
                "full_m_surface_vs_row_total": full_ms / row_ms,
                "row_compute_only_vs_full_m_surface": compute_ms / full_ms,
                "row_total_over_wave2_fullp_dq_dk_diag_productionish": row_ms / 1.0865,
                "row_total_over_stage2_bwd_bwd_productionish": row_ms / 3.6940,
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
def run_probe(
    requested_gpu: str,
    shape_csv: str,
    warmup: int,
    iters: int,
    block_i: int,
    block_j: int,
) -> dict[str, Any]:
    import traceback

    try:
        device = _device_report(requested_gpu)
        results = []
        for shape in _selected_shapes(shape_csv):
            print(
                f"[fused-m-tile-wave5] shape={shape.name} block_i={block_i} block_j={block_j}",
                flush=True,
            )
            results.append(_run_shape(shape, warmup, iters, block_i, block_j))
        return {
            "app_name": APP_NAME,
            "device": device,
            "settings": {
                "shape_csv": shape_csv,
                "block_i": block_i,
                "block_j": block_j,
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
    warmup: int = 1,
    iters: int = 4,
    block_i: int = DEFAULT_BLOCK_I,
    block_j: int = DEFAULT_BLOCK_J,
) -> None:
    result = run_probe.remote(GPU_SPEC, shape_csv, warmup, iters, block_i, block_j)
    print("SUMMARY_JSON_START")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print("SUMMARY_JSON_END")
