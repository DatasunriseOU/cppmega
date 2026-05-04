"""Wave10 Triton tensorized state/LKQ/D prototype for Mamba3 bwd_bwd.

Wave9 proved the state/LKQ/D ownership but used scalar CUDA loops.  This file
targets the largest remaining tensor-core-shaped producer:

* state: ``K[64,64] @ dstates[64,P]``;
* LKQ: ``K[64,64] @ Q[64,64].T``;
* LKQ apply: ``masked(LKQ)[64,64] @ dPhiO[64,P]``;
* direct D: ``D[h] * dPhiO``.

The Triton kernel owns one ``(B, H, chunk, P tile)`` program, uses ``tl.dot``
for the three matrix products, then writes the same DV/DD/DMIMO partial outputs
as the wave9 CUDA skeleton.  It intentionally does not include any scalar-loop
CUDA implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class Shape:
    B: int
    S: int
    H: int
    G: int
    N: int
    P: int
    R: int = 4
    chunk: int = 16

    @property
    def nchunks(self) -> int:
        return math.ceil(self.S / self.chunk)

    @property
    def fcs(self) -> int:
        return self.chunk * self.R


PRESETS: dict[str, dict[str, int]] = {
    "smoke": {"B": 1, "S": 256, "H": 4, "G": 1, "N": 64, "P": 64},
    "representative": {"B": 2, "S": 1024, "H": 16, "G": 1, "N": 64, "P": 64},
    "productionish": {"B": 4, "S": 4096, "H": 32, "G": 1, "N": 64, "P": 128},
}

COMPARISON_CONTEXT: dict[str, Any] = {
    "wave8_diag_qk_dv_qk_dmimov_target_ms": 2.45093,
    "wave9_scalar_state_lkq_d_two_pass_ms": 27.05544,
    "tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms": 3.70674,
}


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _shape_from_args(args: argparse.Namespace) -> Shape:
    if args.shape:
        values = PRESETS[args.shape]
        return Shape(
            B=values["B"],
            S=values["S"],
            H=values["H"],
            G=values["G"],
            N=values["N"],
            P=values["P"],
            R=args.R,
            chunk=args.chunk,
        )
    return Shape(B=args.B, S=args.S, H=args.H, G=args.G, N=args.N, P=args.P, R=args.R, chunk=args.chunk)


def _stats(values_ms: list[float]) -> dict[str, Any]:
    values = sorted(values_ms)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "min_ms": values[0],
        "p50_ms": values[len(values) // 2],
        "max_ms": values[-1],
    }


def _time_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return _stats(times)


def _time_wall(fn: Callable[[], object], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return _stats(times)


def _randn(
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    *size: int,
    scale: float = 0.01,
) -> torch.Tensor:
    return (torch.randn(size, device=device, dtype=dtype, generator=generator) * scale).contiguous()


def make_prepared_inputs(shape: Shape, *, dtype: torch.dtype, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    fcs = shape.fcs
    return {
        "q": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "k": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.N),
        "dstates": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, shape.N, shape.P),
        "dphi": _randn(generator, device, dtype, shape.B, shape.H, shape.nchunks, fcs, shape.P),
        "v": _randn(generator, device, dtype, shape.B, shape.S, shape.H, shape.P),
        "mimo_v": _randn(generator, device, torch.float32, shape.H, shape.R, shape.P),
        "exp_rev": torch.exp(
            _randn(generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, scale=0.01)
        ).contiguous(),
        "segsum": _randn(
            generator, device, torch.float32, shape.B, shape.H, shape.nchunks, shape.chunk, shape.chunk, scale=0.01
        ),
        "D": _randn(generator, device, torch.float32, shape.H),
    }


@torch.no_grad()
def state_lkq_d_reference(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    lkq_apply_dtype: str = "fp32",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference matching the isolated state/LKQ/D contribution."""

    dtype = inputs["q"].dtype
    fcs = shape.fcs
    total = shape.B * shape.H * shape.nchunks
    q = inputs["q"].float().reshape(total, fcs, shape.N)
    k = inputs["k"].float().reshape(total, fcs, shape.N)
    dstates = inputs["dstates"].float().reshape(total, shape.N, shape.P)
    dphi = inputs["dphi"].float().reshape(total, fcs, shape.P)

    state = torch.bmm(k, dstates)
    exp_rev = inputs["exp_rev"].float().reshape(total, shape.chunk)
    state = state * exp_rev.repeat_interleave(shape.R, dim=-1).unsqueeze(-1)

    lkq = torch.bmm(k, q.transpose(1, 2))
    ci = torch.arange(fcs, device=q.device) // shape.R
    causal = ci[:, None] < ci[None, :]
    seg = inputs["segsum"].float().reshape(total, shape.chunk, shape.chunk)
    seg_weight = seg[
        :,
        ci[None, :].expand(fcs, fcs),
        ci[:, None].expand(fcs, fcs),
    ]
    lkq = torch.where(causal[None, :, :], lkq * torch.exp(seg_weight), torch.zeros_like(lkq))
    if lkq_apply_dtype == "bf16":
        lkq = lkq.to(dtype).float()
    elif lkq_apply_dtype != "fp32":
        raise ValueError(f"unsupported lkq_apply_dtype: {lkq_apply_dtype}")
    lkq_dpsi = torch.bmm(lkq, dphi)

    d_per_total = (
        inputs["D"].float()[None, :, None]
        .expand(shape.B, shape.H, shape.nchunks)
        .reshape(total)[:, None, None]
    )
    dpsi = (state + lkq_dpsi + d_per_total * dphi).to(dtype).float()

    dpsi_bh = dpsi.reshape(shape.B, shape.H, shape.nchunks, shape.chunk, shape.R, shape.P)
    dv = (
        dpsi_bh
        * inputs["mimo_v"].float()[None, :, None, None, :, :]
    ).sum(dim=4)
    dv = dv.permute(0, 2, 3, 1, 4).reshape(shape.B, shape.S, shape.H, shape.P).to(dtype).contiguous()

    v = inputs["v"].float().reshape(shape.B, shape.nchunks, shape.chunk, shape.H, shape.P)
    v = v.permute(0, 3, 1, 2, 4).contiguous()
    dmimo_v = (dpsi_bh * v[:, :, :, :, None, :]).sum(dim=(2, 3))

    dd = inputs["dphi"].float().sum(dim=(2, 3, 4))
    return dv, dd.contiguous(), dmimo_v.contiguous()


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


if _has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _state_lkq_d_kernel(
        Q,
        K,
        DSTATES,
        DPHI,
        V,
        MIMO_V,
        EXP_REV,
        SEGSUM,
        D,
        DV,
        DD,
        DMIMO_PARTIALS,
        total_programs: tl.constexpr,
        n_pblocks: tl.constexpr,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        R: tl.constexpr,
        CHUNK: tl.constexpr,
        NCHUNKS: tl.constexpr,
        FCS: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        LKQ_APPLY_BF16: tl.constexpr,
        WRITE_PARTIALS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        p_block = pid % n_pblocks
        chunk = (pid // n_pblocks) % NCHUNKS
        bh = pid // (n_pblocks * NCHUNKS)
        h = bh % H
        b = bh // H

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_c = tl.arange(0, CHUNK)
        offs_r = tl.arange(0, R)

        qk_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * N
        dstate_base = (((b * H + h) * NCHUNKS + chunk) * N) * P
        dphi_base = (((b * H + h) * NCHUNKS + chunk) * FCS) * P
        exp_base = ((b * H + h) * NCHUNKS + chunk) * CHUNK
        seg_base = (((b * H + h) * NCHUNKS + chunk) * CHUNK) * CHUNK
        mimo_base = h * R * P

        k_mat = tl.load(
            K + qk_base + offs_m[:, None] * N + offs_n[None, :],
            mask=(offs_m[:, None] < FCS) & (offs_n[None, :] < N),
            other=0.0,
        )
        dst_mat = tl.load(
            DSTATES + dstate_base + offs_n[:, None] * P + offs_p[None, :],
            mask=(offs_n[:, None] < N) & (offs_p[None, :] < P),
            other=0.0,
        )
        state = tl.dot(k_mat, dst_mat, input_precision="tf32", out_dtype=tl.float32)

        row_c = offs_m // R
        exp_scale = tl.load(EXP_REV + exp_base + row_c, mask=offs_m < FCS, other=0.0)
        state = state * exp_scale[:, None]

        q_t = tl.load(
            Q + qk_base + offs_m[None, :] * N + offs_n[:, None],
            mask=(offs_m[None, :] < FCS) & (offs_n[:, None] < N),
            other=0.0,
        )
        lkq = tl.dot(k_mat, q_t, input_precision="tf32", out_dtype=tl.float32)
        col_c = offs_m
        col_c = col_c // R
        causal = row_c[:, None] < col_c[None, :]
        seg = tl.load(
            SEGSUM + seg_base + col_c[None, :] * CHUNK + row_c[:, None],
            mask=causal,
            other=0.0,
        )
        lkq = tl.where(causal, lkq * tl.exp(seg), 0.0)

        dphi_mat = tl.load(
            DPHI + dphi_base + offs_m[:, None] * P + offs_p[None, :],
            mask=(offs_m[:, None] < FCS) & (offs_p[None, :] < P),
            other=0.0,
        )
        if LKQ_APPLY_BF16:
            lkq_dphi = tl.dot(lkq.to(dphi_mat.dtype), dphi_mat, input_precision="tf32", out_dtype=tl.float32)
        else:
            lkq_dphi = tl.dot(lkq, dphi_mat.to(tl.float32), input_precision="tf32", out_dtype=tl.float32)

        d_h = tl.load(D + h)
        dpsi = state + lkq_dphi + d_h * dphi_mat.to(tl.float32)
        dpsi = dpsi.to(dphi_mat.dtype).to(tl.float32)

        dpsi_crp = tl.reshape(dpsi, (CHUNK, R, BLOCK_P))
        mimo = tl.load(
            MIMO_V + mimo_base + offs_r[None, :, None] * P + offs_p[None, None, :],
            mask=offs_p[None, None, :] < P,
            other=0.0,
        ).to(tl.float32)
        dv_tile = tl.sum(dpsi_crp * mimo, axis=1)

        s_idx = chunk * CHUNK + offs_c
        dv_base = ((b * S + s_idx[:, None]) * H + h) * P + offs_p[None, :]
        tl.store(DV + dv_base, dv_tile, mask=(s_idx[:, None] < S) & (offs_p[None, :] < P))

        dd_part = tl.sum(tl.sum(dphi_mat.to(tl.float32), axis=0), axis=0)
        tl.atomic_add(DD + b * H + h, dd_part, sem="relaxed")

        if WRITE_PARTIALS:
            v_tile = tl.load(
                V + ((b * S + s_idx[:, None]) * H + h) * P + offs_p[None, :],
                mask=(s_idx[:, None] < S) & (offs_p[None, :] < P),
                other=0.0,
            ).to(tl.float32)
            dmimo_tile = tl.sum(dpsi_crp * v_tile[:, None, :], axis=0)
            partial_base = (((b * H + h) * NCHUNKS + chunk) * R) * P
            tl.store(
                DMIMO_PARTIALS + partial_base + offs_r[:, None] * P + offs_p[None, :],
                dmimo_tile,
                mask=offs_p[None, :] < P,
            )

    @triton.jit
    def _reduce_dmimo_partials_kernel(
        PARTIALS,
        DMIMO_V,
        B: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        NCHUNKS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        pid = tl.program_id(0)
        p_block = tl.program_id(1)
        r = pid % R
        bh = pid // R
        h = bh % H
        b = bh // H
        offs_c = tl.arange(0, BLOCK_CHUNKS)
        offs_p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        base = (((b * H + h) * NCHUNKS) * R + r) * P
        vals = tl.load(
            PARTIALS + base + offs_c[:, None] * R * P + offs_p[None, :],
            mask=(offs_c[:, None] < NCHUNKS) & (offs_p[None, :] < P),
            other=0.0,
        )
        out = tl.sum(vals, axis=0)
        tl.store(
            DMIMO_V + ((b * H + h) * R + r) * P + offs_p,
            out,
            mask=offs_p < P,
        )


def state_lkq_d_triton(
    inputs: dict[str, torch.Tensor],
    shape: Shape,
    *,
    block_p: int,
    num_warps: int,
    lkq_apply_dtype: str,
    write_partials: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not _has_triton():
        raise RuntimeError("triton is not importable")
    if not inputs["q"].is_cuda:
        raise RuntimeError("triton path requires CUDA tensors")
    if shape.chunk != 16 or shape.R != 4 or shape.N != 64:
        raise ValueError("wave10 prototype specializes chunk=16, R=4, N=64")

    import triton

    dv = torch.empty_like(inputs["v"])
    dd = torch.empty(shape.B, shape.H, device=inputs["q"].device, dtype=torch.float32)
    dd.zero_()
    partials = None
    if write_partials:
        partials = torch.empty(
            shape.B, shape.H, shape.nchunks, shape.R, shape.P, device=inputs["q"].device, dtype=torch.float32
        )

    n_pblocks = triton.cdiv(shape.P, block_p)
    grid = (shape.B * shape.H * shape.nchunks * n_pblocks,)
    _state_lkq_d_kernel[grid](
        inputs["q"],
        inputs["k"],
        inputs["dstates"],
        inputs["dphi"],
        inputs["v"],
        inputs["mimo_v"],
        inputs["exp_rev"],
        inputs["segsum"],
        inputs["D"],
        dv,
        dd,
        partials if partials is not None else dv,
        grid[0],
        n_pblocks,
        shape.B,
        shape.S,
        shape.H,
        shape.N,
        shape.P,
        shape.R,
        shape.chunk,
        shape.nchunks,
        shape.fcs,
        block_p,
        shape.fcs,
        shape.N,
        lkq_apply_dtype == "bf16",
        write_partials,
        num_warps=num_warps,
    )
    return dv, dd, partials


def reduce_dmimo_partials_triton(
    partials: torch.Tensor,
    shape: Shape,
    *,
    block_p: int,
    block_chunks: int,
    num_warps: int,
) -> torch.Tensor:
    if not _has_triton():
        raise RuntimeError("triton is not importable")
    import triton

    dmimo = torch.empty(shape.B, shape.H, shape.R, shape.P, device=partials.device, dtype=torch.float32)
    grid = (shape.B * shape.H * shape.R, triton.cdiv(shape.P, block_p))
    _reduce_dmimo_partials_kernel[grid](
        partials,
        dmimo,
        shape.B,
        shape.H,
        shape.R,
        shape.P,
        shape.nchunks,
        block_chunks,
        block_p,
        num_warps=num_warps,
    )
    return dmimo


def _max_diff(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


def _operation_model(shape: Shape, *, block_p: int) -> dict[str, Any]:
    chunks = shape.B * shape.H * shape.nchunks
    pblocks = math.ceil(shape.P / block_p)
    fcs = shape.fcs
    causal_entries = shape.chunk * (shape.chunk - 1) // 2 * shape.R * shape.R
    useful_fma = {
        "state_fma": chunks * fcs * shape.N * shape.P,
        "causal_lkq_fma": chunks * causal_entries * shape.N,
        "causal_lkq_apply_fma": chunks * causal_entries * shape.P,
    }
    executed_dot_fma = {
        "state_dot_fma": chunks * fcs * shape.N * shape.P,
        "full_lkq_dot_fma_with_p_tile_recompute": chunks * pblocks * fcs * shape.N * fcs,
        "masked_lkq_apply_dot_fma": chunks * fcs * fcs * shape.P,
    }
    partial_bytes = shape.B * shape.H * shape.nchunks * shape.R * shape.P * 4
    return {
        "chunks": chunks,
        "pblocks": pblocks,
        "block_p": block_p,
        "fused_chunk_size": fcs,
        "causal_lkq_entries_per_chunk": causal_entries,
        "useful_state_lkq_fma": useful_fma,
        "useful_state_lkq_total_fma": sum(useful_fma.values()),
        "executed_dot_fma_in_this_triton_prototype": executed_dot_fma,
        "executed_dot_total_fma_in_this_triton_prototype": sum(executed_dot_fma.values()),
        "dmimov_partial_mib": partial_bytes / (1024**2),
        "dmimov_partial_extra_global_rw_mib": (partial_bytes * 2 + shape.B * shape.H * shape.R * shape.P * 4)
        / (1024**2),
    }


def _projection(timings: dict[str, Any]) -> None:
    base = COMPARISON_CONTEXT["wave8_diag_qk_dv_qk_dmimov_target_ms"]
    tilelang = COMPARISON_CONTEXT["tilelang_stage2_bf1_bb0_productionish_bwd_bwd_ms"]
    for item in timings.values():
        mean_ms = item.get("mean_ms") if isinstance(item, dict) else None
        if mean_ms is None:
            continue
        item["projected_total_with_wave8_target_ms"] = base + mean_ms
        item["projected_total_ratio_vs_tilelang"] = (base + mean_ms) / tilelang
        item["remaining_budget_after_wave8_ms"] = tilelang - base


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = _shape_from_args(args)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    inputs = make_prepared_inputs(shape, dtype=dtype, device=device, seed=args.seed)
    timer = _time_cuda if device.type == "cuda" else _time_wall

    correctness: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    ref_dv = ref_dd = ref_dmimo = None
    if not args.skip_reference:
        ref_dv, ref_dd, ref_dmimo = state_lkq_d_reference(inputs, shape, lkq_apply_dtype=args.lkq_apply_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        dv, dd, partials = state_lkq_d_triton(
            inputs,
            shape,
            block_p=args.block_p,
            num_warps=args.num_warps,
            lkq_apply_dtype=args.lkq_apply_dtype,
            write_partials=True,
        )
        if partials is None:
            raise RuntimeError("partials were requested but not produced")
        dmimo = reduce_dmimo_partials_triton(
            partials,
            shape,
            block_p=args.reduce_block_p,
            block_chunks=args.reduce_block_chunks,
            num_warps=args.reduce_num_warps,
        )
        torch.cuda.synchronize()

        if ref_dv is not None and ref_dd is not None and ref_dmimo is not None:
            correctness["state_lkq_d_triton_vs_torch_reference"] = {
                "dv_delta": _max_diff(ref_dv, dv),
                "dd_delta": _max_diff(ref_dd, dd),
                "dmimo_v_delta": _max_diff(ref_dmimo, dmimo),
            }

        dv_out = torch.empty_like(inputs["v"])
        dd_out = torch.empty(shape.B, shape.H, device=device, dtype=torch.float32)
        partials_out = torch.empty(
            shape.B, shape.H, shape.nchunks, shape.R, shape.P, device=device, dtype=torch.float32
        )
        dmimo_out = torch.empty(shape.B, shape.H, shape.R, shape.P, device=device, dtype=torch.float32)

        def run_producer_with_partials() -> None:
            dd_out.zero_()
            n_pblocks = math.ceil(shape.P / args.block_p)
            _state_lkq_d_kernel[(shape.B * shape.H * shape.nchunks * n_pblocks,)](
                inputs["q"],
                inputs["k"],
                inputs["dstates"],
                inputs["dphi"],
                inputs["v"],
                inputs["mimo_v"],
                inputs["exp_rev"],
                inputs["segsum"],
                inputs["D"],
                dv_out,
                dd_out,
                partials_out,
                shape.B * shape.H * shape.nchunks * n_pblocks,
                n_pblocks,
                shape.B,
                shape.S,
                shape.H,
                shape.N,
                shape.P,
                shape.R,
                shape.chunk,
                shape.nchunks,
                shape.fcs,
                args.block_p,
                shape.fcs,
                shape.N,
                args.lkq_apply_dtype == "bf16",
                True,
                num_warps=args.num_warps,
            )

        def run_reduce() -> None:
            _reduce_dmimo_partials_kernel[(shape.B * shape.H * shape.R, math.ceil(shape.P / args.reduce_block_p))](
                partials_out,
                dmimo_out,
                shape.B,
                shape.H,
                shape.R,
                shape.P,
                shape.nchunks,
                args.reduce_block_chunks,
                args.reduce_block_p,
                num_warps=args.reduce_num_warps,
            )

        def run_two_pass() -> None:
            run_producer_with_partials()
            run_reduce()

        run_producer_with_partials()
        run_reduce()
        torch.cuda.synchronize()

        timings["state_lkq_d_triton_producer_dv_dd_dmimov_partials"] = timer(
            run_producer_with_partials,
            warmup=args.warmup,
            iters=args.iters,
        )
        timings["state_lkq_d_triton_reduce_dmimov_partials"] = timer(
            run_reduce,
            warmup=args.warmup,
            iters=args.iters,
        )
        timings["state_lkq_d_triton_two_pass_total"] = timer(
            run_two_pass,
            warmup=args.warmup,
            iters=args.iters,
        )
        if shape.P <= args.block_p:
            def run_no_partials() -> None:
                dd_out.zero_()
                n_pblocks = math.ceil(shape.P / args.block_p)
                _state_lkq_d_kernel[(shape.B * shape.H * shape.nchunks * n_pblocks,)](
                    inputs["q"],
                    inputs["k"],
                    inputs["dstates"],
                    inputs["dphi"],
                    inputs["v"],
                    inputs["mimo_v"],
                    inputs["exp_rev"],
                    inputs["segsum"],
                    inputs["D"],
                    dv_out,
                    dd_out,
                    partials_out,
                    shape.B * shape.H * shape.nchunks * n_pblocks,
                    n_pblocks,
                    shape.B,
                    shape.S,
                    shape.H,
                    shape.N,
                    shape.P,
                    shape.R,
                    shape.chunk,
                    shape.nchunks,
                    shape.fcs,
                    args.block_p,
                    shape.fcs,
                    shape.N,
                    args.lkq_apply_dtype == "bf16",
                    False,
                    num_warps=args.num_warps,
                )

            timings["state_lkq_d_triton_dv_dd_no_dmimov_partials"] = timer(
                run_no_partials,
                warmup=args.warmup,
                iters=args.iters,
            )

    if (
        args.shape == "productionish"
        or (shape.B, shape.S, shape.H, shape.N, shape.P, shape.R, shape.chunk) == (4, 4096, 32, 64, 128, 4, 16)
    ):
        _projection(timings)

    return {
        "shape_name": args.shape or "custom",
        "shape": asdict(shape),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "triton_importable": _has_triton(),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "block_p": args.block_p,
        "num_warps": args.num_warps,
        "lkq_apply_dtype": args.lkq_apply_dtype,
        "comparison_context": COMPARISON_CONTEXT,
        "operation_model": _operation_model(shape, block_p=args.block_p),
        "correctness": correctness,
        "timings": timings,
        "read": [
            "Triton path uses tl.dot for state, LKQ construction, and LKQ application.",
            "One program owns a P tile, so LKQ is recomputed once per P tile; a final CUDA/CuTe design must compute LKQ once per chunk and reuse it across P tiles.",
            "The producer writes DMIMO_V per-chunk partials, matching wave9's viable ownership for the non-qk path.",
            "This benchmark excludes the remaining DK/DQ state+intra paths and scalar DDA/DSSDA/DFACTOR/DANGLES consumers.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(PRESETS), default=None)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--G", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=128)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--block-p", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--lkq-apply-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--reduce-block-p", type=int, default=64)
    parser.add_argument("--reduce-block-chunks", type=int, default=256)
    parser.add_argument("--reduce-num-warps", type=int, default=8)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
