"""CuTe LKQ/state chain probe built from the correct 64x64 BF16 GEMM tile.

This is a bounded scan-owner composition probe, not a production fused kernel.
It keeps Wave 4's scalar-copy CuTe GEMM as the correctness oracle and composes
the chunk-local pieces that the monolithic owner must eventually keep in one
CTA:

  1. state = K @ DStates
  2. lkq = K @ Q^T
  3. apply = future_mask(lkq) @ dPhi
  4. dpsi = state + apply
  5. DV / DMIMO_V scalar consumers from dpsi

The scalar correctness mode keeps the Wave 5 LKQ/masked-LKQ global tensors.
The fused mode uses a two-WGMMA CuTe tile for ``future_mask(lkq) @ dPhi`` so
LKQ is only spilled to swizzled shared memory inside the kernel.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cuda.bindings.driver as cuda
import torch

from cppmega.megatron.cute_dsl_mimo.masked_lkq_apply import run_masked_lkq_apply
from cppmega.megatron.cute_dsl_mimo.single_gemm_test import run_single_gemm


CHUNK_SIZE = 16
RANK = 4
FCS = CHUNK_SIZE * RANK
N = 64
P = 64


@dataclass(frozen=True)
class ChainInputs:
    name: str
    k: torch.Tensor
    q: torch.Tensor
    dstates: torch.Tensor
    dphi: torch.Tensor
    v: torch.Tensor
    mimo_v: torch.Tensor


def _structured(rows: int, cols: int, *, a: int, b: int, mod: int, denom: float) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float32, device="cuda")[:, None]
    col = torch.arange(cols, dtype=torch.float32, device="cuda")[None, :]
    return (((row * a + col * b) % mod) - (mod // 2)) / denom


def _make_cases(seed: int) -> list[ChainInputs]:
    torch.manual_seed(seed)
    structured = ChainInputs(
        name="structured_mod",
        k=_structured(FCS, N, a=3, b=5, mod=17, denom=8.0).to(torch.bfloat16),
        q=_structured(FCS, N, a=7, b=-2, mod=19, denom=9.0).to(torch.bfloat16),
        dstates=_structured(N, P, a=11, b=3, mod=23, denom=11.0).to(torch.bfloat16),
        dphi=_structured(FCS, P, a=-5, b=13, mod=29, denom=13.0).to(torch.bfloat16),
        v=_structured(CHUNK_SIZE, P, a=2, b=7, mod=31, denom=32.0).to(torch.bfloat16),
        mimo_v=_structured(RANK, P, a=17, b=-3, mod=37, denom=64.0).to(torch.bfloat16),
    )

    scale = 0.125
    random = ChainInputs(
        name="random_seed",
        k=(torch.randn(FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        q=(torch.randn(FCS, N, dtype=torch.bfloat16, device="cuda") * scale),
        dstates=(torch.randn(N, P, dtype=torch.bfloat16, device="cuda") * scale),
        dphi=(torch.randn(FCS, P, dtype=torch.bfloat16, device="cuda") * scale),
        v=(torch.randn(CHUNK_SIZE, P, dtype=torch.bfloat16, device="cuda") * scale),
        mimo_v=(torch.randn(RANK, P, dtype=torch.bfloat16, device="cuda") * scale),
    )
    return [structured, random]


def _future_mask_bf16() -> torch.Tensor:
    f = torch.arange(FCS, device="cuda")
    t = f // RANK
    mask = t[:, None] < t[None, :]
    return mask.to(torch.bfloat16)


def _run_tile(a: torch.Tensor, b_nk: torch.Tensor, stream: cuda.CUstream) -> torch.Tensor:
    """Run C = A @ B^T through the Wave 4 scalar-copy CuTe tile."""

    m, k = a.shape
    n, b_k = b_nk.shape
    if k != b_k:
        raise ValueError(f"K mismatch: A={tuple(a.shape)} B={tuple(b_nk.shape)}")
    c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    run_single_gemm(m, n, k, a.contiguous(), b_nk.contiguous(), c, stream)
    return c


def _scalar_consumers(
    dpsi: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dpsi_t = dpsi.float().view(CHUNK_SIZE, RANK, P)
    dv = (dpsi_t * mimo_v.float()[None, :, :]).sum(dim=1)
    dmimo_v = (dpsi_t * v.float()[:, None, :]).sum(dim=0)
    return dv, dmimo_v


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _reference(inputs: ChainInputs, mask_bf16: torch.Tensor) -> dict[str, torch.Tensor]:
    state = (inputs.k.float() @ inputs.dstates.float()).to(torch.bfloat16)
    lkq = (inputs.k.float() @ inputs.q.float().T).to(torch.bfloat16)
    masked_lkq = (lkq.float() * mask_bf16.float()).to(torch.bfloat16)
    apply = (masked_lkq.float() @ inputs.dphi.float()).to(torch.bfloat16)
    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ideal_dpsi = inputs.k.float() @ inputs.dstates.float()
    ideal_dpsi += ((inputs.k.float() @ inputs.q.float().T) * mask_bf16.float()) @ inputs.dphi.float()
    return {
        "state": state,
        "lkq": lkq,
        "masked_lkq": masked_lkq,
        "apply": apply,
        "dpsi": dpsi,
        "dv": dv,
        "dmimo_v": dmimo_v,
        "ideal_dpsi": ideal_dpsi,
    }


def _run_case_scalar(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    state = _run_tile(inputs.k, inputs.dstates.T.contiguous(), stream)
    lkq = _run_tile(inputs.k, inputs.q, stream)
    torch.cuda.synchronize()

    masked_lkq = (lkq.float() * mask_bf16.float()).to(torch.bfloat16)
    apply = _run_tile(masked_lkq, inputs.dphi.T.contiguous(), stream)
    torch.cuda.synchronize()

    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ref = _reference(inputs, mask_bf16)

    diffs = {
        "state": _max_abs(state, ref["state"]),
        "lkq": _max_abs(lkq, ref["lkq"]),
        "masked_lkq": _max_abs(masked_lkq, ref["masked_lkq"]),
        "apply": _max_abs(apply, ref["apply"]),
        "dpsi": _max_abs(dpsi, ref["dpsi"]),
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "scalar_copy_cute_tiles_plus_torch_mask",
        "lkq_global_materialized": True,
        "diffs": diffs,
        "ideal_dpsi_max_abs": _max_abs(dpsi, ref["ideal_dpsi"]),
        "lkq_checksum": float(lkq.float().sum().item()),
        "dpsi_checksum": float(dpsi.float().sum().item()),
        "dpsi_row0": [float(x) for x in dpsi[0, :4].tolist()],
        "ref_dpsi_row0": [float(x) for x in ref["dpsi"][0, :4].tolist()],
    }


def _run_case_fused_masked_apply(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
) -> dict[str, Any]:
    state = _run_tile(inputs.k, inputs.dstates.T.contiguous(), stream)
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    run_masked_lkq_apply(
        FCS,
        RANK,
        inputs.k.contiguous(),
        inputs.q.contiguous(),
        inputs.dphi.T.contiguous(),
        apply,
        stream,
    )
    torch.cuda.synchronize()

    dpsi = state.float() + apply.float()
    dv, dmimo_v = _scalar_consumers(dpsi, inputs.v, inputs.mimo_v)
    ref = _reference(inputs, mask_bf16)

    diffs = {
        "state": _max_abs(state, ref["state"]),
        "apply": _max_abs(apply, ref["apply"]),
        "dpsi": _max_abs(dpsi, ref["dpsi"]),
        "dv": _max_abs(dv, ref["dv"]),
        "dmimo_v": _max_abs(dmimo_v, ref["dmimo_v"]),
    }
    return {
        "mode": "state_scalar_plus_fused_masked_lkq_apply",
        "lkq_global_materialized": False,
        "remaining_global": [
            "state BF16 tile from scalar-copy CuTe GEMM",
            "apply BF16 tile output from fused masked-apply kernel",
            "dpsi/DV/DMIMO_V torch-side scalar correctness consumers",
        ],
        "diffs": diffs,
        "ideal_dpsi_max_abs": _max_abs(dpsi, ref["ideal_dpsi"]),
        "dpsi_checksum": float(dpsi.float().sum().item()),
        "dpsi_row0": [float(x) for x in dpsi[0, :4].tolist()],
        "ref_dpsi_row0": [float(x) for x in ref["dpsi"][0, :4].tolist()],
    }


def _time_chain(
    inputs: ChainInputs,
    mask_bf16: torch.Tensor,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    state = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    lkq = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    masked_lkq = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    dstates_t = inputs.dstates.T.contiguous()
    dphi_t = inputs.dphi.T.contiguous()

    def launch_once() -> None:
        run_single_gemm(FCS, P, N, inputs.k, dstates_t, state, stream)
        run_single_gemm(FCS, FCS, N, inputs.k, inputs.q, lkq, stream)
        # Keep this benchmark path conservative: the torch mask is a temporary
        # stand-in for an eventual in-kernel masked epilogue, so place explicit
        # CUDA barriers around the framework/CuTe handoff.
        torch.cuda.synchronize()
        torch.mul(lkq, mask_bf16, out=masked_lkq)
        torch.cuda.synchronize()
        run_single_gemm(FCS, P, FCS, masked_lkq, dphi_t, apply, stream)

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_iter_us = elapsed_us / iters
    flops = 2 * FCS * P * N + 2 * FCS * FCS * N + 2 * FCS * P * FCS
    return {
        "mode": "three_scalar_copy_cute_tiles_plus_torch_mask",
        "warmup": warmup,
        "iters": iters,
        "chain_us": per_iter_us,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_iter_us * 1e-6) / 1e12),
    }


def _time_chain_fused_masked_apply(
    inputs: ChainInputs,
    stream: cuda.CUstream,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    state = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    apply = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    dstates_t = inputs.dstates.T.contiguous()
    dphi_t = inputs.dphi.T.contiguous()

    def launch_once() -> None:
        run_single_gemm(FCS, P, N, inputs.k, dstates_t, state, stream)
        run_masked_lkq_apply(FCS, RANK, inputs.k, inputs.q, dphi_t, apply, stream)

    for _ in range(warmup):
        launch_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch_once()
    end.record()
    torch.cuda.synchronize()

    elapsed_us = start.elapsed_time(end) * 1000.0
    per_iter_us = elapsed_us / iters
    flops = 2 * FCS * P * N + 2 * FCS * FCS * N + 2 * FCS * P * FCS
    return {
        "mode": "state_scalar_plus_fused_masked_lkq_apply",
        "warmup": warmup,
        "iters": iters,
        "chain_us": per_iter_us,
        "estimated_tile_flops": flops,
        "estimated_tile_tflops": float(flops / (per_iter_us * 1e-6) / 1e12),
        "lkq_global_materialized": False,
        "launches_per_chain": 2,
    }


def run_lkq_tile_chain(
    *,
    seed: int = 20260430,
    atol: float = 1e-5,
    bench_iters: int = 100,
    bench_warmup: int = 10,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe LKQ tile chain probe")

    print("Wave 6: CuTe LKQ/state chain with fused masked-apply tile")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: chunk={CHUNK_SIZE} rank={RANK} fcs={FCS} N={N} P={P}")
    print(f"atol: {atol}")
    print(f"copy_strategy: scalar BF16 universal copies inherited from SingleGemmWGMMA")
    print("fused_path: LKQ is R2S-spilled to swizzled smem only, no LKQ gmem output")

    mask_bf16 = _future_mask_bf16()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile the shared 64x64x64 tile before collecting case diagnostics.
    t0 = time.time()
    first = _make_cases(seed)[0]
    warm = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    run_single_gemm(FCS, FCS, N, first.k, first.q, warm, stream)
    fused_warm = torch.empty((FCS, P), dtype=torch.bfloat16, device="cuda")
    run_masked_lkq_apply(
        FCS,
        RANK,
        first.k.contiguous(),
        first.q.contiguous(),
        first.dphi.T.contiguous(),
        fused_warm,
        stream,
    )
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    print(f"compile_plus_first_lkq_and_fused_apply_launch_s: {compile_s:.3f}")

    scalar_cases: dict[str, Any] = {}
    fused_cases: dict[str, Any] = {}
    passed = True
    for case in _make_cases(seed):
        scalar_result = _run_case_scalar(case, mask_bf16, stream)
        scalar_cases[case.name] = scalar_result
        scalar_pass = all(value <= atol for value in scalar_result["diffs"].values())
        passed = passed and scalar_pass
        print(f"Scalar case {case.name}: {'PASS' if scalar_pass else 'FAIL'}")
        for name, value in scalar_result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  dpsi_row0[:4]: {scalar_result['dpsi_row0']}")
        print(f"  ref_row0[:4]:  {scalar_result['ref_dpsi_row0']}")
        print(f"  ideal_dpsi_bf16_chain_delta={scalar_result['ideal_dpsi_max_abs']:.6f}")

        fused_result = _run_case_fused_masked_apply(case, mask_bf16, stream)
        fused_cases[case.name] = fused_result
        fused_pass = all(value <= atol for value in fused_result["diffs"].values())
        passed = passed and fused_pass
        print(f"Fused masked-apply case {case.name}: {'PASS' if fused_pass else 'FAIL'}")
        for name, value in fused_result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  lkq_global_materialized: {fused_result['lkq_global_materialized']}")
        print(f"  dpsi_row0[:4]: {fused_result['dpsi_row0']}")
        print(f"  ref_row0[:4]:  {fused_result['ref_dpsi_row0']}")
        print(f"  ideal_dpsi_bf16_chain_delta={fused_result['ideal_dpsi_max_abs']:.6f}")

    timings = None
    if bench_iters > 0:
        timing_case = _make_cases(seed)[-1]
        scalar_timing = _time_chain(
            timing_case,
            mask_bf16,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        fused_timing = _time_chain_fused_masked_apply(
            timing_case,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        timings = {
            "scalar_copy": scalar_timing,
            "fused_masked_apply": fused_timing,
        }
        print(
            f"Scalar timing: {scalar_timing['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, includes torch mask)"
        )
        print(
            f"Fused masked-apply timing: {fused_timing['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, no LKQ gmem/torch mask)"
        )
        print(
            "Estimated tile throughput: "
            f"scalar={scalar_timing['estimated_tile_tflops']:.4f} TFLOP/s "
            f"fused={fused_timing['estimated_tile_tflops']:.4f} TFLOP/s"
        )

    print(f"{'PASS' if passed else 'FAIL'}: LKQ/state chain correctness")
    return {
        "passed": bool(passed),
        "atol": atol,
        "compile_plus_first_lkq_and_fused_apply_launch_s": compile_s,
        "copy_strategy": "scalar_bf16_universal_g2s_s2g",
        "lkq_global_materialized_for_tested_fused_path": False,
        "fused_path_remaining_global": [
            "state BF16 tile from scalar-copy CuTe GEMM",
            "apply BF16 tile output from fused masked-apply kernel",
            "dpsi/DV/DMIMO_V torch-side scalar correctness consumers",
        ],
        "shape": {
            "chunk": CHUNK_SIZE,
            "rank": RANK,
            "fcs": FCS,
            "N": N,
            "P": P,
        },
        "scalar_cases": scalar_cases,
        "fused_cases": fused_cases,
        "timings": timings,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_lkq_tile_chain(), indent=2, sort_keys=True))
