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

The intermediate LKQ and apply tiles are BF16 to mirror the current
register-to-shared spill boundary used by the CuTe prototypes.
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


def _run_case(
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
        "diffs": diffs,
        "ideal_dpsi_max_abs": _max_abs(dpsi, ref["ideal_dpsi"]),
        "lkq_checksum": float(lkq.float().sum().item()),
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


def run_lkq_tile_chain(
    *,
    seed: int = 20260430,
    atol: float = 1e-5,
    bench_iters: int = 100,
    bench_warmup: int = 10,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe LKQ tile chain probe")

    print("Wave 5: CuTe LKQ/state tile chain from scalar-copy 64x64 BF16 GEMM")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: chunk={CHUNK_SIZE} rank={RANK} fcs={FCS} N={N} P={P}")
    print(f"atol: {atol}")
    print(f"copy_strategy: scalar BF16 universal copies inherited from SingleGemmWGMMA")

    mask_bf16 = _future_mask_bf16()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile the shared 64x64x64 tile before collecting case diagnostics.
    t0 = time.time()
    first = _make_cases(seed)[0]
    warm = torch.empty((FCS, FCS), dtype=torch.bfloat16, device="cuda")
    run_single_gemm(FCS, FCS, N, first.k, first.q, warm, stream)
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    print(f"compile_plus_first_lkq_launch_s: {compile_s:.3f}")

    cases: dict[str, Any] = {}
    passed = True
    for case in _make_cases(seed):
        result = _run_case(case, mask_bf16, stream)
        cases[case.name] = result
        case_pass = all(value <= atol for value in result["diffs"].values())
        passed = passed and case_pass
        print(f"Case {case.name}: {'PASS' if case_pass else 'FAIL'}")
        for name, value in result["diffs"].items():
            print(f"  {name}: max_abs={value:.6f}")
        print(f"  dpsi_row0[:4]: {result['dpsi_row0']}")
        print(f"  ref_row0[:4]:  {result['ref_dpsi_row0']}")
        print(f"  ideal_dpsi_bf16_chain_delta={result['ideal_dpsi_max_abs']:.6f}")

    timings = None
    if bench_iters > 0:
        timings = _time_chain(
            _make_cases(seed)[-1],
            mask_bf16,
            stream,
            warmup=bench_warmup,
            iters=bench_iters,
        )
        print(
            f"Timing: {timings['chain_us']:.3f} us/chain "
            f"({bench_iters} iters, includes torch mask)"
        )
        print(f"Estimated tile throughput: {timings['estimated_tile_tflops']:.4f} TFLOP/s")

    print(f"{'PASS' if passed else 'FAIL'}: LKQ/state chain correctness")
    return {
        "passed": bool(passed),
        "atol": atol,
        "compile_plus_first_lkq_launch_s": compile_s,
        "copy_strategy": "scalar_bf16_universal_g2s_s2g",
        "shape": {
            "chunk": CHUNK_SIZE,
            "rank": RANK,
            "fcs": FCS,
            "N": N,
            "P": P,
        },
        "cases": cases,
        "timings": timings,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_lkq_tile_chain(), indent=2, sort_keys=True))
