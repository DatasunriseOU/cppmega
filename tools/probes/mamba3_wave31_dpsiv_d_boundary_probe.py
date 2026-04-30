#!/usr/bin/env python3
"""Wave31 Lane C H100 probe for a small Mamba3 bwd_bwd helper boundary.

Candidate: fuse the post-GEMM dPsiV_D epilogue:

    out_bf16 = bf16(dPsiV_fp32 + dPhi_bf16 * D_fp32
                    + gamma_fp32 * qk_dot_bf16^T @ dPhi_bf16)

The helper deliberately does not own full bwd_bwd. It consumes an already
computed dPsiV accumulator and writes the bf16 boundary tensor used by the
later DV/DPsi paths. It allocates no global scratch and only keeps an R-sized
loop in registers.

This probe is H100-only by policy for Wave31 Lane C. It refuses to build or run
on H200 and on non-H100 devices, including local GB10/B200 hosts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void dpsiv_d_boundary_kernel(
    const float* __restrict__ dpsiv,
    const __nv_bfloat16* __restrict__ dphi,
    const float* __restrict__ d,
    const __nv_bfloat16* __restrict__ qk_dot,
    const float* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int tiles,
    int cs,
    int r,
    int p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tiles * cs * r * p;
    if (idx >= total) {
        return;
    }

    int p_idx = idx % p;
    int fr = (idx / p) % (cs * r);
    int tile = idx / (p * cs * r);
    int c_idx = fr / r;
    int r_out = fr - c_idx * r;

    float acc = dpsiv[idx];
    float dphi_here = __bfloat162float(dphi[idx]);
    acc += dphi_here * d[tile * p + p_idx];

    int qk_base = ((tile * cs + c_idx) * r) * r;
    int dphi_base = (tile * cs * r + c_idx * r) * p + p_idx;
    float qk_acc = 0.0f;
    for (int r_in = 0; r_in < r; ++r_in) {
        float qk = __bfloat162float(qk_dot[qk_base + r_in * r + r_out]);
        float dph = __bfloat162float(dphi[dphi_base + r_in * p]);
        qk_acc += qk * dph;
    }
    acc += gamma[tile * cs + c_idx] * qk_acc;
    out[idx] = __float2bfloat16_rn(acc);
}

}  // namespace

void dpsiv_d_boundary(
    torch::Tensor dpsiv,
    torch::Tensor dphi,
    torch::Tensor d,
    torch::Tensor qk_dot,
    torch::Tensor gamma,
    torch::Tensor out,
    int64_t tiles,
    int64_t cs,
    int64_t r,
    int64_t p
) {
    const int threads = 256;
    const int total = static_cast<int>(tiles * cs * r * p);
    const int blocks = (total + threads - 1) / threads;
    dpsiv_d_boundary_kernel<<<blocks, threads>>>(
        static_cast<const float*>(dpsiv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dphi.data_ptr()),
        static_cast<const float*>(d.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qk_dot.data_ptr()),
        static_cast<const float*>(gamma.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(tiles),
        static_cast<int>(cs),
        static_cast<int>(r),
        static_cast<int>(p)
    );
}
"""

CPP_SRC = r"""
#include <torch/extension.h>

void dpsiv_d_boundary(
    torch::Tensor dpsiv,
    torch::Tensor dphi,
    torch::Tensor d,
    torch::Tensor qk_dot,
    torch::Tensor gamma,
    torch::Tensor out,
    int64_t tiles,
    int64_t cs,
    int64_t r,
    int64_t p
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dpsiv_d_boundary", &dpsiv_d_boundary, "Wave31 dPsiV_D boundary helper");
}
"""


@dataclass
class DeviceGate:
    ok: bool
    reason: str
    name: str | None
    capability: tuple[int, int] | None


def _device_gate() -> DeviceGate:
    if not torch.cuda.is_available():
        return DeviceGate(False, "cuda_unavailable", None, None)
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    capability = torch.cuda.get_device_capability(idx)
    lowered = name.lower()
    if "h200" in lowered:
        return DeviceGate(False, "h200_forbidden_by_wave31_lane_c", name, capability)
    if capability != (9, 0) or "h100" not in lowered:
        return DeviceGate(False, "requires_h100_sm90", name, capability)
    return DeviceGate(True, "ok", name, capability)


def _build_extension(verbose: bool) -> Any:
    flags = [
        "-O3",
        "-lineinfo",
        "-Xptxas=-v",
    ]
    return load_inline(
        name="mamba3_wave31_dpsiv_d_boundary_ext_v1",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        extra_cuda_cflags=flags,
        with_cuda=True,
        verbose=verbose,
    )


def _make_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    shape = (args.tiles, args.cs * args.r, args.p)
    return {
        "dpsiv": torch.randn(shape, device=device, dtype=torch.float32),
        "dphi": torch.randn(shape, device=device, dtype=torch.bfloat16),
        "d": torch.randn((args.tiles, args.p), device=device, dtype=torch.float32),
        "qk_dot": torch.randn(
            (args.tiles, args.cs, args.r, args.r),
            device=device,
            dtype=torch.bfloat16,
        ),
        "gamma": torch.randn((args.tiles, args.cs), device=device, dtype=torch.float32),
    }


def _reference(
    dpsiv: torch.Tensor,
    dphi: torch.Tensor,
    d: torch.Tensor,
    qk_dot: torch.Tensor,
    gamma: torch.Tensor,
    cs: int,
    r: int,
) -> torch.Tensor:
    tiles, _fcs, p = dpsiv.shape
    dpsiv_r = dpsiv.float().reshape(tiles, cs, r, p)
    dphi_r = dphi.float().reshape(tiles, cs, r, p)
    qkd = qk_dot.float()
    out = torch.empty_like(dphi_r, dtype=torch.float32)
    for r_out in range(r):
        acc = dpsiv_r[:, :, r_out, :] + dphi_r[:, :, r_out, :] * d[:, None, :]
        qk_acc = torch.zeros((tiles, cs, p), device=dpsiv.device, dtype=torch.float32)
        for r_in in range(r):
            qk_acc = qk_acc + qkd[:, :, r_in, r_out].unsqueeze(-1) * dphi_r[:, :, r_in, :]
        out[:, :, r_out, :] = acc + gamma[:, :, None] * qk_acc
    return out.reshape(tiles, cs * r, p).to(torch.bfloat16)


def _time_cuda(fn: Any, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _peak_allocated_delta(fn: Any) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return int(max(0, peak - before))


def _memory_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _bytes(args: argparse.Namespace) -> dict[str, int]:
    elems = args.tiles * args.cs * args.r * args.p
    qk = args.tiles * args.cs * args.r * args.r
    d = args.tiles * args.p
    gam = args.tiles * args.cs
    return {
        "read_dpsiv_fp32": elems * 4,
        "read_dphi_bf16": elems * 2,
        "read_d_fp32": d * 4,
        "read_qk_dot_bf16": qk * 2,
        "read_gamma_fp32": gam * 4,
        "write_out_bf16": elems * 2,
        "global_scratch": 0,
    }


def _no_go_payload(gate: DeviceGate, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "candidate": "dpsiv_d_bf16_boundary",
        "status": "NO_GO_RESOURCE_GATE",
        "gate": asdict(gate),
        "policy": "Wave31 Lane C: H100 only; H200 forbidden",
        "compiled": False,
        "ptxas": None,
        "timing_ms": None,
        "memory": None,
        "shape": {"tiles": args.tiles, "cs": args.cs, "r": args.r, "p": args.p},
        "interface": [
            "dpsiv_fp32[tiles, cs*r, p]",
            "dphi_bf16[tiles, cs*r, p]",
            "d_fp32[tiles, p]",
            "qk_dot_bf16[tiles, cs, r, r]",
            "gamma_fp32[tiles, cs]",
            "out_bf16[tiles, cs*r, p]",
        ],
        "scratch_bytes": 0,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "time_unix": int(time.time()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=int, default=512)
    parser.add_argument("--cs", type=int, default=128)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--correctness-atol", type=float, default=0.03125)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    gate = _device_gate()
    if not gate.ok:
        payload = _no_go_payload(gate, args)
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        return 0

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ext = _build_extension(verbose=args.verbose_build)
    inputs = _make_inputs(args, device)
    out = torch.empty_like(inputs["dphi"])
    torch.cuda.synchronize()
    allocated_inputs_outputs = _memory_snapshot()

    def kernel_call() -> None:
        ext.dpsiv_d_boundary(
            inputs["dpsiv"],
            inputs["dphi"],
            inputs["d"],
            inputs["qk_dot"],
            inputs["gamma"],
            out,
            args.tiles,
            args.cs,
            args.r,
            args.p,
        )

    kernel_call()
    torch.cuda.synchronize()
    ref = _reference(
        inputs["dpsiv"],
        inputs["dphi"],
        inputs["d"],
        inputs["qk_dot"],
        inputs["gamma"],
        args.cs,
        args.r,
    )
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mismatch_count = int((diff != 0).sum().item())
    within_tolerance = bool(max_abs <= args.correctness_atol)
    if not within_tolerance:
        raise SystemExit(f"correctness failed: max_abs={max_abs} atol={args.correctness_atol}")

    kernel_ms = _time_cuda(kernel_call, args.warmup, args.iters)
    ref_ms = _time_cuda(
        lambda: _reference(
            inputs["dpsiv"],
            inputs["dphi"],
            inputs["d"],
            inputs["qk_dot"],
            inputs["gamma"],
            args.cs,
            args.r,
        ),
        args.warmup,
        args.iters,
    )
    peak_delta = _peak_allocated_delta(kernel_call)
    torch.cuda.synchronize()
    after_bench = _memory_snapshot()
    theoretical = _bytes(args)
    traffic_bytes = sum(value for key, value in theoretical.items() if key != "global_scratch")
    payload = {
        "candidate": "dpsiv_d_bf16_boundary",
        "status": "GO_COMPILED_H100",
        "gate": asdict(gate),
        "compiled": True,
        "ptxas": "emitted to stderr by --verbose-build via -Xptxas=-v",
        "correctness": {
            "max_abs_vs_torch_bf16_reference": max_abs,
            "mismatch_count": mismatch_count,
            "exact": mismatch_count == 0,
            "atol": args.correctness_atol,
            "within_tolerance": within_tolerance,
        },
        "timing_ms": {"kernel": kernel_ms, "torch_reference": ref_ms},
        "effective_bandwidth_gib_s": (traffic_bytes / (1024**3)) / (kernel_ms / 1000.0),
        "memory": {
            "allocated_inputs_outputs": allocated_inputs_outputs,
            "after_bench": after_bench,
            "theoretical_bytes": theoretical,
            "peak_allocated_delta_bytes": peak_delta,
            "scratch_bytes": 0,
        },
        "shape": {"tiles": args.tiles, "cs": args.cs, "r": args.r, "p": args.p},
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "time_unix": int(time.time()),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    raise SystemExit(main())
