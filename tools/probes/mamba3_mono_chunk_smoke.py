"""Compile and smoke the Mamba3 monolithic chunk CUDA skeleton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cppmega.megatron.mamba3_mono_chunk_skeleton import (
    kernel_metadata,
    mono_chunk_skeleton,
)


def _make_inputs(
    *,
    B: int,
    S: int,
    H: int,
    R: int,
    N: int,
    P: int,
    device: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    scale = 0.05
    q = torch.randn(B, S, H, R, N, device=device, dtype=torch.float16, generator=gen) * scale
    k = torch.randn(B, S, H, R, N, device=device, dtype=torch.float16, generator=gen) * scale
    dout = torch.randn(B, S, H, P, device=device, dtype=torch.float16, generator=gen) * scale
    v = torch.randn(B, S, H, P, device=device, dtype=torch.float16, generator=gen) * scale
    mimo_v = torch.randn(H, R, P, device=device, dtype=torch.float16, generator=gen) * scale
    mimo_o = torch.randn(H, R, P, device=device, dtype=torch.float16, generator=gen) * scale
    qk_dot = torch.randn(B, S, H, R, R, device=device, dtype=torch.float32, generator=gen) * scale
    dt = torch.rand(B, H, S, device=device, dtype=torch.float32, generator=gen) * 0.1
    trap = torch.randn(B, H, S, device=device, dtype=torch.float32, generator=gen) * scale
    dstates = torch.randn(B, H, N, P, device=device, dtype=torch.float16, generator=gen) * scale
    return {
        "q": q,
        "k": k,
        "dout": dout,
        "v": v,
        "mimo_v": mimo_v,
        "mimo_o": mimo_o,
        "qk_dot": qk_dot,
        "dt": dt,
        "trap": trap,
        "dstates": dstates,
    }


@torch.no_grad()
def _reference(inputs: dict[str, torch.Tensor], chunk_size: int) -> tuple[torch.Tensor, ...]:
    q = inputs["q"].float()
    k = inputs["k"].float()
    dout = inputs["dout"].float()
    v = inputs["v"].float()
    mimo_v = inputs["mimo_v"].float()
    mimo_o = inputs["mimo_o"].float()
    qk_dot = inputs["qk_dot"].float()
    dt = inputs["dt"].float()
    trap = inputs["trap"].float()
    dstates = inputs["dstates"].float()

    B, S, H, R, N = q.shape
    P = dout.shape[-1]
    nchunks = S // chunk_size

    dv = torch.zeros(B, S, H, P, device=q.device, dtype=torch.float32)
    dmimo_v = torch.zeros(B, H, R, P, device=q.device, dtype=torch.float32)
    dk_diag = torch.zeros(B, S, H, R, N, device=q.device, dtype=torch.float32)
    dq_diag = torch.zeros(B, S, H, R, N, device=q.device, dtype=torch.float32)
    lkq_checksum = torch.zeros(B, H, nchunks, device=q.device, dtype=torch.float32)

    for b in range(B):
        for h in range(H):
            for c in range(nchunks):
                s0 = c * chunk_size
                q_c = q[b, s0 : s0 + chunk_size, h].reshape(chunk_size * R, N)
                k_c = k[b, s0 : s0 + chunk_size, h].reshape(chunk_size * R, N)
                dphi = (
                    dout[b, s0 : s0 + chunk_size, h, None, :]
                    * mimo_o[h, None, :, :]
                ).reshape(chunk_size * R, P)
                psi = (
                    v[b, s0 : s0 + chunk_size, h, None, :]
                    * mimo_v[h, None, :, :]
                ).reshape(chunk_size * R, P)
                lkq = k_c @ q_c.T
                lkq_checksum[b, h, c] = lkq.sum()
                dpsi = k_c @ dstates[b, h]
                for f in range(chunk_size * R):
                    t = f // R
                    future = torch.arange(chunk_size * R, device=q.device) // R > t
                    dpsi[f] += lkq[f, future] @ dphi[future]
                for t in range(chunk_size):
                    s = s0 + t
                    gamma = dt[b, h, s] * torch.sigmoid(trap[b, h, s])
                    for r_in in range(R):
                        f_in = t * R + r_in
                        for r_out in range(R):
                            f_out = t * R + r_out
                            dpsi[f_in] += gamma * qk_dot[b, s, h, r_out, r_in] * dphi[f_out]

                for t in range(chunk_size):
                    s = s0 + t
                    for p in range(P):
                        dv[b, s, h, p] = (dpsi[t * R : (t + 1) * R, p] * mimo_v[h, :, p]).sum()
                    for r in range(R):
                        dmimo_v[b, h, r] += dpsi[t * R + r] * v[b, s, h]

                    gamma = dt[b, h, s] * torch.sigmoid(trap[b, h, s])
                    for r_out in range(R):
                        f_out = t * R + r_out
                        for r_in in range(R):
                            f_in = t * R + r_in
                            grad_qk = gamma * (dphi[f_out] * psi[f_in]).sum()
                            dq_diag[b, s, h, r_out] += grad_qk * k_c[f_in]
                            dk_diag[b, s, h, r_in] += grad_qk * q_c[f_out]

    return dv, dmimo_v, dk_diag, dq_diag, lkq_checksum


def _max_abs(ref: torch.Tensor, got: torch.Tensor) -> float:
    return float((ref.float() - got.float()).abs().max().item())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")

    metadata = kernel_metadata()
    if args.compile_only:
        return {"compile_only": True, "metadata": metadata}

    inputs = _make_inputs(
        B=args.B,
        S=args.S,
        H=args.H,
        R=args.R,
        N=args.N,
        P=args.P,
        device=args.device,
        seed=args.seed,
    )
    got = mono_chunk_skeleton(**inputs, chunk_size=args.chunk)
    if args.device == "cuda":
        torch.cuda.synchronize()
    ref = _reference(inputs, args.chunk)
    if args.device == "cuda":
        torch.cuda.synchronize()

    names = ("dv", "dmimo_v", "dk_diag", "dq_diag", "lkq_checksum")
    diffs = {name: _max_abs(r, g) for name, r, g in zip(names, ref, got)}
    passed = all(value <= args.atol for value in diffs.values())
    return {
        "passed": passed,
        "atol": args.atol,
        "diffs": diffs,
        "lkq_checksum_nonzero": bool(got[-1].abs().max().item() > 0),
        "shapes": {
            "B": args.B,
            "S": args.S,
            "H": args.H,
            "R": args.R,
            "N": args.N,
            "P": args.P,
            "chunk": args.chunk,
        },
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--atol", type=float, default=8e-3)
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
