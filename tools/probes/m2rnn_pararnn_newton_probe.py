"""Probe Newton-style ParaRNN feasibility for cppmega M2RNN.

This is not a fast implementation.  It solves the Newton linearized M2RNN
system with sequential forward substitution so we can answer the first
question before writing CUDA: does the M2RNN nonlinear system converge in a
small, fixed number of Newton iterations?

Run examples:
    python tools/probes/m2rnn_pararnn_newton_probe.py
    python tools/probes/m2rnn_pararnn_newton_probe.py --device cuda --S 4096 --K 64 --V 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Shape:
    B: int
    S: int
    H: int
    K: int
    V: int


def _make_inputs(shape: Shape, *, device: str, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(shape.B, shape.S, shape.H, shape.K, device=device, dtype=dtype, generator=g)
    k = torch.randn(shape.B, shape.S, shape.H, shape.K, device=device, dtype=dtype, generator=g)
    v = torch.randn(shape.B, shape.S, shape.H, shape.V, device=device, dtype=dtype, generator=g)
    W = (
        torch.eye(shape.V, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(shape.H, -1, -1)
        .contiguous()
        .clone()
    )
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(shape.B, shape.S, shape.H, device=device, dtype=dtype, generator=g))
    return q, k, v, W, xf


def _m2rnn_states_sequential(q, k, v, W, xf):
    B, S, H, K = q.shape
    V = v.size(-1)
    h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    states = torch.empty(B, S, H, K, V, device=q.device, dtype=torch.float32)
    x = (k.float()[..., None] * v.float()[..., None, :])
    Wf = W.float()
    xff = xf.float()
    for s in range(S):
        h_new = torch.tanh(h @ Wf + x[:, s])
        f = xff[:, s, :, None, None]
        h = f * h + (1.0 - f) * h_new
        states[:, s] = h
    out = torch.einsum("bshk,bshkv->bshv", q.float(), states)
    return out, states


def _residual(states, k, v, W, xf):
    B, S, H, K, V = states.shape
    initial = torch.zeros(B, 1, H, K, V, device=states.device, dtype=states.dtype)
    prev = torch.cat([initial, states[:, :-1]], dim=1)
    x = k.float()[..., None] * v.float()[..., None, :]
    cand = torch.tanh(prev @ W.float() + x)
    f = xf.float()[..., None, None]
    return f * prev + (1.0 - f) * cand - states, cand, prev


def _newton_m2rnn_states(q, k, v, W, xf, *, num_iters: int):
    B, S, H, K = q.shape
    V = v.size(-1)
    x = k.float()[..., None] * v.float()[..., None, :]
    f = xf.float()[..., None, None]
    Wf = W.float()

    # h_l^0 = f(0, x_l), as in ParaRNN Appendix A.
    states = (1.0 - f) * torch.tanh(x)
    residual_history: list[float] = []

    for _ in range(num_iters):
        res, cand, _prev = _residual(states, k, v, W, xf)
        residual_history.append(res.abs().max().item())

        # Solve delta_l = J_l delta_{l-1} + res_l.  This probe uses
        # sequential substitution; CUDA work should replace this with a
        # block-affine parallel reduction over sequence length.
        delta_prev = torch.zeros(B, H, K, V, device=states.device, dtype=torch.float32)
        delta = torch.empty_like(states)
        g = 1.0 - cand * cand
        for s in range(S):
            # Row-vector differential:
            # d tanh(h @ W + x) = (delta_prev @ W) * (1 - cand^2).
            mapped = f[:, s] * delta_prev + (1.0 - f[:, s]) * ((delta_prev @ Wf) * g[:, s])
            delta_s = mapped + res[:, s]
            delta[:, s] = delta_s
            delta_prev = delta_s
        states = states + delta

    res, _cand, _prev = _residual(states, k, v, W, xf)
    residual_history.append(res.abs().max().item())
    out = torch.einsum("bshk,bshkv->bshv", q.float(), states)
    return out, states, residual_history


def _dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--V", type=int, default=16)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    shape = Shape(args.B, args.S, args.H, args.K, args.V)
    q, k, v, W, xf = _make_inputs(shape, device=args.device, dtype=_dtype(args.dtype), seed=args.seed)
    out_ref, h_ref = _m2rnn_states_sequential(q, k, v, W, xf)
    out_newton, h_newton, residual_history = _newton_m2rnn_states(q, k, v, W, xf, num_iters=args.iters)

    print(f"shape B={args.B} S={args.S} H={args.H} K={args.K} V={args.V} device={args.device} dtype={args.dtype}")
    print("iter residual_max")
    for i, value in enumerate(residual_history):
        print(f"{i:4d} {value:.6e}")
    print(f"state_max_abs_err {(h_ref - h_newton).abs().max().item():.6e}")
    print(f"out_max_abs_err   {(out_ref - out_newton).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
