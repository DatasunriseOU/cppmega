"""Benchmark the fused Triton M²RNN kernel vs the pure-Python reference.

Run on GB10 or bench3 (H200):

    python scripts/bench_m2rnn.py

Env var knobs:
    BENCH_B, BENCH_S, BENCH_H, BENCH_K, BENCH_V : problem shape
    BENCH_WARMUP, BENCH_ITERS : timing iterations
    BENCH_TORCH : "1" to include the slow torch reference baseline
"""

from __future__ import annotations

import os
import time

import torch


def _torch_m2rnn_forward(q, k, v, W, xf, *, h0=None):
    batch, seq, n_q, k_dim = q.shape
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    v_dim = v.size(-1)
    n = max(n_q, n_k, n_v, n_w, n_f)

    if h0 is None:
        h = torch.zeros(batch, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    else:
        h = h0

    if n_q != n:
        q = q.repeat_interleave(n // n_q, dim=-2)
    if n_k != n:
        k = k.repeat_interleave(n // n_k, dim=-2)
    if n_v != n:
        v = v.repeat_interleave(n // n_v, dim=-2)
    if n_w != n:
        W = W.repeat_interleave(n // n_w, dim=0)
    if n_f != n:
        xf = xf.repeat_interleave(n // n_f, dim=-1)

    x = k[..., None] * v[..., None, :]
    W_expanded = W[None, ...]
    y = torch.empty(batch, seq, n, k_dim, v_dim, device=q.device, dtype=q.dtype)
    for s in range(seq):
        f = xf[:, s, :, None, None]
        h_new = torch.tanh(h @ W_expanded + x[:, s])
        h = f * h + (1 - f) * h_new
        y[:, s] = h
    out = (q[..., None, :] @ y).squeeze(-2)
    return out, h


def _time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    assert torch.cuda.is_available(), "bench_m2rnn requires a CUDA GPU"

    B = int(os.environ.get("BENCH_B", "2"))
    S = int(os.environ.get("BENCH_S", "4096"))
    H = int(os.environ.get("BENCH_H", "8"))
    K = int(os.environ.get("BENCH_K", "64"))
    V = int(os.environ.get("BENCH_V", "16"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    iters = int(os.environ.get("BENCH_ITERS", "10"))
    include_torch = os.environ.get("BENCH_TORCH", "1") == "1"

    device = "cuda"
    dtype = torch.bfloat16
    g = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    k = torch.randn(B, S, H, K, device=device, dtype=dtype, generator=g)
    v = torch.randn(B, S, H, V, device=device, dtype=dtype, generator=g)
    W = torch.eye(V, device=device, dtype=dtype).unsqueeze(0).expand(H, -1, -1).contiguous().clone()
    W += 0.05 * torch.randn(W.shape, device=device, dtype=dtype, generator=g)
    xf = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=dtype, generator=g))

    print(f"shape B={B} S={S} H={H} K={K} V={V} dtype={dtype}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    # --- Triton fwd only ---
    from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton

    def triton_fwd():
        return m2rnn_scan_triton(q, k, v, W, xf)

    t_triton = _time_fn(triton_fwd, warmup, iters)
    print(f"triton fwd : {t_triton:8.2f} ms/iter  ({iters} iters)")

    # --- Triton fwd+bwd ---
    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    W1 = W.detach().clone().requires_grad_(True)
    xf1 = xf.detach().clone().requires_grad_(True)

    def triton_fwdbwd():
        out, _ = m2rnn_scan_triton(q1, k1, v1, W1, xf1)
        out.sum().backward()
        q1.grad = None
        k1.grad = None
        v1.grad = None
        W1.grad = None
        xf1.grad = None

    t_triton_fb = _time_fn(triton_fwdbwd, warmup, iters)
    print(f"triton fwd+bwd : {t_triton_fb:8.2f} ms/iter  ({iters} iters)")

    # --- Torch reference ---
    if include_torch:
        def torch_fwd():
            return _torch_m2rnn_forward(q, k, v, W, xf)

        # Torch is slow; use a smaller iter count.
        torch_iters = max(1, iters // 5)
        t_torch = _time_fn(torch_fwd, 1, torch_iters)
        print(f"torch  fwd : {t_torch:8.2f} ms/iter  ({torch_iters} iters)")
        print(f"speedup    : {t_torch / t_triton:6.2f}x")


if __name__ == "__main__":
    main()
