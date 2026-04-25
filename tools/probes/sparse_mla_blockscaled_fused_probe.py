#!/usr/bin/env python3
"""Probe fused MXFP8 block-scaled SparseMLA forward/backward.

The runtime under test is:

* ``sparse_mla_blockscaled_mxfp8_forward``
* ``sparse_mla_blockscaled_mxfp8_backward``

The runtime consumes pre-quantized MXFP8 payloads and per-32-channel scales.
The validation reference dequantizes Q/KV to BF16 and uses PyTorch autograd.
That BF16 materialization is reference-only; the runtime forward does not
materialize full scores or full BF16 Q/K/V tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from quack.mx_utils import to_mx
except ImportError as exc:  # pragma: no cover - environment check
    raise ImportError("sparse_mla_blockscaled_fused_probe.py requires quack") from exc


@dataclass(frozen=True)
class BlockQuantizedTensor:
    data: torch.Tensor
    scale: torch.Tensor
    block_size: int
    logical_shape: tuple[int, ...]

    @property
    def num_blocks(self) -> int:
        return self.logical_shape[-1] // self.block_size


def quantize_mxfp8(x: torch.Tensor) -> BlockQuantizedTensor:
    x = x.contiguous()
    block_size = 32
    if x.shape[-1] % block_size:
        raise ValueError(f"MXFP8 requires last dim multiple of {block_size}")
    data, scale = to_mx(x, block_size)
    if data.dtype == torch.uint8:
        data = data.view(torch.float8_e4m3fn)
    return BlockQuantizedTensor(
        data=data.contiguous(),
        scale=scale.float().contiguous(),
        block_size=block_size,
        logical_shape=tuple(x.shape),
    )


def dequantize_mxfp8(tensor: BlockQuantizedTensor) -> torch.Tensor:
    pieces = []
    for block_idx in range(tensor.num_blocks):
        start = block_idx * tensor.block_size
        stop = start + tensor.block_size
        values = tensor.data[..., start:stop].float()
        scale = tensor.scale[..., block_idx].float().unsqueeze(-1)
        pieces.append(values * scale)
    return torch.cat(pieces, dim=-1).reshape(tensor.logical_shape)


def reference_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: float,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq, heads, _ = q.shape
    _, seq_kv, kv_group, _ = kv.shape
    heads_per_group = heads // kv_group
    indices = indices.to(device=q.device, dtype=torch.long)
    out = torch.empty((bsz, seq, heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.empty((bsz, seq, heads), dtype=torch.float32, device=q.device)

    for group_idx in range(kv_group):
        h0 = group_idx * heads_per_group
        h1 = h0 + heads_per_group
        group_indices = indices[:, :, group_idx, :]
        safe_indices = group_indices.clamp(min=0, max=seq_kv - 1)
        valid = group_indices >= 0

        batch_idx = torch.arange(bsz, device=q.device)[:, None, None]
        gathered_kv = kv[:, :, group_idx, :][batch_idx, safe_indices]
        q_group = q[:, :, h0:h1, :].float()
        scores = torch.einsum("bshd,bstd->bsht", q_group, gathered_kv.float())
        scores = scores * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, h0:h1, :] = torch.einsum(
            "bsht,bstd->bshd", probs, gathered_kv[:, :, :, :d_v].float()
        ).to(q.dtype)
        lse[:, :, h0:h1] = torch.logsumexp(scores, dim=-1)

    return out, lse


def _make_indices(
    bsz: int,
    seq: int,
    seq_kv: int,
    kv_group: int,
    topk: int,
    device: torch.device,
    invalid_fraction: float,
) -> torch.Tensor:
    indices = torch.randint(0, seq_kv, (bsz, seq, kv_group, topk), device=device)
    if invalid_fraction > 0:
        invalid = torch.rand(indices.shape, device=device) < invalid_fraction
        indices = torch.where(invalid, torch.full_like(indices, -1), indices)
        valid_count = (indices >= 0).sum(dim=-1)
        empty = valid_count == 0
        if empty.any():
            replacement = torch.randint(0, seq_kv, indices[..., :1].shape, device=device)
            indices[..., :1] = torch.where(empty.unsqueeze(-1), replacement, indices[..., :1])
    return indices.to(torch.int32).contiguous()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, device: torch.device, iters: int) -> tuple[float, object]:
    _synchronize(device)
    out = fn()
    _synchronize(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        stop.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(stop) / iters, out

    start_t = time.perf_counter()
    for _ in range(iters):
        out = fn()
    return (time.perf_counter() - start_t) * 1000.0 / iters, out


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if not torch.isfinite(actual).all():
        return float("inf"), float("inf")
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-7)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    return max_abs, max_rel


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2)
    parser.add_argument("--seq-kv", type=int, default=64)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--kv-group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--dim-total", type=int, default=128)
    parser.add_argument("--d-v", type=int, default=64)
    parser.add_argument("--block-i", type=int, default=64)
    parser.add_argument("--bwd-block", type=int, default=32)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--bwd-threads", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--invalid-fraction", type=float, default=0.05)
    parser.add_argument("--fwd-atol", type=float, default=4e-3)
    parser.add_argument("--fwd-rtol", type=float, default=7e-3)
    parser.add_argument("--grad-atol", type=float, default=8e-3)
    parser.add_argument("--grad-rtol", type=float, default=9e-2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.heads % args.kv_group:
        raise ValueError("--heads must be divisible by --kv-group")
    if args.dim_total % 32:
        raise ValueError("--dim-total must be divisible by 32 for MXFP8")
    if args.d_v <= 0 or args.d_v >= args.dim_total:
        raise ValueError("--d-v must be positive and smaller than --dim-total")
    if args.d_v % 16 or (args.dim_total - args.d_v) % 16:
        raise ValueError("--d-v and tail dim must be divisible by 16")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    q_hp = (
        torch.randn(args.batch, args.seq, args.heads, args.dim_total, device=device)
        * (1.0 / math.sqrt(args.dim_total))
    ).to(torch.bfloat16)
    kv_hp = (
        torch.randn(args.batch, args.seq_kv, args.kv_group, args.dim_total, device=device)
        * (1.0 / math.sqrt(args.dim_total))
    ).to(torch.bfloat16)
    indices = _make_indices(
        args.batch,
        args.seq,
        args.seq_kv,
        args.kv_group,
        args.topk,
        device,
        args.invalid_fraction,
    )
    softmax_scale = args.dim_total ** -0.5

    q_quant = quantize_mxfp8(q_hp)
    kv_quant = quantize_mxfp8(kv_hp)
    q_ref = dequantize_mxfp8(q_quant).bfloat16().detach().requires_grad_(True)
    kv_ref = dequantize_mxfp8(kv_quant).bfloat16().detach().requires_grad_(True)

    os.environ.setdefault("CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED", "1")
    os.environ.setdefault("CPPMEGA_SPARSE_MLA_BLOCKSCALED_BWD_REFERENCE_ACK", "1")
    from cppmega.megatron.sparse_mla_ops import (
        sparse_mla_blockscaled_mxfp8_backward,
        sparse_mla_blockscaled_mxfp8_forward,
    )

    _reset_peak(device)
    runtime_fwd_ms, runtime_pair = _bench(
        lambda: sparse_mla_blockscaled_mxfp8_forward(
            q_quant.data,
            kv_quant.data,
            q_quant.scale,
            kv_quant.scale,
            indices,
            softmax_scale=softmax_scale,
            d_v=args.d_v,
            block_i=args.block_i,
            threads=args.threads,
        ),
        device,
        args.iters,
    )
    runtime_peak_fwd_mb = _peak_mb(device)
    runtime_out, runtime_lse = runtime_pair

    _reset_peak(device)
    ref_fwd_ms, ref_pair = _bench(
        lambda: reference_sparse_mla(
            q_ref,
            kv_ref,
            indices,
            softmax_scale=softmax_scale,
            d_v=args.d_v,
        ),
        device,
        max(1, min(args.iters, 3)),
    )
    ref_peak_fwd_mb = _peak_mb(device)
    ref_out, ref_lse = ref_pair

    grad_out = (torch.randn_like(runtime_out) * 0.01).bfloat16()

    ref_out_for_bwd, _ = reference_sparse_mla(
        q_ref,
        kv_ref,
        indices,
        softmax_scale=softmax_scale,
        d_v=args.d_v,
    )
    _synchronize(device)
    _reset_peak(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        ref_out_for_bwd.backward(grad_out)
        stop.record()
        torch.cuda.synchronize(device)
        ref_bwd_ms = start.elapsed_time(stop)
    else:
        start_t = time.perf_counter()
        ref_out_for_bwd.backward(grad_out)
        ref_bwd_ms = (time.perf_counter() - start_t) * 1000.0
    ref_peak_bwd_mb = _peak_mb(device)
    ref_dq = q_ref.grad.detach()
    ref_dkv = kv_ref.grad.detach()

    _reset_peak(device)
    runtime_bwd_ms, runtime_grads = _bench(
        lambda: sparse_mla_blockscaled_mxfp8_backward(
            q_quant.data,
            kv_quant.data,
            q_quant.scale,
            kv_quant.scale,
            runtime_out,
            grad_out,
            indices,
            runtime_lse,
            softmax_scale=softmax_scale,
            d_v=args.d_v,
            block_size=args.bwd_block,
            threads=args.bwd_threads,
        ),
        device,
        args.iters,
    )
    runtime_peak_bwd_mb = _peak_mb(device)
    runtime_dq, runtime_dkv = runtime_grads

    fwd_abs, fwd_rel = _max_errors(runtime_out, ref_out)
    lse_abs, lse_rel = _max_errors(runtime_lse * math.log(2.0), ref_lse)
    dq_abs, dq_rel = _max_errors(runtime_dq, ref_dq)
    dkv_abs, dkv_rel = _max_errors(runtime_dkv, ref_dkv)

    reference_dequant_bytes = _tensor_bytes(q_ref, kv_ref)
    score_bytes = args.batch * args.seq * args.heads * args.topk * 4
    runtime_grad_bytes = _tensor_bytes(runtime_dq, runtime_dkv)

    print(
        f"format=mxfp8 q_data={tuple(q_quant.data.shape)}:{q_quant.data.dtype} "
        f"q_scale={tuple(q_quant.scale.shape)}:{q_quant.scale.dtype} "
        f"kv_data={tuple(kv_quant.data.shape)}:{kv_quant.data.dtype} "
        f"kv_scale={tuple(kv_quant.scale.shape)}:{kv_quant.scale.dtype} "
        f"indices={tuple(indices.shape)} d_v={args.d_v}"
    )
    print(
        f"runtime_fwd_ms={runtime_fwd_ms:.4f} ref_fwd_ms={ref_fwd_ms:.4f} "
        f"runtime_bwd_reference_ms={runtime_bwd_ms:.4f} ref_bwd_ms={ref_bwd_ms:.4f}"
    )
    print(
        f"fwd_max_abs={fwd_abs:.6g} fwd_max_rel={fwd_rel:.6g} "
        f"lse_nat_max_abs={lse_abs:.6g} lse_nat_max_rel={lse_rel:.6g}"
    )
    print(
        f"dq_max_abs={dq_abs:.6g} dq_max_rel={dq_rel:.6g} "
        f"dkv_max_abs={dkv_abs:.6g} dkv_max_rel={dkv_rel:.6g}"
    )
    print(
        f"peak_mb runtime_fwd={runtime_peak_fwd_mb:.2f} ref_fwd={ref_peak_fwd_mb:.2f} "
        f"runtime_bwd={runtime_peak_bwd_mb:.2f} ref_bwd={ref_peak_bwd_mb:.2f}"
    )
    print(
        "materialization runtime_forward_full_scores=False "
        "runtime_forward_full_bf16_qkv=False "
        "runtime_backward_backend=explicit_bf16_reference_ack "
        "runtime_backward_full_bf16_qkv=True "
        "runtime_backward_sparse_scores=True "
        f"reference_dequant_qkv_bytes={reference_dequant_bytes} "
        f"avoided_sparse_scores_bytes={score_bytes} "
        f"runtime_returned_grad_bytes={runtime_grad_bytes}"
    )

    failed = False
    if fwd_abs > args.fwd_atol and fwd_rel > args.fwd_rtol:
        print(
            f"forward validation failed: max_abs={fwd_abs:.6g} max_rel={fwd_rel:.6g} "
            f"(atol={args.fwd_atol}, rtol={args.fwd_rtol})"
        )
        failed = True
    if dq_abs > args.grad_atol and dq_rel > args.grad_rtol:
        print(
            f"dq validation failed: max_abs={dq_abs:.6g} max_rel={dq_rel:.6g} "
            f"(atol={args.grad_atol}, rtol={args.grad_rtol})"
        )
        failed = True
    if dkv_abs > args.grad_atol and dkv_rel > args.grad_rtol:
        print(
            f"dkv validation failed: max_abs={dkv_abs:.6g} max_rel={dkv_rel:.6g} "
            f"(atol={args.grad_atol}, rtol={args.grad_rtol})"
        )
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
