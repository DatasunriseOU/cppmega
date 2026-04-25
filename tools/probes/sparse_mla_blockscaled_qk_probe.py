#!/usr/bin/env python3
"""Probe the experimental SparseMLA block-scaled QK helper.

The runtime under test is
``cppmega.megatron.sparse_mla_ops.sparse_mla_blockscaled_qk_scores``.  It
computes sparse QK logits from already-quantized MXFP8/NVFP4 payloads and
per-block scales without materializing BF16 Q/K tensors.  This probe uses a
BF16-dequantized reference only for validation.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from quack.mx_utils import to_mx, to_nvfp4
except ImportError as exc:  # pragma: no cover - environment check
    raise ImportError("sparse_mla_blockscaled_qk_probe.py requires quack") from exc


QuantFormat = Literal["mxfp8", "nvfp4"]

FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class BlockQuantizedTensor:
    data: torch.Tensor
    scale: torch.Tensor
    block_size: int
    logical_shape: tuple[int, ...]
    fmt: QuantFormat

    @property
    def num_blocks(self) -> int:
        return self.logical_shape[-1] // self.block_size


def _fp4_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(FP4_E2M1_VALUES, dtype=torch.float32, device=device)


def quantize_blockscaled(x: torch.Tensor, fmt: QuantFormat) -> BlockQuantizedTensor:
    x = x.contiguous()
    if fmt == "mxfp8":
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
            fmt=fmt,
        )

    if fmt == "nvfp4":
        block_size = 16
        if x.shape[-1] % block_size:
            raise ValueError(f"NVFP4 requires last dim multiple of {block_size}")
        data, scale, global_scale = to_nvfp4(x, block_size, None)
        scale = (scale.float() * global_scale.float()).contiguous()
        return BlockQuantizedTensor(
            data=data.contiguous().view(torch.uint8),
            scale=scale,
            block_size=block_size,
            logical_shape=tuple(x.shape),
            fmt=fmt,
        )

    raise ValueError(f"unsupported quantization format: {fmt}")


def _fp4_block_values(tensor: BlockQuantizedTensor, block_idx: int) -> torch.Tensor:
    packed_per_block = tensor.block_size // 2
    start = block_idx * packed_per_block
    stop = start + packed_per_block
    packed = tensor.data[..., start:stop].view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], tensor.block_size)
    return _fp4_table(tensor.data.device)[codes.long()]


def _block_values(tensor: BlockQuantizedTensor, block_idx: int) -> torch.Tensor:
    if tensor.fmt == "mxfp8":
        start = block_idx * tensor.block_size
        stop = start + tensor.block_size
        return tensor.data[..., start:stop].float()
    if tensor.fmt == "nvfp4":
        return _fp4_block_values(tensor, block_idx)
    raise ValueError(f"unsupported quantization format: {tensor.fmt}")


def dequantize_blockscaled(tensor: BlockQuantizedTensor) -> torch.Tensor:
    pieces = []
    for block_idx in range(tensor.num_blocks):
        values = _block_values(tensor, block_idx)
        scale = tensor.scale[..., block_idx].float().unsqueeze(-1)
        pieces.append(values * scale)
    return torch.cat(pieces, dim=-1).reshape(tensor.logical_shape)


def reference_sparse_qk(
    q: BlockQuantizedTensor,
    kv: BlockQuantizedTensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    q_dq = dequantize_blockscaled(q).bfloat16()
    kv_dq = dequantize_blockscaled(kv).bfloat16()
    bsz, seq, heads, _ = q_dq.shape
    _, seq_kv, kv_group, _ = kv_dq.shape
    topk = indices.shape[-1]
    heads_per_group = heads // kv_group
    batch_idx = torch.arange(bsz, device=q_dq.device)[:, None, None]
    indices = indices.to(device=q_dq.device, dtype=torch.long)
    scores = torch.empty((bsz, seq, heads, topk), dtype=torch.float32, device=q_dq.device)

    for group_idx in range(kv_group):
        h0 = group_idx * heads_per_group
        h1 = h0 + heads_per_group
        group_indices = indices[:, :, group_idx, :]
        safe_indices = group_indices.clamp(min=0, max=seq_kv - 1)
        valid = group_indices >= 0
        gathered_kv = kv_dq[:, :, group_idx, :][batch_idx, safe_indices]
        group_scores = torch.einsum("bshd,bstd->bsht", q_dq[:, :, h0:h1, :], gathered_kv)
        scores[:, :, h0:h1, :] = group_scores.masked_fill(
            ~valid.unsqueeze(2), float("-inf")
        )

    return scores


def sparse_qk_runtime(
    q: BlockQuantizedTensor,
    kv: BlockQuantizedTensor,
    indices: torch.Tensor,
    *,
    block_i: int,
    threads: int,
) -> torch.Tensor:
    if q.fmt != kv.fmt:
        raise ValueError(f"format mismatch: q={q.fmt}, kv={kv.fmt}")

    os.environ.setdefault("CPPMEGA_SPARSE_MLA_BLOCKSCALED_QK", "1")
    from cppmega.megatron.sparse_mla_ops import sparse_mla_blockscaled_qk_scores

    return sparse_mla_blockscaled_qk_scores(
        q.data,
        kv.data,
        q.scale,
        kv.scale,
        indices.to(device=q.data.device, dtype=torch.int32).contiguous(),
        quant_format=q.fmt,
        block_i=block_i,
        threads=threads,
    )


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
    return indices.to(torch.int32).contiguous()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, device: torch.device, iters: int) -> tuple[float, torch.Tensor]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("mxfp8", "nvfp4"), default="mxfp8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--seq-kv", type=int, default=64)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--kv-group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--dim", type=int, default=576)
    parser.add_argument("--block-i", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--invalid-fraction", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--rtol", type=float, default=4e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.heads % args.kv_group:
        raise ValueError("--heads must be divisible by --kv-group")
    if args.format == "mxfp8" and args.dim % 32:
        raise ValueError("--dim must be divisible by 32 for MXFP8")
    if args.format == "nvfp4" and args.dim % 16:
        raise ValueError("--dim must be divisible by 16 for NVFP4")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    q_hp = (
        torch.randn(args.batch, args.seq, args.heads, args.dim, device=device)
        * (1.0 / math.sqrt(args.dim))
    ).to(torch.bfloat16)
    kv_hp = (
        torch.randn(args.batch, args.seq_kv, args.kv_group, args.dim, device=device)
        * (1.0 / math.sqrt(args.dim))
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

    q_quant = quantize_blockscaled(q_hp, args.format)
    kv_quant = quantize_blockscaled(kv_hp, args.format)

    runtime_ms, scores = _bench(
        lambda: sparse_qk_runtime(
            q_quant,
            kv_quant,
            indices,
            block_i=args.block_i,
            threads=args.threads,
        ),
        device,
        args.iters,
    )
    ref_ms, ref = _bench(
        lambda: reference_sparse_qk(q_quant, kv_quant, indices),
        device,
        max(1, min(args.iters, 5)),
    )

    finite = torch.isfinite(ref)
    diff = (scores[finite] - ref[finite]).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = ref[finite].abs().clamp_min(1e-7)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0

    print(
        f"format={args.format} block_size={q_quant.block_size} "
        f"q_data={tuple(q_quant.data.shape)}:{q_quant.data.dtype} "
        f"q_scale={tuple(q_quant.scale.shape)}:{q_quant.scale.dtype} "
        f"kv_data={tuple(kv_quant.data.shape)}:{kv_quant.data.dtype} "
        f"kv_scale={tuple(kv_quant.scale.shape)}:{kv_quant.scale.dtype} "
        f"indices={tuple(indices.shape)}"
    )
    print(
        f"runtime_qk_ms={runtime_ms:.4f} "
        f"bf16_dequant_reference_ms={ref_ms:.4f} "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
    )

    if max_abs > args.atol and max_rel > args.rtol:
        raise SystemExit(
            f"validation failed: max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
            f"(atol={args.atol}, rtol={args.rtol})"
        )


if __name__ == "__main__":
    main()
