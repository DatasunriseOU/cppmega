# ruff: noqa
"""Experimental block-scaled SparseMLA QK score kernels.

This module is deliberately QK-only.  It consumes already-quantized block
payloads plus block scales and returns sparse attention logits:

    scores[b, s, h, topk] = sum_kblock(
        dot(q_block, k_block) * q_scale_block * k_scale_block
    )

No BF16 Q/K tensor is materialized in this runtime path.  Full SparseMLA
softmax/PV fusion still lives in the existing tensorwise FP8 kernels.
"""

from functools import lru_cache

import tilelang
import torch
from tilelang import language as T


MXFP8_BLOCK_SIZE = 32
NVFP4_BLOCK_SIZE = 16


def _canonical_format(quant_format: str) -> str:
    quant_format = quant_format.strip().lower()
    if quant_format in {"mx", "mx8", "mxfp8"}:
        return "mxfp8"
    if quant_format in {"nv", "nv4", "nvfp4", "fp4"}:
        return "nvfp4"
    raise ValueError(f"unsupported SparseMLA block-scaled QK format: {quant_format!r}")


@lru_cache(maxsize=64)
def _mxfp8_qk_kernel(
    heads: int,
    dim: int,
    topk: int,
    kv_group: int,
    block_i: int,
    threads: int,
):
    block_k = MXFP8_BLOCK_SIZE
    if dim % block_k:
        raise ValueError(f"MXFP8 dim must be divisible by {block_k}, got {dim}")
    if topk % block_i:
        raise ValueError(f"topk={topk} must be divisible by block_i={block_i}")
    if heads % kv_group:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")

    head_kv = heads // kv_group
    padded_head_kv = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_head_kv > 64:
        raise ValueError(
            "experimental MXFP8 QK helper supports at most 64 query heads per KV group"
        )

    num_kblocks = dim // block_k
    num_itiles = topk // block_i

    batch = T.dynamic("batch")
    seq = T.dynamic("seq")
    seq_kv = T.dynamic("seq_kv")

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        @T.prim_func
        def main(
            Q: T.Tensor([batch, seq, heads, dim], T.float8_e4m3fn),
            KV: T.Tensor([batch, seq_kv, kv_group, dim], T.float8_e4m3fn),
            QScale: T.Tensor([batch, seq, heads, num_kblocks], T.float32),
            KVScale: T.Tensor([batch, seq_kv, kv_group, num_kblocks], T.float32),
            Indices: T.Tensor([batch, seq, kv_group, topk], T.int32),
            Scores: T.Tensor([batch, seq, heads, topk], T.float32),
        ):
            with T.Kernel(seq * num_itiles, batch, kv_group, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_shared = T.alloc_shared([padded_head_kv, block_k], T.float8_e4m3fn)
                kv_shared = T.alloc_shared([block_i, block_k], T.float8_e4m3fn)
                partial = T.alloc_fragment([padded_head_kv, block_i], T.float32)
                acc = T.alloc_fragment([padded_head_kv, block_i], T.float32)
                mask = T.alloc_fragment([block_i], "bool")

                s_i = bx // num_itiles
                tile_i = bx - s_i * num_itiles
                b_i = by
                g_i = bz
                h0 = g_i * head_kv

                for bi_i in T.Parallel(block_i):
                    mask[bi_i] = Indices[b_i, s_i, g_i, tile_i * block_i + bi_i] != -1

                T.fill(acc, 0.0)
                for kb in T.serial(num_kblocks):
                    T.fill(q_shared, 0)
                    for h_i, d_i in T.Parallel(padded_head_kv, block_k):
                        if h_i < head_kv:
                            q_shared[h_i, d_i] = Q[
                                b_i,
                                s_i,
                                h0 + h_i,
                                kb * block_k + d_i,
                            ]

                    for bi_i, d_i in T.Parallel(block_i, block_k):
                        idx = Indices[b_i, s_i, g_i, tile_i * block_i + bi_i]
                        safe_idx = T.max(idx, 0)
                        kv_shared[bi_i, d_i] = KV[b_i, safe_idx, g_i, kb * block_k + d_i]

                    T.fill(partial, 0.0)
                    T.gemm(
                        q_shared,
                        kv_shared,
                        partial,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for h_i, bi_i in T.Parallel(padded_head_kv, block_i):
                        if h_i < head_kv:
                            idx = Indices[b_i, s_i, g_i, tile_i * block_i + bi_i]
                            safe_idx = T.max(idx, 0)
                            acc[h_i, bi_i] += (
                                partial[h_i, bi_i]
                                * QScale[b_i, s_i, h0 + h_i, kb]
                                * KVScale[b_i, safe_idx, g_i, kb]
                            )

                for h_i, bi_i in T.Parallel(padded_head_kv, block_i):
                    if h_i < head_kv:
                        Scores[b_i, s_i, h0 + h_i, tile_i * block_i + bi_i] = (
                            T.if_then_else(mask[bi_i], acc[h_i, bi_i], -T.infinity(T.float32))
                        )

        return main

    return kernel_builder()


@lru_cache(maxsize=64)
def _nvfp4_qk_kernel(
    heads: int,
    dim: int,
    topk: int,
    kv_group: int,
    threads: int,
):
    block_k = NVFP4_BLOCK_SIZE
    packed_block_k = block_k // 2
    if dim % block_k:
        raise ValueError(f"NVFP4 dim must be divisible by {block_k}, got {dim}")
    if heads % kv_group:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")

    head_kv = heads // kv_group
    num_kblocks = dim // block_k
    packed_dim = dim // 2
    batch = T.dynamic("batch")
    seq = T.dynamic("seq")
    seq_kv = T.dynamic("seq_kv")

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        @T.prim_func
        def main(
            Q: T.Tensor([batch, seq, heads, packed_dim], T.uint8),
            KV: T.Tensor([batch, seq_kv, kv_group, packed_dim], T.uint8),
            QScale: T.Tensor([batch, seq, heads, num_kblocks], T.float32),
            KVScale: T.Tensor([batch, seq_kv, kv_group, num_kblocks], T.float32),
            Indices: T.Tensor([batch, seq, kv_group, topk], T.int32),
            Fp4Table: T.Tensor([16], T.float32),
            Scores: T.Tensor([batch, seq, heads, topk], T.float32),
        ):
            with T.Kernel(batch * seq * heads * topk, threads=threads) as bx:
                topk_i = bx % topk
                h_i = (bx // topk) % heads
                s_i = (bx // (topk * heads)) % seq
                b_i = bx // (topk * heads * seq)
                g_i = h_i // head_kv
                kv_i = Indices[b_i, s_i, g_i, topk_i]

                acc = T.alloc_fragment([1], T.float32)
                T.fill(acc, 0.0)
                if kv_i != -1:
                    for kb in T.serial(num_kblocks):
                        partial = T.alloc_fragment([1], T.float32)
                        T.fill(partial, 0.0)
                        for packed_j in T.serial(packed_block_k):
                            q_byte = T.cast(
                                Q[b_i, s_i, h_i, kb * packed_block_k + packed_j],
                                T.int32,
                            )
                            kv_byte = T.cast(
                                KV[b_i, kv_i, g_i, kb * packed_block_k + packed_j],
                                T.int32,
                            )
                            q_lo = q_byte & 15
                            q_hi = (q_byte >> 4) & 15
                            kv_lo = kv_byte & 15
                            kv_hi = (kv_byte >> 4) & 15
                            partial[0] += Fp4Table[q_lo] * Fp4Table[kv_lo]
                            partial[0] += Fp4Table[q_hi] * Fp4Table[kv_hi]
                        acc[0] += (
                            partial[0]
                            * QScale[b_i, s_i, h_i, kb]
                            * KVScale[b_i, kv_i, g_i, kb]
                        )

                    Scores[b_i, s_i, h_i, topk_i] = acc[0]
                else:
                    Scores[b_i, s_i, h_i, topk_i] = -T.infinity(T.float32)

        return main

    return kernel_builder()


def _fp4_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (
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
        ),
        dtype=torch.float32,
        device=device,
    )


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for experimental block-scaled QK")


def sparse_mla_blockscaled_qk_scores(
    q_data: torch.Tensor,
    kv_data: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    quant_format: str,
    block_i: int = 64,
    threads: int = 256,
) -> torch.Tensor:
    """Run the experimental SparseMLA block-scaled QK score path.

    Args:
        q_data: MXFP8 ``[B,S,H,D]`` float8 or NVFP4 ``[B,S,H,D/2]`` uint8.
        kv_data: MXFP8 ``[B,SK,G,D]`` float8 or NVFP4 ``[B,SK,G,D/2]`` uint8.
        q_scale: FP32 block scales ``[B,S,H,D/block]``.
        kv_scale: FP32 block scales ``[B,SK,G,D/block]``.
        indices: Sparse indices ``[B,S,G,topk]`` int32 with ``-1`` sentinel.
        quant_format: ``"mxfp8"`` or ``"nvfp4"``.

    Returns:
        FP32 QK scores ``[B,S,H,topk]``.
    """
    quant_format = _canonical_format(quant_format)
    for name, tensor in (
        ("q_data", q_data),
        ("kv_data", kv_data),
        ("q_scale", q_scale),
        ("kv_scale", kv_scale),
        ("indices", indices),
    ):
        _require_contiguous(name, tensor)

    if q_data.device.type != "cuda":
        raise RuntimeError("experimental block-scaled SparseMLA QK requires CUDA tensors")
    if not (kv_data.device == q_data.device == q_scale.device == kv_scale.device == indices.device):
        raise ValueError("all block-scaled QK tensors must be on the same device")
    if q_data.ndim != 4 or kv_data.ndim != 4 or q_scale.ndim != 4 or kv_scale.ndim != 4:
        raise ValueError("q/kv data and scale tensors must be rank 4")
    if indices.ndim != 4:
        raise ValueError("indices must be rank 4 [B,S,G,topk]")
    if indices.dtype != torch.int32:
        raise ValueError("indices must be int32 with -1 sentinel")
    if q_scale.dtype != torch.float32 or kv_scale.dtype != torch.float32:
        raise ValueError("block-scaled QK scale tensors must be float32")

    batch, seq, heads, q_width = q_data.shape
    kv_batch, seq_kv, kv_group, kv_width = kv_data.shape
    if batch != kv_batch:
        raise ValueError(f"batch mismatch: q={batch}, kv={kv_batch}")
    if heads % kv_group:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")
    topk = indices.shape[-1]
    if indices.shape != (batch, seq, kv_group, topk):
        raise ValueError(
            f"indices shape mismatch: expected {(batch, seq, kv_group, topk)}, "
            f"got {tuple(indices.shape)}"
        )

    if quant_format == "mxfp8":
        if q_data.dtype != torch.float8_e4m3fn or kv_data.dtype != torch.float8_e4m3fn:
            raise ValueError("MXFP8 QK data tensors must use torch.float8_e4m3fn")
        if q_width != kv_width:
            raise ValueError(f"q/kv dim mismatch: q={q_width}, kv={kv_width}")
        num_kblocks = q_width // MXFP8_BLOCK_SIZE
        if q_scale.shape != (batch, seq, heads, num_kblocks):
            raise ValueError(
                f"q_scale shape mismatch: expected {(batch, seq, heads, num_kblocks)}, "
                f"got {tuple(q_scale.shape)}"
            )
        if kv_scale.shape != (batch, seq_kv, kv_group, num_kblocks):
            raise ValueError(
                "kv_scale shape mismatch: expected "
                f"{(batch, seq_kv, kv_group, num_kblocks)}, got {tuple(kv_scale.shape)}"
            )
        kernel = _mxfp8_qk_kernel(heads, q_width, topk, kv_group, block_i, threads)
        return kernel(q_data, kv_data, q_scale, kv_scale, indices)

    if q_data.dtype != torch.uint8 or kv_data.dtype != torch.uint8:
        raise ValueError("NVFP4 QK data tensors must use packed torch.uint8 storage")
    if q_width != kv_width:
        raise ValueError(f"packed q/kv dim mismatch: q={q_width}, kv={kv_width}")
    dim = q_width * 2
    num_kblocks = dim // NVFP4_BLOCK_SIZE
    if q_scale.shape != (batch, seq, heads, num_kblocks):
        raise ValueError(
            f"q_scale shape mismatch: expected {(batch, seq, heads, num_kblocks)}, "
            f"got {tuple(q_scale.shape)}"
        )
    if kv_scale.shape != (batch, seq_kv, kv_group, num_kblocks):
        raise ValueError(
            "kv_scale shape mismatch: expected "
            f"{(batch, seq_kv, kv_group, num_kblocks)}, got {tuple(kv_scale.shape)}"
        )
    kernel = _nvfp4_qk_kernel(heads, dim, topk, kv_group, threads)
    return kernel(q_data, kv_data, q_scale, kv_scale, indices, _fp4_table(q_data.device))
