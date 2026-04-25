# ruff: noqa
"""Experimental fused block-scaled SparseMLA kernels.

This is an MXFP8-first prototype that consumes pre-quantized block-scaled
payloads and FP32 block scales:

    q_data:   [B, S,  H, D_total]      torch.float8_e4m3fn
    kv_data:  [B, SK, G, D_total]      torch.float8_e4m3fn
    q_scale:  [B, S,  H, D_total/32]   torch.float32
    kv_scale: [B, SK, G, D_total/32]   torch.float32
    indices:  [B, S,  G, topk]         torch.int32, -1 sentinel

Forward fuses block-scaled QK, online softmax, and PV without materializing a
full sparse score tensor or full BF16 Q/K/V tensors. The in-kernel backward
prototype is kept for debugging but is disabled by the public wrapper until it
passes finite-gradient validation; the public correctness backward is an
explicit-ACK BF16 reference. This module is not yet a replacement for the
default TE tensorwise runtime.
"""

import os
import threading
from collections import OrderedDict

import tilelang
import torch
from tilelang import language as T


MXFP8_BLOCK_SIZE = 32
_SPARSE_MLA_BLOCKSCALED_BWD_BLOCK_SIZE = 32

_fwd_kernel_cache = OrderedDict()
_bwd_kernel_cache = OrderedDict()
_cache_lock = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


_TILELANG_KERNEL_CACHE_MAX = _env_int("MCORE_DSA_TILELANG_KERNEL_CACHE_MAX", 512)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 1:
        return x
    return _ceil_div(x, multiple) * multiple


def _cache_put_lru(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _TILELANG_KERNEL_CACHE_MAX:
        cache.popitem(last=False)


def _normalize_sm_scale(sm_scale):
    if sm_scale is None:
        return None
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.detach().item())
    else:
        sm_scale = float(sm_scale)
    return round(sm_scale, 12)


def _require_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for block-scaled SparseMLA")


def _validate_mxfp8_inputs(
    q_data: torch.Tensor,
    kv_data: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    d_v: int,
) -> tuple[int, int, int, int, int, int, int]:
    for name, tensor in (
        ("q_data", q_data),
        ("kv_data", kv_data),
        ("q_scale", q_scale),
        ("kv_scale", kv_scale),
        ("indices", indices),
    ):
        _require_contiguous(name, tensor)

    if q_data.device.type != "cuda":
        raise RuntimeError("block-scaled SparseMLA requires CUDA tensors")
    if not (q_data.device == kv_data.device == q_scale.device == kv_scale.device == indices.device):
        raise ValueError("all block-scaled SparseMLA tensors must be on the same device")
    if q_data.dtype != torch.float8_e4m3fn or kv_data.dtype != torch.float8_e4m3fn:
        raise ValueError("MXFP8 block-scaled SparseMLA data tensors must be torch.float8_e4m3fn")
    if q_scale.dtype != torch.float32 or kv_scale.dtype != torch.float32:
        raise ValueError("MXFP8 block-scaled SparseMLA scales must be torch.float32")
    if indices.dtype != torch.int32:
        raise ValueError("indices must be torch.int32 with -1 sentinel")
    if q_data.ndim != 4 or kv_data.ndim != 4:
        raise ValueError("q_data and kv_data must be rank 4")
    if q_scale.ndim != 4 or kv_scale.ndim != 4:
        raise ValueError("q_scale and kv_scale must be rank 4")
    if indices.ndim != 4:
        raise ValueError("indices must be rank 4 [B,S,G,topk]")

    batch, seq_len, heads, dim_total = q_data.shape
    kv_batch, seq_len_kv, kv_group, kv_dim = kv_data.shape
    if batch != kv_batch:
        raise ValueError(f"batch mismatch: q={batch}, kv={kv_batch}")
    if dim_total != kv_dim:
        raise ValueError(f"q/kv dim mismatch: q={dim_total}, kv={kv_dim}")
    if dim_total % MXFP8_BLOCK_SIZE:
        raise ValueError(f"MXFP8 D_total must be divisible by {MXFP8_BLOCK_SIZE}, got {dim_total}")
    if d_v <= 0 or d_v > dim_total:
        raise ValueError(f"d_v must be in (0, {dim_total}], got {d_v}")
    if d_v % 16:
        raise ValueError(f"d_v must be divisible by 16 for TileLang GEMM, got {d_v}")
    tail_dim = dim_total - d_v
    if tail_dim <= 0 or tail_dim % 16:
        raise ValueError(
            "block-scaled SparseMLA prototype requires a positive tail_dim divisible by 16; "
            f"got tail_dim={tail_dim}"
        )
    if heads % kv_group:
        raise ValueError(f"heads={heads} must be divisible by kv_group={kv_group}")

    topk = indices.shape[-1]
    if indices.shape != (batch, seq_len, kv_group, topk):
        raise ValueError(
            f"indices shape mismatch: expected {(batch, seq_len, kv_group, topk)}, "
            f"got {tuple(indices.shape)}"
        )
    num_blocks = dim_total // MXFP8_BLOCK_SIZE
    if q_scale.shape != (batch, seq_len, heads, num_blocks):
        raise ValueError(
            f"q_scale shape mismatch: expected {(batch, seq_len, heads, num_blocks)}, "
            f"got {tuple(q_scale.shape)}"
        )
    if kv_scale.shape != (batch, seq_len_kv, kv_group, num_blocks):
        raise ValueError(
            "kv_scale shape mismatch: expected "
            f"{(batch, seq_len_kv, kv_group, num_blocks)}, got {tuple(kv_scale.shape)}"
        )
    return batch, seq_len, seq_len_kv, heads, dim_total, kv_group, topk


def _get_fwd_kernel(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_group: int,
    sm_scale,
    block_i: int,
    num_stages: int,
    threads: int,
):
    key = (
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        _normalize_sm_scale(sm_scale),
        block_i,
        num_stages,
        threads,
    )
    with _cache_lock:
        kernel = _fwd_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = sparse_mla_blockscaled_mxfp8_fwd(
                heads,
                dim,
                tail_dim,
                topk,
                kv_group,
                sm_scale,
                block_i=block_i,
                num_stages=num_stages,
                threads=threads,
            )
        _cache_put_lru(_fwd_kernel_cache, key, kernel)
        return kernel


def _get_bwd_kernel(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_group: int,
    sm_scale,
    block_size: int,
    num_stages: int,
    threads: int,
):
    key = (
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        _normalize_sm_scale(sm_scale),
        block_size,
        num_stages,
        threads,
    )
    with _cache_lock:
        kernel = _bwd_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = sparse_mla_blockscaled_mxfp8_bwd_kernel(
                heads,
                dim,
                tail_dim,
                topk,
                kv_group,
                sm_scale,
                block_size=block_size,
                num_stages=num_stages,
                threads=threads,
            )
        _cache_put_lru(_bwd_kernel_cache, key, kernel)
        return kernel


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def sparse_mla_blockscaled_mxfp8_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_i=64,
    num_stages=2,
    threads=256,
):
    assert dim % 16 == 0, f"dim must be multiple of 16, got {dim}"
    assert tail_dim > 0 and tail_dim % 16 == 0, f"tail_dim must be positive multiple of 16, got {tail_dim}"
    assert (dim + tail_dim) % MXFP8_BLOCK_SIZE == 0
    assert topk % block_i == 0, "topk must be divisible by block_i"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    fp8_dtype = T.float8_e4m3fn
    out_dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    total_dim = dim + tail_dim
    num_qk_blocks = total_dim // MXFP8_BLOCK_SIZE
    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != head_kv:
        assert kv_group == 1, "head padding is only implemented for kv_group=1"
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        replicate_H = head_kv // 64
    else:
        replicate_H = 1
    H_per_block = padded_H if replicate_H == 1 else 64

    q_shape = [batch, seq_len, heads, total_dim]
    kv_shape = [batch, seq_len_kv, kv_group, total_dim]
    scale_q_shape = [batch, seq_len, heads, num_qk_blocks]
    scale_kv_shape = [batch, seq_len_kv, kv_group, num_qk_blocks]
    indices_shape = [batch, seq_len, kv_group, topk]
    out_shape = [batch, seq_len, heads, dim]
    lse_shape = [batch, seq_len, heads]

    BI = block_i
    NI = tilelang.cdiv(topk, block_i)
    BK = MXFP8_BLOCK_SIZE

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, fp8_dtype),
        KV: T.Tensor(kv_shape, fp8_dtype),
        QScale: T.Tensor(scale_q_shape, accum_dtype),
        KVScale: T.Tensor(scale_kv_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(out_shape, out_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * replicate_H, batch, kv_group, threads=threads) as (bx, by, bz):
            q_block = T.alloc_shared([H_per_block, BK], fp8_dtype)
            kv_block = T.alloc_shared([BI, BK], fp8_dtype)
            v_shared = T.alloc_shared([BI, dim], out_dtype)
            s_shared = T.alloc_shared([H_per_block, BI], out_dtype)
            partial = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            acc_o = T.alloc_fragment([H_per_block, dim], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            b_i, g_i = by, bz
            s_i = bx if replicate_H == 1 else (bx // replicate_H)
            h0 = g_i * padded_H + (0 if replicate_H == 1 else (bx % replicate_H) * 64)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))

                for kb in T.serial(num_qk_blocks):
                    for h_i, d_i in T.Parallel(H_per_block, BK):
                        if h0 + h_i < heads:
                            q_block[h_i, d_i] = Q[b_i, s_i, h0 + h_i, kb * BK + d_i]
                        else:
                            q_block[h_i, d_i] = T.cast(0, fp8_dtype)

                    for bi_i, d_i in T.Parallel(BI, BK):
                        idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.max(idx, 0)
                        kv_block[bi_i, d_i] = KV[b_i, safe_idx, g_i, kb * BK + d_i]

                    T.fill(partial, 0)
                    T.gemm(
                        q_block,
                        kv_block,
                        partial,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.max(idx, 0)
                        if h0 + h_i < heads:
                            acc_s[h_i, bi_i] += (
                                partial[h_i, bi_i]
                                * QScale[b_i, s_i, h0 + h_i, kb]
                                * KVScale[b_i, safe_idx, g_i, kb]
                            )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, dim):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                for bi_i, d_i in T.Parallel(BI, dim):
                    idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    safe_idx = T.max(idx, 0)
                    scale_idx = d_i // BK
                    v_shared[bi_i, d_i] = T.cast(
                        T.cast(KV[b_i, safe_idx, g_i, d_i], accum_dtype)
                        * KVScale[b_i, safe_idx, g_i, scale_idx],
                        out_dtype,
                    )

                T.copy(acc_s, s_shared)
                T.gemm(s_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for h_i, d_i in T.Parallel(H_per_block, dim):
                if h0 + h_i < heads:
                    acc_o[h_i, d_i] /= sumexp[h_i]
                    Output[b_i, s_i, h0 + h_i, d_i] = T.cast(acc_o[h_i, d_i], out_dtype)

            for h_i in T.Parallel(H_per_block):
                if h0 + h_i < heads:
                    Lse[b_i, s_i, h0 + h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

    return main


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def sparse_mla_blockscaled_mxfp8_bwd_kernel(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
):
    assert dim % 16 == 0, f"dim must be multiple of 16, got {dim}"
    assert tail_dim > 0 and tail_dim % 16 == 0, f"tail_dim must be positive multiple of 16, got {tail_dim}"
    assert (dim + tail_dim) % MXFP8_BLOCK_SIZE == 0
    assert topk % block_size == 0, "topk must be divisible by block_size"
    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** (-0.5)
    sm_scale_log2 = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    fp8_dtype = T.float8_e4m3fn
    out_dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    total_dim = dim + tail_dim
    num_qk_blocks = total_dim // MXFP8_BLOCK_SIZE
    H_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    if padded_H != H_kv:
        assert kv_group == 1, "head padding is only implemented for kv_group=1"
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H

    q_shape = [batch, seq_len, heads, total_dim]
    kv_shape = [batch, seq_len_kv, kv_group, total_dim]
    o_shape = [batch, seq_len, heads, dim]
    scale_q_shape = [batch, seq_len, heads, num_qk_blocks]
    scale_kv_shape = [batch, seq_len_kv, kv_group, num_qk_blocks]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    delta_shape = [batch, seq_len, heads]

    BS = block_size
    BK = MXFP8_BLOCK_SIZE
    NS = tilelang.cdiv(topk, block_size)
    split_store = 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, fp8_dtype),
        KV: T.Tensor(kv_shape, fp8_dtype),
        QScale: T.Tensor(scale_q_shape, accum_dtype),
        KVScale: T.Tensor(scale_kv_shape, accum_dtype),
        dO: T.Tensor(o_shape, out_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, out_dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, kv_group * NH, threads=threads) as (s_i, by, bz):
            group_i = bz // NH
            h_tile = bz - group_i * NH
            h0 = group_i * padded_H + h_tile * block_H

            q_block = T.alloc_shared([block_H, BK], fp8_dtype)
            kv_block = T.alloc_shared([BS, BK], fp8_dtype)
            q_deq = T.alloc_shared([block_H, dim], out_dtype)
            q_tail_deq = T.alloc_shared([block_H, tail_dim], out_dtype)
            kv_deq = T.alloc_shared([BS, dim], out_dtype)
            kv_tail_deq = T.alloc_shared([BS, tail_dim], out_dtype)
            do_shared = T.alloc_shared([block_H, dim], out_dtype)
            p_shared = T.alloc_shared([block_H, BS], out_dtype)
            dp_shared = T.alloc_shared([block_H, BS], out_dtype)
            dq_shared = T.alloc_shared([block_H, dim], out_dtype)
            dq_tail_shared = T.alloc_shared([block_H, tail_dim], out_dtype)
            dkv_shared = T.alloc_shared([BS // split_store, dim], accum_dtype)
            dkv_tail_shared = T.alloc_shared([BS // split_store, tail_dim], accum_dtype)

            partial = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, dim], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, tail_dim], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, dim], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, tail_dim], accum_dtype)
            mask = T.alloc_fragment([BS], "bool")

            for h_i, d_i in T.Parallel(block_H, dim):
                if h0 + h_i < heads:
                    do_shared[h_i, d_i] = dO[by, s_i, h0 + h_i, d_i]
                    q_deq[h_i, d_i] = T.cast(
                        T.cast(Q[by, s_i, h0 + h_i, d_i], accum_dtype)
                        * QScale[by, s_i, h0 + h_i, d_i // BK],
                        out_dtype,
                    )
                else:
                    do_shared[h_i, d_i] = T.cast(0, out_dtype)
                    q_deq[h_i, d_i] = T.cast(0, out_dtype)

            for h_i, d_i in T.Parallel(block_H, tail_dim):
                if h0 + h_i < heads:
                    q_tail_deq[h_i, d_i] = T.cast(
                        T.cast(Q[by, s_i, h0 + h_i, dim + d_i], accum_dtype)
                        * QScale[by, s_i, h0 + h_i, (dim + d_i) // BK],
                        out_dtype,
                    )
                else:
                    q_tail_deq[h_i, d_i] = T.cast(0, out_dtype)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, group_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for kb in T.serial(num_qk_blocks):
                    for h_i, d_i in T.Parallel(block_H, BK):
                        if h0 + h_i < heads:
                            q_block[h_i, d_i] = Q[by, s_i, h0 + h_i, kb * BK + d_i]
                        else:
                            q_block[h_i, d_i] = T.cast(0, fp8_dtype)

                    for bi_i, d_i in T.Parallel(BS, BK):
                        idx = Indices[by, s_i, group_i, i_i * BS + bi_i]
                        safe_idx = T.max(idx, 0)
                        kv_block[bi_i, d_i] = KV[by, safe_idx, group_i, kb * BK + d_i]

                    T.fill(partial, 0)
                    T.gemm(
                        q_block,
                        kv_block,
                        partial,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for h_i, bi_i in T.Parallel(block_H, BS):
                        idx = Indices[by, s_i, group_i, i_i * BS + bi_i]
                        safe_idx = T.max(idx, 0)
                        if h0 + h_i < heads:
                            acc_p[h_i, bi_i] += (
                                partial[h_i, bi_i]
                                * QScale[by, s_i, h0 + h_i, kb]
                                * KVScale[by, safe_idx, group_i, kb]
                            )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    if h0 + h_i < heads:
                        acc_p[h_i, bi_i] = T.exp2(
                            acc_p[h_i, bi_i] * sm_scale_log2 - Lse[by, s_i, h0 + h_i]
                        )
                    else:
                        acc_p[h_i, bi_i] = T.cast(0, accum_dtype)

                T.copy(acc_p, p_shared)

                for bi_i, d_i in T.Parallel(BS, dim):
                    idx = Indices[by, s_i, group_i, i_i * BS + bi_i]
                    safe_idx = T.max(idx, 0)
                    kv_deq[bi_i, d_i] = T.cast(
                        T.cast(KV[by, safe_idx, group_i, d_i], accum_dtype)
                        * KVScale[by, safe_idx, group_i, d_i // BK],
                        out_dtype,
                    )

                for bi_i, d_i in T.Parallel(BS, tail_dim):
                    idx = Indices[by, s_i, group_i, i_i * BS + bi_i]
                    safe_idx = T.max(idx, 0)
                    kv_tail_deq[bi_i, d_i] = T.cast(
                        T.cast(KV[by, safe_idx, group_i, dim + d_i], accum_dtype)
                        * KVScale[by, safe_idx, group_i, (dim + d_i) // BK],
                        out_dtype,
                    )

                T.gemm(
                    do_shared,
                    kv_deq,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    if h0 + h_i < heads:
                        acc_dp[h_i, bi_i] = (
                            T.cast(p_shared[h_i, bi_i], accum_dtype)
                            * (acc_dp[h_i, bi_i] - Delta[by, s_i, h0 + h_i])
                            * sm_scale
                        )
                    else:
                        acc_dp[h_i, bi_i] = T.cast(0, accum_dtype)

                T.copy(acc_dp, dp_shared)

                T.gemm(dp_shared, kv_deq, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dp_shared, kv_tail_deq, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dp_shared,
                    q_deq,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(
                    p_shared,
                    do_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                T.clear(acc_dkv_tail)
                T.gemm(
                    dp_shared,
                    q_tail_deq,
                    acc_dkv_tail,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, dim):
                        if bi_i < BS // split_store:
                            dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, tail_dim):
                        if bi_i < BS // split_store:
                            dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                                bi_i + s * (BS // split_store), d_i
                            ]

                    for bi_i, d_i in T.Parallel(BS // split_store, dim // 4):
                        idx = Indices[by, s_i, group_i, i_i * BS + bi_i + s * (BS // split_store)]
                        if idx >= 0:
                            T.atomic_addx4(
                                dKV[by, idx, group_i, d_i * 4],
                                dkv_shared[bi_i, d_i * 4],
                            )

                    for bi_i, d_i in T.Parallel(BS // split_store, tail_dim // 4):
                        idx = Indices[by, s_i, group_i, i_i * BS + bi_i + s * (BS // split_store)]
                        if idx >= 0:
                            T.atomic_addx4(
                                dKV[by, idx, group_i, dim + d_i * 4],
                                dkv_tail_shared[bi_i, d_i * 4],
                            )

            T.copy(acc_dq, dq_shared)
            T.copy(acc_dq_tail, dq_tail_shared)

            for h_i, d_i in T.Parallel(block_H, dim):
                if h0 + h_i < heads:
                    dQ[by, s_i, h0 + h_i, d_i] = dq_shared[h_i, d_i]

            for h_i, d_i in T.Parallel(block_H, tail_dim):
                if h0 + h_i < heads:
                    dQ[by, s_i, h0 + h_i, dim + d_i] = dq_tail_shared[h_i, d_i]

    return main


def sparse_mla_blockscaled_mxfp8_fwd_interface(
    q_data: torch.Tensor,
    kv_data: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale=None,
    d_v: int = 512,
    block_i: int = 64,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused MXFP8 block-scaled SparseMLA forward and return (out, lse)."""
    seq_bucket = _env_int("MCORE_DSA_TILELANG_SEQ_BUCKET", 256)
    topk_bucket = _env_int("MCORE_DSA_TILELANG_TOPK_BUCKET", block_i)

    squeeze_batch = q_data.ndim == 3
    if squeeze_batch:
        q_data = q_data.unsqueeze(0)
        kv_data = kv_data.unsqueeze(0)
        q_scale = q_scale.unsqueeze(0)
        kv_scale = kv_scale.unsqueeze(0)
        indices = indices.unsqueeze(0)

    _, seq_len, seq_len_kv, heads, dim_total, kv_group, topk = _validate_mxfp8_inputs(
        q_data,
        kv_data,
        q_scale,
        kv_scale,
        indices,
        d_v=d_v,
    )
    batch = q_data.shape[0]
    tail_dim = dim_total - d_v
    num_blocks = dim_total // MXFP8_BLOCK_SIZE

    seq_len_bucketed = _round_up(seq_len, seq_bucket)
    seq_len_kv_bucketed = _round_up(seq_len_kv, seq_bucket)
    topk_bucketed = _round_up(_round_up(topk, topk_bucket), block_i)

    if seq_len_bucketed != seq_len:
        q_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, dim_total), dtype=q_data.dtype, device=q_data.device
        )
        q_padded[:, :seq_len].copy_(q_data)
        q_data = q_padded
        q_scale_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, num_blocks),
            dtype=q_scale.dtype,
            device=q_scale.device,
        )
        q_scale_padded[:, :seq_len].copy_(q_scale)
        q_scale = q_scale_padded

    if seq_len_kv_bucketed != seq_len_kv:
        kv_padded = torch.zeros(
            (batch, seq_len_kv_bucketed, kv_group, dim_total),
            dtype=kv_data.dtype,
            device=kv_data.device,
        )
        kv_padded[:, :seq_len_kv].copy_(kv_data)
        kv_data = kv_padded
        kv_scale_padded = torch.ones(
            (batch, seq_len_kv_bucketed, kv_group, num_blocks),
            dtype=kv_scale.dtype,
            device=kv_scale.device,
        )
        kv_scale_padded[:, :seq_len_kv].copy_(kv_scale)
        kv_scale = kv_scale_padded

    if seq_len_bucketed != seq_len or topk_bucketed != topk:
        indices_padded = torch.full(
            (batch, seq_len_bucketed, kv_group, topk_bucketed),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices_padded[:, :seq_len, :, :topk].copy_(indices)
        indices = indices_padded

    kernel = _get_fwd_kernel(
        heads=heads,
        dim=d_v,
        tail_dim=tail_dim,
        topk=topk_bucketed,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_i=block_i,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q_data, kv_data, q_scale, kv_scale, indices)
    out = out[:, :seq_len].contiguous()
    lse = lse[:, :seq_len].contiguous()

    if squeeze_batch:
        out = out.squeeze(0)
        lse = lse.squeeze(0)
    return out, lse


def sparse_mla_blockscaled_mxfp8_bwd(
    q_data: torch.Tensor,
    kv_data: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale=None,
    d_v: int | None = None,
    block_size: int = _SPARSE_MLA_BLOCKSCALED_BWD_BLOCK_SIZE,
    num_stages: int = 0,
    threads: int = 128,
    delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MXFP8 block-scaled SparseMLA backward prototype.

    The returned gradients are BF16 gradients with respect to the logical
    dequantized Q and KV tensors, not gradients with respect to FP8 bytes or
    scale tensors.
    """
    from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd_fp8 import (
        _get_postprocess_fp8_kernel,
        _get_preprocess_fp8_kernel,
    )

    if d_v is None:
        d_v = out.shape[-1]
    seq_bucket = _env_int("MCORE_DSA_TILELANG_SEQ_BUCKET", 256)
    topk_bucket = _env_int("MCORE_DSA_TILELANG_TOPK_BUCKET", block_size)

    squeeze_batch = q_data.ndim == 3
    if squeeze_batch:
        q_data = q_data.unsqueeze(0)
        kv_data = kv_data.unsqueeze(0)
        q_scale = q_scale.unsqueeze(0)
        kv_scale = kv_scale.unsqueeze(0)
        out = out.unsqueeze(0)
        grad_out = grad_out.unsqueeze(0)
        indices = indices.unsqueeze(0)
        lse = lse.unsqueeze(0)
        if delta is not None and delta.ndim == 2:
            delta = delta.unsqueeze(0)

    _, seq_len, seq_len_kv, heads, dim_total, kv_group, topk = _validate_mxfp8_inputs(
        q_data,
        kv_data,
        q_scale,
        kv_scale,
        indices,
        d_v=d_v,
    )
    batch = q_data.shape[0]
    tail_dim = dim_total - d_v
    num_blocks = dim_total // MXFP8_BLOCK_SIZE
    if out.shape != (batch, seq_len, heads, d_v):
        raise ValueError(f"out shape mismatch: expected {(batch, seq_len, heads, d_v)}, got {tuple(out.shape)}")
    if grad_out.shape != out.shape:
        raise ValueError(f"grad_out shape mismatch: expected {tuple(out.shape)}, got {tuple(grad_out.shape)}")
    if lse.shape != (batch, seq_len, heads):
        raise ValueError(f"lse shape mismatch: expected {(batch, seq_len, heads)}, got {tuple(lse.shape)}")

    out = out.contiguous()
    grad_out = grad_out.contiguous()
    lse = lse.contiguous()

    seq_len_bucketed = _round_up(seq_len, seq_bucket)
    seq_len_kv_bucketed = _round_up(seq_len_kv, seq_bucket)
    topk_bucketed = _round_up(_round_up(topk, topk_bucket), block_size)

    if seq_len_bucketed != seq_len:
        q_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, dim_total), dtype=q_data.dtype, device=q_data.device
        )
        q_padded[:, :seq_len].copy_(q_data)
        q_data = q_padded

        q_scale_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, num_blocks),
            dtype=q_scale.dtype,
            device=q_scale.device,
        )
        q_scale_padded[:, :seq_len].copy_(q_scale)
        q_scale = q_scale_padded

        out_padded = torch.zeros((batch, seq_len_bucketed, heads, d_v), dtype=out.dtype, device=out.device)
        out_padded[:, :seq_len].copy_(out)
        out = out_padded

        grad_out_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, d_v), dtype=grad_out.dtype, device=grad_out.device
        )
        grad_out_padded[:, :seq_len].copy_(grad_out)
        grad_out = grad_out_padded

        lse_padded = torch.zeros((batch, seq_len_bucketed, heads), dtype=lse.dtype, device=lse.device)
        lse_padded[:, :seq_len].copy_(lse)
        lse = lse_padded

    if seq_len_kv_bucketed != seq_len_kv:
        kv_padded = torch.zeros(
            (batch, seq_len_kv_bucketed, kv_group, dim_total),
            dtype=kv_data.dtype,
            device=kv_data.device,
        )
        kv_padded[:, :seq_len_kv].copy_(kv_data)
        kv_data = kv_padded

        kv_scale_padded = torch.ones(
            (batch, seq_len_kv_bucketed, kv_group, num_blocks),
            dtype=kv_scale.dtype,
            device=kv_scale.device,
        )
        kv_scale_padded[:, :seq_len_kv].copy_(kv_scale)
        kv_scale = kv_scale_padded

    if seq_len_bucketed != seq_len or topk_bucketed != topk:
        indices_padded = torch.full(
            (batch, seq_len_bucketed, kv_group, topk_bucketed),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices_padded[:, :seq_len, :, :topk].copy_(indices)
        indices = indices_padded

    if delta is not None:
        if delta.ndim == 2:
            delta = delta.unsqueeze(0)
        if seq_len_bucketed != seq_len:
            delta_padded = torch.zeros((batch, seq_len_bucketed, heads), dtype=delta.dtype, device=delta.device)
            delta_padded[:, :seq_len].copy_(delta)
            delta = delta_padded

    preprocess_kernel = _get_preprocess_fp8_kernel(heads, d_v)
    bwd_kernel = _get_bwd_kernel(
        heads=heads,
        dim=d_v,
        tail_dim=tail_dim,
        topk=topk_bucketed,
        kv_group=kv_group,
        sm_scale=sm_scale,
        block_size=block_size,
        num_stages=num_stages,
        threads=threads,
    )
    postprocess_kernel = _get_postprocess_fp8_kernel(d_v, tail_dim, kv_group)

    if delta is None:
        delta = preprocess_kernel(out, grad_out)
    dkv_accum = torch.zeros_like(kv_data, dtype=torch.float32)
    dq = bwd_kernel(q_data, kv_data, q_scale, kv_scale, grad_out, indices, lse, delta, dkv_accum)
    dkv = postprocess_kernel(dkv_accum)

    dq = dq[:, :seq_len].contiguous()
    dkv = dkv[:, :seq_len_kv].contiguous()

    if squeeze_batch:
        dq = dq.squeeze(0)
        dkv = dkv.squeeze(0)
    return dq, dkv


def _dequantize_mxfp8_tensor(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pieces = []
    block_size = MXFP8_BLOCK_SIZE
    for block_idx in range(scale.shape[-1]):
        start = block_idx * block_size
        stop = start + block_size
        values = data[..., start:stop].float()
        pieces.append(values * scale[..., block_idx].float().unsqueeze(-1))
    return torch.cat(pieces, dim=-1).bfloat16().contiguous()


def _torch_sparse_mla_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    batch, seq_len, heads, _ = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    heads_per_group = heads // kv_group
    indices = indices.to(device=q.device, dtype=torch.long)
    out = torch.empty((batch, seq_len, heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, seq_len, heads), dtype=torch.float32, device=q.device)

    for group_idx in range(kv_group):
        h0 = group_idx * heads_per_group
        h1 = h0 + heads_per_group
        group_indices = indices[:, :, group_idx, :]
        safe_indices = group_indices.clamp(min=0, max=seq_len_kv - 1)
        valid = group_indices >= 0
        batch_idx = torch.arange(batch, device=q.device)[:, None, None]
        gathered_kv = kv[:, :, group_idx, :][batch_idx, safe_indices]
        scores = torch.einsum(
            "bshd,bstd->bsht",
            q[:, :, h0:h1, :].float(),
            gathered_kv.float(),
        )
        scores = scores * float(sm_scale)
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, h0:h1, :] = torch.einsum(
            "bsht,bstd->bshd",
            probs,
            gathered_kv[:, :, :, :d_v].float(),
        ).to(q.dtype)
        lse[:, :, h0:h1] = torch.logsumexp(scores, dim=-1)

    return out, lse


def sparse_mla_blockscaled_mxfp8_bwd_reference(
    q_data: torch.Tensor,
    kv_data: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale=None,
    d_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit BF16-dequant correctness backward for MXFP8 block-scaled SparseMLA.

    This intentionally materializes BF16 Q/KV and sparse scores through PyTorch
    autograd. It is for correctness probes only and is selected by the public
    wrapper only when the caller sets
    ``CPPMEGA_SPARSE_MLA_BLOCKSCALED_BWD_REFERENCE_ACK=1``.
    """
    del out, lse
    if d_v is None:
        d_v = grad_out.shape[-1]
    _validate_mxfp8_inputs(q_data, kv_data, q_scale, kv_scale, indices, d_v=d_v)

    with torch.enable_grad():
        q_ref = _dequantize_mxfp8_tensor(q_data, q_scale).detach().requires_grad_(True)
        kv_ref = _dequantize_mxfp8_tensor(kv_data, kv_scale).detach().requires_grad_(True)
        ref_out, _ = _torch_sparse_mla_reference(
            q_ref,
            kv_ref,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
        )
        ref_out.backward(grad_out.to(ref_out.dtype))
        return q_ref.grad.detach().bfloat16().contiguous(), kv_ref.grad.detach().bfloat16().contiguous()
