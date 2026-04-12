# ruff: noqa
# FP8 variant of the TileLang sparse MLA forward kernel.
#
# Based on tilelang_sparse_mla_fwd.py (BF16) with FP8 modifications:
# - Q and KV stored in float8_e4m3fn
# - Per-token scale factors q_scale [batch, seq_len] and kv_scale [batch, seq_len_kv]
# - After Q@K GEMM, dequantize by multiplying acc_s by (q_scale_i * kv_scale_j)
# - sm_scale adjusted for FP8 dynamic range
# - Accumulation stays FP32, output stays BF16
#
# Reference: tile-ai/tilelang examples/deepseek_v32/fp8_lighting_indexer.py
import os
import threading
from collections import OrderedDict

import tilelang
import torch
from tilelang import language as T

_tilelang_sparse_mla_fwd_fp8_kernel_cache = OrderedDict()
_tilelang_sparse_mla_fwd_fp8_cache_lock = threading.Lock()


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


def _get_sparse_mla_fwd_fp8_kernel(
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_group: int,
    sm_scale,
    is_causal: bool,
    block_I: int,
    num_stages: int,
    threads: int,
):
    key = (
        "fp8",
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        _normalize_sm_scale(sm_scale),
        is_causal,
        block_I,
        num_stages,
        threads,
    )
    with _tilelang_sparse_mla_fwd_fp8_cache_lock:
        kernel = _tilelang_sparse_mla_fwd_fp8_kernel_cache.pop(key, None)
        if kernel is None:
            kernel = sparse_mla_fwd_fp8(
                heads,
                dim,
                tail_dim,
                topk,
                kv_group,
                sm_scale,
                is_causal,
                block_I=block_I,
                num_stages=num_stages,
                threads=threads,
            )
        _cache_put_lru(_tilelang_sparse_mla_fwd_fp8_kernel_cache, key, kernel)
        return kernel


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_fp8(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """Build sparse-MLA forward kernel — FP8 variant.

    Inputs Q and KV are float8_e4m3fn with per-token scale factors.
    After each Q@K GEMM tile, the FP32 accumulator is dequantized by
    multiplying with q_scale[q_token] * kv_scale[kv_token].
    Output is BF16, same as the BF16 kernel.
    """
    assert dim % 16 == 0, f"dim must be multiple of 16 for warp ops, got {dim}"
    assert tail_dim % 16 == 0, f"tail_dim must be multiple of 16 for warp ops, got {tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    # Q and KV are FP8
    fp8_dtype = T.float8_e4m3fn
    out_dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    # Per-token scale factors: one scalar per token
    q_scale_shape = [batch, seq_len]
    kv_scale_shape = [batch, seq_len_kv]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, otherwise handle Q/Output copy with "
            "your own mask (for kv_group==1, g_i*padded_H:(g_i+1)*padded_H is handled)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, fp8_dtype),  # type: ignore
        KV: T.Tensor(kv_shape, fp8_dtype),  # type: ignore
        QScale: T.Tensor(q_scale_shape, accum_dtype),  # type: ignore
        KVScale: T.Tensor(kv_scale_shape, accum_dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, out_dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            # Shared memory tiles — Q and KV in FP8 for Q@K GEMM
            Q_shared = T.alloc_shared([H_per_block, D], fp8_dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], fp8_dtype)
            KV_shared = T.alloc_shared([BI, D], fp8_dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], fp8_dtype)
            # Dequantized V tile (BF16) for S@V GEMM — per-token kv_scale
            # varies across the BI dimension, so we cannot leave V in FP8
            # and apply a single post-GEMM scale.  Dequantizing V to BF16
            # keeps the Q@K path fully FP8 (the dominant compute) while
            # guaranteeing correct V scaling.
            V_shared = T.alloc_shared([BI, D], out_dtype)
            O_shared = T.alloc_shared([H_per_block, D], out_dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            # Scale fragments for dequantization after GEMM
            kv_scale_frag = T.alloc_fragment([BI], accum_dtype)
            q_scale_val = T.alloc_fragment([1], accum_dtype)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], out_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            # Load Q tile (FP8)
            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # Load q_scale for this token (scalar, broadcast to all heads)
            q_scale_val[0] = QScale[b_i, s_i]

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    # Use -1 sentinel for invalid indices (PR #3674 "thd" convention)
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1

                # Load KV tile (FP8) and per-token KV scales
                for bi_i, d_i in T.Parallel(BI, D):
                    idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    safe_idx = T.max(idx, 0)
                    KV_shared[bi_i, d_i] = KV[b_i, safe_idx, g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    safe_idx = T.max(idx, 0)
                    K_tail_shared[bi_i, d_i] = KV[b_i, safe_idx, g_i, D + d_i]
                for bi_i in T.Parallel(BI):
                    idx = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    safe_idx = T.max(idx, 0)
                    kv_scale_frag[bi_i] = KVScale[b_i, safe_idx]

                # Initialize acc_s: masked positions get -inf, valid get 0
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))

                # Q @ K^T GEMMs (FP8 x FP8 -> FP32)
                T.gemm(
                    Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Dequantize: acc_s *= q_scale * kv_scale (per-element along BI dim)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = acc_s[h_i, bi_i] * q_scale_val[0] * kv_scale_frag[bi_i]

                # Online softmax update
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
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                # Dequantize V from FP8 to BF16 using per-token kv_scale.
                # V shares the first D channels of the packed KV tensor.
                # V_dequant[bi, d] = float(KV_fp8[bi, d]) * kv_scale[bi]
                for bi_i, d_i in T.Parallel(BI, D):
                    V_shared[bi_i, d_i] = T.cast(T.cast(KV_shared[bi_i, d_i], accum_dtype) * kv_scale_frag[bi_i], out_dtype)

                # S @ V GEMM — S (BF16 attention weights) @ V (BF16 dequantized),
                # accumulates into acc_o (FP32).
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale output by sumexp
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]

            # LSE computation
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a tensor to FP8 with per-token (per-row) scaling.

    Matches the scaling convention from tile-ai/tilelang utils.py:
    scale = absmax / 448.0, then x_fp8 = x / scale.

    Args:
        x: Input tensor [..., D]. Scaling is over the last dimension.

    Returns:
        (x_fp8, scales) where x_fp8 is float8_e4m3fn and scales is float32
        with shape [...] (last dim removed).
    """
    # Compute per-token absmax over the last dimension
    x_float = x.float()
    amax = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    # Scale factor: maps max value to 448 (FP8 e4m3 max)
    scale = amax / 448.0
    x_scaled = (x_float / scale).to(torch.float8_e4m3fn)
    return x_scaled, scale.squeeze(-1)


def sparse_mla_fwd_fp8_interface(
    q,
    kv,
    indices,
    q_scale=None,
    kv_scale=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """Run sparse-MLA FP8 forward kernel and return (out, lse).

    Accepts 3D tensors [seq, heads, dim] or 4D [batch, seq, heads, dim].
    Q and KV can be either BF16 (auto-quantized to FP8) or pre-quantized FP8.

    Args:
        q: Query [batch, seq, heads, dim+tail_dim] — BF16 or FP8
        kv: KV [batch, seq_kv, kv_group, dim+tail_dim] — BF16 or FP8
        indices: Sparse indices [batch, seq, kv_group, topk] int32
        q_scale: Per-token Q scale [batch, seq] float32. Required if q is FP8.
        kv_scale: Per-token KV scale [batch, seq_kv] float32. Required if kv is FP8.
        sm_scale: Softmax scale. If None, uses 1/sqrt(dim+tail_dim).
        d_v: Value head dimension.
        block_I: Block size for sparse index dimension.
        num_stages: Pipeline stages.
        threads: Thread count.

    Returns:
        (out, lse) — out is BF16 [batch, seq, heads, d_v], lse is FP32.
    """
    seq_bucket = _env_int("MCORE_DSA_TILELANG_SEQ_BUCKET", 256)
    topk_bucket = _env_int("MCORE_DSA_TILELANG_TOPK_BUCKET", block_I)

    squeeze_batch = q.ndim == 3
    if squeeze_batch:
        q = q.unsqueeze(0)
        kv = kv.unsqueeze(0)
        indices = indices.unsqueeze(0)
        if q_scale is not None:
            q_scale = q_scale.unsqueeze(0)
        if kv_scale is not None:
            kv_scale = kv_scale.unsqueeze(0)

    assert return_p_sum is False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, kv_dim = kv.shape
    assert (
        kv_dim == dim_plus_tail_dim
    ), "q and kv must have the same embedding dimension on the last axis"
    dim = d_v
    assert 0 < dim <= dim_plus_tail_dim, f"d_v must be in (0, {dim_plus_tail_dim}], but got {dim}"
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # Auto-quantize BF16 inputs to FP8 if needed
    if q.dtype != torch.float8_e4m3fn:
        # Reshape to [batch*seq, heads*D] for per-token quantization, then back
        q_flat = q.reshape(batch * seq_len, heads * dim_plus_tail_dim)
        q_fp8_flat, q_scale_flat = per_token_cast_to_fp8(q_flat)
        q = q_fp8_flat.reshape(batch, seq_len, heads, dim_plus_tail_dim)
        q_scale = q_scale_flat.reshape(batch, seq_len)
    else:
        assert q_scale is not None, "q_scale required when q is pre-quantized FP8"

    if kv.dtype != torch.float8_e4m3fn:
        kv_flat = kv.reshape(batch * seq_len_kv, kv_group * dim_plus_tail_dim)
        kv_fp8_flat, kv_scale_flat = per_token_cast_to_fp8(kv_flat)
        kv = kv_fp8_flat.reshape(batch, seq_len_kv, kv_group, dim_plus_tail_dim)
        kv_scale = kv_scale_flat.reshape(batch, seq_len_kv)
    else:
        assert kv_scale is not None, "kv_scale required when kv is pre-quantized FP8"

    q = q.contiguous()
    kv = kv.contiguous()
    q_scale = q_scale.contiguous().to(torch.float32)
    kv_scale = kv_scale.contiguous().to(torch.float32)

    # Bucketing for stable JIT cache
    seq_len_bucketed = _round_up(seq_len, seq_bucket)
    seq_len_kv_bucketed = _round_up(seq_len_kv, seq_bucket)
    topk_bucketed = _round_up(_round_up(topk, topk_bucket), block_I)

    if seq_len_bucketed != seq_len:
        q_padded = torch.zeros(
            (batch, seq_len_bucketed, heads, dim_plus_tail_dim), dtype=q.dtype, device=q.device
        )
        q_padded[:, :seq_len].copy_(q)
        q = q_padded
        q_scale_padded = torch.zeros(
            (batch, seq_len_bucketed), dtype=q_scale.dtype, device=q_scale.device
        )
        q_scale_padded[:, :seq_len].copy_(q_scale)
        q_scale = q_scale_padded

    if seq_len_kv_bucketed != seq_len_kv:
        kv_padded = torch.zeros(
            (batch, seq_len_kv_bucketed, kv_group, dim_plus_tail_dim),
            dtype=kv.dtype,
            device=kv.device,
        )
        kv_padded[:, :seq_len_kv].copy_(kv)
        kv = kv_padded
        kv_scale_padded = torch.ones(
            (batch, seq_len_kv_bucketed), dtype=kv_scale.dtype, device=kv_scale.device
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

    kernel = _get_sparse_mla_fwd_fp8_kernel(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk_bucketed,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=True,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, q_scale, kv_scale, indices)
    out = out[:, :seq_len].contiguous()
    lse = lse[:, :seq_len].contiguous()

    if squeeze_batch:
        out = out.squeeze(0)
        lse = lse.squeeze(0)

    return out, lse
