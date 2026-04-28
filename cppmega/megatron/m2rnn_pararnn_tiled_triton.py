"""Tiled/streaming ParaRNN Newton solve prototype for M2RNN.

This module is deliberately separate from ``m2rnn_pararnn.py``: it is a
Triton-first prototype for the Newton linear solve that never materialises the
full ``A[B, S, H, K, V, V]`` Jacobian.  The only dense affine matrices that
live longer than a kernel/chunk are tile summaries with shape
``[B * H * K, ceil(S / tile), V, V]``.

Pipeline
--------
1. local tile pass: assemble ``M_t = -A_t`` and ``b_t = -F_t`` inside one
   sequence tile, run the zero-boundary affine prefix, write local deltas and a
   tile summary ``(M_tile, b_tile)``.
2. summary scan: Triton scan over tile summaries on CUDA, PyTorch fallback on
   CPU/debug paths.
3. apply pass: replay each tile with the incoming carry to produce full
   Newton deltas.  This pass recomputes local ``M_t, b_t`` instead of storing
   per-token prefix matrices.

The CUDA path uses Triton for steps 1-3.  CPU, missing-Triton, and fp64
debugging paths use the same streaming algorithm in PyTorch so unit tests can
validate the memory shape without a GPU.  fp16/bf16 inputs are up-cast to fp32
for the solve and cast back at the public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:  # pragma: no cover - import availability is environment-dependent.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class TiledTritonConfig:
    """Knobs for the tiled Newton prototype."""

    max_its: int = 3
    omega_sor: float = 1.0
    init_strategy: str = "zero"
    tile_size: int = 64
    prefer_triton: bool = True


@dataclass(frozen=True)
class TiledSolveStats:
    """Memory accounting for one Newton linear solve."""

    B: int
    S: int
    H: int
    K: int
    V: int
    Be: int
    num_tiles: int
    tile_size: int
    dtype_bytes: int
    full_A_bytes: int
    peak_tile_A_bytes: int
    summary_bytes: int
    local_delta_bytes: int
    carry_bytes: int

    @property
    def avoids_full_A(self) -> bool:
        return self.peak_tile_A_bytes < self.full_A_bytes

    @property
    def full_A_to_tile_ratio(self) -> float:
        return self.full_A_bytes / max(1, self.peak_tile_A_bytes)


def _nbytes(*shape: int, itemsize: int) -> int:
    n = itemsize
    for dim in shape:
        n *= dim
    return n


def estimate_tiled_solve_memory(
    *,
    B: int,
    S: int,
    H: int,
    K: int,
    V: int,
    tile_size: int,
    dtype: torch.dtype = torch.float32,
) -> TiledSolveStats:
    """Return the core solve memory sizes.

    ``full_A_bytes`` is the tensor this prototype is designed to avoid:
    ``[B * H * K, S, V, V]``.  ``peak_tile_A_bytes`` is the largest local
    affine block created by the PyTorch fallback or held in Triton registers
    per tile.
    """

    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    Be = B * H * K
    num_tiles = (S + tile_size - 1) // tile_size
    peak_tile = min(S, tile_size)
    return TiledSolveStats(
        B=B,
        S=S,
        H=H,
        K=K,
        V=V,
        Be=Be,
        num_tiles=num_tiles,
        tile_size=tile_size,
        dtype_bytes=dtype_bytes,
        full_A_bytes=_nbytes(Be, S, V, V, itemsize=dtype_bytes),
        peak_tile_A_bytes=_nbytes(Be, peak_tile, V, V, itemsize=dtype_bytes),
        summary_bytes=_nbytes(Be, num_tiles, V, V, itemsize=dtype_bytes)
        + _nbytes(Be, num_tiles, V, itemsize=dtype_bytes),
        local_delta_bytes=_nbytes(Be, S, V, itemsize=dtype_bytes),
        carry_bytes=_nbytes(Be, num_tiles, V, itemsize=dtype_bytes),
    )


def _broadcast_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    n_q = q.size(-2)
    n_k = k.size(-2)
    n_v = v.size(-2)
    n_w = W.size(0)
    n_f = xf.size(-1)
    H = max(n_q, n_k, n_v, n_w, n_f)

    if n_q != H:
        q = q.repeat_interleave(H // n_q, dim=-2)
    if n_k != H:
        k = k.repeat_interleave(H // n_k, dim=-2)
    if n_v != H:
        v = v.repeat_interleave(H // n_v, dim=-2)
    if n_w != H:
        W = W.repeat_interleave(H // n_w, dim=0)
    if n_f != H:
        xf = xf.repeat_interleave(H // n_f, dim=-1)
    return q, k, v, W, xf, H


def _prepare_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int]:
    B, S, _, K = q.shape
    V = v.size(-1)
    q, k, v, W, xf, H = _broadcast_heads(q, k, v, W, xf)
    compute_dtype = torch.promote_types(torch.float32, q.dtype)
    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    Wf = W.to(compute_dtype)
    xff = xf.to(compute_dtype)

    Be = B * H * K
    x_proj = (kf[..., :, None] * vf[..., None, :]).permute(0, 2, 3, 1, 4)
    x_proj = x_proj.reshape(Be, S, V).contiguous()
    f_t = xff.permute(0, 2, 1).unsqueeze(2).expand(B, H, K, S)
    f_t = f_t.reshape(Be, S).contiguous()
    W_be = Wf.unsqueeze(0).unsqueeze(2).expand(B, H, K, V, V)
    W_be = W_be.reshape(Be, V, V).contiguous()
    if h0 is None:
        h0_row = torch.zeros(Be, V, device=q.device, dtype=compute_dtype)
    else:
        h0_row = h0.to(compute_dtype).reshape(Be, V).contiguous()
    return qf, x_proj, f_t, W_be, h0_row, B, S, H, K, V


if TRITON_AVAILABLE:

    @triton.jit
    def _local_tile_kernel(
        h,
        x_proj,
        f_t,
        W_be,
        h0_row,
        local_delta,
        summary_M,
        summary_b,
        S: tl.constexpr,
        V: tl.constexpr,
        NUM_TILES: tl.constexpr,
        TILE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        be = tl.program_id(0)
        tile_id = tl.program_id(1)
        start = tile_id * TILE
        offs = tl.arange(0, BLOCK_V)
        rows = tl.arange(0, BLOCK_V)[:, None]
        cols = tl.arange(0, BLOCK_V)[None, :]
        vmask = offs < V
        mmask = (rows < V) & (cols < V)

        Wt = tl.load(W_be + be * V * V + cols * V + rows, mask=mmask, other=0.0)
        eye = rows == cols
        P = tl.where(eye & mmask, 1.0, 0.0).to(tl.float32)
        c = tl.zeros((BLOCK_V,), tl.float32)

        for step in tl.static_range(0, TILE):
            t = start + step
            valid_t = t < S
            h_cur = tl.load(h + be * S * V + t * V + offs, mask=valid_t & vmask, other=0.0)
            h_prev = tl.load(
                h + be * S * V + (t - 1) * V + offs,
                mask=(valid_t & (t > 0) & vmask),
                other=0.0,
            )
            h0v = tl.load(h0_row + be * V + offs, mask=valid_t & (t == 0) & vmask, other=0.0)
            h_prev = tl.where(t == 0, h0v, h_prev)
            x = tl.load(x_proj + be * S * V + t * V + offs, mask=valid_t & vmask, other=0.0)
            fval = tl.load(f_t + be * S + t, mask=valid_t, other=0.0).to(tl.float32)

            z = tl.sum(h_prev[None, :] * Wt, axis=1) + x
            h_new = tl.inline_asm_elementwise(
                asm="tanh.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[z],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            one_minus_f = 1.0 - fval
            residual = h_cur - fval * h_prev - one_minus_f * h_new
            b = -residual
            sech2 = 1.0 - h_new * h_new
            M = tl.where(eye, fval, 0.0) + one_minus_f * sech2[:, None] * Wt
            M = tl.where(mmask, M, 0.0)

            P = tl.dot(M, P)
            c = tl.sum(M * c[None, :], axis=1) + b
            c = tl.where(vmask, c, 0.0)
            tl.store(local_delta + be * S * V + t * V + offs, c, mask=valid_t & vmask)

        base_m = (be * NUM_TILES + tile_id) * V * V
        base_b = (be * NUM_TILES + tile_id) * V
        tl.store(summary_M + base_m + rows * V + cols, P, mask=mmask)
        tl.store(summary_b + base_b + offs, c, mask=vmask)

    @triton.jit
    def _summary_scan_kernel(
        summary_M,
        summary_b,
        carries,
        V: tl.constexpr,
        NUM_TILES: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        be = tl.program_id(0)
        offs = tl.arange(0, BLOCK_V)
        rows = tl.arange(0, BLOCK_V)[:, None]
        cols = tl.arange(0, BLOCK_V)[None, :]
        vmask = offs < V
        mmask = (rows < V) & (cols < V)
        cur = tl.zeros((BLOCK_V,), tl.float32)

        for tile_id in tl.range(0, NUM_TILES):
            base_b = (be * NUM_TILES + tile_id) * V
            base_m = (be * NUM_TILES + tile_id) * V * V
            tl.store(carries + base_b + offs, cur, mask=vmask)
            M = tl.load(summary_M + base_m + rows * V + cols, mask=mmask, other=0.0)
            b = tl.load(summary_b + base_b + offs, mask=vmask, other=0.0)
            cur = tl.sum(M * cur[None, :], axis=1) + b
            cur = tl.where(vmask, cur, 0.0)

    @triton.jit
    def _apply_carry_kernel(
        h,
        x_proj,
        f_t,
        W_be,
        h0_row,
        carries,
        delta,
        S: tl.constexpr,
        V: tl.constexpr,
        NUM_TILES: tl.constexpr,
        TILE: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        be = tl.program_id(0)
        tile_id = tl.program_id(1)
        start = tile_id * TILE
        offs = tl.arange(0, BLOCK_V)
        rows = tl.arange(0, BLOCK_V)[:, None]
        cols = tl.arange(0, BLOCK_V)[None, :]
        vmask = offs < V
        mmask = (rows < V) & (cols < V)
        d = tl.load(carries + (be * NUM_TILES + tile_id) * V + offs, mask=vmask, other=0.0)
        Wt = tl.load(W_be + be * V * V + cols * V + rows, mask=mmask, other=0.0)
        eye = rows == cols

        for step in tl.static_range(0, TILE):
            t = start + step
            valid_t = t < S
            h_cur = tl.load(h + be * S * V + t * V + offs, mask=valid_t & vmask, other=0.0)
            h_prev = tl.load(
                h + be * S * V + (t - 1) * V + offs,
                mask=(valid_t & (t > 0) & vmask),
                other=0.0,
            )
            h0v = tl.load(h0_row + be * V + offs, mask=valid_t & (t == 0) & vmask, other=0.0)
            h_prev = tl.where(t == 0, h0v, h_prev)
            x = tl.load(x_proj + be * S * V + t * V + offs, mask=valid_t & vmask, other=0.0)
            fval = tl.load(f_t + be * S + t, mask=valid_t, other=0.0).to(tl.float32)

            z = tl.sum(h_prev[None, :] * Wt, axis=1) + x
            h_new = tl.inline_asm_elementwise(
                asm="tanh.approx.f32 $0, $1;",
                constraints="=f,f",
                args=[z],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            one_minus_f = 1.0 - fval
            residual = h_cur - fval * h_prev - one_minus_f * h_new
            b = -residual
            sech2 = 1.0 - h_new * h_new
            M = tl.where(eye, fval, 0.0) + one_minus_f * sech2[:, None] * Wt
            M = tl.where(mmask, M, 0.0)
            d = tl.sum(M * d[None, :], axis=1) + b
            d = tl.where(vmask, d, 0.0)
            tl.store(delta + be * S * V + t * V + offs, d, mask=valid_t & vmask)


def _assemble_tile_torch(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_prev_parts = []
    if start == 0:
        h_prev_parts.append(h0_row[:, None])
        if end - start > 1:
            h_prev_parts.append(h[:, start : end - 1])
    else:
        h_prev_parts.append(h[:, start - 1 : end - 1])
    h_prev = torch.cat(h_prev_parts, dim=1)
    h_cur = h[:, start:end]
    f = f_t[:, start:end]
    z = torch.einsum("btv,bvw->btw", h_prev, W_be) + x_proj[:, start:end]
    h_new = torch.tanh(z)
    residual = h_cur - f[..., None] * h_prev - (1.0 - f[..., None]) * h_new
    b = -residual
    sech2 = 1.0 - h_new * h_new
    V = h.shape[-1]
    eye = torch.eye(V, device=h.device, dtype=h.dtype)
    M = f[..., None, None] * eye[None, None]
    M = M + (1.0 - f[..., None, None]) * sech2[..., :, None] * W_be.transpose(-1, -2)[:, None]
    return M.contiguous(), b.contiguous()


def _local_tiles_torch(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Be, S, V = h.shape
    num_tiles = (S + tile_size - 1) // tile_size
    local_delta = torch.empty_like(h)
    summaries_M = torch.empty(Be, num_tiles, V, V, device=h.device, dtype=h.dtype)
    summaries_b = torch.empty(Be, num_tiles, V, device=h.device, dtype=h.dtype)
    eye = torch.eye(V, device=h.device, dtype=h.dtype).expand(Be, V, V)
    zero = torch.zeros(Be, V, device=h.device, dtype=h.dtype)

    for tile_id, start in enumerate(range(0, S, tile_size)):
        end = min(start + tile_size, S)
        M, b = _assemble_tile_torch(h, x_proj, f_t, W_be, h0_row, start, end)
        P = eye.clone()
        c = zero.clone()
        for j in range(end - start):
            P = torch.bmm(M[:, j], P)
            c = torch.bmm(M[:, j], c[:, :, None]).squeeze(-1) + b[:, j]
            local_delta[:, start + j] = c
        summaries_M[:, tile_id] = P
        summaries_b[:, tile_id] = c
    return local_delta, summaries_M, summaries_b


def _scan_summaries_torch(
    summaries_M: torch.Tensor,
    summaries_b: torch.Tensor,
) -> torch.Tensor:
    Be, num_tiles, V, _ = summaries_M.shape
    carries = torch.empty(Be, num_tiles, V, device=summaries_b.device, dtype=summaries_b.dtype)
    cur = torch.zeros(Be, V, device=summaries_b.device, dtype=summaries_b.dtype)
    for tile_id in range(num_tiles):
        carries[:, tile_id] = cur
        cur = torch.bmm(summaries_M[:, tile_id], cur[:, :, None]).squeeze(-1) + summaries_b[:, tile_id]
    return carries


def _scan_summaries_triton(
    summaries_M: torch.Tensor,
    summaries_b: torch.Tensor,
    block_v: int,
) -> torch.Tensor:
    assert triton is not None
    Be, num_tiles, V, _ = summaries_M.shape
    carries = torch.empty(Be, num_tiles, V, device=summaries_b.device, dtype=summaries_b.dtype)
    _summary_scan_kernel[(Be,)](
        summaries_M,
        summaries_b,
        carries,
        V,
        num_tiles,
        block_v,
        num_warps=1,
    )
    return carries


def _apply_carries_torch(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    carries: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    Be, S, _ = h.shape
    delta = torch.empty_like(h)
    for tile_id, start in enumerate(range(0, S, tile_size)):
        end = min(start + tile_size, S)
        M, b = _assemble_tile_torch(h, x_proj, f_t, W_be, h0_row, start, end)
        d = carries[:, tile_id]
        for j in range(end - start):
            d = torch.bmm(M[:, j], d[:, :, None]).squeeze(-1) + b[:, j]
            delta[:, start + j] = d
    return delta


def _tiled_newton_delta(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    *,
    tile_size: int,
    prefer_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TiledSolveStats]:
    Be, S, V = h.shape
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    num_tiles = (S + tile_size - 1) // tile_size
    stats = estimate_tiled_solve_memory(
        B=1,
        S=S,
        H=1,
        K=Be,
        V=V,
        tile_size=tile_size,
        dtype=h.dtype,
    )

    use_triton = (
        prefer_triton
        and TRITON_AVAILABLE
        and h.is_cuda
        and h.dtype == torch.float32
        and V <= 32
    )
    if use_triton:
        assert triton is not None
        block_v = max(16, triton.next_power_of_2(V))
        local_delta = torch.empty_like(h)
        summaries_M = torch.empty(Be, num_tiles, V, V, device=h.device, dtype=h.dtype)
        summaries_b = torch.empty(Be, num_tiles, V, device=h.device, dtype=h.dtype)
        grid = (Be, num_tiles)
        _local_tile_kernel[grid](
            h,
            x_proj,
            f_t,
            W_be,
            h0_row,
            local_delta,
            summaries_M,
            summaries_b,
            S,
            V,
            num_tiles,
            tile_size,
            block_v,
            num_warps=1,
        )
        carries = _scan_summaries_triton(summaries_M, summaries_b, block_v)
        delta = torch.empty_like(h)
        _apply_carry_kernel[grid](
            h,
            x_proj,
            f_t,
            W_be,
            h0_row,
            carries,
            delta,
            S,
            V,
            num_tiles,
            tile_size,
            block_v,
            num_warps=1,
        )
    else:
        local_delta, summaries_M, summaries_b = _local_tiles_torch(
            h, x_proj, f_t, W_be, h0_row, tile_size
        )
        carries = _scan_summaries_torch(summaries_M, summaries_b)
        delta = _apply_carries_torch(h, x_proj, f_t, W_be, h0_row, carries, tile_size)

    return delta, local_delta, summaries_M, stats


def m2rnn_pararnn_tiled_triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    config: TiledTritonConfig = TiledTritonConfig(),
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, TiledSolveStats]:
    """Run M2RNN forward with tiled/streaming Newton linear solves.

    This is a prototype and is intentionally not wired into the training
    default.  For parity experiments use small shapes and several Newton
    iterations, matching ``m2rnn_pararnn_forward``.
    """

    out_dtype = q.dtype
    qf, x_proj, f_t, W_be, h0_row, B, S, H, K, V = _prepare_rows(q, k, v, W, xf, h0)
    Be = B * H * K

    if config.init_strategy == "zero":
        h = torch.zeros(Be, S, V, device=q.device, dtype=qf.dtype)
    elif config.init_strategy == "chunk":
        h = torch.empty(Be, S, V, device=q.device, dtype=qf.dtype)
        h_cur = h0_row
        for t in range(S):
            z = torch.einsum("bv,bvw->bw", h_cur, W_be) + x_proj[:, t]
            h_new = torch.tanh(z)
            h_cur = f_t[:, t, None] * h_cur + (1.0 - f_t[:, t, None]) * h_new
            h[:, t] = h_cur
    else:
        raise ValueError(f"unknown init_strategy: {config.init_strategy}")

    stats = estimate_tiled_solve_memory(
        B=B,
        S=S,
        H=H,
        K=K,
        V=V,
        tile_size=config.tile_size,
        dtype=qf.dtype,
    )
    for _ in range(config.max_its):
        delta, _local_delta, _summaries_M, _stats = _tiled_newton_delta(
            h,
            x_proj,
            f_t,
            W_be,
            h0_row,
            tile_size=config.tile_size,
            prefer_triton=config.prefer_triton,
        )
        h = h + config.omega_sor * delta

    h_bshkv = h.view(B, H, K, S, V).permute(0, 3, 1, 2, 4)
    out = torch.einsum("bshk,bshkv->bshv", qf, h_bshkv)
    h_final = h_bshkv[:, -1].contiguous()
    if return_stats:
        return out.to(out_dtype), h_final.to(out_dtype), stats
    return out.to(out_dtype), h_final.to(out_dtype)


__all__ = [
    "TRITON_AVAILABLE",
    "TiledSolveStats",
    "TiledTritonConfig",
    "estimate_tiled_solve_memory",
    "m2rnn_pararnn_tiled_triton_forward",
]
