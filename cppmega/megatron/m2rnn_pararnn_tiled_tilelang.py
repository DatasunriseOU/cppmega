"""Tiled/streaming ParaRNN assembly for the M2RNN recurrence.

This variant keeps the Newton Jacobian blocks local to a sequence tile.  A
Newton iteration is split into:

1. assemble each tile's affine summary ``delta_tail = A_tile @ carry + b_tile``;
2. scan the tile summaries on GPU to get one carry per tile;
3. re-assemble each tile and stream the carry application to write deltas.

The full ``A[B, S, H, K, V, V]`` tensor is never materialized.  The optional
TileLang path currently targets the summary and apply passes for
CUDA/fp32/V=16; if JIT compile or launch fails, callers can fall back to the
same tiled torch contract.
"""

from __future__ import annotations

import io
import math
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

import torch

from cppmega.megatron.m2rnn_pararnn import (
    PararnnConfig,
    _parallel_reduce_dense,
    m2rnn_pararnn_forward,
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton-less envs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


Backend = Literal["auto", "torch", "tilelang"]
SummaryVariant = Literal["serial", "parallel_shared_old"]


@dataclass(frozen=True)
class TiledTileLangConfig:
    """Runtime knobs for the tiled TileLang M2RNN path."""

    max_its: int = 3
    omega_sor: float = 1.0
    init_strategy: str = "zero"
    tile_len: int = 32
    backend: Backend = "auto"
    allow_tilelang_fallback: bool = True
    summary_variant: SummaryVariant = "parallel_shared_old"


@dataclass
class TiledTileLangStats:
    """Memory-contract and TileLang viability data for probes/tests."""

    backend_requested: str
    backend_used: str
    tile_len: int
    n_tiles: int
    be: int
    s: int
    v_dim: int
    max_tile_jac_elements: int
    full_jac_elements_avoided: int
    summary_a_elements: int
    summary_b_elements: int
    tilelang_attempted: bool = False
    tilelang_used: bool = False
    tilelang_summary_attempted: bool = False
    tilelang_summary_used: bool = False
    triton_scan_attempted: bool = False
    triton_scan_used: bool = False
    tilelang_scan_attempted: bool = False
    tilelang_scan_used: bool = False
    tilelang_apply_attempted: bool = False
    tilelang_apply_used: bool = False
    tilelang_compile_log: str = ""
    tilelang_summary_compile_log: str = ""
    triton_scan_compile_log: str = ""
    tilelang_scan_compile_log: str = ""
    tilelang_apply_compile_log: str = ""
    torch_materialized_tile_jac_elements: int = 0

    @property
    def max_tile_jac_bytes_fp32(self) -> int:
        return self.max_tile_jac_elements * 4

    @property
    def full_jac_bytes_fp32(self) -> int:
        return self.full_jac_elements_avoided * 4


def _validate_config(config: TiledTileLangConfig) -> None:
    if config.tile_len not in (16, 32, 64):
        raise ValueError(f"tile_len must be one of 16/32/64, got {config.tile_len}")
    if config.max_its < 0:
        raise ValueError(f"max_its must be non-negative, got {config.max_its}")
    if config.backend not in ("auto", "torch", "tilelang"):
        raise ValueError(f"unknown backend: {config.backend}")
    if config.summary_variant not in ("serial", "parallel_shared_old"):
        raise ValueError(f"unknown summary_variant: {config.summary_variant}")


def _broadcast_inputs(
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


def _prepare_flat_problem(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
]:
    B, S, _, k_dim = q.shape
    v_dim = v.size(-1)
    q, k, v, W, xf, H = _broadcast_inputs(q, k, v, W, xf)

    compute_dtype = torch.promote_types(torch.float32, q.dtype)
    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    vf = v.to(compute_dtype)
    Wf = W.to(compute_dtype)
    xff = xf.to(compute_dtype)
    Be = B * H * k_dim

    x_proj = kf[..., :, None] * vf[..., None, :]
    x_proj = x_proj.permute(0, 2, 3, 1, 4).reshape(Be, S, v_dim).contiguous()

    f_t = (
        xff.permute(0, 2, 1)
        .unsqueeze(2)
        .expand(B, H, k_dim, S)
        .reshape(Be, S)
        .contiguous()
    )

    W_be = (
        Wf.unsqueeze(0)
        .unsqueeze(2)
        .expand(B, H, k_dim, v_dim, v_dim)
        .reshape(Be, v_dim, v_dim)
        .contiguous()
    )

    if h0 is None:
        h0_row = torch.zeros(Be, v_dim, device=q.device, dtype=compute_dtype)
    else:
        h0_row = h0.to(compute_dtype).reshape(Be, v_dim).contiguous()

    return qf, x_proj, f_t, W_be, h0_row, q, B, S, H, k_dim


def _initial_guess(
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    *,
    init_strategy: str,
) -> torch.Tensor:
    Be, S, v_dim = x_proj.shape
    if init_strategy == "zero":
        return torch.zeros(Be, S, v_dim, device=x_proj.device, dtype=x_proj.dtype)
    if init_strategy != "chunk":
        raise ValueError(f"unknown init_strategy: {init_strategy}")

    h = torch.empty(Be, S, v_dim, device=x_proj.device, dtype=x_proj.dtype)
    h_cur = h0_row
    for t in range(S):
        z = torch.einsum("bv,bvj->bj", h_cur, W_be) + x_proj[:, t]
        h_new = torch.tanh(z)
        f_bcast = f_t[:, t, None]
        h_cur = f_bcast * h_cur + (1.0 - f_bcast) * h_new
        h[:, t] = h_cur
    return h


def _tile_residual_and_jacobian_be(
    h_traj: torch.Tensor,
    x_proj: torch.Tensor,
    f: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble one tile's Newton residual and subdiagonal Jacobian."""

    h_tile = h_traj[:, start:end]
    if start == 0:
        h_prev0 = h0_row
    else:
        h_prev0 = h_traj[:, start - 1]

    if end - start == 1:
        h_prev = h_prev0[:, None, :]
    else:
        h_prev = torch.cat([h_prev0[:, None, :], h_traj[:, start : end - 1]], dim=1)

    z = torch.einsum("btv,bvw->btw", h_prev, W_be) + x_proj[:, start:end]
    h_new = torch.tanh(z)
    f_tile = f[:, start:end]
    f_b = f_tile[..., None]
    residual = h_tile - f_b * h_prev - (1.0 - f_b) * h_new

    v_dim = h_traj.size(-1)
    eye_v = torch.eye(v_dim, device=W_be.device, dtype=W_be.dtype)
    sech2 = 1.0 - h_new * h_new
    f_bb = f_tile[..., None, None]
    nonlin_block = sech2[..., :, None] * W_be.transpose(-1, -2)[:, None, :, :]
    jac = -f_bb * eye_v[None, None, :, :] - (1.0 - f_bb) * nonlin_block
    return residual, jac


def _tile_summary_torch(jac: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``A_tile, b_tile`` for ``delta_tail = A_tile @ carry + b_tile``."""

    Be, _, v_dim, _ = jac.shape
    P = torch.eye(v_dim, device=jac.device, dtype=jac.dtype).expand(Be, v_dim, v_dim).clone()
    b = torch.zeros(Be, v_dim, device=jac.device, dtype=jac.dtype)
    for local_t in range(jac.size(1)):
        J = jac[:, local_t]
        P = -torch.bmm(J, P)
        b = rhs[:, local_t] - torch.bmm(J, b[:, :, None]).squeeze(-1)
    return P, b


def _tile_apply_scan_torch(jac: torch.Tensor, rhs: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
    """Apply one tile carry and solve the local bidiagonal system."""

    jac_c = jac.contiguous()
    rhs_c = rhs.contiguous()
    rhs_c[:, 0] -= torch.einsum("bij,bj->bi", jac_c[:, 0], carry)
    jac_c[:, 0] = 0
    return _parallel_reduce_dense(jac_c, rhs_c)


def _scan_tile_summaries(
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    Be, n_tiles, v_dim, _ = summary_A.shape
    carries = torch.empty(Be, n_tiles, v_dim, device=summary_A.device, dtype=summary_A.dtype)
    carry = torch.zeros(Be, v_dim, device=summary_A.device, dtype=summary_A.dtype)
    for tile_idx in range(n_tiles):
        carries[:, tile_idx] = carry
        carry = torch.einsum("bij,bj->bi", summary_A[:, tile_idx], carry) + summary_b[:, tile_idx]
    return carries, carry


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_scan_kernel(
        SummaryA,
        SummaryB,
        Carries,
        N_TILES: tl.constexpr,
        V: tl.constexpr,
    ):
        be_i = tl.program_id(0)
        offs = tl.arange(0, 16)
        rows = tl.arange(0, 16)[:, None]
        cols = tl.arange(0, 16)[None, :]
        carry = tl.full((16,), 0.0, tl.float32)

        for tile_i in tl.static_range(0, N_TILES):
            tl.store(Carries + be_i * N_TILES * V + tile_i * V + offs, carry)
            mat = tl.load(
                SummaryA
                + be_i * N_TILES * V * V
                + tile_i * V * V
                + rows * V
                + cols
            )
            bias = tl.load(SummaryB + be_i * N_TILES * V + tile_i * V + offs)
            carry = tl.sum(mat * carry[None, :], axis=1) + bias


def _try_triton_scan(
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
    carries: torch.Tensor,
) -> tuple[bool, str]:
    if not TRITON_AVAILABLE:
        return False, "triton is not importable"
    log = io.StringIO()
    try:
        _triton_scan_kernel[(summary_A.size(0),)](
            summary_A,
            summary_b,
            carries,
            N_TILES=summary_A.size(1),
            V=summary_A.size(2),
            num_warps=1,
        )
        return True, ""
    except Exception:
        log.write(traceback.format_exc())
        return False, log.getvalue()


@lru_cache(maxsize=1)
def _tilelang_scan_kernel():
    import tilelang
    from tilelang import language as T

    V = 16
    be = T.dynamic("be")
    n_tiles = T.dynamic("n_tiles")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        @T.prim_func
        def main(
            SummaryA: T.Tensor([be, n_tiles, V, V], T.float32),
            SummaryB: T.Tensor([be, n_tiles, V], T.float32),
            Carries: T.Tensor([be, n_tiles, V], T.float32),
        ):
            with T.Kernel(be, threads=128) as be_i:
                carry = T.alloc_fragment([V], T.float32)
                carry_next = T.alloc_fragment([V], T.float32)

                for vi in T.serial(V):
                    carry[vi] = 0.0

                for tile_i in T.serial(n_tiles):
                    for vi in T.serial(V):
                        Carries[be_i, tile_i, vi] = carry[vi]

                    for vi in T.serial(V):
                        carry_next[vi] = SummaryB[be_i, tile_i, vi]
                        for vj in T.serial(V):
                            carry_next[vi] += SummaryA[be_i, tile_i, vi, vj] * carry[vj]

                    for vi in T.serial(V):
                        carry[vi] = carry_next[vi]

        return main

    return kernel_builder()


def _try_tilelang_scan(
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
    carries: torch.Tensor,
) -> tuple[bool, str]:
    log = io.StringIO()
    try:
        kernel = _tilelang_scan_kernel()
        kernel(summary_A, summary_b, carries)
        return True, ""
    except Exception:
        log.write(traceback.format_exc())
        return False, log.getvalue()


def _tilelang_eligible(
    config: TiledTileLangConfig,
    x_proj: torch.Tensor,
    v_dim: int,
) -> bool:
    if config.backend == "torch":
        return False
    if v_dim != 16:
        return False
    if x_proj.dtype != torch.float32:
        return False
    return x_proj.is_cuda


@lru_cache(maxsize=8)
def _tilelang_summary_kernel(tile_len: int):
    import tilelang
    from tilelang import language as T

    V = 16
    be = T.dynamic("be")
    seq = T.dynamic("seq")
    n_tiles = T.dynamic("n_tiles")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        @T.prim_func
        def main(
            H: T.Tensor([be, seq, V], T.float32),
            X: T.Tensor([be, seq, V], T.float32),
            F: T.Tensor([be, seq], T.float32),
            W: T.Tensor([be, V, V], T.float32),
            H0: T.Tensor([be, V], T.float32),
            SummaryA: T.Tensor([be, n_tiles, V, V], T.float32),
            SummaryB: T.Tensor([be, n_tiles, V], T.float32),
        ):
            with T.Kernel(be, n_tiles, threads=128) as (be_i, tile_i):
                W_shared = T.alloc_shared([V, V], T.float32)
                P = T.alloc_fragment([V, V], T.float32)
                P_next = T.alloc_fragment([V, V], T.float32)
                b = T.alloc_fragment([V], T.float32)
                b_next = T.alloc_fragment([V], T.float32)
                h_prev = T.alloc_fragment([V], T.float32)
                z = T.alloc_fragment([V], T.float32)
                h_new = T.alloc_fragment([V], T.float32)
                rhs = T.alloc_fragment([V], T.float32)

                T.copy(W[be_i, 0:V, 0:V], W_shared)

                for i in T.serial(V):
                    for j in T.serial(V):
                        P[i, j] = T.if_then_else(i == j, 1.0, 0.0)
                for i in T.serial(V):
                    b[i] = 0.0

                for local_t in T.serial(tile_len):
                    s_i = tile_i * tile_len + local_t
                    if s_i < seq:
                        f_i = F[be_i, s_i]
                        for vi in T.serial(V):
                            if s_i == 0:
                                h_prev[vi] = H0[be_i, vi]
                            else:
                                h_prev[vi] = H[be_i, s_i - 1, vi]

                        for vi in T.serial(V):
                            z[vi] = X[be_i, s_i, vi]
                            for vj in T.unroll(V):
                                z[vi] += h_prev[vj] * W_shared[vj, vi]
                            h_new[vi] = T.tanh(z[vi])
                            rhs[vi] = -(
                                H[be_i, s_i, vi]
                                - f_i * h_prev[vi]
                                - (1.0 - f_i) * h_new[vi]
                            )

                        for vi in T.serial(V):
                            for vj in T.serial(V):
                                sech2 = 1.0 - h_new[vi] * h_new[vi]
                                P_next[vi, vj] = f_i * P[vi, vj]
                                for vk in T.unroll(V):
                                    P_next[vi, vj] += (
                                        (1.0 - f_i) * sech2 * W_shared[vk, vi] * P[vk, vj]
                                    )

                        for vi in T.serial(V):
                            sech2 = 1.0 - h_new[vi] * h_new[vi]
                            b_next[vi] = rhs[vi] + f_i * b[vi]
                            for vk in T.unroll(V):
                                b_next[vi] += (1.0 - f_i) * sech2 * W_shared[vk, vi] * b[vk]

                        for vi in T.serial(V):
                            for vj in T.serial(V):
                                P[vi, vj] = P_next[vi, vj]
                        for vi in T.serial(V):
                            b[vi] = b_next[vi]

                for vi in T.serial(V):
                    for vj in T.serial(V):
                        SummaryA[be_i, tile_i, vi, vj] = P[vi, vj]
                for vi in T.serial(V):
                    SummaryB[be_i, tile_i, vi] = b[vi]

        return main

    return kernel_builder()


@lru_cache(maxsize=8)
def _tilelang_summary_parallel_shared_old_kernel(tile_len: int):
    import tilelang
    from tilelang import language as T

    V = 16
    be = T.dynamic("be")
    seq = T.dynamic("seq")
    n_tiles = T.dynamic("n_tiles")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        mat_layout = T.Fragment(
            [V, V],
            forward_thread_fn=lambda i, j: (i * V + j) % 128,
            forward_index_fn=lambda i, j: (i * V + j) // 128,
        )

        @T.prim_func
        def main(
            H: T.Tensor([be, seq, V], T.float32),
            X: T.Tensor([be, seq, V], T.float32),
            F: T.Tensor([be, seq], T.float32),
            W: T.Tensor([be, V, V], T.float32),
            H0: T.Tensor([be, V], T.float32),
            SummaryA: T.Tensor([be, n_tiles, V, V], T.float32),
            SummaryB: T.Tensor([be, n_tiles, V], T.float32),
        ):
            with T.Kernel(be, n_tiles, threads=128) as (be_i, tile_i):
                W_shared = T.alloc_shared([V, V], T.float32)
                P_old = T.alloc_shared([V, V], T.float32)
                b_old = T.alloc_shared([V], T.float32)
                P_next = T.alloc_fragment([V, V], T.float32)
                b_next = T.alloc_fragment([V], T.float32)
                h_prev = T.alloc_fragment([V], T.float32)
                z = T.alloc_fragment([V], T.float32)
                h_new = T.alloc_fragment([V], T.float32)
                rhs = T.alloc_fragment([V], T.float32)

                T.copy(W[be_i, 0:V, 0:V], W_shared)

                for i, j in T.Parallel(V, V, loop_layout=mat_layout):
                    P_old[i, j] = T.if_then_else(i == j, 1.0, 0.0)
                for i in T.serial(V):
                    b_old[i] = 0.0

                for local_t in T.serial(tile_len):
                    s_i = tile_i * tile_len + local_t
                    if s_i < seq:
                        f_i = F[be_i, s_i]
                        for vi in T.serial(V):
                            if s_i == 0:
                                h_prev[vi] = H0[be_i, vi]
                            else:
                                h_prev[vi] = H[be_i, s_i - 1, vi]

                        for vi in T.serial(V):
                            z[vi] = X[be_i, s_i, vi]
                            for vj in T.unroll(V):
                                z[vi] += h_prev[vj] * W_shared[vj, vi]
                            h_new[vi] = T.tanh(z[vi])
                            rhs[vi] = -(
                                H[be_i, s_i, vi]
                                - f_i * h_prev[vi]
                                - (1.0 - f_i) * h_new[vi]
                            )

                        for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                            sech2 = 1.0 - h_new[vi] * h_new[vi]
                            P_next[vi, vj] = 0.0
                            for vk in T.unroll(V):
                                P_next[vi, vj] += (
                                    T.if_then_else(vk == vi, f_i, 0.0)
                                    + (1.0 - f_i) * sech2 * W_shared[vk, vi]
                                ) * P_old[vk, vj]

                        for vi in T.serial(V):
                            sech2 = 1.0 - h_new[vi] * h_new[vi]
                            b_next[vi] = rhs[vi] + f_i * b_old[vi]
                            for vk in T.unroll(V):
                                b_next[vi] += (1.0 - f_i) * sech2 * W_shared[vk, vi] * b_old[vk]

                        for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                            P_old[vi, vj] = P_next[vi, vj]
                        for vi in T.serial(V):
                            b_old[vi] = b_next[vi]

                for vi, vj in T.Parallel(V, V, loop_layout=mat_layout):
                    SummaryA[be_i, tile_i, vi, vj] = P_old[vi, vj]
                for vi in T.serial(V):
                    SummaryB[be_i, tile_i, vi] = b_old[vi]

        return main

    return kernel_builder()


def _try_tilelang_summary(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
    tile_len: int,
    summary_variant: SummaryVariant = "serial",
) -> tuple[bool, str]:
    log = io.StringIO()
    try:
        if summary_variant == "parallel_shared_old":
            kernel = _tilelang_summary_parallel_shared_old_kernel(tile_len)
        else:
            kernel = _tilelang_summary_kernel(tile_len)
        kernel(h, x_proj, f_t, W_be, h0_row, summary_A, summary_b)
        return True, ""
    except Exception:
        log.write(traceback.format_exc())
        return False, log.getvalue()


def _try_tilelang_summary_with_serial_fallback(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
    tile_len: int,
    summary_variant: SummaryVariant,
) -> tuple[bool, str, SummaryVariant]:
    ok, compile_log = _try_tilelang_summary(
        h,
        x_proj,
        f_t,
        W_be,
        h0_row,
        summary_A,
        summary_b,
        tile_len,
        summary_variant,
    )
    if ok or summary_variant != "parallel_shared_old":
        return ok, compile_log, summary_variant

    serial_ok, serial_log = _try_tilelang_summary(
        h,
        x_proj,
        f_t,
        W_be,
        h0_row,
        summary_A,
        summary_b,
        tile_len,
        "serial",
    )
    if serial_ok:
        return True, compile_log, "serial"
    return False, compile_log + "\nserial_summary_fallback_failed:\n" + serial_log, summary_variant


@lru_cache(maxsize=8)
def _tilelang_apply_kernel(tile_len: int):
    import tilelang
    from tilelang import language as T

    V = 16
    be = T.dynamic("be")
    seq = T.dynamic("seq")
    n_tiles = T.dynamic("n_tiles")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        },
    )
    def kernel_builder():
        @T.prim_func
        def main(
            H: T.Tensor([be, seq, V], T.float32),
            X: T.Tensor([be, seq, V], T.float32),
            F: T.Tensor([be, seq], T.float32),
            W: T.Tensor([be, V, V], T.float32),
            H0: T.Tensor([be, V], T.float32),
            Carries: T.Tensor([be, n_tiles, V], T.float32),
            Delta: T.Tensor([be, seq, V], T.float32),
        ):
            with T.Kernel(be, n_tiles, threads=128) as (be_i, tile_i):
                W_shared = T.alloc_shared([V, V], T.float32)
                h_prev = T.alloc_fragment([V], T.float32)
                z = T.alloc_fragment([V], T.float32)
                h_new = T.alloc_fragment([V], T.float32)
                rhs = T.alloc_fragment([V], T.float32)
                delta_prev = T.alloc_fragment([V], T.float32)
                delta_cur = T.alloc_fragment([V], T.float32)

                T.copy(W[be_i, 0:V, 0:V], W_shared)

                for vi in T.serial(V):
                    delta_prev[vi] = Carries[be_i, tile_i, vi]

                for local_t in T.serial(tile_len):
                    s_i = tile_i * tile_len + local_t
                    if s_i < seq:
                        f_i = F[be_i, s_i]
                        for vi in T.serial(V):
                            if s_i == 0:
                                h_prev[vi] = H0[be_i, vi]
                            else:
                                h_prev[vi] = H[be_i, s_i - 1, vi]

                        for vi in T.serial(V):
                            z[vi] = X[be_i, s_i, vi]
                            for vj in T.unroll(V):
                                z[vi] += h_prev[vj] * W_shared[vj, vi]
                            h_new[vi] = T.tanh(z[vi])
                            rhs[vi] = -(
                                H[be_i, s_i, vi]
                                - f_i * h_prev[vi]
                                - (1.0 - f_i) * h_new[vi]
                            )

                        for vi in T.serial(V):
                            sech2 = 1.0 - h_new[vi] * h_new[vi]
                            delta_cur[vi] = rhs[vi] + f_i * delta_prev[vi]
                            for vj in T.unroll(V):
                                delta_cur[vi] += (
                                    (1.0 - f_i) * sech2 * W_shared[vj, vi] * delta_prev[vj]
                                )

                        for vi in T.serial(V):
                            Delta[be_i, s_i, vi] = delta_cur[vi]
                            delta_prev[vi] = delta_cur[vi]

        return main

    return kernel_builder()


def _try_tilelang_apply(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    carries: torch.Tensor,
    delta: torch.Tensor,
    tile_len: int,
) -> tuple[bool, str]:
    log = io.StringIO()
    try:
        kernel = _tilelang_apply_kernel(tile_len)
        kernel(h, x_proj, f_t, W_be, h0_row, carries, delta)
        return True, ""
    except Exception:
        log.write(traceback.format_exc())
        return False, log.getvalue()


def _torch_summary_pass(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    summary_A: torch.Tensor,
    summary_b: torch.Tensor,
    tile_len: int,
) -> int:
    S = h.size(1)
    max_tile_jac_elements = 0
    for tile_idx, start in enumerate(range(0, S, tile_len)):
        end = min(start + tile_len, S)
        residual, jac = _tile_residual_and_jacobian_be(h, x_proj, f_t, W_be, h0_row, start, end)
        rhs = -residual
        A_tile, b_tile = _tile_summary_torch(jac, rhs)
        summary_A[:, tile_idx] = A_tile
        summary_b[:, tile_idx] = b_tile
        max_tile_jac_elements = max(max_tile_jac_elements, jac.numel())
    return max_tile_jac_elements


def _apply_pass_torch(
    h: torch.Tensor,
    x_proj: torch.Tensor,
    f_t: torch.Tensor,
    W_be: torch.Tensor,
    h0_row: torch.Tensor,
    carries: torch.Tensor,
    tile_len: int,
) -> tuple[torch.Tensor, int]:
    S = h.size(1)
    delta = torch.empty_like(h)
    max_tile_jac_elements = 0
    for tile_idx, start in enumerate(range(0, S, tile_len)):
        end = min(start + tile_len, S)
        residual, jac = _tile_residual_and_jacobian_be(h, x_proj, f_t, W_be, h0_row, start, end)
        rhs = -residual
        delta[:, start:end] = _tile_apply_scan_torch(jac, rhs, carries[:, tile_idx])
        max_tile_jac_elements = max(max_tile_jac_elements, jac.numel())
    return delta, max_tile_jac_elements


def m2rnn_pararnn_tiled_tilelang_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    config: TiledTileLangConfig = TiledTileLangConfig(),
    return_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, TiledTileLangStats]:
    """Forward pass with tiled Newton assembly and scan.

    The torch fallback is the reference implementation for the memory contract.
    The TileLang backend, when eligible, is used for the tile summary assembly;
    the inter-tile scan and carry-application pass remain PyTorch in this first
    staged version.
    """

    _validate_config(config)
    qf, x_proj, f_t, W_be, h0_row, q_broadcast, B, S, H, k_dim = _prepare_flat_problem(
        q, k, v, W, xf, h0
    )
    del q_broadcast
    out_dtype = q.dtype
    Be, _, v_dim = x_proj.shape
    n_tiles = math.ceil(S / config.tile_len)
    h = _initial_guess(
        x_proj,
        f_t,
        W_be,
        h0_row,
        init_strategy=config.init_strategy,
    )

    stats = TiledTileLangStats(
        backend_requested=config.backend,
        backend_used="torch",
        tile_len=config.tile_len,
        n_tiles=n_tiles,
        be=Be,
        s=S,
        v_dim=v_dim,
        max_tile_jac_elements=0,
        full_jac_elements_avoided=Be * S * v_dim * v_dim,
        summary_a_elements=Be * n_tiles * v_dim * v_dim,
        summary_b_elements=Be * n_tiles * v_dim,
    )

    use_tilelang_summary = _tilelang_eligible(config, x_proj, v_dim)
    use_tilelang_scan = _tilelang_eligible(config, x_proj, v_dim)
    use_tilelang_apply = _tilelang_eligible(config, x_proj, v_dim)
    active_summary_variant = config.summary_variant
    for _ in range(config.max_its):
        summary_A = torch.empty(Be, n_tiles, v_dim, v_dim, device=x_proj.device, dtype=x_proj.dtype)
        summary_b = torch.empty(Be, n_tiles, v_dim, device=x_proj.device, dtype=x_proj.dtype)

        if use_tilelang_summary:
            stats.tilelang_attempted = True
            stats.tilelang_summary_attempted = True
            ok, compile_log, summary_variant_used = _try_tilelang_summary_with_serial_fallback(
                h,
                x_proj,
                f_t,
                W_be,
                h0_row,
                summary_A,
                summary_b,
                config.tile_len,
                active_summary_variant,
            )
            if ok:
                active_summary_variant = summary_variant_used
                stats.backend_used = (
                    "tilelang-summary-parallel-shared-old+pending-apply"
                    if summary_variant_used == "parallel_shared_old"
                    else "tilelang-summary+pending-apply"
                )
                stats.tilelang_used = True
                stats.tilelang_summary_used = True
                stats.max_tile_jac_elements = max(
                    stats.max_tile_jac_elements,
                    Be * min(config.tile_len, S) * v_dim * v_dim,
                )
            else:
                stats.tilelang_summary_compile_log = compile_log
                stats.tilelang_compile_log = compile_log
                if config.backend == "tilelang" and not config.allow_tilelang_fallback:
                    raise RuntimeError("TileLang M2RNN summary kernel failed") from None
                use_tilelang_summary = False

        if not use_tilelang_summary:
            max_summary = _torch_summary_pass(
                h,
                x_proj,
                f_t,
                W_be,
                h0_row,
                summary_A,
                summary_b,
                config.tile_len,
            )
            stats.max_tile_jac_elements = max(stats.max_tile_jac_elements, max_summary)
            stats.torch_materialized_tile_jac_elements = max(
                stats.torch_materialized_tile_jac_elements,
                max_summary,
            )

        carries = torch.empty(Be, n_tiles, v_dim, device=x_proj.device, dtype=x_proj.dtype)
        if use_tilelang_scan:
            stats.tilelang_attempted = True
            stats.triton_scan_attempted = True
            ok, compile_log = _try_triton_scan(summary_A, summary_b, carries)
            if ok:
                stats.tilelang_used = True
                stats.triton_scan_used = True
            else:
                stats.triton_scan_compile_log = compile_log
                stats.tilelang_scan_attempted = True
                ok, compile_log = _try_tilelang_scan(summary_A, summary_b, carries)
                if ok:
                    stats.tilelang_used = True
                    stats.tilelang_scan_used = True
                else:
                    stats.tilelang_scan_compile_log = compile_log
                    use_tilelang_scan = False

            if not (stats.triton_scan_used or stats.tilelang_scan_used):
                stats.tilelang_compile_log = "\n".join(
                    log
                    for log in (
                        stats.tilelang_summary_compile_log,
                        stats.triton_scan_compile_log,
                        stats.tilelang_scan_compile_log,
                        stats.tilelang_apply_compile_log,
                    )
                    if log
                )
                if config.backend == "tilelang" and not config.allow_tilelang_fallback:
                    raise RuntimeError("M2RNN GPU scan kernel failed") from None

        if not use_tilelang_scan:
            carries, _final_delta = _scan_tile_summaries(summary_A, summary_b)
        delta = torch.empty_like(h)
        if use_tilelang_apply:
            stats.tilelang_attempted = True
            stats.tilelang_apply_attempted = True
            ok, compile_log = _try_tilelang_apply(
                h,
                x_proj,
                f_t,
                W_be,
                h0_row,
                carries,
                delta,
                config.tile_len,
            )
            if ok:
                stats.tilelang_used = True
                stats.tilelang_apply_used = True
                stats.max_tile_jac_elements = max(
                    stats.max_tile_jac_elements,
                    Be * min(config.tile_len, S) * v_dim * v_dim,
                )
            else:
                stats.tilelang_apply_compile_log = compile_log
                stats.tilelang_compile_log = "\n".join(
                    log
                    for log in (
                        stats.tilelang_summary_compile_log,
                        stats.triton_scan_compile_log,
                        stats.tilelang_scan_compile_log,
                        stats.tilelang_apply_compile_log,
                    )
                    if log
                )
                if config.backend == "tilelang" and not config.allow_tilelang_fallback:
                    raise RuntimeError("TileLang M2RNN apply kernel failed") from None
                use_tilelang_apply = False

        if not use_tilelang_apply:
            delta, max_apply = _apply_pass_torch(
                h,
                x_proj,
                f_t,
                W_be,
                h0_row,
                carries,
                config.tile_len,
            )
            stats.max_tile_jac_elements = max(stats.max_tile_jac_elements, max_apply)
            stats.torch_materialized_tile_jac_elements = max(
                stats.torch_materialized_tile_jac_elements,
                max_apply,
            )

        gpu_scan_used = stats.triton_scan_used or stats.tilelang_scan_used
        summary_backend = (
            "tilelang-summary-parallel-shared-old"
            if active_summary_variant == "parallel_shared_old"
            else "tilelang-summary"
        )
        if stats.tilelang_summary_used and gpu_scan_used and stats.tilelang_apply_used:
            scan_backend = "triton-scan" if stats.triton_scan_used else "tilelang-scan"
            stats.backend_used = f"{summary_backend}+{scan_backend}+tilelang-apply"
        elif stats.tilelang_summary_used and stats.tilelang_apply_used:
            stats.backend_used = f"{summary_backend}+torch-scan+tilelang-apply"
        elif stats.tilelang_summary_used:
            stats.backend_used = f"{summary_backend}+torch-apply"
        elif stats.tilelang_apply_used:
            stats.backend_used = "torch-summary+tilelang-apply"
        else:
            stats.backend_used = "torch"
        h = h + config.omega_sor * delta

    h_btehv = h.view(B, H, k_dim, S, v_dim).permute(0, 3, 1, 2, 4)
    out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)
    h_final = h_btehv[:, -1].contiguous()
    if return_stats:
        return out.to(out_dtype), h_final.to(out_dtype), stats
    return out.to(out_dtype), h_final.to(out_dtype)


def m2rnn_pararnn_tiled_tilelang_one_step_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    tile_len: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TiledTileLangStats]:
    """Convenience helper comparing one Newton step against the full ParaRNN path."""

    tiled_out, tiled_h, stats = m2rnn_pararnn_tiled_tilelang_forward(
        q,
        k,
        v,
        W,
        xf,
        h0=h0,
        config=TiledTileLangConfig(max_its=1, tile_len=tile_len, backend="torch"),
        return_stats=True,
    )
    full_out, full_h = m2rnn_pararnn_forward(
        q,
        k,
        v,
        W,
        xf,
        h0=h0,
        config=PararnnConfig(max_its=1, init_strategy="zero", chunk_size=0),
    )
    return tiled_out, tiled_h, full_out, full_h, stats
