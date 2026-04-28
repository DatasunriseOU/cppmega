"""CUDA local-tile affine scan prototype for M2RNN ParaRNN.

This module is intentionally scoped to the first production-relevant kernel:
one CUDA block owns one ``(batch, head, k_idx, tile)`` chain segment and
assembles the dense ``V x V`` affine map in shared/register state.  It does not
materialise the full per-token Jacobian ``A[B,S,H,K,V,V]`` or the per-token
local prefix ``local_prefix[Be,S,V,V]`` in the production path.  CUDA first
emits tile summaries, then a recompute apply kernel streams the within-tile
prefix and writes the final Newton delta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


_CUDA_EXT: Any | None = None
_CUDA_EXT_ERROR: BaseException | None = None


@dataclass(frozen=True)
class TiledCudaPararnnConfig:
    max_its: int = 3
    omega_sor: float = 1.0
    tile_size: int = 32


def reset_m2rnn_tiled_cuda_ext_cache() -> None:
    global _CUDA_EXT, _CUDA_EXT_ERROR
    _CUDA_EXT = None
    _CUDA_EXT_ERROR = None


def _load_cuda_ext() -> Any:
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        raise RuntimeError("M2RNN tiled CUDA extension failed to load") from _CUDA_EXT_ERROR

    try:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

        src_dir = Path(__file__).resolve().parent / "cuda_ext"
        verbose = os.environ.get("CPPMEGA_VERBOSE_EXT_BUILD", "0") == "1"
        cuda_flags = [
            "-O2",
            "-std=c++17",
            "--use_fast_math",
            "--ptxas-options=-v",
        ]
        _CUDA_EXT = load(
            name="cppmega_m2rnn_tiled_affine_scan_cuda",
            sources=[str(src_dir / "m2rnn_tiled_affine_scan.cu")],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=cuda_flags,
            verbose=verbose,
        )
        return _CUDA_EXT
    except BaseException as exc:  # pragma: no cover - host/build specific
        _CUDA_EXT_ERROR = exc
        raise


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

    for name, n in (("q", n_q), ("k", n_k), ("v", n_v), ("W", n_w), ("xf", n_f)):
        if H % n != 0:
            raise ValueError(f"{name} head count {n} does not divide broadcast head count {H}")

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


def _scan_tile_summaries(tile_A: torch.Tensor, tile_b: torch.Tensor) -> torch.Tensor:
    """Return the input delta for every tile using a CUDA summary scan.

    ``tile_A/tile_b`` describe ``tail = tile_A @ input + tile_b``.  The first
    tile starts from zero because the Newton correction has no state before
    ``t=0``; each later tile starts from the previous tile's tail.
    """

    return _load_cuda_ext().scan_tile_summaries(tile_A.contiguous(), tile_b.contiguous())


def _scan_tile_summaries_python(tile_A: torch.Tensor, tile_b: torch.Tensor) -> torch.Tensor:
    """Reference implementation for tests/debugging."""

    Be, n_tiles, V = tile_b.shape
    tile_inputs = torch.empty_like(tile_b)
    prev = torch.zeros(Be, V, device=tile_b.device, dtype=tile_b.dtype)
    for tile in range(n_tiles):
        tile_inputs[:, tile] = prev
        prev = torch.einsum("bij,bj->bi", tile_A[:, tile], prev) + tile_b[:, tile]
    return tile_inputs


def _use_warprow_v16(V: int, tile_size: int) -> bool:
    """Opt-in experimental V=16 CUDA kernels using one warp per matrix row."""

    return V == 16 and os.environ.get("CPPMEGA_M2RNN_WARPROW_V16", "0") == "1"


def _make_h0_row(
    h0: Optional[torch.Tensor],
    *,
    B: int,
    H: int,
    K: int,
    V: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if h0 is None:
        return torch.zeros(B * H * K, V, device=device, dtype=dtype)
    if h0.shape != (B, H, K, V):
        raise ValueError(f"h0 must have shape {(B, H, K, V)}, got {tuple(h0.shape)}")
    return h0.to(device=device, dtype=dtype).reshape(B * H * K, V).contiguous()


def m2rnn_pararnn_tiled_cuda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    *,
    h0: Optional[torch.Tensor] = None,
    config: TiledCudaPararnnConfig = TiledCudaPararnnConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the tiled CUDA ParaRNN prototype.

    The current implementation is forward/probe only: CUDA tensors,
    ``v_dim <= 16``, and no custom backward.  Inputs may be fp32 or bf16, but
    Newton solve accumulators and the returned tensors are fp32.
    """

    for name, tensor in (("q", q), ("k", k), ("v", v), ("W", W), ("xf", xf)):
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor")
    allowed_dtypes = {torch.float32, torch.bfloat16}
    for name, tensor in (("q", q), ("k", k), ("v", v), ("W", W), ("xf", xf)):
        if tensor.dtype not in allowed_dtypes:
            raise TypeError(f"{name} must be float32 or bfloat16, got {tensor.dtype}")
    if config.max_its < 1:
        raise ValueError("max_its must be positive")
    if config.tile_size < 1:
        raise ValueError("tile_size must be positive")

    q, k, v, W, xf, H = _broadcast_heads(q, k, v, W, xf)
    B, S, _, K = q.shape
    V = v.size(-1)
    if V > 16:
        raise ValueError(f"tiled CUDA prototype supports v_dim <= 16, got {V}")

    qf = q.to(dtype=torch.float32).contiguous()
    kf = k.to(dtype=torch.float32).contiguous()
    vf = v.to(dtype=torch.float32).contiguous()
    Wf = W.to(dtype=torch.float32).contiguous()
    xff = xf.to(dtype=torch.float32).contiguous()
    h0_row = _make_h0_row(h0, B=B, H=H, K=K, V=V, device=q.device, dtype=torch.float32)

    Be = B * H * K
    n_tiles = (S + int(config.tile_size) - 1) // int(config.tile_size)
    h = torch.zeros(Be, S, V, device=q.device, dtype=torch.float32)
    tile_A = torch.empty(Be, n_tiles, V, V, device=q.device, dtype=torch.float32)
    tile_b = torch.empty(Be, n_tiles, V, device=q.device, dtype=torch.float32)
    tile_inputs = torch.empty_like(tile_b)
    delta = torch.empty_like(h)
    ext = _load_cuda_ext()

    for _ in range(config.max_its):
        if _use_warprow_v16(V, int(config.tile_size)):
            summary_out = ext.tile_summaries_v16_warprow_out
            apply_out = ext.apply_tile_prefixes_v16_warprow_out
        else:
            summary_out = ext.tile_summaries_out
            apply_out = ext.apply_tile_prefixes_out

        summary_out(
            qf,
            kf,
            vf,
            Wf,
            xff,
            h.contiguous(),
            h0_row,
            tile_A,
            tile_b,
            int(config.tile_size),
        )
        ext.scan_tile_summaries_out(tile_A, tile_b, tile_inputs)
        apply_out(
            qf,
            kf,
            vf,
            Wf,
            xff,
            h.contiguous(),
            h0_row,
            tile_inputs,
            delta,
            int(config.tile_size),
        )
        h = h + float(config.omega_sor) * delta

    h_btehv = h.view(B, H, K, S, V).permute(0, 3, 1, 2, 4).contiguous()
    out = torch.einsum("bshk,bshkv->bshv", qf, h_btehv)
    h_final = h_btehv[:, -1].contiguous()
    return out, h_final


def local_tile_scan_debug(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h_traj: torch.Tensor,
    h0_row: torch.Tensor,
    *,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the raw CUDA local tile products for tests/probes."""

    ext = _load_cuda_ext()
    return ext.local_tile_scan_debug(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        W.contiguous(),
        xf.contiguous(),
        h_traj.contiguous(),
        h0_row.contiguous(),
        int(tile_size),
    )


def memory_accounting_bytes(B: int, S: int, H: int, K: int, V: int, tile_size: int) -> dict[str, int]:
    Be = B * H * K
    n_tiles = (S + tile_size - 1) // tile_size
    f32 = 4
    return {
        "forbidden_full_jacobian": Be * S * V * V * f32,
        "delta": Be * S * V * f32,
        "tile_A": Be * n_tiles * V * V * f32,
        "tile_b": Be * n_tiles * V * f32,
        "h_trajectory": Be * S * V * f32,
        "tile_inputs": Be * n_tiles * V * f32,
        "debug_only_local_delta": Be * S * V * f32,
        "debug_only_local_prefix": Be * S * V * V * f32,
    }
