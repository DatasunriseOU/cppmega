"""Monolithic Mamba3 bwd_bwd chunk-kernel skeleton.

This module intentionally exposes a narrow, pre-expanded per-head layout.  It
is a compile/smoke vehicle for the chunk-level CUDA mapping, not a production
autograd path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


_CUDA_EXT: Any | None = None
_CUDA_EXT_ERROR: BaseException | None = None


def _load_cuda_ext() -> Any:
    global _CUDA_EXT, _CUDA_EXT_ERROR
    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        raise RuntimeError("Mamba3 monolithic chunk CUDA skeleton failed to load") from _CUDA_EXT_ERROR

    try:
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # The skeleton uses WMMA, so it does not require the architecture
            # accelerated `a` target.  Keep GB10 aligned with the repo's
            # existing CUDA extension convention.
            if (major, minor) == (12, 1):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1a"
            elif (major, minor) == (12, 0):
                os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

        src_dir = Path(__file__).resolve().parent / "cuda_ext"
        verbose = os.environ.get("CPPMEGA_VERBOSE_EXT_BUILD", "0") == "1"
        _CUDA_EXT = load(
            name="cppmega_mamba3_mono_chunk_skeleton_cuda",
            sources=[
                str(src_dir / "mamba3_mono_chunk_skeleton.cpp"),
                str(src_dir / "mamba3_mono_chunk_skeleton.cu"),
            ],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            verbose=verbose,
        )
        return _CUDA_EXT
    except BaseException as exc:  # pragma: no cover - host/compiler specific
        _CUDA_EXT_ERROR = exc
        raise


def cuda_available() -> bool:
    """Return whether the extension can be attempted on this host."""

    return torch.cuda.is_available()


def mono_chunk_skeleton(
    q: torch.Tensor,
    k: torch.Tensor,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dstates: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the CUDA skeleton.

    Shapes:
      * ``q``, ``k``: ``(B, S, H, R, 64)`` fp16
      * ``dout``, ``v``: ``(B, S, H, P)`` fp16, ``P <= 128``
      * ``mimo_v``, ``mimo_o``: ``(H, R, P)`` fp16
      * ``qk_dot``: ``(B, S, H, R, R)`` fp32
      * ``dt``, ``trap``: ``(B, H, S)`` fp32
      * ``dstates``: ``(B, H, 64, P)`` fp16

    Returns ``(dv, dmimo_v, dk_diag, dq_diag, lkq_checksum)`` in fp32.
    """

    ext = _load_cuda_ext()
    return ext.mono_chunk_skeleton(
        q.contiguous(),
        k.contiguous(),
        dout.contiguous(),
        v.contiguous(),
        mimo_v.contiguous(),
        mimo_o.contiguous(),
        qk_dot.contiguous(),
        dt.contiguous(),
        trap.contiguous(),
        dstates.contiguous(),
        int(chunk_size),
    )


def allocate_outputs(
    q: torch.Tensor,
    dout: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the skeleton output tuple without launching the kernel."""

    B, S, H, R, N = q.shape
    P = dout.shape[-1]
    nchunks = S // chunk_size
    opts = {"device": q.device, "dtype": torch.float32}
    return (
        torch.zeros((B, S, H, P), **opts),
        torch.zeros((B, H, R, P), **opts),
        torch.zeros((B, S, H, R, N), **opts),
        torch.zeros((B, S, H, R, N), **opts),
        torch.zeros((B, H, nchunks), **opts),
    )


def mono_chunk_skeleton_out(
    q: torch.Tensor,
    k: torch.Tensor,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dstates: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    chunk_size: int = 16,
    zero_outputs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the CUDA skeleton into caller-provided outputs.

    This is primarily for benchmarking: with ``zero_outputs=False`` the timed
    loop avoids allocation and memset kernels and measures the chunk kernel
    launch itself.
    """

    ext = _load_cuda_ext()
    dv, dmimo_v, dk_diag, dq_diag, lkq_checksum = outputs
    return ext.mono_chunk_skeleton_out(
        q.contiguous(),
        k.contiguous(),
        dout.contiguous(),
        v.contiguous(),
        mimo_v.contiguous(),
        mimo_o.contiguous(),
        qk_dot.contiguous(),
        dt.contiguous(),
        trap.contiguous(),
        dstates.contiguous(),
        dv,
        dmimo_v,
        dk_diag,
        dq_diag,
        lkq_checksum,
        int(chunk_size),
        bool(zero_outputs),
    )


def kernel_metadata() -> dict[str, Any]:
    """Static metadata for docs and smoke output."""

    try:
        ext = _load_cuda_ext()
        return dict(ext.kernel_metadata())
    except Exception as exc:  # pragma: no cover - build failure path
        return {"load_error": f"{type(exc).__name__}: {exc}"}
