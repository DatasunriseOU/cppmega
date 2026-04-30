"""CUDA extension wrapper for the standalone R x R diagonal microbench."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_EXTENSION: Any | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {value}")
    return value


def _env_flag(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value not in (0, 1):
        raise RuntimeError(f"{name} must be 0 or 1, got {value}")
    return value


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value)


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    this_dir = Path(__file__).resolve().parent
    threads = _env_int("RR_DIAG_THREADS", 256)
    p_tile = _env_int("RR_DIAG_DMIMO_P_TILE", 32)
    unroll = _env_int("RR_DIAG_DMIMO_UNROLL", 1)
    broadcast_qk = _env_flag("RR_DIAG_DMIMO_BROADCAST_QK", 0)
    mono_p_tile = _env_int("RR_DIAG_MONO_P_TILE", 32)
    if threads % 32 != 0:
        raise RuntimeError(f"RR_DIAG_THREADS must be a warp multiple, got {threads}")
    suffix = _safe_suffix(os.environ.get("RR_DIAG_CUDA_EXT_SUFFIX", ""))
    name = f"rr_diag_cuda_ext_wave11_t{threads}_p{p_tile}_u{unroll}_b{broadcast_qk}_m{mono_p_tile}"
    if suffix:
        name = f"{name}_{suffix}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    _EXTENSION = load(
        name=name,
        sources=[str(this_dir / "rr_diag_cuda_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "--ptxas-options=-v",
            f"-DRR_DIAG_THREADS={threads}",
            f"-DRR_DIAG_DMIMO_P_TILE={p_tile}",
            f"-DRR_DIAG_DMIMO_UNROLL={unroll}",
            f"-DRR_DIAG_DMIMO_BROADCAST_QK={broadcast_qk}",
            f"-DRR_DIAG_MONO_P_TILE={mono_p_tile}",
        ],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("RR_DIAG_CUDA_VERBOSE_BUILD", "0"))),
    )
    return _EXTENSION


def rr_specialized_cuda(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not inputs["dphi"].is_cuda:
        raise RuntimeError("CUDA extension path requires CUDA tensors")
    ext = _load_extension()
    return ext.rr_diag_forward(
        inputs["dphi"],
        inputs["psiv"],
        inputs["q_pre_rot"],
        inputs["k_pre_rot"],
        inputs["qk_dot"],
        inputs["gamma"],
    )


def rr_cuda_kernel_metadata(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not inputs["dphi"].is_cuda:
        return {}
    ext = _load_extension()
    return ext.rr_diag_kernel_metadata(inputs["dphi"])


def stage2_rr_diag_post_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dk: torch.Tensor,
    dq: torch.Tensor,
    dgamma_diag: torch.Tensor,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 CUDA post path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_post(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dk,
        dq,
        dgamma_diag,
    )


def stage2_rr_diag_post_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_post_metadata(dout)


def stage2_rr_diag_chunk_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the wave6 chunk-owner diagonal slice kernel.

    Returns ``(dgamma_diag, dk_delta, dq_delta)`` using the same production
    tensor layouts as the stage2 bwd_bwd benchmark.  Unlike the wave5 post
    kernel, this writes diagonal DK/DQ contributions directly instead of
    reloading and adding into an already-stored bwd_bwd output tensor.
    """

    if not dout.is_cuda:
        raise RuntimeError("stage2 chunk-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_rr_diag_chunk_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_rr_diag_chunk_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dgamma_diag: torch.Tensor,
    dk_delta: torch.Tensor,
    dq_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 chunk-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_chunk_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dgamma_diag,
        dk_delta,
        dq_delta,
        chunk_size,
    )


def stage2_rr_diag_chunk_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_chunk_owner_metadata(dout)


def stage2_rr_diag_chunk_warp_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the wave6 one-warp-per-timestep chunk-owner variant."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_rr_diag_chunk_warp_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_rr_diag_chunk_warp_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dgamma_diag: torch.Tensor,
    dk_delta: torch.Tensor,
    dq_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_chunk_warp_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dgamma_diag,
        dk_delta,
        dq_delta,
        chunk_size,
    )


def stage2_rr_diag_chunk_warp_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_chunk_warp_owner_metadata(dout)


def stage2_qk_dv_chunk_warp_owner_cuda(
    *,
    dout: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Run the wave7 qk_dot -> dPsiV -> dV chunk-warp consumer."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/dV chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_qk_dv_chunk_warp_owner(
        dout,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_qk_dv_chunk_warp_owner_cuda_out(
    *,
    dout: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dv_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/dV chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_qk_dv_chunk_warp_owner_out(
        dout,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dv_delta,
        chunk_size,
    )


def stage2_qk_dv_chunk_warp_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_qk_dv_chunk_warp_owner_metadata(dout)


def stage2_qk_dmimo_v_sequence_owner_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
) -> torch.Tensor:
    """Run the wave8 qk_dot -> dPsiV -> DMIMO_V sequence owner."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/DMIMO_V CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_qk_dmimo_v_sequence_owner(
        dout,
        v,
        mimo_o,
        qk_dot,
        dt,
        trap,
    )


def stage2_qk_dmimo_v_sequence_owner_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dmimo_v_delta: torch.Tensor,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/DMIMO_V CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_qk_dmimo_v_sequence_owner_out(
        dout,
        v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dmimo_v_delta,
    )


def stage2_qk_dmimo_v_sequence_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_qk_dmimo_v_sequence_owner_metadata(dout)


def stage2_qk_dmimo_v_output_owner_rvec_cuda(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
) -> torch.Tensor:
    """Run the qk_dot -> dPsiV -> DMIMO_V all-R output-owner kernel."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/DMIMO_V all-R CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_qk_dmimo_v_output_owner_rvec(
        dout,
        v,
        mimo_o,
        qk_dot,
        dt,
        trap,
    )


def stage2_qk_dmimo_v_output_owner_rvec_cuda_out(
    *,
    dout: torch.Tensor,
    v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dmimo_v_delta: torch.Tensor,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 qk/DMIMO_V all-R CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_qk_dmimo_v_output_owner_rvec_out(
        dout,
        v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dmimo_v_delta,
    )


def stage2_qk_dmimo_v_output_owner_rvec_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_qk_dmimo_v_output_owner_rvec_metadata(dout)


def stage2_rr_diag_qk_dv_chunk_warp_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the wave7 combined diagonal plus qk/dV chunk-warp prototype."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 combined chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_chunk_warp_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dgamma_diag: torch.Tensor,
    dk_delta: torch.Tensor,
    dq_delta: torch.Tensor,
    dv_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 combined chunk-warp CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_qk_dv_chunk_warp_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dgamma_diag,
        dk_delta,
        dq_delta,
        dv_delta,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_chunk_warp_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_chunk_warp_owner_metadata(dout)


def stage2_rr_diag_qk_dv_dmimo_v_sequence_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the old wave8 combined kernel with sequence-owner DMIMO_V CTAs."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 wave8 sequence-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_dmimo_v_sequence_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_dmimo_v_sequence_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dgamma_diag: torch.Tensor,
    dk_delta: torch.Tensor,
    dq_delta: torch.Tensor,
    dv_delta: torch.Tensor,
    dmimo_v_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 wave8 sequence-owner CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_qk_dv_dmimo_v_sequence_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dgamma_diag,
        dk_delta,
        dq_delta,
        dv_delta,
        dmimo_v_delta,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_dmimo_v_sequence_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_dmimo_v_sequence_owner_metadata(dout)


def stage2_rr_diag_qk_dv_dmimo_v_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the wave9 one-launch diagonal plus qk/dV plus all-R DMIMO_V prototype."""

    if not dout.is_cuda:
        raise RuntimeError("stage2 wave9 combined CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_dmimo_v_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    qk_dot: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    dgamma_diag: torch.Tensor,
    dk_delta: torch.Tensor,
    dq_delta: torch.Tensor,
    dv_delta: torch.Tensor,
    dmimo_v_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 wave9 combined CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_rr_diag_qk_dv_dmimo_v_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        qk_dot,
        dt,
        trap,
        dgamma_diag,
        dk_delta,
        dq_delta,
        dv_delta,
        dmimo_v_delta,
        chunk_size,
    )


def stage2_rr_diag_qk_dv_dmimo_v_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_rr_diag_qk_dv_dmimo_v_owner_metadata(dout)


def stage2_mono_state_lkq_d_chunk_owner_cuda(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    dstates: torch.Tensor,
    da_cs_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Wave 1 monolithic state/LKQ/D chunk-owner prototype.

    The returned tuple is ``(dv_delta, dmimo_v_chunk_delta, dssda_delta)``.
    ``dmimo_v_chunk_delta`` is intentionally per-chunk; this first prototype
    proves local intermediate reuse before choosing the final cross-chunk
    reduction owner.
    """

    if not dout.is_cuda:
        raise RuntimeError("stage2 monolithic chunk CUDA path requires CUDA tensors")
    ext = _load_extension()
    return ext.stage2_mono_state_lkq_d_chunk_owner(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        dstates,
        da_cs_rev,
        segsum,
        D,
        chunk_size,
    )


def stage2_mono_state_lkq_d_chunk_owner_cuda_out(
    *,
    dout: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    mimo_v: torch.Tensor,
    mimo_o: torch.Tensor,
    dstates: torch.Tensor,
    da_cs_rev: torch.Tensor,
    segsum: torch.Tensor,
    D: torch.Tensor,
    dv_delta: torch.Tensor,
    dmimo_v_chunk_delta: torch.Tensor,
    dssda_delta: torch.Tensor,
    chunk_size: int = 16,
) -> None:
    if not dout.is_cuda:
        raise RuntimeError("stage2 monolithic chunk CUDA path requires CUDA tensors")
    ext = _load_extension()
    ext.stage2_mono_state_lkq_d_chunk_owner_out(
        dout,
        q_flat,
        k_flat,
        v,
        q_bias,
        k_bias,
        mimo_v,
        mimo_o,
        dstates,
        da_cs_rev,
        segsum,
        D,
        dv_delta,
        dmimo_v_chunk_delta,
        dssda_delta,
        chunk_size,
    )


def stage2_mono_state_lkq_d_chunk_owner_cuda_metadata(dout: torch.Tensor) -> dict[str, Any]:
    if not dout.is_cuda:
        return {}
    ext = _load_extension()
    return ext.stage2_mono_state_lkq_d_chunk_owner_metadata(dout)
