"""CuTe DSL Mamba-3 MIMO autograd.Function for Hopper (sm_90a).

Drop-in replacement for TileLang's mamba3_mimo() in the NAM56R pipeline.

Architecture:
  - forward(): CuTe DSL fwd kernel (PyTorch reference in Phase 1,
    fused WGMMA kernel in Phase 2 / task #72)
  - backward(): delegates to TileLang's mamba_mimo_bwd_combined()
    (the bwd kernels are production-verified and task #72 handles the
    CuTe DSL bwd port)

Interface matches TileLang's _Mamba3Function exactly:
  - Same tensor names, shapes, dtypes, and ordering
  - Same return semantics (return_state toggle)
  - Same gradient return ordering
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
    compute_dacs_segsum_triton,
    bwd_dadt_fused_triton,
    bwd_dtrap_ddt_triton,
)
from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd import mamba_mimo_bwd_combined

from cppmega.megatron.cute_dsl_mimo.cute_dsl_mimo_fwd import cute_dsl_mimo_fwd


class CuTeDSLMamba3MIMOFunction(torch.autograd.Function):
    """Custom autograd function for Mamba-3 MIMO using CuTe DSL fwd kernel."""

    @staticmethod
    def forward(
        ctx,
        Q: Tensor,          # [B, S, R, G, N]
        K: Tensor,          # [B, S, R, G, N]
        V: Tensor,          # [B, S, H, P]
        ADT: Tensor,        # [B, H, S]
        DT: Tensor,         # [B, H, S]
        Trap: Tensor,       # [B, H, S]
        Q_bias: Tensor,     # [H, R, N]
        K_bias: Tensor,     # [H, R, N]
        MIMO_V: Tensor,     # [H, R, P]
        MIMO_Z: Tensor,     # [H, R, P]
        MIMO_Out: Union[Tensor, None],  # [H, R, P] or None
        Angles: Tensor,     # [B, S, H, N//rotary_dim_divisor]
        D: Tensor,          # [H]
        Z: Tensor,          # [B, S, H, P]
        chunk_size: int,
        rotary_dim_divisor: int,
        dtype: torch.dtype,
        return_state: bool,
        cu_seqlens: Optional[Tensor],
    ) -> Tensor | Tuple[Tensor, ...]:
        """Forward pass using CuTe DSL fwd kernel."""

        if cu_seqlens is not None:
            raise NotImplementedError(
                "CuTe DSL MIMO does not support variable-length sequences (cu_seqlens). "
                "Use TileLang for varlen workloads."
            )

        ctx.chunk_size = chunk_size
        ctx.rotary_dim_divisor = rotary_dim_divisor
        ctx.dtype = dtype

        # Make contiguous
        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out,
         Angles, D, Z) = tuple(
            t.contiguous() if t is not None else None
            for t in (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
                      MIMO_Out, Angles, D, Z)
        )

        # Compute discretization tensors (shared Triton kernels)
        DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(ADT, chunk_size)

        # --- CuTe DSL Forward Kernel ---
        Out, Final_SSM_State, Final_K = cute_dsl_mimo_fwd(
            Q, K, V, Q_bias, K_bias, MIMO_V, MIMO_Out,
            Z, D, MIMO_Z, Angles,
            DA_CS, DA_CS_REV, DT, Trap, Segsum,
            chunk_size=chunk_size,
            rotary_dim_divisor=rotary_dim_divisor,
            dtype=dtype,
            return_state=return_state,
        )

        # Save for backward (same tensors as TileLang's _Mamba3Function)
        ctx.save_for_backward(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
            D, Z,
            MIMO_V, MIMO_Out, MIMO_Z,
            # cu_seqlens is None, save a dummy
            torch.empty(0, device=Q.device),  # placeholder for cu_seqlens
        )

        if not return_state:
            return Out
        else:
            Final_Angle = torch.remainder(
                Angles[:, -1, :, :], 2 * math.pi
            ).contiguous().detach()
            Final_SSM_State = Final_SSM_State.permute(0, 1, 3, 2).contiguous().detach()
            Final_K = Final_K.contiguous().detach()
            Final_V = V[:, -1, :, :].contiguous().detach()
            ctx.mark_non_differentiable(
                Final_Angle, Final_SSM_State, Final_K, Final_V
            )
            return Out, Final_Angle, Final_SSM_State, Final_K, Final_V

    @staticmethod
    def backward(ctx, dout, *args) -> tuple:
        """Backward pass: delegates to TileLang bwd kernels.

        The backward kernels are production-verified TileLang code.
        CuTe DSL bwd port is task #72.
        """
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking."
            )
        dout = dout.contiguous()

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
         D, Z,
         MIMO_V, MIMO_Out, MIMO_Z,
         _cu_seqlens_placeholder,
         ) = ctx.saved_tensors

        # Recompute discretization (same as forward)
        DA_CS, DA_CS_REV, Segsum = compute_dacs_segsum_triton(
            ADT, ctx.chunk_size
        )

        # Delegate to TileLang backward
        (dQ, dK, dV,
         dADT, dDT, dTrap, dQ_bias, dK_bias,
         dMIMO_V, dMIMO_Z, dMIMO_Out, dAngles,
         dD, dZ) = mamba_mimo_bwd_combined(
            dout,
            Q, K, V,
            Q_bias, K_bias,
            MIMO_V, MIMO_Out,
            Z, MIMO_Z, Angles,
            DA_CS, DA_CS_REV, DT, Trap,
            D, Segsum,
            ctx.chunk_size,
            ctx.rotary_dim_divisor,
            ctx.dtype,
        )

        return (
            dQ, dK, dV,
            dADT, dDT, dTrap,
            dQ_bias, dK_bias,
            dMIMO_V, dMIMO_Z, dMIMO_Out,
            dAngles,
            dD, dZ,
            None, None, None, None, None,  # chunk_size, rotary_dim_divisor, dtype, return_state, cu_seqlens
        )


def cute_dsl_mimo_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    MIMO_V: Tensor,
    MIMO_Z: Tensor,
    MIMO_Out: Tensor,
    Angles: Tensor,
    D: Tensor,
    Z: Tensor,
    chunk_size: int,
    rotary_dim_divisor: int,
    dtype: torch.dtype,
    return_state: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, ...]:
    """Drop-in replacement for TileLang's mamba3_mimo().

    Uses CuTe DSL forward kernel + TileLang backward kernels.

    Args: same as mamba_ssm.ops.tilelang.mamba3.mamba3_mimo.mamba3_mimo
    Returns: same as mamba_ssm.ops.tilelang.mamba3.mamba3_mimo.mamba3_mimo
    """
    batch, seqlen, mimo_rank, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape

    assert chunk_size >= 8, f"chunk_size must be at least 8, got {chunk_size}"
    assert nheads % nheads_qk == 0, (
        f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    )
    assert headdim_qk % 2 == 0, (
        f"headdim_qk ({headdim_qk}) must be even for rotary embeddings"
    )
    assert rotary_dim_divisor in [2, 4], (
        f"rotary_dim_divisor must be 2 or 4, got {rotary_dim_divisor}"
    )

    return CuTeDSLMamba3MIMOFunction.apply(
        Q, K, V, ADT, DT, Trap,
        Q_bias, K_bias,
        MIMO_V, MIMO_Z, MIMO_Out,
        Angles, D, Z,
        chunk_size, rotary_dim_divisor, dtype, return_state, cu_seqlens,
    )
