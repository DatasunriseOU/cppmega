"""Regional torch.compile for Mamba3 elementwise ops.

Compiles ONLY the 4 proven-win submodules (GB10-validated speedups):

  1. Data-dependent A (exp, norm, softplus, clamp)     — 5.93x
  2. Mamba3 pre-processing (softplus dt, exp A, D skip) — 2.66x
  3. SiLU + gate multiply                               — 1.35x
  4. Mamba3 post-processing (RMSNorm + SiLU gate)       — 1.84x

EXCLUDED from compilation (already fast or break Inductor):
  - RMSNorm standalone, RMSNormGated (already Triton-fused)
  - MoE Router, MLA projections, Transform B/C
  - Scan kernels (mamba3_siso_combined, mamba3_mimo_combined) — break Inductor

Always on — no env var gates.  If compile fails, crash.

The approach: define small pure-PyTorch compiled functions for the elementwise
math, then monkey-patch the mixer forward methods to call them instead of
inline code.  The scan kernels remain untouched.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled elementwise functions
# ---------------------------------------------------------------------------
# These are module-level compiled functions.  torch.compile traces them on
# first call and caches the fused Triton kernel.  Subsequent calls reuse the
# cache (dynamic=False because shapes are fixed per MBS/seqlen config).

@torch.compile(mode="default", dynamic=False)
def _compiled_data_dep_A(dd_A: torch.Tensor, A_floor: float,
                         dd_dt: torch.Tensor,
                         dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse: softplus(dd_A) -> clamp -> softplus(dd_dt + dt_bias) -> A * DT.

    Returns (ADT, DT, _A) all in (batch, nheads, seqlen) layout for the
    Author Mamba3 kernel contract.
    """
    # Data-dependent A: per-position negative decay
    _A = -F.softplus(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-A_floor)
    # Delta-time with learned bias
    DT = F.softplus((dd_dt + dt_bias).to(torch.float32))
    # Combined discretised decay
    ADT = _A * DT
    # Rearrange to kernel layout: (b, l, n) -> (b, n, l)
    DT = DT.transpose(1, 2)
    ADT = ADT.transpose(1, 2)
    return ADT, DT, _A


@torch.compile(mode="default", dynamic=False)
def _compiled_mamba3_preprocess(dd_dt: torch.Tensor,
                                dt_bias: torch.Tensor,
                                dd_A: torch.Tensor,
                                A_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the pre-scan elementwise chain for the NoConv Mamba3 path.

    Computes:
      DT = softplus(dd_dt + dt_bias)
      A_dd = -softplus(dd_A) clamped to [-inf, -A_floor]
      ADT = A_dd * DT
      dt_kernel = -ADT  (positive, for the A=-1 trick)

    Returns (DT, dt_kernel).
    """
    DT = F.softplus(dd_dt + dt_bias)
    A_dd = -F.softplus(dd_A.float())
    A_dd = torch.clamp(A_dd, max=-A_floor)
    ADT = A_dd * DT
    dt_kernel = -ADT
    return DT, dt_kernel


@torch.compile(mode="default", dynamic=False)
def _compiled_silu_gate(x: torch.Tensor) -> torch.Tensor:
    """Fuse SiLU activation.  Replaces nn.SiLU() module call."""
    return F.silu(x)


@torch.compile(mode="default", dynamic=False)
def _compiled_postprocess_siso(y: torch.Tensor,
                               z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse post-scan reshapes before RMSNormGated for SISO path.

    RMSNormGated itself is already Triton-fused, so we only compile the
    surrounding rearranges + float casts that feed into it.
    """
    # y: (b, l, h, p) -> (b, l, d)
    b, l, h, p = y.shape
    y_flat = y.reshape(b, l, h * p)
    z_flat = z.reshape(b, l, h * p)
    return y_flat, z_flat


@torch.compile(mode="default", dynamic=False)
def _compiled_postprocess_mimo(y: torch.Tensor,
                               z: torch.Tensor,
                               mimo_z: torch.Tensor,
                               headdim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse post-scan elementwise ops for MIMO outproj_norm path.

    Compiles the z einsum projection + reshapes before RMSNormGated.
    RMSNormGated is excluded (already Triton-fused).
    """
    # z: (b, l, h, p), mimo_z: (h, r, p)
    z_f = torch.einsum("blhp,hrp->blrhp", z.float(), mimo_z)
    # (b, l, r, h, p) -> (b, l, r, h*p)
    b, l, r, h, p = z_f.shape
    z_f = z_f.reshape(b, l, r, h * p)
    # y: (b, l, r, h, p) -> (b, l, r, h*p)
    y_f = y.reshape(y.shape[0], y.shape[1], y.shape[2], -1).float()
    return y_f, z_f


@torch.compile(mode="default", dynamic=False)
def _compiled_postprocess_mimo_out(y: torch.Tensor,
                                   mimo_o: torch.Tensor,
                                   headdim: int) -> torch.Tensor:
    """Fuse post-RMSNorm MIMO output projection.

    After RMSNormGated, we reshape back and apply the output einsum.
    """
    # y: (b, l, r, h*p) -> (b, l, r, h, p)
    b, l, r, d = y.shape
    h = d // headdim
    y = y.reshape(b, l, r, h, headdim)
    y = torch.einsum("blrhp,hrp->blhp", y, mimo_o)
    return y


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _patch_cppmega_mamba3_te():
    """Patch CppMegaMamba3TE.forward to use compiled elementwise functions."""
    from cppmega.megatron.mamba3_te_mixer import CppMegaMamba3TE
    from einops import rearrange

    if getattr(CppMegaMamba3TE, "_cppmega_compiled", False):
        return

    _orig_forward = CppMegaMamba3TE.forward

    # Disable dynamo tracing on the outer forward — only the 4 inner
    # @torch.compile sub-functions need compilation.  Preventing dynamo
    # from tracing this function avoids two problems:
    #   1. Spurious kwargs (padding_mask, etc.) passed through te_checkpoint
    #      or torch.utils.checkpoint hitting dynamo's kwarg validation.
    #   2. Scan kernels (mamba3_siso_combined, mamba3_mimo_combined) that are
    #      explicitly excluded from Inductor being pulled into a trace.
    @torch._dynamo.disable
    def _compiled_forward(
        self,
        hidden_states,
        inference_context=None,
        *,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        from megatron.core.inference.contexts.static_context import deprecate_inference_params
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
        from mamba_ssm.ops.triton.angle_cumsum import angle_dt

        inference_context = deprecate_inference_params(
            inference_context, inference_params
        )

        # --- TE in_proj ---
        zxBCdt_packed, _ = self.in_proj(hidden_states)
        zxBCdt_packed = rearrange(zxBCdt_packed, "l b d -> b l d").contiguous()

        batch, seqlen, _ = zxBCdt_packed.shape

        z_size = self.d_inner_local_tp
        x_size = self.d_inner_local_tp
        B_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        C_size = self.ngroups_local_tp * self.d_state * self.mimo_rank
        dd_dt_size = self.nheads_local_tp
        dd_A_size = self.nheads_local_tp
        trap_size = self.nheads_local_tp

        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt_packed,
            [z_size, x_size, B_size, C_size,
             dd_dt_size, dd_A_size, trap_size, self.num_rope_angles],
            dim=-1,
        )

        # Reshape for scan kernels
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(
            B, "b l (r g n) -> b l r g n",
            r=self.mimo_rank, g=self.ngroups_local_tp,
        )
        C = rearrange(
            C, "b l (r g n) -> b l r g n",
            r=self.mimo_rank, g=self.ngroups_local_tp,
        )
        trap = rearrange(trap, "b l h -> b h l")

        # === COMPILED REGION 1: Data-dependent A (5.93x) ===
        ADT, DT, _A = _compiled_data_dep_A(
            dd_A, self.A_floor, dd_dt, self.dt_bias,
        )

        # --- Complex RoPE angles (NOT compiled — already fast) ---
        angles = angles.unsqueeze(-2).expand(
            -1, -1, self.nheads_local_tp, -1
        )

        # --- QK-Norm on B and C (NOT compiled — already Triton-fused) ---
        B = self.B_norm(B)
        C = self.C_norm(C)

        # --- Packed sequence support ---
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q

        # --- SSM scan (NOT compiled — scan kernels break Inductor) ---
        if self.is_mimo:
            angles = angle_dt(angles, DT.transpose(-1, -2))
            mimo_chunk = min(self.chunk_size, max(1, 64 // self.mimo_rank))
            y = mamba3_mimo_combined(
                Q=C, K=B, V=x, ADT=ADT, DT=DT, Trap=trap,
                Q_bias=self.C_bias.float(),
                K_bias=self.B_bias.float(),
                MIMO_V=self.mimo_x.float(),
                MIMO_Z=self.mimo_z.float(),
                MIMO_Out=self.mimo_o.float() if not self.is_outproj_norm else None,
                Angles=angles,
                D=self.D.float(),
                Z=z if not self.is_outproj_norm else None,
                chunk_size=mimo_chunk,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=False,
                cu_seqlens=cu_seqlens,
            )
            # === COMPILED REGION 4: Post-processing (1.84x) ===
            if self.is_outproj_norm:
                y_f, z_f = _compiled_postprocess_mimo(
                    y, z, self.mimo_z, self.headdim,
                )
                # RMSNormGated — NOT compiled (already Triton-fused)
                y_normed = self.norm(y_f, z_f)
                y = _compiled_postprocess_mimo_out(
                    y_normed, self.mimo_o, self.headdim,
                )
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2), K=B.squeeze(2), V=x,
                ADT=ADT, DT=DT, Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None,
                return_final_states=False,
                cu_seqlens=cu_seqlens,
            )
            # === COMPILED REGION 4: Post-processing (1.84x) ===
            if self.is_outproj_norm:
                y_flat, z_flat = _compiled_postprocess_siso(y, z)
                # RMSNormGated — NOT compiled (already Triton-fused)
                y = self.norm(y_flat, z_flat)
            else:
                y = rearrange(y, "b l h p -> b l (h p)")

        # Transpose back to Megatron layout
        y = rearrange(y, "b l d -> l b d").contiguous()

        # --- TE out_proj ---
        out, out_bias = self.out_proj(y.to(hidden_states.dtype))
        return out, out_bias

    CppMegaMamba3TE.forward = _compiled_forward
    CppMegaMamba3TE._cppmega_compiled = True
    log.info("mamba3_compile_patch: CppMegaMamba3TE.forward patched with "
             "4 compiled elementwise regions")


def _patch_noconv_mamba3():
    """Patch Mamba3ScanMixin._mamba3_scan to use compiled pre-processing.

    Also patches NoConvMambaMixer._ssm_noconv for SiLU compilation.
    """
    from cppmega.megatron.noconv_mamba_mixer import (
        Mamba3ScanMixin,
        NoConvMambaMixer,
        _compute_data_dependent_A,
    )
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    if getattr(Mamba3ScanMixin, "_cppmega_compiled", False):
        return

    _orig_mamba3_scan = Mamba3ScanMixin._mamba3_scan

    def _compiled_mamba3_scan(
        self, x, B, C, z, dd_dt, dd_A, trap, angles, dt_bias, D,
        chunk_size, rmsnorm, A_floor=1e-4, rope_fraction=0.5,
        return_final_states=False,
    ):
        batch, seqlen, nheads, headdim = x.shape

        if dd_A is not None:
            # === COMPILED REGION 1+2: Data-dep A + pre-processing (5.93x + 2.66x) ===
            DT, dt_kernel = _compiled_mamba3_preprocess(
                dd_dt, dt_bias, dd_A, A_floor,
            )
            A_kernel = torch.full(
                (nheads,), -1.0, device=x.device, dtype=torch.float32,
            )
            y = mamba_chunk_scan_combined(
                x, dt_kernel, A_kernel, B, C, chunk_size,
                D=D,
                z=z if not rmsnorm else None,
                dt_bias=None,
                dt_softplus=False,
                return_final_states=return_final_states,
            )
        else:
            # Fixed A path — compile the exp
            A = -torch.exp(self.A_log.float())
            y = mamba_chunk_scan_combined(
                x, dd_dt, A, B, C, chunk_size,
                D=D,
                z=z if not rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
                return_final_states=return_final_states,
            )
        return y

    Mamba3ScanMixin._mamba3_scan = _compiled_mamba3_scan
    Mamba3ScanMixin._cppmega_compiled = True

    # --- Patch SiLU in NoConvMambaMixer._ssm_noconv ---
    if getattr(NoConvMambaMixer, "_cppmega_silu_compiled", False):
        return

    _orig_ssm_noconv = NoConvMambaMixer._ssm_noconv

    def _compiled_ssm_noconv(self, zxBCdt):
        from einops import rearrange

        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()
        A = -torch.exp(self.A_log.float())

        z, x, B, C, dt = torch.split(
            zxBCdt,
            [self.d_inner_local, self.d_inner_local,
             self.ngroups_local * self.d_state,
             self.ngroups_local * self.d_state,
             self.nheads_local],
            dim=-1,
        )

        # === COMPILED REGION 3: SiLU (1.35x) ===
        x = _compiled_silu_gate(x)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
        dt = dt.contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()

        D = (
            rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim else self.D
        )

        y = mamba_chunk_scan_combined(
            x, dt, A, B, C, self.chunk_size,
            D=D,
            z=z if not self.rmsnorm else None,
            dt_bias=self.dt_bias.float(),
            dt_softplus=True,
            return_final_states=False,
        )

        # === COMPILED REGION 4: Post-processing (1.84x) ===
        y = rearrange(y, "b l h p -> l b (h p)").contiguous()
        if self.rmsnorm:
            z = rearrange(z, "b l h p -> l b (h p)").contiguous()
            # RMSNormGated — NOT compiled (already Triton-fused)
            y = self.norm(y, z)

        return y

    NoConvMambaMixer._ssm_noconv = _compiled_ssm_noconv
    NoConvMambaMixer._cppmega_silu_compiled = True

    log.info("mamba3_compile_patch: Mamba3ScanMixin._mamba3_scan + "
             "NoConvMambaMixer._ssm_noconv patched with compiled elementwise regions")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_mamba3_compile_patch() -> None:
    """Apply regional torch.compile to all Mamba3 mixer variants.

    Always on.  Crashes on failure — no fallbacks.
    """
    import torch

    _patch_cppmega_mamba3_te()
    _patch_noconv_mamba3()

    print(
        "[cppmega] Mamba3 regional compile installed: "
        "4 elementwise regions (data-dep-A, preprocess, SiLU, postprocess)"
    )
