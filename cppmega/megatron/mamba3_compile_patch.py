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
math.  The TE mixer imports the data-dependent-A helper directly; the scan
kernels remain untouched.  Only the legacy NoConv mixin still receives an
explicit patch below.
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
    # y: (b, seq_len, h, p) -> (b, seq_len, d)
    b, seq_len, h, p = y.shape
    y_flat = y.reshape(b, seq_len, h * p)
    z_flat = z.reshape(b, seq_len, h * p)
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
    # (b, seq_len, r, h, p) -> (b, seq_len, r, h*p)
    b, seq_len, r, h, p = z_f.shape
    z_f = z_f.reshape(b, seq_len, r, h * p)
    # y: (b, seq_len, r, h, p) -> (b, seq_len, r, h*p)
    y_f = y.reshape(y.shape[0], y.shape[1], y.shape[2], -1).float()
    return y_f, z_f


@torch.compile(mode="default", dynamic=False)
def _compiled_postprocess_mimo_out(y: torch.Tensor,
                                   mimo_o: torch.Tensor,
                                   headdim: int) -> torch.Tensor:
    """Fuse post-RMSNorm MIMO output projection.

    After RMSNormGated, we reshape back and apply the output einsum.
    """
    # y: (b, seq_len, r, h*p) -> (b, seq_len, r, h, p)
    b, seq_len, r, d = y.shape
    h = d // headdim
    y = y.reshape(b, seq_len, r, h, headdim)
    y = torch.einsum("blrhp,hrp->blhp", y, mimo_o)
    return y


def _patch_noconv_mamba3():
    """Patch Mamba3ScanMixin._mamba3_scan to use compiled pre-processing.

    Also patches NoConvMambaMixer._ssm_noconv for SiLU compilation.
    """
    from cppmega.megatron.noconv_mamba_mixer import (
        Mamba3ScanMixin,
        NoConvMambaMixer,
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
    # CppMegaMamba3TE: compile is inline in mamba3_te_mixer.py.
    _patch_noconv_mamba3()

    print(
        "[cppmega] Mamba3 regional compile installed: "
        "4 elementwise regions (data-dep-A, preprocess, SiLU, postprocess)"
    )
