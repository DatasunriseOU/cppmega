"""Regional torch.compile for Mamba3 elementwise ops.

Wraps the pre-scan and post-scan elementwise operations in Mamba3.forward()
with torch.compile(mode="default") to fuse the ~89 small kernels (SiLU,
softplus, exp, clamp, rearrange, RMSNorm) into ~4-8 fused Triton kernels.

The scan kernel itself (TileLang mamba3_mimo_combined) causes a graph break,
which is expected and correct — the scan is already optimized via TileLang
WGMMA. torch.compile fuses everything AROUND the scan.

Gate: CPPMEGA_MAMBA3_COMPILE=1

Compatibility:
  - mode="default": safe with TE CUDA graphs (no internal CG conflict)
  - mode="reduce-overhead": CONFLICTS with --cuda-graph-impl transformer_engine
  - TileLang JIT kernels: graph break (expected), surrounding ops fuse normally

Code references:
  - Pre-scan ops: mamba_ssm/modules/mamba3.py:170-179 (softplus, clamp, rearrange)
  - Post-scan ops: mamba_ssm/modules/mamba3.py:210-230 (gating, RMSNorm, out_proj)
  - Scan: mamba3_mimo_combined() → graph break → two fused regions

Expected impact: ~89 kernels → ~4-8 fused. 14.7% of time → ~10%. Net ~5% speedup.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)


def apply_mamba3_compile_patch() -> bool:
    """Wrap Mamba3.forward with torch.compile(mode='default').

    Returns True if patch applied, False if skipped.
    """
    if os.environ.get("CPPMEGA_MAMBA3_COMPILE", "0") != "1":
        return False

    try:
        import torch
        from mamba_ssm.modules.mamba3 import Mamba3
    except ImportError as e:
        log.warning("mamba3_compile_patch: cannot import Mamba3: %s", e)
        return False

    if getattr(Mamba3, "_cppmega_compiled", False):
        log.info("mamba3_compile_patch: already applied")
        return True

    _orig_forward = Mamba3.forward

    # Compile the forward method with mode="default" (Inductor fusion only,
    # no internal CUDA graphs — safe with TE CG impl).
    # dynamic=False because shapes are static (fixed B, S, H, P per MBS).
    _compiled_forward = torch.compile(
        _orig_forward,
        mode="default",
        dynamic=False,
        fullgraph=False,  # allow graph breaks at TileLang scan
    )

    def _forward_wrapper(self, *args, **kwargs):
        return _compiled_forward(self, *args, **kwargs)

    Mamba3.forward = _forward_wrapper
    Mamba3._cppmega_compiled = True

    log.info(
        "mamba3_compile_patch: Mamba3.forward wrapped with "
        "torch.compile(mode='default', dynamic=False)"
    )
    print("[cppmega] Mamba3 regional compile installed (mode=default, dynamic=False)")
    return True
