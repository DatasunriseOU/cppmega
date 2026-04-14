"""Sparse MLA ops — TileLang fused sparse attention ported from Megatron-LM PR #3674.

Provides :class:`SparseMLA` autograd.Function that wraps tilelang fused
sparse MLA forward + backward kernels with LRU kernel caching and
sequence/topk bucketing for JIT recompilation amortisation.

Source:
    NVIDIA/Megatron-LM PR #3674 (``dsa_cp_thd`` branch by HollowMan6).
    TileLang kernels originally from ``tile-ai/tilelang/examples/deepseek_v32/``.
"""

# TileLang bug workaround: on cu13.2, compiling a second TileLang kernel
# variant in the same process aborts with "libnvrtc symbols not found
# globally" because the NVRTC DSO was loaded without RTLD_GLOBAL by an
# earlier CUDA initialisation.  We force-load libnvrtc with RTLD_GLOBAL
# at cppmega-sparse-mla import time so every subsequent TileLang JIT
# (BF16 fwd, FP8 fwd, BF16 bwd, FP8 bwd, dO E5M2 bwd) resolves NVRTC
# symbols correctly.  Candidate for upstream tilelang issue.
import ctypes
import os
import sys


def _preload_nvrtc_global() -> None:
    for _name in ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"):
        try:
            ctypes.CDLL(_name, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    # Not found: fine on CPU-only systems; TileLang would fail loudly
    # later if it actually needs it.  Print a one-line note to stderr
    # so the condition is visible but not fatal.
    if os.environ.get("CPPMEGA_VERBOSE", "0") == "1":
        print("[cppmega] note: libnvrtc not preloaded (non-CUDA env?)", file=sys.stderr)


_preload_nvrtc_global()
del _preload_nvrtc_global
del ctypes, os, sys

from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA  # noqa: E402

__all__ = ["SparseMLA"]
