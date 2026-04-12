"""Sparse MLA ops — TileLang fused sparse attention ported from Megatron-LM PR #3674.

Provides :class:`SparseMLA` autograd.Function that wraps tilelang fused
sparse MLA forward + backward kernels with LRU kernel caching and
sequence/topk bucketing for JIT recompilation amortisation.

Source:
    NVIDIA/Megatron-LM PR #3674 (``dsa_cp_thd`` branch by HollowMan6).
    TileLang kernels originally from ``tile-ai/tilelang/examples/deepseek_v32/``.
"""

from cppmega.megatron.sparse_mla_ops.sparse_mla import SparseMLA

__all__ = ["SparseMLA"]
