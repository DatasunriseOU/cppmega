"""Small runtime patches for Megatron MoE token movement.

The local GB10 lane uses the Megatron alltoall dispatcher with TP=EP=1.  In
that topology the dispatcher still asks Transformer Engine to sort local expert
chunks even though the chunk permutation is identity.  Skipping that no-op sort
removes two fused sort launches per MoE layer without changing token order.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

__all__ = ["apply_moe_dispatcher_identity_sort_patch", "is_identity_permutation"]

_ENV_FLAG = "CPPMEGA_MOE_SKIP_IDENTITY_CHUNK_SORT"
_PATCH_MARKER = "__cppmega_identity_chunk_sort_skip__"
_IDENTITY_CACHE: dict[tuple[str, int, int, Optional[int]], bool] = {}


def _tensor_cache_key(tensor: torch.Tensor) -> tuple[str, int, int, Optional[int]]:
    ptr: Optional[int]
    try:
        ptr = int(tensor.data_ptr())
    except RuntimeError:
        ptr = None
    return (str(tensor.device), int(tensor.numel()), int(tensor.dtype == torch.long), ptr)


def is_identity_permutation(sorted_idxs: torch.Tensor) -> bool:
    """Return True when ``sorted_idxs`` is ``[0, 1, ..., n - 1]``.

    CUDA tensors require a device-to-host read for a Python branch.  The result
    is cached by tensor storage, so the GB10 dispatcher pays at most one small
    sync per static sort-index tensor and then skips per-step sort kernels.
    """

    if sorted_idxs.dim() != 1:
        return False
    key = _tensor_cache_key(sorted_idxs)
    cached = _IDENTITY_CACHE.get(key)
    if cached is not None:
        return cached

    if sorted_idxs.is_cuda and torch.cuda.is_current_stream_capturing():
        return False

    values = sorted_idxs.detach().cpu().tolist()
    result = values == list(range(len(values)))
    _IDENTITY_CACHE[key] = result
    return result


def apply_moe_dispatcher_identity_sort_patch(*, force: bool = False) -> bool:
    """Patch Megatron's MoE chunk sorter to skip identity permutations.

    Returns True when the patch is installed.  The patch is enabled by default;
    set ``CPPMEGA_MOE_SKIP_IDENTITY_CHUNK_SORT=0`` to disable it for A/B runs.
    """

    if os.environ.get(_ENV_FLAG, "1") != "1" and not force:
        return False

    try:
        from megatron.core.transformer.moe import moe_utils
        from megatron.core.transformer.moe import token_dispatcher
    except Exception:
        return False

    original = getattr(moe_utils, "sort_chunks_by_idxs", None)
    if original is None:
        return False
    if getattr(original, _PATCH_MARKER, False):
        return True

    @functools.wraps(original)
    def _cppmega_sort_chunks_by_idxs(input, split_sizes, sorted_idxs, probs=None, fused=False):
        if (
            input.is_contiguous()
            and (probs is None or probs.is_contiguous())
            and is_identity_permutation(sorted_idxs)
        ):
            return input, probs
        return original(input, split_sizes, sorted_idxs, probs=probs, fused=fused)

    setattr(_cppmega_sort_chunks_by_idxs, _PATCH_MARKER, True)
    setattr(_cppmega_sort_chunks_by_idxs, "__wrapped_sort_chunks_by_idxs__", original)
    moe_utils.sort_chunks_by_idxs = _cppmega_sort_chunks_by_idxs
    token_dispatcher.sort_chunks_by_idxs = _cppmega_sort_chunks_by_idxs
    return True
