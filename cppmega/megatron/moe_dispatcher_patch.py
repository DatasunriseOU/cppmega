"""Small runtime patches for Megatron MoE token movement.

The local GB10 lane may use the Megatron alltoall dispatcher with TP=EP=1.  In
some topologies the dispatcher asks Transformer Engine to sort local expert
chunks even though the chunk permutation is identity.  Skipping that no-op sort
removes two fused sort launches per MoE layer, but it is only valid when the
dispatcher has already expanded the token/expert splits to match the grouped
linear input.  Keep the optimization opt-in until every routed-expert path has
that contract.
"""

from __future__ import annotations

import functools
import logging
import os
import weakref

import torch

__all__ = ["apply_moe_dispatcher_identity_sort_patch", "is_identity_permutation"]

log = logging.getLogger(__name__)

_ENV_FLAG = "CPPMEGA_MOE_SKIP_IDENTITY_CHUNK_SORT"
_PATCH_MARKER = "__cppmega_identity_chunk_sort_skip__"
_IDENTITY_CACHE: dict[
    int, tuple[weakref.ReferenceType[torch.Tensor], int, bool]
] = {}


def _identity_cache_finalizer(
    current: weakref.ReferenceType[torch.Tensor], key: int
) -> None:
    """Remove ``current`` from the identity cache if it is still the holder."""
    cached = _IDENTITY_CACHE.get(key)
    if cached is not None and cached[0]() is current():
        _IDENTITY_CACHE.pop(key, None)


def is_identity_permutation(sorted_idxs: torch.Tensor) -> bool:
    """Return True when ``sorted_idxs`` is ``[0, 1, ..., n - 1]``.

    CUDA tensors require a device-to-host read for a Python branch.  The result
    is cached by tensor storage, so the GB10 dispatcher pays at most one small
    sync per static sort-index tensor and then skips per-step sort kernels.
    """

    if sorted_idxs.dim() != 1:
        return False
    # CPU checks are cheap. Caching by storage pointer is unsafe because the
    # allocator may reuse that pointer for a different tensor, which can turn a
    # non-identity permutation into a false cache hit.
    cache_key = id(sorted_idxs)
    version = int(sorted_idxs._version)
    if sorted_idxs.is_cuda:
        cached = _IDENTITY_CACHE.get(cache_key)
        if cached is not None and cached[0]() is sorted_idxs and cached[1] == version:
            return cached[2]

    if sorted_idxs.is_cuda and torch.cuda.is_current_stream_capturing():
        return False

    values = sorted_idxs.detach().cpu().tolist()
    result = values == list(range(len(values)))
    if sorted_idxs.is_cuda:
        reference = weakref.ref(
            sorted_idxs,
            functools.partial(_identity_cache_finalizer, key=cache_key),
        )
        _IDENTITY_CACHE[cache_key] = (reference, version, result)
    return result


def _split_sizes_match_input(input: torch.Tensor, split_sizes: torch.Tensor) -> bool:
    """Return True when a no-op chunk reorder can preserve the input as-is."""

    if split_sizes.dim() != 1:
        return False
    if split_sizes.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    total = int(split_sizes.detach().sum().cpu().item())
    return total == int(input.shape[0])


def apply_moe_dispatcher_identity_sort_patch(*, force: bool = False) -> bool:
    """Patch Megatron's MoE chunk sorter to skip identity permutations.

    Returns True when the patch is installed.  The patch is disabled by default;
    set ``CPPMEGA_MOE_SKIP_IDENTITY_CHUNK_SORT=1`` to enable it for A/B runs.
    """

    if os.environ.get(_ENV_FLAG, "0") != "1" and not force:
        return False

    try:
        from megatron.core.transformer.moe import moe_utils, token_dispatcher
    except Exception:
        log.warning(
            "MoE identity-sort patch not installed: Megatron MoE modules "
            "are not importable",
            exc_info=True,
        )
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
            and _split_sizes_match_input(input, split_sizes)
            and is_identity_permutation(sorted_idxs)
        ):
            return input, probs
        return original(input, split_sizes, sorted_idxs, probs=probs, fused=fused)

    setattr(_cppmega_sort_chunks_by_idxs, _PATCH_MARKER, True)
    _cppmega_sort_chunks_by_idxs.__wrapped_sort_chunks_by_idxs__ = original  # type: ignore[attr-defined]
    moe_utils.sort_chunks_by_idxs = _cppmega_sort_chunks_by_idxs
    token_dispatcher.sort_chunks_by_idxs = _cppmega_sort_chunks_by_idxs
    return True
