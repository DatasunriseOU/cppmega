"""Mamba3 MIMO P2 PsiV cache — skeleton (not active).

Design: `docs/mamba3_mimo_p2_psiv_cache_design.md`.
Plan: `plan.md` §50-53 (+1.5-2.3% TFLOP/s expected on bench3).

Status: **SCAFFOLDING ONLY.** Env gate `CPPMEGA_MAMBA3_P2_PSIV_CACHE=1` is
recognised, but if set the module raises `NotImplementedError`. This lets
integration shims import the module unconditionally while making it
impossible to silently activate a half-built cache. No fallback, no
workarounds — per `feedback_no_silent_fallbacks.md`.

Why this module exists as a skeleton:
  Future sessions will fill in:
    1. `precompute_psi_v` — Phase-A Python-level `v * psi` materialisation,
       used to measure the ceiling gain before any TileLang kernel edit.
    2. `psi_v_gmem_pool` — optional pre-allocated activation pool keyed by
       `(B, S, H, R, P, dtype)` to avoid `torch.empty()` inside CUDA graphs.
    3. Integration glue into `cppmega_tilelang_mimo_combined`: a new kwarg
       on the autograd Function that accepts/returns the PsiV cache tensor,
       plumbed through `save_for_backward`.

PsiV dependency recap (for readers unfamiliar):
  PsiV has shape (B, S, H, R, P) and is defined as
      psi_v[b, s, h, r, p] = V[b, s, h, p] * MIMO_V[h, r, p]
  MIMO_V is a learned parameter (static within a step); V is a derived
  activation (changes every forward). ⇒ cache is INTRA-STEP, not inter-step.
  See §2 of the design doc for the full analysis.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------

_ENV_FLAG = "CPPMEGA_MAMBA3_P2_PSIV_CACHE"


def is_enabled() -> bool:
    """True iff the env gate asks for the P2 cache.

    Design guarantees: if True is returned, every public function in this
    module MUST either implement the real path or raise NotImplementedError.
    Never silently return a non-cached path when the user asked for caching.
    """
    return os.environ.get(_ENV_FLAG, "0") in ("1", "true", "True")


def _refuse_if_gated() -> None:
    """Crash loudly if the gate is on but the implementation is absent.

    Rationale (`feedback_no_silent_fallbacks.md`): the user opts in explicitly
    via CPPMEGA_MAMBA3_P2_PSIV_CACHE=1. Silently falling back to the cold
    kernel path would give them a wrong perf number and obscure the fact
    that the cache is a stub. Better to crash.
    """
    if is_enabled():
        raise NotImplementedError(
            "CPPMEGA_MAMBA3_P2_PSIV_CACHE=1 but the cache is a scaffold only. "
            "See docs/mamba3_mimo_p2_psiv_cache_design.md §12 for status. "
            "Unset the variable to run the baseline kernel path."
        )


# ---------------------------------------------------------------------------
# Phase A: Python-level precompute — **TODO**
# ---------------------------------------------------------------------------

def precompute_psi_v(
    V: torch.Tensor,
    mimo_v: torch.Tensor,
) -> torch.Tensor:
    """Compute PsiV = V * mimo_v (broadcast over R) at the Python level.

    Shapes:
        V:       (B, S, H, P)
        mimo_v:  (H, R, P)
    Returns:
        psi_v:   (B, S, H, R, P), dtype == V.dtype

    This is the Phase-A prototype: materialise `psi_v` BEFORE calling the
    Mamba3 MIMO kernel, so we can measure the upper bound of the P2 win
    without touching TileLang. If this does not produce a measurable nsys
    delta vs baseline, Phase B/C are dead and P2 is archived.

    TODO(Phase A): implement + call from `cppmega_tilelang_mimo_combined`
    behind the env gate. Do NOT activate in production yet.
    """
    _refuse_if_gated()
    raise NotImplementedError(
        "Phase-A Python precompute_psi_v not implemented — see design doc §9."
    )


# ---------------------------------------------------------------------------
# Phase B/C: gmem pool for PsiV cache — **TODO**
# ---------------------------------------------------------------------------

class PsiVCachePool:
    """Per-shape gmem pool for the PsiV activation tensor.

    Needed under CUDA graphs: `torch.empty()` inside the captured region
    allocates in the graph's private memory pool, which is fine for autograd
    saved-tensors. But if the pool is pre-warmed OUTSIDE the capture,
    allocation cost is hidden from the critical path.

    Key: (B, S, H, R, P, dtype, device).
    Value: a `torch.Tensor` we hand out on `acquire` and take back on
    `release`. Reference-counted so double-release is caught.

    TODO(Phase B): implement. Exactly one tensor per shape key should be
    sufficient for intra-step caching since fwd→bwd_fwd→bwd_bwd chain is
    serial per-op.

    TODO(Phase C): verify this works under CUDA graph capture. The tricky
    case is when `acquire` is called inside a captured region but the pool
    was warmed outside — tensor addresses must stay stable. Consult
    `feedback_mandatory_patches.md` for CG flag coverage.
    """

    def __init__(self) -> None:
        self._pool: dict[tuple, torch.Tensor] = {}
        _refuse_if_gated()

    def acquire(
        self,
        batch: int,
        seqlen: int,
        nheads: int,
        rank: int,
        headdim_v: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get a (B, S, H, R, P) tensor matching the shape/dtype/device.

        TODO(Phase B): implement. First cut can be a plain `torch.empty()`
        allocation; optimise to pool on second iteration once perf numbers
        prove the Phase-A ceiling is worth chasing.
        """
        _refuse_if_gated()
        raise NotImplementedError("PsiVCachePool.acquire not implemented.")

    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool. No-op for now.

        TODO(Phase B): implement once we have a real pool. In Phase A
        (no pool) this is trivially a no-op.
        """
        _refuse_if_gated()
        raise NotImplementedError("PsiVCachePool.release not implemented.")


# ---------------------------------------------------------------------------
# Phase B/C: autograd integration — **TODO**
# ---------------------------------------------------------------------------

def forward_with_cache(
    fwd_kernel_callable,
    *args,
    psi_v_out: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Call the Mamba3 MIMO fwd kernel with an extra PsiV-out tensor.

    Phase B: the forward TileLang kernel gets a new output argument that
    materialises PsiV to gmem. This wrapper threads that argument through
    and returns (y, psi_v_cache) where psi_v_cache must be saved via
    ctx.save_for_backward(...).

    TODO(Phase B): implement once the kernel is patched. Signature may
    need adjustment depending on how the upstream kernel is extended.
    """
    _refuse_if_gated()
    raise NotImplementedError(
        "forward_with_cache not implemented — waiting on Phase B kernel edits "
        "per docs/mamba3_mimo_p2_psiv_cache_design.md §9."
    )


def backward_fwd_with_cache(
    bwd_fwd_kernel_callable,
    *args,
    psi_v_in: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Call `mamba_mimo_bwd_fwd` with precomputed PsiV input.

    Phase C: skip the `psi_v = v * psi` recompute inside bwd_fwd.

    TODO(Phase C): implement. Depends on bwd_fwd kernel signature
    extension (separate patch file — see apply_mamba3_p2_psiv_patches.py).
    """
    _refuse_if_gated()
    raise NotImplementedError("backward_fwd_with_cache not implemented.")


def backward_bwd_with_cache(
    bwd_bwd_kernel_callable,
    *args,
    psi_v_in: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Call `mamba_mimo_bwd_bwd` with precomputed PsiV input.

    Phase C: drop ~3 fragment tiles from the bwd_bwd inner live set by
    loading PsiV instead of rematerialising it. Primary win location per
    P3 design doc line 185-189 "Hoist-PsiV alternative".

    TODO(Phase C): implement.
    """
    _refuse_if_gated()
    raise NotImplementedError("backward_bwd_with_cache not implemented.")


# ---------------------------------------------------------------------------
# Memory budget helper (used by integration tests to assert headroom)
# ---------------------------------------------------------------------------

def estimate_cache_bytes(
    batch: int,
    seqlen: int,
    nheads: int,
    rank: int,
    headdim_v: int,
    dtype: torch.dtype,
    num_layers: int = 1,
) -> int:
    """Return bytes needed to cache PsiV for `num_layers` Mamba3 mixers.

    Used by `test_mamba3_psiv_cache.py::test_memory_budget_within_headroom`
    to assert that we stay under the remaining GPU budget at NAM56R MBS=8.

    Does not raise when gate is off — this is a pure arithmetic helper.
    """
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    per_sample = batch * seqlen * nheads * rank * headdim_v * dtype_bytes
    return per_sample * num_layers


__all__ = [
    "is_enabled",
    "precompute_psi_v",
    "PsiVCachePool",
    "forward_with_cache",
    "backward_fwd_with_cache",
    "backward_bwd_with_cache",
    "estimate_cache_bytes",
]
