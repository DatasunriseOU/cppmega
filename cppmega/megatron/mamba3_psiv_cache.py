"""Mamba3 MIMO P2/P3 PsiV cache helpers.

Design: `docs/mamba3_mimo_p2_psiv_cache_design.md`.
Plan: `plan.md` §50-53 (+1.5-2.3% TFLOP/s expected on bench3).

Status: **MEASUREMENT SCAFFOLDING ONLY.** The pure PyTorch PsiV materializer,
memory estimators, and tiny benchmark helper are implemented so we can size
and time the proposal without editing TileLang. The actual fwd/bwd TileLang
integration remains unimplemented and still raises `NotImplementedError`
when called.

What is implemented here:
  1. `precompute_psi_v` — Phase-A Python-level `v * psi` materialisation,
     used to measure allocation and broadcast-multiply cost before TileLang
     kernel edits.
  2. `PsiVCachePool` — optional pre-allocated activation pool keyed by
     `(B, S, H, R, P, dtype, device)` for later CUDA graph lifetime probes.
  3. Memory budget helpers for P2 PsiV and P3 `dstates_per_chunk` estimates.

What is still absent:
  TileLang kernel signatures and autograd integration into
  `cppmega_tilelang_mimo_combined`. Those wrappers still raise loudly.

PsiV dependency recap (for readers unfamiliar):
  PsiV has shape (B, S, H, R, P) and is defined as
      psi_v[b, s, h, r, p] = V[b, s, h, p] * MIMO_V[h, r, p]
  MIMO_V is a learned parameter (static within a step); V is a derived
  activation (changes every forward). ⇒ cache is INTRA-STEP, not inter-step.
  See §2 of the design doc for the full analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
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
    """Crash loudly for entrypoints whose kernel integration is absent.

    Rationale (`feedback_no_silent_fallbacks.md`): the user opts in explicitly
    via CPPMEGA_MAMBA3_P2_PSIV_CACHE=1. Silently falling back to the cold
    kernel path would give them a wrong perf number and obscure the fact
    that the cache is a stub. Better to crash.
    """
    if is_enabled():
        raise NotImplementedError(
            "CPPMEGA_MAMBA3_P2_PSIV_CACHE=1 but TileLang cache integration is "
            "not implemented. "
            "See docs/mamba3_mimo_p2_psiv_cache_design.md §12 for status. "
            "Unset the variable to run the baseline kernel path."
        )


# ---------------------------------------------------------------------------
# Phase A: Python-level precompute / measurement
# ---------------------------------------------------------------------------


def _validate_psiv_inputs(V: torch.Tensor, mimo_v: torch.Tensor) -> tuple[int, int, int, int, int]:
    if V.ndim != 4:
        raise ValueError(f"V must have shape (B, S, H, P); got {tuple(V.shape)}")
    if mimo_v.ndim != 3:
        raise ValueError(f"mimo_v must have shape (H, R, P); got {tuple(mimo_v.shape)}")
    batch, seqlen, nheads, headdim_v = V.shape
    psi_heads, rank, psi_headdim_v = mimo_v.shape
    if psi_heads != nheads or psi_headdim_v != headdim_v:
        raise ValueError(
            "V and mimo_v shape mismatch: "
            f"V has H={nheads}, P={headdim_v}; "
            f"mimo_v has H={psi_heads}, P={psi_headdim_v}"
        )
    if mimo_v.device != V.device:
        raise ValueError(f"V and mimo_v must be on the same device; got {V.device} and {mimo_v.device}")
    return batch, seqlen, nheads, rank, headdim_v


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

    This is a measurement helper, not production integration. TileLang copies
    `MIMO_V` into a bf16/fp16 fragment before multiplying, so this helper casts
    `mimo_v` to `V.dtype` before the broadcast multiply.
    """
    _validate_psiv_inputs(V, mimo_v)
    psi = mimo_v.to(dtype=V.dtype)
    return (V.unsqueeze(3) * psi.unsqueeze(0).unsqueeze(0)).contiguous()


@dataclass(frozen=True)
class PsiVBenchmarkResult:
    """Small timing result for Phase-A `v * psi` materialization."""

    device: str
    shape: tuple[int, int, int, int, int]
    dtype: str
    cache_bytes: int
    mean_ms: float
    min_ms: float
    max_ms: float
    iters: int


def benchmark_precompute_psi_v(
    V: torch.Tensor,
    mimo_v: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> PsiVBenchmarkResult:
    """Time `precompute_psi_v` with CUDA events when available.

    The helper is intentionally narrow: it only measures the materialization
    cost and allocation footprint, which is the first go/no-go signal before
    any TileLang patch work.
    """
    batch, seqlen, nheads, rank, headdim_v = _validate_psiv_inputs(V, mimo_v)
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0")

    for _ in range(warmup):
        out = precompute_psi_v(V, mimo_v)
    if V.is_cuda:
        torch.cuda.synchronize(V.device)
        timings: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            out = precompute_psi_v(V, mimo_v)
            end.record()
            torch.cuda.synchronize(V.device)
            timings.append(start.elapsed_time(end))
    else:
        timings = []
        for _ in range(iters):
            t0 = perf_counter()
            out = precompute_psi_v(V, mimo_v)
            timings.append((perf_counter() - t0) * 1000.0)

    # Keep the last tensor live until after timing so eager execution cannot
    # trivially drop the allocation before synchronization.
    del out
    cache_bytes = estimate_cache_bytes(batch, seqlen, nheads, rank, headdim_v, V.dtype)
    return PsiVBenchmarkResult(
        device=str(V.device),
        shape=(batch, seqlen, nheads, rank, headdim_v),
        dtype=str(V.dtype),
        cache_bytes=cache_bytes,
        mean_ms=sum(timings) / len(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        iters=iters,
    )


# ---------------------------------------------------------------------------
# Phase B/C: gmem pool for PsiV cache
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

    This pool is not wired into the Mamba3 autograd wrapper. It exists so a
    future kernel patch can test allocation lifetime and CUDA graph behavior
    without changing the public shape contract again.
    """

    def __init__(self) -> None:
        self._pool: dict[tuple, list[torch.Tensor]] = {}
        self._checked_out: set[int] = set()

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

        Returned tensors are uninitialized. The caller must fill the entire
        tensor before passing it to any kernel.
        """
        key = (batch, seqlen, nheads, rank, headdim_v, dtype, torch.device(device))
        tensors = self._pool.get(key)
        if tensors:
            tensor = tensors.pop()
        else:
            tensor = torch.empty((batch, seqlen, nheads, rank, headdim_v), dtype=dtype, device=device)
        self._checked_out.add(id(tensor))
        return tensor

    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool."""
        tensor_id = id(tensor)
        if tensor_id not in self._checked_out:
            raise RuntimeError("PsiVCachePool.release called for a tensor that is not checked out")
        self._checked_out.remove(tensor_id)
        if tensor.ndim != 5:
            raise ValueError(f"PsiV cache tensor must be 5D; got {tuple(tensor.shape)}")
        batch, seqlen, nheads, rank, headdim_v = tensor.shape
        key = (batch, seqlen, nheads, rank, headdim_v, tensor.dtype, tensor.device)
        self._pool.setdefault(key, []).append(tensor)


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


def estimate_state_checkpoint_bytes(
    batch: int,
    seqlen: int,
    nheads: int,
    dstate: int,
    headdim_v: int,
    dtype: torch.dtype,
    *,
    chunk_size: int = 16,
    num_layers: int = 1,
) -> int:
    """Return bytes for a P3 `dstates_per_chunk` checkpoint.

    Shape: (B, H, ceil(S / chunk_size), N, P). The original P3 doc assumed
    fp32, but this accepts dtype explicitly so bf16/fp16 variants can be
    budgeted without changing call sites.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    return batch * nheads * nchunks * dstate * headdim_v * dtype_bytes * num_layers


@dataclass(frozen=True)
class Mamba3CacheBudget:
    """Memory budget for P2 PsiV and P3 dstates checkpointing."""

    psiv_bytes: int
    dstates_bytes: int
    total_bytes: int
    batch: int
    seqlen: int
    nheads: int
    rank: int
    dstate: int
    headdim_v: int
    chunk_size: int
    num_layers: int


def estimate_cache_budget(
    *,
    batch: int,
    seqlen: int,
    nheads: int,
    rank: int,
    dstate: int,
    headdim_v: int,
    psiv_dtype: torch.dtype = torch.bfloat16,
    dstates_dtype: torch.dtype = torch.float32,
    chunk_size: int = 16,
    num_layers: int = 1,
) -> Mamba3CacheBudget:
    psiv_bytes = estimate_cache_bytes(
        batch, seqlen, nheads, rank, headdim_v, psiv_dtype, num_layers=num_layers
    )
    dstates_bytes = estimate_state_checkpoint_bytes(
        batch,
        seqlen,
        nheads,
        dstate,
        headdim_v,
        dstates_dtype,
        chunk_size=chunk_size,
        num_layers=num_layers,
    )
    return Mamba3CacheBudget(
        psiv_bytes=psiv_bytes,
        dstates_bytes=dstates_bytes,
        total_bytes=psiv_bytes + dstates_bytes,
        batch=batch,
        seqlen=seqlen,
        nheads=nheads,
        rank=rank,
        dstate=dstate,
        headdim_v=headdim_v,
        chunk_size=chunk_size,
        num_layers=num_layers,
    )


__all__ = [
    "is_enabled",
    "precompute_psi_v",
    "benchmark_precompute_psi_v",
    "PsiVBenchmarkResult",
    "PsiVCachePool",
    "forward_with_cache",
    "backward_fwd_with_cache",
    "backward_bwd_with_cache",
    "estimate_cache_bytes",
    "estimate_state_checkpoint_bytes",
    "estimate_cache_budget",
    "Mamba3CacheBudget",
]
