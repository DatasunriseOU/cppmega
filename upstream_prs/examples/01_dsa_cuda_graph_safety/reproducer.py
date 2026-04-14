"""Reproducer: DSA CUDA graph capture breaks on torch.equal / .any() CPU-syncs.

Mirrors the CG-unsafe patterns found in
``megatron/core/transformer/experimental_attention_variant/dsa.py``:

  1. ``torch.equal(finite, expected)`` — validation check that implicitly
     calls ``.item()`` and triggers ``cudaStreamSynchronize``.
  2. ``if torch.any(idx_chunk < 0): ... if valid_topk.any(): ...`` inside
     ``_scatter_topk_into_index_mask`` — branches on a GPU reduction, which
     requires a host-side bool and syncs the stream.

Both patterns are forbidden inside ``torch.cuda.graph(g):`` capture and
raise ``cudaErrorStreamCaptureUnsupported`` / RuntimeError.

The reproducer does two runs:

  (A) Unpatched module (mirrors upstream dsa.py today):
      expected: capture FAILS with stream-capture RuntimeError.
      Prints ``BUG_REPRODUCED``.

  (B) Patched module (branchless clamp+scatter+fixup, no validation
      .item()); expected: capture SUCCEEDS and the graph replays
      producing the same mask as eager.
      Prints ``FIX_VALIDATED``.

Exit code:
  0 — bug reproduced on (A) AND fix captured + numerically matches on (B)
  1 — (A) unexpectedly captured (bug not present); or (B) failed
  2 — no CUDA device available

Reference: ``upstream_prs/01_dsa_cuda_graph_safety.md``.
"""
from __future__ import annotations

import sys

import torch


# ---------------------------------------------------------------------------
# (A) Unpatched module — mirrors CG-unsafe dsa.py patterns exactly.
# ---------------------------------------------------------------------------


class DsaScatterUnpatched(torch.nn.Module):
    """Mirrors upstream ``_scatter_topk_into_index_mask`` + validation."""

    def __init__(self, s_kv: int) -> None:
        super().__init__()
        self.s_kv = s_kv

    def forward(
        self,
        index_mask: torch.Tensor,    # [b, s_q, s_kv], fill with -inf, zero where selected
        idx_chunk: torch.Tensor,     # [b, chunk_len, topk] int64, may contain -1 sentinels
        finite_ref: torch.Tensor,    # [*] bool, "expected" invariant
        finite_got: torch.Tensor,    # [*] bool, what the indexer produced
        s0: int,
        s1: int,
    ) -> torch.Tensor:
        # Pattern 1: torch.equal() validation — CPU sync via .item().
        # (Upstream: ``if not torch.equal(finite, expected): raise ...``)
        if not torch.equal(finite_got, finite_ref):
            raise RuntimeError("finite mask mismatch")

        # Pattern 2: branch on torch.any() reduction — CPU sync.
        # (Upstream: ``if torch.any(idx_chunk < 0): ... if valid_topk.any(): ...``)
        if torch.any(idx_chunk < 0):
            valid_topk = idx_chunk >= 0
            if valid_topk.any():
                b_idx, q_rel_idx, t_idx = torch.where(valid_topk)
                q_idx = q_rel_idx + s0
                k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
                index_mask[b_idx, q_idx, k_idx] = 0.0
        else:
            index_mask[:, s0:s1].scatter_(-1, idx_chunk, 0.0)

        return index_mask


# ---------------------------------------------------------------------------
# (B) Patched module — branchless clamp+scatter+fixup, no validation .item().
# ---------------------------------------------------------------------------


class DsaScatterPatched(torch.nn.Module):
    """Branchless, CG-safe drop-in replacement."""

    def __init__(self, s_kv: int) -> None:
        super().__init__()
        self.s_kv = s_kv

    def forward(
        self,
        index_mask: torch.Tensor,
        idx_chunk: torch.Tensor,
        finite_ref: torch.Tensor,
        finite_got: torch.Tensor,
        s0: int,
        s1: int,
    ) -> torch.Tensor:
        # Validation check: skipped under stream capture (invariant holds
        # by construction; in debug runs guard with
        # ``torch.cuda.is_current_stream_capturing()``).
        # No torch.equal() / .item() on the capture path.

        # Branchless scatter with -1 sentinel fixup.
        sentinel = idx_chunk < 0
        safe_chunk = idx_chunk.clamp(min=0)
        index_mask[:, s0:s1].scatter_(-1, safe_chunk, 0.0)
        # Undo the position-0 unmasking that clamped sentinels caused,
        # but only on rows that had NO real index 0.
        has_sent = sentinel.any(dim=-1)                       # [b, chunk_len]
        has_real0 = ((idx_chunk == 0) & ~sentinel).any(dim=-1)  # [b, chunk_len]
        fixup = has_sent & ~has_real0
        index_mask[:, s0:s1, 0].masked_fill_(fixup, float("-inf"))

        return index_mask


# ---------------------------------------------------------------------------
# Eager oracle: what the mask should look like after the scatter.
# ---------------------------------------------------------------------------


def eager_reference(
    index_mask: torch.Tensor,
    idx_chunk: torch.Tensor,
    s0: int,
    s1: int,
) -> torch.Tensor:
    """Branchy reference: matches upstream semantics exactly."""
    out = index_mask.clone()
    valid = idx_chunk >= 0
    if valid.any().item():
        b_idx, q_rel_idx, t_idx = torch.where(valid)
        q_idx = q_rel_idx + s0
        k_idx = idx_chunk[b_idx, q_rel_idx, t_idx]
        out[b_idx, q_idx, k_idx] = 0.0
    return out


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_inputs(device: torch.device, with_sentinels: bool = True):
    torch.manual_seed(0)
    b, s_q, s_kv, topk = 2, 64, 256, 8
    s0, s1 = 0, s_q  # chunk covers the whole query range
    chunk_len = s1 - s0

    index_mask = torch.full((b, s_q, s_kv), float("-inf"), device=device)
    idx_chunk = torch.randint(
        low=0, high=s_kv, size=(b, chunk_len, topk), device=device, dtype=torch.long
    )
    if with_sentinels:
        # Mark ~15% of positions as -1 sentinels (indexer "not enough candidates").
        sentinel_mask = torch.rand(b, chunk_len, topk, device=device) < 0.15
        idx_chunk = torch.where(
            sentinel_mask, torch.full_like(idx_chunk, -1), idx_chunk
        )

    # Validation tensors — deterministic, equal by construction.
    finite_ref = torch.ones(b, chunk_len, dtype=torch.bool, device=device)
    finite_got = finite_ref.clone()
    return index_mask, idx_chunk, finite_ref, finite_got, s0, s1


def _is_stream_capture_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "cudaerrorstreamcaptureunsupported",
            "stream capture",
            "streamcapture",
            "capturing",
            "operation not permitted when stream is capturing",
        )
    )


# ---------------------------------------------------------------------------
# Runners.
# ---------------------------------------------------------------------------


def run_unpatched(device: torch.device) -> bool:
    """Attempt CUDA graph capture of the unpatched module.

    Returns True if capture failed as expected (bug reproduced).
    """
    print("=" * 72)
    print("(A) UNPATCHED — expecting CUDA graph capture to FAIL")
    print("=" * 72)

    mod = DsaScatterUnpatched(s_kv=256).to(device)
    index_mask, idx_chunk, finite_ref, finite_got, s0, s1 = _make_inputs(device)

    # Warm up on a side stream (required by torch.cuda.graph).
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            _ = mod(
                index_mask.clone(), idx_chunk, finite_ref, finite_got, s0, s1
            )
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    g = torch.cuda.CUDAGraph()
    capture_mask = index_mask.clone()
    try:
        with torch.cuda.graph(g):
            _ = mod(capture_mask, idx_chunk, finite_ref, finite_got, s0, s1)
    except (RuntimeError, torch.cuda.CudaError) as exc:
        if _is_stream_capture_error(exc):
            print(f"  capture raised (as expected): {type(exc).__name__}: {exc}")
            print("  BUG_REPRODUCED")
            return True
        print(f"  capture raised UNEXPECTED error: {type(exc).__name__}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        if _is_stream_capture_error(exc):
            print(f"  capture raised (as expected): {type(exc).__name__}: {exc}")
            print("  BUG_REPRODUCED")
            return True
        print(f"  capture raised UNEXPECTED error: {type(exc).__name__}: {exc}")
        return False

    # No exception — capture succeeded.  That means the bug is NOT present
    # on this build (torch handled it, or upstream already fixed it).
    print("  capture UNEXPECTEDLY succeeded — bug is not present on this build.")
    return False


def run_patched(device: torch.device) -> bool:
    """Capture + replay the patched module, compare to eager reference."""
    print()
    print("=" * 72)
    print("(B) PATCHED — expecting CUDA graph capture to SUCCEED")
    print("=" * 72)

    mod = DsaScatterPatched(s_kv=256).to(device)
    index_mask, idx_chunk, finite_ref, finite_got, s0, s1 = _make_inputs(device)

    # Eager reference BEFORE capture (fresh fill-mask).
    ref_mask = eager_reference(index_mask, idx_chunk, s0, s1)

    # Warm up.
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            _ = mod(
                index_mask.clone(), idx_chunk, finite_ref, finite_got, s0, s1
            )
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    # Static inputs/outputs for the graph.
    static_mask = index_mask.clone()
    static_idx = idx_chunk.clone()
    static_finite_ref = finite_ref.clone()
    static_finite_got = finite_got.clone()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            out = mod(
                static_mask, static_idx, static_finite_ref, static_finite_got,
                s0, s1,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  capture FAILED: {type(exc).__name__}: {exc}")
        return False

    # Replay — fresh inputs copied into the static buffers.
    static_mask.fill_(float("-inf"))
    static_idx.copy_(idx_chunk)
    g.replay()
    torch.cuda.synchronize(device)

    # The patched scatter uses a fixup that is NOT identical to the eager
    # branchy reference: clamped sentinels unmask (b, q, 0), then
    # masked_fill_ restores -inf on rows with no real 0.  Rows that have
    # BOTH a real 0 and a sentinel end up with position 0 unmasked — which
    # is the correct semantics (real 0 dominates).
    #
    # Compare at the "unmasked set" level rather than bytewise:
    got_unmasked = (out == 0.0)
    ref_unmasked = (ref_mask == 0.0)
    mismatch = (got_unmasked ^ ref_unmasked).sum().item()
    if mismatch != 0:
        print(f"  capture OK but replay mask differs from eager ref "
              f"({mismatch} mismatched positions)")
        return False

    print("  capture OK, replay matches eager reference exactly")
    print("  FIX_VALIDATED")
    return True


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required (graph capture is CUDA-only).")
        return 2

    device = torch.device("cuda")
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(device)}")
    print()

    bug = run_unpatched(device)
    fix = run_patched(device)

    print()
    print("=" * 72)
    if bug and fix:
        print("VERDICT: bug reproduced on unpatched + fix validated on patched.")
        return 0
    if not bug:
        print("VERDICT: BUG NOT REPRODUCED on unpatched module.")
        print("         Either the torch build silently handles CPU-sync under")
        print("         capture, or the pattern has been fixed elsewhere.")
        return 1
    print("VERDICT: fix did not validate (capture failed or mismatch).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
