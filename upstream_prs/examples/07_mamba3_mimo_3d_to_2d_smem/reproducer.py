"""Reproducer: Mamba3 MIMO bwd — 3D → 2D smem refactor for TMA compatibility.

NOTE: This is a reproducer for upstream_prs/07 (Mamba3 *kernel-source*
refactor targeting state-spaces/mamba). It is distinct from
upstream_prs/08 (TileLang `LowerBulkCopy` compiler-side issue). 07 is
our workaround for 08; this reproducer validates that the *kernel-level*
refactor is correctness-neutral AND unblocks the TMA fast-path.

PR site summary (from `upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md`):

  Site 1 — `qk_dot_shared`:
      Before:  T.alloc_shared([chunk_size, R, R], dtype)
      After:   T.alloc_shared([chunk_size, R * R], dtype)

  Site 2/3 — Q/K load smem views (the ones that block TMA):
      Before:  T.copy(Q[..., cs:cs+C, :, i_h_qk, :], q_smem_3d)  # 3D smem
               # gmem source is 5D: Q[B, S, R, G, N]
      After:   T.copy(Q[..., cs*R:cs*R+C*R, i_h_qk, :], q_smem_2d)  # 2D smem
               # gmem source flattened to 4D: Q[B, S*R, G, N]

What this reproducer does (tiny, self-contained — does NOT build the
full MIMO kernel, just the offending load + store indexing pattern):

  Variant A (3D — pre-refactor pattern):
      * Q/K tensors shape [B, S, R, G, N]
      * smem alloc  [chunk_size, R, N]       (Q/K load sites)
      * smem alloc  [chunk_size, R, R]       (qk_dot_shared)
      * Fill qk_dot in pure smem-indexing math, write back to global.
      * With TL_DISABLE_TMA_LOWER=False: expect ASSERTION_HIT_AT_3D
        on pre-PR-746 TileLang, or COMPILE_FALLBACK_AT_3D (cp.async
        fallback) on post-PR-746 TileLang. Either way, no TMA.

  Variant B (2D — post-refactor pattern from PR 07):
      * Q/K tensors shape [B, S*R, G, N]  (signature flattened)
      * smem alloc  [chunk_size*R, N]    (flat load target)
      * smem alloc  [chunk_size, R*R]    (flat qk_dot)
      * Same math, indexing remapped per PR 07.
      * Compiles cleanly with TMA lowering enabled (fast-path).

  Correctness: both variants run on the *same* random Q/K input and
  produce the same qk_dot output (algebraically identical indexing).

Tags printed:
    ASSERTION_HIT_AT_3D     — 3D variant hit the pre-PR-746 ICHECK
    COMPILE_FALLBACK_AT_3D  — 3D variant compiled but fell back from TMA
    CLEAN_COMPILE_AT_2D     — 2D variant compiled with TMA lowering on
    CORRECTNESS_PASS        — 3D and 2D variants produce matching outputs

Exit codes:
    0 — 2D compiles cleanly AND correctness passes. PR 07 refactor is
        valid (regardless of whether 3D hit assert or just fell back).
    1 — 2D failed to compile OR 3D and 2D disagree numerically.

Run on CUDA hardware (TileLang lowers via NVRTC/NVCC).

References:
    upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md
    upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md
    .tmp/mamba3_mimo_cutile/mamba3_mimo_bwd_phase0.py (real kernel source)
"""
from __future__ import annotations

import io
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout


# Tiny shapes — just enough to exercise the indexing pattern.
# C*R*N and C*R*R must fit in GB10 smem (99 KiB). With C=4, R=4, N=16 the
# combined working set is well under 2 KiB, trivially fits anywhere.
B = 1
S = 8            # timesteps (chunk_size * nchunks)
R = 4            # MIMO rank (matches production NAM56R)
G = 1            # GQA groups; we use i_h_qk = 0
N = 16           # head dim
CHUNK_SIZE = 4
NCHUNKS = S // CHUNK_SIZE
FUSED_CHUNK_SIZE = CHUNK_SIZE * R  # 16


_FATAL_PATTERNS = (
    re.compile(r"Cannot detect TMA layout", re.IGNORECASE),
    re.compile(r"Check failed:.*InputDim\(\)\s*==\s*2", re.IGNORECASE),
)


def _classify_exception(exc: BaseException, stderr_text: str) -> str:
    blob = f"{type(exc).__name__}: {exc}\n{stderr_text}"
    for pat in _FATAL_PATTERNS:
        if pat.search(blob):
            return "assert_3d"
    return "other"


def _build_3d_kernel(tl, T, disable_tma: bool):
    """Pre-refactor: 5D gmem Q/K + 3D smem (the site-1/2/3 pattern).

    Real kernel analogue: lines 235-249 and 629 of mamba3_mimo_bwd.py.
    We copy Q/K through 3D smem and re-emit into a 3D qk_dot_shared
    layout, then stream the smem contents back to global so the whole
    3D-smem lowering path is exercised.
    """
    pass_configs = {
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: disable_tma,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }

    @tl.jit(pass_configs=pass_configs)
    def k(
        Q: T.Tensor([B, S, R, G, N], "bfloat16"),
        K: T.Tensor([B, S, R, G, N], "bfloat16"),
        # Output: mirror of (q+k) written back from smem — same data,
        # just round-tripped through the 3D smem descriptor that PR 07
        # flattens.
        OUT: T.Tensor([B, NCHUNKS, CHUNK_SIZE, R, R], "float32"),
    ):
        with T.Kernel(NCHUNKS, B, threads=128) as (i_c, i_b):
            # Site 2/3 pattern: 3D smem view for Q and K.
            q_shared = T.alloc_shared([CHUNK_SIZE, R, N], "bfloat16")
            k_shared = T.alloc_shared([CHUNK_SIZE, R, N], "bfloat16")
            # Site 1 pattern: structural 3D qk_dot_shared.
            qk_dot_shared = T.alloc_shared([CHUNK_SIZE, R, R], "float32")

            i_h_qk = 0
            chunk_start = i_c * CHUNK_SIZE

            # Site 2/3: 5D gmem slice → 3D smem. This is what PR 07 refactors.
            T.copy(Q[i_b, chunk_start:chunk_start + CHUNK_SIZE, :, i_h_qk, :],
                   q_shared)
            T.copy(K[i_b, chunk_start:chunk_start + CHUNK_SIZE, :, i_h_qk, :],
                   k_shared)

            # Site 1: build qk_dot_shared in [chunk_size, R, R] using only
            # smem reads + parallel smem writes (no fragment accumulator —
            # that trips eager-mode layout inference on rank-3 fragments).
            # The quantity is a deterministic, order-independent function
            # of the 3D indices so the output is bit-stable.
            for cs, r1, r2 in T.Parallel(CHUNK_SIZE, R, R):
                # Pick a fixed representative N-slot so we avoid a reduction
                # (which would need either fragment acc or atomic add).
                # The refactor is about the smem *layout* — the arithmetic
                # only needs to access the 3D smem descriptor.
                qk_dot_shared[cs, r1, r2] = \
                    T.Cast("float32", q_shared[cs, r1, 0]) * \
                    T.Cast("float32", k_shared[cs, r2, 0]) + \
                    T.Cast("float32", q_shared[cs, r1, N - 1]) * \
                    T.Cast("float32", k_shared[cs, r2, N - 1])

            # Stream 3D smem → 4D gmem (tile index + 3D).
            for cs, r1, r2 in T.Parallel(CHUNK_SIZE, R, R):
                OUT[i_b, i_c, cs, r1, r2] = qk_dot_shared[cs, r1, r2]

    return k


def _build_2d_kernel(tl, T, disable_tma: bool):
    """Post-refactor: 4D gmem Q/K (S*R flattened) + 2D smem (PR 07 pattern)."""
    pass_configs = {
        tl.PassConfigKey.TL_DISABLE_TMA_LOWER: disable_tma,
        tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }

    @tl.jit(pass_configs=pass_configs)
    def k(
        # Q/K signature flattened S → S*R (caller passes .view(B, S*R, G, N)).
        Q: T.Tensor([B, S * R, G, N], "bfloat16"),
        K: T.Tensor([B, S * R, G, N], "bfloat16"),
        # Output flattened: [C, R, R] → [C, R*R].
        OUT: T.Tensor([B, NCHUNKS, CHUNK_SIZE, R * R], "float32"),
    ):
        with T.Kernel(NCHUNKS, B, threads=128) as (i_c, i_b):
            # 2D smem — TMA-compatible rank-2 descriptors.
            q_shared = T.alloc_shared([FUSED_CHUNK_SIZE, N], "bfloat16")
            k_shared = T.alloc_shared([FUSED_CHUNK_SIZE, N], "bfloat16")
            qk_dot_shared = T.alloc_shared([CHUNK_SIZE, R * R], "float32")

            i_h_qk = 0
            chunk_start = i_c * CHUNK_SIZE
            fused_chunk_start = chunk_start * R

            # PR 07 Site 2/3: direct 2D copy — this IS the TMA fast-path.
            T.copy(Q[i_b, fused_chunk_start:fused_chunk_start + FUSED_CHUNK_SIZE,
                     i_h_qk, :], q_shared)
            T.copy(K[i_b, fused_chunk_start:fused_chunk_start + FUSED_CHUNK_SIZE,
                     i_h_qk, :], k_shared)

            # PR 07 Site 1: flat qk_dot_shared, same math, flattened index.
            # Remap: q_shared[cs, r, n] ↔ q_shared[cs*R + r, n];
            #        qk_dot[cs, r1, r2] ↔ qk_dot[cs, r1*R + r2].
            for cs, r1, r2 in T.Parallel(CHUNK_SIZE, R, R):
                qk_dot_shared[cs, r1 * R + r2] = \
                    T.Cast("float32", q_shared[cs * R + r1, 0]) * \
                    T.Cast("float32", k_shared[cs * R + r2, 0]) + \
                    T.Cast("float32", q_shared[cs * R + r1, N - 1]) * \
                    T.Cast("float32", k_shared[cs * R + r2, N - 1])

            for cs, rr in T.Parallel(CHUNK_SIZE, R * R):
                OUT[i_b, i_c, cs, rr] = qk_dot_shared[cs, rr]

    return k


def _compile_and_run(tl, T, variant: str, disable_tma: bool):
    """Compile + execute the variant kernel. Returns (status, stderr, tensor_or_None, exc)."""
    import torch

    stderr_buf = io.StringIO()
    stdout_buf = io.StringIO()
    try:
        with redirect_stderr(stderr_buf), redirect_stdout(stdout_buf):
            if variant == "3d":
                k = _build_3d_kernel(tl, T, disable_tma=disable_tma)
            else:
                k = _build_2d_kernel(tl, T, disable_tma=disable_tma)

            dev = torch.device("cuda")
            torch.manual_seed(0)
            q_nat = torch.randn(B, S, R, G, N, device=dev, dtype=torch.bfloat16) * 0.1
            k_nat = torch.randn(B, S, R, G, N, device=dev, dtype=torch.bfloat16) * 0.1

            if variant == "3d":
                qk_out = torch.zeros(B, NCHUNKS, CHUNK_SIZE, R, R,
                                     device=dev, dtype=torch.float32)
                k(q_nat, k_nat, qk_out)
                out = qk_out
            else:
                # PR 07 caller convention: .view(B, S*R, G, N).  The memory layout
                # of q_nat is (B, S, R, G, N) contiguous, and S*R slice selects
                # the same (s, r) pairs in row-major order → zero-copy .view.
                q_flat = q_nat.reshape(B, S * R, G, N)
                k_flat = k_nat.reshape(B, S * R, G, N)
                qk_out = torch.zeros(B, NCHUNKS, CHUNK_SIZE, R * R,
                                     device=dev, dtype=torch.float32)
                k(q_flat, k_flat, qk_out)
                out = qk_out
            torch.cuda.synchronize()

        return ("ok", stderr_buf.getvalue() + stdout_buf.getvalue(), out, None)
    except BaseException as exc:  # noqa: BLE001 — catch ICHECK too
        tb = traceback.format_exc()
        combined = stderr_buf.getvalue() + stdout_buf.getvalue() + "\n" + tb
        return (_classify_exception(exc, combined), combined, None, exc)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not importable", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA device required (TileLang lowers via NVRTC/NVCC).",
              file=sys.stderr)
        return 2
    try:
        import tilelang as tl
        import tilelang.language as T
    except ImportError as exc:
        print(f"ERROR: tilelang import failed: {exc}", file=sys.stderr)
        return 2

    tl_ver = getattr(tl, "__version__", "unknown")
    dev_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"TILELANG_VERSION : {tl_ver}")
    print(f"DEVICE           : {dev_name} (sm_{cc[0]}{cc[1]})")
    print(f"SHAPES           : B={B} S={S} R={R} G={G} N={N} chunk={CHUNK_SIZE}")
    print(f"PR 07 target     : state-spaces/mamba — mamba3_mimo_bwd.py")
    print()
    os.environ.setdefault("TVM_LOG_DEBUG", "0")

    failed = False
    tags = []

    # --- 3D variant (pre-refactor pattern), TMA enabled. ---
    print("[A] 3D smem variant  (Q:[B,S,R,G,N] + smem[C,R,N] + qk_dot[C,R,R])")
    print("    TL_DISABLE_TMA_LOWER=False  — this is the site PR 07 refactors.")
    statusA, errA, outA, excA = _compile_and_run(tl, T, "3d", disable_tma=False)
    if statusA == "assert_3d":
        print("    STATUS: ASSERTION_HIT_AT_3D — pre-PR-746 TileLang ICHECK")
        print("            ('Cannot detect TMA layout'). 3D smem cannot reach TMA.")
        tags.append("ASSERTION_HIT_AT_3D")
        for line in errA.strip().splitlines()[-8:]:
            print(f"      {line}")
        # Expected-bug: re-run 3D with TMA off so we have a numerical baseline.
        print("    → re-running 3D with TL_DISABLE_TMA_LOWER=True to get baseline…")
        statusA, errA, outA, excA = _compile_and_run(tl, T, "3d", disable_tma=True)
        if statusA != "ok":
            print(f"    BASELINE FAIL: {type(excA).__name__}: {excA}")
            failed = True
        else:
            print("    3D baseline (TMA disabled) compiled OK — will use for parity.")
    elif statusA == "other":
        print(f"    STATUS: UNEXPECTED FAIL — {type(excA).__name__}: {excA}")
        for line in errA.strip().splitlines()[-8:]:
            print(f"      {line}")
        failed = True
    else:
        # Compiled with TMA enabled — on post-PR-746 TileLang this means it
        # fell back to cp.async via LOG(WARNING). A separate WS-pass warning
        # ("[WS] skipped: no TMA copies in pipeline loop") is also evidence
        # that warp-spec didn't engage because the copy didn't promote to TMA.
        warn_hit = ("fallback to normal copy" in errA.lower()
                    or "cannot detect tma layout" in errA.lower()
                    or "no tma copies" in errA.lower()
                    or "[ws] skipped" in errA.lower())
        if warn_hit:
            print("    STATUS: COMPILE_FALLBACK_AT_3D — compiled but fell back from TMA")
            tags.append("COMPILE_FALLBACK_AT_3D")
            for line in errA.strip().splitlines():
                low = line.lower()
                if "tma" in low or "fallback" in low or "warning" in low:
                    print(f"      {line.strip()}")
        else:
            print("    STATUS: OK (compiled — no TMA fallback warning captured)")
            print("            (TileLang may have inferred a 2D layout internally.")
            print("             PR 07 refactor is still valuable as explicit semantics.)")
    print()

    # --- 2D variant (post-refactor pattern), TMA enabled. ---
    print("[B] 2D smem variant  (Q:[B,S*R,G,N] + smem[C*R,N] + qk_dot[C,R*R])")
    print("    TL_DISABLE_TMA_LOWER=False  — this IS the TMA fast-path PR 07 unlocks.")
    statusB, errB, outB, excB = _compile_and_run(tl, T, "2d", disable_tma=False)
    if statusB != "ok":
        print(f"    STATUS: FAIL — {type(excB).__name__}: {excB}")
        for line in errB.strip().splitlines()[-8:]:
            print(f"      {line}")
        failed = True
    else:
        print("    STATUS: CLEAN_COMPILE_AT_2D — TMA path compiled")
        tags.append("CLEAN_COMPILE_AT_2D")
    print()

    # --- Correctness: 3D vs 2D parity. ---
    print("[C] Correctness: 3D output vs 2D output, same input")
    if outA is None or outB is None:
        print("    STATUS: SKIP (one of the variants did not produce output)")
        failed = True
    else:
        # Reshape to a common shape for comparison. The refactor uses
        # r1*R + r2 in the flat index, matching the C-contiguous layout of
        # a [..., R, R] tensor when reshaped to [..., R*R].
        outA_flat = outA.reshape(B, NCHUNKS, CHUNK_SIZE, R * R)
        diff = (outA_flat.float() - outB.float()).abs().max().item()
        print(f"    max |outA - outB| = {diff:.2e}")
        # Both paths do the same arithmetic on the same inputs — bitwise equal.
        if diff == 0.0:
            print("    STATUS: CORRECTNESS_PASS (bitwise identical)")
            tags.append("CORRECTNESS_PASS")
        elif diff < 1e-3:
            print("    STATUS: CORRECTNESS_PASS (within bf16 rounding)")
            tags.append("CORRECTNESS_PASS")
        else:
            print("    STATUS: CORRECTNESS_FAIL — refactor is NOT neutral; check indexing")
            failed = True
    print()

    print("=" * 72)
    print("TAGS:", " ".join(tags) if tags else "(none)")
    if failed:
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: OK — PR 07 refactor compiles cleanly with TMA and")
    print("         reproduces the same numerical output as the 3D variant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
