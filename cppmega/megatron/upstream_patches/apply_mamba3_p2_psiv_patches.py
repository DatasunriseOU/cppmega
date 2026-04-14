"""Mamba3 MIMO P2 PsiV cache upstream patch applier — SKELETON.

Companion to `cppmega/megatron/mamba3_psiv_cache.py`.
Design: `docs/mamba3_mimo_p2_psiv_cache_design.md`.

Status (2026-04-14): SCAFFOLDING ONLY.

This file mirrors the structure of `apply_mamba3_mimo_p1_patches.py` (atomic
writes, idempotence, rank-0-only flock, line-count-preserving edits) so the
future implementer has a clear template. All real edits are TODOs.

What this patch will do when implemented (Phase B/C):
  1. Extend `mamba_mimo_fwd` in `mamba3_mimo_fwd.py` with a new `PsiV_out`
     kernel argument + a per-chunk `T.copy(psi_v, PsiV_out[...])` into it.
  2. Extend `mamba_mimo_bwd_fwd` in `mamba3_mimo_bwd.py` with a `PsiV_in`
     argument; replace the `psi_v = v * psi` line with a single
     `T.copy(PsiV_in[...], psi_v_shared)` load.
  3. Same for `mamba_mimo_bwd_bwd`.
  4. Extend the Python autograd op `mamba3_mimo` in `mamba3_mimo.py` to
     allocate the PsiV cache tensor, pass it to fwd as an output, save it
     to ctx, and pass it to bwd_fwd / bwd_bwd as an input.

Gotchas the implementer must watch for (learned from P1):
  * **Line-count preservation** — if this patch inserts new lines in the
    upstream .py file AFTER `import mamba_ssm` has happened in the process,
    `inspect.getsource` desyncs. See `_ensure_aggressive_merge` in
    `apply_mamba3_mimo_p1_patches.py` for the fix (merge onto existing
    lines). `reference_py_patch_line_shift_bug.md` has full context.
  * **Multi-rank race** — 8 ranks patching the same file concurrently =
    half-written files + IndentationError. Use the flock + DONE-sentinel
    pattern from `apply_all()` in the P1 applier.
  * **Idempotence** — re-running must be a no-op. Mark patched sites with
    `# cppmega P2` sentinel comments so subsequent runs can detect them.
  * **Atomic write** — use the `_atomic_write_text` helper from the P1
    file (tempfile + `os.replace`).

Env gate: `CPPMEGA_MAMBA3_P2_PSIV_CACHE=1` (same gate as the runtime
module in `mamba3_psiv_cache.py`). Default OFF. Mode: explicit opt-in.

Usage (once implemented):
    python -m cppmega.megatron.upstream_patches.apply_mamba3_p2_psiv_patches

Currently raises NotImplementedError — gate must stay OFF.
"""

from __future__ import annotations

import os
import sys

_ENV_FLAG = "CPPMEGA_MAMBA3_P2_PSIV_CACHE"


def _sites_to_patch() -> dict[str, int]:
    """Which upstream kernel files need edits, and how many sites each.

    TODO(Phase B/C): finalise once kernel signatures are decided.

    Planned sites:
      mamba3_mimo_fwd.py          -> mamba_mimo_fwd              (1 site)
      mamba3_mimo_fwd_varlen.py   -> mamba_mimo_fwd (varlen)     (1 site)
      mamba3_mimo_bwd.py          -> mamba_mimo_bwd_fwd          (1 site)
      mamba3_mimo_bwd.py          -> mamba_mimo_bwd_bwd          (1 site)
      mamba3_mimo_bwd_varlen.py   -> mamba_mimo_bwd_fwd (varlen) (1 site)
      mamba3_mimo_bwd_varlen.py   -> mamba_mimo_bwd_bwd (varlen) (1 site)
      mamba3_mimo.py              -> mamba3_mimo autograd op     (1 site,
                                                                  biggest edit)
    """
    return {
        "mamba3_mimo_fwd.py": 1,
        "mamba3_mimo_fwd_varlen.py": 1,
        "mamba3_mimo_bwd.py": 2,
        "mamba3_mimo_bwd_varlen.py": 2,
        "mamba3_mimo.py": 1,
    }


def apply_all() -> None:
    """Apply P2 patches. **STUB — raises NotImplementedError.**

    TODO(implementer): copy the structure from
    `apply_mamba3_mimo_p1_patches.apply_all()` — specifically the rank-0-only
    flock + DONE-sentinel pattern — and call site-specific edit helpers
    per file.

    Until this is implemented, any call crashes loudly so no silent
    partial patch can slip into production.
    """
    raise NotImplementedError(
        "apply_mamba3_p2_psiv_patches.apply_all is a scaffold. "
        "See docs/mamba3_mimo_p2_psiv_cache_design.md §9 for the roadmap. "
        "Do not enable CPPMEGA_MAMBA3_P2_PSIV_CACHE until Phase B/C are "
        "implemented and H200-perf-validated."
    )


def apply_if_requested() -> bool:
    """Env-gated entry point. Returns True if patches were applied.

    Safe to import/call in any shim — if the gate is off, this is a no-op.
    If the gate is ON but the implementation is absent, raises — matching
    the `mamba3_psiv_cache._refuse_if_gated` contract.
    """
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        return False
    apply_all()  # will raise NotImplementedError
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except NotImplementedError as exc:
        print(f"Not yet implemented: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
