"""Mamba3 MIMO P2 PsiV cache upstream patch/probe helpers.

Companion to `cppmega/megatron/mamba3_psiv_cache.py`.
Design: `docs/mamba3_mimo_p2_psiv_cache_design.md`.

Status: non-mutating probe implemented; mutating patch application is still
disabled. This lets us inspect the source tree and verify expected edit sites
without touching installed packages or upstream checkouts.

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

from dataclasses import dataclass
import os
from pathlib import Path
import sys

_ENV_FLAG = "CPPMEGA_MAMBA3_P2_PSIV_CACHE"


@dataclass(frozen=True)
class PatchSiteProbe:
    """Static analysis result for one candidate TileLang source directory."""

    root: Path
    exists: bool
    files_present: tuple[str, ...]
    files_missing: tuple[str, ...]
    fwd_psiv_materializations: int
    bwd_psiv_materializations: int
    bwd_dstates_updates: int
    autograd_wrapper_present: bool
    ready_for_patch_skeleton: bool

    def summary(self) -> str:
        status = "ready" if self.ready_for_patch_skeleton else "not-ready"
        return (
            f"{self.root} [{status}] files={len(self.files_present)}/"
            f"{len(self.files_present) + len(self.files_missing)} "
            f"fwd_psiv={self.fwd_psiv_materializations} "
            f"bwd_psiv={self.bwd_psiv_materializations} "
            f"dstates_updates={self.bwd_dstates_updates}"
        )


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


def _candidate_roots() -> list[Path]:
    """Return likely `.../mamba_ssm/ops/tilelang/mamba3` source dirs.

    The local venv currently exposes `mamba_ssm` but not `mamba_ssm.ops.tilelang`,
    so import-based discovery alone is insufficient. We also accept explicit
    env vars and the source/build paths used by the local cppmega workspace.
    """
    candidates: list[Path] = []
    for env_name in ("CPPMEGA_MAMBA3_TILELANG_DIR", "MAMBA3_TILELANG_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))

    source_root = os.environ.get("MAMBA_SSM_SOURCE_DIR")
    if source_root:
        candidates.append(Path(source_root) / "mamba_ssm/ops/tilelang/mamba3")

    try:
        import mamba_ssm.ops.tilelang.mamba3 as mamba3_tilelang  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        candidates.extend(Path(p) for p in getattr(mamba3_tilelang, "__path__", []))

    candidates.extend(
        [
            Path("/home/dave/state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3"),
            Path("/home/dave/state-spaces-mamba/build/lib.linux-aarch64-cpython-313/mamba_ssm/ops/tilelang/mamba3"),
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def probe_patch_sites(root: str | os.PathLike[str] | None = None) -> PatchSiteProbe:
    """Inspect one TileLang Mamba3 source directory without modifying it."""
    path = Path(root) if root is not None else _candidate_roots()[0]
    expected_files = tuple(_sites_to_patch())
    present = tuple(name for name in expected_files if (path / name).exists())
    missing = tuple(name for name in expected_files if not (path / name).exists())

    def read(name: str) -> str:
        file_path = path / name
        return file_path.read_text() if file_path.exists() else ""

    fwd_text = read("mamba3_mimo_fwd.py") + read("mamba3_mimo_fwd_varlen.py")
    bwd_text = read("mamba3_mimo_bwd.py") + read("mamba3_mimo_bwd_varlen.py")
    wrapper_text = read("mamba3_mimo.py")

    fwd_psiv = fwd_text.count("PsiV_frag = T.alloc_fragment")
    bwd_psiv = bwd_text.count("PsiV_frag = T.alloc_fragment")
    dstates_updates = bwd_text.count("T.gemm(q_shared, dPhiO_scaled_frag, dstates_frag")
    wrapper_present = "class _Mamba3Function" in wrapper_text and "ctx.save_for_backward" in wrapper_text

    # Expected source/build tree has five files, fwd materialization in regular
    # and varlen fwd, bwd materialization in bwd_bwd regular and varlen, and
    # the reverse-scan dstates update in bwd_bwd regular and varlen.
    ready = (
        len(missing) == 0
        and fwd_psiv >= 2
        and bwd_psiv >= 2
        and dstates_updates >= 2
        and wrapper_present
    )
    return PatchSiteProbe(
        root=path,
        exists=path.exists(),
        files_present=present,
        files_missing=missing,
        fwd_psiv_materializations=fwd_psiv,
        bwd_psiv_materializations=bwd_psiv,
        bwd_dstates_updates=dstates_updates,
        autograd_wrapper_present=wrapper_present,
        ready_for_patch_skeleton=ready,
    )


def probe_all_candidate_roots() -> list[PatchSiteProbe]:
    """Inspect every candidate root without modifying files."""
    return [probe_patch_sites(root) for root in _candidate_roots()]


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
    if "--probe" in sys.argv or "--dry-run" in sys.argv:
        for result in probe_all_candidate_roots():
            print(result.summary())
            if result.files_missing:
                print(f"  missing: {', '.join(result.files_missing)}")
        sys.exit(0)
    try:
        apply_all()
    except NotImplementedError as exc:
        print(f"Not yet implemented: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
