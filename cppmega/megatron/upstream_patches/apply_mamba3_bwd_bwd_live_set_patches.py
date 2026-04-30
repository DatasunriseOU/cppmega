"""Env-gated Mamba3 bwd_bwd live-set experiment applier.

Default behavior is a no-op. The candidate first applies the guarded stage2
force-nonTMA source layout patch, then applies the Wave31 bwd_bwd live-set
patch:

    upstream_prs/examples/14_mamba3_bwd_bwd_live_set/
        mamba3_bwd_bwd_late_dqk_recompute.patch

The candidate removes the long-lived ``dqk_from_diag_shared`` smem tile and
computes only the per-step R x R diagonal qk-dot consumer in a compact shared
cache, avoiding the full ``fused_chunk_size x fused_chunk_size`` intermediate.
Keep it default-off until H100/H200 production gates show a real win.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cppmega.megatron.upstream_patches import (
    apply_mamba3_stage2_force_nontma_patches as stage2,
)

_ENV_FLAG = "CPPMEGA_MAMBA3_BWD_BWD_LIVE_SET"
_ALLOW_MUTATION_FLAG = "MAMBA3_BWD_BWD_LIVE_SET_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_BWD_BWD_LIVE_SET_ROLLBACK"
_BACKUP_SUFFIX = ".cppmega_bwd_bwd_live_set.bak"

_PATCH_REL = Path(
    "upstream_prs/examples/14_mamba3_bwd_bwd_live_set/"
    "mamba3_bwd_bwd_late_dqk_recompute.patch"
)

_PATCHED_MARKERS = {
    "diag_microkernel": "dqk_diag_shared = T.alloc_shared([chunk_size, R, R], accum_dtype)",
    "diag_comment": "compact [chunk, R, R] shared cache",
    "dk_consumer": "dqk_diag_shared[cs, r_out, csr_in % R]",
}

_REMOVED_MARKERS = (
    "dqk_from_diag_shared = T.alloc_shared([fused_chunk_size, fused_chunk_size], accum_dtype)",
    "dqk_from_diag_shared: tilelang.layout.make_swizzled_layout(dqk_from_diag_shared)",
    "T.copy(dqk_from_diag_frag, dqk_from_diag_shared)",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _patch_path() -> Path:
    path = _repo_root() / _PATCH_REL
    if not path.exists():
        raise RuntimeError(f"bwd_bwd live-set patch file missing: {path}")
    return path


def _find_mamba3_bwd_file() -> Path:
    return stage2._find_mamba3_bwd_file()


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _flag_enabled() -> bool:
    value = os.environ.get(_ENV_FLAG, "0")
    return value in ("1", "true", "True", "yes", "on", "late_dqk_recompute")


def _is_patched(text: str) -> bool:
    return all(marker in text for marker in _PATCHED_MARKERS.values()) and not any(
        marker in text for marker in _REMOVED_MARKERS
    )


def _has_partial_markers(text: str) -> bool:
    present = sum(marker in text for marker in _PATCHED_MARKERS.values())
    return 0 < present < len(_PATCHED_MARKERS)


def _validate_patched(path: Path) -> None:
    text = path.read_text()
    stage2._validate_patched(path)
    missing = [name for name, marker in _PATCHED_MARKERS.items() if marker not in text]
    if missing:
        raise RuntimeError(f"{path}: live-set validation failed, missing {missing}")
    removed = [marker for marker in _REMOVED_MARKERS if marker in text]
    if removed:
        raise RuntimeError(f"{path}: live-set validation failed, stale markers {removed}")


def _apply_patch_to_temp(src: Path, patch_file: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_live_set_"))
    dst = work / src.name
    shutil.copy2(src, dst)
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-p4", str(dst)],
        input=patch_file.read_bytes(),
        capture_output=True,
        cwd=work,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "bwd_bwd live-set patch failed\n"
            f"stdout:\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')[-4000:]}"
        )
    return dst


def _stage2_base_for(src: Path) -> Path:
    text = src.read_text()
    if stage2._is_patched(text):
        stage2._validate_patched(src)
        work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_bwd_bwd_live_set_stage2_"))
        dst = work / src.name
        shutil.copy2(src, dst)
        return dst
    if stage2._has_partial_stage2_markers(text):
        raise RuntimeError(
            f"{src}: partial stage2 force-nonTMA markers detected; rollback or "
            "reinstall mamba_ssm before applying bwd_bwd live-set patch"
        )
    return stage2._apply_patch_to_temp(src, stage2._patch_path())


def _do_patch() -> None:
    path = _find_mamba3_bwd_file()
    text = path.read_text()
    print(f"Mamba3 bwd kernel path: {path}")

    if _is_patched(text):
        _validate_patched(path)
        print("  OK   bwd_bwd live-set patch already applied")
        return
    if _has_partial_markers(text):
        raise RuntimeError(
            f"{path}: partial bwd_bwd live-set markers detected. Set "
            f"{_ROLLBACK_FLAG}=1 to rollback from backup, or reinstall "
            "mamba_ssm before retrying."
        )

    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  DONE backup written: {backup}")
    else:
        print(f"  OK   backup already exists: {backup}")

    stage2_base = _stage2_base_for(path)
    patched = _apply_patch_to_temp(stage2_base, _patch_path())
    _validate_patched(patched)
    stage2._atomic_replace_from(patched, path)
    _validate_patched(path)
    print("  DONE bwd_bwd live-set late_dqk_recompute patch applied")


def rollback() -> None:
    path = _find_mamba3_bwd_file()
    backup = _backup_path(path)
    print(f"Mamba3 bwd kernel path: {path}")
    if backup.exists():
        stage2._atomic_replace_from(backup, path)
        print(f"  DONE restored backup: {backup}")
        return
    raise RuntimeError(
        f"No backup found at {backup}. Reinstall mamba_ssm to restore the kernel file."
    )


def _is_live_set_patch_applied() -> bool:
    try:
        path = _find_mamba3_bwd_file()
        if not _is_patched(path.read_text()):
            return False
        _validate_patched(path)
        return True
    except Exception:
        return False


def _is_live_set_patch_absent() -> bool:
    try:
        text = _find_mamba3_bwd_file().read_text()
        return not _is_patched(text) and not _has_partial_markers(text)
    except Exception:
        return False


def apply_all() -> None:
    """Apply the bwd_bwd live-set patch if explicit mutation gates are set."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        stage2._run_once_with_local_rank_guard(rollback, _is_live_set_patch_absent)
        return
    if not _flag_enabled():
        print(f"  SKIP {_ENV_FLAG} is not set to late_dqk_recompute")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    stage2._run_once_with_local_rank_guard(_do_patch, _is_live_set_patch_applied)


def apply_if_requested() -> bool:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if not _flag_enabled():
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
