"""Env-gated applier for the Mamba3 stage2 force-nonTMA bwd patch.

This is a production-control wrapper around the benchmarked patch:

    upstream_prs/examples/13_tilelang_floormod_dbz/
        mamba3_bwd_stage2_force_nontma.patch

Default behavior is a no-op. To mutate the installed ``mamba_ssm`` source,
both gates must be set:

    CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA=1
    MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1

Rollback guard:

    CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1
    python -m cppmega.megatron.upstream_patches.apply_mamba3_stage2_force_nontma_patches

The patch is intentionally asymmetric by default: ``bf_num_stages=1`` and
``bb_num_stages=0``. H200 productionish benchmarking showed that bwd_fwd
benefits from WS/TMA while bwd_bwd regresses when WS/TMA is enabled.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_ENV_FLAG = "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA"
_ALLOW_MUTATION_FLAG = "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK"
_LOCK_NAME = "cppmega_mamba3_stage2_force_nontma.lock"
_BACKUP_SUFFIX = ".cppmega_stage2_force_nontma.bak"

_PATCH_REL = Path(
    "upstream_prs/examples/13_tilelang_floormod_dbz/"
    "mamba3_bwd_stage2_force_nontma.patch"
)

_PATCHED_MARKERS = {
    "flat_q": "Q: T.Tensor([B, S * R, G, N], dtype)",
    "flat_qk": "QK_DOT: T.Tensor([B, H, S, R * R], dtype)",
    "bf_default": "bf_num_stages=1",
    "bb_default": "bb_num_stages=0",
    "direct_qk": "qk_dot_shared[cs, r_out * R + r_in]",
}

_STRUCTURAL_PATCHED_MARKERS = {
    name: marker
    for name, marker in _PATCHED_MARKERS.items()
    if name in ("flat_q", "flat_qk", "direct_qk")
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _patch_path() -> Path:
    path = _repo_root() / _PATCH_REL
    if not path.exists():
        raise RuntimeError(f"stage2 force-nonTMA patch file missing: {path}")
    return path


def _find_mamba3_bwd_file() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "mamba_ssm.ops.tilelang.mamba3 not importable - is mamba_ssm installed?"
        )
    path = Path(next(iter(spec.submodule_search_locations))) / "mamba3_mimo_bwd.py"
    if not path.exists():
        raise RuntimeError(f"Mamba3 bwd kernel file missing: {path}")
    return path


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _is_patched(text: str) -> bool:
    return all(marker in text for marker in _PATCHED_MARKERS.values())


def _has_partial_stage2_markers(text: str) -> bool:
    structural_count = sum(marker in text for marker in _STRUCTURAL_PATCHED_MARKERS.values())
    if structural_count == 0:
        return False
    full_count = sum(marker in text for marker in _PATCHED_MARKERS.values())
    return full_count < len(_PATCHED_MARKERS)


def _validate_patched(path: Path) -> None:
    text = path.read_text()
    missing = [name for name, marker in _PATCHED_MARKERS.items() if marker not in text]
    if missing:
        raise RuntimeError(f"{path}: patched validation failed, missing markers {missing}")
    if "bb_num_stages=1" in text:
        raise RuntimeError(
            f"{path}: rollback guard tripped - found bb_num_stages=1. "
            "The production candidate must stay bf=1,bb=0."
        )
    disable_tma_count = text.count("disable_tma=True")
    if disable_tma_count < 10:
        raise RuntimeError(
            f"{path}: expected targeted per-copy disable_tma guards, "
            f"found only {disable_tma_count}"
        )


def _atomic_replace_from(src: Path, dst: Path) -> None:
    import py_compile

    py_compile.compile(str(src), doraise=True)
    tmp = dst.with_name(f"{dst.name}.cppmega_stage2.tmp.{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _apply_patch_to_temp(src: Path, patch_file: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_stage2_force_nontma_"))
    dst = work / src.name
    shutil.copy2(src, dst)
    patch_bytes = patch_file.read_bytes()
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-p4", str(dst)],
        input=patch_bytes,
        capture_output=True,
        cwd=work,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stage2 force-nonTMA patch failed\n"
            f"stdout:\n{proc.stdout.decode(errors='replace')[-4000:]}\n"
            f"stderr:\n{proc.stderr.decode(errors='replace')[-4000:]}"
        )
    _validate_patched(dst)
    return dst


def _do_patch() -> None:
    path = _find_mamba3_bwd_file()
    text = path.read_text()
    print(f"Mamba3 bwd kernel path: {path}")

    if _is_patched(text):
        _validate_patched(path)
        print("  OK   stage2 force-nonTMA patch already applied")
        return
    if _has_partial_stage2_markers(text):
        raise RuntimeError(
            f"{path}: partial stage2 force-nonTMA markers detected. "
            f"Set {_ROLLBACK_FLAG}=1 to rollback from backup/reverse patch, "
            "or reinstall mamba_ssm before retrying."
        )

    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  DONE backup written: {backup}")
    else:
        print(f"  OK   backup already exists: {backup}")

    patched = _apply_patch_to_temp(path, _patch_path())
    _atomic_replace_from(patched, path)
    _validate_patched(path)
    print("  DONE stage2 force-nonTMA patch applied")
    print("  Active default: bf_num_stages=1, bb_num_stages=0")


def _reverse_patch(path: Path, patch_file: Path) -> bool:
    work = Path(tempfile.mkdtemp(prefix="cppmega_mamba3_stage2_force_nontma_rollback_"))
    dst = work / path.name
    shutil.copy2(path, dst)
    proc = subprocess.run(
        ["patch", "--ignore-whitespace", "-R", "-p4", str(dst)],
        input=patch_file.read_bytes(),
        capture_output=True,
        cwd=work,
        check=False,
    )
    if proc.returncode != 0:
        return False
    _atomic_replace_from(dst, path)
    return True


def rollback() -> None:
    """Restore the pre-patch file from backup, falling back to reverse patch."""
    path = _find_mamba3_bwd_file()
    backup = _backup_path(path)
    print(f"Mamba3 bwd kernel path: {path}")

    if backup.exists():
        _atomic_replace_from(backup, path)
        print(f"  DONE restored backup: {backup}")
        return

    if _is_patched(path.read_text()) and _reverse_patch(path, _patch_path()):
        print("  DONE reverted stage2 force-nonTMA patch with patch -R")
        return

    raise RuntimeError(
        f"No backup found at {backup} and reverse patch failed. "
        "Reinstall mamba_ssm to restore the upstream kernel file."
    )


def _is_stage2_patch_applied() -> bool:
    try:
        path = _find_mamba3_bwd_file()
        if not _is_patched(path.read_text()):
            return False
        _validate_patched(path)
        return True
    except Exception:
        log.debug("stage2 force-nonTMA patch detection failed", exc_info=True)
        return False


def _is_stage2_patch_absent() -> bool:
    try:
        text = _find_mamba3_bwd_file().read_text()
        return not _is_patched(text) and not _has_partial_stage2_markers(text)
    except Exception:
        log.debug("stage2 force-nonTMA patch-absence detection failed", exc_info=True)
        return False


def _run_once_with_local_rank_guard(fn, is_done=None) -> None:
    try:
        import torch.distributed as dist
    except Exception:
        log.debug("torch.distributed unavailable; falling back to file-lock guard", exc_info=True)
        dist = None  # type: ignore[assignment]

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            fn()
        else:
            print(f"[mamba3_stage2_force_nontma] rank={rank} waiting on rank-0")
        dist.barrier()
        return

    import fcntl
    import time

    lock_path = Path(tempfile.gettempdir()) / _LOCK_NAME
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "?"
    sentinel = "DONE\n"

    if local_rank == 0:
        lock_path.unlink(missing_ok=True)
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                print(
                    "[mamba3_stage2_force_nontma] "
                    f"local_rank=0 rank={rank_env} mutating file"
                )
                fn()
                lock_fh.write(sentinel)
                lock_fh.flush()
                os.fsync(lock_fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return

    print(
        "[mamba3_stage2_force_nontma] "
        f"local_rank={local_rank} rank={rank_env} waiting for local_rank=0"
    )
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if lock_path.exists():
            with open(lock_path) as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH)
                try:
                    if sentinel in lock_fh.read() and (
                        is_done is None or is_done()
                    ):
                        return
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        time.sleep(0.1)
    raise RuntimeError(
        f"Timed out waiting for local_rank=0 stage2 patch. lockfile={lock_path}"
    )


def apply_all() -> None:
    """Apply the stage2 patch if explicit mutation gates are set."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        _run_once_with_local_rank_guard(rollback, _is_stage2_patch_absent)
        return
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        print(f"  SKIP {_ENV_FLAG} is not set")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    _run_once_with_local_rank_guard(_do_patch, _is_stage2_patch_applied)


def apply_if_requested() -> bool:
    """Env-gated entry point for a future shim."""
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        log.debug("stage2 force-nonTMA patch not requested: %s is not set", _ENV_FLAG)
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
