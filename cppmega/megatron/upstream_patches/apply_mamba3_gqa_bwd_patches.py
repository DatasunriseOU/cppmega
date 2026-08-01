"""Env-gated applier for the Mamba3 MIMO GQA backward reduction fix.

Upstream ``mamba_mimo_bwd_combined`` still only handles ``G == 1`` and
``G == H`` in the post-kernel dq/dk reduction. NAM56R uses ``1 < G < H``
(``G=8``), so Modal full-boundary runs can reach Mamba backward and then fail
with ``ValueError: G value of 8 is not currently supported!``.

Default behavior is a no-op. To mutate the installed ``mamba_ssm`` source:

    CPPMEGA_MAMBA3_GQA_BWD=1
    MAMBA3_GQA_BWD_ALLOW_FILE_MUTATION=1

Rollback:

    CPPMEGA_MAMBA3_GQA_BWD_ROLLBACK=1
    python -m cppmega.megatron.upstream_patches.apply_mamba3_gqa_bwd_patches
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_ENV_FLAG = "CPPMEGA_MAMBA3_GQA_BWD"
_ALLOW_MUTATION_FLAG = "MAMBA3_GQA_BWD_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_GQA_BWD_ROLLBACK"
_LOCK_NAME = "cppmega_mamba3_gqa_bwd.lock"
_BACKUP_SUFFIX = ".cppmega_gqa_bwd.bak"

_NONVARLEN_UNPATCHED = """    elif G == H:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dmimo_v = dmimo_v.sum(dim=0)
        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
        dD = dD.sum(dim=0) if dD is not None else None
    else:
        raise ValueError(f"G value of {G} is not currently supported!")
"""

_NONVARLEN_PATCHED = """    elif G == H:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dmimo_v = dmimo_v.sum(dim=0)
        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
        dD = dD.sum(dim=0) if dD is not None else None
    elif H % G == 0:
        hpg = H // G
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
        dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
        dmimo_v = dmimo_v.sum(dim=0)
        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
        dD = dD.sum(dim=0) if dD is not None else None
    else:
        raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")
"""

_VARLEN_UNPATCHED = """    elif G == H:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dmimo_v = dmimo_v.sum(dim=(0, 2))
        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None
        dD = dD.sum(dim=(0, 2)) if dD is not None else None
    else:
        raise ValueError(f"G value of {G} is not currently supported!")
"""

_VARLEN_PATCHED = """    elif G == H:
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dmimo_v = dmimo_v.sum(dim=(0, 2))
        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None
        dD = dD.sum(dim=(0, 2)) if dD is not None else None
    elif H % G == 0:
        hpg = H // G
        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
        dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
        dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
        dmimo_v = dmimo_v.sum(dim=(0, 2))
        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None
        dD = dD.sum(dim=(0, 2)) if dD is not None else None
    else:
        raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")
"""

_TARGETS = {
    "mamba3_mimo_bwd.py": (_NONVARLEN_UNPATCHED, _NONVARLEN_PATCHED),
    "mamba3_mimo_bwd_varlen.py": (_VARLEN_UNPATCHED, _VARLEN_PATCHED),
}


def _find_mamba3_bwd_files() -> dict[str, Path]:
    import importlib.util

    spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "mamba_ssm.ops.tilelang.mamba3 not importable - is mamba_ssm installed?"
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    paths = {name: root / name for name in _TARGETS}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Mamba3 bwd kernel file(s) missing: {missing}")
    return paths


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _is_patched_text(text: str) -> bool:
    return (
        "elif H % G == 0:" in text
        and "hpg = H // G" in text
        and "G must divide H" in text
        and "view(B, S, R, G, hpg, N).sum(dim=4)" in text
    )


def _has_partial_gqa_markers(text: str) -> bool:
    markers = (
        "elif H % G == 0:",
        "hpg = H // G",
        "G must divide H",
        "view(B, S, R, G, hpg, N).sum(dim=4)",
    )
    count = sum(marker in text for marker in markers)
    return 0 < count < len(markers)


def _validate_patched(path: Path) -> None:
    text = path.read_text()
    if not _is_patched_text(text):
        raise RuntimeError(f"{path}: GQA bwd validation failed")
    if _has_partial_gqa_markers(text):
        raise RuntimeError(f"{path}: partial GQA bwd markers detected")


def _atomic_write_text(path: Path, text: str) -> None:
    import py_compile

    tmp = path.with_name(f"{path.name}.cppmega_gqa_bwd.tmp.{os.getpid()}")
    tmp.write_text(text)
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _patch_one(path: Path, unpatched: str, patched: str) -> None:
    text = path.read_text()
    if _is_patched_text(text):
        _validate_patched(path)
        print(f"  OK   {path.name}: GQA bwd patch already applied")
        return
    if _has_partial_gqa_markers(text):
        raise RuntimeError(
            f"{path}: partial GQA bwd markers detected. Roll back or reinstall "
            "mamba_ssm before retrying."
        )
    if text.count(unpatched) != 1:
        raise RuntimeError(
            f"{path}: expected one unpatched G/H reduction block; upstream source "
            "changed or patch already diverged."
        )
    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  DONE backup written: {backup}")
    new_text = text.replace(unpatched, patched)
    _atomic_write_text(path, new_text)
    _validate_patched(path)
    print(f"  DONE {path.name}: GQA bwd reduction branch applied")


def _do_patch() -> None:
    paths = _find_mamba3_bwd_files()
    for name, path in paths.items():
        unpatched, patched = _TARGETS[name]
        _patch_one(path, unpatched, patched)


def rollback() -> None:
    paths = _find_mamba3_bwd_files()
    for path in paths.values():
        backup = _backup_path(path)
        if backup.exists():
            _atomic_write_text(path, backup.read_text())
            print(f"  DONE restored backup: {backup}")
        elif _is_patched_text(path.read_text()):
            print(
                f"  OK   {path.name}: GQA bwd patch active with no cppmega backup; "
                "leaving pre-existing source unchanged"
            )
        else:
            print(f"  OK   {path.name}: GQA bwd patch absent")


def _is_gqa_bwd_patch_applied() -> bool:
    try:
        paths = _find_mamba3_bwd_files()
        for path in paths.values():
            _validate_patched(path)
        return True
    except Exception:
        log.debug("GQA bwd patch detection failed", exc_info=True)
        return False


def _is_gqa_bwd_patch_absent() -> bool:
    try:
        return all(
            not _is_patched_text(path.read_text())
            and not _has_partial_gqa_markers(path.read_text())
            for path in _find_mamba3_bwd_files().values()
        )
    except Exception:
        log.debug("GQA bwd patch-absence detection failed", exc_info=True)
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
            print(f"[mamba3_gqa_bwd] rank={rank} waiting on rank-0")
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
                print(f"[mamba3_gqa_bwd] local_rank=0 rank={rank_env} mutating file")
                fn()
                lock_fh.write(sentinel)
                lock_fh.flush()
                os.fsync(lock_fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return

    print(f"[mamba3_gqa_bwd] local_rank={local_rank} rank={rank_env} waiting")
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if lock_path.exists():
            with open(lock_path) as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH)
                try:
                    if sentinel in lock_fh.read() and (is_done is None or is_done()):
                        return
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local_rank=0 GQA bwd patch: {lock_path}")


def apply_all() -> None:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        _run_once_with_local_rank_guard(rollback, _is_gqa_bwd_patch_absent)
        return
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        print(f"  SKIP {_ENV_FLAG} is not set")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    _run_once_with_local_rank_guard(_do_patch, _is_gqa_bwd_patch_applied)


def apply_if_requested() -> bool:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        log.debug("GQA bwd patch not requested: %s is not set", _ENV_FLAG)
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
