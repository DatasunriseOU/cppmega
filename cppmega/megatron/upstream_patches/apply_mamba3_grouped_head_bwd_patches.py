"""Env-gated applier for Mamba3 MIMO backward grouped-head reduction support.

This patches the installed ``mamba_ssm`` TileLang source so
``mamba_mimo_bwd_combined`` handles intermediate grouped-head shapes
(``1 < G < H`` and ``H % G == 0``). NAM56R uses ``G=8`` and currently
hits the upstream fallback:

    ValueError: G value of 8 is not currently supported!

Default behavior is a no-op. Mutating site-packages requires both gates:

    CPPMEGA_MAMBA3_GROUPED_HEAD_BWD=1
    MAMBA3_GROUPED_HEAD_BWD_ALLOW_FILE_MUTATION=1

Rollback:

    CPPMEGA_MAMBA3_GROUPED_HEAD_BWD_ROLLBACK=1
    python -m cppmega.megatron.upstream_patches.apply_mamba3_grouped_head_bwd_patches
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_ENV_FLAG = "CPPMEGA_MAMBA3_GROUPED_HEAD_BWD"
_ALLOW_MUTATION_FLAG = "MAMBA3_GROUPED_HEAD_BWD_ALLOW_FILE_MUTATION"
_ROLLBACK_FLAG = "CPPMEGA_MAMBA3_GROUPED_HEAD_BWD_ROLLBACK"
_LOCK_NAME = "cppmega_mamba3_grouped_head_bwd.lock"
_BACKUP_SUFFIX = ".cppmega_grouped_head_bwd.bak"

_TARGETS = ("mamba3_mimo_bwd.py", "mamba3_mimo_bwd_varlen.py")

_REGULAR_ORIGINAL_BLOCK = (
    "    elif G == H:\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dmimo_v = dmimo_v.sum(dim=0)\n"
    "        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=0) if dD is not None else None\n"
    "    else:\n"
    '        raise ValueError(f"G value of {G} is not currently supported!")\n'
)

_REGULAR_PATCHED_BLOCK = (
    "    elif G == H:\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dmimo_v = dmimo_v.sum(dim=0)\n"
    "        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=0) if dD is not None else None\n"
    "    elif H % G == 0:\n"
    "        # Grouped-head MIMO: 1 < G < H, H divisible by G. Sum over heads_per_group.\n"
    "        hpg = H // G\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        dmimo_v = dmimo_v.sum(dim=0)\n"
    "        dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=0) if dD is not None else None\n"
    "    else:\n"
    '        raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")\n'
)

_VARLEN_ORIGINAL_BLOCK = (
    "    elif G == H:\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dmimo_v = dmimo_v.sum(dim=(0, 2))\n"
    "        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=(0, 2)) if dD is not None else None\n"
    "    else:\n"
    '        raise ValueError(f"G value of {G} is not currently supported!")\n'
)

_VARLEN_PATCHED_BLOCK = (
    "    elif G == H:\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dmimo_v = dmimo_v.sum(dim=(0, 2))\n"
    "        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=(0, 2)) if dD is not None else None\n"
    "    elif H % G == 0:\n"
    "        # Grouped-head MIMO: 1 < G < H, H divisible by G. Sum over heads_per_group.\n"
    "        hpg = H // G\n"
    "        dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))\n"
    "        dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)\n"
    "        dmimo_v = dmimo_v.sum(dim=(0, 2))\n"
    "        dmimo_z = dmimo_z.sum(dim=(0, 2)) if dmimo_z is not None else None\n"
    "        dD = dD.sum(dim=(0, 2)) if dD is not None else None\n"
    "    else:\n"
    '        raise ValueError(f"G value of {G} is not currently supported (H={H}, G must divide H)!")\n'
)


def _find_mamba3_tilelang_dir() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("mamba_ssm.ops.tilelang.mamba3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "mamba_ssm.ops.tilelang.mamba3 not importable - is mamba_ssm installed?"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def _backup_path(path: Path) -> Path:
    return path.with_name(path.name + _BACKUP_SUFFIX)


def _blocks_for(path: Path) -> tuple[str, str]:
    if path.name == "mamba3_mimo_bwd.py":
        return _REGULAR_ORIGINAL_BLOCK, _REGULAR_PATCHED_BLOCK
    if path.name == "mamba3_mimo_bwd_varlen.py":
        return _VARLEN_ORIGINAL_BLOCK, _VARLEN_PATCHED_BLOCK
    raise RuntimeError(f"unsupported Mamba3 bwd target: {path.name}")


def _is_patched_text(text: str) -> bool:
    return (
        "elif H % G == 0:" in text
        and "hpg = H // G" in text
        and "G must divide H" in text
        and "dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)" in text
        and "dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)" in text
    )


def _has_partial_markers(text: str) -> bool:
    markers = (
        "elif H % G == 0:",
        "hpg = H // G",
        "G must divide H",
        "dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)",
        "dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)",
    )
    count = sum(marker in text for marker in markers)
    return 0 < count < len(markers)


def _atomic_write_text(path: Path, content: str) -> None:
    import py_compile

    tmp = path.with_name(f"{path.name}.cppmega_grouped_head_bwd.tmp.{os.getpid()}")
    tmp.write_text(content)
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _patch_text(text: str, path: Path) -> tuple[str, bool]:
    if _is_patched_text(text):
        return text, False
    if _has_partial_markers(text):
        raise RuntimeError(f"{path}: partial Mamba3 grouped-head bwd patch markers detected")

    original, patched = _blocks_for(path)
    count = text.count(original)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected exactly one unpatched grouped-head reduction block, found {count}. "
            "The upstream source shape changed or another patch touched this region."
        )
    return text.replace(original, patched, 1), True


def _unpatch_text(text: str, path: Path) -> tuple[str, bool]:
    original, patched = _blocks_for(path)
    count = text.count(patched)
    if count == 1:
        return text.replace(patched, original, 1), True
    if _is_patched_text(text):
        return text, False
    return text, False


def _validate_file(path: Path) -> None:
    text = path.read_text()
    if not _is_patched_text(text):
        raise RuntimeError(f"{path}: grouped-head bwd validation failed")
    if _has_partial_markers(text):
        raise RuntimeError(f"{path}: partial grouped-head bwd markers after patch")


def _target_paths() -> list[Path]:
    base = _find_mamba3_tilelang_dir()
    paths = [base / name for name in _TARGETS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Mamba3 bwd kernel file(s) missing: {missing}")
    return paths


def _do_patch() -> None:
    paths = _target_paths()
    for path in paths:
        print(f"Mamba3 bwd kernel path: {path}")
        text = path.read_text()
        new_text, changed = _patch_text(text, path)
        if not changed:
            _validate_file(path)
            print(f"  OK   {path.name}: grouped-head bwd branch already applied")
            continue

        backup = _backup_path(path)
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  DONE backup written: {backup}")
        else:
            print(f"  OK   backup already exists: {backup}")

        _atomic_write_text(path, new_text)
        _validate_file(path)
        print(f"  DONE {path.name}: grouped-head bwd branch applied")


def rollback() -> None:
    paths = _target_paths()
    for path in paths:
        backup = _backup_path(path)
        print(f"Mamba3 bwd kernel path: {path}")
        if backup.exists():
            _atomic_write_text(path, backup.read_text())
            print(f"  DONE restored backup: {backup}")
            continue

        text = path.read_text()
        if _is_patched_text(text):
            print(f"  SKIP no backup for {path.name}; leaving existing grouped-head-patched source intact")
            continue

        print(f"  OK   {path.name}: no grouped-head patch present")


def _is_grouped_head_patch_applied() -> bool:
    try:
        return all(_is_patched_text(path.read_text()) for path in _target_paths())
    except Exception:
        return False


def _run_once_with_local_rank_guard(fn, is_done=None) -> None:
    try:
        import torch.distributed as dist
    except Exception:
        dist = None  # type: ignore[assignment]

    if dist is not None and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            fn()
        else:
            print(f"[mamba3_grouped_head_bwd] rank={rank} waiting on rank-0")
        dist.barrier()
        return

    import fcntl
    import tempfile
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
                print(f"[mamba3_grouped_head_bwd] local_rank=0 rank={rank_env} mutating file")
                fn()
                lock_fh.write(sentinel)
                lock_fh.flush()
                os.fsync(lock_fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return

    print(f"[mamba3_grouped_head_bwd] local_rank={local_rank} rank={rank_env} waiting for local_rank=0")
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
    raise RuntimeError(f"Timed out waiting for local_rank=0 grouped-head bwd patch. lockfile={lock_path}")


def apply_all() -> None:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        _run_once_with_local_rank_guard(rollback)
        return
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        print(f"  SKIP {_ENV_FLAG} is not set")
        return
    if os.environ.get(_ALLOW_MUTATION_FLAG, "0") not in ("1", "true", "True"):
        raise RuntimeError(
            f"Refusing to mutate installed mamba_ssm without {_ALLOW_MUTATION_FLAG}=1"
        )
    _run_once_with_local_rank_guard(_do_patch, _is_grouped_head_patch_applied)


def apply_if_requested() -> bool:
    if os.environ.get(_ROLLBACK_FLAG, "0") in ("1", "true", "True"):
        apply_all()
        return True
    if os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        return False
    apply_all()
    return True


if __name__ == "__main__":
    try:
        apply_all()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
