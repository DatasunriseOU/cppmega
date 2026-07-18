"""Same-filesystem staging helpers for corpus publication."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_output_file(output_path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a sibling staging file and atomically replace output on success."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_stage = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=f".staged{output.suffix}",
        dir=output.parent,
    )
    os.close(fd)
    stage = Path(raw_stage)
    try:
        yield stage
        if not stage.exists():
            raise FileNotFoundError(f"staged output disappeared before publish: {stage}")
        with stage.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(stage, output)
        _fsync_directory(output.parent)
    finally:
        stage.unlink(missing_ok=True)


@contextmanager
def atomic_output_directory(
    output_path: str | os.PathLike[str],
) -> Iterator[Path]:
    """Build a sibling directory and replace the published directory on success."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staged", dir=output.parent)
    )
    backup: Path | None = None
    try:
        yield stage
        if output.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{output.name}.", suffix=".replaced", dir=output.parent
                )
            )
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        _fsync_directory(output.parent)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists() and output.exists():
            shutil.rmtree(backup, ignore_errors=True)


__all__ = ["atomic_output_directory", "atomic_output_file"]
