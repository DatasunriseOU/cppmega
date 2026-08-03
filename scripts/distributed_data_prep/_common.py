"""Strict shared helpers for distributed source preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
GCS_URI_RE = re.compile(r"^gs://([^/]+)/(.+[^/])$")
MAX_METADATA_BYTES = 16 * 1024 * 1024


class ContractError(RuntimeError):
    """A receipt, manifest, or immutable input violated its contract."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def require_git_object(value: object, *, where: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise ContractError(f"{where} must be a 40- or 64-character Git object id")
    return value


def require_nonempty(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def require_int(value: object, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where} must be an integer >= {minimum}")
    return value


def require_exact_fields(
    value: Mapping[str, object], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{where} fields drifted: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def load_json_object(
    path: Path,
    *,
    where: str,
    max_bytes: int = MAX_METADATA_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where} must be a regular file: {path}")
    if path.stat().st_size > max_bytes:
        raise ContractError(f"{where} exceeds the {max_bytes}-byte bound")
    raw = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{where} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{where} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{where} must contain a JSON object")
    return raw, payload


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_stage = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    stage = Path(raw_stage)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)


def run_checked(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in argv]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        capture_output=capture_output,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "")[-8000:]
        stdout = (completed.stdout or "")[-8000:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command!r}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return completed


def validate_gcs_uri(uri: object, *, where: str) -> str:
    value = require_nonempty(uri, where=where)
    if GCS_URI_RE.fullmatch(value) is None or "#" in value:
        raise ContractError(f"{where} must be a generation-free gs://bucket/object URI")
    if any(part in {"", ".", ".."} for part in value.split("/", 3)[-1].split("/")):
        raise ContractError(f"{where} contains an unsafe object path")
    return value


def gcs_join(prefix: str, *parts: str) -> str:
    base = validate_gcs_uri(prefix.rstrip("/"), where="GCS prefix")
    clean: list[str] = []
    for part in parts:
        if not part or part in {".", ".."} or "/" in part:
            raise ContractError(f"unsafe GCS object component: {part!r}")
        clean.append(part)
    return "/".join((base, *clean))


def iter_jsonl_bytes(path: Path) -> Iterable[bytes]:
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                raise ContractError(
                    f"JSONL line {line_number} is not newline terminated: {path}"
                )
            yield raw
