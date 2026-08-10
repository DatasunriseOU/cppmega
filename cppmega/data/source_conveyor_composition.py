"""Fail-closed composition of revision-bound source conveyor runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_COMPOSITION_PLAN_SCHEMA = "cppmega_source_conveyor_composition_plan_v1"
SOURCE_COMPOSITION_SCHEMA = "cppmega_source_conveyor_composition_v1"
PACKED_SOURCE_INVENTORY_SCHEMA = "cppmega_packed_source_inventory_v1"
GLOBAL_DEDUP_RECEIPT_SCHEMA = "cppmega_global_dedup_store_receipt_v1"
_FULL_LAUNCH_SCHEMA = "cppmega.canonical_source_launch_v1"
_FULL_EXIT_SCHEMA = "cppmega.canonical_source_exit_v1"
_TARGETED_LAUNCH_SCHEMA = "cppmega.canonical_source_targeted_retry_launch_v1"
_TARGETED_EXIT_SCHEMA = "cppmega.canonical_source_targeted_retry_exit_v1"
_EXIT_SALVAGE_SCHEMA = "cppmega.source_exit_salvage_attestation_v1"
_SALVAGED_EXIT_FILENAME = "exit_receipt.salvaged.json"
_PR_COMPLETION_SCHEMA = "cppmega_pr_completion_v2"
_PR_COMPLETION_BINDING_FIELDS = {
    "schema",
    "status",
    "receipt_sha256",
    "pr_store_sha256",
    "repo_list_sha256",
    "expected_repos_sha256",
    "scan_id",
    "expected_repo_count",
    "stored_pr_count",
    "unverified_store_pr_count",
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ARTIFACT_FILENAME_RE = re.compile(
    r"^(?:(?:[a-z0-9_-])|(?:%[0-9a-f]{2}))+\.parquet$"
)
_MAX_PLAN_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
# ponytail: legacy full-source runs embed parse-recovery records in _done.json;
# keep a finite 2 GiB bridge until those records are stored as bound sidecars.
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024 * 1024
_DEDUP_TABLES = frozenset(
    {
        "exact",
        "minhash",
        "lsh",
        "dedup_meta",
        "chunk_claims",
        "dedup_stages",
        "exact_stage",
        "minhash_stage",
        "lsh_stage",
        "chunk_claims_stage",
    }
)
_STAGED_DEDUP_TABLES = frozenset(
    {
        "dedup_stages",
        "exact_stage",
        "minhash_stage",
        "lsh_stage",
        "chunk_claims_stage",
    }
)


@dataclass(frozen=True)
class SourceComposition:
    """Validated source composition and the exact files that prove it."""

    allowlist: dict[tuple[str, int], dict[str, int]]
    receipt: dict[str, object]
    plan_path: Path
    dedup_receipt_path: Path
    run_files: tuple[dict[str, Path], ...]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def source_composition_receipt_sha256(receipt: Mapping[str, object]) -> str:
    """Return the canonical digest used to bind a validated composition."""
    return _canonical_sha256(receipt)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{where} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _require_positive_int(value: object, *, where: str) -> int:
    result = _require_nonnegative_int(value, where=where)
    if result < 1:
        raise ValueError(f"{where} must be positive")
    return result


def _resolve_regular_file(path: Path, *, where: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{where} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _duplicate_rejector(
    where: str,
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    return reject_duplicates


def _load_json_object(path: Path, *, where: str, max_bytes: int) -> tuple[bytes, dict]:
    path = _resolve_regular_file(path, where=where)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{where} exceeds the {max_bytes}-byte metadata bound")
    raw = path.read_bytes()

    try:
        payload = json.loads(raw, object_pairs_hook=_duplicate_rejector(where))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{where} must be a JSON object: {path}")
    return raw, payload


def _load_json_object_streaming(
    path: Path, *, where: str, max_bytes: int
) -> tuple[str, dict[str, Any]]:
    """Hash and parse a large JSON object without retaining its raw bytes."""

    path = _resolve_regular_file(path, where=where)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{where} exceeds the {max_bytes}-byte metadata bound")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
            stream.seek(0)
            payload = json.load(
                stream,
                object_pairs_hook=_duplicate_rejector(where),
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{where} must be a JSON object: {path}")
    return digest.hexdigest(), payload


def _require_mapping(value: object, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be an object")
    return dict(value)


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} fields drifted: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _resolve_bound_path(raw: object, *, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{where} path must be a non-empty string")
    return _resolve_regular_file(Path(raw), where=where)


def _validate_exit_salvage(
    *,
    exit_path: Path,
    exit_receipt: Mapping[str, object],
    exit_code: int,
    done: Mapping[str, object],
    failed: Mapping[str, object],
    run_id: str,
) -> tuple[dict[str, object] | None, Path | None]:
    raw_salvage = exit_receipt.get("salvage")
    if raw_salvage is None:
        if exit_path.name == _SALVAGED_EXIT_FILENAME:
            raise ValueError(f"{run_id} salvaged exit has no attestation")
        return None, None
    if exit_path.name != _SALVAGED_EXIT_FILENAME:
        raise ValueError(f"{run_id} exit salvage must use {_SALVAGED_EXIT_FILENAME}")
    if exit_code == 0:
        raise ValueError(f"{run_id} successful exit must not be salvaged")

    salvage = _require_mapping(raw_salvage, where=f"{run_id} exit salvage")
    _require_exact_fields(
        salvage,
        {"schema", "created_at", "reason", "original_exit_receipt"},
        where=f"{run_id} exit salvage",
    )
    if salvage.get("schema") != _EXIT_SALVAGE_SCHEMA:
        raise ValueError(f"{run_id} exit salvage schema is unsupported")
    created_at = salvage.get("created_at")
    reason = salvage.get("reason")
    if not isinstance(created_at, str) or not created_at or len(created_at) > 128:
        raise ValueError(f"{run_id} exit salvage created_at is invalid")
    if not isinstance(reason, str) or not reason or len(reason) > 4096:
        raise ValueError(f"{run_id} exit salvage reason is invalid")

    binding = _require_mapping(
        salvage.get("original_exit_receipt"),
        where=f"{run_id} original exit receipt binding",
    )
    _require_exact_fields(
        binding,
        {"path", "sha256", "size_bytes"},
        where=f"{run_id} original exit receipt binding",
    )
    original_path = _resolve_bound_path(
        binding.get("path"), where=f"{run_id} original exit receipt"
    )
    if (
        original_path.parent != exit_path.parent.resolve()
        or original_path.name != "exit_receipt.json"
    ):
        raise ValueError(f"{run_id} exit salvage binds a noncanonical original path")
    original_raw, original = _load_json_object(
        original_path,
        where=f"{run_id} original exit receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    original_size = _require_positive_int(
        binding.get("size_bytes"),
        where=f"{run_id} original exit receipt size_bytes",
    )
    original_sha256 = _require_sha256(
        binding.get("sha256"),
        where=f"{run_id} original exit receipt sha256",
    )
    if (
        len(original_raw) != original_size
        or hashlib.sha256(original_raw).hexdigest() != original_sha256
    ):
        raise ValueError(f"{run_id} original exit receipt binding drifted")
    if (
        _require_nonnegative_int(
            original.get("exit_code"), where=f"{run_id} original exit_code"
        )
        != exit_code
        or original.get("ts") != exit_receipt.get("finished_at")
    ):
        raise ValueError(f"{run_id} salvaged exit projection drifted")

    for name, units in (("done", done), ("failed", failed)):
        raw_units = original.get(f"{name}_units")
        if (
            not isinstance(raw_units, list)
            or any(not isinstance(unit, str) for unit in raw_units)
            or len(raw_units) != len(set(raw_units))
            or sorted(raw_units) != sorted(units)
            or _require_nonnegative_int(
                original.get(f"{name}_count"),
                where=f"{run_id} original {name}_count",
            )
            != len(units)
        ):
            raise ValueError(f"{run_id} original {name} projection drifted")

    return (
        {
            "schema": _EXIT_SALVAGE_SCHEMA,
            "created_at": created_at,
            "reason": reason,
            "original_exit_receipt_sha256": original_sha256,
            "original_exit_receipt_size_bytes": original_size,
        },
        original_path,
    )


def _single_option(command: Sequence[object], name: str) -> str:
    indexes = [index for index, value in enumerate(command) if value == name]
    if len(indexes) != 1:
        raise ValueError(f"source launch command must contain exactly one {name}")
    index = indexes[0]
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError(f"source launch command has no value for {name}")
    return str(command[index + 1])


def _repeated_option(command: Sequence[object], name: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command):
        if value != name:
            continue
        if index + 1 >= len(command) or not isinstance(command[index + 1], str):
            raise ValueError(f"source launch command has no value for {name}")
        values.append(str(command[index + 1]))
    return values


def _revision_binding(value: object, *, where: str) -> dict[str, object]:
    revision = _require_mapping(value, where=where)
    if (
        _require_positive_int(
            revision.get("schema_version"), where=f"{where}.schema_version"
        )
        < 2
        or revision.get("dirty") is not False
        or revision.get("producer_role") != "canonical_source_conveyor"
        or revision.get("repository_identity") != "cppmega"
    ):
        raise ValueError(f"{where} is not a clean canonical cppmega revision")
    commit = revision.get("git_commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError(f"{where}.git_commit is invalid")
    tree_sha256 = _require_sha256(
        revision.get("source_tree_sha256"), where=f"{where}.source_tree_sha256"
    )
    indexer = _require_mapping(
        revision.get("indexer_provenance"), where=f"{where}.indexer_provenance"
    )
    if indexer.get("schema") != "cppmega_indexer_dependency_binding_v1":
        raise ValueError(f"{where} indexer provenance schema is unsupported")
    source_sha256 = _require_sha256(
        indexer.get("source_sha256"), where=f"{where}.indexer.source_sha256"
    )
    closure_sha256 = _require_sha256(
        indexer.get("dependency_closure_sha256"),
        where=f"{where}.indexer.dependency_closure_sha256",
    )
    if revision.get("indexer_dependency_closure_sha256") != closure_sha256:
        raise ValueError(f"{where} indexer dependency closure drifted")
    return {
        "cppmega": {
            "commit": commit,
            "tree_sha256": tree_sha256,
        },
        "clang_indexer": {
            "source_sha256": source_sha256,
            "dependency_closure_sha256": closure_sha256,
        },
    }


def _code_run_identity(value: object, *, where: str) -> dict[str, str]:
    identity = _require_mapping(value, where=where)
    return {
        name: _require_sha256(identity.get(name), where=f"{where}.{name}")
        for name in ("launch_sha256", "exit_sha256", "manifest_sha256")
    }


def _code_repair_root_binding(value: object, *, where: str) -> dict[str, list[str]]:
    binding = _require_mapping(value, where=where)
    _require_exact_fields(
        binding,
        {"launched", "pending", "ordered"},
        where=where,
    )
    roots: dict[str, list[str]] = {}
    for state in ("launched", "pending", "ordered"):
        raw_paths = binding[state]
        if not isinstance(raw_paths, list):
            raise TypeError(f"{where}.{state} must be a list")
        resolved_paths: list[str] = []
        for index, raw_path in enumerate(raw_paths):
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"{where}.{state}[{index}] must be a non-empty path"
                )
            path = Path(raw_path).expanduser()
            if path.is_symlink():
                raise ValueError(
                    f"{where}.{state}[{index}] must not be a symlink: {path}"
                )
            try:
                path = path.resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    f"{where}.{state}[{index}] does not exist: {path}"
                ) from exc
            if not path.is_dir():
                raise ValueError(
                    f"{where}.{state}[{index}] is not a directory: {path}"
                )
            resolved_paths.append(str(path))
        if len(resolved_paths) != len(set(resolved_paths)):
            raise ValueError(f"{where}.{state} contains duplicate paths")
        roots[state] = resolved_paths
    if set(roots["launched"]) & set(roots["pending"]):
        raise ValueError(f"{where} launched and pending roots overlap")
    if roots["ordered"] != [*roots["launched"], *roots["pending"]]:
        raise ValueError(f"{where}.ordered does not preserve root state order")
    return roots


def _unit_repo(unit: str) -> str:
    repo, separator, suffix = unit.rpartition("::")
    if not separator or not repo or not suffix:
        raise ValueError(f"invalid conveyor unit key: {unit!r}")
    return repo


def _terminal_repositories(
    *,
    done: Mapping[str, object],
    failed: Mapping[str, object],
    stream: str,
) -> set[str]:
    if stream == "code":
        successful = {
            _unit_repo(unit) for unit in done if str(unit).endswith("::code")
        }
    else:
        successful = {
            _unit_repo(unit)
            for unit in done
            if str(unit).endswith(("::commits", "::no_git"))
        }
    return successful | {_unit_repo(str(unit)) for unit in failed}


def _manifest_allowlist(
    *,
    manifest: Mapping[str, object],
    buckets: tuple[int, ...],
    run_id: str,
) -> dict[tuple[str, int], dict[str, int]]:
    done = _require_mapping(manifest.get("done"), where=f"{run_id} done")
    allowed: dict[tuple[str, int], dict[str, int]] = {
        (kind, bucket): {} for kind in ("code", "commits") for bucket in buckets
    }
    for unit, raw_info in done.items():
        if not isinstance(unit, str) or not isinstance(raw_info, Mapping):
            continue
        info = dict(raw_info)
        lengths = info.get("lengths")
        if not isinstance(lengths, Mapping) or not lengths:
            continue
        if unit.endswith("::code"):
            kind = "code"
        elif "::r" in unit:
            repo, raw_start = unit.rsplit("::r", 1)
            if not repo or not raw_start.isdecimal():
                raise ValueError(f"{run_id} has an invalid commit range unit {unit!r}")
            kind = "commits"
        else:
            continue
        filename = info.get("artifact_filename")
        if (
            not isinstance(filename, str)
            or _ARTIFACT_FILENAME_RE.fullmatch(filename) is None
        ):
            raise ValueError(
                f"{run_id} unit {unit} has no canonical artifact_filename"
            )
        unknown_lengths = sorted(set(lengths) - {str(bucket) for bucket in buckets})
        if unknown_lengths:
            raise ValueError(
                f"{run_id} unit {unit} has unexpected buckets {unknown_lengths}"
            )
        for bucket in buckets:
            raw_length = lengths.get(str(bucket))
            if raw_length is None:
                continue
            length = _require_mapping(
                raw_length, where=f"{run_id} {unit}/{bucket}"
            )
            rows = _require_positive_int(
                length.get("rows"), where=f"{run_id} {unit}/{bucket}.rows"
            )
            if filename in allowed[(kind, bucket)]:
                raise ValueError(
                    f"{run_id} maps multiple units to source shard "
                    f"{kind}/{bucket}/{filename}"
                )
            allowed[(kind, bucket)][filename] = rows
    return allowed


def _validate_input_artifact(
    inputs: Mapping[str, object],
    name: str,
    *,
    run_id: str,
    max_bytes: int | None = None,
    repository_root: Path | None = None,
    recorded_revision: str | None = None,
    cache_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    binding = _require_mapping(inputs.get(name), where=f"{run_id} {name}")
    expected_sha256 = _require_sha256(
        binding.get("sha256"), where=f"{run_id} {name}.sha256"
    )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{run_id} {name} path must be a non-empty string")
    try:
        path = _resolve_bound_path(raw_path, where=f"{run_id} {name}")
    except FileNotFoundError:
        path = None
    if path is not None:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise ValueError(f"{run_id} {name} exceeds the metadata size bound")
        if _sha256(path) == expected_sha256:
            return binding, path
    if name == "source_quarantine_manifest" and all(
        value is not None
        for value in (repository_root, recorded_revision, cache_root)
    ):
        if max_bytes is None:
            raise ValueError(
                f"{run_id} {name} historical recovery requires a metadata size bound"
            )
        assert repository_root is not None
        assert recorded_revision is not None
        assert cache_root is not None
        path = _resolve_recorded_repository_artifact(
            recorded_path=Path(raw_path),
            expected_sha256=expected_sha256,
            repository_root=repository_root,
            recorded_revision=recorded_revision,
            cache_root=cache_root,
            label=f"{run_id} {name}",
            max_bytes=max_bytes,
        )
    else:
        raise ValueError(f"{run_id} {name} artifact binding drifted")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ValueError(f"{run_id} {name} exceeds the metadata size bound")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{run_id} {name} artifact binding drifted")
    return binding, path


def _resolve_recorded_repository_artifact(
    *,
    recorded_path: Path,
    expected_sha256: str,
    repository_root: Path,
    recorded_revision: str,
    cache_root: Path,
    label: str,
    max_bytes: int,
) -> Path:
    """Resolve a drifted tracked input from its recorded Git revision."""

    expected_sha256 = _require_sha256(expected_sha256, where=f"{label}.sha256")
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
    ):
        raise ValueError(f"{label} metadata bound is invalid")
    if _COMMIT_RE.fullmatch(recorded_revision) is None:
        raise ValueError(f"{label} recorded revision is not an exact Git commit")
    repository_root = repository_root.expanduser()
    if repository_root.is_symlink():
        raise ValueError(f"{label} repository root must not be a symlink")
    repository_root = repository_root.resolve(strict=True)
    recorded_path = recorded_path.expanduser()
    if recorded_path.is_symlink():
        raise ValueError(f"{label} recorded path must not be a symlink")
    recorded_path = recorded_path.resolve(strict=False)
    try:
        relative_path = recorded_path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} is not repository-owned") from exc
    if relative_path != Path("configs/source_quarantine_manifest.json"):
        raise ValueError(f"{label} is not the canonical quarantine manifest")
    if recorded_path.is_file():
        recorded_size = recorded_path.stat().st_size
        if recorded_size <= max_bytes and _sha256(recorded_path) == expected_sha256:
            return recorded_path

    cache_root = _validate_private_cache_root(cache_root, label=label)
    target = cache_root / f"source_quarantine_manifest.{expected_sha256}.json"
    cached_sha256 = _sha256_private_cache_artifact(
        target,
        label=label,
        max_bytes=max_bytes,
    )
    if cached_sha256 is not None:
        if cached_sha256 != expected_sha256:
            raise ValueError(f"{label} historical cache artifact drifted")
        return target

    object_spec = f"{recorded_revision}:{relative_path.as_posix()}"
    try:
        size_result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "cat-file",
                "-s",
                object_spec,
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{label} historical Git blob lookup timed out") from exc
    if size_result.returncode != 0:
        raise ValueError(f"{label} historical Git blob is unavailable")
    try:
        blob_size = int(size_result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"{label} historical Git blob size is invalid") from exc
    if blob_size < 0 or blob_size > max_bytes:
        raise ValueError(f"{label} historical Git blob exceeds the metadata bound")

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "cat-file",
                "blob",
                object_spec,
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{label} historical Git blob lookup timed out") from exc
    if completed.returncode != 0:
        raise ValueError(f"{label} historical Git blob is unavailable")
    payload = completed.stdout
    if len(payload) > max_bytes:
        raise ValueError(f"{label} historical Git blob exceeds the metadata bound")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} historical Git blob does not match its binding")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_root,
        prefix=f".{expected_sha256}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(cache_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if (
        _sha256_private_cache_artifact(
            target,
            label=label,
            max_bytes=max_bytes,
        )
        != expected_sha256
    ):
        raise ValueError(f"{label} historical cache write drifted")
    return target


def _validate_private_cache_root(cache_root: Path, *, label: str) -> Path:
    """Create or validate one current-user-only historical input cache."""

    cache_root = cache_root.expanduser()
    try:
        cache_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    try:
        metadata = os.lstat(cache_root)
    except OSError as exc:
        raise ValueError(f"{label} cache root is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} cache root must be a directory")
    if metadata.st_uid != os.geteuid():
        raise ValueError(f"{label} cache root must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError(f"{label} cache root permissions must be 0700")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(cache_root, flags)
    except OSError as exc:
        raise ValueError(f"{label} cache root is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError(f"{label} cache root changed during validation")
    finally:
        os.close(descriptor)
    return cache_root.resolve(strict=True)


def _sha256_private_cache_artifact(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> str | None:
    """Hash a bounded regular cache file without following links."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{label} historical cache artifact drifted") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} historical cache artifact drifted")
        if metadata.st_uid != os.geteuid():
            raise ValueError(
                f"{label} historical cache artifact must be owned by the current user"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError(
                f"{label} historical cache artifact permissions must be 0600"
            )
        if metadata.st_size > max_bytes:
            raise ValueError(
                f"{label} historical cache artifact exceeds the metadata bound"
            )
        digest = hashlib.sha256()
        hashed_bytes = 0
        while payload := os.read(
            descriptor,
            min(1024 * 1024, max_bytes - hashed_bytes + 1),
        ):
            hashed_bytes += len(payload)
            if hashed_bytes > max_bytes:
                raise ValueError(
                    f"{label} historical cache artifact exceeds the metadata bound"
                )
            digest.update(payload)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _historical_input_cache_root() -> Path:
    """Return a private cache outside immutable source-run evidence."""

    return Path(tempfile.gettempdir()) / (
        f"cppmega-source-composition-input-cache-{os.geteuid()}"
    )


def _validate_pr_provenance(
    *,
    launch: Mapping[str, object],
    manifest: Mapping[str, object],
    command: Sequence[object],
    run_id: str,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Verify the immutable PR corpus used by one commit conveyor run."""

    pr_inputs = _require_mapping(
        launch.get("pr_inputs"), where=f"{run_id} PR inputs"
    )
    repo_binding = _require_mapping(
        pr_inputs.get("repo_list"), where=f"{run_id} PR repo list"
    )
    completion_artifact = _require_mapping(
        pr_inputs.get("completion"), where=f"{run_id} PR completion artifact"
    )
    store_binding = _require_mapping(
        pr_inputs.get("store"), where=f"{run_id} PR store"
    )
    completion_binding = _require_mapping(
        pr_inputs.get("completion_binding"),
        where=f"{run_id} PR completion binding",
    )
    _require_exact_fields(
        completion_binding,
        _PR_COMPLETION_BINDING_FIELDS,
        where=f"{run_id} PR completion binding",
    )

    repo_path = _resolve_bound_path(
        repo_binding.get("path"), where=f"{run_id} PR repo list"
    )
    completion_path = _resolve_bound_path(
        completion_artifact.get("path"), where=f"{run_id} PR completion"
    )
    store_path = _resolve_bound_path(
        store_binding.get("path"), where=f"{run_id} PR store"
    )
    if (
        Path(_single_option(command, "--pr-repo-list")).expanduser().resolve()
        != repo_path
        or Path(_single_option(command, "--pr-completion-receipt"))
        .expanduser()
        .resolve()
        != completion_path
        or Path(_single_option(command, "--pr-store")).expanduser().resolve()
        != store_path
    ):
        raise ValueError(f"{run_id} PR command paths drifted")

    repo_sha256 = _require_sha256(
        repo_binding.get("sha256"), where=f"{run_id} PR repo list.sha256"
    )
    completion_sha256 = _require_sha256(
        completion_artifact.get("sha256"),
        where=f"{run_id} PR completion.sha256",
    )
    store_sha256 = _require_sha256(
        store_binding.get("sha256"), where=f"{run_id} PR store.sha256"
    )
    if _sha256(repo_path) != repo_sha256:
        raise ValueError(f"{run_id} PR repo list artifact binding drifted")
    completion_raw, completion = _load_json_object(
        completion_path,
        where=f"{run_id} PR completion",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    if hashlib.sha256(completion_raw).hexdigest() != completion_sha256:
        raise ValueError(f"{run_id} PR completion artifact binding drifted")

    store_stat = store_path.stat()
    store_identity = (
        store_stat.st_dev,
        store_stat.st_ino,
        store_stat.st_size,
        store_stat.st_mtime_ns,
        store_stat.st_ctime_ns,
    )
    for field, actual in (
        ("device", store_stat.st_dev),
        ("inode", store_stat.st_ino),
        ("size_bytes", store_stat.st_size),
    ):
        if _require_positive_int(
            store_binding.get(field), where=f"{run_id} PR store.{field}"
        ) != actual:
            raise ValueError(f"{run_id} PR store identity drifted")
    if store_binding.get("quick_check") != "ok":
        raise ValueError(f"{run_id} PR store lacks a green quick_check")
    wal_path = Path(f"{store_path}-wal")
    if wal_path.exists() and wal_path.stat().st_size:
        raise ValueError(f"{run_id} PR store has an uncheckpointed WAL")
    if _sha256(store_path) != store_sha256:
        raise ValueError(f"{run_id} PR store artifact binding drifted")
    store_stat_after = store_path.stat()
    if store_identity != (
        store_stat_after.st_dev,
        store_stat_after.st_ino,
        store_stat_after.st_size,
        store_stat_after.st_mtime_ns,
        store_stat_after.st_ctime_ns,
    ):
        raise ValueError(f"{run_id} PR store changed while hashing")
    if wal_path.exists() and wal_path.stat().st_size:
        raise ValueError(f"{run_id} PR store WAL appeared while hashing")

    if (
        completion.get("schema") != _PR_COMPLETION_SCHEMA
        or completion.get("status") != "verified"
    ):
        raise ValueError(f"{run_id} PR completion is not verified")
    receipt_repo = _require_mapping(
        completion.get("repo_list"), where=f"{run_id} PR receipt repo list"
    )
    receipt_store = _require_mapping(
        completion.get("pr_store"), where=f"{run_id} PR receipt store"
    )
    if (
        _resolve_bound_path(
            receipt_repo.get("path"), where=f"{run_id} PR receipt repo list"
        )
        != repo_path
        or _require_sha256(
            receipt_repo.get("sha256"),
            where=f"{run_id} PR receipt repo list.sha256",
        )
        != repo_sha256
        or _resolve_bound_path(
            receipt_store.get("path"), where=f"{run_id} PR receipt store"
        )
        != store_path
        or _require_sha256(
            receipt_store.get("sha256"),
            where=f"{run_id} PR receipt store.sha256",
        )
        != store_sha256
        or _require_positive_int(
            receipt_store.get("size"), where=f"{run_id} PR receipt store.size"
        )
        != store_stat.st_size
    ):
        raise ValueError(f"{run_id} PR completion input bindings drifted")

    manifest_binding = _require_mapping(
        manifest.get("pr_completion"), where=f"{run_id} manifest PR completion"
    )
    if manifest.get("pr_completion_reverified_at_finish") is not True:
        raise ValueError(f"{run_id} PR completion was not reverified at finish")
    if manifest_binding != completion_binding:
        raise ValueError(f"{run_id} manifest PR completion binding drifted")
    if (
        completion_binding.get("schema") != _PR_COMPLETION_SCHEMA
        or completion_binding.get("status") != "verified"
        or completion_binding.get("receipt_sha256") != completion_sha256
        or completion_binding.get("pr_store_sha256") != store_sha256
        or completion_binding.get("repo_list_sha256") != repo_sha256
    ):
        raise ValueError(f"{run_id} normalized PR completion binding drifted")
    for field, minimum in (
        ("expected_repo_count", 1),
        ("stored_pr_count", 0),
        ("unverified_store_pr_count", 0),
    ):
        value = completion_binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{run_id} PR completion {field} is invalid")
    if completion_binding["stored_pr_count"] != _require_nonnegative_int(
        store_binding.get("pr_rows"), where=f"{run_id} PR store.pr_rows"
    ):
        raise ValueError(f"{run_id} PR store row count drifted")
    for field in ("expected_repos_sha256", "scan_id"):
        _require_sha256(
            completion_binding.get(field),
            where=f"{run_id} PR completion {field}",
        )
    for field in (
        "expected_repos_sha256",
        "scan_id",
        "expected_repo_count",
        "stored_pr_count",
        "unverified_store_pr_count",
    ):
        if completion_binding[field] != completion.get(field):
            raise ValueError(f"{run_id} PR completion {field} drifted")

    return (
        dict(completion_binding),
        {
            "pr_completion": completion_path,
            "pr_repo_list": repo_path,
        },
    )


def _archive_identity(
    archive: Mapping[str, object],
    *,
    run_id: str,
) -> tuple[dict[str, object], str]:
    resolved_path = archive.get("resolved_path")
    if not isinstance(resolved_path, str) or not resolved_path:
        raise ValueError(f"{run_id} archive.resolved_path is invalid")
    identity: dict[str, object] = {
        "resolved_path": resolved_path,
        "sha256": _require_sha256(
            archive.get("sha256"), where=f"{run_id} archive.sha256"
        ),
        "size_bytes": _require_positive_int(
            archive.get("size_bytes"), where=f"{run_id} archive.size_bytes"
        ),
        "mtime_epoch": _require_positive_int(
            archive.get("mtime_epoch"), where=f"{run_id} archive.mtime_epoch"
        ),
        "inode": _require_positive_int(
            archive.get("inode"), where=f"{run_id} archive.inode"
        ),
        "device": _require_positive_int(
            archive.get("device"), where=f"{run_id} archive.device"
        ),
    }
    return identity, _canonical_sha256(identity)


def _validate_dedup_receipt(path: Path) -> tuple[dict[str, object], Path]:
    raw, receipt = _load_json_object(
        path, where="global dedup receipt", max_bytes=_MAX_RECEIPT_BYTES
    )
    _require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "created_at",
            "database",
            "checkpoint",
            "integrity_check",
            "sqlite_schema_sha256",
            "logical_hash_algorithm",
            "logical_sha256",
            "tables",
            "policy",
            "verifier",
        },
        where="global dedup receipt",
    )
    if (
        receipt.get("schema") != GLOBAL_DEDUP_RECEIPT_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("integrity_check") != "ok"
    ):
        raise ValueError("global dedup receipt is not verified")
    created_at = receipt.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("global dedup receipt has no creation timestamp")
    database = _require_mapping(receipt.get("database"), where="dedup database")
    _require_exact_fields(
        database, {"path", "size_bytes", "sha256"}, where="dedup database"
    )
    database_path = _resolve_bound_path(
        database.get("path"), where="dedup database"
    )
    size = _require_positive_int(
        database.get("size_bytes"), where="dedup database.size_bytes"
    )
    digest = _require_sha256(
        database.get("sha256"), where="dedup database.sha256"
    )
    if database_path.stat().st_size != size or _sha256(database_path) != digest:
        raise ValueError("global dedup database artifact binding drifted")
    _require_sha256(
        receipt.get("sqlite_schema_sha256"), where="dedup sqlite_schema_sha256"
    )
    if receipt.get("logical_hash_algorithm") != "cppmega_sqlite_rows_lenprefixed_v1":
        raise ValueError("global dedup logical hash algorithm is unsupported")
    _require_sha256(receipt.get("logical_sha256"), where="dedup logical_sha256")
    checkpoint = _require_mapping(receipt.get("checkpoint"), where="dedup checkpoint")
    _require_exact_fields(
        checkpoint,
        {
            "mode",
            "busy",
            "log_frames",
            "checkpointed_frames",
            "wal_size_bytes",
        },
        where="dedup checkpoint",
    )
    if checkpoint != {
        "mode": "TRUNCATE",
        "busy": 0,
        "log_frames": 0,
        "checkpointed_frames": 0,
        "wal_size_bytes": 0,
    }:
        raise ValueError("global dedup WAL is not fully checkpointed")
    tables = _require_mapping(receipt.get("tables"), where="dedup tables")
    if set(tables) != _DEDUP_TABLES:
        raise ValueError("global dedup table inventory drifted")
    for name, raw_table in tables.items():
        table = _require_mapping(raw_table, where=f"dedup table {name}")
        _require_exact_fields(
            table, {"rows", "logical_sha256"}, where=f"dedup table {name}"
        )
        rows = _require_nonnegative_int(
            table.get("rows"), where=f"dedup table {name}.rows"
        )
        _require_sha256(
            table.get("logical_sha256"),
            where=f"dedup table {name}.logical_sha256",
        )
        if name in _STAGED_DEDUP_TABLES and rows:
            raise ValueError(f"global dedup table {name} contains unpromoted rows")
        if name not in _STAGED_DEDUP_TABLES and rows < 1:
            raise ValueError(f"global dedup production table {name} is empty")
    policy = _require_mapping(receipt.get("policy"), where="dedup policy")
    _require_exact_fields(
        policy, {"exact", "chunk", "near"}, where="dedup policy"
    )
    near = _require_mapping(policy.get("near"), where="dedup near policy")
    if (
        policy.get("exact") != "sha1_token_ids_v1"
        or policy.get("chunk") != "tokenized_chunk_claims_v1"
        or near
        != {
            "enabled": True,
            "threshold": 0.7,
            "num_perm": 256,
            "shingle_k": 5,
        }
    ):
        raise ValueError("global dedup policy is not the production exact+near policy")
    verifier = _require_mapping(receipt.get("verifier"), where="dedup verifier")
    _require_exact_fields(
        verifier,
        {"repository_identity", "script", "script_sha256"},
        where="dedup verifier",
    )
    if (
        verifier.get("repository_identity") != "cppmega"
        or verifier.get("script") != "scripts/data/verify_global_dedup_store.py"
    ):
        raise ValueError("global dedup verifier identity drifted")
    _require_sha256(verifier.get("script_sha256"), where="dedup verifier.script_sha256")
    portable = dict(receipt)
    portable["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    portable_database = dict(database)
    portable_database.pop("path")
    portable["database"] = portable_database
    return portable, database_path


def _load_run(
    raw_run: object,
    *,
    buckets: tuple[int, ...],
    code_root: Path,
    commit_root: Path,
) -> tuple[
    dict[str, object],
    dict[tuple[str, int], dict[str, int]],
    dict[str, Path],
    set[str],
    set[str],
    set[str],
    set[str],
    str,
    str,
    dict[str, object],
]:
    run = _require_mapping(raw_run, where="source composition run")
    _require_exact_fields(
        run,
        {"run_id", "launch_receipt", "exit_receipt", "manifest"},
        where="source composition run",
    )
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("source composition run_id is invalid")
    launch_path = _resolve_bound_path(
        run.get("launch_receipt"), where=f"{run_id} launch receipt"
    )
    exit_path = _resolve_bound_path(
        run.get("exit_receipt"), where=f"{run_id} exit receipt"
    )
    manifest_path = _resolve_bound_path(
        run.get("manifest"), where=f"{run_id} conveyor manifest"
    )
    launch_raw, launch = _load_json_object(
        launch_path, where=f"{run_id} launch receipt", max_bytes=_MAX_RECEIPT_BYTES
    )
    exit_raw, exit_receipt = _load_json_object(
        exit_path, where=f"{run_id} exit receipt", max_bytes=_MAX_RECEIPT_BYTES
    )
    manifest_sha256, manifest = _load_json_object_streaming(
        manifest_path,
        where=f"{run_id} conveyor manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )

    launch_schema = launch.get("schema")
    expected_exit_schema = {
        _FULL_LAUNCH_SCHEMA: _FULL_EXIT_SCHEMA,
        _TARGETED_LAUNCH_SCHEMA: _TARGETED_EXIT_SCHEMA,
    }.get(launch_schema)
    if expected_exit_schema is None or exit_receipt.get("schema") != expected_exit_schema:
        raise ValueError(f"{run_id} source supervisor schemas are unsupported")
    if launch.get("status") != "running":
        raise ValueError(f"{run_id} launch receipt is not the executed receipt")
    exit_code = _require_nonnegative_int(
        exit_receipt.get("exit_code"), where=f"{run_id} exit_code"
    )
    expected_status = "success" if exit_code == 0 else "failed"
    if exit_receipt.get("status") != expected_status:
        raise ValueError(f"{run_id} exit status does not match its exit code")
    if exit_receipt.get("launch_receipt_sha256") != hashlib.sha256(
        launch_raw
    ).hexdigest():
        raise ValueError(f"{run_id} exit receipt does not bind the launch receipt")
    done_binding = _require_mapping(
        exit_receipt.get("done_manifest"), where=f"{run_id} exit done_manifest"
    )
    bound_manifest_path = _resolve_bound_path(
        done_binding.get("path"), where=f"{run_id} exit done_manifest"
    )
    if (
        bound_manifest_path != manifest_path
        or _require_sha256(
            done_binding.get("sha256"),
            where=f"{run_id} exit done_manifest.sha256",
        )
        != manifest_sha256
    ):
        raise ValueError(f"{run_id} exit receipt does not bind the conveyor manifest")

    if launch.get("repository_identity") != "cppmega":
        raise ValueError(f"{run_id} launch receipt is not bound to cppmega")
    source_code_runs: list[dict[str, str]] | None = None
    raw_source_code_runs = launch.get("source_code_runs")
    if raw_source_code_runs is not None:
        if not isinstance(raw_source_code_runs, list) or not raw_source_code_runs:
            raise ValueError(f"{run_id} source code run identities are invalid")
        source_code_runs = [
            _code_run_identity(value, where=f"{run_id} source code run {index}")
            for index, value in enumerate(raw_source_code_runs)
        ]
    if "source_code_run" in launch:
        source_code_run = _code_run_identity(
            launch.get("source_code_run"), where=f"{run_id} source code run"
        )
        if source_code_runs is None:
            source_code_runs = [source_code_run]
        elif source_code_run != source_code_runs[0]:
            raise ValueError(f"{run_id} source code run bindings drifted")
    source_code_repair_roots: dict[str, list[str]] | None = None
    raw_source_code_repair_roots = launch.get("source_code_repair_roots")
    if raw_source_code_repair_roots is not None:
        source_code_repair_roots = _code_repair_root_binding(
            raw_source_code_repair_roots,
            where=f"{run_id} source code repair roots",
        )
        run_binding = _require_mapping(
            launch.get("run_binding"), where=f"{run_id} run binding"
        )
        run_binding_sha256 = _require_sha256(
            launch.get("run_binding_sha256"),
            where=f"{run_id} run binding sha256",
        )
        if _canonical_sha256(run_binding) != run_binding_sha256:
            raise ValueError(f"{run_id} run binding digest drifted")
        if run_binding.get("source_code_repair_roots") != raw_source_code_repair_roots:
            raise ValueError(f"{run_id} source code repair root bindings drifted")
    command = launch.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"{run_id} launch command is missing")
    streams = _single_option(command, "--streams")
    if streams not in {"code", "commits", "both"}:
        raise ValueError(f"{run_id} launch streams are invalid")
    if source_code_repair_roots is not None and streams != "commits":
        raise ValueError(
            f"{run_id} code repair root binding is only valid for commits"
        )
    if "--no-near-dedup" in command:
        raise ValueError(f"{run_id} launch explicitly disabled near dedup")
    expected_revision = _single_option(command, "--expected-code-revision")
    if (
        launch.get("code_revision") != expected_revision
        or exit_receipt.get("code_revision") != expected_revision
    ):
        raise ValueError(f"{run_id} source revision binding drifted")
    revision = _revision_binding(
        manifest.get("code_revision"), where=f"{run_id} manifest code revision"
    )
    if revision["cppmega"]["commit"] != expected_revision:
        raise ValueError(f"{run_id} manifest revision does not match its launch")

    target_lengths = launch.get("target_lengths")
    if target_lengths != list(buckets):
        raise ValueError(f"{run_id} target length ladder drifted")
    for option in (
        "--target-lengths-code",
        "--target-lengths-commits",
    ):
        if option in command:
            parsed = [int(value) for value in _single_option(command, option).split(",")]
            if parsed != list(buckets):
                raise ValueError(f"{run_id} {option} drifted")
    outputs = _require_mapping(launch.get("outputs"), where=f"{run_id} outputs")
    if streams in {"code", "both"}:
        raw_code_root = outputs.get("code_output_root")
        if (
            not isinstance(raw_code_root, str)
            or Path(raw_code_root).expanduser().resolve() != code_root
        ):
            raise ValueError(f"{run_id} code output root drifted")
    if streams in {"commits", "both"}:
        raw_commit_root = outputs.get("commit_output_root")
        if (
            not isinstance(raw_commit_root, str)
            or Path(raw_commit_root).expanduser().resolve() != commit_root
        ):
            raise ValueError(f"{run_id} commit output root drifted")
    raw_dedup_path = outputs.get("dedup_db")
    if not isinstance(raw_dedup_path, str) or not raw_dedup_path:
        raise ValueError(f"{run_id} dedup output path is missing")
    dedup_path = Path(raw_dedup_path).expanduser().resolve()
    if Path(_single_option(command, "--dedup-db")).expanduser().resolve() != dedup_path:
        raise ValueError(f"{run_id} dedup path drifted")

    inputs = _require_mapping(launch.get("inputs"), where=f"{run_id} inputs")
    archive = _require_mapping(inputs.get("archive"), where=f"{run_id} archive")
    archive_fields, archive_identity = _archive_identity(
        archive,
        run_id=run_id,
    )
    if "--source-archive" in command:
        command_archive = Path(
            _single_option(command, "--source-archive")
        ).expanduser().resolve()
        receipt_archive = Path(
            str(archive_fields["resolved_path"])
        ).expanduser().resolve()
        if command_archive != receipt_archive:
            raise ValueError(f"{run_id} source archive command path drifted")
    raw_repository_root = launch.get("repository_root")
    repository_root = (
        Path(raw_repository_root)
        if isinstance(raw_repository_root, str) and raw_repository_root
        else None
    )
    input_artifacts: dict[str, tuple[dict[str, Any], Path]] = {
        name: _validate_input_artifact(
            inputs,
            name,
            run_id=run_id,
            max_bytes=(
                _MAX_RECEIPT_BYTES
                if name.endswith("_receipt")
                or name == "source_quarantine_manifest"
                else None
            ),
            repository_root=repository_root,
            recorded_revision=expected_revision,
            cache_root=_historical_input_cache_root(),
        )
        for name in (
            "archive_sha256_receipt",
            "archive_inventory_receipt",
            "repo_list",
            "source_quarantine_manifest",
            "tokenizer",
        )
    }
    input_binding_hashes = {
        name: _require_sha256(
            binding.get("sha256"), where=f"{run_id} {name}.sha256"
        )
        for name, (binding, _path) in input_artifacts.items()
    }
    input_binding = _canonical_sha256(
        {
            name: digest
            for name, digest in input_binding_hashes.items()
            if name != "source_quarantine_manifest"
        }
    )
    if (
        Path(_single_option(command, "--repo-list")).expanduser().resolve()
        != input_artifacts["repo_list"][1]
        or Path(_single_option(command, "--source-quarantine-manifest"))
        .expanduser()
        .resolve()
        != Path(str(input_artifacts["source_quarantine_manifest"][0]["path"]))
        .expanduser()
        .resolve()
    ):
        raise ValueError(f"{run_id} launch command input paths drifted")

    archive_receipt_path = input_artifacts["archive_sha256_receipt"][1]
    _archive_receipt_raw, archive_receipt = _load_json_object(
        archive_receipt_path,
        where=f"{run_id} archive SHA-256 receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    if (
        archive_receipt.get("schema")
        != "cppmega.source_archive_sha256_verification_v1"
        or archive_receipt.get("status") != "verified"
        or archive_receipt.get("exit_code") != 0
        or any(
            archive_receipt.get(name) != value
            for name, value in archive_fields.items()
        )
    ):
        raise ValueError(f"{run_id} archive SHA-256 receipt identity drifted")

    inventory_binding, inventory_path = input_artifacts[
        "archive_inventory_receipt"
    ]
    inventory_raw, inventory = _load_json_object(
        inventory_path,
        where=f"{run_id} archive inventory",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    if _require_sha256(
        inventory_binding.get("sha256"),
        where=f"{run_id} archive inventory receipt.sha256",
    ) != hashlib.sha256(inventory_raw).hexdigest():
        raise ValueError(f"{run_id} archive inventory artifact binding drifted")
    if (
        inventory.get("schema") != "cppmega.source_archive_inventory_binding_v1"
        or inventory.get("status") != "verified"
    ):
        raise ValueError(f"{run_id} archive inventory is not verified")
    inventory_archive_receipt = _require_mapping(
        inventory.get("archive_sha256_receipt"),
        where=f"{run_id} inventory archive SHA-256 receipt",
    )
    inventory_repo_list = _require_mapping(
        inventory.get("canonical_repo_list"),
        where=f"{run_id} inventory canonical repo list",
    )
    for label, binding in (
        ("archive SHA-256 receipt", inventory_archive_receipt),
        ("canonical repo list", inventory_repo_list),
    ):
        recorded_path = binding.get("path")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ValueError(
                f"{run_id} inventory {label} path provenance is invalid"
            )
    if (
        _require_sha256(
            inventory_archive_receipt.get("sha256"),
            where=f"{run_id} inventory archive SHA-256 receipt.sha256",
        )
        != input_binding_hashes["archive_sha256_receipt"]
        or _require_sha256(
            inventory_repo_list.get("sha256"),
            where=f"{run_id} inventory canonical repo list.sha256",
        )
        != input_binding_hashes["repo_list"]
    ):
        raise ValueError(f"{run_id} archive inventory input bindings drifted")

    done = _require_mapping(manifest.get("done"), where=f"{run_id} done")
    failed = _require_mapping(manifest.get("failed"), where=f"{run_id} failed")
    exit_salvage, original_exit_path = _validate_exit_salvage(
        exit_path=exit_path,
        exit_receipt=exit_receipt,
        exit_code=exit_code,
        done=done,
        failed=failed,
        run_id=run_id,
    )
    pr_provenance: dict[str, object] | None = None
    pr_files: dict[str, Path] = {}
    if streams in {"commits", "both"}:
        pr_provenance, pr_files = _validate_pr_provenance(
            launch=launch,
            manifest=manifest,
            command=command,
            run_id=run_id,
        )
    elif launch.get("pr_inputs") is not None or any(
        option in command
        for option in (
            "--pr-repo-list",
            "--pr-store",
            "--pr-completion-receipt",
        )
    ):
        raise ValueError(f"{run_id} code-only run contains PR inputs")
    selected: set[str] = set()
    repair_base_code_run: dict[str, Any] | None = None
    if launch_schema == _TARGETED_LAUNCH_SCHEMA:
        raw_selected = launch.get("selected_repositories")
        if (
            not isinstance(raw_selected, list)
            or not raw_selected
            or any(not isinstance(repo, str) or not repo for repo in raw_selected)
            or len(set(raw_selected)) != len(raw_selected)
        ):
            raise ValueError(f"{run_id} targeted repository selection is invalid")
        selected = set(raw_selected)
        if exit_receipt.get("selected_repositories") != raw_selected:
            raise ValueError(f"{run_id} targeted exit selection drifted")
        if launch.get("expected_selected_repository_count") != len(selected):
            raise ValueError(f"{run_id} targeted repository count drifted")
        if _repeated_option(command, "--only-repo") != raw_selected:
            raise ValueError(f"{run_id} targeted command selection drifted")
        if int(_single_option(command, "--max-repos")) != len(selected):
            raise ValueError(f"{run_id} targeted command repository count drifted")
        repair_base_code_run = _require_mapping(
            launch.get("repair_base_code_run"),
            where=f"{run_id} repair base code run",
        )
        _require_exact_fields(
            repair_base_code_run,
            {"launch_sha256", "exit_sha256", "manifest_sha256"},
            where=f"{run_id} repair base code run",
        )
        for name, value in repair_base_code_run.items():
            _require_sha256(value, where=f"{run_id} repair base {name}")
        if exit_receipt.get("repair_base_code_run") != repair_base_code_run:
            raise ValueError(f"{run_id} targeted exit repair-base binding drifted")
    elif "--only-repo" in command or "--max-repos" in command:
        raise ValueError(f"{run_id} full launch contains a targeted repository limit")

    code_terminal = (
        _terminal_repositories(done=done, failed=failed, stream="code")
        if streams in {"code", "both"}
        else set()
    )
    commit_terminal = (
        _terminal_repositories(done=done, failed=failed, stream="commits")
        if streams in {"commits", "both"}
        else set()
    )
    terminal = code_terminal | commit_terminal
    if selected and (
        (exit_code == 0 and terminal != selected)
        or (exit_code != 0 and (not terminal or not terminal <= selected))
    ):
        raise ValueError(f"{run_id} targeted terminal repository set drifted")
    if selected and exit_code != 0 and not any(
        str(unit).endswith("::code") for unit in done
    ):
        raise ValueError(
            f"{run_id} interrupted targeted repair contributed no code success"
        )
    if selected and any(_unit_repo(str(unit)) not in selected for unit in (*done, *failed)):
        raise ValueError(f"{run_id} contains units outside its targeted selection")

    allowed = _manifest_allowlist(
        manifest=manifest, buckets=buckets, run_id=run_id
    )
    portable = {
        "run_id": run_id,
        "launch": {
            "schema": launch_schema,
            "sha256": hashlib.sha256(launch_raw).hexdigest(),
        },
        "exit": {
            "schema": expected_exit_schema,
            "sha256": hashlib.sha256(exit_raw).hexdigest(),
            "exit_code": exit_code,
        },
        "manifest": {
            "sha256": manifest_sha256,
            "done_units": len(done),
            "failed_units": len(failed),
            "done_unit_set_sha256": _canonical_sha256(sorted(done)),
            "failed_unit_set_sha256": _canonical_sha256(sorted(failed)),
        },
        "streams": streams,
        "selected_repositories": sorted(selected),
        "terminal_repositories": sorted(terminal),
        "terminal_repository_set_sha256": _canonical_sha256(sorted(terminal)),
        "input_artifacts": input_binding_hashes,
        "code_revision": revision,
        "allowlist_counts": {
            f"{kind}/{bucket}": len(files)
            for (kind, bucket), files in sorted(allowed.items())
        },
    }
    if pr_provenance is not None:
        portable["pr_completion"] = pr_provenance
    if exit_salvage is not None:
        portable["exit"]["salvage"] = exit_salvage
    if repair_base_code_run is not None:
        portable["repair_base_code_run"] = repair_base_code_run
    if source_code_runs is not None:
        portable["source_code_runs"] = source_code_runs
    if source_code_repair_roots is not None:
        portable["source_code_repair_roots"] = source_code_repair_roots
    files = {
        "launch": launch_path,
        "exit": exit_path,
        "manifest": manifest_path,
        "archive_sha256_receipt": archive_receipt_path,
        "archive_inventory": inventory_path,
        "repo_list": input_artifacts["repo_list"][1],
        "source_quarantine_manifest": input_artifacts[
            "source_quarantine_manifest"
        ][1],
        "tokenizer": input_artifacts["tokenizer"][1],
    }
    files.update(pr_files)
    if original_exit_path is not None:
        files["original_exit"] = original_exit_path
    return (
        portable,
        allowed,
        files,
        code_terminal,
        commit_terminal,
        {_unit_repo(str(unit)) for unit in failed},
        {str(unit) for unit in failed},
        archive_identity,
        input_binding,
        {
            "dedup_path": str(dedup_path),
            "inventory": inventory,
            "launch_schema": launch_schema,
            "streams": streams,
            "expected_repository_count": launch.get("expected_repository_count"),
            "done": done,
            "failed": failed,
            "repair_base_code_run": repair_base_code_run,
            "source_code_runs": source_code_runs,
            "source_code_repair_roots": source_code_repair_roots,
        },
    )


def load_source_composition(
    plan_path: Path,
    *,
    buckets: tuple[int, ...],
    code_root: Path,
    commit_root: Path,
) -> SourceComposition:
    """Load and fully verify a multi-run source composition plan."""

    plan_path = _resolve_regular_file(
        plan_path, where="source composition plan"
    )
    plan_raw, plan = _load_json_object(
        plan_path, where="source composition plan", max_bytes=_MAX_PLAN_BYTES
    )
    _require_exact_fields(
        plan,
        {"schema", "runs", "dedup_receipt"},
        where="source composition plan",
    )
    if plan.get("schema") != SOURCE_COMPOSITION_PLAN_SCHEMA:
        raise ValueError("source composition plan schema is unsupported")
    raw_runs = plan.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("source composition plan has no runs")
    dedup_receipt_path = _resolve_bound_path(
        plan.get("dedup_receipt"), where="source composition dedup receipt"
    )
    dedup_receipt, dedup_database_path = _validate_dedup_receipt(dedup_receipt_path)

    code_root = code_root.expanduser().resolve()
    commit_root = commit_root.expanduser().resolve()
    combined_allowlist: dict[tuple[str, int], dict[str, int]] = {
        (kind, bucket): {} for kind in ("code", "commits") for bucket in buckets
    }
    run_receipts: list[dict[str, object]] = []
    run_files: list[dict[str, Path]] = []
    run_details: list[dict[str, object]] = []
    run_ids: set[str] = set()
    archive_identities: set[str] = set()
    input_bindings: set[str] = set()
    all_code_terminal: set[str] = set()
    all_commit_terminal: set[str] = set()
    failed_repositories: set[str] = set()
    failed_units: list[tuple[str, str, str]] = []
    producers: dict[str, dict[str, object]] = {}

    for raw_run in raw_runs:
        (
            portable,
            allowed,
            files,
            code_terminal,
            commit_terminal,
            run_failed_repositories,
            run_failed_units,
            archive_identity,
            input_binding,
            details,
        ) = _load_run(
            raw_run,
            buckets=buckets,
            code_root=code_root,
            commit_root=commit_root,
        )
        run_id = str(portable["run_id"])
        if run_id in run_ids:
            raise ValueError(f"duplicate source composition run_id: {run_id}")
        run_ids.add(run_id)
        if Path(str(details["dedup_path"])).resolve() != dedup_database_path:
            raise ValueError(f"{run_id} did not use the receipt-bound global dedup DB")
        for key, files_for_bucket in allowed.items():
            for filename, rows in files_for_bucket.items():
                if filename in combined_allowlist[key]:
                    raise ValueError(
                        f"duplicate source shard across runs: "
                        f"{key[0]}/{key[1]}/{filename}"
                    )
                combined_allowlist[key][filename] = rows
        producer = dict(portable["code_revision"])
        producers[_canonical_sha256(producer)] = producer
        archive_identities.add(archive_identity)
        input_bindings.add(input_binding)
        all_code_terminal.update(code_terminal)
        all_commit_terminal.update(commit_terminal)
        failed_repositories.update(run_failed_repositories)
        streams = str(details["streams"])
        for unit in run_failed_units:
            failed_units.append((run_id, streams, unit))
        run_receipts.append(portable)
        run_files.append(files)
        run_details.append(details)

    if len(archive_identities) != 1 or len(input_bindings) != 1:
        raise ValueError("source composition runs do not share one immutable input set")

    full_code_runs = {
        (
            str(portable["launch"]["sha256"]),
            str(portable["exit"]["sha256"]),
            str(portable["manifest"]["sha256"]),
        ): details
        for portable, details in zip(run_receipts, run_details, strict=True)
        if details["launch_schema"] == _FULL_LAUNCH_SCHEMA
        and details["streams"] in {"code", "both"}
    }
    repairs_by_base: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for portable, details in zip(run_receipts, run_details, strict=True):
        repair_base = details["repair_base_code_run"]
        if repair_base is None:
            continue
        base_identity = (
            str(repair_base["launch_sha256"]),
            str(repair_base["exit_sha256"]),
            str(repair_base["manifest_sha256"]),
        )
        base_details = full_code_runs.get(base_identity)
        if base_details is None:
            raise ValueError(
                f"{portable['run_id']} repair base code run is absent from the plan"
            )
        base_failed = {
            _unit_repo(str(unit))
            for unit in _require_mapping(base_details["failed"], where="base failed")
        }
        selected = set(portable["selected_repositories"])
        if not selected <= base_failed:
            raise ValueError(
                f"{portable['run_id']} targets repositories not failed by its base run"
            )
        repairs_by_base.setdefault(base_identity, []).append(portable)
    for repairs in repairs_by_base.values():
        final_exit = _require_mapping(
            repairs[-1].get("exit"), where="final targeted repair exit"
        )
        if final_exit.get("exit_code") != 0:
            raise ValueError("final targeted code repair exit code is non-zero")
    if any(details["repair_base_code_run"] is not None for details in run_details):
        code_run_identities = [
            {
                "launch_sha256": str(portable["launch"]["sha256"]),
                "exit_sha256": str(portable["exit"]["sha256"]),
                "manifest_sha256": str(portable["manifest"]["sha256"]),
            }
            for portable, details in zip(run_receipts, run_details, strict=True)
            if details["streams"] in {"code", "both"}
        ]
        code_repair_roots = [
            str(files["launch"].parent)
            for files, details in zip(run_files, run_details, strict=True)
            if details["streams"] in {"code", "both"}
            and details["repair_base_code_run"] is not None
        ]
        for portable, details in zip(run_receipts, run_details, strict=True):
            if details["streams"] != "commits":
                continue
            if details["source_code_runs"] != code_run_identities:
                raise ValueError(
                    f"{portable['run_id']} does not bind every composed code run"
                )
            repair_root_binding = details["source_code_repair_roots"]
            if (
                repair_root_binding is not None
                and repair_root_binding["ordered"] != code_repair_roots
            ):
                raise ValueError(
                    f"{portable['run_id']} does not bind every composed code "
                    "repair root"
                )
    if any(not files for files in combined_allowlist.values()):
        missing = [
            f"{kind}/{bucket}"
            for (kind, bucket), files in combined_allowlist.items()
            if not files
        ]
        raise ValueError(
            "source composition has no trainable shards for: " + ", ".join(missing)
        )

    full_code_sets: list[set[str]] = []
    full_commit_sets: list[set[str]] = []
    inventory: dict[str, object] | None = None
    for portable, details in zip(run_receipts, run_details, strict=True):
        if details["launch_schema"] != _FULL_LAUNCH_SCHEMA:
            continue
        current_inventory = dict(details["inventory"])
        if inventory is None:
            inventory = current_inventory
        streams = str(details["streams"])
        terminals = set(portable["terminal_repositories"])
        if streams in {"code", "both"}:
            full_code_sets.append(terminals)
        if streams in {"commits", "both"}:
            full_commit_sets.append(terminals)
    if inventory is None or not full_code_sets or not full_commit_sets:
        raise ValueError("source composition requires full code and commit runs")
    expected_count = _require_positive_int(
        inventory.get("archive_unique_worktree_repo_count"),
        where="archive inventory repository count",
    )
    expected_names_sha256 = _require_sha256(
        inventory.get("archive_sorted_repo_names_json_sha256"),
        where="archive inventory repository names SHA-256",
    )
    for portable, details in zip(run_receipts, run_details, strict=True):
        if details["launch_schema"] != _FULL_LAUNCH_SCHEMA:
            continue
        if details["expected_repository_count"] != expected_count:
            raise ValueError(
                f"{portable['run_id']} full launch repository count drifted"
            )
    expected_repositories = full_code_sets[0]
    if (
        len(expected_repositories) != expected_count
        or _canonical_sha256(sorted(expected_repositories)) != expected_names_sha256
    ):
        raise ValueError("full code run does not match the archive repository inventory")
    if any(repositories != expected_repositories for repositories in full_code_sets):
        raise ValueError("full code run repository sets disagree")
    if any(repositories != expected_repositories for repositories in full_commit_sets):
        raise ValueError("full commit run repository set differs from the archive")
    if all_code_terminal != expected_repositories:
        raise ValueError("final code coverage differs from the archive repository set")
    if all_commit_terminal != expected_repositories:
        raise ValueError("final commit coverage differs from the archive repository set")

    code_success = {
        _unit_repo(str(unit))
        for details in run_details
        for unit in _require_mapping(details["done"], where="run done")
        if str(unit).endswith("::code")
    }
    commit_success = {
        _unit_repo(str(unit))
        for details in run_details
        for unit in _require_mapping(details["done"], where="run done")
        if str(unit).endswith(("::commits", "::no_git"))
    }
    unresolved: list[str] = []
    for run_id, streams, unit in failed_units:
        repo = _unit_repo(unit)
        if streams in {"code", "both"} and repo not in code_success:
            unresolved.append(f"{run_id}:{unit}:code")
        if streams in {"commits", "both"} and repo not in commit_success:
            unresolved.append(f"{run_id}:{unit}:commits")
    if unresolved:
        raise ValueError(
            "source composition has unresolved failed units: "
            + ", ".join(sorted(unresolved)[:20])
        )
    if code_success != expected_repositories:
        raise ValueError("not every archive repository has terminal code success")
    if commit_success != expected_repositories:
        raise ValueError("not every archive repository has terminal commit success")

    source_producers = [producers[digest] for digest in sorted(producers)]
    receipt: dict[str, object] = {
        "schema": SOURCE_COMPOSITION_SCHEMA,
        "status": "complete",
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "buckets": list(buckets),
        "archive": {
            "repository_count": expected_count,
            "repository_names_sha256": expected_names_sha256,
            "input_binding_sha256": next(iter(input_bindings)),
            "archive_identity_sha256": next(iter(archive_identities)),
        },
        "dedup": dedup_receipt,
        "runs": run_receipts,
        "source_producers": source_producers,
        "source_producer_set_sha256": _canonical_sha256(source_producers),
        "coverage": {
            "expected_repositories": expected_count,
            "code_success_repositories": len(code_success),
            "commit_success_repositories": len(commit_success),
            "failed_repositories_observed": len(failed_repositories),
            "failed_units_observed": len(failed_units),
            "unresolved_failed_units": 0,
            "repository_set_sha256": _canonical_sha256(
                sorted(expected_repositories)
            ),
            "allowlist_counts": {
                f"{kind}/{bucket}": len(files)
                for (kind, bucket), files in sorted(combined_allowlist.items())
            },
        },
    }
    return SourceComposition(
        allowlist=combined_allowlist,
        receipt=receipt,
        plan_path=plan_path,
        dedup_receipt_path=dedup_receipt_path,
        run_files=tuple(run_files),
    )


def build_packed_source_inventory_receipt(
    composition: SourceComposition,
    *,
    kind: str,
    input_root: Path,
) -> dict[str, object]:
    """Bind one composed source stream to its exact ZSTD Parquet bytes."""

    if kind not in {"code", "commits"}:
        raise ValueError(f"unsupported source inventory kind: {kind!r}")
    raw_buckets = composition.receipt.get("buckets")
    if not isinstance(raw_buckets, list):
        raise TypeError("source composition has no bucket list")
    buckets = tuple(
        _require_positive_int(bucket, where="source composition bucket")
        for bucket in raw_buckets
    )
    if buckets != tuple(sorted(set(buckets))):
        raise ValueError("source composition buckets are not canonical")

    expanded_root = input_root.expanduser()
    if expanded_root.is_symlink():
        raise ValueError(f"source inventory root must not be a symlink: {expanded_root}")
    root = expanded_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    expected_rows: dict[str, int] = {}
    actual_paths: dict[str, Path] = {}
    for bucket in buckets:
        bucket_root = root / str(bucket)
        if bucket_root.is_symlink() or not bucket_root.is_dir():
            raise ValueError(f"source inventory bucket is invalid: {bucket_root}")
        allowed = composition.allowlist.get((kind, bucket))
        if allowed is None:
            raise ValueError(f"source composition lacks {kind}/{bucket} shards")
        for filename, rows in allowed.items():
            expected_rows[f"{bucket}/{filename}"] = _require_positive_int(
                rows,
                where=f"source composition {kind}/{bucket}/{filename} rows",
            )
        for path in bucket_root.glob("*.parquet"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source inventory artifact is invalid: {path}")
            actual_paths[relative] = path

    if set(actual_paths) != set(expected_rows):
        raise ValueError(
            f"{kind} Parquet inventory differs from source composition: "
            f"missing={sorted(set(expected_rows) - set(actual_paths))[:20]} "
            f"unexpected={sorted(set(actual_paths) - set(expected_rows))[:20]}"
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("pyarrow is required to verify packed source Parquet") from error

    inventory: list[dict[str, object]] = []
    artifact_identities: dict[str, tuple[int, int, int, int]] = {}
    total_rows = 0
    for relative in sorted(actual_paths):
        path = actual_paths[relative]
        before = path.stat()
        metadata = pq.ParquetFile(path).metadata
        codecs = {
            str(metadata.row_group(group).column(column).compression)
            for group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
        }
        rows = expected_rows[relative]
        if int(metadata.num_rows) != rows or codecs != {"ZSTD"}:
            raise ValueError(
                f"source artifact is not receipt-bound ZSTD Parquet: {path}"
            )
        sha256 = _sha256(path)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"source artifact changed while hashing: {path}")
        artifact_identities[relative] = after_identity
        inventory.append(
            {"path": relative, "sha256": sha256, "size": after.st_size}
        )
        total_rows += rows

    observed_paths = {
        path.relative_to(root).as_posix(): path
        for bucket in buckets
        for path in (root / str(bucket)).glob("*.parquet")
    }
    if set(observed_paths) != set(expected_rows):
        raise RuntimeError("source Parquet inventory changed while it was hashed")
    for relative, path in observed_paths.items():
        current = path.stat()
        if path.is_symlink() or (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != artifact_identities[relative]:
            raise RuntimeError(f"source artifact changed while hashing: {path}")
    plan_sha256 = _sha256(composition.plan_path)
    if plan_sha256 != composition.receipt.get("plan_sha256"):
        raise RuntimeError("source composition plan changed while inventory was built")

    return {
        "schema": PACKED_SOURCE_INVENTORY_SCHEMA,
        "status": "complete",
        "kind": kind,
        "input_root": str(root),
        "buckets": list(buckets),
        "source_composition": {
            "plan": {
                "path": str(composition.plan_path),
                "sha256": plan_sha256,
            },
            "receipt_sha256": source_composition_receipt_sha256(
                composition.receipt
            ),
        },
        "source_inventory": inventory,
        "source_inventory_sha256": _canonical_sha256(inventory),
        "totals": {
            "files": len(inventory),
            "rows": total_rows,
            "bytes": sum(int(record["size"]) for record in inventory),
        },
        "unresolved_count": 0,
    }


__all__ = [
    "GLOBAL_DEDUP_RECEIPT_SCHEMA",
    "PACKED_SOURCE_INVENTORY_SCHEMA",
    "SOURCE_COMPOSITION_PLAN_SCHEMA",
    "SOURCE_COMPOSITION_SCHEMA",
    "SourceComposition",
    "build_packed_source_inventory_receipt",
    "load_source_composition",
    "source_composition_receipt_sha256",
]
