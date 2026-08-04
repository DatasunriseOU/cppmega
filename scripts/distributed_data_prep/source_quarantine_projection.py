"""Derive an exact source-quarantine manifest for one pinned Git tree.

The corpus-wide quarantine manifest is built from the local archive.  A
public-cloud checkout is pinned to a different Git commit, so rules for files
which are absent from that exact tree must not be passed to the strict
quarantine consumer.  This module derives a smaller v2 manifest without
changing any present rule: present rules are retained verbatim and still go
through the normal size, digest, and format checks.  The deterministic
projection receipt records every omission and binds it to both the base
manifest and the materialized Git tree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
    require_git_object,
    require_int,
    require_sha256,
    sha256_file,
)
from tools.clang_indexer.source_quarantine import ProjectSourceQuarantine


SOURCE_QUARANTINE_PROJECTION_SCHEMA = "cppmega.source_quarantine_projection_v1"
PINNED_TREE_PROJECTION_MODE = "pinned_source_tree_v1"
_MANIFEST_SCHEMA = "cppmega.source_quarantine_manifest_v2"
_SOURCE_FIELDS = frozenset(
    {"kind", "remote_url", "expected_commit", "resolved_commit", "tree"}
)
_PROJECTED_MANIFEST_FIELDS = frozenset(
    {"schema", "sha256", "size_bytes", "entry_count", "collection_count"}
)
_SELECTION_FIELDS = frozenset(
    {
        "included_entry_paths",
        "omitted_entry_paths",
        "included_collections",
        "omitted_collections",
    }
)


class SourceQuarantineProjectionError(ContractError):
    """A pinned-tree quarantine projection cannot be proved exact."""


def _require_mapping(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SourceQuarantineProjectionError(f"{where} must be an object")
    return dict(value)


def _require_nonempty_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceQuarantineProjectionError(f"{where} must be a non-empty string")
    return value


def _canonical_relative_path(value: object, *, where: str) -> str:
    path = _require_nonempty_string(value, where=where)
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or path != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in path
    ):
        raise SourceQuarantineProjectionError(
            f"{where} is not a canonical safe POSIX path: {path!r}"
        )
    return path


def _canonical_prefix(value: object, *, where: str) -> str:
    prefix = _require_nonempty_string(value, where=where)
    if not prefix.endswith("/"):
        raise SourceQuarantineProjectionError(f"{where} must end with a slash")
    _canonical_relative_path(prefix.removesuffix("/"), where=where)
    return prefix


def _source_binding(source_snapshot: Mapping[str, object]) -> dict[str, str]:
    snapshot = _require_mapping(source_snapshot, where="source snapshot")
    if snapshot.get("kind") != "git_mirror":
        raise SourceQuarantineProjectionError(
            "pinned-tree quarantine projection requires a git_mirror source"
        )
    remote_url = _require_nonempty_string(
        snapshot.get("remote_url"), where="source snapshot remote_url"
    )
    expected_commit = require_git_object(
        snapshot.get("expected_commit"), where="source snapshot expected_commit"
    )
    resolved_commit = require_git_object(
        snapshot.get("resolved_commit"), where="source snapshot resolved_commit"
    )
    tree = require_git_object(snapshot.get("tree"), where="source snapshot tree")
    if expected_commit != resolved_commit:
        raise SourceQuarantineProjectionError(
            "source snapshot resolved commit differs from expected commit"
        )
    return {
        "kind": "git_mirror",
        "remote_url": remote_url,
        "expected_commit": expected_commit,
        "resolved_commit": resolved_commit,
        "tree": tree,
    }


def _regular_file_or_absent(root: Path, relative_path: str) -> Path | None:
    candidate = root / relative_path
    if candidate.is_symlink():
        raise SourceQuarantineProjectionError(
            f"quarantine candidate is a symlink: {relative_path}"
        )
    if not candidate.exists():
        return None
    if not candidate.is_file():
        raise SourceQuarantineProjectionError(
            f"quarantine candidate is not a regular file: {relative_path}"
        )
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise SourceQuarantineProjectionError(
            f"quarantine candidate escapes source root: {relative_path}"
        ) from exc
    return candidate


def _collection_candidates(root: Path, prefix: str, suffix: str) -> list[Path]:
    directory = root / prefix.removesuffix("/")
    if directory.is_symlink():
        raise SourceQuarantineProjectionError(
            f"quarantine collection root is a symlink: {prefix}"
        )
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise SourceQuarantineProjectionError(
            f"quarantine collection root is not a directory: {prefix}"
        )
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.name.endswith(suffix):
            continue
        if path.is_symlink():
            raise SourceQuarantineProjectionError(
                f"quarantine collection candidate is a symlink: {path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise SourceQuarantineProjectionError(
                f"quarantine collection candidate escapes source root: {path}"
            ) from exc
        candidates.append(path)
    return candidates


def _collection_identity(raw: Mapping[str, object]) -> dict[str, str]:
    return {
        "relative_path_prefix": _canonical_prefix(
            raw.get("relative_path_prefix"),
            where="quarantine collection relative_path_prefix",
        ),
        "relative_path_suffix": _require_nonempty_string(
            raw.get("relative_path_suffix"),
            where="quarantine collection relative_path_suffix",
        ),
    }


def _load_base_manifest(path: Path, *, project_id: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise SourceQuarantineProjectionError(
            f"base quarantine manifest is not a regular file: {path}"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceQuarantineProjectionError(
            f"base quarantine manifest is invalid JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SourceQuarantineProjectionError("base quarantine manifest must be an object")
    # This parses every rule, including rules for other projects, before the
    # projection selects anything.  It prevents a projection from laundering a
    # malformed corpus-wide manifest into a valid smaller one.
    ProjectSourceQuarantine.load(path, project_id=project_id)
    if value.get("schema") != _MANIFEST_SCHEMA:
        raise SourceQuarantineProjectionError(
            "pinned-tree projection requires a v2 quarantine manifest"
        )
    return value, raw


def _projection_payload(
    *,
    project_id: str,
    base_manifest_sha256: str,
    base_entry_count: int,
    base_collection_count: int,
    source: Mapping[str, str],
    projected_manifest: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SOURCE_QUARANTINE_PROJECTION_SCHEMA,
        "status": "complete",
        "training_ready": False,
        "mode": PINNED_TREE_PROJECTION_MODE,
        "project_id": project_id,
        "base_manifest_sha256": base_manifest_sha256,
        "base_project_entry_count": base_entry_count,
        "base_project_collection_count": base_collection_count,
        "source": dict(source),
        "projected_manifest": dict(projected_manifest),
        "selection": dict(selection),
    }
    payload["projection_sha256"] = canonical_sha256(payload)
    return payload


def build_pinned_tree_quarantine_projection(
    *,
    base_manifest_path: Path,
    source_root: Path,
    project_id: str,
    source_snapshot: Mapping[str, object],
    projected_manifest_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    """Write and validate the exact per-tree manifest and its audit receipt."""

    root = source_root.resolve()
    if source_root.is_symlink() or not root.is_dir():
        raise SourceQuarantineProjectionError(
            f"source root is not a real directory: {source_root}"
        )
    project = _require_nonempty_string(project_id, where="project_id")
    source = _source_binding(source_snapshot)
    base, base_raw = _load_base_manifest(base_manifest_path, project_id=project)
    raw_entries = base.get("entries")
    raw_collections = base.get("collections")
    if not isinstance(raw_entries, list) or not isinstance(raw_collections, list):
        raise SourceQuarantineProjectionError("base quarantine manifest shape drifted")

    project_entries = [
        dict(entry)
        for entry in raw_entries
        if isinstance(entry, Mapping) and entry.get("project_id") == project
    ]
    project_collections = [
        dict(collection)
        for collection in raw_collections
        if isinstance(collection, Mapping) and collection.get("project_id") == project
    ]
    included_entries: list[dict[str, object]] = []
    omitted_entry_paths: list[str] = []
    candidate_paths: list[Path] = []
    for entry in project_entries:
        relative_path = _canonical_relative_path(
            entry.get("relative_path"), where="quarantine entry relative_path"
        )
        candidate = _regular_file_or_absent(root, relative_path)
        if candidate is None:
            omitted_entry_paths.append(relative_path)
            continue
        included_entries.append(entry)
        candidate_paths.append(candidate)

    included_collections: list[dict[str, object]] = []
    included_collection_ids: list[dict[str, str]] = []
    omitted_collection_ids: list[dict[str, str]] = []
    for collection in project_collections:
        identity = _collection_identity(collection)
        matching = _collection_candidates(
            root,
            identity["relative_path_prefix"],
            identity["relative_path_suffix"],
        )
        if not matching:
            omitted_collection_ids.append(identity)
            continue
        included_collections.append(collection)
        included_collection_ids.append(identity)
        candidate_paths.extend(matching)

    derived = {
        "schema": _MANIFEST_SCHEMA,
        "entries": included_entries,
        "collections": included_collections,
    }
    atomic_write_json(projected_manifest_path, derived)
    # Load and exercise the derived manifest before publishing it.  Present
    # rules therefore retain the exact size/SHA/format verification that the
    # corpus-wide manifest had; only truly absent rules are omitted.
    derived_policy = ProjectSourceQuarantine.load(projected_manifest_path, project_id=project)
    derived_policy.filter_candidates(root, [str(path) for path in candidate_paths])

    derived_raw = projected_manifest_path.read_bytes()
    selection = {
        "included_entry_paths": sorted(
            _canonical_relative_path(
                entry.get("relative_path"), where="included quarantine entry"
            )
            for entry in included_entries
        ),
        "omitted_entry_paths": sorted(omitted_entry_paths),
        "included_collections": sorted(
            included_collection_ids,
            key=lambda item: (item["relative_path_prefix"], item["relative_path_suffix"]),
        ),
        "omitted_collections": sorted(
            omitted_collection_ids,
            key=lambda item: (item["relative_path_prefix"], item["relative_path_suffix"]),
        ),
    }
    receipt = _projection_payload(
        project_id=project,
        base_manifest_sha256=hashlib.sha256(base_raw).hexdigest(),
        base_entry_count=len(project_entries),
        base_collection_count=len(project_collections),
        source=source,
        projected_manifest={
            "schema": _MANIFEST_SCHEMA,
            "sha256": hashlib.sha256(derived_raw).hexdigest(),
            "size_bytes": len(derived_raw),
            "entry_count": len(included_entries),
            "collection_count": len(included_collections),
        },
        selection=selection,
    )
    validate_pinned_tree_quarantine_projection(
        receipt,
        project_id=project,
        base_manifest_sha256=hashlib.sha256(base_raw).hexdigest(),
        source_snapshot=source_snapshot,
        projected_manifest_path=projected_manifest_path,
    )
    atomic_write_json(receipt_path, receipt)
    return receipt


def validate_pinned_tree_quarantine_projection(
    value: Mapping[str, object],
    *,
    project_id: str,
    base_manifest_sha256: str,
    source_snapshot: Mapping[str, object],
    projected_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Validate an immutable projection receipt without trusting its path."""

    receipt = _require_mapping(value, where="source quarantine projection")
    expected_fields = {
        "schema",
        "status",
        "training_ready",
        "mode",
        "project_id",
        "base_manifest_sha256",
        "base_project_entry_count",
        "base_project_collection_count",
        "source",
        "projected_manifest",
        "selection",
        "projection_sha256",
    }
    if set(receipt) != expected_fields:
        raise SourceQuarantineProjectionError(
            "source quarantine projection has an invalid field set"
        )
    if (
        receipt["schema"] != SOURCE_QUARANTINE_PROJECTION_SCHEMA
        or receipt["status"] != "complete"
        or receipt["training_ready"] is not False
        or receipt["mode"] != PINNED_TREE_PROJECTION_MODE
        or receipt["project_id"] != project_id
    ):
        raise SourceQuarantineProjectionError("source quarantine projection header drifted")
    expected_base_sha = require_sha256(
        base_manifest_sha256, where="expected base quarantine manifest sha256"
    )
    if receipt["base_manifest_sha256"] != expected_base_sha:
        raise SourceQuarantineProjectionError(
            "source quarantine projection base manifest binding drifted"
        )
    source = _require_mapping(receipt["source"], where="projection source")
    if set(source) != _SOURCE_FIELDS or source != _source_binding(source_snapshot):
        raise SourceQuarantineProjectionError(
            "source quarantine projection source binding drifted"
        )
    base_entries = require_int(
        receipt["base_project_entry_count"],
        where="projection base project entry count",
        minimum=0,
    )
    base_collections = require_int(
        receipt["base_project_collection_count"],
        where="projection base project collection count",
        minimum=0,
    )
    projected = _require_mapping(
        receipt["projected_manifest"], where="projection projected manifest"
    )
    if set(projected) != _PROJECTED_MANIFEST_FIELDS:
        raise SourceQuarantineProjectionError(
            "projection projected manifest has an invalid field set"
        )
    if projected["schema"] != _MANIFEST_SCHEMA:
        raise SourceQuarantineProjectionError("projection manifest schema drifted")
    projected_sha = require_sha256(
        projected["sha256"], where="projection manifest sha256"
    )
    projected_size = require_int(
        projected["size_bytes"], where="projection manifest size", minimum=1
    )
    projected_entries = require_int(
        projected["entry_count"], where="projection manifest entry count", minimum=0
    )
    projected_collections = require_int(
        projected["collection_count"],
        where="projection manifest collection count",
        minimum=0,
    )
    selection = _require_mapping(receipt["selection"], where="projection selection")
    if set(selection) != _SELECTION_FIELDS:
        raise SourceQuarantineProjectionError(
            "projection selection has an invalid field set"
        )
    included_entries = selection["included_entry_paths"]
    omitted_entries = selection["omitted_entry_paths"]
    if not isinstance(included_entries, list) or not isinstance(omitted_entries, list):
        raise SourceQuarantineProjectionError("projection entry selection must be lists")
    normalized_included = [
        _canonical_relative_path(path, where="included projection entry")
        for path in included_entries
    ]
    normalized_omitted = [
        _canonical_relative_path(path, where="omitted projection entry")
        for path in omitted_entries
    ]
    if (
        normalized_included != sorted(normalized_included)
        or normalized_omitted != sorted(normalized_omitted)
        or len(set(normalized_included)) != len(normalized_included)
        or len(set(normalized_omitted)) != len(normalized_omitted)
        or set(normalized_included) & set(normalized_omitted)
        or len(normalized_included) + len(normalized_omitted) != base_entries
        or len(normalized_included) != projected_entries
    ):
        raise SourceQuarantineProjectionError("projection entry selection does not close")

    def normalize_collections(raw: object, *, where: str) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            raise SourceQuarantineProjectionError(f"{where} must be a list")
        normalized = [_collection_identity(_require_mapping(item, where=where)) for item in raw]
        if normalized != sorted(
            normalized,
            key=lambda item: (item["relative_path_prefix"], item["relative_path_suffix"]),
        ) or len(
            {
                (item["relative_path_prefix"], item["relative_path_suffix"])
                for item in normalized
            }
        ) != len(normalized):
            raise SourceQuarantineProjectionError(f"{where} is not canonical")
        return normalized

    included_collections = normalize_collections(
        selection["included_collections"], where="included projection collections"
    )
    omitted_collections = normalize_collections(
        selection["omitted_collections"], where="omitted projection collections"
    )
    if (
        {
            (item["relative_path_prefix"], item["relative_path_suffix"])
            for item in included_collections
        }
        & {
            (item["relative_path_prefix"], item["relative_path_suffix"])
            for item in omitted_collections
        }
        or len(included_collections) + len(omitted_collections) != base_collections
        or len(included_collections) != projected_collections
    ):
        raise SourceQuarantineProjectionError("projection collection selection does not close")

    expected_projection_sha = receipt.pop("projection_sha256")
    try:
        if require_sha256(
            expected_projection_sha, where="projection receipt sha256"
        ) != canonical_sha256(receipt):
            raise SourceQuarantineProjectionError(
                "source quarantine projection logical digest drifted"
            )
    finally:
        receipt["projection_sha256"] = expected_projection_sha
    if projected_manifest_path is not None:
        if projected_manifest_path.is_symlink() or not projected_manifest_path.is_file():
            raise SourceQuarantineProjectionError(
                "projected quarantine manifest is not a regular file"
            )
        if (
            projected_manifest_path.stat().st_size != projected_size
            or sha256_file(projected_manifest_path) != projected_sha
        ):
            raise SourceQuarantineProjectionError(
                "projected quarantine manifest bytes drifted"
            )
        # Parse the actual file as a v2 manifest before a caller uses it.
        ProjectSourceQuarantine.load(projected_manifest_path, project_id=project_id)
    return receipt


__all__ = [
    "PINNED_TREE_PROJECTION_MODE",
    "SOURCE_QUARANTINE_PROJECTION_SCHEMA",
    "SourceQuarantineProjectionError",
    "build_pinned_tree_quarantine_projection",
    "validate_pinned_tree_quarantine_projection",
]
