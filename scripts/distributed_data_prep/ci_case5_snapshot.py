#!/usr/bin/env python3
"""Prepare and consume immutable CASE5 CI cloud-lane snapshots.

The CASE5 source is a directory of receipt-bound SQLite/pack files.  A cloud
lane must not turn that directory into an opaque archive: each byte artifact is
published independently at a content-addressed GCS URI and the lane manifest
binds the exact generation, size, and digest.  The adapter reconstructs the
small directory layout from those objects and emits one canonical candidate for
each assigned CAS occurrence.

This module intentionally stops at ``canonical_jsonl`` candidates.  The cloud
lane worker and the existing Parquet/Megatron sealing path remain responsible
for global deduplication and declaring training readiness.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterator, Mapping, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in os.sys.path:
        os.sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_nonempty,
    require_sha256,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.cloud_lane import (  # noqa: E402
    build_cloud_lane_manifest,
)


SNAPSHOT_SET_SCHEMA = "cppmega.ci_case5_snapshot_set_v1"
SNAPSHOT_RECEIPT_SCHEMA = SNAPSHOT_SET_SCHEMA
CASE5_PAYLOAD_SCHEMA = "cppmega.ci_case5_candidate_payload_v1"
CASE5_DATASET_SCHEMA = "cppmega.ci_case5_candidate_dataset_v1"
CASE5_MEMBERSHIP_POLICY = "cppmega.ci_case5_occurrence_fetch_membership_v1"
CASE5_ADAPTER_SCHEMA = "cppmega.ci_case5_snapshot_adapter_v1"
ADAPTER_REQUEST_SCHEMA = "cppmega.distributed_cloud_lane_adapter_request_v1"
ADAPTER_OUTPUT_SCHEMA = "cppmega.distributed_cloud_lane_adapter_output_v1"

MODE_THRESHOLD = "threshold"
MODE_PRODUCTION = "inventory-exhaustive"
COMPLETION_MODE_THRESHOLD = MODE_THRESHOLD
COMPLETION_MODE_INVENTORY_EXHAUSTIVE = MODE_PRODUCTION

PRIMARY_SNAPSHOT_NAME = "content-store-index.sqlite3"
MEMBERSHIP_SNAPSHOT_NAME = "fetch-state.sqlite3"
STORE_RECEIPT_NAME = "store-receipt.json"
FETCH_RECEIPT_NAME = "fetch-receipt.json"
TOKENIZER_NAME = "tokenizer.json"
INVENTORY_NAME = "inventory.sqlite3"
INVENTORY_RECEIPT_NAME = "inventory-receipt.json"
MERGE_RECEIPT_NAME = "merge-receipt.json"

_PACK_RE = re.compile(r"^pack-[0-9]{8}\.cicp$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_STORE_RECEIPT_SCHEMA = "cppmega_ci_content_store_receipt_v1"
_FETCH_RECEIPT_THRESHOLD_SCHEMA = "cppmega_ci_stream_fetch_receipt_v3"
_FETCH_RECEIPT_PRODUCTION_SCHEMA = "cppmega_ci_stream_fetch_receipt_v4"
_FETCH_STATE_SCHEMA = "cppmega_ci_stream_fetch_v4"
_SNAPSHOT_ROLES = frozenset({"primary", "membership", "ancillary"})
_FIXED_NAMES = frozenset(
    {
        PRIMARY_SNAPSHOT_NAME,
        MEMBERSHIP_SNAPSHOT_NAME,
        STORE_RECEIPT_NAME,
        FETCH_RECEIPT_NAME,
        TOKENIZER_NAME,
        INVENTORY_NAME,
        INVENTORY_RECEIPT_NAME,
        MERGE_RECEIPT_NAME,
    }
)


def _without_digest(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(dict(value))
    result.pop(field, None)
    return result


def snapshot_set_sha256(value: Mapping[str, object]) -> str:
    """Return the logical digest of a snapshot-set receipt."""

    return canonical_sha256(_without_digest(value, "receipt_sha256"))


def snapshot_receipt_sha256(value: Mapping[str, object]) -> str:
    return snapshot_set_sha256(value)


def _stable_descriptor(path: Path, *, where: str) -> dict[str, object]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ContractError(f"{where} must be a regular file: {candidate}")
    before = candidate.stat()
    digest = sha256_file(candidate)
    after = candidate.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ContractError(f"{where} changed while it was hashed")
    if after.st_size < 1:
        raise ContractError(f"{where} must not be empty")
    return {"size_bytes": after.st_size, "sha256": digest}


def _load_object(path: Path, *, where: str) -> tuple[dict[str, Any], str]:
    raw, value = load_json_object(path, where=where)
    return value, hashlib.sha256(raw).hexdigest()


def _require_hex(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX_RE.fullmatch(value) is None:
        raise ContractError(f"{where} must be lowercase SHA-256")
    return value


def _require_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def _source_file(root: Path, name: str, *, where: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where} is missing or unsafe: {path}")
    return path


def _resolve_input_path(
    path: Path, *, where: str, directory: bool = False
) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not (raw.is_dir() if directory else raw.is_file()):
        kind = "directory" if directory else "file"
        raise ContractError(f"{where} must be a regular {kind}: {raw}")
    return raw.resolve()


def _validate_store_receipt(value: Mapping[str, object]) -> dict[str, Any]:
    if value.get("schema") != _STORE_RECEIPT_SCHEMA:
        raise ContractError("CASE5 store receipt schema is unsupported")
    if value.get("status") != "complete" or value.get("store_schema") != "cppmega_ci_content_store_v1":
        raise ContractError("CASE5 store receipt is not complete")
    verification = value.get("verification")
    if not isinstance(verification, Mapping) or verification.get("mode") != "full" or verification.get("ok") is not True:
        raise ContractError("CASE5 store receipt lacks full verification")
    counters = value.get("counters")
    if not isinstance(counters, Mapping):
        raise ContractError("CASE5 store receipt counters are missing")
    require_int(counters.get("occurrence_count"), where="store counters.occurrence_count", minimum=1)
    require_int(counters.get("unique_content_count"), where="store counters.unique_content_count", minimum=1)
    require_int(counters.get("exact_unique_payload_tokens"), where="store counters.exact_unique_payload_tokens", minimum=0)
    for field in (
        "logical_content_set_sha256",
        "logical_token_sequence_set_sha256",
        "occurrence_set_sha256",
        "sqlite_logical_sha256",
        "sqlite_schema_sha256",
        "policy_sha256",
        "script_sha256",
    ):
        _require_hex(value.get(field), where=f"store receipt {field}")
    packs = value.get("pack_hashes")
    if not isinstance(packs, list) or not packs:
        raise ContractError("CASE5 store receipt pack_hashes is empty")
    names: list[str] = []
    for index, raw in enumerate(packs):
        if not isinstance(raw, Mapping):
            raise ContractError(f"store receipt pack_hashes[{index}] is malformed")
        name = require_nonempty(raw.get("filename"), where=f"pack_hashes[{index}].filename")
        if _PACK_RE.fullmatch(name) is None or name in names:
            raise ContractError(f"store receipt has an unsafe/duplicate pack: {name}")
        require_int(raw.get("committed_end"), where=f"pack_hashes[{index}].committed_end", minimum=1)
        require_int(raw.get("content_count"), where=f"pack_hashes[{index}].content_count", minimum=1)
        _require_hex(raw.get("sha256"), where=f"pack_hashes[{index}].sha256")
        names.append(name)
    if names != sorted(names):
        raise ContractError("store receipt pack_hashes are not name-sorted")
    return dict(value)


def _validate_fetch_receipt(
    value: Mapping[str, object],
    *,
    mode: str,
    store_receipt: Mapping[str, object],
    fetch_state: Path,
) -> dict[str, Any]:
    expected_schema = (
        _FETCH_RECEIPT_THRESHOLD_SCHEMA
        if mode == MODE_THRESHOLD
        else _FETCH_RECEIPT_PRODUCTION_SCHEMA
    )
    if value.get("schema") != expected_schema:
        raise ContractError(
            f"CASE5 {mode} mode requires fetch receipt {expected_schema}"
        )
    if value.get("content_store_receipt") != dict(store_receipt):
        raise ContractError("fetch receipt does not bind the supplied store receipt")
    frozen = value.get("frozen_fetch_state")
    if not isinstance(frozen, Mapping) or frozen.get("schema") != _FETCH_STATE_SCHEMA:
        raise ContractError("fetch receipt frozen_fetch_state binding is missing")
    artifact = frozen.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractError("fetch receipt frozen_fetch_state artifact is missing")
    descriptor = _stable_descriptor(fetch_state, where="fetch state")
    if (
        artifact.get("byte_size") != descriptor["size_bytes"]
        or artifact.get("sha256") != descriptor["sha256"]
    ):
        raise ContractError("fetch receipt does not bind exact fetch-state bytes")
    for field in ("sqlite_schema_sha256", "sqlite_logical_sha256", "sidecar_set_sha256"):
        _require_hex(frozen.get(field), where=f"fetch receipt frozen state {field}")
    settings = frozen.get("settings")
    if not isinstance(settings, Mapping) or settings.get("schema") != _FETCH_STATE_SCHEMA:
        raise ContractError("fetch receipt frozen settings are unsupported")
    summary = frozen.get("summary")
    if not isinstance(summary, Mapping):
        raise ContractError("fetch receipt frozen summary is missing")
    require_int(summary.get("members"), where="fetch summary.members", minimum=1)
    require_int(summary.get("chunks"), where="fetch summary.chunks", minimum=1)
    if mode == MODE_PRODUCTION:
        if value.get("completion_mode") != MODE_PRODUCTION or value.get("production_complete") is not True:
            raise ContractError("production CASE5 source is not inventory-exhaustive")
        if value.get("coverage_semantics") != "exact-production-inventory-attempt-equality":
            raise ContractError("production fetch receipt coverage semantics drifted")
    else:
        if value.get("completion_mode") not in {None, MODE_THRESHOLD}:
            raise ContractError("threshold source carries production completion fields")
        if value.get("production_complete") is True:
            raise ContractError("threshold source cannot claim production completion")
    return dict(value)


def _validate_source_layout(store_root: Path) -> tuple[Path, tuple[str, ...]]:
    root = Path(store_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"content store root is missing or unsafe: {root}")
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink():
            raise ContractError(f"content store contains a symlink: {entry}")
        if entry.is_file() and entry.name.endswith(("-wal", "-shm", "-journal")):
            raise ContractError("content store contains mutable SQLite WAL/journal files")
        if entry.is_dir() and entry.name != "orphaned":
            raise ContractError(f"content store contains an unexpected directory: {entry.name}")
    files = sorted(path.name for path in entries if path.is_file())
    packs = tuple(name for name in files if _PACK_RE.fullmatch(name))
    # The source layout uses index.sqlite3; the cloud snapshot has a stable
    # descriptive name and is mapped back to this filename by the adapter.
    if "index.sqlite3" not in files:
        raise ContractError("content store index.sqlite3 is missing")
    if not packs:
        raise ContractError("content store has no immutable pack files")
    allowed = {"index.sqlite3", *packs}
    if set(files) != allowed:
        raise ContractError("content store contains files outside the frozen layout")
    return root, packs


def _mode(value: object) -> str:
    text = require_nonempty(value, where="source mode")
    if text in {"production", MODE_PRODUCTION}:
        return MODE_PRODUCTION
    if text == MODE_THRESHOLD:
        return MODE_THRESHOLD
    raise ContractError(f"unsupported CASE5 source mode: {text!r}")


def _source_paths(
    *,
    store_root: Path,
    store_receipt: Path,
    fetch_state: Path,
    fetch_receipt: Path,
    tokenizer: Path,
    mode: str,
    inventory: Path | None,
    inventory_receipt: Path | None,
    merge_receipt: Path | None,
) -> list[tuple[str, Path, str]]:
    _validate_source_layout(store_root)
    paths: list[tuple[str, Path, str]] = [
        (PRIMARY_SNAPSHOT_NAME, _source_file(store_root, "index.sqlite3", where="store index"), "primary"),
        (MEMBERSHIP_SNAPSHOT_NAME, fetch_state, "membership"),
        (STORE_RECEIPT_NAME, store_receipt, "ancillary"),
        (FETCH_RECEIPT_NAME, fetch_receipt, "ancillary"),
        (TOKENIZER_NAME, tokenizer, "ancillary"),
    ]
    for pack_name in _validate_source_layout(store_root)[1]:
        paths.append((pack_name, _source_file(store_root, pack_name, where=f"store pack {pack_name}"), "ancillary"))
    production_values = (inventory, inventory_receipt, merge_receipt)
    if mode == MODE_PRODUCTION:
        if any(value is None for value in production_values):
            raise ContractError("inventory-exhaustive mode requires inventory, inventory receipt, and merge receipt")
        if inventory is None or inventory_receipt is None or merge_receipt is None:
            raise ContractError("inventory-exhaustive production inputs are unavailable")
        paths.extend(
            [
                (INVENTORY_NAME, inventory, "ancillary"),
                (INVENTORY_RECEIPT_NAME, inventory_receipt, "ancillary"),
                (MERGE_RECEIPT_NAME, merge_receipt, "ancillary"),
            ]
        )
    elif any(value is not None for value in production_values):
        raise ContractError("threshold mode must not receive production evidence")
    return paths


def _validate_store_artifacts(
    store_root: Path, store_receipt: Mapping[str, object]
) -> None:
    """Check the physical store files before any object is published."""

    _root, pack_names = _validate_source_layout(store_root)
    raw_packs = store_receipt.get("pack_hashes")
    if not isinstance(raw_packs, list):
        raise ContractError("store receipt pack_hashes is missing")
    expected: dict[str, Mapping[str, object]] = {
        str(item["filename"]): item
        for item in raw_packs
        if isinstance(item, Mapping)
    }
    if tuple(sorted(expected)) != pack_names:
        raise ContractError("store receipt pack set differs from the source directory")
    for name in pack_names:
        item = expected[name]
        path = store_root / name
        descriptor = _stable_descriptor(path, where=f"store pack {name}")
        committed_end = require_int(item.get("committed_end"), where=f"store pack {name}.committed_end", minimum=1)
        if descriptor["size_bytes"] != committed_end or descriptor["sha256"] != item.get("sha256"):
            raise ContractError(f"store pack {name} differs from the receipt")
    index = store_root / "index.sqlite3"
    _stable_descriptor(index, where="store index")
    try:
        connection = sqlite3.connect(f"{index.as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
            if integrity != ["ok"]:
                raise ContractError("store SQLite integrity_check failed")
            row = connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()
            counters = _require_mapping(
                store_receipt.get("counters"), where="store receipt counters"
            )
            if row is None or int(row[0]) != int(counters["occurrence_count"]):
                raise ContractError("store occurrence count differs from receipt")
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ContractError("store SQLite cannot be opened immutable") from exc


def _reject_sqlite_sidecars(path: Path, *, where: str) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        candidate = path.with_name(path.name + suffix)
        if candidate.exists() or candidate.is_symlink():
            raise ContractError(f"{where} has a mutable SQLite sidecar: {candidate.name}")


def _verify_production_provenance(
    *,
    store_root: Path,
    store_receipt: Path,
    fetch_state: Path,
    inventory: Path,
    inventory_receipt: Path,
    fetch_receipt: Path,
    merge_receipt: Path,
) -> None:
    """Reuse the existing full inventory/fetch/merge equality proof."""

    try:
        from scripts.export_ci_content_store_case5 import (
            _verify_exhaustive_export_provenance,
        )

        _verify_exhaustive_export_provenance(
            store_root=store_root,
            store_receipt_path=store_receipt,
            fetch_state_path=fetch_state,
            inventory_path=inventory,
            inventory_receipt_path=inventory_receipt,
            fetch_receipt_path=fetch_receipt,
            merge_receipt_path=merge_receipt,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            "CASE5 production inventory/fetch/merge provenance was refused"
        ) from exc


def _schema_digest(schema: str) -> str:
    return canonical_sha256({"schema": schema})


def _schema_for(
    name: str,
    *,
    store_receipt: Mapping[str, object],
    fetch_receipt: Mapping[str, object],
) -> tuple[str, str]:
    if name == PRIMARY_SNAPSHOT_NAME:
        return (
            "ci-case5-content-store-sqlite-v1",
            _require_hex(
                store_receipt["sqlite_schema_sha256"], where="store SQLite schema"
            ),
        )
    if name == MEMBERSHIP_SNAPSHOT_NAME:
        frozen = _require_mapping(
            fetch_receipt.get("frozen_fetch_state"), where="fetch frozen state"
        )
        return (
            "ci-case5-fetch-state-sqlite-v4",
            _require_hex(
                frozen["sqlite_schema_sha256"], where="fetch SQLite schema"
            ),
        )
    if _PACK_RE.fullmatch(name):
        policy = store_receipt.get("policy")
        if not isinstance(policy, Mapping):
            raise ContractError("store receipt policy is missing")
        return (
            "ci-case5-content-pack-v1",
            _schema_digest(str(policy.get("frame_schema", ""))),
        )
    if name in {INVENTORY_NAME}:
        return "ci-inventory-sqlite-v1", _schema_digest("ci-inventory-sqlite-v1")
    if name == TOKENIZER_NAME:
        return "tokenizer-json-v1", _schema_digest("tokenizer-json-v1")
    return "ci-case5-receipt-json-v1", _schema_digest("ci-case5-receipt-json-v1")


def _content_set_for(name: str, *, store_receipt: Mapping[str, object], fetch_receipt: Mapping[str, object], descriptor: Mapping[str, object]) -> str:
    if name == PRIMARY_SNAPSHOT_NAME:
        return _require_hex(store_receipt["logical_content_set_sha256"], where="store logical content set")
    if name == MEMBERSHIP_SNAPSHOT_NAME:
        frozen = _require_mapping(
            fetch_receipt.get("frozen_fetch_state"), where="fetch frozen state"
        )
        return _require_hex(frozen["sidecar_set_sha256"], where="fetch sidecar set")
    return str(descriptor["sha256"])


def _record_count_for(name: str, *, store_receipt: Mapping[str, object], fetch_receipt: Mapping[str, object]) -> int:
    if name == PRIMARY_SNAPSHOT_NAME:
        counters = _require_mapping(
            store_receipt.get("counters"), where="store receipt counters"
        )
        return require_int(counters["occurrence_count"], where="store occurrence count", minimum=1)
    if name == MEMBERSHIP_SNAPSHOT_NAME:
        frozen = _require_mapping(
            fetch_receipt.get("frozen_fetch_state"), where="fetch frozen state"
        )
        summary = _require_mapping(frozen.get("summary"), where="fetch summary")
        return require_int(summary["members"], where="fetch member count", minimum=1)
    return 1


def _publish_exact(path: Path, uri: str, *, object_store: Any, scratch_root: Path) -> dict[str, object]:
    published = dict(object_store.publish_if_absent(path, uri))
    generation = require_nonempty(published.get("generation"), where=f"published generation {uri}")
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError(f"published object has invalid generation: {uri}")
    if str(published.get("uri")) != uri or int(published.get("size_bytes", -1)) != path.stat().st_size:
        raise ContractError(f"published object metadata drifted: {uri}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case5-publish-", dir=scratch_root) as raw:
        verify_path = Path(raw) / "object"
        metadata = object_store.download(uri, verify_path, generation=generation)
        if (
            str(metadata.get("generation")) != generation
            or int(metadata.get("size_bytes", -1)) != path.stat().st_size
            or verify_path.stat().st_size != path.stat().st_size
            or sha256_file(verify_path) != sha256_file(path)
        ):
            raise ContractError(f"published object exact-generation verification failed: {uri}")
    return {
        "uri": uri,
        "generation": generation,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_immutable_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Write once locally; a resume may only reuse byte-compatible content."""

    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        _raw, existing = load_json_object(target, where="CASE5 snapshot receipt")
        if existing != dict(receipt):
            raise ContractError("existing CASE5 snapshot receipt differs from resume")
        return
    atomic_write_json(target, receipt)


def validate_snapshot_set(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the preparer's canonical receipt and return a deep copy."""

    receipt = copy.deepcopy(dict(value))
    require_exact_fields(
        receipt,
        {
            "schema",
            "status",
            "source_mode",
            "input_snapshots",
            "input_snapshot_set_sha256",
            "primary_record_count",
            "source_bindings",
            "production_complete",
            "training_ready",
            "receipt_sha256",
        },
        where="CASE5 snapshot receipt",
    )
    if receipt["schema"] != SNAPSHOT_SET_SCHEMA or receipt["status"] != "ready":
        raise ContractError("CASE5 snapshot receipt schema/status is unsupported")
    mode = _mode(receipt["source_mode"])
    snapshots = receipt["input_snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise ContractError("CASE5 snapshot receipt has no snapshots")
    names: list[str] = []
    roles: list[str] = []
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(snapshots):
        if not isinstance(raw, Mapping):
            raise ContractError(f"snapshot {index} is malformed")
        require_exact_fields(
            raw,
            {
                "name",
                "role",
                "uri",
                "generation",
                "size_bytes",
                "sha256",
                "content_set_sha256",
                "schema_sha256",
                "format",
                "record_count",
            },
            where=f"snapshot {index}",
        )
        name = require_nonempty(raw["name"], where=f"snapshot {index}.name")
        role = require_nonempty(raw["role"], where=f"snapshot {index}.role")
        if role not in _SNAPSHOT_ROLES or name in names:
            raise ContractError("CASE5 snapshot names/roles are invalid")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", name):
            raise ContractError(f"snapshot {index}.name is not canonical")
        generation = require_nonempty(raw["generation"], where=f"snapshot {index}.generation")
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError(f"snapshot {index}.generation is invalid")
        descriptor = {
            "name": name,
            "role": role,
            "uri": validate_gcs_uri(raw["uri"], where=f"snapshot {index}.uri"),
            "generation": generation,
            "size_bytes": require_int(raw["size_bytes"], where=f"snapshot {index}.size_bytes", minimum=1),
            "sha256": _require_hex(raw["sha256"], where=f"snapshot {index}.sha256"),
            "content_set_sha256": _require_hex(raw["content_set_sha256"], where=f"snapshot {index}.content_set_sha256"),
            "schema_sha256": _require_hex(raw["schema_sha256"], where=f"snapshot {index}.schema_sha256"),
            "format": require_nonempty(raw["format"], where=f"snapshot {index}.format"),
            "record_count": require_int(raw["record_count"], where=f"snapshot {index}.record_count", minimum=0),
        }
        if role in {"primary", "membership"} and descriptor["record_count"] < 1:
            raise ContractError(f"snapshot {index} has no records")
        names.append(name)
        roles.append(role)
        normalized.append(descriptor)
    if normalized != sorted(normalized, key=lambda item: str(item["name"])):
        raise ContractError("CASE5 snapshots are not name-sorted")
    if roles.count("primary") != 1 or roles.count("membership") != 1:
        raise ContractError("CASE5 snapshots require one primary and one membership")
    if canonical_sha256(normalized) != require_sha256(receipt["input_snapshot_set_sha256"], where="snapshot set digest"):
        raise ContractError("CASE5 snapshot set digest drifted")
    primary = next(item for item in normalized if item["role"] == "primary")
    if receipt["primary_record_count"] != primary["record_count"]:
        raise ContractError("CASE5 primary count drifted")
    bindings = receipt["source_bindings"]
    if not isinstance(bindings, Mapping):
        raise ContractError("CASE5 source bindings are missing")
    for field in ("store_receipt_sha256", "fetch_receipt_sha256", "tokenizer_sha256"):
        _require_hex(bindings.get(field), where=f"source_bindings.{field}")
    if mode == MODE_PRODUCTION:
        for field in ("inventory_sha256", "inventory_receipt_sha256", "merge_receipt_sha256"):
            _require_hex(bindings.get(field), where=f"source_bindings.{field}")
    elif any(field in bindings for field in ("inventory_sha256", "inventory_receipt_sha256", "merge_receipt_sha256")):
        raise ContractError("threshold receipt contains production bindings")
    if receipt["production_complete"] is not (mode == MODE_PRODUCTION) or receipt["training_ready"] is not False:
        raise ContractError("CASE5 snapshot readiness flags are invalid")
    digest = require_sha256(receipt["receipt_sha256"], where="snapshot receipt SHA-256")
    if snapshot_set_sha256(receipt) != digest:
        raise ContractError("CASE5 snapshot receipt digest drifted")
    receipt["input_snapshots"] = normalized
    return receipt


def prepare_ci_case5_snapshot(
    *,
    store_root: Path,
    store_receipt: Path,
    fetch_state: Path,
    fetch_receipt: Path,
    tokenizer: Path,
    object_store: Any,
    gcs_input_prefix: str,
    source_mode: str = MODE_THRESHOLD,
    inventory: Path | None = None,
    inventory_receipt: Path | None = None,
    merge_receipt: Path | None = None,
    scratch_root: Path | None = None,
    receipt_path: Path | None = None,
    publish_receipt: bool = False,
) -> dict[str, object]:
    """Validate and publish an immutable CASE5 snapshot set.

    ``source_mode`` is mandatory in spirit (the default is the explicitly
    non-production threshold mode).  Production evidence is never inferred
    from a file that happens to be present.
    """

    mode = _mode(source_mode)
    input_prefix = validate_gcs_uri(gcs_input_prefix.rstrip("/"), where="CASE5 input prefix")
    root = _resolve_input_path(store_root, where="content store", directory=True)
    store_receipt = _resolve_input_path(store_receipt, where="store receipt")
    fetch_state = _resolve_input_path(fetch_state, where="fetch state")
    fetch_receipt = _resolve_input_path(fetch_receipt, where="fetch receipt")
    tokenizer = _resolve_input_path(tokenizer, where="tokenizer")
    inventory = (
        None
        if inventory is None
        else _resolve_input_path(inventory, where="inventory")
    )
    inventory_receipt = (
        None
        if inventory_receipt is None
        else _resolve_input_path(inventory_receipt, where="inventory receipt")
    )
    merge_receipt = (
        None
        if merge_receipt is None
        else _resolve_input_path(merge_receipt, where="merge receipt")
    )
    _reject_sqlite_sidecars(fetch_state, where="fetch state")
    store_value, store_receipt_sha = _load_object(store_receipt, where="store receipt")
    store_value = _validate_store_receipt(store_value)
    _validate_store_artifacts(root, store_value)
    fetch_value, fetch_receipt_sha = _load_object(fetch_receipt, where="fetch receipt")
    fetch_value = _validate_fetch_receipt(
        fetch_value,
        mode=mode,
        store_receipt=store_value,
        fetch_state=fetch_state,
    )
    tokenizer_descriptor = _stable_descriptor(tokenizer, where="tokenizer")
    token_contract = fetch_value.get("tokenizer_contract")
    if not isinstance(token_contract, Mapping) or token_contract.get("artifact_sha256") != tokenizer_descriptor["sha256"]:
        raise ContractError("tokenizer bytes differ from fetch receipt contract")
    if mode == MODE_PRODUCTION:
        if inventory is None or inventory_receipt is None or merge_receipt is None:
            raise ContractError("inventory-exhaustive production inputs are unavailable")
        _reject_sqlite_sidecars(inventory, where="inventory")
        _verify_production_provenance(
            store_root=root,
            store_receipt=store_receipt,
            fetch_state=fetch_state,
            inventory=inventory,
            inventory_receipt=inventory_receipt,
            fetch_receipt=fetch_receipt,
            merge_receipt=merge_receipt,
        )
    paths = _source_paths(
        store_root=root,
        store_receipt=store_receipt,
        fetch_state=fetch_state,
        fetch_receipt=fetch_receipt,
        tokenizer=tokenizer,
        mode=mode,
        inventory=inventory,
        inventory_receipt=inventory_receipt,
        merge_receipt=merge_receipt,
    )
    stage_root = Path(scratch_root or tempfile.gettempdir()).expanduser().resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    snapshots: list[dict[str, object]] = []
    publications: list[dict[str, object]] = []
    for name, path, role in paths:
        descriptor = _stable_descriptor(path, where=f"CASE5 source {name}")
        uri = gcs_join(input_prefix, "ci-case5", mode, str(descriptor["sha256"]), name)
        published = _publish_exact(path, uri, object_store=object_store, scratch_root=stage_root)
        snapshot_format, schema_sha = _schema_for(
            name, store_receipt=store_value, fetch_receipt=fetch_value
        )
        snapshot = {
            "name": name,
            "role": role,
            "uri": uri,
            "generation": str(published["generation"]),
            "size_bytes": descriptor["size_bytes"],
            "sha256": descriptor["sha256"],
            "content_set_sha256": _content_set_for(name, store_receipt=store_value, fetch_receipt=fetch_value, descriptor=descriptor),
            "schema_sha256": schema_sha,
            "format": snapshot_format,
            "record_count": _record_count_for(name, store_receipt=store_value, fetch_receipt=fetch_value),
        }
        snapshots.append(snapshot)
        publications.append(published)
    snapshots.sort(key=lambda item: str(item["name"]))
    source_bindings: dict[str, object] = {
        "store_receipt_sha256": store_receipt_sha,
        "fetch_receipt_sha256": fetch_receipt_sha,
        "tokenizer_sha256": tokenizer_descriptor["sha256"],
    }
    for field, path in (
        ("inventory_sha256", inventory),
        ("inventory_receipt_sha256", inventory_receipt),
        ("merge_receipt_sha256", merge_receipt),
    ):
        if path is not None:
            source_bindings[field] = _stable_descriptor(Path(path), where=field)["sha256"]
    receipt: dict[str, object] = {
        "schema": SNAPSHOT_SET_SCHEMA,
        "status": "ready",
        "source_mode": mode,
        "input_snapshots": snapshots,
        "input_snapshot_set_sha256": canonical_sha256(snapshots),
        "primary_record_count": next(item["record_count"] for item in snapshots if item["role"] == "primary"),
        "source_bindings": source_bindings,
        "production_complete": mode == MODE_PRODUCTION,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = snapshot_set_sha256(receipt)
    receipt = validate_snapshot_set(receipt)
    if receipt_path is not None:
        _write_immutable_receipt(Path(receipt_path), receipt)
    receipt_publication: dict[str, object] | None = None
    if publish_receipt:
        if receipt_path is None:
            raise ContractError("publish_receipt requires receipt_path")
        receipt_uri = gcs_join(input_prefix, "ci-case5", mode, str(receipt["receipt_sha256"]), "snapshot-set.receipt.json")
        receipt_publication = _publish_exact(Path(receipt_path), receipt_uri, object_store=object_store, scratch_root=stage_root)
    return {
        "receipt": receipt,
        "input_snapshots": receipt["input_snapshots"],
        "input_snapshot_set_sha256": receipt["input_snapshot_set_sha256"],
        "publications": publications,
        "receipt_publication": receipt_publication,
        "training_ready": False,
    }


def partition_ci_occurrences(
    primary_record_count: int,
    *,
    records_per_item: int,
    input_snapshot_set_sha256: str,
) -> list[dict[str, object]]:
    """Create deterministic contiguous work items for the cloud lane."""

    total = require_int(primary_record_count, where="primary_record_count", minimum=1)
    size = require_int(records_per_item, where="records_per_item", minimum=1)
    snapshot_set = require_sha256(input_snapshot_set_sha256, where="input snapshot set SHA-256")
    items: list[dict[str, object]] = []
    start = 0
    while start < total:
        count = min(size, total - start)
        end = start + count
        payload = {
            "schema": "cppmega.ci_case5_occurrence_partition_v1",
            "input_snapshot_set_sha256": snapshot_set,
            "record_start": start,
            "record_count": count,
        }
        items.append(
            {
                "item_id": f"occurrences/{start:012d}-{end:012d}",
                "record_start": start,
                "record_count": count,
                "partition_sha256": canonical_sha256(payload),
            }
        )
        start = end
    return items


def build_ci_case5_manifest(
    snapshot_receipt: Mapping[str, object],
    *,
    worker_count: int,
    records_per_item: int,
    gcs_output_prefix: str,
    code_revision: str,
    adapter_path: Path | None = None,
    runner_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
    dataset_schema_sha256: str | None = None,
    membership_policy_sha256: str | None = None,
    candidate_schema_sha256: str | None = None,
) -> dict[str, object]:
    """Build the existing generic cloud-lane manifest from a CASE5 receipt."""

    receipt = validate_snapshot_set(snapshot_receipt)
    if adapter_path is not None:
        observed_runner = _stable_descriptor(Path(adapter_path), where="CASE5 adapter")["sha256"]
        if runner_sha256 is not None and require_sha256(runner_sha256, where="runner_sha256") != observed_runner:
            raise ContractError("runner_sha256 differs from adapter bytes")
        runner_sha256 = str(observed_runner)
    if runner_sha256 is None:
        raise ContractError("runner_sha256 or adapter_path is required")
    runner = require_sha256(runner_sha256, where="runner_sha256")
    bindings = _require_mapping(
        receipt.get("source_bindings"), where="snapshot receipt source_bindings"
    )
    tokenizer = require_sha256(tokenizer_sha256 or str(bindings["tokenizer_sha256"]), where="tokenizer_sha256")
    dataset = require_sha256(dataset_schema_sha256 or canonical_sha256({"schema": CASE5_DATASET_SCHEMA}), where="dataset_schema_sha256")
    policy = require_sha256(membership_policy_sha256 or canonical_sha256({"policy": CASE5_MEMBERSHIP_POLICY}), where="membership_policy_sha256")
    candidate = require_sha256(candidate_schema_sha256 or canonical_sha256({"schema": CASE5_PAYLOAD_SCHEMA}), where="candidate_schema_sha256")
    work_items = partition_ci_occurrences(
        int(receipt["primary_record_count"]),
        records_per_item=records_per_item,
        input_snapshot_set_sha256=str(receipt["input_snapshot_set_sha256"]),
    )
    return build_cloud_lane_manifest(
        kind="ci",
        input_snapshots=receipt["input_snapshots"],
        work_items=work_items,
        worker_count=worker_count,
        gcs_output_prefix=gcs_output_prefix,
        code_revision=require_git_object(code_revision, where="code_revision"),
        runner_sha256=runner,
        tokenizer_sha256=tokenizer,
        dataset_schema_sha256=dataset,
        membership_policy_sha256=policy,
        candidate_schema_sha256=candidate,
    )


def _request_snapshots(value: object) -> tuple[list[dict[str, object]], str]:
    if not isinstance(value, list) or not value:
        raise ContractError("adapter request snapshots must be a non-empty list")
    descriptors: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ContractError(f"adapter snapshot {index} is malformed")
        expected = {
            "name", "role", "uri", "generation", "size_bytes", "sha256",
            "content_set_sha256", "schema_sha256", "format", "record_count", "local_path",
        }
        require_exact_fields(raw, expected, where=f"adapter snapshot {index}")
        name = require_nonempty(raw["name"], where=f"adapter snapshot {index}.name")
        if name in names:
            raise ContractError("adapter snapshot names are duplicated")
        if _PACK_RE.fullmatch(name) is None and name not in _FIXED_NAMES:
            raise ContractError(f"adapter snapshot name is unsupported: {name}")
        expected_role = (
            "primary"
            if name == PRIMARY_SNAPSHOT_NAME
            else "membership"
            if name == MEMBERSHIP_SNAPSHOT_NAME
            else "ancillary"
        )
        if raw["role"] != expected_role:
            raise ContractError(f"adapter snapshot role drifted: {name}")
        local = Path(require_nonempty(raw["local_path"], where=f"adapter snapshot {index}.local_path")).expanduser()
        if local.is_symlink() or not local.is_file():
            raise ContractError(f"adapter snapshot local path is unsafe: {name}")
        descriptor = {key: raw[key] for key in expected if key != "local_path"}
        # The full receipt validator requires one primary and one membership;
        # validate request descriptors field-by-field here instead.
        validate_gcs_uri(raw["uri"], where=f"adapter snapshot {index}.uri")
        generation = require_nonempty(raw["generation"], where=f"adapter snapshot {index}.generation")
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError("adapter snapshot generation is invalid")
        require_int(raw["size_bytes"], where=f"adapter snapshot {index}.size_bytes", minimum=1)
        record_count = require_int(
            raw["record_count"],
            where=f"adapter snapshot {index}.record_count",
            minimum=0,
        )
        if expected_role in {"primary", "membership"} and record_count < 1:
            raise ContractError(f"adapter snapshot has no {expected_role} records")
        _require_hex(raw["sha256"], where=f"adapter snapshot {index}.sha256")
        _require_hex(raw["content_set_sha256"], where=f"adapter snapshot {index}.content_set_sha256")
        _require_hex(raw["schema_sha256"], where=f"adapter snapshot {index}.schema_sha256")
        if _stable_descriptor(local, where=f"adapter snapshot {name}") != {
            "size_bytes": raw["size_bytes"],
            "sha256": raw["sha256"],
        }:
            raise ContractError(f"adapter snapshot {name} bytes differ from manifest")
        descriptor["local_path"] = str(local)
        descriptors.append(descriptor)
        names.add(name)
    if descriptors != sorted(descriptors, key=lambda item: str(item["name"])):
        raise ContractError("adapter snapshots are not name-sorted")
    plain = [{key: item[key] for key in item if key != "local_path"} for item in descriptors]
    return descriptors, canonical_sha256(plain)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"adapter output contains duplicate key {key!r}")
        result[key] = value
    return result


@contextmanager
def _snapshot_layout(snapshots: Sequence[Mapping[str, object]], *, parent: Path) -> Iterator[dict[str, Path]]:
    """Materialize a flat snapshot list into the immutable CASE5 directory layout."""

    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case5-layout-", dir=parent) as raw:
        root = Path(raw)
        store_root = root / "content_store"
        store_root.mkdir()
        mapped: dict[str, Path] = {}
        for snapshot in snapshots:
            name = str(snapshot["name"])
            source = Path(str(snapshot["local_path"])).expanduser()
            if name in {PRIMARY_SNAPSHOT_NAME}:
                destination = store_root / "index.sqlite3"
            elif name == MEMBERSHIP_SNAPSHOT_NAME:
                destination = root / "fetch_state.sqlite3"
            elif _PACK_RE.fullmatch(name):
                destination = store_root / name
            elif name in _FIXED_NAMES - {PRIMARY_SNAPSHOT_NAME, MEMBERSHIP_SNAPSHOT_NAME}:
                destination = root / name
            else:
                raise ContractError(f"adapter snapshot layout/name is unsupported: {name}")
            if destination.exists() or destination.is_symlink():
                raise ContractError(f"adapter snapshot layout has duplicate destination: {name}")
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            if destination.is_symlink() or not destination.is_file():
                raise ContractError(f"adapter failed to materialize snapshot: {name}")
            mapped[name] = destination
        required = {PRIMARY_SNAPSHOT_NAME, MEMBERSHIP_SNAPSHOT_NAME, STORE_RECEIPT_NAME, FETCH_RECEIPT_NAME, TOKENIZER_NAME}
        if not required.issubset(mapped):
            raise ContractError("adapter snapshot set is missing required CASE5 artifacts")
        yield {"root": root, "store_root": store_root, **mapped}


def _read_range(connection: sqlite3.Connection, start: int, count: int) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT repo,run_attempt,job,step,chunk_ordinal,
               content_sha256,provenance_sha256,
               provenance_raw_size,provenance_zlib
        FROM occurrences
        ORDER BY repo,run_attempt,job,step,chunk_ordinal
        LIMIT ? OFFSET ?
        """,
        (count, start),
    ).fetchall()
    return list(rows)


def _candidate_payload(
    *,
    occurrence: Any,
    content: Any,
    content_bytes: bytes,
    member: Any,
    store_receipt_sha256: str,
    fetch_receipt_sha256: str,
) -> dict[str, object]:
    try:
        text = content_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ContractError("CASE5 content frame is not strict UTF-8") from exc
    return {
        "schema": CASE5_PAYLOAD_SCHEMA,
        "occurrence_key": occurrence.key_dict,
        "content": {
            "sha256": content.sha256,
            "raw_size": content.raw_size,
            "text": text,
        },
        "token": {
            "count": content.token_count,
            "sequence_sha256": content.token_sequence_sha256,
            "tokenizer_fingerprint": content.tokenizer_fingerprint,
        },
        "provenance_sha256": occurrence.provenance_sha256,
        "provenance": occurrence.provenance,
        "membership": {
            "key": {
                "repository": member.key[0],
                "run_id": member.key[1],
                "attempt": member.key[2],
                "archive_member": member.key[3],
            },
            "job_key": member.job_key,
            "job": member.job,
            "raw_sha256": member.raw_sha256,
            "raw_size": member.raw_size,
            "canonical_sha256": member.canonical_sha256,
            "dedup_sha256": member.dedup_sha256,
            "sidecar_sha256": member.sidecar_sha256,
            "chunk_count": member.chunk_count,
            "occurrence_tokens": member.occurrence_tokens,
            "sidecar": member.sidecar,
            "opaque": member.opaque,
            "exclusion_reason": member.exclusion_reason,
            "decode_status": member.decode_status,
            "invalid_sequence_count": member.invalid_sequence_count,
            "replacement_char_count": member.replacement_char_count,
            "invalid_ratio_ppm": member.invalid_ratio_ppm,
        },
        "source_bindings": {
            "store_receipt_sha256": store_receipt_sha256,
            "fetch_receipt_sha256": fetch_receipt_sha256,
        },
    }


def _validate_snapshot_semantics(
    snapshots: Sequence[Mapping[str, object]],
    *,
    mode: str,
    store_receipt: Mapping[str, object],
    fetch_receipt: Mapping[str, object],
) -> None:
    """Bind lane descriptors back to the receipts reconstructed by the adapter."""

    by_name = {str(snapshot["name"]): snapshot for snapshot in snapshots}
    pack_hashes = store_receipt.get("pack_hashes")
    if not isinstance(pack_hashes, list):
        raise ContractError("CASE5 store receipt pack_hashes is missing")
    pack_names = {str(item["filename"]) for item in pack_hashes if isinstance(item, Mapping)}
    expected_names = {
        PRIMARY_SNAPSHOT_NAME,
        MEMBERSHIP_SNAPSHOT_NAME,
        STORE_RECEIPT_NAME,
        FETCH_RECEIPT_NAME,
        TOKENIZER_NAME,
        *pack_names,
    }
    if mode == MODE_PRODUCTION:
        expected_names.update(
            {INVENTORY_NAME, INVENTORY_RECEIPT_NAME, MERGE_RECEIPT_NAME}
        )
    if set(by_name) != expected_names:
        raise ContractError("CASE5 adapter snapshot layout/name drifted")
    for name, snapshot in by_name.items():
        expected_format, expected_schema = _schema_for(
            name, store_receipt=store_receipt, fetch_receipt=fetch_receipt
        )
        if (
            snapshot.get("format") != expected_format
            or snapshot.get("schema_sha256") != expected_schema
            or int(snapshot["record_count"])
            != _record_count_for(
                name, store_receipt=store_receipt, fetch_receipt=fetch_receipt
            )
        ):
            raise ContractError(f"CASE5 adapter snapshot descriptor drifted: {name}")
        descriptor = {
            "size_bytes": snapshot["size_bytes"],
            "sha256": snapshot["sha256"],
        }
        if _stable_descriptor(
            Path(str(snapshot["local_path"])), where=f"adapter snapshot {name}"
        ) != descriptor:
            raise ContractError(f"CASE5 adapter snapshot bytes drifted: {name}")
        expected_content_set = _content_set_for(
            name,
            store_receipt=store_receipt,
            fetch_receipt=fetch_receipt,
            descriptor=descriptor,
        )
        if snapshot.get("content_set_sha256") != expected_content_set:
            raise ContractError(f"CASE5 adapter snapshot content-set drifted: {name}")
    for item in pack_hashes:
        item = _require_mapping(item, where="CASE5 store receipt pack")
        name = str(item["filename"])
        snapshot = by_name[name]
        if (
            snapshot["sha256"] != item.get("sha256")
            or snapshot["size_bytes"] != item.get("committed_end")
        ):
            raise ContractError(f"CASE5 adapter pack binding drifted: {name}")


def run_ci_case5_adapter(*, request_path: Path, output_path: Path) -> dict[str, object]:
    """Run the CASE5 adapter protocol used by ``cloud_lane_worker``."""

    _raw, request = load_json_object(Path(request_path), where="CASE5 adapter request")
    require_exact_fields(
        request,
        {
            "schema", "kind", "manifest_sha256", "input_snapshot_set_sha256",
            "assignment", "snapshots", "output_schema", "training_ready",
        },
        where="CASE5 adapter request",
    )
    if request["schema"] != ADAPTER_REQUEST_SCHEMA or request["kind"] != "ci" or request["output_schema"] != ADAPTER_OUTPUT_SCHEMA or request["training_ready"] is not False:
        raise ContractError("CASE5 adapter request contract is unsupported")
    manifest_sha = require_sha256(request["manifest_sha256"], where="manifest_sha256")
    snapshot_set = require_sha256(request["input_snapshot_set_sha256"], where="input snapshot set SHA-256")
    assignment = request["assignment"]
    if not isinstance(assignment, Mapping):
        raise ContractError("CASE5 adapter assignment is malformed")
    require_exact_fields(
        assignment,
        {"ordinal", "item_id", "record_start", "record_count", "partition_sha256", "worker", "assignment_sha256"},
        where="CASE5 adapter assignment",
    )
    start = require_int(assignment["record_start"], where="assignment.record_start", minimum=0)
    count = require_int(assignment["record_count"], where="assignment.record_count", minimum=1)
    partition_sha = require_sha256(assignment["partition_sha256"], where="assignment.partition_sha256")
    require_sha256(assignment["assignment_sha256"], where="assignment.assignment_sha256")
    ordinal = require_int(assignment["ordinal"], where="assignment.ordinal", minimum=0)
    item_id = require_nonempty(assignment["item_id"], where="assignment.item_id")
    require_nonempty(assignment["worker"], where="assignment.worker")
    snapshots, observed_set = _request_snapshots(request["snapshots"])
    if observed_set != snapshot_set:
        raise ContractError("adapter snapshot set digest differs from manifest")
    expected_partition = canonical_sha256(
        {
            "schema": "cppmega.ci_case5_occurrence_partition_v1",
            "input_snapshot_set_sha256": snapshot_set,
            "record_start": start,
            "record_count": count,
        }
    )
    expected_item_id = f"occurrences/{start:012d}-{start + count:012d}"
    if partition_sha != expected_partition or item_id != expected_item_id:
        raise ContractError("CASE5 assignment partition binding drifted")
    output = Path(output_path).expanduser()
    if output.is_symlink() or output.exists():
        raise ContractError(f"CASE5 adapter output must be a new regular file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _snapshot_layout(snapshots, parent=output.parent) as layout:
        store_receipt_value, store_receipt_sha = _load_object(layout[STORE_RECEIPT_NAME], where="CASE5 store receipt")
        store_receipt_value = _validate_store_receipt(store_receipt_value)
        fetch_receipt_value, fetch_receipt_sha = _load_object(layout[FETCH_RECEIPT_NAME], where="CASE5 fetch receipt")
        mode = MODE_PRODUCTION if fetch_receipt_value.get("schema") == _FETCH_RECEIPT_PRODUCTION_SCHEMA else MODE_THRESHOLD
        if mode == MODE_PRODUCTION and not {
            INVENTORY_NAME,
            INVENTORY_RECEIPT_NAME,
            MERGE_RECEIPT_NAME,
        }.issubset(layout):
            raise ContractError("production adapter snapshot set lacks inventory/merge evidence")
        fetch_receipt_value = _validate_fetch_receipt(
            fetch_receipt_value,
            mode=mode,
            store_receipt=store_receipt_value,
            fetch_state=layout[MEMBERSHIP_SNAPSHOT_NAME],
        )
        token_contract = fetch_receipt_value.get("tokenizer_contract")
        tokenizer_path = layout[TOKENIZER_NAME]
        tokenizer_descriptor = _stable_descriptor(tokenizer_path, where="CASE5 tokenizer")
        if not isinstance(token_contract, Mapping) or token_contract.get("artifact_sha256") != tokenizer_descriptor["sha256"]:
            raise ContractError("CASE5 tokenizer snapshot differs from fetch receipt")
        _validate_snapshot_semantics(
            snapshots,
            mode=mode,
            store_receipt=store_receipt_value,
            fetch_receipt=fetch_receipt_value,
        )
        # Imports are lazy so receipt-only tooling does not need the optional
        # tokenizer/Parquet stack until an adapter actually executes.
        try:
            from scripts.ci_stream_fetch import ExactTokenizer
            from scripts.export_ci_content_store_case5 import FrozenFetchState, FrozenStore
        except (ImportError, ModuleNotFoundError) as exc:
            raise ContractError("CASE5 exporter dependencies are unavailable") from exc
        try:
            tokenizer_adapter = ExactTokenizer(tokenizer_path)
            with FrozenStore(layout["store_root"], layout[STORE_RECEIPT_NAME]) as store:
                settings = sqlite3.connect(
                    f"{layout[MEMBERSHIP_SNAPSHOT_NAME].as_uri()}?mode=ro&immutable=1",
                    uri=True,
                )
                try:
                    settings.row_factory = sqlite3.Row
                    row = settings.execute("SELECT value FROM settings WHERE key='content_store_path'").fetchone()
                    bound_path = Path(str(row[0])) if row is not None else layout["store_root"]
                finally:
                    settings.close()
                with FrozenFetchState(
                    layout[MEMBERSHIP_SNAPSHOT_NAME],
                    tokenizer=tokenizer_adapter,
                    store=store,
                    bound_store_path=bound_path,
                ) as frozen_fetch:
                    rows = _read_range(store.connection, start, count)
                    if len(rows) != count:
                        raise ContractError("CASE5 assignment range exceeds primary occurrence set")
                    with output.open("xb") as stream:
                        previous_key: tuple[str, str, str, str, int] | None = None
                        for ordinal, row in enumerate(rows, start):
                            occurrence = store._occurrence_record(row)
                            if previous_key is not None and occurrence.key <= previous_key:
                                raise ContractError("CASE5 occurrence order is not canonical")
                            previous_key = occurrence.key
                            member = frozen_fetch.validate_occurrence(occurrence)
                            content = store.get_content_record(occurrence.content_sha256)
                            content_bytes = store.read_content(content)
                            payload = _candidate_payload(
                                occurrence=occurrence,
                                content=content,
                                content_bytes=content_bytes,
                                member=member,
                                store_receipt_sha256=store_receipt_sha,
                                fetch_receipt_sha256=fetch_receipt_sha,
                            )
                            envelope = {
                                "schema": ADAPTER_OUTPUT_SCHEMA,
                                "source_record_ordinal": ordinal,
                                "document_ordinal": 0,
                                "valid_tokens": content.token_count,
                                "payload": payload,
                            }
                            stream.write(canonical_json_bytes(envelope) + b"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    store.require_unchanged()
                    frozen_fetch.require_unchanged()
        except ContractError:
            raise
        except Exception as exc:
            # The exporter uses ExportError (a RuntimeError subclass).  Keep
            # adapter failures on the lane's stable ContractError boundary and
            # do not expose receipt contents or credentials in diagnostics.
            raise ContractError(
                "CASE5 immutable adapter validation failed: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            ) from exc
    for snapshot in snapshots:
        path = Path(str(snapshot["local_path"]))
        if _stable_descriptor(path, where=f"adapter snapshot {snapshot['name']}") != {
            "size_bytes": snapshot["size_bytes"],
            "sha256": snapshot["sha256"],
        }:
            raise ContractError("CASE5 input snapshot changed during adapter execution")
    return {
        "schema": CASE5_ADAPTER_SCHEMA,
        "manifest_sha256": manifest_sha,
        "input_snapshot_set_sha256": snapshot_set,
        "record_start": start,
        "record_count": count,
        "assignment_ordinal": ordinal,
        "output": str(output),
        "training_ready": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--store-receipt", type=Path)
    parser.add_argument("--fetch-state", type=Path)
    parser.add_argument("--fetch-receipt", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--source-mode",
        choices=(MODE_THRESHOLD, MODE_PRODUCTION, "production"),
        default=MODE_THRESHOLD,
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-receipt", type=Path)
    parser.add_argument("--merge-receipt", type=Path)
    parser.add_argument("--gcs-input-prefix")
    parser.add_argument("--snapshot-receipt", type=Path)
    parser.add_argument("--publish-receipt", action="store_true")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gcs-output-prefix")
    parser.add_argument("--code-revision")
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--records-per-item", type=int)
    args = parser.parse_args(argv)
    adapter_requested = args.request is not None or args.output is not None
    if adapter_requested and (args.request is None or args.output is None or args.prepare):
        parser.error("adapter mode requires --request and --output only")
    if not adapter_requested and not args.prepare:
        parser.error("use --request/--output for adapter mode or --prepare")
    try:
        if adapter_requested:
            run_ci_case5_adapter(request_path=args.request, output_path=args.output)
            return 0
        required = {
            "--store-root": args.store_root,
            "--store-receipt": args.store_receipt,
            "--fetch-state": args.fetch_state,
            "--fetch-receipt": args.fetch_receipt,
            "--tokenizer": args.tokenizer,
            "--gcs-input-prefix": args.gcs_input_prefix,
            "--snapshot-receipt": args.snapshot_receipt,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error("--prepare requires " + ", ".join(missing))
        from scripts.distributed_data_prep.source_worker import GcloudObjectStore

        prepared = prepare_ci_case5_snapshot(
            store_root=args.store_root,
            store_receipt=args.store_receipt,
            fetch_state=args.fetch_state,
            fetch_receipt=args.fetch_receipt,
            tokenizer=args.tokenizer,
            object_store=GcloudObjectStore(),
            gcs_input_prefix=str(args.gcs_input_prefix),
            source_mode=args.source_mode,
            inventory=args.inventory,
            inventory_receipt=args.inventory_receipt,
            merge_receipt=args.merge_receipt,
            scratch_root=args.scratch_root,
            receipt_path=args.snapshot_receipt,
            publish_receipt=args.publish_receipt,
        )
        if args.manifest is not None:
            manifest_required = {
                "--gcs-output-prefix": args.gcs_output_prefix,
                "--code-revision": args.code_revision,
                "--worker-count": args.worker_count,
                "--records-per-item": args.records_per_item,
            }
            missing_manifest = [
                flag for flag, value in manifest_required.items() if value is None
            ]
            if missing_manifest:
                parser.error(
                    "--manifest requires " + ", ".join(missing_manifest)
                )
            manifest = build_ci_case5_manifest(
                prepared["receipt"],
                worker_count=int(args.worker_count),
                records_per_item=int(args.records_per_item),
                gcs_output_prefix=str(args.gcs_output_prefix),
                code_revision=str(args.code_revision),
                adapter_path=Path(__file__).resolve(),
            )
            atomic_write_json(args.manifest, manifest)
        print(
            canonical_json_bytes(
                {
                    "snapshot_receipt_sha256": prepared["receipt"]["receipt_sha256"],
                    "input_snapshot_set_sha256": prepared[
                        "input_snapshot_set_sha256"
                    ],
                    "training_ready": False,
                }
            ).decode("ascii")
        )
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"CASE5 snapshot operation failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "ADAPTER_OUTPUT_SCHEMA",
    "CASE5_PAYLOAD_SCHEMA",
    "COMPLETION_MODE_INVENTORY_EXHAUSTIVE",
    "COMPLETION_MODE_THRESHOLD",
    "MEMBERSHIP_SNAPSHOT_NAME",
    "MODE_PRODUCTION",
    "MODE_THRESHOLD",
    "PRIMARY_SNAPSHOT_NAME",
    "SNAPSHOT_RECEIPT_SCHEMA",
    "SNAPSHOT_SET_SCHEMA",
    "build_ci_case5_manifest",
    "partition_ci_occurrences",
    "prepare_ci_case5_snapshot",
    "run_ci_case5_adapter",
    "snapshot_receipt_sha256",
    "snapshot_set_sha256",
    "validate_snapshot_set",
]
