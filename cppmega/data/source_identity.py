"""Canonical uint64 source identities and their collision-checking registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


MAX_ROW_LOCAL_DOC_ID = (1 << 32) - 1
MAX_SOURCE_ID = (1 << 64) - 1


@dataclass(frozen=True)
class SourceIdentity:
    source_identity_id: int
    canonical_sha256: str
    source: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "source_identity_id": self.source_identity_id,
            "canonical_sha256": self.canonical_sha256,
            "source": self.source,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_source(record: Mapping[str, Any]) -> str:
    """Return the full canonical provenance string used by the registry."""

    fields = {
        key: record.get(key)
        for key in (
            "repo_stable_id",
            "filepath_stable_id",
            "repo",
            "filepath",
            "source_path",
        )
        if record.get(key) not in (None, "")
    }
    legacy = record.get("source_doc_id")
    if not fields and isinstance(legacy, str) and legacy:
        fields["legacy_source_doc_id"] = legacy
    if fields:
        return _canonical_json(fields)

    content = record.get("text", record.get("source_text"))
    if content is None:
        content = record.get("token_ids", [])
    serialized = content if isinstance(content, str) else _canonical_json(content)
    return _canonical_json(
        {"content_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}
    )


def _id_from_digest(digest: bytes) -> int:
    value = int.from_bytes(digest[:8], "big", signed=False)
    if value == 0:
        value = int.from_bytes(digest[8:16], "big", signed=False)
    if value == 0:
        raise ValueError("SHA256 digest cannot produce reserved source identity 0")
    return value


def source_identity(record: Mapping[str, Any]) -> SourceIdentity:
    """Build an identity with a uint64 key and full SHA256 collision witness."""

    source = canonical_source(record)
    digest_bytes = hashlib.sha256(source.encode("utf-8")).digest()
    canonical_sha256 = digest_bytes.hex()
    derived_id = _id_from_digest(digest_bytes)

    explicit = record.get("source_identity_id")
    if explicit is not None and int(explicit) != derived_id:
        raise ValueError(
            "source_identity_id does not match canonical source SHA256: "
            f"provided={int(explicit)} derived={derived_id}"
        )
    return SourceIdentity(derived_id, canonical_sha256, source)


def stable_source_identity_id(record: Mapping[str, Any]) -> int:
    return source_identity(record).source_identity_id


def source_identity_for_path(
    path: str | Path,
    *,
    text: str = "",
    source_root: str | Path | None = None,
) -> SourceIdentity:
    path_obj = Path(path)
    if source_root is not None:
        try:
            source_path = path_obj.resolve().relative_to(Path(source_root).resolve()).as_posix()
        except (OSError, ValueError):
            source_path = path_obj.as_posix()
    else:
        source_path = path_obj.as_posix()
    return source_identity({"source_path": source_path, "text": text})


def stable_source_identity_for_path(
    path: str | Path,
    *,
    text: str = "",
    source_root: str | Path | None = None,
) -> int:
    return source_identity_for_path(
        path,
        text=text,
        source_root=source_root,
    ).source_identity_id


def normalize_positive_source_ids(
    values: Sequence[Any] | None,
    *,
    length: int,
    fallback_source_id: int,
) -> list[int]:
    """Validate a uint64 source-identity vector and replace legacy zero sentinels."""

    fallback = int(fallback_source_id)
    if not 0 < fallback <= MAX_SOURCE_ID:
        raise ValueError(f"fallback_source_id must be uint64 and positive, got {fallback}")
    if not values:
        return [fallback] * length
    if len(values) != length:
        raise ValueError(f"source identity vector length {len(values)} != expected {length}")
    normalized: list[int] = []
    for raw in values:
        value = int(raw)
        if value < 0 or value > MAX_SOURCE_ID:
            raise ValueError(f"source identity must be uint64, got {value}")
        normalized.append(value or fallback)
    return normalized


def normalize_row_local_doc_ids(
    values: Sequence[Any] | None,
    *,
    length: int,
    fallback_doc_id: int = 1,
) -> list[int]:
    fallback = int(fallback_doc_id)
    if not 0 < fallback <= MAX_ROW_LOCAL_DOC_ID:
        raise ValueError(f"fallback_doc_id must be positive uint32, got {fallback}")
    if not values:
        return [fallback] * length
    if len(values) != length:
        raise ValueError(f"row-local doc vector length {len(values)} != expected {length}")
    result: list[int] = []
    for raw in values:
        value = int(raw)
        if value < 0 or value > MAX_ROW_LOCAL_DOC_ID:
            raise ValueError(f"row-local doc id must be uint32, got {value}")
        result.append(value or fallback)
    return result


def validate_source_identity_registry(
    entries: Sequence[Mapping[str, Any]],
    *,
    referenced_ids: Sequence[Any] = (),
) -> dict[int, SourceIdentity]:
    """Validate full digests, canonical strings, collisions, and registry FKs."""

    registry: dict[int, SourceIdentity] = {}
    for raw in entries:
        source = raw.get("source")
        digest = raw.get("canonical_sha256")
        if not isinstance(source, str) or not source:
            raise ValueError("source identity registry entry has empty source")
        if not isinstance(digest, str):
            raise ValueError("source identity registry entry has invalid SHA256 digest")
        expected_bytes = hashlib.sha256(source.encode("utf-8")).digest()
        expected_digest = expected_bytes.hex()
        expected_id = _id_from_digest(expected_bytes)
        identity_id = int(raw.get("source_identity_id", 0))
        if digest != expected_digest or identity_id != expected_id:
            raise ValueError(
                "source identity registry entry does not match canonical source: "
                f"id={identity_id} digest={digest!r}"
            )
        identity = SourceIdentity(identity_id, digest, source)
        previous = registry.get(identity_id)
        if previous is not None and previous != identity:
            raise ValueError(f"source identity uint64 collision for id {identity_id}")
        registry[identity_id] = identity

    missing = sorted({int(value) for value in referenced_ids if int(value) not in registry})
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise ValueError(f"source identity registry missing referenced ids: {preview}")
    return registry


__all__ = [
    "MAX_ROW_LOCAL_DOC_ID",
    "MAX_SOURCE_ID",
    "SourceIdentity",
    "canonical_source",
    "normalize_positive_source_ids",
    "normalize_row_local_doc_ids",
    "source_identity",
    "source_identity_for_path",
    "stable_source_identity_for_path",
    "stable_source_identity_id",
    "validate_source_identity_registry",
]
