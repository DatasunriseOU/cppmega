"""Canonical identity binding for CASE6 production receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any

RECEIPT_BINDING_SCHEMA = "cppmega_case6_receipt_binding_v1"
NO_CHECKPOINT_SHA256 = hashlib.sha256(b"cppmega:no-checkpoint:v1").hexdigest()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIELDS = (
    "schema",
    "bundle_id",
    "artifact_set_sha256",
    "prefix_manifest_sha256s",
    "checkpoint_sha256",
    "config_sha256",
    "command_sha256",
    "run_id",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def validate_binding_shape(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} binding must be an object")
    binding = dict(value)
    if set(binding) != set(_FIELDS):
        raise ValueError(
            f"{where} binding fields mismatch: "
            f"missing={sorted(set(_FIELDS) - set(binding))} "
            f"extra={sorted(set(binding) - set(_FIELDS))}"
        )
    if binding["schema"] != RECEIPT_BINDING_SCHEMA:
        raise ValueError(f"{where} binding schema mismatch")
    bundle_id = binding["bundle_id"]
    if not isinstance(bundle_id, str) or not _RUN_ID_RE.fullmatch(bundle_id):
        raise ValueError(f"{where} binding bundle_id is invalid")
    run_id = binding["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"{where} binding run_id is invalid")
    for field in (
        "artifact_set_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "command_sha256",
    ):
        _require_sha256(binding[field], field=f"{where} binding {field}")
    prefix_hashes = binding["prefix_manifest_sha256s"]
    if not isinstance(prefix_hashes, Mapping) or not prefix_hashes:
        raise ValueError(
            f"{where} binding prefix_manifest_sha256s must be a nonempty object"
        )
    normalized_prefixes: dict[str, str] = {}
    for path, digest in prefix_hashes.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError(f"{where} binding has unsafe prefix manifest path")
        normalized_prefixes[path] = _require_sha256(
            digest, field=f"{where} binding prefix {path}"
        )
    binding["prefix_manifest_sha256s"] = dict(sorted(normalized_prefixes.items()))
    return binding


def build_receipt_binding(
    *,
    bundle_id: str,
    artifact_set_sha256: str,
    prefix_manifest_sha256s: Mapping[str, str],
    checkpoint_sha256: str,
    config: Mapping[str, Any],
    command: Sequence[str],
    run_id: str,
) -> dict[str, object]:
    if not command or any(not isinstance(part, str) for part in command):
        raise ValueError("receipt command must be a nonempty string sequence")
    binding: dict[str, object] = {
        "schema": RECEIPT_BINDING_SCHEMA,
        "bundle_id": bundle_id,
        "artifact_set_sha256": artifact_set_sha256,
        "prefix_manifest_sha256s": dict(prefix_manifest_sha256s),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": canonical_sha256(dict(config)),
        "command_sha256": canonical_sha256(list(command)),
        "run_id": run_id,
    }
    return validate_binding_shape(binding, where="constructed receipt")


def validate_receipt_binding(
    actual: object,
    *,
    expected: object,
    where: str,
) -> dict[str, object]:
    actual_binding = validate_binding_shape(actual, where=where)
    expected_binding = validate_binding_shape(expected, where="expected")
    for field in _FIELDS:
        if actual_binding[field] != expected_binding[field]:
            raise RuntimeError(f"{where} binding mismatch for {field}")
    return actual_binding


__all__ = [
    "NO_CHECKPOINT_SHA256",
    "RECEIPT_BINDING_SCHEMA",
    "build_receipt_binding",
    "canonical_sha256",
    "validate_binding_shape",
    "validate_receipt_binding",
]
