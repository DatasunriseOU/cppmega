"""Canonical identity binding for CASE6 production receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any

RECEIPT_BINDING_SCHEMA = "cppmega_case6_receipt_binding_v2"
IMPLEMENTATION_BINDING_SCHEMA = "cppmega_implementation_binding_v1"
NO_CHECKPOINT_SHA256 = hashlib.sha256(b"cppmega:no-checkpoint:v1").hexdigest()
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$|^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMPLEMENTATION_COMPONENTS = (
    "cppmega",
    "megatron",
    "cppmega_mlx",
    "clang_indexer",
)
_FIELDS = (
    "schema",
    "bundle_id",
    "artifact_set_sha256",
    "prefix_manifest_sha256s",
    "checkpoint_sha256",
    "config_sha256",
    "command_sha256",
    "run_id",
    "implementation",
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


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase Git commit SHA")
    return value


def validate_implementation_binding(
    value: object,
    *,
    where: str,
    required_components: Sequence[str] = _IMPLEMENTATION_COMPONENTS,
) -> dict[str, object]:
    """Validate the source implementation identities carried by a receipt.

    The graph sidecars are produced by a different checkout from the CUDA
    consumer.  A bundle receipt therefore has to bind both repositories and
    the exact indexer dependency closure; a bare repository name or a single
    cppmega commit is insufficient evidence.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{where} implementation binding must be an object")
    binding = dict(value)
    expected = {"schema", "components"}
    if set(binding) != expected:
        raise ValueError(
            f"{where} implementation binding fields mismatch: "
            f"missing={sorted(expected - set(binding))} "
            f"extra={sorted(set(binding) - expected)}"
        )
    if binding["schema"] != IMPLEMENTATION_BINDING_SCHEMA:
        raise ValueError(f"{where} implementation binding schema mismatch")
    components = binding["components"]
    if not isinstance(components, Mapping):
        raise ValueError(f"{where} implementation binding components must be an object")
    component_names = set(components)
    required = set(required_components)
    unknown_required = required - set(_IMPLEMENTATION_COMPONENTS)
    if unknown_required:
        raise ValueError(
            f"{where} implementation binding has unsupported required components: "
            f"{sorted(unknown_required)}"
        )
    missing = required - component_names
    if missing:
        raise ValueError(
            f"{where} implementation binding is missing components: {sorted(missing)}"
        )
    normalized: dict[str, dict[str, str]] = {}
    for name, raw_component in components.items():
        if not isinstance(name, str) or not _RUN_ID_RE.fullmatch(name):
            raise ValueError(f"{where} implementation component name is invalid")
        if not isinstance(raw_component, Mapping):
            raise ValueError(f"{where} implementation component {name} must be an object")
        component = dict(raw_component)
        allowed = {"commit", "tree_sha256", "source_sha256", "dependency_closure_sha256"}
        if set(component) - allowed:
            raise ValueError(
                f"{where} implementation component {name} has unknown fields: "
                f"{sorted(set(component) - allowed)}"
            )
        if "commit" in component:
            component["commit"] = _require_commit(
                component["commit"], field=f"{where} implementation {name}.commit"
            )
        for field in ("tree_sha256", "source_sha256", "dependency_closure_sha256"):
            if field in component:
                component[field] = _require_sha256(
                    component[field],
                    field=f"{where} implementation {name}.{field}",
                )
        if name in {"cppmega", "cppmega_mlx"} and name in required and not {
            "commit",
            "tree_sha256",
        }.issubset(component):
            raise ValueError(
                f"{where} implementation {name} requires commit and tree_sha256"
            )
        if name == "megatron" and name in required and "commit" not in component:
            raise ValueError(f"{where} implementation megatron requires commit")
        if name == "clang_indexer" and name in required and not {
            "source_sha256",
            "dependency_closure_sha256",
        }.issubset(component):
            raise ValueError(
                f"{where} implementation clang_indexer requires source_sha256 "
                "and dependency_closure_sha256"
            )
        normalized[name] = dict(sorted(component.items()))
    binding["components"] = dict(sorted(normalized.items()))
    return binding


def build_implementation_binding(
    *,
    cppmega_commit: str,
    cppmega_tree_sha256: str,
    megatron_commit: str,
    cppmega_mlx_commit: str,
    cppmega_mlx_tree_sha256: str,
    clang_indexer_sha256: str,
    clang_indexer_dependency_closure_sha256: str,
    optional_components: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, object]:
    components: dict[str, Mapping[str, str]] = {
        "cppmega": {
            "commit": cppmega_commit,
            "tree_sha256": cppmega_tree_sha256,
        },
        "megatron": {"commit": megatron_commit},
        "cppmega_mlx": {
            "commit": cppmega_mlx_commit,
            "tree_sha256": cppmega_mlx_tree_sha256,
        },
        "clang_indexer": {
            "source_sha256": clang_indexer_sha256,
            "dependency_closure_sha256": clang_indexer_dependency_closure_sha256,
        },
    }
    if optional_components:
        for name, component in optional_components.items():
            if name in components:
                raise ValueError(f"optional implementation component is core: {name}")
            components[name] = component
    return validate_implementation_binding(
        {
            "schema": IMPLEMENTATION_BINDING_SCHEMA,
            "components": components,
        },
        where="constructed implementation",
    )


def build_data_producer_binding(
    *,
    cppmega_commit: str,
    cppmega_tree_sha256: str,
    cppmega_mlx_commit: str,
    cppmega_mlx_tree_sha256: str,
    clang_indexer_sha256: str,
    clang_indexer_dependency_closure_sha256: str,
) -> dict[str, object]:
    """Build the producer half embedded in a data bundle manifest."""

    return validate_implementation_binding(
        {
            "schema": IMPLEMENTATION_BINDING_SCHEMA,
            "components": {
                "cppmega": {
                    "commit": cppmega_commit,
                    "tree_sha256": cppmega_tree_sha256,
                },
                "cppmega_mlx": {
                    "commit": cppmega_mlx_commit,
                    "tree_sha256": cppmega_mlx_tree_sha256,
                },
                "clang_indexer": {
                    "source_sha256": clang_indexer_sha256,
                    "dependency_closure_sha256": (
                        clang_indexer_dependency_closure_sha256
                    ),
                },
            },
        },
        where="constructed data producer",
        required_components=("cppmega", "cppmega_mlx", "clang_indexer"),
    )


def complete_training_implementation_binding(
    producer_binding: Mapping[str, object],
    *,
    megatron_commit: str,
) -> dict[str, object]:
    """Add the exact external Megatron source to a bundle producer binding."""

    producer = validate_implementation_binding(
        producer_binding,
        where="producer implementation",
        required_components=("cppmega", "cppmega_mlx", "clang_indexer"),
    )
    components = dict(producer["components"])
    components["megatron"] = {"commit": megatron_commit}
    return validate_implementation_binding(
        {"schema": IMPLEMENTATION_BINDING_SCHEMA, "components": components},
        where="constructed training implementation",
    )


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
    binding["implementation"] = validate_implementation_binding(
        binding["implementation"],
        where=where,
    )
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
    implementation: Mapping[str, object],
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
        "implementation": validate_implementation_binding(
            implementation,
            where="constructed receipt",
        ),
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
    "IMPLEMENTATION_BINDING_SCHEMA",
    "NO_CHECKPOINT_SHA256",
    "RECEIPT_BINDING_SCHEMA",
    "build_receipt_binding",
    "canonical_sha256",
    "build_implementation_binding",
    "build_data_producer_binding",
    "complete_training_implementation_binding",
    "validate_implementation_binding",
    "validate_binding_shape",
    "validate_receipt_binding",
]
