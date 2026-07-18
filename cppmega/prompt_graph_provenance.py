"""Shared production provenance and integrity checks for prompt-graph indexes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .symbol_identity import (
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    is_repo_file_location_identity,
    parse_repo_file_location_identity,
)


PRODUCTION_INDEX_PRODUCER = "ClangPromptProjectIndexProducer"
PRODUCTION_INDEX_VERSION = "3"
PRODUCTION_IDENTITY_PROVENANCE_CONTRACT = (
    "case4_symbol_reference_v3_repo_binding_v1"
)
INDEX_INTEGRITY_VERSION = "1"
INDEX_PAYLOAD_HASH_KEY = "index_payload_sha256"
TRUSTED_IDENTITY_ADAPTERS = frozenset(
    {
        "case4_symbol_reference_for_cursor_v3",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha_json(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def integrity_payload(index: Any) -> dict[str, Any]:
    payload = index.to_dict()
    provenance = dict(payload.get("provenance") or {})
    provenance.pop(INDEX_PAYLOAD_HASH_KEY, None)
    payload["provenance"] = provenance
    return payload


def payload_sha256(index: Any) -> str:
    return _sha_json(integrity_payload(index))


def with_integrity(index: Any) -> Any:
    payload = integrity_payload(index)
    provenance = dict(payload["provenance"])
    provenance[INDEX_PAYLOAD_HASH_KEY] = _sha_json(payload)
    payload["provenance"] = provenance
    return type(index).from_dict(payload)


def verify_integrity(index: Any) -> None:
    expected = index.provenance.get(INDEX_PAYLOAD_HASH_KEY)
    if not _is_sha256(expected):
        raise ValueError(
            "prompt graph index is missing a valid "
            f"{INDEX_PAYLOAD_HASH_KEY}"
        )
    actual = payload_sha256(index)
    if expected != actual:
        raise ValueError(
            "prompt graph index payload integrity mismatch: "
            f"expected={expected} actual={actual}"
        )


def _project_marker(symbol_key: str) -> str | None:
    for part in symbol_key.split("\x1f"):
        if part.startswith("project="):
            return part.removeprefix("project=")
    if "\x1fscope=" in symbol_key:
        scope = symbol_key.split("\x1fscope=", 1)[1].split("\x1f", 1)[0]
        for part in scope.split("|"):
            if part.startswith("project="):
                return part.removeprefix("project=")
    return None


def _validate_symbol_identity(
    symbol: Any,
    *,
    project_id: str,
    definition_kinds: set[str],
) -> None:
    if not symbol.symbol_key or symbol.semantic_identity != symbol.symbol_key:
        raise ValueError(
            f"production repository index symbol {symbol.identity!r} "
            "has mismatched semantic identity and symbol key"
        )
    if is_repo_file_location_identity(symbol.symbol_key):
        identity = parse_repo_file_location_identity(
            symbol.symbol_key,
            source=f"production symbol {symbol.identity}",
        )
        if identity.project != project_id:
            raise ValueError(
                f"production repository index symbol {symbol.identity!r} "
                "does not preserve project identity"
            )
        if symbol.usr or symbol.canonical_signature:
            raise ValueError(
                f"production repository index location identity "
                f"{symbol.identity!r} has semantic fields"
            )
        if identity.qname != symbol.qname:
            raise ValueError(
                f"production repository index symbol {symbol.identity!r} "
                "location qname does not match record"
            )
        if (
            symbol.kind in definition_kinds
            and identity.file != symbol.source_path
        ):
            raise ValueError(
                f"production repository index definition {symbol.identity!r} "
                "location file does not match source_path"
            )
        return

    if symbol.usr:
        if not symbol.canonical_signature:
            raise ValueError(
                f"production repository index symbol {symbol.identity!r} "
                "lacks a canonical signature"
            )
        expected_key = (
            f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
            f"project={project_id}\x1fusr={symbol.usr}"
        )
        if symbol.symbol_key != expected_key:
            marker = _project_marker(symbol.symbol_key)
            if symbol.kind in definition_kinds or marker is not None:
                raise ValueError(
                    f"production repository index symbol {symbol.identity!r} "
                    "USR/project identity is inconsistent"
                )
        return

    if not symbol.canonical_signature or not symbol.symbol_key.startswith(
        "fallback:"
    ):
        raise ValueError(
            f"production repository index symbol {symbol.identity!r} "
            "lacks a supported USR/signature identity"
        )
    fields: dict[str, str] = {}
    for part in symbol.symbol_key.removeprefix("fallback:").split("\x1f"):
        key, separator, value = part.partition("=")
        if separator:
            fields[key] = value
    if fields.get("schema") != f"v{SYMBOL_IDENTITY_SCHEMA_VERSION}":
        raise ValueError(
            f"production repository index symbol {symbol.identity!r} "
            "signature identity schema is inconsistent"
        )
    if fields.get("sig") != symbol.canonical_signature:
        raise ValueError(
            f"production repository index symbol {symbol.identity!r} "
            "canonical signature is inconsistent"
        )
    if symbol.kind in definition_kinds:
        scope = fields.get("scope", "").split("|")
        if f"project={project_id}" not in scope:
            raise ValueError(
                f"production repository index symbol {symbol.identity!r} "
                "does not preserve project identity"
            )


def validate_production_repository_index(
    index: Any,
    *,
    expected_project_id: str | None,
    repository_root: str | Path | None,
    expected_indexer_root: str | Path | None,
    index_schema: str,
    relation_names: Sequence[str],
    repository_snapshot: Callable[[str | Path], tuple[str, dict[str, str]]],
    validate_relative_source_path: Callable[[str, str], str],
    sha_file: Callable[[Path], str],
    require_project_id: Callable[[object, str], str],
) -> None:
    index.validate()
    provenance = dict(index.provenance)
    if provenance.get("producer") != PRODUCTION_INDEX_PRODUCER:
        raise ValueError(
            "production repository index requires producer "
            f"{PRODUCTION_INDEX_PRODUCER!r}"
        )
    if str(provenance.get("producer_version")) != PRODUCTION_INDEX_VERSION:
        raise ValueError(
            "production repository index has unsupported producer_version "
            f"{provenance.get('producer_version')!r}"
        )
    if provenance.get("index_integrity_version") != INDEX_INTEGRITY_VERSION:
        raise ValueError("production repository index integrity version mismatch")
    if provenance.get("schema") != index_schema:
        raise ValueError("production repository index schema provenance mismatch")
    if provenance.get("identity_provenance_contract") != (
        PRODUCTION_IDENTITY_PROVENANCE_CONTRACT
    ):
        raise ValueError(
            "production repository index identity provenance contract mismatch"
        )
    if provenance.get("project_id") != index.project_id:
        raise ValueError("production repository index project identity mismatch")
    if expected_project_id is not None:
        expected_project_id = require_project_id(
            expected_project_id, where="expected prompt graph project_id"
        )
        if index.project_id != expected_project_id:
            raise ValueError(
                "production repository index project identity mismatch: "
                f"expected={expected_project_id!r} actual={index.project_id!r}"
            )
    if provenance.get("strict_diagnostics") is not True:
        raise ValueError(
            "production repository index must be built with strict diagnostics"
        )
    if int(provenance.get("symbol_identity_schema_version") or 0) != (
        SYMBOL_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError(
            "production repository index symbol identity schema mismatch"
        )
    verify_integrity(index)

    hashes = provenance.get("hashes")
    required_hashes = {
        "repository_sha256",
        "dependency_closure_sha256",
        "compile_args_sha256",
        "indexer_sha256",
        "libclang_version_sha256",
    }
    if not isinstance(hashes, Mapping) or not required_hashes <= set(hashes):
        raise ValueError(
            "production repository index hashes are incomplete or invalid"
        )
    if any(not _is_sha256(hashes.get(name)) for name in required_hashes):
        raise ValueError(
            "production repository index hashes are incomplete or invalid"
        )

    repository_manifest = provenance.get("repository_manifest")
    dependency_manifest = provenance.get("dependency_manifest")
    if not isinstance(repository_manifest, Mapping) or not repository_manifest:
        raise ValueError(
            "production repository index repository_manifest is missing"
        )
    if not isinstance(dependency_manifest, Mapping) or not dependency_manifest:
        raise ValueError(
            "production repository index dependency_manifest is missing"
        )
    for name, manifest in (
        ("repository_manifest", repository_manifest),
        ("dependency_manifest", dependency_manifest),
    ):
        for relative, digest in manifest.items():
            if not isinstance(relative, str):
                raise ValueError(
                    f"production repository index {name} has a non-string path"
                )
            validate_relative_source_path(relative, where=f"{name} path")
            if not _is_sha256(digest):
                raise ValueError(
                    f"production repository index {name} has an invalid digest"
                )
    document_paths = {document.source_path for document in index.documents}
    if not document_paths <= set(repository_manifest):
        raise ValueError(
            "production repository index documents are absent from "
            "repository_manifest"
        )
    if not set(dependency_manifest) <= set(repository_manifest):
        raise ValueError(
            "production repository index dependency_manifest is not covered "
            "by repository_manifest"
        )

    toolchain = provenance.get("toolchain")
    if not isinstance(toolchain, Mapping):
        raise ValueError(
            "production repository index toolchain provenance is missing"
        )
    if not isinstance(toolchain.get("libclang_version"), str) or not toolchain.get(
        "libclang_version"
    ):
        raise ValueError(
            "production repository index toolchain provenance is missing"
        )
    if not isinstance(toolchain.get("libclang_path"), str) or not toolchain.get(
        "libclang_path"
    ):
        raise ValueError(
            "production repository index libclang path provenance is missing"
        )
    compile_args = toolchain.get("compile_args_by_file")
    if not isinstance(compile_args, Mapping) or not compile_args:
        raise ValueError(
            "production repository index compile argument provenance is missing"
        )
    if not set(dependency_manifest) <= set(compile_args):
        raise ValueError(
            "production repository index compile arguments do not cover "
            "the dependency manifest"
        )
    for relative, args in compile_args.items():
        validate_relative_source_path(
            str(relative), where="compile_args_by_file path"
        )
        if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
            raise ValueError(
                "production repository index compile arguments must be sequences"
            )
        if any(not isinstance(arg, str) for arg in args):
            raise ValueError(
                "production repository index compile arguments must be strings"
            )

    indexer_path_value = provenance.get("indexer_path")
    checkout_root_value = provenance.get("indexer_checkout_root")
    if not isinstance(indexer_path_value, str) or not isinstance(
        checkout_root_value, str
    ):
        raise ValueError("production repository indexer provenance is missing")
    indexer_path = Path(indexer_path_value).expanduser()
    checkout_root = Path(checkout_root_value).expanduser()
    if (
        not indexer_path.is_absolute()
        or indexer_path.name != "index_project.py"
        or not checkout_root.is_absolute()
    ):
        raise ValueError("production repository indexer provenance is not canonical")
    if expected_indexer_root is not None:
        expected_root = Path(expected_indexer_root).expanduser().resolve()
        expected_path = expected_root / "tools" / "clang_indexer" / "index_project.py"
        if indexer_path.resolve(strict=False) != expected_path:
            raise ValueError(
                "production repository indexer provenance does not match "
                "the current checkout"
            )
        if checkout_root.resolve() != expected_root:
            raise ValueError(
                "production repository checkout provenance does not match "
                "the current checkout"
            )
        if not expected_path.is_file() or sha_file(expected_path) != hashes[
            "indexer_sha256"
        ]:
            raise ValueError(
                "production repository indexer hash does not match the "
                "current checkout"
            )
    elif indexer_path.is_file() and sha_file(indexer_path) != hashes[
        "indexer_sha256"
    ]:
        raise ValueError("production repository indexer hash mismatch")
    elif repository_root is None and not indexer_path.is_file():
        raise ValueError("production repository indexer cannot be verified")

    adapters = provenance.get("identity_adapters")
    if not isinstance(adapters, Sequence) or isinstance(adapters, (str, bytes)):
        raise ValueError(
            "production repository index identity adapters are missing"
        )
    adapter_names = {str(adapter) for adapter in adapters}
    if adapter_names != TRUSTED_IDENTITY_ADAPTERS:
        raise ValueError(
            "production repository index uses an untrusted identity adapter"
        )

    for key, expected in (
        ("document_count", len(index.documents)),
        ("symbol_count", len(index.symbols)),
        ("chunk_count", len(index.chunks)),
    ):
        try:
            actual = int(provenance.get(key, -1))
        except (TypeError, ValueError):
            actual = -1
        if actual != expected:
            raise ValueError(
                f"production repository index {key} does not match payload"
            )
    declared_edge_counts = provenance.get("edge_counts")
    actual_edge_counts = {
        relation: sum(1 for edge in index.edges if edge.relation == relation)
        for relation in relation_names
    }
    if not isinstance(declared_edge_counts, Mapping):
        raise ValueError(
            "production repository index edge_counts do not match payload"
        )
    try:
        declared = {
            relation: int(declared_edge_counts.get(relation, -1))
            for relation in relation_names
        }
    except (TypeError, ValueError):
        declared = {}
    if declared != actual_edge_counts:
        raise ValueError(
            "production repository index edge_counts do not match payload"
        )
    diagnostics = provenance.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != document_paths:
        raise ValueError(
            "production repository index diagnostics do not match documents"
        )

    chunks = {chunk.identity: chunk for chunk in index.chunks}
    symbols = {symbol.identity: symbol for symbol in index.symbols}
    documents_by_id = {document.id: document for document in index.documents}
    definitions = {"function", "type", "variable"}
    contracts: dict[
        str, tuple[str, str, str, str, str, str, int, int, str, str, str]
    ] = {}
    definitions_by_semantic: dict[str, list[Any]] = {}
    for symbol in index.symbols:
        if (
            symbol.identity_project != index.project_id
            or not symbol.identity_file
            or symbol.identity_file not in repository_manifest
            or symbol.identity_line <= 0
            or symbol.identity_column <= 0
            or not symbol.identity_kind
        ):
            raise ValueError(
                "production repository index symbol identity provenance "
                f"is incomplete or not repository-bound for {symbol.identity!r}"
            )
        validate_relative_source_path(
            symbol.identity_file,
            where="production repository index symbol identity provenance file",
        )
        _validate_symbol_identity(
            symbol,
            project_id=index.project_id,
            definition_kinds=definitions,
        )
        if symbol.usr:
            expected_prefix = (
                f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
                f"project={index.project_id}\x1fusr="
            )
            if not symbol.symbol_key.startswith(expected_prefix) or (
                symbol.symbol_key.removeprefix(expected_prefix) != symbol.usr
            ):
                raise ValueError(
                    "production repository index USR/project identity "
                    f"mismatch for {symbol.identity!r}"
                )
        elif symbol.canonical_signature:
            parts = symbol.symbol_key.split("\x1f")
            fields = {
                key: value
                for key, separator, value in (
                    part.partition("=") for part in parts[1:]
                )
                if separator == "="
            }
            scope = fields.get("scope", "")
            if (
                parts[0]
                != f"fallback:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
                or not (
                    scope == f"project={index.project_id}"
                    or scope.startswith(f"project={index.project_id}|")
                )
                or fields.get("sig")
                != " ".join(symbol.canonical_signature.split())
            ):
                raise ValueError(
                    "production repository index signature/project identity "
                    f"mismatch for {symbol.identity!r}"
                )
        else:
            location_identity = parse_repo_file_location_identity(
                symbol.symbol_key,
                source=f"production repository index {symbol.identity}",
            )
            if (
                location_identity.project != index.project_id
                or location_identity.project != symbol.identity_project
                or location_identity.file != symbol.identity_file
                or location_identity.file != symbol.source_path
                or location_identity.line != symbol.identity_line
                or location_identity.column != symbol.identity_column
                or location_identity.kind != symbol.identity_kind
                or location_identity.qname != symbol.qname
            ):
                raise ValueError(
                    "production repository index location/project identity "
                    f"mismatch for {symbol.identity!r}"
                )
        contract = (
            symbol.symbol_key,
            symbol.usr,
            symbol.canonical_signature,
            symbol.qname,
            symbol.identity_project,
            symbol.identity_file,
            symbol.identity_line,
            symbol.identity_column,
            symbol.identity_kind,
            symbol.identity_provider,
            symbol.identity_include_provenance,
        )
        previous = contracts.setdefault(
            symbol.semantic_identity,
            contract,
        )
        if previous != contract:
            raise ValueError(
                "production repository index semantic identity contract "
                "mismatch (identity provenance) for "
                f"{symbol.semantic_identity!r}"
            )
        if symbol.chunk_identity:
            chunk = chunks.get(symbol.chunk_identity)
            if chunk is None or (
                chunk.document_id != symbol.document_id
                or chunk.source_path != symbol.source_path
                or not (chunk.start <= symbol.start <= symbol.end <= chunk.end)
            ):
                raise ValueError(
                    f"production repository index symbol {symbol.identity!r} "
                    "has an invalid owning chunk"
                )
        if symbol.kind in definitions:
            definitions_by_semantic.setdefault(
                symbol.semantic_identity, []
            ).append(symbol)
            document = documents_by_id[symbol.document_id]
            line_start = document.source.rfind("\n", 0, symbol.start) + 1
            expected_line = document.source.count("\n", 0, symbol.start) + 1
            expected_column = (
                len(document.source[line_start : symbol.start].encode("utf-8"))
                + 1
            )
            if (
                symbol.identity_file != symbol.source_path
                or symbol.identity_line != expected_line
                or symbol.identity_column != expected_column
            ):
                raise ValueError(
                    "production repository index definition identity "
                    f"provenance does not match source for {symbol.identity!r}"
                )

    for semantic_identity, definition_symbols in definitions_by_semantic.items():
        definition_contract = (
            definition_symbols[0].identity_project,
            definition_symbols[0].identity_file,
            definition_symbols[0].identity_line,
            definition_symbols[0].identity_column,
            definition_symbols[0].identity_kind,
            definition_symbols[0].identity_provider,
            definition_symbols[0].identity_include_provenance,
        )
        if any(
            (
                symbol.identity_project,
                symbol.identity_file,
                symbol.identity_line,
                symbol.identity_column,
                symbol.identity_kind,
                symbol.identity_provider,
                symbol.identity_include_provenance,
            )
            != definition_contract
            for symbol in definition_symbols[1:]
        ):
            raise ValueError(
                "production repository index definition identity provenance "
                f"mismatch for {semantic_identity!r}"
            )

    for edge in index.edges:
        if edge.relation not in {"call", "type", "def_use", "domain"}:
            continue
        source = symbols[edge.source]
        target = symbols[edge.target]
        if target.kind not in definitions:
            raise ValueError(
                "production repository index edge target is not a definition: "
                f"{edge.target!r}"
            )
        if not source.chunk_identity or not target.chunk_identity:
            raise ValueError(
                "production repository index core edge endpoints must have "
                "owning chunks"
            )
        if source.semantic_identity != target.semantic_identity:
            raise ValueError(
                "production repository index edge changes semantic identity: "
                f"{edge.relation} {edge.source!r}->{edge.target!r}"
            )

    if repository_root is not None:
        root = index.verify_repository(repository_root)
        actual_hash, actual_manifest = repository_snapshot(root)
        if actual_hash != hashes["repository_sha256"] or dict(
            actual_manifest
        ) != dict(repository_manifest):
            raise ValueError(
                "production repository index repository provenance is stale"
            )
        if any(
            actual_manifest.get(relative) != digest
            for relative, digest in dependency_manifest.items()
        ):
            raise ValueError(
                "production repository index dependency provenance is stale"
            )
