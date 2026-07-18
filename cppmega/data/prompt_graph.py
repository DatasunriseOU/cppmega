"""Typed prompt-time projection of repository graphs onto tokenizer coordinates.

The builder consumes an existing project index with stable source offsets. It
never parses source code. Graph inference therefore has one deterministic path
and fails closed when the index, tokenizer offsets, cache, or visible routes are
invalid.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .domain_schema import DomainEdgeKind, VALID_DOMAIN_EDGE_KINDS
from cppmega.symbol_identity import (
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SYMBOL_ID_MAX,
    compute_symbol_id,
    is_repo_file_location_identity,
)
from cppmega.prompt_graph_provenance import (
    INDEX_INTEGRITY_VERSION,
    INDEX_PAYLOAD_HASH_KEY,
    PRODUCTION_IDENTITY_PROVENANCE_CONTRACT,
    PRODUCTION_INDEX_PRODUCER,
    PRODUCTION_INDEX_VERSION,
    TRUSTED_IDENTITY_ADAPTERS,
    integrity_payload as _integrity_payload_for_index,
    payload_sha256 as _payload_sha256_for_index,
    validate_production_repository_index as _validate_production_index,
    verify_integrity as _verify_index_integrity,
    with_integrity as _with_index_integrity,
)


INDEX_SCHEMA = "cppmega_prompt_graph_index_v3"
LEGACY_INDEX_SCHEMA = "cppmega_prompt_graph_index_v1"
ARTIFACT_SCHEMA = "cppmega_prompt_graph_artifact_v3"
WINDOW_SCHEMA = "cppmega_prompt_graph_window_v3"
BUILDER_VERSION = "3"

TOKEN_SIDECAR_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "confidence_ids",
)
TOKEN_SIDECAR_DEFAULTS = {
    **{name: 0 for name in TOKEN_SIDECAR_NAMES},
    "domain_ids": 1,
    "confidence_ids": 4,
}
GENERATED_TOKEN_SIDECAR_DEFAULTS = {
    **TOKEN_SIDECAR_DEFAULTS,
    "confidence_ids": 1,
    "source_doc_ids": 0,
}
GENERATED_TOKEN_POLICY = (
    "generated_continuation_chunk_with_repository_summary_v1"
)
CPP_SPACE_SENTINEL = "<SPACE>"
CPP_NEWLINE_SENTINEL = "<NL>"

PAIR_ROUTE_KEYS = {
    "call": ("graph_call_edges", "graph_call_edge_counts"),
    "type": ("graph_type_edges", "graph_type_edge_counts"),
}
PAIR_ROUTE_EDGE_KINDS = {
    "call": int(DomainEdgeKind.CALL),
    "type": int(DomainEdgeKind.TYPE),
}
TRIPLE_ROUTE_KEYS = {
    "domain": ("graph_domain_edges", "graph_domain_edge_counts"),
    "build": ("graph_build_edges", "graph_build_edge_counts"),
    "shell": ("graph_shell_edges", "graph_shell_edge_counts"),
    "diagnostic": ("graph_diagnostic_edges", "graph_diagnostic_edge_counts"),
    "cross_domain": (
        "graph_cross_domain_edges",
        "graph_cross_domain_edge_counts",
    ),
}
ROUTE_KIND_FAMILIES = {
    "domain": frozenset(range(1, 14)),
    "build": frozenset(range(20, 27)),
    "shell": frozenset(range(40, 45)),
    "diagnostic": frozenset({60, 61, 62, 63, 64, 70, 71, 80, 90}),
    "cross_domain": frozenset({100}),
}
GENERATED_QUERY_ROUTE_KEY = "graph_generated_query_edges"
GENERATED_QUERY_COUNT_KEY = "graph_generated_query_edge_counts"
RELATION_NAMES = tuple(
    sorted((*PAIR_ROUTE_KEYS, "def_use", *TRIPLE_ROUTE_KEYS))
)

_REPOSITORY_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".h++",
        ".hh",
        ".hpp",
        ".hxx",
        ".inc",
        ".inl",
        ".ipp",
        ".tpp",
        ".txx",
    }
)
_REPOSITORY_INPUT_NAMES = frozenset(
    {
        "compile_commands.json",
        "CMakeLists.txt",
        "Makefile",
        "meson.build",
        "BUILD",
        "BUILD.bazel",
        "WORKSPACE",
        "WORKSPACE.bazel",
    }
)
_REPOSITORY_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", ".venv", "node_modules", "__pycache__"}
)
_PROJECT_IDENTITY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_OPAQUE_SYMBOL_ID_SIDECARS = frozenset(
    {"symbol_ids", "call_targets", "type_refs"}
)
_DEF_USE_VALUES = frozenset({0, 1, 2})


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha_json(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return _sha_json(json.loads(path.read_text(encoding="utf-8")))
    return sha256(path.read_bytes()).hexdigest()


def repository_snapshot(project_root: str | Path) -> tuple[str, dict[str, str]]:
    """Hash deterministic source/build inputs used by the prompt index producer."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"prompt graph repository not found: {root}")
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _REPOSITORY_SKIP_DIRS for part in relative.parts):
            continue
        if (
            path.suffix.lower() not in _REPOSITORY_SOURCE_SUFFIXES
            and path.name not in _REPOSITORY_INPUT_NAMES
        ):
            continue
        manifest[relative.as_posix()] = sha256(path.read_bytes()).hexdigest()
    if not any(
        Path(path).suffix.lower() in _REPOSITORY_SOURCE_SUFFIXES
        for path in manifest
    ):
        raise ValueError(f"prompt graph repository has no C/C++ sources: {root}")
    return _sha_json(manifest), manifest


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_relative_source_path(value: str, *, where: str) -> str:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{where} must be a contained relative path, got {value!r}")
    return relative.as_posix()


def _route_kind_family(kind: int) -> str | None:
    for family, kinds in ROUTE_KIND_FAMILIES.items():
        if kind in kinds:
            return family
    return None


def _validate_route_kind(relation: str, kind: int | None, *, where: str) -> int:
    if kind is None:
        raise ValueError(f"{where}: {relation} requires an explicit kind")
    actual = _route_kind_family(kind)
    if actual is None:
        raise ValueError(f"{where}: unknown domain edge kind {kind}")
    if actual != relation:
        raise ValueError(
            f"{where}: edge kind {kind} belongs to {actual}, not {relation}"
        )
    return kind


def normalize_cpp_whitespace_with_offsets(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Normalize C++ whitespace while retaining exact source coordinates."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    length = len(text)
    index = 0
    state: str | None = None
    raw_delimiter = ""
    while index < length:
        char = text[index]
        if state is None:
            if text.startswith("//", index):
                _append_literal(chars, spans, text, index, index + 2)
                state = "line-comment"
                index += 2
                continue
            if text.startswith("/*", index):
                _append_literal(chars, spans, text, index, index + 2)
                state = "block-comment"
                index += 2
                continue
            raw_opening = _cpp_raw_string_opening(text, index)
            if raw_opening is not None:
                opening_end, raw_delimiter = raw_opening
                _append_literal(chars, spans, text, index, opening_end)
                state = "raw-string"
                index = opening_end
                continue
            if char in {'"', "'"}:
                chars.append(char)
                spans.append((index, index + 1))
                state = "string" if char == '"' else "character"
                index += 1
                continue
            if char in "\r\n":
                end = index + 1
                while end < length and text[end] in "\r\n":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_NEWLINE_SENTINEL, index, end
                )
                index = end
                continue
            if char in " \t":
                end = index + 1
                while end < length and text[end] in " \t":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_SPACE_SENTINEL, index, end
                )
                index = end
                continue
            chars.append(char)
            spans.append((index, index + 1))
            index += 1
            continue

        if state == "line-comment":
            if char in "\r\n":
                end = index + 1
                while end < length and text[end] in "\r\n":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_NEWLINE_SENTINEL, index, end
                )
                state = None
                index = end
            elif char in " \t":
                end = index + 1
                while end < length and text[end] in " \t":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_SPACE_SENTINEL, index, end
                )
                index = end
            else:
                chars.append(char)
                spans.append((index, index + 1))
                index += 1
            continue

        if state == "block-comment":
            if text.startswith("*/", index):
                _append_literal(chars, spans, text, index, index + 2)
                state = None
                index += 2
            elif char in "\r\n":
                end = index + 1
                while end < length and text[end] in "\r\n":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_NEWLINE_SENTINEL, index, end
                )
                index = end
            elif char in " \t":
                end = index + 1
                while end < length and text[end] in " \t":
                    end += 1
                _append_sentinel(
                    chars, spans, CPP_SPACE_SENTINEL, index, end
                )
                index = end
            else:
                chars.append(char)
                spans.append((index, index + 1))
                index += 1
            continue

        if state == "raw-string":
            if text.startswith(raw_delimiter, index):
                end = index + len(raw_delimiter)
                _append_literal(chars, spans, text, index, end)
                state = None
                raw_delimiter = ""
                index = end
            else:
                chars.append(char)
                spans.append((index, index + 1))
                index += 1
            continue

        quote = '"' if state == "string" else "'"
        chars.append(char)
        spans.append((index, index + 1))
        if char == "\\" and index + 1 < length:
            chars.append(text[index + 1])
            spans.append((index + 1, index + 2))
            index += 2
        else:
            if char == quote:
                state = None
            index += 1
    return "".join(chars), spans


def _cpp_raw_string_opening(
    text: str,
    start: int,
) -> tuple[int, str] | None:
    index = start
    if text.startswith("u8", index):
        index += 2
    elif index < len(text) and text[index] in "uUL":
        index += 1
    if not text.startswith('R"', index):
        return None
    delimiter_start = index + 2
    opening_end = text.find("(", delimiter_start, delimiter_start + 17)
    if opening_end < 0:
        return None
    delimiter = text[delimiter_start:opening_end]
    if any(char.isspace() or char in '()\\"' for char in delimiter):
        return None
    return opening_end + 1, ")" + delimiter + '"'


def _append_literal(
    chars: list[str],
    spans: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    for index in range(start, end):
        chars.append(text[index])
        spans.append((index, index + 1))


def _append_sentinel(
    chars: list[str],
    spans: list[tuple[int, int]],
    sentinel: str,
    start: int,
    end: int,
) -> None:
    for char in sentinel:
        chars.append(char)
        spans.append((start, end))


def _require_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be int, got {type(value).__name__}")
    return int(value)


def _require_str(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{where} must be a non-empty string")
    return value


def require_prompt_graph_project_id(value: object, *, where: str) -> str:
    """Validate the stable ``owner/repo`` identity used by v3 symbol keys."""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError(
            f"{where} must be an exact owner/repo string, got {value!r}"
        )
    if not _PROJECT_IDENTITY_RE.fullmatch(value):
        raise ValueError(
            f"{where} must be stable owner/repo, got {value!r}"
        )
    owner, repo = value.split("/", 1)
    if owner in {".", ".."} or repo in {".", ".."} or repo.endswith(".git"):
        raise ValueError(
            f"{where} must be canonical owner/repo without .git, got {value!r}"
        )
    return value


def _require_rows(
    value: Any,
    *,
    where: str,
) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{where} must be a sequence of objects")
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise TypeError(f"{where}[{index}] must be an object")
    return value


def _validate_span(start: int, end: int, upper: int, *, where: str) -> None:
    if start < 0 or end <= start or end > upper:
        raise ValueError(
            f"{where}: invalid span [{start},{end}) for length {upper}"
        )


@dataclass(frozen=True)
class PromptGraphDocument:
    id: int
    source_path: str
    source: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PromptGraphDocument":
        source = row.get("source")
        if not isinstance(source, str):
            raise TypeError("document.source must be a string")
        return cls(
            id=_require_int(row.get("id"), where="document.id"),
            source_path=_validate_relative_source_path(
                _require_str(row.get("source_path"), where="document.source_path"),
                where="document.source_path",
            ),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "source": self.source,
        }


@dataclass(frozen=True)
class PromptGraphSymbol:
    id: int
    symbol_id: int
    identity: str
    kind: str
    document_id: int
    source_path: str
    start: int
    end: int
    semantic_identity: str
    symbol_key: str = ""
    usr: str = ""
    canonical_signature: str = ""
    qname: str = ""
    chunk_identity: str = ""
    identity_project: str = ""
    identity_file: str = ""
    identity_line: int = 0
    identity_column: int = 0
    identity_kind: str = ""
    identity_provider: str = ""
    identity_include_provenance: str = ""

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PromptGraphSymbol":
        identity = _require_str(row.get("identity"), where="symbol.identity")
        return cls(
            id=_require_int(row.get("id"), where="symbol.id"),
            symbol_id=_require_int(
                row.get("symbol_id"), where="symbol.symbol_id"
            ),
            identity=identity,
            kind=str(row.get("kind") or "symbol"),
            document_id=_require_int(
                row.get("document_id"), where="symbol.document_id"
            ),
            source_path=_validate_relative_source_path(
                _require_str(row.get("source_path"), where="symbol.source_path"),
                where="symbol.source_path",
            ),
            start=_require_int(row.get("start"), where="symbol.start"),
            end=_require_int(row.get("end"), where="symbol.end"),
            semantic_identity=_require_str(
                row.get("semantic_identity"), where="symbol.semantic_identity"
            ),
            symbol_key=str(row.get("symbol_key") or ""),
            usr=str(row.get("usr") or ""),
            canonical_signature=str(row.get("canonical_signature") or ""),
            qname=str(row.get("qname") or ""),
            chunk_identity=str(row.get("chunk_identity") or ""),
            identity_project=str(row.get("identity_project") or ""),
            identity_file=str(row.get("identity_file") or ""),
            identity_line=_require_int(
                row.get("identity_line", 0), where="symbol.identity_line"
            ),
            identity_column=_require_int(
                row.get("identity_column", 0), where="symbol.identity_column"
            ),
            identity_kind=str(row.get("identity_kind") or ""),
            identity_provider=str(row.get("identity_provider") or ""),
            identity_include_provenance=str(
                row.get("identity_include_provenance") or ""
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symbol_id": self.symbol_id,
            "identity": self.identity,
            "semantic_identity": self.semantic_identity,
            "symbol_key": self.symbol_key,
            "usr": self.usr,
            "canonical_signature": self.canonical_signature,
            "qname": self.qname,
            "kind": self.kind,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "start": self.start,
            "end": self.end,
            "chunk_identity": self.chunk_identity,
            "identity_project": self.identity_project,
            "identity_file": self.identity_file,
            "identity_line": self.identity_line,
            "identity_column": self.identity_column,
            "identity_kind": self.identity_kind,
            "identity_provider": self.identity_provider,
            "identity_include_provenance": self.identity_include_provenance,
        }


@dataclass(frozen=True)
class PromptGraphChunk:
    id: int
    identity: str
    document_id: int
    source_path: str
    start: int
    end: int
    kind: int = 1
    dep_level: int = 0

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PromptGraphChunk":
        chunk_id = _require_int(row.get("id"), where="chunk.id")
        return cls(
            id=chunk_id,
            identity=str(row.get("identity") or f"chunk:{chunk_id}"),
            document_id=_require_int(
                row.get("document_id"), where="chunk.document_id"
            ),
            source_path=_validate_relative_source_path(
                _require_str(row.get("source_path"), where="chunk.source_path"),
                where="chunk.source_path",
            ),
            start=_require_int(row.get("start"), where="chunk.start"),
            end=_require_int(row.get("end"), where="chunk.end"),
            kind=_require_int(row.get("kind", 1), where="chunk.kind"),
            dep_level=_require_int(row.get("dep_level", 0), where="chunk.dep_level"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "identity": self.identity,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "dep_level": self.dep_level,
        }


@dataclass(frozen=True)
class PromptGraphEdge:
    relation: str
    source: str
    target: str
    kind: int | None = None

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PromptGraphEdge":
        kind = row.get("kind")
        return cls(
            relation=_require_str(
                row.get("relation"),
                where="edge.relation",
            ),
            source=_require_str(row.get("source"), where="edge.source"),
            target=_require_str(row.get("target"), where="edge.target"),
            kind=None
            if kind is None
            else _require_int(kind, where="edge.kind"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "relation": self.relation,
            "source": self.source,
            "target": self.target,
        }
        if self.kind is not None:
            out["kind"] = self.kind
        return out


@dataclass(frozen=True)
class PromptProjectIndex:
    project_id: str
    documents: tuple[PromptGraphDocument, ...]
    symbols: tuple[PromptGraphSymbol, ...]
    chunks: tuple[PromptGraphChunk, ...]
    edges: tuple[PromptGraphEdge, ...]
    provenance: Mapping[str, Any]
    schema: str = INDEX_SCHEMA

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_legacy_single_document: bool = False,
    ) -> "PromptProjectIndex":
        schema = payload.get("schema")
        if schema == LEGACY_INDEX_SCHEMA:
            if not allow_legacy_single_document:
                raise ValueError(
                    "legacy prompt graph index requires explicit "
                    "allow_legacy_single_document=True"
                )
            payload = _upgrade_legacy_single_document_index(payload)
            schema = payload.get("schema")
        if schema != INDEX_SCHEMA:
            raise ValueError(
                f"unsupported prompt graph index schema {schema!r}"
            )
        index = cls(
            project_id=require_prompt_graph_project_id(
                payload.get("project_id"),
                where="project_id",
            ),
            documents=tuple(
                PromptGraphDocument.from_dict(row)
                for row in _require_rows(payload.get("documents"), where="documents")
            ),
            symbols=tuple(
                PromptGraphSymbol.from_dict(row)
                for row in _require_rows(
                    payload.get("symbols"),
                    where="symbols",
                )
            ),
            chunks=tuple(
                PromptGraphChunk.from_dict(row)
                for row in _require_rows(
                    payload.get("chunks"),
                    where="chunks",
                )
            ),
            edges=tuple(
                PromptGraphEdge.from_dict(row)
                for row in _require_rows(
                    payload.get("edges"),
                    where="edges",
                )
            ),
            provenance=dict(payload.get("provenance") or {}),
            schema=schema,
        )
        index.validate()
        return index

    @classmethod
    def from_json_path(
        cls,
        path: str | Path,
        *,
        allow_legacy_single_document: bool = False,
    ) -> "PromptProjectIndex":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"{path}: prompt graph index must be a JSON object"
            )
        return cls.from_dict(
            payload,
            allow_legacy_single_document=allow_legacy_single_document,
        )

    def validate(self) -> None:
        require_prompt_graph_project_id(self.project_id, where="project_id")
        if not self.documents:
            raise ValueError("prompt graph index has no documents")
        documents_by_id: dict[int, PromptGraphDocument] = {}
        documents_by_path: dict[str, PromptGraphDocument] = {}
        for document in self.documents:
            if document.id <= 0:
                raise ValueError("prompt graph document ids must be positive")
            if document.id in documents_by_id:
                raise ValueError(f"duplicate document id {document.id}")
            if document.source_path in documents_by_path:
                raise ValueError(f"duplicate document source_path {document.source_path!r}")
            documents_by_id[document.id] = document
            documents_by_path[document.source_path] = document

        seen_ids: set[int] = set()
        seen_identities: set[str] = set()
        legacy_adapter = (
            self.provenance.get("producer")
            == "explicit_legacy_single_document_adapter"
        )
        if (
            not legacy_adapter
            and self.provenance.get("symbol_identity_schema_version")
            != SYMBOL_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "prompt graph index requires symbol_identity_schema_version="
                f"{SYMBOL_IDENTITY_SCHEMA_VERSION}"
            )
        canonical_key_by_id: dict[int, str] = {}
        canonical_id_by_key: dict[str, int] = {}
        for symbol in self.symbols:
            document = documents_by_id.get(symbol.document_id)
            if document is None:
                raise ValueError(
                    f"symbol {symbol.identity}: unknown document_id {symbol.document_id}"
                )
            if document.source_path != symbol.source_path:
                raise ValueError(
                    f"symbol {symbol.identity}: document_id {symbol.document_id} "
                    f"source_path mismatch {symbol.source_path!r} != "
                    f"{document.source_path!r}"
                )
            _validate_span(
                symbol.start,
                symbol.end,
                len(document.source),
                where=f"symbol {symbol.identity} in {document.source_path}",
            )
            if symbol.id <= 0:
                raise ValueError(
                    f"symbol {symbol.identity}: id must be positive"
                )
            if symbol.id in seen_ids:
                raise ValueError(f"duplicate symbol id {symbol.id}")
            if not 0 < symbol.symbol_id <= SYMBOL_ID_MAX:
                raise ValueError(
                    f"symbol {symbol.identity}: symbol_id must be canonical uint64"
                )
            if not symbol.symbol_key:
                raise ValueError(
                    f"symbol {symbol.identity}: symbol_key must be non-empty"
                )
            if symbol.semantic_identity != symbol.symbol_key:
                raise ValueError(
                    f"symbol {symbol.identity}: semantic_identity must equal symbol_key"
                )
            expected_symbol_id = compute_symbol_id(symbol.symbol_key)
            if symbol.symbol_id != expected_symbol_id:
                raise ValueError(
                    f"symbol {symbol.identity}: symbol_id {symbol.symbol_id} does not "
                    f"match v{SYMBOL_IDENTITY_SCHEMA_VERSION} canonical ID "
                    f"{expected_symbol_id}"
                )
            existing_key = canonical_key_by_id.get(symbol.symbol_id)
            if existing_key is not None and existing_key != symbol.symbol_key:
                raise ValueError(
                    "canonical symbol ID collision in prompt graph index: "
                    f"id={symbol.symbol_id} first={existing_key!r} "
                    f"second={symbol.symbol_key!r}"
                )
            existing_id = canonical_id_by_key.get(symbol.symbol_key)
            if existing_id is not None and existing_id != symbol.symbol_id:
                raise ValueError(
                    f"canonical symbol key {symbol.symbol_key!r} maps to both "
                    f"{existing_id} and {symbol.symbol_id}"
                )
            if symbol.identity in seen_identities:
                raise ValueError(
                    f"duplicate symbol identity {symbol.identity!r}"
                )
            if (
                not legacy_adapter
                and symbol.kind in {"function", "type", "variable"}
                and not symbol.canonical_signature
                and not is_repo_file_location_identity(symbol.symbol_key)
            ):
                raise ValueError(
                    f"symbol {symbol.identity}: v3 definitions require a "
                    "canonical_signature; qname-only indexes are unsupported"
                )
            seen_ids.add(symbol.id)
            seen_identities.add(symbol.identity)
            canonical_key_by_id[symbol.symbol_id] = symbol.symbol_key
            canonical_id_by_key[symbol.symbol_key] = symbol.symbol_id

        seen_chunk_ids: set[int] = set()
        seen_chunk_identities: set[str] = set()
        last_end_by_document: dict[int, int] = {}
        for chunk in sorted(
            self.chunks,
            key=lambda item: (item.document_id, item.start, item.end, item.id),
        ):
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                raise ValueError(
                    f"chunk {chunk.identity}: unknown document_id {chunk.document_id}"
                )
            if document.source_path != chunk.source_path:
                raise ValueError(
                    f"chunk {chunk.identity}: document_id {chunk.document_id} "
                    f"source_path mismatch {chunk.source_path!r} != "
                    f"{document.source_path!r}"
                )
            _validate_span(
                chunk.start,
                chunk.end,
                len(document.source),
                where=f"chunk {chunk.identity} in {document.source_path}",
            )
            if chunk.id < 0:
                raise ValueError(
                    f"chunk {chunk.identity}: id must be non-negative"
                )
            if chunk.kind < 0 or chunk.dep_level < 0:
                raise ValueError(
                    f"chunk {chunk.identity}: kind and dep_level "
                    "must be non-negative"
                )
            if chunk.id in seen_chunk_ids:
                raise ValueError(f"duplicate chunk id {chunk.id}")
            if chunk.identity in seen_chunk_identities:
                raise ValueError(
                    f"duplicate chunk identity {chunk.identity!r}"
                )
            last_end = last_end_by_document.get(chunk.document_id, -1)
            if chunk.start < last_end:
                raise ValueError(
                    f"prompt graph chunks must not overlap in {chunk.source_path}"
                )
            seen_chunk_ids.add(chunk.id)
            seen_chunk_identities.add(chunk.identity)
            last_end_by_document[chunk.document_id] = chunk.end

        valid_relations = set(RELATION_NAMES)
        seen_edges: set[tuple[str, str, str, int | None]] = set()
        for edge in self.edges:
            if edge.relation not in valid_relations:
                raise ValueError(
                    f"unsupported prompt graph relation {edge.relation!r}"
                )
            if edge.source not in seen_identities:
                raise ValueError(
                    f"edge source identity not found: {edge.source!r}"
                )
            if edge.target not in seen_identities:
                raise ValueError(
                    f"edge target identity not found: {edge.target!r}"
                )
            if edge.relation in TRIPLE_ROUTE_KEYS:
                _validate_route_kind(
                    edge.relation,
                    edge.kind,
                    where=f"edge {edge.source!r}->{edge.target!r}",
                )
            elif edge.kind is not None:
                raise ValueError(
                    f"{edge.relation} edge must not carry a domain route kind"
                )
            edge_key = (
                edge.relation,
                edge.source,
                edge.target,
                edge.kind,
            )
            if edge_key in seen_edges:
                raise ValueError(f"duplicate prompt graph edge {edge_key!r}")
            seen_edges.add(edge_key)

    def verify_repository(self, project_root: str | Path) -> Path:
        root = Path(project_root).resolve()
        for document in self.documents:
            source_path = (root / document.source_path).resolve()
            try:
                source_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"prompt graph source path escapes repository: "
                    f"{document.source_path!r}"
                ) from exc
            if not source_path.is_file():
                raise FileNotFoundError(f"prompt graph source file not found: {source_path}")
            actual = source_path.read_text(encoding="utf-8")
            if actual != document.source:
                raise ValueError(
                    f"prompt graph repository freshness mismatch for {source_path}"
                )
        expected = dict(self.provenance.get("hashes") or {}).get(
            "repository_sha256"
        )
        if expected is not None:
            actual, _manifest = repository_snapshot(root)
            if actual != expected:
                raise ValueError(
                    "prompt graph repository freshness mismatch: "
                    f"expected={expected} actual={actual}"
                )
        return root

    def document_for_path(self, source_path: str) -> PromptGraphDocument:
        normalized = _validate_relative_source_path(source_path, where="source_path")
        matches = [doc for doc in self.documents if doc.source_path == normalized]
        if len(matches) != 1:
            raise KeyError(f"prompt graph document not found: {normalized!r}")
        return matches[0]

    def document_for_id(self, document_id: int) -> PromptGraphDocument:
        matches = [doc for doc in self.documents if doc.id == document_id]
        if len(matches) != 1:
            raise KeyError(f"prompt graph document id not found: {document_id}")
        return matches[0]

    def symbol_for_identity(self, identity: str) -> PromptGraphSymbol:
        matches = [symbol for symbol in self.symbols if symbol.identity == identity]
        if len(matches) != 1:
            raise KeyError(f"prompt graph symbol identity not found: {identity!r}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "project_id": self.project_id,
            "documents": [
                document.to_dict()
                for document in sorted(self.documents, key=lambda item: item.id)
            ],
            "symbols": [
                symbol.to_dict()
                for symbol in sorted(
                    self.symbols,
                    key=lambda item: (item.id, item.identity),
                )
            ],
            "chunks": [
                chunk.to_dict()
                for chunk in sorted(
                    self.chunks,
                    key=lambda item: (
                        item.document_id,
                        item.start,
                        item.end,
                        item.id,
                    ),
                )
            ],
            "edges": [
                edge.to_dict()
                for edge in sorted(
                    self.edges,
                    key=lambda item: (
                        item.relation,
                        item.source,
                        item.target,
                        -1 if item.kind is None else item.kind,
                    ),
                )
            ],
            "provenance": json.loads(_stable_json(self.provenance)),
        }

    @property
    def source_sha256(self) -> str:
        return _sha_json(
            {
                document.source_path: _sha_text(document.source)
                for document in sorted(self.documents, key=lambda item: item.source_path)
            }
        )

    @property
    def index_sha256(self) -> str:
        return _sha_json(self.to_dict())

    def _integrity_payload(self) -> dict[str, Any]:
        return _integrity_payload_for_index(self)

    @property
    def payload_sha256(self) -> str:
        return _payload_sha256_for_index(self)

    def with_integrity(self) -> "PromptProjectIndex":
        return _with_index_integrity(self)

    def verify_integrity(self) -> None:
        _verify_index_integrity(self)

    def validate_production_repository_index(
        self,
        *,
        expected_project_id: str | None = None,
        repository_root: str | Path | None = None,
        expected_indexer_root: str | Path | None = None,
    ) -> None:
        _validate_production_index(
            self,
            expected_project_id=expected_project_id,
            repository_root=repository_root,
            expected_indexer_root=expected_indexer_root,
            index_schema=INDEX_SCHEMA,
            relation_names=RELATION_NAMES,
            repository_snapshot=repository_snapshot,
            validate_relative_source_path=_validate_relative_source_path,
            sha_file=_sha_file,
            require_project_id=require_prompt_graph_project_id,
        )


def _upgrade_legacy_single_document_index(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    project_id = require_prompt_graph_project_id(
        payload.get("project_id"), where="project_id"
    )
    source_path = _validate_relative_source_path(
        _require_str(payload.get("source_path"), where="source_path"),
        where="source_path",
    )
    source = payload.get("source")
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    upgraded = dict(payload)
    upgraded.pop("source_path", None)
    upgraded.pop("source", None)
    upgraded["schema"] = INDEX_SCHEMA
    upgraded["documents"] = [
        {"id": 1, "source_path": source_path, "source": source}
    ]
    upgraded_symbols: list[dict[str, Any]] = []
    for row in _require_rows(payload.get("symbols"), where="symbols"):
        identity = _require_str(row.get("identity"), where="symbol.identity")
        symbol_key = (
            f"legacy:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
            f"project={project_id}\x1fidentity={identity}"
        )
        upgraded_symbols.append(
            {
                **dict(row),
                "symbol_id": compute_symbol_id(symbol_key),
                "document_id": 1,
                "source_path": source_path,
                "semantic_identity": symbol_key,
                "symbol_key": symbol_key,
            }
        )
    upgraded["symbols"] = upgraded_symbols
    upgraded["chunks"] = [
        {
            **dict(row),
            "document_id": 1,
            "source_path": source_path,
        }
        for row in _require_rows(payload.get("chunks"), where="chunks")
    ]
    upgraded["provenance"] = {
        "producer": "explicit_legacy_single_document_adapter",
        "legacy_schema": LEGACY_INDEX_SCHEMA,
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
    }
    return upgraded


@dataclass(frozen=True)
class PromptGraphSegment:
    text: str
    document_id: int | None = None
    source_path: str | None = None
    source_start: int | None = None
    role: str = "code"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("prompt graph segment text must be non-empty")
        if (
            self.source_start is not None
            and (
                isinstance(self.source_start, bool)
                or not isinstance(self.source_start, int)
                or self.source_start < 0
            )
        ):
            raise ValueError(
                "prompt graph segment source_start must be a non-negative integer"
            )
        if self.source_start is not None:
            if (
                isinstance(self.document_id, bool)
                or not isinstance(self.document_id, int)
                or self.document_id <= 0
            ):
                raise ValueError(
                    "source-mapped prompt graph segment requires positive document_id"
                )
            if not isinstance(self.source_path, str) or not self.source_path:
                raise ValueError(
                    "source-mapped prompt graph segment requires source_path"
                )
            _validate_relative_source_path(
                self.source_path, where="prompt graph segment source_path"
            )
        elif self.document_id is not None or self.source_path is not None:
            raise ValueError(
                "unmapped prompt graph segment must not declare document metadata"
            )
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("prompt graph segment role must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_start": self.source_start,
            "role": self.role,
        }


@dataclass(frozen=True)
class PromptGraphContext:
    segments: tuple[PromptGraphSegment, ...]
    language: str = "cpp"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("prompt graph context needs at least one segment")
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("prompt graph context language must be non-empty")

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        document_id: int | None = None,
        source_path: str | None = None,
        source_start: int | None = None,
        language: str = "cpp",
        role: str = "code",
    ) -> "PromptGraphContext":
        return cls(
            segments=(
                PromptGraphSegment(
                    prompt,
                    document_id=document_id,
                    source_path=source_path,
                    source_start=source_start,
                    role=role,
                ),
            ),
            language=language,
        )

    @classmethod
    def from_repository_prompt(
        cls,
        project_index: PromptProjectIndex,
        prompt: str,
        *,
        document_id: int,
        source_path: str,
        source_start: int,
        language: str = "cpp",
    ) -> "PromptGraphContext":
        """Prepend the transitive definitions referenced by a source prompt."""
        prompt_segment = PromptGraphSegment(
            prompt,
            document_id=document_id,
            source_path=source_path,
            source_start=source_start,
            role="target",
        )
        dependencies = _repository_dependency_chunks(
            project_index,
            document_id=document_id,
            source_start=source_start,
            source_end=source_start + len(prompt),
        )
        if not dependencies:
            return cls(segments=(prompt_segment,), language=language)

        segments: list[PromptGraphSegment] = []
        for chunk in dependencies:
            document = project_index.document_for_id(chunk.document_id)
            segments.extend(
                (
                    PromptGraphSegment(
                        f"\n// cppmega dependency: {chunk.source_path}\n",
                        role="dependency_boundary",
                    ),
                    PromptGraphSegment(
                        document.source[chunk.start : chunk.end],
                        document_id=chunk.document_id,
                        source_path=chunk.source_path,
                        source_start=chunk.start,
                        role="dependency",
                    ),
                )
            )
        segments.append(
            PromptGraphSegment(
                f"\n// cppmega target: {source_path}\n",
                role="target_boundary",
            )
        )
        segments.append(prompt_segment)
        return cls(segments=tuple(segments), language=language)

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)

    @property
    def context_sha256(self) -> str:
        return _sha_json(self.to_dict())

    def source_positions(self) -> list[tuple[int, int] | None]:
        positions: list[tuple[int, int] | None] = []
        for segment in self.segments:
            if segment.source_start is None:
                positions.extend([None] * len(segment.text))
            else:
                assert segment.document_id is not None
                positions.extend(
                    (
                        int(segment.document_id),
                        int(segment.source_start) + offset,
                    )
                    for offset in range(len(segment.text))
                )
        return positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [
                segment.to_dict() for segment in self.segments
            ],
        }


def _repository_dependency_chunks(
    project_index: PromptProjectIndex,
    *,
    document_id: int,
    source_start: int,
    source_end: int,
) -> tuple[PromptGraphChunk, ...]:
    """Return every indexed definition chunk reachable from visible uses."""
    project_index.validate()
    source_document = project_index.document_for_id(document_id)
    if source_start < 0 or source_end <= source_start:
        raise ValueError("repository prompt source span must be non-empty")
    if source_end > len(source_document.source):
        raise ValueError(
            f"repository prompt source span [{source_start},{source_end}) "
            f"exceeds {source_document.source_path} length "
            f"{len(source_document.source)}"
        )

    symbols_by_identity = {
        symbol.identity: symbol for symbol in project_index.symbols
    }
    chunks_by_identity = {
        chunk.identity: chunk for chunk in project_index.chunks
    }
    symbols_by_chunk: dict[str, list[PromptGraphSymbol]] = {}
    for symbol in sorted(
        project_index.symbols,
        key=lambda item: (
            item.chunk_identity,
            item.document_id,
            item.start,
            item.end,
            item.identity,
        ),
    ):
        symbols_by_chunk.setdefault(symbol.chunk_identity, []).append(symbol)
    outgoing: dict[str, list[PromptGraphEdge]] = {}
    for edge in sorted(
        project_index.edges,
        key=lambda item: (
            item.source,
            item.relation,
            item.target,
            -1 if item.kind is None else item.kind,
        ),
    ):
        outgoing.setdefault(edge.source, []).append(edge)

    visible = {
        symbol.identity
        for symbol in project_index.symbols
        if symbol.document_id == document_id
        and symbol.start < source_end
        and source_start < symbol.end
    }
    queue = deque(sorted(visible))
    seen_or_queued = set(visible)
    visited: set[str] = set()
    dependencies: dict[str, PromptGraphChunk] = {}
    while queue:
        source_identity = queue.popleft()
        if source_identity in visited:
            continue
        visited.add(source_identity)
        for edge in outgoing.get(source_identity, ()):
            target = symbols_by_identity[edge.target]
            chunk = chunks_by_identity.get(target.chunk_identity)
            if chunk is None:
                raise ValueError(
                    f"repository dependency edge {edge.source!r}->"
                    f"{edge.target!r} has no target definition chunk"
                )
            overlaps_prompt = (
                chunk.document_id == document_id
                and chunk.start < source_end
                and source_start < chunk.end
            )
            if overlaps_prompt:
                continue
            dependencies[chunk.identity] = chunk
            for candidate in symbols_by_chunk.get(chunk.identity, ()):
                if candidate.identity not in seen_or_queued:
                    queue.append(candidate.identity)
                    seen_or_queued.add(candidate.identity)

    return tuple(
        sorted(
            dependencies.values(),
            key=lambda item: (
                item.source_path,
                item.start,
                item.end,
                item.identity,
            ),
        )
    )


class CppPromptTokenizerAdapter:
    """Apply cppmega whitespace normalization to a raw tokenizer backend."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.path = None if tokenizer_path is None else Path(tokenizer_path)
        self.name_or_path = getattr(tokenizer, "name_or_path", None)

    def encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        normalized, normalized_to_source = (
            normalize_cpp_whitespace_with_offsets(text)
        )
        token_ids, normalized_offsets = self._encode_normalized(normalized)
        if len(token_ids) != len(normalized_offsets):
            raise ValueError(
                "raw tokenizer ids/offsets length mismatch: "
                f"ids={len(token_ids)} offsets={len(normalized_offsets)}"
            )
        source_offsets: list[tuple[int, int]] = []
        for token_index, (start, end) in enumerate(normalized_offsets):
            if (
                start < 0
                or end <= start
                or end > len(normalized_to_source)
            ):
                raise ValueError(
                    f"token {token_index} has invalid normalized offset "
                    f"[{start},{end}) for length "
                    f"{len(normalized_to_source)}"
                )
            covered = normalized_to_source[start:end]
            source_offsets.append(
                (
                    min(span[0] for span in covered),
                    max(span[1] for span in covered),
                )
            )
        return token_ids, source_offsets

    def _encode_normalized(
        self,
        normalized: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                normalized,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = (
                encoded["input_ids"]
                if isinstance(encoded, Mapping)
                else encoded.input_ids
            )
            offsets = (
                encoded["offset_mapping"]
                if isinstance(encoded, Mapping)
                else encoded.offset_mapping
            )
        else:
            encode = getattr(self.tokenizer, "encode", None)
            if not callable(encode):
                raise TypeError(
                    "CppPromptTokenizerAdapter requires a callable Hugging "
                    "Face tokenizer or a backend with encode(text)"
                )
            encoded = encode(normalized)
            ids = encoded.ids
            offsets = encoded.offsets
        if ids and isinstance(ids[0], Sequence):
            raise ValueError("raw tokenizer returned batched ids for one prompt")
        return (
            [int(value) for value in ids],
            [(int(start), int(end)) for start, end in offsets],
        )


@dataclass(frozen=True)
class PromptGraphModelInputs:
    side_channels: Mapping[str, list[int]]
    graph_routes: Mapping[str, list[Any]]
    receipt: Mapping[str, Any]
    token_count: int
    schema: str = WINDOW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WINDOW_SCHEMA:
            raise ValueError(
                f"unsupported prompt graph window schema {self.schema!r}"
            )
        if self.receipt.get("schema") != WINDOW_SCHEMA:
            raise ValueError("prompt graph window receipt schema mismatch")
        if (
            self.receipt.get("symbol_identity_schema_version")
            != SYMBOL_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "prompt graph window requires canonical symbol identity v3"
            )
        if self.token_count <= 0:
            raise ValueError("prompt graph model input token_count must be positive")
        _validate_side_channels(self.side_channels, self.token_count)
        _validate_graph_routes(
            self.graph_routes,
            token_count=self.token_count,
        )

    def dense_relation_attention_bias(
        self,
        *,
        relation_weights: Mapping[str, float] | None = None,
    ) -> list[list[float]]:
        weights = {
            relation: 1.0
            for relation in (*PAIR_ROUTE_KEYS, *TRIPLE_ROUTE_KEYS)
        }
        if relation_weights is not None:
            unknown = set(relation_weights) - set(weights)
            if unknown:
                raise ValueError(
                    "unknown prompt graph relation weights: "
                    f"{sorted(unknown)}"
                )
            weights.update(
                {
                    relation: float(value)
                    for relation, value in relation_weights.items()
                }
            )
        for relation, weight in weights.items():
            if not math.isfinite(weight):
                raise ValueError(
                    f"prompt graph relation {relation!r} weight must be finite, "
                    f"got {weight}"
                )

        bias = [
            [0.0 for _ in range(self.token_count)]
            for _ in range(self.token_count)
        ]
        starts = [
            int(value)
            for value in self.graph_routes["graph_chunk_starts"]
        ]
        ends = [
            int(value)
            for value in self.graph_routes["graph_chunk_ends"]
        ]
        for relation, (route_key, _count_key) in PAIR_ROUTE_KEYS.items():
            weight = weights[relation]
            for source_chunk, target_chunk in self.graph_routes[route_key]:
                for query_index in range(
                    starts[int(source_chunk)],
                    ends[int(source_chunk)],
                ):
                    for key_index in range(
                        starts[int(target_chunk)],
                        ends[int(target_chunk)],
                    ):
                        bias[query_index][key_index] += weight
        for relation, (route_key, _count_key) in TRIPLE_ROUTE_KEYS.items():
            weight = weights[relation]
            for source_token, target_token, _kind in self.graph_routes[
                route_key
            ]:
                bias[int(source_token)][int(target_token)] += weight
        for source_token, target_token in self.graph_routes[
            GENERATED_QUERY_ROUTE_KEY
        ]:
            bias[int(source_token)][int(target_token)] += 1.0
        return bias

    def dense_edge_kind_attention_bias(
        self,
        *,
        edge_kind_weights: Mapping[int, float] | None = None,
        default_weight: float = 1.0,
    ) -> list[list[float]]:
        """Return the categorical edge-kind prior independently of relations."""

        default_weight = float(default_weight)
        if not math.isfinite(default_weight):
            raise ValueError(
                "prompt graph edge-kind default weight must be finite, "
                f"got {default_weight}"
            )
        if default_weight == 0.0:
            raise ValueError(
                "prompt graph edge-kind default weight must be nonzero; "
                "zero kind priors are reserved for explicit graph-off ablation"
            )
        weights: dict[int, float] = {}
        if edge_kind_weights is not None:
            known_kinds = {int(kind) for kind in VALID_DOMAIN_EDGE_KINDS}
            unknown = sorted(
                set(int(kind) for kind in edge_kind_weights) - known_kinds
            )
            if unknown:
                raise ValueError(
                    f"unknown prompt graph edge-kind weights: {unknown}"
                )
            weights = {
                int(kind): float(value)
                for kind, value in edge_kind_weights.items()
            }
        for kind, weight in weights.items():
            if not math.isfinite(weight):
                raise ValueError(
                    f"prompt graph edge kind {kind} weight must be finite, "
                    f"got {weight}"
                )
            if weight == 0.0:
                raise ValueError(
                    f"prompt graph edge kind {kind} weight must be nonzero; "
                    "zero kind priors are reserved for explicit graph-off ablation"
                )

        bias = [
            [0.0 for _ in range(self.token_count)]
            for _ in range(self.token_count)
        ]
        starts = [
            int(value)
            for value in self.graph_routes["graph_chunk_starts"]
        ]
        ends = [
            int(value)
            for value in self.graph_routes["graph_chunk_ends"]
        ]
        for relation, (route_key, _count_key) in PAIR_ROUTE_KEYS.items():
            kind = PAIR_ROUTE_EDGE_KINDS[relation]
            weight = weights.get(kind, default_weight)
            for source_chunk, target_chunk in self.graph_routes[route_key]:
                for query_index in range(
                    starts[int(source_chunk)],
                    ends[int(source_chunk)],
                ):
                    for key_index in range(
                        starts[int(target_chunk)],
                        ends[int(target_chunk)],
                    ):
                        bias[query_index][key_index] += weight
        for relation, (route_key, _count_key) in TRIPLE_ROUTE_KEYS.items():
            for source_token, target_token, kind in self.graph_routes[
                route_key
            ]:
                kind = _validate_route_kind(
                    relation,
                    int(kind),
                    where=(
                        f"{route_key} triple "
                        f"({source_token},{target_token},{kind})"
                    ),
                )
                bias[int(source_token)][int(target_token)] += weights.get(
                    kind,
                    default_weight,
                )
        return bias

    def edge_kind_route_counts(self) -> dict[int, int]:
        """Return categorical route counts before token-span expansion."""

        counts: dict[int, int] = {}
        for relation, (route_key, _count_key) in PAIR_ROUTE_KEYS.items():
            kind = PAIR_ROUTE_EDGE_KINDS[relation]
            counts[kind] = counts.get(kind, 0) + len(
                self.graph_routes[route_key]
            )
        for relation, (route_key, _count_key) in TRIPLE_ROUTE_KEYS.items():
            for source_token, target_token, kind in self.graph_routes[
                route_key
            ]:
                kind = _validate_route_kind(
                    relation,
                    int(kind),
                    where=(
                        f"{route_key} triple "
                        f"({source_token},{target_token},{kind})"
                    ),
                )
                counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))

    def dense_attention_bias(
        self,
        *,
        relation_weights: Mapping[str, float] | None = None,
        edge_kind_weights: Mapping[int, float] | None = None,
        default_edge_kind_weight: float = 1.0,
    ) -> list[list[float]]:
        """Return the combined relation and categorical edge-kind prior."""

        relation_bias = self.dense_relation_attention_bias(
            relation_weights=relation_weights,
        )
        edge_kind_bias = self.dense_edge_kind_attention_bias(
            edge_kind_weights=edge_kind_weights,
            default_weight=default_edge_kind_weight,
        )
        return [
            [
                relation + kind
                for relation, kind in zip(relation_row, kind_row)
            ]
            for relation_row, kind_row in zip(
                relation_bias,
                edge_kind_bias,
            )
        ]


@dataclass(frozen=True)
class PromptGraphArtifact:
    token_ids: tuple[int, ...]
    token_spans: tuple[tuple[int, int], ...]
    side_channels: Mapping[str, list[int]]
    graph_routes: Mapping[str, list[Any]]
    identity_token_spans: Mapping[str, tuple[int, int]]
    receipt: Mapping[str, Any]
    schema: str = ARTIFACT_SCHEMA

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def edge_counts(self) -> Mapping[str, int]:
        counts = self.receipt.get("edge_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("prompt graph receipt.edge_counts must be an object")
        return {
            relation: int(counts.get(relation, 0))
            for relation in RELATION_NAMES
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PromptGraphArtifact":
        if payload.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(
                "unsupported prompt graph artifact schema "
                f"{payload.get('schema')!r}"
            )
        try:
            artifact = cls(
                token_ids=tuple(
                    _require_int(value, where="token_ids[]")
                    for value in payload["token_ids"]
                ),
                token_spans=tuple(
                    (
                        _require_int(span[0], where="token_spans[][0]"),
                        _require_int(span[1], where="token_spans[][1]"),
                    )
                    for span in payload["token_spans"]
                ),
                side_channels={
                    str(key): [
                        _require_int(value, where=f"side_channels.{key}[]")
                        for value in values
                    ]
                    for key, values in dict(
                        payload["side_channels"]
                    ).items()
                },
                graph_routes={
                    str(key): _copy_route_values(values)
                    for key, values in dict(
                        payload["graph_routes"]
                    ).items()
                },
                identity_token_spans={
                    str(key): (
                        _require_int(value[0], where=f"identity.{key}[0]"),
                        _require_int(value[1], where=f"identity.{key}[1]"),
                    )
                    for key, value in dict(
                        payload["identity_token_spans"]
                    ).items()
                },
                receipt=dict(payload["receipt"]),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(
                "malformed prompt graph artifact payload"
            ) from exc
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.schema != ARTIFACT_SCHEMA:
            raise ValueError(
                f"unsupported prompt graph artifact schema {self.schema!r}"
            )
        if not self.token_ids:
            raise ValueError("prompt graph artifact has no token ids")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("prompt graph token ids must be non-negative")
        if len(self.token_spans) != self.token_count:
            raise ValueError(
                "prompt graph token ids/offsets length mismatch"
            )
        prompt_length = int(self.receipt.get("prompt_length", -1))
        if prompt_length <= 0:
            raise ValueError(
                "prompt graph receipt.prompt_length must be positive"
            )
        _validate_token_spans(self.token_spans, prompt_length)
        _validate_side_channels(self.side_channels, self.token_count)
        _validate_graph_routes(
            self.graph_routes,
            token_count=self.token_count,
        )
        for identity, (start, end) in self.identity_token_spans.items():
            _validate_span(
                int(start),
                int(end),
                self.token_count,
                where=f"identity token span {identity}",
            )

        if self.receipt.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(
                "prompt graph receipt schema does not match artifact"
            )
        if (
            self.receipt.get("symbol_identity_schema_version")
            != SYMBOL_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError(
                "prompt graph artifact requires canonical symbol identity v3"
            )
        if int(self.receipt.get("token_count", -1)) != self.token_count:
            raise ValueError(
                "prompt graph receipt token_count does not match artifact"
            )
        chunk_count = int(
            self.graph_routes["graph_chunk_counts"][0]
        )
        if int(self.receipt.get("chunk_count", -1)) != chunk_count:
            raise ValueError(
                "prompt graph receipt chunk_count does not match routes"
            )
        cache_key = self.receipt.get("cache_key")
        if not _is_sha256(cache_key):
            raise ValueError("prompt graph receipt.cache_key must be a sha256")
        hashes = self.receipt.get("hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError("prompt graph receipt.hashes is missing")
        required_hashes = {
            "source_sha256",
            "index_sha256",
            "prompt_sha256",
            "tokenizer_sha256",
            "context_sha256",
            "artifact_sha256",
        }
        if set(hashes) != required_hashes:
            raise ValueError(
                "prompt graph receipt hashes mismatch: "
                f"expected={sorted(required_hashes)} "
                f"actual={sorted(hashes)}"
            )
        if any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError("prompt graph receipt hashes must be sha256 values")
        expected_artifact_hash = _sha_json(self._hash_payload())
        if hashes["artifact_sha256"] != expected_artifact_hash:
            raise ValueError(
                "prompt graph artifact_sha256 mismatch: "
                f"receipt={hashes['artifact_sha256']} "
                f"actual={expected_artifact_hash}"
            )
        _ = self.edge_counts

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "token_ids": list(self.token_ids),
            "token_spans": [
                [start, end] for start, end in self.token_spans
            ],
            "side_channels": {
                key: list(values)
                for key, values in sorted(self.side_channels.items())
            },
            "graph_routes": {
                key: _copy_route_values(values)
                for key, values in sorted(self.graph_routes.items())
            },
            "identity_token_spans": {
                key: [value[0], value[1]]
                for key, value in sorted(
                    self.identity_token_spans.items()
                )
            },
        }

    def _hash_payload(self) -> dict[str, Any]:
        receipt = json.loads(_stable_json(self.receipt))
        hashes = receipt.get("hashes")
        if isinstance(hashes, dict):
            hashes.pop("artifact_sha256", None)
        return {"artifact": self._core_dict(), "receipt": receipt}

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._core_dict(),
            "receipt": json.loads(_stable_json(self.receipt)),
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())

    def first_token_for_identity(self, identity: str) -> int:
        if identity not in self.identity_token_spans:
            raise KeyError(
                f"identity {identity!r} is not present in prompt graph"
            )
        return int(self.identity_token_spans[identity][0])

    def source_token_kwargs(self) -> dict[str, list[int]]:
        return {
            key: list(value)
            for key, value in self.side_channels.items()
        }

    def model_inputs(
        self,
        *,
        total_token_count: int,
        window_start: int,
        window_end: int,
    ) -> PromptGraphModelInputs:
        if total_token_count < self.token_count:
            raise ValueError(
                "total_token_count cannot be shorter than prompt graph "
                f"artifact: total={total_token_count} "
                f"prompt={self.token_count}"
            )
        if (
            window_start < 0
            or window_end <= window_start
            or window_end > total_token_count
        ):
            raise ValueError(
                "invalid prompt graph model window "
                f"[{window_start},{window_end}) for total "
                f"{total_token_count}"
            )

        generated_count = total_token_count - self.token_count
        anchor_candidates = [
            index
            for index, value in enumerate(self.side_channels["structure_ids"])
            if int(value) > 0
        ]
        if not anchor_candidates:
            raise ValueError(
                "prompt graph artifact has no live structure token for generated decode"
            )
        anchor = anchor_candidates[-1]
        generated_values = dict(GENERATED_TOKEN_SIDECAR_DEFAULTS)
        for name in (
            "structure_ids",
            "dep_levels",
            "ast_depth_ids",
            "sibling_index_ids",
            "node_type_ids",
            "domain_ids",
        ):
            generated_values[name] = int(self.side_channels[name][anchor])
        generated_values["domain_ids"] = 1
        generated_values["confidence_ids"] = 1
        generated_values["source_doc_ids"] = 0
        side_channels = {
            name: (
                list(self.side_channels[name])
                + [generated_values[name]] * generated_count
            )[window_start:window_end]
            for name in TOKEN_SIDECAR_NAMES
        }

        old_starts = [
            int(value)
            for value in self.graph_routes["graph_chunk_starts"]
        ]
        old_ends = [
            int(value)
            for value in self.graph_routes["graph_chunk_ends"]
        ]
        old_kinds = [
            int(value)
            for value in self.graph_routes["graph_chunk_kinds"]
        ]
        old_dep_levels = [
            int(value)
            for value in self.graph_routes[
                "graph_chunk_dep_levels"
            ]
        ]
        old_to_new: dict[int, int] = {}
        starts: list[int] = []
        ends: list[int] = []
        kinds: list[int] = []
        dep_levels: list[int] = []
        for old_index, (start, end, kind, dep_level) in enumerate(
            zip(old_starts, old_ends, old_kinds, old_dep_levels)
        ):
            visible_start = max(start, window_start)
            visible_end = min(end, window_end)
            if visible_start >= visible_end:
                continue
            old_to_new[old_index] = len(starts)
            starts.append(visible_start - window_start)
            ends.append(visible_end - window_start)
            kinds.append(kind)
            dep_levels.append(dep_level)

        graph_routes: dict[str, list[Any]] = {}
        visible_route_counts: dict[str, int] = {}
        for relation, (route_key, count_key) in PAIR_ROUTE_KEYS.items():
            rows: list[list[int]] = []
            for source_chunk, target_chunk in self.graph_routes[route_key]:
                source_chunk = int(source_chunk)
                target_chunk = int(target_chunk)
                if (
                    source_chunk not in old_to_new
                    or target_chunk not in old_to_new
                ):
                    continue
                row = [
                    old_to_new[source_chunk],
                    old_to_new[target_chunk],
                ]
                if row not in rows:
                    rows.append(row)
            graph_routes[route_key] = rows
            graph_routes[count_key] = [len(rows)]
            visible_route_counts[relation] = len(rows)

        for relation, (route_key, count_key) in TRIPLE_ROUTE_KEYS.items():
            rows = []
            for source_token, target_token, kind in self.graph_routes[
                route_key
            ]:
                source_token = int(source_token)
                target_token = int(target_token)
                if not (
                    window_start <= source_token < window_end
                    and window_start <= target_token < window_end
                ):
                    continue
                row = [
                    source_token - window_start,
                    target_token - window_start,
                    int(kind),
                ]
                if row not in rows:
                    rows.append(row)
            graph_routes[route_key] = rows
            graph_routes[count_key] = [len(rows)]
            visible_route_counts[relation] = len(rows)

        summary_tokens: set[int] = set()
        for _relation, (route_key, _count_key) in PAIR_ROUTE_KEYS.items():
            for source_chunk, target_chunk in graph_routes[route_key]:
                summary_tokens.add(starts[int(source_chunk)])
                summary_tokens.add(starts[int(target_chunk)])
        for _relation, (route_key, _count_key) in TRIPLE_ROUTE_KEYS.items():
            for source_token, target_token, _kind in graph_routes[route_key]:
                summary_tokens.add(int(source_token))
                summary_tokens.add(int(target_token))

        generated_visible_start = max(self.token_count, window_start)
        generated_visible_end = window_end
        generated_rows: list[list[int]] = []
        if generated_visible_start < generated_visible_end:
            if not summary_tokens:
                raise ValueError(
                    "generated prompt graph queries have no visible repository summary"
                )
            starts.append(generated_visible_start - window_start)
            ends.append(generated_visible_end - window_start)
            kinds.append(int(generated_values["structure_ids"]))
            dep_levels.append(int(generated_values["dep_levels"]))
            for query_token in range(
                generated_visible_start - window_start,
                generated_visible_end - window_start,
            ):
                generated_rows.extend(
                    [query_token, target_token]
                    for target_token in sorted(summary_tokens)
                )
        graph_routes[GENERATED_QUERY_ROUTE_KEY] = generated_rows
        graph_routes[GENERATED_QUERY_COUNT_KEY] = [len(generated_rows)]

        graph_routes.update(
            {
                "graph_chunk_starts": starts,
                "graph_chunk_ends": ends,
                "graph_chunk_kinds": kinds,
                "graph_chunk_dep_levels": dep_levels,
                "graph_chunk_counts": [len(starts)],
            }
        )
        receipt = {
            "schema": WINDOW_SCHEMA,
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "artifact_cache_key": self.receipt["cache_key"],
            "hashes": dict(self.receipt["hashes"]),
            "edge_counts": dict(self.edge_counts),
            "visible_route_edge_counts": visible_route_counts,
            "chunk_count": len(starts),
            "window_start": window_start,
            "window_end": window_end,
            "total_token_count": total_token_count,
            "token_count": window_end - window_start,
            "generated_token_policy": GENERATED_TOKEN_POLICY,
            "generated_token_count": max(
                0, generated_visible_end - generated_visible_start
            ),
            "repository_summary_token_count": len(summary_tokens),
            "generated_query_edge_count": len(generated_rows),
        }
        return PromptGraphModelInputs(
            side_channels=side_channels,
            graph_routes=graph_routes,
            receipt=receipt,
            token_count=window_end - window_start,
        )


class PromptGraphBuilder:
    """Build and cache deterministic graph artifacts from a typed project index."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        tokenizer_hash: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_hash = str(
            tokenizer_hash or _infer_tokenizer_hash(tokenizer)
        ).lower()
        if not _is_sha256(self.tokenizer_hash):
            raise ValueError(
                "PromptGraphBuilder tokenizer_hash must be a sha256 hex digest"
            )
        self.cache_dir = (
            None if cache_dir is None else Path(cache_dir)
        )

    def cache_key(
        self,
        project_index: PromptProjectIndex,
        context: PromptGraphContext,
    ) -> str:
        return _sha_json(
            {
                "artifact_schema": ARTIFACT_SCHEMA,
                "builder_version": BUILDER_VERSION,
                "source_sha256": project_index.source_sha256,
                "index_sha256": project_index.index_sha256,
                "tokenizer_sha256": self.tokenizer_hash,
                "prompt_sha256": _sha_text(context.text),
                "context_sha256": context.context_sha256,
            }
        )

    def build(
        self,
        project_index: PromptProjectIndex,
        context: PromptGraphContext,
    ) -> PromptGraphArtifact:
        project_index.validate()
        prompt = context.text
        if not prompt:
            raise ValueError("PromptGraphBuilder: prompt text is empty")
        _validate_context_source_text(project_index, context)
        cache_key = self.cache_key(project_index, context)
        hashes = {
            "source_sha256": project_index.source_sha256,
            "index_sha256": project_index.index_sha256,
            "prompt_sha256": _sha_text(prompt),
            "tokenizer_sha256": self.tokenizer_hash,
            "context_sha256": context.context_sha256,
        }

        cache_path = (
            None
            if self.cache_dir is None
            else self.cache_dir / f"{cache_key}.json"
        )
        if cache_path is not None and cache_path.exists():
            return self._load_cached(
                cache_path,
                cache_key=cache_key,
                hashes=hashes,
            )

        artifact = self._build_uncached(
            project_index,
            context,
            cache_key=cache_key,
            hashes=hashes,
        )
        if cache_path is not None:
            self._write_cached(cache_path, artifact)
        return artifact

    def _build_uncached(
        self,
        project_index: PromptProjectIndex,
        context: PromptGraphContext,
        *,
        cache_key: str,
        hashes: Mapping[str, str],
    ) -> PromptGraphArtifact:
        prompt = context.text
        token_ids, token_spans = _encode_with_offsets(
            self.tokenizer,
            prompt,
        )
        if len(token_ids) != len(token_spans):
            raise ValueError(
                "PromptGraphBuilder: tokenizer ids/offsets length mismatch"
            )
        if not token_ids:
            raise ValueError(
                "PromptGraphBuilder: prompt tokenized to zero tokens"
            )
        _validate_token_spans(token_spans, len(prompt))

        prompt_to_source = context.source_positions()
        if len(prompt_to_source) != len(prompt):
            raise ValueError(
                "PromptGraphBuilder: prompt/source position map "
                "length mismatch"
            )
        _validate_prompt_source_bounds(
            project_index,
            prompt_to_source,
        )
        _validate_token_source_alignment(token_spans, prompt_to_source)
        token_source_sets = _token_source_sets(
            token_spans,
            prompt_to_source,
        )
        side_channels = _empty_side_channels(len(token_ids))
        _map_source_documents_to_tokens(token_source_sets, side_channels)
        identity_token_spans = _map_symbols_to_tokens(
            project_index,
            token_source_sets,
            side_channels,
        )
        chunk_rows = _map_chunks_to_token_rows(
            project_index,
            token_source_sets,
            side_channels,
        )
        graph_routes, edge_counts, route_edge_counts = _map_edges(
            project_index,
            chunk_rows=chunk_rows,
            identity_token_spans=identity_token_spans,
            side_channels=side_channels,
        )
        if sum(edge_counts.values()) == 0:
            raise ValueError(
                "PromptGraphBuilder: project index produced no visible "
                "graph relations for the requested prompt"
            )
        if sum(route_edge_counts.values()) == 0:
            raise ValueError(
                "PromptGraphBuilder: project index produced no model-routable "
                "graph relations for the requested prompt"
            )

        graph_routes.update(
            {
                "graph_chunk_starts": [
                    row[1] for row in chunk_rows
                ],
                "graph_chunk_ends": [
                    row[2] for row in chunk_rows
                ],
                "graph_chunk_kinds": [
                    row[3] for row in chunk_rows
                ],
                "graph_chunk_dep_levels": [
                    row[4] for row in chunk_rows
                ],
                "graph_chunk_counts": [len(chunk_rows)],
            }
        )
        _validate_graph_routes(
            graph_routes,
            token_count=len(token_ids),
        )

        receipt: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "cache_key": cache_key,
            "project_id": project_index.project_id,
            "document_count": len(project_index.documents),
            "source_paths": [
                document.source_path
                for document in sorted(
                    project_index.documents, key=lambda item: item.id
                )
            ],
            "prompt_length": len(prompt),
            "token_count": len(token_ids),
            "symbol_count": len(identity_token_spans),
            "chunk_count": len(chunk_rows),
            "edge_counts": {
                relation: int(edge_counts[relation])
                for relation in RELATION_NAMES
            },
            "route_edge_counts": {
                relation: int(route_edge_counts.get(relation, 0))
                for relation in (
                    *PAIR_ROUTE_KEYS,
                    *TRIPLE_ROUTE_KEYS,
                )
            },
            "hashes": dict(hashes),
            "provenance": {
                "builder": "PromptGraphBuilder",
                "builder_version": BUILDER_VERSION,
                "index_schema": project_index.schema,
                "context_language": context.language,
                "context_segments": [
                    {
                        "length": len(segment.text),
                        "role": segment.role,
                        "document_id": segment.document_id,
                        "source_path": segment.source_path,
                        "source_start": segment.source_start,
                        "text_sha256": _sha_text(segment.text),
                    }
                    for segment in context.segments
                ],
            },
        }
        artifact_without_hash = PromptGraphArtifact(
            token_ids=tuple(token_ids),
            token_spans=tuple(token_spans),
            side_channels=side_channels,
            graph_routes=graph_routes,
            identity_token_spans=identity_token_spans,
            receipt=receipt,
        )
        final_receipt = json.loads(_stable_json(receipt))
        final_receipt["hashes"]["artifact_sha256"] = _sha_json(
            artifact_without_hash._hash_payload()
        )
        artifact = PromptGraphArtifact(
            token_ids=artifact_without_hash.token_ids,
            token_spans=artifact_without_hash.token_spans,
            side_channels=artifact_without_hash.side_channels,
            graph_routes=artifact_without_hash.graph_routes,
            identity_token_spans=(
                artifact_without_hash.identity_token_spans
            ),
            receipt=final_receipt,
        )
        artifact.validate()
        return artifact

    @staticmethod
    def _load_cached(
        path: Path,
        *,
        cache_key: str,
        hashes: Mapping[str, str],
    ) -> PromptGraphArtifact:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("cache payload is not an object")
            artifact = PromptGraphArtifact.from_dict(payload)
            if artifact.receipt["cache_key"] != cache_key:
                raise ValueError(
                    "cache key does not match requested graph"
                )
            actual_hashes = artifact.receipt["hashes"]
            for name, expected in hashes.items():
                if actual_hashes.get(name) != expected:
                    raise ValueError(
                        f"cache {name} does not match requested graph"
                    )
            return artifact
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                f"cached prompt graph artifact {path} is invalid: {exc}"
            ) from exc

    def _write_cached(
        self,
        path: Path,
        artifact: PromptGraphArtifact,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(artifact.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


def _copy_route_values(values: Any) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(
        values,
        (str, bytes),
    ):
        raise TypeError("graph route values must be a sequence")
    copied: list[Any] = []
    for value in values:
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            copied.append(
                [
                    _require_int(item, where="graph route item")
                    for item in value
                ]
            )
        else:
            copied.append(
                _require_int(value, where="graph route value")
            )
    return copied


def _infer_tokenizer_hash(tokenizer: Any) -> str:
    for raw_path in (
        getattr(tokenizer, "path", None),
        getattr(tokenizer, "name_or_path", None),
    ):
        if raw_path is None:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if candidate.is_dir():
            candidate = candidate / "tokenizer.json"
        if candidate.is_file():
            return _sha_file(candidate)

    for backend in (
        getattr(tokenizer, "_tokenizer", None),
        getattr(tokenizer, "backend_tokenizer", None),
    ):
        to_str = getattr(backend, "to_str", None)
        if callable(to_str):
            return _sha_text(str(to_str()))

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, Mapping) and vocab:
            return _sha_json(
                {
                    "class": (
                        f"{tokenizer.__class__.__module__}."
                        f"{tokenizer.__class__.__qualname__}"
                    ),
                    "vocab": {
                        str(key): int(value)
                        for key, value in vocab.items()
                    },
                }
            )

    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return _sha_json(
            {
                "class": (
                    f"{tokenizer.__class__.__module__}."
                    f"{tokenizer.__class__.__qualname__}"
                ),
                "name_or_path": name,
                "vocab_size": getattr(
                    tokenizer,
                    "vocab_size",
                    None,
                ),
            }
        )
    raise TypeError(
        "PromptGraphBuilder cannot derive a tokenizer hash; pass "
        "tokenizer_hash=<sha256> or use a tokenizer backed by tokenizer.json"
    )


def _encode_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    method = getattr(tokenizer, "encode_with_offsets", None)
    if callable(method):
        ids, offsets = method(text)
        return (
            [int(value) for value in ids],
            [
                (int(start), int(end))
                for start, end in offsets
            ],
        )

    if callable(tokenizer):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = (
            encoded["input_ids"]
            if isinstance(encoded, Mapping)
            else encoded.input_ids
        )
        offsets = (
            encoded["offset_mapping"]
            if isinstance(encoded, Mapping)
            else encoded.offset_mapping
        )
        if ids and isinstance(ids[0], Sequence):
            raise ValueError(
                "PromptGraphBuilder tokenizer returned batched ids "
                "for one prompt"
            )
        return (
            [int(value) for value in ids],
            [
                (int(start), int(end))
                for start, end in offsets
            ],
        )

    raise TypeError(
        "PromptGraphBuilder graph mode requires a tokenizer with "
        "encode_with_offsets(text) or a fast tokenizer callable with offsets"
    )


def _validate_token_spans(
    spans: Sequence[tuple[int, int]],
    prompt_len: int,
) -> None:
    last_start = 0
    last_end = 0
    for index, (start, end) in enumerate(spans):
        if start < 0 or end <= start or end > prompt_len:
            raise ValueError(
                f"token offset {index} has invalid span "
                f"[{start},{end}) for prompt length {prompt_len}"
            )
        if start < last_start or end < last_end:
            raise ValueError(
                "token offsets must be monotonic"
            )
        last_start = start
        last_end = end


def _validate_prompt_source_bounds(
    index: PromptProjectIndex,
    positions: Sequence[tuple[int, int] | None],
) -> None:
    for reference in positions:
        if reference is None:
            continue
        document_id, position = reference
        try:
            document = index.document_for_id(document_id)
        except KeyError as exc:
            raise ValueError(
                f"prompt source references unknown document_id {document_id}"
            ) from exc
        if not 0 <= position < len(document.source):
            raise ValueError(
                f"prompt source offset {position} outside {document.source_path} "
                f"bounds length {len(document.source)}"
            )


def _validate_context_source_text(
    index: PromptProjectIndex,
    context: PromptGraphContext,
) -> None:
    for segment_number, segment in enumerate(context.segments):
        if segment.source_start is None:
            continue
        assert segment.document_id is not None
        assert segment.source_path is not None
        document = index.document_for_id(segment.document_id)
        if document.source_path != segment.source_path:
            raise ValueError(
                f"prompt segment {segment_number} document_id/source_path mismatch"
            )
        start = int(segment.source_start)
        end = start + len(segment.text)
        if start < 0 or end > len(document.source):
            raise ValueError(
                f"prompt segment {segment_number} source span "
                f"[{start},{end}) is outside indexed source length "
                f"{len(document.source)}"
            )
        if document.source[start:end] != segment.text:
            raise ValueError(
                f"prompt segment {segment_number} text does not match "
                f"indexed source span [{start},{end})"
            )


def _token_source_sets(
    token_spans: Sequence[tuple[int, int]],
    prompt_to_source: Sequence[tuple[int, int] | None],
) -> list[set[tuple[int, int]]]:
    return [
        {
            (int(reference[0]), int(reference[1]))
            for reference in prompt_to_source[start:end]
            if reference is not None
        }
        for start, end in token_spans
    ]


def _validate_token_source_alignment(
    token_spans: Sequence[tuple[int, int]],
    prompt_to_source: Sequence[tuple[int, int] | None],
) -> None:
    """Reject tokenizer pieces that mix synthetic and source-mapped text."""

    for token_index, (start, end) in enumerate(token_spans):
        references = prompt_to_source[start:end]
        mapped_count = sum(reference is not None for reference in references)
        if mapped_count and mapped_count != len(references):
            raise ValueError(
                "prompt graph token sidecar alignment crosses mapped and "
                f"unmapped text at token {token_index}: [{start},{end})"
            )


def _empty_side_channels(
    token_count: int,
) -> dict[str, list[int]]:
    return {
        name: [TOKEN_SIDECAR_DEFAULTS[name]] * token_count
        for name in TOKEN_SIDECAR_NAMES
    }


def _map_source_documents_to_tokens(
    token_source_sets: Sequence[set[tuple[int, int]]],
    side_channels: dict[str, list[int]],
) -> None:
    for token_index, references in enumerate(token_source_sets):
        document_ids = {document_id for document_id, _position in references}
        if len(document_ids) > 1:
            raise ValueError(
                f"token {token_index} crosses prompt graph documents: "
                f"{sorted(document_ids)}"
            )
        if document_ids:
            side_channels["source_doc_ids"][token_index] = next(
                iter(document_ids)
            )


def _tokens_overlapping_span(
    token_source_sets: Sequence[set[tuple[int, int]]],
    document_id: int,
    start: int,
    end: int,
) -> list[int]:
    return [
        index
        for index, positions in enumerate(token_source_sets)
        if any(
            source_document_id == document_id and start <= position < end
            for source_document_id, position in positions
        )
    ]


def _map_symbols_to_tokens(
    index: PromptProjectIndex,
    token_source_sets: Sequence[set[tuple[int, int]]],
    side_channels: dict[str, list[int]],
) -> dict[str, tuple[int, int]]:
    identity_token_spans: dict[str, tuple[int, int]] = {}
    for symbol in sorted(
        index.symbols,
        key=lambda item: (
            item.document_id,
            -(item.end - item.start),
            item.start,
            item.id,
        ),
    ):
        token_indexes = _tokens_overlapping_span(
            token_source_sets,
            symbol.document_id,
            symbol.start,
            symbol.end,
        )
        if not token_indexes:
            continue
        token_span = (
            token_indexes[0],
            token_indexes[-1] + 1,
        )
        identity_token_spans[symbol.identity] = token_span
        for token_index in token_indexes:
            side_channels["symbol_ids"][token_index] = symbol.symbol_id
            if symbol.kind in {
                "struct",
                "class",
                "type",
                "enum",
                "type_use",
            }:
                side_channels["type_refs"][token_index] = symbol.symbol_id
            if symbol.kind in {"function", "type", "variable"}:
                side_channels["def_use"][token_index] = 1
            elif symbol.kind == "use":
                side_channels["def_use"][token_index] = 2
    return identity_token_spans


def _map_chunks_to_token_rows(
    index: PromptProjectIndex,
    token_source_sets: Sequence[set[tuple[int, int]]],
    side_channels: dict[str, list[int]],
) -> list[tuple[int, int, int, int, int, PromptGraphChunk]]:
    rows: list[
        tuple[int, int, int, int, int, PromptGraphChunk]
    ] = []
    for chunk in sorted(
        index.chunks,
        key=lambda item: (
            item.document_id,
            item.start,
            item.end,
            item.id,
        ),
    ):
        token_indexes = _tokens_overlapping_span(
            token_source_sets,
            chunk.document_id,
            chunk.start,
            chunk.end,
        )
        if not token_indexes:
            continue
        start_token = token_indexes[0]
        end_token = token_indexes[-1] + 1
        rows.append(
            (
                chunk.id,
                start_token,
                end_token,
                chunk.kind,
                chunk.dep_level,
                chunk,
            )
        )
        for token_index in token_indexes:
            side_channels["structure_ids"][token_index] = chunk.kind
            side_channels["dep_levels"][token_index] = (
                chunk.dep_level
            )
            side_channels["ast_depth_ids"][token_index] = max(
                side_channels["ast_depth_ids"][token_index],
                chunk.dep_level + 1,
            )
            side_channels["node_type_ids"][token_index] = chunk.kind

    rows.sort(key=lambda row: (row[1], row[2], row[0]))
    for previous, current in zip(rows, rows[1:]):
        if current[1] < previous[2]:
            raise ValueError(
                "prompt graph chunk token spans overlap or are out of order"
            )
    return rows


def _map_edges(
    index: PromptProjectIndex,
    *,
    chunk_rows: Sequence[
        tuple[int, int, int, int, int, PromptGraphChunk]
    ],
    identity_token_spans: Mapping[str, tuple[int, int]],
    side_channels: dict[str, list[int]],
) -> tuple[
    dict[str, list[Any]],
    dict[str, int],
    dict[str, int],
]:
    graph_routes: dict[str, list[Any]] = {}
    edge_counts = {relation: 0 for relation in RELATION_NAMES}
    route_edge_counts: dict[str, int] = {}
    symbol_by_identity = {
        symbol.identity: symbol for symbol in index.symbols
    }
    symbol_to_chunk = _symbol_to_chunk_rows(
        index,
        chunk_rows,
    )

    for edge in index.edges:
        if (
            edge.source not in identity_token_spans
            or edge.target not in identity_token_spans
        ):
            continue
        edge_counts[edge.relation] += 1
        source_symbol = symbol_by_identity[edge.source]
        target_symbol = symbol_by_identity[edge.target]
        source_span = identity_token_spans[edge.source]
        source_first = source_span[0]
        target_first = identity_token_spans[edge.target][0]

        if edge.relation == "call":
            _fill_side_channel_range(
                side_channels["call_targets"],
                source_span,
                target_symbol.symbol_id,
            )
        elif edge.relation == "type":
            _fill_side_channel_range(
                side_channels["type_refs"],
                source_span,
                target_symbol.symbol_id,
            )
        elif edge.relation == "def_use":
            _fill_side_channel_range(
                side_channels["def_use"],
                source_span,
                2,
            )
            _fill_side_channel_range(
                side_channels["def_use"],
                identity_token_spans[edge.target],
                1,
            )

        if edge.relation in PAIR_ROUTE_KEYS:
            source_chunk = symbol_to_chunk.get(
                source_symbol.identity
            )
            target_chunk = symbol_to_chunk.get(
                target_symbol.identity
            )
            if source_chunk is None or target_chunk is None:
                raise ValueError(
                    f"{edge.relation} edge "
                    f"{edge.source!r}->{edge.target!r} does not map "
                    "to prompt chunks"
                )
            route_key, _count_key = PAIR_ROUTE_KEYS[
                edge.relation
            ]
            pair = [source_chunk, target_chunk]
            if pair not in graph_routes.setdefault(route_key, []):
                graph_routes[route_key].append(pair)
        elif edge.relation in TRIPLE_ROUTE_KEYS:
            route_key, _count_key = TRIPLE_ROUTE_KEYS[
                edge.relation
            ]
            kind = _validate_route_kind(
                edge.relation,
                edge.kind,
                where=f"edge {edge.source!r}->{edge.target!r}",
            )
            triple = [source_first, target_first, kind]
            if triple not in graph_routes.setdefault(route_key, []):
                graph_routes[route_key].append(triple)

    for relation, (route_key, count_key) in PAIR_ROUTE_KEYS.items():
        rows = graph_routes.setdefault(route_key, [])
        graph_routes[count_key] = [len(rows)]
        route_edge_counts[relation] = len(rows)
    for relation, (route_key, count_key) in TRIPLE_ROUTE_KEYS.items():
        rows = graph_routes.setdefault(route_key, [])
        graph_routes[count_key] = [len(rows)]
        route_edge_counts[relation] = len(rows)
    graph_routes[GENERATED_QUERY_ROUTE_KEY] = []
    graph_routes[GENERATED_QUERY_COUNT_KEY] = [0]
    return graph_routes, edge_counts, route_edge_counts


def _fill_side_channel_range(
    values: list[int],
    token_span: tuple[int, int],
    value: int,
) -> None:
    start, end = token_span
    for index in range(start, end):
        values[index] = int(value)


def _symbol_to_chunk_rows(
    index: PromptProjectIndex,
    chunk_rows: Sequence[
        tuple[int, int, int, int, int, PromptGraphChunk]
    ],
) -> dict[str, int]:
    row_for_chunk_id = {
        chunk.id: row_index
        for row_index, (*_prefix, chunk) in enumerate(chunk_rows)
    }
    result: dict[str, int] = {}
    for symbol in index.symbols:
        if symbol.chunk_identity:
            matches = [
                chunk
                for *_prefix, chunk in chunk_rows
                if chunk.identity == symbol.chunk_identity
            ]
            if len(matches) == 1:
                result[symbol.identity] = row_for_chunk_id[matches[0].id]
                continue
        containing = [
            chunk
            for *_prefix, chunk in chunk_rows
            if (
                chunk.document_id == symbol.document_id
                and
                chunk.start <= symbol.start
                and symbol.end <= chunk.end
            )
        ]
        if not containing:
            containing = [
                chunk
                for *_prefix, chunk in chunk_rows
                if chunk.document_id == symbol.document_id
                and max(chunk.start, symbol.start)
                < min(chunk.end, symbol.end)
            ]
        if containing:
            selected = min(
                containing,
                key=lambda item: (
                    item.end - item.start,
                    item.id,
                ),
            )
            result[symbol.identity] = row_for_chunk_id[
                selected.id
            ]
    return result


def _validate_side_channels(
    side_channels: Mapping[str, Sequence[int]],
    token_count: int,
) -> None:
    if set(side_channels) != set(TOKEN_SIDECAR_NAMES):
        raise ValueError(
            "prompt graph side-channel names mismatch: "
            f"expected={sorted(TOKEN_SIDECAR_NAMES)} "
            f"actual={sorted(side_channels)}"
        )
    for name in TOKEN_SIDECAR_NAMES:
        values = side_channels[name]
        if len(values) != token_count:
            raise ValueError(
                f"prompt graph side channel {name} length "
                f"{len(values)} != token_count {token_count}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            raise ValueError(
                f"prompt graph side channel {name} must contain "
                "non-negative integers"
            )
        if name in _OPAQUE_SYMBOL_ID_SIDECARS and any(
            value > SYMBOL_ID_MAX for value in values
        ):
            raise ValueError(
                f"prompt graph side channel {name} must contain uint64 IDs"
            )
        if name == "def_use" and any(
            value not in _DEF_USE_VALUES for value in values
        ):
            raise ValueError(
                "prompt graph token_def_use values must be only 0/1/2"
            )


def _validate_graph_routes(
    graph_routes: Mapping[str, list[Any]],
    *,
    token_count: int,
) -> None:
    required = {
        "graph_chunk_starts",
        "graph_chunk_ends",
        "graph_chunk_kinds",
        "graph_chunk_dep_levels",
        "graph_chunk_counts",
        GENERATED_QUERY_ROUTE_KEY,
        GENERATED_QUERY_COUNT_KEY,
    }
    for route_key, count_key in (
        *PAIR_ROUTE_KEYS.values(),
        *TRIPLE_ROUTE_KEYS.values(),
    ):
        required.update((route_key, count_key))
    missing = required - set(graph_routes)
    if missing:
        raise ValueError(
            f"prompt graph routes missing keys: {sorted(missing)}"
        )

    starts = [
        int(value)
        for value in graph_routes["graph_chunk_starts"]
    ]
    ends = [
        int(value)
        for value in graph_routes["graph_chunk_ends"]
    ]
    kinds = [
        int(value)
        for value in graph_routes["graph_chunk_kinds"]
    ]
    dep_levels = [
        int(value)
        for value in graph_routes["graph_chunk_dep_levels"]
    ]
    chunk_counts = graph_routes["graph_chunk_counts"]
    if len(chunk_counts) != 1:
        raise ValueError(
            "graph_chunk_counts must contain one batch count"
        )
    chunk_count = int(chunk_counts[0])
    if not (
        len(starts)
        == len(ends)
        == len(kinds)
        == len(dep_levels)
        == chunk_count
    ):
        raise ValueError(
            "prompt graph chunk arrays/count length mismatch"
        )
    last_end = -1
    for start, end, kind, dep_level in zip(
        starts,
        ends,
        kinds,
        dep_levels,
    ):
        if (
            start < 0
            or end <= start
            or end > token_count
            or kind < 0
            or dep_level < 0
        ):
            raise ValueError(
                "prompt graph chunk span/metadata out of bounds: "
                f"[{start},{end}) kind={kind} dep_level={dep_level} "
                f"token_count={token_count}"
            )
        if start < last_end:
            raise ValueError(
                "prompt graph chunk spans overlap or are out of order"
            )
        last_end = end

    for route_key, count_key in PAIR_ROUTE_KEYS.values():
        rows = graph_routes[route_key]
        counts = graph_routes[count_key]
        if len(counts) != 1 or int(counts[0]) != len(rows):
            raise ValueError(
                f"{route_key} count does not match route rows"
            )
        for row in rows:
            if len(row) != 2:
                raise ValueError(
                    f"{route_key} rows must be edge pairs"
                )
            source, target = (int(row[0]), int(row[1]))
            if not (
                0 <= source < chunk_count
                and 0 <= target < chunk_count
            ):
                raise ValueError(
                    f"{route_key} pair ({source},{target}) out of "
                    f"graph chunk bounds {chunk_count}"
                )

    for relation, (route_key, count_key) in TRIPLE_ROUTE_KEYS.items():
        rows = graph_routes[route_key]
        counts = graph_routes[count_key]
        if len(counts) != 1 or int(counts[0]) != len(rows):
            raise ValueError(
                f"{route_key} count does not match route rows"
            )
        for row in rows:
            if len(row) != 3:
                raise ValueError(
                    f"{route_key} rows must be edge triples"
                )
            source, target, kind = (
                int(row[0]),
                int(row[1]),
                int(row[2]),
            )
            if not (
                0 <= source < token_count
                and 0 <= target < token_count
            ):
                raise ValueError(
                    f"{route_key} triple ({source},{target},{kind}) "
                    f"out of token bounds {token_count}"
                )
            _validate_route_kind(
                relation,
                kind,
                where=f"{route_key} triple ({source},{target},{kind})",
            )

    generated_rows = graph_routes[GENERATED_QUERY_ROUTE_KEY]
    generated_counts = graph_routes[GENERATED_QUERY_COUNT_KEY]
    if len(generated_counts) != 1 or int(generated_counts[0]) != len(
        generated_rows
    ):
        raise ValueError(
            f"{GENERATED_QUERY_ROUTE_KEY} count does not match route rows"
        )
    for row in generated_rows:
        if len(row) != 2:
            raise ValueError(
                f"{GENERATED_QUERY_ROUTE_KEY} rows must be token pairs"
            )
        source, target = int(row[0]), int(row[1])
        if not (
            0 <= source < token_count and 0 <= target < token_count
        ):
            raise ValueError(
                f"{GENERATED_QUERY_ROUTE_KEY} pair ({source},{target}) out of "
                f"token bounds {token_count}"
            )


__all__ = [
    "ARTIFACT_SCHEMA",
    "BUILDER_VERSION",
    "CPP_NEWLINE_SENTINEL",
    "CPP_SPACE_SENTINEL",
    "CppPromptTokenizerAdapter",
    "GENERATED_QUERY_COUNT_KEY",
    "GENERATED_QUERY_ROUTE_KEY",
    "GENERATED_TOKEN_POLICY",
    "GENERATED_TOKEN_SIDECAR_DEFAULTS",
    "INDEX_INTEGRITY_VERSION",
    "INDEX_SCHEMA",
    "INDEX_PAYLOAD_HASH_KEY",
    "LEGACY_INDEX_SCHEMA",
    "PAIR_ROUTE_KEYS",
    "PAIR_ROUTE_EDGE_KINDS",
    "PromptGraphArtifact",
    "PromptGraphBuilder",
    "PromptGraphChunk",
    "PromptGraphContext",
    "PromptGraphDocument",
    "PromptGraphEdge",
    "PromptGraphModelInputs",
    "PromptGraphSegment",
    "PromptGraphSymbol",
    "PromptProjectIndex",
    "PRODUCTION_INDEX_PRODUCER",
    "PRODUCTION_INDEX_VERSION",
    "PRODUCTION_IDENTITY_PROVENANCE_CONTRACT",
    "TOKEN_SIDECAR_DEFAULTS",
    "TOKEN_SIDECAR_NAMES",
    "TRIPLE_ROUTE_KEYS",
    "TRUSTED_IDENTITY_ADAPTERS",
    "WINDOW_SCHEMA",
    "normalize_cpp_whitespace_with_offsets",
    "require_prompt_graph_project_id",
    "repository_snapshot",
]
