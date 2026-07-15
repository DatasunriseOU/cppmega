"""Shared helpers for deterministic build-domain parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cppmega.data.domain_packet import DomainPacket

from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    domain_edge_family,
    validate_domain_edge_kind,
)
from cppmega.data.source_identity import (
    MAX_ROW_LOCAL_DOC_ID,
    MAX_SOURCE_ID,
    SourceIdentity,
    validate_source_identity_registry,
)


_TOKEN_RE = re.compile(
    r'"[^"\n]*"|\'[^\'\n]*\'|\$\(?[A-Za-z_][A-Za-z0-9_]*\)?|::|&&|\|\||>>|2>|&>|'
    r'[A-Za-z0-9_./@%+$\\<>-]+|[:=(),;\[\]{}|<>]|[^\s]',
    re.MULTILINE,
)


@dataclass(frozen=True)
class LexedToken:
    text: str
    start: int
    end: int
    line: int
    column: int


@dataclass
class ParsedDomainDocument:
    domain: DomainKind
    text: str
    tokens: list[LexedToken]
    domain_ids: list[int]
    role_ids: list[int]
    entity_ids: list[int]
    scope_ids: list[int]
    source_doc_ids: list[int]
    source_identity_ids: list[int]
    confidence_ids: list[int]
    source_identity_registry: list[dict[str, int | str]] = field(default_factory=list)
    edges: list[tuple[int, int, int]] = field(default_factory=list)
    embedded_blocks: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        domain: DomainKind,
        text: str,
        confidence: ParseConfidence = ParseConfidence.HEURISTIC,
        source_doc_id: int = 0,
        source_identity: SourceIdentity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ParsedDomainDocument":
        tokens = lex_text(text)
        n = len(tokens)
        return cls(
            domain=domain,
            text=text,
            tokens=tokens,
            domain_ids=[int(domain)] * n,
            role_ids=[int(DomainRoleKind.NONE)] * n,
            entity_ids=[0] * n,
            scope_ids=[0] * n,
            source_doc_ids=[int(source_doc_id)] * n,
            source_identity_ids=[
                0 if source_identity is None else int(source_identity.source_identity_id)
            ] * n,
            confidence_ids=[int(confidence)] * n,
            source_identity_registry=(
                [] if source_identity is None else [source_identity.as_dict()]
            ),
            metadata=dict(metadata or {}),
        )

    def mark_raw(self, reason: str) -> "ParsedDomainDocument":
        """Keep the text/domain but explicitly mark unsupported syntax."""

        self.confidence_ids = [int(ParseConfidence.RAW)] * len(self.tokens)
        self.metadata["unsupported_syntax"] = reason
        self.metadata["raw_reason"] = reason
        return self

    def set_source_doc_id(self, source_doc_id: int) -> None:
        if not 0 <= int(source_doc_id) <= MAX_ROW_LOCAL_DOC_ID:
            raise ValueError("source_doc_id must be uint32")
        self.source_doc_ids = [int(source_doc_id)] * len(self.tokens)

    def set_source_identity(self, identity: SourceIdentity) -> None:
        identity_id = int(identity.source_identity_id)
        if not 0 < identity_id <= MAX_SOURCE_ID:
            raise ValueError("source_identity_id must be positive uint64")
        self.source_identity_ids = [identity_id] * len(self.tokens)
        self.source_identity_registry = [identity.as_dict()]

    def set_role(
        self,
        index: int,
        role: DomainRoleKind,
        *,
        entity: int | None = None,
        scope: int | None = None,
        confidence: ParseConfidence | None = None,
    ) -> None:
        if not (0 <= index < len(self.tokens)):
            raise IndexError(f"token index {index} out of range")
        self.role_ids[index] = int(role)
        if entity is not None:
            self.entity_ids[index] = int(entity)
        if scope is not None:
            self.scope_ids[index] = int(scope)
        if confidence is not None:
            self.confidence_ids[index] = int(confidence)

    def add_edge(
        self,
        src: int,
        dst: int,
        kind: DomainEdgeKind,
    ) -> None:
        if not (0 <= src < len(self.tokens)) or not (0 <= dst < len(self.tokens)):
            raise IndexError(f"edge endpoint out of range: {src}->{dst}")
        edge_kind = validate_domain_edge_kind(kind)
        self.edges.append((int(src), int(dst), int(edge_kind)))

    def token_indices_on_line(self, line_no: int) -> list[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.line == line_no]

    def validate(self) -> None:
        expected = len(self.tokens)
        vectors = {
            "domain_ids": self.domain_ids,
            "role_ids": self.role_ids,
            "entity_ids": self.entity_ids,
            "scope_ids": self.scope_ids,
            "source_doc_ids": self.source_doc_ids,
            "source_identity_ids": self.source_identity_ids,
            "confidence_ids": self.confidence_ids,
        }
        for name, values in vectors.items():
            if len(values) != expected:
                raise ValueError(
                    f"ParsedDomainDocument.{name}: length {len(values)} != "
                    f"token count {expected}"
                )
        for value in self.domain_ids:
            DomainKind(int(value))
        for value in self.role_ids:
            DomainRoleKind(int(value))
        for value in self.confidence_ids:
            ParseConfidence(int(value))
        for name in ("entity_ids", "scope_ids", "source_doc_ids", "source_identity_ids"):
            if any(int(value) < 0 for value in vectors[name]):
                raise ValueError(f"ParsedDomainDocument.{name} must be non-negative")
        if any(int(value) > MAX_ROW_LOCAL_DOC_ID for value in self.source_doc_ids):
            raise ValueError("ParsedDomainDocument.source_doc_ids must be uint32")
        if any(int(value) > MAX_SOURCE_ID for value in self.source_identity_ids):
            raise ValueError("ParsedDomainDocument.source_identity_ids must be uint64")
        referenced = [value for value in self.source_identity_ids if int(value) != 0]
        if referenced or self.source_identity_registry:
            validate_source_identity_registry(
                self.source_identity_registry,
                referenced_ids=referenced,
            )
        for src, dst, kind in self.edges:
            if not (0 <= int(src) < expected and 0 <= int(dst) < expected):
                raise ValueError(f"edge endpoint out of range: {src}->{dst}")
            validate_domain_edge_kind(kind)

    def to_enriched_document(self) -> dict[str, Any]:
        """Project token parser output onto character-aligned ingestion sidecars."""

        self.validate()
        text_len = len(self.text)
        default_confidence = (
            min(self.confidence_ids)
            if self.confidence_ids
            else int(ParseConfidence.ABSENT)
        )
        default_source = self.source_doc_ids[0] if self.source_doc_ids else 0
        default_source_identity = (
            self.source_identity_ids[0] if self.source_identity_ids else 0
        )
        result: dict[str, Any] = {
            "text": self.text,
            "domain_kind": int(self.domain),
            "domain_ids": [int(self.domain)] * text_len,
            "domain_role_ids": [int(DomainRoleKind.NONE)] * text_len,
            "domain_entity_ids": [0] * text_len,
            "domain_scope_ids": [0] * text_len,
            "domain_source_doc_ids": [int(default_source)] * text_len,
            "domain_source_identity_ids": [int(default_source_identity)] * text_len,
            "source_identity_registry": [
                dict(entry) for entry in self.source_identity_registry
            ],
            "domain_confidence_ids": [int(default_confidence)] * text_len,
            "domain_edges": [],
            "build_edges": [],
            "shell_edges": [],
            "diagnostic_edges": [],
            "cross_domain_edges": [],
            "embedded_domain_spans": [],
            "domain_parse_info": {
                key: value
                for key, value in self.metadata.items()
                if not key.startswith("_")
            },
        }

        def apply_token(
            token: LexedToken,
            *,
            domain_id: int,
            role_id: int,
            entity_id: int,
            scope_id: int,
            source_doc_id: int,
            source_identity_id: int,
            confidence_id: int,
            offset: int = 0,
        ) -> None:
            start = max(0, min(offset + int(token.start), text_len))
            end = max(start, min(offset + int(token.end), text_len))
            for char_idx in range(start, end):
                result["domain_ids"][char_idx] = int(domain_id)
                result["domain_role_ids"][char_idx] = int(role_id)
                result["domain_entity_ids"][char_idx] = int(entity_id)
                result["domain_scope_ids"][char_idx] = int(scope_id)
                result["domain_source_doc_ids"][char_idx] = int(source_doc_id)
                result["domain_source_identity_ids"][char_idx] = int(source_identity_id)
                result["domain_confidence_ids"][char_idx] = int(confidence_id)

        for idx, token in enumerate(self.tokens):
            apply_token(
                token,
                domain_id=self.domain_ids[idx],
                role_id=self.role_ids[idx],
                entity_id=self.entity_ids[idx],
                scope_id=self.scope_ids[idx],
                source_doc_id=self.source_doc_ids[idx],
                source_identity_id=self.source_identity_ids[idx],
                confidence_id=self.confidence_ids[idx],
            )

        edge_columns = {
            "domain": "domain_edges",
            "build": "build_edges",
            "shell": "shell_edges",
            "diagnostic": "diagnostic_edges",
            "cross_domain": "cross_domain_edges",
        }

        def append_token_edges(
            parsed: "ParsedDomainDocument",
            *,
            offset: int = 0,
        ) -> None:
            for src, dst, kind in parsed.edges:
                family = domain_edge_family(kind)
                edge = {
                    "from_char": offset + int(parsed.tokens[src].start),
                    "to_char": offset + int(parsed.tokens[dst].start),
                    "kind": int(kind),
                }
                result["domain_edges"].append(edge)
                if family != "domain":
                    result[edge_columns[family]].append(dict(edge))

        append_token_edges(self)
        for block in self.embedded_blocks:
            parsed = block.parsed
            parsed.validate()
            block_start = max(0, min(int(block.start), text_len))
            block_end = max(block_start, min(int(block.end), text_len))
            block_confidence = (
                min(parsed.confidence_ids)
                if parsed.confidence_ids
                else int(ParseConfidence.ABSENT)
            )
            block_source = parsed.source_doc_ids[0] if parsed.source_doc_ids else default_source
            block_source_identity = (
                parsed.source_identity_ids[0]
                if parsed.source_identity_ids
                else default_source_identity
            )
            registry_by_id = {
                int(entry["source_identity_id"]): dict(entry)
                for entry in result["source_identity_registry"]
            }
            for entry in parsed.source_identity_registry:
                identity_id = int(entry["source_identity_id"])
                previous = registry_by_id.get(identity_id)
                if previous is not None and previous != dict(entry):
                    raise ValueError(f"source identity uint64 collision for id {identity_id}")
                registry_by_id[identity_id] = dict(entry)
            result["source_identity_registry"] = list(registry_by_id.values())
            for char_idx in range(block_start, block_end):
                result["domain_ids"][char_idx] = int(parsed.domain)
                result["domain_source_doc_ids"][char_idx] = int(block_source)
                result["domain_source_identity_ids"][char_idx] = int(block_source_identity)
                result["domain_confidence_ids"][char_idx] = int(block_confidence)
            for idx, token in enumerate(parsed.tokens):
                apply_token(
                    token,
                    domain_id=parsed.domain_ids[idx],
                    role_id=parsed.role_ids[idx],
                    entity_id=parsed.entity_ids[idx],
                    scope_id=parsed.scope_ids[idx],
                    source_doc_id=parsed.source_doc_ids[idx],
                    source_identity_id=parsed.source_identity_ids[idx],
                    confidence_id=parsed.confidence_ids[idx],
                    offset=int(block.start),
                )
            append_token_edges(parsed, offset=int(block.start))
            result["embedded_domain_spans"].append(
                {
                    "start": int(block.start),
                    "end": int(block.end),
                    "domain_kind": int(block.domain),
                }
            )
            src, dst, kind = block.cross_domain_edge
            validate_domain_edge_kind(kind, family="cross_domain")
            edge = {"from_char": int(src), "to_char": int(dst), "kind": int(kind)}
            result["domain_edges"].append(edge)
            result["cross_domain_edges"].append(dict(edge))

        result["domain_parse_info"].update(
            {
                "tokens": len(self.tokens),
                "edges": sum(len(result[column]) for column in edge_columns.values()),
                "confidence": int(default_confidence),
            }
        )
        return result

    def to_packet(self, token_ids: list[int] | None = None) -> DomainPacket:
        import numpy as np

        from cppmega.data.domain_packet import DomainEdgeIndex, DomainPacket

        self.validate()
        if token_ids is None:
            token_ids = [debug_token_id(token.text) for token in self.tokens]
            metadata = dict(self.metadata)
            metadata["debug_token_ids"] = True
        else:
            metadata = dict(self.metadata)
        if len(token_ids) != len(self.tokens):
            raise ValueError(
                f"token_ids length {len(token_ids)} != parsed token count {len(self.tokens)}"
            )
        return DomainPacket(
            token_ids=np.asarray(token_ids, dtype=np.int32),
            domain=self.domain,
            domain_ids=np.asarray(self.domain_ids, dtype=np.int32),
            role_ids=np.asarray(self.role_ids, dtype=np.int32),
            entity_ids=np.asarray(self.entity_ids, dtype=np.int32),
            scope_ids=np.asarray(self.scope_ids, dtype=np.int32),
            source_doc_ids=np.asarray(self.source_doc_ids, dtype=np.int32),
            confidence_ids=np.asarray(self.confidence_ids, dtype=np.int32),
            graph_edges=DomainEdgeIndex.from_triples(self.edges),
            metadata=metadata,
        )


def lex_text(text: str) -> list[LexedToken]:
    line_starts = [0]
    for match in re.finditer(r"\n", text):
        line_starts.append(match.end())
    tokens: list[LexedToken] = []
    for match in _TOKEN_RE.finditer(text):
        start = match.start()
        line = 0
        # Texts here are small build/script files.  Linear scan keeps the helper
        # dependency-free and deterministic.
        for idx, line_start in enumerate(line_starts):
            if line_start > start:
                break
            line = idx
        tokens.append(
            LexedToken(
                text=match.group(0),
                start=start,
                end=match.end(),
                line=line,
                column=start - line_starts[line],
            )
        )
    return tokens


def debug_token_id(text: str) -> int:
    """Stable non-training token id for parser unit tests."""

    return 1000 + (sum(text.encode("utf-8")) % 30000)


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def is_source_path(value: str) -> bool:
    lower = strip_quotes(value).lower()
    return lower.endswith(
        (
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hh",
            ".hpp",
            ".hxx",
            ".cu",
            ".cuh",
            ".s",
            ".asm",
        )
    )


def is_option(value: str) -> bool:
    return value.startswith("-") or value.startswith("--")


__all__ = [
    "LexedToken",
    "ParsedDomainDocument",
    "debug_token_id",
    "is_option",
    "is_source_path",
    "lex_text",
    "strip_quotes",
]
