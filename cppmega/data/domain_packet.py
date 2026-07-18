"""Typed packet for domain-routed non-C++ and cross-domain documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
    validate_domain_edge_kind,
)


_TOKEN_SIDECAR_FIELDS = (
    "domain_ids",
    "role_ids",
    "entity_ids",
    "scope_ids",
    "source_doc_ids",
    "confidence_ids",
)


def _as_int_vector(value: Any, *, where: str) -> np.ndarray:
    import numpy as np

    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{where}: expected 1-D vector, got shape {tuple(arr.shape)}")
    return arr.astype(np.int32, copy=False)


@dataclass(frozen=True)
class DomainEdgeIndex:
    """Directed token-local edges with an explicit edge-kind column.

    ``src``, ``dst``, and ``kind`` are all ``(E,)`` int32 vectors.  They index the
    token positions in the same packet after any domain delimiter wrapping.
    """

    src: np.ndarray
    dst: np.ndarray
    kind: np.ndarray

    def __post_init__(self) -> None:
        import numpy as np

        object.__setattr__(self, "src", _as_int_vector(self.src, where="edge.src"))
        object.__setattr__(self, "dst", _as_int_vector(self.dst, where="edge.dst"))
        object.__setattr__(self, "kind", _as_int_vector(self.kind, where="edge.kind"))
        if self.src.shape != self.dst.shape or self.src.shape != self.kind.shape:
            raise ValueError(
                "DomainEdgeIndex src/dst/kind length mismatch: "
                f"src={tuple(self.src.shape)} dst={tuple(self.dst.shape)} "
                f"kind={tuple(self.kind.shape)}"
            )
        if int(self.src.shape[0]) and (
            bool(np.any(self.src < 0)) or bool(np.any(self.dst < 0))
        ):
            raise ValueError("DomainEdgeIndex endpoints must be non-negative")
        for value in np.asarray(self.kind).tolist():
            validate_domain_edge_kind(int(value))

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @classmethod
    def empty(cls) -> "DomainEdgeIndex":
        import numpy as np

        z = np.zeros((0,), dtype=np.int32)
        return cls(src=z, dst=z, kind=z)

    @classmethod
    def from_triples(
        cls,
        triples: Sequence[Sequence[int] | tuple[int, int, DomainEdgeKind | int]],
    ) -> "DomainEdgeIndex":
        import numpy as np

        if not triples:
            return cls.empty()
        arr = np.asarray(
            [
                (int(triple[0]), int(triple[1]), int(triple[2]))
                for triple in triples
            ],
            dtype=np.int32,
        )
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"DomainEdgeIndex.from_triples expected (E,3), got {arr.shape}"
            )
        return cls(src=arr[:, 0], dst=arr[:, 1], kind=arr[:, 2])

    def shifted(self, offset: int) -> "DomainEdgeIndex":
        if self.num_edges == 0:
            return self
        return DomainEdgeIndex(src=self.src + offset, dst=self.dst + offset, kind=self.kind)

    def to_triples(self) -> list[tuple[int, int, int]]:
        import numpy as np

        src = np.asarray(self.src)
        dst = np.asarray(self.dst)
        kind = np.asarray(self.kind)
        return [(int(s), int(d), int(k)) for s, d, k in zip(src, dst, kind)]


@dataclass(frozen=True)
class DomainPacket:
    """A single tokenized domain document with token-aligned sidecars."""

    token_ids: np.ndarray
    domain: DomainKind
    domain_ids: np.ndarray
    role_ids: np.ndarray
    entity_ids: np.ndarray
    scope_ids: np.ndarray
    source_doc_ids: np.ndarray
    confidence_ids: np.ndarray
    graph_edges: DomainEdgeIndex = field(default_factory=DomainEdgeIndex.empty)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import numpy as np

        object.__setattr__(self, "domain", DomainKind(self.domain))
        object.__setattr__(self, "token_ids", _as_int_vector(self.token_ids, where="token_ids"))
        expected = int(self.token_ids.shape[0])
        for name in _TOKEN_SIDECAR_FIELDS:
            arr = _as_int_vector(getattr(self, name), where=name)
            if int(arr.shape[0]) != expected:
                raise ValueError(
                    f"DomainPacket.{name}: length {int(arr.shape[0])} != "
                    f"token_ids length {expected}"
                )
            object.__setattr__(self, name, arr)
        if not isinstance(self.graph_edges, DomainEdgeIndex):
            raise TypeError(
                f"DomainPacket.graph_edges must be DomainEdgeIndex, got "
                f"{type(self.graph_edges).__name__}"
            )
        if self.graph_edges.num_edges:
            max_endpoint = int(
                max(np.max(self.graph_edges.src), np.max(self.graph_edges.dst))
            )
            if max_endpoint >= expected:
                raise ValueError(
                    f"DomainPacket.graph_edges endpoint {max_endpoint} outside "
                    f"token range 0..{expected - 1}"
                )
        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                f"DomainPacket.metadata must be a Mapping, got {type(self.metadata).__name__}"
            )

    @property
    def token_axis_len(self) -> int:
        return int(self.token_ids.shape[0])

    @classmethod
    def filled(
        cls,
        token_ids: Sequence[int] | np.ndarray,
        *,
        domain: DomainKind,
        role: DomainRoleKind = DomainRoleKind.NONE,
        confidence: ParseConfidence = ParseConfidence.EXACT,
        graph_edges: DomainEdgeIndex | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DomainPacket":
        import numpy as np

        ids = _as_int_vector(token_ids, where="token_ids")
        n = int(ids.shape[0])
        zeros = np.zeros((n,), dtype=np.int32)
        return cls(
            token_ids=ids,
            domain=domain,
            domain_ids=np.full((n,), int(domain), dtype=np.int32),
            role_ids=np.full((n,), int(role), dtype=np.int32),
            entity_ids=zeros,
            scope_ids=zeros,
            source_doc_ids=zeros,
            confidence_ids=np.full((n,), int(confidence), dtype=np.int32),
            graph_edges=graph_edges or DomainEdgeIndex.empty(),
            metadata=metadata or {},
        )


def wrap_with_domain_tokens(packet: DomainPacket) -> DomainPacket:
    """Insert the domain opener/closer token ids and shift graph edges by one."""

    import numpy as np

    start_id, end_id = delimiter_token_ids(packet.domain)
    token_ids = np.concatenate(
        [
            np.asarray([start_id], dtype=np.int32),
            packet.token_ids,
            np.asarray([end_id], dtype=np.int32),
        ],
        axis=0,
    )
    delimiter_role = int(DomainRoleKind.DELIMITER)
    exact = int(ParseConfidence.EXACT)
    domain_id = int(packet.domain)

    def wrap_sidecar(values: np.ndarray, *, delimiter_value: int) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray([delimiter_value], dtype=np.int32),
                values.astype(np.int32),
                np.asarray([delimiter_value], dtype=np.int32),
            ],
            axis=0,
        )

    metadata = dict(packet.metadata)
    metadata["domain_wrapped"] = True
    metadata["domain_start_token_id"] = int(start_id)
    metadata["domain_end_token_id"] = int(end_id)
    source_values = np.asarray(packet.source_doc_ids).tolist()
    start_source = int(source_values[0]) if source_values else 0
    end_source = int(source_values[-1]) if source_values else 0

    return DomainPacket(
        token_ids=token_ids,
        domain=packet.domain,
        domain_ids=wrap_sidecar(packet.domain_ids, delimiter_value=domain_id),
        role_ids=wrap_sidecar(packet.role_ids, delimiter_value=delimiter_role),
        entity_ids=wrap_sidecar(packet.entity_ids, delimiter_value=0),
        scope_ids=wrap_sidecar(packet.scope_ids, delimiter_value=0),
        source_doc_ids=np.concatenate(
            [
                np.asarray([start_source], dtype=np.int32),
                packet.source_doc_ids.astype(np.int32),
                np.asarray([end_source], dtype=np.int32),
            ],
            axis=0,
        ),
        confidence_ids=wrap_sidecar(packet.confidence_ids, delimiter_value=exact),
        graph_edges=packet.graph_edges.shifted(1),
        metadata=metadata,
    )


__all__ = [
    "DomainEdgeIndex",
    "DomainPacket",
    "wrap_with_domain_tokens",
]
