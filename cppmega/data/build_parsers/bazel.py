"""Bazel/Starlark build-domain parser."""

from __future__ import annotations

from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    is_option,
    is_source_path,
    strip_quotes,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_RULES = {"cc_binary", "cc_library", "cc_test", "cc_import", "proto_library"}


def parse_bazel(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.BAZEL,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "bazel", "parser_adapter": "bazel-starlark"},
    )
    next_entity = 1
    current_target: int | None = None
    current_attr: str | None = None
    pending_target_edges: list[tuple[int, DomainEdgeKind]] = []

    for idx, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        if value in _RULES:
            doc.set_role(idx, DomainRoleKind.RULE, entity=next_entity)
            next_entity += 1
            current_target = None
            current_attr = None
            pending_target_edges = []
        elif value in {"name", "srcs", "deps", "copts", "linkopts", "hdrs"}:
            doc.set_role(idx, DomainRoleKind.ATTRIBUTE, entity=next_entity)
            next_entity += 1
            current_attr = value
        elif current_attr == "name" and value not in {"=", ",", "[", "]", "(", ")"}:
            doc.set_role(idx, DomainRoleKind.TARGET, entity=next_entity)
            current_target = idx
            next_entity += 1
            for dst, kind in pending_target_edges:
                doc.add_edge(current_target, dst, kind)
            pending_target_edges = []
            current_attr = None
        elif current_attr in {"srcs", "hdrs"} and is_source_path(value):
            doc.set_role(idx, DomainRoleKind.SOURCE, entity=next_entity)
            next_entity += 1
            if current_target is not None:
                doc.add_edge(current_target, idx, DomainEdgeKind.BUILD_TARGET_SOURCE)
            else:
                pending_target_edges.append((idx, DomainEdgeKind.BUILD_TARGET_SOURCE))
        elif current_attr == "deps" and (value.startswith("//") or value.startswith(":") or value.startswith("@")):
            doc.set_role(idx, DomainRoleKind.LABEL, entity=next_entity)
            next_entity += 1
            if current_target is not None:
                doc.add_edge(current_target, idx, DomainEdgeKind.BUILD_TARGET_DEP)
            else:
                pending_target_edges.append((idx, DomainEdgeKind.BUILD_TARGET_DEP))
        elif current_attr in {"copts", "linkopts"} and is_option(value):
            doc.set_role(idx, DomainRoleKind.OPTION)
        elif value == "]":
            current_attr = None

    return doc


__all__ = ["parse_bazel"]
