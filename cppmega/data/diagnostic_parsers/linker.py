"""Linker diagnostic parser."""

from __future__ import annotations

import re

from cppmega.data.diagnostic_parsers.base import new_diagnostic_doc
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_UNDEF_RE = re.compile(r"(undefined reference to|Undefined symbols for architecture).*?[`'\"](?P<sym>[^`'\"]+)[`'\"]")


def _linker_severity(text: str) -> str:
    lower = text.lower()
    if "fatal" in lower or "error" in lower or "undefined reference" in lower or "unresolved external symbol" in lower:
        return "error"
    if "warning" in lower:
        return "warning"
    return "unknown"


def parse_linker_error(text: str, *, tool: str = "linker") -> object:
    severity = _linker_severity(text)
    domain = (
        DomainKind.LINKER_ERROR
        if severity == "error"
        else DomainKind.LINKER_DIAGNOSTIC
    )
    doc = new_diagnostic_doc(
        text,
        domain=domain,
        tool=tool,
        severity=severity,
        stage="link",
        confidence=ParseConfidence.HEURISTIC if severity != "unknown" else ParseConfidence.RAW,
    )
    first_error = 0 if doc.tokens else None
    for idx, token in enumerate(doc.tokens):
        value = token.text.strip("`'\"")
        if value in {"ld", "lld", "link", "link.exe"}:
            doc.set_role(idx, DomainRoleKind.COMMAND)
            if first_error is None:
                first_error = idx
        elif value.lower() in {"error", "warning", "undefined"}:
            doc.set_role(idx, DomainRoleKind.SEVERITY)
            if first_error is None:
                first_error = idx
        elif "/" in value or value.endswith((".o", ".obj", ".a", ".lib", ".so", ".dylib")):
            doc.set_role(idx, DomainRoleKind.PATH)

    for match in _UNDEF_RE.finditer(text):
        start, end = match.span("sym")
        first_symbol_idx: int | None = None
        for idx, token in enumerate(doc.tokens):
            if token.start < end and token.end > start:
                doc.set_role(idx, DomainRoleKind.SYMBOL)
                if first_symbol_idx is None:
                    first_symbol_idx = idx
        if first_error is not None and first_symbol_idx is not None:
            doc.add_edge(first_error, first_symbol_idx, DomainEdgeKind.LINK_UNDEFINED_SYMBOL)
    return doc


__all__ = ["parse_linker_error"]
