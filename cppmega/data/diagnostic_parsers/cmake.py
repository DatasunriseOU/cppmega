"""CMake/build-system diagnostic parser."""

from __future__ import annotations

from cppmega.data.diagnostic_parsers.base import new_diagnostic_doc
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


def parse_build_error(text: str, *, tool: str = "build") -> object:
    lower_text = text.lower()
    severity = (
        "error"
        if any(token in lower_text for token in ("error", "failed", "failure", "build stopped"))
        else "warning"
        if "warning" in lower_text
        else "unknown"
    )
    domain = DomainKind.BUILD_ERROR if severity == "error" else DomainKind.BUILD_DIAGNOSTIC
    doc = new_diagnostic_doc(
        text,
        domain=domain,
        tool=tool,
        severity=severity,
        stage="build",
        confidence=ParseConfidence.HEURISTIC if severity != "unknown" else ParseConfidence.RAW,
    )
    primary: int | None = None
    for idx, token in enumerate(doc.tokens):
        value = token.text
        lower = value.lower()
        if lower in {"cmake", "make", "ninja", "bazel"}:
            doc.set_role(idx, DomainRoleKind.COMMAND)
            primary = primary if primary is not None else idx
        elif lower in {"error", "failed", "failure"}:
            doc.set_role(idx, DomainRoleKind.SEVERITY)
            primary = primary if primary is not None else idx
        elif value.endswith(":") and not value.startswith("-"):
            doc.set_role(idx, DomainRoleKind.TARGET)
            if primary is not None:
                doc.add_edge(primary, idx, DomainEdgeKind.DIAG_BUILD_TARGET)
        elif "/" in value or value.endswith((".cmake", "CMakeLists.txt", ".ninja", ".mk")):
            doc.set_role(idx, DomainRoleKind.PATH)
            if primary is not None:
                doc.add_edge(primary, idx, DomainEdgeKind.DIAG_PRIMARY_LOCATION)
        elif value.isdigit():
            doc.set_role(idx, DomainRoleKind.EXIT_CODE)
    return doc


__all__ = ["parse_build_error"]
