"""Clang/GCC text diagnostic parser."""

from __future__ import annotations

import re

from cppmega.data.diagnostic_parsers.base import mark_message_tail, new_diagnostic_doc
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_DIAG_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<severity>fatal error|error|warning|note):\s+(?P<message>.*)$"
)


def _diagnostic_domain(severities: list[str]) -> DomainKind:
    return (
        DomainKind.COMPILER_ERROR
        if any(severity in {"fatal error", "error"} for severity in severities)
        else DomainKind.COMPILER_DIAGNOSTIC
    )


def parse_clang_diagnostic(text: str, *, tool: str = "clang") -> object:
    matches = [match for line in text.splitlines() if (match := _DIAG_RE.match(line))]
    severities = [match.group("severity") for match in matches]
    primary_severity = severities[0] if severities else "unknown"
    doc = new_diagnostic_doc(
        text,
        domain=_diagnostic_domain(severities),
        tool=tool,
        severity=primary_severity,
        stage="compile",
        confidence=ParseConfidence.HEURISTIC if matches else ParseConfidence.RAW,
    )
    if not matches:
        doc.metadata["unsupported_syntax"] = "unrecognized_compiler_diagnostic"
        return doc
    first_diag_idx: int | None = None
    for line_no, raw_line in enumerate(text.splitlines()):
        match = _DIAG_RE.match(raw_line)
        if not match:
            continue
        line_indices = doc.token_indices_on_line(line_no)
        if not line_indices:
            continue
        file_idx = line_indices[0]
        severity_token = match.group("severity").split()[-1]
        severity_idx = next(
            (idx for idx in line_indices if doc.tokens[idx].text == severity_token),
            line_indices[-1],
        )
        doc.set_role(file_idx, DomainRoleKind.FILE)
        seen_number = 0
        for idx in line_indices[1:severity_idx]:
            if doc.tokens[idx].text.isdigit():
                seen_number += 1
                role = (
                    DomainRoleKind.LINE
                    if seen_number == 1
                    else DomainRoleKind.COLUMN
                )
                doc.set_role(idx, role)
        doc.set_role(severity_idx, DomainRoleKind.SEVERITY)
        mark_message_tail(doc, line_no=line_no, after_index=severity_idx)
        doc.add_edge(severity_idx, file_idx, DomainEdgeKind.DIAG_PRIMARY_LOCATION)
        if (
            severity_token == "note"
            and first_diag_idx is not None
            and first_diag_idx != severity_idx
        ):
            doc.add_edge(first_diag_idx, severity_idx, DomainEdgeKind.DIAG_NOTE)
        if first_diag_idx is None:
            first_diag_idx = severity_idx
    return doc


def parse_gcc_diagnostic(text: str) -> object:
    return parse_clang_diagnostic(text, tool="gcc")


__all__ = ["parse_clang_diagnostic", "parse_gcc_diagnostic"]
