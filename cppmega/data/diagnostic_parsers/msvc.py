"""MSVC diagnostic parser."""

from __future__ import annotations

import re

from cppmega.data.diagnostic_parsers.base import mark_message_tail, new_diagnostic_doc
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_MSVC_RE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+)(?:,(?P<col>\d+))?\):\s+"
    r"(?P<severity>fatal error|error|warning)\s+(?P<code>[A-Z]+\d+):\s+(?P<message>.*)$"
)


def parse_msvc_diagnostic(text: str) -> object:
    matches = [match for line in text.splitlines() if (match := _MSVC_RE.match(line))]
    severities = [match.group("severity") for match in matches]
    primary_severity = severities[0] if severities else "unknown"
    domain = (
        DomainKind.COMPILER_ERROR
        if any(severity in {"fatal error", "error"} for severity in severities)
        else DomainKind.COMPILER_DIAGNOSTIC
    )
    doc = new_diagnostic_doc(
        text,
        domain=domain,
        tool="msvc",
        severity=primary_severity,
        stage="compile",
        platform="windows",
        confidence=ParseConfidence.HEURISTIC if matches else ParseConfidence.RAW,
    )
    for line_no, raw_line in enumerate(text.splitlines()):
        match = _MSVC_RE.match(raw_line)
        if not match:
            continue
        line_indices = doc.token_indices_on_line(line_no)
        if not line_indices:
            continue
        file_idx = line_indices[0]
        code_idx = next(
            (idx for idx in line_indices if doc.tokens[idx].text == match.group("code")),
            line_indices[-1],
        )
        doc.set_role(file_idx, DomainRoleKind.FILE)
        seen_number = 0
        for idx in line_indices[1:code_idx]:
            if doc.tokens[idx].text.isdigit():
                seen_number += 1
                doc.set_role(idx, DomainRoleKind.LINE if seen_number == 1 else DomainRoleKind.COLUMN)
            elif doc.tokens[idx].text in {"error", "warning"}:
                doc.set_role(idx, DomainRoleKind.SEVERITY)
        doc.set_role(code_idx, DomainRoleKind.MESSAGE)
        mark_message_tail(doc, line_no=line_no, after_index=code_idx)
        doc.add_edge(code_idx, file_idx, DomainEdgeKind.DIAG_PRIMARY_LOCATION)
    return doc


__all__ = ["parse_msvc_diagnostic"]
