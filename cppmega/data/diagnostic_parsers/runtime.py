"""Test-runner and sanitizer diagnostic adapters."""

from __future__ import annotations

import re

from cppmega.data.build_parsers.base import ParsedDomainDocument
from cppmega.data.diagnostic_parsers.base import new_diagnostic_doc
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_TEST_NAME_RE = re.compile(
    r"(?P<file>[A-Za-z0-9_./\\-]+\.(?:py|cc|cpp|cxx|c|m|mm))::"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_\[\].:-]*)"
)
_LOCATION_RE = re.compile(
    r"(?P<file>[A-Za-z0-9_./\\-]+\.(?:py|cc|cpp|cxx|c|h|hpp|m|mm)):"
    r"(?P<line>\d+)(?::(?P<col>\d+))?"
)
_SANITIZER_TOOLS = (
    "AddressSanitizer",
    "LeakSanitizer",
    "MemorySanitizer",
    "ThreadSanitizer",
    "UndefinedBehaviorSanitizer",
)
_STATUS_COUNT_RE = re.compile(
    r"\b(?P<count>\d+)\s+(?:tests?\s+)?"
    r"(?P<status>passed|failed|failures?|errors?)\b",
    re.IGNORECASE,
)
_STATUS_FIRST_COUNT_RE = re.compile(
    r"\b(?P<status>passed|failed|failures?|errors?)\s*[:=]\s*"
    r"(?P<count>\d+)\b",
    re.IGNORECASE,
)


def _test_result_severity(text: str) -> str:
    counts = [
        (int(match.group("count")), match.group("status").lower())
        for pattern in (_STATUS_COUNT_RE, _STATUS_FIRST_COUNT_RE)
        for match in pattern.finditer(text)
    ]
    failure_count = sum(
        count
        for count, status in counts
        if status in {"failed", "failure", "failures", "error", "errors"}
    )
    if failure_count > 0:
        return "failure"

    residual = _STATUS_FIRST_COUNT_RE.sub("", _STATUS_COUNT_RE.sub("", text))
    if re.search(r"(?mi)^(?:FAILED|FAIL|ERROR)\b", residual) or re.search(
        r"\b(?:AssertionError|Traceback)\b",
        residual,
        re.IGNORECASE,
    ):
        return "failure"
    if counts:
        return "pass"
    if re.search(r"(?mi)^(?:PASSED|PASS|OK)\b|\bpassed\b", residual):
        return "pass"
    return "unknown"


def _token_at_span(
    doc: ParsedDomainDocument,
    start: int,
    end: int,
) -> int | None:
    return next(
        (
            idx
            for idx, token in enumerate(doc.tokens)
            if token.start < end and token.end > start
        ),
        None,
    )


def _mark_location(
    doc: ParsedDomainDocument,
    match: re.Match[str],
) -> int | None:
    file_idx = _token_at_span(doc, *match.span("file"))
    line_idx = _token_at_span(doc, *match.span("line"))
    col_span = match.span("col") if match.group("col") is not None else None
    if file_idx is not None:
        doc.set_role(file_idx, DomainRoleKind.FILE)
    if line_idx is not None:
        doc.set_role(line_idx, DomainRoleKind.LINE)
    if col_span is not None:
        col_idx = _token_at_span(doc, *col_span)
        if col_idx is not None:
            doc.set_role(col_idx, DomainRoleKind.COLUMN)
    return file_idx


def parse_test_output(text: str, *, tool: str = "test") -> ParsedDomainDocument:
    severity = _test_result_severity(text)
    doc = new_diagnostic_doc(
        text,
        domain=DomainKind.TEST_OUTPUT,
        tool=tool,
        severity=severity,
        stage="test",
        confidence=ParseConfidence.HEURISTIC if text.strip() else ParseConfidence.RAW,
    )
    doc.metadata["parser_adapter"] = "test-output"
    doc.metadata["result_status"] = severity

    primary_idx: int | None = None
    for idx, token in enumerate(doc.tokens):
        if token.text.upper() in {"FAILED", "FAIL", "ERROR"}:
            doc.set_role(idx, DomainRoleKind.SEVERITY)
            primary_idx = primary_idx if primary_idx is not None else idx

    result_counts: list[dict[str, int | str]] = []
    seen_result_edges: set[tuple[int, int]] = set()
    for pattern in (_STATUS_COUNT_RE, _STATUS_FIRST_COUNT_RE):
        for match in pattern.finditer(text):
            status_idx = _token_at_span(doc, *match.span("status"))
            count_idx = _token_at_span(doc, *match.span("count"))
            if status_idx is None or count_idx is None:
                continue
            doc.set_role(status_idx, DomainRoleKind.SEVERITY)
            doc.set_role(count_idx, DomainRoleKind.MESSAGE)
            edge = (status_idx, count_idx)
            if edge not in seen_result_edges:
                doc.add_edge(status_idx, count_idx, DomainEdgeKind.TOOL_ACTION_RESULT)
                seen_result_edges.add(edge)
            primary_idx = primary_idx if primary_idx is not None else status_idx
            result_counts.append(
                {
                    "status": match.group("status").lower(),
                    "count": int(match.group("count")),
                }
            )
    doc.metadata["result_counts"] = result_counts

    test_name_idx: int | None = None
    for match in _TEST_NAME_RE.finditer(text):
        file_idx = _token_at_span(doc, *match.span("file"))
        name_idx = _token_at_span(doc, *match.span("name"))
        if file_idx is not None:
            doc.set_role(file_idx, DomainRoleKind.FILE)
        if name_idx is not None:
            doc.set_role(name_idx, DomainRoleKind.TEST_NAME)
            test_name_idx = test_name_idx if test_name_idx is not None else name_idx

    locations = list(_LOCATION_RE.finditer(text))
    for match in locations:
        file_idx = _mark_location(doc, match)
        source_idx = test_name_idx if test_name_idx is not None else primary_idx
        if source_idx is not None and file_idx is not None:
            doc.add_edge(source_idx, file_idx, DomainEdgeKind.TEST_FAILURE_LOCATION)
    return doc


def parse_sanitizer_output(text: str) -> ParsedDomainDocument:
    tool = next((name for name in _SANITIZER_TOOLS if name in text), "sanitizer")
    recognized = tool != "sanitizer"
    doc = new_diagnostic_doc(
        text,
        domain=DomainKind.SANITIZER_OUTPUT,
        tool=tool,
        severity="error" if recognized else "unknown",
        stage="runtime",
        confidence=ParseConfidence.HEURISTIC if recognized else ParseConfidence.RAW,
    )
    doc.metadata["parser_adapter"] = "sanitizer-output"
    if not recognized:
        doc.mark_raw("unrecognized_sanitizer_output")

    primary_idx: int | None = None
    for idx, token in enumerate(doc.tokens):
        if token.text.upper() == "ERROR":
            doc.set_role(idx, DomainRoleKind.SEVERITY)
            primary_idx = primary_idx if primary_idx is not None else idx
        elif token.text in _SANITIZER_TOOLS:
            doc.set_role(idx, DomainRoleKind.COMMAND)
            primary_idx = primary_idx if primary_idx is not None else idx

    for match in _LOCATION_RE.finditer(text):
        file_idx = _mark_location(doc, match)
        if primary_idx is not None and file_idx is not None:
            doc.add_edge(primary_idx, file_idx, DomainEdgeKind.DIAG_PRIMARY_LOCATION)
    return doc


__all__ = ["parse_sanitizer_output", "parse_test_output"]
