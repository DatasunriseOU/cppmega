"""CMake domain parser."""

from __future__ import annotations

import re

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


_TARGET_COMMANDS = {
    "add_executable",
    "add_library",
    "target_sources",
    "target_link_libraries",
    "target_include_directories",
    "target_compile_options",
    "target_compile_definitions",
}
_COMMANDS = _TARGET_COMMANDS | {"project", "set", "option", "find_package", "include"}
_SCOPE_KEYWORDS = {
    "PRIVATE",
    "PUBLIC",
    "INTERFACE",
    "STATIC",
    "SHARED",
    "MODULE",
    "OBJECT",
    "EXCLUDE_FROM_ALL",
}


def _parentheses_are_balanced(text: str) -> bool:
    """Check CMake command parens outside comments and quoted/bracket text."""

    depth = 0
    idx = 0
    quote = False
    escaped = False
    bracket_end: str | None = None
    while idx < len(text):
        if bracket_end is not None:
            end = text.find(bracket_end, idx)
            if end < 0:
                return False
            idx = end + len(bracket_end)
            bracket_end = None
            continue

        char = text[idx]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            idx += 1
            continue

        if char == '"':
            quote = True
            idx += 1
            continue
        if char == "#":
            match = re.match(r"#\[(=*)\[", text[idx:])
            if match is not None:
                bracket_end = "]" + match.group(1) + "]"
                idx += len(match.group(0))
                continue
            newline = text.find("\n", idx + 1)
            idx = len(text) if newline < 0 else newline + 1
            continue
        if char == "[":
            match = re.match(r"\[(=*)\[", text[idx:])
            if match is not None:
                bracket_end = "]" + match.group(1) + "]"
                idx += len(match.group(0))
                continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
        idx += 1
    return depth == 0 and not quote and bracket_end is None


def _comment_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    idx = 0
    quote = False
    escaped = False
    while idx < len(text):
        char = text[idx]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            idx += 1
            continue
        if char == '"':
            quote = True
            idx += 1
            continue
        if char == "[":
            bracket_argument = re.match(r"\[(=*)\[", text[idx:])
            if bracket_argument is not None:
                marker = "]" + bracket_argument.group(1) + "]"
                end = text.find(marker, idx + len(bracket_argument.group(0)))
                end = len(text) if end < 0 else end + len(marker)
                spans.append((idx, end))
                idx = end
                continue
        if char != "#":
            idx += 1
            continue
        bracket = re.match(r"#\[(=*)\[", text[idx:])
        if bracket is not None:
            marker = "]" + bracket.group(1) + "]"
            end = text.find(marker, idx + len(bracket.group(0)))
            end = len(text) if end < 0 else end + len(marker)
        else:
            newline = text.find("\n", idx + 1)
            end = len(text) if newline < 0 else newline
        spans.append((idx, end))
        idx = end
    return spans


def _is_commented(offset: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in spans)


def _balanced_command_arguments(
    doc: ParsedDomainDocument,
    command_idx: int,
    comment_spans: list[tuple[int, int]],
) -> list[int] | None:
    open_idx = command_idx + 1
    while open_idx < len(doc.tokens) and _is_commented(
        doc.tokens[open_idx].start, comment_spans
    ):
        open_idx += 1
    if open_idx >= len(doc.tokens) or doc.tokens[open_idx].text != "(":
        return []
    depth = 0
    arguments: list[int] = []
    for idx in range(open_idx, len(doc.tokens)):
        if _is_commented(doc.tokens[idx].start, comment_spans):
            continue
        value = doc.tokens[idx].text
        if value == "(":
            depth += 1
            continue
        if value == ")":
            depth -= 1
            if depth == 0:
                return arguments
            if depth < 0:
                return None
            continue
        if depth > 0 and value not in {",", ";"}:
            arguments.append(idx)
    return None


def parse_cmake(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.CMAKE,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "cmake", "parser_adapter": "cmake"},
    )
    next_entity = 1
    targets_seen = 0
    comment_spans = _comment_spans(text)

    if not _parentheses_are_balanced(text):
        return doc.mark_raw("malformed_cmake_command_parentheses")

    for idx, token in enumerate(doc.tokens):
        if _is_commented(token.start, comment_spans) or token.text.startswith(("\"", "'")):
            continue
        word = strip_quotes(token.text)
        lower = word.lower()
        if lower in _COMMANDS:
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            command_entity = next_entity
            next_entity += 1
            after = _balanced_command_arguments(doc, idx, comment_spans)
            open_idx = idx + 1
            while open_idx < len(doc.tokens) and _is_commented(
                doc.tokens[open_idx].start, comment_spans
            ):
                open_idx += 1
            if after is None or (
                open_idx >= len(doc.tokens) or doc.tokens[open_idx].text != "("
            ):
                return doc.mark_raw("malformed_cmake_command_parentheses")
            if lower in _TARGET_COMMANDS and after:
                target_idx = after[0]
                doc.set_role(target_idx, DomainRoleKind.TARGET, entity=next_entity, scope=command_entity)
                target_entity = next_entity
                next_entity += 1
                targets_seen += 1
                doc.add_edge(idx, target_idx, DomainEdgeKind.BUILD_COMMAND_TARGET)
                for arg_idx in after[1:]:
                    value = strip_quotes(doc.tokens[arg_idx].text)
                    if value.upper() in _SCOPE_KEYWORDS:
                        doc.set_role(arg_idx, DomainRoleKind.KEYWORD, scope=target_entity)
                    elif is_source_path(value):
                        doc.set_role(arg_idx, DomainRoleKind.SOURCE, entity=next_entity, scope=target_entity)
                        next_entity += 1
                        doc.add_edge(target_idx, arg_idx, DomainEdgeKind.BUILD_TARGET_SOURCE)
                    elif is_option(value):
                        doc.set_role(arg_idx, DomainRoleKind.OPTION, scope=target_entity)
                    elif lower == "target_link_libraries":
                        doc.set_role(arg_idx, DomainRoleKind.LIBRARY, entity=next_entity, scope=target_entity)
                        next_entity += 1
                        doc.add_edge(target_idx, arg_idx, DomainEdgeKind.BUILD_TARGET_DEP)
                    elif lower == "target_include_directories":
                        doc.set_role(arg_idx, DomainRoleKind.PATH, scope=target_entity)
            elif lower == "set" and after:
                doc.set_role(after[0], DomainRoleKind.VARIABLE, entity=next_entity, scope=command_entity)
                next_entity += 1
        elif word.startswith("${") or word.startswith("$("):
            doc.set_role(idx, DomainRoleKind.VARIABLE)

    doc.metadata["targets_seen"] = targets_seen
    return doc


__all__ = ["parse_cmake"]
