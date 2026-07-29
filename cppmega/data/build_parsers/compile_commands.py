"""Deterministic parser for ``compile_commands.json`` build evidence."""

from __future__ import annotations

import json

from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    is_option,
    strip_quotes,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)

_VALUE_ROLES = {
    "directory": DomainRoleKind.PATH,
    "file": DomainRoleKind.SOURCE,
    "output": DomainRoleKind.OUTPUT,
    "command": DomainRoleKind.COMMAND,
}


def _load_compile_command_entries(text: str) -> tuple[list[dict], bool]:
    """Validate a complete array or one structurally chunked array fragment."""

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        stripped = text.strip()
        if stripped.startswith("["):
            stripped = stripped[1:].lstrip()
        if stripped.endswith("]"):
            stripped = stripped[:-1].rstrip()
        if stripped.startswith(","):
            stripped = stripped[1:].lstrip()
        if stripped.endswith(","):
            stripped = stripped[:-1].rstrip()
        try:
            payload = json.loads(f"[{stripped}]") if stripped else []
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed_compile_commands_json") from exc
        fragment = True
    else:
        fragment = False

    if isinstance(payload, dict):
        payload = [payload]
        fragment = True
    if not isinstance(payload, list) or any(
        not isinstance(entry, dict) for entry in payload
    ):
        raise ValueError("compile_commands_root_is_not_object_array")
    return payload, fragment


def _next_value_token(
    doc: ParsedDomainDocument,
    key_index: int,
) -> int | None:
    for index in range(key_index + 1, len(doc.tokens)):
        value = doc.tokens[index].text
        if value == ":":
            continue
        return index
    return None


def parse_compile_commands(text: str) -> ParsedDomainDocument:
    """Parse compilation-database entries into roles and action edges.

    JSON validation is exact; role and edge attachment is deliberately
    heuristic because the lexical token stream preserves source offsets rather
    than rebuilding a second JSON position map.
    """

    doc = ParsedDomainDocument.new(
        domain=DomainKind.COMPILE_COMMANDS,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={
            "parser_adapter": "compile-commands-json",
            "build_kind": "compile_commands",
        },
    )
    try:
        payload, fragment = _load_compile_command_entries(text)
    except ValueError as exc:
        return doc.mark_raw(str(exc))

    doc.metadata["json_entries"] = len(payload)
    if fragment:
        doc.metadata["json_fragment"] = True
    current_entry: dict[str, list[int]] = {}
    object_depth = 0

    def finish_entry() -> None:
        commands = current_entry.get("command", []) + current_entry.get(
            "arguments",
            [],
        )
        inputs = current_entry.get("file", [])
        outputs = current_entry.get("output", [])
        for command in commands[:1]:
            for source in inputs:
                doc.add_edge(command, source, DomainEdgeKind.BUILD_ACTION_INPUT)
            for output in outputs:
                doc.add_edge(command, output, DomainEdgeKind.BUILD_ACTION_OUTPUT)
        current_entry.clear()

    for index, token in enumerate(doc.tokens):
        value = strip_quotes(token.text)
        if token.text == "{":
            object_depth += 1
            continue
        if token.text == "}":
            if object_depth == 1:
                finish_entry()
            object_depth = max(0, object_depth - 1)
            continue
        if object_depth != 1:
            continue
        if value not in {*_VALUE_ROLES, "arguments"}:
            continue

        doc.set_role(index, DomainRoleKind.ATTRIBUTE)
        value_index = _next_value_token(doc, index)
        if value_index is None:
            continue
        value_token = doc.tokens[value_index]
        if value == "arguments" and value_token.text == "[":
            argument_indices: list[int] = []
            depth = 0
            for argument_index in range(value_index, len(doc.tokens)):
                argument_token = doc.tokens[argument_index]
                if argument_token.text == "[":
                    depth += 1
                    continue
                if argument_token.text == "]":
                    depth -= 1
                    if depth == 0:
                        break
                    continue
                if depth != 1 or argument_token.text == ",":
                    continue
                argument_indices.append(argument_index)
            if argument_indices:
                doc.set_role(argument_indices[0], DomainRoleKind.COMMAND)
                for argument_index in argument_indices[1:]:
                    argument = strip_quotes(doc.tokens[argument_index].text)
                    doc.set_role(
                        argument_index,
                        (
                            DomainRoleKind.OPTION
                            if is_option(argument)
                            else (
                                DomainRoleKind.PATH
                                if "/" in argument or "\\" in argument
                                else DomainRoleKind.STRING
                            )
                        ),
                    )
                current_entry.setdefault("arguments", []).append(argument_indices[0])
            continue

        role = _VALUE_ROLES[value]
        doc.set_role(value_index, role)
        current_entry.setdefault(value, []).append(value_index)

    if current_entry:
        finish_entry()
    return doc


__all__ = ["parse_compile_commands"]
