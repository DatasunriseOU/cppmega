"""Dockerfile domain parser using tree-sitter AST."""

from __future__ import annotations

import json
import re
from typing import Any

from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    strip_quotes,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)

_COPY_ADD_INSTRUCTIONS = {"COPY", "ADD"}
_RUN_INSTRUCTION = "RUN"
_FROM_INSTRUCTION = "FROM"
_INSTRUCTION_RE = re.compile(
    r"(?im)^(?P<indent>[ \t]*)(?P<instruction>"
    r"FROM|COPY|ADD|RUN|WORKDIR|ENV|ARG"
    r")\b(?P<body>[^\r\n]*(?:\\\r?\n[^\r\n]*)*)"
)
_ARGUMENT_RE = re.compile(r'"[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s]+')
_JSON_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def _get_parser():
    from tree_sitter_language_pack import get_parser

    return get_parser("dockerfile")


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def _utf8_byte_to_char_offsets(source: bytes) -> list[int]:
    """Map tree-sitter byte offsets to ParsedDomainDocument char offsets."""

    offsets = [0] * (len(source) + 1)
    byte_offset = 0
    for char_offset, char in enumerate(source.decode("utf-8")):
        encoded_length = len(char.encode("utf-8"))
        for index in range(byte_offset, byte_offset + encoded_length):
            offsets[index] = char_offset
        byte_offset += encoded_length
        offsets[byte_offset] = char_offset + 1
    return offsets


def _token_at_char(doc: ParsedDomainDocument, char_offset: int) -> int | None:
    best: int | None = None
    for index, token in enumerate(doc.tokens):
        if token.start <= char_offset < token.end:
            return index
        if token.start <= char_offset:
            best = index
        else:
            break
    return best


def _lexical_arguments(
    doc: ParsedDomainDocument,
    body: str,
    body_start: int,
) -> list[tuple[str, int]]:
    """Return shell-form or JSON-array Docker arguments with source tokens."""

    json_start = body.find("[")
    if json_start >= 0:
        try:
            payload, json_end = json.JSONDecoder().raw_decode(body[json_start:])
        except (TypeError, ValueError):
            payload = None
            json_end = 0
        if isinstance(payload, list) and all(
            isinstance(value, str) for value in payload
        ):
            arguments = _lexical_arguments(
                doc,
                body[:json_start],
                body_start,
            )
            json_text = body[json_start : json_start + json_end]
            value_matches = list(_JSON_STRING_RE.finditer(json_text))
            if len(value_matches) == len(payload):
                for value, value_match in zip(
                    payload,
                    value_matches,
                    strict=True,
                ):
                    token_index = _token_at_char(
                        doc,
                        body_start + json_start + value_match.start(),
                    )
                    if token_index is not None:
                        arguments.append((value, token_index))
                return arguments

    arguments = []
    for argument in _ARGUMENT_RE.finditer(body):
        value = strip_quotes(argument.group(0).rstrip("\\"))
        if not value:
            continue
        token_index = _token_at_char(doc, body_start + argument.start())
        if token_index is not None:
            arguments.append((value, token_index))
    return arguments


def _attach_lexical_roles_and_edges(
    doc: ParsedDomainDocument,
    text: str,
) -> int:
    """Attach a deterministic Docker instruction graph without dependencies."""

    parsed_instructions = 0
    next_entity = 1
    for match in _INSTRUCTION_RE.finditer(text):
        parsed_instructions += 1
        instruction = match.group("instruction").upper()
        instruction_index = _token_at_char(doc, match.start("instruction"))
        if instruction_index is None:
            continue
        doc.set_role(
            instruction_index,
            DomainRoleKind.KEYWORD,
            entity=next_entity,
        )
        next_entity += 1

        body = match.group("body")
        body_start = match.start("body")
        arguments = _lexical_arguments(doc, body, body_start)
        if not arguments:
            continue

        if instruction == "FROM":
            non_options = [
                item for item in arguments if not item[0].startswith("--")
            ]
            if non_options:
                image_index = non_options[0][1]
                doc.set_role(
                    image_index,
                    DomainRoleKind.SOURCE,
                    entity=next_entity,
                )
                image_entity = next_entity
                next_entity += 1
                for position, (value, token_index) in enumerate(non_options[1:]):
                    if value.upper() == "AS":
                        doc.set_role(token_index, DomainRoleKind.KEYWORD)
                        if position + 2 <= len(non_options) - 1:
                            stage_index = non_options[position + 2][1]
                            doc.set_role(
                                stage_index,
                                DomainRoleKind.TARGET,
                                entity=next_entity,
                                scope=image_entity,
                            )
                            doc.add_edge(
                                stage_index,
                                image_index,
                                DomainEdgeKind.BUILD_TARGET_DEP,
                            )
                            next_entity += 1
                        break
        elif instruction in _COPY_ADD_INSTRUCTIONS:
            non_options = [
                item for item in arguments if not item[0].startswith("--")
            ]
            if len(non_options) >= 2:
                destination_index = non_options[-1][1]
                doc.set_role(
                    destination_index,
                    DomainRoleKind.OUTPUT,
                    entity=next_entity,
                )
                destination_entity = next_entity
                next_entity += 1
                for _value, source_index in non_options[:-1]:
                    doc.set_role(
                        source_index,
                        DomainRoleKind.INPUT,
                        entity=next_entity,
                        scope=destination_entity,
                    )
                    doc.add_edge(
                        destination_index,
                        source_index,
                        DomainEdgeKind.BUILD_ACTION_INPUT,
                    )
                    next_entity += 1
            for value, token_index in arguments:
                if value.startswith("--"):
                    doc.set_role(token_index, DomainRoleKind.OPTION)
        elif instruction == _RUN_INSTRUCTION:
            command_index = arguments[0][1]
            doc.set_role(
                command_index,
                DomainRoleKind.COMMAND,
                entity=next_entity,
            )
            doc.add_edge(
                instruction_index,
                command_index,
                DomainEdgeKind.BUILD_RULE_COMMAND,
            )
            command_entity = next_entity
            next_entity += 1
            for value, token_index in arguments[1:]:
                if value.startswith("-"):
                    doc.set_role(
                        token_index,
                        DomainRoleKind.OPTION,
                        scope=command_entity,
                    )
                elif "/" in value:
                    doc.set_role(
                        token_index,
                        DomainRoleKind.PATH,
                        scope=command_entity,
                    )
        elif instruction == "WORKDIR":
            doc.set_role(arguments[0][1], DomainRoleKind.PATH)
        elif instruction in {"ENV", "ARG"}:
            doc.set_role(
                arguments[0][1],
                DomainRoleKind.VARIABLE,
                entity=next_entity,
            )
            next_entity += 1
    for index, token in enumerate(doc.tokens):
        if token.text.startswith("#"):
            doc.set_role(index, DomainRoleKind.COMMENT)
    return parsed_instructions


def parse_dockerfile(text: str) -> ParsedDomainDocument:
    """Parse Dockerfile syntax on the frozen CONFIGURE build domain.

    The frozen tokenizer contract predates a dedicated Dockerfile delimiter.
    ``CONFIGURE`` is therefore the shared build-configuration domain while the
    exact ``dockerfile`` dialect remains explicit in parser metadata and the
    packed ``source_build_kinds`` sidecar.
    """

    doc = ParsedDomainDocument.new(
        domain=DomainKind.CONFIGURE,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={
            "parser": "tree-sitter",
            "parser_adapter": "dockerfile",
            "build_dialect": "dockerfile",
            "shared_domain": "configure",
        },
    )
    parsed_instructions = _attach_lexical_roles_and_edges(doc, text)
    doc.metadata["lexical_instruction_count"] = parsed_instructions

    try:
        parser = _get_parser()
    except Exception as exc:
        doc.metadata["parse_engine"] = "deterministic-lexical"
        doc.metadata["tree_sitter_unavailable"] = type(exc).__name__
        return (
            doc
            if parsed_instructions
            else doc.mark_raw("unrecognized_dockerfile_syntax")
        )

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    if root.has_error and root.child_count == 0:
        doc.metadata["parse_engine"] = "deterministic-lexical"
        return (
            doc
            if parsed_instructions
            else doc.mark_raw("tree-sitter produced empty parse")
        )

    edges: list[tuple[int, int, int]] = []
    stages: dict[str, int] = {}

    for node in _walk(root):
        if node.type not in {"from_instruction", "copy_instruction", "run_instruction",
                             "add_instruction", "instruction"}:
            continue

        instruction_node = None
        for c in node.children:
            if c.type in {"instruction", "keyword"} or (
                c.type == "ERROR" and c.child_count == 0
            ):
                instruction_node = c
                break
            if c.child_count == 0:
                text_upper = _node_text(c, source_bytes).upper()
                if text_upper in {"FROM", "COPY", "ADD", "RUN", "WORKDIR", "ENV", "ARG"}:
                    instruction_node = c
                    break

        if instruction_node is None:
            continue

        instr = _node_text(instruction_node, source_bytes).upper()

        if instr == _FROM_INSTRUCTION:
            for c in node.children:
                if c.type in {"image_spec", "image_name", "word"}:
                    img = _node_text(c, source_bytes)
                    stages[img.split(":")[0]] = c.start_byte
                    break
            as_nodes = [
                c for c in _walk(node) if _node_text(c, source_bytes).upper() == "AS"
            ]
            if as_nodes:
                next_sib = as_nodes[0].next_sibling
                if next_sib is not None:
                    stage_name = _node_text(next_sib, source_bytes)
                    stages[stage_name] = next_sib.start_byte

        elif instr in _COPY_ADD_INSTRUCTIONS:
            words = [
                c
                for c in _walk(node)
                if c.child_count == 0
                and c.type not in {"instruction", "keyword", "comment"}
                and not _node_text(c, source_bytes).upper() in {"COPY", "ADD", "--FROM", "AS"}
                and not _node_text(c, source_bytes).startswith("--")
            ]
            if len(words) >= 2:
                src_word = words[-2] if len(words) >= 2 else words[0]
                dst_word = words[-1]
                src_text = strip_quotes(_node_text(src_word, source_bytes))
                if src_text in stages:
                    edges.append(
                        (
                            stages[src_text],
                            src_word.start_byte,
                            int(DomainEdgeKind.BUILD_TARGET_DEP),
                        )
                    )
                edges.append(
                    (
                        src_word.start_byte,
                        dst_word.start_byte,
                        int(DomainEdgeKind.BUILD_ACTION_INPUT),
                    )
                )

        elif instr == _RUN_INSTRUCTION:
            run_nodes = [
                c
                for c in _walk(node)
                if c.child_count == 0
                and c.type not in {"instruction", "keyword", "comment"}
                and _node_text(c, source_bytes).upper() != "RUN"
            ]
            if len(run_nodes) >= 2:
                cmd_node = run_nodes[0]
                for arg_node in run_nodes[1:]:
                    arg_text = strip_quotes(_node_text(arg_node, source_bytes))
                    if "/" in arg_text or arg_text.endswith(
                        (".c", ".cpp", ".h", ".py", ".sh", ".txt")
                    ):
                        edges.append(
                            (
                                cmd_node.start_byte,
                                arg_node.start_byte,
                                int(DomainEdgeKind.BUILD_RULE_COMMAND),
                            )
                        )

    _attach_edges_to_doc(doc, edges, source_bytes)
    doc.metadata["parse_engine"] = "tree-sitter+deterministic-lexical"
    doc.metadata["tree_sitter_language"] = "dockerfile"
    doc.metadata["ast_root_type"] = root.type
    doc.metadata["ast_has_error"] = root.has_error
    return doc


def _attach_edges_to_doc(
    doc: ParsedDomainDocument,
    raw_edges: list[tuple[int, int, int]],
    source: bytes,
) -> None:
    tokens = doc.tokens
    if not tokens:
        return
    byte_to_char_offsets = _utf8_byte_to_char_offsets(source)

    def byte_to_token_idx(byte_offset: int) -> int | None:
        if not 0 <= byte_offset <= len(source):
            return None
        char_offset = byte_to_char_offsets[byte_offset]
        best: int | None = None
        for idx, tok in enumerate(tokens):
            if tok.start <= char_offset < tok.end:
                return idx
            if tok.start <= char_offset:
                best = idx
            else:
                break
        return best

    seen: set[tuple[int, int, int]] = set(doc.edges)
    for src_byte, dst_byte, kind in raw_edges:
        src_idx = byte_to_token_idx(src_byte)
        dst_idx = byte_to_token_idx(dst_byte)
        if src_idx is None or dst_idx is None or src_idx == dst_idx:
            continue
        triple = (src_idx, dst_idx, kind)
        if triple not in seen:
            seen.add(triple)
            doc.edges.append(triple)


__all__ = ["parse_dockerfile"]
