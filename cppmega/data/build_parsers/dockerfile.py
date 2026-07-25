"""Dockerfile domain parser using tree-sitter AST."""

from __future__ import annotations

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


def _get_parser():
    from tree_sitter_language_pack import get_parser

    return get_parser("dockerfile")


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def parse_dockerfile(text: str) -> ParsedDomainDocument:
    """Parse a Dockerfile using tree-sitter and extract domain edges."""
    doc = ParsedDomainDocument.new(
        domain=DomainKind.DOCKERFILE,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser": "tree-sitter"},
    )

    try:
        parser = _get_parser()
    except Exception:
        return doc.mark_raw("tree-sitter dockerfile grammar unavailable")

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    if root.has_error and root.child_count == 0:
        return doc.mark_raw("tree-sitter produced empty parse")

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

    _attach_edges_to_doc(doc, edges)
    doc.metadata["tree_sitter_language"] = "dockerfile"
    doc.metadata["ast_root_type"] = root.type
    doc.metadata["ast_has_error"] = root.has_error
    return doc


def _attach_edges_to_doc(
    doc: ParsedDomainDocument, raw_edges: list[tuple[int, int, int]]
) -> None:
    tokens = doc.tokens
    if not tokens:
        return

    def byte_to_token_idx(byte_offset: int) -> int | None:
        best: int | None = None
        for idx, tok in enumerate(tokens):
            if tok.start <= byte_offset < tok.end:
                return idx
            if tok.start <= byte_offset:
                best = idx
            else:
                break
        return best

    seen: set[tuple[int, int, int]] = set()
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
