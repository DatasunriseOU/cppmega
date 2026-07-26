"""Meson build system domain parser using tree-sitter AST."""

from __future__ import annotations

from typing import Any

from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    is_source_path,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    ParseConfidence,
)

_TARGET_FUNCTIONS = {
    "executable",
    "shared_library",
    "static_library",
    "library",
    "both_libraries",
    "shared_module",
    "custom_target",
    "run_target",
    "jar",
}

_DEP_FUNCTIONS = {
    "dependency",
    "declare_dependency",
    "find_library",
}

_SOURCE_FUNCTIONS = {
    "files",
    "configure_file",
    "custom_target",
}


def _get_parser():
    from tree_sitter_language_pack import get_parser

    return get_parser("meson")


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _utf8_byte_to_char_offsets(source: bytes) -> list[int]:
    """Map UTF-8 byte positions to the character offsets used by lexed tokens."""

    offsets = [0] * (len(source) + 1)
    byte_offset = 0
    for char_offset, char in enumerate(source.decode("utf-8")):
        encoded_length = len(char.encode("utf-8"))
        for index in range(byte_offset, byte_offset + encoded_length):
            offsets[index] = char_offset
        byte_offset += encoded_length
        offsets[byte_offset] = char_offset + 1
    return offsets


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_string_args(node: Any, source: bytes) -> list[tuple[str, int]]:
    """Extract string literal arguments from a function call node."""
    results: list[tuple[str, int]] = []
    for child in node.children:
        if child.type == "variableunit":
            for sub in child.children:
                if sub.type == "string":
                    text = _node_text(sub, source).strip("'\"")
                    results.append((text, sub.start_byte))
        elif child.type == "string":
            text = _node_text(child, source).strip("'\"")
            results.append((text, child.start_byte))
    return results


def parse_meson(text: str) -> ParsedDomainDocument:
    """Parse a meson.build file using tree-sitter and extract domain edges."""
    doc = ParsedDomainDocument.new(
        domain=DomainKind.MESON,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser": "tree-sitter"},
    )

    try:
        parser = _get_parser()
    except Exception:
        return doc.mark_raw("tree-sitter meson grammar unavailable")

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    if root.has_error and root.child_count == 0:
        return doc.mark_raw("tree-sitter produced empty parse")

    edges: list[tuple[int, int, int]] = []
    targets: dict[str, int] = {}

    for node in _walk(root):
        if node.type != "normal_command":
            continue

        func_name_node = None
        for c in node.children:
            if c.type == "identifier":
                func_name_node = c
                break
        if func_name_node is None:
            continue

        func_name = _node_text(func_name_node, source_bytes)
        args = _extract_string_args(node, source_bytes)

        if func_name in _TARGET_FUNCTIONS and args:
            target_name = args[0][0]
            target_offset = func_name_node.start_byte
            targets[target_name] = target_offset
            for arg_text, arg_offset in args[1:]:
                if is_source_path(arg_text) or arg_text.endswith((".build", ".py")):
                    edges.append(
                        (target_offset, arg_offset, int(DomainEdgeKind.BUILD_TARGET_SOURCE))
                    )
                elif not arg_text.startswith("-"):
                    edges.append(
                        (target_offset, arg_offset, int(DomainEdgeKind.BUILD_TARGET_DEP))
                    )

        elif func_name in _DEP_FUNCTIONS and args:
            dep_name = args[0][0]
            dep_offset = func_name_node.start_byte
            targets.setdefault(dep_name, dep_offset)

        elif func_name == "subdir" and args:
            edges.append(
                (
                    func_name_node.start_byte,
                    args[0][1],
                    int(DomainEdgeKind.BUILD_TARGET_DEP),
                )
            )

    _attach_edges_to_doc(doc, edges, source_bytes)
    doc.metadata["tree_sitter_language"] = "meson"
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


__all__ = ["parse_meson"]
