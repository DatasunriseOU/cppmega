"""Python domain parser backed only by the stdlib AST and tokenizer."""

from __future__ import annotations

import ast
from bisect import bisect_right
from collections import defaultdict
import io
import keyword
import tokenize

from cppmega.data.build_parsers.base import LexedToken, ParsedDomainDocument
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_SKIPPED_TOKEN_TYPES = {
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}
_SCOPE_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    return starts


def _token_offset(
    line_starts: list[int],
    position: tuple[int, int],
    *,
    text_length: int,
) -> int:
    line, column = position
    if line <= 0:
        return 0
    if line > len(line_starts):
        return text_length
    return min(text_length, line_starts[line - 1] + max(0, int(column)))


def _tokenize_source(
    text: str,
) -> tuple[list[LexedToken], list[tokenize.TokenInfo], str | None]:
    starts = _line_starts(text)
    tokens: list[LexedToken] = []
    token_infos: list[tokenize.TokenInfo] = []
    error: str | None = None
    try:
        for info in tokenize.generate_tokens(io.StringIO(text).readline):
            if info.type in _SKIPPED_TOKEN_TYPES:
                continue
            if info.type == tokenize.ERRORTOKEN and info.string.isspace():
                continue
            start = _token_offset(starts, info.start, text_length=len(text))
            end = _token_offset(starts, info.end, text_length=len(text))
            token_text = info.string
            if tokenize.tok_name.get(info.type) == "FSTRING_MIDDLE":
                leading_whitespace = len(token_text) - len(token_text.lstrip())
                start += leading_whitespace
                token_text = token_text[leading_whitespace:]
            if end <= start:
                continue
            line = max(0, bisect_right(starts, start) - 1)
            tokens.append(
                LexedToken(
                    text=token_text,
                    start=start,
                    end=end,
                    line=line,
                    column=start - starts[line],
                )
            )
            token_infos.append(info)
    except (IndentationError, tokenize.TokenError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return tokens, token_infos, error


def _new_document(
    text: str,
    tokens: list[LexedToken],
) -> ParsedDomainDocument:
    token_count = len(tokens)
    return ParsedDomainDocument(
        domain=DomainKind.PYTHON,
        text=text,
        tokens=tokens,
        domain_ids=[int(DomainKind.PYTHON)] * token_count,
        role_ids=[int(DomainRoleKind.NONE)] * token_count,
        entity_ids=[0] * token_count,
        scope_ids=[0] * token_count,
        source_doc_ids=[0] * token_count,
        source_identity_ids=[0] * token_count,
        confidence_ids=[int(ParseConfidence.EXACT)] * token_count,
        metadata={
            "language": "python",
            "parser_adapter": "python-ast-tokenize",
            "syntax_parser": "stdlib-ast",
            "lexical_parser": "stdlib-tokenize",
        },
    )


def _ast_offset_maps(text: str) -> tuple[list[int], list[list[int]]]:
    lines = text.splitlines(keepends=True) or [""]
    starts: list[int] = []
    byte_to_char: list[list[int]] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
        mapping = [0] * (len(line.encode("utf-8")) + 1)
        byte_cursor = 0
        for char_index, char in enumerate(line):
            char_width = len(char.encode("utf-8"))
            for byte_index in range(char_width):
                mapping[byte_cursor + byte_index] = char_index
            byte_cursor += char_width
        mapping[byte_cursor] = len(line)
        byte_to_char.append(mapping)
    return starts, byte_to_char


def _ast_offset(
    line_starts: list[int],
    byte_to_char: list[list[int]],
    line: int | None,
    byte_column: int | None,
    *,
    text_length: int,
) -> int | None:
    if line is None or byte_column is None or line <= 0 or line > len(line_starts):
        return None
    line_map = byte_to_char[line - 1]
    bounded_column = min(len(line_map) - 1, max(0, int(byte_column)))
    return min(text_length, line_starts[line - 1] + line_map[bounded_column])


class _TokenSpans:
    def __init__(self, tokens: list[LexedToken]) -> None:
        self.tokens = tokens
        self.starts = [token.start for token in tokens]

    def overlapping(self, start: int, end: int) -> list[int]:
        if end <= start or not self.tokens:
            return []
        index = max(0, bisect_right(self.starts, start) - 1)
        while index > 0 and self.tokens[index - 1].end > start:
            index -= 1
        result: list[int] = []
        while index < len(self.tokens) and self.tokens[index].start < end:
            if self.tokens[index].end > start:
                result.append(index)
            index += 1
        return result


def _node_span(
    node: ast.AST,
    *,
    line_starts: list[int],
    byte_to_char: list[list[int]],
    text_length: int,
) -> tuple[int, int] | None:
    start = _ast_offset(
        line_starts,
        byte_to_char,
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        text_length=text_length,
    )
    end = _ast_offset(
        line_starts,
        byte_to_char,
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
        text_length=text_length,
    )
    if start is None or end is None or end <= start:
        return None
    return start, end


def _classify_lexical_tokens(
    doc: ParsedDomainDocument,
    token_infos: list[tokenize.TokenInfo],
) -> None:
    for index, info in enumerate(token_infos):
        token_name = tokenize.tok_name.get(info.type, "")
        if info.type == tokenize.COMMENT:
            doc.set_role(index, DomainRoleKind.COMMENT)
        elif info.type == tokenize.NAME:
            role = (
                DomainRoleKind.KEYWORD
                if keyword.iskeyword(info.string) or keyword.issoftkeyword(info.string)
                else DomainRoleKind.IDENTIFIER
            )
            doc.set_role(index, role)
        elif info.type == tokenize.STRING or token_name.startswith("FSTRING"):
            doc.set_role(index, DomainRoleKind.STRING)


def _parse_error_message(exc: SyntaxError) -> str:
    location = ""
    if exc.lineno is not None:
        location = f" at line {exc.lineno}"
        if exc.offset is not None:
            location += f", column {exc.offset}"
    return f"{exc.msg}{location}"


def _apply_ast(
    doc: ParsedDomainDocument,
    tree: ast.AST,
) -> None:
    line_starts, byte_to_char = _ast_offset_maps(doc.text)
    spans = _TokenSpans(doc.tokens)
    parents: dict[ast.AST, ast.AST] = {}
    depths: dict[ast.AST, int] = {tree: 0}
    for parent in ast.walk(tree):
        parent_depth = depths.get(parent, 0)
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
            depths[child] = parent_depth + 1

    span_cache: dict[ast.AST, tuple[int, int] | None] = {}
    representative_cache: dict[ast.AST, int | None] = {}

    def node_span(node: ast.AST) -> tuple[int, int] | None:
        if node not in span_cache:
            span_cache[node] = _node_span(
                node,
                line_starts=line_starts,
                byte_to_char=byte_to_char,
                text_length=len(doc.text),
            )
        return span_cache[node]

    def named_token(node: ast.AST, name: str, *, last: bool = False) -> int | None:
        span = node_span(node)
        if span is None:
            return None
        matches = [
            index
            for index in spans.overlapping(*span)
            if doc.tokens[index].text == name
        ]
        if not matches and "." in name:
            terminal = name.rsplit(".", 1)[-1]
            matches = [
                index
                for index in spans.overlapping(*span)
                if doc.tokens[index].text == terminal
            ]
        if not matches:
            return None
        return matches[-1] if last else matches[0]

    def representative(node: ast.AST) -> int | None:
        if node in representative_cache:
            return representative_cache[node]
        index: int | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index = named_token(node, node.name)
        elif isinstance(node, ast.Name):
            index = named_token(node, node.id)
        elif isinstance(node, ast.arg):
            index = named_token(node, node.arg)
        elif isinstance(node, ast.Attribute):
            index = named_token(node, node.attr, last=True)
        elif isinstance(node, ast.alias):
            index = named_token(node, node.asname or node.name, last=True)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            index = named_token(node, node.arg)
        if index is None:
            span = node_span(node)
            if span is not None:
                matches = spans.overlapping(*span)
                index = matches[0] if matches else None
        representative_cache[node] = index
        return index

    edge_set: set[tuple[int, int, int]] = set()

    def add_edge(src: int | None, dst: int | None, kind: DomainEdgeKind) -> None:
        if src is None or dst is None or src == dst:
            return
        edge = (int(src), int(dst), int(kind))
        if edge in edge_set:
            return
        doc.add_edge(src, dst, kind)
        edge_set.add(edge)

    scope_entity_by_node: dict[ast.AST, int] = {}
    definition_entity_by_token: dict[int, int] = {}
    definitions: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    next_entity = 1

    def enclosing_scope_nodes(node: ast.AST) -> list[ast.AST]:
        result: list[ast.AST] = []
        current = parents.get(node)
        while current is not None:
            if isinstance(current, _SCOPE_NODE_TYPES):
                result.append(current)
            current = parents.get(current)
        return result

    def scope_id_for(node: ast.AST) -> int:
        for scope_node in enclosing_scope_nodes(node):
            entity = scope_entity_by_node.get(scope_node)
            if entity is not None:
                return entity
        return 0

    def register_definition(name: str, node: ast.AST, index: int | None) -> int | None:
        nonlocal next_entity
        if index is None:
            return None
        entity = definition_entity_by_token.get(index)
        if entity is None:
            entity = next_entity
            next_entity += 1
            definition_entity_by_token[index] = entity
        scope_id = scope_id_for(node)
        doc.set_role(
            index,
            DomainRoleKind.IDENTIFIER,
            entity=entity,
            scope=scope_id,
        )
        definitions[name].append((doc.tokens[index].start, entity, index))
        return entity

    ordered_nodes = sorted(
        ast.walk(tree),
        key=lambda node: (
            node_span(node)[0] if node_span(node) is not None else len(doc.text),
            depths.get(node, 0),
            type(node).__name__,
        ),
    )
    for node in ordered_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index = named_token(node, node.name)
            entity = register_definition(node.name, node, index)
            if entity is not None:
                scope_entity_by_node[node] = entity
        elif isinstance(node, ast.arg):
            register_definition(node.arg, node, named_token(node, node.arg))
        elif isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            register_definition(node.id, node, representative(node))
        elif isinstance(node, ast.alias):
            name = node.asname or node.name.split(".", 1)[0]
            register_definition(name, node, named_token(node, name, last=True))
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            register_definition(node.name, node, named_token(node, node.name))

    for owner in ast.walk(tree):
        if not isinstance(owner, (ast.Module, *_SCOPE_NODE_TYPES)) or not owner.body:
            continue
        first = owner.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        span = node_span(first.value)
        if span is None:
            continue
        for index in spans.overlapping(*span):
            doc.set_role(index, DomainRoleKind.DOCSTRING)

    scope_depths = [-1] * len(doc.tokens)
    for node in ordered_nodes:
        span = node_span(node)
        scope_id = scope_id_for(node)
        if span is None or scope_id == 0:
            continue
        depth = len(enclosing_scope_nodes(node))
        for index in spans.overlapping(*span):
            if depth >= scope_depths[index]:
                doc.scope_ids[index] = scope_id
                scope_depths[index] = depth

    for child, parent in parents.items():
        add_edge(representative(parent), representative(child), DomainEdgeKind.AST_PARENT)

    for node in ordered_nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            use_index = representative(node)
            if use_index is None:
                continue
            scope_chain = [
                scope_entity_by_node[scope_node]
                for scope_node in enclosing_scope_nodes(node)
                if scope_node in scope_entity_by_node
            ]
            scope_chain.append(0)
            candidates = [
                definition
                for definition in definitions.get(node.id, [])
                if definition[0] <= doc.tokens[use_index].start
                and doc.scope_ids[definition[2]] in scope_chain
            ]
            if not candidates:
                continue
            _position, entity, definition_index = max(candidates, key=lambda item: item[0])
            doc.set_role(
                use_index,
                DomainRoleKind.IDENTIFIER,
                entity=entity,
                scope=doc.scope_ids[use_index],
            )
            add_edge(definition_index, use_index, DomainEdgeKind.DEF_USE)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            index = named_token(node, node.arg)
            if index is not None:
                doc.set_role(index, DomainRoleKind.ATTRIBUTE)
        elif isinstance(node, ast.Call):
            callee_index = representative(node.func)
            owner_index = next(
                (
                    representative(scope_node)
                    for scope_node in enclosing_scope_nodes(node)
                    if representative(scope_node) is not None
                ),
                None,
            )
            add_edge(owner_index, callee_index, DomainEdgeKind.CALL)

    for node in ordered_nodes:
        if isinstance(node, ast.arg) and node.annotation is not None:
            add_edge(representative(node), representative(node.annotation), DomainEdgeKind.TYPE)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_edge(representative(node), representative(node.returns), DomainEdgeKind.TYPE)
        elif isinstance(node, ast.AnnAssign):
            add_edge(representative(node.target), representative(node.annotation), DomainEdgeKind.TYPE)


def parse_python(text: str) -> ParsedDomainDocument:
    """Parse Python into token-aligned roles, entities, scopes, and graph edges."""

    tokens, token_infos, tokenize_error = _tokenize_source(text)
    doc = _new_document(text, tokens)
    _classify_lexical_tokens(doc, token_infos)
    try:
        tree = ast.parse(text, type_comments=True)
    except SyntaxError as exc:
        doc.metadata["syntax_error"] = _parse_error_message(exc)
        if tokenize_error is not None:
            doc.metadata["tokenize_error"] = tokenize_error
        doc.mark_raw("malformed_python_syntax")
        doc.validate()
        return doc
    if tokenize_error is not None:
        doc.metadata["tokenize_error"] = tokenize_error
        doc.mark_raw("malformed_python_token_stream")
        doc.validate()
        return doc

    _apply_ast(doc, tree)
    doc.validate()
    return doc


__all__ = ["parse_python"]
