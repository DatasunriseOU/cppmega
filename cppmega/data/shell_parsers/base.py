"""Shared shell-domain parser helpers."""

from __future__ import annotations

import re

from cppmega.data.build_parsers.base import ParsedDomainDocument, is_option
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_SHELL_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "do", "done", "case",
    "esac", "while", "until", "export", "readonly", "unset",
}
_BASH_KEYWORDS = {"declare", "local", "mapfile", "readarray", "select", "shopt"}
_ZSH_KEYWORDS = {"autoload", "emulate", "setopt", "unsetopt", "zmodload"}
_KSH_KEYWORDS = {"compound", "function", "integer", "nameref", "typeset"}
_TCSH_KEYWORDS = {"setenv", "unsetenv", "alias", "foreach", "endif", "switch", "endsw"}
_REDIR_OUT = {">", ">>", "2>", "&>"}
_REDIR_IN = {"<"}
_HEREDOC_RE = re.compile(
    r"<<(?P<strip>-?)[ \t]*(?P<quote>['\"]?)"
    r"(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)


def _shell_lexical_spans(
    text: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return comment and quoted spans without losing source offsets."""

    comments: list[tuple[int, int]] = []
    quotes: list[tuple[int, int]] = []
    quote: str | None = None
    quote_start = 0
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quotes.append((quote_start, index + 1))
                quote = None
            index += 1
            continue
        if quote == '"':
            if char == "\\":
                escaped = True
            elif char == '"':
                quotes.append((quote_start, index + 1))
                quote = None
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            quote_start = index
            index += 1
            continue
        if char == "#" and (
            index == 0
            or text[index - 1].isspace()
            or text[index - 1] in ";|&(){}<>"
        ):
            end = text.find("\n", index)
            if end < 0:
                end = len(text)
            comments.append((index, end))
            index = end
            continue
        index += 1
    if quote is not None:
        quotes.append((quote_start, len(text)))
    return comments, quotes


def _shell_heredoc_spans(
    text: str,
    *,
    protected_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return heredoc declaration and payload spans, including terminators."""

    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    def protected(position: int) -> bool:
        return any(start <= position < end for start, end in protected_spans)

    spans: list[tuple[int, int]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        pending: list[tuple[int, int, str, bool]] = []
        for match in _HEREDOC_RE.finditer(line):
            declaration_start = offsets[line_index] + match.start()
            if protected(declaration_start):
                continue
            if match.start() > 0 and line[match.start() - 1] == "<":
                continue
            prefix = line[: match.start()]
            if prefix.rfind("((") > prefix.rfind("))"):
                continue
            pending.append(
                (
                    declaration_start,
                    offsets[line_index] + match.end(),
                    match.group("delimiter"),
                    match.group("strip") == "-",
                )
            )
        if not pending:
            line_index += 1
            continue

        scan_index = line_index + 1
        for declaration_start, declaration_end, delimiter, strip_tabs in pending:
            spans.append((declaration_start, declaration_end))
            body_start = offsets[scan_index] if scan_index < len(lines) else len(text)
            close_end = len(text)
            while scan_index < len(lines):
                candidate = lines[scan_index].rstrip("\r\n")
                comparison = candidate.lstrip("\t") if strip_tabs else candidate
                if comparison == delimiter:
                    close_end = offsets[scan_index] + len(lines[scan_index])
                    scan_index += 1
                    break
                scan_index += 1
            spans.append((body_start, close_end))
        line_index = max(line_index + 1, scan_index)
    return spans


def _ignored_shell_spans(text: str) -> list[tuple[int, int]]:
    """Find regions that cannot contain shell commands or graph operators."""

    comments, quotes = _shell_lexical_spans(text)
    ignored = list(comments)
    ignored.extend(
        span
        for span in quotes
        if "\n" in text[span[0] : span[1]] or "\r" in text[span[0] : span[1]]
    )
    ignored.extend(
        _shell_heredoc_spans(
            text,
            protected_spans=comments + quotes,
        )
    )
    return sorted(ignored)


def _mask_ignored_shell_spans(text: str) -> str:
    masked = list(text)
    for start, end in _ignored_shell_spans(text):
        for index in range(start, end):
            if masked[index] not in "\r\n":
                masked[index] = " "
    return "".join(masked)


def _shell_syntax_words(text: str) -> tuple[list[str], bool]:
    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    comment = False

    def flush() -> None:
        if current:
            words.append("".join(current).lower())
            current.clear()

    for index, char in enumerate(text):
        if comment:
            if char == "\n":
                comment = False
            continue
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            flush()
            quote = char
            continue
        comment_start = (
            char == "#"
            and (
                index == 0
                or text[index - 1].isspace()
                or text[index - 1] in ";|&(){}"
            )
        )
        if comment_start:
            flush()
            comment = True
            continue
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            flush()
    flush()
    return words, quote is None


def _has_unbalanced_shell_syntax(text: str, shell_kind: str) -> bool:
    words, quotes_balanced = _shell_syntax_words(text)
    if not quotes_balanced:
        return True

    if shell_kind == "tcsh":
        openers = {
            "if": "endif",
            "foreach": "end",
            "while": "end",
            "switch": "endsw",
        }
        closers = {"end", "endsw", "endif"}
    else:
        openers = {
            "if": "fi",
            "case": "esac",
            "for": "done",
            "while": "done",
            "until": "done",
        }
        if shell_kind in {"bash", "zsh", "ksh"}:
            openers["select"] = "done"
        closers = {"fi", "esac", "done"}

    expected_closers: list[str] = []
    for word in words:
        if word in openers:
            expected_closers.append(openers[word])
        elif word in closers:
            if not expected_closers or expected_closers.pop() != word:
                return True
    return bool(expected_closers)


def parse_shell(
    text: str,
    *,
    domain: DomainKind,
    shell_kind: str,
    malformed_reason: str | None = None,
) -> ParsedDomainDocument:
    adapter = {
        "sh": "posix-sh",
        "configure": "configure-shell",
    }.get(shell_kind, shell_kind)
    doc = ParsedDomainDocument.new(
        domain=domain,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={
            "shell_kind": shell_kind,
            "parser_adapter": adapter,
        },
    )
    next_entity = 1
    previous_command: int | None = None
    command_expected = True
    pending_redir: tuple[int, DomainEdgeKind] | None = None
    ignored_spans = _ignored_shell_spans(text)

    def ignored(token_start: int, token_end: int) -> bool:
        return any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in ignored_spans
        )

    idx = 0
    while idx < len(doc.tokens):
        token = doc.tokens[idx]
        if idx > 0 and token.line != doc.tokens[idx - 1].line:
            previous_command = None
            command_expected = True
            pending_redir = None
        if ignored(token.start, token.end):
            idx += 1
            continue
        value = token.text
        if shell_kind == "tcsh" and value in _TCSH_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
            keyword_entity = next_entity
            next_entity += 1
            if value == "setenv" and idx + 1 < len(doc.tokens):
                env_idx = idx + 1
                doc.set_role(env_idx, DomainRoleKind.ENVIRONMENT, entity=next_entity, scope=keyword_entity)
                if idx + 2 < len(doc.tokens):
                    doc.set_role(idx + 2, DomainRoleKind.STRING, entity=next_entity, scope=keyword_entity)
                next_entity += 1
                idx += 3
                command_expected = True
                continue
            command_expected = True
            idx += 1
            continue
        if shell_kind == "zsh" and value in _ZSH_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
            next_entity += 1
            command_expected = False
            idx += 1
            continue
        if shell_kind == "ksh" and value in _KSH_KEYWORDS:
            doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
            next_entity += 1
            command_expected = False
            idx += 1
            continue
        if value == "|":
            doc.set_role(idx, DomainRoleKind.PIPE)
            if previous_command is not None:
                pending_pipe_src = previous_command
            else:
                pending_pipe_src = None
            command_expected = True
            pending_redir = None
            # Store temporary source in metadata-like scope via negative-free local.
            previous_command = pending_pipe_src
            idx += 1
            continue
        if value in _REDIR_OUT | _REDIR_IN:
            edge_kind = (
                DomainEdgeKind.SHELL_REDIR_OUT
                if value in _REDIR_OUT
                else DomainEdgeKind.SHELL_REDIR_IN
            )
            doc.set_role(idx, DomainRoleKind.REDIRECT)
            pending_redir = (previous_command if previous_command is not None else idx, edge_kind)
            idx += 1
            continue
        if (
            command_expected
            and value.isidentifier()
            and idx + 2 < len(doc.tokens)
            and doc.tokens[idx + 1].text == "="
        ):
            doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
            doc.set_role(idx + 2, DomainRoleKind.STRING, entity=next_entity)
            next_entity += 1
            idx += 3
            continue
        if "=" in value and not value.startswith("=") and value.split("=", 1)[0].isidentifier():
            doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
            next_entity += 1
            command_expected = True
            idx += 1
            continue
        if value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
            idx += 1
            continue
        dialect_keywords = _SHELL_KEYWORDS | (
            _BASH_KEYWORDS
            if shell_kind == "bash"
            else _ZSH_KEYWORDS
            if shell_kind == "zsh"
            else _KSH_KEYWORDS
            if shell_kind == "ksh"
            else set()
        )
        if value in dialect_keywords:
            doc.set_role(idx, DomainRoleKind.KEYWORD)
            command_expected = True
            idx += 1
            continue
        if pending_redir is not None:
            src, edge_kind = pending_redir
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity)
            next_entity += 1
            doc.add_edge(src, idx, edge_kind)
            pending_redir = None
            idx += 1
            continue
        if command_expected and value not in {";", "&&", "||"}:
            old_command = previous_command
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            previous_command = idx
            next_entity += 1
            if old_command is not None and any(
                doc.tokens[j].text == "|" for j in range(max(0, old_command + 1), idx)
            ):
                doc.add_edge(old_command, idx, DomainEdgeKind.SHELL_PIPE)
            command_expected = False
            idx += 1
            continue
        if is_option(value):
            doc.set_role(idx, DomainRoleKind.OPTION, scope=previous_command or 0)
        elif "/" in value or "." in value:
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity, scope=previous_command or 0)
            next_entity += 1
            if previous_command is not None:
                doc.add_edge(previous_command, idx, DomainEdgeKind.SHELL_COMMAND_FILE)

        if value in {";", "&&", "||"}:
            command_expected = True
        idx += 1

    if _has_unbalanced_shell_syntax(
        _mask_ignored_shell_spans(text),
        shell_kind,
    ):
        return doc.mark_raw(malformed_reason or f"malformed_{shell_kind}_shell")
    return doc


__all__ = ["parse_shell"]
