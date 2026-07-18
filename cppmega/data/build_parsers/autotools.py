"""Autotools build-domain parsers."""

from __future__ import annotations

from cppmega.data.build_parsers.base import ParsedDomainDocument
from cppmega.data.build_parsers.make import parse_automake
from cppmega.data.shell_parsers.base import parse_shell
from cppmega.data.domain_schema import (
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_AUTOCONF_MACROS = {
    "AC_INIT",
    "AC_CONFIG_FILES",
    "AC_PROG_CC",
    "AC_PROG_CXX",
    "AM_INIT_AUTOMAKE",
    "PKG_CHECK_MODULES",
}


def _is_balanced(text: str) -> bool:
    paren_depth = 0
    quote_depth = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if quote_depth:
            if char == "[":
                quote_depth += 1
            elif char == "]":
                quote_depth -= 1
            idx += 1
            continue
        if char == "[":
            quote_depth = 1
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return False
        elif char == "#":
            newline = text.find("\n", idx)
            idx = len(text) if newline < 0 else newline
            continue
        elif text[idx:idx + 3].lower() == "dnl" and (
            idx == 0 or not (text[idx - 1].isalnum() or text[idx - 1] == "_")
        ):
            after = idx + 3
            if after == len(text) or not (text[after].isalnum() or text[after] == "_"):
                newline = text.find("\n", after)
                idx = len(text) if newline < 0 else newline
                continue
        idx += 1
    return paren_depth == 0 and quote_depth == 0


def parse_autoconf(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.AUTOCONF,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "autoconf", "parser_adapter": "autoconf"},
    )
    if not _is_balanced(text):
        return doc.mark_raw("malformed_autoconf_macro")

    next_entity = 1
    for idx, token in enumerate(doc.tokens):
        value = token.text
        if value in _AUTOCONF_MACROS or value.startswith("AC_") or value.startswith("AM_"):
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            next_entity += 1
        elif value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
    return doc


def parse_configure(text: str) -> ParsedDomainDocument:
    doc = parse_shell(
        text,
        domain=DomainKind.CONFIGURE,
        shell_kind="configure",
        malformed_reason="malformed_configure_shell",
    )
    doc.metadata["build_kind"] = "configure"
    return doc


__all__ = ["parse_autoconf", "parse_automake", "parse_configure"]
