"""KornShell parser wrapper."""

from __future__ import annotations

from cppmega.data.domain_schema import DomainKind
from cppmega.data.shell_parsers.base import parse_shell


def parse_ksh(text: str):
    return parse_shell(text, domain=DomainKind.KSH, shell_kind="ksh")


__all__ = ["parse_ksh"]
