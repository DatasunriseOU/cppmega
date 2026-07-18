"""POSIX sh parser."""

from __future__ import annotations

from cppmega.data.domain_schema import DomainKind
from cppmega.data.shell_parsers.base import parse_shell


def parse_sh(text: str):
    return parse_shell(text, domain=DomainKind.SH, shell_kind="sh")


__all__ = ["parse_sh"]
