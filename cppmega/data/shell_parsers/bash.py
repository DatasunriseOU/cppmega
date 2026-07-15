"""Bash parser wrapper."""

from __future__ import annotations

from cppmega.data.domain_schema import DomainKind
from cppmega.data.shell_parsers.base import parse_shell


def parse_bash(text: str):
    return parse_shell(text, domain=DomainKind.BASH, shell_kind="bash")


__all__ = ["parse_bash"]
