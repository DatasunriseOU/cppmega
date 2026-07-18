"""Zsh parser wrapper."""

from __future__ import annotations

from cppmega.data.domain_schema import DomainKind
from cppmega.data.shell_parsers.base import parse_shell


def parse_zsh(text: str):
    return parse_shell(text, domain=DomainKind.ZSH, shell_kind="zsh")


__all__ = ["parse_zsh"]
