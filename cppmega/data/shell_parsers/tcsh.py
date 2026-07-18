"""Tcsh parser wrapper."""

from __future__ import annotations

from cppmega.data.domain_schema import DomainKind
from cppmega.data.shell_parsers.base import parse_shell


def parse_tcsh(text: str):
    return parse_shell(text, domain=DomainKind.TCSH, shell_kind="tcsh")


__all__ = ["parse_tcsh"]
