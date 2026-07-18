"""Diagnostic-domain parser entrypoints."""

from cppmega.data.diagnostic_parsers.clang import parse_clang_diagnostic
from cppmega.data.diagnostic_parsers.cmake import parse_build_error
from cppmega.data.diagnostic_parsers.gcc import parse_gcc_diagnostic
from cppmega.data.diagnostic_parsers.linker import parse_linker_error
from cppmega.data.diagnostic_parsers.msvc import parse_msvc_diagnostic
from cppmega.data.diagnostic_parsers.runtime import (
    parse_sanitizer_output,
    parse_test_output,
)

__all__ = [
    "parse_build_error",
    "parse_clang_diagnostic",
    "parse_gcc_diagnostic",
    "parse_linker_error",
    "parse_msvc_diagnostic",
    "parse_sanitizer_output",
    "parse_test_output",
]
