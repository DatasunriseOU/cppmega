"""GCC diagnostic parser entrypoint.

GCC's common text diagnostic shape is close enough to Clang's for this
lightweight fallback parser, but it stays a separate module so callers can route
tool-specific structured JSON/SARIF parsers here later without changing imports.
"""

from __future__ import annotations

from cppmega.data.diagnostic_parsers.clang import parse_gcc_diagnostic

__all__ = ["parse_gcc_diagnostic"]
