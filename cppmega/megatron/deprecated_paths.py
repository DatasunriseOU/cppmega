"""Fail-closed gates for deprecated cppmega runtime paths."""

from __future__ import annotations

import os
import sys

_warned: set[tuple[str, str]] = set()


def require_deprecated_ack(
    *,
    feature: str,
    ack_env: str,
    replacement: str,
    reason: str | None = None,
) -> None:
    """Require an explicit env acknowledgement before running old code paths."""
    if os.environ.get(ack_env, "0") != "1":
        detail = f" {reason}" if reason else ""
        raise RuntimeError(
            f"{feature} is DEPRECATED and disabled by default.{detail} "
            f"Use {replacement}. To force this old path, set {ack_env}=1."
        )

    key = (feature, ack_env)
    if key not in _warned:
        print(
            f"[cppmega] DEPRECATED: {feature} enabled because {ack_env}=1. "
            f"Use {replacement}.",
            file=sys.stderr,
        )
        _warned.add(key)
