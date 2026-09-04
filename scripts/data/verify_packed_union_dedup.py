#!/usr/bin/env python3
"""Verify the packed-union 'no global dedup' receipt. Does not invent overlay 501."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA = "cppmega_packed_union_dedup_absent_v1"


def main(argv: list[str] | None = None) -> int:
    path = Path((argv or sys.argv)[1])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "not_applicable"
        or receipt.get("overlay_501_claimed") is not False
    ):
        raise SystemExit("packed union dedup receipt is unsupported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
