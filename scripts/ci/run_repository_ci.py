#!/usr/bin/env python3
"""Run cppmega CI directly on repository-owned machines."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.repository_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
