"""Small RSS guard for long-running data-generation wrappers."""

from __future__ import annotations

import os
import resource
import sys
import threading
import time

_BYTES_PER_GIB = 1024**3


def max_rss_bytes() -> int:
    """Return max RSS for this process in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def check_memory_limit(limit_gb: float, *, label: str) -> None:
    """Fail fast if this process has crossed the configured RSS budget."""
    if limit_gb <= 0:
        return
    limit_bytes = int(limit_gb * _BYTES_PER_GIB)
    rss = max_rss_bytes()
    if rss <= limit_bytes:
        return
    print(
        f"ERROR: {label} exceeded memory limit: "
        f"max_rss={rss / _BYTES_PER_GIB:.2f} GiB "
        f"limit={limit_gb:.2f} GiB",
        file=sys.stderr,
        flush=True,
    )
    raise MemoryError(f"{label} exceeded memory limit")


def start_memory_guard(
    limit_gb: float,
    *,
    label: str,
    interval_seconds: float = 1.0,
) -> None:
    """Start a daemon watchdog that exits before runaway RSS continues."""
    if limit_gb <= 0:
        return
    limit_bytes = int(limit_gb * _BYTES_PER_GIB)

    def _watch() -> None:
        while True:
            rss = max_rss_bytes()
            if rss > limit_bytes:
                print(
                    f"ERROR: {label} exceeded memory limit: "
                    f"max_rss={rss / _BYTES_PER_GIB:.2f} GiB "
                    f"limit={limit_gb:.2f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(137)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_watch, name=f"{label}-memory-guard", daemon=True)
    thread.start()
