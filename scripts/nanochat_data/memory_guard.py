"""Small RSS guard for long-running data-generation wrappers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import os
import resource
import subprocess
import sys
import threading
import time

_BYTES_PER_GIB = 1024**3
_WARNED_PROBE_FAILURES: set[str] = set()


def max_rss_bytes() -> int:
    """Return max RSS for this process in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def _current_rss_procfs_bytes() -> int | None:
    """Return current RSS from Linux procfs when it is available."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            return None
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if resident_pages <= 0 or page_size <= 0:
            return None
        return resident_pages * page_size
    except (FileNotFoundError, OSError, ValueError):
        return None


def _current_rss_psutil_bytes() -> int | None:
    """Use psutil when available on platforms without a native probe."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        return rss if rss > 0 else None
    except (psutil.Error, OSError, ValueError):
        return None


def _current_rss_ps_bytes() -> int | None:
    """Return current RSS from the portable Unix ``ps`` utility."""
    if os.name == "nt":
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if result.returncode != 0:
            return None
        # ps reports RSS in KiB on macOS and the BSDs.
        rss_kib = int(result.stdout.strip().splitlines()[0])
        return rss_kib * 1024 if rss_kib > 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def current_rss_bytes(
    *,
    probes: Iterable[Callable[[], int | None]] | None = None,
) -> int:
    """Return current resident memory, not the process high-water mark.

    Long-lived data workers routinely release memory between documents. The
    ``resource.ru_maxrss`` value is monotonic for the lifetime of the process,
    so using it as a live admission check makes every later document inherit
    an earlier transient peak. Require a live current-RSS probe instead of
    silently falling back to the historical high-water value.
    """
    active_probes = tuple(probes) if probes is not None else (
        _current_rss_procfs_bytes,
        _current_rss_psutil_bytes,
        _current_rss_ps_bytes,
    )
    for probe in active_probes:
        probe_name = getattr(probe, "__name__", repr(probe))
        try:
            rss = probe()
        except Exception as exc:
            if probe_name not in _WARNED_PROBE_FAILURES:
                _WARNED_PROBE_FAILURES.add(probe_name)
                print(
                    f"WARNING: RSS probe {probe_name} failed: "
                    f"{type(exc).__name__}: {exc}; trying next probe",
                    file=sys.stderr,
                    flush=True,
                )
            rss = None
        if rss is not None:
            return rss
    raise RuntimeError(
        "current RSS is unavailable from all configured probes: "
        + ", ".join(
            getattr(probe, "__name__", repr(probe)) for probe in active_probes
        )
    )


def check_memory_limit(
    limit_gb: float,
    *,
    label: str,
    rss_reader: Callable[[], int] | None = None,
) -> None:
    """Fail fast if this process has crossed the configured RSS budget."""
    if limit_gb <= 0:
        return
    limit_bytes = int(limit_gb * _BYTES_PER_GIB)
    rss = (current_rss_bytes if rss_reader is None else rss_reader)()
    if rss <= limit_bytes:
        return
    print(
        f"ERROR: {label} exceeded memory limit: "
        f"rss={rss / _BYTES_PER_GIB:.2f} GiB "
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
    rss_reader: Callable[[], int] | None = None,
) -> None:
    """Start a daemon watchdog that exits before runaway RSS continues."""
    if limit_gb <= 0:
        return
    limit_bytes = int(limit_gb * _BYTES_PER_GIB)
    read_rss = current_rss_bytes if rss_reader is None else rss_reader

    def _watch() -> None:
        while True:
            try:
                rss = read_rss()
            except Exception as exc:
                print(
                    f"ERROR: {label} RSS probe failed closed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(137)
            if rss > limit_bytes:
                print(
                    f"ERROR: {label} exceeded memory limit: "
                    f"rss={rss / _BYTES_PER_GIB:.2f} GiB "
                    f"limit={limit_gb:.2f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(137)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_watch, name=f"{label}-memory-guard", daemon=True)
    thread.start()
