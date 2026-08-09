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


def _current_rss_darwin_task_info_bytes() -> int | None:
    """Current RSS via mach task_info (no subprocess fork).

    Under heavy macro-scan / memory pressure, forking ``ps`` can fail with
    ENOMEM or hit short subprocess timeouts. Darwin exposes live resident_size
    through task_info without allocating a child process.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        class _TimeValue(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int),
                ("microseconds", ctypes.c_int),
            ]

        class _MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", _TimeValue),
                ("system_time", _TimeValue),
                ("policy", ctypes.c_int),
                ("suspend_count", ctypes.c_int),
            ]

        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return None
        libc = ctypes.CDLL(libc_name, use_errno=True)
        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        task_info.restype = ctypes.c_int

        info = _MachTaskBasicInfo()
        count = ctypes.c_uint(
            ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint)
        )
        # MACH_TASK_BASIC_INFO == 20; KERN_SUCCESS == 0
        kr = task_info(mach_task_self(), 20, ctypes.byref(info), ctypes.byref(count))
        if kr != 0:
            return None
        rss = int(info.resident_size)
        return rss if rss > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _current_rss_ps_bytes() -> int | None:
    """Return current RSS from the portable Unix ``ps`` utility.

    Retries briefly: under load, a single 2s ``ps`` call can time out or fail
    to fork even when the process is healthy. Still returns None on total
    failure so the caller fails closed rather than inventing an RSS value.
    """
    if os.name == "nt":
        return None
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            if result.returncode != 0:
                last_err = None
                time.sleep(0.05 * (attempt + 1))
                continue
            lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            if not lines:
                time.sleep(0.05 * (attempt + 1))
                continue
            # ps reports RSS in KiB on macOS and the BSDs.
            rss_kib = int(lines[0].strip())
            if rss_kib > 0:
                return rss_kib * 1024
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
            last_err = exc
            time.sleep(0.05 * (attempt + 1))
    if last_err is not None and "ps" not in _WARNED_PROBE_FAILURES:
        # one-shot diagnostic; still return None (fail closed upstream)
        _WARNED_PROBE_FAILURES.add("ps")
        print(
            f"WARNING: RSS probe _current_rss_ps_bytes exhausted retries: "
            f"{type(last_err).__name__}: {last_err}",
            file=sys.stderr,
            flush=True,
        )
    return None


def _default_rss_probes() -> tuple[Callable[[], int | None], ...]:
    """Platform-ordered live RSS probes (no historical ru_maxrss)."""
    if sys.platform == "darwin":
        # Prefer no-fork Darwin task_info first: macro-scan can exhaust fork budget.
        return (
            _current_rss_darwin_task_info_bytes,
            _current_rss_psutil_bytes,
            _current_rss_ps_bytes,
            _current_rss_procfs_bytes,
        )
    return (
        _current_rss_procfs_bytes,
        _current_rss_psutil_bytes,
        _current_rss_ps_bytes,
        _current_rss_darwin_task_info_bytes,
    )


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
    active_probes = tuple(probes) if probes is not None else _default_rss_probes()
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
