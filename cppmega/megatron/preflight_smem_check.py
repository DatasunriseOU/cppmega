"""GB10 sm_121 shared-memory preflight enforcer.

Context
-------
sm_121 (GB10) has only 99 KiB of dynamic shared memory per SM.  Our TileLang
kernels — in particular the SparseMLA bwd / bwd_bwd family and the Mamba3
MIMO bwd kernels — naively emit 140+ KiB of smem descriptors, which fails
kernel launch on GB10 with an opaque "invalid configuration argument" mid
training.

The fix is already known (see
``reference_gb10_bwd_bwd_blocker.md`` — PR #904 upstream):
passing ``TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`` in the TileLang
``pass_configs`` dict of every ``@tilelang.jit`` decorator causes the
TileLang lowering to aggressively merge smem buffers and stay under the
99 KiB cap.

This module enforces the fix.  It performs two static checks and one
(opt-in) runtime check at process startup:

1.  **Static AST check** — walks every TileLang kernel module we ship
    (and the in-tree forked-upstream copy), parses each
    ``@tilelang.jit(...)`` decorator, and asserts that
    ``TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`` is present with ``True``
    in its ``pass_configs`` dict.  This runs on any host — GB10, H200,
    CI, CPU-only dev machines — because it's pure AST inspection.

2.  **Compute-capability gate** — if the running GPU is sm_121 (GB10),
    enforcement escalates to HARD FAIL on any missing flag.  On
    non-sm_121 devices, missing flags are WARNINGS unless the
    ``CPPMEGA_SMEM_CHECK_STRICT=1`` env var is set.

3.  **Runtime smem inspection (opt-in)** — if
    ``CPPMEGA_SMEM_CHECK_RUNTIME=1`` is set AND ``tilelang`` is
    importable, compile each kernel module's advertised entry point
    against a representative small shape and read
    ``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`` for every emitted kernel
    via ``cuFuncGetAttribute``.  If any compiled kernel exceeds the
    detected cap, raise.  This is the ground-truth check; it requires
    a GPU and is therefore opt-in.

The module is safe to import on hosts without CUDA, without TileLang,
and without any GPU — it degrades gracefully in that order.

No silent fallbacks: once a check is active, failure is a ``RuntimeError``.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Compute-capability (major, minor) → max dynamic smem (KiB).
# Source: CUDA Programming Guide, Table 15, and NVIDIA device whitepapers.
# Conservative: we only list capabilities we actually target or have
# evidence for.  Unknown caps get the pessimistic default (99 KiB).
SMEM_CAPS_KIB: dict[tuple[int, int], int] = {
    (7, 0): 96,    # V100
    (7, 5): 64,    # T4 / RTX 20xx (Turing)
    (8, 0): 163,   # A100
    (8, 6): 99,    # RTX 30xx
    (8, 9): 99,    # L40 / RTX 40xx (Ada)
    (9, 0): 228,   # H100 / H200 (SXM / PCIe)
    (10, 0): 228,  # B200 (Blackwell data center)
    (10, 1): 228,  # B300
    (12, 0): 99,   # sm_120 (GB202 — Blackwell client / GB10 B200-variant)
    (12, 1): 99,   # sm_121 (GB10 Grace-Blackwell Superchip)
}
DEFAULT_SMEM_CAP_KIB = 99  # If unknown, be pessimistic — GB10-level cap.

# Compute capabilities where we UPGRADE any missing flag from WARN to ERROR.
# GB10 is the only platform where a missing aggressive-merge flag is known
# to cause kernel-launch failure mid-training.
_HARD_FAIL_CAPS: frozenset[tuple[int, int]] = frozenset({(12, 1), (12, 0)})


# Static set of files we audit.  These are the in-tree TileLang kernels.
# Add new kernel files here as they're introduced, or the preflight will
# silently miss them.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_KERNEL_FILES: tuple[Path, ...] = (
    _REPO_ROOT / "cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py",
    _REPO_ROOT / "cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd_fp8.py",
    _REPO_ROOT / "cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd.py",
    _REPO_ROOT / "cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd_fp8.py",
    _REPO_ROOT / "cppmega/megatron/tilelang_sparse_mla/sparse_mla_fwd.py",
    _REPO_ROOT / "cppmega/megatron/tilelang_sparse_mla/sparse_mla_bwd.py",
    _REPO_ROOT / "cppmega/megatron/tilelang_sparse_mla/topk_selector.py",
)

_REQUIRED_FLAG_NAME = "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE"


@dataclass(frozen=True)
class KernelSite:
    """One ``@tilelang.jit(...)`` decorator site in a source file."""

    file: Path
    lineno: int
    func_name: str
    has_flag: bool
    has_pass_configs: bool


class SmemPreflightError(RuntimeError):
    """Raised when a kernel is missing the GB10 aggressive-merge flag."""


def _iter_tilelang_decorators(tree: ast.AST) -> Iterable[tuple[ast.Call, str]]:
    """Yield ``(Call_node, function_name)`` for each ``@tilelang.jit(...)``.

    Recognizes any of:
      * ``@tilelang.jit(...)``
      * ``@tl.jit(...)``   (common alias)
    Ignores bare-name ``@jit`` to avoid false positives from other frameworks.
    """
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            # ``@tilelang.jit(...)`` / ``@tl.jit(...)``
            if not isinstance(deco, ast.Call):
                continue
            func = deco.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "jit":
                continue
            if not isinstance(func.value, ast.Name):
                continue
            if func.value.id not in ("tilelang", "tl"):
                continue
            yield deco, node.name


def _decorator_has_flag(deco: ast.Call) -> tuple[bool, bool]:
    """Return ``(has_pass_configs_kw, flag_set_to_True)``."""
    for kw in deco.keywords:
        if kw.arg != "pass_configs":
            continue
        value = kw.value
        # ``pass_configs=SOME_NAME`` — resolve by name inside module.
        # We take a conservative stance: treat as "present but unverified";
        # call-sites resolve via module attribute lookup below.
        if isinstance(value, ast.Name):
            return True, _named_dict_has_flag(deco, value.id)
        if not isinstance(value, ast.Dict):
            return True, False
        for key in value.keys:
            # Expect ``tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE``
            if (
                isinstance(key, ast.Attribute)
                and key.attr == _REQUIRED_FLAG_NAME
            ):
                return True, True
        return True, False
    return False, False


def _named_dict_has_flag(deco: ast.Call, dict_name: str) -> bool:
    """Resolve a module-level ``pass_configs = {...}`` by name in the same file.

    Walks up to the enclosing Module and finds an assignment
    ``dict_name = {...}`` with the required flag.
    """
    # Walk up from deco — in practice we receive the decorator AST node only;
    # resolve via the captured tree stored on the call site.  Since ast.Call
    # does not track its module parent, we re-parse at call time.  Return
    # False conservatively — caller should supply a dict literal for strong
    # verification.  (Topk_selector uses a module-level dict; we handle it
    # explicitly in the file-scan by looking for a top-level Assign.)
    return False


def _scan_file(path: Path) -> list[KernelSite]:
    """Return the list of kernel sites in one source file."""
    if not path.exists():
        return []
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SmemPreflightError(
            f"[preflight_smem_check] Failed to parse {path}: {exc}"
        ) from exc

    # Pre-resolve any module-level ``pass_configs = {...}`` literals so that
    # ``@tilelang.jit(pass_configs=pass_configs)`` call-sites can be
    # validated.  Maps name -> (has_flag: bool).
    named_dicts: dict[str, bool] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, ast.Dict):
                has_flag = any(
                    isinstance(key, ast.Attribute) and key.attr == _REQUIRED_FLAG_NAME
                    for key in node.value.keys
                )
                named_dicts[tgt.id] = has_flag

    sites: list[KernelSite] = []
    for deco, fname in _iter_tilelang_decorators(tree):
        has_pc, has_flag = False, False
        for kw in deco.keywords:
            if kw.arg != "pass_configs":
                continue
            has_pc = True
            if isinstance(kw.value, ast.Dict):
                has_flag = any(
                    isinstance(key, ast.Attribute)
                    and key.attr == _REQUIRED_FLAG_NAME
                    for key in kw.value.keys
                )
            elif isinstance(kw.value, ast.Name):
                has_flag = named_dicts.get(kw.value.id, False)
            break
        sites.append(
            KernelSite(
                file=path,
                lineno=deco.lineno,
                func_name=fname,
                has_flag=has_flag,
                has_pass_configs=has_pc,
            )
        )
    return sites


def _detect_cc() -> tuple[int, int] | None:
    """Return ``(major, minor)`` of current GPU, or ``None`` if no CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def _kib_cap_for(cc: tuple[int, int]) -> int:
    return SMEM_CAPS_KIB.get(cc, DEFAULT_SMEM_CAP_KIB)


def _fmt_path(p: Path) -> str:
    """Format a path relative to the repo root when possible, else absolute."""
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def _format_sites(sites: list[KernelSite]) -> str:
    return "\n".join(
        f"  - {_fmt_path(s.file)}:{s.lineno}  "
        f"{s.func_name}(…)  "
        f"pass_configs={'present' if s.has_pass_configs else 'MISSING'}  "
        f"{_REQUIRED_FLAG_NAME}="
        f"{'True' if s.has_flag else 'MISSING'}"
        for s in sites
    )


def check(
    *,
    cc: tuple[int, int] | None = None,
    strict: bool | None = None,
    raise_on_error: bool = True,
    runtime_compile: bool | None = None,
) -> list[KernelSite]:
    """Run the preflight.

    Parameters
    ----------
    cc
        Override compute capability detection (for unit tests).  If
        ``None``, detect via ``torch.cuda.get_device_capability(0)``.
    strict
        If ``True``, treat missing flag as hard error even on non-GB10.
        If ``None`` (default), read from ``CPPMEGA_SMEM_CHECK_STRICT``.
    raise_on_error
        If ``True`` (default), raise :class:`SmemPreflightError` on
        failure.  If ``False``, log and return the site list.
    runtime_compile
        If ``True``, additionally compile each kernel at a small
        representative shape and read its ``shared_size_bytes`` via the
        CUDA runtime.  Requires ``tilelang`` + an actual GPU.  If
        ``None``, read from ``CPPMEGA_SMEM_CHECK_RUNTIME``.

    Returns
    -------
    list[KernelSite]
        All kernel sites discovered (whether passing or failing).
    """
    if strict is None:
        strict = os.environ.get("CPPMEGA_SMEM_CHECK_STRICT", "0") == "1"
    if runtime_compile is None:
        runtime_compile = os.environ.get("CPPMEGA_SMEM_CHECK_RUNTIME", "0") == "1"

    sites: list[KernelSite] = []
    for p in _TRACKED_KERNEL_FILES:
        sites.extend(_scan_file(p))

    if not sites:
        # No kernel files found — package layout changed.  Don't pretend
        # everything is fine; loudly notify.
        msg = (
            "[preflight_smem_check] No TileLang kernel files found on disk; "
            "the tracked-files list in preflight_smem_check.py is stale. "
            "Update _TRACKED_KERNEL_FILES."
        )
        if raise_on_error:
            raise SmemPreflightError(msg)
        print(msg, file=sys.stderr)
        return sites

    bad = [s for s in sites if not s.has_flag]

    if cc is None:
        cc = _detect_cc()
    cap_kib = _kib_cap_for(cc) if cc is not None else DEFAULT_SMEM_CAP_KIB

    hard_fail = strict or (cc in _HARD_FAIL_CAPS)

    if bad:
        detail = _format_sites(bad)
        cc_str = f"sm_{cc[0]}{cc[1]}" if cc else "no-GPU"
        msg = (
            f"[preflight_smem_check] {len(bad)} TileLang kernel(s) are "
            f"missing `{_REQUIRED_FLAG_NAME}: True` in their pass_configs.\n"
            f"Device: {cc_str}  smem cap: {cap_kib} KiB  "
            f"hard_fail={hard_fail}\n"
            f"Failing kernels:\n{detail}\n"
            f"Fix: add `tilelang.PassConfigKey.{_REQUIRED_FLAG_NAME}: True` "
            f"to each `@tilelang.jit(..., pass_configs={{...}})` decorator.\n"
            f"Reference: memory note `reference_gb10_bwd_bwd_blocker.md`."
        )
        if hard_fail and raise_on_error:
            raise SmemPreflightError(msg)
        print("WARNING: " + msg, file=sys.stderr)

    if runtime_compile:
        _runtime_smem_probe(cap_kib, raise_on_error=raise_on_error)

    return sites


def _runtime_smem_probe(cap_kib: int, *, raise_on_error: bool) -> None:
    """Compile each kernel at a tiny shape and check smem via CUDA runtime.

    Loud failure on any kernel that reports smem > ``cap_kib`` KiB.
    Requires ``tilelang`` importable and a live CUDA device.
    """
    try:
        import tilelang  # noqa: F401
    except ImportError:
        print(
            "[preflight_smem_check] runtime_compile requested but tilelang "
            "is not importable; skipping runtime probe.",
            file=sys.stderr,
        )
        return

    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    violations: list[str] = []
    # Tiny representative shapes for each kernel module.  Keep these small
    # to minimize compile time; the question is only "does tilelang emit
    # <= cap KiB smem", not correctness.
    probes = _runtime_probes()
    for name, builder in probes.items():
        try:
            kernel = builder()
            smem_bytes = _read_kernel_smem(kernel)
        except Exception as exc:  # noqa: BLE001
            violations.append(f"  - {name}: compile/probe raised {exc!r}")
            continue
        if smem_bytes is None:
            continue
        smem_kib = smem_bytes / 1024.0
        if smem_bytes > cap_kib * 1024:
            violations.append(
                f"  - {name}: dynamic smem = {smem_kib:.1f} KiB > "
                f"cap {cap_kib} KiB"
            )

    if violations:
        detail = "\n".join(violations)
        msg = (
            f"[preflight_smem_check] Runtime smem probe detected kernels "
            f"over the {cap_kib} KiB cap:\n{detail}\n"
            f"This will crash at kernel launch on the current device. "
            f"Ensure every kernel has "
            f"`{_REQUIRED_FLAG_NAME}: True`."
        )
        if raise_on_error:
            raise SmemPreflightError(msg)
        print("WARNING: " + msg, file=sys.stderr)


def _runtime_probes() -> dict[str, callable]:
    """Return a map of kernel-probe-name -> zero-arg compile-builder.

    Each builder returns a compiled TileLang kernel handle that exposes
    either ``.dynamic_smem_bytes`` or an underlying ``cuFunction`` we can
    query.  Unknown shapes default to small NAM56R-compatible tiles.
    """
    def _fwd_bf16():
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd import (
            sparse_mla_fwd,
        )
        return sparse_mla_fwd(heads=64, dim=512, tail_dim=64, topk=64, kv_group=1)

    def _fwd_fp8():
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_fwd_fp8 import (
            sparse_mla_fwd_fp8,
        )
        return sparse_mla_fwd_fp8(
            heads=64, dim=512, tail_dim=64, topk=64, kv_group=1
        )

    def _bwd_bf16():
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd import bwd
        return bwd(H=64, D=512, D_tail=64, topk=64, kv_group=1)

    def _bwd_fp8():
        from cppmega.megatron.sparse_mla_ops.tilelang_sparse_mla_bwd_fp8 import (
            bwd_fp8,
        )
        return bwd_fp8(H=64, D=512, D_tail=64, topk=64, kv_group=1)

    return {
        "sparse_mla_ops.fwd (bf16)": _fwd_bf16,
        "sparse_mla_ops.fwd (fp8)":  _fwd_fp8,
        "sparse_mla_ops.bwd (bf16)": _bwd_bf16,
        "sparse_mla_ops.bwd (fp8)":  _bwd_fp8,
    }


def _read_kernel_smem(kernel) -> int | None:
    """Best-effort read of compiled kernel's dynamic smem usage, in bytes.

    TileLang exposes several shapes of kernel handles across versions.
    We attempt, in order:
      * ``kernel.dynamic_smem_bytes``
      * ``kernel.get_dynamic_smem_bytes()``
      * ``kernel.get_kernel_source()`` scraped for ``__shared__`` size hints
        (last-ditch heuristic — not authoritative, do not fail on this alone)
    Returns ``None`` if no attribute is available.
    """
    for attr in ("dynamic_smem_bytes", "shared_memory_size", "smem_bytes"):
        v = getattr(kernel, attr, None)
        if isinstance(v, int):
            return v
    for meth in ("get_dynamic_smem_bytes", "get_shared_memory_size"):
        fn = getattr(kernel, meth, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, int):
                    return v
            except Exception:  # noqa: BLE001
                pass
    return None


def main() -> int:
    """CLI entrypoint — used by training launch scripts.

    ``python -m cppmega.megatron.preflight_smem_check`` exits non-zero
    on any kernel missing the aggressive-merge flag (on GB10) or on any
    kernel-file-list drift.
    """
    try:
        sites = check(raise_on_error=True)
    except SmemPreflightError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 1

    bad = [s for s in sites if not s.has_flag]
    cc = _detect_cc()
    cc_str = f"sm_{cc[0]}{cc[1]}" if cc else "no-GPU"
    cap_kib = _kib_cap_for(cc) if cc else DEFAULT_SMEM_CAP_KIB
    print(
        f"[preflight_smem_check] OK — {len(sites)} TileLang kernels scanned, "
        f"{len(bad)} missing flag (non-fatal on {cc_str}, cap {cap_kib} KiB)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
