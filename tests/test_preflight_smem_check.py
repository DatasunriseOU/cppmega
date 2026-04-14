"""Unit tests for cppmega.megatron.preflight_smem_check.

These tests run on any host — no GPU, no tilelang required.  They
verify:

1.  The static AST scanner finds every ``@tilelang.jit(...)`` in our
    tracked kernel files and reports ``has_flag=True`` for all of
    them (regression: if anyone strips the aggressive-merge flag, this
    test fails).
2.  A forged kernel file WITHOUT the flag is correctly flagged.
3.  Compute-capability gating: sm_121 / sm_120 hard-fails, others warn.
4.  The CLI entrypoint returns non-zero on failure.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cppmega.megatron import preflight_smem_check as pf


# ---- fixtures --------------------------------------------------------


@pytest.fixture()
def bad_kernel_file(tmp_path: Path) -> Path:
    """A TileLang kernel file missing the aggressive-merge flag."""
    p = tmp_path / "fake_bad_kernel.py"
    p.write_text(
        textwrap.dedent(
            """
            import tilelang
            from tilelang import language as T

            @tilelang.jit(out_idx=[-1])
            def missing_all(H, D):
                @T.prim_func
                def k():
                    pass
                return k

            @tilelang.jit(
                out_idx=[-1],
                pass_configs={
                    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                },
            )
            def has_pc_but_no_flag(H, D):
                @T.prim_func
                def k():
                    pass
                return k
            """
        ).strip()
    )
    return p


@pytest.fixture()
def good_kernel_file(tmp_path: Path) -> Path:
    """A TileLang kernel file with the aggressive-merge flag."""
    p = tmp_path / "fake_good_kernel.py"
    p.write_text(
        textwrap.dedent(
            """
            import tilelang
            from tilelang import language as T

            @tilelang.jit(
                out_idx=[-1],
                pass_configs={
                    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
                },
            )
            def good_kernel(H, D):
                @T.prim_func
                def k():
                    pass
                return k

            # Module-level pass_configs dict:
            pass_configs = {
                tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
            }

            @tilelang.jit(pass_configs=pass_configs)
            def good_kernel_via_name(H, D):
                @T.prim_func
                def k():
                    pass
                return k
            """
        ).strip()
    )
    return p


# ---- core AST tests --------------------------------------------------


def test_scan_finds_missing_flag(bad_kernel_file: Path) -> None:
    sites = pf._scan_file(bad_kernel_file)
    assert len(sites) == 2
    by_name = {s.func_name: s for s in sites}
    assert by_name["missing_all"].has_pass_configs is False
    assert by_name["missing_all"].has_flag is False
    assert by_name["has_pc_but_no_flag"].has_pass_configs is True
    assert by_name["has_pc_but_no_flag"].has_flag is False


def test_scan_accepts_good_file(good_kernel_file: Path) -> None:
    sites = pf._scan_file(good_kernel_file)
    assert len(sites) == 2
    assert all(s.has_flag for s in sites), (
        f"expected every kernel to have the flag, got: {sites}"
    )


def test_scan_missing_file_returns_empty(tmp_path: Path) -> None:
    assert pf._scan_file(tmp_path / "nope.py") == []


# ---- compute-capability gating --------------------------------------


def test_gb10_hard_fails_on_missing_flag(
    monkeypatch: pytest.MonkeyPatch, bad_kernel_file: Path
) -> None:
    """On sm_121 (GB10), any missing flag is a RuntimeError."""
    monkeypatch.setattr(pf, "_TRACKED_KERNEL_FILES", (bad_kernel_file,))
    with pytest.raises(pf.SmemPreflightError, match=r"missing.*pass_configs"):
        pf.check(cc=(12, 1))


def test_h200_warns_but_passes_on_missing_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bad_kernel_file: Path,
) -> None:
    """On sm_90 (H200), missing flag is a warning, not a fatal error."""
    monkeypatch.setattr(pf, "_TRACKED_KERNEL_FILES", (bad_kernel_file,))
    sites = pf.check(cc=(9, 0))
    assert len(sites) == 2
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE" in captured.err


def test_strict_mode_hard_fails_everywhere(
    monkeypatch: pytest.MonkeyPatch, bad_kernel_file: Path
) -> None:
    """``strict=True`` upgrades warnings to errors on any device."""
    monkeypatch.setattr(pf, "_TRACKED_KERNEL_FILES", (bad_kernel_file,))
    with pytest.raises(pf.SmemPreflightError):
        pf.check(cc=(9, 0), strict=True)


def test_good_file_passes_on_gb10(
    monkeypatch: pytest.MonkeyPatch, good_kernel_file: Path
) -> None:
    monkeypatch.setattr(pf, "_TRACKED_KERNEL_FILES", (good_kernel_file,))
    sites = pf.check(cc=(12, 1))
    assert all(s.has_flag for s in sites)


def test_sm120_is_also_hard_fail(
    monkeypatch: pytest.MonkeyPatch, bad_kernel_file: Path
) -> None:
    """sm_120 (GB202 / Blackwell client) shares the 99 KiB cap."""
    monkeypatch.setattr(pf, "_TRACKED_KERNEL_FILES", (bad_kernel_file,))
    with pytest.raises(pf.SmemPreflightError):
        pf.check(cc=(12, 0))


# ---- smem-cap table --------------------------------------------------


@pytest.mark.parametrize(
    "cc,expected_kib",
    [
        ((9, 0), 228),   # H100/H200
        ((10, 0), 228),  # B200
        ((12, 1), 99),   # GB10
        ((12, 0), 99),   # sm_120
        ((8, 0), 163),   # A100
        ((99, 99), pf.DEFAULT_SMEM_CAP_KIB),  # unknown
    ],
)
def test_kib_cap_table(cc: tuple[int, int], expected_kib: int) -> None:
    assert pf._kib_cap_for(cc) == expected_kib


# ---- integration: shipped kernels must all pass --------------------


def test_shipped_kernels_all_have_flag() -> None:
    """Regression guard: every in-tree TileLang kernel has the flag.

    This is the real test — if anyone strips the aggressive-merge flag
    from any shipped kernel, this fails in CI.
    """
    sites = []
    for p in pf._TRACKED_KERNEL_FILES:
        if not p.exists():
            pytest.fail(f"Tracked kernel file disappeared: {p}")
        sites.extend(pf._scan_file(p))
    assert sites, "no kernel sites found — _TRACKED_KERNEL_FILES is stale"
    missing = [
        f"{s.file.name}:{s.lineno} {s.func_name}" for s in sites if not s.has_flag
    ]
    assert not missing, (
        "Shipped TileLang kernels missing "
        "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: "
        + ", ".join(missing)
    )


def test_shipped_kernels_pass_preflight_on_gb10() -> None:
    """End-to-end: running ``check`` against real tracked files on sm_121
    must succeed (no missing flags, no exceptions)."""
    # No monkeypatch — use the real _TRACKED_KERNEL_FILES list.
    sites = pf.check(cc=(12, 1))
    assert sites, "expected non-empty site list"
