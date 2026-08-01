"""Archive guard for the Mamba3 MIMO P2 PsiV cache.

The PsiV cache scaffolding was removed in 2026-08-01 because Phase A could
not be scheduled on available GPU time while higher-priority work (packed-
document isolation, FA4 beta23, data release blockers) was in flight.
`docs/mamba3_mimo_p2_psiv_cache_design.md` contains the superseded design and
the archive rationale.

If Phase A is ever resurrected, start from the design doc §9 and re-create
the module; do not resurrect the deleted skeleton verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_psiv_cache_module_is_archived():
    """The scaffold module must not be importable after archival."""
    module_path = REPO_ROOT / "cppmega" / "megatron" / "mamba3_psiv_cache.py"
    patch_path = (
        REPO_ROOT
        / "cppmega"
        / "megatron"
        / "upstream_patches"
        / "apply_mamba3_p2_psiv_patches.py"
    )
    assert not module_path.exists(), (
        "mamba3_psiv_cache.py was archived; delete this test if P2 is revived"
    )
    assert not patch_path.exists(), (
        "apply_mamba3_p2_psiv_patches.py was archived; delete this test if P2 is revived"
    )
    with pytest.raises(ImportError):
        import cppmega.megatron.mamba3_psiv_cache  # noqa: F401


def test_psiv_cache_design_doc_has_archive_addendum():
    """The design doc must record why the work was archived."""
    doc_path = REPO_ROOT / "docs" / "mamba3_mimo_p2_psiv_cache_design.md"
    text = doc_path.read_text(encoding="utf-8")
    assert "## 15. Archive addendum" in text
    assert "superseded" in text.lower() or "archived" in text.lower()
    assert "2026-08-01" in text
