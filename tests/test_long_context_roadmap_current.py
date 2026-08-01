"""Verify long_context_roadmap.md reflects the latest SWA/CP decisions."""

from pathlib import Path


def test_long_context_roadmap_has_current_header_and_decisions():
    doc = Path("docs/long_context_roadmap.md").read_text(encoding="utf-8")
    assert "**Last updated**: 2026-08-01" in doc
    assert "SWA plumbing landed (P083)" in doc
    assert "CP 128k design landed (P084)" in doc
    assert "docs/document_isolation_swa_design.md" in doc
    assert "docs/document_isolation_cp128k_design.md" in doc
    assert "## Current blockers and owners" in doc
