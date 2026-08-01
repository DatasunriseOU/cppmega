"""Verify the 2026-08-01 docs sweep is reflected in indexes."""

from pathlib import Path


def test_status_index_lists_new_isolation_design_docs():
    status = Path("docs/status/README.md").read_text(encoding="utf-8")
    assert "../document_isolation_swa_design.md" in status
    assert "../document_isolation_cp128k_design.md" in status
    assert "../document_isolation_varlen_design.md" in status
    assert "Quarterly sweep" in status


def test_sessions_index_marks_april_2026_archived():
    sessions = Path("docs/sessions/README.md").read_text(encoding="utf-8")
    assert "## Earlier Session Notes" in sessions
    # All April 2026 rows in the earlier-notes table should be archived.
    in_section = False
    for line in sessions.splitlines():
        if "## Earlier Session Notes" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and "2026-04-" in line:
            assert "| archived |" in line, f"April note not archived: {line}"


def test_sweep_doc_exists_and_has_schedule():
    sweep = Path("docs/quarterly_docs_sweep_2026_08_01.md").read_text(
        encoding="utf-8"
    )
    assert "# Quarterly docs sweep — 2026-08-01" in sweep
    assert "Schedule: 2026-11-01" in sweep
