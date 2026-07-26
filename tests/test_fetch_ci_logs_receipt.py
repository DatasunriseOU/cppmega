from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts.fetch_ci_logs import (
    _canonical_sha256,
    _classify_job_log_process_result,
    _completion_receipt,
    _load_ci_source_records,
)


def _source_record(*, repo: str, job_id: int) -> dict:
    return {
        "repo": repo,
        "job_id": job_id,
        "run_id": 99,
        "job_name": "build",
        "diagnostics": [],
    }


def test_fetch_result_distinguishes_expired_from_transient_failure() -> None:
    fetched = _classify_job_log_process_result(
        "owner",
        "repo",
        7,
        subprocess.CompletedProcess([], 0, stdout="the log", stderr=""),
    )
    assert fetched.status == "fetched"
    assert fetched.text == "the log"

    expired = _classify_job_log_process_result(
        "owner",
        "repo",
        7,
        subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="gh: Server Error (HTTP 410)",
        ),
    )
    assert expired.status == "expired"
    assert "410" in expired.detail

    with pytest.raises(RuntimeError, match="HTTP 503"):
        _classify_job_log_process_result(
            "owner",
            "repo",
            7,
            subprocess.CompletedProcess(
                [],
                1,
                stdout="",
                stderr="gh: Service Unavailable (HTTP 503)",
            ),
        )


def test_ci_source_inventory_accounts_for_repo_alias_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ci.jsonl"
    first = _source_record(repo="new/project", job_id=7)
    alias = {**first, "repo": "old/project"}
    source.write_text(
        "\n".join(
            (
                json.dumps(first),
                json.dumps(alias),
                json.dumps({"workflow": "non-job summary"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    records, inventory, non_jobs, aliases = _load_ci_source_records(tmp_path)

    assert records == [first]
    assert len(inventory) == 1
    assert non_jobs == 1
    assert aliases == [
        {
            "job_id": 7,
            "canonical_repo": "new/project",
            "alias_repo": "old/project",
        }
    ]


def test_ci_completion_receipt_allows_only_exact_accounted_outcomes(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "ci.jsonl"
    state_path = tmp_path / "state.jsonl"
    document = {
        "repo": "owner/one",
        "ci_metadata": {"job_id": 7},
        "text": "build log",
    }
    output_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    states = {
        7: {
            "job_id": 7,
            "repo": "owner/one",
            "status": "fetched",
            "source_sha256": "a" * 64,
            "document_sha256": _canonical_sha256(document),
        },
        9: {
            "job_id": 9,
            "repo": "owner/two",
            "status": "expired",
            "source_sha256": "b" * 64,
            "detail": "HTTP 410",
        },
    }
    state_path.write_text(
        "".join(json.dumps(state) + "\n" for state in states.values()),
        encoding="utf-8",
    )
    records = [
        _source_record(repo="owner/one", job_id=7),
        _source_record(repo="owner/two", job_id=9),
    ]

    complete = _completion_receipt(
        source_inventory=[],
        records=records,
        source_row_count=2,
        non_job_records=0,
        aliases=[],
        output_path=output_path,
        output={7: document},
        state_path=state_path,
        states=states,
        errors=[],
        max_jobs=0,
    )
    assert complete["status"] == "complete"
    assert complete["fetched_count"] == 1
    assert complete["expired_count"] == 1
    assert complete["unresolved_count"] == 0

    incomplete = _completion_receipt(
        source_inventory=[],
        records=records,
        source_row_count=2,
        non_job_records=0,
        aliases=[],
        output_path=output_path,
        output={7: document},
        state_path=state_path,
        states={7: states[7]},
        errors=[],
        max_jobs=0,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["unresolved_jobs"] == [9]
