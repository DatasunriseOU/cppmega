from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import urllib.parse
import zlib

import pytest

from scripts import ci_stream_inventory as ci


START = "2026-01-01T00:00:00Z"
END = "2026-01-01T00:00:04Z"


def _write_repo_list(path: Path, names: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "repo_names": names,
                "repos": [],
                "unresolved": [],
            },
            indent=2,
        )
    )
    return path


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(
    run_id: int,
    created_epoch: int,
    *,
    attempt: int = 1,
    status: str = "completed",
    conclusion: str | None = "success",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "run_attempt": attempt,
        "created_at": _iso(created_epoch),
        "updated_at": _iso(created_epoch + 1),
        "run_started_at": _iso(created_epoch),
        "status": status,
        "conclusion": conclusion,
        "workflow_id": 700 + run_id % 3,
        "name": f"workflow-{run_id % 3}",
        "event": "push",
        "head_branch": "main",
        "head_sha": f"{run_id:040x}"[-40:],
        "run_number": run_id,
        "html_url": f"https://github.com/Owner/Repo/actions/runs/{run_id}",
        "url": f"https://api.github.com/repos/Owner/Repo/actions/runs/{run_id}",
        "actor": {"login": f"actor-{run_id % 5}"},
        "pull_requests": [{"number": run_id % 17}],
        "future_field": {"must": ["survive", "compression"]},
    }


class DatasetAPI:
    def __init__(self, runs: list[dict[str, Any]]):
        self.runs = list(runs)
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _query(url: str) -> tuple[int, int, int, int]:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        created = query["created"][0]
        start_text, inclusive_end_text = created.split("..", 1)
        start = ci.parse_utc_instant(start_text)
        inclusive_end = ci.parse_utc_instant(inclusive_end_text)
        page = int(query["page"][0])
        per_page = int(query["per_page"][0])
        return start, inclusive_end + 1, page, per_page

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        assert method == "GET"
        assert timeout > 0
        assert headers["Authorization"].startswith("Bearer ")
        start, end, page, per_page = self._query(url)
        selected = sorted(
            (
                run
                for run in self.runs
                if start <= ci.parse_utc_instant(str(run["created_at"])) < end
            ),
            key=lambda run: (str(run["created_at"]), int(run["id"])),
        )
        offset = (page - 1) * per_page
        page_runs = selected[offset : offset + per_page]
        self.calls.append(
            {
                "start": start,
                "end": end,
                "page": page,
                "per_page": per_page,
                "authorization": headers["Authorization"],
            }
        )
        return ci.HTTPResponse(
            status=200,
            headers={"X-RateLimit-Remaining": "4999"},
            body=json.dumps(
                {"total_count": len(selected), "workflow_runs": page_runs}
            ).encode(),
        )


def _scope(tmp_path: Path, names: list[str] | None = None, **kwargs: Any) -> ci.RepoScope:
    repo_list = _write_repo_list(
        tmp_path / "repo_list.json", names or ["Owner/Repo"]
    )
    return ci.load_repo_scope(repo_list, **kwargs)


def _inventory(
    tmp_path: Path,
    api: Any,
    *,
    scope: ci.RepoScope | None = None,
    resume: bool = False,
    max_attempts: int = 3,
    db_name: str = "inventory.sqlite",
    start: str = START,
    end: str = END,
    allow_script_upgrade_from_sha256: str | None = None,
    script_upgrade_reason: str | None = None,
    script_path: Path | None = None,
) -> ci.GitHubActionsInventory:
    return ci.GitHubActionsInventory(
        db_path=tmp_path / db_name,
        scope=scope or _scope(tmp_path),
        start=start,
        end=end,
        tokens=["secret-one", "secret-two"],
        resume=resume,
        allow_script_upgrade_from_sha256=(
            allow_script_upgrade_from_sha256
        ),
        script_upgrade_reason=script_upgrade_reason,
        progress_path=tmp_path / f"{db_name}.progress.json",
        requester=api,
        sleeper=lambda _: None,
        max_attempts=max_attempts,
        script_path=script_path or ci.__file__,
    )


def test_repo_scope_deduplicates_github_identity_and_requires_smoke_limit(
    tmp_path: Path,
) -> None:
    path = _write_repo_list(
        tmp_path / "repo_list.json",
        [
            "Owner/Repo",
            "owner/repo",
            "https://github.com/Second/Thing.git",
            "git@github.com:SECOND/thing.git",
            "github.com/Third/Bare.git",
        ],
    )
    scope = ci.load_repo_scope(path)
    assert [repo.key for repo in scope.repos] == [
        "owner/repo",
        "second/thing",
        "third/bare",
    ]
    with pytest.raises(ci.ScopeError, match="only with explicit --smoke"):
        ci.load_repo_scope(path, max_repos=1)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/github.com/Owner/Repo",
        "github.com.evil.invalid/Owner/Repo",
        "github.com@evil.invalid/Owner/Repo",
    ],
)
def test_repo_scope_rejects_spoofed_github_hosts(
    tmp_path: Path, value: str
) -> None:
    path = _write_repo_list(tmp_path / "repo_list.json", [value])
    with pytest.raises(ci.ScopeError, match="GitHub"):
        ci.load_repo_scope(path)


def test_paginates_and_preserves_full_compressed_metadata_and_attempts(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    runs = [
        _run(
            run_id,
            start + 1,
            attempt=1 + run_id % 2,
            status="in_progress" if run_id == 205 else "completed",
            conclusion=None if run_id == 205 else "success",
        )
        for run_id in range(1, 206)
    ]
    api = DatasetAPI(runs)
    inventory = _inventory(tmp_path, api)

    progress = inventory.run()
    receipt = inventory.write_completion_receipt(tmp_path / "receipt.json")

    assert [call["page"] for call in api.calls] == [1, 2, 3]
    assert progress["runs"] == 205
    assert receipt["enumeration_complete"] is True
    assert receipt["source_snapshot_stable"] is True
    assert receipt["production_complete"] is True
    assert receipt["source_count_drift"]["windows"] == 0
    assert receipt["run_count"] == 205
    assert receipt["metadata_encoding"] == ci.METADATA_ENCODING
    final_progress = json.loads(
        (tmp_path / "inventory.sqlite.progress.json").read_text()
    )
    assert final_progress["runs"] == 205
    assert final_progress["repos_closed"] == 1

    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT item_count FROM window_pages ORDER BY page_no"
        ).fetchall() == [(100,), (100,), (5,)]
        row = conn.execute(
            """
            SELECT metadata_blob,run_attempt,status,conclusion
            FROM runs WHERE run_id=205
            """
        ).fetchone()
        assert row is not None
        restored = json.loads(zlib.decompress(row[0]))
        assert restored == runs[-1]
        assert row[1:] == (2, "in_progress", None)


def test_recursively_splits_over_1000_and_assigns_boundary_to_right_window(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    # The exact midpoint is start+2.  A [start,end) split must put every
    # midpoint run in the right child and must not duplicate it.
    runs = [
        *[_run(run_id, start + 1) for run_id in range(1, 601)],
        *[_run(run_id, start + 2) for run_id in range(601, 1202)],
    ]
    inventory = _inventory(tmp_path, DatasetAPI(runs))

    inventory.run()
    receipt = inventory.write_completion_receipt(tmp_path / "receipt.json")

    assert receipt["run_count"] == 1201
    assert receipt["leaf_window_count"] == 2
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        windows = conn.execute(
            """
            SELECT start_epoch,end_epoch,status,expected_total
            FROM search_windows ORDER BY depth,start_epoch
            """
        ).fetchall()
        assert [row["status"] for row in windows] == ["split", "done", "done"]
        assert [row["expected_total"] for row in windows] == [1201, 600, 601]
        boundary_window = conn.execute(
            """
            SELECT w.start_epoch,w.end_epoch
            FROM window_runs wr
            JOIN search_windows w ON w.id=wr.window_id
            WHERE wr.run_id=601
            """
        ).fetchone()
        assert tuple(boundary_window) == (start + 2, start + 4)


class DuplicatePageAPI(DatasetAPI):
    def __init__(
        self, runs: list[dict[str, Any]], *, duplicate_responses: int = 1
    ):
        super().__init__(runs)
        self.duplicate_responses = duplicate_responses

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        response = super().__call__(method, url, headers, timeout)
        _, _, page, _ = self._query(url)
        if page == 2 and self.duplicate_responses > 0:
            self.duplicate_responses -= 1
            payload = json.loads(response.body)
            payload["workflow_runs"] = [self.runs[99]]
            return ci.HTTPResponse(
                status=200, headers=response.headers, body=json.dumps(payload).encode()
            )
        return response


class CrossWindowDuplicateAPI(DatasetAPI):
    def __init__(self, start: int):
        self.start = start
        self.left = [_run(run_id, start + 1) for run_id in range(1, 501)]
        self.right = [_run(run_id, start + 2) for run_id in range(501, 1001)]
        # Same complete metadata as a left-window run.  Returning it for the
        # right child is an overlapping enumeration and must fail closed.
        self.right.insert(0, self.left[-1])
        super().__init__([*self.left, *self.right[1:]])

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        start, end, page, per_page = self._query(url)
        if start == self.start and end == self.start + 4:
            selected = [*self.left, *self.right[1:], _run(1001, self.start + 3)]
        elif end == self.start + 2:
            selected = self.left
        else:
            selected = self.right
        offset = (page - 1) * per_page
        return ci.HTTPResponse(
            status=200,
            headers={"X-RateLimit-Remaining": "4999"},
            body=json.dumps(
                {
                    "total_count": len(selected),
                    "workflow_runs": selected[offset : offset + per_page],
                }
            ).encode(),
        )


class SplitCountDriftAPI(DatasetAPI):
    """Return one contradictory parent count, then stable child counts."""

    def __init__(self, start: int):
        runs = [
            *[_run(run_id, start + 1) for run_id in range(1, 51)],
            *[_run(run_id, start + 2) for run_id in range(51, 101)],
        ]
        super().__init__(runs)
        self.start = start

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        start, end, page, per_page = self._query(url)
        if start != self.start or end != self.start + 4:
            return super().__call__(method, url, headers, timeout)
        assert method == "GET"
        assert timeout > 0
        assert headers["Authorization"].startswith("Bearer ")
        page_runs = self.runs if page == 1 else [self.runs[-1]]
        return ci.HTTPResponse(
            status=200,
            headers={"X-RateLimit-Remaining": "4999"},
            body=json.dumps(
                {
                    "total_count": 101,
                    "workflow_runs": page_runs[:per_page],
                }
            ).encode(),
        )


def test_cross_page_duplicate_invalidates_leaf_and_recovers_by_split(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    page_inventory = _inventory(
        tmp_path,
        DuplicatePageAPI([_run(run_id, start + 1) for run_id in range(1, 102)]),
    )
    page_inventory.run()
    receipt = page_inventory.write_completion_receipt(tmp_path / "receipt.json")
    assert receipt["run_count"] == 101
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            """
            SELECT COUNT(*) FROM request_ledger
            WHERE outcome='pagination_drift_split'
            """
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM search_windows WHERE status='split'"
        ).fetchone()[0] == 1


def test_split_parent_count_drift_is_explicit_and_not_production_complete(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    inventory = _inventory(tmp_path, SplitCountDriftAPI(start))

    inventory.run()
    receipt = inventory.write_completion_receipt(tmp_path / "receipt.json")

    drift_line = (
        f"S\towner/repo\t{start}\t{start + 4}\t101\t100\t1"
    )
    assert receipt["schema"] == ci.RECEIPT_SCHEMA
    assert receipt["enumeration_complete"] is True
    assert receipt["source_snapshot_stable"] is False
    assert receipt["production_complete"] is False
    assert receipt["run_count"] == 100
    assert receipt["source_count_drift"] == {
        "windows": 1,
        "parent_total": 101,
        "child_total": 100,
        "net_parent_minus_children": 1,
        "absolute_delta": 1,
        "sha256": ci._hash_lines([drift_line]),
        "semantics": (
            "GitHub total_count observations at each split parent versus "
            "its later child enumeration; nonzero means the source "
            "cardinality changed or pagination contradicted itself during "
            "inventory"
        ),
    }


def test_cross_window_overlap_still_fails_closed(tmp_path: Path) -> None:
    start = ci.parse_utc_instant(START)
    cross_window_dir = tmp_path / "window"
    cross_window_dir.mkdir()
    window_inventory = _inventory(
        cross_window_dir, CrossWindowDuplicateAPI(start)
    )
    with pytest.raises(ci.InventoryError, match="outside"):
        window_inventory.run()
    with pytest.raises(ci.CompletionError, match="open/failed"):
        window_inventory.db.completion_receipt()


def test_one_second_drift_requires_two_stable_complete_sets(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    api = DuplicatePageAPI(
        [_run(run_id, start) for run_id in range(1, 102)],
        duplicate_responses=1,
    )
    inventory = _inventory(
        tmp_path,
        api,
        end=_iso(start + 1),
    )
    inventory.run()
    receipt = inventory.write_completion_receipt(tmp_path / "receipt.json")

    assert receipt["run_count"] == 101
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            """
            SELECT COUNT(*) FROM request_ledger
            WHERE outcome='pagination_drift_converge'
            """
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM window_convergence"
        ).fetchone()[0] == 0
    # Ordinary failed pass (2 requests) plus two complete proof passes
    # (2 requests each).
    assert len(api.calls) == 6


class UnstableTiePaginationAPI:
    def __init__(
        self,
        start: int,
        patterns: list[list[int]],
    ):
        self.start = start
        self.patterns = patterns
        self.pass_index = -1
        self.calls: list[tuple[int, int]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        assert method == "GET"
        start, end, page, per_page = DatasetAPI._query(url)
        assert (start, end, per_page) == (
            self.start,
            self.start + 1,
            ci.DEFAULT_PER_PAGE,
        )
        if page == 1:
            self.pass_index += 1
        if self.pass_index >= len(self.patterns):
            raise AssertionError("unexpected extra pagination pass")
        unique_ids = self.patterns[self.pass_index]
        assert len(unique_ids) == 100
        page_ids = unique_ids if page == 1 else [unique_ids[-1]]
        self.calls.append((self.pass_index, page))
        return ci.HTTPResponse(
            status=200,
            headers={"X-RateLimit-Remaining": "4999"},
            body=json.dumps(
                {
                    "total_count": 101,
                    "workflow_runs": [
                        _run(run_id, self.start) for run_id in page_ids
                    ],
                }
            ).encode(),
        )


class FailOnCallAPI:
    def __init__(self, delegate: Any, call_no: int):
        self.delegate = delegate
        self.call_no = call_no
        self.calls = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        self.calls += 1
        if self.calls == self.call_no:
            return ci.HTTPResponse(
                status=503,
                headers={},
                body=b'{"message":"temporary proof interruption"}',
            )
        return self.delegate(method, url, headers, timeout)


def _tie_patterns() -> tuple[list[int], list[int], list[int]]:
    missing_last = list(range(1, 101))
    missing_first = list(range(2, 102))
    missing_middle = [1, *range(3, 102)]
    return missing_last, missing_first, missing_middle


def test_one_second_tie_pagination_closes_from_audited_cardinality_union(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    missing_last, missing_first, missing_middle = _tie_patterns()
    api = UnstableTiePaginationAPI(
        start,
        [
            missing_last,
            missing_last,
            missing_first,
            missing_middle,
        ],
    )
    inventory = _inventory(tmp_path, api, end=_iso(start + 1))

    inventory.run()
    receipt = inventory.write_completion_receipt(tmp_path / "receipt.json")

    assert receipt["run_count"] == 101
    assert len(api.calls) == 8
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM window_pages"
        ).fetchone() == (0,)
        assert conn.execute(
            """
            SELECT distinct_item_count,accumulated_distinct_count,
                   min_observation_count
            FROM convergence_passes ORDER BY pass_no
            """
        ).fetchall() == [
            (100, 100, 1),
            (100, 101, 1),
            (100, 101, 2),
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM convergence_pass_pages"
        ).fetchone() == (6,)
        assert conn.execute(
            "SELECT COUNT(*) FROM convergence_pass_runs"
        ).fetchone() == (300,)
        assert conn.execute(
            """
            SELECT pass_count,observed_page_count,observed_item_count,
                   distinct_run_count,min_observation_count
            FROM window_union_closures
            """
        ).fetchone() == (3, 6, 303, 101, 2)
        assert conn.execute(
            """
            SELECT MIN(observation_count),MAX(observation_count)
            FROM convergence_runs
            """
        ).fetchone() == (2, 3)

        conn.execute(
            """
            UPDATE convergence_runs SET observation_count=1
            WHERE run_id=(SELECT MIN(run_id) FROM convergence_runs)
            """
        )
    with pytest.raises(ci.CompletionError, match="observation proof"):
        inventory.db.completion_receipt()


def test_cardinality_union_proof_survives_process_resume(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    missing_last, missing_first, missing_middle = _tie_patterns()
    interrupted_api = FailOnCallAPI(
        UnstableTiePaginationAPI(
            start,
            [missing_last, missing_last],
        ),
        call_no=5,
    )
    first = _inventory(
        tmp_path,
        interrupted_api,
        end=_iso(start + 1),
        max_attempts=1,
    )
    with pytest.raises(ci.InventoryError, match="server retries exhausted"):
        first.run()
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT pass_no FROM convergence_passes"
        ).fetchall() == [(1,)]
        assert conn.execute(
            "SELECT status FROM search_windows"
        ).fetchall() == [("failed",)]

    resumed_api = UnstableTiePaginationAPI(
        start,
        [missing_first, missing_middle],
    )
    resumed = _inventory(
        tmp_path,
        resumed_api,
        end=_iso(start + 1),
        resume=True,
    )
    resumed.run()
    receipt = resumed.write_completion_receipt(tmp_path / "receipt.json")

    assert receipt["run_count"] == 101
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT pass_no FROM convergence_passes ORDER BY pass_no"
        ).fetchall() == [(1,), (2,), (3,)]
        assert conn.execute(
            "SELECT COUNT(*) FROM window_convergence"
        ).fetchone() == (0,)


class RateLimitOnceAPI:
    def __init__(self, delegate: DatasetAPI):
        self.delegate = delegate
        self.authorizations: list[str] = []
        self.calls = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        self.calls += 1
        self.authorizations.append(headers["Authorization"])
        if self.calls == 1:
            return ci.HTTPResponse(
                status=429,
                headers={"Retry-After": "1", "X-RateLimit-Remaining": "0"},
                body=json.dumps(
                    {"message": "secondary rate limit mentions secret-one"}
                ).encode(),
            )
        return self.delegate(method, url, headers, timeout)


def test_rate_limit_rotates_tokens_and_ledger_redacts_secrets(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    api = RateLimitOnceAPI(DatasetAPI([_run(1, start + 1)]))
    inventory = _inventory(tmp_path, api)
    inventory.run()

    assert api.authorizations == ["Bearer secret-one", "Bearer secret-two"]
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        rows = conn.execute(
            "SELECT outcome,error_message FROM request_ledger ORDER BY id"
        ).fetchall()
        assert [row[0] for row in rows] == ["rate_limit_retry", "success"]
        assert "secret-one" not in (rows[0][1] or "")
        assert "<redacted>" in rows[0][1]


class ServerFailureOnceAPI:
    def __init__(self, delegate: DatasetAPI):
        self.delegate = delegate
        self.failed = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        if not self.failed:
            self.failed = True
            return ci.HTTPResponse(
                status=503,
                headers={},
                body=b'{"message":"temporary outage"}',
            )
        return self.delegate(method, url, headers, timeout)


def test_failed_page_resumes_from_sqlite_without_replaying_completed_pages(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    runs = [_run(run_id, start + 1) for run_id in range(1, 151)]
    first_api = ServerFailureOnceAPI(DatasetAPI(runs))
    first = _inventory(tmp_path, first_api, max_attempts=1)
    with pytest.raises(ci.InventoryError, match="server retries exhausted"):
        first.run()
    # The audited v1 -> v2 migration is the sole script-hash upgrade path,
    # allowing the real failed smoke DB to resume under pagination recovery.
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='schema'",
            ("cppmega_ci_stream_inventory_v1",),
        )
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='script_sha256'",
            ("legacy-script-sha",),
        )

    # The failed first request committed no page.  --resume resets only the
    # failed window and reuses the exact scope/interval/script binding.
    second_api = DatasetAPI(runs)
    resumed = _inventory(tmp_path, second_api, resume=True)
    resumed.run()
    receipt = resumed.write_completion_receipt(tmp_path / "receipt.json")

    assert [call["page"] for call in second_api.calls] == [1, 2]
    assert receipt["run_count"] == 150
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT value FROM inventory_meta WHERE key='schema'"
        ).fetchone() == (ci.SCHEMA_VERSION,)
        assert conn.execute(
            """
            SELECT from_schema,to_schema,from_script_sha256
            FROM inventory_upgrades
            """
        ).fetchall() == [
            (
                "cppmega_ci_stream_inventory_v1",
                ci.SCHEMA_VERSION,
                "legacy-script-sha",
            )
        ]
        assert conn.execute(
            "SELECT outcome FROM request_ledger ORDER BY id"
        ).fetchall() == [
            ("server_retry",),
            ("window_error",),
            ("success",),
            ("success",),
        ]


def test_v2_to_v3_resume_requires_exact_audited_producer_migration(
    tmp_path: Path,
) -> None:
    old_v1_script = "1" * 64
    old_v2_script = "2" * 64
    reason = (
        "recover unstable one-second pagination with cardinality union proof"
    )
    _inventory(tmp_path, DatasetAPI([]))
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='schema'",
            (ci.PREVIOUS_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='script_sha256'",
            (old_v2_script,),
        )
        conn.execute(
            """
            INSERT INTO inventory_upgrades(
                from_schema,to_schema,from_script_sha256,
                to_script_sha256,upgraded_at
            ) VALUES (?,?,?,?,?)
            """,
            (
                "cppmega_ci_stream_inventory_v1",
                ci.PREVIOUS_SCHEMA_VERSION,
                old_v1_script,
                old_v2_script,
                "2026-01-01T00:00:00Z",
            ),
        )

    with pytest.raises(ci.BindingError, match="exact bound producer"):
        _inventory(tmp_path, DatasetAPI([]), resume=True)
    with pytest.raises(ci.BindingError, match="exact bound producer"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            allow_script_upgrade_from_sha256="3" * 64,
            script_upgrade_reason=reason,
        )
    with pytest.raises(ci.BindingError, match="requires a reason"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            allow_script_upgrade_from_sha256=old_v2_script,
        )

    migrated = _inventory(
        tmp_path,
        DatasetAPI([]),
        resume=True,
        allow_script_upgrade_from_sha256=old_v2_script,
        script_upgrade_reason=reason,
    )
    migrated.run()
    receipt = migrated.write_completion_receipt(
        tmp_path / "receipt.json"
    )

    assert [
        upgrade["reason"] for upgrade in receipt["binding_upgrades"]
    ] == [ci.IMPORTED_UPGRADE_REASON, reason]
    assert receipt["binding_upgrades"][-1][
        "from_script_sha256"
    ] == old_v2_script
    assert receipt["binding_upgrades"][-1][
        "to_script_sha256"
    ] == migrated.script_sha256
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM inventory_upgrades"
        ).fetchone() == (2,)
        assert conn.execute(
            "SELECT COUNT(*) FROM inventory_binding_upgrades"
        ).fetchone() == (2,)
        assert conn.execute(
            "SELECT value FROM inventory_meta WHERE key='schema'"
        ).fetchone() == (ci.SCHEMA_VERSION,)

    repeated_api = DatasetAPI([])
    repeated = _inventory(
        tmp_path,
        repeated_api,
        resume=True,
        allow_script_upgrade_from_sha256=old_v2_script,
        script_upgrade_reason=reason,
    )
    repeated.run()
    assert repeated_api.calls == []
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM inventory_binding_upgrades"
        ).fetchone() == (2,)
    with pytest.raises(ci.BindingError, match="does not exactly replay"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            allow_script_upgrade_from_sha256=old_v2_script,
            script_upgrade_reason=f"{reason} but changed",
        )


def test_same_schema_resume_requires_exact_audited_producer_migration(
    tmp_path: Path,
) -> None:
    old_script = tmp_path / "old_inventory_producer.py"
    new_script = tmp_path / "new_inventory_producer.py"
    old_script.write_text("old producer\n", encoding="utf-8")
    new_script.write_text("new producer\n", encoding="utf-8")
    reason = "record split-parent source count drift in the receipt"

    original = _inventory(
        tmp_path,
        DatasetAPI([]),
        script_path=old_script,
    )
    original.run()

    with pytest.raises(ci.BindingError, match="resume binding mismatch"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            script_path=new_script,
        )
    with pytest.raises(ci.BindingError, match="exact bound producer"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            script_path=new_script,
            allow_script_upgrade_from_sha256="3" * 64,
            script_upgrade_reason=reason,
        )
    with pytest.raises(ci.BindingError, match="requires a reason"):
        _inventory(
            tmp_path,
            DatasetAPI([]),
            resume=True,
            script_path=new_script,
            allow_script_upgrade_from_sha256=original.script_sha256,
        )

    migrated_api = DatasetAPI([])
    migrated = _inventory(
        tmp_path,
        migrated_api,
        resume=True,
        script_path=new_script,
        allow_script_upgrade_from_sha256=original.script_sha256,
        script_upgrade_reason=reason,
    )
    migrated.run()
    receipt = migrated.write_completion_receipt(tmp_path / "receipt.json")

    assert migrated_api.calls == []
    assert receipt["binding_upgrades"][-1] == {
        "from_schema": ci.SCHEMA_VERSION,
        "to_schema": ci.SCHEMA_VERSION,
        "from_script_sha256": original.script_sha256,
        "to_script_sha256": migrated.script_sha256,
        "reason": reason,
        "upgraded_at": receipt["binding_upgrades"][-1]["upgraded_at"],
    }
    repeated_api = DatasetAPI([])
    repeated = _inventory(
        tmp_path,
        repeated_api,
        resume=True,
        script_path=new_script,
        allow_script_upgrade_from_sha256=original.script_sha256,
        script_upgrade_reason=reason,
    )
    repeated.run()
    assert repeated_api.calls == []
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM inventory_binding_upgrades"
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT from_schema,to_schema FROM inventory_upgrades"
        ).fetchone() == (ci.SCHEMA_VERSION, ci.SCHEMA_VERSION)


def test_v2_incomplete_convergence_resumes_with_persisted_attempt_number(
    tmp_path: Path,
) -> None:
    start = ci.parse_utc_instant(START)
    missing_last, missing_first, missing_middle = _tie_patterns()
    interrupted = _inventory(
        tmp_path,
        FailOnCallAPI(
            UnstableTiePaginationAPI(start, [missing_last]),
            call_no=3,
        ),
        end=_iso(start + 1),
        max_attempts=1,
    )
    with pytest.raises(ci.InventoryError, match="server retries exhausted"):
        interrupted.run()

    old_v2_script = "4" * 64
    reason = "resume the exact failed v2 one-second convergence state"
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        conn.execute(
            """
            UPDATE window_convergence
            SET attempts=12,candidate_total=101,
                candidate_sha256=?,stable_observations=0
            """,
            ("5" * 64,),
        )
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='schema'",
            (ci.PREVIOUS_SCHEMA_VERSION,),
        )
        conn.execute(
            "UPDATE inventory_meta SET value=? WHERE key='script_sha256'",
            (old_v2_script,),
        )

    resumed = _inventory(
        tmp_path,
        UnstableTiePaginationAPI(
            start,
            [missing_last, missing_first, missing_middle],
        ),
        end=_iso(start + 1),
        resume=True,
        allow_script_upgrade_from_sha256=old_v2_script,
        script_upgrade_reason=reason,
    )
    resumed.run()
    receipt = resumed.write_completion_receipt(
        tmp_path / "receipt.json"
    )

    assert receipt["run_count"] == 101
    assert receipt["binding_upgrades"][-1]["reason"] == reason
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT pass_no FROM convergence_passes ORDER BY pass_no"
        ).fetchall() == [(13,), (14,), (15,)]
        assert conn.execute(
            """
            SELECT first_pass_no,last_pass_no,pass_count
            FROM window_union_closures
            """
        ).fetchone() == (13, 15, 3)


class MalformedAPI:
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> ci.HTTPResponse:
        return ci.HTTPResponse(
            status=200,
            headers={},
            body=b'{"total_count":"not-an-int","workflow_runs":[]}',
        )


def test_malformed_api_is_ledgered_and_receipt_is_refused(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, MalformedAPI())
    with pytest.raises(ci.InventoryError, match="malformed GitHub response"):
        inventory.run()
    with sqlite3.connect(tmp_path / "inventory.sqlite") as conn:
        assert conn.execute(
            "SELECT outcome,error_class FROM request_ledger"
        ).fetchall() == [
            ("malformed", "MalformedAPI"),
            ("window_error", "MalformedAPIError"),
        ]
        assert conn.execute(
            "SELECT status FROM search_windows"
        ).fetchall() == [("failed",)]
    with pytest.raises(ci.CompletionError, match="open/failed windows"):
        inventory.write_completion_receipt(tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_receipt_refuses_open_windows_and_smoke_never_claims_production(
    tmp_path: Path,
) -> None:
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_inventory = _inventory(open_dir, DatasetAPI([]))
    with pytest.raises(ci.CompletionError, match="open/failed windows"):
        open_inventory.write_completion_receipt(open_dir / "receipt.json")

    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    scope = _scope(
        smoke_dir,
        ["Owner/Repo", "Other/Repo"],
        smoke=True,
        max_repos=1,
    )
    smoke_inventory = _inventory(
        smoke_dir, DatasetAPI([]), scope=scope, db_name="smoke.sqlite"
    )
    smoke_inventory.run()
    receipt = smoke_inventory.write_completion_receipt(smoke_dir / "receipt.json")
    assert receipt["mode"] == "smoke"
    assert receipt["enumeration_complete"] is True
    assert receipt["source_snapshot_stable"] is True
    assert receipt["production_complete"] is False
    assert receipt["repo_list"]["repos"] == 1
    assert receipt["repo_list"]["original_repos"] == 2
