#!/usr/bin/env python3
"""Unified root PR store for GraphQL, GH Archive, and training export.

The canonical API is the production function surface:
connect, upsert_record, get_by_pr, and get_by_sha. The root checkout also has
established callers of PRStore that persist state, author/timestamp/raw metadata
and GraphQL cursor checkpoints. This module keeps both surfaces on one SQLite
schema and migrates older root or MLX-created stores in place by adding missing
columns only.

RULE #1: malformed rows, unknown event types, missing stores, and SQLite errors
are surfaced immediately. There is no sibling-checkout or degraded fallback.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS prs (
    repo             TEXT NOT NULL,
    pr_number        INTEGER NOT NULL,
    title            TEXT,
    body             TEXT,
    pr_title         TEXT,
    pr_body          TEXT,
    state            TEXT,
    author           TEXT,
    created_at       TEXT,
    merged_at       TEXT,
    merge_commit_sha TEXT,
    comments_json    TEXT NOT NULL DEFAULT '[]',
    reviews_json     TEXT NOT NULL DEFAULT '[]',
    raw_json         TEXT,
    fetched_at       TEXT,
    scan_id          TEXT,
    PRIMARY KEY (repo, pr_number)
);
CREATE TABLE IF NOT EXISTS pr_by_sha (
    repo             TEXT NOT NULL,
    merge_commit_sha TEXT NOT NULL,
    pr_number        INTEGER NOT NULL,
    PRIMARY KEY (repo, merge_commit_sha)
);
CREATE INDEX IF NOT EXISTS idx_pr_by_sha_pr ON pr_by_sha (repo, pr_number);
CREATE TABLE IF NOT EXISTS fetch_cursor (
    repo       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    cursor     TEXT,
    page_count INTEGER NOT NULL DEFAULT 0,
    pr_count   INTEGER NOT NULL DEFAULT 0,
    done       INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (repo, kind)
);
CREATE TABLE IF NOT EXISTS comments (
    repo       TEXT NOT NULL,
    pr_number  INTEGER NOT NULL,
    user       TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    path       TEXT,
    created_at TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT 'comment',
    UNIQUE (repo, pr_number, kind, created_at, body)
);
CREATE TABLE IF NOT EXISTS reviews (
    repo       TEXT NOT NULL,
    pr_number  INTEGER NOT NULL,
    user       TEXT NOT NULL DEFAULT '',
    state      TEXT NOT NULL DEFAULT '',
    body       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    UNIQUE (repo, pr_number, created_at, body)
);
CREATE TABLE IF NOT EXISTS linked_issues (
    repo      TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    number    INTEGER NOT NULL,
    title     TEXT NOT NULL DEFAULT '',
    body      TEXT NOT NULL DEFAULT '',
    UNIQUE (repo, pr_number, number)
);
"""

_PR_COLUMN_DEFINITIONS = {
    "title": "TEXT",
    "body": "TEXT",
    "pr_title": "TEXT",
    "pr_body": "TEXT",
    "state": "TEXT",
    "author": "TEXT",
    "created_at": "TEXT",
    "merged_at": "TEXT",
    "merge_commit_sha": "TEXT",
    "comments_json": "TEXT NOT NULL DEFAULT '[]'",
    "reviews_json": "TEXT NOT NULL DEFAULT '[]'",
    "raw_json": "TEXT",
    "fetched_at": "TEXT",
    "scan_id": "TEXT",
}


def _ensure_pr_columns(conn: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(prs)")
    }
    added: set[str] = set()
    for name, definition in _PR_COLUMN_DEFINITIONS.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE prs ADD COLUMN {name} {definition}")
            added.add(name)

    # These aliases only need backfilling when one side was introduced by this
    # migration. Running four whole-table UPDATEs on every connection turned a
    # resumable multi-gigabyte PR scan into a long writer transaction and made
    # concurrent workers fail with ``database is locked``.
    if {"title", "pr_title"} & added:
        conn.execute(
            "UPDATE prs SET title=pr_title "
            "WHERE (title IS NULL OR title='') AND pr_title IS NOT NULL"
        )
        conn.execute(
            "UPDATE prs SET pr_title=title "
            "WHERE (pr_title IS NULL OR pr_title='') AND title IS NOT NULL"
        )
    if {"body", "pr_body"} & added:
        conn.execute(
            "UPDATE prs SET body=pr_body "
            "WHERE (body IS NULL OR body='') AND pr_body IS NOT NULL"
        )
        conn.execute(
            "UPDATE prs SET pr_body=body "
            "WHERE (pr_body IS NULL OR pr_body='') AND body IS NOT NULL"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prs_scan_repo "
        "ON prs(scan_id, repo, pr_number)"
    )


def connect(
    store_path: str,
    create: bool = True,
    *,
    readonly: bool = False,
    initialize: bool = True,
) -> sqlite3.Connection:
    """Open the PR store, optionally creating or compatibly migrating it.

    ``initialize=False`` is the worker-connection path after one parent
    connection has already initialized the schema. It deliberately performs no
    DDL or compatibility UPDATEs, so concurrent stream workers do not contend
    on an otherwise-idempotent migration transaction.
    """

    if readonly and create:
        raise ValueError("readonly PR store connections cannot create a database")
    if create and not initialize:
        raise ValueError("non-initializing PR store connections cannot create a database")
    if not create and not os.path.exists(store_path):
        raise SystemExit(f"[pr_store] store does not exist: {store_path}")
    if create:
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)

    if readonly:
        uri = Path(store_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=60.0)
    else:
        conn = sqlite3.connect(store_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    if readonly:
        return conn

    if not initialize:
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if journal_mode != "wal":
            conn.close()
            raise RuntimeError(
                "non-initializing PR store connection requires a parent-initialized "
                f"WAL database, got journal_mode={journal_mode!r}"
            )
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    _ensure_pr_columns(conn)
    conn.commit()
    return conn


def _int_or_none(value: object) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    return int(value)


def _nonempty(new_value: object, old_value: object) -> object:
    return old_value if new_value in (None, "") else new_value


def _json_list(value: object, *, field: str) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid PR store {field}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"PR store {field} must be a list of objects")
    return [dict(item) for item in value]


def _normalize_comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "user": str(
            comment.get("user")
            or comment.get("author")
            or comment.get("login")
            or ""
        ),
        "body": str(comment.get("body") or ""),
        "path": comment.get("path"),
        "created_at": str(comment.get("created_at") or ""),
        "kind": str(comment.get("kind") or "comment"),
    }


def _normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "user": str(
            review.get("user")
            or review.get("author")
            or review.get("login")
            or ""
        ),
        "state": str(review.get("state") or ""),
        "body": str(review.get("body") or ""),
        "created_at": str(review.get("created_at") or ""),
    }


def record_content_sha256(rec: dict[str, Any]) -> str:
    """Hash the authoritative PR fields populated by the GraphQL gap filler."""

    comments = [_normalize_comment(dict(item)) for item in rec.get("comments", [])]
    reviews = [_normalize_review(dict(item)) for item in rec.get("reviews", [])]
    linked_issues = [
        {
            "number": int(item["number"]),
            "title": str(item.get("title") or ""),
            "body": str(item.get("body") or ""),
        }
        for item in rec.get("linked_issues", [])
    ]

    def canonical_item(item: dict[str, Any]) -> str:
        return json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    payload = {
        "repo": str(rec["repo"]),
        "pr_number": int(rec["pr_number"]),
        "merge_commit_sha": rec.get("merge_commit_sha"),
        "pr_title": str(rec.get("pr_title", rec.get("title")) or ""),
        "pr_body": str(rec.get("pr_body", rec.get("body")) or ""),
        "comments": sorted(comments, key=canonical_item),
        "reviews": sorted(reviews, key=canonical_item),
        "linked_issues": sorted(linked_issues, key=canonical_item),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _upsert_pr_meta(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
    *,
    merge_commit_sha: Optional[str] = None,
    title: object = None,
    body: object = None,
    state: object = None,
    author: object = None,
    created_at: object = None,
    merged_at: object = None,
    raw: object = None,
    fetched_at: object = None,
    scan_id: object = None,
) -> None:
    if not repo or pr_number is None:
        raise ValueError(
            f"PR metadata requires repo+pr_number, got {repo!r} {pr_number!r}"
        )
    row = conn.execute(
        "SELECT * FROM prs WHERE repo=? AND pr_number=?",
        (repo, int(pr_number)),
    ).fetchone()
    raw_json = (
        json.dumps(raw, ensure_ascii=False)
        if raw is not None
        else (row["raw_json"] if row is not None else None)
    )
    if row is None:
        normalized_title = "" if title is None else str(title)
        normalized_body = "" if body is None else str(body)
        conn.execute(
            """
            INSERT INTO prs(
                repo, pr_number, title, body, pr_title, pr_body, state, author,
                created_at, merged_at, merge_commit_sha, comments_json,
                reviews_json, raw_json, fetched_at, scan_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,'[]','[]',?,?,?)
            """,
            (
                repo,
                int(pr_number),
                normalized_title,
                normalized_body,
                normalized_title,
                normalized_body,
                state,
                author,
                created_at,
                merged_at,
                merge_commit_sha,
                raw_json,
                fetched_at,
                scan_id,
            ),
        )
    else:
        normalized_title = str(_nonempty(title, row["pr_title"] or row["title"] or ""))
        normalized_body = str(_nonempty(body, row["pr_body"] or row["body"] or ""))
        new_sha = _nonempty(merge_commit_sha, row["merge_commit_sha"])
        conn.execute(
            """
            UPDATE prs
            SET title=?, body=?, pr_title=?, pr_body=?, state=?, author=?,
                created_at=?, merged_at=?, merge_commit_sha=?, raw_json=?,
                fetched_at=?, scan_id=?
            WHERE repo=? AND pr_number=?
            """,
            (
                normalized_title,
                normalized_body,
                normalized_title,
                normalized_body,
                _nonempty(state, row["state"]),
                _nonempty(author, row["author"]),
                _nonempty(created_at, row["created_at"]),
                _nonempty(merged_at, row["merged_at"]),
                new_sha,
                raw_json,
                _nonempty(fetched_at, row["fetched_at"]),
                _nonempty(scan_id, row["scan_id"]),
                repo,
                int(pr_number),
            ),
        )

    effective_sha = merge_commit_sha
    if effective_sha is None:
        effective_sha = conn.execute(
            "SELECT merge_commit_sha FROM prs WHERE repo=? AND pr_number=?",
            (repo, int(pr_number)),
        ).fetchone()["merge_commit_sha"]
    if effective_sha:
        conn.execute(
            "INSERT OR REPLACE INTO pr_by_sha(repo, merge_commit_sha, pr_number) "
            "VALUES (?,?,?)",
            (repo, str(effective_sha), int(pr_number)),
        )


def _insert_comments(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
    comments: Iterable[dict[str, Any]],
) -> None:
    for raw_comment in comments:
        comment = _normalize_comment(dict(raw_comment))
        conn.execute(
            """
            INSERT OR IGNORE INTO comments(
                repo, pr_number, user, body, path, created_at, kind
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                repo,
                int(pr_number),
                comment["user"],
                comment["body"],
                comment["path"],
                comment["created_at"],
                comment["kind"],
            ),
        )


def _insert_reviews(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
    reviews: Iterable[dict[str, Any]],
) -> None:
    for raw_review in reviews:
        review = _normalize_review(dict(raw_review))
        conn.execute(
            """
            INSERT OR IGNORE INTO reviews(
                repo, pr_number, user, state, body, created_at
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                repo,
                int(pr_number),
                review["user"],
                review["state"],
                review["body"],
                review["created_at"],
            ),
        )


def _child_comments(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
) -> list[dict[str, Any]]:
    return [
        {
            "user": row["user"],
            "body": row["body"],
            "path": row["path"],
            "created_at": row["created_at"],
            "kind": row["kind"],
        }
        for row in conn.execute(
            "SELECT * FROM comments WHERE repo=? AND pr_number=? "
            "ORDER BY created_at, rowid",
            (repo, int(pr_number)),
        )
    ]


def _child_reviews(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
) -> list[dict[str, Any]]:
    return [
        {
            "user": row["user"],
            "state": row["state"],
            "body": row["body"],
            "created_at": row["created_at"],
        }
        for row in conn.execute(
            "SELECT * FROM reviews WHERE repo=? AND pr_number=? "
            "ORDER BY created_at, rowid",
            (repo, int(pr_number)),
        )
    ]


def _sync_json_blobs(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
) -> None:
    conn.execute(
        "UPDATE prs SET comments_json=?, reviews_json=? "
        "WHERE repo=? AND pr_number=?",
        (
            json.dumps(_child_comments(conn, repo, pr_number), ensure_ascii=False),
            json.dumps(_child_reviews(conn, repo, pr_number), ensure_ascii=False),
            repo,
            int(pr_number),
        ),
    )


class PRStore:
    """Compatibility wrapper retained for root ingestion and cursor callers."""

    def __init__(self, path: str):
        self.path = path
        self.conn = connect(path, create=True)

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "PRStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def upsert_pr(
        self,
        repo: str,
        pr_number: int,
        *,
        title: Optional[str],
        body: Optional[str],
        state: Optional[str],
        author: Optional[str],
        created_at: Optional[str],
        merged_at: Optional[str],
        merge_commit_sha: Optional[str],
        comments: Iterable[dict[str, Any]],
        reviews: Iterable[dict[str, Any]],
        raw: Any,
        fetched_at: str,
    ) -> None:
        comment_rows = [dict(item) for item in comments]
        review_rows = [dict(item) for item in reviews]
        _upsert_pr_meta(
            self.conn,
            repo,
            int(pr_number),
            merge_commit_sha=merge_commit_sha,
            title=title,
            body=body,
            state=state,
            author=author,
            created_at=created_at,
            merged_at=merged_at,
            raw=raw,
            fetched_at=fetched_at,
        )
        self.conn.execute(
            "DELETE FROM comments WHERE repo=? AND pr_number=?",
            (repo, int(pr_number)),
        )
        self.conn.execute(
            "DELETE FROM reviews WHERE repo=? AND pr_number=?",
            (repo, int(pr_number)),
        )
        _insert_comments(self.conn, repo, int(pr_number), comment_rows)
        _insert_reviews(self.conn, repo, int(pr_number), review_rows)
        _sync_json_blobs(self.conn, repo, int(pr_number))

    def commit(self) -> None:
        self.conn.commit()

    def get_cursor(self, repo: str, kind: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM fetch_cursor WHERE repo=? AND kind=?",
            (repo, kind),
        ).fetchone()

    def set_cursor(
        self,
        repo: str,
        kind: str,
        cursor: Optional[str],
        page_count: int,
        pr_count: int,
        done: bool,
        updated_at: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO fetch_cursor(
                repo, kind, cursor, page_count, pr_count, done, updated_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                repo,
                kind,
                cursor,
                int(page_count),
                int(pr_count),
                1 if done else 0,
                updated_at,
            ),
        )
        self.conn.commit()

    def get_by_number(self, repo: str, pr_number: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM prs WHERE repo=? AND pr_number=?",
            (repo, int(pr_number)),
        ).fetchone()

    def get_by_sha(
        self,
        repo: str,
        merge_commit_sha: str,
    ) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT p.* FROM prs p
            JOIN pr_by_sha s ON s.repo=p.repo AND s.pr_number=p.pr_number
            WHERE s.repo=? AND s.merge_commit_sha=?
            """,
            (repo, merge_commit_sha),
        ).fetchone()

    def count(self, repo: Optional[str] = None) -> int:
        if repo is None:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM prs").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM prs WHERE repo=?",
                (repo,),
            ).fetchone()
        return int(row["n"])

    def count_by_sha(self, repo: Optional[str] = None) -> int:
        if repo is None:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM pr_by_sha"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM pr_by_sha WHERE repo=?",
                (repo,),
            ).fetchone()
        return int(row["n"])


def upsert_record(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    *,
    commit: bool = True,
    replace_children: bool = False,
    scan_id: str | None = None,
) -> None:
    """Insert an assembled production record while retaining root metadata."""

    repo = rec["repo"]
    pr_number = int(rec["pr_number"])
    if replace_children:
        for table in ("comments", "reviews", "linked_issues"):
            conn.execute(
                f"DELETE FROM {table} WHERE repo=? AND pr_number=?",
                (repo, pr_number),
            )
    _upsert_pr_meta(
        conn,
        repo,
        pr_number,
        merge_commit_sha=rec.get("merge_commit_sha"),
        title=rec.get("pr_title", rec.get("title")),
        body=rec.get("pr_body", rec.get("body")),
        state=rec.get("state"),
        author=rec.get("author"),
        created_at=rec.get("created_at"),
        merged_at=rec.get("merged_at"),
        raw=rec.get("raw"),
        fetched_at=rec.get("fetched_at"),
        scan_id=scan_id if scan_id is not None else rec.get("scan_id"),
    )
    _insert_comments(conn, repo, pr_number, rec.get("comments", []))
    _insert_reviews(conn, repo, pr_number, rec.get("reviews", []))
    for issue in rec.get("linked_issues", []):
        conn.execute(
            """
            INSERT OR IGNORE INTO linked_issues(
                repo, pr_number, number, title, body
            ) VALUES (?,?,?,?,?)
            """,
            (
                repo,
                pr_number,
                int(issue["number"]),
                str(issue.get("title") or ""),
                str(issue.get("body") or ""),
            ),
        )
    _sync_json_blobs(conn, repo, pr_number)
    if commit:
        conn.commit()


def ingest_gharchive_rows(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Fold projected rows from gharchive_query.sql into the unified store."""

    counts = {
        "pr": 0,
        "comments": 0,
        "review_comments": 0,
        "reviews": 0,
        "skipped": 0,
    }
    touched: set[tuple[str, int]] = set()
    for index, raw_row in enumerate(rows):
        row = _project_raw_gharchive_row(raw_row)
        repo = row.get("repo_name")
        pr_number = _int_or_none(row.get("pr_number"))
        if not repo or pr_number is None:
            counts["skipped"] += 1
            continue
        event_type = str(row.get("event_type") or "")
        merge_sha = row.get("merge_commit_sha") or None
        event_created_at = str(row.get("created_at") or "")
        touched.add((str(repo), pr_number))

        if event_type == "PullRequestEvent":
            _upsert_pr_meta(
                conn,
                str(repo),
                pr_number,
                merge_commit_sha=merge_sha,
                title=row.get("pr_title") or "",
                body=row.get("pr_body") or "",
            )
            counts["pr"] += 1
        elif event_type == "IssueCommentEvent":
            _upsert_pr_meta(
                conn,
                str(repo),
                pr_number,
                merge_commit_sha=merge_sha,
            )
            _insert_comments(
                conn,
                str(repo),
                pr_number,
                [
                    {
                        "user": row.get("comment_user") or "",
                        "body": row.get("comment_body") or "",
                        "created_at": event_created_at,
                        "kind": "comment",
                    }
                ],
            )
            counts["comments"] += 1
        elif event_type == "PullRequestReviewCommentEvent":
            _upsert_pr_meta(
                conn,
                str(repo),
                pr_number,
                merge_commit_sha=merge_sha,
            )
            _insert_comments(
                conn,
                str(repo),
                pr_number,
                [
                    {
                        "user": row.get("comment_user") or "",
                        "body": row.get("comment_body") or "",
                        "path": row.get("comment_path"),
                        "created_at": event_created_at,
                        "kind": "review_comment",
                    }
                ],
            )
            counts["review_comments"] += 1
        elif event_type == "PullRequestReviewEvent":
            _upsert_pr_meta(
                conn,
                str(repo),
                pr_number,
                merge_commit_sha=merge_sha,
            )
            _insert_reviews(
                conn,
                str(repo),
                pr_number,
                [
                    {
                        "user": row.get("review_user") or "",
                        "state": row.get("review_state") or "",
                        "body": row.get("review_body") or "",
                        "created_at": event_created_at,
                    }
                ],
            )
            counts["reviews"] += 1
        else:
            raise SystemExit(
                f"[pr_store] row {index}: unknown event_type "
                f"{event_type!r} (corrupt export?)"
            )

    for repo, pr_number in touched:
        _sync_json_blobs(conn, repo, pr_number)
    conn.commit()
    return counts


def _project_raw_gharchive_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a root raw event into the production query row shape."""

    if row.get("event_type"):
        return row
    event_type = row.get("type")
    if not event_type:
        return row
    payload = row.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"GH Archive event {row.get('id')!r} has invalid payload JSON: {exc}"
            ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"GH Archive event {row.get('id')!r} payload must be an object"
        )

    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        pull_request = {}
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        issue = {}
    comment = payload.get("comment")
    if not isinstance(comment, dict):
        comment = {}
    review = payload.get("review")
    if not isinstance(review, dict):
        review = {}

    def login(value: object) -> str:
        return (
            str(value.get("login") or "")
            if isinstance(value, dict)
            else ""
        )

    return {
        **row,
        "event_type": event_type,
        "pr_number": pull_request.get("number") or issue.get("number"),
        "merge_commit_sha": pull_request.get("merge_commit_sha"),
        "pr_title": pull_request.get("title"),
        "pr_body": pull_request.get("body"),
        "comment_body": comment.get("body"),
        "comment_user": login(comment.get("user")),
        "comment_path": comment.get("path"),
        "review_body": review.get("body"),
        "review_state": review.get("state"),
        "review_user": login(review.get("user")),
    }


def _assemble(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
    *,
    scan_id: str | None = None,
) -> Optional[dict[str, Any]]:
    if scan_id is None:
        row = conn.execute(
            "SELECT * FROM prs WHERE repo=? AND pr_number=?",
            (repo, int(pr_number)),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM prs WHERE repo=? AND pr_number=? AND scan_id=?",
            (repo, int(pr_number), scan_id),
        ).fetchone()
    if row is None:
        return None

    comments = _child_comments(conn, repo, int(pr_number))
    reviews = _child_reviews(conn, repo, int(pr_number))
    if not comments:
        comments = [
            _normalize_comment(item)
            for item in _json_list(row["comments_json"], field="comments_json")
        ]
    if not reviews:
        reviews = [
            _normalize_review(item)
            for item in _json_list(row["reviews_json"], field="reviews_json")
        ]
    linked = [
        {
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
        }
        for issue in conn.execute(
            "SELECT * FROM linked_issues WHERE repo=? AND pr_number=? "
            "ORDER BY number",
            (repo, int(pr_number)),
        )
    ]
    raw = None
    if row["raw_json"]:
        try:
            raw = json.loads(row["raw_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid PR store raw_json for {repo}#{pr_number}: {exc}"
            ) from exc

    title = row["pr_title"] or row["title"] or ""
    body = row["pr_body"] or row["body"] or ""
    return {
        "repo": row["repo"],
        "pr_number": row["pr_number"],
        "merge_commit_sha": row["merge_commit_sha"],
        "pr_title": title,
        "pr_body": body,
        "title": title,
        "body": body,
        "state": row["state"],
        "author": row["author"],
        "created_at": row["created_at"],
        "merged_at": row["merged_at"],
        "fetched_at": row["fetched_at"],
        "scan_id": row["scan_id"],
        "raw": raw,
        "comments": comments,
        "reviews": reviews,
        "linked_issues": linked,
    }


def get_by_pr(
    conn: sqlite3.Connection,
    repo: str,
    pr_number: int,
    *,
    scan_id: str | None = None,
) -> Optional[dict[str, Any]]:
    return _assemble(conn, repo, int(pr_number), scan_id=scan_id)


def get_by_sha(
    conn: sqlite3.Connection,
    repo: str,
    sha: str,
    *,
    scan_id: str | None = None,
) -> Optional[dict[str, Any]]:
    if scan_id is None:
        row = conn.execute(
            """
            SELECT p.pr_number
            FROM pr_by_sha AS s
            JOIN prs AS p
              ON p.repo=s.repo
             AND p.pr_number=s.pr_number
             AND p.merge_commit_sha=s.merge_commit_sha
            WHERE s.repo=? AND s.merge_commit_sha=?
            """,
            (repo, sha),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT p.pr_number
            FROM pr_by_sha AS s
            JOIN prs AS p
              ON p.repo=s.repo
             AND p.pr_number=s.pr_number
             AND p.merge_commit_sha=s.merge_commit_sha
            WHERE s.repo=? AND s.merge_commit_sha=? AND p.scan_id=?
            """,
            (repo, sha, scan_id),
        ).fetchone()
    if row is None:
        return None
    return _assemble(
        conn,
        repo,
        int(row["pr_number"]),
        scan_id=scan_id,
    )


def _iter_ndjson(paths: list[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            first = handle.read(1)
            while first and first.isspace():
                first = handle.read(1)
            handle.seek(0)
            if first == "[":
                values = json.load(handle)
                if not isinstance(values, list):
                    raise SystemExit(f"[pr_store] {path}: JSON root must be a list")
                for index, value in enumerate(values):
                    if not isinstance(value, dict):
                        raise SystemExit(
                            f"[pr_store] {path}: row {index} must be a JSON object"
                        )
                    yield value
                continue
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"[pr_store] {path}:{lineno} not valid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise SystemExit(
                        f"[pr_store] {path}:{lineno} must be a JSON object"
                    )
                yield value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser(
        "ingest-gharchive",
        help="Fold projected GH Archive NDJSON shards.",
    )
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--input", required=True, help="Glob of NDJSON shards.")

    get = sub.add_parser("get", help="Look up a PR by number or merge SHA.")
    get.add_argument("--store", required=True)
    get.add_argument("--repo", required=True)
    get.add_argument("--pr", type=int)
    get.add_argument("--sha")

    stats = sub.add_parser("stats", help="Print store row counts.")
    stats.add_argument("--store", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "ingest-gharchive":
        paths = sorted(glob.glob(args.input))
        if not paths:
            raise SystemExit(
                f"[pr_store] no files matched --input {args.input!r}"
            )
        conn = connect(args.store, create=True)
        try:
            counts = ingest_gharchive_rows(conn, _iter_ndjson(paths))
        finally:
            conn.close()
        print(
            f"[pr_store] ingested {paths} -> {counts}",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "get":
        conn = connect(args.store, create=False, readonly=True)
        try:
            if args.sha:
                record = get_by_sha(conn, args.repo, args.sha)
            elif args.pr is not None:
                record = get_by_pr(conn, args.repo, args.pr)
            else:
                raise SystemExit("[pr_store] get requires --pr or --sha")
        finally:
            conn.close()
        if record is None:
            raise SystemExit(
                f"[pr_store] MISS: {args.repo} pr={args.pr} sha={args.sha}"
            )
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0

    conn = connect(args.store, create=False, readonly=True)
    try:
        for table in (
            "prs",
            "pr_by_sha",
            "comments",
            "reviews",
            "linked_issues",
            "fetch_cursor",
        ):
            count = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
            print(f"  {table:16s} {count}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
