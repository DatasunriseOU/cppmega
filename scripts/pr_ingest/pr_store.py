#!/usr/bin/env python3
"""SQLite store for ingested GitHub pull requests (Tier-2 PR corpus).

Two query paths, both first-class (RULE #1: one clear path each, fail loud):

  * table ``prs``        -- primary key (repo, pr_number)
  * table ``pr_by_sha``  -- (repo, merge_commit_sha) -> pr_number, so a PR can be
                            looked up by the commit it landed as.

A PR row carries the full real payload: number, title, body, state, author,
created/merged timestamps, merge_commit_sha, and a JSON blob of comments +
reviews (each with author + body). Cursor checkpoints for resumable streaming
live in ``fetch_cursor`` keyed by (repo, kind).

Nothing here silently swallows errors: a SQLite error propagates. The only
"soft" path is upsert (INSERT OR REPLACE), which is intentional idempotency for
re-runs, not a fallback that hides failure.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prs (
    repo             TEXT NOT NULL,
    pr_number        INTEGER NOT NULL,
    title            TEXT,
    body             TEXT,
    state            TEXT,
    author           TEXT,
    created_at       TEXT,
    merged_at        TEXT,
    merge_commit_sha TEXT,
    comments_json    TEXT NOT NULL DEFAULT '[]',
    reviews_json     TEXT NOT NULL DEFAULT '[]',
    raw_json         TEXT,
    fetched_at       TEXT,
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
"""


class PRStore:
    """Thin, fail-loud wrapper around the PR SQLite database."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "PRStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- writes -----------------------------------------------------------
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
        comments: Iterable[dict],
        reviews: Iterable[dict],
        raw: Any,
        fetched_at: str,
    ) -> None:
        if not repo or pr_number is None:
            raise ValueError(f"upsert_pr needs repo+pr_number, got {repo!r} {pr_number!r}")
        comments_json = json.dumps(list(comments), ensure_ascii=False)
        reviews_json = json.dumps(list(reviews), ensure_ascii=False)
        raw_json = json.dumps(raw, ensure_ascii=False) if raw is not None else None
        self.conn.execute(
            """INSERT OR REPLACE INTO prs
               (repo, pr_number, title, body, state, author, created_at,
                merged_at, merge_commit_sha, comments_json, reviews_json,
                raw_json, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (repo, pr_number, title, body, state, author, created_at,
             merged_at, merge_commit_sha, comments_json, reviews_json,
             raw_json, fetched_at),
        )
        if merge_commit_sha:
            self.conn.execute(
                """INSERT OR REPLACE INTO pr_by_sha
                   (repo, merge_commit_sha, pr_number) VALUES (?,?,?)""",
                (repo, merge_commit_sha, pr_number),
            )

    def commit(self) -> None:
        self.conn.commit()

    # ---- cursor checkpoint (resumable) ------------------------------------
    def get_cursor(self, repo: str, kind: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM fetch_cursor WHERE repo=? AND kind=?", (repo, kind)
        )
        return cur.fetchone()

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
            """INSERT OR REPLACE INTO fetch_cursor
               (repo, kind, cursor, page_count, pr_count, done, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (repo, kind, cursor, page_count, pr_count, 1 if done else 0, updated_at),
        )
        self.conn.commit()

    # ---- reads (both query paths) -----------------------------------------
    def get_by_number(self, repo: str, pr_number: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM prs WHERE repo=? AND pr_number=?", (repo, pr_number)
        )
        return cur.fetchone()

    def get_by_sha(self, repo: str, merge_commit_sha: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute(
            """SELECT p.* FROM prs p
               JOIN pr_by_sha s ON s.repo=p.repo AND s.pr_number=p.pr_number
               WHERE s.repo=? AND s.merge_commit_sha=?""",
            (repo, merge_commit_sha),
        )
        return cur.fetchone()

    def count(self, repo: Optional[str] = None) -> int:
        if repo is None:
            cur = self.conn.execute("SELECT COUNT(*) AS n FROM prs")
        else:
            cur = self.conn.execute("SELECT COUNT(*) AS n FROM prs WHERE repo=?", (repo,))
        return int(cur.fetchone()["n"])

    def count_by_sha(self, repo: Optional[str] = None) -> int:
        if repo is None:
            cur = self.conn.execute("SELECT COUNT(*) AS n FROM pr_by_sha")
        else:
            cur = self.conn.execute(
                "SELECT COUNT(*) AS n FROM pr_by_sha WHERE repo=?", (repo,)
            )
        return int(cur.fetchone()["n"])
