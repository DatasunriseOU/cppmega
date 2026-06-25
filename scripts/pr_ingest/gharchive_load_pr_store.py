#!/usr/bin/env python3
"""Load GH Archive BigQuery PR events into the PRStore schema.

``gharchive_run.sh`` extracts raw PullRequest/Review/Comment events as JSON.
This loader is the ingestion half of that fallback: it materializes those raw
events into the same ``prs`` and ``pr_by_sha`` tables used by the GraphQL path.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from pr_store import PRStore


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, str):
        return json.loads(payload) if payload else {}
    if isinstance(payload, dict):
        return payload
    return {}


def _events(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"{path} JSON root must be a list")
        return data
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} JSONL row must be an object")
        events.append(item)
    return events


def _event_repo(event: dict[str, Any]) -> str:
    repo = event.get("repo_name") or (event.get("repo") or {}).get("name")
    if not isinstance(repo, str) or "/" not in repo:
        raise ValueError(f"GH Archive event missing repo_name: id={event.get('id')!r}")
    return repo


def _event_author(event: dict[str, Any]) -> str | None:
    actor = event.get("actor_login")
    if isinstance(actor, str) and actor:
        return actor
    actor_obj = event.get("actor")
    if isinstance(actor_obj, dict):
        login = actor_obj.get("login")
        return login if isinstance(login, str) and login else None
    return None


def _issue_number(payload: dict[str, Any]) -> int | None:
    issue = payload.get("issue")
    if isinstance(issue, dict) and isinstance(issue.get("number"), int):
        return int(issue["number"])
    return None


def _pr_number(payload: dict[str, Any]) -> int | None:
    pr = payload.get("pull_request")
    if isinstance(pr, dict) and isinstance(pr.get("number"), int):
        return int(pr["number"])
    return _issue_number(payload)


def load_gharchive_events(events_path: Path, db_path: Path) -> int:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}

    for event in _events(events_path):
        event_type = event.get("type")
        payload = _payload(event)
        repo = _event_repo(event)
        number = _pr_number(payload)
        if number is None:
            continue

        row = grouped.setdefault(
            (repo, number),
            {
                "repo": repo,
                "number": number,
                "title": None,
                "body": None,
                "state": None,
                "author": _event_author(event),
                "created_at": None,
                "merged_at": None,
                "merge_commit_sha": None,
                "comments": [],
                "reviews": [],
                "raw": [],
            },
        )
        row["raw"].append(event)

        if event_type == "PullRequestEvent":
            pr = payload.get("pull_request") or {}
            if not isinstance(pr, dict):
                continue
            row["title"] = pr.get("title") or row["title"]
            row["body"] = pr.get("body") or row["body"]
            row["state"] = pr.get("state") or row["state"]
            user = pr.get("user") if isinstance(pr.get("user"), dict) else {}
            row["author"] = user.get("login") or row["author"]
            row["created_at"] = pr.get("created_at") or row["created_at"]
            row["merged_at"] = pr.get("merged_at") or row["merged_at"]
            row["merge_commit_sha"] = (
                pr.get("merge_commit_sha") or row["merge_commit_sha"]
            )
        elif event_type == "PullRequestReviewEvent":
            review = payload.get("review") or {}
            if isinstance(review, dict):
                user = review.get("user") if isinstance(review.get("user"), dict) else {}
                row["reviews"].append(
                    {
                        "author": user.get("login") or _event_author(event),
                        "body": review.get("body"),
                        "state": review.get("state"),
                    }
                )
        elif event_type in {"IssueCommentEvent", "PullRequestReviewCommentEvent"}:
            comment = payload.get("comment") or {}
            if isinstance(comment, dict):
                user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
                row["comments"].append(
                    {
                        "author": user.get("login") or _event_author(event),
                        "body": comment.get("body"),
                    }
                )

    fetched_at = _now_utc()
    with PRStore(str(db_path)) as store:
        for row in grouped.values():
            store.upsert_pr(
                row["repo"],
                row["number"],
                title=row["title"],
                body=row["body"],
                state=row["state"],
                author=row["author"],
                created_at=row["created_at"],
                merged_at=row["merged_at"],
                merge_commit_sha=row["merge_commit_sha"],
                comments=row["comments"],
                reviews=row["reviews"],
                raw=row["raw"],
                fetched_at=fetched_at,
            )
        store.commit()
    return len(grouped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="GH Archive JSON/JSONL output")
    parser.add_argument("--db", default="outputs/pr_ingest/prs.sqlite")
    args = parser.parse_args()

    count = load_gharchive_events(Path(args.events), Path(args.db))
    print(json.dumps({"events": str(Path(args.events)), "db": args.db, "prs": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
