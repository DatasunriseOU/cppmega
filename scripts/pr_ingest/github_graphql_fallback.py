#!/usr/bin/env python3
"""Fetch ONE specific PR by (repo, pr_number) via GraphQL, with --tokens rotation.

This is the targeted fallback used when the stream missed a PR or we need to
backfill a single number (e.g. one referenced by a commit). It reuses the same
TokenRotator (multi-token, fail-loud on full exhaustion) and the same PRStore so
the PR lands queryable by BOTH (repo,pr_number) AND (repo,merge_commit_sha).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import Optional

from graphql_pr_stream import (
    TokenRotator,
    _graphql_post,
    _reset_epoch_from_headers,
    load_tokens,
)
from pr_store import PRStore

_ONE_PR_QUERY = """
query($owner:String!, $name:String!, $number:Int!) {
  rateLimit { remaining resetAt }
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      number title body state createdAt mergedAt mergeCommit { oid }
      author { login }
      comments(first:20) { nodes { author { login } body } }
      reviews(first:20) { nodes { author { login } body state } }
    }
  }
}
"""


def fetch_one(store: PRStore, rotator: TokenRotator, owner: str, name: str,
              number: int, comment_cap: int) -> dict:
    repo = f"{owner}/{name}"
    while True:
        i, token = rotator.current()
        resp = _graphql_post(token, _ONE_PR_QUERY,
                             {"owner": owner, "name": name, "number": number})
        if resp.status_code in (403, 429):
            import time
            reset = _reset_epoch_from_headers(resp) or (int(time.time()) + 60)
            rotator.block(i, reset)
            rotator.advance_to_available()
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"[{repo}#{number}] HTTP {resp.status_code}: {resp.text[:400]}")
        payload = resp.json()
        if payload.get("errors"):
            if any(e.get("type") == "RATE_LIMITED" for e in payload["errors"]):
                import time
                rotator.block(i, _reset_epoch_from_headers(resp) or int(time.time()) + 60)
                rotator.advance_to_available()
                continue
            raise RuntimeError(f"[{repo}#{number}] GraphQL errors: {json.dumps(payload['errors'])[:400]}")
        rotator.mark_used(i)
        nd = payload["data"]["repository"]["pullRequest"]
        if nd is None:
            raise RuntimeError(f"[{repo}#{number}] pullRequest is null (not found)")
        comments = [{"author": (c["author"] or {}).get("login"), "body": c["body"]}
                    for c in nd["comments"]["nodes"][:comment_cap]]
        reviews = [{"author": (r["author"] or {}).get("login"), "body": r["body"],
                    "state": r.get("state")} for r in nd["reviews"]["nodes"][:comment_cap]]
        store.upsert_pr(
            repo, nd["number"], title=nd.get("title"), body=nd.get("body"),
            state=nd.get("state"), author=(nd.get("author") or {}).get("login"),
            created_at=nd.get("createdAt"), merged_at=nd.get("mergedAt"),
            merge_commit_sha=(nd.get("mergeCommit") or {}).get("oid"),
            comments=comments, reviews=reviews, raw=nd,
            fetched_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        )
        store.commit()
        return {"repo": repo, "pr_number": nd["number"],
                "merge_commit_sha": (nd.get("mergeCommit") or {}).get("oid")}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--db", default="outputs/pr_ingest/prs.sqlite")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--no-gh-cli", action="store_true")
    ap.add_argument("--comment-cap", type=int, default=20)
    args = ap.parse_args(argv)
    owner, name = args.repo.split("/", 1)
    rotator = TokenRotator(load_tokens(args.tokens, use_gh_cli=not args.no_gh_cli))
    with PRStore(args.db) as store:
        res = fetch_one(store, rotator, owner, name, args.pr, args.comment_cap)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
