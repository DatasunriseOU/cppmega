#!/usr/bin/env python3
"""GraphQL-primary, resumable PR fetch stream -> pr_store (GH Archive fallback path).

HYBRID strategy (David): GitHub GraphQL is PRIMARY (free, 5000 pts/hr/token).
Multiple tokens (secrets/gh_tokens.txt PATs + the gh CLI token) are ROTATED so
the effective budget is N*5000/hr. GH Archive is only a *fallback* hook invoked
when GraphQL truly cannot proceed (all tokens rate-limited with a far reset, or
a repo GraphQL cannot serve).

RULE #1 (fail loud, no silent fallback):
  * If a token hits its rate limit we ROTATE to the next token. If ALL tokens are
    exhausted we do NOT silently skip / drop PRs -- we RAISE TokenExhausted with
    the soonest reset time so the caller decides (wait or hand off to GH Archive).
  * Any unexpected GraphQL error body is raised with the repo + cursor context.
  * Resume is exact: the per-(repo,'pr') cursor checkpoint in pr_store means a
    re-run continues from the last saved endCursor and never refetches a page.

Each PR is stored with its REAL number, title, body, state, author, timestamps,
merge_commit_sha, plus the first N comments and reviews (author + body each).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from typing import Any, Callable, Optional

from pr_store import PRStore

GITHUB_GRAPHQL = "https://api.github.com/graphql"


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class TokenExhausted(RuntimeError):
    """All tokens are rate-limited. Carries soonest reset epoch for fallback."""

    def __init__(self, soonest_reset_epoch: Optional[int]):
        self.soonest_reset_epoch = soonest_reset_epoch
        when = (
            _dt.datetime.fromtimestamp(soonest_reset_epoch, _dt.timezone.utc).isoformat()
            if soonest_reset_epoch
            else "unknown"
        )
        super().__init__(
            f"ALL GraphQL tokens rate-limited; soonest reset at {when}. "
            f"Run GH Archive fallback (gharchive_run.sh with PR_STORE_DB) or wait."
        )


class TokenRotator:
    """Round-robin over real tokens; tracks which were actually used."""

    def __init__(self, tokens: list[str]):
        if not tokens:
            raise ValueError("TokenRotator needs at least one token")
        self.tokens = tokens
        self.idx = 0
        self.used: set[int] = set()
        # epoch seconds each token is rate-limited until (0 = available)
        self.blocked_until: dict[int, int] = {i: 0 for i in range(len(tokens))}

    def current(self) -> tuple[int, str]:
        return self.idx, self.tokens[self.idx]

    def mark_used(self, i: int) -> None:
        self.used.add(i)

    def block(self, i: int, until_epoch: int) -> None:
        self.blocked_until[i] = until_epoch

    def advance_to_available(self) -> tuple[int, str]:
        """Return next non-blocked token; raise TokenExhausted if all blocked."""
        now = int(time.time())
        n = len(self.tokens)
        for step in range(1, n + 1):
            j = (self.idx + step) % n
            if self.blocked_until.get(j, 0) <= now:
                self.idx = j
                return j, self.tokens[j]
        soonest = min(self.blocked_until.values()) if self.blocked_until else None
        raise TokenExhausted(soonest)

    def used_count(self) -> int:
        return len(self.used)


_PR_QUERY = """
query($owner:String!, $name:String!, $after:String) {
  rateLimit { remaining resetAt }
  repository(owner:$owner, name:$name) {
    pullRequests(first:25, after:$after, orderBy:{field:CREATED_AT, direction:ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state createdAt mergedAt mergeCommit { oid }
        author { login }
        comments(first:20) { nodes { author { login } body } }
        reviews(first:20) { nodes { author { login } body state } }
      }
    }
  }
}
"""


def _graphql_post(token: str, query: str, variables: dict) -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GraphQL PR ingestion requires the requests package for live HTTP calls"
        ) from exc
    return requests.post(
        GITHUB_GRAPHQL,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "cppmega-pr-ingest",
        },
        json={"query": query, "variables": variables},
        timeout=60,
    )


def _reset_epoch_from_headers(resp: Any) -> Optional[int]:
    v = resp.headers.get("X-RateLimit-Reset")
    if v and v.isdigit():
        return int(v)
    return None


def fetch_repo(
    store: PRStore,
    rotator: TokenRotator,
    owner: str,
    name: str,
    max_pages: Optional[int],
    max_prs: Optional[int],
    comment_cap: int,
    verbose: bool = True,
    graphql_post: Callable[[str, str, dict], Any] = _graphql_post,
) -> dict:
    """Fetch PRs for owner/name into store, resuming from the saved cursor."""
    repo = f"{owner}/{name}"
    cur_row = store.get_cursor(repo, "pr")
    after = cur_row["cursor"] if cur_row else None
    page_count = cur_row["page_count"] if cur_row else 0
    pr_count = cur_row["pr_count"] if cur_row else 0
    if cur_row and cur_row["done"]:
        if verbose:
            sys.stderr.write(f"[{repo}] cursor marked done; nothing to do.\n")
        return {"repo": repo, "fetched": 0, "resumed_at": after, "already_done": True}

    fetched_this_run = 0
    pages_this_run = 0
    refetch_guard = after  # for resume proof: first request reuses saved cursor

    while True:
        if max_prs is not None and fetched_this_run >= max_prs:
            break
        if max_pages is not None and pages_this_run >= max_pages:
            break
        i, token = rotator.current()
        resp = graphql_post(token, _PR_QUERY, {"owner": owner, "name": name, "after": after})
        if resp.status_code == 401:
            raise RuntimeError(f"[{repo}] token #{i} unauthorized (401): {resp.text[:300]}")
        if resp.status_code in (403, 429):
            reset = _reset_epoch_from_headers(resp) or (int(time.time()) + 60)
            rotator.block(i, reset)
            if verbose:
                sys.stderr.write(f"[{repo}] token #{i} rate-limited; rotating.\n")
            rotator.advance_to_available()
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"[{repo}] HTTP {resp.status_code}: {resp.text[:500]}")
        payload = resp.json()
        if "errors" in payload and payload["errors"]:
            # Rate-limit surfaced as a GraphQL error -> rotate; else fail loud.
            errs = payload["errors"]
            if any(e.get("type") == "RATE_LIMITED" for e in errs):
                reset = _reset_epoch_from_headers(resp) or (int(time.time()) + 60)
                rotator.block(i, reset)
                rotator.advance_to_available()
                continue
            raise RuntimeError(f"[{repo}] GraphQL errors: {json.dumps(errs)[:600]}")

        rotator.mark_used(i)
        data = payload["data"]
        rl = data.get("rateLimit") or {}
        repo_node = data.get("repository")
        if repo_node is None:
            raise RuntimeError(f"[{repo}] repository is null (not found / no access)")
        prs = repo_node["pullRequests"]
        nodes = prs["nodes"]
        page_info = prs["pageInfo"]

        processed_nodes = 0
        hit_pr_cap = False
        for nd in nodes:
            number = nd["number"]
            comments = [
                {"author": (c["author"] or {}).get("login"), "body": c["body"]}
                for c in nd["comments"]["nodes"][:comment_cap]
            ]
            reviews = [
                {
                    "author": (r["author"] or {}).get("login"),
                    "body": r["body"],
                    "state": r.get("state"),
                }
                for r in nd["reviews"]["nodes"][:comment_cap]
            ]
            store.upsert_pr(
                repo,
                number,
                title=nd.get("title"),
                body=nd.get("body"),
                state=nd.get("state"),
                author=(nd.get("author") or {}).get("login"),
                created_at=nd.get("createdAt"),
                merged_at=nd.get("mergedAt"),
                merge_commit_sha=(nd.get("mergeCommit") or {}).get("oid"),
                comments=comments,
                reviews=reviews,
                raw=nd,
                fetched_at=_now_utc(),
            )
            pr_count += 1
            fetched_this_run += 1
            processed_nodes += 1
            if max_prs is not None and fetched_this_run >= max_prs:
                hit_pr_cap = True
                break

        if hit_pr_cap and processed_nodes < len(nodes):
            store.commit()
            if verbose:
                sys.stderr.write(
                    f"[{repo}] max_prs reached mid-page after {processed_nodes}/"
                    f"{len(nodes)} PRs; cursor not advanced.\n"
                )
                sys.stderr.flush()
            break

        page_count += 1
        pages_this_run += 1
        has_next = page_info["hasNextPage"]
        after = page_info["endCursor"]
        done = not has_next
        store.set_cursor(repo, "pr", after, page_count, pr_count, done, _now_utc())
        store.commit()
        if verbose:
            sys.stderr.write(
                f"[{repo}] page {page_count} (+{processed_nodes} PRs, total {pr_count}) "
                f"rl_remaining={rl.get('remaining')} tok#{i} cursor={after}\n"
            )
            sys.stderr.flush()
        if done or hit_pr_cap:
            break

    return {
        "repo": repo,
        "fetched": fetched_this_run,
        "pages_this_run": pages_this_run,
        "total_in_store": store.count(repo),
        "resumed_from_cursor": refetch_guard,
    }


def load_tokens(tokens_file: Optional[str], use_gh_cli: bool) -> list[str]:
    toks: list[str] = []
    if tokens_file:
        with open(tokens_file) as fh:
            for line in fh:
                t = line.strip()
                if t and not t.startswith("#"):
                    toks.append(t)
    if use_gh_cli:
        try:
            out = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=True
            ).stdout.strip()
            if out and out not in toks:
                toks.append(out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gh auth token failed: {e.stderr}")
    # de-dup, preserve order
    seen: set[str] = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        raise RuntimeError("no tokens loaded (need --tokens file and/or gh CLI login)")
    return uniq


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--db", default="outputs/pr_ingest/prs.sqlite")
    ap.add_argument("--tokens", default=None, help="file of GitHub tokens, one per line")
    ap.add_argument("--no-gh-cli", action="store_true", help="do not append the gh CLI token")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--max-prs", type=int, default=None)
    ap.add_argument("--comment-cap", type=int, default=20)
    args = ap.parse_args(argv)

    if "/" not in args.repo:
        raise SystemExit(f"--repo must be owner/repo, got {args.repo!r}")
    owner, name = args.repo.split("/", 1)
    tokens = load_tokens(args.tokens, use_gh_cli=not args.no_gh_cli)
    sys.stderr.write(f"[tokens] loaded {len(tokens)} token(s)\n")
    rotator = TokenRotator(tokens)

    with PRStore(args.db) as store:
        result = fetch_repo(
            store, rotator, owner, name,
            max_pages=args.max_pages, max_prs=args.max_prs,
            comment_cap=args.comment_cap,
        )
        result["tokens_used"] = rotator.used_count()
        result["tokens_available"] = len(tokens)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
