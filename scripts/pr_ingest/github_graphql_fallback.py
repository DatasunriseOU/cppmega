#!/usr/bin/env python3
"""Tier-2 PR ingest, component (4): GitHub GraphQL gap-filler.

For (repo, pr_number) pairs that are NOT in the pr_store (events GH Archive
dropped: pre-2015, missing days, truncated bodies), fetch the PR directly from
the GitHub GraphQL API and write the assembled record into the same pr_store.

Fetched per PR (one GraphQL query, paginated):
  * title, body, mergeCommit.oid  (the merge SHA, for the (repo, sha) key)
  * comments (issue comments)              -> comments[]   (kind=comment)
  * reviews (with body + state)            -> reviews[]
  * reviewThreads.comments (inline)        -> comments[]   (kind=review_comment)
  * closingIssuesReferences (title+body)   -> linked_issues[]

Token rotation: --tokens points to a file with one PAT per line. We rotate to
the next token when one hits the primary rate limit (X-RateLimit-Remaining=0)
or a secondary/abuse limit; we honor Retry-After / reset with backoff.

RULE #1 (fail loud, no silent skip):
  * 401/403-bad-credential on ALL tokens  -> RAISE (auth failure, not a miss).
  * A PR that GitHub says does not exist (data.repository.pullRequest == null)
    -> recorded as a MISS in the misses file (a real, logged outcome), NOT
       silently swallowed. The run still completes; misses are reported.
  * Exhausting every token's quota with work remaining -> RAISE (quota), so the
    user knows to add tokens / wait, rather than getting a partial store that
    looks complete.

Usage:
  python3 scripts/pr_ingest/github_graphql_fallback.py \
      --store out/pr_store.sqlite \
      --tokens ~/.gh_pats.txt \
      --targets targets.jsonl \
      --misses out/graphql_misses.jsonl \
      --completion-receipt out/graphql_gap_completion.json

  targets.jsonl: one JSON per line, {"repo":"owner/repo","pr_number":1234}
  (Typically produced by diffing the corpus PR set against `pr_store stats`.)
"""

import argparse
import datetime as _dt
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.pr_ingest import pr_store  # noqa: E402

GQL_URL = "https://api.github.com/graphql"
TRANSIENT_RETRY_MAX_BACKOFF_S = 30.0
_TRANSIENT_TRANSPORT_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    TimeoutError,
    ConnectionError,
    urllib.error.URLError,
)

PR_QUERY = """
query($owner:String!, $name:String!, $number:Int!,
      $cComments:String, $cReviews:String, $cThreads:String,
      $cLinked:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      number
      title
      body
      mergeCommit { oid }
      comments(first:100, after:$cComments) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { author { login } body createdAt }
      }
      reviews(first:100, after:$cReviews) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { author { login } body state submittedAt }
      }
      reviewThreads(first:50, after:$cThreads) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          comments(first:100) {
            totalCount
            pageInfo { hasNextPage endCursor }
            nodes { id author { login } body path createdAt }
          }
        }
      }
      closingIssuesReferences(first:50, after:$cLinked) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { number title body }
      }
    }
  }
}
"""

THREAD_COMMENTS_QUERY = """
query($threadId:ID!, $cursor:String) {
  node(id:$threadId) {
    ... on PullRequestReviewThread {
      id
      comments(first:100, after:$cursor) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes { id author { login } body path createdAt }
      }
    }
  }
}
"""


class TokenPool:
    """Rotating pool of GitHub PATs with primary/secondary limit handling."""

    def __init__(self, tokens: list[str]):
        if not tokens:
            raise SystemExit("[graphql] --tokens file had no tokens")
        self.tokens = tokens
        self.idx = 0
        # Per-token next-usable epoch (set when a token is rate-limited).
        self.cooldown_until = [0.0] * len(tokens)

    def current(self) -> str:
        return self.tokens[self.idx]

    def advance(self) -> bool:
        """Move to the next non-cooling token. Return False if all are cooling."""
        n = len(self.tokens)
        for _ in range(n):
            self.idx = (self.idx + 1) % n
            if self.cooldown_until[self.idx] <= time.time():
                return True
        return False

    def cool(self, seconds: float) -> None:
        self.cooldown_until[self.idx] = time.time() + max(1.0, seconds)


def _post_query(
    query: str,
    token: str,
    variables: dict,
) -> tuple[int, dict, dict]:
    """POST one GraphQL query. Return (status, headers, json_body)."""
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(GQL_URL, data=payload, method="POST")
    req.add_header("Authorization", f"bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "cppmega-pr-ingest")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        headers = dict(e.headers or {})
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"message": str(e)}
        return e.code, headers, body


def _post(token: str, variables: dict) -> tuple[int, dict, dict]:
    return _post_query(PR_QUERY, token, variables)


def _post_thread_comments(
    token: str,
    variables: dict,
) -> tuple[int, dict, dict]:
    return _post_query(THREAD_COMMENTS_QUERY, token, variables)


def _request_graphql_page(
    pool: TokenPool,
    variables: dict,
    *,
    context: str,
    max_retries: int,
    post_fn: Callable[[str, dict], tuple[int, dict, dict]],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    attempts = 0
    transient_attempts = 0
    while True:
        try:
            status, headers, payload = post_fn(pool.current(), variables)
        except urllib.error.HTTPError:
            raise
        except _TRANSIENT_TRANSPORT_ERRORS as exc:
            transient_reason = type(exc).__name__
            headers = {}
        else:
            transient_reason = (
                f"HTTP {status}" if 500 <= status <= 599 else None
            )
        if transient_reason is not None:
            transient_attempts += 1
            if transient_attempts > max_retries:
                raise SystemExit(
                    f"[graphql] transient retry budget exhausted for {context} "
                    f"after {transient_attempts} failed attempts "
                    f"({transient_reason})"
                )
            retry_after = headers.get("Retry-After")
            wait = min(
                TRANSIENT_RETRY_MAX_BACKOFF_S,
                float(2 ** min(5, transient_attempts - 1)),
            )
            if retry_after is not None:
                try:
                    wait = min(
                        TRANSIENT_RETRY_MAX_BACKOFF_S,
                        max(1.0, float(retry_after)),
                    )
                except (TypeError, ValueError):
                    pass
            sys.stderr.write(
                f"[graphql] transient {transient_reason} for {context}; "
                f"retrying same token/query in {wait:.1f}s "
                f"({transient_attempts}/{max_retries})\n"
            )
            sys.stderr.flush()
            sleep_fn(wait)
            continue

        if status == 401:
            raise SystemExit(
                f"[graphql] 401 Unauthorized for token #{pool.idx} "
                f"({context}); fix the PAT(s) in --tokens"
            )

        errors = payload.get("errors") if isinstance(payload, dict) else None
        rate_limited_error = bool(
            errors
            and any(
                isinstance(error, dict)
                and error.get("type") == "RATE_LIMITED"
                for error in errors
            )
        )
        remaining = headers.get("X-RateLimit-Remaining")
        retry_after = headers.get("Retry-After")
        serialized_payload = json.dumps(payload).lower()
        is_secondary = status == 403 and (
            retry_after is not None
            or "secondary rate limit" in serialized_payload
            or "abuse" in serialized_payload
        )
        is_rate_limited = (
            rate_limited_error
            or is_secondary
            or str(remaining) == "0"
            or status == 429
        )
        if status == 403 and not is_rate_limited:
            raise SystemExit(
                f"[graphql] 403 Forbidden (not a rate limit) for {context}: "
                f"{payload.get('message', payload)}"
            )
        if is_rate_limited:
            attempts += 1
            if attempts > max_retries:
                raise SystemExit(
                    f"[graphql] exceeded {max_retries} retries on rate limits for "
                    f"{context}; add more PATs or wait"
                )
            if retry_after is not None:
                wait = float(retry_after)
            else:
                reset = headers.get("X-RateLimit-Reset")
                wait = (
                    max(1.0, float(reset) - time.time())
                    if reset
                    else 30.0
                )
            pool.cool(wait)
            if not pool.advance():
                soonest = min(pool.cooldown_until) - time.time()
                raise SystemExit(
                    f"[graphql] ALL tokens rate-limited (soonest reset in "
                    f"~{soonest:.0f}s); add more PATs or rerun later. "
                    f"FAILING LOUD rather than returning a partial store."
                )
            continue

        if status != 200:
            raise SystemExit(f"[graphql] HTTP {status} for {context}: {payload}")
        if errors:
            raise SystemExit(
                f"[graphql] GraphQL errors for {context}: {errors}"
            )
        return payload


def _normalize_review_comment(node: dict) -> dict:
    return {
        "user": (node.get("author") or {}).get("login", ""),
        "body": node.get("body", ""),
        "path": node.get("path"),
        "created_at": node.get("createdAt", ""),
        "kind": "review_comment",
    }


def _fetch_complete_thread_comments(
    pool: TokenPool,
    *,
    owner: str,
    name: str,
    number: int,
    thread_id: str,
    first_connection: dict,
    max_retries: int,
    post_fn: Callable[[str, dict], tuple[int, dict, dict]],
) -> list[dict]:
    expected_total = int(first_connection.get("totalCount") or 0)
    comments: list[dict] = []
    seen_comment_ids: set[str] = set()

    def consume(connection: dict) -> tuple[bool, str | None]:
        page_total = int(connection.get("totalCount") or 0)
        if page_total != expected_total:
            raise RuntimeError(
                f"[graphql] {owner}/{name}#{number}: review thread "
                f"{thread_id} comment totalCount changed during pagination "
                f"({expected_total}->{page_total})"
            )
        for node in connection.get("nodes", []):
            comment_id = node.get("id")
            if isinstance(comment_id, str) and comment_id:
                if comment_id in seen_comment_ids:
                    raise RuntimeError(
                        f"[graphql] {owner}/{name}#{number}: duplicate review "
                        f"comment during thread pagination: {comment_id}"
                    )
                seen_comment_ids.add(comment_id)
            comments.append(_normalize_review_comment(node))
        page_info = connection.get("pageInfo") or {}
        has_next = bool(page_info.get("hasNextPage"))
        cursor = page_info.get("endCursor")
        if has_next and (not isinstance(cursor, str) or not cursor):
            raise RuntimeError(
                f"[graphql] {owner}/{name}#{number}: review thread "
                f"{thread_id} hasNextPage without an endCursor"
            )
        return has_next, cursor

    has_next, cursor = consume(first_connection)
    while has_next:
        payload = _request_graphql_page(
            pool,
            {"threadId": thread_id, "cursor": cursor},
            context=f"{owner}/{name}#{number} review-thread {thread_id}",
            max_retries=max_retries,
            post_fn=post_fn,
        )
        node = (payload.get("data") or {}).get("node")
        if not isinstance(node, dict) or node.get("id") != thread_id:
            raise RuntimeError(
                f"[graphql] {owner}/{name}#{number}: review thread "
                f"{thread_id} disappeared or changed identity during pagination"
            )
        has_next, cursor = consume(node.get("comments") or {})

    if len(comments) != expected_total:
        raise RuntimeError(
            f"[graphql] {owner}/{name}#{number}: review thread {thread_id} "
            f"comment pagination incomplete ({len(comments)}/{expected_total})"
        )
    return comments


def fetch_pr(pool: TokenPool, owner: str, name: str, number: int,
             max_retries: int = 8,
             post_fn: Callable[[str, dict], tuple[int, dict, dict]] = _post,
             thread_post_fn: Callable[
                 [str, dict], tuple[int, dict, dict]
             ] = _post_thread_comments) -> dict:
    """Fetch one PR, paginating connections. RAISE on auth/quota exhaustion."""
    comments: list[dict] = []
    reviews: list[dict] = []
    linked: list[dict] = []
    title = body = ""
    merge_sha = None

    c_comments = c_reviews = c_threads = c_linked = None
    comments_active = reviews_active = threads_active = linked_active = True
    seen_first_page = False
    expected_comments = expected_reviews = expected_threads = None
    expected_linked = None
    seen_thread_ids: set[str] = set()

    while True:
        variables = {
            "owner": owner, "name": name, "number": number,
            "cComments": c_comments, "cReviews": c_reviews, "cThreads": c_threads,
            "cLinked": c_linked,
        }
        jb = _request_graphql_page(
            pool,
            variables,
            context=f"{owner}/{name}#{number}",
            max_retries=max_retries,
            post_fn=post_fn,
        )

        pr = (jb.get("data") or {}).get("repository", {})
        pr = (pr or {}).get("pullRequest")
        if pr is None:
            # Real outcome: GitHub has no such PR. Signal a logged MISS.
            return {"__miss__": True, "repo": f"{owner}/{name}", "pr_number": number}

        if not seen_first_page:
            title = pr.get("title") or ""
            body = pr.get("body") or ""
            mc = pr.get("mergeCommit")
            merge_sha = mc.get("oid") if mc else None
            seen_first_page = True

        more = False
        if comments_active:
            cn = pr.get("comments") or {}
            page_expected_comments = int(cn.get("totalCount") or 0)
            if expected_comments is None:
                expected_comments = page_expected_comments
            elif expected_comments != page_expected_comments:
                raise RuntimeError(
                    f"[graphql] {owner}/{name}#{number}: comment totalCount changed "
                    f"during pagination ({expected_comments}->{page_expected_comments})"
                )
            for n in cn.get("nodes", []):
                comments.append({
                    "user": (n.get("author") or {}).get("login", ""),
                    "body": n.get("body", ""), "created_at": n.get("createdAt", ""),
                    "kind": "comment",
                })
            cpi = cn.get("pageInfo") or {}
            comments_active = bool(cpi.get("hasNextPage"))
            if comments_active:
                c_comments = cpi.get("endCursor")
                more = True

        if reviews_active:
            rv = pr.get("reviews") or {}
            page_expected_reviews = int(rv.get("totalCount") or 0)
            if expected_reviews is None:
                expected_reviews = page_expected_reviews
            elif expected_reviews != page_expected_reviews:
                raise RuntimeError(
                    f"[graphql] {owner}/{name}#{number}: review totalCount changed "
                    f"during pagination ({expected_reviews}->{page_expected_reviews})"
                )
            for n in rv.get("nodes", []):
                reviews.append({
                    "user": (n.get("author") or {}).get("login", ""),
                    "state": n.get("state", ""), "body": n.get("body", ""),
                    "created_at": n.get("submittedAt", ""),
                })
            rpi = rv.get("pageInfo") or {}
            reviews_active = bool(rpi.get("hasNextPage"))
            if reviews_active:
                c_reviews = rpi.get("endCursor")
                more = True

        if threads_active:
            rt = pr.get("reviewThreads") or {}
            page_expected_threads = int(rt.get("totalCount") or 0)
            if expected_threads is None:
                expected_threads = page_expected_threads
            elif expected_threads != page_expected_threads:
                raise RuntimeError(
                    f"[graphql] {owner}/{name}#{number}: review-thread totalCount "
                    f"changed during pagination "
                    f"({expected_threads}->{page_expected_threads})"
                )
            for thread in rt.get("nodes", []):
                thread_id = thread.get("id")
                if not isinstance(thread_id, str) or not thread_id:
                    raise RuntimeError(
                        f"[graphql] {owner}/{name}#{number}: review thread lacks id"
                    )
                if thread_id in seen_thread_ids:
                    raise RuntimeError(
                        f"[graphql] {owner}/{name}#{number}: duplicate review "
                        f"thread during pagination: {thread_id}"
                    )
                seen_thread_ids.add(thread_id)
                thread_comments = thread.get("comments") or {}
                comments.extend(
                    _fetch_complete_thread_comments(
                        pool,
                        owner=owner,
                        name=name,
                        number=number,
                        thread_id=thread_id,
                        first_connection=thread_comments,
                        max_retries=max_retries,
                        post_fn=thread_post_fn,
                    )
                )
            tpi = rt.get("pageInfo") or {}
            threads_active = bool(tpi.get("hasNextPage"))
            if threads_active:
                c_threads = tpi.get("endCursor")
                more = True

        if linked_active:
            li_connection = pr.get("closingIssuesReferences") or {}
            page_expected_linked = int(li_connection.get("totalCount") or 0)
            if expected_linked is None:
                expected_linked = page_expected_linked
            elif expected_linked != page_expected_linked:
                raise RuntimeError(
                    f"[graphql] {owner}/{name}#{number}: linked-issue totalCount "
                    f"changed during pagination ({expected_linked}->"
                    f"{page_expected_linked})"
                )
            for li in li_connection.get("nodes", []):
                linked.append({
                    "number": li["number"],
                    "title": li.get("title", ""),
                    "body": li.get("body", ""),
                })
            lpi = li_connection.get("pageInfo") or {}
            linked_active = bool(lpi.get("hasNextPage"))
            if linked_active:
                c_linked = lpi.get("endCursor")
                more = True

        if not more:
            break

    # Comments arrived across two connections; order by created_at.
    comments.sort(key=lambda c: c.get("created_at") or "")
    reviews.sort(key=lambda r: r.get("created_at") or "")
    plain_comment_count = sum(
        1 for comment in comments if comment.get("kind") == "comment"
    )
    if plain_comment_count != int(expected_comments or 0):
        raise RuntimeError(
            f"[graphql] {owner}/{name}#{number}: comment pagination incomplete "
            f"({plain_comment_count}/{expected_comments})"
        )
    if len(reviews) != int(expected_reviews or 0):
        raise RuntimeError(
            f"[graphql] {owner}/{name}#{number}: review pagination incomplete "
            f"({len(reviews)}/{expected_reviews})"
        )
    if len(seen_thread_ids) != int(expected_threads or 0):
        raise RuntimeError(
            f"[graphql] {owner}/{name}#{number}: review-thread pagination incomplete "
            f"({len(seen_thread_ids)}/{expected_threads})"
        )
    if len(linked) != int(expected_linked or 0):
        raise RuntimeError(
            f"[graphql] {owner}/{name}#{number}: linked-issue projection incomplete "
            f"({len(linked)}/{expected_linked})"
        )
    return {
        "repo": f"{owner}/{name}", "pr_number": number,
        "merge_commit_sha": merge_sha, "pr_title": title, "pr_body": body,
        "comments": comments, "reviews": reviews, "linked_issues": linked,
    }


def load_tokens(path: str | None, use_gh_cli: bool = False) -> list[str]:
    toks: list[str] = []
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"[graphql] --tokens file not found: {path}")
        with open(path) as f:
            toks.extend(
                ln.strip()
                for ln in f
                if ln.strip() and not ln.startswith("#")
            )
    if use_gh_cli:
        try:
            completed = subprocess.run(
                ["gh", "auth", "token"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise RuntimeError(f"gh auth token failed: {exc}") from exc
        token = completed.stdout.strip()
        if token:
            toks.append(token)
    deduplicated = list(dict.fromkeys(toks))
    if not deduplicated:
        raise RuntimeError(
            "no tokens loaded (need --tokens file and/or gh CLI login)"
        )
    return deduplicated



# Root compatibility for the established targeted fetch API.
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



def _graphql_post(token: str, query: str, variables: dict) -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GraphQL PR ingestion requires the requests package for live HTTP calls"
        ) from exc
    return requests.post(
        GQL_URL,
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


def fetch_one(store: pr_store.PRStore, rotator: TokenRotator, owner: str, name: str,
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



def _canonical_target_set_sha256(
    targets: list[tuple[str, int]] | tuple[tuple[str, int], ...],
) -> str:
    normalized = [
        {"repo": repo, "pr_number": number}
        for repo, number in sorted(
            {(str(repo), int(number)) for repo, number in targets}
        )
    ]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_gap_completion_receipt(
    *,
    targets: tuple[tuple[str, int], ...],
    completed: tuple[tuple[str, int], ...],
    completed_record_sha256s: dict[tuple[str, int], str],
    misses: tuple[tuple[str, int], ...],
    skipped: tuple[tuple[str, int], ...],
) -> dict:
    target_set = set(targets)
    completed_set = set(completed)
    miss_set = set(misses)
    skipped_set = set(skipped)
    unresolved = target_set - completed_set
    hashes_complete = (
        set(completed_record_sha256s) == completed_set
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in completed_record_sha256s.values()
        )
    )
    status = (
        "verified"
        if completed_set == target_set
        and not miss_set
        and not skipped_set
        and hashes_complete
        else "incomplete"
    )
    return {
        "schema": "cppmega_pr_gap_completion_v1",
        "status": status,
        "targets_sha256": _canonical_target_set_sha256(tuple(target_set)),
        "target_count": len(target_set),
        "completed_count": len(completed_set),
        "miss_count": len(miss_set),
        "skipped_count": len(skipped_set),
        "unresolved_count": len(unresolved),
        "completed": [
            {
                "repo": repo,
                "pr_number": number,
                "record_sha256": completed_record_sha256s.get((repo, number)),
            }
            for repo, number in sorted(completed_set)
        ],
        "misses": [
            {"repo": repo, "pr_number": number}
            for repo, number in sorted(miss_set)
        ],
        "skipped": [
            {"repo": repo, "pr_number": number}
            for repo, number in sorted(skipped_set)
        ],
    }


def _atomic_write_json(path: str, payload: dict) -> None:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp = f"{destination}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    directory_fd = os.open(os.path.dirname(destination), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_misses(
    path: str,
    misses: list[tuple[str, int]] | tuple[tuple[str, int], ...],
) -> None:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp = f"{destination}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        for repo, number in sorted(set(misses)):
            handle.write(
                json.dumps(
                    {"repo": repo, "pr_number": number},
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    directory_fd = os.open(os.path.dirname(destination), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_gap_resume_state(
    *,
    completion_receipt: str,
    targets: tuple[tuple[str, int], ...],
    conn: Any,
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], str]]:
    """Load only receipt-proven completed targets from a prior interrupted run.

    A row's mere presence in ``prs.sqlite`` is not completion evidence.  The
    prior receipt must bind the exact target set and each completed row's
    canonical content hash must still match the store.
    """

    if not os.path.exists(completion_receipt):
        return [], {}
    try:
        with open(completion_receipt, encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"[graphql] cannot resume from completion receipt "
            f"{completion_receipt}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise RuntimeError(
            f"[graphql] completion receipt must be an object: "
            f"{completion_receipt}"
        )
    if receipt.get("schema") != "cppmega_pr_gap_completion_v1":
        raise RuntimeError(
            f"[graphql] cannot resume unsupported completion receipt schema: "
            f"{receipt.get('schema')!r}"
        )

    target_set = set(targets)
    expected_target_hash = _canonical_target_set_sha256(targets)
    if (
        receipt.get("targets_sha256") != expected_target_hash
        or receipt.get("target_count") != len(target_set)
    ):
        raise RuntimeError(
            "[graphql] existing completion receipt is bound to a different "
            "target set; archive it before starting this gap-fill"
        )

    completed_raw = receipt.get("completed")
    if not isinstance(completed_raw, list):
        raise RuntimeError(
            "[graphql] existing completion receipt lacks a completed list"
        )
    completed: list[tuple[str, int]] = []
    completed_hashes: dict[tuple[str, int], str] = {}
    for index, item in enumerate(completed_raw):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"[graphql] completion receipt completed[{index}] is not an object"
            )
        repo = item.get("repo")
        number = item.get("pr_number")
        record_sha256 = item.get("record_sha256")
        target = (repo, number)
        if (
            not isinstance(repo, str)
            or not isinstance(number, int)
            or isinstance(number, bool)
            or target not in target_set
        ):
            raise RuntimeError(
                f"[graphql] completion receipt completed[{index}] is not in "
                f"the bound target set: {target!r}"
            )
        if target in completed_hashes:
            raise RuntimeError(
                f"[graphql] duplicate completed target in receipt: "
                f"{repo}#{number}"
            )
        if (
            not isinstance(record_sha256, str)
            or len(record_sha256) != 64
            or any(char not in "0123456789abcdef" for char in record_sha256)
        ):
            raise RuntimeError(
                f"[graphql] invalid record hash for resumed target "
                f"{repo}#{number}"
            )
        stored = pr_store.get_by_pr(conn, repo, number)
        if stored is None:
            raise RuntimeError(
                f"[graphql] receipt-proven target disappeared from store: "
                f"{repo}#{number}"
            )
        actual_sha256 = pr_store.record_content_sha256(stored)
        if actual_sha256 != record_sha256:
            raise RuntimeError(
                f"[graphql] receipt/store content hash mismatch for "
                f"{repo}#{number}; refusing to silently trust or overwrite "
                "corrupted resume state"
            )
        completed.append(target)
        completed_hashes[target] = record_sha256
    if receipt.get("completed_count") != len(completed):
        raise RuntimeError(
            "[graphql] completion receipt completed_count does not match its "
            "completed list"
        )
    return completed, completed_hashes


def _persist_gap_progress(
    *,
    completion_receipt: str,
    misses_path: str,
    targets: tuple[tuple[str, int], ...],
    completed: list[tuple[str, int]],
    completed_record_sha256s: dict[tuple[str, int], str],
    misses: list[tuple[str, int]],
    skipped: list[tuple[str, int]],
) -> dict:
    receipt = build_gap_completion_receipt(
        targets=targets,
        completed=tuple(completed),
        completed_record_sha256s=completed_record_sha256s,
        misses=tuple(misses),
        skipped=tuple(skipped),
    )
    # The misses file is diagnostic; publish it before the authoritative
    # receipt so a crash can never expose a newer receipt with older misses.
    _atomic_write_misses(misses_path, misses)
    _atomic_write_json(completion_receipt, receipt)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description="GitHub GraphQL PR gap-filler.")
    ap.add_argument("--store", required=True)
    ap.add_argument("--tokens", required=True, help="File of GitHub PATs, one/line.")
    ap.add_argument("--targets", required=True, help="JSONL of {repo,pr_number}.")
    ap.add_argument("--misses", required=True, help="JSONL output for PR misses.")
    ap.add_argument(
        "--completion-receipt",
        required=True,
        help="Atomic cppmega_pr_gap_completion_v1 receipt. The command exits "
             "non-zero unless every target was fetched completely.",
    )
    ap.add_argument("--skip-present", action="store_true",
                    help="Skip targets already in the store.")
    args = ap.parse_args()

    pool = TokenPool(load_tokens(args.tokens))
    conn = pr_store.connect(args.store, create=True)

    targets: list[tuple[str, int]] = []
    with open(args.targets, encoding="utf-8") as tf:
        for line_number, raw_line in enumerate(tf, start=1):
            line = raw_line.strip()
            if not line:
                continue
            target = json.loads(line)
            if not isinstance(target, dict):
                raise ValueError(
                    f"{args.targets}:{line_number}: target must be an object"
                )
            repo = target.get("repo")
            number = target.get("pr_number")
            if not isinstance(repo, str) or "/" not in repo:
                raise ValueError(
                    f"{args.targets}:{line_number}: invalid repo {repo!r}"
                )
            if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                raise ValueError(
                    f"{args.targets}:{line_number}: invalid pr_number {number!r}"
                )
            targets.append((repo, number))
    if len(set(targets)) != len(targets):
        raise ValueError(f"{args.targets}: duplicate PR gap targets are forbidden")

    target_tuple = tuple(targets)
    completed, completed_record_sha256s = load_gap_resume_state(
        completion_receipt=args.completion_receipt,
        targets=target_tuple,
        conn=conn,
    )
    resumed = len(completed)
    completed_set = set(completed)
    misses: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []
    receipt = _persist_gap_progress(
        completion_receipt=args.completion_receipt,
        misses_path=args.misses,
        targets=target_tuple,
        completed=completed,
        completed_record_sha256s=completed_record_sha256s,
        misses=misses,
        skipped=skipped,
    )
    try:
        for repo, number in targets:
            target = (repo, number)
            if target in completed_set:
                continue
            owner, name = repo.split("/", 1)

            if args.skip_present and pr_store.get_by_pr(conn, repo, number):
                skipped.append(target)
                receipt = _persist_gap_progress(
                    completion_receipt=args.completion_receipt,
                    misses_path=args.misses,
                    targets=target_tuple,
                    completed=completed,
                    completed_record_sha256s=completed_record_sha256s,
                    misses=misses,
                    skipped=skipped,
                )
                continue

            rec = fetch_pr(pool, owner, name, number)
            if rec.get("__miss__"):
                misses.append(target)
                receipt = _persist_gap_progress(
                    completion_receipt=args.completion_receipt,
                    misses_path=args.misses,
                    targets=target_tuple,
                    completed=completed,
                    completed_record_sha256s=completed_record_sha256s,
                    misses=misses,
                    skipped=skipped,
                )
                continue
            expected_record_sha256 = pr_store.record_content_sha256(rec)
            pr_store.upsert_record(conn, rec, replace_children=True)
            stored = pr_store.get_by_pr(conn, repo, number)
            if stored is None:
                raise RuntimeError(
                    f"[graphql] {repo}#{number}: record disappeared after upsert"
                )
            stored_record_sha256 = pr_store.record_content_sha256(stored)
            if stored_record_sha256 != expected_record_sha256:
                raise RuntimeError(
                    f"[graphql] {repo}#{number}: PR store cannot losslessly "
                    "represent the complete GraphQL record"
                )
            completed.append(target)
            completed_set.add(target)
            completed_record_sha256s[target] = expected_record_sha256
            receipt = _persist_gap_progress(
                completion_receipt=args.completion_receipt,
                misses_path=args.misses,
                targets=target_tuple,
                completed=completed,
                completed_record_sha256s=completed_record_sha256s,
                misses=misses,
                skipped=skipped,
            )
    finally:
        conn.close()
    print(
        f"[graphql] fetched={len(completed) - resumed} resumed={resumed} "
        f"completed={len(completed)} miss={len(misses)} "
        f"skipped={len(skipped)} misses->{args.misses} "
        f"completion->{args.completion_receipt} status={receipt['status']}",
        file=sys.stderr,
    )
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
