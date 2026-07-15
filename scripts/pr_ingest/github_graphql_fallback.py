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
      --misses out/graphql_misses.jsonl

  targets.jsonl: one JSON per line, {"repo":"owner/repo","pr_number":1234}
  (Typically produced by diffing the corpus PR set against `pr_store stats`.)
"""

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

if __package__ in (None, ""):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.pr_ingest import pr_store  # noqa: E402

GQL_URL = "https://api.github.com/graphql"

PR_QUERY = """
query($owner:String!, $name:String!, $number:Int!,
      $cComments:String, $cReviews:String, $cThreads:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      number
      title
      body
      mergeCommit { oid }
      comments(first:100, after:$cComments) {
        pageInfo { hasNextPage endCursor }
        nodes { author { login } body createdAt }
      }
      reviews(first:100, after:$cReviews) {
        pageInfo { hasNextPage endCursor }
        nodes { author { login } body state submittedAt }
      }
      reviewThreads(first:50, after:$cThreads) {
        pageInfo { hasNextPage endCursor }
        nodes { comments(first:100) {
          nodes { author { login } body path createdAt } } }
      }
      closingIssuesReferences(first:50) {
        nodes { number title body }
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


def _post(token: str, variables: dict) -> tuple[int, dict, dict]:
    """POST the GraphQL query. Return (status, headers, json_body)."""
    payload = json.dumps({"query": PR_QUERY, "variables": variables}).encode()
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


def fetch_pr(pool: TokenPool, owner: str, name: str, number: int,
             max_retries: int = 8) -> dict:
    """Fetch one PR, paginating connections. RAISE on auth/quota exhaustion."""
    comments: list[dict] = []
    reviews: list[dict] = []
    linked: list[dict] = []
    title = body = ""
    merge_sha = None

    c_comments = c_reviews = c_threads = None
    seen_first_page = False
    attempts = 0

    while True:
        variables = {
            "owner": owner, "name": name, "number": number,
            "cComments": c_comments, "cReviews": c_reviews, "cThreads": c_threads,
        }
        status, headers, jb = _post(pool.current(), variables)

        # --- Auth failures: fail loud across all tokens ---------------------
        if status in (401,):
            raise SystemExit(
                f"[graphql] 401 Unauthorized for token #{pool.idx} "
                f"({owner}/{name}#{number}); fix the PAT(s) in --tokens"
            )

        # --- Rate / abuse limits: rotate + backoff -------------------------
        remaining = headers.get("X-RateLimit-Remaining")
        retry_after = headers.get("Retry-After")
        is_secondary = status == 403 and (
            retry_after is not None
            or "secondary rate limit" in json.dumps(jb).lower()
            or "abuse" in json.dumps(jb).lower()
        )
        if status == 403 and not is_secondary and "rate limit" not in json.dumps(jb).lower():
            raise SystemExit(
                f"[graphql] 403 Forbidden (not a rate limit) for {owner}/{name}#{number}: "
                f"{jb.get('message', jb)}"
            )
        if is_secondary or (remaining is not None and remaining == "0") or status in (429,):
            attempts += 1
            if attempts > max_retries:
                raise SystemExit(
                    f"[graphql] exceeded {max_retries} retries on rate limits for "
                    f"{owner}/{name}#{number}; add more PATs or wait"
                )
            if retry_after is not None:
                wait = float(retry_after)
            else:
                reset = headers.get("X-RateLimit-Reset")
                wait = max(1.0, float(reset) - time.time()) if reset else 30.0
            pool.cool(wait)
            if not pool.advance():
                # Every token is cooling -> quota exhausted. Fail loud.
                soonest = min(pool.cooldown_until) - time.time()
                raise SystemExit(
                    f"[graphql] ALL tokens rate-limited (soonest reset in "
                    f"~{soonest:.0f}s); add more PATs or rerun later. "
                    f"FAILING LOUD rather than returning a partial store."
                )
            continue

        if status != 200:
            raise SystemExit(
                f"[graphql] HTTP {status} for {owner}/{name}#{number}: {jb}"
            )
        if "errors" in jb and jb["errors"]:
            raise SystemExit(
                f"[graphql] GraphQL errors for {owner}/{name}#{number}: {jb['errors']}"
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
            for li in (pr.get("closingIssuesReferences") or {}).get("nodes", []):
                linked.append({"number": li["number"], "title": li.get("title", ""),
                               "body": li.get("body", "")})
            seen_first_page = True

        cn = pr.get("comments") or {}
        for n in cn.get("nodes", []):
            comments.append({
                "user": (n.get("author") or {}).get("login", ""),
                "body": n.get("body", ""), "created_at": n.get("createdAt", ""),
                "kind": "comment",
            })
        rv = pr.get("reviews") or {}
        for n in rv.get("nodes", []):
            reviews.append({
                "user": (n.get("author") or {}).get("login", ""),
                "state": n.get("state", ""), "body": n.get("body", ""),
                "created_at": n.get("submittedAt", ""),
            })
        rt = pr.get("reviewThreads") or {}
        for thread in rt.get("nodes", []):
            for n in (thread.get("comments") or {}).get("nodes", []):
                comments.append({
                    "user": (n.get("author") or {}).get("login", ""),
                    "body": n.get("body", ""), "path": n.get("path"),
                    "created_at": n.get("createdAt", ""), "kind": "review_comment",
                })

        # Advance any connection that still has pages.
        more = False
        cpi = cn.get("pageInfo") or {}
        if cpi.get("hasNextPage"):
            c_comments = cpi.get("endCursor")
            more = True
        rpi = rv.get("pageInfo") or {}
        if rpi.get("hasNextPage"):
            c_reviews = rpi.get("endCursor")
            more = True
        tpi = rt.get("pageInfo") or {}
        if tpi.get("hasNextPage"):
            c_threads = tpi.get("endCursor")
            more = True
        if not more:
            break
        attempts = 0  # reset retry budget on a successful page

    # Comments arrived across two connections; order by created_at.
    comments.sort(key=lambda c: c.get("created_at") or "")
    reviews.sort(key=lambda r: r.get("created_at") or "")
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



def main() -> None:
    ap = argparse.ArgumentParser(description="GitHub GraphQL PR gap-filler.")
    ap.add_argument("--store", required=True)
    ap.add_argument("--tokens", required=True, help="File of GitHub PATs, one/line.")
    ap.add_argument("--targets", required=True, help="JSONL of {repo,pr_number}.")
    ap.add_argument("--misses", required=True, help="JSONL output for PR misses.")
    ap.add_argument("--skip-present", action="store_true",
                    help="Skip targets already in the store.")
    args = ap.parse_args()

    pool = TokenPool(load_tokens(args.tokens))
    conn = pr_store.connect(args.store, create=True)

    n_fetched = n_miss = n_skip = 0
    with open(args.targets) as tf, open(args.misses, "w") as mf:
        for line in tf:
            line = line.strip()
            if not line:
                continue
            tgt = json.loads(line)
            repo = tgt["repo"]
            number = int(tgt["pr_number"])
            owner, name = repo.split("/", 1)

            if args.skip_present and pr_store.get_by_pr(conn, repo, number):
                n_skip += 1
                continue

            rec = fetch_pr(pool, owner, name, number)
            if rec.get("__miss__"):
                mf.write(json.dumps({"repo": repo, "pr_number": number}) + "\n")
                n_miss += 1
                continue
            pr_store.upsert_record(conn, rec)
            n_fetched += 1

    print(
        f"[graphql] fetched={n_fetched} miss={n_miss} skipped={n_skip} "
        f"misses->{args.misses}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
