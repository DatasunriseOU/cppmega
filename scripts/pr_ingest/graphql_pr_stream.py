#!/usr/bin/env python3
"""Tier-2 PR ingest, component (5): GraphQL-PRIMARY whole-repo PR streamer.

GraphQL is the canonical whole-repository path. Where
``github_graphql_fallback.py`` fetches ONE (repo, pr_number) at a time for
explicit, receipt-bound gap filling, THIS module streams EVERY PR of EVERY repo
in ``repo_list.json`` by paginating the ``pullRequests`` connection, and folds
each PR into the same ``pr_store`` (by (repo,pr_number) AND by
(repo, mergeCommit.oid)).

For each repo we page::

  repository(owner,name){
    pullRequests(first:100, after:$cursor, states:[MERGED,CLOSED,OPEN],
                 orderBy:{field:CREATED_AT, direction:ASC}){
      nodes{ number mergeCommit{oid} title body
             comments(first:100){nodes{author{login} body createdAt}}
             reviews(first:50){nodes{author{login} state body submittedAt}}
             closingIssuesReferences(first:20){nodes{number title body}} }
      pageInfo{ hasNextPage endCursor } } }

Token rotation (REUSED from github_graphql_fallback.TokenPool): round-robin over
the 6 tokens (the 5 PATs in ``secrets/gh_tokens.txt`` PLUS the ``gh auth token``
of the logged-in CLI account). On a token's primary limit
(X-RateLimit-Remaining==0), secondary/abuse limit, or 429 we cool that token and
rotate to the next; when EVERY token is cooling we FAIL LOUD (RULE #1) after
recording the (repo, cursor) so a later run resumes mid-repo — we never silently
skip a repo on token exhaustion.

Checkpointing / resume (RULE #1: a crash must not lose progress, and must not
silently drop a repo):
  * A MANIFEST (JSON) records, per repo: status (pending|in_progress|done|
    fallback), the last successfully-committed ``endCursor``, and PR counts.
  * On (re)start we skip ``done`` repos and resume ``in_progress`` repos FROM
    their saved cursor. The manifest is rewritten atomically after each
    committed page, so resume is per-repo AND per-cursor.

Legacy fallback entries are non-terminal. They are retried only when
``--retry-fallback`` is explicit, and the process exits non-zero until every
expected repository is ``done``. The fallback thresholds default to disabled;
enabling them creates resumable work, never a successful completion receipt.
Every PR whose nested discussion connections exceed the inline page is recorded
in the persistent ``--truncated-targets`` set for the explicit gap filler.

Usage:
  python3 scripts/pr_ingest/graphql_pr_stream.py \
      --repo-list outputs/pr_ingest/repo_list.json \
      --store out/pr_store.sqlite \
      --tokens secrets/gh_tokens.txt \
      --manifest out/graphql_stream_manifest.json \
      --fallback-list out/graphql_fallback_repos.jsonl \
      --truncated-targets out/graphql_truncated_targets.jsonl

  # Validate on ONE explicit repo before repo_list.json exists:
  python3 scripts/pr_ingest/graphql_pr_stream.py \
      --repo tilelang/tilelang --store out/pr_store.sqlite \
      --tokens secrets/gh_tokens.txt --manifest out/m.json \
      --fallback-list out/fb.jsonl --max-prs 5
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

# Package-safe for tests/imports; direct-script execution adds only the repo root.
if __package__ in (None, ""):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.pr_ingest import pr_store  # noqa: E402
from scripts.pr_ingest.github_graphql_fallback import load_tokens  # noqa: E402

GQL_URL = "https://api.github.com/graphql"

# One bounded page of a repo's pullRequests connection. Review-thread comments
# must be fetched inline too: treating every PR with any review thread as a gap
# target produced hundreds of thousands of unnecessary per-PR follow-up calls.
# The 50 x (20 x 50) nested bound stays well below GitHub's 500,000-node query
# cap; only a connection that actually exceeds an inline bound is routed to the
# exact per-PR gap filler.
REPO_PR_QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:50, after:$cursor,
                 states:[MERGED,CLOSED,OPEN],
                 orderBy:{field:CREATED_AT, direction:ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        mergeCommit { oid }
        comments(first:100) {
          totalCount
          nodes { author { login } body createdAt }
        }
        reviews(first:50) {
          totalCount
          nodes { author { login } state body submittedAt }
        }
        reviewThreads(first:20) {
          totalCount
          nodes {
            id
            comments(first:50) {
              totalCount
              nodes { id author { login } body path createdAt }
            }
          }
        }
        closingIssuesReferences(first:20) {
          totalCount
          nodes { number title body }
        }
      }
    }
  }
}
"""
GRAPHQL_MANIFEST_SCHEMA = "cppmega_graphql_pr_stream_manifest_v4"
GRAPHQL_STREAM_SEMANTICS_VERSION = 4
GRAPHQL_QUERY_CONTRACT_SHA256 = hashlib.sha256(
    (
        REPO_PR_QUERY
        + f"\nsemantic-normalization-v{GRAPHQL_STREAM_SEMANTICS_VERSION}\n"
    ).encode("utf-8")
).hexdigest()


# --------------------------------------------------------------------------- #
# Token loading: 5 PATs from the file + the gh CLI account token = 6 tokens.   #
# --------------------------------------------------------------------------- #
def load_all_tokens(tokens_file: str, include_gh_cli: bool = True) -> list[str]:
    """Reuse github_graphql_fallback.load_tokens, then append `gh auth token`.

    RULE #1: if --include-gh-cli is on and `gh auth token` cannot be obtained we
    RAISE (we do not silently run with fewer tokens than the user asked for).
    """
    toks = load_tokens(tokens_file)
    if include_gh_cli:
        try:
            out = subprocess.run(
                ["gh", "auth", "token"],
                check=True, capture_output=True, text=True, timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise SystemExit(
                f"[graphql-stream] --include-gh-cli set but `gh auth token` "
                f"failed: {e}. Run `gh auth login`, or pass --no-gh-cli."
            )
        gh_tok = (out.stdout or "").strip()
        if not gh_tok:
            raise SystemExit(
                "[graphql-stream] `gh auth token` returned empty; "
                "run `gh auth login` or pass --no-gh-cli."
            )
        if gh_tok not in toks:
            toks.append(gh_tok)
    if not toks:
        raise SystemExit("[graphql-stream] no tokens available (file + gh cli both empty)")
    return toks


# --------------------------------------------------------------------------- #
# Manifest: per-repo resumable checkpoint.                                     #
# --------------------------------------------------------------------------- #
class Manifest:
    """Per-(repo, endCursor) resumable checkpoint, persisted atomically."""

    def __init__(self, path: str, *, restart_query_contract: bool = False):
        self.path = path
        self._lock = threading.RLock()
        self.restart_tag: str | None = None
        empty = {
            "schema": GRAPHQL_MANIFEST_SCHEMA,
            "query_contract_sha256": GRAPHQL_QUERY_CONTRACT_SHA256,
            "scan_id": hashlib.sha256(
                (
                    f"{GRAPHQL_QUERY_CONTRACT_SHA256}:"
                    f"{time.time_ns()}:{os.getpid()}"
                ).encode("utf-8")
            ).hexdigest(),
            "repos": {},
        }
        self.data: dict = dict(empty)
        if os.path.exists(path):
            with open(path) as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError as e:
                    raise SystemExit(
                        f"[graphql-stream] manifest {path} is corrupt JSON: {e}. "
                        f"Move it aside to start fresh (RULE #1: not silently overwriting)."
                    )
            contract_matches = (
                self.data.get("schema") == GRAPHQL_MANIFEST_SCHEMA
                and self.data.get("query_contract_sha256")
                == GRAPHQL_QUERY_CONTRACT_SHA256
            )
            loaded_scan_id = self.data.get("scan_id")
            scan_id_matches = (
                isinstance(loaded_scan_id, str)
                and len(loaded_scan_id) == 64
                and all(
                    char in "0123456789abcdef"
                    for char in loaded_scan_id
                )
            )
            if (
                not contract_matches or not scan_id_matches
            ) and restart_query_contract:
                self.restart_tag = (
                    f"{GRAPHQL_MANIFEST_SCHEMA}-"
                    f"{int(time.time())}-{os.getpid()}"
                )
                backup = f"{path}.pre-{self.restart_tag}.json"
                os.replace(path, backup)
                sys.stderr.write(
                    "[graphql-stream] archived incompatible manifest to "
                    f"{backup}; restarting every repo under query contract "
                    f"{GRAPHQL_QUERY_CONTRACT_SHA256}\n"
                )
                self.data = dict(empty)
            elif not contract_matches:
                raise SystemExit(
                    "[graphql-stream] manifest query contract is missing or stale; "
                    "refusing to reuse done/cursor state produced without complete "
                    "review-thread and linked-issue accounting. Rerun once with "
                    "--restart-query-contract to archive it and rescan every repo."
                )
            elif not scan_id_matches:
                raise SystemExit(
                    f"[graphql-stream] manifest {path} has invalid scan_id; "
                    "rerun once with --restart-query-contract to archive it "
                    "and rescan every repo"
                )
            if "repos" not in self.data:
                raise SystemExit(
                    f"[graphql-stream] manifest {path} missing 'repos' key (wrong file?)"
                )
        scan_id = self.data.get("scan_id")
        if (
            not isinstance(scan_id, str)
            or len(scan_id) != 64
            or any(char not in "0123456789abcdef" for char in scan_id)
        ):
            raise SystemExit(
                f"[graphql-stream] manifest {path} has invalid scan_id; "
                "restart the query contract instead of mixing scan identities"
            )

    @property
    def scan_id(self) -> str:
        return str(self.data["scan_id"])

    def get(self, repo: str) -> dict:
        with self._lock:
            return dict(self.data["repos"].get(repo, {}))

    def status(self, repo: str) -> str:
        return self.get(repo).get("status", "pending")

    def cursor(self, repo: str):
        return self.get(repo).get("cursor")

    def update(self, repo: str, **fields) -> None:
        with self._lock:
            rec = self.data["repos"].setdefault(repo, {})
            rec.update(fields)
            self._flush()

    def _flush(self) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)  # atomic on POSIX


def archive_query_bound_side_files(
    manifest: Manifest,
    paths: tuple[str, ...],
) -> list[str]:
    """Archive side files whose contents were produced by a stale query."""

    if manifest.restart_tag is None:
        return []
    archived: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        backup = f"{path}.pre-{manifest.restart_tag}"
        os.replace(path, backup)
        archived.append(backup)
        sys.stderr.write(
            f"[graphql-stream] archived query-bound side file to {backup}\n"
        )
    return archived


class StreamAborted(Exception):
    """Raised inside a worker when a sibling worker hit a fatal condition.

    The shared ``stop_event`` is set by the failing worker; every other worker
    checks it between pages / HTTP attempts and raises this so the pool tears
    down promptly instead of hanging on ``shutdown(wait=True)``."""


class SharedTokenPool:
    """ONE thread-safe token pool shared by ALL workers (fixes the H4 double-spend).

    The previous per-worker ``TokenPool`` gave every worker a PRIVATE
    ``cooldown_until``, so a 401/rate-limit cooldown on a PAT was invisible to
    the other workers -- two workers would keep hammering the SAME PAT past its
    GLOBAL (per-token) GitHub limit, and one worker's cooldown could not stop the
    others. This single pool serializes ALL token state (rotation + cooldowns)
    under one lock so a cooldown set by any worker is honored by every worker,
    and ``AllTokensExhausted`` is raised only when every token is genuinely
    cooling (RULE #1: fail loud, no silent double-spend).
    """

    def __init__(self, tokens: list[str]):
        toks = list(tokens)
        if not toks:
            raise SystemExit("[graphql-stream] SharedTokenPool got no tokens")
        self.tokens = toks
        self.cooldown_until = [0.0] * len(toks)
        self._idx = 0
        self._lock = threading.Lock()

    def acquire(self) -> tuple[int, str]:
        """Reserve the next non-cooling token (round-robin across workers).

        Returns ``(idx, token)``. RAISES :class:`AllTokensExhausted` when every
        token is cooling, carrying the soonest reset so the caller can report it.
        """
        with self._lock:
            now = time.time()
            n = len(self.tokens)
            for _ in range(n):
                i = self._idx
                self._idx = (self._idx + 1) % n
                if self.cooldown_until[i] <= now:
                    return i, self.tokens[i]
            soonest = min(self.cooldown_until) - now
            raise AllTokensExhausted(soonest)

    def cool(self, idx: int, seconds: float) -> None:
        """Cool token ``idx`` for ``seconds`` -- visible to ALL workers at once."""
        with self._lock:
            self.cooldown_until[idx] = time.time() + max(1.0, seconds)


# --------------------------------------------------------------------------- #
# HTTP POST with the same rate-limit semantics as the fallback gap-filler.     #
# --------------------------------------------------------------------------- #
def _post(token: str, variables: dict) -> tuple[int, dict, dict]:
    payload = json.dumps({"query": REPO_PR_QUERY, "variables": variables}).encode()
    req = urllib.request.Request(GQL_URL, data=payload, method="POST")
    req.add_header("Authorization", f"bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "cppmega-pr-ingest")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        headers = dict(e.headers or {})
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"message": str(e)}
        return e.code, headers, body


class AllTokensExhausted(Exception):
    """Raised when every token is cooling. Carries the soonest reset (s)."""

    def __init__(self, soonest_s: float):
        super().__init__(f"all tokens rate-limited; soonest reset ~{soonest_s:.0f}s")
        self.soonest_s = soonest_s


class RepoRateLimited(Exception):
    """Soft signal: this page kept rate-limiting; caller may route to fallback."""


def _post_with_rotation(pool: "SharedTokenPool", variables: dict, owner: str, name: str,
                        max_retries: int,
                        stop_event: threading.Event | None = None,
                        post_fn: Callable[
                            [str, dict], tuple[int, dict, dict]
                        ] = _post) -> dict:
    """POST one page via the SHARED pool, rotating tokens on rate limits.

    Returns the JSON body on success. RAISES:
      * StreamAborted when ``stop_event`` was set by a sibling worker's fatal
        failure (so we abort promptly instead of hanging the pool).
      * SystemExit on auth/other-403/GraphQL-errors (fail loud — real bug).
      * AllTokensExhausted when every (shared) token is cooling (caller records
        cursor and re-raises as a loud failure).
      * RepoRateLimited when we burn the per-page retry budget on THIS repo
        (caller decides: route to GH Archive fallback, do NOT abort the stream).
    """
    attempts = 0
    unauthorized = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            raise StreamAborted(f"{owner}/{name}: aborted by sibling worker failure")
        # Reserve a usable token from the SHARED pool (raises AllTokensExhausted
        # when every token is cooling) -- cooldowns set below are seen by all.
        idx, token = pool.acquire()
        status, headers, jb = post_fn(token, variables)

        if status == 401:
            unauthorized += 1
            sys.stderr.write(
                f"[graphql-stream] token #{idx} unauthorized for "
                f"{owner}/{name}; cooling it for ALL workers\n"
            )
            sys.stderr.flush()
            pool.cool(idx, 24 * 3600)
            if unauthorized >= len(pool.tokens):
                raise SystemExit(
                    f"[graphql-stream] all tokens unauthorized while streaming "
                    f"{owner}/{name}; fix the PAT(s)/gh token"
                )
            continue

        remaining = headers.get("X-RateLimit-Remaining")
        retry_after = headers.get("Retry-After")
        jb_lower = json.dumps(jb).lower()
        graphql_rate_limited = any(
            isinstance(error, dict) and error.get("type") == "RATE_LIMITED"
            for error in jb.get("errors", ())
        )
        is_secondary = status == 403 and (
            retry_after is not None
            or "secondary rate limit" in jb_lower
            or "abuse" in jb_lower
        )
        if status == 403 and not is_secondary and "rate limit" not in jb_lower:
            raise SystemExit(
                f"[graphql-stream] 403 Forbidden (not a rate limit) for "
                f"{owner}/{name}: {jb.get('message', jb)}"
            )
        if (
            graphql_rate_limited
            or is_secondary
            or (remaining is not None and remaining == "0")
            or status == 429
        ):
            attempts += 1
            if attempts > max_retries:
                # Burned the per-page budget on THIS repo -> soft signal up.
                raise RepoRateLimited()
            if retry_after is not None:
                wait = float(retry_after)
            else:
                reset = headers.get("X-RateLimit-Reset")
                wait = max(1.0, float(reset) - time.time()) if reset else 30.0
            # Cool THIS token for all workers; the next loop's acquire() rotates
            # to the next usable token or raises AllTokensExhausted when none.
            pool.cool(idx, wait)
            continue

        if status != 200:
            raise SystemExit(
                f"[graphql-stream] HTTP {status} for {owner}/{name}: {jb}"
            )
        if jb.get("errors"):
            # A NOT_FOUND on the repository is a real, recordable outcome, not a
            # transient. Surface it loudly with the repo so the caller logs it.
            raise SystemExit(
                f"[graphql-stream] GraphQL errors for {owner}/{name}: {jb['errors']}"
            )
        return jb


# --------------------------------------------------------------------------- #
# Map a GraphQL PR node -> the pr_store.upsert_record() record shape.          #
# --------------------------------------------------------------------------- #
def _pr_node_to_record(repo: str, node: dict) -> tuple[dict, bool]:
    """Return (record, truncated). ``truncated`` is True when this PR had MORE
    nested records than an inline connection bound."""
    number = node["number"]
    mc = node.get("mergeCommit")
    merge_sha = mc.get("oid") if mc else None

    comments: list[dict] = []
    reviews: list[dict] = []
    linked: list[dict] = []
    truncated = False

    cn = node.get("comments") or {}
    for n in cn.get("nodes", []):
        comments.append({
            "user": (n.get("author") or {}).get("login", "") if n.get("author") else "",
            "body": n.get("body", "") or "",
            "created_at": n.get("createdAt", "") or "",
            "kind": "comment",
        })
    if (cn.get("totalCount") or 0) > len(cn.get("nodes", [])):
        truncated = True

    rv = node.get("reviews") or {}
    for n in rv.get("nodes", []):
        reviews.append({
            "user": (n.get("author") or {}).get("login", "") if n.get("author") else "",
            "state": n.get("state", "") or "",
            "body": n.get("body", "") or "",
            "created_at": n.get("submittedAt", "") or "",
        })
    if (rv.get("totalCount") or 0) > len(rv.get("nodes", [])):
        truncated = True

    review_threads = node.get("reviewThreads") or {}
    thread_nodes = review_threads.get("nodes", [])
    if (review_threads.get("totalCount") or 0) > len(thread_nodes):
        truncated = True
    for thread in thread_nodes:
        thread_comments = thread.get("comments")
        if not isinstance(thread_comments, dict):
            truncated = True
            continue
        inline_nodes = thread_comments.get("nodes", [])
        for comment in inline_nodes:
            comments.append(
                {
                    "user": (
                        (comment.get("author") or {}).get("login", "")
                        if comment.get("author")
                        else ""
                    ),
                    "body": comment.get("body", "") or "",
                    "path": comment.get("path"),
                    "created_at": comment.get("createdAt", "") or "",
                    "kind": "review_comment",
                }
            )
        if (thread_comments.get("totalCount") or 0) > len(inline_nodes):
            truncated = True

    linked_connection = node.get("closingIssuesReferences") or {}
    for li in linked_connection.get("nodes", []):
        linked.append({
            "number": li["number"],
            "title": li.get("title", "") or "",
            "body": li.get("body", "") or "",
        })
    if (linked_connection.get("totalCount") or 0) > len(
        linked_connection.get("nodes", [])
    ):
        truncated = True

    comments.sort(key=lambda c: c.get("created_at") or "")
    reviews.sort(key=lambda r: r.get("created_at") or "")
    rec = {
        "repo": repo,
        "pr_number": number,
        "merge_commit_sha": merge_sha,
        "pr_title": node.get("title", "") or "",
        "pr_body": node.get("body", "") or "",
        "comments": comments,
        "reviews": reviews,
        "linked_issues": linked,
    }
    return rec, truncated


# --------------------------------------------------------------------------- #
# Stream ONE repo, paginating + checkpointing per page.                        #
# --------------------------------------------------------------------------- #
def stream_repo(
    pool: "SharedTokenPool",
    conn,
    manifest: Manifest,
    repo: str,
    *,
    fallback_pr_threshold: int,
    fallback_ratelimit_trips: int,
    fallback_list_path: str,
    max_prs: int | None = None,
    max_retries: int = 8,
    truncated_targets_path: str | None = None,
    append_lock: threading.Lock | None = None,
    truncated_target_keys: set[tuple[str, int]] | None = None,
    stop_event: threading.Event | None = None,
    post_fn: Callable[[str, dict], tuple[int, dict, dict]] = _post,
) -> dict:
    """Stream all PRs of ``repo``. Resumes from manifest cursor if present.

    Returns a per-repo stats dict. RAISES AllTokensExhausted up to the caller
    (after the manifest already holds the resumable cursor). RAISES StreamAborted
    if ``stop_event`` is set (a sibling worker failed fatally) so the pool tears
    down promptly. Routes the repo to the GH Archive fallback (and returns) when
    the fallback heuristic trips.
    """
    if "/" not in repo:
        raise SystemExit(f"[graphql-stream] repo must be 'owner/name', got {repo!r}")
    owner, name = repo.split("/", 1)

    prior_status = manifest.status(repo)
    cursor = manifest.cursor(repo)  # None on a fresh repo, else resume point
    scan_id = manifest.scan_id
    if prior_status == "in_progress" and cursor is None:
        # The prior process may have committed a page before it could persist
        # that page's cursor, or may be retrying a terminal count mismatch.
        # Restarting the repo from the first page must also restart its exact
        # membership proof; stale content remains available but is unverified.
        conn.execute(
            "UPDATE prs SET scan_id=NULL WHERE repo=? AND scan_id=?",
            (repo, scan_id),
        )
        conn.commit()
    manifest.update(repo, status="in_progress")
    stats = {"repo": repo, "prs": 0, "truncated": 0, "pages": 0, "ratelimit_trips": 0}
    total_count = None

    while True:
        if stop_event is not None and stop_event.is_set():
            raise StreamAborted(f"{repo}: aborted by sibling worker failure")
        variables = {"owner": owner, "name": name, "cursor": cursor}
        try:
            jb = _post_with_rotation(pool, variables, owner, name, max_retries,
                                     stop_event=stop_event, post_fn=post_fn)
        except RepoRateLimited:
            stats["ratelimit_trips"] += 1
            if (
                fallback_ratelimit_trips > 0
                and stats["ratelimit_trips"] >= fallback_ratelimit_trips
            ):
                _route_to_fallback(
                    manifest, repo, fallback_list_path, cursor,
                    reason="ratelimit", stats=stats, total_count=total_count,
                    append_lock=append_lock,
                )
                return stats
            if fallback_ratelimit_trips <= 0:
                raise SystemExit(
                    f"[graphql-stream] {repo}: per-page rate-limit retry budget "
                    f"exhausted at cursor={cursor!r}; progress is saved. Automated "
                    "fallback is disabled, so rerun after reset or explicitly set "
                    "--fallback-ratelimit-trips."
                )
            # Another wall on the same page but under the trip cap: brief pause,
            # then retry the SAME cursor (no progress lost).
            time.sleep(2.0)
            continue

        prconn = ((jb.get("data") or {}).get("repository") or {}).get("pullRequests")
        if prconn is None:
            # repository present but pullRequests null is anomalous -> fail loud.
            raise SystemExit(
                f"[graphql-stream] {repo}: data.repository.pullRequests is null "
                f"(repo missing or no PR access): {jb}"
            )

        page_total_count = prconn.get("totalCount")
        if (
            not isinstance(page_total_count, int)
            or isinstance(page_total_count, bool)
            or page_total_count < 0
        ):
            raise SystemExit(
                f"[graphql-stream] {repo}: pullRequests.totalCount is invalid: "
                f"{page_total_count!r}"
            )
        if total_count is None:
            total_count = page_total_count
            # Fallback heuristic on size: a huge repo is routed to GH Archive at
            # the very first page, BEFORE we burn quota paging it here.
            if (
                fallback_pr_threshold > 0
                and total_count > fallback_pr_threshold
            ):
                _route_to_fallback(
                    manifest, repo, fallback_list_path, cursor,
                    reason="too_many_prs", stats=stats, total_count=total_count,
                    append_lock=append_lock,
                )
                return stats
        elif page_total_count != total_count:
            manifest.update(
                repo,
                status="in_progress",
                cursor=None,
                note=(
                    "pull request membership changed during scan: "
                    f"first_total_count={total_count} "
                    f"page_total_count={page_total_count}; "
                    "next resume restarts this repo from the first page"
                ),
            )
            raise SystemExit(
                f"[graphql-stream] {repo}: pull request membership changed "
                f"during scan: first_total_count={total_count} "
                f"page_total_count={page_total_count}; progress remains "
                "fail-closed and the next resume will rescan this repo"
            )

        nodes = prconn.get("nodes")
        page_info = prconn.get("pageInfo")
        if not isinstance(nodes, list) or any(
            not isinstance(node, dict) for node in nodes
        ):
            raise SystemExit(
                f"[graphql-stream] {repo}: pullRequests.nodes is not a list "
                "of objects"
            )
        if not isinstance(page_info, dict):
            raise SystemExit(
                f"[graphql-stream] {repo}: pullRequests.pageInfo is invalid"
            )
        has_next = page_info.get("hasNextPage")
        end_cursor = page_info.get("endCursor")
        if not isinstance(has_next, bool):
            raise SystemExit(
                f"[graphql-stream] {repo}: pageInfo.hasNextPage is invalid: "
                f"{has_next!r}"
            )
        if has_next and (
            not isinstance(end_cursor, str) or not end_cursor
        ):
            raise SystemExit(
                f"[graphql-stream] {repo}: a non-terminal page lacks a valid "
                "endCursor"
            )
        if not has_next and end_cursor is not None and not isinstance(
            end_cursor, str
        ):
            raise SystemExit(
                f"[graphql-stream] {repo}: terminal page endCursor is invalid: "
                f"{end_cursor!r}"
            )

        for node in nodes:
            rec, truncated = _pr_node_to_record(repo, node)
            pr_store.upsert_record(
                conn,
                rec,
                commit=False,
                replace_children=True,
                scan_id=scan_id,
            )
            stats["prs"] += 1
            if truncated:
                stats["truncated"] += 1
                if truncated_targets_path is None or truncated_target_keys is None:
                    raise SystemExit(
                        f"[graphql-stream] {repo}#{rec['pr_number']}: truncated "
                        "discussion requires --truncated-targets and exact gap-fill "
                        "accounting"
                    )
                target_key = (repo, int(rec["pr_number"]))
                should_append = False
                if append_lock is not None:
                    with append_lock:
                        if target_key not in truncated_target_keys:
                            truncated_target_keys.add(target_key)
                            should_append = True
                elif target_key not in truncated_target_keys:
                    truncated_target_keys.add(target_key)
                    should_append = True
                if should_append:
                    _append_jsonl(
                        truncated_targets_path,
                        {"repo": repo, "pr_number": rec["pr_number"]},
                        lock=append_lock,
                    )
        stats["pages"] += 1
        conn.commit()

        # Checkpoint AFTER the page's PRs are committed: resume continues from
        # this exact endCursor (per-repo + per-cursor resumability).
        cursor = end_cursor
        scanned_prs = int(
            conn.execute(
                "SELECT COUNT(*) FROM prs WHERE repo=? AND scan_id=?",
                (repo, scan_id),
            ).fetchone()[0]
        )
        terminal_count_mismatch = (
            not has_next
            and isinstance(total_count, int)
            and scanned_prs != total_count
        )
        manifest.update(
            repo,
            status="in_progress"
            if has_next or terminal_count_mismatch
            else "done",
            cursor=cursor if has_next else None,
            prs=scanned_prs,
            truncated=sum(
                1 for target_repo, _number in truncated_target_keys
                if target_repo == repo
            ),
            total_count=total_count,
            **(
                {
                    "note": (
                        "scan membership count mismatch at terminal page: "
                        f"scanned={scanned_prs} total_count={total_count}; "
                        "next resume restarts this repo from the first page"
                    )
                }
                if terminal_count_mismatch
                else {}
            ),
        )
        if terminal_count_mismatch:
            raise SystemExit(
                f"[graphql-stream] {repo}: scan membership count mismatch at "
                f"terminal page: scanned={scanned_prs} total_count={total_count}; "
                "progress remains fail-closed and the next resume will rescan "
                "this repo from the first page"
            )
        if max_prs is not None and stats["prs"] >= max_prs and has_next:
            # A GraphQL cursor addresses a complete page, not a prefix of its
            # nodes. Finish and commit the page before applying the test/debug
            # cap so resume always advances instead of replaying the same page.
            manifest.update(
                repo,
                status="in_progress",
                cursor=cursor,
                prs=scanned_prs,
                truncated=sum(
                    1 for target_repo, _number in truncated_target_keys
                    if target_repo == repo
                ),
                total_count=total_count,
                note=(
                    f"stopped after complete page at --max-prs={max_prs}; "
                    f"processed_this_run={stats['prs']}"
                ),
            )
            return stats
        if not has_next:
            break

    return stats


def _route_to_fallback(manifest: Manifest, repo: str, fallback_list_path: str,
                       cursor, *, reason: str, stats: dict, total_count,
                       append_lock: threading.Lock | None = None) -> None:
    """Mark repo for the GH Archive path WITHOUT aborting the stream."""
    manifest.update(
        repo, status="fallback", cursor=cursor,
        fallback_reason=reason, prs=stats["prs"], total_count=total_count,
    )
    _append_jsonl(fallback_list_path, {
        "repo": repo, "reason": reason, "cursor": cursor,
        "total_count": total_count, "prs_streamed": stats["prs"],
    }, lock=append_lock)
    stats["fallback"] = reason
    sys.stderr.write(
        f"[graphql-stream] {repo}: routed to GH Archive fallback ({reason}; "
        f"total_count={total_count}); continuing stream.\n"
    )
    sys.stderr.flush()


def _append_jsonl(path: str, obj: dict, *, lock: threading.Lock | None = None) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if lock is not None:
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(obj) + "\n")
        return
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def load_truncated_target_keys(path: str) -> set[tuple[str, int]]:
    """Load the durable unique gap-target set used for resume accounting."""

    if not os.path.exists(path):
        return set()
    targets: set[tuple[str, int]] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[graphql-stream] {path}:{line_number}: invalid truncated "
                    f"target JSON: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise SystemExit(
                    f"[graphql-stream] {path}:{line_number}: target must be an object"
                )
            repo = item.get("repo")
            number = item.get("pr_number")
            if (
                not isinstance(repo, str)
                or "/" not in repo
                or not isinstance(number, int)
                or isinstance(number, bool)
                or number < 1
            ):
                raise SystemExit(
                    f"[graphql-stream] {path}:{line_number}: invalid target {item!r}"
                )
            targets.add((repo, number))
    return targets


# --------------------------------------------------------------------------- #
# Repo list loading.                                                          #
# --------------------------------------------------------------------------- #
def load_repo_list(path: str) -> list[str]:
    """Load only GitHub ``owner_repo`` keys from a mixed-forge repo list."""
    if not os.path.exists(path):
        raise SystemExit(
            f"[graphql-stream] --repo-list not found: {path}. "
            f"Pass --repo owner/name to validate on a single repo instead."
        )
    with open(path) as f:
        data = json.load(f)
    repos = data.get("repos")
    if not isinstance(repos, list):
        raise SystemExit(
            f"[graphql-stream] {path} has no 'repos' list (wrong file?)"
        )
    out = []
    seen: set[str] = set()
    for index, r in enumerate(repos):
        if not isinstance(r, dict):
            raise SystemExit(
                f"[graphql-stream] {path}: repos[{index}] must be an object"
            )
        owner_repo = r.get("owner_repo")
        if owner_repo is None:
            continue
        if not isinstance(owner_repo, str) or not owner_repo:
            raise SystemExit(
                f"[graphql-stream] invalid owner_repo in repos[{index}]: {r}"
            )
        project_identity = r.get("project_identity")
        if (
            project_identity is not None
            and project_identity != owner_repo
        ):
            raise SystemExit(
                f"[graphql-stream] repos[{index}] has conflicting project_identity "
                f"{project_identity!r} and owner_repo {owner_repo!r}"
            )
        if owner_repo in seen:
            continue
        seen.add(owner_repo)
        out.append(owner_repo)
    if not out:
        raise SystemExit(f"[graphql-stream] {path} resolved zero repos")
    return out


def manifest_completion_summary(manifest: Manifest, repos: list[str]) -> dict:
    """Return an exact terminal-state summary for the requested repo set.

    A GraphQL stream invocation is successful only when every expected repo is
    ``done``.  ``fallback`` is intentionally non-terminal here: a separate
    reconciliation/completion receipt must prove any non-GraphQL source before
    training can consume it.
    """

    counts = {
        "done": 0,
        "fallback": 0,
        "in_progress": 0,
        "pending": 0,
        "other": 0,
    }
    incomplete: list[dict[str, str]] = []
    for repo in repos:
        status = manifest.status(repo)
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
        if status != "done":
            incomplete.append({"repo": repo, "status": status})
    return {
        "status": "complete" if not incomplete else "incomplete",
        "expected": len(repos),
        **counts,
        "incomplete_repos": incomplete,
    }



# Root compatibility: exact cursor semantics used by existing callers/tests.
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


def fetch_repo(
    store: pr_store.PRStore,
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



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-list", help="repo_list.json from build_repo_list.py")
    src.add_argument("--repo", help="single owner/name (validate before repo_list exists)")

    ap.add_argument("--store", required=True, help="pr_store SQLite path")
    ap.add_argument("--tokens", required=True, help="File of GitHub PATs, one/line")
    ap.add_argument("--manifest", required=True, help="Per-repo resume manifest JSON")
    ap.add_argument("--fallback-list", required=True,
                    help="JSONL of repos routed to the GH Archive fallback")
    ap.add_argument("--truncated-targets", required=True,
                    help="JSONL of (repo,pr_number) whose comments/reviews "
                         "overflowed one page (finish via github_graphql_fallback.py)")
    ap.add_argument("--no-gh-cli", dest="include_gh_cli", action="store_false",
                    help="Do NOT append `gh auth token` as the 6th token")
    ap.add_argument("--fallback-pr-threshold", type=int, default=0,
                    help="Route a repo to GH Archive if totalCount exceeds this. "
                         "0 disables automated fallback (production default).")
    ap.add_argument("--fallback-ratelimit-trips", type=int, default=0,
                    help="Route a repo to GH Archive after this many page-level "
                         "rate-limit walls. 0 disables automated fallback.")
    ap.add_argument(
        "--retry-fallback",
        action="store_true",
        help="Explicitly resume repos currently marked fallback from their saved "
             "cursor. With production defaults they stay on the GraphQL path.",
    )
    ap.add_argument(
        "--restart-query-contract",
        action="store_true",
        help=(
            "Archive an incompatible legacy manifest and rescan every repo under "
            "the current complete nested-connection query contract. The PR store "
            "is retained and authoritatively refreshed."
        ),
    )
    ap.add_argument("--max-retries", type=int, default=8,
                    help="Per-page rate-limit retry budget before a fallback trip")
    ap.add_argument("--max-prs", type=int,
                    help="Stop each repo after N PRs (validation/smoke)")
    ap.add_argument("--max-repos", type=int,
                    help="Process at most N repos this run (validation/smoke)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of repos to stream concurrently. Each worker "
                         "gets its own SQLite connection; ALL workers share ONE "
                         "token pool so per-PAT rate limits are accounted once.")
    args = ap.parse_args(argv)

    tokens = load_all_tokens(args.tokens, include_gh_cli=args.include_gh_cli)
    sys.stderr.write(f"[graphql-stream] token pool size = {len(tokens)}\n")
    # ONE shared, thread-safe token pool for ALL workers: a cooldown set by any
    # worker is honored by every worker, so concurrent workers cannot double-spend
    # the SAME PAT's global rate limit (H4 fix).
    pool = SharedTokenPool(tokens)
    # Cross-worker abort flag: a worker that hits a fatal condition sets this so
    # the others stop between pages instead of the pool hanging on shutdown.
    stop_event = threading.Event()
    # Open once in the parent to create schema and switch the DB into WAL before
    # workers open their own connections.
    parent_conn = pr_store.connect(args.store, create=True)
    parent_conn.close()
    manifest = Manifest(
        args.manifest,
        restart_query_contract=args.restart_query_contract,
    )
    archive_query_bound_side_files(
        manifest,
        (args.truncated_targets, args.fallback_list),
    )
    append_lock = threading.Lock()
    truncated_target_keys = load_truncated_target_keys(args.truncated_targets)

    expected_repos = [args.repo] if args.repo else load_repo_list(args.repo_list)
    repos = list(expected_repos)
    if args.max_repos is not None:
        repos = repos[: args.max_repos]

    totals = {"repos_done": 0, "repos_fallback": 0, "prs": 0, "truncated": 0}
    runnable: list[str] = []
    seen_runnable: set[str] = set()
    for repo in repos:
        if repo in seen_runnable:
            continue
        seen_runnable.add(repo)
        st = manifest.status(repo)
        if st == "done":
            sys.stderr.write(f"[graphql-stream] {repo}: already done, skipping\n")
            continue
        if st == "fallback":
            if args.retry_fallback:
                manifest.update(repo, status="in_progress")
                sys.stderr.write(
                    f"[graphql-stream] {repo}: explicitly retrying prior fallback "
                    f"from cursor={manifest.cursor(repo)!r}\n"
                )
            else:
                sys.stderr.write(
                    f"[graphql-stream] {repo}: already routed to fallback; "
                    "remaining incomplete (use --retry-fallback after choosing "
                    "the GraphQL completion path)\n"
                )
                totals["repos_fallback"] += 1
                continue
        runnable.append(repo)

    def run_one(repo: str, worker_index: int) -> dict:
        conn = pr_store.connect(
            args.store,
            create=False,
            initialize=False,
        )
        try:
            st = manifest.status(repo)
            sys.stderr.write(
                f"[graphql-stream] streaming {repo}"
                + (f" (resume cursor={manifest.cursor(repo)})" if st == "in_progress" else "")
                + f" worker={worker_index}\n"
            )
            sys.stderr.flush()
            return stream_repo(
                pool, conn, manifest, repo,
                fallback_pr_threshold=args.fallback_pr_threshold,
                fallback_ratelimit_trips=args.fallback_ratelimit_trips,
                fallback_list_path=args.fallback_list,
                max_prs=args.max_prs, max_retries=args.max_retries,
                truncated_targets_path=args.truncated_targets,
                append_lock=append_lock,
                truncated_target_keys=truncated_target_keys,
                stop_event=stop_event,
            )
        finally:
            conn.close()

    def record_stats(repo: str, stats: dict) -> None:
        totals["prs"] += stats["prs"]
        totals["truncated"] += stats["truncated"]
        if stats.get("fallback"):
            totals["repos_fallback"] += 1
        elif manifest.status(repo) == "done":
            totals["repos_done"] += 1
        sys.stderr.write(
            f"[graphql-stream] {repo}: prs={stats['prs']} pages={stats['pages']} "
            f"truncated={stats['truncated']} "
            f"{'FALLBACK=' + stats['fallback'] if stats.get('fallback') else 'done'}\n"
        )
        sys.stderr.flush()

    workers = max(1, int(args.workers or 1))
    if args.repo:
        workers = 1
    if workers == 1:
        for i, repo in enumerate(runnable):
            try:
                record_stats(repo, run_one(repo, i))
            except AllTokensExhausted as e:
                cur = manifest.cursor(repo)
                raise SystemExit(
                    f"[graphql-stream] ALL {len(tokens)} tokens rate-limited while "
                    f"streaming {repo} (soonest reset ~{e.soonest_s:.0f}s). "
                    f"Progress saved: manifest={args.manifest} repo={repo} "
                    f"cursor={cur!r}. Re-run the SAME command after the reset "
                    f"(or add more PATs) to RESUME mid-repo. FAILING LOUD per RULE #1."
                )
    else:
        # NOTE: do NOT use `with ThreadPoolExecutor(...)` here. Its __exit__ runs
        # shutdown(wait=True), which on a fatal error would BLOCK until every
        # already-submitted repo finishes streaming (a hang, not a fast abort).
        # Instead, on a fatal error we set stop_event (workers bail between pages)
        # and shutdown(wait=False, cancel_futures=True) so the process exits
        # promptly. FAILING LOUD per RULE #1, without hanging.
        executor = ThreadPoolExecutor(max_workers=workers)
        future_repo = {
            executor.submit(run_one, repo, i): repo
            for i, repo in enumerate(runnable)
        }
        try:
            for fut in as_completed(future_repo):
                repo = future_repo[fut]
                try:
                    record_stats(repo, fut.result())
                except StreamAborted:
                    # A sibling worker already hit the fatal condition and raises
                    # the loud SystemExit below; this worker just bailed cleanly.
                    continue
                except AllTokensExhausted as e:
                    stop_event.set()
                    cur = manifest.cursor(repo)
                    raise SystemExit(
                        f"[graphql-stream] ALL {len(tokens)} tokens rate-limited while "
                        f"streaming {repo} (soonest reset ~{e.soonest_s:.0f}s). "
                        f"Progress saved: manifest={args.manifest} repo={repo} "
                        f"cursor={cur!r}. Re-run the SAME command after the reset "
                        f"(or add more PATs) to RESUME mid-repo. FAILING LOUD per RULE #1."
                    )
        finally:
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)

    completion = manifest_completion_summary(manifest, expected_repos)
    sys.stderr.write(
        f"[graphql-stream] DONE repos_done={totals['repos_done']} "
        f"repos_fallback={totals['repos_fallback']} prs={totals['prs']} "
        f"truncated={totals['truncated']} store={args.store} "
        f"completion={completion['status']} "
        f"done={completion['done']}/{completion['expected']}\n"
    )
    if completion["status"] != "complete":
        sample = completion["incomplete_repos"][:20]
        sys.stderr.write(
            "[graphql-stream] INCOMPLETE: expected repo set still contains "
            f"non-terminal entries; sample={sample}. Returning non-zero.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
