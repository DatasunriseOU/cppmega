#!/usr/bin/env python3
"""Resumable, receipt-bound GitHub Actions workflow-run inventory.

This stage inventories workflow-run metadata only.  It deliberately does not
download logs, artifacts, or jobs.  GitHub caps a filtered workflow-run listing
at 1,000 results, so every repository starts with one explicit UTC ``[start,
end)`` window and dense windows are bisected recursively before pagination.

Progress is committed page-by-page to SQLite.  A completion receipt is emitted
only after the database proves that every repository has a gap-free,
non-overlapping set of closed leaf windows, every page/count closes exactly,
and the repository scope still matches the canonical input.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import zlib


SCHEMA_VERSION = "cppmega_ci_stream_inventory_v3"
RECEIPT_SCHEMA = "cppmega_ci_stream_inventory_receipt_v3"
PROGRESS_SCHEMA = "cppmega_ci_stream_inventory_progress_v3"
PREVIOUS_SCHEMA_VERSION = "cppmega_ci_stream_inventory_v2"
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_REPO_LIST = "../cppmega.mlx/outputs/pr_ingest/repo_list.json"
DEFAULT_PER_PAGE = 100
GITHUB_FILTER_LIMIT = 1000
METADATA_ENCODING = "zlib6-canonical-json-utf8-v1"
CONVERGENCE_MAX_PASSES = 64
MAX_UPGRADE_REASON_CHARS = 1000
IMPORTED_UPGRADE_REASON = (
    "imported pre-v3 inventory producer upgrade audit record"
)

_OWNER_REPO_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,99}))/"
    r"(?P<repo>[A-Za-z0-9_.-]{1,100})$"
)
class InventoryError(RuntimeError):
    """Base class for fail-closed inventory errors."""


class ScopeError(InventoryError):
    """The canonical repository list is malformed or unresolved."""


class BindingError(InventoryError):
    """An existing database does not match this invocation."""


class APIError(InventoryError):
    """GitHub returned a permanent or exhausted-retry error."""


class MalformedAPIError(APIError):
    """GitHub returned a response that cannot prove complete enumeration."""


class UnstableEnumerationError(APIError):
    """Repeated observations disagree, so no stable snapshot can be claimed."""


class PaginationDrift(UnstableEnumerationError):
    """A paginated leaf shifted and must be invalidated and subdivided."""

    def __init__(self, message: str, *, observed_total: int):
        super().__init__(message)
        self.observed_total = observed_total


class CompletionError(InventoryError):
    """The SQLite inventory cannot support a completion receipt."""


@dataclass(frozen=True)
class Repo:
    key: str
    owner: str
    name: str
    canonical: str
    ordinal: int


@dataclass(frozen=True)
class RepoScope:
    path: str
    source_sha256: str
    scope_sha256: str
    repos: tuple[Repo, ...]
    original_repo_count: int
    unresolved_count: int
    smoke: bool
    max_repos: int | None


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class PageResponse:
    total_count: int
    workflow_runs: tuple[dict[str, Any], ...]
    payload_sha256: str


@dataclass
class _TokenState:
    token: str
    remaining: int | None = None
    reset_epoch: float = 0.0
    cooldown_until: float = 0.0


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_upgrade_reason(value: str | None) -> str:
    if value is None:
        raise BindingError(
            "an explicit inventory script upgrade requires a reason"
        )
    reason = value.strip()
    if (
        not reason
        or len(reason) > MAX_UPGRADE_REASON_CHARS
        or any(not character.isprintable() for character in reason)
    ):
        raise BindingError(
            "inventory script upgrade reason must be non-empty printable text "
            f"of at most {MAX_UPGRADE_REASON_CHARS} characters"
        )
    return reason


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_instant(value: str) -> int:
    """Parse a second-precision UTC timestamp and return its Unix epoch."""

    raw = value.strip()
    if not raw:
        raise ValueError("UTC timestamp must not be empty")
    normalized = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp must include an explicit UTC offset: {value!r}")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"timestamp must be UTC, got {value!r}")
    if parsed.microsecond:
        raise ValueError(
            "GitHub workflow-run timestamps are second precision; "
            f"fractional boundary is not allowed: {value!r}"
        )
    return int(parsed.timestamp())


def format_utc_instant(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_owner_repo(value: str) -> tuple[str, str]:
    candidate = value.strip()
    if candidate.startswith("git@github.com:"):
        candidate = candidate[len("git@github.com:") :]
    elif "://" in candidate:
        parsed = urllib.parse.urlparse(candidate)
        if parsed.hostname is None or parsed.hostname.casefold() != "github.com":
            raise ScopeError(f"not a GitHub repository: {value!r}")
        candidate = urllib.parse.unquote(parsed.path).strip("/")
    elif "/" in candidate:
        parsed = urllib.parse.urlparse(f"//{candidate}")
        if parsed.netloc.casefold() == "github.com":
            candidate = urllib.parse.unquote(parsed.path).strip("/")
    candidate = candidate.strip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    match = _OWNER_REPO_RE.fullmatch(candidate)
    if match is None:
        raise ScopeError(f"invalid GitHub owner/repository: {value!r}")
    owner = match.group("owner")
    name = match.group("repo")
    if name in {".", ".."}:
        raise ScopeError(f"invalid GitHub repository name: {value!r}")
    return owner, name


def load_repo_scope(
    path: str | os.PathLike[str] = DEFAULT_REPO_LIST,
    *,
    smoke: bool = False,
    max_repos: int | None = None,
) -> RepoScope:
    """Load and case-insensitively deduplicate the canonical GitHub repo scope."""

    source_path = Path(path).expanduser().resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise ScopeError(f"cannot read repository list {source_path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScopeError(f"repository list is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ScopeError("repository list root must be an object")

    unresolved = document.get("unresolved")
    if not isinstance(unresolved, list):
        raise ScopeError("repository list must contain an 'unresolved' array")
    if unresolved:
        raise ScopeError(
            f"repository list has {len(unresolved)} unresolved entries; "
            "production inventory requires zero"
        )
    names = document.get("repo_names")
    if not isinstance(names, list) or not names:
        raise ScopeError("repository list must contain a non-empty 'repo_names' array")

    deduplicated: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(names):
        if not isinstance(item, str):
            raise ScopeError(f"repo_names[{index}] must be a string")
        owner, name = _normalize_owner_repo(item)
        key = f"{owner}/{name}".casefold()
        previous = deduplicated.get(key)
        if previous is not None:
            # GitHub names are case-insensitive.  Preserve the first spelling,
            # but reject a duplicate that is not actually the same identity.
            if (previous[0].casefold(), previous[1].casefold()) != (
                owner.casefold(),
                name.casefold(),
            ):
                raise ScopeError(f"ambiguous GitHub repository identity: {item!r}")
            continue
        deduplicated[key] = (owner, name)

    ordered_pairs = sorted(
        deduplicated.values(),
        key=lambda pair: (pair[0].casefold(), pair[1].casefold()),
    )
    original_repo_count = len(ordered_pairs)
    if max_repos is not None:
        if not smoke:
            raise ScopeError("--max-repos is allowed only with explicit --smoke")
        if max_repos <= 0:
            raise ScopeError("--max-repos must be positive")
        ordered_pairs = ordered_pairs[:max_repos]
    repos = tuple(
        Repo(
            key=f"{owner}/{name}".casefold(),
            owner=owner,
            name=name,
            canonical=f"{owner}/{name}",
            ordinal=ordinal,
        )
        for ordinal, (owner, name) in enumerate(ordered_pairs)
    )
    scope_hash = _hash_lines(repo.key for repo in repos)
    return RepoScope(
        path=str(source_path),
        source_sha256=_sha256_bytes(raw),
        scope_sha256=scope_hash,
        repos=repos,
        original_repo_count=original_repo_count,
        unresolved_count=0,
        smoke=smoke,
        max_repos=max_repos,
    )


def load_token_pool(
    token_file: str | os.PathLike[str] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Load newline-delimited tokens and append ``GH_TOKEN`` when present."""

    tokens: list[str] = []
    if token_file is not None:
        path = Path(token_file).expanduser()
        try:
            lines = path.read_text().splitlines()
        except OSError as exc:
            raise InventoryError(f"cannot read token pool {path}: {exc}") from exc
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tokens.append(stripped)
    env = os.environ if environ is None else environ
    env_token = env.get("GH_TOKEN", "").strip()
    if env_token:
        tokens.append(env_token)
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    if not unique:
        raise InventoryError(
            "no GitHub tokens available; provide --tokens and/or GH_TOKEN"
        )
    return unique


class TokenPool:
    """Thread-safe token rotation driven by observed rate-limit state."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        unique = list(dict.fromkeys(token.strip() for token in tokens if token.strip()))
        if not unique:
            raise ValueError("token pool must not be empty")
        self._states = [_TokenState(token=token) for token in unique]
        self._clock = clock
        self._sleeper = sleeper
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(state.token for state in self._states)

    def acquire(self) -> tuple[int, str]:
        with self._lock:
            now = self._clock()
            count = len(self._states)
            available = [
                (offset, self._states[(self._cursor + offset) % count])
                for offset in range(count)
                if self._states[(self._cursor + offset) % count].cooldown_until
                <= now
            ]
            if available:
                # The cursor provides fair rotation; cooldown state removes
                # exhausted tokens from the candidate set.
                offset, state = available[0]
                index = (self._cursor + offset) % count
                self._cursor = (index + 1) % count
                return index, state.token
            index = min(
                range(count), key=lambda item: self._states[item].cooldown_until
            )
            wait_seconds = max(0.0, self._states[index].cooldown_until - now)
        # Do not hold the pool lock while waiting.  A bounded sleep keeps
        # progress/status reporting responsive; the API response will enforce
        # the limit again if its reset time has not arrived.
        self._sleeper(min(wait_seconds, 60.0))
        with self._lock:
            self._cursor = (index + 1) % len(self._states)
            return index, self._states[index].token

    def observe(self, index: int, headers: Mapping[str, str]) -> None:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        with self._lock:
            state = self._states[index]
            try:
                state.remaining = int(lowered["x-ratelimit-remaining"])
            except (KeyError, ValueError):
                pass
            try:
                state.reset_epoch = float(lowered["x-ratelimit-reset"])
            except (KeyError, ValueError):
                pass
            if state.remaining == 0:
                state.cooldown_until = max(
                    state.cooldown_until,
                    state.reset_epoch or self._clock() + 60.0,
                )

    def rate_limited(
        self,
        index: int,
        headers: Mapping[str, str],
        *,
        secondary: bool,
    ) -> None:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        now = self._clock()
        retry_after = 0.0
        try:
            retry_after = max(0.0, float(lowered.get("retry-after", "0")))
        except ValueError:
            retry_after = 0.0
        try:
            reset = float(lowered.get("x-ratelimit-reset", "0"))
        except ValueError:
            reset = 0.0
        fallback = 60.0 if secondary else 5.0
        until = max(now + retry_after, reset, now + fallback)
        with self._lock:
            state = self._states[index]
            state.remaining = 0
            state.reset_epoch = reset
            state.cooldown_until = max(state.cooldown_until, until)


def _default_requester(
    method: str,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> HTTPResponse:
    request = urllib.request.Request(url, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HTTPResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HTTPResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            body=exc.read(),
        )


class GitHubClient:
    def __init__(
        self,
        token_pool: TokenPool,
        *,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_requester,
        sleeper: Callable[[float], None] = time.sleep,
        timeout: float = 60.0,
        max_attempts: int = 12,
        api_base: str = "https://api.github.com",
    ):
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.token_pool = token_pool
        self.requester = requester
        self.sleeper = sleeper
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.api_base = api_base.rstrip("/")

    def redact(self, message: object) -> str:
        text = str(message)
        for secret in self.token_pool.secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text[:4000]

    @staticmethod
    def _rate_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
        lowered = {str(key).casefold(): str(value) for key, value in headers.items()}
        return {
            "rate_remaining": lowered.get("x-ratelimit-remaining"),
            "rate_reset": lowered.get("x-ratelimit-reset"),
            "retry_after": lowered.get("retry-after"),
        }

    @staticmethod
    def _body_message(body: bytes) -> str:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:1000]
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)[:1000]
        return str(payload)[:1000]

    def get_workflow_runs(
        self,
        *,
        repo: Repo,
        start_epoch: int,
        end_epoch: int,
        page: int,
        per_page: int,
        ledger: Callable[..., None],
    ) -> PageResponse:
        if end_epoch <= start_epoch:
            raise ValueError("empty workflow-run search interval")
        created = (
            f"{format_utc_instant(start_epoch)}.."
            f"{format_utc_instant(end_epoch - 1)}"
        )
        endpoint = f"/repos/{repo.owner}/{repo.name}/actions/runs"
        query = urllib.parse.urlencode(
            {
                "created": created,
                "exclude_pull_requests": "false",
                "per_page": per_page,
                "page": page,
            }
        )
        url = f"{self.api_base}{endpoint}?{query}"

        for attempt in range(1, self.max_attempts + 1):
            token_index, token = self.token_pool.acquire()
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "cppmega-ci-stream-inventory/1",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            }
            started = time.monotonic()
            try:
                response = self.requester("GET", url, headers, self.timeout)
            except Exception as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                message = self.redact(exc)
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=None,
                    outcome="transport_retry",
                    latency_ms=elapsed,
                    error_class=type(exc).__name__,
                    error_message=message,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"transport retries exhausted for {repo.canonical} "
                        f"window {created}: {message}"
                    ) from exc
                self.sleeper(min(2 ** (attempt - 1), 30))
                continue

            elapsed = int((time.monotonic() - started) * 1000)
            self.token_pool.observe(token_index, response.headers)
            rate = self._rate_headers(response.headers)
            body_message = self._body_message(response.body)
            message_lower = body_message.casefold()
            remaining_zero = rate["rate_remaining"] == "0"
            secondary = (
                "secondary rate limit" in message_lower
                or "abuse detection" in message_lower
            )
            rate_limited = response.status == 429 or (
                response.status == 403
                and (remaining_zero or secondary or "rate limit" in message_lower)
            )
            if rate_limited:
                self.token_pool.rate_limited(
                    token_index, response.headers, secondary=secondary
                )
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="rate_limit_retry",
                    latency_ms=elapsed,
                    error_class="RateLimit",
                    error_message=self.redact(body_message),
                    **rate,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"rate-limit retries exhausted for {repo.canonical} "
                        f"window {created}"
                    )
                continue

            if response.status >= 500:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="server_retry",
                    latency_ms=elapsed,
                    error_class="GitHubServerError",
                    error_message=self.redact(body_message),
                    **rate,
                )
                if attempt == self.max_attempts:
                    raise APIError(
                        f"GitHub server retries exhausted for {repo.canonical}: "
                        f"HTTP {response.status}"
                    )
                self.sleeper(min(2 ** (attempt - 1), 30))
                continue

            if response.status != 200:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="permanent_error",
                    latency_ms=elapsed,
                    error_class="GitHubHTTPError",
                    error_message=self.redact(body_message),
                    **rate,
                )
                raise APIError(
                    f"GitHub HTTP {response.status} for {repo.canonical}: "
                    f"{self.redact(body_message)}"
                )

            try:
                payload = json.loads(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="malformed",
                    latency_ms=elapsed,
                    error_class=type(exc).__name__,
                    error_message=self.redact(exc),
                    **rate,
                )
                raise MalformedAPIError(
                    f"GitHub returned invalid JSON for {repo.canonical}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                problem = "response root is not an object"
            elif (
                isinstance(payload.get("total_count"), bool)
                or not isinstance(payload.get("total_count"), int)
                or int(payload["total_count"]) < 0
            ):
                problem = "total_count must be a non-negative integer"
            elif not isinstance(payload.get("workflow_runs"), list):
                problem = "workflow_runs must be an array"
            elif any(not isinstance(item, dict) for item in payload["workflow_runs"]):
                problem = "workflow_runs must contain objects"
            else:
                problem = ""
            if problem:
                ledger(
                    endpoint=endpoint,
                    page=page,
                    per_page=per_page,
                    attempt=attempt,
                    http_status=response.status,
                    outcome="malformed",
                    latency_ms=elapsed,
                    error_class="MalformedAPI",
                    error_message=problem,
                    **rate,
                )
                raise MalformedAPIError(
                    f"malformed GitHub response for {repo.canonical}: {problem}"
                )
            canonical_payload = {
                "total_count": int(payload["total_count"]),
                "workflow_runs": payload["workflow_runs"],
            }
            ledger(
                endpoint=endpoint,
                page=page,
                per_page=per_page,
                attempt=attempt,
                http_status=response.status,
                outcome="success",
                latency_ms=elapsed,
                error_class=None,
                error_message=None,
                **rate,
            )
            return PageResponse(
                total_count=int(payload["total_count"]),
                workflow_runs=tuple(dict(item) for item in payload["workflow_runs"]),
                payload_sha256=_sha256_json(canonical_payload),
            )
        raise AssertionError("unreachable retry loop")


_SCHEMA_SQL = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS inventory_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_schema TEXT NOT NULL,
    to_schema TEXT NOT NULL,
    from_script_sha256 TEXT NOT NULL,
    to_script_sha256 TEXT NOT NULL,
    upgraded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_binding_upgrades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_schema TEXT NOT NULL,
    to_schema TEXT NOT NULL,
    from_script_sha256 TEXT NOT NULL,
    to_script_sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    upgraded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS repos (
    repo_key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical TEXT NOT NULL,
    ordinal INTEGER NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS search_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
    start_epoch INTEGER NOT NULL,
    end_epoch INTEGER NOT NULL,
    parent_id INTEGER REFERENCES search_windows(id),
    depth INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('open','fetching','split','done','failed')
    ),
    expected_total INTEGER,
    expected_pages INTEGER,
    pages_done INTEGER NOT NULL DEFAULT 0,
    raw_items INTEGER NOT NULL DEFAULT 0,
    distinct_items INTEGER NOT NULL DEFAULT 0,
    duplicate_items INTEGER NOT NULL DEFAULT 0,
    run_keys_sha256 TEXT,
    failure_class TEXT,
    failure_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(repo_key, start_epoch, end_epoch)
);
CREATE INDEX IF NOT EXISTS idx_windows_work
    ON search_windows(repo_key, status, start_epoch);
CREATE TABLE IF NOT EXISTS runs (
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    run_started_at TEXT,
    status TEXT,
    conclusion TEXT,
    workflow_id INTEGER,
    workflow_name TEXT,
    event TEXT,
    head_branch TEXT,
    head_sha TEXT,
    run_number INTEGER,
    html_url TEXT,
    api_url TEXT,
    metadata_blob BLOB NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY(repo_key, run_id, run_attempt)
);
CREATE INDEX IF NOT EXISTS idx_runs_created
    ON runs(repo_key, created_at, run_id, run_attempt);
CREATE TABLE IF NOT EXISTS window_runs (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    repo_key TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    PRIMARY KEY(window_id, repo_key, run_id, run_attempt),
    FOREIGN KEY(repo_key, run_id, run_attempt)
        REFERENCES runs(repo_key, run_id, run_attempt)
);
CREATE INDEX IF NOT EXISTS idx_window_runs_identity
    ON window_runs(repo_key,run_id,run_attempt,window_id);
CREATE TABLE IF NOT EXISTS window_pages (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    page_no INTEGER NOT NULL,
    total_count INTEGER NOT NULL,
    item_count INTEGER NOT NULL,
    distinct_item_count INTEGER NOT NULL,
    duplicate_item_count INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY(window_id, page_no)
);
CREATE TABLE IF NOT EXISTS window_convergence (
    window_id INTEGER PRIMARY KEY REFERENCES search_windows(id),
    attempts INTEGER NOT NULL DEFAULT 0,
    candidate_total INTEGER,
    candidate_sha256 TEXT,
    stable_observations INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS convergence_passes (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    pass_no INTEGER NOT NULL CHECK(pass_no >= 1),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    page_count INTEGER NOT NULL CHECK(page_count >= 1),
    raw_item_count INTEGER NOT NULL CHECK(raw_item_count >= 0),
    distinct_item_count INTEGER NOT NULL CHECK(distinct_item_count >= 0),
    duplicate_item_count INTEGER NOT NULL CHECK(duplicate_item_count >= 0),
    page_payload_set_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    accumulated_distinct_count INTEGER NOT NULL
        CHECK(accumulated_distinct_count >= 0),
    min_observation_count INTEGER NOT NULL CHECK(min_observation_count >= 0),
    observed_at TEXT NOT NULL,
    PRIMARY KEY(window_id, pass_no)
);
CREATE TABLE IF NOT EXISTS convergence_pass_pages (
    window_id INTEGER NOT NULL,
    pass_no INTEGER NOT NULL,
    page_no INTEGER NOT NULL CHECK(page_no >= 1),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    item_count INTEGER NOT NULL CHECK(item_count >= 0),
    distinct_item_count INTEGER NOT NULL CHECK(distinct_item_count >= 0),
    duplicate_item_count INTEGER NOT NULL CHECK(duplicate_item_count >= 0),
    payload_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    PRIMARY KEY(window_id, pass_no, page_no),
    FOREIGN KEY(window_id,pass_no)
        REFERENCES convergence_passes(window_id,pass_no)
);
CREATE TABLE IF NOT EXISTS convergence_runs (
    window_id INTEGER NOT NULL REFERENCES search_windows(id),
    repo_key TEXT NOT NULL REFERENCES repos(repo_key),
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    run_started_at TEXT,
    status TEXT,
    conclusion TEXT,
    workflow_id INTEGER,
    workflow_name TEXT,
    event TEXT,
    head_branch TEXT,
    head_sha TEXT,
    run_number INTEGER,
    html_url TEXT,
    api_url TEXT,
    metadata_blob BLOB NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    first_pass INTEGER NOT NULL CHECK(first_pass >= 1),
    last_pass INTEGER NOT NULL CHECK(last_pass >= first_pass),
    observation_count INTEGER NOT NULL CHECK(observation_count >= 1),
    PRIMARY KEY(window_id, repo_key, run_id, run_attempt)
);
CREATE INDEX IF NOT EXISTS idx_convergence_runs_identity
    ON convergence_runs(repo_key,run_id,run_attempt,window_id);
CREATE TABLE IF NOT EXISTS convergence_pass_runs (
    window_id INTEGER NOT NULL,
    pass_no INTEGER NOT NULL,
    repo_key TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    run_attempt INTEGER NOT NULL,
    metadata_sha256 TEXT NOT NULL,
    PRIMARY KEY(
        window_id,pass_no,repo_key,run_id,run_attempt
    ),
    FOREIGN KEY(window_id,pass_no)
        REFERENCES convergence_passes(window_id,pass_no),
    FOREIGN KEY(window_id,repo_key,run_id,run_attempt)
        REFERENCES convergence_runs(
            window_id,repo_key,run_id,run_attempt
        )
);
CREATE TABLE IF NOT EXISTS window_union_closures (
    window_id INTEGER PRIMARY KEY REFERENCES search_windows(id),
    total_count INTEGER NOT NULL CHECK(total_count >= 0),
    pass_count INTEGER NOT NULL CHECK(pass_count >= 2),
    first_pass_no INTEGER NOT NULL CHECK(first_pass_no >= 1),
    last_pass_no INTEGER NOT NULL CHECK(last_pass_no >= first_pass_no),
    observed_page_count INTEGER NOT NULL CHECK(observed_page_count >= 2),
    observed_item_count INTEGER NOT NULL CHECK(observed_item_count >= 0),
    distinct_run_count INTEGER NOT NULL CHECK(distinct_run_count >= 0),
    min_observation_count INTEGER NOT NULL CHECK(min_observation_count >= 2),
    pass_set_sha256 TEXT NOT NULL,
    run_keys_sha256 TEXT NOT NULL,
    closed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at TEXT NOT NULL,
    repo_key TEXT NOT NULL,
    window_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    page_no INTEGER NOT NULL,
    per_page INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    http_status INTEGER,
    outcome TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    rate_remaining TEXT,
    rate_reset TEXT,
    retry_after TEXT,
    error_class TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_ledger_window
    ON request_ledger(window_id, id);
"""


class InventoryDB:
    """SQLite state and fail-closed completion validation."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        conn = self.connect()
        try:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _meta(conn: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key,value FROM inventory_meta")
        }

    @staticmethod
    def _backfill_binding_upgrade_history_locked(
        conn: sqlite3.Connection,
        *,
        current_schema: str,
        current_script_sha256: str,
    ) -> None:
        legacy = list(
            conn.execute(
                """
                SELECT from_schema,to_schema,from_script_sha256,
                       to_script_sha256,upgraded_at
                FROM inventory_upgrades ORDER BY id
                """
            )
        )
        binding = list(
            conn.execute(
                """
                SELECT from_schema,to_schema,from_script_sha256,
                       to_script_sha256,reason,upgraded_at
                FROM inventory_binding_upgrades ORDER BY id
                """
            )
        )
        projected = [
            (
                str(row["from_schema"]),
                str(row["to_schema"]),
                str(row["from_script_sha256"]),
                str(row["to_script_sha256"]),
                str(row["upgraded_at"]),
            )
            for row in binding
        ]
        legacy_values = [
            (
                str(row["from_schema"]),
                str(row["to_schema"]),
                str(row["from_script_sha256"]),
                str(row["to_script_sha256"]),
                str(row["upgraded_at"]),
            )
            for row in legacy
        ]
        if binding:
            if projected != legacy_values:
                raise BindingError(
                    "inventory producer upgrade ledgers disagree before "
                    "migration"
                )
            for row in binding:
                _validate_upgrade_reason(str(row["reason"]))
        elif legacy_values:
            conn.executemany(
                """
                INSERT INTO inventory_binding_upgrades(
                    from_schema,to_schema,from_script_sha256,
                    to_script_sha256,reason,upgraded_at
                ) VALUES (?,?,?,?,?,?)
                """,
                [
                    (*row[:4], IMPORTED_UPGRADE_REASON, row[4])
                    for row in legacy_values
                ],
            )
        if legacy_values:
            for index, row in enumerate(legacy_values):
                if index and (
                    legacy_values[index - 1][1] != row[0]
                    or legacy_values[index - 1][3] != row[2]
                ):
                    raise BindingError(
                        "legacy inventory producer upgrade chain is broken"
                    )
            if (
                legacy_values[-1][1] != current_schema
                or legacy_values[-1][3] != current_script_sha256
            ):
                raise BindingError(
                    "legacy inventory producer upgrade chain does not bind "
                    "the current database producer"
                )

    def bind(
        self,
        *,
        scope: RepoScope,
        start_epoch: int,
        end_epoch: int,
        script_sha256: str,
        resume: bool,
        allow_script_upgrade_from_sha256: str | None = None,
        script_upgrade_reason: str | None = None,
    ) -> None:
        if start_epoch >= end_epoch:
            raise BindingError("inventory interval must satisfy start < end")
        expected = {
            "schema": SCHEMA_VERSION,
            "repo_list_path": scope.path,
            "repo_list_sha256": scope.source_sha256,
            "repo_scope_sha256": scope.scope_sha256,
            "repo_count": str(len(scope.repos)),
            "original_repo_count": str(scope.original_repo_count),
            "unresolved_count": str(scope.unresolved_count),
            "start_epoch": str(start_epoch),
            "end_epoch": str(end_epoch),
            "start_utc": format_utc_instant(start_epoch),
            "end_utc": format_utc_instant(end_epoch),
            "script_sha256": script_sha256,
            "metadata_encoding": METADATA_ENCODING,
            "smoke": "1" if scope.smoke else "0",
            "max_repos": "" if scope.max_repos is None else str(scope.max_repos),
        }
        conn = self.connect()
        try:
            with self._write_lock, conn:
                current = self._meta(conn)
                if current:
                    if not resume:
                        raise BindingError(
                            f"inventory database already exists at {self.path}; "
                            "pass --resume after verifying its binding"
                        )
                    current_schema = current.get("schema")
                    previous_script = current.get("script_sha256", "")
                    upgrade_v1 = (
                        current_schema == "cppmega_ci_stream_inventory_v1"
                    )
                    upgrade_v2 = current_schema == PREVIOUS_SCHEMA_VERSION
                    upgrade_reason: str | None = None
                    if upgrade_v2:
                        if (
                            allow_script_upgrade_from_sha256
                            != previous_script
                        ):
                            raise BindingError(
                                "inventory v2 to v3 migration requires "
                                "--allow-inventory-script-upgrade-from-sha256 "
                                "to match the exact bound producer"
                            )
                        upgrade_reason = _validate_upgrade_reason(
                            script_upgrade_reason
                        )
                    elif (
                        allow_script_upgrade_from_sha256 is not None
                        or script_upgrade_reason is not None
                    ):
                        repeated_reason = _validate_upgrade_reason(
                            script_upgrade_reason
                        )
                        repeated_upgrade = conn.execute(
                            """
                            SELECT from_schema,to_schema,from_script_sha256,
                                   to_script_sha256,reason
                            FROM inventory_binding_upgrades
                            ORDER BY id DESC LIMIT 1
                            """
                        ).fetchone()
                        if (
                            current_schema != SCHEMA_VERSION
                            or previous_script != script_sha256
                            or repeated_upgrade is None
                            or str(repeated_upgrade["to_schema"])
                            != SCHEMA_VERSION
                            or str(
                                repeated_upgrade[
                                    "from_script_sha256"
                                ]
                            )
                            != allow_script_upgrade_from_sha256
                            or str(
                                repeated_upgrade["to_script_sha256"]
                            )
                            != script_sha256
                            or str(repeated_upgrade["reason"])
                            != repeated_reason
                        ):
                            raise BindingError(
                                "inventory script upgrade authorization does "
                                "not exactly replay the latest completed "
                                "producer migration"
                            )
                    ignored_upgrade_keys = (
                        {"schema", "script_sha256"}
                        if upgrade_v1 or upgrade_v2
                        else set()
                    )
                    mismatches = {
                        key: (current.get(key), value)
                        for key, value in expected.items()
                        if key not in ignored_upgrade_keys
                        and current.get(key) != value
                    }
                    if mismatches:
                        rendered = ", ".join(
                            f"{key}={old!r}->{new!r}"
                            for key, (old, new) in sorted(mismatches.items())
                        )
                        raise BindingError(
                            f"resume binding mismatch in {self.path}: {rendered}"
                        )
                    if upgrade_v1 or upgrade_v2:
                        self._backfill_binding_upgrade_history_locked(
                            conn,
                            current_schema=str(current_schema),
                            current_script_sha256=previous_script,
                        )
                        reason = (
                            "audited legacy inventory v1 recovery migration"
                            if upgrade_v1
                            else upgrade_reason
                        )
                        assert reason is not None
                        upgraded_at = _utc_now()
                        conn.execute(
                            """
                            INSERT INTO inventory_upgrades(
                                from_schema,to_schema,from_script_sha256,
                                to_script_sha256,upgraded_at
                            ) VALUES (?,?,?,?,?)
                            """,
                            (
                                current["schema"],
                                SCHEMA_VERSION,
                                previous_script,
                                script_sha256,
                                upgraded_at,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO inventory_binding_upgrades(
                                from_schema,to_schema,from_script_sha256,
                                to_script_sha256,reason,upgraded_at
                            ) VALUES (?,?,?,?,?,?)
                            """,
                            (
                                current["schema"],
                                SCHEMA_VERSION,
                                previous_script,
                                script_sha256,
                                reason,
                                upgraded_at,
                            ),
                        )
                        conn.execute(
                            """
                            UPDATE inventory_meta SET value=?
                            WHERE key='schema'
                            """,
                            (SCHEMA_VERSION,),
                        )
                        conn.execute(
                            """
                            UPDATE inventory_meta SET value=?
                            WHERE key='script_sha256'
                            """,
                            (script_sha256,),
                        )
                    elif current.get("script_sha256") != script_sha256:
                        raise BindingError(
                            "resume script hash mismatch; no authorized "
                            "inventory producer migration applies"
                        )
                else:
                    if (
                        allow_script_upgrade_from_sha256 is not None
                        or script_upgrade_reason is not None
                    ):
                        raise BindingError(
                            "inventory script upgrade authorization cannot be "
                            "used when creating a new database"
                        )
                    conn.executemany(
                        "INSERT INTO inventory_meta(key,value) VALUES (?,?)",
                        sorted(expected.items()),
                    )
                    conn.execute(
                        "INSERT INTO inventory_meta(key,value) VALUES ('created_at',?)",
                        (_utc_now(),),
                    )
                    conn.executemany(
                        """
                        INSERT INTO repos(repo_key,owner,name,canonical,ordinal)
                        VALUES (?,?,?,?,?)
                        """,
                        [
                            (
                                repo.key,
                                repo.owner,
                                repo.name,
                                repo.canonical,
                                repo.ordinal,
                            )
                            for repo in scope.repos
                        ],
                    )
                    now = _utc_now()
                    conn.executemany(
                        """
                        INSERT INTO search_windows(
                            repo_key,start_epoch,end_epoch,parent_id,depth,status,
                            created_at,updated_at
                        ) VALUES (?,?,?,NULL,0,'open',?,?)
                        """,
                        [
                            (repo.key, start_epoch, end_epoch, now, now)
                            for repo in scope.repos
                        ],
                    )
                database_repos = {
                    str(row["repo_key"]): (
                        str(row["owner"]),
                        str(row["name"]),
                        int(row["ordinal"]),
                    )
                    for row in conn.execute(
                        "SELECT repo_key,owner,name,ordinal FROM repos"
                    )
                }
                expected_repos = {
                    repo.key: (repo.owner, repo.name, repo.ordinal)
                    for repo in scope.repos
                }
                if database_repos != expected_repos:
                    raise BindingError(
                        "database repository scope is not exactly the canonical scope"
                    )
                if resume:
                    # A page transaction is atomic.  Retrying a failed window
                    # therefore resumes at its first absent page without loss.
                    conn.execute(
                        """
                        UPDATE search_windows
                        SET status=CASE
                              WHEN expected_total IS NULL THEN 'open'
                              ELSE 'fetching'
                            END,
                            failure_class=NULL,
                            failure_message=NULL,
                            updated_at=?
                        WHERE status='failed'
                        """,
                        (_utc_now(),),
                    )
                    conn.execute(
                        """
                        UPDATE search_windows
                        SET status='fetching',updated_at=?
                        WHERE id IN (SELECT window_id FROM window_convergence)
                        """,
                        (_utc_now(),),
                    )
        finally:
            conn.close()

    def record_request(
        self,
        conn: sqlite3.Connection,
        *,
        repo_key: str,
        window_id: int,
        endpoint: str,
        page: int,
        per_page: int,
        attempt: int,
        http_status: int | None,
        outcome: str,
        latency_ms: int,
        rate_remaining: str | None = None,
        rate_reset: str | None = None,
        retry_after: str | None = None,
        error_class: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._write_lock, conn:
            conn.execute(
                """
                INSERT INTO request_ledger(
                    requested_at,repo_key,window_id,endpoint,page_no,per_page,
                    attempt,http_status,outcome,latency_ms,rate_remaining,
                    rate_reset,retry_after,error_class,error_message
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _utc_now(),
                    repo_key,
                    window_id,
                    endpoint,
                    page,
                    per_page,
                    attempt,
                    http_status,
                    outcome,
                    latency_ms,
                    rate_remaining,
                    rate_reset,
                    retry_after,
                    error_class,
                    error_message,
                ),
            )

    @staticmethod
    def _run_int(
        run: Mapping[str, Any],
        field: str,
        *,
        required: bool = False,
        minimum: int | None = None,
    ) -> int | None:
        value = run.get(field)
        if value is None and not required:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise MalformedAPIError(f"workflow run {field!r} must be an integer")
        if minimum is not None and value < minimum:
            raise MalformedAPIError(
                f"workflow run {field!r} must be >= {minimum}, got {value}"
            )
        return value

    @staticmethod
    def _run_text(run: Mapping[str, Any], field: str) -> str | None:
        value = run.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise MalformedAPIError(f"workflow run {field!r} must be text or null")
        return value

    def _normalize_run(
        self,
        repo_key: str,
        run: dict[str, Any],
        *,
        start_epoch: int,
        end_epoch: int,
    ) -> tuple[dict[str, Any], str, tuple[int, int]]:
        run_id = self._run_int(run, "id", required=True, minimum=1)
        assert run_id is not None
        attempt_value = run.get("run_attempt")
        if attempt_value is None:
            run_attempt = 0
        else:
            parsed_attempt = self._run_int(
                run, "run_attempt", required=True, minimum=1
            )
            assert parsed_attempt is not None
            run_attempt = parsed_attempt
        created_at = self._run_text(run, "created_at")
        if not created_at:
            raise MalformedAPIError("workflow run is missing created_at")
        try:
            created_epoch = parse_utc_instant(created_at)
        except ValueError as exc:
            raise MalformedAPIError(
                f"workflow run {run_id} has invalid created_at: {exc}"
            ) from exc
        if not start_epoch <= created_epoch < end_epoch:
            raise UnstableEnumerationError(
                f"workflow run {run_id} created_at={created_at} is outside "
                f"[{format_utc_instant(start_epoch)},"
                f"{format_utc_instant(end_epoch)})"
            )
        status = self._run_text(run, "status")
        conclusion = self._run_text(run, "conclusion")
        metadata_json = _canonical_json(run)
        metadata_sha = _sha256_bytes(metadata_json.encode("utf-8"))
        metadata_blob = zlib.compress(metadata_json.encode("utf-8"), level=6)
        normalized = {
            "repo_key": repo_key,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "created_at": created_at,
            "updated_at": self._run_text(run, "updated_at"),
            "run_started_at": self._run_text(run, "run_started_at"),
            "status": status,
            "conclusion": conclusion,
            "workflow_id": self._run_int(run, "workflow_id"),
            "workflow_name": self._run_text(run, "name"),
            "event": self._run_text(run, "event"),
            "head_branch": self._run_text(run, "head_branch"),
            "head_sha": self._run_text(run, "head_sha"),
            "run_number": self._run_int(run, "run_number"),
            "html_url": self._run_text(run, "html_url"),
            "api_url": self._run_text(run, "url"),
            "metadata_blob": sqlite3.Binary(metadata_blob),
            "metadata_sha256": metadata_sha,
        }
        return normalized, metadata_sha, (run_id, run_attempt)

    def split_window(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        observed_total: int,
    ) -> None:
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        if end - start <= 1:
            raise UnstableEnumerationError(
                f"{row['repo_key']} has {observed_total} workflow runs in the "
                f"unsplittable one-second interval "
                f"[{format_utc_instant(start)},{format_utc_instant(end)})"
            )
        midpoint = start + (end - start) // 2
        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT status,expected_total FROM search_windows WHERE id=?",
                (int(row["id"]),),
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["status"] == "split":
                if int(current["expected_total"]) != observed_total:
                    raise UnstableEnumerationError(
                        "split-window total changed across resume"
                    )
                return
            if current["status"] not in {"open", "fetching"}:
                raise UnstableEnumerationError(
                    f"cannot split window in status {current['status']!r}"
                )
            conn.execute(
                """
                UPDATE search_windows
                SET status='split',expected_total=?,expected_pages=NULL,
                    updated_at=?
                WHERE id=?
                """,
                (observed_total, now, int(row["id"])),
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO search_windows(
                    repo_key,start_epoch,end_epoch,parent_id,depth,status,
                    created_at,updated_at
                ) VALUES (?,?,?,?,?,'open',?,?)
                """,
                [
                    (
                        str(row["repo_key"]),
                        start,
                        midpoint,
                        int(row["id"]),
                        int(row["depth"]) + 1,
                        now,
                        now,
                    ),
                    (
                        str(row["repo_key"]),
                        midpoint,
                        end,
                        int(row["id"]),
                        int(row["depth"]) + 1,
                        now,
                        now,
                    ),
                ],
            )

    def _clear_window_payload_locked(
        self, conn: sqlite3.Connection, *, window_id: int
    ) -> None:
        keys = [
            (str(row["repo_key"]), int(row["run_id"]), int(row["run_attempt"]))
            for row in conn.execute(
                """
                SELECT repo_key,run_id,run_attempt
                FROM window_runs WHERE window_id=?
                """,
                (window_id,),
            )
        ]
        conn.execute("DELETE FROM window_pages WHERE window_id=?", (window_id,))
        conn.execute("DELETE FROM window_runs WHERE window_id=?", (window_id,))
        for repo_key, run_id, run_attempt in keys:
            conn.execute(
                """
                DELETE FROM runs
                WHERE repo_key=? AND run_id=? AND run_attempt=?
                  AND NOT EXISTS (
                      SELECT 1 FROM window_runs wr
                      WHERE wr.repo_key=runs.repo_key
                        AND wr.run_id=runs.run_id
                        AND wr.run_attempt=runs.run_attempt
                  )
                """,
                (repo_key, run_id, run_attempt),
            )

    @staticmethod
    def _clear_convergence_proof_locked(
        conn: sqlite3.Connection, *, window_id: int
    ) -> None:
        conn.execute(
            "DELETE FROM window_union_closures WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_pass_runs WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_pass_pages WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_passes WHERE window_id=?",
            (window_id,),
        )
        conn.execute(
            "DELETE FROM convergence_runs WHERE window_id=?",
            (window_id,),
        )

    def recover_pagination_drift(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        observed_total: int,
        reason: str,
    ) -> str:
        """Invalidate an unstable leaf and atomically split or converge it."""

        window_id = int(row["id"])
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT * FROM search_windows WHERE id=?", (window_id,)
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["status"] not in {"open", "fetching", "failed", "done"}:
                raise UnstableEnumerationError(
                    f"cannot recover window in status {current['status']!r}"
                )
            self._clear_window_payload_locked(conn, window_id=window_id)
            self._clear_convergence_proof_locked(
                conn, window_id=window_id
            )
            if end - start > 1:
                midpoint = start + (end - start) // 2
                conn.execute(
                    "DELETE FROM window_convergence WHERE window_id=?", (window_id,)
                )
                conn.execute(
                    """
                    UPDATE search_windows
                    SET status='split',expected_total=?,expected_pages=NULL,
                        pages_done=0,raw_items=0,distinct_items=0,
                        duplicate_items=0,run_keys_sha256=NULL,
                        failure_class=NULL,failure_message=NULL,updated_at=?
                    WHERE id=?
                    """,
                    (observed_total, now, window_id),
                )
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO search_windows(
                        repo_key,start_epoch,end_epoch,parent_id,depth,status,
                        created_at,updated_at
                    ) VALUES (?,?,?,?,?,'open',?,?)
                    """,
                    [
                        (
                            str(row["repo_key"]),
                            start,
                            midpoint,
                            window_id,
                            int(row["depth"]) + 1,
                            now,
                            now,
                        ),
                        (
                            str(row["repo_key"]),
                            midpoint,
                            end,
                            window_id,
                            int(row["depth"]) + 1,
                            now,
                            now,
                        ),
                    ],
                )
                return "split"

            if observed_total > GITHUB_FILTER_LIMIT:
                raise UnstableEnumerationError(
                    f"{row['repo_key']} has {observed_total} runs in one second; "
                    "the repository endpoint cannot prove a complete set above "
                    f"{GITHUB_FILTER_LIMIT}"
                )
            conn.execute(
                """
                UPDATE search_windows
                SET status='fetching',expected_total=?,expected_pages=?,
                    pages_done=0,raw_items=0,distinct_items=0,
                    duplicate_items=0,run_keys_sha256=NULL,
                    failure_class=NULL,failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (
                    observed_total,
                    max(1, math.ceil(observed_total / DEFAULT_PER_PAGE)),
                    now,
                    window_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO window_convergence(
                    window_id,attempts,candidate_total,candidate_sha256,
                    stable_observations,last_error,updated_at
                ) VALUES (?,0,NULL,NULL,0,?,?)
                ON CONFLICT(window_id) DO UPDATE SET
                    candidate_total=NULL,
                    candidate_sha256=NULL,
                    stable_observations=0,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (window_id, reason[:4000], now),
            )
            return "converge"

    def convergence_state(
        self, conn: sqlite3.Connection, window_id: int
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM window_convergence WHERE window_id=?", (window_id,)
        ).fetchone()

    def prepare_convergence(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> None:
        window_id = int(row["id"])
        with self._write_lock, conn:
            state = conn.execute(
                "SELECT 1 FROM window_convergence WHERE window_id=?", (window_id,)
            ).fetchone()
            if state is None:
                raise UnstableEnumerationError(
                    f"window {window_id} lost convergence state"
                )
            self._clear_window_payload_locked(conn, window_id=window_id)
            conn.execute(
                """
                UPDATE search_windows
                SET status='fetching',pages_done=0,raw_items=0,
                    distinct_items=0,duplicate_items=0,run_keys_sha256=NULL,
                    failure_class=NULL,failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (_utc_now(), window_id),
            )

    @staticmethod
    def _convergence_pass_set_sha256(
        conn: sqlite3.Connection, *, window_id: int
    ) -> str:
        return _hash_lines(
            "\t".join(
                str(row[field])
                for field in (
                    "pass_no",
                    "total_count",
                    "page_count",
                    "raw_item_count",
                    "distinct_item_count",
                    "duplicate_item_count",
                    "page_payload_set_sha256",
                    "run_keys_sha256",
                    "accumulated_distinct_count",
                    "min_observation_count",
                )
            )
            for row in conn.execute(
                """
                SELECT pass_no,total_count,page_count,raw_item_count,
                       distinct_item_count,duplicate_item_count,
                       page_payload_set_sha256,run_keys_sha256,
                       accumulated_distinct_count,min_observation_count
                FROM convergence_passes
                WHERE window_id=?
                ORDER BY pass_no
                """,
                (window_id,),
            )
        )

    def accumulate_convergence_pass(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        pages: Sequence[PageResponse],
    ) -> tuple[bool, str | None]:
        """Accumulate one complete API pass into a cardinality-bound union.

        GitHub does not provide a stable tie-breaker when more than one page of
        workflow runs shares the same ``created_at`` second.  A single pass can
        therefore contain duplicates across pages.  The union is allowed to
        close only after it contains exactly ``total_count`` unique run keys
        and every key has appeared with identical metadata in at least two
        distinct passes.
        """

        if not pages:
            raise PaginationDrift(
                "convergence pass returned no pages", observed_total=0
            )
        total = pages[0].total_count
        if total > GITHUB_FILTER_LIMIT:
            raise UnstableEnumerationError(
                f"one-second convergence total {total} exceeds "
                f"{GITHUB_FILTER_LIMIT}"
            )
        expected_pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
        if len(pages) != expected_pages:
            raise PaginationDrift(
                f"convergence pass has {len(pages)} pages, expected "
                f"{expected_pages}",
                observed_total=total,
            )

        repo_key = str(row["repo_key"])
        normalized: dict[
            tuple[int, int], tuple[dict[str, Any], str]
        ] = {}
        page_lines: list[str] = []
        page_proofs: list[
            tuple[int, int, int, int, int, str, str]
        ] = []
        raw_item_count = 0
        for page_no, page in enumerate(pages, start=1):
            if page.total_count != total:
                raise PaginationDrift(
                    f"convergence total_count changed {total} -> "
                    f"{page.total_count}",
                    observed_total=page.total_count,
                )
            expected_items = (
                DEFAULT_PER_PAGE
                if page_no < expected_pages
                else total - DEFAULT_PER_PAGE * (expected_pages - 1)
            )
            if len(page.workflow_runs) != expected_items:
                raise PaginationDrift(
                    f"convergence page {page_no} has "
                    f"{len(page.workflow_runs)} items, expected {expected_items}",
                    observed_total=total,
                )
            page_keys: list[str] = []
            for run in page.workflow_runs:
                record, metadata_sha, key = self._normalize_run(
                    repo_key,
                    run,
                    start_epoch=int(row["start_epoch"]),
                    end_epoch=int(row["end_epoch"]),
                )
                previous = normalized.get(key)
                if previous is not None and previous[1] != metadata_sha:
                    raise PaginationDrift(
                        f"convergence run {key[0]} attempt {key[1]} "
                        "changed metadata within one pass",
                        observed_total=total,
                    )
                normalized[key] = (record, metadata_sha)
                page_keys.append(
                    f"{repo_key}\t{key[0]}\t{key[1]}\t{metadata_sha}"
                )
            raw_item_count += len(page.workflow_runs)
            page_key_digest = _hash_lines(sorted(page_keys))
            page_line = (
                f"{page_no}\t{page.total_count}\t"
                f"{len(page.workflow_runs)}\t{len(set(page_keys))}\t"
                f"{len(page_keys) - len(set(page_keys))}\t"
                f"{page.payload_sha256}\t{page_key_digest}"
            )
            page_lines.append(page_line)
            page_proofs.append(
                (
                    page_no,
                    page.total_count,
                    len(page.workflow_runs),
                    len(set(page_keys)),
                    len(page_keys) - len(set(page_keys)),
                    page.payload_sha256,
                    page_key_digest,
                )
            )
        if raw_item_count != total:
            raise PaginationDrift(
                f"convergence pass returned {raw_item_count} raw items "
                f"for total_count={total}",
                observed_total=total,
            )

        pass_run_sha256 = _hash_lines(
            f"{repo_key}\t{run_id}\t{run_attempt}\t{metadata_sha}"
            for (run_id, run_attempt), (_record, metadata_sha) in sorted(
                normalized.items()
            )
        )
        page_payload_set_sha256 = _hash_lines(page_lines)
        now = _utc_now()
        window_id = int(row["id"])
        with self._write_lock, conn:
            state = conn.execute(
                "SELECT * FROM window_convergence WHERE window_id=?",
                (window_id,),
            ).fetchone()
            if state is None:
                raise UnstableEnumerationError(
                    f"window {window_id} lost convergence state"
                )
            if (
                state["candidate_total"] is not None
                and int(state["candidate_total"]) != total
            ):
                raise PaginationDrift(
                    f"convergence total_count changed "
                    f"{state['candidate_total']} -> {total}",
                    observed_total=total,
                )
            pass_no = int(state["attempts"]) + 1
            for (run_id, run_attempt), (
                record,
                metadata_sha,
            ) in sorted(normalized.items()):
                existing = conn.execute(
                    """
                    SELECT metadata_sha256,last_pass,observation_count
                    FROM convergence_runs
                    WHERE window_id=? AND repo_key=? AND run_id=?
                      AND run_attempt=?
                    """,
                    (window_id, repo_key, run_id, run_attempt),
                ).fetchone()
                if existing is not None:
                    if str(existing["metadata_sha256"]) != metadata_sha:
                        raise PaginationDrift(
                            f"convergence run {run_id} attempt {run_attempt} "
                            "changed metadata across passes",
                            observed_total=total,
                        )
                    if int(existing["last_pass"]) != pass_no:
                        conn.execute(
                            """
                            UPDATE convergence_runs
                            SET last_pass=?,observation_count=observation_count+1
                            WHERE window_id=? AND repo_key=? AND run_id=?
                              AND run_attempt=?
                            """,
                            (
                                pass_no,
                                window_id,
                                repo_key,
                                run_id,
                                run_attempt,
                            ),
                        )
                    continue
                conn.execute(
                    """
                    INSERT INTO convergence_runs(
                        window_id,repo_key,run_id,run_attempt,created_at,
                        updated_at,run_started_at,status,conclusion,workflow_id,
                        workflow_name,event,head_branch,head_sha,run_number,
                        html_url,api_url,metadata_blob,metadata_sha256,
                        first_seen_at,first_pass,last_pass,observation_count
                    ) VALUES (
                        :window_id,:repo_key,:run_id,:run_attempt,:created_at,
                        :updated_at,:run_started_at,:status,:conclusion,
                        :workflow_id,:workflow_name,:event,:head_branch,
                        :head_sha,:run_number,:html_url,:api_url,:metadata_blob,
                        :metadata_sha256,:first_seen_at,:first_pass,:last_pass,1
                    )
                    """,
                    {
                        **record,
                        "window_id": window_id,
                        "first_seen_at": now,
                        "first_pass": pass_no,
                        "last_pass": pass_no,
                    },
                )

            aggregate = conn.execute(
                """
                SELECT COUNT(*) AS distinct_count,
                       COALESCE(MIN(observation_count),0) AS min_observations
                FROM convergence_runs WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            distinct_count = int(aggregate["distinct_count"])
            min_observations = int(aggregate["min_observations"])
            if distinct_count > total:
                raise UnstableEnumerationError(
                    f"convergence union for window {window_id} contains "
                    f"{distinct_count} runs, above total_count={total}"
                )
            union_sha256 = _hash_lines(
                f"{item['repo_key']}\t{item['run_id']}\t"
                f"{item['run_attempt']}\t{item['metadata_sha256']}"
                for item in conn.execute(
                    """
                    SELECT repo_key,run_id,run_attempt,metadata_sha256
                    FROM convergence_runs WHERE window_id=?
                    ORDER BY repo_key,run_id,run_attempt
                    """,
                    (window_id,),
                )
            )
            conn.execute(
                """
                INSERT INTO convergence_passes(
                    window_id,pass_no,total_count,page_count,raw_item_count,
                    distinct_item_count,duplicate_item_count,
                    page_payload_set_sha256,run_keys_sha256,
                    accumulated_distinct_count,min_observation_count,observed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    pass_no,
                    total,
                    expected_pages,
                    raw_item_count,
                    len(normalized),
                    raw_item_count - len(normalized),
                    page_payload_set_sha256,
                    pass_run_sha256,
                    distinct_count,
                    min_observations,
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO convergence_pass_pages(
                    window_id,pass_no,page_no,total_count,item_count,
                    distinct_item_count,duplicate_item_count,payload_sha256,
                    run_keys_sha256
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    (window_id, pass_no, *page_proof)
                    for page_proof in page_proofs
                ],
            )
            conn.executemany(
                """
                INSERT INTO convergence_pass_runs(
                    window_id,pass_no,repo_key,run_id,run_attempt,
                    metadata_sha256
                ) VALUES (?,?,?,?,?,?)
                """,
                [
                    (
                        window_id,
                        pass_no,
                        repo_key,
                        run_id,
                        run_attempt,
                        metadata_sha,
                    )
                    for (run_id, run_attempt), (
                        _record,
                        metadata_sha,
                    ) in sorted(normalized.items())
                ],
            )
            conn.execute(
                """
                UPDATE window_convergence
                SET attempts=?,candidate_total=?,candidate_sha256=?,
                    stable_observations=?,last_error=?,updated_at=?
                WHERE window_id=?
                """,
                (
                    pass_no,
                    total,
                    union_sha256,
                    min_observations if distinct_count == total else 0,
                    (
                        None
                        if distinct_count == total and min_observations >= 2
                        else (
                            f"cardinality union has {distinct_count}/{total} "
                            f"runs; minimum distinct-pass observations="
                            f"{min_observations}"
                        )
                    ),
                    now,
                    window_id,
                ),
            )
            if distinct_count != total or min_observations < 2:
                return False, None

            mismatch = conn.execute(
                """
                SELECT candidate.repo_key,candidate.run_id,
                       candidate.run_attempt
                FROM convergence_runs candidate
                JOIN runs existing
                  ON existing.repo_key=candidate.repo_key
                 AND existing.run_id=candidate.run_id
                 AND existing.run_attempt=candidate.run_attempt
                WHERE candidate.window_id=?
                  AND existing.metadata_sha256 != candidate.metadata_sha256
                LIMIT 1
                """,
                (window_id,),
            ).fetchone()
            if mismatch is not None:
                raise UnstableEnumerationError(
                    "convergence metadata differs from an adjacent window for "
                    f"{mismatch['repo_key']}#{mismatch['run_id']} attempt "
                    f"{mismatch['run_attempt']}"
                )
            conn.execute(
                """
                INSERT OR IGNORE INTO runs(
                    repo_key,run_id,run_attempt,created_at,updated_at,
                    run_started_at,status,conclusion,workflow_id,workflow_name,
                    event,head_branch,head_sha,run_number,html_url,api_url,
                    metadata_blob,metadata_sha256,first_seen_at
                )
                SELECT repo_key,run_id,run_attempt,created_at,updated_at,
                       run_started_at,status,conclusion,workflow_id,
                       workflow_name,event,head_branch,head_sha,run_number,
                       html_url,api_url,metadata_blob,metadata_sha256,
                       first_seen_at
                FROM convergence_runs WHERE window_id=?
                """,
                (window_id,),
            )
            conn.execute(
                """
                INSERT INTO window_runs(
                    window_id,repo_key,run_id,run_attempt,metadata_sha256
                )
                SELECT window_id,repo_key,run_id,run_attempt,metadata_sha256
                FROM convergence_runs WHERE window_id=?
                ORDER BY repo_key,run_id,run_attempt
                """,
                (window_id,),
            )
            pass_stats = conn.execute(
                """
                SELECT COUNT(*) AS pass_count,MIN(pass_no) AS first_pass,
                       MAX(pass_no) AS last_pass,
                       SUM(page_count) AS observed_pages,
                       SUM(raw_item_count) AS observed_items
                FROM convergence_passes WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            pass_set_sha256 = self._convergence_pass_set_sha256(
                conn, window_id=window_id
            )
            pass_count = int(pass_stats["pass_count"])
            observed_pages = int(pass_stats["observed_pages"])
            observed_items = int(pass_stats["observed_items"])
            conn.execute(
                """
                INSERT INTO window_union_closures(
                    window_id,total_count,pass_count,first_pass_no,last_pass_no,
                    observed_page_count,observed_item_count,distinct_run_count,
                    min_observation_count,pass_set_sha256,run_keys_sha256,
                    closed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    total,
                    pass_count,
                    int(pass_stats["first_pass"]),
                    int(pass_stats["last_pass"]),
                    observed_pages,
                    observed_items,
                    distinct_count,
                    min_observations,
                    pass_set_sha256,
                    union_sha256,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE search_windows
                SET status='done',expected_total=?,expected_pages=?,
                    pages_done=?,raw_items=?,distinct_items=?,
                    duplicate_items=?,run_keys_sha256=?,failure_class=NULL,
                    failure_message=NULL,updated_at=?
                WHERE id=?
                """,
                (
                    total,
                    expected_pages,
                    observed_pages,
                    observed_items,
                    distinct_count,
                    observed_items - distinct_count,
                    union_sha256,
                    now,
                    window_id,
                ),
            )
            conn.execute(
                "DELETE FROM window_convergence WHERE window_id=?",
                (window_id,),
            )
            return True, union_sha256

    def store_page(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        page_no: int,
        page: PageResponse,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> bool:
        """Atomically commit a page.  Return true when the leaf closes."""

        window_id = int(row["id"])
        repo_key = str(row["repo_key"])
        start = int(row["start_epoch"])
        end = int(row["end_epoch"])
        total = page.total_count
        if total > GITHUB_FILTER_LIMIT:
            raise ValueError("dense response must be split before storing a page")
        expected_pages = max(1, math.ceil(total / per_page))
        if page_no < 1 or page_no > expected_pages:
            raise MalformedAPIError(
                f"page {page_no} outside expected 1..{expected_pages}"
            )
        expected_items = (
            per_page
            if page_no < expected_pages
            else total - per_page * (expected_pages - 1)
        )
        if len(page.workflow_runs) != expected_items:
            raise PaginationDrift(
                f"incomplete page {page_no}: expected {expected_items} items "
                f"from total_count={total}, got {len(page.workflow_runs)}",
                observed_total=total,
            )

        normalized: list[tuple[dict[str, Any], str, tuple[int, int]]] = []
        for run in page.workflow_runs:
            normalized.append(
                self._normalize_run(
                    repo_key, run, start_epoch=start, end_epoch=end
                )
            )
        page_keys = [
            f"{repo_key}\t{run_id}\t{attempt}\t{metadata_sha}"
            for _, metadata_sha, (run_id, attempt) in normalized
        ]
        page_key_digest = _hash_lines(sorted(page_keys))
        duplicate_in_page = len(page_keys) - len(set(page_keys))

        now = _utc_now()
        with self._write_lock, conn:
            current = conn.execute(
                "SELECT * FROM search_windows WHERE id=?", (window_id,)
            ).fetchone()
            if current is None:
                raise UnstableEnumerationError("search window disappeared")
            if current["expected_total"] is not None and (
                int(current["expected_total"]) != total
            ):
                raise PaginationDrift(
                    f"total_count changed for window {window_id}: "
                    f"{current['expected_total']} -> {total}",
                    observed_total=total,
                )
            old_page = conn.execute(
                """
                SELECT payload_sha256,run_keys_sha256
                FROM window_pages WHERE window_id=? AND page_no=?
                """,
                (window_id, page_no),
            ).fetchone()
            if old_page is not None:
                if (
                    old_page["payload_sha256"] != page.payload_sha256
                    or old_page["run_keys_sha256"] != page_key_digest
                ):
                    raise PaginationDrift(
                        f"page {page_no} of window {window_id} changed on replay",
                        observed_total=total,
                    )
                return str(current["status"]) == "done"
            if current["status"] not in {"open", "fetching"}:
                raise UnstableEnumerationError(
                    f"cannot store page in window status {current['status']!r}"
                )

            for record, metadata_sha, (run_id, run_attempt) in normalized:
                existing = conn.execute(
                    """
                    SELECT metadata_sha256 FROM runs
                    WHERE repo_key=? AND run_id=? AND run_attempt=?
                    """,
                    (repo_key, run_id, run_attempt),
                ).fetchone()
                if existing is not None:
                    if str(existing["metadata_sha256"]) != metadata_sha:
                        raise PaginationDrift(
                            f"workflow run {repo_key}#{run_id} attempt "
                            f"{run_attempt} changed during enumeration",
                            observed_total=total,
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO runs(
                            repo_key,run_id,run_attempt,created_at,updated_at,
                            run_started_at,status,conclusion,workflow_id,
                            workflow_name,event,head_branch,head_sha,run_number,
                            html_url,api_url,metadata_blob,metadata_sha256,
                            first_seen_at
                        ) VALUES (
                            :repo_key,:run_id,:run_attempt,:created_at,:updated_at,
                            :run_started_at,:status,:conclusion,:workflow_id,
                            :workflow_name,:event,:head_branch,:head_sha,
                            :run_number,:html_url,:api_url,:metadata_blob,
                            :metadata_sha256,:first_seen_at
                        )
                        """,
                        {**record, "first_seen_at": now},
                    )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO window_runs(
                        window_id,repo_key,run_id,run_attempt,metadata_sha256
                    ) VALUES (?,?,?,?,?)
                    """,
                    (window_id, repo_key, run_id, run_attempt, metadata_sha),
                )

            conn.execute(
                """
                INSERT INTO window_pages(
                    window_id,page_no,total_count,item_count,
                    distinct_item_count,duplicate_item_count,payload_sha256,
                    run_keys_sha256,fetched_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    window_id,
                    page_no,
                    total,
                    len(page.workflow_runs),
                    len(set(page_keys)),
                    duplicate_in_page,
                    page.payload_sha256,
                    page_key_digest,
                    now,
                ),
            )
            aggregates = conn.execute(
                """
                SELECT COUNT(*) AS pages_done,
                       COALESCE(SUM(item_count),0) AS raw_items,
                       COALESCE(SUM(duplicate_item_count),0) AS duplicate_items
                FROM window_pages WHERE window_id=?
                """,
                (window_id,),
            ).fetchone()
            distinct_items = int(
                conn.execute(
                    "SELECT COUNT(*) FROM window_runs WHERE window_id=?",
                    (window_id,),
                ).fetchone()[0]
            )
            pages_done = int(aggregates["pages_done"])
            raw_items = int(aggregates["raw_items"])
            duplicate_items = raw_items - distinct_items
            status = "fetching"
            digest: str | None = None
            if pages_done == expected_pages:
                page_numbers = [
                    int(item[0])
                    for item in conn.execute(
                        """
                        SELECT page_no FROM window_pages
                        WHERE window_id=? ORDER BY page_no
                        """,
                        (window_id,),
                    )
                ]
                if page_numbers != list(range(1, expected_pages + 1)):
                    raise MalformedAPIError(
                        f"window {window_id} has non-contiguous page closure"
                    )
                if raw_items != total:
                    raise PaginationDrift(
                        f"window {window_id} raw item count {raw_items} "
                        f"does not equal total_count {total}",
                        observed_total=total,
                    )
                if distinct_items != total:
                    raise PaginationDrift(
                        f"window {window_id} returned {distinct_items} distinct "
                        f"runs for total_count={total}; duplicates make the "
                        "enumeration incomplete",
                        observed_total=total,
                    )
                digest = _hash_lines(
                    str(item["repo_key"])
                    + "\t"
                    + str(item["run_id"])
                    + "\t"
                    + str(item["run_attempt"])
                    + "\t"
                    + str(item["metadata_sha256"])
                    for item in conn.execute(
                        """
                        SELECT repo_key,run_id,run_attempt,metadata_sha256
                        FROM window_runs WHERE window_id=?
                        ORDER BY repo_key,run_id,run_attempt
                        """,
                        (window_id,),
                    )
                )
                status = "done"
            conn.execute(
                """
                UPDATE search_windows
                SET status=?,expected_total=?,expected_pages=?,pages_done=?,
                    raw_items=?,distinct_items=?,duplicate_items=?,
                    run_keys_sha256=?,failure_class=NULL,failure_message=NULL,
                    updated_at=?
                WHERE id=?
                """,
                (
                    status,
                    total,
                    expected_pages,
                    pages_done,
                    raw_items,
                    distinct_items,
                    duplicate_items,
                    digest,
                    now,
                    window_id,
                ),
            )
            return status == "done"

    def mark_failed(
        self,
        conn: sqlite3.Connection,
        window_id: int,
        exc: BaseException,
        *,
        redacted_message: str | None = None,
    ) -> None:
        with self._write_lock, conn:
            conn.execute(
                """
                UPDATE search_windows
                SET status='failed',failure_class=?,failure_message=?,updated_at=?
                WHERE id=? AND status NOT IN ('done','split')
                """,
                (
                    type(exc).__name__,
                    (redacted_message if redacted_message is not None else str(exc))[
                        :4000
                    ],
                    _utc_now(),
                    window_id,
                ),
            )

    def next_window(
        self, conn: sqlite3.Connection, repo_key: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM search_windows
            WHERE repo_key=? AND status IN ('open','fetching')
            ORDER BY depth,start_epoch,id LIMIT 1
            """,
            (repo_key,),
        ).fetchone()

    def progress(self) -> dict[str, Any]:
        conn = self.connect()
        try:
            conn.execute("BEGIN")
            meta = self._meta(conn)
            status_counts = {
                str(row["status"]): int(row["count"])
                for row in conn.execute(
                    """
                    SELECT status,COUNT(*) AS count
                    FROM search_windows GROUP BY status
                    """
                )
            }
            repo_done = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM repos r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM search_windows w
                        WHERE w.repo_key=r.repo_key
                          AND w.status IN ('open','fetching','failed')
                    )
                    """
                ).fetchone()[0]
            )
            return {
                "schema": PROGRESS_SCHEMA,
                "generated_at": _utc_now(),
                "database": self.path,
                "repo_list_sha256": meta.get("repo_list_sha256"),
                "repo_scope_sha256": meta.get("repo_scope_sha256"),
                "interval": {
                    "start": meta.get("start_utc"),
                    "end": meta.get("end_utc"),
                    "semantics": "[start,end)",
                },
                "smoke": meta.get("smoke") == "1",
                "repos_total": int(meta.get("repo_count", "0")),
                "repos_closed": repo_done,
                "runs": int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]),
                "requests": int(
                    conn.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
                ),
                "windows": status_counts,
            }
        finally:
            conn.close()

    def _validate_and_digests(self) -> dict[str, Any]:
        conn = self.connect()
        try:
            # One SQLite read transaction makes every count and digest belong
            # to the same WAL snapshot, even if a separate process is still
            # writing request/page progress.
            conn.execute("BEGIN")
            meta = self._meta(conn)
            required_meta = {
                "schema",
                "repo_list_sha256",
                "repo_scope_sha256",
                "repo_count",
                "unresolved_count",
                "start_epoch",
                "end_epoch",
                "script_sha256",
                "metadata_encoding",
                "smoke",
            }
            missing = sorted(required_meta - meta.keys())
            if missing:
                raise CompletionError(f"database metadata missing: {missing}")
            if meta["schema"] != SCHEMA_VERSION:
                raise CompletionError(f"unsupported database schema {meta['schema']!r}")
            if meta["metadata_encoding"] != METADATA_ENCODING:
                raise CompletionError(
                    "database workflow-run metadata encoding does not match "
                    f"{METADATA_ENCODING}"
                )
            if int(meta["unresolved_count"]) != 0:
                raise CompletionError("unresolved repository count is not zero")
            legacy_upgrades = [
                (
                    str(row["from_schema"]),
                    str(row["to_schema"]),
                    str(row["from_script_sha256"]),
                    str(row["to_script_sha256"]),
                    str(row["upgraded_at"]),
                )
                for row in conn.execute(
                    """
                    SELECT from_schema,to_schema,from_script_sha256,
                           to_script_sha256,upgraded_at
                    FROM inventory_upgrades ORDER BY id
                    """
                )
            ]
            binding_upgrades = [
                {
                    "from_schema": str(row["from_schema"]),
                    "to_schema": str(row["to_schema"]),
                    "from_script_sha256": str(
                        row["from_script_sha256"]
                    ),
                    "to_script_sha256": str(row["to_script_sha256"]),
                    "reason": str(row["reason"]),
                    "upgraded_at": str(row["upgraded_at"]),
                }
                for row in conn.execute(
                    """
                    SELECT from_schema,to_schema,from_script_sha256,
                           to_script_sha256,reason,upgraded_at
                    FROM inventory_binding_upgrades ORDER BY id
                    """
                )
            ]
            projected_upgrades = [
                (
                    row["from_schema"],
                    row["to_schema"],
                    row["from_script_sha256"],
                    row["to_script_sha256"],
                    row["upgraded_at"],
                )
                for row in binding_upgrades
            ]
            if legacy_upgrades != projected_upgrades:
                raise CompletionError(
                    "inventory producer upgrade ledgers disagree"
                )
            for index, upgrade in enumerate(binding_upgrades):
                try:
                    _validate_upgrade_reason(upgrade["reason"])
                except BindingError as exc:
                    raise CompletionError(
                        f"inventory producer upgrade {index} reason is invalid"
                    ) from exc
                if index and (
                    binding_upgrades[index - 1]["to_schema"]
                    != upgrade["from_schema"]
                    or binding_upgrades[index - 1]["to_script_sha256"]
                    != upgrade["from_script_sha256"]
                ):
                    raise CompletionError(
                        f"inventory producer upgrade {index} breaks the "
                        "upgrade chain"
                    )
            if binding_upgrades and (
                binding_upgrades[-1]["to_schema"] != SCHEMA_VERSION
                or binding_upgrades[-1]["to_script_sha256"]
                != meta["script_sha256"]
            ):
                raise CompletionError(
                    "inventory producer upgrade chain does not bind the "
                    "completed producer"
                )
            start = int(meta["start_epoch"])
            end = int(meta["end_epoch"])

            repos = list(
                conn.execute(
                    """
                    SELECT repo_key,owner,name,canonical,ordinal
                    FROM repos ORDER BY ordinal
                    """
                )
            )
            if len(repos) != int(meta["repo_count"]):
                raise CompletionError("database repository count differs from binding")
            scope_digest = _hash_lines(str(row["repo_key"]) for row in repos)
            if scope_digest != meta["repo_scope_sha256"]:
                raise CompletionError("database repository scope digest mismatch")

            unfinished = list(
                conn.execute(
                    """
                    SELECT id,repo_key,status,failure_class
                    FROM search_windows
                    WHERE status IN ('open','fetching','failed')
                    ORDER BY repo_key,start_epoch
                    LIMIT 10
                    """
                )
            )
            if unfinished:
                sample = ", ".join(
                    f"{row['repo_key']}:{row['id']}={row['status']}"
                    for row in unfinished
                )
                raise CompletionError(f"inventory has open/failed windows: {sample}")
            convergence_left = int(
                conn.execute("SELECT COUNT(*) FROM window_convergence").fetchone()[0]
            )
            if convergence_left:
                raise CompletionError(
                    f"inventory has {convergence_left} unresolved convergence proofs"
                )
            orphan_proof = conn.execute(
                """
                SELECT proof.window_id
                FROM (
                    SELECT window_id FROM convergence_passes
                    UNION
                    SELECT window_id FROM convergence_pass_pages
                    UNION
                    SELECT window_id FROM convergence_pass_runs
                    UNION
                    SELECT window_id FROM convergence_runs
                ) proof
                LEFT JOIN window_union_closures closure
                  ON closure.window_id=proof.window_id
                WHERE closure.window_id IS NULL
                LIMIT 1
                """
            ).fetchone()
            if orphan_proof is not None:
                raise CompletionError(
                    "inventory has convergence proof rows without a union "
                    f"closure for window {orphan_proof['window_id']}"
                )
            invalid_union_window = conn.execute(
                """
                SELECT closure.window_id
                FROM window_union_closures closure
                JOIN search_windows window ON window.id=closure.window_id
                WHERE window.status != 'done'
                   OR window.end_epoch - window.start_epoch != 1
                LIMIT 1
                """
            ).fetchone()
            if invalid_union_window is not None:
                raise CompletionError(
                    "inventory union closure is attached to an invalid window "
                    f"{invalid_union_window['window_id']}"
                )

            all_windows = list(
                conn.execute(
                    """
                    SELECT * FROM search_windows
                    ORDER BY repo_key,start_epoch,end_epoch,id
                    """
                )
            )
            by_repo: dict[str, list[sqlite3.Row]] = {}
            by_parent: dict[int, list[sqlite3.Row]] = {}
            for row in all_windows:
                by_repo.setdefault(str(row["repo_key"]), []).append(row)
                if row["parent_id"] is not None:
                    by_parent.setdefault(int(row["parent_id"]), []).append(row)

            leaf_ids: list[int] = []
            union_closure_lines: list[str] = []
            for repo in repos:
                repo_key = str(repo["repo_key"])
                windows = by_repo.get(repo_key, [])
                roots = [row for row in windows if row["parent_id"] is None]
                if len(roots) != 1:
                    raise CompletionError(
                        f"{repo_key} has {len(roots)} root search windows"
                    )
                root = roots[0]
                if (
                    int(root["start_epoch"]) != start
                    or int(root["end_epoch"]) != end
                ):
                    raise CompletionError(f"{repo_key} root interval binding mismatch")

                leaves = [row for row in windows if row["status"] == "done"]
                leaves.sort(key=lambda row: int(row["start_epoch"]))
                cursor = start
                for leaf in leaves:
                    leaf_start = int(leaf["start_epoch"])
                    leaf_end = int(leaf["end_epoch"])
                    if leaf_start != cursor:
                        relation = "overlap" if leaf_start < cursor else "gap"
                        raise CompletionError(
                            f"{repo_key} leaf-window {relation} at "
                            f"{format_utc_instant(cursor)}"
                        )
                    if leaf_end <= leaf_start:
                        raise CompletionError(f"{repo_key} has an empty leaf window")
                    cursor = leaf_end
                    leaf_ids.append(int(leaf["id"]))
                    total = int(leaf["expected_total"])
                    expected_pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
                    if total > GITHUB_FILTER_LIMIT:
                        raise CompletionError(
                            f"dense leaf window {leaf['id']} was not split"
                        )
                    window_id = int(leaf["id"])
                    pages = list(
                        conn.execute(
                            """
                            SELECT * FROM window_pages
                            WHERE window_id=? ORDER BY page_no
                            """,
                            (window_id,),
                        )
                    )
                    union = conn.execute(
                        """
                        SELECT * FROM window_union_closures
                        WHERE window_id=?
                        """,
                        (window_id,),
                    ).fetchone()
                    if union is None:
                        if (
                            int(leaf["expected_pages"]) != expected_pages
                            or int(leaf["pages_done"]) != expected_pages
                            or int(leaf["raw_items"]) != total
                            or int(leaf["distinct_items"]) != total
                            or int(leaf["duplicate_items"]) != 0
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} has incomplete "
                                "page/count closure"
                            )
                        if [int(page["page_no"]) for page in pages] != list(
                            range(1, expected_pages + 1)
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} page sequence "
                                "is incomplete"
                            )
                        if any(
                            int(page["total_count"]) != total for page in pages
                        ):
                            raise CompletionError(
                                f"leaf window {leaf['id']} has unstable "
                                "total_count"
                            )
                        stale_proof_rows = int(
                            conn.execute(
                                """
                                SELECT
                                  (SELECT COUNT(*)
                                   FROM convergence_passes
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_pass_pages
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_pass_runs
                                   WHERE window_id=?)
                                + (SELECT COUNT(*)
                                   FROM convergence_runs
                                   WHERE window_id=?)
                                """,
                                (
                                    window_id,
                                    window_id,
                                    window_id,
                                    window_id,
                                ),
                            ).fetchone()[0]
                        )
                        if stale_proof_rows:
                            raise CompletionError(
                                f"ordinary leaf window {leaf['id']} retains "
                                "convergence proof rows"
                            )
                    else:
                        if leaf_end - leaf_start != 1:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} is not one second"
                            )
                        if pages:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} also has "
                                "ordinary page rows"
                            )
                        passes = list(
                            conn.execute(
                                """
                                SELECT * FROM convergence_passes
                                WHERE window_id=? ORDER BY pass_no
                                """,
                                (window_id,),
                            )
                        )
                        pass_numbers = [
                            int(item["pass_no"]) for item in passes
                        ]
                        first_pass_no = int(union["first_pass_no"])
                        last_pass_no = int(union["last_pass_no"])
                        if (
                            len(passes) != int(union["pass_count"])
                            or not passes
                            or pass_numbers
                            != list(
                                range(first_pass_no, last_pass_no + 1)
                            )
                        ):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} has an "
                                "invalid pass sequence"
                            )
                        observed_run_passes: dict[
                            tuple[str, int, int], list[int]
                        ] = {}
                        observed_run_metadata: dict[
                            tuple[str, int, int], str
                        ] = {}
                        for proof_pass in passes:
                            pass_no = int(proof_pass["pass_no"])
                            pass_raw = int(proof_pass["raw_item_count"])
                            pass_distinct = int(
                                proof_pass["distinct_item_count"]
                            )
                            if (
                                int(proof_pass["total_count"]) != total
                                or int(proof_pass["page_count"])
                                != expected_pages
                                or pass_raw != total
                                or pass_distinct > total
                                or int(proof_pass["duplicate_item_count"])
                                != pass_raw - pass_distinct
                                or int(
                                    proof_pass["accumulated_distinct_count"]
                                )
                                > total
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} has "
                                    "invalid pass accounting"
                                )
                            proof_pages = list(
                                conn.execute(
                                    """
                                    SELECT * FROM convergence_pass_pages
                                    WHERE window_id=? AND pass_no=?
                                    ORDER BY page_no
                                    """,
                                    (window_id, pass_no),
                                )
                            )
                            if [
                                int(page["page_no"])
                                for page in proof_pages
                            ] != list(range(1, expected_pages + 1)):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} page sequence is incomplete"
                                )
                            page_lines: list[str] = []
                            for page in proof_pages:
                                page_no = int(page["page_no"])
                                item_count = int(page["item_count"])
                                page_distinct = int(
                                    page["distinct_item_count"]
                                )
                                expected_items = (
                                    DEFAULT_PER_PAGE
                                    if page_no < expected_pages
                                    else total
                                    - DEFAULT_PER_PAGE
                                    * (expected_pages - 1)
                                )
                                if (
                                    int(page["total_count"]) != total
                                    or item_count != expected_items
                                    or page_distinct > item_count
                                    or int(page["duplicate_item_count"])
                                    != item_count - page_distinct
                                ):
                                    raise CompletionError(
                                        f"union leaf window {leaf['id']} "
                                        f"pass {pass_no} page accounting "
                                        "is invalid"
                                    )
                                page_lines.append(
                                    f"{page_no}\t{page['total_count']}\t"
                                    f"{item_count}\t{page_distinct}\t"
                                    f"{page['duplicate_item_count']}\t"
                                    f"{page['payload_sha256']}\t"
                                    f"{page['run_keys_sha256']}"
                                )
                            if (
                                sum(
                                    int(page["item_count"])
                                    for page in proof_pages
                                )
                                != pass_raw
                                or _hash_lines(page_lines)
                                != str(
                                    proof_pass[
                                        "page_payload_set_sha256"
                                    ]
                                )
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} page proof digest mismatch"
                                )
                            pass_members = list(
                                conn.execute(
                                    """
                                    SELECT repo_key,run_id,run_attempt,
                                           metadata_sha256
                                    FROM convergence_pass_runs
                                    WHERE window_id=? AND pass_no=?
                                    ORDER BY repo_key,run_id,run_attempt
                                    """,
                                    (window_id, pass_no),
                                )
                            )
                            pass_member_digest = _hash_lines(
                                f"{member['repo_key']}\t"
                                f"{member['run_id']}\t"
                                f"{member['run_attempt']}\t"
                                f"{member['metadata_sha256']}"
                                for member in pass_members
                            )
                            if (
                                len(pass_members) != pass_distinct
                                or pass_member_digest
                                != str(proof_pass["run_keys_sha256"])
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} run-set proof mismatch"
                                )
                            for member in pass_members:
                                key = (
                                    str(member["repo_key"]),
                                    int(member["run_id"]),
                                    int(member["run_attempt"]),
                                )
                                metadata_sha256 = str(
                                    member["metadata_sha256"]
                                )
                                previous_metadata = (
                                    observed_run_metadata.get(key)
                                )
                                if (
                                    previous_metadata is not None
                                    and previous_metadata
                                    != metadata_sha256
                                ):
                                    raise CompletionError(
                                        f"union leaf window {leaf['id']} "
                                        f"run {key} changed metadata "
                                        "across passes"
                                    )
                                observed_run_metadata[key] = (
                                    metadata_sha256
                                )
                                observed_run_passes.setdefault(
                                    key, []
                                ).append(pass_no)
                            reconstructed_minimum = min(
                                (
                                    len(observed)
                                    for observed in (
                                        observed_run_passes.values()
                                    )
                                ),
                                default=0,
                            )
                            if (
                                int(
                                    proof_pass[
                                        "accumulated_distinct_count"
                                    ]
                                )
                                != len(observed_run_passes)
                                or int(
                                    proof_pass[
                                        "min_observation_count"
                                    ]
                                )
                                != reconstructed_minimum
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} pass "
                                    f"{pass_no} cumulative proof mismatch"
                                )
                        observed_pages = sum(
                            int(item["page_count"]) for item in passes
                        )
                        observed_items = sum(
                            int(item["raw_item_count"]) for item in passes
                        )
                        pass_set_sha256 = (
                            self._convergence_pass_set_sha256(
                                conn, window_id=window_id
                            )
                        )
                        candidates = list(
                            conn.execute(
                                """
                                SELECT * FROM convergence_runs
                                WHERE window_id=?
                                ORDER BY repo_key,run_id,run_attempt
                                """,
                                (window_id,),
                            )
                        )
                        if len(candidates) != total:
                            raise CompletionError(
                                f"union leaf window {leaf['id']} candidate "
                                "count differs from total_count"
                            )
                        candidate_digest = hashlib.sha256()
                        candidate_keys: set[
                            tuple[str, int, int]
                        ] = set()
                        for candidate in candidates:
                            candidate_key = (
                                str(candidate["repo_key"]),
                                int(candidate["run_id"]),
                                int(candidate["run_attempt"]),
                            )
                            candidate_keys.add(candidate_key)
                            observed_passes = observed_run_passes.get(
                                candidate_key, []
                            )
                            observation_count = int(
                                candidate["observation_count"]
                            )
                            first_pass = int(candidate["first_pass"])
                            last_pass = int(candidate["last_pass"])
                            if (
                                observation_count < 2
                                or observation_count
                                != len(observed_passes)
                                or not observed_passes
                                or first_pass != observed_passes[0]
                                or last_pass != observed_passes[-1]
                                or str(candidate["metadata_sha256"])
                                != observed_run_metadata.get(candidate_key)
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} candidate "
                                    "observation proof is invalid"
                                )
                            metadata_blob = candidate["metadata_blob"]
                            if not isinstance(metadata_blob, bytes):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} metadata "
                                    "is not a BLOB"
                                )
                            try:
                                metadata_bytes = zlib.decompress(metadata_blob)
                            except zlib.error as exc:
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} metadata "
                                    f"is corrupt: {exc}"
                                ) from exc
                            metadata_sha256 = _sha256_bytes(metadata_bytes)
                            if metadata_sha256 != str(
                                candidate["metadata_sha256"]
                            ):
                                raise CompletionError(
                                    f"union leaf window {leaf['id']} metadata "
                                    "digest mismatch"
                                )
                            candidate_digest.update(
                                (
                                    f"{candidate['repo_key']}\t"
                                    f"{candidate['run_id']}\t"
                                    f"{candidate['run_attempt']}\t"
                                    f"{metadata_sha256}\n"
                                ).encode()
                            )
                        if candidate_keys != set(observed_run_passes):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} pass "
                                "membership and candidate sets disagree"
                            )
                        candidate_sha256 = candidate_digest.hexdigest()
                        minimum_observations = min(
                            (
                                len(observations)
                                for observations in (
                                    observed_run_passes.values()
                                )
                            ),
                            default=0,
                        )
                        if (
                            int(union["total_count"]) != total
                            or int(union["distinct_run_count"]) != total
                            or int(union["min_observation_count"])
                            != minimum_observations
                            or minimum_observations < 2
                            or int(union["observed_page_count"])
                            != observed_pages
                            or int(union["observed_item_count"])
                            != observed_items
                            or str(union["pass_set_sha256"])
                            != pass_set_sha256
                            or str(union["run_keys_sha256"])
                            != candidate_sha256
                            or int(leaf["expected_pages"]) != expected_pages
                            or int(leaf["pages_done"]) != observed_pages
                            or int(leaf["raw_items"]) != observed_items
                            or int(leaf["distinct_items"]) != total
                            or int(leaf["duplicate_items"])
                            != observed_items - total
                        ):
                            raise CompletionError(
                                f"union leaf window {leaf['id']} closure "
                                "accounting is invalid"
                            )
                        union_closure_lines.append(
                            f"U\t{repo_key}\t{leaf_start}\t{leaf_end}\t"
                            f"{union['pass_count']}\t"
                            f"{union['first_pass_no']}\t"
                            f"{union['last_pass_no']}\t{observed_pages}\t"
                            f"{observed_items}\t{total}\t"
                            f"{minimum_observations}\t{pass_set_sha256}\t"
                            f"{candidate_sha256}"
                        )
                    actual_leaf_digest = _hash_lines(
                        str(item["repo_key"])
                        + "\t"
                        + str(item["run_id"])
                        + "\t"
                        + str(item["run_attempt"])
                        + "\t"
                        + str(item["metadata_sha256"])
                        for item in conn.execute(
                            """
                            SELECT repo_key,run_id,run_attempt,metadata_sha256
                            FROM window_runs WHERE window_id=?
                            ORDER BY repo_key,run_id,run_attempt
                            """,
                            (window_id,),
                        )
                    )
                    if actual_leaf_digest != str(leaf["run_keys_sha256"]):
                        raise CompletionError(
                            f"leaf window {leaf['id']} run-set digest mismatch"
                        )
                if cursor != end:
                    raise CompletionError(
                        f"{repo_key} leaf windows stop at "
                        f"{format_utc_instant(cursor)}, expected "
                        f"{format_utc_instant(end)}"
                    )

                for window in windows:
                    window_id = int(window["id"])
                    children = sorted(
                        by_parent.get(window_id, []),
                        key=lambda row: int(row["start_epoch"]),
                    )
                    if window["status"] == "done":
                        if children:
                            raise CompletionError(
                                f"done window {window_id} unexpectedly has children"
                            )
                        continue
                    if window["status"] != "split":
                        raise CompletionError(
                            f"window {window_id} has nonterminal status "
                            f"{window['status']!r}"
                        )
                    if len(children) != 2:
                        raise CompletionError(
                            f"split window {window_id} has {len(children)} children"
                        )
                    if (
                        int(children[0]["start_epoch"])
                        != int(window["start_epoch"])
                        or int(children[0]["end_epoch"])
                        != int(children[1]["start_epoch"])
                        or int(children[1]["end_epoch"]) != int(window["end_epoch"])
                    ):
                        raise CompletionError(
                            f"split window {window_id} children overlap or leave a gap"
                        )
                    child_total = sum(int(child["expected_total"]) for child in children)
                    if child_total != int(window["expected_total"]):
                        raise CompletionError(
                            f"split window {window_id} count changed: parent "
                            f"{window['expected_total']} != children {child_total}"
                        )

            if not leaf_ids and repos:
                raise CompletionError("inventory has no closed leaf windows")
            if leaf_ids:
                overlap = conn.execute(
                    """
                    SELECT wr.repo_key,wr.run_id,wr.run_attempt,
                           COUNT(*) AS appearances
                    FROM window_runs wr
                    JOIN search_windows w ON w.id=wr.window_id
                    WHERE w.status='done'
                    GROUP BY wr.repo_key,wr.run_id,wr.run_attempt
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                ).fetchone()
                if overlap is not None:
                    raise CompletionError(
                        "workflow run appears in overlapping leaf windows: "
                        f"{overlap['repo_key']}#{overlap['run_id']} attempt "
                        f"{overlap['run_attempt']}"
                    )

            unlinked = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM runs r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM window_runs wr
                        WHERE wr.repo_key=r.repo_key
                          AND wr.run_id=r.run_id
                          AND wr.run_attempt=r.run_attempt
                    )
                    """
                ).fetchone()[0]
            )
            if unlinked:
                raise CompletionError(f"database contains {unlinked} unlinked runs")

            run_count = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            run_digest = hashlib.sha256()
            for row in conn.execute(
                """
                SELECT repo_key,run_id,run_attempt,metadata_blob,metadata_sha256
                FROM runs ORDER BY repo_key,run_id,run_attempt
                """
            ):
                blob = row["metadata_blob"]
                if not isinstance(blob, bytes):
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} metadata is not a BLOB"
                    )
                try:
                    metadata_bytes = zlib.decompress(blob)
                except zlib.error as exc:
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} metadata is corrupt: "
                        f"{exc}"
                    ) from exc
                actual_metadata_sha = _sha256_bytes(metadata_bytes)
                if actual_metadata_sha != str(row["metadata_sha256"]):
                    raise CompletionError(
                        f"run {row['repo_key']}#{row['run_id']} metadata digest mismatch"
                    )
                line = (
                    f"{row['repo_key']}\t{row['run_id']}\t{row['run_attempt']}\t"
                    f"{actual_metadata_sha}\n"
                )
                run_digest.update(line.encode("utf-8"))
            run_set_sha = run_digest.hexdigest()

            closure_lines = [
                f"W\t{row['repo_key']}\t{row['start_epoch']}\t"
                f"{row['end_epoch']}\t{row['status']}\t"
                f"{row['expected_total']}\t{row['run_keys_sha256'] or ''}"
                for row in all_windows
            ]
            closure_lines.extend(
                f"P\t{row['repo_key']}\t{row['start_epoch']}\t"
                f"{row['end_epoch']}\t{row['page_no']}\t"
                f"{row['total_count']}\t{row['item_count']}\t"
                f"{row['distinct_item_count']}\t{row['duplicate_item_count']}\t"
                f"{row['payload_sha256']}\t{row['run_keys_sha256']}"
                for row in conn.execute(
                    """
                    SELECT w.repo_key,w.start_epoch,w.end_epoch,p.page_no,
                           p.total_count,p.item_count,p.distinct_item_count,
                           p.duplicate_item_count,p.payload_sha256,
                           p.run_keys_sha256
                    FROM window_pages p
                    JOIN search_windows w ON w.id=p.window_id
                    ORDER BY w.repo_key,w.start_epoch,w.end_epoch,p.page_no
                    """
                )
            )
            closure_lines.extend(sorted(union_closure_lines))
            closure_sha = _hash_lines(closure_lines)
            logical_document = {
                "schema": SCHEMA_VERSION,
                "repo_list_sha256": meta["repo_list_sha256"],
                "repo_scope_sha256": meta["repo_scope_sha256"],
                "start_epoch": start,
                "end_epoch": end,
                "script_sha256": meta["script_sha256"],
                "repo_count": len(repos),
                "run_count": run_count,
                "run_set_sha256": run_set_sha,
                "window_closure_sha256": closure_sha,
                "binding_upgrades_sha256": _sha256_json(
                    binding_upgrades
                ),
            }
            return {
                "meta": meta,
                "repo_count": len(repos),
                "run_count": run_count,
                "run_set_sha256": run_set_sha,
                "window_closure_sha256": closure_sha,
                "db_logical_sha256": _sha256_json(logical_document),
                "leaf_window_count": len(leaf_ids),
                "request_count": int(
                    conn.execute("SELECT COUNT(*) FROM request_ledger").fetchone()[0]
                ),
                "binding_upgrades": binding_upgrades,
            }
        finally:
            conn.close()

    def completion_receipt(self) -> dict[str, Any]:
        validated = self._validate_and_digests()
        meta = validated["meta"]
        smoke = meta["smoke"] == "1"
        return {
            "schema": RECEIPT_SCHEMA,
            "completed_at": _utc_now(),
            "production_complete": not smoke,
            "mode": "smoke" if smoke else "production",
            "database": self.path,
            "repo_list": {
                "path": meta["repo_list_path"],
                "sha256": meta["repo_list_sha256"],
                "scope_sha256": meta["repo_scope_sha256"],
                "repos": validated["repo_count"],
                "original_repos": int(meta["original_repo_count"]),
                "unresolved": int(meta["unresolved_count"]),
            },
            "interval": {
                "start": meta["start_utc"],
                "end": meta["end_utc"],
                "semantics": "[start,end)",
            },
            "script_sha256": meta["script_sha256"],
            "metadata_encoding": meta["metadata_encoding"],
            "run_count": validated["run_count"],
            "leaf_window_count": validated["leaf_window_count"],
            "request_count": validated["request_count"],
            "run_set_sha256": validated["run_set_sha256"],
            "window_closure_sha256": validated["window_closure_sha256"],
            "db_logical_sha256": validated["db_logical_sha256"],
            "binding_upgrades": validated["binding_upgrades"],
        }


def atomic_write_json(path: str | os.PathLike[str], document: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class GitHubActionsInventory:
    """Orchestrate bounded window discovery, pagination, and receipts."""

    def __init__(
        self,
        *,
        db_path: str | os.PathLike[str],
        scope: RepoScope,
        start: str | int,
        end: str | int,
        tokens: Sequence[str],
        resume: bool = False,
        allow_script_upgrade_from_sha256: str | None = None,
        script_upgrade_reason: str | None = None,
        progress_path: str | os.PathLike[str] | None = None,
        requester: Callable[
            [str, str, Mapping[str, str], float], HTTPResponse
        ] = _default_requester,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        max_attempts: int = 12,
        progress_interval_seconds: float = 5.0,
        script_path: str | os.PathLike[str] = __file__,
    ):
        self.scope = scope
        self.start_epoch = parse_utc_instant(start) if isinstance(start, str) else start
        self.end_epoch = parse_utc_instant(end) if isinstance(end, str) else end
        if self.start_epoch >= self.end_epoch:
            raise BindingError("inventory interval must satisfy start < end")
        script_bytes = Path(script_path).resolve().read_bytes()
        self.script_sha256 = _sha256_bytes(script_bytes)
        self.db = InventoryDB(db_path)
        self.db.bind(
            scope=scope,
            start_epoch=self.start_epoch,
            end_epoch=self.end_epoch,
            script_sha256=self.script_sha256,
            resume=resume,
            allow_script_upgrade_from_sha256=(
                allow_script_upgrade_from_sha256
            ),
            script_upgrade_reason=script_upgrade_reason,
        )
        self.progress_path = (
            str(Path(progress_path).expanduser().resolve())
            if progress_path is not None
            else None
        )
        if progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be non-negative")
        self.progress_interval_seconds = progress_interval_seconds
        self._progress_clock = time.monotonic
        self._last_progress_monotonic: float | None = None
        pool = TokenPool(tokens, clock=clock, sleeper=sleeper)
        self.client = GitHubClient(
            pool,
            requester=requester,
            sleeper=sleeper,
            max_attempts=max_attempts,
        )
        self._progress_lock = threading.Lock()

    def _write_progress(self, *, force: bool = False) -> None:
        if self.progress_path is None:
            return
        with self._progress_lock:
            now = self._progress_clock()
            if (
                not force
                and self._last_progress_monotonic is not None
                and now - self._last_progress_monotonic
                < self.progress_interval_seconds
            ):
                return
            atomic_write_json(self.progress_path, self.db.progress())
            self._last_progress_monotonic = now

    def _converge_one_second(
        self,
        conn: sqlite3.Connection,
        repo: Repo,
        row: sqlite3.Row,
        ledger: Callable[..., None],
    ) -> None:
        window_id = int(row["id"])
        self.db.prepare_convergence(conn, row)
        expected_total = (
            None if row["expected_total"] is None else int(row["expected_total"])
        )
        for _ in range(CONVERGENCE_MAX_PASSES):
            pages: list[PageResponse] = []
            first = self.client.get_workflow_runs(
                repo=repo,
                start_epoch=int(row["start_epoch"]),
                end_epoch=int(row["end_epoch"]),
                page=1,
                per_page=DEFAULT_PER_PAGE,
                ledger=ledger,
            )
            pages.append(first)
            page_count = max(
                1, math.ceil(first.total_count / DEFAULT_PER_PAGE)
            )
            if first.total_count > GITHUB_FILTER_LIMIT:
                raise UnstableEnumerationError(
                    f"{repo.canonical} one-second convergence returned "
                    f"{first.total_count} runs, above the provable REST limit"
                )
            for page_no in range(2, page_count + 1):
                pages.append(
                    self.client.get_workflow_runs(
                        repo=repo,
                        start_epoch=int(row["start_epoch"]),
                        end_epoch=int(row["end_epoch"]),
                        page=page_no,
                        per_page=DEFAULT_PER_PAGE,
                        ledger=ledger,
                    )
                )
            try:
                total = pages[0].total_count
                if expected_total is not None and total != expected_total:
                    raise PaginationDrift(
                        f"one-second convergence total changed "
                        f"{expected_total} -> {total}",
                        observed_total=total,
                    )
                complete, _digest = self.db.accumulate_convergence_pass(
                    conn, row, pages
                )
            except PaginationDrift as exc:
                raise UnstableEnumerationError(
                    f"window {window_id} convergence pass is malformed: {exc}"
                ) from exc
            if not complete:
                self._write_progress()
                continue
            self._write_progress()
            return
        raise UnstableEnumerationError(
            f"window {window_id} did not accumulate total_count unique runs "
            "with two stable metadata observations each in "
            f"{CONVERGENCE_MAX_PASSES} passes"
        )

    def _process_repo(self, repo: Repo) -> None:
        conn = self.db.connect()
        try:
            while True:
                row = self.db.next_window(conn, repo.key)
                if row is None:
                    return
                window_id = int(row["id"])
                active_page = 1

                def ledger(**fields: Any) -> None:
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        **fields,
                    )

                try:
                    if self.db.convergence_state(conn, window_id) is not None:
                        self._converge_one_second(conn, repo, row, ledger)
                        continue
                    expected_total = row["expected_total"]
                    if expected_total is None:
                        active_page = 1
                        first_page = self.client.get_workflow_runs(
                            repo=repo,
                            start_epoch=int(row["start_epoch"]),
                            end_epoch=int(row["end_epoch"]),
                            page=1,
                            per_page=DEFAULT_PER_PAGE,
                            ledger=ledger,
                        )
                        if first_page.total_count > GITHUB_FILTER_LIMIT:
                            self.db.split_window(
                                conn, row, observed_total=first_page.total_count
                            )
                            self._write_progress()
                            continue
                        self.db.store_page(
                            conn, row, page_no=1, page=first_page
                        )
                        self._write_progress()
                        row = conn.execute(
                            "SELECT * FROM search_windows WHERE id=?",
                            (window_id,),
                        ).fetchone()
                        assert row is not None
                    if row["status"] == "done":
                        continue
                    total = int(row["expected_total"])
                    pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
                    completed_pages = {
                        int(item[0])
                        for item in conn.execute(
                            "SELECT page_no FROM window_pages WHERE window_id=?",
                            (window_id,),
                        )
                    }
                    for page_no in range(1, pages + 1):
                        if page_no in completed_pages:
                            continue
                        active_page = page_no
                        response = self.client.get_workflow_runs(
                            repo=repo,
                            start_epoch=int(row["start_epoch"]),
                            end_epoch=int(row["end_epoch"]),
                            page=page_no,
                            per_page=DEFAULT_PER_PAGE,
                            ledger=ledger,
                        )
                        self.db.store_page(
                            conn, row, page_no=page_no, page=response
                        )
                        self._write_progress()
                except PaginationDrift as exc:
                    action = self.db.recover_pagination_drift(
                        conn,
                        row,
                        observed_total=exc.observed_total,
                        reason=str(exc),
                    )
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        endpoint=f"/repos/{repo.owner}/{repo.name}/actions/runs",
                        page=active_page,
                        per_page=DEFAULT_PER_PAGE,
                        attempt=0,
                        http_status=None,
                        outcome=f"pagination_drift_{action}",
                        latency_ms=0,
                        error_class=type(exc).__name__,
                        error_message=self.client.redact(exc),
                    )
                    self._write_progress(force=True)
                    continue
                except BaseException as exc:
                    self.db.record_request(
                        conn,
                        repo_key=repo.key,
                        window_id=window_id,
                        endpoint=f"/repos/{repo.owner}/{repo.name}/actions/runs",
                        page=active_page,
                        per_page=DEFAULT_PER_PAGE,
                        attempt=0,
                        http_status=None,
                        outcome="window_error",
                        latency_ms=0,
                        error_class=type(exc).__name__,
                        error_message=self.client.redact(exc),
                    )
                    self.db.mark_failed(
                        conn,
                        window_id,
                        exc,
                        redacted_message=self.client.redact(exc),
                    )
                    self._write_progress(force=True)
                    raise
        finally:
            conn.close()

    def run(self, *, workers: int = 1) -> dict[str, Any]:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self._write_progress(force=True)
        errors: list[tuple[str, BaseException]] = []
        if workers == 1:
            for repo in self.scope.repos:
                try:
                    self._process_repo(repo)
                except BaseException as exc:
                    errors.append((repo.canonical, exc))
                    break
        else:
            with ThreadPoolExecutor(
                max_workers=min(workers, len(self.scope.repos)),
                thread_name_prefix="ci-inventory",
            ) as executor:
                futures = {
                    executor.submit(self._process_repo, repo): repo
                    for repo in self.scope.repos
                }
                for future in as_completed(futures):
                    repo = futures[future]
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append((repo.canonical, exc))
        self._write_progress(force=True)
        if errors:
            details = "; ".join(
                f"{repo}: {type(exc).__name__}: {self.client.redact(exc)}"
                for repo, exc in errors[:10]
            )
            raise InventoryError(
                f"inventory failed for {len(errors)} repository/repositories: {details}"
            ) from errors[0][1]
        return self.db.progress()

    def write_completion_receipt(
        self, path: str | os.PathLike[str]
    ) -> dict[str, Any]:
        receipt = self.db.completion_receipt()
        atomic_write_json(path, receipt)
        return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory GitHub Actions workflow-run metadata only"
    )
    parser.add_argument(
        "--mode",
        choices=("inventory-only",),
        default="inventory-only",
        help="this stage intentionally supports metadata inventory only",
    )
    parser.add_argument("--repo-list", default=DEFAULT_REPO_LIST)
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", help="inclusive UTC boundary")
    parser.add_argument("--end", help="exclusive UTC boundary")
    parser.add_argument("--tokens", help="newline-delimited GitHub token pool")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-inventory-script-upgrade-from-sha256",
        help=(
            "explicitly authorize one v2 to v3 resume migration from this "
            "exact previously bound producer SHA-256"
        ),
    )
    parser.add_argument(
        "--inventory-script-upgrade-reason",
        help=(
            "required printable audit reason for an explicitly authorized "
            "inventory producer migration"
        ),
    )
    parser.add_argument(
        "--progress",
        help="atomic progress JSON (default: <db>.progress.json)",
    )
    parser.add_argument(
        "--receipt",
        help="atomic completion JSON (default: <db>.completion.json)",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="print current SQLite progress without making requests",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-repos", type=int)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="minimum seconds between nonterminal atomic progress snapshots",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.progress_only:
        progress = InventoryDB(args.db).progress()
        print(json.dumps(progress, indent=2, sort_keys=True))
        return 0
    if not args.start or not args.end:
        parser.error("--start and --end are required unless --progress-only is used")
    if args.max_repos is not None and not args.smoke:
        parser.error("--max-repos requires explicit --smoke")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")

    try:
        scope = load_repo_scope(
            args.repo_list, smoke=args.smoke, max_repos=args.max_repos
        )
        tokens = load_token_pool(args.tokens)
        progress_path = args.progress or f"{args.db}.progress.json"
        receipt_path = args.receipt or f"{args.db}.completion.json"
        inventory = GitHubActionsInventory(
            db_path=args.db,
            scope=scope,
            start=args.start,
            end=args.end,
            tokens=tokens,
            resume=args.resume,
            allow_script_upgrade_from_sha256=(
                args.allow_inventory_script_upgrade_from_sha256
            ),
            script_upgrade_reason=args.inventory_script_upgrade_reason,
            progress_path=progress_path,
            max_attempts=args.max_attempts,
            progress_interval_seconds=args.progress_interval,
        )
        progress = inventory.run(workers=args.workers)
        receipt = inventory.write_completion_receipt(receipt_path)
    except InventoryError as exc:
        print(f"[ci-stream-inventory] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"progress": progress, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
