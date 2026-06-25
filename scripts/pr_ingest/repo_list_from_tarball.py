#!/usr/bin/env python3
"""Resolve owner/repo for ALL C/C++ corpus repos from the data-cpp_all tarball.

Single-pass streaming over the (huge, ~235 GiB) zstd tarball whose members are
``cpp_all/<repo>/...`` (a real, curated C/C++ training corpus where each repo
keeps its ``.git`` directory). For every ``cpp_all/<repo>/.git/config`` member we
read ONLY that one file's bytes, parse the ``[remote "origin"]`` url, and derive
``owner/repo``. Every other member is skipped (its data block is seeked over) so
the pass is I/O-bound on decompression but never materializes whole repos.

We stream by piping the ``zstd`` CLI into Python's ``tarfile`` in streaming mode
(``tarfile.open(fileobj=..., mode="r|")``) -- the same ``zstd|tarfile r|`` shape
used by scripts/streaming_reindex_commits.py -- so we do not depend on the
``zstandard`` Python module being installed in the .venv.

RULE #1 (fail loud): if a repo has a ``.git/config`` we cannot parse into an
owner/repo, we DO NOT silently drop it -- we record it under ``unresolved`` with
the reason. A repo subtree that has no ``.git/config`` at all is also recorded as
unresolved (reason ``no-git-config``). The only thing that aborts the whole run
is an unexpected structural error (e.g. the zstd pipe dying), which raises.

Outputs ``outputs/pr_ingest/repo_list.json``::

    {
      "tarball": "...",
      "generated_utc": "...",
      "counts": {"repos": N, "unresolved": M, "subtrees_seen": N+M},
      "repos": [{"name": "...", "owner_repo": "owner/repo", "url": "..."}],
      "unresolved": [{"name": "...", "reason": "...", "url": "..."|null}]
    }
"""

from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tarfile
from typing import Iterator, Optional

# Working-tree layout:  cpp_all/<repo>/.git/config
# Bare-clone layout:     cpp_all/<repo>.bare/config   (config at top level, no .git/)
# Both capture <repo> (no slashes allowed in <repo>).
_GIT_CONFIG_RE = re.compile(r"^cpp_all/([^/]+)/(?:\.git/config|config)$")
# cpp_all/<repo>/...          -- capture <repo> to count distinct subtrees seen.
_SUBTREE_RE = re.compile(r"^cpp_all/([^/]+)/")

# github.com/OWNER/REPO(.git)  or  git@github.com:OWNER/REPO(.git)
_GH_HTTPS_RE = re.compile(
    r"github\.com[/:]+(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def _parse_owner_repo(url: str) -> Optional[str]:
    """Return ``owner/repo`` for a GitHub remote URL, else None.

    Handles https://github.com/OWNER/REPO(.git), http://, git://,
    ssh://git@github.com/OWNER/REPO(.git), and git@github.com:OWNER/REPO(.git).
    """
    url = url.strip()
    if not url:
        return None
    m = _GH_HTTPS_RE.search(url)
    if not m:
        return None
    owner = m.group("owner").strip()
    repo = m.group("repo").strip()
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _remote_url_from_config(raw: bytes) -> Optional[str]:
    """Extract the [remote "origin"] url from a git config file's bytes."""
    text = raw.decode("utf-8", errors="replace")
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read_string(text)
    except configparser.Error:
        cp = None
    if cp is not None:
        # git config section header is literally: [remote "origin"]
        for sect in cp.sections():
            if sect.strip().lower() in ('remote "origin"', "remote origin"):
                if cp.has_option(sect, "url"):
                    return cp.get(sect, "url")
        # No origin remote -- fall back to ANY remote url so we still resolve.
        for sect in cp.sections():
            if sect.strip().lower().startswith("remote") and cp.has_option(
                sect, "url"
            ):
                return cp.get(sect, "url")
    # Last-resort manual scan inside [remote "origin"] (handles odd formatting
    # configparser rejects) -- still deterministic, still fails loud if absent.
    in_origin = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_origin = s.lower().replace(" ", "").startswith('[remote"origin"]')
            continue
        if in_origin and s.lower().startswith("url"):
            _, _, val = s.partition("=")
            val = val.strip()
            if val:
                return val
    return None


def _open_stream(tarball: str) -> tuple[subprocess.Popen, tarfile.TarFile]:
    """Open the zstd->tarfile streaming pipeline. Raises on failure."""
    if not os.path.isfile(tarball):
        raise FileNotFoundError(f"tarball not found: {tarball}")
    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=31", tarball],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )
    if proc.stdout is None:  # pragma: no cover - defensive
        raise RuntimeError("zstd stdout pipe was not created")
    tf = tarfile.open(fileobj=proc.stdout, mode="r|")
    return proc, tf


def iter_repo_configs(
    tarball: str, progress_every: int = 200
) -> Iterator[tuple[str, Optional[bytes], str]]:
    """Yield (repo_name, config_bytes_or_None, kind) for every repo subtree.

    ``kind`` is "config" when config_bytes is the .git/config payload, or
    "subtree" the first time a new repo subtree is seen without (yet) a config.
    We yield a "subtree" marker on first sight of each repo so the caller can
    detect repos that NEVER produced a .git/config (unresolved, fail-loud).
    """
    proc, tf = _open_stream(tarball)
    seen_subtrees: set[str] = set()
    members = 0
    try:
        for member in tf:
            members += 1
            if progress_every and members % progress_every == 0:
                sys.stderr.write(
                    f"  ... scanned {members:,} members "
                    f"({len(seen_subtrees):,} repo subtrees)\n"
                )
                sys.stderr.flush()
            name = member.name
            sm = _SUBTREE_RE.match(name)
            if sm:
                repo = sm.group(1)
                if repo not in seen_subtrees:
                    seen_subtrees.add(repo)
                    yield repo, None, "subtree"
            if not member.isfile():
                continue
            cm = _GIT_CONFIG_RE.match(name)
            if cm is None:
                continue  # skip data block -- tarfile seeks past it for us
            repo = cm.group(1)
            # ".git/config" is authoritative (working tree); top-level "config"
            # is the bare-clone layout. Tag the source so the builder prefers
            # the authoritative one when both somehow appear.
            is_git = name.endswith("/.git/config")
            f = tf.extractfile(member)
            if f is None:  # pragma: no cover - defensive
                raise RuntimeError(f"could not extract member: {name}")
            data = f.read()
            yield repo, data, ("config_git" if is_git else "config_bare")
    finally:
        try:
            tf.close()
        except Exception:
            pass
        # Drain/await zstd; surface a nonzero exit loudly (RULE #1).
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        err = b""
        if proc.stderr is not None:
            try:
                err = proc.stderr.read() or b""
            except Exception:
                err = b""
            try:
                proc.stderr.close()
            except Exception:
                pass
        rc = proc.wait()
        # rc == -13 (SIGPIPE) can happen if we stop early; we never stop early
        # here, so any nonzero rc is a real failure.
        if rc not in (0, None):
            raise RuntimeError(
                f"zstd exited with code {rc}: {err.decode('utf-8', 'replace')[:2000]}"
            )


def build_repo_list(tarball: str, progress_every: int = 200) -> dict:
    resolved: dict[str, dict] = {}
    unresolved: dict[str, dict] = {}
    subtrees: set[str] = set()
    # Track which config source produced a repo's verdict so that an
    # authoritative .git/config is never overwritten by a bare top-level config.
    verdict_src: dict[str, str] = {}  # repo -> "config_git" | "config_bare"

    for repo, data, kind in iter_repo_configs(tarball, progress_every):
        if kind == "subtree":
            subtrees.add(repo)
            continue
        # kind in ("config_git", "config_bare")
        assert data is not None
        # If we already resolved this repo from an authoritative .git/config,
        # do not let a later bare "config" downgrade it.
        if verdict_src.get(repo) == "config_git" and kind == "config_bare":
            continue
        url = _remote_url_from_config(data)
        if url is None:
            unresolved[repo] = {
                "name": repo,
                "reason": "no-remote-url-in-git-config",
                "url": None,
            }
            resolved.pop(repo, None)
            verdict_src[repo] = kind
            continue
        owner_repo = _parse_owner_repo(url)
        if owner_repo is None:
            unresolved[repo] = {
                "name": repo,
                "reason": "remote-url-not-github",
                "url": url,
            }
            resolved.pop(repo, None)
            verdict_src[repo] = kind
            continue
        resolved[repo] = {"name": repo, "owner_repo": owner_repo, "url": url}
        unresolved.pop(repo, None)
        verdict_src[repo] = kind

    # Any repo subtree that never produced a parseable .git/config -> unresolved.
    for repo in sorted(subtrees):
        if repo not in resolved and repo not in unresolved:
            unresolved[repo] = {
                "name": repo,
                "reason": "no-git-config",
                "url": None,
            }

    repos = sorted(resolved.values(), key=lambda d: d["name"].lower())
    unres = sorted(unresolved.values(), key=lambda d: d["name"].lower())
    return {
        "tarball": os.path.abspath(tarball),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "counts": {
            "repos": len(repos),
            "unresolved": len(unres),
            "subtrees_seen": len(subtrees),
        },
        "repos": repos,
        "unresolved": unres,
    }


_GHA_EVENT_TYPES = (
    "PullRequestEvent",
    "PullRequestReviewEvent",
    "PullRequestReviewCommentEvent",
    "IssueCommentEvent",
)


def _build_query(owner_repos: list[str], table_glob: str) -> str:
    """Build the GH Archive extraction SQL with the REAL repo list inlined."""
    if not owner_repos:
        raise ValueError("refusing to build query with an empty repo list")
    in_list = ", ".join("'" + r.replace("'", "''") + "'" for r in owner_repos)
    types = ", ".join("'" + t + "'" for t in _GHA_EVENT_TYPES)
    return (
        "SELECT type, repo.name AS repo_name, actor.login AS actor_login, "
        "created_at, id, payload\n"
        f"FROM `{table_glob}`\n"
        f"WHERE type IN ({types})\n"
        f"  AND repo.name IN ({in_list})\n"
    )


def _bq_dry_run_bytes(project: str, sql: str) -> int:
    """Run a bq --dry_run and return total bytes processed. Fails loud."""
    proc = subprocess.run(
        [
            "bq",
            f"--project_id={project}",
            "query",
            "--use_legacy_sql=false",
            "--dry_run",
            "--format=json",
            sql,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bq --dry_run failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    # bq prints a human line on stderr AND json on stdout with statistics.
    out = proc.stdout.strip()
    data = json.loads(out)
    stats = data.get("statistics", data)
    tbp = stats.get("totalBytesProcessed") or stats.get("query", {}).get(
        "totalBytesProcessed"
    )
    if tbp is None:
        raise RuntimeError(
            f"could not find totalBytesProcessed in dry-run output: {out[:2000]}"
        )
    return int(tbp)


# GH Archive monthly tables span 201501 .. present. Full history months count.
def _months_in_span(first: str = "201501", last: str = "202606") -> int:
    fy, fm = int(first[:4]), int(first[4:])
    ly, lm = int(last[:4]), int(last[4:])
    return (ly - fy) * 12 + (lm - fm) + 1


def run_bq_dry_run(
    repo_list_path: str,
    project: str,
    dry_month: str,
    first_month: str,
    last_month: str,
    price_per_tib: float,
) -> dict:
    with open(repo_list_path) as fh:
        rl = json.load(fh)
    all_owner_repos = [r["owner_repo"] for r in rl["repos"]]
    if not all_owner_repos:
        raise RuntimeError("repo_list.json has zero resolved repos -- aborting")
    # repo_list.json keeps one entry per corpus subtree (e.g. a working-tree
    # clone AND its ".bare" mirror both point at the same GitHub repo), so the
    # owner/repo values contain duplicates. The GH Archive filter must use the
    # UNIQUE set (order-preserving) -- duplicates would waste the IN list.
    owner_repos = list(dict.fromkeys(all_owner_repos))
    table_glob = f"githubarchive.month.{dry_month}"
    sql = _build_query(owner_repos, table_glob)
    sys.stderr.write(
        f"[bq] dry-run over {table_glob} with {len(owner_repos)} unique repos "
        f"({len(all_owner_repos)} subtree entries)...\n"
    )
    sys.stderr.flush()
    bytes_month = _bq_dry_run_bytes(project, sql)
    n_months = _months_in_span(first_month, last_month)
    bytes_full = bytes_month * n_months
    tib = 2 ** 40
    cost_month = bytes_month / tib * price_per_tib
    cost_full = bytes_full / tib * price_per_tib
    result = {
        "project": project,
        "dry_run_month": dry_month,
        "repos_in_filter": len(owner_repos),
        "subtree_entries_total": len(all_owner_repos),
        "one_month_bytes": bytes_month,
        "one_month_gib": round(bytes_month / 2 ** 30, 4),
        "one_month_cost_usd": round(cost_month, 6),
        "history_first_month": first_month,
        "history_last_month": last_month,
        "history_months": n_months,
        "full_history_bytes_est": bytes_full,
        "full_history_tib_est": round(bytes_full / tib, 4),
        "price_per_tib_usd": price_per_tib,
        "full_history_cost_usd_est": round(cost_full, 4),
        "query_table_glob_for_full_run": "githubarchive.month.20*",
    }
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tarball",
        default="/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst",
        help="path to data-cpp_all.tar.zst",
    )
    ap.add_argument(
        "--skip-tarball",
        action="store_true",
        help="skip the tarball pass; only run the bq dry-run on an existing repo_list.json",
    )
    ap.add_argument("--bq-dry-run", action="store_true", help="run the bq dry-run")
    ap.add_argument("--bq-project", default="natural-bison-491019-t9")
    ap.add_argument("--bq-dry-month", default="202512")
    ap.add_argument("--bq-first-month", default="201501")
    ap.add_argument("--bq-last-month", default="202606")
    ap.add_argument("--price-per-tib", type=float, default=6.25)
    ap.add_argument(
        "--bq-out",
        default=None,
        help="where to write the bq dry-run cost JSON (default alongside repo_list.json)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "outputs",
            "pr_ingest",
            "repo_list.json",
        ),
        help="output repo_list.json path",
    )
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args(argv)

    if not args.skip_tarball:
        sys.stderr.write(f"[repo_list_from_tarball] streaming {args.tarball}\n")
        sys.stderr.flush()
        result = build_repo_list(args.tarball, args.progress_every)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=False)
            fh.write("\n")
        c = result["counts"]
        sys.stderr.write(
            f"[repo_list_from_tarball] resolved={c['repos']} "
            f"unresolved={c['unresolved']} subtrees={c['subtrees_seen']}\n"
            f"[repo_list_from_tarball] wrote {args.out}\n"
        )
        sys.stderr.flush()

    if args.bq_dry_run:
        cost = run_bq_dry_run(
            repo_list_path=args.out,
            project=args.bq_project,
            dry_month=args.bq_dry_month,
            first_month=args.bq_first_month,
            last_month=args.bq_last_month,
            price_per_tib=args.price_per_tib,
        )
        bq_out = args.bq_out or os.path.join(
            os.path.dirname(args.out), "gharchive_cost_estimate.json"
        )
        with open(bq_out, "w") as fh:
            json.dump(cost, fh, indent=2)
            fh.write("\n")
        sys.stderr.write(
            f"[bq] one_month_bytes={cost['one_month_bytes']:,} "
            f"({cost['one_month_gib']} GiB)\n"
            f"[bq] full_history({cost['history_months']} months) "
            f"~{cost['full_history_tib_est']} TiB -> "
            f"${cost['full_history_cost_usd_est']} @ ${cost['price_per_tib_usd']}/TiB\n"
            f"[bq] wrote {bq_out}\n"
        )
        sys.stderr.flush()
        print(json.dumps(cost, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
