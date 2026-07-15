#!/usr/bin/env python3
"""Build stable project identities and GitHub PR keys for the corpus.

The git extractor (scripts/data/extract_git_history.py) records only the *bare*
repo name (the clone directory basename, e.g. ``opencv``). Symbol identity needs
a stable namespace for every network forge, while GH Archive / GitHub GraphQL
need GitHub's ``owner/repo`` slug. The authoritative source for both is the
clone's git remote:

    git -C <clone> config --get remote.origin.url

This script resolves a canonical ``project_identity`` for each network remote.
GitHub rows retain ``owner/repo`` as their project identity and additionally
carry ``owner_repo`` for PR lookup. Other forges use a lossless, exact-one-slash
``host/percent-encoded-path`` identity and intentionally omit ``owner_repo``.

RULE #1 (fail loud): a clone whose project identity cannot be resolved is NOT
silently dropped. Every unresolved clone is collected and, unless
``--allow-unresolved`` is given, the script raises and lists them. There is no
silent fallback.

Usage:
    # From one or more directories that each contain cloned repos:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo_dir ~/data/cpp_raw \
        --out /mnt/nvme/cppmega_data/pr_ingest/repo_list.json

    # From explicit clone paths:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo /path/to/opencv --repo /path/to/llvm-project \
        --out repo_list.json

    # Offline re-run: seed from / merge into a previously-stored map so clones
    # that are no longer on disk keep their explicit resolved identity:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo_dir ~/data/cpp_raw --stored_map prior_repo_list.json \
        --out repo_list.json

Output schema (repo_list.json):
    {
      "schema_version": 2,
      "repos": [
        {"bare_name": "opencv", "project_identity": "opencv/opencv",
         "owner_repo": "opencv/opencv",
         "owner": "opencv", "repo": "opencv",
         "remote_url": "git@github.com:opencv/opencv.git",
         "clone_path": "/abs/path/opencv", "source": "git_remote"},
        {"bare_name": "aosp-frameworks-av",
         "project_identity": "android.googlesource.com/platform%2Fframeworks%2Fav",
         "remote_url": "https://android.googlesource.com/platform/frameworks/av",
         "clone_path": "/abs/path/aosp-frameworks-av", "source": "git_remote"}
      ],
      "by_bare_name": {"opencv": "opencv/opencv", ...},
      "project_identities": ["android.googlesource.com/...", "opencv/opencv"],
      "repo_names": ["opencv/opencv"],  # GitHub-only legacy PR input
      "unresolved": []
    }
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

MLX_ROOT = Path(__file__).resolve().parents[2]
if str(MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_ROOT))

from cppmega.data.symbol_identity import (  # noqa: E402
    SymbolIdentityError,
    require_project_identity,
    resolve_remote_project_identity,
)


REPO_LIST_SCHEMA_VERSION = 2


def run_git(repo_path: str, args: list[str], timeout: int = 30) -> Optional[str]:
    """Run ``git -C repo_path <args>``; return stdout or None on failure.

    Mirrors scripts/data/extract_git_history.py:run_git so resolution behaves
    identically to the extractor that produced the bare names.
    """
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_owner_repo(remote_url: str) -> Optional[tuple[str, str]]:
    """Parse ``(owner, repo)`` from a GitHub remote URL, or None if not GitHub."""
    try:
        resolved = resolve_remote_project_identity(
            remote_url,
            source="git remote",
        )
    except SymbolIdentityError:
        return None
    if resolved.owner_repo is None:
        return None
    owner, repo = resolved.owner_repo.split("/", 1)
    return owner, repo


def resolve_clone(clone_path: str) -> dict:
    """Resolve a single clone to a repo_list entry (resolved or unresolved)."""
    bare_name = Path(clone_path).name
    if not os.path.isdir(os.path.join(clone_path, ".git")):
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "resolved": False,
            "reason": "no .git directory (not a clone)",
        }
    remote_url = run_git(clone_path, ["config", "--get", "remote.origin.url"])
    if remote_url is None:
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "resolved": False,
            "reason": "git config --get remote.origin.url failed (no origin?)",
        }
    remote_url = remote_url.strip()
    try:
        identity = resolve_remote_project_identity(
            remote_url,
            source=f"{clone_path}:remote.origin.url",
        )
    except SymbolIdentityError as exc:
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "remote_url": remote_url,
            "resolved": False,
            "reason": f"could not resolve project identity: {exc}",
        }
    entry = {
        "bare_name": bare_name,
        "project_identity": identity.project_identity,
        "remote_url": remote_url,
        "clone_path": os.path.abspath(clone_path),
        "source": "git_remote",
        "resolved": True,
    }
    if identity.owner_repo is not None:
        owner, repo = identity.owner_repo.split("/", 1)
        entry.update(
            {
                "owner_repo": identity.owner_repo,
                "owner": owner,
                "repo": repo,
            }
        )
    return entry


def discover_clones(repo_dirs: list[str], repos: list[str]) -> list[str]:
    """Return the list of clone paths from --repo_dir entries and --repo paths."""
    clone_paths: list[str] = []
    for rd in repo_dirs:
        if not os.path.isdir(rd):
            raise SystemExit(f"[build_repo_list] --repo_dir does not exist: {rd}")
        for entry in sorted(os.listdir(rd)):
            path = os.path.join(rd, entry)
            if os.path.isdir(os.path.join(path, ".git")):
                clone_paths.append(path)
    for r in repos:
        clone_paths.append(r)
    # De-dup preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in clone_paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def load_stored_map(path: Optional[str]) -> dict[str, dict]:
    """Load and migrate a prior repo_list into ``{bare_name: entry}``.

    Schema-v1 GitHub rows carried only ``owner_repo``. That explicit field is a
    lossless project identity, so it is the sole backwards-compatible fallback.
    Rows with neither field are not guessed from names or stale remote strings.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f"[build_repo_list] --stored_map does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("repos")
    if not isinstance(entries, list):
        raise SystemExit(f"[build_repo_list] {path}: expected a repos list")
    out: dict[str, dict] = {}
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise SystemExit(
                f"[build_repo_list] {path}: repos[{index}] must be an object"
            )
        entry = dict(raw_entry)
        bare_name = entry.get("bare_name") or entry.get("name")
        if not isinstance(bare_name, str) or not bare_name:
            raise SystemExit(
                f"[build_repo_list] {path}: repos[{index}] has no bare_name"
            )
        project_identity = entry.get("project_identity")
        owner_repo = entry.get("owner_repo")
        if project_identity is None:
            project_identity = owner_repo
        try:
            project_identity = require_project_identity(
                project_identity,
                source=f"{path}:repos[{index}].project_identity",
            )
            if owner_repo is not None:
                owner_repo = require_project_identity(
                    owner_repo,
                    source=f"{path}:repos[{index}].owner_repo",
                )
        except SymbolIdentityError as exc:
            raise SystemExit(f"[build_repo_list] {exc}") from exc
        if owner_repo is not None and owner_repo != project_identity:
            raise SystemExit(
                f"[build_repo_list] {path}:repos[{index}] has conflicting "
                f"project_identity {project_identity!r} and owner_repo "
                f"{owner_repo!r}"
            )
        entry["bare_name"] = bare_name
        entry["project_identity"] = project_identity
        previous = out.get(bare_name)
        if (
            previous is not None
            and previous["project_identity"] != project_identity
        ):
            raise SystemExit(
                "[build_repo_list] project identity collision for stored bare "
                f"name {bare_name!r}: {previous['project_identity']!r} vs "
                f"{project_identity!r}"
            )
        if (
            previous is not None
            and previous.get("owner_repo") != entry.get("owner_repo")
        ):
            raise SystemExit(
                "[build_repo_list] GitHub capability collision for stored bare "
                f"name {bare_name!r}: owner_repo "
                f"{previous.get('owner_repo')!r} vs {entry.get('owner_repo')!r}"
            )
        out.setdefault(bare_name, entry)

    # Older manifests placed non-GitHub remotes in ``unresolved`` even when
    # the URL contains a lossless forge identity. Recover those entries from
    # the recorded URL; do not infer an identity from the bare directory name.
    unresolved_entries = data.get("unresolved", [])
    if not isinstance(unresolved_entries, list):
        raise SystemExit(f"[build_repo_list] {path}: expected an unresolved list")
    for index, raw_entry in enumerate(unresolved_entries):
        if not isinstance(raw_entry, dict):
            raise SystemExit(
                f"[build_repo_list] {path}: unresolved[{index}] must be an object"
            )
        bare_name = raw_entry.get("bare_name") or raw_entry.get("name")
        remote_url = raw_entry.get("remote_url") or raw_entry.get("url")
        if not isinstance(bare_name, str) or not bare_name:
            continue
        if not isinstance(remote_url, str) or not remote_url:
            continue
        try:
            identity = resolve_remote_project_identity(
                remote_url,
                source=f"{path}:unresolved[{index}].url",
            )
        except SymbolIdentityError:
            continue
        migrated = {
            "bare_name": bare_name,
            "project_identity": identity.project_identity,
            "remote_url": remote_url,
            "source": "stored_unresolved_remote",
        }
        if identity.owner_repo is not None:
            migrated["owner_repo"] = identity.owner_repo
        previous = out.get(bare_name)
        if previous is not None:
            if previous["project_identity"] != migrated["project_identity"]:
                raise SystemExit(
                    "[build_repo_list] project identity collision while migrating "
                    f"unresolved bare name {bare_name!r}"
                )
            continue
        out[bare_name] = migrated
    return out


def _add_resolved_entry(
    resolved: dict[str, dict],
    entry: dict,
    *,
    context: str,
) -> None:
    bare_name = entry["bare_name"]
    project_identity = entry["project_identity"]
    previous = resolved.get(bare_name)
    if previous is None:
        resolved[bare_name] = entry
        return
    if previous["project_identity"] != project_identity:
        raise SystemExit(
            f"[build_repo_list] project identity collision for bare name "
            f"{bare_name!r} ({context}): {previous['project_identity']!r} vs "
            f"{project_identity!r}"
        )
    if previous.get("owner_repo") != entry.get("owner_repo"):
        raise SystemExit(
            f"[build_repo_list] GitHub capability collision for bare name "
            f"{bare_name!r} ({context}): owner_repo "
            f"{previous.get('owner_repo')!r} vs {entry.get('owner_repo')!r}"
        )


def build(
    repo_dirs: list[str],
    repos: list[str],
    stored_map_path: Optional[str],
    allow_unresolved: bool,
) -> dict:
    stored = load_stored_map(stored_map_path)
    clone_paths = discover_clones(repo_dirs, repos)
    if not clone_paths and not stored:
        raise SystemExit(
            "[build_repo_list] no clones found and no --stored_map given; nothing to do"
        )

    resolved: dict[str, dict] = {}
    unresolved_by_bare: dict[str, dict] = {}

    for clone_path in clone_paths:
        entry = resolve_clone(clone_path)
        if entry.get("resolved"):
            entry.pop("resolved", None)
            bare = entry["bare_name"]
            stored_entry = stored.get(bare)
            if (
                stored_entry is not None
                and stored_entry["project_identity"] != entry["project_identity"]
            ):
                raise SystemExit(
                    f"[build_repo_list] project identity collision for bare name "
                    f"{bare!r} between stored map "
                    f"{stored_entry['project_identity']!r} and live remote "
                    f"{entry['project_identity']!r}"
                )
            if (
                stored_entry is not None
                and stored_entry.get("owner_repo") != entry.get("owner_repo")
            ):
                raise SystemExit(
                    f"[build_repo_list] GitHub capability collision for bare name "
                    f"{bare!r} between stored and live owner_repo values"
                )
            _add_resolved_entry(resolved, entry, context=clone_path)
            unresolved_by_bare.pop(bare, None)
        else:
            # Offline fallback that is NOT silent: only an EXPLICIT prior
            # resolution (from --stored_map) can rescue a clone we cannot read
            # now. We record that the value came from the stored map.
            bare = entry["bare_name"]
            if bare in stored:
                rescued = dict(stored[bare])
                rescued["source"] = "stored_map"
                rescued["clone_path"] = entry.get(
                    "clone_path", rescued.get("clone_path", "")
                )
                _add_resolved_entry(resolved, rescued, context=clone_path)
                unresolved_by_bare.pop(bare, None)
            elif bare not in resolved:
                unresolved_by_bare.setdefault(bare, entry)

    # Merge in stored entries whose clones were not on disk at all this run.
    for bare, entry in stored.items():
        if bare not in resolved:
            merged = dict(entry)
            merged["source"] = "stored_map"
            _add_resolved_entry(resolved, merged, context="stored map")

    repos_out = sorted(
        resolved.values(),
        key=lambda entry: (entry["project_identity"], entry["bare_name"]),
    )
    unresolved = sorted(
        unresolved_by_bare.values(), key=lambda entry: entry["bare_name"]
    )
    by_bare = {
        entry["bare_name"]: entry["project_identity"] for entry in repos_out
    }
    project_identities = sorted(
        {entry["project_identity"] for entry in repos_out}
    )
    repo_names = sorted(
        {
            entry["owner_repo"]
            for entry in repos_out
            if entry.get("owner_repo")
        }
    )

    result = {
        "schema_version": REPO_LIST_SCHEMA_VERSION,
        "repos": repos_out,
        "by_bare_name": by_bare,
        "project_identities": project_identities,
        "repo_names": repo_names,
        "unresolved": unresolved,
    }

    if unresolved and not allow_unresolved:
        names = ", ".join(sorted(e["bare_name"] for e in unresolved))
        reasons = "\n".join(
            f"  - {e['bare_name']}: {e.get('reason', 'unknown')}" for e in unresolved
        )
        raise SystemExit(
            "[build_repo_list] FAILING LOUD: could not resolve project identity for "
            f"{len(unresolved)} clone(s): {names}\n{reasons}\n"
            "Fix the network remotes, provide a --stored_map with explicit prior "
            "identities, or pass --allow-unresolved to "
            "proceed WITHOUT them (they will be excluded from the repo_list)."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stable project identities from authoritative git remotes."
    )
    parser.add_argument(
        "--repo_dir", action="append", default=[],
        help="Directory containing cloned repos (repeatable).",
    )
    parser.add_argument(
        "--repo", action="append", default=[],
        help="Explicit clone path (repeatable).",
    )
    parser.add_argument(
        "--stored_map",
        help="Prior repo_list.json to seed/merge from for offline re-runs.",
    )
    parser.add_argument("--out", required=True, help="Output repo_list.json path.")
    parser.add_argument(
        "--allow-unresolved", action="store_true",
        help="Do not fail on unresolved clones; exclude them (still listed).",
    )
    args = parser.parse_args()

    result = build(
        repo_dirs=args.repo_dir,
        repos=args.repo,
        stored_map_path=args.stored_map,
        allow_unresolved=args.allow_unresolved,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(
        f"[build_repo_list] resolved {len(result['repos'])} repo(s), "
        f"{len(result['unresolved'])} unresolved -> {args.out}",
        file=sys.stderr,
    )
    for e in result["repos"]:
        github_key = f" (GitHub {e['owner_repo']})" if e.get("owner_repo") else ""
        print(
            f"  {e['bare_name']:30s} -> {e['project_identity']}{github_key}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
