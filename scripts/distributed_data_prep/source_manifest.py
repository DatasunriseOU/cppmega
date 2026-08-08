#!/usr/bin/env python3
"""Build and validate deterministic distributed source-worker manifests.

The manifest contains no live discovery result.  A network-backed repository is
accepted only with an expected commit.  Repositories that cannot be cloned from
an authoritative network remote (including ``corpus.local`` identities) must
name an immutable, generation-pinned GCS source object instead.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_sha256,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_nonempty,
    require_sha256,
    validate_gcs_uri,
)

SOURCE_MANIFEST_SCHEMA = "cppmega.distributed_source_manifest_v1"
ASSIGNMENT_ALGORITHM = "canonical_round_robin_v1"
PRE_GLOBAL_SCHEMA = "cppmega.pre_global_enriched_jsonl_v1"
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
LOSSLESS_INDEX_MAX_TOKENS = (1 << 63) - 1
_WORKER_RE = re.compile(r"^worker-[0-9]{4}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}$")
_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+%:-]*/[A-Za-z0-9][A-Za-z0-9._+%/-]*$")
_REMOTE_SCHEMES = frozenset({"https", "ssh"})


def _manifest_digest_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_sha256", None)
    return payload


def manifest_sha256(manifest: Mapping[str, object]) -> str:
    return canonical_sha256(_manifest_digest_payload(manifest))


def _validate_remote_url(value: object, *, where: str) -> str:
    remote = require_nonempty(value, where=where)
    parsed = urlsplit(remote)
    if parsed.scheme not in _REMOTE_SCHEMES:
        raise ContractError(
            f"{where} must use an authoritative https:// or ssh:// remote"
        )
    if not parsed.hostname or parsed.username or parsed.password:
        raise ContractError(f"{where} must not contain credentials and needs a host")
    if parsed.query or parsed.fragment:
        raise ContractError(f"{where} must not contain query or fragment components")
    if parsed.hostname.lower() == "corpus.local":
        raise ContractError(
            f"{where} is corpus.local; use immutable_gcs_tar instead of a fake remote"
        )
    return remote


def _validate_source(source: object, *, where: str) -> dict[str, object]:
    if not isinstance(source, Mapping):
        raise ContractError(f"{where} must be an object")
    result = dict(source)
    kind = result.get("kind")
    if kind == "git_mirror":
        require_exact_fields(
            result,
            {"kind", "remote_url", "expected_commit", "expected_tree"},
            where=where,
        )
        return {
            "kind": kind,
            "remote_url": _validate_remote_url(
                result["remote_url"], where=f"{where}.remote_url"
            ),
            "expected_commit": require_git_object(
                result["expected_commit"], where=f"{where}.expected_commit"
            ),
            "expected_tree": (
                require_git_object(
                    result["expected_tree"], where=f"{where}.expected_tree"
                )
                if result["expected_tree"] is not None
                else None
            ),
        }
    if kind == "immutable_gcs_tar":
        require_exact_fields(
            result,
            {
                "kind",
                "uri",
                "generation",
                "sha256",
                "archive_format",
                "strip_components",
            },
            where=where,
        )
        generation = require_nonempty(
            result["generation"], where=f"{where}.generation"
        )
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError(f"{where}.generation must be a positive decimal")
        if result["archive_format"] != "tar.zst":
            raise ContractError(f"{where}.archive_format must be 'tar.zst'")
        return {
            "kind": kind,
            "uri": validate_gcs_uri(result["uri"], where=f"{where}.uri"),
            "generation": generation,
            "sha256": require_sha256(result["sha256"], where=f"{where}.sha256"),
            "archive_format": "tar.zst",
            "strip_components": require_int(
                result["strip_components"],
                where=f"{where}.strip_components",
                minimum=0,
            ),
        }
    raise ContractError(f"{where}.kind is unsupported: {kind!r}")


def _validate_lengths(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError("pipeline.target_lengths must be a non-empty list")
    lengths = [
        require_int(item, where="pipeline target length", minimum=1) for item in value
    ]
    if lengths != sorted(set(lengths)):
        raise ContractError("pipeline.target_lengths must be unique and ascending")
    return lengths


def build_source_manifest(
    repositories: Sequence[Mapping[str, object]],
    *,
    worker_count: int,
    gcs_output_prefix: str,
    code_revision: str,
    indexer_sha256: str,
    tokenizer_sha256: str,
    quarantine_manifest_sha256: str,
    target_lengths: Sequence[int] = DEFAULT_TARGET_LENGTHS,
) -> dict[str, object]:
    """Create one deterministic, fully assigned source-work manifest."""

    if isinstance(worker_count, bool) or int(worker_count) < 1:
        raise ContractError("worker_count must be positive")
    workers = [f"worker-{index:04d}" for index in range(int(worker_count))]
    prefix = validate_gcs_uri(gcs_output_prefix.rstrip("/"), where="gcs_output_prefix")
    revision = require_git_object(code_revision, where="code_revision")
    bindings = {
        "indexer_sha256": require_sha256(indexer_sha256, where="indexer_sha256"),
        "tokenizer_sha256": require_sha256(tokenizer_sha256, where="tokenizer_sha256"),
        "quarantine_manifest_sha256": require_sha256(
            quarantine_manifest_sha256, where="quarantine_manifest_sha256"
        ),
    }
    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    projects: set[str] = set()
    for raw_index, raw in enumerate(repositories):
        if not isinstance(raw, Mapping):
            raise ContractError(f"repository[{raw_index}] must be an object")
        require_exact_fields(
            raw, {"repo", "project_id", "source"}, where=f"repository[{raw_index}]"
        )
        repo = require_nonempty(raw["repo"], where=f"repository[{raw_index}].repo")
        project_id = require_nonempty(
            raw["project_id"], where=f"repository[{raw_index}].project_id"
        )
        if _REPO_RE.fullmatch(repo) is None:
            raise ContractError(f"repository[{raw_index}].repo is not canonical")
        if _PROJECT_RE.fullmatch(project_id) is None:
            raise ContractError(f"repository[{raw_index}].project_id is not canonical")
        if repo in names:
            raise ContractError(f"duplicate repository name: {repo}")
        if project_id in projects:
            raise ContractError(f"duplicate project identity: {project_id}")
        names.add(repo)
        projects.add(project_id)
        normalized.append(
            {
                "repo": repo,
                "project_id": project_id,
                "source": _validate_source(
                    raw["source"], where=f"repository[{raw_index}].source"
                ),
            }
        )

    # The reducer's primary-copy decision is defined by this order, never by
    # worker completion time or GCS listing order.
    normalized.sort(key=lambda row: (str(row["project_id"]), str(row["repo"])))
    jobs: list[dict[str, object]] = []
    for ordinal, row in enumerate(normalized):
        worker = workers[ordinal % len(workers)]
        identity = canonical_sha256(
            {
                "ordinal": ordinal,
                "repo": row["repo"],
                "project_id": row["project_id"],
                "source": row["source"],
                "worker": worker,
            }
        )
        jobs.append(
            {
                "ordinal": ordinal,
                **row,
                "worker": worker,
                "assignment_sha256": identity,
            }
        )

    lengths = _validate_lengths(list(target_lengths))
    manifest: dict[str, object] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "status": "ready",
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "workers": workers,
        "gcs_output_prefix": prefix,
        "repository_count": len(jobs),
        "repository_order_sha256": canonical_sha256(
            [str(job["project_id"]) for job in jobs]
        ),
        "code_revision": revision,
        "pipeline": {
            "candidate_schema": PRE_GLOBAL_SCHEMA,
            "dedup_applied_on_worker": False,
            "document_order": "canonical_enriched_json_v1",
            "index_max_tokens": LOSSLESS_INDEX_MAX_TOKENS,
            "indexer_sha256": bindings["indexer_sha256"],
            "tokenizer_sha256": bindings["tokenizer_sha256"],
            "quarantine_manifest_sha256": bindings[
                "quarantine_manifest_sha256"
            ],
            "target_lengths": lengths,
        },
        "repositories": jobs,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return validate_source_manifest(manifest)


def validate_source_manifest(value: Mapping[str, object]) -> dict[str, object]:
    manifest = copy.deepcopy(dict(value))
    require_exact_fields(
        manifest,
        {
            "schema",
            "status",
            "manifest_sha256",
            "assignment_algorithm",
            "workers",
            "gcs_output_prefix",
            "repository_count",
            "repository_order_sha256",
            "code_revision",
            "pipeline",
            "repositories",
        },
        where="source manifest",
    )
    if manifest["schema"] != SOURCE_MANIFEST_SCHEMA or manifest["status"] != "ready":
        raise ContractError("source manifest schema/status is unsupported")
    if manifest["assignment_algorithm"] != ASSIGNMENT_ALGORITHM:
        raise ContractError("source manifest assignment algorithm drifted")
    expected_digest = require_sha256(
        manifest["manifest_sha256"], where="source manifest manifest_sha256"
    )
    if manifest_sha256(manifest) != expected_digest:
        raise ContractError("source manifest logical digest is invalid")
    validate_gcs_uri(manifest["gcs_output_prefix"], where="gcs_output_prefix")
    require_git_object(manifest["code_revision"], where="code_revision")

    workers = manifest["workers"]
    if (
        not isinstance(workers, list)
        or not workers
        or workers != [f"worker-{index:04d}" for index in range(len(workers))]
    ):
        raise ContractError("source manifest workers are not canonical")
    if any(not isinstance(worker, str) or _WORKER_RE.fullmatch(worker) is None for worker in workers):
        raise ContractError("source manifest contains an invalid worker id")

    pipeline = manifest["pipeline"]
    if not isinstance(pipeline, Mapping):
        raise ContractError("source manifest pipeline must be an object")
    require_exact_fields(
        pipeline,
        {
            "candidate_schema",
            "dedup_applied_on_worker",
            "document_order",
            "index_max_tokens",
            "indexer_sha256",
            "tokenizer_sha256",
            "quarantine_manifest_sha256",
            "target_lengths",
        },
        where="source manifest pipeline",
    )
    if (
        pipeline["candidate_schema"] != PRE_GLOBAL_SCHEMA
        or pipeline["dedup_applied_on_worker"] is not False
        or pipeline["document_order"] != "canonical_enriched_json_v1"
    ):
        raise ContractError("worker output is not canonical pre-global-dedup data")
    if (
        require_int(
            pipeline["index_max_tokens"],
            where="pipeline.index_max_tokens",
            minimum=1,
        )
        != LOSSLESS_INDEX_MAX_TOKENS
    ):
        raise ContractError("pipeline index_max_tokens is not lossless")
    for field in (
        "indexer_sha256",
        "tokenizer_sha256",
        "quarantine_manifest_sha256",
    ):
        require_sha256(pipeline[field], where=f"pipeline.{field}")
    _validate_lengths(pipeline["target_lengths"])

    raw_jobs = manifest["repositories"]
    if not isinstance(raw_jobs, list):
        raise ContractError("source manifest repositories must be a list")
    expected_count = require_int(
        manifest["repository_count"], where="repository_count", minimum=0
    )
    if len(raw_jobs) != expected_count:
        raise ContractError("source manifest repository_count drifted")
    normalized_jobs: list[dict[str, object]] = []
    names: set[str] = set()
    projects: set[str] = set()
    for ordinal, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, Mapping):
            raise ContractError(f"repository[{ordinal}] must be an object")
        job = dict(raw_job)
        require_exact_fields(
            job,
            {
                "ordinal",
                "repo",
                "project_id",
                "source",
                "worker",
                "assignment_sha256",
            },
            where=f"repository[{ordinal}]",
        )
        if job["ordinal"] != ordinal:
            raise ContractError("repository ordinals are not contiguous")
        repo = require_nonempty(job["repo"], where=f"repository[{ordinal}].repo")
        project = require_nonempty(
            job["project_id"], where=f"repository[{ordinal}].project_id"
        )
        if _REPO_RE.fullmatch(repo) is None or _PROJECT_RE.fullmatch(project) is None:
            raise ContractError(f"repository[{ordinal}] identity is invalid")
        if repo in names or project in projects:
            raise ContractError("source manifest contains duplicate repositories")
        names.add(repo)
        projects.add(project)
        worker = job["worker"]
        if worker != workers[ordinal % len(workers)]:
            raise ContractError(f"repository[{ordinal}] worker assignment drifted")
        source = _validate_source(job["source"], where=f"repository[{ordinal}].source")
        assignment = canonical_sha256(
            {
                "ordinal": ordinal,
                "repo": repo,
                "project_id": project,
                "source": source,
                "worker": worker,
            }
        )
        if require_sha256(
            job["assignment_sha256"],
            where=f"repository[{ordinal}].assignment_sha256",
        ) != assignment:
            raise ContractError(f"repository[{ordinal}] assignment digest drifted")
        normalized_jobs.append(
            {
                "ordinal": ordinal,
                "repo": repo,
                "project_id": project,
                "source": source,
                "worker": worker,
                "assignment_sha256": assignment,
            }
        )
    if normalized_jobs != sorted(
        normalized_jobs, key=lambda row: (str(row["project_id"]), str(row["repo"]))
    ):
        raise ContractError("source manifest repository order is not canonical")
    order_digest = require_sha256(
        manifest["repository_order_sha256"], where="repository_order_sha256"
    )
    if order_digest != canonical_sha256(
        [str(job["project_id"]) for job in normalized_jobs]
    ):
        raise ContractError("source manifest repository order digest drifted")
    manifest["repositories"] = normalized_jobs
    return manifest


def load_source_manifest(path: Path) -> tuple[dict[str, object], str]:
    raw, payload = load_json_object(path, where="distributed source manifest")
    return validate_source_manifest(payload), hashlib.sha256(raw).hexdigest()


def repositories_for_worker(
    manifest: Mapping[str, object], worker: str
) -> tuple[dict[str, object], ...]:
    validated = validate_source_manifest(manifest)
    if worker not in validated["workers"]:
        raise ContractError(f"worker is not assigned by this manifest: {worker}")
    return tuple(
        dict(job)
        for job in validated["repositories"]
        if job["worker"] == worker
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repositories", required=True, type=Path)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--gcs-output-prefix", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--indexer-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--quarantine-manifest-sha256", required=True)
    parser.add_argument(
        "--target-lengths",
        default=",".join(str(value) for value in DEFAULT_TARGET_LENGTHS),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    _raw, document = load_json_object(
        args.repositories, where="distributed repository specification"
    )
    repositories = document.get("repositories")
    if not isinstance(repositories, list):
        parser.error("--repositories must contain a repositories list")
    try:
        lengths = [int(value) for value in args.target_lengths.split(",")]
        manifest = build_source_manifest(
            repositories,
            worker_count=args.worker_count,
            gcs_output_prefix=args.gcs_output_prefix,
            code_revision=args.code_revision,
            indexer_sha256=args.indexer_sha256,
            tokenizer_sha256=args.tokenizer_sha256,
            quarantine_manifest_sha256=args.quarantine_manifest_sha256,
            target_lengths=lengths,
        )
        atomic_write_json(args.output, manifest)
    except (ContractError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "ASSIGNMENT_ALGORITHM",
    "DEFAULT_TARGET_LENGTHS",
    "LOSSLESS_INDEX_MAX_TOKENS",
    "PRE_GLOBAL_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "build_source_manifest",
    "load_source_manifest",
    "manifest_sha256",
    "repositories_for_worker",
    "validate_source_manifest",
]
