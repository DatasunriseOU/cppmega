from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega.data.nanochat_pipeline.packed_rows_schema import (
    NUM_DOCS_COLUMN,
    SOURCE_COMMIT_HASHES_COLUMN,
    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
    SOURCE_PR_NUMBERS_COLUMN,
)
from cppmega.data.source_conveyor_composition import SourceComposition
from scripts.pr_ingest import build_gitlab_primary_membership as bridge
from scripts.pr_ingest.pr_store import connect, upsert_record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _store(tmp_path: Path, *, scan_id: str) -> Path:
    path = tmp_path / "gitlab-primary.sqlite"
    conn = connect(str(path), create=True)
    try:
        for iid in (7, 9):
            commit = f"{iid:040x}"
            upsert_record(
                conn,
                {
                    "repo": "gitlab.com/libeigen%2FEigen",
                    "pr_number": iid,
                    "merge_commit_sha": commit,
                    "pr_title": f"MR {iid}",
                    "pr_body": "body",
                    "comments": [],
                    "reviews": [],
                    "linked_issues": [],
                },
                scan_id=scan_id,
            )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return path


def _composition(tmp_path: Path) -> tuple[SourceComposition, Path]:
    commit_root = tmp_path / "commits"
    bucket_root = commit_root / "1024"
    bucket_root.mkdir(parents=True)
    artifact = bucket_root / "primary.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "repo": "gitlab.com/libeigen%2FEigen",
                NUM_DOCS_COLUMN: 2,
                SOURCE_PR_NUMBERS_COLUMN: [None, None],
                SOURCE_COMMIT_HASHES_COLUMN: [f"{7:040x}", f"{9:040x}"],
                SOURCE_HAS_PR_DISCUSSIONS_COLUMN: [False, False],
            }
        ],
        schema=pa.schema(
            [
                pa.field("repo", pa.string()),
                pa.field(NUM_DOCS_COLUMN, pa.int32()),
                pa.field(SOURCE_PR_NUMBERS_COLUMN, pa.list_(pa.int64())),
                pa.field(SOURCE_COMMIT_HASHES_COLUMN, pa.list_(pa.string())),
                pa.field(
                    SOURCE_HAS_PR_DISCUSSIONS_COLUMN,
                    pa.list_(pa.bool_()),
                ),
            ]
        ),
    )
    pq.write_table(table, artifact, compression="zstd")
    plan = tmp_path / "source-composition.json"
    _canonical_write(plan, {"schema": "test_source_composition_plan_v1"})
    composition = SourceComposition(
        allowlist={
            ("code", 1024): {"unused.parquet": 1},
            ("commits", 1024): {artifact.name: table.num_rows},
        },
        receipt={
            "schema": "cppmega_source_conveyor_composition_v1",
            "status": "complete",
            "plan_sha256": _sha256(plan),
        },
        plan_path=plan,
        dedup_receipt_path=tmp_path / "dedup.json",
        run_files=(),
    )
    return composition, commit_root


def _fixture(
    tmp_path: Path,
) -> tuple[
    argparse.Namespace,
    SourceComposition,
    dict[str, object],
    list[int],
]:
    scan_id = "9" * 64
    store = _store(tmp_path, scan_id=scan_id)
    composition, commit_root = _composition(tmp_path)
    repo_list = tmp_path / "gitlab-repos.json"
    _canonical_write(
        repo_list,
        {
            "schema_version": 2,
            "repos": [
                {
                    "bare_name": "eigen",
                    "project_identity": "gitlab.com/libeigen%2FEigen",
                }
            ],
            "unresolved": [],
        },
    )
    completion = tmp_path / "gitlab-completion.json"
    completion_document = {
        "schema": bridge.GITLAB_COMPLETION_SCHEMA,
        "status": "verified",
        "platform": "gitlab",
        "contract_sha256": "8" * 64,
        "scan_id": scan_id,
        "training_ready_without_membership": False,
        "required_training_gate": "exact_primary_pr_membership_receipt",
        "manifest": {"path": "/immutable/manifest.json", "sha256": "1" * 64},
        "ancillary_store": {
            "path": "/immutable/ancillary.sqlite",
            "sha256": "2" * 64,
            "size": 4096,
        },
        "sidecars": {
            "root": "/immutable/sidecars",
            "logical_set_sha256": "3" * 64,
            "files": 4,
            "byte_size": 8192,
        },
    }
    _canonical_write(completion, completion_document)
    verifier_script = tmp_path / "gitlab_mr_stream.py"
    verifier_script.write_text("# immutable verifier fixture\n", encoding="utf-8")
    normalized = {
        "schema": bridge.GITLAB_COMPLETION_SCHEMA,
        "status": "verified",
        "platform": "gitlab",
        "receipt_sha256": _sha256(completion),
        "pr_store_sha256": _sha256(store),
        "repo_list_sha256": _sha256(repo_list),
        "expected_repos_sha256": "4" * 64,
        "scan_id": scan_id,
        "expected_repo_count": 1,
        "stored_pr_count": 2,
        "unverified_store_pr_count": 0,
        "training_ready_without_membership": False,
    }
    calls: list[int] = []

    def verifier(
        script_path: Path,
        receipt_path: Path,
        pr_store: Path,
        repos: Path,
    ) -> dict[str, object]:
        assert script_path == verifier_script
        assert receipt_path == completion
        assert pr_store == store
        assert repos == repo_list
        calls.append(len(calls) + 1)
        return dict(normalized)

    args = argparse.Namespace(
        store=str(store),
        gitlab_completion_receipt=str(completion),
        repo_list=str(repo_list),
        gitlab_verifier_script=str(verifier_script),
        source_composition=str(composition.plan_path),
        code_root=str(tmp_path / "unused-code"),
        commit_root=str(commit_root),
        target_lengths="1024",
        output_root=str(tmp_path / "membership"),
    )
    args.completion_verifier = verifier
    return args, composition, normalized, calls


def _verifier_source(script: Path) -> dict[str, object]:
    return {
        "schema": bridge.VERIFIER_SOURCE_SCHEMA,
        "repository_identity": "cppmega.mlx",
        "git_commit": "a" * 40,
        "dependency_tree_sha256": "b" * 64,
        "script": bridge._file_descriptor(
            script,
            role="GitLab completion verifier",
        ),
    }


def test_bridge_publishes_receipt_bound_zstd_membership_idempotently(
    tmp_path: Path,
) -> None:
    args, composition, normalized, calls = _fixture(tmp_path)

    first = bridge.build_gitlab_primary_membership(
        args,
        source_composition=composition,
        completion_verifier=args.completion_verifier,
        verifier_binding_loader=_verifier_source,
    )
    second = bridge.build_gitlab_primary_membership(
        args,
        source_composition=composition,
        completion_verifier=args.completion_verifier,
        verifier_binding_loader=_verifier_source,
    )

    assert first == second
    assert calls == [1, 2, 3, 4, 5, 6]
    assert first["status"] == "complete"
    assert first["selected_pr_count"] == 2
    assert first["training_ready"] is False
    output_root = Path(args.output_root)
    artifact = output_root / "primary_pr_membership.parquet"
    parquet = pq.ParquetFile(artifact)
    assert parquet.read().to_pylist() == [
        {"repo": "gitlab.com/libeigen%2FEigen", "pr_number": 7},
        {"repo": "gitlab.com/libeigen%2FEigen", "pr_number": 9},
    ]
    assert {
        str(parquet.metadata.row_group(group).column(column).compression)
        for group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    } == {"ZSTD"}

    bridge_path = output_root / bridge.BRIDGE_RECEIPT_NAME
    receipt = json.loads(bridge_path.read_bytes())
    membership = receipt["primary_membership"]
    assert receipt["schema"] == bridge.BRIDGE_RECEIPT_SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["training_ready_without_export"] is False
    assert receipt["gitlab_completion"]["binding"] == normalized
    assert receipt["source"]["plan"]["sha256"] == composition.receipt["plan_sha256"]
    assert (
        receipt["source"]["composition_receipt_sha256"]
        == membership["commit_artifacts"]["source_composition_sha256"]
    )
    assert (
        receipt["source"]["commit_artifacts"]["artifact_set_sha256"]
        == membership["commit_artifacts"]["artifact_set_sha256"]
    )
    assert receipt["primary_membership_receipt"] == first[
        "primary_membership_receipt"
    ]


def test_bridge_fails_before_publish_when_completion_drifts(
    tmp_path: Path,
) -> None:
    args, composition, normalized, calls = _fixture(tmp_path)

    def drifting_verifier(
        _script: Path,
        _receipt: Path,
        _store: Path,
        _repos: Path,
    ) -> dict[str, object]:
        calls.append(len(calls) + 1)
        current = dict(normalized)
        if len(calls) > 1:
            current["expected_repos_sha256"] = "5" * 64
        return current

    with pytest.raises(
        bridge.GitLabMembershipBridgeError,
        match="changed before membership publication",
    ):
        bridge.build_gitlab_primary_membership(
            args,
            source_composition=composition,
            completion_verifier=drifting_verifier,
            verifier_binding_loader=_verifier_source,
        )

    assert calls == [1, 2]
    assert not Path(args.output_root).exists()


def test_bridge_fails_before_publish_when_source_plan_drifts(
    tmp_path: Path,
) -> None:
    args, composition, normalized, calls = _fixture(tmp_path)

    def mutating_verifier(
        _script: Path,
        _receipt: Path,
        _store: Path,
        _repos: Path,
    ) -> dict[str, object]:
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            composition.plan_path.write_text("mutated\n", encoding="utf-8")
        return dict(normalized)

    with pytest.raises(
        bridge.GitLabMembershipBridgeError,
        match="canonical membership does not bind the exact source composition",
    ):
        bridge.build_gitlab_primary_membership(
            args,
            source_composition=composition,
            completion_verifier=mutating_verifier,
            verifier_binding_loader=_verifier_source,
        )

    assert calls == [1, 2]
    assert not Path(args.output_root).exists()


def test_external_verifier_calls_read_only_function_not_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cppmega.mlx"
    script = root / "scripts" / "pr_ingest" / "gitlab_mr_stream.py"
    script.parent.mkdir(parents=True)
    sentinel = tmp_path / "writer-was-called"
    script.write_text(
        """
from pathlib import Path
import json

def verify_gitlab_completion_receipt(path, *, pr_store, repo_list):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def run(*_args, **_kwargs):
    Path(%r).write_text("bad", encoding="utf-8")
    raise RuntimeError("writer entrypoint must not run")

if __name__ == "__main__":
    run()
"""
        % str(sentinel),
        encoding="utf-8",
    )
    binding = {"schema": "read-only-verifier-result", "ok": True}
    receipt = tmp_path / "completion.json"
    _canonical_write(receipt, binding)
    store = tmp_path / "store.sqlite"
    store.write_bytes(b"store")
    repos = tmp_path / "repos.json"
    _canonical_write(repos, {})

    assert bridge.verify_gitlab_completion_external(
        script,
        receipt,
        store,
        repos,
    ) == binding
    assert not sentinel.exists()


def test_verifier_source_binding_rejects_dirty_dependency_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cppmega.mlx"
    script = root / "scripts" / "pr_ingest" / "gitlab_mr_stream.py"
    store_module = root / "scripts" / "pr_ingest" / "pr_store.py"
    data_module = root / "cppmega_mlx" / "data" / "commit_scope.py"
    for path in (script, store_module, data_module):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    binding = bridge.gitlab_verifier_source_binding(script)
    assert binding["schema"] == bridge.VERIFIER_SOURCE_SCHEMA
    assert binding["git_commit"] == subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    data_module.write_text("# dirty\n", encoding="utf-8")
    with pytest.raises(
        bridge.GitLabMembershipBridgeError,
        match="dependency subtree is not clean",
    ):
        bridge.gitlab_verifier_source_binding(script)
