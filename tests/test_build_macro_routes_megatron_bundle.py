from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

import scripts.data.build_macro_routes_megatron_bundle as builder
from scripts.data.build_macro_routes_megatron_bundle import (
    _artifact_set_sha256,
    build_arg_parser,
    _load_manifest_allowlist,
    _portable_bucket_results,
    _run_snapshot_audit,
    _snapshot_sources,
    _stage_tokenizer,
    _write_repaired_snapshot_manifest,
)


def test_artifact_set_fingerprint_is_order_independent_and_content_bound() -> None:
    first = {"path": "b.bin", "size": 2, "sha256": "bb"}
    second = {"path": "a.bin", "size": 1, "sha256": "aa"}

    digest = _artifact_set_sha256([first, second])

    assert digest == _artifact_set_sha256([second, first])
    assert digest != _artifact_set_sha256(
        [first, {"path": "a.bin", "size": 1, "sha256": "changed"}]
    )


def test_builder_discards_intermediate_snapshot_by_default() -> None:
    parser = build_arg_parser()

    assert parser.parse_args([]).keep_snapshot is False
    assert parser.parse_args(["--keep-snapshot"]).keep_snapshot is True


def test_builder_stages_and_hashes_the_production_tokenizer(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data/tokenizer_v2"
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    descriptor = _stage_tokenizer(source, bundle)

    assert descriptor["path"] == "tokenizer"
    assert descriptor["vocab_size"] == 65536
    assert {record["path"] for record in descriptor["files"]} == {
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    for record in descriptor["files"]:
        staged = bundle / record["path"]
        assert staged.stat().st_size == record["size"]
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == record["sha256"]


def test_bucket_prefixes_are_bundle_relative_and_cannot_escape(tmp_path: Path) -> None:
    bundle = tmp_path / ".bundle.partial"
    prefix = bundle / "data/seq_1024/cppmega_train"

    results = _portable_bucket_results(
        bundle,
        [{"bucket": 1024, "prefix": str(prefix), "manifest": {}}],
    )

    assert results[0]["prefix"] == "data/seq_1024/cppmega_train"
    with pytest.raises(RuntimeError, match="escapes bundle root"):
        _portable_bucket_results(
            bundle,
            [{"bucket": 1024, "prefix": str(tmp_path / "outside"), "manifest": {}}],
        )


def test_builder_rejects_unbound_existing_audit_receipt(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    (audit_root / "sidecar_parquet_audit.json").write_text(
        json.dumps({"total": {"bad_files": 0, "bad_rows": 0}, "bad_files": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not bound"):
        _run_snapshot_audit(
            snapshot_root=tmp_path / "snapshot",
            audit_script=tmp_path / "audit.py",
            audit_root=audit_root,
            buckets=(1024,),
            workers=1,
            snapshot_manifest_sha256="abc",
        )


def test_snapshot_audit_passes_explicit_empty_pr_root(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, list[str]] = {}
    audit_root = tmp_path / "audit"

    def fake_run(cmd: list[str], *, check: bool) -> None:
        assert check is True
        captured["cmd"] = cmd
        audit_root.mkdir(parents=True, exist_ok=True)
        (audit_root / "sidecar_parquet_audit.json").write_text(
            json.dumps(
                {
                    "total": {"bad_files": 0, "bad_rows": 0},
                    "bad_files": [],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    _run_snapshot_audit(
        snapshot_root=tmp_path / "snapshot",
        audit_script=tmp_path / "audit.py",
        audit_root=audit_root,
        buckets=(1024,),
        workers=1,
        snapshot_manifest_sha256="abc",
    )

    cmd = captured["cmd"]
    pr_root = Path(cmd[cmd.index("--pr-root") + 1])
    assert pr_root == audit_root / "empty_standalone_pr_root"
    assert pr_root.is_dir()
    assert "outputs/reindexed_pr" not in " ".join(cmd)


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_manifest_allowlist_excludes_uncommitted_parquet_orphans(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    _write(code_root / "1024" / "repo.parquet", b"code")
    _write(code_root / "1024" / "orphan.parquet", b"orphan")
    _write(commit_root / "1024" / "repo_r0.parquet", b"commit")
    _write(commit_root / "1024" / "orphan_r0.parquet", b"orphan")
    manifest_path = tmp_path / "_done.json"
    manifest_path.write_text(
        json.dumps(
            {
                "done": {
                    "repo::code": {"lengths": {"1024": {"rows": 1}}},
                    "repo::r0": {"lengths": {"1024": {"rows": 1}}},
                },
                "failed": {},
            }
        ),
        encoding="utf-8",
    )

    allowed, conveyor = _load_manifest_allowlist(manifest_path, (1024,))
    snapshot = tmp_path / "snapshot"
    receipt = _snapshot_sources(
        code_root=code_root,
        commit_root=commit_root,
        snapshot_root=snapshot,
        buckets=(1024,),
        min_age_seconds=0,
        hash_jobs=1,
        allowed=allowed,
        conveyor_manifest=conveyor,
    )

    assert receipt["by_kind_bucket"] == {"code/1024": 1, "commits/1024": 1}
    assert sorted(path.name for path in (snapshot / "code/1024").glob("*.parquet")) == [
        "repo.parquet"
    ]
    assert sorted(
        path.name for path in (snapshot / "commits/1024").glob("*.parquet")
    ) == ["repo_r0.parquet"]


def test_repaired_snapshot_manifest_hashes_only_replaced_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    unchanged = snapshot / "code/1024/a.parquet"
    changed = snapshot / "commits/1024/b.parquet"
    _write(unchanged, b"same")
    _write(changed, b"before")
    source_manifest = {
        "files": [
            {
                "kind": "code",
                "bucket": 1024,
                "snapshot": "code/1024/a.parquet",
                "sha256": hashlib.sha256(b"same").hexdigest(),
            },
            {
                "kind": "commits",
                "bucket": 1024,
                "snapshot": "commits/1024/b.parquet",
                "sha256": hashlib.sha256(b"before").hexdigest(),
            },
        ]
    }
    replacement = changed.with_suffix(".new")
    replacement.write_bytes(b"after")
    os.replace(replacement, changed)
    repaired = _write_repaired_snapshot_manifest(
        snapshot_root=snapshot,
        source_manifest=source_manifest,
        repair_receipt={"file_scans": [{"path": str(changed)}]},
        hash_jobs=1,
    )

    by_path = {record["snapshot"]: record for record in repaired["files"]}
    assert by_path["code/1024/a.parquet"]["boundary_repaired"] is False
    assert (
        by_path["code/1024/a.parquet"]["snapshot_sha256"]
        == by_path["code/1024/a.parquet"]["source_sha256"]
    )
    assert by_path["commits/1024/b.parquet"]["boundary_repaired"] is True
    assert (
        by_path["commits/1024/b.parquet"]["snapshot_sha256"]
        != by_path["commits/1024/b.parquet"]["source_sha256"]
    )
