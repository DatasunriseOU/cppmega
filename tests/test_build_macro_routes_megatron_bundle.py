from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.data.build_macro_routes_megatron_bundle as builder
from scripts.data.build_macro_routes_megatron_bundle import (
    BUNDLE_KNOWN_LIMITATIONS,
    _acquire_build_lock,
    _artifact_set_sha256,
    build_arg_parser,
    _canonical_sha256,
    _ensure_partial_build_plan,
    _load_manifest_allowlist,
    _parse_objective_artifacts,
    _portable_bucket_results,
    _producer_binding_from_conveyor,
    _publish_validated_bundle,
    _run_snapshot_audit,
    _snapshot_sources,
    _stage_data_contracts,
    _stage_tokenizer,
    _validate_objective_source_binding,
    _write_repaired_snapshot_manifest,
)


def test_bundle_known_limitations_do_not_claim_retired_qname_or_domain_gaps() -> None:
    assert BUNDLE_KNOWN_LIMITATIONS == (
        "the source snapshot is the manifest-complete subset; failed or live "
        "conveyor units are excluded",
    )
    text = " ".join(BUNDLE_KNOWN_LIMITATIONS).lower()
    assert "qname" not in text
    assert "no observed shell" not in text


def _conveyor_revision_binding() -> dict[str, object]:
    return {
        "code_revision": {
            "schema_version": 2,
            "git_commit": "a" * 40,
            "dirty": False,
            "source_tree_sha256": "b" * 64,
            "indexer_dependency_closure_sha256": "d" * 64,
            "indexer_provenance": {
                "schema": "cppmega_indexer_dependency_binding_v1",
                "path": "tools/clang_indexer/index_project.py",
                "source_sha256": "c" * 64,
                "dependency_closure_sha256": "d" * 64,
                "dependency_manifest": {
                    "tools/clang_indexer/index_project.py": "c" * 64,
                },
            },
        }
    }


def test_bundle_producer_binding_covers_cppmega_mlx_and_indexer_closure() -> None:
    binding = _producer_binding_from_conveyor(
        _conveyor_revision_binding(),
        cppmega_commit="e" * 40,
        cppmega_tree_sha256="f" * 64,
    )

    assert set(binding["components"]) == {
        "cppmega",
        "cppmega_mlx",
        "clang_indexer",
    }
    assert binding["components"]["clang_indexer"][
        "dependency_closure_sha256"
    ] == "d" * 64


def test_bundle_producer_binding_rejects_legacy_revision_receipt() -> None:
    with pytest.raises(RuntimeError, match="schema v2"):
        _producer_binding_from_conveyor(
            {"code_revision": {"schema_version": 1, "dirty": False}},
            cppmega_commit="e" * 40,
            cppmega_tree_sha256="f" * 64,
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


def test_builder_accepts_explicit_bucketed_objective_artifacts() -> None:
    args = build_arg_parser().parse_args(
        ["--objective-artifact", "1024=/checked-out/objective_materialization.json"]
    )

    assert args.objective_artifact == [
        "1024=/checked-out/objective_materialization.json"
    ]


def test_builder_requires_exactly_one_objective_artifact_per_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "objective_materialization.json"
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        builder, "load_objective_materialization_artifact", lambda _path: object()
    )

    assert _parse_objective_artifacts(
        [f"1024={artifact}", f"2048={artifact}"], (1024, 2048)
    ) == {1024: artifact.resolve(), 2048: artifact.resolve()}
    with pytest.raises(ValueError, match="exactly match"):
        _parse_objective_artifacts([f"1024={artifact}"], (1024, 2048))


def test_every_bucket_conversion_receives_hash_bound_objective_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    objective_artifact = tmp_path / "objective_materialization.json"
    objective_artifact.write_text("{}", encoding="utf-8")

    def fake_convert(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(builder, "convert_parquet_to_megatron", fake_convert)
    monkeypatch.setattr(
        builder,
        "_current_generation_directory",
        lambda _prefix: generation_dir,
    )
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            artifact_set_sha256="a" * 64,
            file_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_objective_expected_counts",
        lambda _path: {"rows": 1, "valid_tokens": 3, "trained_tokens": 2},
    )
    monkeypatch.setattr(
        builder,
        "_verify_prefix",
        lambda _prefix, _expected: {
            "objective_contract": {
                "sha256": "a" * 64,
                "objective_id_sidecar": {
                    "path": "objective_ids.bin",
                    "dtype": "uint8",
                    "document_aligned": True,
                },
            },
            "objective_materialization": {
                "artifact_set_sha256": "a" * 64,
                "artifact_file_sha256": "b" * 64,
            },
        },
    )

    builder._build_bucket(
        bucket=1024,
        data_root=tmp_path / "data",
        objective_artifact_path=objective_artifact,
    )

    assert captured["objective_artifact_path"] == str(objective_artifact.resolve())
    assert captured["input_dir"] is None
    assert captured["token_column"] == "input_ids"


def test_build_bucket_seals_pointer_generation_as_regular_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objective_artifact = tmp_path / "objective_materialization.json"
    objective_artifact.write_text("{}", encoding="utf-8")
    verified: list[Path] = []

    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            artifact_set_sha256="a" * 64,
            file_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        builder,
        "_objective_expected_counts",
        lambda _path: {"rows": 1, "valid_tokens": 3, "trained_tokens": 2},
    )

    def fake_convert(**kwargs: object) -> None:
        prefix = Path(str(kwargs["output_prefix"]))
        generation = (
            prefix.parent
            / "snapshot"
            / "megatron_generations"
            / prefix.name
            / "generation-test"
        )
        generation.mkdir(parents=True)
        for suffix in (".bin", ".idx", ".json"):
            (generation / f"{prefix.name}{suffix}").write_bytes(suffix.encode())
        current = prefix.parent / f".{prefix.name}.current"
        current.symlink_to(os.path.relpath(generation, prefix.parent))
        for suffix in (".bin", ".idx", ".json"):
            alias = prefix.with_suffix(suffix)
            alias.symlink_to(f"{current.name}/{alias.name}")

    manifest = {
        "objective_materialization": {
            "artifact_set_sha256": "a" * 64,
            "artifact_file_sha256": "b" * 64,
        }
    }

    def fake_verify(prefix: Path, _expected: dict[str, int]) -> dict[str, object]:
        verified.append(prefix)
        return manifest

    monkeypatch.setattr(builder, "convert_parquet_to_megatron", fake_convert)
    monkeypatch.setattr(builder, "_verify_prefix", fake_verify)

    result = builder._build_bucket(
        bucket=1024,
        data_root=tmp_path / "data",
        objective_artifact_path=objective_artifact,
    )

    final_dir = tmp_path / "data" / "seq_1024"
    final_prefix = final_dir / "cppmega_macro_routes_seq1024_train"
    assert result["prefix"] == str(final_prefix)
    assert verified[-1] == final_prefix
    assert not (tmp_path / "data" / ".seq_1024.building").exists()
    assert all(path.is_file() and not path.is_symlink() for path in final_dir.iterdir())


def test_objective_sources_must_match_repaired_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = b"code"
    second = b"commit"
    files = [
        {
            "path": "code/1024/repo.parquet",
            "size_bytes": len(first),
            "sha256": hashlib.sha256(first).hexdigest(),
            "rows": 2,
        },
        {
            "path": "commits/1024/repo_r0.parquet",
            "size_bytes": len(second),
            "sha256": hashlib.sha256(second).hexdigest(),
            "rows": 3,
        },
    ]
    digest = _artifact_set_sha256(
        [
            {"path": row["path"], "size": row["size_bytes"], "sha256": row["sha256"]}
            for row in files
        ]
    )
    source_snapshot = {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": 1024,
        "file_count": 2,
        "row_count": 5,
        "files": files,
        "sampling": {
            "mode": "deterministic_epoch_shuffle_v1",
            "seed": 17,
            "requested_samples": 60,
            "full_passes": 12,
            "tail_rows": 0,
            "min_row_reuse": 12,
            "max_row_reuse": 12,
        },
        "artifact_set_sha256": digest,
    }
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )
    repaired = {
        "files": [
            {
                "kind": Path(row["path"]).parts[0],
                "bucket": 1024,
                "snapshot": row["path"],
                "size": row["size_bytes"],
                "snapshot_sha256": row["sha256"],
                "rows": row["rows"],
            }
            for row in files
        ]
    }

    result = _validate_objective_source_binding(
        objective_artifact_path=tmp_path / "objective.json",
        repaired_snapshot_manifest=repaired,
        bucket=1024,
    )
    assert result["artifact_set_sha256"] == digest

    for field, value in (
        ("kind", "commits"),
        ("bucket", 2048),
        ("snapshot", "code/1024/other.parquet"),
        ("size", 999),
        ("snapshot_sha256", "0" * 64),
        ("rows", 999),
    ):
        drifted = json.loads(json.dumps(repaired))
        drifted["files"][0][field] = value
        with pytest.raises(RuntimeError, match="do not match repaired snapshot"):
            _validate_objective_source_binding(
                objective_artifact_path=tmp_path / "objective.json",
                repaired_snapshot_manifest=drifted,
                bucket=1024,
            )

    reversed_snapshot = {"files": list(reversed(repaired["files"]))}
    with pytest.raises(RuntimeError, match="do not match repaired snapshot"):
        _validate_objective_source_binding(
            objective_artifact_path=tmp_path / "objective.json",
            repaired_snapshot_manifest=reversed_snapshot,
            bucket=1024,
        )


def _bounded_v2_sampling() -> dict[str, object]:
    return {
        "mode": "deterministic_shard_row_group_record_batch_shuffle_v2",
        "seed": 31,
        "requested_samples": 5,
        "full_passes": 2,
        "tail_rows": 1,
        "min_row_reuse": 2,
        "max_row_reuse": 3,
        "record_batch_rows": 2,
        "ordering": {
            "permutation": "sha256_sort_key_v1",
            "epochs": "ascending",
            "shards": "seeded_permutation_per_epoch",
            "row_groups": "seeded_permutation_per_shard_epoch",
            "record_batches": "physical_order_within_row_group",
            "rows": "seeded_permutation_within_record_batch",
        },
        "cursor_semantics": "last_yielded_row_v1",
        "producer": {
            "name": "pyarrow.parquet.ParquetFile.iter_batches",
            "version": 1,
            "row_group_rows": [[2]],
        },
        "final_cursor": {
            "epoch": 2,
            "shard_position": 0,
            "shard_index": 0,
            "row_group_position": 0,
            "row_group_index": 0,
            "record_batch_index": 0,
            "row_shuffle_position": 0,
            "row_index_in_record_batch": 0,
            "source_index": 4,
        },
    }


def _bounded_v2_source_binding() -> tuple[dict[str, object], dict[str, object]]:
    payload = b"code"
    files = [
        {
            "path": "code/1024/repo.parquet",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": 2,
        }
    ]
    digest = _artifact_set_sha256(
        [
            {
                "path": files[0]["path"],
                "size": files[0]["size_bytes"],
                "sha256": files[0]["sha256"],
            }
        ]
    )
    source_snapshot = {
        "schema": "cppmega_objective_source_snapshot_v1",
        "sequence_length": 1024,
        "file_count": 1,
        "row_count": 2,
        "files": files,
        "sampling": _bounded_v2_sampling(),
        "artifact_set_sha256": digest,
    }
    repaired_snapshot = {
        "files": [
            {
                "kind": "code",
                "bucket": 1024,
                "snapshot": files[0]["path"],
                "size": files[0]["size_bytes"],
                "snapshot_sha256": files[0]["sha256"],
                "rows": files[0]["rows"],
            }
        ]
    }
    return source_snapshot, repaired_snapshot


def test_builder_accepts_exact_bounded_v2_sampling_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_snapshot, repaired_snapshot = _bounded_v2_source_binding()
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )

    result = _validate_objective_source_binding(
        objective_artifact_path=tmp_path / "objective.json",
        repaired_snapshot_manifest=repaired_snapshot,
        bucket=1024,
    )

    assert result["sampling"] == _bounded_v2_sampling()


@pytest.mark.parametrize(
    "malformation",
    (
        "missing_record_batch_rows",
        "record_batch_size_alias",
        "ordering_drift",
        "missing_cursor_coordinate",
        "wrong_source_index",
    ),
)
def test_builder_rejects_malformed_bounded_v2_sampling_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    source_snapshot, repaired_snapshot = _bounded_v2_source_binding()
    sampling = source_snapshot["sampling"]
    assert isinstance(sampling, dict)
    if malformation == "missing_record_batch_rows":
        sampling.pop("record_batch_rows")
    elif malformation == "record_batch_size_alias":
        sampling["record_batch_size"] = sampling.pop("record_batch_rows")
    elif malformation == "ordering_drift":
        sampling["ordering"]["rows"] = "physical_order_within_record_batch"
    elif malformation == "missing_cursor_coordinate":
        sampling["final_cursor"].pop("row_index_in_record_batch")
    else:
        sampling["final_cursor"]["source_index"] = 3
    monkeypatch.setattr(
        builder,
        "load_objective_materialization_artifact",
        lambda _path: SimpleNamespace(
            contract=SimpleNamespace(payload={"source_snapshot": source_snapshot})
        ),
    )

    with pytest.raises(RuntimeError, match="objective source"):
        _validate_objective_source_binding(
            objective_artifact_path=tmp_path / "objective.json",
            repaired_snapshot_manifest=repaired_snapshot,
            bucket=1024,
        )


def test_builder_stages_and_hashes_the_production_tokenizer(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data/tokenizer_v2"
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    descriptor = _stage_tokenizer(source, bundle)
    resumed = _stage_tokenizer(source, bundle)

    assert resumed == descriptor
    assert descriptor["path"] == "tokenizer"
    assert descriptor["vocab_size"] == 65536
    assert {record["path"] for record in descriptor["files"]} == {
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_contract_v1.json",
        "tokenizer/tokenizer_config.json",
    }
    for record in descriptor["files"]:
        staged = bundle / record["path"]
        assert staged.stat().st_size == record["size"]
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == record["sha256"]


def test_builder_stages_frozen_domain_and_tokenizer_contracts(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    descriptors = _stage_data_contracts(bundle)
    resumed = _stage_data_contracts(bundle)

    assert resumed == descriptors
    assert set(descriptors) == {"domain_schema", "tokenizer_contract"}
    for descriptor in descriptors.values():
        staged = bundle / str(descriptor["path"])
        assert staged.stat().st_size == descriptor["size"]
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == descriptor["sha256"]


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


def test_snapshot_audit_passes_explicit_empty_pr_root(
    tmp_path: Path, monkeypatch
) -> None:
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


def test_manifest_allowlist_excludes_uncommitted_parquet_orphans(
    tmp_path: Path,
) -> None:
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
    assert receipt["files"][0]["rows"] == 1
    assert (
        (snapshot / "code/1024/repo.parquet").stat().st_ino
        != (code_root / "1024/repo.parquet").stat().st_ino
    )


def test_manifest_allowlist_rejects_failed_conveyor_units(tmp_path: Path) -> None:
    manifest_path = tmp_path / "_done.json"
    manifest_path.write_text(
        json.dumps(
            {
                "done": {
                    "repo::code": {"lengths": {"1024": {"rows": 1}}},
                    "repo::r0": {"lengths": {"1024": {"rows": 1}}},
                },
                "failed": {
                    "broken::code": {
                        "stage": "index_project",
                        "detail": "exit 137",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="refusing to freeze.*1 failed units"):
        _load_manifest_allowlist(manifest_path, (1024,))


def test_repaired_snapshot_manifest_hashes_and_binds_replaced_files(
    tmp_path: Path,
) -> None:
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
                "size": len(b"same"),
                "rows": 2,
                "sha256": hashlib.sha256(b"same").hexdigest(),
            },
            {
                "kind": "commits",
                "bucket": 1024,
                "snapshot": "commits/1024/b.parquet",
                "size": len(b"before"),
                "rows": 3,
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
        repair_receipt={"file_scans": [{"path": str(changed), "rows": 3}]},
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
    assert by_path["code/1024/a.parquet"]["rows"] == 2
    assert by_path["commits/1024/b.parquet"]["rows"] == 3


def test_snapshot_rejects_source_mutation_during_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    code_source = code_root / "1024/repo.parquet"
    _write(code_source, b"code-before")
    _write(commit_root / "1024/repo_r0.parquet", b"commit")
    real_copy = builder._copy_private

    def copy_then_mutate(source: Path, target: Path) -> None:
        real_copy(source, target)
        if source == code_source:
            source.write_bytes(b"code-after")

    monkeypatch.setattr(builder, "_copy_private", copy_then_mutate)
    snapshot = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="source changed while snapshotting"):
        _snapshot_sources(
            code_root=code_root,
            commit_root=commit_root,
            snapshot_root=snapshot,
            buckets=(1024,),
            min_age_seconds=0,
            hash_jobs=1,
            allowed={
                ("code", 1024): {"repo.parquet": 2},
                ("commits", 1024): {"repo_r0.parquet": 3},
            },
            conveyor_manifest={"sha256": "a" * 64},
        )

    assert not (snapshot / "source_manifest.json").exists()
    assert not (snapshot / "code/1024/repo.parquet").exists()


def _test_build_plan(objective_digest: str) -> dict[str, object]:
    objectives = [
        {
            "bucket": 1024,
            "artifact_path": "/objective.json",
            "artifact_set_sha256": objective_digest,
            "artifact_file_sha256": "b" * 64,
            "contract_path": "/contract.json",
            "contract_sha256": "c" * 64,
            "contract_file_sha256": "d" * 64,
        }
    ]
    plan = {"buckets": [1024], "objective_artifacts": objectives}
    return {
        "schema": builder.BUILD_PLAN_SCHEMA,
        "objective_artifacts_sha256": _canonical_sha256(objectives),
        "build_plan_sha256": _canonical_sha256(plan),
        "plan": plan,
    }


def test_stale_partial_cannot_reuse_different_objective_build_plan(
    tmp_path: Path,
) -> None:
    partial = tmp_path / ".bundle.partial"
    original = _test_build_plan("a" * 64)
    _ensure_partial_build_plan(partial, original)
    _ensure_partial_build_plan(partial, original)

    with pytest.raises(RuntimeError, match="stale partial build plan mismatch"):
        _ensure_partial_build_plan(partial, _test_build_plan("e" * 64))

    changed_plan = json.loads(json.dumps(original))
    changed_plan["plan"]["keep_snapshot"] = True
    changed_plan["build_plan_sha256"] = _canonical_sha256(changed_plan["plan"])
    assert (
        changed_plan["objective_artifacts_sha256"]
        == original["objective_artifacts_sha256"]
    )
    with pytest.raises(RuntimeError, match="stale partial build plan mismatch"):
        _ensure_partial_build_plan(partial, changed_plan)


def test_unbound_partial_is_rejected_as_stale(tmp_path: Path) -> None:
    partial = tmp_path / ".bundle.partial"
    partial.mkdir()

    with pytest.raises(RuntimeError, match="no canonical build plan"):
        _ensure_partial_build_plan(partial, _test_build_plan("a" * 64))


def test_concurrent_build_lock_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    first = _acquire_build_lock(output)
    try:
        with pytest.raises(RuntimeError, match="bundle build already active"):
            _acquire_build_lock(output)
    finally:
        first.close()

    second = _acquire_build_lock(output)
    second.close()


def test_strict_validation_runs_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    partial = tmp_path / ".bundle.partial"
    output = tmp_path / "bundle"

    def validate(bundle: Path, hash_jobs: int) -> None:
        assert bundle == partial
        assert hash_jobs == 3
        events.append("validate")

    def publish(source: Path, target: Path) -> None:
        assert source == partial
        assert target == output
        events.append("publish")

    monkeypatch.setattr(builder, "_validate_bundle", validate)
    monkeypatch.setattr(builder.os, "replace", publish)

    _publish_validated_bundle(partial, output, hash_jobs=3)

    assert events == ["validate", "publish"]
