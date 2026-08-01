from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cppmega.data.source_conveyor_composition import (
    GLOBAL_DEDUP_RECEIPT_SCHEMA,
    SOURCE_COMPOSITION_PLAN_SCHEMA,
    load_source_composition,
)
from scripts import commit_source_conveyor_supervisor as commit_supervisor

_BUCKETS = (1024,)
_REPOSITORIES = ("alpha", "beta")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(digit: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "git_commit": digit * 40,
        "dirty": False,
        "source_tree_sha256": digit * 64,
        "producer_role": "canonical_source_conveyor",
        "repository_identity": "cppmega",
        "indexer_dependency_closure_sha256": digit * 64,
        "indexer_provenance": {
            "schema": "cppmega_indexer_dependency_binding_v1",
            "source_sha256": digit * 64,
            "dependency_closure_sha256": digit * 64,
        },
    }


def _lengths(rows: int = 1) -> dict[str, object]:
    return {"1024": {"rows": rows}}


def _write_pr_inputs(root: Path) -> dict[str, object]:
    repo_list = root / "pr_repo_list.json"
    store = root / "prs.sqlite"
    completion = root / "pr_completion.json"
    _write_json(repo_list, {"repo_names": ["owner/alpha", "owner/beta"]})
    store.write_bytes(b"immutable PR store fixture")
    repo_list_sha256 = _sha256(repo_list)
    store_sha256 = _sha256(store)
    expected_repos_sha256 = _canonical_sha256(["owner/alpha", "owner/beta"])
    scan_id = "6" * 64
    _write_json(
        completion,
        {
            "schema": "cppmega_pr_completion_v2",
            "status": "verified",
            "repo_list": {
                "path": str(repo_list),
                "sha256": repo_list_sha256,
            },
            "pr_store": {
                "path": str(store),
                "sha256": store_sha256,
                "size": store.stat().st_size,
            },
            "expected_repos_sha256": expected_repos_sha256,
            "scan_id": scan_id,
            "expected_repo_count": 2,
            "stored_pr_count": 2,
            "declared_pr_count": 2,
            "unverified_store_pr_count": 0,
        },
    )
    completion_sha256 = _sha256(completion)
    store_stat = store.stat()
    completion_binding = {
        "schema": "cppmega_pr_completion_v2",
        "status": "verified",
        "receipt_sha256": completion_sha256,
        "pr_store_sha256": store_sha256,
        "repo_list_sha256": repo_list_sha256,
        "expected_repos_sha256": expected_repos_sha256,
        "scan_id": scan_id,
        "expected_repo_count": 2,
        "stored_pr_count": 2,
        "unverified_store_pr_count": 0,
    }
    return {
        "repo_list": {
            "path": str(repo_list),
            "sha256": repo_list_sha256,
        },
        "completion": {
            "path": str(completion),
            "sha256": completion_sha256,
        },
        "completion_binding": completion_binding,
        "store": {
            "path": str(store),
            "sha256": store_sha256,
            "device": store_stat.st_dev,
            "inode": store_stat.st_ino,
            "size_bytes": store_stat.st_size,
            "quick_check": "ok",
            "pr_rows": 2,
        },
    }


def _write_run(
    root: Path,
    *,
    run_id: str,
    digit: str,
    streams: str,
    done: dict[str, object],
    failed: dict[str, object],
    code_root: Path,
    commit_root: Path,
    dedup_db: Path,
    inventory: Path,
    archive_receipt: Path,
    repo_list: Path,
    quarantine: Path,
    tokenizer: Path,
    targeted: tuple[str, ...] = (),
    pr_inputs: dict[str, object] | None = None,
) -> dict[str, str]:
    run_root = root / run_id
    manifest_path = run_root / "conveyor" / "_done.json"
    launch_path = run_root / "launch_receipt.json"
    exit_path = run_root / "exit_receipt.json"
    manifest = {
        "code_revision": _revision(digit),
        "done": done,
        "failed": failed,
    }
    if streams in {"commits", "both"} and pr_inputs is not None:
        manifest["pr_completion"] = copy.deepcopy(
            pr_inputs["completion_binding"]
        )
        manifest["pr_completion_reverified_at_finish"] = True
    _write_json(manifest_path, manifest)

    command = [
        "python",
        "scripts/streaming_conveyor.py",
        "--streams",
        streams,
        "--expected-code-revision",
        digit * 40,
        "--dedup-db",
        str(dedup_db),
        "--repo-list",
        str(repo_list),
        "--source-quarantine-manifest",
        str(quarantine),
    ]
    if streams in {"code", "both"}:
        command.extend(
            [
                "--target-lengths-code",
                "1024",
                "--code-output-root",
                str(code_root),
            ]
        )
    if streams in {"commits", "both"}:
        command.extend(
            [
                "--target-lengths-commits",
                "1024",
                "--commit-output-root",
                str(commit_root),
            ]
        )
        if pr_inputs is not None:
            command.extend(
                [
                    "--pr-repo-list",
                    str(pr_inputs["repo_list"]["path"]),
                    "--pr-store",
                    str(pr_inputs["store"]["path"]),
                    "--pr-completion-receipt",
                    str(pr_inputs["completion"]["path"]),
                ]
            )
    for repository in targeted:
        command.extend(["--only-repo", repository])
    if targeted:
        command.extend(["--max-repos", str(len(targeted))])

    launch_schema = (
        "cppmega.canonical_source_targeted_retry_launch_v1"
        if targeted
        else "cppmega.canonical_source_launch_v1"
    )
    launch: dict[str, object] = {
        "schema": launch_schema,
        "status": "running",
        "repository_identity": "cppmega",
        "code_revision": digit * 40,
        "target_lengths": [1024],
        "command": command,
        "inputs": {
            "archive": {
                "resolved_path": "/immutable/source.tar.zst",
                "sha256": "a" * 64,
                "size_bytes": 123,
                "mtime_epoch": 456,
                "inode": 789,
                "device": 10,
            },
            "archive_sha256_receipt": {
                "path": str(archive_receipt),
                "sha256": _sha256(archive_receipt),
            },
            "archive_inventory_receipt": {
                "path": str(inventory),
                "sha256": _sha256(inventory),
            },
            "repo_list": {
                "path": str(repo_list),
                "sha256": _sha256(repo_list),
            },
            "source_quarantine_manifest": {
                "path": str(quarantine),
                "sha256": _sha256(quarantine),
            },
            "tokenizer": {
                "path": str(tokenizer),
                "sha256": _sha256(tokenizer),
            },
        },
        "outputs": {
            "dedup_db": str(dedup_db),
            "code_output_root": str(code_root),
            "commit_output_root": str(commit_root),
            "conveyor_manifest": str(manifest_path),
        },
    }
    if targeted:
        launch["selected_repositories"] = list(targeted)
        launch["expected_selected_repository_count"] = len(targeted)
    else:
        launch["expected_repository_count"] = len(_REPOSITORIES)
    if pr_inputs is not None:
        launch["pr_inputs"] = copy.deepcopy(pr_inputs)
    _write_json(launch_path, launch)

    exit_code = 0 if not failed else 1
    exit_schema = (
        "cppmega.canonical_source_targeted_retry_exit_v1"
        if targeted
        else "cppmega.canonical_source_exit_v1"
    )
    exit_receipt: dict[str, object] = {
        "schema": exit_schema,
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "code_revision": digit * 40,
        "launch_receipt_sha256": _sha256(launch_path),
        "done_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
    }
    if targeted:
        exit_receipt["selected_repositories"] = list(targeted)
    _write_json(exit_path, exit_receipt)
    return {
        "run_id": run_id,
        "launch_receipt": str(launch_path),
        "exit_receipt": str(exit_path),
        "manifest": str(manifest_path),
    }


def _write_dedup_receipt(root: Path, dedup_db: Path) -> Path:
    dedup_db.write_bytes(b"verified global dedup fixture")
    empty_digest = hashlib.sha256(b"").hexdigest()
    receipt = {
        "schema": GLOBAL_DEDUP_RECEIPT_SCHEMA,
        "status": "verified",
        "created_at": "2026-07-28T00:00:00Z",
        "database": {
            "path": str(dedup_db),
            "size_bytes": dedup_db.stat().st_size,
            "sha256": _sha256(dedup_db),
        },
        "checkpoint": {
            "mode": "TRUNCATE",
            "busy": 0,
            "log_frames": 0,
            "checkpointed_frames": 0,
            "wal_size_bytes": 0,
        },
        "integrity_check": "ok",
        "sqlite_schema_sha256": "1" * 64,
        "logical_hash_algorithm": "cppmega_sqlite_rows_lenprefixed_v1",
        "logical_sha256": "2" * 64,
        "tables": {
            name: {
                "rows": 0 if name.endswith("_stage") or name == "dedup_stages" else 1,
                "logical_sha256": empty_digest,
            }
            for name in (
                "exact",
                "minhash",
                "lsh",
                "dedup_meta",
                "chunk_claims",
                "dedup_stages",
                "exact_stage",
                "minhash_stage",
                "lsh_stage",
                "chunk_claims_stage",
            )
        },
        "policy": {
            "exact": "sha1_token_ids_v1",
            "chunk": "tokenized_chunk_claims_v1",
            "near": {
                "enabled": True,
                "threshold": 0.7,
                "num_perm": 256,
                "shingle_k": 5,
            },
        },
        "verifier": {
            "repository_identity": "cppmega",
            "script": "scripts/data/verify_global_dedup_store.py",
            "script_sha256": _sha256(
                Path(__file__).parents[1]
                / "scripts/data/verify_global_dedup_store.py"
            ),
        },
    }
    path = root / "dedup_receipt.json"
    _write_json(path, receipt)
    return path


def _composition_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    code_root = tmp_path / "code"
    commit_root = tmp_path / "commits"
    code_root.mkdir()
    commit_root.mkdir()
    dedup_db = tmp_path / "dedup.sqlite"
    archive_receipt = tmp_path / "archive_sha256_receipt.json"
    repo_list = tmp_path / "repo_list.json"
    quarantine = tmp_path / "source_quarantine_manifest.json"
    tokenizer = tmp_path / "tokenizer.json"
    _write_json(
        archive_receipt,
        {
            "schema": "cppmega.source_archive_sha256_verification_v1",
            "status": "verified",
            "resolved_path": "/immutable/source.tar.zst",
            "sha256": "a" * 64,
            "size_bytes": 123,
            "mtime_epoch": 456,
            "inode": 789,
            "device": 10,
            "exit_code": 0,
        },
    )
    _write_json(repo_list, {"alpha": {}, "beta": {}})
    _write_json(quarantine, {"schema": "fixture", "entries": []})
    _write_json(tokenizer, {"schema": "fixture", "vocab": {}})
    inventory = tmp_path / "archive_inventory.json"
    _write_json(
        inventory,
        {
            "schema": "cppmega.source_archive_inventory_binding_v1",
            "status": "verified",
            "archive_unique_worktree_repo_count": len(_REPOSITORIES),
            "archive_sorted_repo_names_json_sha256": _canonical_sha256(
                sorted(_REPOSITORIES)
            ),
            "archive_sha256_receipt": {
                "path": str(archive_receipt),
                "sha256": _sha256(archive_receipt),
            },
            "canonical_repo_list": {
                "path": str(repo_list),
                "sha256": _sha256(repo_list),
            },
        },
    )
    dedup_receipt = _write_dedup_receipt(tmp_path, dedup_db)
    pr_inputs = _write_pr_inputs(tmp_path)
    failed_record = {"stage": "index_project", "detail": "USR failure"}
    runs = [
        _write_run(
            tmp_path,
            run_id="base-code",
            digit="3",
            streams="code",
            done={"alpha::code": {"lengths": _lengths()}},
            failed={"beta::repo": failed_record},
            code_root=code_root,
            commit_root=commit_root,
            dedup_db=dedup_db,
            inventory=inventory,
            archive_receipt=archive_receipt,
            repo_list=repo_list,
            quarantine=quarantine,
            tokenizer=tokenizer,
        ),
        _write_run(
            tmp_path,
            run_id="repair-code",
            digit="4",
            streams="code",
            done={"beta::code": {"lengths": _lengths()}},
            failed={},
            code_root=code_root,
            commit_root=commit_root,
            dedup_db=dedup_db,
            inventory=inventory,
            archive_receipt=archive_receipt,
            repo_list=repo_list,
            quarantine=quarantine,
            tokenizer=tokenizer,
            targeted=("beta",),
        ),
        _write_run(
            tmp_path,
            run_id="full-commits",
            digit="5",
            streams="commits",
            done={
                "alpha::r0": {"lengths": _lengths()},
                "alpha::commits": {"source": "commits", "complete": True},
                "beta::r0": {"lengths": _lengths()},
                "beta::commits": {"source": "commits", "complete": True},
            },
            failed={},
            code_root=code_root,
            commit_root=commit_root,
            dedup_db=dedup_db,
            inventory=inventory,
            archive_receipt=archive_receipt,
            repo_list=repo_list,
            quarantine=quarantine,
            tokenizer=tokenizer,
            pr_inputs=pr_inputs,
        ),
    ]
    plan = {
        "schema": SOURCE_COMPOSITION_PLAN_SCHEMA,
        "runs": runs,
        "dedup_receipt": str(dedup_receipt),
    }
    plan_path = tmp_path / "source_composition_plan.json"
    _write_json(plan_path, plan)
    return plan_path, code_root, commit_root


def test_source_composition_resolves_revision_bound_code_repair(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)

    composition = load_source_composition(
        plan_path,
        buckets=_BUCKETS,
        code_root=code_root,
        commit_root=commit_root,
    )

    assert composition.receipt["status"] == "complete"
    assert composition.receipt["coverage"] == {
        "expected_repositories": 2,
        "code_success_repositories": 2,
        "commit_success_repositories": 2,
        "failed_repositories_observed": 1,
        "failed_units_observed": 1,
        "unresolved_failed_units": 0,
        "repository_set_sha256": _canonical_sha256(sorted(_REPOSITORIES)),
        "allowlist_counts": {
            "code/1024": 2,
            "commits/1024": 2,
        },
    }
    assert set(composition.allowlist[("code", 1024)]) == {
        "alpha.parquet",
        "beta.parquet",
    }
    assert set(composition.allowlist[("commits", 1024)]) == {
        "alpha_r0.parquet",
        "beta_r0.parquet",
    }
    assert len(composition.receipt["source_producers"]) == 3
    commit_run = next(
        run
        for run in composition.receipt["runs"]
        if run["run_id"] == "full-commits"
    )
    assert commit_run["pr_completion"]["status"] == "verified"
    assert commit_run["pr_completion"]["stored_pr_count"] == 2


def test_commit_supervisor_plan_is_accepted_by_source_composition(
    tmp_path: Path,
) -> None:
    original_plan, code_root, commit_root = _composition_fixture(tmp_path)
    fixture = json.loads(original_plan.read_text(encoding="utf-8"))
    code_run = fixture["runs"][0]
    manifest_path = Path(code_run["manifest"])
    exit_path = Path(code_run["exit_receipt"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["done"]["beta::code"] = {"lengths": _lengths()}
    manifest["failed"] = {}
    _write_json(manifest_path, manifest)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt.update(status="success", exit_code=0)
    exit_receipt["done_manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(exit_path, exit_receipt)

    plan_path = tmp_path / "commit-supervisor-plan.json"
    commit_supervisor._write_composition_plan(
        plan_path,
        code_run={
            "launch_path": Path(code_run["launch_receipt"]),
            "exit_path": exit_path,
            "manifest_path": manifest_path,
        },
        commit_run_root=Path(fixture["runs"][2]["launch_receipt"]).parent,
        dedup_receipt=Path(fixture["dedup_receipt"]),
    )

    composition = load_source_composition(
        plan_path,
        buckets=_BUCKETS,
        code_root=code_root,
        commit_root=commit_root,
    )
    assert composition.receipt["status"] == "complete"
    assert len(composition.receipt["runs"]) == 2


def test_bundle_stages_every_source_composition_proof(tmp_path: Path) -> None:
    from scripts.data.build_macro_routes_megatron_bundle import (
        _stage_source_composition,
    )

    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    composition = load_source_composition(
        plan_path,
        buckets=_BUCKETS,
        code_root=code_root,
        commit_root=commit_root,
    )
    partial_dir = tmp_path / "bundle.partial"
    provenance_root = partial_dir / "provenance"
    provenance_root.mkdir(parents=True)

    staged = _stage_source_composition(
        composition,
        partial_dir=partial_dir,
        provenance_root=provenance_root,
    )

    assert staged["schema"] == composition.receipt["schema"]
    assert len(staged["runs"]) == 3
    assert {
        name
        for run in staged["runs"]
        for name in run["artifacts"]
    } == {
        "launch",
        "exit",
        "manifest",
        "archive_sha256_receipt",
        "archive_inventory",
        "repo_list",
        "source_quarantine_manifest",
        "tokenizer",
        "pr_completion",
        "pr_repo_list",
    }
    for descriptor in (
        staged["receipt"],
        staged["plan"],
        staged["dedup_receipt"],
        staged["dedup_verifier"],
    ):
        path = partial_dir / descriptor["path"]
        assert _sha256(path) == descriptor["sha256"]


def test_source_composition_rejects_unresolved_failed_repository(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["runs"].pop(1)
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="no trainable shards|unresolved|coverage"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_dedup_substitution(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][1]["launch_receipt"])
    exit_path = Path(plan["runs"][1]["exit_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["outputs"]["dedup_db"] = str(tmp_path / "other.sqlite")
    command_index = launch["command"].index("--dedup-db") + 1
    launch["command"][command_index] = str(tmp_path / "other.sqlite")
    _write_json(launch_path, launch)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="receipt-bound global dedup DB"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_duplicate_repair_output(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repair_manifest_path = Path(plan["runs"][1]["manifest"])
    repair_exit_path = Path(plan["runs"][1]["exit_receipt"])
    repair_manifest = json.loads(repair_manifest_path.read_text(encoding="utf-8"))
    repair_manifest["done"]["alpha::code"] = {"lengths": _lengths()}
    _write_json(repair_manifest_path, repair_manifest)
    repair_exit = json.loads(repair_exit_path.read_text(encoding="utf-8"))
    repair_exit["done_manifest"]["sha256"] = _sha256(repair_manifest_path)
    _write_json(repair_exit_path, repair_exit)

    with pytest.raises(ValueError, match="targeted terminal repository set|outside"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_weakened_near_dedup_receipt(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt_path = Path(plan["dedup_receipt"])
    receipt = copy.deepcopy(json.loads(receipt_path.read_text(encoding="utf-8")))
    receipt["policy"]["near"]["enabled"] = False
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="production exact\\+near policy"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_symlinked_plan(tmp_path: Path) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    linked_plan = tmp_path / "linked_plan.json"
    linked_plan.symlink_to(plan_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        load_source_composition(
            linked_plan,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_targeted_command_selection_drift(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][1]["launch_receipt"])
    exit_path = Path(plan["runs"][1]["exit_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    only_repo_index = launch["command"].index("--only-repo") + 1
    launch["command"][only_repo_index] = "alpha"
    _write_json(launch_path, launch)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="targeted command selection drifted"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_input_artifact_mutation(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][0]["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    Path(launch["inputs"]["repo_list"]["path"]).write_text(
        '{"mutated": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repo_list artifact binding drifted"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_accepts_hash_identical_relocated_repo_list(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    first_launch = json.loads(
        Path(plan["runs"][0]["launch_receipt"]).read_text(encoding="utf-8")
    )
    original_repo_list = Path(first_launch["inputs"]["repo_list"]["path"])
    relocated_repo_list = tmp_path / "relocated" / original_repo_list.name
    relocated_repo_list.parent.mkdir()
    relocated_repo_list.write_bytes(original_repo_list.read_bytes())

    for run in plan["runs"]:
        launch_path = Path(run["launch_receipt"])
        exit_path = Path(run["exit_receipt"])
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        launch["inputs"]["repo_list"]["path"] = str(relocated_repo_list)
        repo_list_index = launch["command"].index("--repo-list") + 1
        launch["command"][repo_list_index] = str(relocated_repo_list)
        _write_json(launch_path, launch)
        exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
        exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
        _write_json(exit_path, exit_receipt)

    original_repo_list.unlink()

    composition = load_source_composition(
        plan_path,
        buckets=_BUCKETS,
        code_root=code_root,
        commit_root=commit_root,
    )

    assert composition.receipt["status"] == "complete"


def test_source_composition_rejects_source_archive_command_drift(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][0]["launch_receipt"])
    exit_path = Path(plan["runs"][0]["exit_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["command"].extend(
        ["--source-archive", "/different/source.tar.zst"]
    )
    _write_json(launch_path, launch)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(
        ValueError,
        match="source archive command path drifted",
    ):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_duplicate_shard_within_run(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = Path(plan["runs"][2]["manifest"])
    exit_path = Path(plan["runs"][2]["exit_receipt"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["done"]["alpha::r00"] = {"lengths": _lengths()}
    _write_json(manifest_path, manifest)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["done_manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="maps multiple units to source shard"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_full_launch_count_drift(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][0]["launch_receipt"])
    exit_path = Path(plan["runs"][0]["exit_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["expected_repository_count"] = 1
    _write_json(launch_path, launch)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="full launch repository count drifted"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_nonverified_pr_completion(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][2]["launch_receipt"])
    exit_path = Path(plan["runs"][2]["exit_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    completion_path = Path(launch["pr_inputs"]["completion"]["path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = "complete"
    _write_json(completion_path, completion)
    launch["pr_inputs"]["completion"]["sha256"] = _sha256(completion_path)
    _write_json(launch_path, launch)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["launch_receipt_sha256"] = _sha256(launch_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="PR completion is not verified"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_rejects_pr_store_mutation(tmp_path: Path) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    launch_path = Path(plan["runs"][2]["launch_receipt"])
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    store_path = Path(launch["pr_inputs"]["store"]["path"])
    with store_path.open("ab") as stream:
        stream.write(b"mutated")

    with pytest.raises(ValueError, match="PR store identity drifted"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )


def test_source_composition_requires_pr_reverification_at_finish(
    tmp_path: Path,
) -> None:
    plan_path, code_root, commit_root = _composition_fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = Path(plan["runs"][2]["manifest"])
    exit_path = Path(plan["runs"][2]["exit_receipt"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pr_completion_reverified_at_finish"] = False
    _write_json(manifest_path, manifest)
    exit_receipt = json.loads(exit_path.read_text(encoding="utf-8"))
    exit_receipt["done_manifest"]["sha256"] = _sha256(manifest_path)
    _write_json(exit_path, exit_receipt)

    with pytest.raises(ValueError, match="not reverified at finish"):
        load_source_composition(
            plan_path,
            buckets=_BUCKETS,
            code_root=code_root,
            commit_root=commit_root,
        )
