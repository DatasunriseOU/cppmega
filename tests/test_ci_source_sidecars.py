from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from scripts.ci_content_store import CIContentStore, hash_token_sequence
from scripts.ci_source_sidecars import (
    AMBIGUOUS_PATH,
    COMMIT_ABSENT,
    CONTENT_SEMANTICS,
    DELETED_FORK,
    ExtractionError,
    FETCH_RECEIPT_SCHEMA,
    GENERATED_OR_MUTATED_UNRESOLVABLE,
    INVENTORY_SCHEMA,
    LocalGitResolver,
    PATH_ABSENT,
    RECEIPT_SCHEMA,
    REPO_UNAVAILABLE,
    RESOLVED,
    ResolutionIntegrityError,
    SourceSidecarStore,
    SourceStoreError,
    UNSUPPORTED_OBJECT,
    _hash_records,
    extract_binding_inventory,
    materialize_inventory,
    normalize_source_candidates,
    normalize_source_path,
    verify_binding_inventory,
)


def _run(
    *args: str,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{' '.join(args)} failed:\n{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def _git_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    work = tmp_path / "work"
    mirror = tmp_path / "repo.git"
    work.mkdir()
    _run("git", "init", "-q", str(work))
    _run("git", "config", "user.name", "CI Source Test", cwd=work)
    _run("git", "config", "user.email", "ci-source@example.test", cwd=work)

    (work / "src" / "nested").mkdir(parents=True)
    (work / "src" / "nested" / "main.cpp").write_text(
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    (work / "src" / "copy.cpp").write_text(
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    (work / "assets").mkdir()
    (work / "assets" / "bytes.bin").write_bytes(b"\x00\xff\x10binary\r\n")
    (work / "model.lfs").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 123456\n"
    )
    os.symlink("src/nested/main.cpp", work / "main-link")
    _run("git", "add", ".", cwd=work)
    _run("git", "commit", "-q", "-m", "base", cwd=work)
    base = _run("git", "rev-parse", "HEAD", cwd=work).decode().strip()

    # Add a gitlink without fetching or dereferencing another repository.
    _run(
        "git",
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{base},vendor/component",
        cwd=work,
    )
    _run("git", "commit", "-q", "-m", "gitlink", cwd=work)
    head = _run("git", "rev-parse", "HEAD", cwd=work).decode().strip()
    _run("git", "clone", "-q", "--bare", str(work), str(mirror))
    return mirror, head, base


def _binding(
    head: str,
    source_path: str,
    *,
    repository: str = "owner/repo",
    status: str = RESOLVED,
) -> dict[str, object]:
    candidates = [source_path] if status == RESOLVED else []
    evidence = [
        {
            "source_input": source_path,
            "cwd": ".",
            "normalization_status": status,
        }
    ]
    return {
        "repository": repository,
        "head_sha": head,
        "source_path": source_path,
        "normalization_status": status,
        "normalized_candidates": candidates,
        "evidence": evidence,
        "evidence_sha256": _hash_records(
            "cppmega-ci-source-binding-evidence-v1",
            evidence,
        ),
    }


def _inventory(bindings: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(
        bindings,
        key=lambda item: (
            str(item["repository"]),
            str(item["head_sha"]),
            str(item["source_path"]),
        ),
    )
    inventory_hash = _hash_records(
        "cppmega-ci-source-binding-inventory-v1",
        (
            {
                "repository": binding["repository"],
                "head_sha": binding["head_sha"],
                "source_path": binding["source_path"],
                "normalization_status": binding["normalization_status"],
                "normalized_candidates": binding["normalized_candidates"],
                "evidence_sha256": binding["evidence_sha256"],
                "evidence": binding["evidence"],
            }
            for binding in ordered
        ),
    )
    return {
        "schema": INVENTORY_SCHEMA,
        "occurrence_set_sha256": "1" * 64,
        "upstream_fetch_receipt_sha256": "2" * 64,
        "binding_count": len(ordered),
        "binding_inventory_sha256": inventory_hash,
        "bindings": ordered,
    }


def _new_source_store(
    root: Path,
    bindings: list[dict[str, object]],
) -> SourceSidecarStore:
    frozen_inventory = _inventory(bindings)
    return SourceSidecarStore(
        root,
        occurrence_set_sha256="1" * 64,
        upstream_fetch_receipt_sha256="2" * 64,
        binding_inventory_sha256=str(frozen_inventory["binding_inventory_sha256"]),
        input_binding_count=len(bindings),
        max_pack_bytes=1024,
    )


def test_path_normalization_handles_posix_windows_relative_cd_and_escape() -> None:
    assert (
        normalize_source_path(
            "../src/main.cpp",
            "/home/runner/work/repo/repo/build",
            repository="owner/repo",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            r"..\src\main.cpp",
            r"D:\a\repo\repo\build",
            repository="owner/repo",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            "../../src/main.cpp",
            "build/nested",
            repository="owner/repo",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            "src/main.cpp",
            "build/..",
            repository="owner/repo",
        )
        == "src/main.cpp"
    )
    assert (
        normalize_source_path(
            "/__w/repo/repo/src/main.cpp",
            None,
            repository="owner/repo",
        )
        == "src/main.cpp"
    )

    escaped = normalize_source_candidates(
        "../../outside.cpp",
        "build",
        repository="owner/repo",
    )
    assert escaped.status == GENERATED_OR_MUTATED_UNRESOLVABLE
    assert escaped.candidates == ()

    outside = normalize_source_candidates(
        "/tmp/generated.cpp",
        ".",
        repository="owner/repo",
    )
    assert outside.status == GENERATED_OR_MUTATED_UNRESOLVABLE

    ambiguous = normalize_source_candidates(
        "/home/runner/work/repo/repo/generated/repo/repo/source.cpp",
        ".",
        repository="owner/repo",
    )
    assert ambiguous.status == AMBIGUOUS_PATH
    assert ambiguous.candidates == (
        "generated/repo/repo/source.cpp",
        "source.cpp",
    )


def test_exact_local_commit_and_component_tree_traversal(tmp_path: Path) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    resolver = LocalGitResolver({"owner/repo": mirror})
    result = resolver.resolve(_binding(head, "src/nested/main.cpp"))

    expected = b"int main() { return 0; }\n"
    assert result.status == RESOLVED
    assert result.commit_oid == head
    assert result.root_tree_oid
    assert result.parent_tree_oid
    assert result.object_format == "sha1"
    assert result.mode == "100644"
    assert result.object_type == "blob"
    assert result.content_kind == "text"
    assert result.content == expected
    assert result.content_sha256 == hashlib.sha256(expected).hexdigest()
    assert (
        result.blob_oid
        == hashlib.sha1(f"blob {len(expected)}\0".encode() + expected).hexdigest()
    )
    assert [entry["component"] for entry in result.traversal] == [
        "src",
        "nested",
        "main.cpp",
    ]
    assert all(entry["tree_oid"] for entry in result.traversal)
    assert result.evidence["runner_filesystem_equivalence_claimed"] is False


def test_binary_symlink_submodule_and_lfs_pointer_are_not_dereferenced(
    tmp_path: Path,
) -> None:
    mirror, head, base = _git_fixture(tmp_path)
    resolver = LocalGitResolver({"owner/repo": mirror})

    binary = resolver.resolve(_binding(head, "assets/bytes.bin"))
    assert binary.status == RESOLVED
    assert binary.content_kind == "binary"
    assert binary.content == b"\x00\xff\x10binary\r\n"

    symlink = resolver.resolve(_binding(head, "main-link"))
    assert symlink.status == RESOLVED
    assert symlink.mode == "120000"
    assert symlink.object_type == "symlink"
    assert symlink.content_kind == "symlink"
    assert symlink.content == b"src/nested/main.cpp"
    assert symlink.evidence["dereferenced"] is False

    lfs = resolver.resolve(_binding(head, "model.lfs"))
    assert lfs.status == RESOLVED
    assert lfs.content_kind == "lfs_pointer"
    assert lfs.lfs_oid_sha256 == "a" * 64
    assert lfs.lfs_size == 123456
    assert lfs.content is not None and lfs.content.startswith(
        b"version https://git-lfs.github.com/spec/v1\n"
    )

    submodule = resolver.resolve(_binding(head, "vendor/component"))
    assert submodule.status == UNSUPPORTED_OBJECT
    assert submodule.mode == "160000"
    assert submodule.object_type == "submodule"
    assert submodule.object_oid == base
    assert submodule.content is None
    assert submodule.evidence["dereferenced"] is False


def test_gap_statuses_are_distinct_and_fail_closed(tmp_path: Path) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")

    assert LocalGitResolver({}).resolve(binding).status == REPO_UNAVAILABLE
    assert (
        LocalGitResolver({"owner/repo": {"status": DELETED_FORK}})
        .resolve(binding)
        .status
        == DELETED_FORK
    )

    wrong_commit = _binding("f" * 40, "src/nested/main.cpp")
    assert (
        LocalGitResolver({"owner/repo": mirror}).resolve(wrong_commit).status
        == COMMIT_ABSENT
    )
    absent = _binding(head, "src/does-not-exist.cpp")
    assert (
        LocalGitResolver({"owner/repo": mirror}).resolve(absent).status == PATH_ABSENT
    )


def test_one_blob_referenced_by_multiple_bindings_is_stored_once(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    resolver = LocalGitResolver({"owner/repo": mirror})
    first_binding = _binding(head, "src/nested/main.cpp")
    second_binding = _binding(head, "src/copy.cpp")
    first = resolver.resolve(first_binding)
    second = resolver.resolve(second_binding)
    assert first.content_sha256 == second.content_sha256
    assert first.blob_oid == second.blob_oid

    with _new_source_store(
        tmp_path / "store",
        [first_binding, second_binding],
    ) as store:
        store.add_resolution(first)
        store.add_resolution(second)
        verification = store.verify()
        assert verification["binding_count"] == 2
        assert verification["blob_count"] == 1
        assert verification["git_object_count"] == 1
        assert store.read_blob(str(first.content_sha256)) == first.content
        receipt = store.receipt()
        assert receipt["status"] == "complete"
        assert receipt["content_semantics"] == CONTENT_SEMANTICS
        ledger = store.reference_ledger()
        assert ledger["reference_count"] == 2
        assert all(entry["content_sha256"] for entry in ledger["entries"])
        assert all("content_bytes" not in entry for entry in ledger["entries"])
        assert all("body" not in entry for entry in ledger["entries"])

    with sqlite3.connect(tmp_path / "store" / "index.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM bindings").fetchone()[0] == 2


class _CorruptBlobResolver(LocalGitResolver):
    def _run_git(
        self,
        mirror: Path,
        args: list[str] | tuple[str, ...],
        *,
        absent_ok: bool = False,
    ) -> bytes:
        value = super()._run_git(mirror, args, absent_ok=absent_ok)
        if len(args) == 3 and list(args[:2]) == ["cat-file", "blob"]:
            return value + b"corrupt"
        return value


def test_corrupt_git_bytes_and_corrupt_pack_fail_verification(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    with pytest.raises(ResolutionIntegrityError, match="hash to"):
        _CorruptBlobResolver({"owner/repo": mirror}).resolve(binding)

    resolution = LocalGitResolver({"owner/repo": mirror}).resolve(binding)
    root = tmp_path / "store"
    with _new_source_store(root, [binding]) as store:
        store.add_resolution(resolution)
        pack = root / str(store.verify()["pack_hashes"][0]["filename"])
    with pack.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        byte = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([byte[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())

    with _new_source_store(root, [binding]) as reopened:
        with pytest.raises(SourceStoreError, match="frame verification"):
            reopened.verify()


def test_inventory_order_is_frozen_and_noncanonical_insertion_order_is_refused(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    inventory = _inventory(
        [
            _binding(head, "src/nested/main.cpp"),
            _binding(head, "assets/bytes.bin"),
            _binding(head, "src/copy.cpp"),
        ]
    )
    receipt_a = materialize_inventory(
        inventory,
        {"owner/repo": mirror},
        tmp_path / "store-a",
        max_pack_bytes=100,
    )
    with pytest.raises(ExtractionError, match="not sorted and unique"):
        materialize_inventory(
            {**inventory, "bindings": list(reversed(inventory["bindings"]))},
            {"owner/repo": mirror},
            tmp_path / "store-b",
            max_pack_bytes=100,
        )
    assert receipt_a["status"] == "complete"


def test_canonical_materialization_has_identical_logical_and_pack_receipts(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    inventory = _inventory(
        [
            _binding(head, "assets/bytes.bin"),
            _binding(head, "src/nested/main.cpp"),
        ]
    )
    first = materialize_inventory(
        inventory,
        {"owner/repo": mirror},
        tmp_path / "first",
        max_pack_bytes=100,
    )
    second = materialize_inventory(
        inventory,
        {"owner/repo": mirror},
        tmp_path / "second",
        max_pack_bytes=100,
    )
    for field in (
        "logical_blob_set_sha256",
        "logical_git_object_set_sha256",
        "logical_binding_set_sha256",
        "binding_reference_ledger_sha256",
        "pack_hashes",
    ):
        assert first[field] == second[field]


def test_uncommitted_pack_tail_is_quarantined_and_truncated_on_reopen(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    binding = _binding(head, "src/nested/main.cpp")
    resolution = LocalGitResolver({"owner/repo": mirror}).resolve(binding)
    root = tmp_path / "store"
    with _new_source_store(root, [binding]) as store:
        store.add_resolution(resolution)
        pack_record = store.verify()["pack_hashes"][0]
        pack = root / str(pack_record["filename"])
        committed_end = int(pack_record["committed_end"])
    with pack.open("ab") as handle:
        handle.write(b"simulated-crash-tail\x00\xff")
        handle.flush()
        os.fsync(handle.fileno())

    with _new_source_store(root, [binding]) as reopened:
        verification = reopened.verify()
        assert pack.stat().st_size == committed_end
        assert verification["recovery"]["orphan_count"] == 1
        record = verification["recovery"]["records"][0]
        assert record["reason"] == "uncommitted_pack_tail"
        assert record["byte_size"] == len(b"simulated-crash-tail\x00\xff")
        assert reopened.read_blob(str(resolution.content_sha256)) == (
            resolution.content
        )


def test_receipt_is_explicitly_incomplete_for_any_gap_or_missing_binding(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    resolver = LocalGitResolver({"owner/repo": mirror})
    resolved_binding = _binding(head, "src/nested/main.cpp")
    gap_binding = _binding(head, "missing.cpp")
    unattempted_binding = _binding(head, "not-attempted.cpp")
    resolved = resolver.resolve(resolved_binding)
    gap = resolver.resolve(gap_binding)
    with _new_source_store(
        tmp_path / "store",
        [resolved_binding, gap_binding, unattempted_binding],
    ) as store:
        store.add_resolution(resolved)
        store.add_resolution(gap)
        receipt = store.receipt()

    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["status"] == "incomplete"
    assert receipt["input_binding_count"] == 3
    assert receipt["resolved_binding_count"] == 1
    assert receipt["missing_binding_count"] == 1
    assert receipt["gap_status_counts"] == {PATH_ABSENT: 1}
    assert receipt["content_semantics"] == "repository_blob_content"
    assert "exact_runner" not in json.dumps(receipt)


def test_receipt_refuses_same_count_bindings_outside_frozen_inventory(
    tmp_path: Path,
) -> None:
    mirror, head, _base = _git_fixture(tmp_path)
    required = _binding(head, "src/nested/main.cpp")
    wrong = _binding(head, "assets/bytes.bin")
    resolution = LocalGitResolver({"owner/repo": mirror}).resolve(wrong)

    with _new_source_store(tmp_path / "store", [required]) as store:
        store.add_resolution(resolution)
        with pytest.raises(
            SourceStoreError,
            match="differs from frozen input inventory",
        ):
            store.receipt()


def _ci_occurrence_provenance(
    head: str,
    *,
    exact_attempt_match: bool = True,
) -> dict[str, object]:
    return {
        "schema": "cppmega_ci_chunk_occurrence_v3",
        "source_repository": "owner/repo",
        "workflow": {"head_sha": head},
        "run_metadata_evidence": {
            "exact_attempt_match": exact_attempt_match,
        },
        "chunk": {
            "training_sidecars": {
                "schema": "cppmega_ci_chunk_training_sidecars_v2",
                "build_actions": [
                    {
                        "action_entity_id": "entity:compile",
                        "action_shape_sha256": "4" * 64,
                        "command_sha256": "5" * 64,
                        "cwd": "/home/runner/work/repo/repo/build",
                        "source_inputs": ["../src/main.cpp"],
                        "repository_source_bindings": [
                            {
                                "repository": "wrong/heuristic",
                                "head_sha": "f" * 40,
                                "source_path": "wrong.cpp",
                            }
                        ],
                    },
                    {
                        "action_entity_id": "entity:duplicate",
                        "action_shape_sha256": "6" * 64,
                        "command_sha256": "7" * 64,
                        "cwd": "build",
                        "source_inputs": ["../src/main.cpp"],
                        "repository_source_bindings": [],
                    },
                ],
            }
        },
    }


def _frozen_ci_fixture(
    tmp_path: Path,
    head: str,
    *,
    exact_attempt_match: bool = True,
) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "ci-store"
    with CIContentStore(root, max_pack_bytes=1024) as store:
        store.add_chunk(
            "compile output\n",
            _ci_occurrence_provenance(
                head,
                exact_attempt_match=exact_attempt_match,
            ),
            {
                "repo": "owner/repo",
                "run_attempt": "1:1",
                "job": "linux",
                "step": "compile:0",
                "chunk_ordinal": 0,
            },
            token_count=1,
            tokenizer_fingerprint="tokenizer-test-v1",
            token_sequence_sha256=hash_token_sequence([1]),
        )
        content_receipt = store.completion_receipt(target_unique_tokens=1)
    fetch_receipt = {
        "schema": FETCH_RECEIPT_SCHEMA,
        "content_store_receipt": content_receipt,
    }
    fetch_path = tmp_path / "fetch-receipt.json"
    encoded = (json.dumps(fetch_receipt, indent=2, sort_keys=True) + "\n").encode()
    fetch_path.write_bytes(encoded)
    return root, fetch_path, fetch_receipt


def test_tiny_immutable_ci_fixture_extracts_unique_exact_binding_inventory(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    root, fetch_path, _receipt = _frozen_ci_fixture(tmp_path, head)

    inventory = extract_binding_inventory(root, fetch_path)
    verify_binding_inventory(inventory)

    assert inventory["schema"] == INVENTORY_SCHEMA
    assert inventory["binding_count"] == 1
    assert len(str(inventory["binding_inventory_sha256"])) == 64
    assert (
        inventory["upstream_fetch_receipt_sha256"]
        == hashlib.sha256(fetch_path.read_bytes()).hexdigest()
    )
    binding = inventory["bindings"][0]
    assert binding["repository"] == "owner/repo"
    assert binding["head_sha"] == head
    assert binding["source_path"] == "src/main.cpp"
    assert binding["normalization_status"] == RESOLVED
    assert len(binding["evidence"]) == 2
    assert all(
        evidence["normalization"]["candidates"] == ["src/main.cpp"]
        for evidence in binding["evidence"]
    )
    assert all(
        evidence["discarded_heuristic_bindings_sha256"]
        for evidence in binding["evidence"]
    )


def test_inventory_extraction_refuses_non_exact_attempt_metadata(
    tmp_path: Path,
) -> None:
    _mirror, head, _base = _git_fixture(tmp_path)
    root, fetch_path, _receipt = _frozen_ci_fixture(
        tmp_path,
        head,
        exact_attempt_match=False,
    )
    with pytest.raises(
        ExtractionError,
        match="exact-attempt run metadata evidence",
    ):
        extract_binding_inventory(root, fetch_path)
