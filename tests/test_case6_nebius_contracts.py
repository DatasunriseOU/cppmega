from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cppmega.megatron.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    STAGE1_GRAPH_TOPK,
    stage1_graph_recipe_binding,
)
from cppmega.megatron.objective_contract import (
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_IDS,
    validate_objective_contract,
)
from cppmega.megatron.graph_objective_loss import graph_bias_beta_binding
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _head_matches,
    _resolve_s3_env,
    _validate_logical_manifest_contract,
)
from scripts.data.restore_megatron_bundle_from_nebius_s3 import (
    _acquire_restore_lock,
    _remove_partial_tree,
    _validate_run_id,
)
from scripts.nebius_h200_megatron_cpp_world_sweep import (
    _host_key_fingerprint,
    _ssh_host_key_failure,
    scp_base,
    ssh_base,
    validate_ssh_host_key_contract,
)
from scripts.h200_megatron_preflight import (
    STACK_REQUIRED_IMPORTS,
    _derive_graph_capacity_from_manifest,
    _iteration_evidence,
    _validate_graph_prior_receipt,
    _profile_environment,
    build_megatron_command,
    validate_stack_compatibility,
)


TASKS = (
    "causal_lm",
    "fim",
    "ast_fim",
    "ifim",
    "commit_diff",
    "pre_to_post",
)


PINNED_NEBIUS_HOST_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIJRwravCVfVsFZfdgfvC/OlW0K7vrJ7pBjl5p86YKSSs"
)
PINNED_NEBIUS_HOST_KEY_FINGERPRINT = (
    "SHA256:xGOQHYDUpAKZPiHLlYNYp01FiayrndE1tGC9wBoA+xw"
)
OTHER_NEBIUS_HOST_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIASmjUl/IUCFfvXkXpCWWGJJ04Tx5FWEevIdFRYJCBic"
)


def _ssh_contract_args(
    *,
    host_key: str | None = PINNED_NEBIUS_HOST_KEY,
    host_key_file: Path | None = None,
    fingerprint: str | None = PINNED_NEBIUS_HOST_KEY_FINGERPRINT,
) -> SimpleNamespace:
    return SimpleNamespace(
        ssh_key=Path("/tmp/cppmega-test-ssh-key"),
        ssh_user="dave",
        ssh_host_key=host_key,
        ssh_host_key_file=host_key_file,
        ssh_host_key_fingerprint=fingerprint,
    )


def _valid_objective_contract() -> dict[str, object]:
    return {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "algorithm": "hamilton_eligibility_bipartite_v1",
        "seed": 17,
        "quota_window_samples": 6,
        "task_order": list(TASKS),
        "objective_ids": {task: OBJECTIVE_IDS[task] for task in TASKS},
        "configured_rates": {task: "1/6" for task in TASKS},
        "planned_samples": {task: 1 for task in TASKS},
        "realized": {
            task: {
                "samples": 1,
                "input_tokens": 3,
                "loss_tokens": 3 if task == "causal_lm" else 2,
            }
            for task in TASKS
        },
        "totals": {"samples": 6, "input_tokens": 18, "loss_tokens": 13},
        "typed_sources": {
            "ifim_instruction": "ifim_instruction_token_ids",
            "commit_message": "commit_msg_token_ids",
            "diff": "diff_token_ids",
            "pre": "pre_token_ids",
            "post": "post_token_ids",
            "missing_fields": "ineligible",
            "rendered_text_parsing": False,
        },
        "graph_auxiliary": {
            "recipe": stage1_graph_recipe_binding(),
            "relations": list(STAGE1_GRAPH_RELATIONS),
            "bias_beta": "1",
            "score_formula": "i_neural_plus_beta_s_graph_v1",
            "score_stage": "before_topk",
            "eligible_samples": 1,
            "positive_edges": 5,
            "global_weight": "1",
            "indexer_weight": "1/1000",
            "layer_weight": "1",
            "layer_reduction": "sum",
            "bce_weight": "1/10",
            "coverage_weight": "1/20",
            "topk": STAGE1_GRAPH_TOPK,
            "pos_weight": "1",
            "margin": "1",
            "included_in_total_loss": True,
            "runtime": "megatron_dsa_indexer_v1",
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
        "materialization": {
            "format": "shifted_lm_document_v1",
            "token_column": "input_ids",
            "loss_mask_column": "loss_mask",
            "loss_mask_alignment": (
                LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
            ),
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
    }


def test_nebius_ssh_contract_has_no_unpinned_default() -> None:
    args = _ssh_contract_args(host_key=None, fingerprint=None)

    with pytest.raises(RuntimeError, match="host-key pin is required"):
        validate_ssh_host_key_contract(args)


def test_nebius_ssh_contract_rejects_missing_host_key_file(tmp_path: Path) -> None:
    args = _ssh_contract_args(
        host_key=None,
        host_key_file=tmp_path / "missing-host-key.pub",
    )

    with pytest.raises(FileNotFoundError, match="host public-key file not found"):
        validate_ssh_host_key_contract(args)


def test_nebius_ssh_contract_rejects_mismatched_key_fingerprint() -> None:
    args = _ssh_contract_args(host_key=OTHER_NEBIUS_HOST_KEY)

    with pytest.raises(ValueError, match="fingerprint does not match"):
        validate_ssh_host_key_contract(args)


def test_nebius_ssh_contract_treats_presented_key_mismatch_as_fatal() -> None:
    assert _ssh_host_key_failure(
        "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!\n"
        "Host key verification failed."
    )


def test_nebius_ssh_contract_accepts_pinned_key_for_ssh_and_scp(
    tmp_path: Path,
) -> None:
    host_key_file = tmp_path / "nebius-host-ed25519.pub"
    host_key_file.write_text(PINNED_NEBIUS_HOST_KEY + "\n", encoding="ascii")
    args = _ssh_contract_args(host_key=None, host_key_file=host_key_file)
    args._nebius_ssh_known_hosts_dir = tmp_path

    contract = validate_ssh_host_key_contract(args)
    ssh_command = ssh_base(args, "203.0.113.10")
    scp_command = scp_base(args, "203.0.113.10")

    assert contract == (
        "ssh-ed25519",
        PINNED_NEBIUS_HOST_KEY.split()[1],
        PINNED_NEBIUS_HOST_KEY_FINGERPRINT,
    )
    assert _host_key_fingerprint(contract[1]) == PINNED_NEBIUS_HOST_KEY_FINGERPRINT
    known_hosts = Path(args._nebius_ssh_known_hosts_path)
    assert known_hosts.read_text(encoding="ascii") == (
        f"203.0.113.10 {PINNED_NEBIUS_HOST_KEY}\n"
    )
    assert known_hosts.stat().st_mode & 0o777 == 0o600

    for command in (ssh_command, scp_command):
        rendered = " ".join(command)
        assert "BatchMode=yes" in rendered
        assert "PasswordAuthentication=no" in rendered
        assert "KbdInteractiveAuthentication=no" in rendered
        assert "PreferredAuthentications=publickey" in rendered
        assert "StrictHostKeyChecking=yes" in rendered
        assert f"UserKnownHostsFile={known_hosts}" in rendered
        assert "GlobalKnownHostsFile=/dev/null" in rendered
        assert "HostKeyAlgorithms=ssh-ed25519" in rendered
        assert "IdentitiesOnly=yes" in rendered
        assert "ForwardAgent=no" in rendered
        assert "StrictHostKeyChecking=no" not in rendered
        assert "UserKnownHostsFile=/dev/null" not in rendered
    assert ssh_command[-1] == "dave@203.0.113.10"


def test_remote_logical_manifest_rejects_legacy_training_before_archive() -> None:
    manifest = {
        "schema": "cppmega_megatron_bundle_v1",
        "tokenizer_contract": "megacpp-vocab-65536",
        "vocab_size": 65536,
        "training_contract": "legacy_causal",
    }

    with pytest.raises(ValueError, match="training_contract"):
        _validate_logical_manifest_contract(manifest)


def test_s3_credentials_do_not_mix_nebius_and_aws_families() -> None:
    resolved = _resolve_s3_env(
        {
            "NEBIUS_S3_ACCESS_KEY_ID": "nebius-a",
            "NEBIUS_S3_SECRET_ACCESS_KEY": "nebius-s",
            "AWS_ACCESS_KEY_ID": "aws-a",
            "AWS_SECRET_ACCESS_KEY": "aws-s",
            "AWS_SESSION_TOKEN": "stale",
        }
    )
    assert resolved["AWS_ACCESS_KEY_ID"] == "nebius-a"
    assert resolved["AWS_SECRET_ACCESS_KEY"] == "nebius-s"
    assert "AWS_SESSION_TOKEN" not in resolved

    with pytest.raises(SystemExit, match="complete Nebius"):
        _resolve_s3_env(
            {
                "NEBIUS_S3_ACCESS_KEY_ID": "nebius-a",
                "AWS_SECRET_ACCESS_KEY": "aws-s",
            }
        )


def test_s3_already_verified_requires_exact_full_object_checksum() -> None:
    digest = hashlib.sha256(b"remote-bytes").hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
    common = {
        "ContentLength": 12,
        "Metadata": {"sha256": digest},
        "ChecksumSHA256": checksum,
    }

    assert _head_matches(
        {**common, "ChecksumType": "FULL_OBJECT"}, size=12, sha256=digest
    )
    assert not _head_matches(common, size=12, sha256=digest)
    assert not _head_matches(
        {
            **common,
            "ChecksumSHA256": checksum + "-2",
            "ChecksumType": "COMPOSITE",
        },
        size=12,
        sha256=digest,
    )


def test_restore_lock_is_scoped_to_bundle_not_run_id(tmp_path: Path) -> None:
    first = _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")
    try:
        with pytest.raises(RuntimeError, match="restore already active"):
            _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-2")
    finally:
        first.close()


def test_restore_refuses_symlink_lock_path(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.write_text("do not touch", encoding="utf-8")
    (tmp_path / ".bundle-1.restore.lock").symlink_to(target)

    with pytest.raises(ValueError, match="restore lock path"):
        _acquire_restore_lock(tmp_path, bundle_id="bundle-1", run_id="run-1")

    assert target.read_text(encoding="utf-8") == "do not touch"


@pytest.mark.parametrize("run_id", ["../escape", "bad/id", "x" * 129, "space id"])
def test_restore_run_id_is_validated_before_path_use(run_id: str) -> None:
    with pytest.raises(ValueError, match="restore run_id"):
        _validate_run_id(run_id)


def test_restore_refuses_symlink_partial_tree(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    partial = tmp_path / ".bundle.run.partial"
    partial.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="partial path"):
        _remove_partial_tree(partial)

    assert target.is_dir()


def test_case6_runbook_requires_fresh_restore_and_trusted_host_key() -> None:
    runbook = (
        Path(__file__).resolve().parents[1] / "docs/case6_nebius_h200_runbook.md"
    ).read_text(encoding="utf-8")

    assert "--require-empty-output-root" in runbook
    assert "NEBIUS_SSH_HOST_KEY_FILE" in runbook
    assert "NEBIUS_SSH_HOST_KEY_FINGERPRINT" in runbook
    assert "--ssh-host-key-file" in runbook
    assert "--ssh-host-key-fingerprint" in runbook
    assert "out-of-band trusted host key" in runbook
    assert "ssh-keyscan" in runbook


def test_fresh_restore_rejects_symlink_root_before_remote_reads(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output_root = tmp_path / "output"
    output_root.symlink_to(target, target_is_directory=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "aws-called"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        "#!/bin/sh\nprintf called > \"$CASE6_AWS_MARKER\"\nexit 99\n",
        encoding="ascii",
    )
    fake_aws.chmod(0o700)
    env = {
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "AWS_ACCESS_KEY_ID": "test-access-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "CASE6_AWS_MARKER": str(marker),
        "HOME": str(tmp_path),
    }
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/data/restore_megatron_bundle_from_nebius_s3.py"),
            "--output-root",
            str(output_root),
            "--bundle-id",
            "bundle-1",
            "--run-id",
            "fresh",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--require-empty-output-root",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "fresh restore requires an empty output root directory" in result.stderr
    assert not marker.exists()


def test_iteration_proof_cannot_borrow_metrics_from_a_later_iteration() -> None:
    log = (
        "iteration 1/ 1 | lm loss: nan | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 1 |\n"
        "iteration 2/ 2 | lm loss: 6.5 | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 0 |\n"
    )

    with pytest.raises(RuntimeError, match="finite positive LM loss"):
        _iteration_evidence(log, expected_iteration=1)


def test_graph_prior_receipt_uses_canonical_recipe_and_beta_binding() -> None:
    receipt = {
        "status": "verified",
        "consumer": "dsa_indexer",
        "graph_recipe": stage1_graph_recipe_binding(),
        "bias_beta": graph_bias_beta_binding(1.0),
        "prior": {"nonzero": 1},
    }

    assert _validate_graph_prior_receipt(receipt, expected_beta=1.0) is receipt

    stale = dict(receipt)
    stale["graph_recipe"] = {
        "schema": stage1_graph_recipe_binding()["schema"],
        "sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="graph recipe"):
        _validate_graph_prior_receipt(stale, expected_beta=1.0)


def test_profile_requires_derived_graph_capacity() -> None:
    with pytest.raises(ValueError, match="capacities derived"):
        _profile_environment(
            sequence_length=1024,
            micro_batch_size=1,
            fp8_recipe="off",
            graph_max_edges=None,
            graph_max_chunks=None,
            enable_dsa_patch=True,
        )
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=7,
        graph_max_chunks=5,
        enable_dsa_patch=True,
    )
    assert environment["CPPMEGA_GRAPH_MAX_EDGES"] == "7"
    assert environment["CPPMEGA_GRAPH_MAX_CHUNKS"] == "5"


def test_h200_training_command_keeps_nan_checks_enabled(tmp_path: Path) -> None:
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=7,
        graph_max_chunks=5,
        enable_dsa_patch=True,
    )
    command = build_megatron_command(
        wrapper=tmp_path / "wrapper.py",
        data_prefix=tmp_path / "train",
        tokenizer_model=tmp_path / "tokenizer",
        checkpoint_root=tmp_path / "checkpoint",
        sequence_length=1024,
        micro_batch_size=1,
        train_iters=1,
        environment=environment,
        load_checkpoint=False,
    )
    assert "--no-check-for-nan-in-loss-and-grad" not in command


def test_graph_capacity_receipt_comes_from_csr_offsets(tmp_path: Path) -> None:
    data_prefix = tmp_path / "train"
    data_prefix.with_suffix(".json").write_text("{}", encoding="utf-8")
    offsets = (0, 2, 5)
    graph_paths: dict[str, dict[str, object]] = {}
    for name in (
        "token_chunk_starts",
        "token_chunk_ends",
        "token_chunk_kinds",
        "token_chunk_dep_levels",
    ):
        path = tmp_path / f"{name}.offsets"
        path.write_bytes(struct.pack("<3q", *offsets))
        graph_paths[name] = {
            "kind": "ragged_1d",
            "offsets_path": path.name,
            "offset_dtype": "int64",
            "item_count": offsets[-1],
        }

    edge_path = tmp_path / "domain.offsets"
    edge_path.write_bytes(struct.pack("<3q", 0, 1, 1))
    graph_paths["token_domain_edges"] = {
        "kind": "edge_triples",
        "offsets_path": edge_path.name,
        "offset_dtype": "int64",
        "item_count": 1,
    }

    receipt = _derive_graph_capacity_from_manifest(
        data_prefix,
        manifest={
            "document_count": 2,
            "source_capacity_token_count": 2048,
            "graph_sidecar_paths": graph_paths,
        },
        sequence_length=1024,
    )

    assert receipt["status"] == "verified"
    assert receipt["graph_max_edges"] == 1
    assert receipt["graph_max_chunks"] == 3
    assert len(receipt["sidecars"]) == 5


def test_stack_contract_matches_pinned_runtime_and_import_set() -> None:
    lock = {
        "base": {
            "cuda_image": "nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04",
            "python": "3.13",
            "torch": "2.13.0.dev20260611+cu132",
        },
        "wheels": {"transformer_engine": {"version": 2.16}},
    }
    contract = validate_stack_compatibility(
        lock,
        python_version=(3, 13),
        torch_version="2.13.0.dev20260611+cu132",
        cuda_runtime="13.2",
        transformer_engine_version="2.16.0.dev0+local",
        imported_modules=STACK_REQUIRED_IMPORTS,
    )
    assert contract["status"] == "verified"

    with pytest.raises(RuntimeError, match="torch version mismatch"):
        validate_stack_compatibility(
            lock,
            python_version=(3, 13),
            torch_version="2.13.0.dev20260612+cu132",
            cuda_runtime="13.2",
            transformer_engine_version="2.16.0.dev0+local",
            imported_modules=STACK_REQUIRED_IMPORTS,
        )

    with pytest.raises(RuntimeError, match="required H200 extension imports"):
        validate_stack_compatibility(
            lock,
            python_version=(3, 13),
            torch_version="2.13.0.dev20260611+cu132",
            cuda_runtime="13.2",
            transformer_engine_version="2.16.0.dev0+local",
            imported_modules=STACK_REQUIRED_IMPORTS[:-1],
        )


def test_objective_graph_relation_contract_rejects_unknown_relation() -> None:
    contract = _valid_objective_contract()
    graph = contract["graph_auxiliary"]
    assert isinstance(graph, dict)
    graph["relations"] = ["unknown"]

    with pytest.raises(ValueError, match="unknown relations"):
        validate_objective_contract(contract)


def test_objective_graph_eligible_samples_cannot_exceed_total() -> None:
    contract = _valid_objective_contract()
    graph = contract["graph_auxiliary"]
    assert isinstance(graph, dict)
    graph["eligible_samples"] = 7

    with pytest.raises(ValueError, match="cannot exceed totals.samples"):
        validate_objective_contract(contract)
