import json
import struct
import subprocess
import tarfile

import pytest

import scripts.data.publish_megatron_bundle_to_nebius_s3 as publisher
from scripts.nebius_h200_megatron_cpp_world_curriculum import (
    S3RestorePlan,
    Stage,
    _assert_prefix_contract,
    _default_stages,
    _make_s3_auth_tar,
    _make_curriculum_manifest,
    _parse_stage,
    _remote_script,
    _resolve_s3_credentials,
    _validate_training_source_revision,
)
from scripts.nebius_h200_megatron_cpp_world_sweep import DEFAULT_DOCKER_IMAGE


_MEGATRON_COMMIT = "b" * 40


def _capacity(stage, *, edges=23, chunks=17):
    return {
        "schema": "cppmega_graph_capacity_v1",
        "status": "verified",
        "sequence_length": stage.seq,
        "graph_max_edges": edges,
        "graph_max_chunks": chunks,
    }


def _write_prefix_contract_fixture(
    prefix,
    *,
    missing_token_sidecar=None,
    missing_graph_sidecar=None,
):
    prefix.with_suffix(".bin").write_bytes(struct.pack("<H", 1))
    prefix.with_suffix(".idx").write_bytes(
        b"MMIDIDX\x00\x00"
        + struct.pack("<QBQQ", 1, 8, 1, 2)
        + struct.pack("<i", 1)
        + struct.pack("<q", 0)
        + struct.pack("<2q", 0, 1)
    )

    side_channel_paths = {}
    for name in sorted(publisher.REQUIRED_TOKEN_SIDECARS):
        if name == missing_token_sidecar:
            continue
        dtype = publisher.TOKEN_SIDECAR_DTYPES[name]
        relative = f"{prefix.name}_{name}.bin"
        (prefix.parent / relative).write_bytes(
            b"\x01" * publisher.DTYPE_SIZES[dtype]
        )
        side_channel_paths[name] = {"path": relative, "dtype": dtype}

    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "tokenizer_contract": "megacpp",
                "vocab_size": 65536,
                "token_count": 1,
                "document_count": 1,
                "dtype": "uint16",
                "loss_mask_alignment": "source_token_predicts_next_v1",
                "side_channel_paths": side_channel_paths,
                "symbol_identity_schema_version": 3,
                "graph_sidecar_schema": "cppmega_graph_routes_v2",
                "graph_sidecar_paths": {
                    name: {}
                    for name in publisher.REQUIRED_GRAPH_SIDECARS
                    if name != missing_graph_sidecar
                },
                "source_platform_sidecar": {
                    "schema": "cppmega_source_platform_v1",
                },
            }
        ),
        encoding="utf-8",
    )
    return Stage(
        index=1,
        seq=1024,
        batch=1,
        micro_batch=1,
        iters=1,
        prefix=prefix,
    )


@pytest.mark.parametrize(
    "missing_token_sidecar",
    ["token_source_doc_ids", "token_source_identity_ids"],
)
def test_curriculum_prefix_contract_rejects_missing_canonical_source_sidecars(
    tmp_path,
    missing_token_sidecar,
):
    stage = _write_prefix_contract_fixture(
        tmp_path / "train",
        missing_token_sidecar=missing_token_sidecar,
    )

    with pytest.raises(
        ValueError,
        match=rf"missing token sidecars.*{missing_token_sidecar}",
    ):
        _assert_prefix_contract([stage])


@pytest.mark.parametrize(
    "missing_graph_sidecar",
    [
        "token_call_edges",
        "token_domain_edges",
        "token_build_edges",
        "token_shell_edges",
        "token_diagnostic_edges",
        "token_cross_domain_edges",
        "token_chunk_starts",
    ],
)
def test_curriculum_prefix_contract_rejects_missing_canonical_graph_families(
    tmp_path,
    missing_graph_sidecar,
):
    stage = _write_prefix_contract_fixture(
        tmp_path / "train",
        missing_graph_sidecar=missing_graph_sidecar,
    )

    with pytest.raises(
        ValueError,
        match=rf"graph sidecar key set.*{missing_graph_sidecar}",
    ):
        _assert_prefix_contract([stage])


def test_default_curriculum_keeps_token_budget_with_receipt_backed_dsa_microbatch():
    stages = _default_stages()

    by_seq = {stage.seq: stage for stage in stages}

    assert by_seq[1024].batch == 192
    assert by_seq[1024].micro_batch == 1
    assert by_seq[2048].batch == 96
    assert by_seq[2048].micro_batch == 1
    assert by_seq[4096].batch == 40
    assert by_seq[4096].micro_batch == 1
    assert by_seq[4096].iters == 2311
    assert by_seq[8192].batch == 16
    assert by_seq[8192].micro_batch == 1
    assert by_seq[8192].iters == 2756
    assert by_seq[16384].batch == 8
    assert by_seq[16384].micro_batch == 1
    assert by_seq[16384].iters == 2391


def test_parse_stage_accepts_explicit_micro_batch(tmp_path):
    prefix = tmp_path / "train"
    for suffix in (".bin", ".idx", ".json"):
        prefix.with_suffix(suffix).write_text("x")

    stage = _parse_stage(f"8192=16=4=2756={prefix}", 4)

    assert stage.seq == 8192
    assert stage.batch == 16
    assert stage.micro_batch == 4
    assert stage.iters == 2756
    assert stage.remote_checkpoint_root.endswith("stage_04_seq_8192_gbs_16_mbs_4")


def test_parse_stage_keeps_legacy_batch_as_micro_batch(tmp_path):
    prefix = tmp_path / "train"
    for suffix in (".bin", ".idx", ".json"):
        prefix.with_suffix(suffix).write_text("x")

    stage = _parse_stage(f"2048=96=1686={prefix}", 2)

    assert stage.batch == 96
    assert stage.micro_batch == 96


def test_curriculum_manifest_records_global_and_micro_batch(tmp_path):
    stage = _default_stages()[3]
    out = tmp_path / "manifest.json"

    _make_curriculum_manifest(
        [stage],
        out,
        graph_capacities={stage.index: _capacity(stage)},
        remote_prefixes={stage.index: "data/seq_8192/train"},
        bundle_identity={"bundle_id": "bundle-1", "artifact_set_sha256": "a" * 64},
    )

    payload = json.loads(out.read_text())
    assert payload["schema"] == "cppmega_h200_curriculum_v2"
    assert payload["stages"][0]["global_batch"] == 16
    assert payload["stages"][0]["micro_batch"] == 1
    assert payload["stages"][0]["graph_capacity"]["graph_max_edges"] == 23
    assert payload["stages"][0]["remote_prefix"] == "data/seq_8192/train"
    assert payload["checkpoint_transition"] == {
        "mode": "model_weights_warm_start",
        "optimizer_state": "reset",
        "rng_state": "reset",
        "scheduler_state": "reset",
        "data_iterator_state": "reset_per_stage",
        "exact_resume": False,
    }


def test_curriculum_overlay_must_match_clean_bundle_producer_revision():
    manifest = {
        "implementation": {
            "components": {
                "cppmega": {
                    "commit": "a" * 40,
                    "tree_sha256": "b" * 64,
                }
            }
        }
    }
    revision = {
        "producer_role": "canonical_source_conveyor",
        "repository_identity": "cppmega",
        "dirty": False,
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }

    assert _validate_training_source_revision(manifest, revision) == revision

    with pytest.raises(RuntimeError, match="clean canonical"):
        _validate_training_source_revision(
            manifest,
            {**revision, "dirty": True},
        )
    with pytest.raises(RuntimeError, match="commit differs"):
        _validate_training_source_revision(
            manifest,
            {**revision, "git_commit": "c" * 40},
        )


def test_curriculum_container_is_fail_closed_and_bash_parseable(tmp_path):
    stage = _default_stages()[3]
    script = _remote_script(
        [stage],
        docker_image=DEFAULT_DOCKER_IMAGE,
        fp8_recipe="tensorwise",
        remote_prefixes={stage.index: "data/seq_8192/train"},
        graph_capacities={stage.index: _capacity(stage)},
        megatron_commit=_MEGATRON_COMMIT,
        initial_checkpoint_root="/data/cppmega_curriculum_checkpoints/initial",
    )
    script_path = tmp_path / "curriculum.sh"
    script_path.write_text(script, encoding="utf-8")

    assert "export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=0" in script
    assert 'export CPPMEGA_DSA_PATCH_ENABLED="1"' in script
    assert "--enable-dsa-patch" in script
    assert "if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1'" in script
    assert "export CPPMEGA_DSA_GRAPH_AUX_ENABLED=1" in script
    assert "export CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001" in script
    assert "GRAPH_PRIOR_RECEIPT=\"/data/cppmega_h200_results/stage_${STAGE_IDX}_graph_prior.json\"" in script
    assert "reason=dsa_selector_gate" in script
    assert "--experimental-attention-variant dsa" in script
    assert "--dsa-indexer-loss-coeff 0.001" in script
    assert (
        "--spec cppmega.megatron.nam56r_full_spec "
        "build_cppmega_nam56r_full_stack_spec"
    ) in script
    assert "unset CPPMEGA_DENSE_GQA" in script
    assert "--group-query-attention" not in script
    assert "apply_graph_route_attention_bias_patch()" in script
    assert "--micro-batch-size ${MBS}" in script
    assert "--global-batch-size ${BS}" in script
    assert "--eval-interval ${EVAL_INTERVAL}" in script
    assert "--save-interval ${TARGET_ITERS}" in script
    assert "--no-check-for-nan-in-loss-and-grad" not in script
    assert "write_graph_capacity_receipt" in script
    assert "write_training_loss_receipt" in script
    assert "validate_production_batch_receipt" in script
    assert "validate_embedding_consumption_receipt" in script
    assert "reason=sidecar_consumption_gate" in script
    assert "CPPMEGA_H200_FULL_SIDECAR_RECEIPT=1" in script
    assert "cppmega_overlay_revision.json" in script
    assert "source_tree_sha256" in script
    assert "model_weights_warm_start optimizer=reset rng=reset scheduler=reset exact_resume=false" in script
    assert "--finetune --no-load-optim --no-load-rng --override-opt-param-scheduler" in script
    assert "checkpoint_iteration expected=${TARGET_ITERS}" in script
    preflight = "python /opt/cppmega/scripts/h200_megatron_preflight.py"
    assert preflight in script
    assert script.index(preflight) < script.index('for SPEC in "${STAGES[@]}"')
    subprocess.run(["bash", "-n", str(script_path)], check=True)

    dsa_script = _remote_script(
        [stage],
        docker_image=DEFAULT_DOCKER_IMAGE,
        fp8_recipe="tensorwise",
        remote_prefixes={stage.index: "data/seq_8192/train"},
        graph_capacities={stage.index: _capacity(stage)},
        megatron_commit=_MEGATRON_COMMIT,
        enable_dsa_patch=True,
    )
    assert 'export CPPMEGA_DSA_PATCH_ENABLED="1"' in dsa_script
    assert "--enable-dsa-patch" in dsa_script


def test_curriculum_rejects_dense_only_mode():
    stage = _default_stages()[0]
    with pytest.raises(ValueError, match="requires the fused DSA patch"):
        _remote_script(
            [stage],
            docker_image=DEFAULT_DOCKER_IMAGE,
            fp8_recipe="tensorwise",
            remote_prefixes={stage.index: "data/seq_1024/train"},
            graph_capacities={stage.index: _capacity(stage)},
            megatron_commit=_MEGATRON_COMMIT,
            enable_dsa_patch=False,
        )


def test_s3_restore_is_verified_before_curriculum_training(tmp_path):
    stage = _default_stages()[0]
    plan = S3RestorePlan(
        bundle_id="bundle-verified",
        artifact_set_sha256="a" * 64,
        bucket="bucket",
        prefix="cppmega/full",
        endpoint_url="https://storage.example.invalid",
        megatron_commit=_MEGATRON_COMMIT,
        run_id="curriculum-001",
    )
    script = _remote_script(
        [stage],
        docker_image=DEFAULT_DOCKER_IMAGE,
        fp8_recipe="tensorwise",
        remote_prefixes={stage.index: "data/seq_1024/train"},
        graph_capacities={stage.index: _capacity(stage)},
        megatron_commit=_MEGATRON_COMMIT,
        bundle_root=plan.remote_bundle_root,
        tokenizer_model=f"{plan.remote_bundle_root}/tokenizer",
        s3_restore=plan,
    )
    script_path = tmp_path / "s3-curriculum.sh"
    script_path.write_text(script, encoding="utf-8")

    restore = "restore_megatron_bundle_from_nebius_s3.py"
    preflight = "python /opt/cppmega/scripts/h200_megatron_preflight.py"
    assert restore in script
    assert "--require-empty-output-root" not in script
    assert "--bundle-id bundle-verified" in script
    assert f"--megatron-commit {_MEGATRON_COMMIT}" in script
    assert "--s3-client python" in script
    assert "--s3-region eu-north1" in script
    assert "command -v aws" not in script
    assert "command -v zstd" in script
    assert "CPPMEGA_S3_RESTORE_STATUS=PASS" in script
    assert "s3_restore_receipt.json" in script
    assert script.index(restore) < script.index(preflight)
    assert script.index("trap cleanup_cppmega_remote_secrets EXIT") < script.index(
        "sudo docker pull"
    )
    assert "AWS_SECRET_ACCESS_KEY" not in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_s3_auth_archive_contains_only_normalized_s3_credentials(tmp_path):
    credentials = _resolve_s3_credentials(
        {
            "NEBIUS_S3_ACCESS_KEY_ID": "access-test",
            "NEBIUS_S3_SECRET_ACCESS_KEY": "secret-test",
            "UNRELATED_SECRET": "must-not-be-archived",
        }
    )
    archive_path = tmp_path / "s3-auth.tgz"

    _make_s3_auth_tar(archive_path, credentials)

    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile("cppmega_s3_auth/.env")
        assert member is not None
        payload = member.read().decode("utf-8")
    assert payload == (
        "AWS_ACCESS_KEY_ID=access-test\n"
        "AWS_SECRET_ACCESS_KEY=secret-test\n"
    )
    assert "NEBIUS_S3_" not in payload
    assert "UNRELATED_SECRET" not in payload


def test_s3_restore_must_match_curriculum_bundle_and_megatron_identity():
    stage = _default_stages()[0]
    plan = S3RestorePlan(
        bundle_id="bundle-verified",
        artifact_set_sha256="a" * 64,
        bucket="bucket",
        prefix="cppmega/full",
        endpoint_url="https://storage.example.invalid",
        megatron_commit=_MEGATRON_COMMIT,
        run_id="curriculum-001",
    )

    with pytest.raises(ValueError, match="bundle root disagree"):
        _remote_script(
            [stage],
            docker_image=DEFAULT_DOCKER_IMAGE,
            fp8_recipe="tensorwise",
            remote_prefixes={stage.index: "data/seq_1024/train"},
            graph_capacities={stage.index: _capacity(stage)},
            megatron_commit=_MEGATRON_COMMIT,
            bundle_root="/data/not-the-restored-bundle",
            s3_restore=plan,
        )
