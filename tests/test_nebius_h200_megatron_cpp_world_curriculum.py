import json
import subprocess

import pytest

from scripts.nebius_h200_megatron_cpp_world_curriculum import (
    _default_stages,
    _make_curriculum_manifest,
    _parse_stage,
    _remote_script,
)
from scripts.nebius_h200_megatron_cpp_world_sweep import DEFAULT_DOCKER_IMAGE


def _capacity(stage, *, edges=23, chunks=17):
    return {
        "schema": "cppmega_graph_capacity_v1",
        "status": "verified",
        "sequence_length": stage.seq,
        "graph_max_edges": edges,
        "graph_max_chunks": chunks,
    }


def test_default_curriculum_uses_h200_observed_long_context_batches():
    stages = _default_stages()

    by_seq = {stage.seq: stage for stage in stages}

    assert by_seq[1024].batch == 192
    assert by_seq[1024].micro_batch == 192
    assert by_seq[2048].batch == 96
    assert by_seq[2048].micro_batch == 96
    assert by_seq[4096].batch == 40
    assert by_seq[4096].micro_batch == 40
    assert by_seq[4096].iters == 2311
    assert by_seq[8192].batch == 16
    assert by_seq[8192].micro_batch == 4
    assert by_seq[8192].iters == 2756
    assert by_seq[16384].batch == 8
    assert by_seq[16384].micro_batch == 2
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
    assert payload["stages"][0]["micro_batch"] == 4
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


def test_curriculum_container_is_fail_closed_and_bash_parseable(tmp_path):
    stage = _default_stages()[3]
    script = _remote_script(
        [stage],
        docker_image=DEFAULT_DOCKER_IMAGE,
        fp8_recipe="tensorwise",
        remote_prefixes={stage.index: "data/seq_8192/train"},
        graph_capacities={stage.index: _capacity(stage)},
        initial_checkpoint_root="/data/cppmega_curriculum_checkpoints/initial",
    )
    script_path = tmp_path / "curriculum.sh"
    script_path.write_text(script, encoding="utf-8")

    assert "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS" in script
    assert 'export CPPMEGA_DSA_PATCH_ENABLED="1"' in script
    assert "--enable-dsa-patch" in script
    assert "if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1'" in script
    assert "export CPPMEGA_DSA_GRAPH_AUX_ENABLED=1" in script
    assert "export CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001" in script
    assert "GRAPH_PRIOR_RECEIPT=\"/data/cppmega_h200_results/stage_${STAGE_IDX}_graph_prior.json\"" in script
    assert "reason=dsa_selector_gate" in script
    assert "apply_graph_route_attention_bias_patch()" in script
    assert "--micro-batch-size ${MBS}" in script
    assert "--global-batch-size ${BS}" in script
    assert "--eval-interval ${EVAL_INTERVAL}" in script
    assert "--save-interval ${TARGET_ITERS}" in script
    assert "--no-check-for-nan-in-loss-and-grad" not in script
    assert "write_graph_capacity_receipt" in script
    assert "write_training_loss_receipt" in script
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
            enable_dsa_patch=False,
        )
