import json

from scripts.nebius_h200_megatron_cpp_world_curriculum import (
    _default_stages,
    _make_curriculum_manifest,
    _parse_stage,
)
from pathlib import Path


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

    _make_curriculum_manifest([stage], out)

    payload = json.loads(out.read_text())
    assert payload["stages"][0]["global_batch"] == 16
    assert payload["stages"][0]["micro_batch"] == 4


def test_curriculum_container_installs_graph_route_attention_patches():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "nebius_h200_megatron_cpp_world_curriculum.py"
    ).read_text()

    assert "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS" in source
    assert "apply_dsa_indexer_fused_patch()" in source
    assert "apply_graph_route_attention_bias_patch()" in source
    assert "--micro-batch-size ${{MBS}}" in source
    assert "--global-batch-size ${{BS}}" in source
