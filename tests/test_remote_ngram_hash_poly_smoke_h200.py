from pathlib import Path


def test_remote_ngram_hash_poly_smoke_uses_cppmega_builder_route():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash_poly.sh").read_text()
    assert "from cppmega.megatron.gpt_builder import cppmega_gpt_builder" in script
    assert "return cppmega_gpt_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)" in script


def test_remote_ngram_hash_poly_smoke_exports_env_configuration():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash_poly.sh").read_text()
    assert 'export CPPMEGA_NGRAM_HASH_ENABLED=1' in script
    assert 'export CPPMEGA_NGRAM_HASH_ORDERS="2,3"' in script
    assert 'export CPPMEGA_NGRAM_HASH_HEADS=4' in script


def test_remote_ngram_hash_poly_smoke_uses_pretrain_gpt_lane():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash_poly.sh").read_text()
    assert "pretrain_gpt.py" in script
    assert "--use-mcore-models \\" in script


def test_remote_ngram_hash_poly_smoke_saves_and_loads_checkpoints():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash_poly.sh").read_text()
    assert '--save "${REMOTE_CKPT_DIR}" \\' in script
    assert '--load "${REMOTE_CKPT_DIR}" \\' in script
    assert '--save-interval 1 \\' in script
