from pathlib import Path


def test_remote_ngram_hash_smoke_patches_language_model_embedding():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash.sh").read_text()
    assert "from cppmega.remote.ngram_patch import patch_language_model_embedding_for_ngram_hash" in script
    assert "language_model_embedding.py" in script


def test_remote_ngram_hash_smoke_exports_env_configuration():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash.sh").read_text()
    assert 'export CPPMEGA_NGRAM_HASH_ENABLED=1' in script
    assert 'export CPPMEGA_NGRAM_HASH_ORDERS="2,3"' in script
    assert 'export CPPMEGA_NGRAM_HASH_HEADS=4' in script


def test_remote_ngram_hash_smoke_uses_pretrain_gpt_lane():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash.sh").read_text()
    assert "pretrain_gpt.py" in script
    assert "--use-mcore-models \\" in script


def test_remote_ngram_hash_smoke_saves_and_loads_checkpoints():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_ngram_hash.sh").read_text()
    assert '--save "${REMOTE_CKPT_DIR}" \\' in script
    assert '--load "${REMOTE_CKPT_DIR}" \\' in script
    assert '--save-interval 1 \\' in script
