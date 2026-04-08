from pathlib import Path


def test_remote_structure_smoke_patches_language_model_embedding():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure.sh").read_text()
    assert "patch_language_model_embedding_for_structure" in script
    assert "language_model_embedding.py" in script


def test_remote_structure_smoke_exports_env_configuration():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure.sh").read_text()
    assert 'export CPPMEGA_STRUCTURE_ENABLED=1' in script
    assert 'export CPPMEGA_STRUCTURE_COMPONENTS="core"' in script


def test_remote_structure_smoke_saves_and_loads_checkpoints():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure.sh").read_text()
    assert '--save "${REMOTE_CKPT_DIR}" \\' in script
    assert '--load "${REMOTE_CKPT_DIR}" \\' in script
