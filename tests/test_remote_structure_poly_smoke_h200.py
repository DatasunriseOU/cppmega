from pathlib import Path


def test_remote_structure_poly_smoke_uses_cppmega_builder_route():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure_poly.sh").read_text()
    assert "from cppmega.megatron.gpt_builder import cppmega_gpt_builder" in script
    assert "return cppmega_gpt_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)" in script


def test_remote_structure_poly_smoke_exports_env_configuration():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure_poly.sh").read_text()
    assert 'export CPPMEGA_STRUCTURE_ENABLED=1' in script
    assert 'export CPPMEGA_STRUCTURE_COMPONENTS="core"' in script


def test_remote_structure_poly_smoke_saves_and_loads_checkpoints():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_structure_poly.sh").read_text()
    assert '--save "${REMOTE_CKPT_DIR}" \\' in script
    assert '--load "${REMOTE_CKPT_DIR}" \\' in script
