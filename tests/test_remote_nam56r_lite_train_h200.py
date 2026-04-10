from pathlib import Path


def test_remote_nam56r_lite_train_uses_cppmega_mamba_builder_override():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_train_h200_nam56r_lite.sh").read_text()
    assert "from cppmega.megatron.mamba_builder import cppmega_mamba_builder" in script
    assert 'from cppmega.megatron.nam56r_lite_spec import build_default_hybrid_layer_pattern' in script
    assert "build_cppmega_nam56r_lite_stack_spec" in script


def test_remote_nam56r_lite_train_exports_custom_feature_env():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_train_h200_nam56r_lite.sh").read_text()
    assert 'export CPPMEGA_NGRAM_HASH_ENABLED=1' in script
    assert 'export CPPMEGA_STRUCTURE_ENABLED=1' in script
    assert 'export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"' in script


def test_remote_nam56r_lite_train_enables_native_mla_mtp_moe():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_train_h200_nam56r_lite.sh").read_text()
    assert "build_nam56r_megatron_native_args" in script
    assert "enable_mla=True" in script
    assert "enable_mtp=True" in script
    assert "enable_moe=True" in script


def test_remote_nam56r_lite_train_uses_mla_without_gqa_flags():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_train_h200_nam56r_lite.sh").read_text()
    assert "--multi-latent-attention" not in script
    assert "--group-query-attention" not in script
    assert "--num-query-groups" not in script
