from pathlib import Path


def test_remote_dsa_smoke_enables_native_megatron_dsa_flags():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "pretrain_gpt.py" in script
    assert "--no-gradient-accumulation-fusion \\" in script
    assert "from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args" in script
    assert "from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan" in script
    assert "${NATIVE_ARGS} \\" in script


def test_remote_dsa_smoke_keeps_cp_at_one():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "--context-parallel-size 1 \\" in script


def test_remote_dsa_smoke_patches_megatron_config_for_rope_fusion():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "kw_args['apply_rope_fusion'] = False" in script
    assert "failed to find Megatron config insertion point for DSA rope fusion override" in script


def test_remote_dsa_smoke_patches_missing_megatron_dsa_router_branch():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert 'if config.experimental_attention_variant == "dsa":\\n' in script
    assert 'return get_dsa_module_spec_for_backend(config=config, backend=backend)\\n' in script
    assert "failed to find Megatron experimental attention variant routing block for DSA patch" in script


def test_remote_dsa_smoke_patches_missing_dsa_metainfo_guard():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert 'if attention.metainfo.get("fuse_input_layernorm", False)\\n' in script
    assert "failed to find Megatron DSA input-layernorm metainfo access for smoke patch" in script
