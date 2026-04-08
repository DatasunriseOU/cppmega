from pathlib import Path


def test_remote_dsa_smoke_enables_native_megatron_dsa_flags():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "pretrain_gpt.py" in script
    assert "--multi-latent-attention \\" in script
    assert "--experimental-attention-variant dsa \\" in script
    assert "--dsa-indexer-topk 16 \\" in script
    assert "--dsa-indexer-loss-coeff 0.0 \\" in script


def test_remote_dsa_smoke_keeps_cp_at_one():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "--context-parallel-size 1 \\" in script


def test_remote_dsa_smoke_patches_megatron_config_for_rope_fusion():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200_dsa.sh").read_text()
    assert "kw_args['apply_rope_fusion'] = False" in script
    assert "failed to find Megatron config insertion point for DSA rope fusion override" in script
