from pathlib import Path

from cppmega.recipes.nam56r_launch import build_nam56r_lite_main_pattern, get_custom_layer_indices


def test_build_nam56r_lite_main_pattern_maps_r_to_mamba_and_adds_mtp_suffix():
    pattern = build_nam56r_lite_main_pattern(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)
    assert pattern.endswith("/*-")
    assert "R" not in pattern
    assert pattern.count("M") == 17
    assert pattern.count("E") == 22
    assert pattern.count("*") == 14


def test_get_custom_layer_indices_finds_r_layers_in_nam56r_reference_pattern():
    indices = get_custom_layer_indices(pattern="AEMEAEMEAEMR", depth=52, custom_symbols=("R",))
    assert indices == (12, 24, 36, 48)


def test_mla_adapters_accept_pp_layer_offset_kwarg():
    spec = (
        Path(__file__).resolve().parents[1] / "cppmega" / "megatron" / "nam56r_lite_spec.py"
    ).read_text()

    assert "class _CppMegaMLASelfAttentionAdapter(MLASelfAttention):" in spec
    assert "class _CppMegaFusedMLASelfAttentionAdapter(FusedMLASelfAttention):" in spec
    assert "def __init__(self, *args, pp_layer_offset=None, **kwargs):" in spec
    assert "def forward(self, *args, rotary_pos_emb=None, **kwargs):" in spec
    assert "return super().forward(*args, rotary_pos_emb=None, **kwargs)" in spec
    assert "mlp_bda=IdentityFuncOp" in spec
