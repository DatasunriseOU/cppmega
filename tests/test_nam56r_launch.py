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


def test_build_nam56r_lite_main_pattern_dsa_symbol():
    """With use_dsa_symbol=True, ALL attention layers become 'D' (PR #3553)."""
    pattern = build_nam56r_lite_main_pattern(
        pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1, use_dsa_symbol=True,
    )
    # No '*' should appear when using DSA symbol for all attention layers.
    assert "*" not in pattern
    # 13 A-layers in the main pattern + 1 MTP attention = 14 D symbols.
    assert pattern.count("D") == 14
    assert pattern.count("M") == 17
    assert pattern.count("E") == 22
    assert pattern.endswith("/D-")


def test_build_nam56r_lite_main_pattern_no_dsa_symbol_unchanged():
    """Without use_dsa_symbol, behavior matches the original (all '*')."""
    pattern = build_nam56r_lite_main_pattern(
        pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1, use_dsa_symbol=False,
    )
    assert "D" not in pattern
    assert pattern.count("*") == 14
    assert pattern.count("M") == 17
    assert pattern.count("E") == 22


def test_mla_adapters_accept_pp_layer_offset_kwarg():
    mla_shared = (
        Path(__file__).resolve().parents[1] / "cppmega" / "megatron" / "mla_shared.py"
    ).read_text()

    assert "class CppMegaMLASelfAttentionAdapter(MLASelfAttention):" in mla_shared
    assert "class CppMegaFusedMLASelfAttentionAdapter(FusedMLASelfAttention):" in mla_shared
    assert "def __init__(self, *args, pp_layer_offset=None, **kwargs):" in mla_shared
    assert "def forward(self, *args, rotary_pos_emb=None, **kwargs):" in mla_shared
    assert "return super().forward(*args, rotary_pos_emb=None, **kwargs)" in mla_shared
    assert "mlp_bda=IdentityFuncOp" in mla_shared
