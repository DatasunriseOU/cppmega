from cppmega.recipes.nam56r_megatron import (
    build_nam56r_reference_plan,
    build_nam56r_feature_plan,
    count_layer_types,
    parse_nem_pattern,
    translate_nanochat_pattern_to_megatron,
)


def test_parse_nem_pattern_tiles_like_nanochat():
    result = parse_nem_pattern("AE", depth=6)
    assert result == ["A", "E", "A", "E", "A", "E"]


def test_count_layer_types_for_aemeaemeaemr_depth_52():
    counts = count_layer_types("AEMEAEMEAEMR", depth=52)
    assert counts == {"A": 13, "E": 22, "M": 13, "R": 4}


def test_translation_marks_custom_seams_fail_closed():
    plan = translate_nanochat_pattern_to_megatron(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mtp_depths=1,
    )
    assert plan.translated_pattern.endswith("/*-")
    assert "R" in plan.translated_pattern
    assert plan.requires_custom_mamba3 is True
    assert plan.requires_custom_m2rnn is True
    assert plan.is_fully_native is False
    assert any(issue.symbol == "R" for issue in plan.issues)


def test_reference_plan_matches_grounded_nam56r_story():
    plan = build_nam56r_reference_plan()
    assert plan.source_pattern == "AEMEAEMEAEMR"
    assert plan.requires_custom_mamba3 is True
    assert plan.requires_custom_m2rnn is True


def test_feature_plan_captures_custom_engram_and_ngram_hash_surfaces():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mtp_depths=1,
        engram_enabled=True,
        engram_layers="0,3,6",
        engram_gated=True,
        engram_conv_kernel=4,
        ngram_hash_enabled=True,
        ngram_hash_orders="2,3",
        ngram_hash_heads=8,
        ngram_hash_table_size=500_000,
        ngram_hash_embed_dim=16,
    )
    assert plan.engram is not None
    assert plan.engram.layer_indices == (0, 3, 6)
    assert plan.engram.ngram_orders == (2, 3, 4)
    assert plan.engram.gated is True
    assert plan.engram.conv_kernel == 4
    assert plan.ngram_hash is not None
    assert plan.ngram_hash.orders == (2, 3)
    assert plan.ngram_hash.heads == 8


def test_feature_plan_rejects_engram_layers_outside_depth():
    try:
        build_nam56r_feature_plan(
            pattern="AE",
            depth=4,
            engram_enabled=True,
            engram_layers="0,4",
        )
    except ValueError as exc:
        assert "exceed depth=4" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range Engram layer")


def test_feature_plan_captures_mhc_mod_moda_and_structure_surfaces():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mhc_enabled=True,
        mhc_layers="1,5,9",
        mhc_n_streams=4,
        mod_enabled=True,
        mod_layers="10,11",
        mod_capacity=0.25,
        mod_routing="topk",
        moda_enabled=True,
        structure_enabled=True,
        structure_components="core,ast",
        max_ast_depth=32,
        structure_bottleneck_dim=128,
        relation_bias_enabled=True,
    )
    assert plan.mhc is not None
    assert plan.mhc.layer_indices == (1, 5, 9)
    assert plan.mhc.n_streams == 4
    assert plan.mod is not None
    assert plan.mod.layer_indices == (10, 11)
    assert plan.mod.capacity == 0.25
    assert plan.moda is not None
    assert plan.moda.enabled is True
    assert plan.structure is not None
    assert plan.structure.components == "core,ast"
    assert plan.structure.max_ast_depth == 32
    assert plan.structure.bottleneck_dim == 128
    assert plan.structure.relation_bias_enabled is True


def test_feature_plan_rejects_mhc_layers_outside_depth():
    try:
        build_nam56r_feature_plan(
            pattern="AE",
            depth=4,
            mhc_enabled=True,
            mhc_layers="0,4",
        )
    except ValueError as exc:
        assert "mHC layer indices" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range mHC layer")


def test_feature_plan_rejects_mod_layers_outside_depth():
    try:
        build_nam56r_feature_plan(
            pattern="AE",
            depth=4,
            mod_enabled=True,
            mod_layers="1,4",
        )
    except ValueError as exc:
        assert "MoD layer indices" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range MoD layer")
