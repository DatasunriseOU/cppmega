from cppmega.recipes.nam56r_megatron import (
    build_nam56r_reference_plan,
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
