from cppmega.recipes.megatron_args import build_megatron_args_bundle
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan


def test_megatron_args_bundle_emits_native_mla_mtp_fim_moe_dsa_flags():
    plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)
    bundle = build_megatron_args_bundle(
        plan=plan,
        use_mla=True,
        use_mtp=True,
        mtp_num_predictors=1,
        use_fim=True,
        use_moe=True,
        use_dsa=True,
    )

    assert "--multi-latent-attention" in bundle.args
    assert "--multi-token-prediction" in bundle.args
    assert "--fim-rate" in bundle.args
    assert "--num-experts" in bundle.args
    assert "--experimental-attention-variant" in bundle.args
    assert bundle.args[bundle.args.index("--experimental-attention-variant") + 1] == "dsa"


def test_megatron_args_bundle_uses_hybrid_mtp_contract_without_gpt_toggle():
    plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52, mtp_depths=1)
    bundle = build_megatron_args_bundle(
        plan=plan,
        use_mtp=True,
        mtp_mode="hybrid",
        mtp_num_predictors=1,
    )

    assert "--mtp-num-layers" in bundle.args
    assert "--multi-token-prediction" not in bundle.args


def test_megatron_args_bundle_marks_custom_features_as_notes_only():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        engram_enabled=True,
        engram_layers="0,3",
        ngram_hash_enabled=True,
        mhc_enabled=True,
        mhc_layers="1,5",
        mod_enabled=True,
        mod_layers="8,9",
        moda_enabled=True,
        structure_enabled=True,
    )
    bundle = build_megatron_args_bundle(plan=plan)

    assert any("Engram remains custom" in note for note in bundle.custom_notes)
    assert any("ngram hash enrichment remains custom" in note for note in bundle.custom_notes)
    assert any("mHC remains custom" in note for note in bundle.custom_notes)
    assert any("MoD remains custom" in note for note in bundle.custom_notes)
    assert any("MoDA remains custom" in note for note in bundle.custom_notes)
    assert any("structure-aware enrichment remains custom" in note for note in bundle.custom_notes)
