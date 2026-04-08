from cppmega.recipes.megatron_args import build_megatron_args_bundle
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan


def test_megatron_args_bundle_renders_shell_fragment():
    plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52)
    bundle = build_megatron_args_bundle(plan=plan, use_mla=True, use_dsa=True)

    fragment = bundle.to_shell_fragment()

    assert "--multi-latent-attention" in fragment
    assert "--experimental-attention-variant dsa" in fragment
