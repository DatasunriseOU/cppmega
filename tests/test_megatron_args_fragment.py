import sys

from cppmega.recipes.megatron_args import build_megatron_args_bundle
from cppmega.recipes.nam56r_launch import build_nam56r_megatron_native_args
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan
from cppmega.recipes import nam56r_launch


def test_megatron_args_bundle_renders_shell_fragment():
    plan = build_nam56r_feature_plan(pattern="AEMEAEMEAEMR", depth=52)
    bundle = build_megatron_args_bundle(plan=plan, use_mla=True, use_dsa=True)

    fragment = bundle.to_shell_fragment()

    assert "--multi-latent-attention" in fragment
    assert "--experimental-attention-variant dsa" in fragment


def test_nam56r_launch_helper_emits_native_fragment_only_for_builtins():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        engram_enabled=True,
        engram_layers="0,3",
        mhc_enabled=True,
        mhc_layers="1,5",
    )
    bundle = build_nam56r_megatron_native_args(
        plan=plan,
        enable_mla=True,
        enable_mtp=True,
        enable_fim=True,
        enable_moe=True,
        enable_dsa=True,
    )

    fragment = bundle.to_shell_fragment()
    assert "--multi-latent-attention" in fragment
    assert "--multi-token-prediction" in fragment
    assert "--fim-rate" in fragment
    assert "--num-experts" in fragment
    assert "--experimental-attention-variant dsa" in fragment
    assert any("Engram remains custom" in note for note in bundle.custom_notes)
    assert any("mHC remains custom" in note for note in bundle.custom_notes)


def test_nam56r_launch_helper_uses_hybrid_mtp_fragment_for_mamba_lane():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mtp_depths=1,
    )
    bundle = build_nam56r_megatron_native_args(
        plan=plan,
        enable_mtp=True,
        mtp_mode="hybrid",
    )

    fragment = bundle.to_shell_fragment()
    assert "--mtp-num-layers 1" in fragment
    assert "--multi-token-prediction" not in fragment


def test_nam56r_launch_helper_uses_requested_mtp_predictor_count():
    plan = build_nam56r_feature_plan(
        pattern="AEMEAEMEAEMR",
        depth=52,
        mtp_depths=2,
    )
    bundle = build_nam56r_megatron_native_args(
        plan=plan,
        enable_mtp=True,
        mtp_mode="hybrid",
        mtp_num_predictors=2,
    )

    assert "--mtp-num-layers 2" in bundle.to_shell_fragment()


def test_nam56r_launch_cli_emits_native_fragment(capsys, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["nam56r_launch", "--enable-mla", "--enable-dsa", "--enable-mtp", "--mtp-depths", "2"],
    )
    exit_code = nam56r_launch.main()
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "--multi-latent-attention" in out
    assert "--experimental-attention-variant dsa" in out
    assert "--mtp-num-layers 2" in out
