from pathlib import Path


def test_remote_launchers_do_not_sed_mtp_num_layers():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders = []
    for script in scripts_dir.glob("remote_*.sh"):
        text = script.read_text()
        if 'sed "s/--mtp-num-layers 1/--mtp-num-layers' in text:
            offenders.append(script.name)

    assert offenders == []


def test_remote_launchers_do_not_postprocess_native_args_with_sed():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    offenders = []
    for script in scripts_dir.glob("remote_*.sh"):
        text = script.read_text()
        if "NATIVE_ARGS=$(echo" in text and "| sed " in text:
            offenders.append(script.name)

    assert offenders == []


def test_remote_mtp2_launchers_pass_predictor_depth_to_helper():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    checked = []
    for script in scripts_dir.glob("remote_*.sh"):
        text = script.read_text()
        if "MTP_DEPTHS" in text and "build_nam56r_megatron_native_args(" in text:
            checked.append(script.name)
            assert "mtp_num_predictors=mtp_depths" in text or "mtp_num_predictors=max(mtp_depths" in text

    assert checked


def test_p092_and_production_use_stable_attention_only_cuda_graphs():
    repo_root = Path(__file__).resolve().parents[1]
    gate = (repo_root / "scripts/modal_mamba3_wave32_h200_20step_gate.py").read_text()
    production = (repo_root / "scripts/remote_production_h200_nam56r_v1.sh").read_text()
    smoke = (repo_root / "scripts/remote_smoke_h200_dsa_9_4_m.sh").read_text()

    assert '"--cuda-graph-scope",\n            "attn",\n            "--cuda-graph-warmup-steps"' in gate
    assert "--cuda-graph-scope attn --cuda-graph-warmup-steps 3" in production
    assert 'CG_FLAGS="--cuda-graph-impl transformer_engine --cuda-graph-scope attn"' in smoke
    assert "--cuda-graph-scope attn mamba" not in production + smoke
