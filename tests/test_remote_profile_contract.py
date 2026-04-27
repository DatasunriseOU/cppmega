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
