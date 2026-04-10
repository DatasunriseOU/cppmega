from pathlib import Path


def test_remote_setup_has_safe_transformer_engine_lane():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_setup_h200.sh").read_text()
    assert 'INSTALL_TE_PYTORCH="${INSTALL_TE_PYTORCH:-0}"' in script
    assert 'python -m pip install --no-deps --force-reinstall "transformer_engine_cu13==${TE_VERSION}" "transformer_engine_torch==${TE_VERSION}"' in script
    assert 'python -m pip install --no-deps "transformer-engine==${TE_VERSION}" onnx onnxscript onnx_ir nvdlfw-inspect ml_dtypes' in script
    assert 'export CPATH="${CUDNN_ROOT}/include:${NCCL_ROOT}/include:${CPATH:-}"' in script


def test_remote_setup_can_upload_te_artifacts_to_gcs():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_setup_h200.sh").read_text()
    assert 'GCS_ARTIFACT_PREFIX="${GCS_ARTIFACT_PREFIX:-}"' in script
    assert 'python -m pip wheel --no-deps --wheel-dir "${ART_DIR}"' in script
    assert '"transformer-engine==${TE_VERSION}" onnx onnxscript onnx_ir nvdlfw-inspect ml_dtypes' in script
    assert 'gcloud storage cp "${ART_DIR}"/* "${GCS_ARTIFACT_PREFIX}/"' in script


def test_remote_setup_uses_base_env_as_runtime_source_of_truth():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_setup_h200.sh").read_text()
    assert '"${REMOTE_BASE_VENV}/bin/python" -m venv --system-site-packages "${REMOTE_VENV}"' in script
    assert 'BASE_TORCH_VERSION="$(${REMOTE_BASE_VENV}/bin/python -c ' in script
    assert 'CLONED_TORCH_VERSION="$(python -c ' in script
    assert 'mkdir -p "${REMOTE_SITE_PACKAGES}/nvidia"' in script
    assert 'ln -s "${entry}" "${REMOTE_SITE_PACKAGES}/nvidia/${name}"' in script


def test_remote_setup_force_reinstalls_author_mamba3_over_base_env_shadow():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_setup_h200.sh").read_text()
    assert 'MAMBA_FORCE_BUILD=TRUE python -m pip install --no-deps --no-build-isolation --force-reinstall --ignore-installed .' in script
    assert 'find_spec("mamba_ssm.modules.mamba3") is not None' in script
