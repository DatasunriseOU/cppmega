from pathlib import Path


def test_remote_smoke_defaults_to_cppmega_local_mamba_spec():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert 'CPPMEGA_SPEC_MODULE="${CPPMEGA_SPEC_MODULE:-cppmega.megatron.mamba_local_spec}"' in script
    assert 'CPPMEGA_SPEC_NAME="${CPPMEGA_SPEC_NAME:-cppmega_mamba_stack_spec}"' in script


def test_remote_smoke_uses_parameterized_spec_vars():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert '--spec "${CPPMEGA_SPEC_MODULE}" "${CPPMEGA_SPEC_NAME}" \\' in script
    assert "CPPMEGA_SPEC_MODULE='" in script
    assert "CPPMEGA_SPEC_NAME='" in script


def test_remote_smoke_disables_non_te_incompatible_fusions():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert "--no-gradient-accumulation-fusion \\" in script
    assert "--no-persist-layer-norm \\" in script


def test_remote_smoke_has_explicit_eval_contract():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert "--eval-interval 50000000 \\" in script
    assert "--eval-iters 0 \\" in script


def test_remote_smoke_disables_masked_softmax_fusion():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert "--no-masked-softmax-fusion \\" in script


def test_remote_smoke_uses_per_spec_run_id_for_checkpoints_and_logs():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert 'CPPMEGA_RUN_ID="${CPPMEGA_RUN_ID:-${CPPMEGA_SPEC_NAME}}"' in script
    assert 'REMOTE_LOG="${REMOTE_LOG:-${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}.log}"' in script
    assert 'REMOTE_CKPT_DIR="${REMOTE_ROOT}/cppmega/${CPPMEGA_RUN_ID}_ckpt"' in script


def test_remote_smoke_can_target_m2rnn_spec_via_env():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "remote_smoke_h200.sh").read_text()
    assert 'CPPMEGA_SPEC_MODULE="${CPPMEGA_SPEC_MODULE:-cppmega.megatron.mamba_local_spec}"' in script
    assert 'CPPMEGA_SPEC_NAME="${CPPMEGA_SPEC_NAME:-cppmega_mamba_stack_spec}"' in script
