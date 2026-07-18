from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cppmega.megatron.graph_objective_loss import validate_runtime_graph_contract
from cppmega.megatron.graph_recipe import (
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
)
from scripts import h200_megatron_preflight as preflight
from scripts.h200_megatron_preflight import (
    _profile_environment,
    _write_wrappers,
    build_megatron_command,
    write_training_loss_receipt,
)


def _included_graph_contract() -> dict[str, object]:
    return {
        **stage1_graph_recipe_payload(),
        "recipe": stage1_graph_recipe_binding(),
        "eligible_samples": 1,
        "positive_edges": 1,
        "included_in_total_loss": True,
    }


def test_pytest_bootstrap_finds_selected_real_megatron_without_pythonpath():
    repo_root = Path(__file__).resolve().parents[1]

    script = """
import conftest
import megatron
import megatron.core
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.moe import moe_utils
from pathlib import Path

expected = Path(conftest.MEGATRON_SOURCE_ROOT).resolve() / "megatron"
for module in (megatron.core, dsa, moe_utils):
    assert Path(module.__file__).resolve().is_relative_to(expected), module.__file__
assert megatron.__spec__ is not None
assert not getattr(megatron, "__cppmega_stub__", False)
"""
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "MEGATRON_LM_REPO", "MEGATRON_ROOT"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pytest_bootstrap_fails_loudly_for_invalid_explicit_megatron_root(tmp_path):
    missing_root = tmp_path / "missing-megatron"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("MEGATRON_ROOT", None)
    environment["MEGATRON_LM_REPO"] = str(missing_root)
    result = subprocess.run(
        [sys.executable, "-c", "import conftest"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "MEGATRON_LM_REPO" in result.stderr
    assert "real Megatron-LM" in result.stderr


def test_portable_profile_requires_only_allowlisted_test_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conftest

    monkeypatch.setenv("CPPMEGA_TEST_PROFILE", "portable-data")
    assert conftest._portable_profile_requested()


def test_portable_profile_accepts_node_id_and_pytest_option_value(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["CPPMEGA_TEST_PROFILE"] = "portable-data"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_data_pipeline_contracts.py::test_verify_dataset_default_vocab_is_canonical_contract",
            "--basetemp",
            str(tmp_path / "pytest-base"),
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_megatron_root_without_receipt_requires_exact_commit(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = tmp_path / "megatron"
    (source / "megatron" / "core").mkdir(parents=True)
    (source / "megatron" / "core" / "__init__.py").write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("CPPMEGA_MEGATRON_COMMIT", None)
    environment["MEGATRON_LM_REPO"] = str(source)
    result = subprocess.run(
        [sys.executable, "-c", "import conftest"],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "CPPMEGA_MEGATRON_COMMIT" in result.stderr


def test_manifest_source_validation_rejects_dirty_or_drifted_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import conftest

    source = tmp_path / "megatron"
    (source / "megatron" / "core").mkdir(parents=True)
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "cppmega-test"],
        check=True,
    )
    (source / "megatron" / "core" / "__init__.py").write_text("\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "megatron_commit": commit,
        "source_dirty": False,
        "megatron_root": str(source),
    }
    manifest_path = tmp_path / "cppmega-environment.json"

    (source / "dirty.txt").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="changed after receipt"):
        conftest._validate_manifest_source(manifest_path, manifest, source)

    (source / "dirty.txt").unlink()
    manifest["megatron_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="differs from receipt"):
        conftest._validate_manifest_source(manifest_path, manifest, source)


def test_production_preflight_uses_derived_capacity_and_active_dsa_auxiliary():
    environment = _profile_environment(
        sequence_length=8192,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=713,
        graph_max_chunks=419,
        enable_dsa_patch=True,
    )

    assert environment["CPPMEGA_GRAPH_MAX_EDGES"] == "713"
    assert environment["CPPMEGA_GRAPH_MAX_CHUNKS"] == "419"
    assert environment["CPPMEGA_DSA_PATCH_ENABLED"] == "1"
    assert environment["CPPMEGA_DSA_GRAPH_AUX_ENABLED"] == "1"
    assert environment["CPPMEGA_DSA_GRAPH_AUX_WEIGHT"] == "1"
    assert environment["CPPMEGA_DSA_INDEXER_LOSS_COEFF"] == "0.001"
    assert environment["CPPMEGA_DSA_SKIP_INDEXER_LOSS"] == "0"
    assert environment["CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS"] == "0"
    assert environment["CPPMEGA_SPEC_MODULE"] == "cppmega.megatron.nam56r_full_spec"
    assert environment["CPPMEGA_SPEC_FUNCTION"] == "build_cppmega_nam56r_full_stack_spec"


def test_h200_command_selects_real_dsa_capable_spec_and_contract():
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=8,
        graph_max_chunks=4,
        enable_dsa_patch=True,
    )
    command = build_megatron_command(
        wrapper=Path("/tmp/pretrain_mamba.py"),
        data_prefix=Path("/tmp/data"),
        tokenizer_model=Path("/tmp/tokenizer"),
        checkpoint_root=Path("/tmp/checkpoint"),
        sequence_length=1024,
        micro_batch_size=1,
        train_iters=1,
        environment=environment,
        load_checkpoint=False,
    )

    spec_index = command.index("--spec")
    assert command[spec_index + 1 : spec_index + 3] == [
        "cppmega.megatron.nam56r_full_spec",
        "build_cppmega_nam56r_full_stack_spec",
    ]


def test_selected_spec_contains_real_sibling_dsattention_and_indexer(monkeypatch):
    from cppmega.megatron.dsa_local_spec import get_cppmega_dsa_layer_spec
    from cppmega.megatron.nam56r_full_spec import (
        CppMegaSelectiveAttentionLayer,
        build_cppmega_nam56r_full_stack_spec,
    )
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexer,
        DSAttention,
    )

    class Backend:
        @staticmethod
        def column_parallel_layer_norm_linear():
            return object

        @staticmethod
        def column_parallel_linear():
            return object

        @staticmethod
        def linear():
            return object

        @staticmethod
        def layer_norm(**_kwargs):
            return object

        @staticmethod
        def row_parallel_linear():
            return object

    dsa_spec = get_cppmega_dsa_layer_spec(
        SimpleNamespace(
            multi_latent_attention=True,
            qk_l2_norm=False,
            qk_layernorm=False,
            normalization="RMSNorm",
            experimental_attention_variant="dsa",
        ),
        backend=Backend(),
    )
    core_attention = dsa_spec.submodules.core_attention

    assert core_attention.module is DSAttention
    assert core_attention.submodules.indexer.module is DSAIndexer

    monkeypatch.setenv("CPPMEGA_NEM_PATTERN", "AF")
    monkeypatch.setenv("CPPMEGA_LAYER_DEPTH", "48")
    monkeypatch.setenv("CPPMEGA_DSA_A_LAYER_RANKS", "1,2,3")
    stack_spec = build_cppmega_nam56r_full_stack_spec(
        SimpleNamespace(
            transformer_impl="transformer_engine",
            tensor_model_parallel_size=1,
        )
    )
    assert stack_spec.submodules.attention_layer.module is CppMegaSelectiveAttentionLayer


def test_included_graph_contract_rejects_aux_off_preflight_environment():
    environment = _profile_environment(
        sequence_length=1024,
        micro_batch_size=1,
        fp8_recipe="off",
        graph_max_edges=8,
        graph_max_chunks=4,
        enable_dsa_patch=True,
    )
    environment["CPPMEGA_DSA_GRAPH_AUX_ENABLED"] = "0"
    with pytest.raises(ValueError, match="included_in_total_loss.*auxiliary"):
        validate_runtime_graph_contract(
            _included_graph_contract(),
            environment=environment,
            require_included_auxiliary=True,
        )


def test_preflight_validates_graph_contract_before_model_wrapper() -> None:
    source = inspect.getsource(preflight.main)
    validation = "validate_runtime_graph_contract("

    assert validation in source
    assert source.index(validation) < source.index("_write_wrappers(")


def test_dsa_graph_receipt_requires_actual_module_loss_coefficient_and_gradients():
    text = (
        'CPPMEGA_DSA_GRAPH_OBJECTIVE {"layer_number": 3, '
        '"actual_dsa_module": '
        '"megatron.core.transformer.experimental_attention_variant.dsa.DSAttention", '
        '"effective_coefficient": 0.001, "graph_loss": 0.25}\n'
        'CPPMEGA_DSA_INDEXER_GRAD {"layer_number": 3, '
        '"actual_indexer_module": '
        '"megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer", '
        '"grad_norm": 0.5, "parameter_grad_norms": {"linear_wk.weight": 0.5}}\n'
    )

    evidence = preflight._dsa_graph_gradient_evidence(
        text,
        expected_coefficient=0.001,
    )

    assert evidence["actual_dsa_modules"] == [
        "megatron.core.transformer.experimental_attention_variant.dsa.DSAttention"
    ]
    assert evidence["effective_coefficient"] == pytest.approx(0.001)
    assert evidence["per_indexer"][0]["grad_norm"] == pytest.approx(0.5)


def test_graph_prior_receipt_requires_dsa_indexer_consumer():
    with pytest.raises(RuntimeError, match="consumer.*dsa_indexer"):
        preflight._validate_graph_prior_receipt(
            {
                "status": "verified",
                "consumer": "dense_attention",
                "prior": {"nonzero": 1},
            }
        )


def test_preflight_wrapper_only_applies_dsa_patch_after_explicit_opt_in(tmp_path):
    wrapper = _write_wrappers(tmp_path)

    source = wrapper.read_text(encoding="utf-8")
    gate = "os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1'"
    assert gate in source
    assert source.index(gate) < source.index("apply_dsa_indexer_fused_patch()")


def test_training_loss_gate_writes_hash_bound_finite_receipt(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "iteration 9/ 9 | lm loss: 6.5 | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 0 |\n",
        encoding="utf-8",
    )
    output = tmp_path / "loss-receipt.json"

    receipt = write_training_loss_receipt(
        log,
        expected_iteration=9,
        output=output,
    )

    assert receipt["schema"] == "cppmega_h200_training_loss_gate_v1"
    assert receipt["status"] == "verified"
    assert receipt["evidence"]["lm_loss"] == 6.5
    assert len(receipt["log_sha256"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == receipt


def test_training_loss_gate_rejects_nan_or_skipped_stage(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text(
        "iteration 2/ 2 | lm loss: nan | grad norm: 0.25 | "
        "number of skipped iterations: 0 | number of nan iterations: 1 |\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="finite positive LM loss"):
        write_training_loss_receipt(
            log,
            expected_iteration=2,
            output=tmp_path / "loss-receipt.json",
        )
