from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_format_default_side_channels_preserve_graph_training_labels() -> None:
    prepare_format = _load_module(
        "prepare_format_megacpp",
        "scripts/data/prepare_format_megacpp.py",
    )

    names = {name for name, _dtype in prepare_format.DEFAULT_SIDE_CHANNELS}

    converter = _load_module(
        "data_prep_parquet_to_megatron_contract",
        "scripts/data_prep_parquet_to_megatron.py",
    )
    assert names == {
        name for name, _dtype in converter.DEFAULT_CPPMEGA_TOKEN_SIDE_CHANNELS
    }
    assert {"loss_mask", "doc_ids", "token_domain_ids", "token_role_ids"} <= names


def test_verify_dataset_default_vocab_is_canonical_contract() -> None:
    verify = _load_module(
        "verify_dataset_megacpp",
        "scripts/data/verify_dataset_megacpp.py",
    )

    parser = verify.build_arg_parser()

    assert parser.parse_args([]).vocab_size == 65536
    assert parser.parse_args(["--raw-only"]).raw_only is True


def _write_minimal_mmididx(idx_path: Path, dtype_code: int, num_sequences: int = 1) -> None:
    with open(idx_path, "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", num_sequences))
        f.write(struct.pack("<Q", num_sequences + 1))
        np.array([1] * num_sequences, dtype=np.int32).tofile(f)
        np.array([0] * num_sequences, dtype=np.int64).tofile(f)
        np.arange(num_sequences + 1, dtype=np.int64).tofile(f)


def test_verify_dataset_raw_fallback_checks_full_token_range(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    megatron_dir = data_root / "megatron"
    megatron_dir.mkdir(parents=True)
    prefix = megatron_dir / "bad_train"
    np.array([65535], dtype=np.uint16).tofile(prefix.with_suffix(".bin"))
    _write_minimal_mmididx(prefix.with_suffix(".idx"), dtype_code=8)

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts/data/verify_dataset_megacpp.py"),
            "--data-root",
            str(data_root),
            "--dataset-name",
            "bad",
            "--splits",
            "train",
            "--vocab-size",
            "65535",
            "--raw-only",
        ],
        cwd=str(_REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "max token id 65535 >= vocab_size 65535" in (
        result.stdout + result.stderr
    )


def test_manifest_requires_tokenizer_fingerprints_when_building_trust_metadata(
    tmp_path: Path,
) -> None:
    manifest = _load_module(
        "build_dataset_manifest",
        "scripts/data/build_dataset_manifest.py",
    )

    with pytest.raises(manifest.ManifestError, match="contract.*required"):
        manifest.build(
            dataset_dir=tmp_path,
            seq_len=4096,
            contract=None,
            tokenizer=None,
            repo_sample=20,
            batch_size=4096,
        )


def test_package_data_includes_cuda_headers_needed_by_extension_build() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["cppmega"]

    assert "megatron/cuda_ext/*.hpp" in package_data
    assert "data/domain_schema_v1.json" in package_data
    assert "tokenizer/tokenizer.json" in package_data
    assert "tokenizer/tokenizer_contract_v1.json" in package_data


def test_package_discovery_does_not_publish_sibling_mlx_namespace() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    assert include == ["cppmega", "cppmega.*"]
    assert "cppmega*" not in include


def test_prepare_data_dispatcher_runs_fail_closed_gates_before_trainable_verify() -> None:
    script = (_REPO_ROOT / "scripts/data/prepare_data.sh").read_text()

    assert "verify_tokenizer_contract.py" in script
    assert "verify_provenance.py" in script
    assert "verify_side_channel_shapes.py" in script
    assert "--require-full-sidecars" in script
    assert "audit_megacpp_4k.py" in script
    assert "build_dataset_manifest.py" in script


def test_tokenizer_contract_verifier_rejects_unpaired_domain_delimiter() -> None:
    verify = _load_module(
        "verify_tokenizer_contract",
        "scripts/data/verify_tokenizer_contract.py",
    )
    contract = {
        "reserved_role_assignments": {
            "CPP_CODE_START": 53,
        },
    }
    id_to_token = {53: "<RESERVED_53>"}

    with pytest.raises(verify.ContractError, match="CPP_CODE_END"):
        verify.check_domain_delimiter_roles(contract, id_to_token)


def test_tokenizer_contract_verifier_derives_all_case5_delimiters_from_schema() -> None:
    verify = _load_module(
        "verify_tokenizer_contract_case5_schema",
        "scripts/data/verify_tokenizer_contract.py",
    )

    pairs = verify.load_domain_delimiter_role_pairs(
        _REPO_ROOT / "data/domain_schema_v1.json"
    )

    assert len(pairs) == 29
    assert ("CONFIGURE_START", "CONFIGURE_END") in pairs
    assert ("KSH_START", "KSH_END") in pairs
    assert ("PYTHON_START", "PYTHON_END") in pairs
    assert ("SQL_START", "SQL_END") in pairs
    assert ("LINKER_DIAGNOSTIC_START", "LINKER_DIAGNOSTIC_END") in pairs
    assert ("SANITIZER_OUTPUT_START", "SANITIZER_OUTPUT_END") in pairs


def test_self_hosted_ci_verifies_only_explicit_checked_out_tokenizer_contracts() -> None:
    workflow = (_REPO_ROOT / ".github/workflows/ci-self-hosted.yml").read_text()
    invocation = workflow.split("scripts/data/verify_tokenizer_contract.py", 1)[1]

    assert '--contract "${GITHUB_WORKSPACE}/data/tokenizer_v2/tokenizer_contract_v1.json"' in invocation
    assert '--tokenizer "${GITHUB_WORKSPACE}/data/tokenizer_v2/tokenizer.json"' in invocation
    assert '--domain-schema "${GITHUB_WORKSPACE}/data/domain_schema_v1.json"' in invocation
    assert "--root /Volumes/external/sources" not in invocation


def test_tokenizer_contract_verifier_rejects_reserved_whitespace_slot(
    tmp_path: Path,
) -> None:
    verify = _load_module(
        "verify_tokenizer_contract_special_strings",
        "scripts/data/verify_tokenizer_contract.py",
    )
    tokenizer = tmp_path / "tokenizer.json"
    vocab = {f"token-{idx}": idx for idx in range(46)}
    vocab["<RESERVED_46>"] = 46
    tokenizer.write_text(
        json.dumps(
            {
                "model": {"vocab": vocab},
                "added_tokens": [{"id": 46, "content": "<RESERVED_46>"}],
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "vocab_size": 47,
        "special_tokens": {"SPACE": 46},
        "reserved_role_assignments": {},
    }

    with pytest.raises(verify.ContractError, match="expected '<SPACE>'"):
        verify.check_contract_vs_tokenizer(contract, tokenizer)


def test_side_channel_checker_has_full_sidecar_gate() -> None:
    script = (_REPO_ROOT / "scripts/data/verify_side_channel_shapes.py").read_text()

    assert "--require-full-sidecars" in script
    assert "--allow-partial-sidecars" in script
    assert "full sidecar dataset missing required token-aligned" in script


def test_tp_sp_angle_gather_test_mirrors_production_default_backward_path() -> None:
    test_source = (
        _REPO_ROOT / "tests/test_cppmega_mamba3_tp_mixer.py"
    ).read_text()
    production_source = (
        _REPO_ROOT / "cppmega/megatron/cppmega_mamba3_tp_mixer.py"
    ).read_text()

    assert "tensor_parallel_output_grad=False" not in test_source
    assert "tensor_parallel_output_grad=False" not in production_source
