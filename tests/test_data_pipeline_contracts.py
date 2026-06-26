from __future__ import annotations

import importlib.util
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

    assert {
        "token_structure_ids",
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_def_use",
        "token_change_mask_pre",
        "token_change_mask_post",
        "token_platform_ids",
    }.issubset(names)


def test_verify_dataset_default_vocab_is_canonical_contract() -> None:
    verify = _load_module(
        "verify_dataset_megacpp",
        "scripts/data/verify_dataset_megacpp.py",
    )

    parser = verify.build_arg_parser()

    assert parser.parse_args([]).vocab_size == 65536


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


def test_prepare_data_dispatcher_runs_fail_closed_gates_before_trainable_verify() -> None:
    script = (_REPO_ROOT / "scripts/data/prepare_data.sh").read_text()

    assert "verify_tokenizer_contract.py" in script
    assert "verify_provenance.py" in script
    assert "verify_side_channel_shapes.py" in script
    assert "--require-full-sidecars" in script
    assert "audit_megacpp_4k.py" in script
    assert "build_dataset_manifest.py" in script


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
