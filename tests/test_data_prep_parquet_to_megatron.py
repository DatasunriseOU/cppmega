from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_converter_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data_prep_parquet_to_megatron.py"
    )
    spec = importlib.util.spec_from_file_location(
        "data_prep_parquet_to_megatron",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_megatron_dtype_codes_match_mmididx_enum() -> None:
    converter = _load_converter_module()

    assert converter._megatron_dtype_code(np.uint8) == 1
    assert converter._megatron_dtype_code(np.int32) == 4
    assert converter._megatron_dtype_code(np.int64) == 5
    assert converter._megatron_dtype_code(np.uint16) == 8


def test_numpy_uint32_index_dtype_fails_closed() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="unsupported Megatron MMIDIDX dtype uint32"):
        converter._megatron_dtype_code(np.uint32)


def test_legacy_uint32_cli_dtype_is_explicit_int32_alias(
    capsys: pytest.CaptureFixture[str],
) -> None:
    converter = _load_converter_module()

    dtype = converter._resolve_output_dtype("uint32")

    assert dtype is np.int32
    assert converter._megatron_dtype_code(dtype) == 4
    assert "no uint32 dtype code" in capsys.readouterr().err


def test_side_channel_length_mismatch_fails_closed() -> None:
    converter = _load_converter_module()

    with pytest.raises(ValueError, match="token_def_use.*length 2.*token_ids length 3"):
        converter._require_token_aligned_side_channel(
            "token_def_use",
            [1, 0],
            [10, 11, 12],
            shard_path="shard_00000.parquet",
            row_idx=7,
        )
