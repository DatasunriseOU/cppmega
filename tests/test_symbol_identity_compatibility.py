from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data_prep_parquet_to_megatron.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("data_prep_parquet_to_megatron", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symbol_identity_schema_metadata_accepts_current_version() -> None:
    converter = _load_converter()
    metadata = {
        converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode(): str(
            converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
        ).encode()
    }

    assert (
        converter._require_symbol_identity_schema_metadata(metadata, "current.parquet")
        == converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {b"cppmega.symbol_identity_schema_version": b"1"},
        {b"cppmega.symbol_identity_schema_version": b"not-an-int"},
    ],
)
def test_symbol_identity_schema_metadata_rejects_missing_stale_or_invalid_versions(
    metadata: dict[bytes, bytes] | None,
) -> None:
    converter = _load_converter()

    with pytest.raises(RuntimeError, match="regenerate.*clang USR",):
        converter._require_symbol_identity_schema_metadata(metadata, "stale.parquet")


def test_megatron_manifest_records_symbol_identity_schema_version() -> None:
    converter = _load_converter()
    manifest: dict[str, object] = {}

    converter._add_symbol_identity_manifest(
        manifest,
        converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION,
    )

    assert manifest["symbol_identity_schema_version"] == (
        converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
    )


def test_partial_semantic_identity_columns_are_rejected(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter()
    table = pa.table(
        {
            "input_ids": [[1, 2]],
            "token_symbol_ids": [[7, 7]],
        }
    ).replace_schema_metadata(
        {
            converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode(): str(
                converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION
            ).encode()
        }
    )
    path = tmp_path / "partial.parquet"
    pq.write_table(table, path)

    with pytest.raises(RuntimeError, match="partial semantic-symbol columns"):
        converter._require_symbol_identity_schema([str(path)])
