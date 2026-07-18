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
    assert converter.REQUIRED_SYMBOL_IDENTITY_SCHEMA_VERSION == 3
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


def test_parquet_validation_rejects_cross_project_symbol_id_collision(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    converter = _load_converter()
    first_key = (
        "fallback:schema=v3\x1fqname=left::route\x1fkind=FUNCTION_DECL\x1f"
        "sig=display=route(int)|type=int (int)\x1fscope=project=owner/repo-a|file=route.cpp"
    )
    second_key = (
        "fallback:schema=v3\x1fqname=right::route\x1fkind=FUNCTION_DECL\x1f"
        "sig=display=route(int)|type=int (int)\x1fscope=project=owner/repo-b|file=route.cpp"
    )
    claimed_id = converter._compute_symbol_id(first_key)
    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.uint32())),
            pa.field("token_symbol_ids", pa.list_(pa.uint64())),
            pa.field("token_call_targets", pa.list_(pa.uint64())),
            pa.field("token_type_refs", pa.list_(pa.uint64())),
            pa.field("token_def_use", pa.list_(pa.uint8())),
            pa.field(
                "symbol_identities",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("symbol_id", pa.uint64()),
                            pa.field("symbol_key", pa.string()),
                        ]
                    )
                ),
            ),
        ],
        metadata={
            converter.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii"): b"3"
        },
    )
    paths = []
    for shard_index, key in enumerate((first_key, second_key)):
        path = tmp_path / f"shard_{shard_index:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "input_ids": [1],
                        "token_symbol_ids": [claimed_id],
                        "token_call_targets": [0],
                        "token_type_refs": [0],
                        "token_def_use": [0],
                        "symbol_identities": [
                            {"symbol_id": claimed_id, "symbol_key": key}
                        ],
                    }
                ],
                schema=schema,
            ),
            path,
        )
        paths.append(str(path))

    with pytest.raises(RuntimeError, match="symbol ID collision"):
        converter._require_symbol_identity_schema(paths)
