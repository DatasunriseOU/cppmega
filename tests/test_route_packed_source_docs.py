from __future__ import annotations

import argparse
import hashlib
import json

import pyarrow.parquet as pq
import pytest

from cppmega.data.nanochat_pipeline import packed_rows_schema as packed
from cppmega.data.nanochat_pipeline import tokenized_enriched_schema as enriched
from cppmega.data.source_identity import source_identity
from cppmega.symbol_identity import compute_symbol_id
from scripts.nanochat_data.pack_enriched_rows import (
    normalize_document_record,
    pack_documents,
    rows_to_table,
)
from scripts.nanochat_data.route_packed_source_docs import (
    _discover,
    _implementation,
    _parse_buckets,
    main as route_main,
    route_file,
    route_packed_row,
)


def _doc(
    *,
    source_doc_index: int,
    tokens: list[int],
    build_kind: str | None,
    symbol_key: str,
    filepath: str | None = None,
):
    source_path = filepath or f"src/{source_doc_index}.cpp"
    symbol_id = compute_symbol_id(symbol_key)
    identity = source_identity(
        {
            "repo": "owner/repo",
            "filepath": source_path,
            "source_doc_id": source_doc_index + 100,
        }
    )
    return normalize_document_record(
        {
            "token_ids": tokens,
            "repo": "owner/repo",
            "filepath": source_path,
            "doc_type": "shell" if build_kind == "sh" else "code",
            "build_kind": build_kind,
            "platform_ids": [7 + source_doc_index],
            "token_symbol_ids": [symbol_id] + [0] * (len(tokens) - 1),
            "symbol_identities": [{"symbol_id": symbol_id, "symbol_key": symbol_key}],
            "token_chunk_starts": [0, 2],
            "token_chunk_ends": [2, len(tokens)],
            "token_chunk_kinds": [1, 2],
            "token_chunk_dep_levels": [0, 1],
            "token_call_edges": [{"from": 0, "to": 1}],
            "token_type_edges": [{"from": 1, "to": 0}],
            "token_domain_edges": [{"from": 0, "to": len(tokens) - 1, "kind": 1}],
            "changed_chunk_ids": [1],
            "changed_chunk_spans": [{"start": 2, "end": len(tokens)}],
            "source_identity_registry": [identity.as_dict()],
            "token_source_identity_ids": [identity.source_identity_id] * len(tokens),
            "ifim_instruction_token_ids": [source_doc_index + 1],
        },
        source_doc_index=source_doc_index,
    )


def _mixed_row() -> dict:
    docs = [
        _doc(
            source_doc_index=0,
            tokens=[10, 11, 12, 13],
            build_kind=None,
            symbol_key="symbol:cpp",
        ),
        _doc(
            source_doc_index=1,
            tokens=[20, 21, 22, 23],
            build_kind="python",
            symbol_key="symbol:python",
        ),
        _doc(
            source_doc_index=2,
            tokens=[30, 31, 32, 33],
            build_kind="cmake",
            symbol_key="symbol:cmake",
        ),
        _doc(
            source_doc_index=3,
            tokens=[40, 41, 42, 43],
            build_kind="sh",
            symbol_key="symbol:build-shell",
            filepath="scripts/build_native.sh",
        ),
        _doc(
            source_doc_index=4,
            tokens=[50, 51, 52, 53],
            build_kind="sh",
            symbol_key="symbol:generic-shell",
            filepath="scripts/release.sh",
        ),
    ]
    rows, overflow = pack_documents(docs, target_length=24, strategy="sequential")
    assert overflow == []
    assert len(rows) == 1
    return rows[0]


def test_bucket_routes_are_unique_and_ordered() -> None:
    assert _parse_buckets("1024,2048,4096") == (1024, 2048, 4096)
    with pytest.raises(argparse.ArgumentTypeError, match="strictly increasing"):
        _parse_buckets("2048,1024,1024")


def test_discovery_and_resume_bindings_are_canonical(tmp_path) -> None:
    for relative in ("16/z.parquet", "1024/a.parquet"):
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
    assert [
        path.relative_to(tmp_path).as_posix()
        for path in _discover(tmp_path, (16, 1024))
    ] == ["1024/a.parquet", "16/z.parquet"]
    assert {
        "router_sha256",
        "packer_sha256",
        "packed_schema_sha256",
        "enriched_schema_sha256",
        "symbol_identity_schema_sha256",
    } <= _implementation().keys()


def test_routes_mixed_row_losslessly_and_writes_resumable_zstd(tmp_path) -> None:
    source = _mixed_row()
    primary, auxiliary, excluded, mixed = route_packed_row(source)
    assert mixed is True
    assert primary is not None
    assert auxiliary is not None
    assert excluded is not None
    assert primary[packed.INPUT_IDS_COLUMN][:12] == [
        10,
        11,
        12,
        13,
        30,
        31,
        32,
        33,
        40,
        41,
        42,
        43,
    ]
    assert auxiliary[packed.INPUT_IDS_COLUMN][:4] == [20, 21, 22, 23]
    assert excluded[packed.INPUT_IDS_COLUMN][:4] == [50, 51, 52, 53]
    assert primary[packed.SOURCE_BUILD_KINDS_COLUMN] == [None, "cmake", "sh"]
    assert auxiliary[packed.SOURCE_BUILD_KINDS_COLUMN] == ["python"]
    assert excluded[packed.SOURCE_BUILD_KINDS_COLUMN] == ["sh"]
    assert primary[enriched.TOKEN_CHUNK_STARTS_COLUMN] == [0, 2, 4, 6, 8, 10]
    assert primary[enriched.TOKEN_CALL_EDGES_COLUMN] == [
        {"from": 0, "to": 1},
        {"from": 2, "to": 3},
        {"from": 4, "to": 5},
    ]
    assert primary[enriched.TOKEN_DOMAIN_EDGES_COLUMN] == [
        {"from": 0, "to": 3, "kind": 1},
        {"from": 4, "to": 7, "kind": 1},
        {"from": 8, "to": 11, "kind": 1},
    ]
    assert [item["symbol_id"] for item in primary["symbol_identities"]] == sorted(
        [
            compute_symbol_id("symbol:cpp"),
            compute_symbol_id("symbol:cmake"),
            compute_symbol_id("symbol:build-shell"),
        ]
    )
    assert [item["symbol_id"] for item in auxiliary["symbol_identities"]] == [
        compute_symbol_id("symbol:python")
    ]
    assert (
        primary[packed.VALID_TOKEN_COUNT_COLUMN]
        + auxiliary[packed.VALID_TOKEN_COUNT_COLUMN]
        + excluded[packed.VALID_TOKEN_COUNT_COLUMN]
        == source[packed.VALID_TOKEN_COUNT_COLUMN]
    )
    assert (
        primary["trained_token_count"]
        + auxiliary["trained_token_count"]
        + excluded["trained_token_count"]
        == source["trained_token_count"]
    )

    input_root = tmp_path / "input"
    input_path = input_root / "16" / "fixture.parquet"
    input_path.parent.mkdir(parents=True)
    pq.write_table(rows_to_table([source]), input_path, compression="zstd")
    output_root = tmp_path / "output"
    receipt = route_file(
        str(input_path),
        input_root_str=str(input_root),
        output_root_str=str(output_root),
        compression_level=6,
        resume=False,
    )
    assert receipt["unresolved_count"] == 0
    assert receipt["mixed_rows_split"] == 1
    assert receipt["routes"]["primary"]["valid_tokens"] == 12
    assert receipt["routes"]["aux_python"]["valid_tokens"] == 4
    assert receipt["routes"]["excluded_non_primary"]["valid_tokens"] == 4
    assert {
        pq.ParquetFile(output_root / route / "16" / "fixture.parquet")
        .metadata.row_group(0)
        .column(0)
        .compression
        for route in ("primary", "aux_python", "excluded_non_primary")
    } == {"ZSTD"}

    marker = output_root / "state" / "16" / "fixture.route.json"
    assert json.loads(marker.read_text())["status"] == "complete"
    assert (
        route_file(
            str(input_path),
            input_root_str=str(input_root),
            output_root_str=str(output_root),
            compression_level=6,
            resume=True,
        )
        == receipt
    )
    tampered = json.loads(marker.read_text())
    tampered["implementation"]["router_sha256"] = "0" * 64
    marker.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="implementation changed after routing"):
        route_file(
            str(input_path),
            input_root_str=str(input_root),
            output_root_str=str(output_root),
            compression_level=6,
            resume=True,
        )
    marker.write_text(json.dumps(receipt))

    input_receipt = tmp_path / "input.receipt.json"
    source_inventory = [
        {
            "path": "16/fixture.parquet",
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "size": input_path.stat().st_size,
        }
    ]
    input_receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "unresolved_count": 0,
                "source_inventory": source_inventory,
                "source_inventory_sha256": hashlib.sha256(
                    json.dumps(
                        source_inventory,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    )
    wrong_inventory = [{**source_inventory[0], "path": "16/other.parquet"}]
    wrong_receipt = tmp_path / "wrong-input.receipt.json"
    wrong_receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "unresolved_count": 0,
                "source_inventory": wrong_inventory,
                "source_inventory_sha256": hashlib.sha256(
                    json.dumps(
                        wrong_inventory,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    )
    with pytest.raises(RuntimeError, match="paths differ from completion receipt"):
        route_main(
            [
                "--input-root",
                str(input_root),
                "--input-receipt",
                str(wrong_receipt),
                "--output-root",
                str(tmp_path / "wrong-output"),
                "--buckets",
                "16",
                "--workers",
                "1",
            ]
        )
    cli_output = tmp_path / "cli-output"
    assert (
        route_main(
            [
                "--input-root",
                str(input_root),
                "--input-receipt",
                str(input_receipt),
                "--output-root",
                str(cli_output),
                "--buckets",
                "16",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    global_receipt = json.loads((cli_output / "route.receipt.json").read_text())
    assert global_receipt["totals"]["source"]["valid_tokens"] == 20
    assert global_receipt["totals"]["primary"]["valid_tokens"] == 12
    assert global_receipt["totals"]["aux_python"]["valid_tokens"] == 4
    assert (
        global_receipt["totals"]["excluded_non_primary"]["valid_tokens"] == 4
    )
