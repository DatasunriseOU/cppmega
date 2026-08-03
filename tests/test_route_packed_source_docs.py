from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cppmega.data.nanochat_pipeline import packed_rows_schema as packed
from cppmega.data.nanochat_pipeline import tokenized_enriched_schema as enriched
from cppmega.data.source_conveyor_composition import (
    SourceComposition,
    build_packed_source_inventory_receipt,
)
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
    load_primary_route_receipt,
    main as route_main,
    route_file,
    route_packed_row,
)
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
    _validate_source_route_payloads,
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
    plan_path = tmp_path / "source_composition_plan.json"
    plan_path.write_text(json.dumps({"schema": "fixture"}))
    source_composition = SourceComposition(
        allowlist={
            ("code", 16): {"fixture.parquet": 1},
            ("commits", 16): {"fixture.parquet": 1},
        },
        receipt={
            "buckets": [16],
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        },
        plan_path=plan_path,
        dedup_receipt_path=plan_path,
        run_files=(),
    )
    input_payload = build_packed_source_inventory_receipt(
        source_composition,
        kind="code",
        input_root=input_root,
    )
    input_receipt.write_text(
        json.dumps(input_payload)
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
    primary_route = load_primary_route_receipt(
        cli_output / "route.receipt.json",
        input_root=input_root,
        source_composition=source_composition,
        kind="code",
        buckets=(16,),
    )
    assert primary_route["primary_root"] == cli_output / "primary"
    assert primary_route["allowlist"] == {
        ("code", 16): {"fixture.parquet": 1}
    }
    assert primary_route["sha256"][("code", 16, "fixture.parquet")] == (
        global_receipt["files"][0]["routes"]["primary"]["sha256"]
    )

    tampered_input_receipt = tmp_path / "tampered-input.receipt.json"
    tampered_input_payload = json.loads(json.dumps(input_payload))
    tampered_input_payload["source_composition"]["receipt_sha256"] = "0" * 64
    tampered_input_receipt.write_text(json.dumps(tampered_input_payload))
    tampered_output = tmp_path / "tampered-output"
    assert route_main(
        [
            "--input-root",
            str(input_root),
            "--input-receipt",
            str(tampered_input_receipt),
            "--output-root",
            str(tampered_output),
            "--buckets",
            "16",
            "--workers",
            "1",
        ]
    ) == 0
    with pytest.raises(ValueError, match="provenance binding drifted"):
        load_primary_route_receipt(
            tampered_output / "route.receipt.json",
            input_root=input_root,
            source_composition=source_composition,
            kind="code",
            buckets=(16,),
        )

    commit_input_receipt = tmp_path / "commit-input.receipt.json"
    commit_input_receipt.write_text(
        json.dumps(
            build_packed_source_inventory_receipt(
                source_composition,
                kind="commits",
                input_root=input_root,
            )
        )
    )
    commit_output = tmp_path / "commit-output"
    assert route_main(
        [
            "--input-root",
            str(input_root),
            "--input-receipt",
            str(commit_input_receipt),
            "--output-root",
            str(commit_output),
            "--buckets",
            "16",
            "--workers",
            "1",
        ]
    ) == 0
    commit_route = load_primary_route_receipt(
        commit_output / "route.receipt.json",
        input_root=input_root,
        source_composition=source_composition,
        kind="commits",
        buckets=(16,),
    )
    bundle = tmp_path / "bundle"
    provenance = bundle / "provenance"
    route_provenance = provenance / "source_routes"
    route_provenance.mkdir(parents=True)
    staged_routes = {}
    snapshot_files = []
    for kind, route in (("code", primary_route), ("commits", commit_route)):
        route_receipt_bytes = route["receipt_path"].read_bytes()
        route_digest = hashlib.sha256(route_receipt_bytes).hexdigest()
        staged = route_provenance / f"{kind}.route.receipt.json"
        staged.write_bytes(route_receipt_bytes)
        binding = route["binding"]
        staged_routes[kind] = {
            "path": staged.relative_to(bundle).as_posix(),
            "sha256": route_digest,
            "route_schema": binding["schema"],
            "input_inventory_sha256": binding["input_inventory_sha256"],
            "output_inventory_sha256": binding["output_inventory_sha256"],
            "primary": binding["totals"]["primary"],
        }
        route_receipt = json.loads(route_receipt_bytes)
        primary_artifact = route_receipt["files"][0]["routes"]["primary"]
        primary_path = Path(route_receipt["output_root"]) / primary_artifact["path"]
        snapshot_files.append(
            {
                "kind": kind,
                "source": str(primary_path.resolve()),
                "size": primary_artifact["size"],
                "sha256": primary_artifact["sha256"],
                "rows": primary_artifact["rows"],
            }
        )
    source_manifest = {
        "source_routes": {
            "code": primary_route["binding"],
            "commits": commit_route["binding"],
        },
        "files": snapshot_files,
    }
    source_manifest_path = provenance / "source_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    _validate_source_route_payloads(
        bundle,
        {
            "source_snapshot": {
                "manifest": source_manifest_path.relative_to(bundle).as_posix(),
                "source_routes": {
                    "schema": "cppmega_packed_source_primary_routes_v1",
                    "policy": "primary-only-code-and-commit-snapshot",
                    "routes": staged_routes,
                },
            }
        },
    )
