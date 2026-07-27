from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega.data.symbol_identity import SYMBOL_IDENTITY_SCHEMA_VERSION
from cppmega.data.source_identity import source_identity
from scripts import streaming_reindex as sr
from scripts import streaming_reindex_commits as src
from scripts.nanochat_data import clang_enriched_to_parquet as materializer
from scripts.nanochat_data.token_budget import chunk_enriched_document


class _CharacterTokenizer:
    """Small deterministic tokenizer: one token per Unicode code point."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def get_vocab(self) -> dict[str, int]:
        return {"<char>": 0}


def test_empty_index_log_accepts_new_lossless_build_receipt() -> None:
    log = "\n".join(
        [
            "Found 0 C/C++ source files",
            "Found 1 build/compilation files",
            (
                "Build docs: emitted=1 source_chars_in=1 source_chars_out=1 "
                "skipped_zero_length=0 "
                "source_chunk_dedup=disabled_for_lossless_spans"
            ),
            "Generated 0 total training documents",
        ]
    )

    assert sr._classify_empty_index_project_log(log) == "no_training_documents"


def test_empty_index_log_rejects_lossy_build_receipt() -> None:
    log = "\n".join(
        [
            "Found 0 C/C++ source files",
            "Found 1 build/compilation files",
            (
                "Build docs: emitted=1 source_chars_in=9 source_chars_out=8 "
                "skipped_zero_length=0 "
                "source_chunk_dedup=disabled_for_lossless_spans"
            ),
            "Generated 0 total training documents",
        ]
    )

    assert sr._classify_empty_index_project_log(log) == "domain_source_loss"


def test_tar_member_path_collision_fails_instead_of_dropping_source(
    tmp_path: Path,
) -> None:
    target = tmp_path / "occupied"
    target.mkdir()

    with pytest.raises(sr.RepoFailure, match="refusing to drop source content"):
        sr._copy_tar_member_file(
            io.BytesIO(b"preserve me"),
            target,
            repo="owner/repo",
            member_name="cpp_all/owner/repo/occupied",
        )


def _aligned_overlong_doc() -> dict:
    text = "abcdefghijklmnopqrst"
    width = len(text)
    source_registry = [
        {
            "source_identity_id": 101,
            "canonical_sha256": "a" * 64,
            "source": "owner/repo:a.cc",
        },
        {
            "source_identity_id": 202,
            "canonical_sha256": "b" * 64,
            "source": "owner/repo:b.cc",
        },
    ]
    return {
        "text": text,
        "source_text": text,
        "structure_ids": list(range(width)),
        "ast_depth": [100 + index for index in range(width)],
        "sibling_index": [200 + index for index in range(width)],
        "ast_node_type": [300 + index for index in range(width)],
        "symbol_ids": [400 + index for index in range(width)],
        "call_targets": [500 + index for index in range(width)],
        "type_refs": [600 + index for index in range(width)],
        "def_use": [index % 3 for index in range(width)],
        "change_mask_pre": [index % 2 for index in range(width)],
        "change_mask_post": [(index + 1) % 2 for index in range(width)],
        "hunk_id_per_char": [700 + index for index in range(width)],
        "edit_op_per_char": [index % 4 for index in range(width)],
        "domain_ids": [10 + index for index in range(width)],
        "domain_role_ids": [20 + index for index in range(width)],
        "domain_entity_ids": [30 + index for index in range(width)],
        "domain_scope_ids": [40 + index for index in range(width)],
        "domain_source_doc_ids": [1] * 10 + [2] * 10,
        "domain_source_identity_ids": [101] * 10 + [202] * 10,
        "domain_confidence_ids": [3] * width,
        "source_identity_registry": source_registry,
        # Deliberately leave prefix [0:2] and suffix [18:20] outside boundaries:
        # lossless splitting must still emit them.
        "chunk_boundaries": [
            {
                "start": 2,
                "end": 8,
                "kind": 1,
                "dep_level": 0,
                "name": "left",
                "symbol_id": 402,
            },
            {
                "start": 8,
                "end": 18,
                "kind": 2,
                "dep_level": 1,
                "name": "right",
                "symbol_id": 408,
            },
        ],
        "call_edges": [{"from": 0, "to": 1}],
        "type_edges": [{"from": 1, "to": 0}],
        "domain_edges": [
            {"from_char": 1, "to_char": 2, "kind": 1},
            {"from_char": 1, "to_char": 15, "kind": 1},
        ],
    }


def test_overlong_split_is_text_lossless_and_slices_all_char_sidecars() -> None:
    doc = _aligned_overlong_doc()
    pieces = chunk_enriched_document(doc, 7, _CharacterTokenizer())

    assert len(pieces) == 3
    assert "".join(piece["text"] for piece in pieces) == doc["text"]
    assert "".join(piece["source_text"] for piece in pieces) == doc["source_text"]

    aligned_fields = (
        "structure_ids",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        "change_mask_pre",
        "change_mask_post",
        "hunk_id_per_char",
        "edit_op_per_char",
        "domain_ids",
        "domain_role_ids",
        "domain_entity_ids",
        "domain_scope_ids",
        "domain_source_doc_ids",
        "domain_source_identity_ids",
        "domain_confidence_ids",
    )
    for field in aligned_fields:
        assert [
            value for piece in pieces for value in piece[field]
        ] == doc[field], field
        assert all(len(piece[field]) == len(piece["text"]) for piece in pieces), field

    assert all(piece["source_identity_registry"] == doc["source_identity_registry"] for piece in pieces)
    assert all(piece["actual_token_count"] <= 7 for piece in pieces)
    assert all(
        0 <= boundary["start"] < boundary["end"] <= len(piece["text"])
        for piece in pieces
        for boundary in piece["chunk_boundaries"]
    )
    assert all(
        0 <= edge["from"] < len(piece["chunk_boundaries"])
        and 0 <= edge["to"] < len(piece["chunk_boundaries"])
        for piece in pieces
        for field in ("call_edges", "type_edges")
        for edge in piece[field]
    )
    assert pieces[0]["_lossless_split_audit"]["cross_piece_edges"] == {
        "domain_edges": 1,
        "build_edges": 0,
        "shell_edges": 0,
        "diagnostic_edges": 0,
        "cross_domain_edges": 0,
        "call_edges": 0,
        "type_edges": 0,
    }


def test_overlong_split_rejects_misaligned_commit_sidecar() -> None:
    doc = _aligned_overlong_doc()
    doc["change_mask_pre"] = doc["change_mask_pre"][:-1]

    with pytest.raises(ValueError, match=r"change_mask_pre.*length"):
        chunk_enriched_document(doc, 7, _CharacterTokenizer())


def test_materializer_rejects_misaligned_structure_without_padding() -> None:
    with pytest.raises(ValueError, match=r"structure_ids.*length"):
        materializer.process_record_with_policy(
            {
                "text": "int value = 7;",
                "repo": "owner/repo",
                "filepath": "src/value.cc",
                "doc_type": "code",
                "structure_ids": [3, 3],
                "chunk_boundaries": [],
            },
            _CharacterTokenizer(),
            128,
            overflow_policy="split",
        )


def test_split_preserves_document_larger_than_65k_tokens() -> None:
    width = 70_123
    text = "x" * width
    identity_id = 909
    doc = {
        "text": text,
        "source_text": text,
        "structure_ids": [3] * width,
        "change_mask_pre": [1] * width,
        "change_mask_post": [2] * width,
        "hunk_id_per_char": [7] * width,
        "edit_op_per_char": [3] * width,
        "domain_source_doc_ids": [1] * width,
        "domain_source_identity_ids": [identity_id] * width,
        "source_identity_registry": [
            {
                "source_identity_id": identity_id,
                "canonical_sha256": "c" * 64,
                "source": "owner/repo:huge.cc",
            }
        ],
        "chunk_boundaries": [],
    }

    pieces = chunk_enriched_document(doc, 16_381, _CharacterTokenizer())

    assert len(pieces) == 5
    assert "".join(piece["text"] for piece in pieces) == text
    for field in (
        "structure_ids",
        "change_mask_pre",
        "change_mask_post",
        "hunk_id_per_char",
        "edit_op_per_char",
        "domain_source_doc_ids",
        "domain_source_identity_ids",
    ):
        assert sum((piece[field] for piece in pieces), []) == doc[field]
    assert max(piece["actual_token_count"] for piece in pieces) <= 16_381


def test_split_rechecks_preferred_break_for_nonmonotonic_tokenizer() -> None:
    class NonmonotonicPrefixTokenizer:
        def encode(self, text: str) -> list[int]:
            token_count = {
                4: 6,
                5: 5,
                6: 5,
            }.get(len(text), len(text))
            return list(range(token_count))

    text = "abc\ndefghi"
    doc = {
        "text": text,
        "structure_ids": [1] * len(text),
        "chunk_boundaries": [],
    }

    pieces = chunk_enriched_document(
        doc,
        max_tokens=5,
        tokenizer=NonmonotonicPrefixTokenizer(),
    )

    assert "".join(piece["text"] for piece in pieces) == text
    assert pieces[0]["text"] == text[:6]
    assert max(piece["actual_token_count"] for piece in pieces) <= 5


def test_route_by_fit_fails_closed_instead_of_dropping_overlong_rows(
    tmp_path: Path,
) -> None:
    tokenized = tmp_path / "tokenized.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": [[1, 2], list(range(17))],
                "token_structure_ids": [[1, 1], [1] * 17],
            }
        ),
        tokenized,
    )

    with pytest.raises(sr.RepoFailure, match=r"overlong_rows=1.*overlong_tokens=17"):
        src.route_by_fit(tokenized, (8, 16), tmp_path / "routed")

    assert not (tmp_path / "routed" / "route_8.parquet").exists()
    assert not (tmp_path / "routed" / "dropped_overlong.json").exists()


def test_route_by_fit_preserves_every_fitting_row_once(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": [
                    [1, 2],
                    list(range(8)),
                    list(range(9)),
                    list(range(16)),
                ],
                "source_doc_id": ["a", "b", "c", "d"],
            }
        ),
        tokenized,
    )

    routed = src.route_by_fit(tokenized, (8, 16), tmp_path / "routed")

    assert set(routed) == {8, 16}
    assert pq.read_table(routed[8]).column("source_doc_id").to_pylist() == [
        "a",
        "b",
    ]
    assert pq.read_table(routed[16]).column("source_doc_id").to_pylist() == [
        "c",
        "d",
    ]


def test_local_materializer_writes_durable_exact_split_receipt(
    tmp_path: Path,
) -> None:
    text = "0123456789abcdef"
    record = {
        "text": text,
        "repo": "owner/repo",
        "filepath": "src/main.cc",
        "doc_type": "code",
        "structure_ids": [1] * len(text),
        "chunk_boundaries": [],
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        "symbol_identities": [],
    }
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "tokenized.parquet"
    receipt = sr.materialize_stats_path(output)

    summary = materializer.convert_local_jsonl_to_parquet(
        source,
        output,
        tokenizer=_CharacterTokenizer(),
        max_tokens=7,
        overflow_policy="split",
        stats_file=receipt,
    )
    assert receipt.exists()
    assert json.loads(receipt.read_text(encoding="utf-8")) == summary
    assert summary["schema"] == "cppmega.materialize_split_stats_v1"
    assert summary["docs_in"] == 1
    assert summary["source_docs_emitted"] == 1
    assert summary["split_input_docs"] == 1
    assert summary["split_output_docs"] == summary["docs_out"]
    assert summary["dropped_input_docs"] == 0
    assert summary["emitted_valid_tokens"] == summary["emitted_chars"]
    assert summary["max_emitted_tokens"] <= 7


def test_real_materialized_split_keeps_token_sidecars_aligned(
    tmp_path: Path,
) -> None:
    tokenizer = materializer.load_tokenizer(str(sr.TOKENIZER_PATH))
    text = "\n".join(f"int value_{index} = {index};" for index in range(80))
    width = len(text)
    record = {
        "text": text,
        "repo": "owner/repo",
        "filepath": "src/overlong.cc",
        "doc_type": "code",
        "build_kind": "cmake",
        "structure_ids": [3] * width,
        "chunk_boundaries": [],
        "change_mask_pre": [index % 2 for index in range(width)],
        "change_mask_post": [(index + 1) % 2 for index in range(width)],
        "hunk_id_per_char": [1 + index // 20 for index in range(width)],
        "edit_op_per_char": [1 + index % 3 for index in range(width)],
        "domain_source_doc_ids": [1] * width,
        "domain_edges": [
            {"from_char": 0, "to_char": width - 1, "kind": 1}
        ],
        "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
        "symbol_identities": [],
    }
    identity = source_identity(record)
    record["source_identity_registry"] = [identity.as_dict()]
    record["domain_source_identity_ids"] = [identity.source_identity_id] * width

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "tokenized.parquet"
    receipt = sr.materialize_stats_path(output)
    summary = materializer.convert_local_jsonl_to_parquet(
        source,
        output,
        tokenizer=tokenizer,
        max_tokens=125,
        overflow_policy="split",
        materialize_tokenized_enriched=True,
        stats_file=receipt,
    )
    assert sr.read_materialize_stats(
        output, fixed_shape_max_tokens=128
    ) == summary
    with pytest.raises(
        ValueError, match="materialized token row exceeds the largest fixed bucket"
    ):
        sr.read_materialize_stats(
            output,
            fixed_shape_max_tokens=int(summary["max_materialized_tokens"]) - 1,
        )

    table = pq.read_table(output)
    rows = table.to_pylist()
    assert summary["split_input_docs"] == 1
    assert summary["materialized_rows"] == len(rows) == summary["docs_out"]
    assert summary["max_materialized_tokens"] == max(
        len(row["token_ids"]) for row in rows
    )
    assert summary["max_materialized_tokens"] <= 128
    assert summary["cross_piece_domain_edges"] == 1
    assert "".join(row["text"] for row in rows).endswith(text)
    assert "".join(row["source_text"] for row in rows) == "".join(
        row["text"] for row in rows
    )
    for row in rows:
        assert row["build_kind"] == "cmake"
        token_count = len(row["token_ids"])
        for field in (
            "token_source_doc_ids",
            "token_source_identity_ids",
            "token_change_mask_pre",
            "token_change_mask_post",
            "hunk_id_per_token",
            "edit_op_per_token",
        ):
            assert len(row[field]) == token_count, field
        assert set(row["token_source_identity_ids"]) == {
            identity.source_identity_id
        }

    original_bytes = output.read_bytes()
    tampered_bytes = bytearray(original_bytes)
    tampered_bytes[len(tampered_bytes) // 2] ^= 1
    output.write_bytes(tampered_bytes)
    with pytest.raises(
        ValueError, match="materialized parquet hash does not match its receipt"
    ):
        sr.read_materialize_stats(output, fixed_shape_max_tokens=128)


def test_streaming_stage_materializes_and_routes_overlong_input(
    tmp_path: Path,
) -> None:
    text = "\n".join(f"int stage_{index} = {index};" for index in range(80))
    enriched = tmp_path / "fixture.enriched.jsonl"
    enriched.write_text(
        json.dumps(
            {
                "text": text,
                "repo": "owner/repo",
                "filepath": "src/stage.cc",
                "doc_type": "code",
                "build_kind": "cmake",
                "structure_ids": [3] * len(text),
                "chunk_boundaries": [],
                "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
                "symbol_identities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split_budget = sr.lossless_materialize_budget((64, 128))

    tokenized = sr.stage_materialize(
        "fixture",
        enriched,
        tmp_path,
        memory_limit_gb=0.0,
        project_id="owner/repo",
        max_tokens=split_budget,
        fixed_shape_max_tokens=128,
    )
    stats = sr.read_materialize_stats(
        tokenized,
        fixed_shape_max_tokens=128,
    )
    routed = src.route_by_fit(tokenized, (64, 128), tmp_path / "routed")

    assert stats["split_input_docs"] == 1
    assert stats["dropped_input_docs"] == 0
    assert stats["max_materialized_tokens"] <= 128
    assert sum(pq.read_table(path).num_rows for path in routed.values()) == stats[
        "docs_out"
    ]
    routed_build_kinds = [
        kind
        for path in routed.values()
        for kind in pq.read_table(
            path,
            columns=["build_kind"],
        ).column("build_kind").to_pylist()
    ]
    assert routed_build_kinds == ["cmake"] * stats["docs_out"]
    packed = {
        bucket: sr.stage_pack("fixture", path, bucket, tmp_path)
        for bucket, path in routed.items()
    }
    packed_build_kinds = [
        kind
        for path in packed.values()
        for row_kinds in pq.read_table(
            path,
            columns=["source_build_kinds"],
        ).column("source_build_kinds").to_pylist()
        for kind in row_kinds
    ]
    assert packed_build_kinds == ["cmake"] * stats["docs_out"]
    aggregate = sr.summarize_materialize_stats(
        [{"materialize_stats": stats}]
    )
    assert aggregate["receipts"] == 1
    assert aggregate["split_input_docs"] == 1
    assert aggregate["materialized_output_tokens"] == stats[
        "materialized_output_tokens"
    ]


def test_streaming_materialize_budget_reserves_bos_and_domain_delimiters() -> None:
    assert sr.lossless_materialize_budget((1024, 2048, 4096, 8192, 16384)) == 16381
    assert sr.LOSSLESS_INDEX_MAX_TOKENS > 65536
