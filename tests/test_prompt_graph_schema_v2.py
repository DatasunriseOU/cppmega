from __future__ import annotations

from copy import deepcopy

import pytest

from cppmega.prompt_graph import (
    PromptGraphBuilder,
    PromptGraphContext,
    PromptGraphSegment,
    PromptProjectIndex,
)


class _CharacterOffsetTokenizer:
    name_or_path = "case3-schema-character-tokenizer"

    def encode_with_offsets(self, text: str):
        return list(range(1, len(text) + 1)), [
            (index, index + 1) for index in range(len(text))
        ]


def _v2_payload() -> dict[str, object]:
    first = "int first() { return 1; }\n"
    second = "int second() { return first(); }\n"
    return {
        "schema": "cppmega_prompt_graph_index_v2",
        "project_id": "multi",
        "documents": [
            {"id": 1, "source_path": "src/first.cpp", "source": first},
            {"id": 2, "source_path": "src/second.cpp", "source": second},
        ],
        "symbols": [
            {
                "id": 1,
                "identity": "definition:first",
                "semantic_identity": "c:@F@first#",
                "usr": "c:@F@first#",
                "canonical_signature": "display=first()|type=int ()",
                "qname": "first",
                "kind": "function",
                "document_id": 1,
                "source_path": "src/first.cpp",
                "start": 4,
                "end": 9,
                "chunk_identity": "chunk:first",
            },
            {
                "id": 2,
                "identity": "definition:second",
                "semantic_identity": "c:@F@second#",
                "usr": "c:@F@second#",
                "canonical_signature": "display=second()|type=int ()",
                "qname": "second",
                "kind": "function",
                "document_id": 2,
                "source_path": "src/second.cpp",
                "start": 4,
                "end": 10,
                "chunk_identity": "chunk:second",
            },
            {
                "id": 3,
                "identity": "occurrence:call:first",
                "semantic_identity": "c:@F@first#",
                "usr": "c:@F@first#",
                "canonical_signature": "display=first()|type=int ()",
                "qname": "first",
                "kind": "callsite",
                "document_id": 2,
                "source_path": "src/second.cpp",
                "start": second.index("first"),
                "end": second.index("first") + len("first"),
                "chunk_identity": "chunk:second",
            },
        ],
        "chunks": [
            {
                "id": 0,
                "identity": "chunk:first",
                "document_id": 1,
                "source_path": "src/first.cpp",
                "start": 0,
                "end": len(first) - 1,
                "kind": 1,
                "dep_level": 0,
            },
            {
                "id": 1,
                "identity": "chunk:second",
                "document_id": 2,
                "source_path": "src/second.cpp",
                "start": 0,
                "end": len(second) - 1,
                "kind": 1,
                "dep_level": 1,
            },
        ],
        "edges": [
            {
                "relation": "call",
                "source": "occurrence:call:first",
                "target": "definition:first",
            },
            {
                "relation": "domain",
                "source": "occurrence:call:first",
                "target": "definition:first",
                "kind": 2,
            },
        ],
        "provenance": {
            "producer": "test",
            "hashes": {"repository_sha256": "0" * 64},
        },
    }


def test_v2_index_validates_spans_per_document_and_cross_document_edges() -> None:
    index = PromptProjectIndex.from_dict(_v2_payload())
    assert index.symbol_for_identity("occurrence:call:first").document_id == 2
    assert index.symbol_for_identity("definition:first").document_id == 1

    bad = _v2_payload()
    bad["symbols"][0]["end"] = len(bad["documents"][1]["source"]) + 10
    with pytest.raises(ValueError, match="src/first.cpp.*invalid span"):
        PromptProjectIndex.from_dict(bad)

    bad = _v2_payload()
    bad["chunks"][0]["source_path"] = "src/second.cpp"
    with pytest.raises(ValueError, match="document_id.*source_path"):
        PromptProjectIndex.from_dict(bad)


def test_builder_projects_cross_document_route_and_document_ids() -> None:
    index = PromptProjectIndex.from_dict(_v2_payload())
    first, second = index.documents
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index,
        PromptGraphContext(
            segments=(
                PromptGraphSegment(
                    first.source,
                    document_id=first.id,
                    source_path=first.source_path,
                    source_start=0,
                ),
                PromptGraphSegment("\n", role="separator"),
                PromptGraphSegment(
                    second.source,
                    document_id=second.id,
                    source_path=second.source_path,
                    source_start=0,
                ),
            )
        ),
    )

    assert artifact.graph_routes["graph_call_edges"] == [[1, 0]]
    assert artifact.graph_routes["graph_domain_edges"][0][2] == 2
    assert {value for value in artifact.side_channels["source_doc_ids"] if value} == {
        1,
        2,
    }


@pytest.mark.parametrize(
    ("relation", "kind", "message"),
    [
        ("domain", None, "requires an explicit kind"),
        ("domain", 0, "unknown domain edge kind"),
        ("domain", 20, "belongs to build, not domain"),
        ("build", 2, "belongs to domain, not build"),
        ("shell", 90, "belongs to diagnostic, not shell"),
        ("cross_domain", 90, "belongs to diagnostic, not cross_domain"),
    ],
)
def test_triple_route_kinds_are_explicit_and_family_checked(
    relation: str,
    kind: int | None,
    message: str,
) -> None:
    payload = _v2_payload()
    edge = payload["edges"][1]
    edge["relation"] = relation
    if kind is None:
        edge.pop("kind")
    else:
        edge["kind"] = kind

    with pytest.raises(ValueError, match=message):
        PromptProjectIndex.from_dict(payload)


def test_canonical_cross_domain_kind_100_is_accepted() -> None:
    payload = _v2_payload()
    payload["edges"][1].update(relation="cross_domain", kind=100)
    index = PromptProjectIndex.from_dict(payload)
    assert index.edges[1].kind == 100


def test_generated_tokens_keep_live_structure_and_repository_summary_routes() -> None:
    index = PromptProjectIndex.from_dict(_v2_payload())
    first, second = index.documents
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(
        index,
        PromptGraphContext(
            segments=(
                PromptGraphSegment(
                    first.source,
                    document_id=first.id,
                    source_path=first.source_path,
                    source_start=0,
                ),
                PromptGraphSegment("\n", role="separator"),
                PromptGraphSegment(
                    second.source,
                    document_id=second.id,
                    source_path=second.source_path,
                    source_start=0,
                ),
            )
        ),
    )
    generated = 3
    model_inputs = artifact.model_inputs(
        total_token_count=artifact.token_count + generated,
        window_start=0,
        window_end=artifact.token_count + generated,
    )
    bias = model_inputs.dense_attention_bias()

    assert model_inputs.receipt["generated_token_policy"] == (
        "generated_continuation_chunk_with_repository_summary_v1"
    )
    assert model_inputs.receipt["generated_token_count"] == generated
    assert model_inputs.receipt["repository_summary_token_count"] > 0
    assert all(
        sum(bias[token_index]) > 0
        for token_index in range(artifact.token_count, artifact.token_count + generated)
    )
    assert all(
        value > 0
        for value in model_inputs.side_channels["structure_ids"][-generated:]
    )
    assert model_inputs.side_channels["domain_ids"][-generated:] == [1] * generated
    assert model_inputs.side_channels["confidence_ids"][-generated:] == [1] * generated
    assert model_inputs.side_channels["source_doc_ids"][-generated:] == [0] * generated


def test_legacy_single_document_schema_requires_explicit_opt_in() -> None:
    payload = _v2_payload()
    document = payload["documents"][0]
    legacy = {
        "schema": "cppmega_prompt_graph_index_v1",
        "project_id": "legacy",
        "source_path": document["source_path"],
        "source": document["source"],
        "symbols": [
            {
                "id": 1,
                "identity": "legacy:first",
                "kind": "function",
                "start": 4,
                "end": 9,
            }
        ],
        "chunks": [
            {
                "id": 0,
                "identity": "legacy:chunk",
                "start": 0,
                "end": len(document["source"]) - 1,
                "kind": 1,
                "dep_level": 0,
            }
        ],
        "edges": [],
    }

    with pytest.raises(ValueError, match="legacy.*explicit"):
        PromptProjectIndex.from_dict(legacy)
    index = PromptProjectIndex.from_dict(
        deepcopy(legacy), allow_legacy_single_document=True
    )
    assert len(index.documents) == 1
    assert index.documents[0].source_path == "src/first.cpp"
