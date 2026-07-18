from __future__ import annotations

from copy import deepcopy
import importlib

import pytest

from cppmega.prompt_graph import (
    INDEX_SCHEMA,
    PromptGraphBuilder,
    PromptGraphContext,
    PromptGraphModelInputs,
    PromptGraphSegment,
    PromptProjectIndex,
)
from cppmega.symbol_identity import (
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    compute_symbol_id,
)


class _CharacterOffsetTokenizer:
    name_or_path = "case3-schema-character-tokenizer"

    def encode_with_offsets(self, text: str):
        return list(range(1, len(text) + 1)), [
            (index, index + 1) for index in range(len(text))
        ]


class _BoundaryMergingTokenizer:
    name_or_path = "case3-schema-boundary-tokenizer"

    def __init__(self, mapped_length: int):
        self.mapped_length = mapped_length

    def encode_with_offsets(self, text: str):
        merge_start = self.mapped_length - 1
        spans = [(index, index + 1) for index in range(merge_start)]
        spans.append((merge_start, len(text)))
        return list(range(1, len(spans) + 1)), spans


PROJECT_ID = "tests/prompt-graph"


def _symbol_key(usr: str) -> str:
    return (
        f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
        f"project={PROJECT_ID}\x1fusr={usr}"
    )


def _v3_payload() -> dict[str, object]:
    first = "int first() { return 1; }\n"
    second = "int second() { return first(); }\n"
    first_key = _symbol_key("c:@F@first#")
    second_key = _symbol_key("c:@F@second#")
    return {
        "schema": INDEX_SCHEMA,
        "project_id": PROJECT_ID,
        "documents": [
            {"id": 1, "source_path": "src/first.cpp", "source": first},
            {"id": 2, "source_path": "src/second.cpp", "source": second},
        ],
        "symbols": [
            {
                "id": 1,
                "symbol_id": compute_symbol_id(first_key),
                "identity": "definition:first",
                "semantic_identity": first_key,
                "symbol_key": first_key,
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
                "symbol_id": compute_symbol_id(second_key),
                "identity": "definition:second",
                "semantic_identity": second_key,
                "symbol_key": second_key,
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
                "symbol_id": compute_symbol_id(first_key),
                "identity": "occurrence:call:first",
                "semantic_identity": first_key,
                "symbol_key": first_key,
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
            {
                "id": 4,
                "symbol_id": compute_symbol_id(first_key),
                "identity": "occurrence:use:first",
                "semantic_identity": first_key,
                "symbol_key": first_key,
                "usr": "c:@F@first#",
                "canonical_signature": "display=first()|type=int ()",
                "qname": "first",
                "kind": "use",
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
            {
                "relation": "def_use",
                "source": "occurrence:use:first",
                "target": "definition:first",
            },
        ],
        "provenance": {
            "producer": "test",
            "symbol_identity_schema_version": SYMBOL_IDENTITY_SCHEMA_VERSION,
            "hashes": {"repository_sha256": "0" * 64},
        },
    }


def test_v3_index_validates_spans_per_document_and_cross_document_edges() -> None:
    index = PromptProjectIndex.from_dict(_v3_payload())
    assert index.symbol_for_identity("occurrence:call:first").document_id == 2
    assert index.symbol_for_identity("definition:first").document_id == 1

    bad = _v3_payload()
    bad["symbols"][0]["end"] = len(bad["documents"][1]["source"]) + 10
    with pytest.raises(ValueError, match="src/first.cpp.*invalid span"):
        PromptProjectIndex.from_dict(bad)

    bad = _v3_payload()
    bad["chunks"][0]["source_path"] = "src/second.cpp"
    with pytest.raises(ValueError, match="document_id.*source_path"):
        PromptProjectIndex.from_dict(bad)


def test_builder_projects_cross_document_route_and_document_ids() -> None:
    index = PromptProjectIndex.from_dict(_v3_payload())
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
    first_symbol = index.symbol_for_identity("definition:first")
    use_symbol = index.symbol_for_identity("occurrence:use:first")
    assert first_symbol.id == 1
    assert first_symbol.symbol_id == compute_symbol_id(first_symbol.symbol_key)
    assert use_symbol.symbol_id == first_symbol.symbol_id
    assert first_symbol.symbol_id != first_symbol.id
    assert (
        artifact.side_channels["symbol_ids"][
            artifact.first_token_for_identity("occurrence:use:first")
        ]
        == first_symbol.symbol_id
    )
    assert (
        artifact.side_channels["call_targets"][
            artifact.first_token_for_identity("occurrence:call:first")
        ]
        == first_symbol.symbol_id
    )
    assert (
        artifact.side_channels["def_use"][
            artifact.first_token_for_identity("definition:first")
        ]
        == 1
    )
    assert (
        artifact.side_channels["def_use"][
            artifact.first_token_for_identity("occurrence:use:first")
        ]
        == 2
    )
    assert set(artifact.side_channels["def_use"]) <= {0, 1, 2}


def test_v3_index_rejects_noncanonical_project_and_symbol_ids() -> None:
    for project_id in ("repo", " owner/repo", "owner/repo.git", "a/b/c"):
        payload = _v3_payload()
        payload["project_id"] = project_id
        with pytest.raises(ValueError, match="owner/repo"):
            PromptProjectIndex.from_dict(payload)

    payload = _v3_payload()
    payload["symbols"][0]["symbol_id"] += 1
    with pytest.raises(ValueError, match="does not match.*canonical ID"):
        PromptProjectIndex.from_dict(payload)


@pytest.mark.parametrize(
    "module_name",
    ("cppmega.prompt_graph", "cppmega.data.prompt_graph"),
)
def test_structural_v3_payload_with_checksum_is_not_production_provenance(
    module_name: str,
) -> None:
    prompt_graph = importlib.import_module(module_name)
    index = prompt_graph.PromptProjectIndex.from_dict(_v3_payload()).with_integrity()

    with pytest.raises(ValueError, match="production repository index requires producer"):
        index.validate_production_repository_index()


@pytest.mark.parametrize(
    "module_name",
    ("cppmega.prompt_graph", "cppmega.data.prompt_graph"),
)
def test_builder_rejects_token_spanning_mapped_and_unmapped_text(
    module_name: str,
) -> None:
    prompt_graph = importlib.import_module(module_name)
    index = prompt_graph.PromptProjectIndex.from_dict(_v3_payload())
    first, second = index.documents
    mapped = first.source + "\n" + second.source
    context = prompt_graph.PromptGraphContext(
        segments=(
            prompt_graph.PromptGraphSegment(
                first.source,
                document_id=first.id,
                source_path=first.source_path,
                source_start=0,
            ),
            prompt_graph.PromptGraphSegment("\n", role="separator"),
            prompt_graph.PromptGraphSegment(
                second.source,
                document_id=second.id,
                source_path=second.source_path,
                source_start=0,
            ),
            prompt_graph.PromptGraphSegment(" synthetic", role="synthetic"),
        )
    )

    with pytest.raises(ValueError, match="mapped and unmapped"):
        prompt_graph.PromptGraphBuilder(
            _BoundaryMergingTokenizer(len(mapped))
        ).build(
            index,
            context,
        )


def test_root_and_data_prompt_graph_artifacts_have_identical_payload_contract() -> None:
    modules = [
        importlib.import_module("cppmega.prompt_graph"),
        importlib.import_module("cppmega.data.prompt_graph"),
    ]
    payloads: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []

    for prompt_graph in modules:
        index = prompt_graph.PromptProjectIndex.from_dict(_v3_payload())
        first, second = index.documents
        artifact = prompt_graph.PromptGraphBuilder(
            _CharacterOffsetTokenizer()
        ).build(
            index,
            prompt_graph.PromptGraphContext(
                segments=(
                    prompt_graph.PromptGraphSegment(
                        first.source,
                        document_id=first.id,
                        source_path=first.source_path,
                        source_start=0,
                    ),
                    prompt_graph.PromptGraphSegment("\n", role="separator"),
                    prompt_graph.PromptGraphSegment(
                        second.source,
                        document_id=second.id,
                        source_path=second.source_path,
                        source_start=0,
                    ),
                )
            ),
        )
        payloads.append(artifact.to_dict())
        model_inputs = artifact.model_inputs(
            total_token_count=artifact.token_count,
            window_start=0,
            window_end=artifact.token_count,
        )
        windows.append(
            {
                "side_channels": model_inputs.side_channels,
                "graph_routes": model_inputs.graph_routes,
                "receipt": model_inputs.receipt,
                "token_count": model_inputs.token_count,
            }
        )

    assert payloads[0] == payloads[1]
    assert windows[0] == windows[1]


def test_prompt_graph_model_inputs_reject_non_enum_token_def_use() -> None:
    index = PromptProjectIndex.from_dict(_v3_payload())
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
    model_inputs = artifact.model_inputs(
        total_token_count=artifact.token_count,
        window_start=0,
        window_end=artifact.token_count,
    )
    invalid_side_channels = {
        name: list(values)
        for name, values in model_inputs.side_channels.items()
    }
    invalid_side_channels["def_use"][0] = 3

    with pytest.raises(ValueError, match="token_def_use.*0/1/2"):
        PromptGraphModelInputs(
            side_channels=invalid_side_channels,
            graph_routes=model_inputs.graph_routes,
            receipt=model_inputs.receipt,
            token_count=model_inputs.token_count,
        )


def test_repository_context_includes_referenced_cross_file_definition() -> None:
    index = PromptProjectIndex.from_dict(_v3_payload())
    second = index.document_for_path("src/second.cpp")

    context = PromptGraphContext.from_repository_prompt(
        index,
        second.source,
        document_id=second.id,
        source_path=second.source_path,
        source_start=0,
    )
    artifact = PromptGraphBuilder(_CharacterOffsetTokenizer()).build(index, context)

    dependencies = [
        segment for segment in context.segments if segment.role == "dependency"
    ]
    assert len(dependencies) == 1
    assert dependencies[0].source_path == "src/first.cpp"
    assert context.segments[-1].role == "target"
    assert artifact.edge_counts["call"] == 1
    assert artifact.graph_routes["graph_call_edges"]


def test_from_prompt_default_remains_unmapped_and_usable() -> None:
    context = PromptGraphContext.from_prompt("int value = 1;")

    assert context.text == "int value = 1;"
    assert context.segments[0].source_start is None
    assert set(context.source_positions()) == {None}


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
    payload = _v3_payload()
    edge = payload["edges"][1]
    edge["relation"] = relation
    if kind is None:
        edge.pop("kind")
    else:
        edge["kind"] = kind

    with pytest.raises(ValueError, match=message):
        PromptProjectIndex.from_dict(payload)


def test_canonical_cross_domain_kind_100_is_accepted() -> None:
    payload = _v3_payload()
    payload["edges"][1].update(relation="cross_domain", kind=100)
    index = PromptProjectIndex.from_dict(payload)
    assert index.edges[1].kind == 100


def test_generated_tokens_keep_live_structure_and_repository_summary_routes() -> None:
    index = PromptProjectIndex.from_dict(_v3_payload())
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
    payload = _v3_payload()
    document = payload["documents"][0]
    legacy = {
        "schema": "cppmega_prompt_graph_index_v1",
        "project_id": "tests/legacy",
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
    assert index.symbols[0].symbol_id == compute_symbol_id(
        index.symbols[0].symbol_key
    )
