import pytest

torch = pytest.importorskip("torch")

from cppmega.megatron import structure_dataset_patch as patch


def test_pop_structure_batch_removes_sidecars_and_sets_thread_local():
    batch = {
        "tokens": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, 4]]),
        "domain_ids": torch.tensor([[1, 2, 2]]),
        "role_ids": torch.tensor([[1, 6, 4]]),
        "confidence_ids": torch.tensor([[4, 4, 4]]),
        "structure_ids": torch.tensor([[5, 6, 7]]),
        "dep_levels": torch.tensor([[0, 1, 2]]),
        "symbol_ids": torch.tensor([[11, 12, 13]]),
        "change_mask_post": torch.tensor([[0, 1, 0]]),
        "graph_call_edges": torch.tensor([[[0, 2], [-1, -1]]]),
        "graph_call_edge_counts": torch.tensor([1]),
        "graph_build_edges": torch.tensor([[[1, 2, 20], [-1, -1, -1]]]),
        "graph_build_edge_counts": torch.tensor([1]),
    }

    structure = patch._pop_structure_batch(batch)

    assert set(batch) == {"tokens", "labels"}
    assert structure is not None
    assert torch.equal(structure["domain_ids"], torch.tensor([[1, 2, 2]]))
    assert torch.equal(structure["role_ids"], torch.tensor([[1, 6, 4]]))
    assert torch.equal(structure["confidence_ids"], torch.tensor([[4, 4, 4]]))
    assert torch.equal(structure["structure_ids"], torch.tensor([[5, 6, 7]]))
    assert torch.equal(structure["symbol_ids"], torch.tensor([[11, 12, 13]]))
    assert torch.equal(structure["graph_call_edge_counts"], torch.tensor([1]))
    assert torch.equal(structure["graph_build_edge_counts"], torch.tensor([1]))
    assert patch._get_current_structure_batch() is structure


def test_safe_sidecar_path_allows_plain_relative_and_blocks_escape():
    base = "/data/cppmega_sidecar"
    ok = patch._safe_sidecar_path(
        base, "train_token_ast_depth.bin", col="c", field="path", json_path="m.json"
    )
    assert ok == "/data/cppmega_sidecar/train_token_ast_depth.bin"

    for evil in ("../../etc/passwd", "/etc/passwd", "sub/../../escape.bin"):
        with pytest.raises(ValueError):
            patch._safe_sidecar_path(base, evil, col="c", field="path", json_path="m.json")


def test_build_graph_route_tensors_offsets_caps_and_clips():
    graph_sidecars = {
        "token_call_edges": {
            "offsets": [0, 3],
            "data": torch.tensor([[1, 2], [2, 4], [7, 8]], dtype=torch.int32).numpy(),
        },
        "token_type_edges": {
            "offsets": [0, 2],
            "data": torch.tensor([[3, 4], [8, 9]], dtype=torch.int32).numpy(),
        },
        "token_domain_edges": {
            "offsets": [0, 2],
            "data": torch.tensor([[1, 4, 20], [8, 9, 60]], dtype=torch.int32).numpy(),
        },
        "token_build_edges": {
            "offsets": [0, 1],
            "data": torch.tensor([[2, 5, 21]], dtype=torch.int32).numpy(),
        },
        "token_shell_edges": {
            "offsets": [0, 1],
            "data": torch.tensor([[0, 3, 40]], dtype=torch.int32).numpy(),
        },
        "token_diagnostic_edges": {
            "offsets": [0, 1],
            "data": torch.tensor([[4, 1, 60]], dtype=torch.int32).numpy(),
        },
        "token_cross_domain_edges": {
            "offsets": [0, 1],
            "data": torch.tensor([[5, 2, 62]], dtype=torch.int32).numpy(),
        },
        "token_chunk_starts": {
            "offsets": [0, 3],
            "data": torch.tensor([0, 2, 8], dtype=torch.int32).numpy(),
        },
        "token_chunk_ends": {
            "offsets": [0, 3],
            "data": torch.tensor([2, 6, 10], dtype=torch.int32).numpy(),
        },
        "token_chunk_kinds": {
            "offsets": [0, 3],
            "data": torch.tensor([1, 2, 3], dtype=torch.int32).numpy(),
        },
        "token_chunk_dep_levels": {
            "offsets": [0, 3],
            "data": torch.tensor([0, 4, 9], dtype=torch.int32).numpy(),
        },
    }
    spans = [
        {
            "real_doc": 0,
            "doc_start_token": 0,
            "source_start": 1,
            "source_end": 6,
            "target_start": 0,
        }
    ]

    routed = patch._build_graph_route_tensors(
        graph_sidecars,
        spans,
        target_len=5,
        max_edges=2,
        max_chunks=2,
    )

    assert torch.equal(routed["graph_call_edges"], torch.tensor([[0, 1], [1, 3]]))
    assert routed["graph_call_edge_counts"].item() == 2
    assert torch.equal(routed["graph_type_edges"], torch.tensor([[2, 3], [-1, -1]]))
    assert routed["graph_type_edge_counts"].item() == 1
    assert torch.equal(routed["graph_domain_edges"], torch.tensor([[0, 3, 20], [-1, -1, -1]]))
    assert routed["graph_domain_edge_counts"].item() == 1
    assert torch.equal(routed["graph_build_edges"], torch.tensor([[1, 4, 21], [-1, -1, -1]]))
    assert routed["graph_build_edge_counts"].item() == 1
    assert torch.equal(routed["graph_shell_edges"], torch.tensor([[-1, -1, -1], [-1, -1, -1]]))
    assert routed["graph_shell_edge_counts"].item() == 0
    assert torch.equal(routed["graph_diagnostic_edges"], torch.tensor([[3, 0, 60], [-1, -1, -1]]))
    assert routed["graph_diagnostic_edge_counts"].item() == 1
    assert torch.equal(routed["graph_cross_domain_edges"], torch.tensor([[4, 1, 62], [-1, -1, -1]]))
    assert routed["graph_cross_domain_edge_counts"].item() == 1
    assert torch.equal(routed["graph_chunk_starts"], torch.tensor([0, 1]))
    assert torch.equal(routed["graph_chunk_ends"], torch.tensor([1, 5]))
    assert torch.equal(routed["graph_chunk_kinds"], torch.tensor([1, 2]))
    assert torch.equal(routed["graph_chunk_dep_levels"], torch.tensor([0, 4]))
    assert routed["graph_chunk_counts"].item() == 2
