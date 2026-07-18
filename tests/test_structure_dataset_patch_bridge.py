import ast
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cppmega.megatron import structure_dataset_patch as patch  # noqa: E402
from cppmega.megatron.structure_batch import extract_structure_inputs  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPAQUE_VALUES = [2**63 + 17, 2**64 - 1]


def _load_pinned_megatron_tp_helper():
    candidates = [
        Path(os.environ["MEGATRON_LM_REPO"])
        if os.environ.get("MEGATRON_LM_REPO")
        else None,
        _REPO_ROOT.parent / "Megatron-LM",
        Path("/opt/megatron-lm"),
    ]
    megatron_root = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None and (candidate / ".git").exists()
        ),
        None,
    )
    if megatron_root is None:
        pytest.skip("pinned Megatron-LM git checkout is unavailable")

    stack_lock = (_REPO_ROOT / "STACK.lock").read_text(encoding="utf-8")
    assert "ref: core_v0.18.0" in stack_lock
    source = subprocess.run(
        [
            "git",
            "-C",
            str(megatron_root),
            "show",
            "core_v0.18.0:megatron/core/utils.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_batch_on_this_tp_rank"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<core_v0.18.0>", "exec"),
        namespace,
    )
    return namespace["get_batch_on_this_tp_rank"]


class _SequentialBroadcast:
    def __init__(self) -> None:
        self.receiving = False
        self.tensors: list[torch.Tensor] = []
        self.index = 0

    def __call__(self, tensor, src, group=None):
        del src, group
        if not self.receiving:
            self.tensors.append(tensor.detach().clone())
            return None
        expected = self.tensors[self.index]
        self.index += 1
        assert tensor.dtype == expected.dtype
        assert tensor.shape == expected.shape
        tensor.copy_(expected)
        return None


def _distributed_uint64_worker(
    rank: int,
    backend: str,
    init_file: str,
    result_dir: str,
) -> None:
    device = torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    source_sidecars = {}
    if rank == 0:
        source_sidecars = {
            "symbol_ids": torch.tensor([_OPAQUE_VALUES], dtype=torch.uint64),
            "call_targets": torch.tensor(
                [[_OPAQUE_VALUES[1], _OPAQUE_VALUES[0]]], dtype=torch.uint64
            ),
            "type_refs": torch.tensor([[0, _OPAQUE_VALUES[1]]], dtype=torch.uint64),
        }
    try:
        received = patch._broadcast_cppmega_sidecars(
            source_sidecars,
            batch={"tokens": torch.empty((1, 2), dtype=torch.long, device=device)},
            tp_rank=rank,
            broadcast_src_rank=0,
            broadcast_group=torch.distributed.group.WORLD,
        )
        result = {
            "ok": True,
            "dtype": str(received["symbol_ids"].dtype),
            "values": received["symbol_ids"].cpu().tolist(),
        }
    except RuntimeError as exc:
        result = {"ok": False, "error": str(exc)}
    finally:
        torch.distributed.destroy_process_group()
    Path(result_dir, f"{rank}.json").write_text(json.dumps(result), encoding="utf-8")


def _assert_distributed_uint64_results(tmp_path: Path) -> None:
    results = [
        json.loads((tmp_path / f"{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    if all(result["ok"] for result in results):
        assert all(result["dtype"] == "torch.uint64" for result in results)
        assert all(result["values"] == [_OPAQUE_VALUES] for result in results)
    else:
        assert not any(result["ok"] for result in results)
        assert all("refusing to narrow" in result["error"] for result in results)


def test_opaque_identity_tensor_preserves_full_uint64_bits():
    identity_id = 2**63 + 0x1234

    tensor = patch._token_sidecar_tensor(
        np.array([identity_id, 0], dtype=np.uint64),
        col="source_identity_ids",
    )

    assert tensor.dtype == torch.uint64
    assert tensor.tolist() == [identity_id, 0]

    with pytest.raises(ValueError, match="must arrive as uint64"):
        patch._token_sidecar_tensor(
            np.array([17], dtype=np.uint32),
            col="symbol_ids",
        )


def test_loss_mask_sidecar_requires_explicit_source_transition_alignment(tmp_path):
    prefix = tmp_path / "train"
    prefix.with_suffix(".bin").write_bytes(b"\0")
    loss_path = tmp_path / "train_loss_mask.bin"
    np.array([0], dtype=np.uint8).tofile(loss_path)
    manifest = {
        "side_channel_paths": {
            "loss_mask": {"path": loss_path.name, "dtype": "uint8"}
        }
    }
    manifest_path = prefix.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=f"{prefix}.bin"))

    with pytest.raises(ValueError, match="loss_mask_alignment"):
        patch._load_sidecar_manifest(dataset)

    manifest["loss_mask_alignment"] = "source_token_predicts_next_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _path, loaded = patch._load_sidecar_manifest(dataset)
    assert loaded["loss_mask_alignment"] == "source_token_predicts_next_v1"


def test_case5_training_loader_requires_success_receipt_and_registry(
    tmp_path,
    monkeypatch,
):
    from cppmega.megatron.domain_route_contract import (
        CASE5_RECEIPT_KEY,
        CASE5_SCHEMA_VERSION,
        DOMAIN_DELIMITER_CONTRACT_SHA256,
        SOURCE_IDENTITY_REGISTRY_SCHEMA,
    )

    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    prefix = tmp_path / "train"
    prefix.with_suffix(".bin").write_bytes(b"\0")
    route_dtypes = {
        "token_domain_ids": "uint16",
        "token_role_ids": "uint16",
        "token_entity_ids": "uint32",
        "token_scope_ids": "uint32",
        "token_source_doc_ids": "uint32",
        "token_source_identity_ids": "uint64",
        "token_confidence_ids": "uint8",
    }
    paths = {}
    for column, dtype in route_dtypes.items():
        path = tmp_path / f"train_{column}.bin"
        np.array([1], dtype=dtype).tofile(path)
        paths[column] = {"path": path.name, "dtype": dtype}
    registry_path = tmp_path / "train_source_identity_registry.sqlite"
    registry_path.write_bytes(b"receipt")
    manifest = {
        "side_channel_paths": paths,
        CASE5_RECEIPT_KEY: {
            "status": "success",
            "schema": CASE5_SCHEMA_VERSION,
            "delimiter_contract_sha256": DOMAIN_DELIMITER_CONTRACT_SHA256,
            "domain_schema_sha256": patch.DOMAIN_SCHEMA_SHA256,
            "tokenizer_contract_sha256": patch.TOKENIZER_CONTRACT_SHA256,
        },
        "source_identity_registry": {
            "schema": SOURCE_IDENTITY_REGISTRY_SCHEMA,
            "path": registry_path.name,
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(manifest))
    dataset = SimpleNamespace(dataset=SimpleNamespace(bin_path=f"{prefix}.bin"))

    loaded = patch._lazy_init_side_channels(dataset)

    assert loaded["token_source_identity_ids"]["dtype"] == np.dtype(np.uint64)

    del manifest[CASE5_RECEIPT_KEY]
    prefix.with_suffix(".json").write_text(json.dumps(manifest))
    unreceipted = SimpleNamespace(dataset=SimpleNamespace(bin_path=f"{prefix}.bin"))
    with pytest.raises(ValueError, match="successful case5_domain_ingestion_receipt"):
        patch._lazy_init_side_channels(unreceipted)


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


def test_pop_structure_batch_carries_objective_ids_for_unified_mix_receipt():
    batch = {
        "tokens": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, 4]]),
        "objective_ids": torch.tensor([[2, 4, 5]]),
    }

    structure = patch._pop_structure_batch(batch)

    assert structure is not None
    assert "objective_ids" not in batch
    assert torch.equal(structure["objective_ids"], torch.tensor([[2, 4, 5]]))


def test_sample_objective_ids_expand_document_aligned_ids_over_packed_spans():
    objective_ids = np.array([2, 5], dtype=np.uint8)
    spans = [
        {"real_doc": 0, "source_start": 1, "source_end": 3, "target_start": 0},
        {"real_doc": 1, "source_start": 0, "source_end": 2, "target_start": 2},
    ]

    sampled = patch._sample_objective_ids(
        objective_ids,
        spans,
        target_len=4,
    )

    assert sampled.tolist() == [2, 2, 5, 5]


def test_sample_objective_ids_clip_megatron_extra_source_token():
    objective_ids = np.array([2], dtype=np.uint8)
    spans = [
        {"real_doc": 0, "source_start": 0, "source_end": 5, "target_start": 0},
    ]

    sampled = patch._sample_objective_ids(
        objective_ids,
        spans,
        target_len=4,
    )

    assert sampled.tolist() == [2, 2, 2, 2]


def test_symbol_sidecar_tensor_preserves_unsigned_values_above_int64() -> None:
    tensor = patch._token_sidecar_tensor(
        np.array(_OPAQUE_VALUES, dtype=np.uint64),
        col="symbol_ids",
    )

    assert tensor.dtype == torch.uint64
    assert tensor.tolist() == _OPAQUE_VALUES


def test_padded_samples_keep_opaque_channels_uint64() -> None:
    tokens = torch.tensor([1, 2, 0, 0], dtype=torch.long)

    for col in patch._TOKEN_BATCH_COLS:
        sidecar = patch._padded_token_sidecar_tensor(tokens, col=col)
        expected_dtype = (
            torch.uint64 if col in patch._OPAQUE_UINT64_ID_COLS else torch.long
        )
        assert sidecar.dtype == expected_dtype
        assert sidecar.shape == tokens.shape
        assert sidecar.tolist() == [0, 0, 0, 0]


def test_pinned_megatron_dataloader_tp_bridge_preserves_opaque_ids(
    monkeypatch,
) -> None:
    pinned_helper = _load_pinned_megatron_tp_helper()
    bridge = patch._make_get_batch_on_this_tp_rank_bridge(pinned_helper)
    broadcasts = _SequentialBroadcast()
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setattr(torch.distributed, "broadcast", broadcasts)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    sample = {
        "tokens": torch.tensor([1, 2], dtype=torch.long),
        "labels": torch.tensor([2, 3], dtype=torch.long),
        "loss_mask": torch.ones(2, dtype=torch.float32),
        "position_ids": torch.arange(2, dtype=torch.long),
        "structure_ids": torch.tensor([4, 5], dtype=torch.long),
        "symbol_ids": torch.tensor(_OPAQUE_VALUES, dtype=torch.uint64),
        "call_targets": torch.tensor(
            [_OPAQUE_VALUES[1], _OPAQUE_VALUES[0]], dtype=torch.uint64
        ),
        "type_refs": torch.tensor([0, _OPAQUE_VALUES[1]], dtype=torch.uint64),
        "graph_call_edges": torch.tensor([[0, 1]], dtype=torch.long),
        "graph_call_edge_counts": torch.tensor(1, dtype=torch.long),
    }
    source_batch = next(iter(torch.utils.data.DataLoader([sample], batch_size=1)))
    common = {
        "is_sft": False,
        "is_hybrid_cp": False,
        "create_attention_mask_in_dataloader": False,
        "broadcast_src_rank": 0,
        "broadcast_group": object(),
        "cp_size": 1,
        "micro_batch_size": 1,
        "seq_length": 2,
        "mtp_on_this_rank": True,
        "pipeline_model_parallel_size": 1,
        "is_pipeline_first_stage": True,
        "is_pipeline_last_stage": True,
    }

    source_core = bridge(source_batch, tp_rank=0, **common)
    source_routes = patch._get_current_structure_batch()
    assert source_routes is not None
    assert not (set(source_core) & set(patch._CPPMEGA_BATCH_COLS))
    assert source_routes["symbol_ids"].dtype == torch.uint64
    assert source_routes["symbol_ids"].tolist() == [_OPAQUE_VALUES]

    broadcasts.receiving = True
    receiver_core = bridge({}, tp_rank=1, **common)
    receiver_routes = patch._get_current_structure_batch()
    assert receiver_routes is not None
    assert len(broadcasts.tensors) == 7  # Four Megatron plus three bridge collectives.
    assert broadcasts.index == len(broadcasts.tensors)
    assert not (set(receiver_core) & set(patch._CPPMEGA_BATCH_COLS))
    for key in ("symbol_ids", "call_targets", "type_refs"):
        assert receiver_routes[key].dtype == torch.uint64
        assert torch.equal(receiver_routes[key], source_routes[key])

    embedding_inputs = extract_structure_inputs(receiver_routes)
    assert embedding_inputs is not None
    assert embedding_inputs["structure_ids"].tolist() == [[4, 5]]
    assert not ({"symbol_ids", "call_targets", "type_refs"} & set(embedding_inputs))
    assert receiver_routes["graph_call_edges"].tolist() == [[[0, 1]]]
    assert receiver_routes["symbol_ids"].tolist() == [_OPAQUE_VALUES]


def test_uint64_transport_failure_refuses_to_narrow(monkeypatch) -> None:
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")

    def unsupported_broadcast(tensor, src, group=None):
        del src, group
        if tensor.dtype == torch.uint64:
            raise RuntimeError("Invalid scalar type")

    monkeypatch.setattr(torch.distributed, "broadcast", unsupported_broadcast)
    bridge = patch._make_get_batch_on_this_tp_rank_bridge(
        lambda batch, *, tp_rank, broadcast_src_rank, broadcast_group: batch
    )

    with pytest.raises(
        RuntimeError,
        match=r"cannot transport opaque uint64 sidecar 'symbol_ids'.*refusing to narrow",
    ):
        bridge(
            {
                "tokens": torch.tensor([[1, 2]], dtype=torch.long),
                "symbol_ids": torch.tensor([_OPAQUE_VALUES], dtype=torch.uint64),
            },
            tp_rank=0,
            broadcast_src_rank=0,
            broadcast_group=object(),
        )


def test_uint64_torch_coalescing_failure_refuses_to_narrow(monkeypatch) -> None:
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, src, group=None: None
    )
    original_cat = torch.cat

    def unsupported_uint64_cat(tensors, *args, **kwargs):
        if tensors and tensors[0].dtype == torch.uint64:
            raise RuntimeError("uint64 cat is unsupported")
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", unsupported_uint64_cat)
    bridge = patch._make_get_batch_on_this_tp_rank_bridge(
        lambda batch, *, tp_rank, broadcast_src_rank, broadcast_group: batch
    )

    with pytest.raises(
        RuntimeError,
        match=r"cannot transport opaque uint64 sidecar 'symbol_ids'.*refusing to narrow",
    ):
        bridge(
            {
                "tokens": torch.tensor([[1, 2]], dtype=torch.long),
                "symbol_ids": torch.tensor([_OPAQUE_VALUES], dtype=torch.uint64),
            },
            tp_rank=0,
            broadcast_src_rank=0,
            broadcast_group=object(),
        )


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_two_rank_gloo_uint64_transport_is_exact_or_fails_loudly(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _distributed_uint64_worker,
        args=("gloo", str(tmp_path / "gloo-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    _assert_distributed_uint64_results(tmp_path)


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available()
    or torch.cuda.device_count() < 2,
    reason="two-rank NCCL is unavailable",
)
def test_two_rank_nccl_uint64_transport_is_exact_or_fails_loudly(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _distributed_uint64_worker,
        args=("nccl", str(tmp_path / "nccl-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    _assert_distributed_uint64_results(tmp_path)


def test_pop_structure_batch_emits_h200_production_receipt(tmp_path, monkeypatch):
    receipt_path = tmp_path / "production-batch.json"
    monkeypatch.setenv("CPPMEGA_H200_BATCH_RECEIPT", str(receipt_path))
    batch = {
        "tokens": torch.tensor([[11, 12, 13, 14]]),
        "labels": torch.tensor([[12, 13, 14, 0]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "source_doc_ids": torch.tensor([[1, 1, 1, 1]]),
        "structure_ids": torch.tensor([[1, 2, 2, 1]]),
        "graph_call_edges": torch.tensor([[[0, 1]]]),
        "graph_call_edge_counts": torch.tensor([1]),
        "graph_type_edges": torch.tensor([[[-1, -1]]]),
        "graph_type_edge_counts": torch.tensor([0]),
        "graph_domain_edges": torch.tensor([[[-1, -1, -1]]]),
        "graph_domain_edge_counts": torch.tensor([0]),
        "graph_build_edges": torch.tensor([[[-1, -1, -1]]]),
        "graph_build_edge_counts": torch.tensor([0]),
        "graph_shell_edges": torch.tensor([[[-1, -1, -1]]]),
        "graph_shell_edge_counts": torch.tensor([0]),
        "graph_diagnostic_edges": torch.tensor([[[-1, -1, -1]]]),
        "graph_diagnostic_edge_counts": torch.tensor([0]),
        "graph_cross_domain_edges": torch.tensor([[[-1, -1, -1]]]),
        "graph_cross_domain_edge_counts": torch.tensor([0]),
        "graph_chunk_starts": torch.tensor([[0, 2]]),
        "graph_chunk_ends": torch.tensor([[2, 4]]),
        "graph_chunk_kinds": torch.tensor([[1, 2]]),
        "graph_chunk_dep_levels": torch.tensor([[0, 1]]),
        "graph_chunk_counts": torch.tensor([2]),
    }

    structure = patch._pop_structure_batch(batch)

    assert structure is not None
    assert set(batch) == {"tokens", "labels", "loss_mask"}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "verified"
    assert receipt["batch"]["tokens"]["shape"] == [1, 4]
    assert receipt["structure"]["structure_ids"]["nonzero"] == 4
    assert receipt["structure"]["graph_chunk_counts"]["sum"] == 2


def test_safe_sidecar_path_allows_plain_relative_and_blocks_escape():
    base = "/data/cppmega_sidecar"
    ok = patch._safe_sidecar_path(
        base, "train_token_ast_depth.bin", col="c", field="path", json_path="m.json"
    )
    assert ok == "/data/cppmega_sidecar/train_token_ast_depth.bin"

    for evil in ("../../etc/passwd", "/etc/passwd", "sub/../../escape.bin"):
        with pytest.raises(ValueError):
            patch._safe_sidecar_path(
                base, evil, col="c", field="path", json_path="m.json"
            )


def test_safe_sidecar_path_blocks_symlink_escape(tmp_path):
    base = tmp_path / "dataset"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"x")
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        patch._safe_sidecar_path(
            str(base), "link/secret.bin", col="c", field="path", json_path="m.json"
        )
    (base / "ok.bin").write_bytes(b"y")
    assert patch._safe_sidecar_path(
        str(base), "ok.bin", col="c", field="path", json_path="m.json"
    ).endswith("ok.bin")


def test_build_graph_route_tensors_offsets_caps_and_clips():
    graph_sidecars = {
        "token_call_edges": {
            "offsets": [0, 3],
            "data": torch.tensor([[0, 1], [1, 2], [2, 2]], dtype=torch.int32).numpy(),
        },
        "token_type_edges": {
            "offsets": [0, 2],
            "data": torch.tensor([[1, 0], [2, 1]], dtype=torch.int32).numpy(),
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

    assert torch.equal(routed["graph_call_edges"], torch.tensor([[0, 1], [-1, -1]]))
    assert routed["graph_call_edge_counts"].item() == 1
    assert torch.equal(routed["graph_type_edges"], torch.tensor([[1, 0], [-1, -1]]))
    assert routed["graph_type_edge_counts"].item() == 1
    assert torch.equal(
        routed["graph_domain_edges"], torch.tensor([[0, 3, 20], [-1, -1, -1]])
    )
    assert routed["graph_domain_edge_counts"].item() == 1
    assert torch.equal(
        routed["graph_build_edges"], torch.tensor([[1, 4, 21], [-1, -1, -1]])
    )
    assert routed["graph_build_edge_counts"].item() == 1
    assert torch.equal(
        routed["graph_shell_edges"], torch.tensor([[-1, -1, -1], [-1, -1, -1]])
    )
    assert routed["graph_shell_edge_counts"].item() == 0
    assert torch.equal(
        routed["graph_diagnostic_edges"], torch.tensor([[3, 0, 60], [-1, -1, -1]])
    )
    assert routed["graph_diagnostic_edge_counts"].item() == 1
    assert torch.equal(
        routed["graph_cross_domain_edges"], torch.tensor([[4, 1, 62], [-1, -1, -1]])
    )
    assert routed["graph_cross_domain_edge_counts"].item() == 1
    assert torch.equal(routed["graph_chunk_starts"], torch.tensor([0, 1]))
    assert torch.equal(routed["graph_chunk_ends"], torch.tensor([1, 5]))
    assert torch.equal(routed["graph_chunk_kinds"], torch.tensor([1, 2]))
    assert torch.equal(routed["graph_chunk_dep_levels"], torch.tensor([0, 4]))
    assert routed["graph_chunk_counts"].item() == 2


def test_build_graph_route_tensors_rejects_invalid_chunk_endpoint():
    graph_sidecars = {
        name: {
            "offsets": [0, 0],
            "data": torch.empty((0, width), dtype=torch.int32).numpy(),
        }
        for name, width in (
            ("token_type_edges", 2),
            ("token_domain_edges", 3),
            ("token_build_edges", 3),
            ("token_shell_edges", 3),
            ("token_diagnostic_edges", 3),
            ("token_cross_domain_edges", 3),
        )
    }
    graph_sidecars.update(
        {
            "token_call_edges": {
                "offsets": [0, 1],
                "data": torch.tensor([[7, 8]], dtype=torch.int32).numpy(),
            },
            "token_chunk_starts": {
                "offsets": [0, 1],
                "data": torch.tensor([0]).numpy(),
            },
            "token_chunk_ends": {"offsets": [0, 1], "data": torch.tensor([4]).numpy()},
            "token_chunk_kinds": {"offsets": [0, 1], "data": torch.tensor([1]).numpy()},
            "token_chunk_dep_levels": {
                "offsets": [0, 1],
                "data": torch.tensor([0]).numpy(),
            },
        }
    )

    with pytest.raises(ValueError, match="chunk endpoint out of range"):
        patch._build_graph_route_tensors(
            graph_sidecars,
            [
                {
                    "real_doc": 0,
                    "doc_start_token": 0,
                    "source_start": 0,
                    "source_end": 4,
                    "target_start": 0,
                }
            ],
            target_len=4,
            max_edges=2,
            max_chunks=2,
        )


def test_build_graph_route_tensors_rejects_capacity_truncation():
    empty_pairs = torch.empty((0, 2), dtype=torch.int32).numpy()
    empty_triples = torch.empty((0, 3), dtype=torch.int32).numpy()
    graph_sidecars = {
        "token_call_edges": {
            "offsets": [0, 1],
            "data": torch.tensor([[0, 1]], dtype=torch.int32).numpy(),
        },
        "token_type_edges": {"offsets": [0, 0], "data": empty_pairs},
        "token_domain_edges": {"offsets": [0, 0], "data": empty_triples},
        "token_build_edges": {"offsets": [0, 0], "data": empty_triples},
        "token_shell_edges": {"offsets": [0, 0], "data": empty_triples},
        "token_diagnostic_edges": {"offsets": [0, 0], "data": empty_triples},
        "token_cross_domain_edges": {"offsets": [0, 0], "data": empty_triples},
        "token_chunk_starts": {
            "offsets": [0, 2],
            "data": torch.tensor([0, 2], dtype=torch.int32).numpy(),
        },
        "token_chunk_ends": {
            "offsets": [0, 2],
            "data": torch.tensor([2, 4], dtype=torch.int32).numpy(),
        },
        "token_chunk_kinds": {
            "offsets": [0, 2],
            "data": torch.tensor([1, 1], dtype=torch.int32).numpy(),
        },
        "token_chunk_dep_levels": {
            "offsets": [0, 2],
            "data": torch.tensor([0, 0], dtype=torch.int32).numpy(),
        },
    }
    spans = [
        {
            "real_doc": 0,
            "doc_start_token": 0,
            "source_start": 0,
            "source_end": 4,
            "target_start": 0,
        }
    ]

    with pytest.raises(ValueError, match="chunk capacity exceeded"):
        patch._build_graph_route_tensors(
            graph_sidecars,
            spans,
            target_len=4,
            max_edges=2,
            max_chunks=1,
        )
    with pytest.raises(ValueError, match="edge capacity exceeded"):
        patch._build_graph_route_tensors(
            graph_sidecars,
            spans,
            target_len=4,
            max_edges=0,
            max_chunks=2,
        )
