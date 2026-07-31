import multiprocessing
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from cppmega.megatron import structure_dataset_patch
from cppmega.megatron.document_isolation import (
    _exchange_pipeline_document_ids,
    _patch_dsa_attention,
    _patch_model_input_transport,
    _patch_te_attention,
    _received_document_ids,
    bind_current_structure_batch,
    map_sequence_by_document,
    mask_sparse_topk_by_document,
    roll_tensor_by_document,
)


def _pipeline_worker(rank, init_file, results):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    activation = torch.zeros(4, 1, 2)
    communicator = SimpleNamespace(
        next_rank=1,
        prev_rank=0,
        pp_group=torch.distributed.group.WORLD,
    )
    try:
        if rank == 0:
            structure_dataset_patch._set_current_structure_batch(
                {"document_ids": torch.tensor([[1, 1, 2, 2]])}
            )
            _exchange_pipeline_document_ids(
                communicator,
                tensor_send_next=activation,
                tensor_recv_prev=None,
            )
        else:
            _exchange_pipeline_document_ids(
                communicator,
                tensor_send_next=None,
                tensor_recv_prev=activation,
            )
            results.put(_received_document_ids.pop(id(activation)).tolist())
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        torch.distributed.destroy_process_group()


def _pipeline_rejection_worker(rank, init_file, failure_mode, results):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    activation = torch.zeros(3 if rank == 1 and failure_mode == "shape" else 4, 1, 2)
    communicator = SimpleNamespace(
        next_rank=1,
        prev_rank=0,
        pp_group=torch.distributed.group.WORLD,
    )
    try:
        if rank == 0:
            if failure_mode == "shape":
                structure_dataset_patch._set_current_structure_batch(
                    {"document_ids": torch.tensor([[1, 1, 2, 2]])}
                )
            _exchange_pipeline_document_ids(
                communicator,
                tensor_send_next=activation,
                tensor_recv_prev=None,
            )
        else:
            _exchange_pipeline_document_ids(
                communicator,
                tensor_send_next=None,
                tensor_recv_prev=activation,
            )
    except Exception as exc:
        results.put((rank, type(exc).__name__, str(exc)))
    else:
        results.put((rank, "none", "document_ids rejection unexpectedly passed"))
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        torch.distributed.destroy_process_group()


def test_packed_documents_isolate_state_sparse_attention_and_mtp_loss():
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": torch.tensor([[1, 1, 2, 2]])}
    )
    try:
        hidden = torch.ones(4, 1, 1)
        assert map_sequence_by_document(
            hidden, lambda x: x.cumsum(dim=0)
        ).flatten().tolist() == [
            1,
            2,
            1,
            2,
        ]

        selected = torch.arange(4).repeat(1, 4, 1)
        isolated = mask_sparse_topk_by_document(selected)
        assert isolated.tolist() == [
            [[0, -1, -1, -1], [0, 1, -1, -1], [-1, -1, 2, -1], [-1, -1, 2, 3]]
        ]

        labels = torch.tensor([[10, 11, 20, 21]])
        rolled, _ = roll_tensor_by_document(
            labels,
            fallback=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("multi-document roll used the fallback")
            ),
        )
        assert rolled.tolist() == [[11, 0, 21, 0]]

        structure_dataset_patch._validate_document_loss_boundaries(
            torch.tensor([1, 1, 2, 2]),
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        with pytest.raises(ValueError, match="document transition"):
            structure_dataset_patch._validate_document_loss_boundaries(
                torch.tensor([1, 1, 2, 2]),
                torch.ones(4),
            )
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_stateful_documents_share_one_padded_kernel_launch():
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": torch.tensor([[1, 2, 2, 2], [1, 1, 2, 2]])}
    )
    calls = []
    try:

        def kernel(hidden):
            calls.append(tuple(hidden.shape))
            assert hidden.shape[0] % 4 == 0
            return hidden.cumsum(dim=0)

        hidden = torch.ones(4, 2, 1, requires_grad=True)
        actual = map_sequence_by_document(hidden, kernel, pad_to=4)
        assert calls == [(4, 4, 1)]
        assert actual[:, 0, 0].tolist() == [1, 1, 2, 3]
        assert actual[:, 1, 0].tolist() == [1, 2, 1, 2]
        actual.sum().backward()
        assert hidden.grad[:, :, 0].tolist() == [[1, 2], [3, 1], [2, 2], [1, 1]]
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_torch_attention_cannot_read_another_document(monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    class FakeGroup:
        def size(self):
            return 2

    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setattr(
        parallel_state,
        "get_global_memory_buffer",
        lambda: SimpleNamespace(
            get_tensor=lambda shape, dtype, _name: torch.empty(shape, dtype=dtype)
        ),
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=2,
        attention_dropout=0.0,
        masked_softmax_fusion=False,
        sequence_parallel=True,
        tensor_model_parallel_size=2,
        use_cpu_initialization=True,
    )
    attention = DotProductAttention(
        config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=FakeGroup()),
    )
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": torch.tensor([[1, 1, 2, 2]])}
    )
    try:
        query = torch.ones(4, 1, 1, 2)
        value = (
            torch.tensor([1.0, 2.0, 10.0, 20.0])
            .view(4, 1, 1, 1)
            .expand(-1, -1, -1, 2)
            .contiguous()
        )
        output = attention(query, query, value, None)
        torch.testing.assert_close(
            output[:, 0, 0],
            torch.tensor([1.0, 1.5, 10.0, 15.0]),
        )
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_te_attention_receives_document_packed_thd_boundaries(monkeypatch):
    from megatron.core.extensions import transformer_engine as te_module
    from megatron.core.transformer.enums import AttnMaskType

    calls = []

    class FakeTEAttention:
        def __init__(self):
            self.config = SimpleNamespace(context_parallel_size=1)

        def forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=None,
            packed_seq_params=None,
        ):
            calls.append((query, key, value, attention_mask, packed_seq_params))
            if query.dim() == 3:
                return query.reshape(query.shape[0], -1)
            return query.reshape(query.shape[0], query.shape[1], -1)

    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setattr(te_module, "TEDotProductAttention", FakeTEAttention)
    _patch_te_attention()
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": torch.tensor([[1, 1, 2, 2], [1, 2, 2, 0]])}
    )
    try:
        query = torch.arange(16.0).view(4, 2, 1, 2)
        output = FakeTEAttention().forward(
            query,
            query,
            query,
            None,
            AttnMaskType.causal,
        )
        packed = calls[0][4]
        assert packed.qkv_format == "thd"
        assert packed.cu_seqlens_q.tolist() == [0, 2, 4, 5, 7, 8]
        assert calls[0][3] is None
        torch.testing.assert_close(output, query.reshape(4, 2, 2))

        calls.clear()
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": torch.tensor([[1, 1, 2, 2]])}
        )
        query = torch.arange(8.0).view(4, 1, 1, 2)
        output = FakeTEAttention().forward(
            query,
            query,
            query,
            None,
            AttnMaskType.causal,
            attention_bias=torch.zeros(1, 1, 4, 4),
        )
        assert len(calls) == 2
        assert [tuple(call[0].shape) for call in calls] == [
            (2, 1, 1, 2),
            (2, 1, 1, 2),
        ]
        torch.testing.assert_close(output, query.reshape(4, 1, 2))
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_dsa_indexer_receives_document_causal_mask(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant import (
        dsa as dsa_module,
    )
    from megatron.core.transformer.enums import AttnMaskType

    calls = []

    class FakeDSAttention:
        def forward(
            self,
            query,
            key,
            value,
            attention_mask,
            x,
            qr,
            attn_mask_type=None,
            attention_bias=None,
            packed_seq_params=None,
        ):
            calls.append((attention_mask, attn_mask_type))
            return query

    def sparse_backend(*_args):
        raise AssertionError("fake DSAttention should not call the backend")

    setattr(sparse_backend, "__cppmega_document_isolation__", True)
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setattr(dsa_module, "DSAttention", FakeDSAttention)
    monkeypatch.setattr(dsa_module, "unfused_dsa_fn", sparse_backend)
    _patch_dsa_attention()
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": torch.tensor([[1, 1, 2, 2]])}
    )
    try:
        query = torch.ones(4, 1, 1, 2)
        FakeDSAttention().forward(
            query,
            query,
            query,
            None,
            query.reshape(4, 1, 2),
            query.reshape(4, 1, 2),
            AttnMaskType.causal,
        )
        mask, mask_type = calls[0]
        assert tuple(mask.shape) == (1, 1, 4, 4)
        assert mask_type is None
        assert mask[0, 0, 2, 0]
        assert not mask[0, 0, 2, 2]
        assert mask[0, 0, 2, 3]
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_pipeline_model_input_restores_received_document_ids(monkeypatch):
    class Model:
        def set_input_tensor(self, input_tensor):
            self.input_tensor = input_tensor

    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    _patch_model_input_transport(Model)
    activation = torch.zeros(4, 1, 2)
    document_ids = torch.tensor([[1, 1, 2, 2]])
    _received_document_ids[id(activation)] = document_ids
    try:
        model = Model()
        input_tensors = [activation]
        model.set_input_tensor(input_tensors)
        assert model.input_tensor is input_tensors
        assert (
            structure_dataset_patch._get_current_structure_batch()["document_ids"]
            is document_ids
        )
    finally:
        _received_document_ids.pop(id(activation), None)
        structure_dataset_patch._set_current_structure_batch(None)


def test_checkpoint_callable_restores_its_microbatch_document_ids():
    first = torch.tensor([[1, 1, 2, 2]])
    second = torch.tensor([[1, 1, 1, 1]])
    structure_dataset_patch._set_current_structure_batch({"document_ids": first})
    try:
        bound = bind_current_structure_batch(
            lambda: structure_dataset_patch._get_current_structure_batch()[
                "document_ids"
            ]
        )
        structure_dataset_patch._set_current_structure_batch({"document_ids": second})
        assert bound() is first
        assert (
            structure_dataset_patch._get_current_structure_batch()["document_ids"]
            is second
        )
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_checkpoint_backward_recomputes_with_the_originating_documents():
    first = torch.tensor([[1, 1, 2, 2]])
    structure_dataset_patch._set_current_structure_batch({"document_ids": first})
    hidden = torch.arange(1.0, 5.0).view(4, 1, 1).requires_grad_()
    try:

        def forward(value):
            return map_sequence_by_document(
                value,
                lambda segment: segment.sin().cumsum(dim=0),
            )

        output = checkpoint(
            bind_current_structure_batch(forward),
            hidden,
            use_reentrant=False,
        )
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": torch.tensor([[1, 1, 1, 1]])}
        )
        output.sum().backward()

        expected = torch.tensor(
            [
                2 * torch.cos(torch.tensor(1.0)),
                torch.cos(torch.tensor(2.0)),
                2 * torch.cos(torch.tensor(3.0)),
                torch.cos(torch.tensor(4.0)),
            ]
        )
        torch.testing.assert_close(hidden.grad.flatten(), expected)
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_document_ids_cross_a_real_pipeline_process_group(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_pipeline_worker,
            args=(rank, str(tmp_path / "gloo-init"), results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(25)
        assert process.exitcode == 0
    assert results.get(timeout=1) == [[1, 1, 2, 2]]


@pytest.mark.parametrize(
    ("failure_mode", "expected_messages"),
    (
        (
            "missing",
            {
                0: "pipeline activation send requires document_ids",
                1: "upstream pipeline stage rejected its document_ids",
            },
        ),
        (
            "shape",
            {
                0: "downstream pipeline stage rejected document_ids metadata",
                1: "is incompatible with activation",
            },
        ),
    ),
)
def test_pipeline_document_id_rejection_is_coordinated(
    tmp_path, failure_mode, expected_messages
):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    init_file = tmp_path / f"gloo-{failure_mode}-init"
    processes = [
        context.Process(
            target=_pipeline_rejection_worker,
            args=(rank, str(init_file), failure_mode, results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
    messages = [results.get(timeout=1) for _ in range(2)]

    assert all(process.exitcode == 0 for process in processes), messages
    assert {rank for rank, _error_type, _message in messages} == {0, 1}
    for rank, error_type, message in messages:
        assert error_type in {"RuntimeError", "ValueError"}
        assert expected_messages[rank] in message
