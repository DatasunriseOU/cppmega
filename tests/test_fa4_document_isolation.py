"""CPU regressions for FA4 packed-document aux construction."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest
import torch

from cppmega.megatron.fa4_graph_attention import (
    FA4GraphRouteAux,
    CppMegaFA4DotProductAttention,
)
from cppmega.megatron.fa4_score_mod_adapter import _build_document_mask_aux
from cppmega.megatron.structure_dataset_patch import (
    _get_current_structure_batch,
    _set_current_structure_batch,
)


@contextmanager
def _structure_batch(batch: dict[str, torch.Tensor]) -> Iterator[None]:
    previous = _get_current_structure_batch()
    _set_current_structure_batch(batch)
    try:
        yield
    finally:
        _set_current_structure_batch(previous)


def test_multi_document_aux_uses_singleton_head_axis() -> None:
    document_ids = torch.tensor(
        [
            [1, 1, 1, 2, 2, 2, 0, 0],
            [1, 1, 2, 2, 2, 2, 2, 0],
        ],
        dtype=torch.long,
    )
    with _structure_batch({"document_ids": document_ids}):
        built = _build_document_mask_aux(
            batch_size=2,
            seqlen_q=8,
            seqlen_k=8,
            device=torch.device("cpu"),
        )

    assert built is not None
    document_ids_q, document_ids_k = built
    assert document_ids_q.shape == (2, 1, 8)
    assert document_ids_k.shape == (2, 1, 8)
    assert document_ids_q.dtype == torch.int32
    assert document_ids_k.dtype == torch.int32
    assert document_ids_q.is_contiguous()
    assert document_ids_k.is_contiguous()
    assert document_ids_q[:, 0].tolist() == document_ids.tolist()
    assert document_ids_k[:, 0].tolist() == document_ids.tolist()


def test_rectangular_decode_aux_uses_absolute_query_tail() -> None:
    document_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.long)
    with _structure_batch({"document_ids": document_ids}):
        built = _build_document_mask_aux(
            batch_size=1,
            seqlen_q=2,
            seqlen_k=8,
            device=torch.device("cpu"),
        )

    assert built is not None
    document_ids_q, document_ids_k = built
    assert document_ids_q.shape == (1, 1, 2)
    assert document_ids_k.shape == (1, 1, 8)
    assert document_ids_q.tolist() == [[[2, 2]]]
    assert document_ids_k.tolist() == [[[1, 1, 1, 1, 2, 2, 2, 2]]]


def test_single_document_keeps_native_fa4_fast_path() -> None:
    with _structure_batch({"document_ids": torch.ones((2, 129), dtype=torch.int32)}):
        built = _build_document_mask_aux(
            batch_size=2,
            seqlen_q=129,
            seqlen_k=129,
            device=torch.device("cpu"),
        )

    assert built is None


def test_legacy_graph_route_aux_fails_closed_for_packed_documents() -> None:
    sequence_length, heads, head_dim = 4, 2, 8
    document_ids = torch.tensor([[1, 1, 2, 2]], dtype=torch.int32)
    graph_aux = FA4GraphRouteAux(
        csr_row_offsets=torch.zeros((1, sequence_length + 1), dtype=torch.int32),
        csr_col_idx=torch.zeros((1, 1), dtype=torch.int32),
        csr_weight=torch.zeros((1, 1), dtype=torch.float32),
        csr_meta=torch.tensor(
            [sequence_length, sequence_length, 1, 0],
            dtype=torch.int32,
        ),
    )
    query = torch.randn(sequence_length, 1, heads, head_dim)
    module = CppMegaFA4DotProductAttention(num_attention_heads=heads)

    with _structure_batch({"document_ids": document_ids}):
        with pytest.raises(
            RuntimeError,
            match="not beta23-compatible with packed-document mask aux",
        ):
            module(query, query, query, attention_bias=graph_aux)
