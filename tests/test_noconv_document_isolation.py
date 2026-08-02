from __future__ import annotations

from datetime import timedelta
import multiprocessing
from pathlib import Path
import traceback
from typing import Any

import pytest
import torch
from torch import nn

from cppmega.megatron import structure_dataset_patch
from cppmega.megatron.document_isolation import map_sharded_sequence_by_document


class _IdentityProjection(nn.Module):
    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, None]:
        return tensor, None


class _StatefulScanHarness(nn.Module):
    """Exercise the production post-projection seam with a CPU stateful scan."""

    def __init__(
        self,
        *,
        context_parallel_group: Any = None,
        chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = _IdentityProjection()
        self.out_proj = _IdentityProjection()
        self.context_parallel_group = context_parallel_group
        self.chunk_size = chunk_size
        self.scale = nn.Parameter(torch.tensor(0.75))

    def _ssm_noconv(self, projected: torch.Tensor) -> torch.Tensor:
        return projected.cumsum(dim=0) * self.scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_context: Any = None,
        *,
        inference_params: Any = None,
        packed_seq_params: Any = None,
    ) -> tuple[torch.Tensor, None]:
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError
        if packed_seq_params is not None:
            raise NotImplementedError(
                "NoConvMambaMixer does not support packed sequences yet"
            )
        projected, _ = self.in_proj(hidden_states)
        mapped = map_sharded_sequence_by_document(
            projected,
            self._ssm_noconv,
            context_parallel_group=self.context_parallel_group,
            pad_to=self.chunk_size,
        )
        return self.out_proj(mapped)


def _reference(
    values: torch.Tensor,
    weights: torch.Tensor,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_values = values.detach().clone().requires_grad_(True)
    scale = torch.tensor(0.75, requires_grad=True)
    parts = []
    row = document_ids[0]
    start = 0
    for position in range(1, row.numel() + 1):
        if position == row.numel() or row[position] != row[start]:
            parts.append(reference_values[start:position].cumsum(dim=0) * scale)
            start = position
    output = torch.cat(parts, dim=0)
    (output * weights).sum().backward()
    assert reference_values.grad is not None
    assert scale.grad is not None
    return output.detach(), reference_values.grad.detach(), scale.grad.detach()


def test_noconv_forward_resets_state_and_gradients_at_document_boundaries() -> None:
    document_ids = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3]])
    values = torch.arange(1.0, 9.0).view(8, 1, 1).requires_grad_(True)
    weights = torch.linspace(0.25, 2.0, 8).view(8, 1, 1)
    expected, expected_input_grad, expected_scale_grad = _reference(
        values,
        weights,
        document_ids,
    )
    module = _StatefulScanHarness(chunk_size=4)

    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": document_ids}
    )
    try:
        output, output_bias = module(values)
        assert output_bias is None
        (output * weights).sum().backward()
    finally:
        structure_dataset_patch._set_current_structure_batch(None)

    torch.testing.assert_close(output, expected)
    assert values.grad is not None
    torch.testing.assert_close(values.grad, expected_input_grad)
    torch.testing.assert_close(module.scale.grad, expected_scale_grad)


def test_noconv_forward_prevents_cross_document_perturbation() -> None:
    document_ids = torch.tensor([[1, 1, 2, 2, 2]])
    values = torch.arange(1.0, 6.0).view(5, 1, 1)
    module = _StatefulScanHarness(chunk_size=4)

    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": document_ids}
    )
    try:
        baseline, _ = module(values)
        perturbed = values.clone()
        perturbed[:2].add_(1000)
        changed, _ = module(perturbed)
    finally:
        structure_dataset_patch._set_current_structure_batch(None)

    torch.testing.assert_close(changed[2:], baseline[2:], atol=0, rtol=0)


def test_noconv_forward_keeps_upstream_packed_sequences_fail_closed() -> None:
    module = _StatefulScanHarness()
    with pytest.raises(NotImplementedError, match="packed sequences"):
        module(torch.ones(4, 1, 1), packed_seq_params=object())


def _balance_context_parallel(
    tensor: torch.Tensor,
    *,
    cp_size: int,
    undo: bool,
) -> torch.Tensor:
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    function = (
        _undo_attention_load_balancing
        if undo
        else _redo_attention_load_balancing
    )
    return function(tensor, cp_size, packed_seq_params=None)


def _context_parallel_worker(
    rank: int,
    init_method: str,
    results: Any,
) -> None:
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=60),
        )
        group = torch.distributed.group.WORLD
        document_ids = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3]])
        values = torch.arange(1.0, 9.0).view(8, 1, 1)
        weights = torch.linspace(0.25, 2.0, 8).view(8, 1, 1)
        expected, expected_input_grad, expected_scale_grad = _reference(
            values,
            weights,
            document_ids,
        )

        balanced_values = _balance_context_parallel(
            values,
            cp_size=2,
            undo=False,
        )
        balanced_weights = _balance_context_parallel(
            weights,
            cp_size=2,
            undo=False,
        )
        local_values = (
            balanced_values.chunk(2, dim=0)[rank]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        local_weights = balanced_weights.chunk(2, dim=0)[rank]
        module = _StatefulScanHarness(context_parallel_group=group)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        output, _ = module(local_values)
        (output * local_weights).sum().backward()
        assert local_values.grad is not None
        assert module.scale.grad is not None
        torch.distributed.all_reduce(module.scale.grad, group=group)

        output_parts = [torch.empty_like(output) for _ in range(2)]
        input_grad_parts = [torch.empty_like(local_values.grad) for _ in range(2)]
        torch.distributed.all_gather(output_parts, output.detach(), group=group)
        torch.distributed.all_gather(
            input_grad_parts,
            local_values.grad.detach(),
            group=group,
        )
        full_output = _balance_context_parallel(
            torch.cat(output_parts),
            cp_size=2,
            undo=True,
        )
        full_input_grad = _balance_context_parallel(
            torch.cat(input_grad_parts),
            cp_size=2,
            undo=True,
        )
        torch.testing.assert_close(full_output, expected)
        torch.testing.assert_close(full_input_grad, expected_input_grad)
        torch.testing.assert_close(module.scale.grad, expected_scale_grad)
        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_noconv_context_parallel_forward_backward_matches_unsharded(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    init_method = f"file://{tmp_path / 'noconv-cp-init'}"
    processes = [
        context.Process(
            target=_context_parallel_worker,
            args=(rank, init_method, results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
    messages = [results.get(timeout=5) for _ in range(2)]
    assert all(process.exitcode == 0 for process in processes), messages
    assert sorted(messages) == [("ok", 0), ("ok", 1)]
