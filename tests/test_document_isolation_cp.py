import multiprocessing
import socket
import traceback
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from cppmega.megatron import structure_dataset_patch
from cppmega.megatron.document_isolation import (
    _exchange_pipeline_document_ids,
    _received_document_ids,
    _validate_model_parallel_topology,
    map_sequence_by_document,
    map_sharded_sequence_by_document,
    roll_tensor_by_document,
)


def _reference(
    values: torch.Tensor,
    weights: torch.Tensor,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = values.detach().clone().requires_grad_(True)
    scale = torch.tensor(0.75, device=values.device, requires_grad=True)
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": document_ids}
    )
    output = map_sequence_by_document(
        source, lambda segment: segment.cumsum(dim=0) * scale
    )
    (output * weights).sum().backward()
    if source.grad is None or scale.grad is None:
        raise RuntimeError("reference backward did not populate gradients")
    return output.detach(), source.grad.detach(), scale.grad.detach()


def _balanced(tensor: torch.Tensor, cp_size: int, *, undo: bool) -> torch.Tensor:
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    function = (
        _undo_attention_load_balancing if undo else _redo_attention_load_balancing
    )
    return function(tensor, cp_size, packed_seq_params=None)


def _distributed_worker(
    rank: int,
    backend: str,
    init_method: str,
    results,
) -> None:
    try:
        device = torch.device("cpu")
        if backend == "nccl":
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        group = torch.distributed.group.WORLD
        aliased_group = torch.distributed.new_group([0, 1])
        with pytest.raises(ValueError, match="distinct Cartesian TP/CP axes"):
            _validate_model_parallel_topology(
                SimpleNamespace(
                    tensor_model_parallel_size=2,
                    context_parallel_size=2,
                ),
                tp_group=group,
                cp_group=aliased_group,
                component="test mixer",
            )
        with pytest.raises(
            ValueError,
            match=r"tensor_model_parallel_size=1.*world_size=2",
        ):
            _validate_model_parallel_topology(
                SimpleNamespace(
                    tensor_model_parallel_size=1,
                    context_parallel_size=1,
                ),
                tp_group=group,
                cp_group=None,
                component="test mixer",
            )
        document_ids = torch.tensor(
            [[1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.long, device=device
        )
        values = torch.arange(
            1.0, 9.0, device=device, dtype=torch.float32
        ).view(8, 1, 1)
        weights = torch.linspace(
            0.25, 2.0, 8, device=device, dtype=torch.float32
        ).view(8, 1, 1)
        expected, expected_grad, expected_scale_grad = _reference(
            values, weights, document_ids
        )

        # TP sequence parallel uses contiguous sequence shards. The stateful
        # module is replicated, so output-gradient gather gives every replica
        # the same complete parameter gradient.
        local_values = values.chunk(2, dim=0)[rank].detach().clone().requires_grad_(True)
        local_weights = weights.chunk(2, dim=0)[rank]
        scale = torch.tensor(0.75, device=device, requires_grad=True)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        local_output = map_sharded_sequence_by_document(
            local_values,
            lambda segment: segment.cumsum(dim=0) * scale,
            sequence_parallel_group=group,
        )
        (local_output * local_weights).sum().backward()
        if local_values.grad is None or scale.grad is None:
            raise RuntimeError("SP backward did not populate gradients")
        output_parts = [torch.empty_like(local_output) for _ in range(2)]
        grad_parts = [torch.empty_like(local_values.grad) for _ in range(2)]
        torch.distributed.all_gather(output_parts, local_output.detach())
        torch.distributed.all_gather(grad_parts, local_values.grad.detach())
        torch.testing.assert_close(torch.cat(output_parts), expected)
        torch.testing.assert_close(torch.cat(grad_parts), expected_grad)
        torch.testing.assert_close(scale.grad, expected_scale_grad)

        torch.distributed.barrier()

        # Megatron CP stores two zigzag chunks on each rank. CP parameter
        # gradients are partial here and are summed by the DP+CP reducer.
        balanced_values = _balanced(values, 2, undo=False)
        balanced_weights = _balanced(weights, 2, undo=False)
        local_values = (
            balanced_values.chunk(2, dim=0)[rank]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        local_weights = balanced_weights.chunk(2, dim=0)[rank]
        scale = torch.tensor(0.75, device=device, requires_grad=True)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        local_output = map_sharded_sequence_by_document(
            local_values,
            lambda segment: segment.cumsum(dim=0) * scale,
            context_parallel_group=group,
        )
        (local_output * local_weights).sum().backward()
        if local_values.grad is None or scale.grad is None:
            raise RuntimeError("CP backward did not populate gradients")
        torch.distributed.all_reduce(scale.grad, group=group)

        output_parts = [torch.empty_like(local_output) for _ in range(2)]
        grad_parts = [torch.empty_like(local_values.grad) for _ in range(2)]
        torch.distributed.all_gather(output_parts, local_output.detach())
        torch.distributed.all_gather(grad_parts, local_values.grad.detach())
        output_full = _balanced(torch.cat(output_parts), 2, undo=True)
        grad_full = _balanced(torch.cat(grad_parts), 2, undo=True)
        torch.testing.assert_close(output_full, expected)
        torch.testing.assert_close(grad_full, expected_grad)
        torch.testing.assert_close(scale.grad, expected_scale_grad)

        torch.distributed.barrier()

        # Pipeline activations are locally sequence-sharded, while the sidecar
        # remains global and is transmitted with its explicit shape.
        activation = torch.zeros(4, 1, 2, device=device)
        communicator = SimpleNamespace(
            next_rank=1,
            prev_rank=0,
            pp_group=group,
        )
        if rank == 0:
            structure_dataset_patch._set_current_structure_batch(
                {"document_ids": document_ids}
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
            received = _received_document_ids.pop(id(activation))
            torch.testing.assert_close(received, document_ids)

        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_world(backend: str, init_method: str) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, backend, init_method, results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
    messages = [results.get(timeout=5) for _ in range(2)]
    assert all(process.exitcode == 0 for process in processes), messages
    assert messages == [("ok", 0), ("ok", 1)] or messages == [
        ("ok", 1),
        ("ok", 0),
    ]


def _mtp_roll_worker(rank: int, init_method: str, results) -> None:
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        cp_group = torch.distributed.group.WORLD
        document_ids = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3]])

        def reject_fallback(*_args, **_kwargs):
            raise AssertionError("packed-document CP roll used the fallback")

        for sequence_dim, values, weights in (
            (
                0,
                torch.arange(1.0, 9.0).view(8, 1, 1),
                torch.linspace(0.25, 2.0, 8).view(8, 1, 1),
            ),
            (
                1,
                torch.arange(1.0, 9.0).view(1, 8),
                torch.linspace(0.25, 2.0, 8).view(1, 8),
            ),
        ):
            structure_dataset_patch._set_current_structure_batch(
                {"document_ids": document_ids}
            )
            reference_input = values.detach().clone().requires_grad_(True)
            reference_rolled, reference_sum = roll_tensor_by_document(
                reference_input,
                shifts=-1,
                dims=sequence_dim,
                fallback=reject_fallback,
            )
            ((reference_rolled * weights).sum() + 0.25 * reference_sum).backward()
            if reference_input.grad is None:
                raise RuntimeError("MTP reference backward did not populate gradients")

            balanced_input = _balanced(
                values.movedim(sequence_dim, 0),
                2,
                undo=False,
            )
            local_input = (
                balanced_input.chunk(2, dim=0)[rank]
                .movedim(0, sequence_dim)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            local_weights = (
                _balanced(weights.movedim(sequence_dim, 0), 2, undo=False)
                .chunk(2, dim=0)[rank]
                .movedim(0, sequence_dim)
            )
            expected_local = (
                _balanced(
                    reference_rolled.detach().movedim(sequence_dim, 0),
                    2,
                    undo=False,
                )
                .chunk(2, dim=0)[rank]
                .movedim(0, sequence_dim)
            )

            actual, local_sum = roll_tensor_by_document(
                local_input,
                shifts=-1,
                dims=sequence_dim,
                cp_group=cp_group,
                fallback=reject_fallback,
            )
            torch.testing.assert_close(actual, expected_local)
            torch.testing.assert_close(local_sum, expected_local.sum())
            ((actual * local_weights).sum() + 0.25 * local_sum).backward()
            if local_input.grad is None:
                raise RuntimeError("MTP CP backward did not populate gradients")

            expected_local_grad = (
                _balanced(
                    reference_input.grad.movedim(sequence_dim, 0),
                    2,
                    undo=False,
                )
                .chunk(2, dim=0)[rank]
                .movedim(0, sequence_dim)
            )
            torch.testing.assert_close(local_input.grad, expected_local_grad)

        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _combined_worker(rank: int, init_method: str, results) -> None:
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=4,
            timeout=timedelta(seconds=90),
        )
        tp_group: Any = None
        cp_group: Any = None
        for ranks in ([0, 1], [2, 3]):
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                tp_group = group
        for ranks in ([0, 2], [1, 3]):
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                cp_group = group
        if tp_group is None or cp_group is None:
            raise RuntimeError("rank was not assigned to TP and CP groups")

        document_ids = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 3]])
        values = torch.arange(1.0, 9.0).view(8, 1, 1)
        weights = torch.linspace(0.25, 2.0, 8).view(8, 1, 1)
        expected, expected_grad, expected_scale_grad = _reference(
            values, weights, document_ids
        )

        cp_rank = torch.distributed.get_rank(group=cp_group)
        tp_rank = torch.distributed.get_rank(group=tp_group)
        balanced_values = _balanced(values, 2, undo=False)
        balanced_weights = _balanced(weights, 2, undo=False)
        local_values = (
            balanced_values.chunk(2, dim=0)[cp_rank]
            .chunk(2, dim=0)[tp_rank]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        local_weights = (
            balanced_weights.chunk(2, dim=0)[cp_rank].chunk(2, dim=0)[tp_rank]
        )
        scale = torch.tensor(0.75, requires_grad=True)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        local_output = map_sharded_sequence_by_document(
            local_values,
            lambda segment: segment.cumsum(dim=0) * scale,
            sequence_parallel_group=tp_group,
            context_parallel_group=cp_group,
        )
        (local_output * local_weights).sum().backward()
        if local_values.grad is None or scale.grad is None:
            raise RuntimeError("combined SP/CP backward did not populate gradients")
        torch.distributed.all_reduce(scale.grad, group=cp_group)

        output_tp = [torch.empty_like(local_output) for _ in range(2)]
        grad_tp = [torch.empty_like(local_values.grad) for _ in range(2)]
        torch.distributed.all_gather(output_tp, local_output.detach(), group=tp_group)
        torch.distributed.all_gather(
            grad_tp, local_values.grad.detach(), group=tp_group
        )
        cp_local_output = torch.cat(output_tp)
        cp_local_grad = torch.cat(grad_tp)
        output_cp = [torch.empty_like(cp_local_output) for _ in range(2)]
        grad_cp = [torch.empty_like(cp_local_grad) for _ in range(2)]
        torch.distributed.all_gather(output_cp, cp_local_output, group=cp_group)
        torch.distributed.all_gather(grad_cp, cp_local_grad, group=cp_group)

        torch.testing.assert_close(
            _balanced(torch.cat(output_cp), 2, undo=True), expected
        )
        torch.testing.assert_close(
            _balanced(torch.cat(grad_cp), 2, undo=True), expected_grad
        )
        torch.testing.assert_close(scale.grad, expected_scale_grad)
        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _m2rnn_config(*, sequence_parallel: bool, context_parallel_size: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
        ffn_hidden_size=16,
        tensor_model_parallel_size=2 if sequence_parallel else 1,
        pipeline_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    )
    for name, value in (
        ("m2rnn_kernel", "torch"),
        ("m2rnn_k_head_dim", 2),
        ("m2rnn_v_head_dim", 2),
        ("m2rnn_conv_kernel", 0),
        ("m2rnn_A_init_min", 1.0),
        ("m2rnn_A_init_max", 2.0),
        ("m2rnn_num_q_heads", 1),
        ("m2rnn_num_k_heads", 1),
        ("m2rnn_num_v_heads", 2),
        ("m2rnn_num_f_heads", 2),
        ("m2rnn_num_g_heads", 2),
        ("m2rnn_num_weight_heads", 2),
    ):
        object.__setattr__(config, name, value)
    return config


def _m2rnn_parameter_grads(module) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"M2RNN parameter {name} has no gradient")
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _m2rnn_integration_worker(
    rank: int,
    init_method: str,
    results,
) -> None:
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=90),
        )
        group = torch.distributed.group.WORLD

        from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer

        torch.manual_seed(20260731)
        reference = CppMegaM2RNNMixer(
            config=_m2rnn_config(
                sequence_parallel=False,
                context_parallel_size=1,
            ),
            d_model=8,
        ).to(device)
        document_ids = torch.tensor(
            [[1, 1, 2, 2, 2, 3, 3, 3]],
            dtype=torch.long,
            device=device,
        )
        values = torch.linspace(
            -0.75,
            0.75,
            8 * 8,
            dtype=torch.float32,
            device=device,
        ).view(8, 1, 8)
        weights = torch.linspace(
            0.25,
            1.5,
            8 * 8,
            dtype=torch.float32,
            device=device,
        ).view(8, 1, 8)
        reference_input = values.detach().clone().requires_grad_(True)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        reference_output, _ = reference(reference_input)
        (reference_output * weights).sum().backward()
        if reference_input.grad is None:
            raise RuntimeError("M2RNN reference input has no gradient")
        expected_input_grad = reference_input.grad.detach()
        expected_parameter_grads = _m2rnn_parameter_grads(reference)

        sp_model = CppMegaM2RNNMixer(
            config=_m2rnn_config(
                sequence_parallel=True,
                context_parallel_size=1,
            ),
            d_model=8,
            pg_collection=SimpleNamespace(tp=group, cp=None),
        ).to(device)
        sp_model.load_state_dict(reference.state_dict())
        for parameter in getattr(sp_model.g_norm, "parameters")():
            if getattr(parameter, "sequence_parallel", False):
                raise AssertionError(
                    "M2RNN gathered SP norm must not be reduced twice"
                )
        sp_input = (
            values.chunk(2, dim=0)[rank]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        sp_weights = weights.chunk(2, dim=0)[rank]
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        sp_output, _ = sp_model(sp_input)
        (sp_output * sp_weights).sum().backward()
        if sp_input.grad is None:
            raise RuntimeError("M2RNN SP input has no gradient")
        sp_outputs = [torch.empty_like(sp_output) for _ in range(2)]
        sp_input_grads = [torch.empty_like(sp_input.grad) for _ in range(2)]
        torch.distributed.all_gather(sp_outputs, sp_output.detach())
        torch.distributed.all_gather(sp_input_grads, sp_input.grad.detach())
        torch.testing.assert_close(
            torch.cat(sp_outputs),
            reference_output,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            torch.cat(sp_input_grads),
            expected_input_grad,
            atol=2e-5,
            rtol=2e-5,
        )
        for name, actual in _m2rnn_parameter_grads(sp_model).items():
            torch.testing.assert_close(
                actual,
                expected_parameter_grads[name],
                atol=2e-5,
                rtol=2e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        torch.distributed.barrier()

        cp_model = CppMegaM2RNNMixer(
            config=_m2rnn_config(
                sequence_parallel=False,
                context_parallel_size=2,
            ),
            d_model=8,
            pg_collection=SimpleNamespace(tp=None, cp=group),
        ).to(device)
        cp_model.load_state_dict(reference.state_dict())
        cp_values = _balanced(values, 2, undo=False)
        cp_weights = _balanced(weights, 2, undo=False)
        cp_input = (
            cp_values.chunk(2, dim=0)[rank]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": document_ids}
        )
        cp_output, _ = cp_model(cp_input)
        (cp_output * cp_weights.chunk(2, dim=0)[rank]).sum().backward()
        if cp_input.grad is None:
            raise RuntimeError("M2RNN CP input has no gradient")
        cp_parameter_grads = _m2rnn_parameter_grads(cp_model)
        for gradient in cp_parameter_grads.values():
            torch.distributed.all_reduce(gradient, group=group)
        cp_outputs = [torch.empty_like(cp_output) for _ in range(2)]
        cp_input_grads = [torch.empty_like(cp_input.grad) for _ in range(2)]
        torch.distributed.all_gather(cp_outputs, cp_output.detach())
        torch.distributed.all_gather(cp_input_grads, cp_input.grad.detach())
        torch.testing.assert_close(
            _balanced(torch.cat(cp_outputs), 2, undo=True),
            reference_output,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            _balanced(torch.cat(cp_input_grads), 2, undo=True),
            expected_input_grad,
            atol=2e-5,
            rtol=2e-5,
        )
        for name, actual in cp_parameter_grads.items():
            torch.testing.assert_close(
                actual,
                expected_parameter_grads[name],
                atol=2e-5,
                rtol=2e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        results.put(("ok", rank))
    except Exception:
        results.put(("error", rank, traceback.format_exc()))
        raise
    finally:
        structure_dataset_patch._set_current_structure_batch(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_gloo_combined_tp_sequence_and_context_parallel_gradients(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    init_method = f"file://{tmp_path / 'combined-init'}"
    processes = [
        context.Process(
            target=_combined_worker,
            args=(rank, init_method, results),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
    messages = [results.get(timeout=5) for _ in range(4)]
    assert all(process.exitcode == 0 for process in processes), messages
    assert sorted(messages) == [("ok", rank) for rank in range(4)]


def test_gloo_sp_cp_reassembly_isolation_pipeline_and_gradients(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    _run_world("gloo", f"file://{tmp_path / 'gloo-init'}")


def test_gloo_packed_document_mtp_cp_roll_parity(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    init_method = f"file://{tmp_path / 'mtp-roll-init'}"
    processes = [
        context.Process(
            target=_mtp_roll_worker,
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


def test_document_mapping_allows_projected_input_and_state_output_widths():
    document_ids = torch.tensor([[1, 1, 2, 2]])
    structure_dataset_patch._set_current_structure_batch(
        {"document_ids": document_ids}
    )
    try:
        projected = torch.arange(8.0).view(4, 1, 2).requires_grad_(True)
        output = map_sequence_by_document(
            projected,
            lambda segment: segment.sum(dim=-1, keepdim=True).cumsum(dim=0),
        )
        assert output.shape == (4, 1, 1)
        assert output.flatten().tolist() == [1.0, 6.0, 9.0, 22.0]
        output.sum().backward()
        assert projected.grad is not None
        assert projected.grad[:, 0, 0].tolist() == [2.0, 1.0, 2.0, 1.0]
    finally:
        structure_dataset_patch._set_current_structure_batch(None)


def test_stateful_mixers_fail_closed_when_configured_group_is_missing():
    from cppmega.megatron.author_mamba3_spec import AuthorMamba3Mixer
    from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer

    missing_groups = SimpleNamespace(tp=None, cp=None)
    with pytest.raises(ValueError, match=r"pg_collection\.tp"):
        CppMegaM2RNNMixer(
            config=cast(
                Any,
                SimpleNamespace(
                    sequence_parallel=True,
                    context_parallel_size=1,
                ),
            ),
            d_model=4,
            pg_collection=missing_groups,
        )
    with pytest.raises(
        ValueError,
        match=r"context_parallel_size=2.*world_size=1",
    ):
        CppMegaM2RNNMixer(
            config=cast(
                Any,
                SimpleNamespace(
                    sequence_parallel=False,
                    context_parallel_size=2,
                ),
            ),
            d_model=4,
            pg_collection=missing_groups,
        )
    with pytest.raises(
        ValueError,
        match=r"context_parallel_size=2.*world_size=1",
    ):
        AuthorMamba3Mixer(
            config=cast(
                Any,
                SimpleNamespace(
                    tensor_model_parallel_size=1,
                    context_parallel_size=2,
                ),
            ),
            d_model=4,
            pg_collection=missing_groups,
        )
    with pytest.raises(
        ValueError,
        match=r"tensor_model_parallel_size=2.*world_size=1",
    ):
        AuthorMamba3Mixer(
            config=cast(
                Any,
                SimpleNamespace(
                    context_parallel_size=1,
                    tensor_model_parallel_size=2,
                ),
            ),
            d_model=4,
            pg_collection=missing_groups,
        )


def test_mamba3_sp_marks_only_replicated_parameters_for_tp_grad_sum():
    from cppmega.megatron.cppmega_mamba3_tp_mixer import CppmegaMamba3TPMixer

    mixer = object.__new__(CppmegaMamba3TPMixer)
    torch.nn.Module.__init__(mixer)
    mixer.config = cast(Any, SimpleNamespace(sequence_parallel=True))
    mixer.tp_world_size = 1
    mixer.tp_group = cast(Any, None)
    mixer.angle_proj = torch.nn.Linear(4, 2, bias=False)
    mixer.B_norm = torch.nn.LayerNorm(2, elementwise_affine=True)
    mixer.C_norm = torch.nn.LayerNorm(2, elementwise_affine=True)
    mixer.norm = torch.nn.LayerNorm(2, elementwise_affine=True)

    mixer._configure_replicated_tp_gradients()

    for parameter in (
        mixer.angle_proj.weight,
        mixer.B_norm.weight,
        mixer.C_norm.weight,
    ):
        assert getattr(parameter, "sequence_parallel", None) is True
        assert getattr(parameter, "tensor_model_parallel", None) is False
    assert not hasattr(mixer.norm.weight, "sequence_parallel")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs with NCCL",
)
def test_nccl_two_gpu_sp_cp_document_isolation_forward_backward_parity():
    port = _free_port()
    _run_world("nccl", f"tcp://127.0.0.1:{port}")


@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs with NCCL and Transformer Engine",
)
def test_nccl_two_gpu_actual_m2rnn_sp_cp_document_isolation_parity():
    port = _free_port()
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_m2rnn_integration_worker,
            args=(rank, f"tcp://127.0.0.1:{port}", results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=180)
    messages = [results.get(timeout=5) for _ in range(2)]
    assert all(process.exitcode == 0 for process in processes), messages
    assert sorted(messages) == [("ok", 0), ("ok", 1)]
