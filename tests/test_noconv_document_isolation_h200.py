"""Real H200 coverage for the production NAM56R no-conv mixer."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import socket
import time
import traceback
from typing import Any

import pytest
import torch


_TOPOLOGIES = {
    "tp1": {"tp_size": 1, "cp_size": 1, "sequence_parallel": False},
    "tp2_sp": {"tp_size": 2, "cp_size": 1, "sequence_parallel": True},
    "cp2": {"tp_size": 1, "cp_size": 2, "sequence_parallel": False},
    "tp2_cp2": {"tp_size": 2, "cp_size": 2, "sequence_parallel": True},
}
_DOCUMENT_LENGTH = 32
_DOCUMENT_COUNT = 3
_SEQUENCE_LENGTH = _DOCUMENT_LENGTH * _DOCUMENT_COUNT
_HIDDEN_SIZE = 64
_FORWARD_ATOL = 2e-5
_FORWARD_RTOL = 2e-3
_INPUT_GRAD_ATOL = 2e-6
_INPUT_GRAD_RTOL = 1e-2
_PARAM_GRAD_ABS_FLOOR = 2e-7
_PARAM_GRAD_RELATIVE_BOUND = 4e-2
_ISOLATION_PARAM_GRAD_ABS_FLOOR = 1e-10
_ISOLATION_PARAM_GRAD_RELATIVE_BOUND = 1e-4


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _grad_fn_names(tensor: Any) -> list[str]:
    pending = [tensor.grad_fn]
    seen: set[int] = set()
    names: set[str] = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.add(type(node).__name__)
        pending.extend(next_node for next_node, _index in node.next_functions)
    return sorted(names)


def _max_error(actual: Any, expected: Any) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max().item()),
        "max_rel": float((difference / denominator).max().item()),
    }


def _parameter_gradient_error(
    actual: Any,
    expected: Any,
    *,
    abs_floor: float,
    relative_bound: float,
) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    max_abs = float(difference.abs().max().item())
    max_abs_scale = max(
        float(actual_float.abs().max().item()),
        float(expected_float.abs().max().item()),
    )
    l2 = float(difference.norm().item())
    l2_scale = max(
        float(actual_float.norm().item()),
        float(expected_float.norm().item()),
    )
    return {
        "max_abs": max_abs,
        "max_abs_scale": max_abs_scale,
        "max_abs_bound": max(
            abs_floor,
            relative_bound * max_abs_scale,
        ),
        "l2": l2,
        "l2_scale": l2_scale,
        "l2_bound": max(
            abs_floor,
            relative_bound * l2_scale,
        ),
    }


def _worker(
    rank: int,
    topology_name: str,
    init_method: str,
    messages: Any,
) -> None:
    topology = _TOPOLOGIES[topology_name]
    tp_size = int(topology["tp_size"])
    cp_size = int(topology["cp_size"])
    sequence_parallel = bool(topology["sequence_parallel"])
    world_size = tp_size * cp_size

    import torch.distributed as dist

    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "CPPMEGA_STRUCTURE_ENABLED": "1",
        }
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )

    try:
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig
        from mamba_ssm.ops.triton import ssd_combined

        from cppmega.megatron import structure_dataset_patch
        from cppmega.megatron.nam56r_noconv_spec import (
            CppMegaNoConvSelectiveMambaMixer,
            build_cppmega_nam56r_noconv_stack_spec,
        )
        from cppmega.megatron.noconv_mamba_mixer import (
            NoConvMamba3BCMixer,
            mamba_chunk_scan_combined,
        )

        device_name = torch.cuda.get_device_name(rank)
        if "H200" not in device_name:
            raise RuntimeError(f"expected an H200, got {device_name!r}")

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(0xC0FFEE)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        tp_rank = dist.get_rank(group=tp_group)
        cp_rank = dist.get_rank(group=cp_group)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=_HIDDEN_SIZE,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=4 * _HIDDEN_SIZE,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.bfloat16,
            bf16=True,
            mamba_state_dim=16,
            mamba_head_dim=16,
            mamba_num_heads=4,
            mamba_num_groups=2,
            use_cpu_initialization=False,
        )
        object.__setattr__(config, "cppmega_noconv_mamba_chunk_size", 16)

        stack_spec = build_cppmega_nam56r_noconv_stack_spec(config)
        mixer_spec = stack_spec.submodules.mamba_layer.submodules.mixer
        if mixer_spec.module is not CppMegaNoConvSelectiveMambaMixer:
            raise AssertionError(
                f"production spec selected {mixer_spec.module!r}, "
                "not CppMegaNoConvSelectiveMambaMixer"
            )
        mixer_params = dict(mixer_spec.params or {})
        if 1 in mixer_params["r_layer_indices"]:
            raise AssertionError("NAM56R layer 1 unexpectedly routes to the R mixer")

        torch.manual_seed(0xBAD5EED)
        torch.cuda.manual_seed_all(0xBAD5EED)
        selector = mixer_spec.module(
            config=config,
            d_model=config.hidden_size,
            submodules=mixer_spec.submodules,
            layer_number=1,
            pg_collection=pg_collection,
            pp_layer_offset=0,
            **mixer_params,
        ).cuda()
        selector.train()
        if type(selector.impl) is not NoConvMamba3BCMixer:
            raise AssertionError(
                "production NAM56R M-layer selected "
                f"{type(selector.impl).__module__}.{type(selector.impl).__name__}"
            )
        if mamba_chunk_scan_combined is not ssd_combined.mamba_chunk_scan_combined:
            raise AssertionError("NoConv mixer is not bound to the installed SSD kernel")

        kernel_path = Path(inspect.getfile(mamba_chunk_scan_combined)).resolve()
        if not kernel_path.is_file():
            raise RuntimeError(f"SSD kernel module is missing: {kernel_path}")

        def shard_sequence(full: torch.Tensor) -> torch.Tensor:
            local = full
            if cp_size > 1:
                from megatron.core.ssm.mamba_context_parallel import (
                    _redo_attention_load_balancing,
                )

                local = _redo_attention_load_balancing(
                    local,
                    cp_size,
                    packed_seq_params=None,
                ).chunk(cp_size, dim=0)[cp_rank]
            if sequence_parallel:
                local = local.chunk(tp_size, dim=0)[tp_rank]
            return local.contiguous()

        def restore_sequence(local: torch.Tensor) -> torch.Tensor:
            restored = local
            if sequence_parallel:
                parts = [torch.empty_like(restored) for _ in range(tp_size)]
                dist.all_gather(parts, restored.contiguous(), group=tp_group)
                restored = torch.cat(parts, dim=0)
            if cp_size > 1:
                from megatron.core.ssm.mamba_context_parallel import (
                    _undo_attention_load_balancing,
                )

                parts = [torch.empty_like(restored) for _ in range(cp_size)]
                dist.all_gather(parts, restored.contiguous(), group=cp_group)
                restored = _undo_attention_load_balancing(
                    torch.cat(parts, dim=0),
                    cp_size,
                    packed_seq_params=None,
                )
            return restored

        def parameter_gradients(
            *,
            require_local_nonzero: bool,
        ) -> tuple[
            dict[str, torch.Tensor],
            dict[str, float],
            dict[str, dict[str, Any]],
        ]:
            gradients: dict[str, torch.Tensor] = {}
            norms: dict[str, float] = {}
            metadata: dict[str, dict[str, Any]] = {}
            violations = []
            for name, parameter in selector.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None:
                    violations.append(f"{name}: missing local gradient")
                    local_gradient = torch.zeros_like(
                        parameter,
                        dtype=torch.float32,
                    )
                else:
                    local_gradient = parameter.grad.detach().float()
                    if not bool(torch.isfinite(local_gradient).all()):
                        violations.append(f"{name}: non-finite local gradient")
                        local_gradient = torch.nan_to_num(local_gradient)
                local_norm = float(local_gradient.norm().item())
                if require_local_nonzero and local_norm == 0.0:
                    violations.append(f"{name}: zero local gradient")
                comparable = local_gradient.clone()
                reductions = []
                parameter_is_sequence_parallel = bool(
                    getattr(parameter, "sequence_parallel", False)
                )
                average_across_tp = bool(
                    getattr(parameter, "average_gradients_across_tp_domain", False)
                )
                if sequence_parallel and parameter_is_sequence_parallel:
                    dist.all_reduce(
                        comparable,
                        op=dist.ReduceOp.SUM,
                        group=tp_group,
                    )
                    reductions.append("tp_sum_sequence_parallel")
                elif average_across_tp:
                    dist.all_reduce(
                        comparable,
                        op=dist.ReduceOp.AVG,
                        group=tp_group,
                    )
                    reductions.append("tp_average")
                if cp_size > 1:
                    dist.all_reduce(
                        comparable,
                        op=dist.ReduceOp.SUM,
                        group=cp_group,
                    )
                    reductions.append("cp_sum")
                comparable_is_finite = bool(torch.isfinite(comparable).all())
                comparable_norm = float(comparable.norm().item())
                if not comparable_is_finite:
                    violations.append(f"{name}: non-finite canonical gradient")
                if comparable_norm == 0.0:
                    violations.append(f"{name}: zero canonical gradient")
                gradients[name] = comparable
                norms[name] = local_norm
                metadata[name] = {
                    "sequence_parallel": parameter_is_sequence_parallel,
                    "tensor_model_parallel": bool(
                        getattr(parameter, "tensor_model_parallel", False)
                    ),
                    "average_gradients_across_tp_domain": average_across_tp,
                    "reductions": reductions,
                    "local_shape": list(parameter.shape),
                    "local_gradient_nonzero": local_norm != 0.0,
                    "canonical_gradient_nonzero": comparable_norm != 0.0,
                }
            world_failure = torch.tensor(
                int(bool(violations)),
                device="cuda",
                dtype=torch.int32,
            )
            dist.all_reduce(world_failure, op=dist.ReduceOp.MAX)
            if bool(world_failure.item()):
                raise RuntimeError(
                    "invalid trainable parameter gradients after completing "
                    f"the collective schedule; rank={rank}, local={violations}"
                )
            return gradients, norms, metadata

        torch.manual_seed(0x12345)
        full_input = torch.randn(
            _SEQUENCE_LENGTH,
            1,
            _HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        full_loss_weight = torch.randn_like(full_input)
        dist.broadcast(full_input, src=0)
        dist.broadcast(full_loss_weight, src=0)
        packed_document_ids = torch.tensor(
            [[doc for doc in range(1, _DOCUMENT_COUNT + 1) for _ in range(_DOCUMENT_LENGTH)]],
            device="cuda",
            dtype=torch.long,
        )

        packed_input = shard_sequence(full_input).detach().clone().requires_grad_(True)
        packed_weight = shard_sequence(full_loss_weight)
        structure_dataset_patch._set_current_structure_batch(
            {"document_ids": packed_document_ids}
        )
        packed_output_local, output_bias = selector(packed_input)
        if output_bias is not None:
            raise AssertionError("NoConv production out_proj unexpectedly returned a bias")
        autograd_nodes = _grad_fn_names(packed_output_local)
        kernel_nodes = [
            name for name in autograd_nodes if "mambachunkscancombined" in name.lower()
        ]
        if not kernel_nodes:
            raise AssertionError(
                "actual output autograd graph does not contain "
                f"MambaChunkScanCombinedFn: {autograd_nodes}"
            )
        (
            packed_output_local.float() * packed_weight.float()
        ).sum().div(full_input.numel()).backward()
        torch.cuda.synchronize()
        if packed_input.grad is None or not bool(torch.isfinite(packed_input.grad).all()):
            raise RuntimeError("packed input gradient is missing or non-finite")
        packed_output = restore_sequence(packed_output_local.detach())
        packed_input_grad = restore_sequence(packed_input.grad.detach())
        (
            packed_parameter_grads,
            packed_gradient_norms,
            parameter_gradient_metadata,
        ) = parameter_gradients(require_local_nonzero=True)

        perturbed = full_input.clone()
        perturbed[:_DOCUMENT_LENGTH].add_(4)

        selector.zero_grad(set_to_none=True)
        separate_outputs = []
        separate_inputs = []
        separate_loss = None
        for document_index in range(_DOCUMENT_COUNT):
            start = document_index * _DOCUMENT_LENGTH
            end = start + _DOCUMENT_LENGTH
            document_input = (
                shard_sequence(full_input[start:end])
                .detach()
                .clone()
                .requires_grad_(True)
            )
            document_weight = shard_sequence(full_loss_weight[start:end])
            structure_dataset_patch._set_current_structure_batch(
                {
                    "document_ids": torch.ones(
                        1,
                        _DOCUMENT_LENGTH,
                        device="cuda",
                        dtype=torch.long,
                    )
                }
            )
            document_output_local, document_bias = selector(document_input)
            if document_bias is not None:
                raise AssertionError("per-document out_proj unexpectedly returned a bias")
            separate_outputs.append(restore_sequence(document_output_local.detach()))
            separate_inputs.append(document_input)
            document_loss = (
                document_output_local.float() * document_weight.float()
            ).sum().div(full_input.numel())
            separate_loss = (
                document_loss if separate_loss is None else separate_loss + document_loss
            )
        if separate_loss is None:
            raise AssertionError("no per-document loss was built")
        separate_loss.backward()
        torch.cuda.synchronize()
        separate_input_grads = []
        for document_input in separate_inputs:
            if document_input.grad is None or not bool(
                torch.isfinite(document_input.grad).all()
            ):
                raise RuntimeError("per-document input gradient is missing or non-finite")
            separate_input_grads.append(restore_sequence(document_input.grad.detach()))
        separate_output = torch.cat(separate_outputs, dim=0)
        separate_input_grad = torch.cat(separate_input_grads, dim=0)
        (
            separate_parameter_grads,
            separate_gradient_norms,
            separate_parameter_gradient_metadata,
        ) = parameter_gradients(require_local_nonzero=True)

        torch.testing.assert_close(
            packed_output,
            separate_output,
            atol=_FORWARD_ATOL,
            rtol=_FORWARD_RTOL,
        )
        torch.testing.assert_close(
            packed_input_grad,
            separate_input_grad,
            atol=_INPUT_GRAD_ATOL,
            rtol=_INPUT_GRAD_RTOL,
        )
        if packed_parameter_grads.keys() != separate_parameter_grads.keys():
            raise AssertionError("packed and per-document parameter gradient sets differ")
        parameter_errors = {}
        for name in packed_parameter_grads:
            error = _parameter_gradient_error(
                packed_parameter_grads[name],
                separate_parameter_grads[name],
                abs_floor=_PARAM_GRAD_ABS_FLOOR,
                relative_bound=_PARAM_GRAD_RELATIVE_BOUND,
            )
            if (
                error["max_abs"] > error["max_abs_bound"]
                or error["l2"] > error["l2_bound"]
            ):
                raise AssertionError(
                    f"{name}: packed/per-document parameter-gradient mismatch: "
                    f"{error}"
                )
            parameter_errors[name] = error

        if parameter_gradient_metadata != separate_parameter_gradient_metadata:
            raise AssertionError(
                "packed and per-document parameter gradient metadata differs"
            )

        later_loss_weight = full_loss_weight.clone()
        later_loss_weight[:_DOCUMENT_LENGTH].zero_()
        isolation_runs = {}
        for label, isolation_input in (
            ("base", full_input),
            ("document_a_perturbed", perturbed),
        ):
            selector.zero_grad(set_to_none=True)
            local_input = (
                shard_sequence(isolation_input)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            structure_dataset_patch._set_current_structure_batch(
                {"document_ids": packed_document_ids}
            )
            local_output, isolation_bias = selector(local_input)
            if isolation_bias is not None:
                raise AssertionError(
                    "later-document isolation out_proj unexpectedly returned a bias"
                )
            (
                local_output.float()
                * shard_sequence(later_loss_weight).float()
            ).sum().div(full_input.numel()).backward()
            torch.cuda.synchronize()
            if local_input.grad is None or not bool(
                torch.isfinite(local_input.grad).all()
            ):
                raise RuntimeError(
                    f"{label} later-document input gradient is missing or non-finite"
                )
            (
                isolation_parameter_grads,
                _isolation_gradient_norms,
                isolation_parameter_metadata,
            ) = parameter_gradients(require_local_nonzero=False)
            isolation_runs[label] = {
                "output": restore_sequence(local_output.detach()),
                "input_grad": restore_sequence(local_input.grad.detach()),
                "parameter_grads": isolation_parameter_grads,
                "parameter_metadata": isolation_parameter_metadata,
            }

        base_isolation = isolation_runs["base"]
        perturbed_isolation = isolation_runs["document_a_perturbed"]
        cross_document_max_abs = float(
            (
                perturbed_isolation["output"][_DOCUMENT_LENGTH:].float()
                - base_isolation["output"][_DOCUMENT_LENGTH:].float()
            )
            .abs()
            .max()
            .item()
        )
        if cross_document_max_abs != 0.0:
            raise AssertionError(
                "perturbing document A changed a later document: "
                f"max_abs={cross_document_max_abs}"
            )
        base_document_a_input_grad_max_abs = float(
            base_isolation["input_grad"][:_DOCUMENT_LENGTH]
            .float()
            .abs()
            .max()
            .item()
        )
        perturbed_document_a_input_grad_max_abs = float(
            perturbed_isolation["input_grad"][:_DOCUMENT_LENGTH]
            .float()
            .abs()
            .max()
            .item()
        )
        if (
            base_document_a_input_grad_max_abs != 0.0
            or perturbed_document_a_input_grad_max_abs != 0.0
        ):
            raise AssertionError(
                "later-document-only loss reached document A inputs: "
                f"base={base_document_a_input_grad_max_abs}, "
                f"perturbed={perturbed_document_a_input_grad_max_abs}"
            )
        if not torch.equal(
            base_isolation["output"][_DOCUMENT_LENGTH:],
            perturbed_isolation["output"][_DOCUMENT_LENGTH:],
        ):
            raise AssertionError(
                "perturbing document A changed later-document outputs in the "
                "later-document-only gradient check"
            )
        if not torch.equal(
            base_isolation["input_grad"][_DOCUMENT_LENGTH:],
            perturbed_isolation["input_grad"][_DOCUMENT_LENGTH:],
        ):
            raise AssertionError(
                "perturbing document A changed later-document input gradients"
            )
        if (
            base_isolation["parameter_metadata"]
            != perturbed_isolation["parameter_metadata"]
        ):
            raise AssertionError(
                "parameter metadata changed across the isolation-gradient runs"
            )
        isolation_parameter_errors = {}
        isolation_parameter_failures = {}
        for name, base_gradient in base_isolation["parameter_grads"].items():
            perturbed_gradient = perturbed_isolation["parameter_grads"][name]
            error = _parameter_gradient_error(
                base_gradient,
                perturbed_gradient,
                abs_floor=_ISOLATION_PARAM_GRAD_ABS_FLOOR,
                relative_bound=_ISOLATION_PARAM_GRAD_RELATIVE_BOUND,
            )
            if (
                error["max_abs"] > error["max_abs_bound"]
                or error["l2"] > error["l2_bound"]
            ):
                isolation_parameter_failures[name] = error
            isolation_parameter_errors[name] = error
        if isolation_parameter_failures:
            raise AssertionError(
                "perturbing document A changed later-document-only parameter "
                f"gradients: {isolation_parameter_failures}"
            )

        messages.put(
            {
                "status": "passed",
                "rank": rank,
                "global_world_size": world_size,
                "tp_size": tp_size,
                "tp_rank": tp_rank,
                "cp_size": cp_size,
                "cp_rank": cp_rank,
                "sequence_parallel": sequence_parallel,
                "device": device_name,
                "device_capability": list(torch.cuda.get_device_capability(rank)),
                "production_spec_builder": (
                    "cppmega.megatron.nam56r_noconv_spec."
                    "build_cppmega_nam56r_noconv_stack_spec"
                ),
                "selector_class": (
                    f"{type(selector).__module__}.{type(selector).__name__}"
                ),
                "mixer_class": (
                    f"{type(selector.impl).__module__}.{type(selector.impl).__name__}"
                ),
                "kernel_callable_module": mamba_chunk_scan_combined.__module__,
                "kernel_module_path": str(kernel_path),
                "kernel_module_sha256": hashlib.sha256(
                    kernel_path.read_bytes()
                ).hexdigest(),
                "kernel_autograd_nodes": kernel_nodes,
                "packed_vs_documents_forward": _max_error(
                    packed_output,
                    separate_output,
                ),
                "packed_vs_documents_input_grad": _max_error(
                    packed_input_grad,
                    separate_input_grad,
                ),
                "cross_document_perturbation_max_abs": cross_document_max_abs,
                "trainable_parameter_count": len(packed_parameter_grads),
                "packed_parameter_gradient_norms": packed_gradient_norms,
                "per_document_parameter_gradient_norms": separate_gradient_norms,
                "parameter_gradient_metadata": parameter_gradient_metadata,
                "parameter_gradient_errors": parameter_errors,
                "later_document_only_isolation": {
                    "document_a_input_grad_max_abs": {
                        "base": base_document_a_input_grad_max_abs,
                        "document_a_perturbed": (
                            perturbed_document_a_input_grad_max_abs
                        ),
                    },
                    "later_output_error": _max_error(
                        base_isolation["output"][_DOCUMENT_LENGTH:],
                        perturbed_isolation["output"][_DOCUMENT_LENGTH:],
                    ),
                    "later_input_grad_error": _max_error(
                        base_isolation["input_grad"][_DOCUMENT_LENGTH:],
                        perturbed_isolation["input_grad"][_DOCUMENT_LENGTH:],
                    ),
                    "parameter_gradient_errors": isolation_parameter_errors,
                },
            }
        )
    except Exception:
        messages.put(
            {
                "status": "failed",
                "rank": rank,
                "topology": topology_name,
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        try:
            from cppmega.megatron import structure_dataset_patch

            structure_dataset_patch._set_current_structure_batch(None)
        except Exception:
            pass
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _run_topology(topology_name: str) -> None:
    topology = _TOPOLOGIES[topology_name]
    world_size = int(topology["tp_size"]) * int(topology["cp_size"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"{topology_name} requires {world_size} CUDA GPUs")

    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    processes = [
        context.Process(
            target=_worker,
            args=(rank, topology_name, init_method, messages),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 360
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    for process in timed_out:
        process.terminate()
    for process in timed_out:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

    reports = []
    for _index in range(world_size):
        try:
            reports.append(messages.get(timeout=2))
        except Empty:
            break
    if timed_out:
        reports.append(
            {
                "status": "failed",
                "topology": topology_name,
                "error": "worker timeout",
                "timed_out_pids": [process.pid for process in timed_out],
            }
        )
    if len(reports) != world_size:
        reports.append(
            {
                "status": "failed",
                "topology": topology_name,
                "error": (
                    f"expected {world_size} worker reports, got {len(reports)}"
                ),
            }
        )

    evidence_dir = os.environ.get("CPPMEGA_NOCONV_EVIDENCE_DIR")
    if evidence_dir:
        path = Path(evidence_dir) / f"{topology_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2, sort_keys=True))
    messages.close()
    messages.join_thread()

    assert not timed_out, reports
    assert all(process.exitcode == 0 for process in processes), reports
    assert len(reports) == world_size, reports
    assert all(report["status"] == "passed" for report in reports), reports
    assert sorted(int(report["rank"]) for report in reports) == list(range(world_size))


def test_real_noconv_tp1_document_isolation() -> None:
    _run_topology("tp1")


def test_real_noconv_tp2_sequence_parallel_document_isolation() -> None:
    _run_topology("tp2_sp")


def test_real_noconv_cp2_zigzag_document_isolation() -> None:
    _run_topology("cp2")


def test_real_noconv_tp2_cp2_cartesian_document_isolation() -> None:
    _run_topology("tp2_cp2")
