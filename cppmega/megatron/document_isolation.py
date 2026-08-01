"""Runtime isolation for logical documents packed into one Megatron row."""

from __future__ import annotations

import inspect
import os
from functools import wraps
from itertools import accumulate
from typing import Any, Callable

import torch

_PATCH_MARKER = "__cppmega_document_isolation_patched__"
_layout_cache: dict[str, Any] = {"key": None, "value": None}

# Maps the Python object identity (``id(tensor_recv_prev)``) of a received
# pipeline activation to the document_ids tensor that accompanied it.
#
# IMPORTANT CONTRACT: the key is the exact tensor object identity seen by
# ``_exchange_pipeline_document_ids`` when it performs the irecv. Any call
# that creates a new tensor object---``.contiguous()`` on a non-contiguous
# buffer, a slicing op that returns a new tensor, or an explicit ``.clone()``
# or ``.detach()``---changes ``id()`` and breaks the lookup in
# ``_patch_model_input_transport``. Keep the activation reference stable
# between the P2P receive and ``set_input_tensor``.
_received_document_ids: dict[int, torch.Tensor] = {}


def _structure_enabled() -> bool:
    return os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _current_structure_batch() -> dict[str, torch.Tensor] | None:
    from cppmega.megatron.structure_dataset_patch import _get_current_structure_batch

    return _get_current_structure_batch()


def _set_current_document_ids(document_ids: torch.Tensor) -> None:
    from cppmega.megatron.structure_dataset_patch import (
        _get_current_structure_batch,
        _set_current_structure_batch,
    )

    current = dict(_get_current_structure_batch() or {})
    current["document_ids"] = document_ids
    _set_current_structure_batch(current)


def bind_current_structure_batch(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the originating microbatch sidecars during checkpoint recomputation."""

    current = _current_structure_batch()
    if current is None:
        if _structure_enabled():
            _raw_document_ids()
        return function
    snapshot = dict(current)
    if "document_ids" not in snapshot and "graph_document_ids" not in snapshot:
        if _structure_enabled():
            _raw_document_ids()
        return function

    @wraps(function)
    def bound(*args, **kwargs):
        from cppmega.megatron.structure_dataset_patch import (
            _get_current_structure_batch,
            _set_current_structure_batch,
        )

        previous = _get_current_structure_batch()
        _set_current_structure_batch(snapshot)
        try:
            return function(*args, **kwargs)
        finally:
            _set_current_structure_batch(previous)

    return bound


def _raw_document_ids(*, required: bool | None = None) -> torch.Tensor | None:
    current = _current_structure_batch()
    document_ids = None if current is None else current.get("document_ids")
    if document_ids is None and current is not None:
        # Compatibility with graph-route batches produced before document_ids
        # became a model-wide sidecar.
        document_ids = current.get("graph_document_ids")
    if document_ids is None:
        if _structure_enabled() if required is None else required:
            raise RuntimeError(
                "CPPMEGA_STRUCTURE_ENABLED=1 requires document_ids on every model stage"
            )
        return None
    if not isinstance(document_ids, torch.Tensor):
        raise TypeError("document_ids must be a torch.Tensor")
    return document_ids


def document_layout(
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device | None = None,
    required: bool | None = None,
) -> tuple[torch.Tensor | None, tuple[tuple[tuple[int, int], ...], ...], bool]:
    """Return validated IDs, row spans, and whether any row packs multiple docs."""

    raw = _raw_document_ids(required=required)
    if raw is None:
        return None, (), False
    if raw.dim() == 1:
        raw = raw.unsqueeze(0)
    if tuple(raw.shape) != (batch_size, sequence_length):
        raise ValueError(
            "document_ids shape "
            f"{tuple(raw.shape)} != expected {(batch_size, sequence_length)}"
        )
    if raw.is_floating_point() or raw.dtype == torch.bool:
        raise TypeError(f"document_ids must use an integer dtype, got {raw.dtype}")

    target_device = raw.device if device is None else torch.device(device)
    key = (id(raw), raw._version, tuple(raw.shape), target_device)
    if _layout_cache["key"] == key:
        return _layout_cache["value"]

    rows = raw.detach().cpu().tolist()
    all_spans: list[tuple[tuple[int, int], ...]] = []
    multiple = False
    for batch_index, row in enumerate(rows):
        spans: list[tuple[int, int]] = []
        start = 0
        previous = int(row[0]) if row else 0
        expected_positive = 1
        if previous not in (0, 1):
            raise ValueError(
                f"document_ids row {batch_index} must start at 1 or be padding, got {previous}"
            )
        for position, value_raw in enumerate(row):
            value = int(value_raw)
            if value < 0:
                raise ValueError("document_ids cannot contain negative values")
            if previous == 0 and value != 0:
                raise ValueError(
                    f"document_ids row {batch_index} contains non-padding after padding"
                )
            if value != previous:
                spans.append((start, position))
                start = position
                if value == 0:
                    previous = value
                    continue
                if value != expected_positive + 1:
                    raise ValueError(
                        f"document_ids row {batch_index} must contain contiguous runs 1..N"
                    )
                expected_positive = value
                previous = value
        if row:
            spans.append((start, len(row)))
        positive_spans = sum(1 for start, _ in spans if row[start] > 0)
        multiple |= positive_spans > 1
        all_spans.append(tuple(spans))

    ids = raw.to(device=target_device, dtype=torch.long, non_blocking=True)
    value = (ids, tuple(all_spans), multiple)
    _layout_cache["key"] = key
    _layout_cache["value"] = value
    return value


def map_sequence_by_document(
    hidden_states: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    *,
    pad_to: int = 1,
) -> torch.Tensor:
    """Run a stateful sequence function independently for each packed document."""

    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [S,B,H]")
    if pad_to < 1:
        raise ValueError("pad_to must be positive")
    raw = _raw_document_ids()
    if raw is None:
        return function(hidden_states)
    if raw.dim() == 1:
        raw = raw.unsqueeze(0)
    batch_size, sequence_length = raw.shape
    _ids, row_spans, multiple = document_layout(
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        device=hidden_states.device,
    )
    if not multiple:
        return function(hidden_states)
    if hidden_states.shape[:2] != (sequence_length, batch_size):
        raise NotImplementedError(
            "multi-document state isolation does not support sequence-sharded hidden states"
        )

    # ponytail: batch padded document segments for exact state resets. A true
    # varlen path is documented in docs/document_isolation_varlen_design.md;
    # it is deferred until a cppmega custom autograd kernel accepts cu_seqlens.
    segments = [
        hidden_states[start:end, batch_index]
        for batch_index, spans in enumerate(row_spans)
        for start, end in spans
    ]
    target_length = max(segment.shape[0] for segment in segments)
    target_length += (-target_length) % pad_to
    padded = torch.stack(
        [
            torch.cat(
                (
                    segment,
                    segment.new_zeros(
                        (target_length - segment.shape[0], segment.shape[1])
                    ),
                )
            )
            for segment in segments
        ],
        dim=1,
    )
    mapped = function(padded)
    if mapped.dim() != 3 or mapped.shape[:2] != padded.shape[:2]:
        raise RuntimeError(
            f"isolated sequence function returned {tuple(mapped.shape)} "
            f"for input sequence/batch shape {tuple(padded.shape[:2])}"
        )
    cursor = 0
    rows = []
    for spans in row_spans:
        parts = []
        for start, end in spans:
            parts.append(mapped[: end - start, cursor])
            cursor += 1
        rows.append(torch.cat(parts, dim=0))
    return torch.stack(rows, dim=1)


def _group_size_rank(group: Any) -> tuple[int, int]:
    if group is None:
        return 1, 0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("sequence sharding requires an initialized process group")
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    if rank < 0:
        raise ValueError("the current rank is not a member of the process group")
    return world_size, rank


def _process_group_ranks(group: Any) -> tuple[int, ...]:
    if group is None:
        return ()
    world_size, _rank = _group_size_rank(group)
    get_ranks = getattr(torch.distributed, "get_process_group_ranks", None)
    if get_ranks is not None:
        return tuple(int(rank) for rank in get_ranks(group))
    get_global_rank = getattr(torch.distributed, "get_global_rank", None)
    if get_global_rank is None:
        raise RuntimeError(
            "this torch.distributed build cannot inspect process-group membership"
        )
    return tuple(int(get_global_rank(group, rank)) for rank in range(world_size))


def _validate_model_parallel_topology(
    config: Any,
    *,
    tp_group: Any,
    cp_group: Any,
    component: str,
) -> tuple[int, int]:
    """Validate that configured TP/CP axes match the initialized process mesh."""

    configured_tp_size = int(
        getattr(config, "tensor_model_parallel_size", 1)
    )
    configured_cp_size = int(getattr(config, "context_parallel_size", 1))
    if configured_tp_size < 1 or configured_cp_size < 1:
        raise ValueError(
            f"{component} requires positive TP/CP sizes, got "
            f"tp={configured_tp_size}, cp={configured_cp_size}"
        )

    actual_tp_size, _tp_rank = _group_size_rank(tp_group)
    actual_cp_size, _cp_rank = _group_size_rank(cp_group)
    if actual_tp_size != configured_tp_size:
        raise ValueError(
            f"{component} configured tensor_model_parallel_size="
            f"{configured_tp_size}, but pg_collection.tp has world_size="
            f"{actual_tp_size}"
        )
    if actual_cp_size != configured_cp_size:
        raise ValueError(
            f"{component} configured context_parallel_size="
            f"{configured_cp_size}, but pg_collection.cp has world_size="
            f"{actual_cp_size}"
        )

    if actual_tp_size > 1 and actual_cp_size > 1:
        tp_ranks = _process_group_ranks(tp_group)
        cp_ranks = _process_group_ranks(cp_group)
        overlap = set(tp_ranks).intersection(cp_ranks)
        current_global_rank = torch.distributed.get_rank()
        if overlap != {current_global_rank}:
            raise ValueError(
                f"{component} requires distinct Cartesian TP/CP axes; "
                f"tp ranks={tp_ranks} and cp ranks={cp_ranks} overlap at "
                f"{tuple(sorted(overlap))}"
            )
    return actual_tp_size, actual_cp_size


class _GatherSequence(torch.autograd.Function):
    """Device-agnostic sequence gather with selectable backward semantics."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any, reduce_backward: bool):
        world_size, rank = _group_size_rank(group)
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.local_length = tensor.shape[0]
        ctx.reduce_backward = reduce_backward
        if world_size == 1:
            return tensor
        parts = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        (grad_output,) = grad_outputs
        if ctx.world_size == 1:
            return grad_output, None, None
        if ctx.reduce_backward:
            local = grad_output.new_empty(
                (ctx.local_length, *grad_output.shape[1:])
            )
            if torch.distributed.get_backend(ctx.group) == "gloo":
                # Gloo lacks reduce_scatter_tensor on supported macOS builds.
                reduced = grad_output.contiguous().clone()
                torch.distributed.all_reduce(reduced, group=ctx.group)
                local.copy_(
                    reduced.narrow(
                        0, ctx.rank * ctx.local_length, ctx.local_length
                    )
                )
            else:
                torch.distributed.reduce_scatter_tensor(
                    local, grad_output.contiguous(), group=ctx.group
                )
        else:
            local = grad_output.narrow(
                0, ctx.rank * ctx.local_length, ctx.local_length
            ).contiguous()
        return local, None, None


class _ScatterSequence(torch.autograd.Function):
    """Sequence scatter paired with either gathered or local-only backward."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any, gather_backward: bool):
        world_size, rank = _group_size_rank(group)
        if tensor.shape[0] % world_size:
            raise ValueError(
                f"sequence length {tensor.shape[0]} is not divisible by group size "
                f"{world_size}"
            )
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.gather_backward = gather_backward
        ctx.global_shape = tuple(tensor.shape)
        local_length = tensor.shape[0] // world_size
        return tensor.narrow(0, rank * local_length, local_length).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        (grad_output,) = grad_outputs
        if ctx.world_size == 1:
            return grad_output, None, None
        if ctx.gather_backward:
            parts = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
            torch.distributed.all_gather(
                parts, grad_output.contiguous(), group=ctx.group
            )
            full = torch.cat(parts, dim=0)
        else:
            full = grad_output.new_zeros(ctx.global_shape)
            local_length = grad_output.shape[0]
            full.narrow(
                0, ctx.rank * local_length, local_length
            ).copy_(grad_output)
        return full, None, None


def _assert_mamba_cp_signatures() -> None:
    """Fail fast if Megatron's private CP helpers change their contract.

    ``document_isolation`` calls ``_redo_attention_load_balancing`` and
    ``_undo_attention_load_balancing`` from
    ``megatron.core.ssm.mamba_context_parallel`` with positional arguments
    ``(tensor, cp_size, packed_seq_params=None)``.  Megatron treats these as
    private helpers and may change their signatures without notice; a silent
    mismatch would corrupt the context-parallel zigzag restore path.  This
    guard checks the current signature at patch time and raises a descriptive
    error instead of allowing downstream shape corruption.
    """
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    def _check(name: str, function: Callable[..., Any]) -> None:
        try:
            signature = inspect.signature(function)
        except ValueError as exc:
            raise RuntimeError(
                f"cannot inspect Megatron mamba_context_parallel.{name} signature"
            ) from exc

        parameters = list(signature.parameters.values())
        if len(parameters) != 3:
            raise RuntimeError(
                f"Megatron mamba_context_parallel.{name} signature changed: "
                f"expected 3 parameters, got {len(parameters)} ({signature}). "
                "Update document_isolation before using this Megatron commit."
            )

        input_param, cp_param, packed_param = parameters
        input_name = input_param.name
        input_annotation = str(input_param.annotation)
        cp_annotation = str(cp_param.annotation)
        packed_annotation = str(packed_param.annotation)

        if input_name not in ("input", "input_") or "Tensor" not in input_annotation:
            raise RuntimeError(
                f"Megatron mamba_context_parallel.{name} first parameter changed: "
                f"expected Tensor input, got {input_name}: {input_annotation} "
                f"({signature})"
            )
        if cp_param.name != "cp_size" or cp_annotation != "<class 'int'>":
            raise RuntimeError(
                f"Megatron mamba_context_parallel.{name} second parameter changed: "
                f"expected int cp_size, got {cp_param.name}: {cp_annotation} "
                f"({signature})"
            )
        if (
            packed_param.name != "packed_seq_params"
            or "PackedSeqParams" not in packed_annotation
        ):
            raise RuntimeError(
                f"Megatron mamba_context_parallel.{name} third parameter changed: "
                f"expected Optional[PackedSeqParams] packed_seq_params, got "
                f"{packed_param.name}: {packed_annotation} ({signature})"
            )

        return_annotation = signature.return_annotation
        if return_annotation is not inspect.Signature.empty:
            return_name = str(return_annotation)
            if "Tensor" not in return_name:
                raise RuntimeError(
                    f"Megatron mamba_context_parallel.{name} return annotation changed: "
                    f"expected Tensor, got {return_name} ({signature})"
                )

    _check("_redo_attention_load_balancing", _redo_attention_load_balancing)
    _check("_undo_attention_load_balancing", _undo_attention_load_balancing)


def _context_parallel_reorder(
    tensor: torch.Tensor, *, cp_size: int, restore_global_order: bool
) -> torch.Tensor:
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    reorder = (
        _undo_attention_load_balancing
        if restore_global_order
        else _redo_attention_load_balancing
    )
    return reorder(tensor, cp_size, packed_seq_params=None)


def gather_context_parallel_sequence(
    tensor: torch.Tensor, cp_group: Any
) -> torch.Tensor:
    """Restore Megatron's CP-zigzag sequence with reduce-scatter backward."""

    cp_size, _rank = _group_size_rank(cp_group)
    if cp_size == 1:
        return tensor
    gathered = _GatherSequence.apply(tensor, cp_group, True)
    if gathered.shape[0] % (2 * cp_size):
        raise NotImplementedError(
            "Megatron CP zigzag requires global sequence length divisible by "
            f"2 * context_parallel_size ({2 * cp_size})"
        )
    return _context_parallel_reorder(
        gathered, cp_size=cp_size, restore_global_order=True
    )


def scatter_context_parallel_sequence(
    tensor: torch.Tensor, cp_group: Any
) -> torch.Tensor:
    """Return a global sequence to Megatron's CP-zigzag local layout."""

    cp_size, _rank = _group_size_rank(cp_group)
    if cp_size == 1:
        return tensor
    if tensor.shape[0] % (2 * cp_size):
        raise NotImplementedError(
            "Megatron CP zigzag requires global sequence length divisible by "
            f"2 * context_parallel_size ({2 * cp_size})"
        )
    balanced = _context_parallel_reorder(
        tensor, cp_size=cp_size, restore_global_order=False
    )
    return _ScatterSequence.apply(balanced, cp_group, False)


def map_sharded_sequence_by_document(
    hidden_states: torch.Tensor,
    function: Callable[[torch.Tensor], torch.Tensor],
    *,
    sequence_parallel_group: Any = None,
    context_parallel_group: Any = None,
    pad_to: int = 1,
) -> torch.Tensor:
    """Reassemble SP/CP input, isolate documents, then restore local layout."""

    sp_size, _sp_rank = _group_size_rank(sequence_parallel_group)
    cp_size, _cp_rank = _group_size_rank(context_parallel_group)
    if sp_size > 1 and cp_size > 1:
        sp_ranks = _process_group_ranks(sequence_parallel_group)
        cp_ranks = _process_group_ranks(context_parallel_group)
        overlap = set(sp_ranks).intersection(cp_ranks)
        current_global_rank = torch.distributed.get_rank()
        if overlap != {current_global_rank}:
            raise ValueError(
                "SP and CP must use distinct Cartesian process-group axes; "
                f"SP ranks={sp_ranks}, CP ranks={cp_ranks}"
            )

    full = hidden_states
    if sp_size > 1:
        # The stateful module is replicated across TP ranks. Its output
        # scatter gathers gradients so every replica receives the full loss.
        full = _GatherSequence.apply(full, sequence_parallel_group, False)
    if cp_size > 1:
        full = gather_context_parallel_sequence(full, context_parallel_group)

    mapped = map_sequence_by_document(full, function, pad_to=pad_to)

    if cp_size > 1:
        mapped = scatter_context_parallel_sequence(mapped, context_parallel_group)
    if sp_size > 1:
        mapped = _ScatterSequence.apply(mapped, sequence_parallel_group, True)
    return mapped


def roll_tensor_by_document(
    tensor: torch.Tensor,
    *,
    shifts: int = -1,
    dims: int = -1,
    cp_group: Any = None,
    packed_seq_params: Any = None,
    fallback: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron MTP roll that zeroes every logical-document boundary."""

    raw = _raw_document_ids()
    if raw is None:
        return fallback(
            tensor,
            shifts=shifts,
            dims=dims,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
        )
    if raw.dim() == 1:
        raw = raw.unsqueeze(0)
    batch_size, sequence_length = raw.shape
    ids, _spans, multiple = document_layout(
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        device=tensor.device,
    )
    if not multiple:
        return fallback(
            tensor,
            shifts=shifts,
            dims=dims,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
        )
    if ids is None:
        raise RuntimeError("validated document layout did not return document_ids")
    if shifts != -1:
        raise NotImplementedError("packed-document MTP supports only shifts=-1")
    if packed_seq_params is not None:
        raise RuntimeError(
            "document_ids and upstream packed_seq_params cannot both drive MTP"
        )

    sequence_dim = dims % tensor.dim()
    cp_size, _cp_rank = _group_size_rank(cp_group)
    rolled_input = tensor
    if cp_size > 1:
        if sequence_length % cp_size:
            raise ValueError(
                f"global document sequence length {sequence_length} is not "
                f"divisible by context parallel size {cp_size}"
            )
        if tensor.shape[sequence_dim] * cp_size != sequence_length:
            raise ValueError(
                "cannot align global document_ids with context-parallel MTP "
                f"tensor shape {tuple(tensor.shape)} along dim {sequence_dim}: "
                f"local length {tensor.shape[sequence_dim]} * cp_size {cp_size} "
                f"!= global length {sequence_length}"
            )
        sequence_first = tensor.movedim(sequence_dim, 0)
        sequence_first = gather_context_parallel_sequence(
            sequence_first,
            cp_group,
        )
        rolled_input = sequence_first.movedim(0, sequence_dim)

    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, :-1] = (ids[:, :-1] > 0) & (ids[:, :-1] == ids[:, 1:])
    rolled = torch.roll(rolled_input, shifts=-1, dims=sequence_dim)
    if (
        rolled_input.shape[:2] == (batch_size, sequence_length)
        and sequence_dim == 1
    ):
        mask = valid
    elif (
        rolled_input.shape[:2] == (sequence_length, batch_size)
        and sequence_dim == 0
    ):
        mask = valid.transpose(0, 1)
    else:
        raise ValueError(
            "cannot align document_ids with rolled tensor shape "
            f"{tuple(rolled_input.shape)} along dim {sequence_dim}"
        )
    while mask.dim() < rolled.dim():
        mask = mask.unsqueeze(-1)
    rolled = torch.where(
        mask, rolled, torch.zeros((), dtype=rolled.dtype, device=rolled.device)
    )
    if cp_size > 1:
        sequence_first = scatter_context_parallel_sequence(
            rolled.movedim(sequence_dim, 0),
            cp_group,
        )
        rolled = sequence_first.movedim(0, sequence_dim)
    return rolled, rolled.sum()


def mask_sparse_topk_by_document(topk_indices: torch.Tensor) -> torch.Tensor:
    """Replace cross-document DSA selections with the kernels' -1 sentinel."""

    if topk_indices.dim() != 3:
        raise ValueError("topk_indices must have shape [B,S,K]")
    batch_size, sequence_length, _ = topk_indices.shape
    ids, _spans, multiple = document_layout(
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=topk_indices.device,
    )
    if not multiple:
        return topk_indices
    if ids is None:
        raise RuntimeError("validated document layout did not return document_ids")

    safe = topk_indices.clamp(0, sequence_length - 1).long()
    selected_docs = ids.gather(1, safe.reshape(batch_size, -1)).view_as(topk_indices)
    query_docs = ids.unsqueeze(-1)
    query_positions = torch.arange(
        sequence_length, device=topk_indices.device, dtype=topk_indices.dtype
    ).view(1, -1, 1)
    invalid = (
        (topk_indices < 0)
        | (topk_indices >= sequence_length)
        | (query_docs <= 0)
        | (selected_docs != query_docs)
        | (topk_indices > query_positions)
    )
    return topk_indices.masked_fill(invalid, -1)


def _normalize_window_size(value: Any) -> tuple[int | None, int | None]:
    """Return one validated ``(left, right)`` attention window."""
    if value is None:
        return None, None
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, 0)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("window_size must be an integer or a two-item sequence")
    window = tuple(value)
    if any(
        item is not None
        and (isinstance(item, bool) or not isinstance(item, int) or item < 0)
        for item in window
    ):
        raise ValueError("window_size entries must be non-negative integers or None")
    return window


def _window_size_from_config(
    config: Any,
    *,
    layer_number: int | None = None,
) -> tuple[int | None, int | None]:
    """Return the active per-layer window from a TransformerConfig-like object."""
    if config is None:
        return None, None
    window_size = getattr(config, "window_size", None)
    if window_size is None:
        return None, None
    skip_frequency = getattr(config, "window_attn_skip_freq", None)
    if skip_frequency is not None:
        if layer_number is None:
            raise ValueError(
                "window_attn_skip_freq requires the attention layer number"
            )
        from megatron.core.transformer.utils import is_layer_window_attention

        if not is_layer_window_attention(
            window_size,
            skip_frequency,
            layer_number,
        ):
            return None, None
    return _normalize_window_size(window_size)


def _require_packed_attention_cp1(config: Any, *, backend: str) -> None:
    """Reject packed attention until its document-aware CP path is implemented."""
    if int(getattr(config, "context_parallel_size", 1)) != 1:
        raise NotImplementedError(
            f"packed-document {backend} attention does not yet support context "
            "parallelism; see docs/document_isolation_cp128k_design.md"
        )


def _document_attention_mask(
    ids: torch.Tensor,
    *,
    causal: bool = True,
    window_size: tuple[int | None, int | None] = (None, None),
) -> torch.Tensor:
    sequence_length = ids.shape[1]
    q_idx = torch.arange(sequence_length, device=ids.device).view(1, -1, 1)
    kv_idx = torch.arange(sequence_length, device=ids.device).view(1, 1, -1)
    cross_document = ids[:, :, None] != ids[:, None, :]
    mask = cross_document
    if causal:
        mask = mask | (q_idx < kv_idx)
    window_left, window_right = window_size
    if window_left is not None and window_left >= 0:
        mask |= kv_idx < q_idx - window_left
    if window_right is not None and window_right >= 0:
        mask |= kv_idx > q_idx + window_right
    return mask.unsqueeze(1)


def _reshape_te_output(result: Any, *, batch_size: int, sequence_length: int) -> Any:
    if isinstance(result, tuple):
        return (
            result[0]
            .reshape(batch_size, sequence_length, -1)
            .transpose(0, 1)
            .contiguous(),
            result[1],
        )
    return result.reshape(batch_size, sequence_length, -1).transpose(0, 1).contiguous()


def _slice_attention_tensor(
    tensor: torch.Tensor | None,
    *,
    batch_index: int,
    batch_size: int,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() < 2:
        raise ValueError("attention mask/bias must have query and key dimensions")
    index: list[Any] = [slice(None)] * tensor.dim()
    if tensor.dim() >= 3 and tensor.shape[0] in (1, batch_size):
        source_batch = 0 if tensor.shape[0] == 1 else batch_index
        index[0] = slice(source_batch, source_batch + 1)
    index[-2] = slice(start, end)
    index[-1] = slice(start, end)
    return tensor[tuple(index)]


def _patch_te_attention() -> None:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.packed_seq_params import PackedSeqParams

    installed = getattr(TEDotProductAttention, "forward", None)
    if not isinstance(TEDotProductAttention, type) or not callable(installed):
        # Megatron exposes a MagicMock placeholder when TE is not installed.
        return
    if getattr(installed, _PATCH_MARKER, False):
        return
    signature = inspect.signature(installed)

    @wraps(installed)
    def isolated_forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query"]
        if query.dim() != 4:
            return installed(*bound.args, **bound.kwargs)
        sequence_length, batch_size = query.shape[:2]
        _ids, row_spans, multiple = document_layout(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=query.device,
        )
        if not multiple:
            return installed(*bound.args, **bound.kwargs)
        if bound.arguments["packed_seq_params"] is not None:
            raise RuntimeError(
                "document_ids isolation cannot be combined with upstream packed_seq_params"
            )
        _require_packed_attention_cp1(getattr(self, "config", None), backend="TE")

        attention_bias = bound.arguments["attention_bias"]
        if attention_bias is None:
            lengths = [end - start for spans in row_spans for start, end in spans]
            cu_seqlens = torch.tensor(
                [0, *accumulate(lengths)],
                dtype=torch.int32,
                device=query.device,
            )
            bound.arguments["query"] = query.transpose(0, 1).reshape(
                batch_size * sequence_length, *query.shape[2:]
            )
            for name in ("key", "value"):
                tensor = bound.arguments[name]
                bound.arguments[name] = tensor.transpose(0, 1).reshape(
                    batch_size * sequence_length, *tensor.shape[2:]
                )
            bound.arguments["attention_mask"] = None
            bound.arguments["packed_seq_params"] = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(lengths),
                max_seqlen_kv=max(lengths),
            )
            return _reshape_te_output(
                installed(*bound.args, **bound.kwargs),
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        if not isinstance(attention_bias, torch.Tensor):
            raise RuntimeError(
                "FA4 chunk-native graph bias does not yet support multi-document rows"
            )

        row_outputs = []
        auxiliary = None
        base_arguments = dict(bound.arguments)
        for batch_index, spans in enumerate(row_spans):
            parts = []
            for start, end in spans:
                call = dict(base_arguments)
                for name in ("query", "key", "value"):
                    call[name] = base_arguments[name][
                        start:end, batch_index : batch_index + 1
                    ]
                for name in ("attention_mask", "attention_bias"):
                    call[name] = _slice_attention_tensor(
                        base_arguments[name],
                        batch_index=batch_index,
                        batch_size=batch_size,
                        start=start,
                        end=end,
                    )
                result = installed(
                    self,
                    **{name: value for name, value in call.items() if name != "self"},
                )
                if isinstance(result, tuple):
                    parts.append(result[0])
                    if result[1] is not None:
                        auxiliary = (
                            result[1]
                            if auxiliary is None
                            else torch.maximum(auxiliary, result[1])
                        )
                else:
                    parts.append(result)
            row_outputs.append(torch.cat(parts, dim=0))
        output = torch.cat(row_outputs, dim=1)
        return (output, auxiliary) if auxiliary is not None else output

    setattr(isolated_forward, _PATCH_MARKER, True)
    setattr(TEDotProductAttention, "forward", isolated_forward)


def _patch_torch_attention() -> None:
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    installed = DotProductAttention.forward
    if getattr(installed, _PATCH_MARKER, False):
        return
    signature = inspect.signature(installed)

    @wraps(installed)
    def isolated_forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query"]
        key = bound.arguments["key"]
        sequence_length, batch_size = query.shape[:2]
        if key.shape[0] != sequence_length:
            if _raw_document_ids() is not None:
                raise NotImplementedError(
                    "packed-document torch attention does not support rectangular decode"
                )
            return installed(*bound.args, **bound.kwargs)
        ids, _spans, multiple = document_layout(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=query.device,
        )
        if not multiple:
            return installed(*bound.args, **bound.kwargs)
        if ids is None:
            raise RuntimeError("validated document layout did not return document_ids")
        _require_packed_attention_cp1(getattr(self, "config", None), backend="torch")
        window_size = _window_size_from_config(
            getattr(self, "config", None),
            layer_number=getattr(self, "layer_number", None),
        )
        mask = _document_attention_mask(ids, window_size=window_size)
        existing_mask = bound.arguments["attention_mask"]
        if isinstance(existing_mask, torch.Tensor):
            mask |= existing_mask.to(device=mask.device, dtype=torch.bool)
        bound.arguments["attention_mask"] = mask
        return installed(*bound.args, **bound.kwargs)

    setattr(isolated_forward, _PATCH_MARKER, True)
    DotProductAttention.forward = isolated_forward


def _patch_dsa_attention() -> None:
    from megatron.core.transformer.experimental_attention_variant import dsa as dsa_mod

    installed = dsa_mod.DSAttention.forward
    if getattr(installed, _PATCH_MARKER, False):
        return
    signature = inspect.signature(installed)

    @wraps(installed)
    def isolated_forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        query = bound.arguments["query"]
        sequence_length, batch_size = query.shape[:2]
        ids, _spans, multiple = document_layout(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=query.device,
        )
        if not multiple:
            return installed(*bound.args, **bound.kwargs)
        if ids is None:
            raise RuntimeError("validated document layout did not return document_ids")
        if bound.arguments["packed_seq_params"] is not None:
            raise RuntimeError("DSA document isolation owns packed boundaries")
        _require_packed_attention_cp1(getattr(self, "config", None), backend="DSA")
        backend = dsa_mod.unfused_dsa_fn
        if not getattr(backend, "__cppmega_document_isolation__", False):
            raise RuntimeError(
                "active DSA sparse backend does not support document-isolated -1 indices"
            )
        window_size = _window_size_from_config(
            getattr(self, "config", None),
            layer_number=getattr(self, "layer_number", None),
        )
        mask = _document_attention_mask(ids, window_size=window_size)
        existing_mask = bound.arguments["attention_mask"]
        if isinstance(existing_mask, torch.Tensor):
            mask |= existing_mask.to(device=mask.device, dtype=torch.bool)
        bound.arguments["attention_mask"] = mask
        bound.arguments["attn_mask_type"] = None
        return installed(*bound.args, **bound.kwargs)

    setattr(isolated_forward, _PATCH_MARKER, True)
    dsa_mod.DSAttention.forward = isolated_forward


def _patch_mtp_roll() -> None:
    from megatron.core.transformer import multi_token_prediction as mtp

    installed = mtp.roll_tensor
    if getattr(installed, _PATCH_MARKER, False):
        return

    @wraps(installed)
    def isolated_roll(
        tensor,
        shifts=-1,
        dims=-1,
        cp_group=None,
        packed_seq_params=None,
    ):
        return roll_tensor_by_document(
            tensor,
            shifts=shifts,
            dims=dims,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            fallback=installed,
        )

    setattr(isolated_roll, _PATCH_MARKER, True)
    mtp.roll_tensor = isolated_roll


def _patch_megatron_checkpoint() -> None:
    from megatron.core import tensor_parallel
    from megatron.core.tensor_parallel import random as random_module

    installed = random_module.checkpoint
    if getattr(installed, _PATCH_MARKER, False):
        return

    @wraps(installed)
    def isolated_checkpoint(function, distribute_saved_activations, *args):
        return installed(
            bind_current_structure_batch(function),
            distribute_saved_activations,
            *args,
        )

    setattr(isolated_checkpoint, _PATCH_MARKER, True)
    random_module.checkpoint = isolated_checkpoint
    tensor_parallel.checkpoint = isolated_checkpoint


def _exchange_pipeline_document_ids(
    communicator: Any,
    *,
    tensor_send_next: torch.Tensor | None,
    tensor_recv_prev: torch.Tensor | None,
) -> None:
    send_ids = None
    send_error: Exception | None = None
    send_header = None
    if tensor_send_next is not None:
        try:
            if tensor_send_next.dim() < 2:
                raise ValueError(
                    "pipeline activation must have [S,B,...] dimensions"
                )
            local_sequence_length, batch_size = tensor_send_next.shape[:2]
            raw = _raw_document_ids()
            if raw is not None and raw.dim() == 1:
                raw = raw.unsqueeze(0)
            if raw is None:
                raise RuntimeError(
                    "pipeline activation send requires document_ids"
                )
            if raw.shape[0] != batch_size:
                raise ValueError(
                    f"document_ids batch {raw.shape[0]} != activation batch "
                    f"{batch_size}"
                )
            if raw.shape[1] % local_sequence_length:
                raise ValueError(
                    f"document_ids sequence {raw.shape[1]} is not divisible by "
                    f"local activation sequence {local_sequence_length}"
                )
            send_ids, _spans, _multiple = document_layout(
                batch_size=batch_size,
                sequence_length=raw.shape[1],
                device=tensor_send_next.device,
            )
            if send_ids is None:
                raise RuntimeError(
                    "validated document layout did not return document_ids"
                )
            header_values = (0, *send_ids.shape)
        except Exception as exc:
            send_error = exc
            header_values = (1, 0, 0)
        send_header = torch.tensor(
            header_values,
            dtype=torch.long,
            device=tensor_send_next.device,
        )

    recv_header = None
    recv_pre_error: Exception | None = None
    if tensor_recv_prev is not None:
        if tensor_recv_prev.dim() < 2:
            recv_pre_error = ValueError(
                "pipeline activation must have [S,B,...] dimensions"
            )
        recv_header = torch.empty(
            3,
            dtype=torch.long,
            device=tensor_recv_prev.device,
        )

    header_ops = []
    if send_header is not None:
        header_ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend,
                send_header,
                communicator.next_rank,
                communicator.pp_group,
            )
        )
    if recv_header is not None:
        header_ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_header,
                communicator.prev_rank,
                communicator.pp_group,
            )
        )
    if header_ops:
        for request in torch.distributed.batch_isend_irecv(header_ops):
            request.wait()

    recv_ids = None
    recv_error: Exception | None = None
    recv_ack = None
    if recv_header is not None:
        try:
            status, recv_batch, recv_sequence = (
                int(value) for value in recv_header.tolist()
            )
            if status != 0:
                raise RuntimeError(
                    "upstream pipeline stage rejected its document_ids"
                )
            if recv_pre_error is not None:
                raise recv_pre_error
            if tensor_recv_prev is None:
                raise RuntimeError(
                    "received document_ids shape without an activation"
                )
            local_sequence_length, activation_batch = tensor_recv_prev.shape[:2]
            if (
                recv_batch != activation_batch
                or recv_sequence % local_sequence_length
            ):
                raise ValueError(
                    "received document_ids shape "
                    f"{(recv_batch, recv_sequence)} is incompatible with "
                    f"activation {tuple(tensor_recv_prev.shape[:2])}"
                )
            recv_ids = torch.empty(
                (recv_batch, recv_sequence),
                dtype=torch.long,
                device=tensor_recv_prev.device,
            )
            ack_value = 0
        except Exception as exc:
            recv_error = exc
            ack_value = 1
        recv_ack = torch.tensor(
            [ack_value],
            dtype=torch.long,
            device=tensor_recv_prev.device,
        )

    send_ack = None
    if send_header is not None:
        send_ack = torch.empty(
            1,
            dtype=torch.long,
            device=send_header.device,
        )

    ack_ops = []
    if recv_ack is not None:
        ack_ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend,
                recv_ack,
                communicator.prev_rank,
                communicator.pp_group,
            )
        )
    if send_ack is not None:
        ack_ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv,
                send_ack,
                communicator.next_rank,
                communicator.pp_group,
            )
        )
    if ack_ops:
        for request in torch.distributed.batch_isend_irecv(ack_ops):
            request.wait()

    data_ops = []
    downstream_accepted = (
        send_ack is not None and int(send_ack.item()) == 0
    )
    if send_ids is not None and downstream_accepted:
        data_ops.append(
            torch.distributed.P2POp(
                torch.distributed.isend,
                send_ids.contiguous(),
                communicator.next_rank,
                communicator.pp_group,
            )
        )
    if recv_ids is not None:
        data_ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_ids,
                communicator.prev_rank,
                communicator.pp_group,
            )
        )
    if data_ops:
        for request in torch.distributed.batch_isend_irecv(data_ops):
            request.wait()
    if recv_ids is not None:
        _received_document_ids[id(tensor_recv_prev)] = recv_ids
    if send_error is not None:
        raise send_error
    if send_header is not None and not downstream_accepted:
        raise RuntimeError(
            "downstream pipeline stage rejected document_ids metadata"
        )
    if recv_error is not None:
        raise recv_error


def _patch_pipeline_transport() -> None:
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

    installed = P2PCommunicator._communicate
    if getattr(installed, _PATCH_MARKER, False):
        return

    @wraps(installed)
    def isolated_communicate(self, **kwargs):
        result = installed(self, **kwargs)
        if _structure_enabled():
            _exchange_pipeline_document_ids(
                self,
                tensor_send_next=kwargs["tensor_send_next"],
                tensor_recv_prev=result[0],
            )
        return result

    setattr(isolated_communicate, _PATCH_MARKER, True)
    P2PCommunicator._communicate = isolated_communicate


def _patch_model_input_transport(model_class: type) -> None:
    installed = model_class.set_input_tensor
    if getattr(installed, _PATCH_MARKER, False):
        return

    @wraps(installed)
    def isolated_set_input_tensor(self, input_tensor):
        tensor = input_tensor[0] if isinstance(input_tensor, list) else input_tensor
        if _structure_enabled() and isinstance(tensor, torch.Tensor):
            # Look up by the exact tensor object identity populated during the
            # P2P receive in ``_exchange_pipeline_document_ids``. Any rewrite of
            # ``tensor`` into a different object (``.clone()``, ``.detach()``,
            # or a non-contiguous ``.contiguous()`` copy) changes ``id()`` and
            # breaks this lookup. Keep the activation reference stable.
            document_ids = _received_document_ids.pop(id(tensor), None)
            if document_ids is None:
                raise RuntimeError(
                    f"{model_class.__name__} received a pipeline activation without document_ids"
                )
            _set_current_document_ids(document_ids)
        return installed(self, input_tensor)

    setattr(isolated_set_input_tensor, _PATCH_MARKER, True)
    model_class.set_input_tensor = isolated_set_input_tensor


def apply_document_isolation_patch() -> bool:
    """Install all pinned Megatron seams used by packed cppmega training.

    Design references:
      - `docs/document_isolation_varlen_design.md` — varlen/cu_seqlens path.
      - `docs/document_isolation_swa_design.md` — sliding-window `window_size`
        plumbing for packed-document attention.
      - `docs/document_isolation_cp128k_design.md` — open CP implementation
        contract and H200 parity gates.
    """

    _assert_mamba_cp_signatures()
    _patch_mtp_roll()
    _patch_te_attention()
    _patch_torch_attention()
    _patch_dsa_attention()
    _patch_pipeline_transport()
    _patch_megatron_checkpoint()

    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.mamba.mamba_model import MambaModel

    for model_class in (GPTModel, HybridModel, MambaModel):
        _patch_model_input_transport(model_class)
    return True


__all__ = [
    "apply_document_isolation_patch",
    "bind_current_structure_batch",
    "document_layout",
    "gather_context_parallel_sequence",
    "map_sequence_by_document",
    "map_sharded_sequence_by_document",
    "mask_sparse_topk_by_document",
    "roll_tensor_by_document",
    "scatter_context_parallel_sequence",
]
