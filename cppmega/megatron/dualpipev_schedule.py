"""DualPipeV schedule for NAM56R Megatron training on H200x8.

Integrates DeepSeek's DualPipeV bidirectional pipeline schedule with
Megatron-Core's training loop for the NAM56R hybrid Mamba3+MLA+DSA+MoE model.

DualPipeV with PP=2 creates 4 stages (2N=4), 13 layers each for a 52-layer
model.  Rank 0 holds stages (0, 3), rank 1 holds stages (1, 2).  Only rank 0
provides input/labels and computes loss.

Usage (monkey-patch in pretrain_mamba.py or launch script):

    from cppmega.megatron.dualpipev_schedule import (
        patch_forward_backward_func,
        build_dualpipev_from_megatron_model,
    )

    # After model creation, before training loop:
    dualpipev_state = build_dualpipev_from_megatron_model(
        model, pp_group, pp_rank, pp_degree=2,
        micro_batch_size=mbs, seq_length=seq_len,
        hidden_size=hidden_size,
    )

    # Replace Megatron's forward_backward_func:
    patch_forward_backward_func(dualpipev_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

# ---------------------------------------------------------------------------
# DualPipeV import (must be installed on the training machine)
# ---------------------------------------------------------------------------

from dualpipe import DualPipeV, set_p2p_tensor_shapes, set_p2p_tensor_dtype

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAM56R_NUM_LAYERS = 52
DUALPIPEV_PP_DEGREE = 2
DUALPIPEV_TOTAL_STAGES = 2 * DUALPIPEV_PP_DEGREE  # 4
LAYERS_PER_STAGE = NAM56R_NUM_LAYERS // DUALPIPEV_TOTAL_STAGES  # 13

assert NAM56R_NUM_LAYERS % DUALPIPEV_TOTAL_STAGES == 0, (
    f"NAM56R layer count {NAM56R_NUM_LAYERS} must be divisible by "
    f"{DUALPIPEV_TOTAL_STAGES} stages"
)


# ---------------------------------------------------------------------------
# Stage module wrapper
# ---------------------------------------------------------------------------


class DualPipeVStageModule(nn.Module):
    """Wraps a slice of MambaStack layers into a single DualPipeV stage.

    DualPipeV calls ``module(input_tensor)`` and expects a single output tensor.

    For stage 0 (first stage): includes embedding
    For stage 3 (last stage):  includes final norm + output layer (lm_head)
    For intermediate stages:   pure layer stack pass-through

    The wrapper also handles:
    - requires_grad on inter-stage activations
    - Saved-tensor hooks to clone integer tensors (prevent cross-microbatch
      version conflicts in DualPipeV's overlapped F/B schedule)
    - Forward sequence counter for loss attribution
    """

    def __init__(
        self,
        stage_id: int,
        layers: nn.ModuleList,
        *,
        embedding: Optional[nn.Module] = None,
        final_norm: Optional[nn.Module] = None,
        output_layer: Optional[nn.Module] = None,
        rotary_pos_emb: Optional[nn.Module] = None,
        config: Any = None,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.layers = layers
        self.embedding = embedding
        self.final_norm = final_norm
        self.output_layer = output_layer
        self.rotary_pos_emb = rotary_pos_emb
        self.config = config

        self._is_first = (stage_id == 0)
        self._is_last = (stage_id == DUALPIPEV_TOTAL_STAGES - 1)
        self._forward_seq = 0

    @property
    def is_first_stage(self) -> bool:
        return self._is_first

    @property
    def is_last_stage(self) -> bool:
        return self._is_last

    @staticmethod
    def _pack_integer(tensor: torch.Tensor) -> torch.Tensor:
        """Clone integer tensors to isolate from cross-microbatch inplace ops."""
        if tensor.is_floating_point() or tensor.is_complex():
            return tensor
        return tensor.clone()

    @staticmethod
    def _unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stage.

        DualPipeV works in batch-first format ``[B, T, D]`` with ``batch_dim=0``.
        Megatron layers expect sequence-first format ``[T, B, D]``.

        This method handles the transposition between the two conventions:
        * On entry from DualPipeV: ``[B, T, D]``  (or ``[B, T]`` for input_ids)
        * Transpose to ``[T, B, D]`` for Megatron layers
        * Transpose back to ``[B, T, D]`` before returning to DualPipeV
        """
        self._forward_seq += 1

        # Import lazily to avoid circular imports
        from megatron.core.transformer.transformer_layer import TransformerLayer

        with torch.autograd.graph.saved_tensors_hooks(
            self._pack_integer, self._unpack,
        ):
            if self._is_first:
                assert self.embedding is not None, (
                    "Stage 0 must have embedding module"
                )
                # x is token ids [B, T] from DualPipeV (batch-first)
                x = self.embedding(
                    input_ids=x.long(),
                    position_ids=None,
                )
                # Embedding outputs [T, B, D] (Megatron transposes internally)
                # We keep it in [T, B, D] for the layer loop below
            else:
                # Inter-stage hidden states arrive as [B, T, D] from DualPipeV.
                # Transpose to [T, B, D] for Megatron layers.
                x = x.transpose(0, 1).contiguous()

            # Compute rotary positional embeddings if available.
            rotary_pos_emb_tensor = None
            if self.rotary_pos_emb is not None:
                # x is [T, B, D] -- dim 0 is sequence length
                seq_len = x.size(0)
                rotary_pos_emb_tensor = self.rotary_pos_emb(seq_len)

            # Run through layers (all expect [T, B, D] format)
            for layer in self.layers:
                # Dispatch based on layer type:
                # - TransformerLayer needs rotary_pos_emb for attention
                # - MambaLayer / MoE / MLP just need hidden_states
                if isinstance(layer, TransformerLayer):
                    x = layer(
                        hidden_states=x,
                        attention_mask=None,
                        rotary_pos_emb=rotary_pos_emb_tensor,
                    )
                else:
                    x = layer(
                        hidden_states=x,
                        attention_mask=None,
                    )
                # Megatron layers may return tuples; extract hidden_states
                if isinstance(x, tuple):
                    x = x[0]

            if self._is_last:
                if self.final_norm is not None:
                    x = self.final_norm(x)
                if self.output_layer is not None:
                    # output_layer is ColumnParallelLinear -> returns (output, bias)
                    x, _ = self.output_layer(x)

            # Transpose back to [B, T, D] for DualPipeV P2P and loss
            x = x.transpose(0, 1).contiguous()

        # Ensure requires_grad for inter-stage gradient flow
        if not x.requires_grad and torch.is_grad_enabled():
            x.requires_grad_()

        return x


# ---------------------------------------------------------------------------
# Loss function wrapper
# ---------------------------------------------------------------------------


class DualPipeVLossFn:
    """Callable loss function for DualPipeV's criterion argument.

    DualPipeV calls ``criterion(logits, labels)`` on the stage that produces
    output (only on rank 0).  This is called once per chunk (so
    ``num_chunks`` times per ``.step()`` call).

    Computes:
    1. Cross-entropy loss (token-level, with loss_mask if available)
    2. MTP loss (via FastMTP if the model has it configured)

    MoE auxiliary losses are NOT collected here.  Megatron-Core's
    ``MoEAuxLossAutoScaler`` already injects them into the backward graph
    during each layer's forward pass.  Collecting them again in the
    criterion would double-count.

    Loss is scaled by ``1 / num_microbatches`` to match Megatron's gradient
    scaling convention (Megatron does ``output_tensor /= num_microbatches``
    in its forward_step).  DualPipeV calls ``.backward()`` on each chunk's
    loss independently, so the scaling ensures accumulated gradients over
    all chunks and all DualPipeV steps equal the non-PP path.
    """

    def __init__(
        self,
        stage_modules: Tuple[DualPipeVStageModule, DualPipeVStageModule],
        model: nn.Module,
        *,
        num_microbatches: int,
        num_chunks: int,
    ):
        self.stage_modules = stage_modules
        self.model = model
        self.num_microbatches = num_microbatches
        self.num_chunks = num_chunks

        # Track which chunk we are on for loss_mask indexing
        self._chunk_idx = 0
        self._loss_masks: List[torch.Tensor] = []

        # Cache FastMTP state
        self._has_fastmtp = (
            hasattr(model, '_fastmtp')
            and model._fastmtp is not None
        )

    def set_loss_masks(self, loss_masks: List[torch.Tensor]):
        """Pre-load loss masks for all chunks before calling DualPipeV.step()."""
        self._loss_masks = loss_masks
        self._chunk_idx = 0

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for one DualPipeV chunk.

        Args:
            logits: (B_chunk, T, V) from the last stage
            labels: (B_chunk, T) target token ids

        Returns:
            Scalar loss tensor with gradient attached.
        """
        # DualPipeV's bidirectional schedule interleaves forward and reverse
        # chunks.  The criterion receives (logits, labels) for each chunk
        # regardless of which direction produced it.  We simply compute the
        # cross-entropy loss without tracking which stage module produced the
        # logits -- DualPipeV handles the mapping internally.

        # Cross-entropy loss
        logits_flat = logits.view(-1, logits.size(-1)).float()
        labels_flat = labels.view(-1)

        if self._loss_masks and self._chunk_idx < len(self._loss_masks):
            loss_mask = self._loss_masks[self._chunk_idx].view(-1).float()
            per_token_loss = torch.nn.functional.cross_entropy(
                logits_flat, labels_flat, reduction='none',
            )
            loss = torch.sum(per_token_loss * loss_mask) / loss_mask.sum().clamp(min=1)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits_flat, labels_flat, ignore_index=-1,
            )

        self._chunk_idx += 1

        # FastMTP loss: the last-stage module captures hidden states before
        # the final norm during forward.  If FastMTP is configured, compute
        # the auxiliary multi-token prediction loss here.
        if self._has_fastmtp:
            _model = self.model
            # Navigate wrapping
            for _ in range(5):
                if hasattr(_model, '_fastmtp'):
                    break
                if hasattr(_model, 'module'):
                    _model = _model.module
            if hasattr(_model, '_fastmtp') and _model._fastmtp is not None:
                if (hasattr(_model, '_fastmtp_hidden_cache')
                        and _model._fastmtp_hidden_cache is not None):
                    mtp_loss = _model._fastmtp(
                        hidden_states=_model._fastmtp_hidden_cache,
                        labels=labels,
                        output_layer=_model.output_layer,
                        loss_mask=(
                            self._loss_masks[self._chunk_idx - 1]
                            if self._loss_masks else None
                        ),
                    )
                    _model._fastmtp_hidden_cache = None
                    mtp_lambda = getattr(_model, '_fastmtp_lambda', 0.3)
                    if mtp_lambda > 0:
                        loss = loss + mtp_lambda * mtp_loss

        # Scale: each chunk's loss is backward'd independently.
        # Total backward calls per optimizer step = num_microbatches
        # (num_chunks calls per step, num_steps = num_microbatches/num_chunks
        #  steps per optimizer iteration, total = num_chunks * num_steps = num_microbatches).
        if self.num_microbatches > 1:
            loss = loss / self.num_microbatches

        return loss


# ---------------------------------------------------------------------------
# Model splitting
# ---------------------------------------------------------------------------


def _split_layers(
    model: nn.Module,
) -> Tuple[
    nn.Module,           # embedding
    nn.ModuleList,       # all decoder layers
    Optional[nn.Module], # final_norm
    Optional[nn.Module], # output_layer
    Optional[nn.Module], # rotary_pos_emb
    Any,                 # config
]:
    """Extract components from a MambaModel/CppMegaMambaModel.

    Works with both megatron.core.models.mamba.MambaModel and
    cppmega.megatron.custom_mamba_model.CppMegaMambaModel.

    Handles Megatron wrapping layers: DDP -> Float16Module -> actual model.
    """
    # Navigate wrapped model (DDP, Float16Module, etc.)
    unwrapped = model
    # Peel up to 5 levels of .module wrapping (DDP -> Float16Module -> ...)
    for _ in range(5):
        if hasattr(unwrapped, 'decoder'):
            break
        if hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        else:
            break

    assert hasattr(unwrapped, 'decoder'), (
        f"Cannot find 'decoder' attribute on model type {type(unwrapped).__name__}. "
        f"Expected MambaModel or CppMegaMambaModel."
    )

    decoder = unwrapped.decoder  # MambaStack
    assert hasattr(decoder, 'layers'), (
        f"decoder (type {type(decoder).__name__}) has no 'layers' attribute"
    )

    num_layers = len(decoder.layers)
    assert num_layers == NAM56R_NUM_LAYERS, (
        f"Expected {NAM56R_NUM_LAYERS} layers, got {num_layers}"
    )

    embedding = getattr(unwrapped, 'embedding', None)
    final_norm = getattr(decoder, 'final_norm', None)
    output_layer = getattr(unwrapped, 'output_layer', None)
    rotary_pos_emb = getattr(unwrapped, 'rotary_pos_emb', None)
    config = getattr(unwrapped, 'config', None)

    return embedding, decoder.layers, final_norm, output_layer, rotary_pos_emb, config


def build_stages(
    model: nn.Module,
    pp_rank: int,
    device: torch.device,
) -> Tuple[DualPipeVStageModule, DualPipeVStageModule]:
    """Split NAM56R model into the two stages owned by this PP rank.

    DualPipeV PP=2:
        Rank 0: stages (0, 3) -> layers [0:13] and [39:52]
        Rank 1: stages (1, 2) -> layers [13:26] and [26:39]

    Stage 0 gets embedding, stage 3 gets final_norm + output_layer.

    Args:
        model: Full MambaModel (all 52 layers on every rank initially).
        pp_rank: This rank's position (0 or 1 for PP=2).
        device: Target CUDA device.

    Returns:
        (forward_stage, reverse_stage) ready for DualPipeV.
    """
    assert pp_rank in (0, 1), f"DualPipeV PP=2 requires pp_rank in {{0, 1}}, got {pp_rank}"

    embedding, all_layers, final_norm, output_layer, rotary_pos_emb, config = _split_layers(model)

    fwd_stage_id = pp_rank
    rev_stage_id = DUALPIPEV_TOTAL_STAGES - 1 - pp_rank

    def get_stage_layers(stage_id: int) -> nn.ModuleList:
        start = stage_id * LAYERS_PER_STAGE
        end = start + LAYERS_PER_STAGE
        return nn.ModuleList(list(all_layers[start:end]))

    fwd_layers = get_stage_layers(fwd_stage_id)
    rev_layers = get_stage_layers(rev_stage_id)

    # rotary_pos_emb is shared across all stages that contain attention layers.
    # Both forward and reverse stages may have TransformerLayer layers that need it.
    fwd_stage = DualPipeVStageModule(
        stage_id=fwd_stage_id,
        layers=fwd_layers,
        embedding=embedding if fwd_stage_id == 0 else None,
        final_norm=final_norm if fwd_stage_id == DUALPIPEV_TOTAL_STAGES - 1 else None,
        output_layer=output_layer if fwd_stage_id == DUALPIPEV_TOTAL_STAGES - 1 else None,
        rotary_pos_emb=rotary_pos_emb,
        config=config,
    )
    rev_stage = DualPipeVStageModule(
        stage_id=rev_stage_id,
        layers=rev_layers,
        embedding=embedding if rev_stage_id == 0 else None,
        final_norm=final_norm if rev_stage_id == DUALPIPEV_TOTAL_STAGES - 1 else None,
        output_layer=output_layer if rev_stage_id == DUALPIPEV_TOTAL_STAGES - 1 else None,
        rotary_pos_emb=rotary_pos_emb,
        config=config,
    )

    fwd_stage.to(device)
    rev_stage.to(device)

    return fwd_stage, rev_stage


# ---------------------------------------------------------------------------
# DualPipeV state
# ---------------------------------------------------------------------------


@dataclass
class DualPipeVState:
    """Holds all state needed for DualPipeV training."""
    dualpipev: Any  # DualPipeV instance
    fwd_stage: DualPipeVStageModule
    rev_stage: DualPipeVStageModule
    loss_fn: DualPipeVLossFn
    pp_rank: int
    pp_degree: int
    num_chunks: int
    micro_batch_size: int
    seq_length: int
    hidden_size: int
    model: nn.Module  # original unwrapped model reference


def build_dualpipev_from_megatron_model(
    model: nn.Module,
    pp_group: dist.ProcessGroup,
    pp_rank: int,
    *,
    pp_degree: int = DUALPIPEV_PP_DEGREE,
    micro_batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_microbatches: int = 1,
    num_chunks: int = 4,
    dtype: torch.dtype = torch.bfloat16,
) -> DualPipeVState:
    """Build DualPipeV from a Megatron MambaModel.

    DualPipeV splits its input along the batch dimension into ``num_chunks``
    pieces.  Each piece is one "chunk" that flows through the pipeline.
    The P2P tensor shape must match the chunk size, not the full input.

    We feed ``num_chunks`` Megatron micro-batches per DualPipeV ``.step()``
    call, so the total input is ``(micro_batch_size * num_chunks, T)`` and
    each chunk is ``(micro_batch_size, T)``.  This way any ``micro_batch_size``
    works regardless of divisibility.

    Args:
        model: MambaModel or CppMegaMambaModel with all 52 layers.
        pp_group: NCCL process group for PP communication.
        pp_rank: This rank's PP position (0 or 1).
        pp_degree: Number of PP ranks (must be 2).
        micro_batch_size: Per-device micro-batch size (Megatron's --micro-batch-size).
        seq_length: Sequence length.
        hidden_size: Model hidden dimension (4096 for NAM56R).
        num_microbatches: Number of gradient accumulation micro-batches.
        num_chunks: DualPipeV chunking factor (>= 2 * pp_degree = 4).
            Each DualPipeV step consumes num_chunks Megatron micro-batches.
        dtype: Communication dtype.

    Returns:
        DualPipeVState with everything needed for training.
    """
    assert pp_degree == DUALPIPEV_PP_DEGREE, (
        f"This module is configured for PP={DUALPIPEV_PP_DEGREE}, got {pp_degree}"
    )
    assert num_chunks >= 2 * pp_degree, (
        f"DualPipeV requires num_chunks >= 2*pp_degree={2*pp_degree}, got {num_chunks}"
    )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Split model into stages
    fwd_stage, rev_stage = build_stages(model, pp_rank, device)

    # Configure P2P communication shapes.
    # Each chunk is one Megatron micro-batch: (micro_batch_size, T, D).
    # DualPipeV splits the full input (micro_batch_size * num_chunks, T)
    # into num_chunks pieces along batch_dim, each of size micro_batch_size.
    set_p2p_tensor_shapes([(micro_batch_size, seq_length, hidden_size)])
    set_p2p_tensor_dtype(dtype)

    # Create DualPipeV instance
    dualpipev = DualPipeV(
        modules=(fwd_stage, rev_stage),
        batch_dim=0,
        process_group=pp_group,
    )

    # Create loss function
    loss_fn = DualPipeVLossFn(
        stage_modules=(fwd_stage, rev_stage),
        model=model,
        num_microbatches=num_microbatches,
        num_chunks=num_chunks,
    )

    return DualPipeVState(
        dualpipev=dualpipev,
        fwd_stage=fwd_stage,
        rev_stage=rev_stage,
        loss_fn=loss_fn,
        pp_rank=pp_rank,
        pp_degree=pp_degree,
        num_chunks=num_chunks,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        model=model,
    )


# ---------------------------------------------------------------------------
# Training step replacement
# ---------------------------------------------------------------------------


def dualpipev_forward_backward(
    state: DualPipeVState,
    *,
    forward_step_func: Callable,
    data_iterator: Any,
    model: Any,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[Any] = None,
    pg_collection: Optional[Any] = None,
    force_all_reduce: Optional[bool] = False,
) -> List[Dict[str, torch.Tensor]]:
    """Drop-in replacement for Megatron's forward_backward_no_pipelining.

    Accepts the same kwargs as forward_backward_no_pipelining but internally
    uses DualPipeV for the forward+backward pass.

    Each DualPipeV ``.step()`` call processes ``num_chunks`` micro-batches
    (accumulated from the data iterator).  Megatron's ``num_microbatches``
    must be divisible by ``num_chunks``.  We call ``.step()``
    ``num_microbatches // num_chunks`` times per training iteration.

    Only rank 0 provides input and computes loss.  All ranks participate in
    every DualPipeV step for NCCL P2P communication.

    Returns:
        forward_data_store: List of loss dicts, one per DualPipeV step, in
        the same format as Megatron's forward_backward_no_pipelining:
        ``[{'lm loss': tensor([loss_val, num_tokens])}]``
    """
    import contextlib
    from megatron.core.utils import get_model_config

    if isinstance(model, list):
        assert len(model) == 1
        model_ref = model[0]
    else:
        model_ref = model

    config = get_model_config(model_ref)

    num_chunks = state.num_chunks
    assert num_microbatches >= num_chunks, (
        f"num_microbatches ({num_microbatches}) must be >= num_chunks ({num_chunks}). "
        f"Increase --num-microbatches or decrease num_chunks."
    )
    assert num_microbatches % num_chunks == 0, (
        f"num_microbatches ({num_microbatches}) must be divisible by "
        f"num_chunks ({num_chunks})"
    )
    num_steps = num_microbatches // num_chunks

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )

    # Set up no_sync context for DDP.  DualPipeV calls backward multiple
    # times inside .step(), so we must defer DDP gradient sync.
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):
        import contextlib as _ctxlib
        _no_sync_funcs = no_sync_func
        @_ctxlib.contextmanager
        def no_sync_func():
            with _ctxlib.ExitStack() as stack:
                for fn in _no_sync_funcs:
                    stack.enter_context(fn())
                yield
    elif no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    forward_data_store: List[Dict[str, torch.Tensor]] = []

    for step_idx in range(num_steps):
        # Accumulate num_chunks micro-batches from the data iterator.
        # Each micro-batch has (micro_batch_size, T); concatenated along
        # batch dim gives (micro_batch_size * num_chunks, T).
        input_ids = None
        labels = None
        loss_masks: List[torch.Tensor] = []

        if state.pp_rank == 0:
            all_tokens: List[torch.Tensor] = []
            all_labels: List[torch.Tensor] = []

            for chunk_idx in range(num_chunks):
                batch = _get_batch_from_iterator(data_iterator)
                assert batch['tokens'] is not None, (
                    f"Data iterator returned None tokens on rank 0 "
                    f"(step {step_idx}, chunk {chunk_idx})"
                )
                assert batch['labels'] is not None, (
                    f"Data iterator returned None labels on rank 0 "
                    f"(step {step_idx}, chunk {chunk_idx})"
                )
                all_tokens.append(batch['tokens'])
                all_labels.append(batch['labels'])

                if batch.get('loss_mask') is not None:
                    loss_masks.append(batch['loss_mask'])

            # Concatenate along batch dimension
            input_ids = torch.cat(all_tokens, dim=0)
            labels = torch.cat(all_labels, dim=0)

            state.loss_fn.set_loss_masks(loss_masks)

        # Run DualPipeV step inside no_sync context to defer DDP gradient
        # all-reduce until after finalize_model_grads.  DualPipeV calls
        # backward() per-chunk internally, which would otherwise trigger
        # premature gradient sync.
        with no_sync_func():
            loss_tensor, _ = state.dualpipev.step(
                input_ids,
                num_chunks=num_chunks,
                criterion=state.loss_fn if state.pp_rank == 0 else None,
                labels=[labels] if state.pp_rank == 0 else [],
                return_outputs=False,
            )

        # Collect loss data for Megatron's training loop
        if state.pp_rank == 0:
            assert loss_tensor is not None, (
                "DualPipeV returned None loss on rank 0"
            )
            # loss_tensor shape: [num_chunks] -- sum is the total scaled loss
            # (each chunk's loss was already divided by num_microbatches*num_chunks
            #  inside DualPipeVLossFn)
            total_loss = loss_tensor.sum()
            num_tokens = torch.tensor(
                [micro_batch_size * seq_length * num_chunks],
                dtype=torch.int, device=total_loss.device,
            )

            forward_data_store.append({
                'lm loss': torch.cat([
                    total_loss.clone().detach().view(1),
                    num_tokens.float().view(1),
                ]),
            })
        else:
            # Non-rank-0 still needs to return something for Megatron
            forward_data_store.append({
                'lm loss': torch.zeros(2, device='cuda'),
            })

    # Finalize gradients (DP all-reduce / reduce-scatter)
    if config.finalize_model_grads_func is not None and not forward_only:
        total_num_tokens = torch.tensor(
            [micro_batch_size * seq_length * num_microbatches],
            dtype=torch.int, device='cuda',
        )
        config.finalize_model_grads_func(
            [model_ref],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def _get_batch_from_iterator(
    data_iterator: Any,
) -> Dict[str, Optional[torch.Tensor]]:
    """Extract one micro-batch dict from Megatron's data iterator.

    Megatron data iterators yield dicts with keys:
    tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen

    For DualPipeV, we need tokens and labels.  Loss mask is used for
    proper loss scaling.
    """
    assert data_iterator is not None, (
        "_get_batch_from_iterator called with None iterator on rank 0"
    )

    if isinstance(data_iterator, list):
        data_iterator = data_iterator[0]

    try:
        batch = next(data_iterator)
    except StopIteration:
        assert False, "Data iterator exhausted during DualPipeV training step"

    def _to_cuda(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Move tensor to current CUDA device if it's on CPU."""
        if t is None:
            return None
        if isinstance(t, torch.Tensor) and not t.is_cuda:
            return t.cuda(non_blocking=True)
        return t

    # Handle different batch formats
    if isinstance(batch, dict):
        return {
            'tokens': _to_cuda(batch.get('tokens')),
            'labels': _to_cuda(batch.get('labels')),
            'loss_mask': _to_cuda(batch.get('loss_mask')),
        }
    elif isinstance(batch, (tuple, list)):
        # pretrain_mamba.get_batch returns values as a tuple:
        # (tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen)
        keys = ['tokens', 'labels', 'loss_mask', 'attention_mask',
                'position_ids', 'cu_seqlens', 'max_seqlen']
        result: Dict[str, Optional[torch.Tensor]] = {}
        for i, key in enumerate(keys):
            result[key] = _to_cuda(batch[i]) if i < len(batch) else None
        return result
    else:
        assert False, f"Unexpected batch type from data iterator: {type(batch)}"


# ---------------------------------------------------------------------------
# Megatron integration: monkey-patch get_forward_backward_func
# ---------------------------------------------------------------------------


def make_dualpipev_forward_backward_func(
    state: DualPipeVState,
) -> Callable:
    """Create a forward_backward function compatible with Megatron's interface.

    Returns a function with the same signature as
    forward_backward_no_pipelining that internally uses DualPipeV.
    """
    def _forward_backward(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
        adjust_tensor_shapes_fn=None,
        p2p_communicator=None,
        pg_collection=None,
        force_all_reduce=False,
    ):
        return dualpipev_forward_backward(
            state,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            forward_only=forward_only,
            collect_non_loss_data=collect_non_loss_data,
            first_val_step=first_val_step,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            p2p_communicator=p2p_communicator,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return _forward_backward


def patch_forward_backward_func(state: DualPipeVState) -> None:
    """Monkey-patch Megatron's get_forward_backward_func to return DualPipeV.

    After calling this, any code that calls
    ``get_forward_backward_func()`` will get the DualPipeV schedule.

    This is the recommended integration point -- call it after model
    creation but before the training loop starts.

    Usage in a launch script or pretrain_mamba.py modification:

        import megatron.core.pipeline_parallel.schedules as schedules

        dualpipev_state = build_dualpipev_from_megatron_model(...)
        patch_forward_backward_func(dualpipev_state)

        # Now Megatron's train() will use DualPipeV
    """
    import megatron.core.pipeline_parallel.schedules as schedules
    import megatron.core.pipeline_parallel as pp_module

    dualpipev_func = make_dualpipev_forward_backward_func(state)

    def _patched_get(*args, **kwargs):
        return dualpipev_func

    schedules.get_forward_backward_func = _patched_get

    # Also patch the re-export in the pipeline_parallel __init__
    if hasattr(pp_module, 'get_forward_backward_func'):
        pp_module.get_forward_backward_func = _patched_get

    # Patch the binding in megatron.training.training which uses
    # `from megatron.core.pipeline_parallel import get_forward_backward_func`
    # (the `from ... import` creates a local name that our module-level
    # patches above do not affect).
    try:
        import megatron.training.training as _training_mod
        _training_mod.get_forward_backward_func = _patched_get
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Convenience: full setup from Megatron args
# ---------------------------------------------------------------------------


def setup_dualpipev_from_args(
    model: nn.Module,
    args: Any,
) -> DualPipeVState:
    """One-call setup using Megatron's parsed args.

    Call this after model creation:

        from megatron.training import get_args
        from cppmega.megatron.dualpipev_schedule import (
            setup_dualpipev_from_args, patch_forward_backward_func,
        )

        args = get_args()
        # model is a list of [wrapped_model] from Megatron's setup_model_and_optimizer
        state = setup_dualpipev_from_args(model[0], args)
        patch_forward_backward_func(state)

    The ``num_chunks`` parameter is set to ``2 * pp_degree`` (=4 for PP=2).
    Megatron's ``num_microbatches`` must be >= ``num_chunks`` and divisible
    by it.  Each DualPipeV step consumes ``num_chunks`` micro-batches from
    the data iterator.

    Args:
        model: Single model module (not a list).  May be wrapped in
            Float16Module / DDP -- the unwrapping logic handles that.
        args: Megatron args namespace with standard fields.

    Returns:
        DualPipeVState ready for training.
    """
    from megatron.core import parallel_state

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_group = parallel_state.get_pipeline_model_parallel_group()

    # Validate PP world size matches our expectation
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pp_world_size == DUALPIPEV_PP_DEGREE, (
        f"DualPipeV requires --pipeline-model-parallel-size={DUALPIPEV_PP_DEGREE}, "
        f"got {pp_world_size}"
    )

    micro_batch_size = args.micro_batch_size
    seq_length = args.seq_length
    hidden_size = args.hidden_size

    # Compute num_microbatches from Megatron's global/micro batch sizes
    # global_batch_size = micro_batch_size * num_microbatches * dp_size
    dp_size = parallel_state.get_data_parallel_world_size()
    global_batch_size = args.global_batch_size
    num_microbatches = global_batch_size // (micro_batch_size * dp_size)

    # num_chunks = 2 * pp_degree is the minimum for DualPipeV
    num_chunks = 2 * DUALPIPEV_PP_DEGREE  # = 4

    assert num_microbatches >= num_chunks, (
        f"DualPipeV needs num_microbatches ({num_microbatches}) >= "
        f"num_chunks ({num_chunks}). Increase global_batch_size or "
        f"decrease micro_batch_size."
    )
    assert num_microbatches % num_chunks == 0, (
        f"num_microbatches ({num_microbatches}) must be divisible by "
        f"num_chunks ({num_chunks}). Adjust global_batch_size."
    )

    return build_dualpipev_from_megatron_model(
        model,
        pp_group,
        pp_rank,
        pp_degree=DUALPIPEV_PP_DEGREE,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_microbatches=num_microbatches,
        num_chunks=num_chunks,
        dtype=torch.bfloat16,
    )
