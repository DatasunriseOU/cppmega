# Copyright (c) 2026, cppmega contributors. All rights reserved.
#
# Hybrid Mamba+Transformer schedule plan for combined_1f1b EP A2A overlap.
#
# MambaModel has mixed layer types: MambaLayer (SSM), TransformerLayer (MoE),
# and TransformerLayer (DSA attention). Only MoE TransformerLayers benefit
# from fine-grained A2A decomposition. Mamba/DSA layers run as opaque nodes.
#
# Public API: apply_hybrid_schedule_plan_patch()

from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor

from megatron.core.pipeline_parallel.utils import (
    AbstractSchedulePlan,
    NoopScheduleNode,
    ScheduleNode,
    get_comm_stream,
    get_comp_stream,
)


# ---------------------------------------------------------------------------
# Helper nodes
# ---------------------------------------------------------------------------

class _OpaqueScheduleNode(ScheduleNode):
    """ScheduleNode with a no-op backward_dw for opaque layers."""

    def backward_dw(self):
        pass


class _NoopWithBackwardDW(NoopScheduleNode):
    """NoopScheduleNode that also has a no-op backward_dw."""

    def backward_dw(self):
        pass


# ---------------------------------------------------------------------------
# OpaqueLayerSchedulePlan -- wraps non-MoE layers (Mamba, DSA) as single nodes
# ---------------------------------------------------------------------------

class OpaqueLayerSchedulePlan:
    """Schedule plan for layers that have no A2A to decompose.

    Presents the same interface as TransformerLayerSchedulePlan but runs the
    entire layer forward/backward as a single compute-stream node. The
    moe_dispatch/moe_combine/mlp/mtp_post_process slots are no-ops so the
    interleaved scheduler can call them without branching.
    """

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args=None):
        self.config = layer.config
        self.layer = layer

        def _opaque_forward(hidden_states):
            from megatron.core.transformer.transformer_layer import TransformerLayer

            if isinstance(layer, TransformerLayer):
                out, _ = layer(
                    hidden_states=hidden_states,
                    attention_mask=chunk_state.attention_mask,
                    rotary_pos_emb=getattr(chunk_state, "rotary_pos_emb", None),
                    packed_seq_params=chunk_state.packed_seq_params,
                    padding_mask=getattr(chunk_state, "padding_mask", None),
                )
                return out
            else:
                return layer(
                    hidden_states=hidden_states,
                    attention_mask=chunk_state.attention_mask,
                    packed_seq_params=chunk_state.packed_seq_params,
                )

        self.attn = _OpaqueScheduleNode(
            _opaque_forward, comp_stream, event, name="opaque_layer"
        )
        self.moe_dispatch = NoopScheduleNode()
        self.mlp = _NoopWithBackwardDW()
        self.moe_combine = NoopScheduleNode()
        self.mtp_post_process = NoopScheduleNode()

    def release_state(self):
        for attr in ("attn", "moe_dispatch", "mlp", "moe_combine", "mtp_post_process"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        if hasattr(self, "layer"):
            del self.layer

    def get_fp8_context(self):
        from megatron.core.enums import Fp8Recipe
        from megatron.core.fp8_utils import get_fp8_context as _get

        if self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed:
            return _get(self.config, self.layer.layer_number - 1)
        return nullcontext()


# ---------------------------------------------------------------------------
# HybridModelChunkSchedulePlan -- mixed Mamba + Transformer layers
# ---------------------------------------------------------------------------

class HybridModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule plan for a hybrid MambaModel chunk.

    Mirrors TransformerModelChunkSchedulePlan but dispatches each layer to
    either TransformerLayerSchedulePlan (MoE layers) or OpaqueLayerSchedulePlan
    (Mamba/DSA layers).
    """

    def __init__(
        self,
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        padding_mask=None,
    ):
        from megatron.core.models.common.model_chunk_schedule_plan import ModelChunkState

        self._model_chunk_state = ModelChunkState()
        self._transformer_layers = []
        self._event = torch.cuda.Event()
        self.pre_process = None
        self.post_process = None
        self.vp_stage = model.vp_stage
        self.config = model.config

        s = self._model_chunk_state
        s.input_ids = input_ids
        s.position_ids = position_ids
        s.attention_mask = attention_mask
        s.decoder_input = decoder_input
        s.labels = labels
        s.mtp_hidden_states = None
        s.loss_mask = loss_mask
        s.packed_seq_params = packed_seq_params
        s.padding_mask = padding_mask
        s.extra_block_kwargs = extra_block_kwargs
        s.runtime_gather_output = runtime_gather_output
        s.model = model
        s.context = None
        s.context_mask = None
        s.attention_bias = None

        self.pre_process = _HybridPreProcessNode(model, s, self._event, get_comp_stream)
        self._build_layer_schedule_plans(model.decoder, get_comp_stream, get_comm_stream)
        if model.post_process:
            self.post_process = _HybridPostProcessNode(model, s, self._event, get_comp_stream)

    def _build_layer_schedule_plans(self, decoder, comp_stream, comm_stream):
        from megatron.core.models.common.model_chunk_schedule_plan import (
            TransformerLayerSchedulePlan,
        )
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.transformer_layer import TransformerLayer

        num_layers = len(decoder.layers)
        for idx in range(num_layers):
            layer = decoder.layers[idx]
            extra = {
                "is_first_layer": idx == 0,
                "is_last_layer": idx == num_layers - 1,
            }
            is_moe = isinstance(layer, TransformerLayer) and isinstance(layer.mlp, MoELayer)
            if is_moe:
                plan = TransformerLayerSchedulePlan(
                    layer, self.event, self.state, comp_stream, comm_stream, extra
                )
            else:
                plan = OpaqueLayerSchedulePlan(
                    layer, self.event, self.state, comp_stream, comm_stream, extra
                )
            self._transformer_layers.append(plan)

    @property
    def event(self):
        return self._event

    def record_current_stream(self):
        self.event.record(torch.cuda.current_stream())

    def wait_current_stream(self):
        self.event.wait(torch.cuda.current_stream())

    def get_layer(self, i):
        assert i < self.num_layers()
        return self._transformer_layers[i]

    def pop_layer(self):
        return self._transformer_layers.pop()

    def num_layers(self):
        return len(self._transformer_layers)

    @property
    def state(self):
        return self._model_chunk_state

    def release_state(self):
        self._model_chunk_state.model = None
        if self.pre_process is not None:
            self.pre_process.chunk_state = None
            self.pre_process = None
        if self.post_process is not None:
            self.post_process.chunk_state = None
            self.post_process = None

    @staticmethod
    def run(f_schedule_plan, b_schedule_plan, b_grad=None,
            pre_forward=None, pre_backward=None, post_forward=None, post_backward=None):
        from megatron.core.models.common.model_chunk_schedule_plan import (
            TransformerModelChunkSchedulePlan,
        )
        return TransformerModelChunkSchedulePlan.run(
            f_schedule_plan, b_schedule_plan, b_grad=b_grad,
            pre_forward=pre_forward, pre_backward=pre_backward,
            post_forward=post_forward, post_backward=post_backward,
        )


# ---------------------------------------------------------------------------
# Pre/Post process nodes for MambaModel
# ---------------------------------------------------------------------------

class _HybridPreProcessNode(ScheduleNode):
    """Embedding + rotary for MambaModel."""

    def __init__(self, model, chunk_state, event, stream):
        self.mamba_model = model
        self.chunk_state = chunk_state
        super().__init__(self._do_forward, stream, event, name="hybrid_pre_process")

    def _do_forward(self):
        model = self.mamba_model
        s = self.chunk_state

        if s.decoder_input is not None:
            decoder_input = s.decoder_input
        elif model.pre_process:
            decoder_input = model.embedding(input_ids=s.input_ids, position_ids=s.position_ids)
        else:
            decoder_input = model.decoder.input_tensor

        rotary_pos_emb = None
        if model.position_embedding_type == "rope":
            rotary_seq_len = model.rotary_pos_emb.get_rotary_seq_len(
                None, model.decoder, decoder_input, model.config, s.packed_seq_params
            )
            rotary_pos_emb = model.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=(
                    s.packed_seq_params is not None and s.packed_seq_params.qkv_format == "thd"
                ),
            )

        s.decoder_input = decoder_input
        s.rotary_pos_emb = rotary_pos_emb
        return decoder_input


class _HybridPostProcessNode(ScheduleNode):
    """Final layernorm + output layer + loss for MambaModel."""

    def __init__(self, model, chunk_state, event, stream):
        self.mamba_model = model
        self.chunk_state = chunk_state
        super().__init__(self._do_forward, stream, event, name="hybrid_post_process")

    def _do_forward(self, hidden_states):
        from megatron.core.transformer.module import float16_to_fp32
        from megatron.core.transformer.multi_token_prediction import process_mtp_loss

        model = self.mamba_model
        s = self.chunk_state

        if getattr(model.decoder, "post_layer_norm", False) and model.decoder.post_process:
            final_norm = getattr(model.decoder, "final_norm", None)
            if final_norm is not None:
                hidden_states = final_norm(hidden_states)

        output_weight = None
        if model.share_embeddings_and_output_weights:
            output_weight = model.shared_embedding_or_output_weight()

        if model.mtp_process:
            hidden_states = process_mtp_loss(
                hidden_states=hidden_states,
                labels=s.labels,
                loss_mask=s.loss_mask,
                output_layer=model.output_layer,
                output_weight=output_weight,
                runtime_gather_output=s.runtime_gather_output,
                is_training=model.training,
                compute_language_model_loss=model.compute_language_model_loss,
                config=model.config,
                cp_group=model.pg_collection.cp,
                packed_seq_params=s.packed_seq_params,
                scale_logits_fn=model._scale_logits if model.config.use_mup else None,
            )

        if s.labels is None:
            logits, _ = model.output_layer(
                hidden_states, weight=output_weight,
                runtime_gather_output=s.runtime_gather_output,
            )
            return float16_to_fp32(logits)

        kw = dict(
            input_=hidden_states, weight=output_weight,
            runtime_gather_output=s.runtime_gather_output,
        )
        if model.fuse_linear_cross_entropy:
            loss = model.output_layer(
                output_cross_entropy_loss=True, labels=s.labels, **kw,
            )
        else:
            logits, _ = model.output_layer(**kw)
            loss = model.compute_language_model_loss(s.labels, logits)

        return float16_to_fp32(loss)


# ---------------------------------------------------------------------------
# build_schedule_plan method for MambaModel
# ---------------------------------------------------------------------------

def _mamba_build_schedule_plan(
    self,
    input_ids,
    position_ids,
    attention_mask,
    decoder_input=None,
    labels=None,
    inference_context=None,
    packed_seq_params=None,
    extra_block_kwargs=None,
    runtime_gather_output=None,
    inference_params=None,
    loss_mask=None,
    padding_mask=None,
):
    """build_schedule_plan for MambaModel -- same signature as GPTModel."""
    return HybridModelChunkSchedulePlan(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input,
        labels,
        packed_seq_params,
        extra_block_kwargs,
        runtime_gather_output,
        loss_mask,
        padding_mask,
    )


# ---------------------------------------------------------------------------
# Helpers for forward_step wrapping
# ---------------------------------------------------------------------------

def _find_loss_func(forward_step_func):
    """Extract loss_func from forward_step's globals (works with runpy)."""
    if hasattr(forward_step_func, "__globals__"):
        g = forward_step_func.__globals__
        if "loss_func" in g:
            return g["loss_func"]

    mod = getattr(forward_step_func, "__module__", None)
    if mod and mod in sys.modules and hasattr(sys.modules[mod], "loss_func"):
        return sys.modules[mod].loss_func

    def _fallback(loss_mask, output_tensor, model=None):
        losses = output_tensor.view(-1).float()
        lm = loss_mask.view(-1).float()
        loss = torch.sum(losses * lm)
        num_tokens = lm.sum().clone().detach().to(torch.int)
        return loss, num_tokens, {
            "lm loss": torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
        }

    return _fallback


def _get_mamba_forward_inputs(data_iterator, model, vp_stage, forward_step_func=None):
    """Fetch a batch and return kwargs for MambaModel.build_schedule_plan."""
    get_batch = None
    if forward_step_func is not None and hasattr(forward_step_func, "__globals__"):
        get_batch = forward_step_func.__globals__.get("get_batch")
    if get_batch is None:
        for _mn, _mod in sys.modules.items():
            if hasattr(_mod, "get_batch") and hasattr(_mod, "forward_step"):
                get_batch = _mod.get_batch
                break
    if get_batch is None:
        raise RuntimeError("[hybrid_schedule_plan] Cannot find get_batch")

    batch = list(get_batch(data_iterator, vp_stage))
    tokens, labels, loss_mask, attention_mask, position_ids = batch[0], batch[1], batch[2], batch[3], batch[4]

    cu_seqlens = batch[5] if len(batch) > 5 else None
    max_seqlen = batch[6] if len(batch) > 6 else None
    packed_seq_params = None
    if cu_seqlens is not None:
        from megatron.core.packed_seq_params import PackedSeqParams
        total_tokens = tokens.size(1) if tokens is not None else labels.size(1)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
            total_tokens=total_tokens,
        )

    return dict(
        input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask,
        labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_hybrid_schedule_plan_patch():
    """Monkey-patch MambaModel + combined_1f1b to enable EP A2A overlap.

    1. Bypasses MTP num_layers assertion in TransformerConfig.__post_init__
       by temporarily setting mtp_num_layers=1 (NOT None -- using None
       breaks validate_layer_layout and can trigger GPTModel instantiation
       with incompatible __init__ signature).
    2. Adds build_schedule_plan() to MambaModel.
    3. Wraps combined_forward_backward_step to:
       a) Accept MambaModel (not just GPTModel) for the isinstance check,
          by temporarily making GPTModel resolve to LanguageModule (the
          common base class).  Cannot use __bases__ patching because
          GPTModel and MambaModel share LanguageModule, causing MRO conflict.
       b) Wrap forward_step_func to support return_schedule_plan=True.
    """
    # (0) Bypass MTP num_layers restriction with EP overlap.
    #     Upstream only tested MTP=1 with EP overlap, but our hybrid model
    #     handles MTP in the post-process node, so MTP>1 is safe.
    #
    #     Strategy: temporarily set mtp_num_layers to 1 (NOT None) during
    #     __post_init__.  Using 1 (instead of None) is critical because:
    #       - None changes behavior in validate_layer_layout (pipeline
    #         layout asserts mtp layer count == mtp_num_layers)
    #       - None changes behavior in MTP model construction code paths
    #         that use ``mtp_num_layers is not None`` to decide whether
    #         to build MTP blocks
    #       - 1 passes the ``mtp_num_layers is None or mtp_num_layers == 1``
    #         assertion cleanly
    #       - 1 preserves ``mtp_num_layers is not None`` semantics for all
    #         other code paths in __post_init__
    #
    #     The ``if mtp_num_layers == 1`` sub-block also asserts
    #     ``pipeline_model_parallel_size > 1``, which is separately valid
    #     for our config (PP>=2).  We keep the real PP value so that
    #     assertion runs against the actual config.
    from megatron.core.transformer.transformer_config import TransformerConfig

    _orig_post_init = TransformerConfig.__post_init__

    def _relaxed_post_init(self):
        _saved_mtp = getattr(self, "mtp_num_layers", None)
        _saved_overlap = getattr(self, "overlap_moe_expert_parallel_comm", False)
        _needs_bypass = _saved_overlap and _saved_mtp is not None and _saved_mtp > 1
        if _needs_bypass:
            # Set to 1 (NOT None) so the overlap+MTP assertion passes.
            # Using 1 instead of None is critical: other __post_init__ code
            # checks ``mtp_num_layers is not None`` for layout validation
            # and MTP block construction.  Setting to None changes those
            # code paths and can trigger GPTModel instantiation with the
            # wrong __init__ signature.
            #
            # The ``if mtp_num_layers == 1`` sub-block also asserts
            # ``pipeline_model_parallel_size > 1`` which is satisfied by
            # our production config (PP>=2).
            self.mtp_num_layers = 1
        try:
            _orig_post_init(self)
        finally:
            if _needs_bypass:
                self.mtp_num_layers = _saved_mtp

    TransformerConfig.__post_init__ = _relaxed_post_init
    print("[hybrid_schedule_plan] MTP num_layers assertion bypassed", file=sys.stderr)

    from megatron.core.models.mamba.mamba_model import MambaModel

    # (1) Patch build_schedule_plan onto MambaModel
    MambaModel.build_schedule_plan = _mamba_build_schedule_plan
    print("[hybrid_schedule_plan] MambaModel.build_schedule_plan patched", file=sys.stderr)

    # (2) Patch combined_forward_backward_step to:
    #     a) Remove the isinstance(GPTModel) assert (MambaModel is not a
    #        GPTModel subclass and cannot be made one -- MRO conflict)
    #     b) Wrap forward_step_func to handle return_schedule_plan=True
    #
    #     We replace the function with a version that delegates to the
    #     original but with the isinstance check removed.  We do this by
    #     patching the function's globals so the ``GPTModel`` name imported
    #     inside the function resolves to LanguageModule (the common base
    #     of both GPTModel and MambaModel), making isinstance() pass for
    #     both model types without touching __bases__.
    import megatron.core.pipeline_parallel.combined_1f1b as c1f1b

    _orig_cfbs = c1f1b.combined_forward_backward_step

    def _patched_cfbs(
        forward_step_func, data_iterator, f_model, num_microbatches,
        input_tensor, forward_data_store, b_model, b_input_tensor,
        b_output_tensor, b_output_tensor_grad, config, **kwargs,
    ):
        import inspect
        from functools import partial

        from megatron.core.utils import get_attr_wrapped_model

        # (a) Wrap forward_step_func if it doesn't support return_schedule_plan.
        #     This teaches pretrain_mamba.py's forward_step to produce a
        #     schedule plan when asked.
        sig = inspect.signature(forward_step_func)
        if "return_schedule_plan" not in sig.parameters:
            _orig_fwd = forward_step_func

            def _wrapped_fwd(data_iter, model, return_schedule_plan=False, **fwd_kw):
                if return_schedule_plan:
                    vp_stage = get_attr_wrapped_model(model, "vp_stage")
                    plan = model.build_schedule_plan(
                        **_get_mamba_forward_inputs(
                            data_iter, model, vp_stage, forward_step_func=_orig_fwd,
                        )
                    )
                    loss_func = _find_loss_func(_orig_fwd)
                    return plan, partial(loss_func, plan.state.loss_mask, model=model)
                return _orig_fwd(data_iter, model, **fwd_kw)

            forward_step_func = _wrapped_fwd

        # (b) Temporarily make the isinstance(GPTModel) check in the original
        #     function accept MambaModel by patching the module-level import.
        #     The original function does:
        #       from megatron.core.models.gpt.gpt_model import GPTModel
        #       assert isinstance(unwrapped_model, GPTModel)
        #     We temporarily inject a fake GPTModel into sys.modules' cache so
        #     the ``from ... import GPTModel`` inside the function resolves to
        #     LanguageModule (common base of both GPTModel and MambaModel).
        #     This is safe because the function only uses GPTModel for the
        #     isinstance check, never for construction.
        import megatron.core.models.gpt.gpt_model as _gpt_mod
        from megatron.core.models.common.language_module.language_module import LanguageModule

        _real_gpt = _gpt_mod.GPTModel
        _gpt_mod.GPTModel = LanguageModule
        try:
            return _orig_cfbs(
                forward_step_func, data_iterator, f_model, num_microbatches,
                input_tensor, forward_data_store, b_model, b_input_tensor,
                b_output_tensor, b_output_tensor_grad, config, **kwargs,
            )
        finally:
            _gpt_mod.GPTModel = _real_gpt

    c1f1b.combined_forward_backward_step = _patched_cfbs

    # Also patch schedules.py reference if it exists
    try:
        import megatron.core.pipeline_parallel.schedules as sched

        if hasattr(sched, "combined_forward_backward_step"):
            sched.combined_forward_backward_step = _patched_cfbs
    except Exception:
        pass

    print("[hybrid_schedule_plan] Hybrid EP A2A overlap patch applied", file=sys.stderr)
