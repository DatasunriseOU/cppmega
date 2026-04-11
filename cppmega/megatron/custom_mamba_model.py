"""Cppmega-owned Mamba model wrapper using derived embedding modules."""

from __future__ import annotations

try:
    from megatron.core.models.mamba.mamba_model import MambaModel
except ModuleNotFoundError:  # local macOS/dev environments without Megatron checkout
    MambaModel = object  # type: ignore[assignment]

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding
from cppmega.megatron.fastmtp_layer import (
    FastMTPLayer,
    fastmtp_enabled,
    get_fastmtp_decay,
    get_fastmtp_depth,
    get_fastmtp_lambda,
)


class CppMegaMambaModel(MambaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cppmega_structure_inputs = None
        self._fastmtp = None
        self._fastmtp_lambda = 0.0
        if self.pre_process or self.mtp_process:
            self.embedding = CppMegaLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=True,
                tp_group=self.pg_collection.tp,
            )
        # FastMTP: when enabled on the post_process stage, create our own MTP
        # layer. This works WITHOUT Megatron's MTP pipeline stages -- the
        # hybrid_layer_pattern should have mtp_depths=0 and --mtp-num-layers
        # should NOT be passed. FastMTP creates its own embedding (weight-tied
        # with the main model) and shared transformer block.
        if fastmtp_enabled() and self.post_process:
            depth = get_fastmtp_depth()
            decay = get_fastmtp_decay()
            self._fastmtp_lambda = get_fastmtp_lambda()
            self._fastmtp = FastMTPLayer(
                config=self.config,
                vocab_size=self.vocab_size,
                depth=depth,
                decay=decay,
                recompute=True,
                tp_group=self.pg_collection.tp if self.pg_collection else None,
            )
            # Tie FastMTP word embeddings with the main model's embedding.
            # On the post_process stage without MTP, self.embedding might not
            # exist. In that case, the FastMTP's own word_embeddings will be
            # trained independently (same init, just not tied). For best
            # results with PP>1, ensure the main model's embedding weight is
            # synchronized at startup.
            if hasattr(self, "embedding") and self.embedding is not None:
                self._fastmtp.tie_word_embeddings(self.embedding)

    def set_cppmega_structure_inputs(self, structure_inputs):
        self._cppmega_structure_inputs = structure_inputs

    def _cppmega_take_structure_inputs(self):
        structure_inputs = self._cppmega_structure_inputs
        self._cppmega_structure_inputs = None
        return structure_inputs

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        packed_seq_params=None,
        padding_mask=None,
        is_spec_decode=None,
    ):
        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids,
                structure_inputs=self._cppmega_take_structure_inputs(),
            )

        # FastMTP: use a hook to capture decoder hidden states, then let
        # the base model forward run normally. After the base forward
        # returns the loss, compute FastMTP loss and add it.
        if self._fastmtp is not None and labels is not None and self.training and self.post_process:
            self._fastmtp_hidden_cache = None

            def _capture_hook(module, input, output):
                # MambaStack (decoder) output is hidden_states tensor
                self._fastmtp_hidden_cache = output

            hook = self.decoder.register_forward_hook(_capture_hook)
            try:
                main_loss = super().forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    decoder_input=decoder_input,
                    labels=labels,
                    inference_context=inference_context,
                    runtime_gather_output=runtime_gather_output,
                    inference_params=inference_params,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    is_spec_decode=is_spec_decode,
                )
            finally:
                hook.remove()

            # Compute FastMTP loss from captured hidden states
            if self._fastmtp_hidden_cache is not None:
                fastmtp_loss = self._fastmtp(
                    hidden_states=self._fastmtp_hidden_cache,
                    labels=labels,
                    output_layer=self.output_layer,
                    loss_mask=loss_mask,
                )
                self._fastmtp_hidden_cache = None
                return main_loss + self._fastmtp_lambda * fastmtp_loss

            return main_loss

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            is_spec_decode=is_spec_decode,
        )
