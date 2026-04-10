"""Cppmega-owned Mamba model wrapper using derived embedding modules."""

from __future__ import annotations

try:
    from megatron.core.models.mamba.mamba_model import MambaModel
except ModuleNotFoundError:  # local macOS/dev environments without Megatron checkout
    MambaModel = object  # type: ignore[assignment]

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding


class CppMegaMambaModel(MambaModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cppmega_structure_inputs = None
        if self.pre_process or self.mtp_process:
            self.embedding = CppMegaLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=True,
                tp_group=self.pg_collection.tp,
            )

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
