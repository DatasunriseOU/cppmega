"""Cppmega-owned GPT model wrapper using derived embedding modules."""

from __future__ import annotations

from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding
from cppmega.megatron.mxfp8_storage_islands import apply_mxfp8_storage_islands_from_env


class CppMegaGPTModel(GPTModel):
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
        apply_mxfp8_storage_islands_from_env(self)

    def set_cppmega_structure_inputs(self, structure_inputs):
        self._cppmega_structure_inputs = structure_inputs

    def _cppmega_take_structure_inputs(self):
        structure_inputs = self._cppmega_structure_inputs
        self._cppmega_structure_inputs = None
        return structure_inputs

    def _preprocess(
        self,
        input_ids,
        position_ids,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
        padding_mask=None,
    ):
        in_inference_mode = inference_context is not None and not self.training

        if decoder_input is None and self.pre_process:
            if padding_mask is not None:
                assert padding_mask.shape == input_ids.shape, (
                    f"padding_mask shape {padding_mask.shape} does not match "
                    f"input_ids shape {input_ids.shape}"
                )
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids,
                structure_inputs=self._cppmega_take_structure_inputs(),
            )
            if padding_mask is not None and self.config.sequence_parallel:
                padding_mask = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        padding_mask.transpose(0, 1).contiguous()
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        return super()._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
