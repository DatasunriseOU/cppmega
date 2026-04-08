"""Cppmega-owned GPT model wrapper using derived embedding modules."""

from __future__ import annotations

from megatron.core.models.gpt.gpt_model import GPTModel

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding


class CppMegaGPTModel(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pre_process or self.mtp_process:
            self.embedding = CppMegaLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=True,
                tp_group=self.pg_collection.tp,
            )
