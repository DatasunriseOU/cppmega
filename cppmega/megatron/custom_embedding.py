"""Cppmega-owned Megatron embedding extensions.

This is the non-patch path for custom input enrichments. It subclasses the
upstream LanguageModelEmbedding and adds optional cppmega features while keeping
the rest of Megatron behavior unchanged.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

try:
    from megatron.core import tensor_parallel
except ModuleNotFoundError:
    tensor_parallel = None  # type: ignore[assignment]

from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding
from cppmega.features.structure.embedding import CppMegaStructureEmbedding

try:
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
except ModuleNotFoundError:  # local macOS/dev environments without Megatron checkout
    LanguageModelEmbedding = object  # type: ignore[assignment]


class CppMegaLanguageModelEmbedding(LanguageModelEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cppmega_ngram_hash = None
        if os.environ.get("CPPMEGA_NGRAM_HASH_ENABLED", "0") == "1":
            orders = tuple(
                int(x)
                for x in os.environ.get("CPPMEGA_NGRAM_HASH_ORDERS", "2,3").split(",")
                if x.strip()
            )
            self.cppmega_ngram_hash = CppMegaNgramHashEmbedding(
                hidden_size=self.config.hidden_size,
                orders=orders,
                num_heads=int(os.environ.get("CPPMEGA_NGRAM_HASH_HEADS", "8")),
                table_size=int(os.environ.get("CPPMEGA_NGRAM_HASH_TABLE_SIZE", "500000")),
                embed_dim=int(os.environ.get("CPPMEGA_NGRAM_HASH_EMBED_DIM", "16")),
                offload=os.environ.get("CPPMEGA_NGRAM_HASH_OFFLOAD", "0") == "1",
            )

        self.cppmega_structure = None
        if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") == "1":
            self.cppmega_structure = CppMegaStructureEmbedding(
                hidden_size=self.config.hidden_size,
                active_components=os.environ.get("CPPMEGA_STRUCTURE_COMPONENTS", "core"),
                max_ast_depth=int(os.environ.get("CPPMEGA_MAX_AST_DEPTH", "20")),
                max_sibling_index=int(os.environ.get("CPPMEGA_MAX_SIBLING_INDEX", "10")),
                num_node_types=int(os.environ.get("CPPMEGA_NUM_NODE_TYPES", "64")),
                bottleneck_dim=int(os.environ.get("CPPMEGA_STRUCTURE_BOTTLENECK_DIM", "64")),
            )

    @staticmethod
    def _normalize_structure_inputs(
        input_ids: Tensor,
        structure_inputs: dict[str, Tensor] | None,
    ) -> dict[str, Tensor | None]:
        names = (
            "structure_ids",
            "dep_levels",
            "ast_depth_ids",
            "sibling_index_ids",
            "node_type_ids",
        )
        normalized: dict[str, Tensor | None] = {name: None for name in names}
        if structure_inputs is None:
            return normalized
        for name in names:
            tensor = structure_inputs.get(name)
            if tensor is None:
                continue
            if tensor.shape != input_ids.shape:
                raise ValueError(
                    f"structure input {name} shape {tuple(tensor.shape)} does not match input_ids {tuple(input_ids.shape)}"
                )
            normalized[name] = tensor.to(device=input_ids.device, dtype=torch.long)
        return normalized

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: int = None,
        structure_inputs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        word_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if self.cppmega_ngram_hash is not None:
            ngram_embeddings = self.cppmega_ngram_hash(input_ids)
            embeddings = embeddings + ngram_embeddings

        if self.cppmega_structure is not None:
            normalized_structure = self._normalize_structure_inputs(input_ids, structure_inputs)
            structure_embeddings = self.cppmega_structure(
                structure_ids=normalized_structure["structure_ids"],
                dep_levels=normalized_structure["dep_levels"],
                ast_depth_ids=normalized_structure["ast_depth_ids"],
                sibling_index_ids=normalized_structure["sibling_index_ids"],
                node_type_ids=normalized_structure["node_type_ids"],
                target_dtype=embeddings.dtype,
            )
            if isinstance(structure_embeddings, torch.Tensor) and structure_embeddings.ndim == embeddings.ndim:
                embeddings = embeddings + structure_embeddings

        if not self.reduce_scatter_embeddings:
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        if self.config.use_mup and self.config.mup_embedding_mult != 1.0:
            embeddings = embeddings * self.config.mup_embedding_mult

        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings and self.scatter_to_sequence_parallel:
                assert tensor_parallel is not None
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    embeddings, group=self.tp_group
                )
            if self.config.clone_scatter_output_in_embedding and self.scatter_to_sequence_parallel:
                embeddings = embeddings.clone()
            assert tensor_parallel is not None
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
