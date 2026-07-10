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

try:
    from megatron.core import parallel_state as _mcore_parallel_state
except ModuleNotFoundError:  # local macOS/dev environments without Megatron checkout
    _mcore_parallel_state = None  # type: ignore[assignment]

from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding
from cppmega.features.domain.embedding import CppMegaDomainEmbedding
from cppmega.features.structure.embedding import CppMegaStructureEmbedding

try:
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
except ModuleNotFoundError:  # local macOS/dev environments without Megatron checkout
    LanguageModelEmbedding = object  # type: ignore[assignment]

# Prefixes (relative to the CppMegaLanguageModelEmbedding module) of submodules
# whose ShardedTensor replica_id must be rewritten to reflect the current
# pipeline stage. The default Megatron walker stamps replica_id=(0, tp, dp) on
# every rank, which produces a duplicate "main replica" when MTP replicates the
# embedding on a non-first PP stage. See
# :meth:`CppMegaLanguageModelEmbedding.sharded_state_dict`.
_CPPMEGA_CUSTOM_PREFIXES: tuple[str, ...] = (
    "cppmega_ngram_hash.",
    "cppmega_domain.",
    "cppmega_structure.",
)


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

        self.cppmega_domain = None
        domain_enabled = os.environ.get(
            "CPPMEGA_DOMAIN_EMBEDDING_ENABLED",
            "1" if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") == "1" else "0",
        ) == "1"
        if domain_enabled:
            self.cppmega_domain = CppMegaDomainEmbedding(
                hidden_size=self.config.hidden_size,
                num_domains=int(os.environ.get("CPPMEGA_NUM_DOMAINS", "64")),
                num_roles=int(os.environ.get("CPPMEGA_NUM_DOMAIN_ROLES", "128")),
                num_confidences=int(os.environ.get("CPPMEGA_NUM_PARSE_CONFIDENCES", "8")),
                bottleneck_dim=int(os.environ.get("CPPMEGA_DOMAIN_BOTTLENECK_DIM", "32")),
            )

        # Diagnostic (RULE #1 visibility): report whether the cppmega feature
        # embeddings were actually constructed and how many params they add, so a
        # silently-absent feature cannot masquerade as "wired".
        _ng = (
            sum(p.numel() for p in self.cppmega_ngram_hash.parameters())
            if self.cppmega_ngram_hash is not None else 0
        )
        _st = (
            sum(p.numel() for p in self.cppmega_structure.parameters())
            if self.cppmega_structure is not None else 0
        )
        _dm = (
            sum(p.numel() for p in self.cppmega_domain.parameters())
            if self.cppmega_domain is not None else 0
        )
        print(
            f"[cppmega-embed] CppMegaLanguageModelEmbedding built: "
            f"ngram={'ON' if self.cppmega_ngram_hash is not None else 'OFF'}({_ng} params) "
            f"domain={'ON' if self.cppmega_domain is not None else 'OFF'}({_dm} params) "
            f"structure={'ON' if self.cppmega_structure is not None else 'OFF'}({_st} params) "
            f"env NGRAM={os.environ.get('CPPMEGA_NGRAM_HASH_ENABLED')} "
            f"STRUCT={os.environ.get('CPPMEGA_STRUCTURE_ENABLED')} "
            f"DOMAIN={os.environ.get('CPPMEGA_DOMAIN_EMBEDDING_ENABLED')}",
            flush=True,
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

    @staticmethod
    def _normalize_domain_inputs(
        input_ids: Tensor,
        structure_inputs: dict[str, Tensor] | None,
    ) -> dict[str, Tensor | None]:
        names = (
            "domain_ids",
            "role_ids",
            "confidence_ids",
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
                    f"domain input {name} shape {tuple(tensor.shape)} does not match input_ids {tuple(input_ids.shape)}"
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

        if self.cppmega_domain is not None:
            normalized_domain = self._normalize_domain_inputs(input_ids, structure_inputs)
            domain_embeddings = self.cppmega_domain(
                domain_ids=normalized_domain["domain_ids"],
                role_ids=normalized_domain["role_ids"],
                confidence_ids=normalized_domain["confidence_ids"],
                target_dtype=embeddings.dtype,
            )
            if domain_embeddings is not None:
                # RULE #1: a wrong-shape domain embedding must not be silently
                # dropped -- that would train with the domain signal disabled.
                if domain_embeddings.shape != embeddings.shape:
                    raise RuntimeError(
                        f"[cppmega] domain embedding shape {tuple(domain_embeddings.shape)} "
                        f"!= token embedding shape {tuple(embeddings.shape)}; "
                        f"refusing to silently drop the domain signal."
                    )
                embeddings = embeddings + domain_embeddings

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

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ):
        """Sharded state dict with pipeline-stage-aware replica ids for cppmega submodules.

        Upstream ``LanguageModelEmbedding`` inherits the default
        ``MegatronModule.sharded_state_dict`` walker. For any custom submodule
        (``cppmega_ngram_hash`` / ``cppmega_structure``) that walker produces
        ShardedTensors with ``replica_id=(0, tp_rank, dp_rank)`` on every PP
        stage where the embedding exists. When Multi-Token Prediction (MTP) is
        enabled, ``CppMegaMambaModel`` / ``CppMegaGPTModel`` build the
        embedding on *both* the first PP stage (``pre_process``) **and** the
        MTP stage (``mtp_process``), so Megatron's
        ``validate_sharding_integrity`` sees two "main replicas"
        (``shard_access_cnt=2``) for the same key and aborts the checkpoint
        save with ``Invalid access pattern for ShardedTensor(...)``.

        For every cppmega-owned submodule under ``self``, rewrite the replica
        id to match the upstream convention used for the tied word embedding
        (see ``tie_word_embeddings_state_dict`` in
        ``megatron.core.transformer.multi_token_prediction``):

        * PP-first / ``pre_process`` stage: ``(0, tp_rank, dp_rank)`` (main)
        * MTP copy (any later PP stage that still holds the embedding):
          ``(1, tp_rank, dp_rank)`` (secondary copy)

        On PP stages that do not hold the embedding, the parent
        ``CppMegaMambaModel.__init__`` simply does not instantiate the
        embedding, so this method is never reached.
        """

        sharded_state_dict = super().sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

        if _mcore_parallel_state is None:
            # Local dev environment without Megatron; nothing to rewrite.
            return sharded_state_dict

        # Determine whether this embedding lives on the pipeline-first stage.
        # If Megatron's parallel state is not initialized (unit-test mode),
        # treat the module as pipeline-first so the rewrite is a no-op.
        try:
            is_first_pipeline_stage = _mcore_parallel_state.is_pipeline_first_stage(
                ignore_virtual=True
            )
        except (AssertionError, RuntimeError, AttributeError):
            is_first_pipeline_stage = True

        leading_replica_axis = 0 if is_first_pipeline_stage else 1

        custom_key_prefixes = tuple(
            f"{prefix}{suffix}" for suffix in _CPPMEGA_CUSTOM_PREFIXES
        )
        if not custom_key_prefixes:
            return sharded_state_dict

        for key, value in sharded_state_dict.items():
            if not key.startswith(custom_key_prefixes):
                continue
            existing = getattr(value, "replica_id", None)
            if existing is None:
                continue
            if isinstance(existing, int):
                # Should not happen for tensor shardings created by the default
                # walker, but handle it conservatively: a scalar replica id is
                # already "main" (0) or "copy" (non-zero). Preserve copy-ness.
                new_replica = leading_replica_axis if existing == 0 else existing
            else:
                tail = tuple(existing[1:]) if len(existing) >= 1 else ()
                new_replica = (leading_replica_axis,) + tail
            try:
                value.replica_id = new_replica
            except AttributeError:
                # ShardedObject / ShardedTensor dataclasses are frozen in some
                # Megatron versions; fall back to object.__setattr__.
                object.__setattr__(value, "replica_id", new_replica)

        return sharded_state_dict
