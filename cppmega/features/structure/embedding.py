"""Minimal structure-aware token embedding for cppmega.

This ports only the first narrow runtime seam from nanochat: additive token-level
structure embeddings that can be injected beside Megatron token embeddings.
Relation bias, platform embeddings, and tree FFN stay out of scope here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CppMegaStructureEmbedding(nn.Module):
    ALL_COMPONENTS = (
        "structure",
        "dep_level",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
    )
    CORE_COMPONENTS = ("structure", "dep_level")

    def __init__(
        self,
        *,
        hidden_size: int,
        num_categories: int = 9,
        max_dep_level: int = 16,
        max_ast_depth: int = 64,
        max_sibling_index: int = 64,
        num_node_types: int = 256,
        active_components: str = "core",
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.active_component_names = self._parse_components(active_components)

        vocab_sizes = {
            "structure": num_categories,
            "dep_level": max_dep_level,
            "ast_depth": max_ast_depth,
            "sibling_index": max_sibling_index,
            "ast_node_type": num_node_types,
        }
        offsets: list[int] = []
        total_vocab = 0
        for name in self.active_component_names:
            offsets.append(total_vocab)
            total_vocab += vocab_sizes[name]

        self.register_buffer("_comp_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_comp_clamp_max",
            torch.tensor([vocab_sizes[name] - 1 for name in self.active_component_names], dtype=torch.long),
            persistent=False,
        )

        num_active = len(self.active_component_names)
        self.stacked_emb = nn.Embedding(total_vocab, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.stacked_emb.weight.is_embedding_or_output_parameter = True
        self.up_proj.weight.is_embedding_or_output_parameter = True
        self.component_scales = nn.Parameter(torch.full((num_active,), 1.0 / max(num_active, 1)))
        nn.init.zeros_(self.stacked_emb.weight)
        nn.init.zeros_(self.up_proj.weight)

    @classmethod
    def _parse_components(cls, spec: str) -> tuple[str, ...]:
        if spec == "all":
            return cls.ALL_COMPONENTS
        if spec == "core":
            return cls.CORE_COMPONENTS
        requested = {item.strip() for item in spec.split(",") if item.strip()}
        invalid = sorted(requested - set(cls.ALL_COMPONENTS))
        if invalid:
            raise ValueError(f"unknown structure components: {invalid!r}")
        return tuple(name for name in cls.ALL_COMPONENTS if name in requested)

    def forward(
        self,
        *,
        structure_ids: torch.Tensor | None,
        dep_levels: torch.Tensor | None,
        ast_depth_ids: torch.Tensor | None = None,
        sibling_index_ids: torch.Tensor | None = None,
        node_type_ids: torch.Tensor | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        inputs = {
            "structure": structure_ids,
            "dep_level": dep_levels,
            "ast_depth": ast_depth_ids,
            "sibling_index": sibling_index_ids,
            "ast_node_type": node_type_ids,
        }
        ref = next((inputs[name] for name in self.active_component_names if inputs[name] is not None), None)
        if ref is None:
            return torch.tensor(0.0, dtype=target_dtype or torch.float32)

        batch, seq = ref.shape[:2]
        ids_list: list[torch.Tensor] = []
        present: list[bool] = []
        for index, name in enumerate(self.active_component_names):
            tensor = inputs[name]
            if tensor is None:
                ids_list.append(torch.zeros(batch, seq, dtype=torch.long, device=ref.device))
                present.append(False)
                continue
            clamped = tensor.clamp(0, int(self._comp_clamp_max[index].item()))
            ids_list.append(clamped + int(self._comp_offsets[index].item()))
            present.append(True)

        stacked_ids = torch.stack(ids_list, dim=-1).reshape(batch * seq, len(self.active_component_names))
        emb = F.embedding(stacked_ids, self.stacked_emb.weight).reshape(
            batch, seq, len(self.active_component_names), self.bottleneck_dim
        )
        scales = self.component_scales.to(device=ref.device)
        if not all(present):
            scales = scales * torch.tensor(present, dtype=scales.dtype, device=ref.device)
        weighted = (emb * scales.view(1, 1, -1, 1)).sum(dim=2)
        if target_dtype is not None and weighted.dtype != target_dtype:
            weighted = weighted.to(target_dtype)
        return F.linear(weighted, self.up_proj.weight.to(weighted.dtype))
