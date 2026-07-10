"""Domain/role/confidence additive embeddings for cppmega world-code models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CppMegaDomainEmbedding(nn.Module):
    """Small additive embedding over domain-routing token sidecars.

    The module intentionally ignores entity/scope ids by default: those ids are
    per-document and can be very high-cardinality. Domain, role, and confidence
    are low-cardinality, stable enums and are the right signal for separating
    C++/CMake/Make/shell/diagnostic semantics at the input layer.
    """

    COMPONENTS = ("domain", "role", "confidence")

    def __init__(
        self,
        *,
        hidden_size: int,
        num_domains: int = 64,
        num_roles: int = 128,
        num_confidences: int = 8,
        bottleneck_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        vocab_sizes = {
            "domain": int(num_domains),
            "role": int(num_roles),
            "confidence": int(num_confidences),
        }
        offsets: list[int] = []
        total_vocab = 0
        for name in self.COMPONENTS:
            offsets.append(total_vocab)
            total_vocab += vocab_sizes[name]

        self.register_buffer("_comp_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_comp_clamp_max",
            torch.tensor([vocab_sizes[name] - 1 for name in self.COMPONENTS], dtype=torch.long),
            persistent=False,
        )
        self.stacked_emb = nn.Embedding(total_vocab, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        self.component_scales = nn.Parameter(torch.full((len(self.COMPONENTS),), 1.0 / len(self.COMPONENTS)))

        self.stacked_emb.weight.is_embedding_or_output_parameter = True
        self.up_proj.weight.is_embedding_or_output_parameter = True
        nn.init.zeros_(self.stacked_emb.weight)
        nn.init.zeros_(self.up_proj.weight)

    def forward(
        self,
        *,
        domain_ids: torch.Tensor | None,
        role_ids: torch.Tensor | None,
        confidence_ids: torch.Tensor | None,
        target_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        inputs = {
            "domain": domain_ids,
            "role": role_ids,
            "confidence": confidence_ids,
        }
        ref = next((value for value in inputs.values() if value is not None), None)
        if ref is None:
            # No domain sidecars for this step -> no additive signal. Return None
            # (not a CPU scalar) so the caller skips the add cleanly on any device.
            return None

        batch, seq = ref.shape[:2]
        ids_list: list[torch.Tensor] = []
        present: list[bool] = []
        for index, name in enumerate(self.COMPONENTS):
            tensor = inputs[name]
            if tensor is None:
                ids_list.append(torch.zeros(batch, seq, dtype=torch.long, device=ref.device))
                present.append(False)
                continue
            tensor = tensor.to(dtype=torch.long)
            comp_max = int(self._comp_clamp_max[index].item())
            # RULE #1: out-of-range ids mean a corrupt/misconfigured sidecar.
            # Raise instead of clamping into the boundary bucket (silent wrong data).
            if bool((tensor < 0).any().item()) or bool((tensor > comp_max).any().item()):
                bad = tensor[(tensor < 0) | (tensor > comp_max)].flatten()[:8].tolist()
                raise ValueError(
                    f"[cppmega-domain] {name}_ids out of range [0,{comp_max}]: "
                    f"offending values {bad} -- corrupt/misconfigured sidecar; "
                    f"refusing to clamp."
                )
            ids_list.append(tensor + int(self._comp_offsets[index].item()))
            present.append(True)

        stacked_ids = torch.stack(ids_list, dim=-1).reshape(batch * seq, len(self.COMPONENTS))
        emb = F.embedding(stacked_ids, self.stacked_emb.weight).reshape(
            batch, seq, len(self.COMPONENTS), self.bottleneck_dim
        )
        scales = self.component_scales.to(device=ref.device)
        if not all(present):
            scales = scales * torch.tensor(present, dtype=scales.dtype, device=ref.device)
        weighted = (emb * scales.view(1, 1, -1, 1)).sum(dim=2)
        if target_dtype is not None and weighted.dtype != target_dtype:
            weighted = weighted.to(target_dtype)
        return F.linear(weighted, self.up_proj.weight.to(weighted.dtype))


__all__ = ["CppMegaDomainEmbedding"]
