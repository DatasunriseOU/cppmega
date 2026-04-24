"""Minimal n-gram hash embedding port for cppmega custom input enrichment.

This keeps the custom seam narrow: a single additive embedding module that can
be attached near Megatron token embeddings without copying nanochat's training
loop or broader GPT stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRIMES = [
    499_801,
    499_819,
    499_853,
    499_879,
    499_883,
    499_897,
    499_903,
    499_927,
    499_943,
    499_957,
    499_969,
    499_973,
    499_979,
    500_009,
    500_029,
    500_041,
]


def _pick_primes(count: int, target_size: int) -> list[int]:
    candidates = [p for p in _PRIMES if abs(p - target_size) / target_size < 0.5]
    if len(candidates) >= count:
        return candidates[:count]
    return [target_size + i for i in range(count)]


class CppMegaNgramHashEmbedding(nn.Module):
    """Hash-based token n-gram enrichment.

    The module intentionally mirrors only the narrow nanochat runtime seam:
    hash token n-grams into a unified table, concatenate table embeddings, and
    project back into the model hidden dimension for additive input enrichment.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        orders: tuple[int, ...] = (2, 3),
        num_heads: int = 8,
        table_size: int = 500_000,
        embed_dim: int = 16,
        dropout: float = 0.0,
        offload: bool = False,
    ) -> None:
        super().__init__()
        if not orders:
            raise ValueError("orders must contain at least one n-gram order")
        if any(order <= 0 for order in orders):
            raise ValueError("all n-gram orders must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.hidden_size = hidden_size
        self.orders = tuple(orders)
        self.num_heads = num_heads
        self.num_tables = len(self.orders) * num_heads
        self.embed_dim = embed_dim
        self.offload = offload

        self.table_sizes = _pick_primes(self.num_tables, table_size)
        offsets: list[int] = []
        total_entries = 0
        for size in self.table_sizes:
            offsets.append(total_entries)
            total_entries += size

        self.register_buffer("table_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("table_sizes_t", torch.tensor(self.table_sizes, dtype=torch.long))
        self.unified_table = nn.Embedding(total_entries, embed_dim)
        self.unified_table.weight.is_embedding_or_output_parameter = True

        max_order = max(self.orders)
        self.max_order = max_order
        mults = torch.randint(1, 2**31, (self.num_tables, max_order), dtype=torch.long)
        self.register_buffer("hash_mults", mults | 1)
        self.register_buffer(
            "hash_bias",
            torch.randint(0, 2**31, (self.num_tables,), dtype=torch.long),
        )

        order_list: list[int] = []
        for order in self.orders:
            for _ in range(self.num_heads):
                order_list.append(order)
        self.register_buffer("order_for_table", torch.tensor(order_list, dtype=torch.long))

        order_mask = torch.zeros(max_order, self.num_tables, dtype=torch.long)
        for table_index, order in enumerate(order_list):
            for position in range(order):
                order_mask[position, table_index] = 1
        self.register_buffer("order_mask", order_mask)

        self.out_proj = nn.Linear(self.num_tables * embed_dim, hidden_size, bias=False)
        self.out_proj.weight.is_embedding_or_output_parameter = True
        torch.nn.init.zeros_(self.out_proj.weight)
        self.dropout = nn.Dropout(dropout)

    def _hash_all(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = token_ids.shape
        shifted = torch.zeros(
            self.max_order,
            batch,
            seq,
            dtype=torch.long,
            device=token_ids.device,
        )
        shifted[0] = token_ids
        for position in range(1, self.max_order):
            shifted[position, :, position:] = token_ids[:, :-position]

        mults = self.hash_mults.t().unsqueeze(-1).unsqueeze(-1)
        mask = self.order_mask.unsqueeze(-1).unsqueeze(-1)
        product = (mults * shifted.unsqueeze(1)) * mask

        hashed = product[0]
        for position in range(1, self.max_order):
            hashed = hashed ^ product[position]

        hashed = hashed ^ self.hash_bias.unsqueeze(-1).unsqueeze(-1)
        hashed = hashed % self.table_sizes_t.unsqueeze(-1).unsqueeze(-1)
        hashed = hashed + self.table_offsets.unsqueeze(-1).unsqueeze(-1)
        return hashed.permute(1, 0, 2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        target_device = token_ids.device
        if self.offload:
            token_ids = token_ids.cpu()

        unified_indices = self._hash_all(token_ids)
        batch, num_tables, seq = unified_indices.shape
        flat_embeddings = F.embedding(unified_indices.reshape(-1), self.unified_table.weight)
        embeddings = flat_embeddings.view(batch, num_tables, seq, self.embed_dim)
        embeddings = embeddings.permute(0, 2, 1, 3).reshape(
            batch,
            seq,
            num_tables * self.embed_dim,
        )
        if self.offload:
            embeddings = embeddings.to(target_device, dtype=self.out_proj.weight.dtype)
        return self.dropout(self.out_proj(embeddings))
