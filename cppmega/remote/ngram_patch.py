"""Helpers for applying the narrow cppmega ngram-hash patch to Megatron.

The intent is to keep the custom port surface minimal: patch only the shared
LanguageModelEmbedding path so GPT/Mamba models can pick up additive token
enrichment without copying Megatron model wrappers or training loops.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_ngram_hash_env() -> dict[str, object]:
    return {
        "enabled": os.environ.get("CPPMEGA_NGRAM_HASH_ENABLED", "0") == "1",
        "orders": tuple(
            int(x)
            for x in os.environ.get("CPPMEGA_NGRAM_HASH_ORDERS", "2,3").split(",")
            if x.strip()
        ),
        "heads": int(os.environ.get("CPPMEGA_NGRAM_HASH_HEADS", "8")),
        "table_size": int(os.environ.get("CPPMEGA_NGRAM_HASH_TABLE_SIZE", "500000")),
        "embed_dim": int(os.environ.get("CPPMEGA_NGRAM_HASH_EMBED_DIM", "16")),
        "offload": os.environ.get("CPPMEGA_NGRAM_HASH_OFFLOAD", "0") == "1",
    }


def patch_language_model_embedding_for_ngram_hash(
    embedding_file: str | Path,
) -> None:
    path = Path(embedding_file)
    text = path.read_text()

    import_line = "from megatron.core.utils import get_tensor_model_parallel_group_if_none, nvtx_decorator\n"
    import_replacement = (
        import_line
        + "from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding\n"
        + "from cppmega.remote.ngram_patch import read_ngram_hash_env\n"
    )
    if "from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding\n" not in text:
        if import_line not in text:
            raise ValueError("failed to find LanguageModelEmbedding import insertion point")
        text = text.replace(import_line, import_replacement, 1)

    init_needle = "        # Embeddings dropout\n        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)\n"
    init_replacement = (
        "        # Embeddings dropout\n"
        "        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)\n"
        "\n"
        "        self.cppmega_ngram_hash = None\n"
        "        cppmega_ngram_hash_cfg = read_ngram_hash_env()\n"
        "        if cppmega_ngram_hash_cfg['enabled']:\n"
        "            self.cppmega_ngram_hash = CppMegaNgramHashEmbedding(\n"
        "                hidden_size=self.config.hidden_size,\n"
        "                orders=tuple(cppmega_ngram_hash_cfg['orders']),\n"
        "                num_heads=int(cppmega_ngram_hash_cfg['heads']),\n"
        "                table_size=int(cppmega_ngram_hash_cfg['table_size']),\n"
        "                embed_dim=int(cppmega_ngram_hash_cfg['embed_dim']),\n"
        "                offload=bool(cppmega_ngram_hash_cfg['offload']),\n"
        "            )\n"
    )
    if "self.cppmega_ngram_hash = None" not in text:
        if init_needle not in text:
            raise ValueError("failed to find LanguageModelEmbedding init insertion point")
        text = text.replace(init_needle, init_replacement, 1)

    forward_needle = "        if self.add_position_embedding:\n            position_embeddings = self.position_embeddings(position_ids)\n            embeddings = word_embeddings + position_embeddings\n        else:\n            embeddings = word_embeddings\n\n"
    forward_replacement = (
        "        if self.add_position_embedding:\n"
        "            position_embeddings = self.position_embeddings(position_ids)\n"
        "            embeddings = word_embeddings + position_embeddings\n"
        "        else:\n"
        "            embeddings = word_embeddings\n"
        "\n"
        "        if self.cppmega_ngram_hash is not None:\n"
        "            ngram_input = input_ids if not self.reduce_scatter_embeddings else input_ids.contiguous()\n"
        "            ngram_embeddings = self.cppmega_ngram_hash(ngram_input)\n"
        "            if self.reduce_scatter_embeddings:\n"
        "                embeddings = embeddings + ngram_embeddings\n"
        "            else:\n"
        "                embeddings = embeddings + ngram_embeddings.transpose(0, 1).contiguous()\n"
        "\n"
    )
    if "if self.cppmega_ngram_hash is not None:" not in text:
        if forward_needle not in text:
            raise ValueError("failed to find LanguageModelEmbedding forward insertion point")
        text = text.replace(forward_needle, forward_replacement, 1)

    path.write_text(text)
