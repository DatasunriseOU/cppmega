"""Helpers for applying the narrow cppmega structure embedding patch to Megatron."""

from __future__ import annotations

import os
from pathlib import Path


def read_structure_env() -> dict[str, object]:
    return {
        "enabled": os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") == "1",
        "components": os.environ.get("CPPMEGA_STRUCTURE_COMPONENTS", "core"),
        "max_ast_depth": int(os.environ.get("CPPMEGA_MAX_AST_DEPTH", "20")),
        "max_sibling_index": int(os.environ.get("CPPMEGA_MAX_SIBLING_INDEX", "10")),
        "num_node_types": int(os.environ.get("CPPMEGA_NUM_NODE_TYPES", "64")),
        "bottleneck_dim": int(os.environ.get("CPPMEGA_STRUCTURE_BOTTLENECK_DIM", "64")),
    }


def patch_language_model_embedding_for_structure(embedding_file: str | Path) -> None:
    path = Path(embedding_file)
    text = path.read_text()

    import_line = "from megatron.core.utils import get_tensor_model_parallel_group_if_none, nvtx_decorator\n"
    import_replacement = (
        import_line
        + "from cppmega.features.structure.embedding import CppMegaStructureEmbedding\n"
        + "from cppmega.remote.structure_patch import read_structure_env\n"
    )
    if "from cppmega.features.structure.embedding import CppMegaStructureEmbedding\n" not in text:
        if import_line not in text:
            raise ValueError("failed to find LanguageModelEmbedding import insertion point")
        text = text.replace(import_line, import_replacement, 1)

    init_needle = "        # Embeddings dropout\n        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)\n"
    init_replacement = (
        "        # Embeddings dropout\n"
        "        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)\n"
        "\n"
        "        self.cppmega_structure = None\n"
        "        cppmega_structure_cfg = read_structure_env()\n"
        "        if cppmega_structure_cfg['enabled']:\n"
        "            self.cppmega_structure = CppMegaStructureEmbedding(\n"
        "                hidden_size=self.config.hidden_size,\n"
        "                active_components=str(cppmega_structure_cfg['components']),\n"
        "                max_ast_depth=int(cppmega_structure_cfg['max_ast_depth']),\n"
        "                max_sibling_index=int(cppmega_structure_cfg['max_sibling_index']),\n"
        "                num_node_types=int(cppmega_structure_cfg['num_node_types']),\n"
        "                bottleneck_dim=int(cppmega_structure_cfg['bottleneck_dim']),\n"
        "            )\n"
    )
    if "self.cppmega_structure = None" not in text:
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
        "        if self.cppmega_structure is not None:\n"
        "            structure_shape = input_ids.shape\n"
        "            zeros = torch.zeros(structure_shape, dtype=torch.long, device=input_ids.device)\n"
        "            structure_embeddings = self.cppmega_structure(\n"
        "                structure_ids=zeros,\n"
        "                dep_levels=zeros,\n"
        "                target_dtype=embeddings.dtype,\n"
        "            )\n"
        "            if self.reduce_scatter_embeddings:\n"
        "                embeddings = embeddings + structure_embeddings\n"
        "            else:\n"
        "                embeddings = embeddings + structure_embeddings.transpose(0, 1).contiguous()\n"
        "\n"
    )
    if "if self.cppmega_structure is not None:" not in text:
        if forward_needle not in text:
            raise ValueError("failed to find LanguageModelEmbedding forward insertion point")
        text = text.replace(forward_needle, forward_replacement, 1)

    path.write_text(text)
