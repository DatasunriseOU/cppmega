from pathlib import Path

from cppmega.remote.structure_patch import patch_language_model_embedding_for_structure


def test_structure_patch_injects_structure_module(tmp_path: Path):
    target = tmp_path / "language_model_embedding.py"
    target.write_text(
        "from megatron.core.utils import get_tensor_model_parallel_group_if_none, nvtx_decorator\n"
        "\n"
        "class LanguageModelEmbedding:\n"
        "    def __init__(self):\n"
        "        # Embeddings dropout\n"
        "        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)\n"
        "\n"
        "    def forward(self, input_ids, position_ids, tokentype_ids=None):\n"
        "        word_embeddings = self.word_embeddings(input_ids)\n"
        "        if self.add_position_embedding:\n"
        "            position_embeddings = self.position_embeddings(position_ids)\n"
        "            embeddings = word_embeddings + position_embeddings\n"
        "        else:\n"
        "            embeddings = word_embeddings\n"
        "\n"
    )

    patch_language_model_embedding_for_structure(target)
    text = target.read_text()

    assert "CppMegaStructureEmbedding" in text
    assert "read_structure_env" in text
    assert "self.cppmega_structure = None" in text
    assert "if self.cppmega_structure is not None:" in text
