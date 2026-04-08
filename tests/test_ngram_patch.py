from pathlib import Path

from cppmega.remote.ngram_patch import patch_language_model_embedding_for_ngram_hash


def test_ngram_patch_injects_cppmega_module_and_forward_hook(tmp_path: Path):
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

    patch_language_model_embedding_for_ngram_hash(target)
    text = target.read_text()

    assert "from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding" in text
    assert "from cppmega.remote.ngram_patch import read_ngram_hash_env" in text
    assert "self.cppmega_ngram_hash = None" in text
    assert "cppmega_ngram_hash_cfg = read_ngram_hash_env()" in text
    assert "if self.cppmega_ngram_hash is not None:" in text


def test_ngram_patch_is_idempotent(tmp_path: Path):
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

    patch_language_model_embedding_for_ngram_hash(target)
    first = target.read_text()
    patch_language_model_embedding_for_ngram_hash(target)
    second = target.read_text()

    assert first == second
