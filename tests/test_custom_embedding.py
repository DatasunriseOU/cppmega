import os

import torch
import pytest

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding


if CppMegaLanguageModelEmbedding.__mro__[1] is object:
    pytest.skip("Megatron is not importable in local test environment", allow_module_level=True)


class _Config:
    hidden_size = 16
    hidden_dropout = 0.0
    sequence_parallel = False
    clone_scatter_output_in_embedding = False
    fp32_residual_connection = False
    use_mup = False
    mup_embedding_mult = 1.0
    perform_initialization = True
    use_cpu_initialization = False

    @staticmethod
    def embedding_init_method(weight):
        torch.nn.init.zeros_(weight)


def test_custom_embedding_ngram_hash_path_runs(monkeypatch):
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_NGRAM_HASH_ORDERS", "2,3")
    embedding = CppMegaLanguageModelEmbedding(
        config=_Config(),
        vocab_size=128,
        max_sequence_length=16,
        position_embedding_type="rope",
        scatter_to_sequence_parallel=False,
        tp_group=None,
    )
    input_ids = torch.randint(0, 16, (2, 4))
    position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

    out = embedding(input_ids, position_ids)

    assert out.shape == (4, 2, 16)
    monkeypatch.delenv("CPPMEGA_NGRAM_HASH_ENABLED")


def test_custom_embedding_structure_path_runs(monkeypatch):
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_COMPONENTS", "core")
    embedding = CppMegaLanguageModelEmbedding(
        config=_Config(),
        vocab_size=128,
        max_sequence_length=16,
        position_embedding_type="rope",
        scatter_to_sequence_parallel=False,
        tp_group=None,
    )
    input_ids = torch.randint(0, 16, (2, 4))
    position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

    out = embedding(input_ids, position_ids)

    assert out.shape == (4, 2, 16)
    monkeypatch.delenv("CPPMEGA_STRUCTURE_ENABLED")


def test_custom_embedding_structure_path_accepts_non_zero_metadata(monkeypatch):
    monkeypatch.setenv("CPPMEGA_STRUCTURE_ENABLED", "1")
    monkeypatch.setenv("CPPMEGA_STRUCTURE_COMPONENTS", "core")
    embedding = CppMegaLanguageModelEmbedding(
        config=_Config(),
        vocab_size=128,
        max_sequence_length=16,
        position_embedding_type="rope",
        scatter_to_sequence_parallel=False,
        tp_group=None,
    )
    with torch.no_grad():
        embedding.cppmega_structure.stacked_emb.weight.fill_(0.25)
        embedding.cppmega_structure.up_proj.weight.fill_(0.5)

    input_ids = torch.randint(0, 16, (2, 4))
    position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)
    structure_inputs = {
        "structure_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long),
        "dep_levels": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long),
    }

    out = embedding(input_ids, position_ids, structure_inputs=structure_inputs)

    assert out.shape == (4, 2, 16)
    assert torch.count_nonzero(out).item() > 0
    monkeypatch.delenv("CPPMEGA_STRUCTURE_ENABLED")
