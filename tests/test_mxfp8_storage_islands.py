from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from cppmega.megatron.mxfp8_storage_islands import (
    Mxfp8StorageIslandConfig,
    apply_mxfp8_storage_islands,
    is_te_quantized_tensor,
    maybe_dequantize_te_tensor,
    mxfp8_storage_island_config_from_env,
    selected_mxfp8_storage_island_paths,
)


def test_storage_island_config_from_env_and_path_selection(monkeypatch):
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLANDS", "trainable_mxfp8")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_TABLE", "0")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_OUT_PROJ", "1")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_STRUCTURE_TABLE", "1")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_PAD_COLUMNS", "0")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_COLUMNWISE", "1")

    config = mxfp8_storage_island_config_from_env()

    assert config.mode == "trainable_mxfp8"
    assert config.ngram_table is False
    assert config.ngram_out_proj is True
    assert config.pad_columns is False
    assert config.columnwise is True
    assert selected_mxfp8_storage_island_paths(config) == (
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        "embedding.cppmega_ngram_hash.out_proj.weight",
        "embedding.cppmega_structure.stacked_emb.weight",
        "embedding.cppmega_structure.up_proj.weight",
    )


def test_storage_island_apply_is_noop_when_disabled():
    model = nn.Module()

    assert apply_mxfp8_storage_islands(model, Mxfp8StorageIslandConfig()) == ()


def test_storage_island_apply_reports_no_cuda_without_touching_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace()
    config = Mxfp8StorageIslandConfig(mode="trainable_mxfp8")

    results = apply_mxfp8_storage_islands(model, config)

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].reason == "cuda_not_available"


def test_storage_island_rejects_frozen_probe_mode(monkeypatch):
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLANDS", "frozen_mxfp8")

    with pytest.raises(ValueError, match="not a trainable storage path"):
        mxfp8_storage_island_config_from_env()


def test_maybe_dequantize_te_tensor_accepts_attr_based_quantized_tensor():
    dense = torch.randn(2, 3, dtype=torch.bfloat16)

    class FakeQuantizedTensor:
        _rowwise_data = object()

        def dequantize(self, *, dtype=None):
            return dense.to(dtype or dense.dtype)

    got = maybe_dequantize_te_tensor(FakeQuantizedTensor(), dtype=torch.float32)

    assert got.dtype == torch.float32
    assert torch.allclose(got, dense.to(torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA MXFP8 kernels")
def test_storage_island_cuda_conversion_keeps_parameters_trainable():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Module()
            self.embedding.word_embeddings = nn.Embedding(33, 32)
            self.embedding.cppmega_ngram_hash = nn.Module()
            self.embedding.cppmega_ngram_hash.unified_table = nn.Embedding(65, 16)
            self.output_layer = nn.Linear(32, 64, bias=False)

    model = ToyModel().cuda().to(dtype=torch.bfloat16)
    config = Mxfp8StorageIslandConfig(
        mode="trainable_mxfp8",
        ngram_out_proj=False,
        structure_table=False,
        structure_up_proj=False,
    )

    results = apply_mxfp8_storage_islands(model, config)

    converted = {result.path: result for result in results if result.status == "converted"}
    assert set(converted) == {
        "embedding.word_embeddings.weight",
        "embedding.cppmega_ngram_hash.unified_table.weight",
        "output_layer.weight",
    }
    assert model.embedding.word_embeddings.weight.requires_grad
    assert model.output_layer.weight.requires_grad
    assert getattr(model.output_layer.weight, "_cppmega_mxfp8_trainable_storage")
    assert not hasattr(model.output_layer.weight, "_cppmega_mxfp8_frozen_storage")

    ids = torch.randint(0, 33, (4, 8), device="cuda")
    hidden = F.embedding(ids, model.embedding.word_embeddings.weight)
    ngram_ids = torch.randint(0, 65, (4, 8), device="cuda")
    ngram = F.embedding(
        ngram_ids, model.embedding.cppmega_ngram_hash.unified_table.weight
    )
    loss = model.output_layer(hidden).float().sum() + ngram[..., :16].float().sum()
    loss.backward()

    assert model.embedding.word_embeddings.weight.grad is not None
    assert model.embedding.cppmega_ngram_hash.unified_table.weight.grad is not None
    assert model.output_layer.weight.grad is not None
    assert model.embedding.word_embeddings.weight.grad.dtype == torch.bfloat16

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.step()

    assert is_te_quantized_tensor(model.embedding.word_embeddings.weight)
    assert is_te_quantized_tensor(model.embedding.cppmega_ngram_hash.unified_table.weight)
    assert is_te_quantized_tensor(model.output_layer.weight)
