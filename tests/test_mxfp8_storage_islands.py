from types import SimpleNamespace

import torch
import torch.nn as nn

from cppmega.megatron.mxfp8_storage_islands import (
    Mxfp8StorageIslandConfig,
    apply_mxfp8_storage_islands,
    maybe_dequantize_te_tensor,
    mxfp8_storage_island_config_from_env,
    selected_mxfp8_storage_island_paths,
)


def test_storage_island_config_from_env_and_path_selection(monkeypatch):
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLANDS", "frozen_mxfp8")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_TABLE", "0")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_NGRAM_OUT_PROJ", "1")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_STRUCTURE_TABLE", "1")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_PAD_COLUMNS", "0")
    monkeypatch.setenv("CPPMEGA_MXFP8_STORAGE_ISLAND_COLUMNWISE", "1")

    config = mxfp8_storage_island_config_from_env()

    assert config.mode == "frozen_mxfp8"
    assert config.ngram_table is False
    assert config.ngram_out_proj is True
    assert config.pad_columns is False
    assert config.columnwise is True
    assert selected_mxfp8_storage_island_paths(config) == (
        "embedding.word_embeddings.weight",
        "output_layer.weight",
        "embedding.cppmega_ngram_hash.out_proj.weight",
        "embedding.cppmega_structure.stacked_emb.weight",
    )


def test_storage_island_apply_is_noop_when_disabled():
    model = nn.Module()

    assert apply_mxfp8_storage_islands(model, Mxfp8StorageIslandConfig()) == ()


def test_storage_island_apply_reports_no_cuda_without_touching_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace()
    config = Mxfp8StorageIslandConfig(mode="frozen_mxfp8")

    results = apply_mxfp8_storage_islands(model, config)

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].reason == "cuda_not_available"


def test_maybe_dequantize_te_tensor_accepts_attr_based_quantized_tensor():
    dense = torch.randn(2, 3, dtype=torch.bfloat16)

    class FakeQuantizedTensor:
        _rowwise_data = object()

        def dequantize(self, *, dtype=None):
            return dense.to(dtype or dense.dtype)

    got = maybe_dequantize_te_tensor(FakeQuantizedTensor(), dtype=torch.float32)

    assert got.dtype == torch.float32
    assert torch.allclose(got, dense.to(torch.float32))
