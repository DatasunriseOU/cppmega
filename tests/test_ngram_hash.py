import torch

from cppmega.features.engram.config import NgramHashConfig
from cppmega.features.engram.ngram_hash import CppMegaNgramHashEmbedding


def test_ngram_hash_config_validates_and_parses_orders():
    config = NgramHashConfig.from_nanochat_args(
        enabled=True,
        orders="2,3",
        heads=4,
        table_size=1024,
        embed_dim=8,
        offload=True,
    )

    assert config is not None
    assert config.orders == (2, 3)
    assert config.heads == 4
    assert config.offload is True


def test_ngram_hash_embedding_returns_hidden_sized_tensor_and_zero_init_projection():
    module = CppMegaNgramHashEmbedding(
        hidden_size=32,
        orders=(2, 3),
        num_heads=2,
        table_size=257,
        embed_dim=8,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)

    out = module(token_ids)

    assert out.shape == (2, 4, 32)
    assert torch.count_nonzero(out) == 0


def test_ngram_hash_embedding_backward_reaches_unified_table():
    module = CppMegaNgramHashEmbedding(
        hidden_size=16,
        orders=(2,),
        num_heads=2,
        table_size=257,
        embed_dim=4,
    )
    with torch.no_grad():
        module.out_proj.weight.fill_(1.0)
    token_ids = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)

    out = module(token_ids)
    loss = out.sum()
    loss.backward()

    assert module.unified_table.weight.grad is not None
    assert torch.isfinite(module.unified_table.weight.grad).all().item()
