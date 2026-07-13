from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cppmega.features.domain.embedding import CppMegaDomainEmbedding


def test_domain_embedding_returns_additive_tensor_for_domain_role_confidence():
    emb = CppMegaDomainEmbedding(hidden_size=16, bottleneck_dim=4)
    domain_ids = torch.tensor([[1, 2, 42]], dtype=torch.long)
    role_ids = torch.tensor([[1, 6, 30]], dtype=torch.long)
    confidence_ids = torch.tensor([[4, 4, 1]], dtype=torch.long)

    out = emb(
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        target_dtype=torch.bfloat16,
    )

    assert tuple(out.shape) == (1, 3, 16)
    assert out.dtype == torch.bfloat16


def test_domain_embedding_accepts_missing_optional_sidecars():
    emb = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    domain_ids = torch.tensor([[1, 2]], dtype=torch.long)

    out = emb(
        domain_ids=domain_ids,
        role_ids=None,
        confidence_ids=None,
    )

    assert tuple(out.shape) == (1, 2, 8)


def test_domain_embedding_fails_when_all_sidecars_absent():
    emb = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    with pytest.raises(ValueError, match="all domain sidecars are absent"):
        emb(domain_ids=None, role_ids=None, confidence_ids=None)


def test_domain_embedding_raises_on_out_of_range_ids():
    emb = CppMegaDomainEmbedding(hidden_size=8, num_domains=64, bottleneck_dim=4)
    with pytest.raises(ValueError, match="out of range"):
        emb(domain_ids=torch.tensor([[999]], dtype=torch.long), role_ids=None, confidence_ids=None)


def test_domain_embedding_default_init_has_live_gradient():
    emb = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    out = emb(
        domain_ids=torch.tensor([[1, 2]], dtype=torch.long),
        role_ids=torch.tensor([[1, 2]], dtype=torch.long),
        confidence_ids=torch.tensor([[1, 1]], dtype=torch.long),
    )

    assert torch.count_nonzero(out).item() == 0
    out.sum().backward()
    assert emb.stacked_emb.weight.grad is not None
    assert torch.count_nonzero(emb.stacked_emb.weight.grad).item() > 0
