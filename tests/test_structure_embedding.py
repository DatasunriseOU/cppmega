import torch

from cppmega.features.structure.embedding import CppMegaStructureEmbedding


def test_structure_embedding_core_returns_hidden_sized_tensor():
    module = CppMegaStructureEmbedding(hidden_size=32, active_components="core", bottleneck_dim=8)
    structure_ids = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long)
    dep_levels = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)

    out = module(structure_ids=structure_ids, dep_levels=dep_levels, target_dtype=torch.float32)

    assert out.shape == (2, 3, 32)
    assert torch.count_nonzero(out) == 0


def test_structure_embedding_backward_reaches_tables():
    module = CppMegaStructureEmbedding(hidden_size=16, active_components="core", bottleneck_dim=4)
    structure_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    dep_levels = torch.tensor([[0, 1, 2]], dtype=torch.long)

    out = module(structure_ids=structure_ids, dep_levels=dep_levels, target_dtype=torch.float32)
    out.sum().backward()

    assert module.stacked_emb.weight.grad is not None
    assert torch.isfinite(module.stacked_emb.weight.grad).all().item()
    assert torch.count_nonzero(module.stacked_emb.weight.grad).item() > 0


def test_structure_embedding_rejects_missing_or_invalid_sidecars():
    module = CppMegaStructureEmbedding(hidden_size=16, active_components="core", bottleneck_dim=4)

    try:
        module(structure_ids=None, dep_levels=None)
    except ValueError as exc:
        assert "all active sidecars are absent" in str(exc)
    else:
        raise AssertionError("missing enabled structure sidecars must fail closed")

    try:
        module(
            structure_ids=torch.tensor([[999]], dtype=torch.long),
            dep_levels=torch.tensor([[0]], dtype=torch.long),
        )
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("invalid structure ids must fail closed")
