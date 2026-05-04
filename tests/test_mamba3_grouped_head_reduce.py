import pytest
import torch

from cppmega.megatron import mamba3_grouped_head_reduce as reduce


def test_torch_grouped_head_pair_matches_manual_sum():
    dq = torch.arange(2 * 3 * 2 * 4 * 5, dtype=torch.float32).reshape(2, 3, 2, 4, 5)
    dk = dq + 0.5

    got_dq, got_dk = reduce.reduce_grouped_heads_torch(dq, dk, groups=2)

    expected_dq = dq.view(2, 3, 2, 2, 2, 5).sum(dim=4)
    expected_dk = dk.view(2, 3, 2, 2, 2, 5).sum(dim=4)
    torch.testing.assert_close(got_dq, expected_dq)
    torch.testing.assert_close(got_dk, expected_dk)


def test_default_backend_is_torch(monkeypatch):
    monkeypatch.delenv("CPPMEGA_MAMBA3_GROUPED_HEAD_REDUCE_BACKEND", raising=False)
    dq = torch.randn(1, 4, 2, 8, 3)
    dk = torch.randn_like(dq)

    got_dq, got_dk = reduce.reduce_grouped_heads(dq, dk, groups=4)

    assert got_dq.shape == (1, 4, 2, 4, 3)
    assert got_dk.shape == (1, 4, 2, 4, 3)
    torch.testing.assert_close(got_dq, dq.view(1, 4, 2, 4, 2, 3).sum(dim=4))
    torch.testing.assert_close(got_dk, dk.view(1, 4, 2, 4, 2, 3).sum(dim=4))


def test_invalid_group_shape_rejected():
    dq = torch.randn(1, 4, 2, 7, 3)
    dk = torch.randn_like(dq)

    with pytest.raises(ValueError, match="H must be divisible"):
        reduce.reduce_grouped_heads_torch(dq, dk, groups=4)


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("CPPMEGA_MAMBA3_GROUPED_HEAD_REDUCE_BACKEND", "cuda")
    dq = torch.randn(1, 4, 2, 8, 3)
    dk = torch.randn_like(dq)

    with pytest.raises(ValueError, match="unsupported"):
        reduce.reduce_grouped_heads(dq, dk, groups=4)
