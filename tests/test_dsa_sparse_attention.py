from __future__ import annotations

import torch

from cppmega.megatron.dsa_sparse_attention import sparse_dsa_fn


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    query = torch.randn(3, 1, 2, 4)
    key = torch.randn(3, 1, 2, 4)
    value = torch.randn(3, 1, 2, 3)
    return query, key, value


def test_invalid_topk_sentinels_are_masked_instead_of_gathered() -> None:
    query, key, value = _inputs()
    with_sentinels = torch.tensor(
        [[[0, -1, 9], [1, -1, 9], [2, -1, 99]]], dtype=torch.int64
    )
    duplicated_valid = torch.tensor(
        [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]], dtype=torch.int64
    )

    actual = sparse_dsa_fn(query, key, value, with_sentinels, 0.5)
    expected = sparse_dsa_fn(query, key, value, duplicated_valid, 0.5)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_all_invalid_topk_row_produces_zero_without_nan() -> None:
    query, key, value = _inputs()
    indices = torch.tensor(
        [[[-1, -1], [0, 1], [1, 2]]], dtype=torch.int64
    )

    output = sparse_dsa_fn(query, key, value, indices, 0.5)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[0]) == 0
