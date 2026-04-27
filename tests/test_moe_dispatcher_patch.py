import torch

from cppmega.megatron.moe_dispatcher_patch import (
    apply_moe_dispatcher_identity_sort_patch,
    is_identity_permutation,
)


def test_identity_permutation_detection():
    assert is_identity_permutation(torch.arange(4))
    assert not is_identity_permutation(torch.tensor([0, 2, 1, 3]))


def test_identity_chunk_sort_patch_skips_noop_sort():
    assert apply_moe_dispatcher_identity_sort_patch(force=True)

    from megatron.core.transformer.moe import moe_utils

    tokens = torch.arange(12).view(6, 2)
    probs = torch.arange(6, dtype=torch.float32)
    split_sizes = torch.tensor([2, 1, 3])
    sorted_idxs = torch.arange(3)

    sorted_tokens, sorted_probs = moe_utils.sort_chunks_by_idxs(
        tokens, split_sizes, sorted_idxs, probs=probs, fused=False
    )

    assert sorted_tokens is tokens
    assert sorted_probs is probs


def test_identity_chunk_sort_patch_preserves_non_identity_sort():
    assert apply_moe_dispatcher_identity_sort_patch(force=True)

    from megatron.core.transformer.moe import moe_utils

    tokens = torch.arange(12).view(6, 2)
    probs = torch.arange(6, dtype=torch.float32)
    split_sizes = torch.tensor([2, 1, 3])
    sorted_idxs = torch.tensor([2, 0, 1])

    sorted_tokens, sorted_probs = moe_utils.sort_chunks_by_idxs(
        tokens, split_sizes, sorted_idxs, probs=probs, fused=False
    )

    expected_tokens = torch.cat((tokens[3:6], tokens[0:2], tokens[2:3]), dim=0)
    expected_probs = torch.cat((probs[3:6], probs[0:2], probs[2:3]), dim=0)
    assert torch.equal(sorted_tokens, expected_tokens)
    assert torch.equal(sorted_probs, expected_probs)
