from __future__ import annotations

import pytest

from cppmega.megatron.sparse_mla_ops import (
    sparse_mla_blockscaled_mxfp8_backward,
    sparse_mla_blockscaled_mxfp8_forward,
)


def test_blockscaled_fused_forward_requires_env(monkeypatch):
    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED", raising=False)

    with pytest.raises(RuntimeError, match="CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED"):
        sparse_mla_blockscaled_mxfp8_forward(
            None,
            None,
            None,
            None,
            None,
            softmax_scale=1.0,
            d_v=64,
        )


def test_blockscaled_fused_backward_requires_reference_or_unsafe_ack(monkeypatch):
    monkeypatch.setenv("CPPMEGA_SPARSE_MLA_BLOCKSCALED_FUSED", "1")
    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_BLOCKSCALED_BWD_REFERENCE_ACK", raising=False)
    monkeypatch.delenv("CPPMEGA_SPARSE_MLA_BLOCKSCALED_TILELANG_BWD_UNSAFE", raising=False)

    with pytest.raises(RuntimeError, match="CPPMEGA_SPARSE_MLA_BLOCKSCALED_BWD_REFERENCE_ACK"):
        sparse_mla_blockscaled_mxfp8_backward(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            softmax_scale=1.0,
            d_v=64,
        )
