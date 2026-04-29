import torch

from cppmega.megatron.cute_dsl_mimo.full_bwd_bwd_epilogue import (
    compute_dstates_before_chunks_pytorch,
    full_bwd_bwd_pytorch,
    full_bwd_bwd_pytorch_state_chunk_split,
)


def _inputs():
    torch.manual_seed(20260429)
    B, S, H, G, N, P, R = 1, 32, 2, 1, 8, 8, 2
    chunk = 16
    rotary_dim_divisor = 4
    nchunks = S // chunk
    rdim = N // rotary_dim_divisor
    return {
        "shape": (B, S, H, G, N, P, R, chunk, rotary_dim_divisor),
        "dout": torch.randn(B, S, H, P) * 0.01,
        "q": torch.randn(B, S, R, G, N) * 0.01,
        "k": torch.randn(B, S, R, G, N) * 0.01,
        "v": torch.randn(B, S, H, P) * 0.01,
        "q_bias": torch.randn(H, R, N) * 0.01,
        "k_bias": torch.randn(H, R, N) * 0.01,
        "mimo_v": torch.randn(H, R, P) * 0.01,
        "mimo_o": torch.randn(H, R, P) * 0.01,
        "angles": torch.randn(B, S, H, rdim) * 0.01,
        "dA_cs": -torch.rand(B, H, S) * 0.01,
        "dA_cs_rev": -torch.rand(B, H, S) * 0.01,
        "dt": torch.randn(B, H, S) * 0.01,
        "trap": torch.randn(B, H, S) * 0.01,
        "D": torch.randn(H) * 0.01,
        "segsum": torch.randn(B, H, nchunks, chunk, chunk) * 0.01,
        "states": torch.randn(B, H, nchunks, N, P) * 0.01,
        "qk_dot": torch.randn(B, H, S, R, R) * 0.01,
    }


def test_state_chunk_split_matches_monolithic_reference():
    data = _inputs()
    B, S, H, G, N, P, R, chunk, rotary_dim_divisor = data["shape"]
    del B, S, H, G, N, P

    kwargs = {
        "dout": data["dout"],
        "q_raw": data["q"],
        "k_raw": data["k"],
        "v": data["v"],
        "q_bias": data["q_bias"],
        "k_bias": data["k_bias"],
        "mimo_v": data["mimo_v"],
        "mimo_o": data["mimo_o"],
        "angles": data["angles"],
        "dA_cs": data["dA_cs"],
        "dA_cs_rev": data["dA_cs_rev"],
        "dt": data["dt"],
        "trap": data["trap"],
        "D": data["D"],
        "segsum": data["segsum"],
        "states": data["states"],
        "qk_dot": data["qk_dot"],
        "chunk_size": chunk,
        "R": R,
        "rotary_dim_divisor": rotary_dim_divisor,
    }
    monolithic = full_bwd_bwd_pytorch(**kwargs)
    split = full_bwd_bwd_pytorch_state_chunk_split(**kwargs)

    for name, expected in monolithic.items():
        actual = split[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_state_chunk_split_pass1_buffer_matches_monolithic_capture():
    data = _inputs()
    _, _, _, _, _, _, R, chunk, rotary_dim_divisor = data["shape"]

    captured = full_bwd_bwd_pytorch(
        data["dout"],
        data["q"],
        data["k"],
        data["v"],
        data["q_bias"],
        data["k_bias"],
        data["mimo_v"],
        data["mimo_o"],
        data["angles"],
        data["dA_cs"],
        data["dA_cs_rev"],
        data["dt"],
        data["trap"],
        data["D"],
        data["segsum"],
        data["states"],
        data["qk_dot"],
        chunk_size=chunk,
        R=R,
        rotary_dim_divisor=rotary_dim_divisor,
        return_dstates_before=True,
    )["DSTATES_BEFORE_CHUNKS"]
    pass1 = compute_dstates_before_chunks_pytorch(
        data["dout"],
        data["q"],
        data["q_bias"],
        data["mimo_o"],
        data["angles"],
        data["dA_cs"],
        chunk_size=chunk,
        rotary_dim_divisor=rotary_dim_divisor,
    )

    torch.testing.assert_close(pass1, captured, rtol=0, atol=0)
