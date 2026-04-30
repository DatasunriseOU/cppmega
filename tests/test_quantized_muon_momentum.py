from __future__ import annotations

import pytest
import torch

from cppmega.megatron.quantized_muon_momentum import (
    TRITON_AVAILABLE,
    QuantizedMuonNormSegment,
    build_quantized_muon_norm_plan,
    dequantize_mxfp8_carrier,
    dequantize_momentum,
    empty_mxfp8_carrier_like,
    empty_quantized_momentum_like,
    quantize_momentum_,
    quantized_muon_momentum_update_mxfp8_carrier_,
    quantized_muon_momentum_update_multi_and_normalize_groups_,
    quantized_muon_momentum_update_multi_and_normalize_,
    quantized_muon_momentum_update_multi_with_group_sumsq_,
    quantized_muon_momentum_update_multi_with_sumsq_,
    quantized_muon_momentum_update_,
    quantized_muon_momentum_update_multi_,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="quantized Muon momentum prototype requires CUDA and Triton",
)


@pytest.mark.parametrize("storage_dtype", [torch.int8, torch.uint8])
def test_quantized_momentum_update_matches_dequantized_reference(storage_dtype: torch.dtype):
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(123)
    old_m = torch.randn((1027,), device=device, dtype=torch.bfloat16, generator=gen)
    grad = torch.randn((1027,), device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(state, old_m)
    old_m_dequant = dequantize_momentum(state)

    scratch = quantized_muon_momentum_update_(state, grad, beta=beta)

    expected = (beta * old_m_dequant + (1.0 - beta) * grad.float()).to(torch.bfloat16)
    torch.testing.assert_close(scratch, expected, atol=4.0e-3, rtol=0.0)


@pytest.mark.parametrize("storage_dtype", [torch.int8, torch.uint8])
def test_quantized_momentum_repeated_steps_stay_close_to_bf16(storage_dtype: torch.dtype):
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(456)
    old_m = 0.2 * torch.randn((513, 769), device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m, storage_dtype=storage_dtype)
    quantize_momentum_(state, old_m)

    bf16_m = old_m.clone()
    scratch = None
    for _ in range(5):
        grad = 0.2 * torch.randn(old_m.shape, device=device, dtype=torch.bfloat16, generator=gen)
        bf16_m = (beta * bf16_m.float() + (1.0 - beta) * grad.float()).to(torch.bfloat16)
        scratch = quantized_muon_momentum_update_(state, grad, beta=beta)

    assert scratch is not None
    diff = (scratch.float() - bf16_m.float()).abs()
    assert diff.max().item() < 2.0e-2
    assert diff.mean().item() < 3.5e-3


def test_quantized_momentum_can_reuse_grad_as_scratch():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(789)
    old_m = torch.randn((2049,), device=device, dtype=torch.bfloat16, generator=gen)
    grad = torch.randn((2049,), device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m)
    quantize_momentum_(state, old_m)
    old_m_dequant = dequantize_momentum(state)

    expected = (beta * old_m_dequant + (1.0 - beta) * grad.float()).to(torch.bfloat16)
    returned = quantized_muon_momentum_update_(state, grad, beta=beta, scratch=grad)

    assert returned is grad
    torch.testing.assert_close(grad, expected, atol=4.0e-3, rtol=0.0)


def test_mxfp8_carrier_layout_is_uint8_rowwise():
    device = "cuda"
    tensor = torch.empty((128, 256), device=device, dtype=torch.float16)

    carrier = empty_mxfp8_carrier_like(tensor)

    assert carrier.rowwise_data.shape == tensor.shape
    assert carrier.rowwise_data.dtype == torch.uint8
    assert carrier.rowwise_scale_inv.shape == (128, 8)
    assert carrier.rowwise_scale_inv.dtype == torch.uint8
    assert carrier.rows == 128
    assert carrier.cols == 256


def test_mxfp8_carrier_update_avoids_bf16_scratch_and_matches_normalized_momentum():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(86420)
    old_m = 0.2 * torch.randn((128, 256), device=device, dtype=torch.float16, generator=gen)
    grad = 0.2 * torch.randn(old_m.shape, device=device, dtype=torch.float16, generator=gen)

    state = empty_quantized_momentum_like(old_m)
    quantize_momentum_(state, old_m)
    old_dequant = dequantize_momentum(state)
    expected_update = beta * old_dequant + (1.0 - beta) * grad.float()
    inv_expected = expected_update.square().sum().clamp_min(1e-14).rsqrt()
    carrier = empty_mxfp8_carrier_like(grad)

    inv_norm = quantized_muon_momentum_update_mxfp8_carrier_(
        state,
        grad,
        carrier,
        beta=beta,
    )
    dequant = dequantize_mxfp8_carrier(carrier)

    torch.testing.assert_close(inv_norm, inv_expected, rtol=2.0e-5, atol=1.0e-8)
    diff = (dequant - expected_update * inv_expected).abs()
    assert diff.max().item() < 2.0e-2
    assert diff.mean().item() < 8.0e-4
    assert carrier.rowwise_data.dtype == torch.uint8
    assert carrier.rowwise_scale_inv.dtype == torch.uint8
    assert grad.dtype == torch.float16


def test_mxfp8_carrier_update_supports_rows_above_legacy_grid_y_limit():
    device = "cuda"
    rows = 65536
    cols = 32
    grad = torch.full((rows, cols), 0.125, device=device, dtype=torch.float16)
    state = empty_quantized_momentum_like(grad)
    carrier = empty_mxfp8_carrier_like(grad)

    inv_norm = quantized_muon_momentum_update_mxfp8_carrier_(
        state,
        grad,
        carrier,
        beta=0.0,
    )

    torch.cuda.synchronize()
    assert inv_norm.dtype == torch.float32
    assert carrier.rowwise_data.shape == (rows, cols)
    assert carrier.rowwise_scale_inv.shape == (rows, 1)
    assert torch.isfinite(inv_norm).all()
    assert bool((carrier.rowwise_data != 0).any().item())


def test_cuda_multi_update_reuses_grads_for_several_tensors():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(2468)
    old_ms = [
        torch.randn((1027,), device=device, dtype=torch.bfloat16, generator=gen),
        torch.randn((513, 769), device=device, dtype=torch.bfloat16, generator=gen),
    ]
    grads = [
        torch.randn_like(old_ms[0], generator=gen),
        torch.randn_like(old_ms[1], generator=gen),
    ]

    states = [empty_quantized_momentum_like(old_m) for old_m in old_ms]
    for state, old_m in zip(states, old_ms):
        quantize_momentum_(state, old_m)
    old_dequants = [dequantize_momentum(state) for state in states]
    expected = [
        (beta * old_dequant + (1.0 - beta) * grad.float()).to(torch.bfloat16)
        for old_dequant, grad in zip(old_dequants, grads)
    ]

    returned = quantized_muon_momentum_update_multi_(states, grads, beta=beta)

    assert returned is grads
    for grad, exp in zip(grads, expected):
        diff = (grad.float() - exp.float()).abs()
        assert diff.max().item() < 2.0e-2
        assert diff.mean().item() < 3.5e-3


def test_cuda_multi_update_with_sumsq_matches_updated_momentum_norm():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(1357)
    old_m = torch.randn((2049,), device=device, dtype=torch.bfloat16, generator=gen)
    grad = torch.randn((2049,), device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m)
    quantize_momentum_(state, old_m)
    old_dequant = dequantize_momentum(state)
    expected = beta * old_dequant + (1.0 - beta) * grad.float()

    partial_sumsq = quantized_muon_momentum_update_multi_with_sumsq_([state], [grad], beta=beta)

    torch.testing.assert_close(partial_sumsq.sum(), expected.square().sum(), rtol=2.0e-5, atol=1.0e-3)
    torch.testing.assert_close(grad, expected.to(torch.bfloat16), rtol=0.0, atol=2.0e-2)


def test_cuda_multi_update_and_normalize_reuses_grad_as_ns_input():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(97531)
    old_ms = [
        torch.randn((257,), device=device, dtype=torch.bfloat16, generator=gen),
        torch.randn((256, 64), device=device, dtype=torch.bfloat16, generator=gen),
    ]
    grads = [torch.randn_like(old_m, generator=gen) for old_m in old_ms]

    states = [empty_quantized_momentum_like(old_m) for old_m in old_ms]
    for state, old_m in zip(states, old_ms):
        quantize_momentum_(state, old_m)
    old_dequants = [dequantize_momentum(state) for state in states]
    updated = [
        beta * old_dequant + (1.0 - beta) * grad.float()
        for old_dequant, grad in zip(old_dequants, grads)
    ]
    inv_norm_expected = torch.cat([u.reshape(-1) for u in updated]).square().sum().clamp_min(1e-14).rsqrt()

    inv_norm = quantized_muon_momentum_update_multi_and_normalize_(states, grads, beta=beta)

    torch.testing.assert_close(inv_norm, inv_norm_expected, rtol=2.0e-5, atol=1.0e-8)
    for grad, upd in zip(grads, updated):
        torch.testing.assert_close(grad, (upd * inv_norm_expected).to(torch.bfloat16), rtol=0.0, atol=2.0e-2)


def test_grouped_update_and_normalize_matches_qkv_slice_norms():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(112233)
    num_query_groups = 3
    qkv_split = (2, 1, 1)
    rows_per_group = sum(qkv_split)
    cols = 256
    shape = (num_query_groups * rows_per_group, cols)
    old_m = 0.2 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
    grad = 0.2 * torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)

    state = empty_quantized_momentum_like(old_m)
    quantize_momentum_(state, old_m)
    old_dequant = dequantize_momentum(state)
    updated = beta * old_dequant + (1.0 - beta) * grad.float()

    segments = []
    for query_group in range(num_query_groups):
        base_row = query_group * rows_per_group
        cursor = base_row
        for group_id, split_rows in enumerate(qkv_split):
            start = cursor * cols
            length = split_rows * cols
            segments.append(
                QuantizedMuonNormSegment(
                    tensor_index=0,
                    start=start,
                    length=length,
                    group_id=group_id,
                )
            )
            cursor += split_rows
    norm_plan = build_quantized_muon_norm_plan([state], segments, num_groups=3)

    q_expected = []
    for group_id, split_rows in enumerate(qkv_split):
        pieces = []
        for query_group in range(num_query_groups):
            base = query_group * rows_per_group
            row_start = base + sum(qkv_split[:group_id])
            pieces.append(updated[row_start : row_start + split_rows])
        q_expected.append(torch.cat([piece.reshape(-1) for piece in pieces]))
    sumsq_expected = torch.stack([piece.square().sum() for piece in q_expected])
    inv_expected = sumsq_expected.clamp_min(1e-14).rsqrt()

    inv_norms = quantized_muon_momentum_update_multi_and_normalize_groups_(
        [state],
        [grad],
        norm_plan,
        beta=beta,
    )

    torch.testing.assert_close(inv_norms, inv_expected, rtol=2.0e-5, atol=1.0e-8)
    for group_id, split_rows in enumerate(qkv_split):
        for query_group in range(num_query_groups):
            base = query_group * rows_per_group
            row_start = base + sum(qkv_split[:group_id])
            expected = (updated[row_start : row_start + split_rows] * inv_expected[group_id]).to(
                torch.bfloat16
            )
            torch.testing.assert_close(
                grad[row_start : row_start + split_rows],
                expected,
                rtol=0.0,
                atol=2.0e-2,
            )


def test_grouped_sumsq_can_accumulate_one_logical_group_across_shards():
    device = "cuda"
    beta = 0.95
    gen = torch.Generator(device=device).manual_seed(445566)
    old_ms = [
        0.2 * torch.randn((2, 256), device=device, dtype=torch.bfloat16, generator=gen),
        0.2 * torch.randn((3, 256), device=device, dtype=torch.bfloat16, generator=gen),
    ]
    grads = [0.2 * torch.randn_like(old_m, generator=gen) for old_m in old_ms]
    states = [empty_quantized_momentum_like(old_m) for old_m in old_ms]
    for state, old_m in zip(states, old_ms):
        quantize_momentum_(state, old_m)

    updated = [
        beta * dequantize_momentum(state) + (1.0 - beta) * grad.float()
        for state, grad in zip(states, grads)
    ]
    segments = [
        QuantizedMuonNormSegment(0, 0, old_ms[0].numel(), 0),
        QuantizedMuonNormSegment(1, 0, old_ms[1].numel(), 0),
    ]
    norm_plan = build_quantized_muon_norm_plan(states, segments, num_groups=1)

    group_sumsq = quantized_muon_momentum_update_multi_with_group_sumsq_(
        states,
        grads,
        norm_plan,
        beta=beta,
    )
    expected = torch.cat([u.reshape(-1) for u in updated]).square().sum().reshape(1)
    torch.testing.assert_close(group_sumsq, expected, rtol=2.0e-5, atol=1.0e-3)


def test_norm_plan_rejects_mid_block_qkv_boundary():
    device = "cuda"
    state = empty_quantized_momentum_like(torch.empty((513,), device=device, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="segment end must align"):
        build_quantized_muon_norm_plan(
            [state],
            [QuantizedMuonNormSegment(tensor_index=0, start=0, length=257, group_id=0)],
            num_groups=1,
        )
