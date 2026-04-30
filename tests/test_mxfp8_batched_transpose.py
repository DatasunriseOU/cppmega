import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_batched_transpose_copies_uint8_payloads_and_compact_scales():
    from cppmega.megatron import mxfp8_batched_transpose as batched

    device = torch.device("cuda")
    first = torch.arange(64 * 96, device=device).remainder(251).to(torch.uint8).reshape(64, 96)
    second = torch.arange(96 * 64, device=device).remainder(241).to(torch.uint8).reshape(96, 64)
    first_scale = torch.arange(4 * 128, device=device).remainder(251).to(torch.uint8).reshape(4, 128)
    second_scale = torch.arange(4 * 128, device=device).remainder(241).to(torch.uint8).reshape(4, 128)
    first_out = torch.empty((96, 64), device=device, dtype=torch.uint8)
    second_out = torch.empty((64, 96), device=device, dtype=torch.uint8)
    first_scale_out = torch.empty((128, 4), device=device, dtype=torch.uint8)
    second_scale_out = torch.empty((128, 4), device=device, dtype=torch.uint8)

    batched.batched_transpose(
        [
            {
                "kind": batched.KIND_UINT8_TRANSPOSE,
                "input": first,
                "columnwise_scale_inv": first_scale,
                "output_rowwise_data": first_out,
                "output_rowwise_scale_inv": first_scale_out,
            },
            {
                "kind": batched.KIND_UINT8_TRANSPOSE,
                "input": second,
                "columnwise_scale_inv": second_scale,
                "output_rowwise_data": second_out,
                "output_rowwise_scale_inv": second_scale_out,
            },
        ]
    )
    torch.cuda.synchronize()

    assert torch.equal(first_out, first.t().contiguous())
    assert torch.equal(second_out, second.t().contiguous())
    assert torch.equal(first_scale_out, first_scale.t().contiguous())
    assert torch.equal(second_scale_out, second_scale.t().contiguous())


def test_batched_bf16_emit_matches_transformer_engine_transpose_cast():
    pytest.importorskip("transformer_engine_torch")
    import transformer_engine_torch as tex
    from cppmega.megatron import mxfp8_batched_transpose as batched

    if not hasattr(tex, "mxfp8_scaling_transpose_cast"):
        pytest.skip("TransformerEngine extension lacks mxfp8_scaling_transpose_cast")

    device = torch.device("cuda")
    source = torch.randn((64, 96), device=device, dtype=torch.bfloat16)
    scale = torch.full((4, 128), 127, device=device, dtype=torch.uint8)
    out = torch.empty((96, 64), device=device, dtype=torch.uint8)
    out_scale = torch.empty((128, 4), device=device, dtype=torch.uint8)
    ref = torch.empty_like(out)
    ref_scale = torch.empty_like(out_scale)

    batched.batched_transpose(
        [
            {
                "kind": batched.KIND_BF16_EMIT,
                "input": source,
                "columnwise_scale_inv": scale,
                "output_rowwise_data": out,
                "output_rowwise_scale_inv": out_scale,
                "with_gemm_swizzled_scales": True,
            }
        ]
    )
    tex.mxfp8_scaling_transpose_cast(
        source,
        scale,
        ref,
        ref_scale,
        64,
        96,
        int(tex.DType.kFloat8E4M3),
        True,
    )
    torch.cuda.synchronize()

    assert torch.equal(out, ref)
    assert torch.equal(out_scale, ref_scale)
