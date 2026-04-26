import pytest
import torch

from cppmega.megatron import fp8_activations as fp8


requires_te_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8._TE_AVAILABLE,
    reason="TE FP8 activation packer requires CUDA and Transformer Engine",
)


def _fp8_pinned_square_input(*, device: str, dtype: torch.dtype) -> torch.Tensor:
    codes = torch.tensor(
        [-448, -224, -112, -56, -28, -14, -7, 0, 7, 14, 28, 56, 112, 224, 448],
        device=device,
        dtype=torch.float32,
    )
    values = codes / 256.0
    row = values.repeat((256 + values.numel() - 1) // values.numel())[:256]
    return row.repeat(128, 1).to(dtype)


def _square_loss_grad(
    x: torch.Tensor,
    *,
    pack=None,
    unpack=None,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    if pack is None or unpack is None:
        y = (x * x).sum()
    else:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            y = (x * x).sum()
    y.backward()
    if x.is_cuda:
        torch.cuda.synchronize()
    assert x.grad is not None
    return x.grad.detach()


def test_backward_reference_input_is_fp8_pinned():
    if fp8._FP8_DTYPE is None:
        pytest.skip("torch FP8 dtype is not available")
    x = _fp8_pinned_square_input(device="cpu", dtype=torch.float32)
    scale = x.abs().amax(dim=-1, keepdim=True) / torch.finfo(fp8._FP8_DTYPE).max

    try:
        roundtrip = (x / scale).to(fp8._FP8_DTYPE).to(torch.float32) * scale
    except RuntimeError as exc:
        pytest.skip(f"torch FP8 CPU cast is not available: {exc}")

    torch.testing.assert_close(roundtrip, x, rtol=0, atol=0)


@requires_te_cuda
def test_te_fp8_activation_roundtrip(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "te")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    packed = fp8.FP8ActivationPacker.pack(x)
    assert isinstance(packed, tuple) and len(packed) == 3
    assert packed[0] == fp8._TE_PACK_SENTINEL

    out = fp8.FP8ActivationPacker.unpack(packed)
    torch.cuda.synchronize()

    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    assert out.is_contiguous()
    assert torch.isfinite(out).all()

    # Compare against a direct TE quantize/dequantize of the same input
    # rather than against the original bf16 tensor. The previous bound
    # (rtol=0.2, atol=0.08) measured FP8 quantization noise rather than
    # the packer wrapper, and would have passed even if the packer
    # silently routed through a different quantizer.
    quantizer = fp8._te_quantizer_for(x.device)
    te_packed = fp8.tex.quantize(x, quantizer)
    te_ref = fp8.tex.dequantize(te_packed, fp8._torch_dtype_to_te(x.dtype))
    torch.testing.assert_close(out, te_ref, rtol=0, atol=0)


@requires_te_cuda
def test_te_fp8_saved_tensors_hooks_backward_matches_reference(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "te")
    monkeypatch.setattr(fp8, "_FP8_MIN_ELEMENTS", 1)
    fp8._TE_QUANTIZER_CACHE.clear()
    x = _fp8_pinned_square_input(device="cuda", dtype=torch.bfloat16)
    reference_grad = _square_loss_grad(x)

    te_pack_count = 0
    te_unpack_count = 0

    def pack(tensor):
        nonlocal te_pack_count
        packed = fp8.FP8ActivationPacker.pack(tensor)
        if (
            isinstance(packed, tuple)
            and len(packed) == 3
            and packed[0] == fp8._TE_PACK_SENTINEL
        ):
            te_pack_count += 1
        return packed

    def unpack(packed):
        nonlocal te_unpack_count
        if (
            isinstance(packed, tuple)
            and len(packed) == 3
            and packed[0] == fp8._TE_PACK_SENTINEL
        ):
            te_unpack_count += 1
        return fp8.FP8ActivationPacker.unpack(packed)

    fp8._TE_QUANTIZER_CACHE.clear()
    fp8_grad = _square_loss_grad(x, pack=pack, unpack=unpack)

    assert te_pack_count >= 2
    assert te_unpack_count >= 2
    assert fp8_grad.dtype == torch.bfloat16
    torch.testing.assert_close(fp8_grad, reference_grad, rtol=0, atol=0)


@requires_te_cuda
def test_legacy_fp8_activation_backend_fails_without_ack(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "legacy")
    monkeypatch.delenv(
        "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY",
        raising=False,
    )
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="FP8_ACTIVATION_LEGACY_BACKEND"):
        fp8.FP8ActivationPacker.pack(x)


@requires_te_cuda
def test_legacy_fp8_activation_backend_still_available_with_ack(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "legacy")
    monkeypatch.setenv(
        "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY",
        "1",
    )
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    packed = fp8.FP8ActivationPacker.pack(x)
    assert isinstance(packed, tuple) and len(packed) == 3
    assert packed[0] != fp8._TE_PACK_SENTINEL

    out = fp8.FP8ActivationPacker.unpack(packed)
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
