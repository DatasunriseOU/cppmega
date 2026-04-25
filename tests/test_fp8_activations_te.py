import pytest
import torch

from cppmega.megatron import fp8_activations as fp8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8._TE_AVAILABLE,
    reason="TE FP8 activation packer requires CUDA and Transformer Engine",
)


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


def test_te_fp8_saved_tensors_hooks_backward(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "te")
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.autograd.graph.saved_tensors_hooks(
        fp8.FP8ActivationPacker.pack,
        fp8.FP8ActivationPacker.unpack,
    ):
        y = (x * x).sum()
    y.backward()
    torch.cuda.synchronize()

    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all()


def test_legacy_fp8_activation_backend_fails_without_ack(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "legacy")
    monkeypatch.delenv(
        "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY",
        raising=False,
    )
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="FP8_ACTIVATION_LEGACY_BACKEND"):
        fp8.FP8ActivationPacker.pack(x)


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
