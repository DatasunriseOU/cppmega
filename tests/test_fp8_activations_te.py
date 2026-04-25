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
    torch.testing.assert_close(out, x, rtol=0.2, atol=0.08)


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


def test_legacy_fp8_activation_backend_still_available(monkeypatch):
    monkeypatch.setenv("CPPMEGA_FP8_ACTIVATION_BACKEND", "legacy")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)

    packed = fp8.FP8ActivationPacker.pack(x)
    assert isinstance(packed, tuple) and len(packed) == 3
    assert packed[0] != fp8._TE_PACK_SENTINEL

    out = fp8.FP8ActivationPacker.unpack(packed)
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
