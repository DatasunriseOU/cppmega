"""Dispatch tests for the ``pararnn`` kernel option in ``CppMegaM2RNNMixer``.

The ParaRNN-style Newton + Brent-Kung scan is a third option alongside the
default ``triton`` (sequential, fast) and ``torch`` (Python loop reference)
paths in ``cppmega.megatron.m2rnn_spec``. Selection is via the
``CPPMEGA_M2RNN_KERNEL=pararnn`` env var. We don't check parity vs the other
kernels here — the kernels compute the same recurrence to within Newton
residual + bf16 roundoff, but the constant-factor differences make a
brittle parity test pointless. Just verify dispatch + finiteness + shape.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch


def _has(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ImportError, ValueError):
        return False


_HAS_MEGATRON = _has("megatron.core.transformer")
_HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _HAS_CUDA, reason="pararnn dispatch requires CUDA")
@pytest.mark.skipif(not _HAS_MEGATRON, reason="megatron not installed locally")
def test_m2rnn_kernel_pararnn_dispatch(monkeypatch):
    """Setting ``CPPMEGA_M2RNN_KERNEL=pararnn`` routes ``CppMegaM2RNNMixer.forward``
    through ``m2rnn_pararnn_forward`` and produces a finite output of the right
    shape. Parity vs ``triton`` is not checked (different roundoff, different
    Newton residual).
    """
    # Lightweight TP=1 process-group init so TENorm can resolve its sequence
    # parallel info. Skip if it's already initialised by a previous test.
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )

    from megatron.core.transformer.transformer_config import TransformerConfig

    monkeypatch.setenv("CPPMEGA_M2RNN_KERNEL", "pararnn")

    # Small config so the test stays fast; the dispatch path is the same as
    # the production NAM56R recipe.
    hidden_size = 320  # 4 heads at (k=64, v=16) -> hidden = 4 * 80
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=hidden_size * 2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )

    from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    mixer = CppMegaM2RNNMixer(config=config, d_model=hidden_size).cuda()

    seq, batch = 64, 2
    x = torch.randn(
        seq, batch, hidden_size, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )

    # Spy on the pararnn forward to confirm we actually go through that path.
    import cppmega.megatron.m2rnn_spec as spec_mod

    call_log = {"pararnn": 0, "triton": 0}
    real_pararnn = spec_mod._m2rnn_pararnn_forward
    real_triton = spec_mod._m2rnn_scan_triton

    def _spy_pararnn(*args, **kwargs):
        call_log["pararnn"] += 1
        return real_pararnn(*args, **kwargs)

    def _spy_triton(*args, **kwargs):  # pragma: no cover -- shouldn't fire
        call_log["triton"] += 1
        return real_triton(*args, **kwargs)

    monkeypatch.setattr(spec_mod, "_m2rnn_pararnn_forward", _spy_pararnn)
    monkeypatch.setattr(spec_mod, "_m2rnn_scan_triton", _spy_triton)

    out, residual = mixer(x)

    assert call_log["pararnn"] == 1, (
        f"pararnn path was not taken (calls: {call_log})"
    )
    assert call_log["triton"] == 0, (
        f"triton path was taken instead of pararnn (calls: {call_log})"
    )

    assert out.shape == (seq, batch, hidden_size), (
        f"unexpected output shape: {out.shape}"
    )
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "non-finite output from pararnn dispatch"
    assert residual is None  # mixer always returns None for the residual slot

    # Quick backward sanity check: gradient should be finite.
    loss = out.float().pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "non-finite grad through pararnn path"
