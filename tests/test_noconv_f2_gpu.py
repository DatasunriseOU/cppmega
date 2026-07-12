import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
pytest.importorskip("mamba_ssm")
if not torch.cuda.is_available():
    pytest.skip("noconv F2 test needs a CUDA GPU + Megatron + mamba_ssm (H200 container)", allow_module_level=True)


"""Standalone CUDA test for F2: build the real Mamba3NoConvMixer with nheads=8,
ngroups=2 (ngroups<nheads → the trapezoidal per-head-expand path), run forward+
backward, and assert the SSD kernel accepts per-head B/C [b,l,nheads,d_state] and
produces finite output+grads. No dataset / dataloader / loss_mask.
"""
import os
import torch.distributed as dist


def test_noconv_f2_per_head_trapezoidal():
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    # ColumnParallelLinear weight init forks the 'model-parallel-rng' state, which
    # must be registered first (production does this in megatron init).
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from cppmega.megatron.noconv_mamba_mixer import (
        Mamba3NoConvMixer, NoConvMambaMixerSubmodules,
    )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=8,
        mamba_state_dim=16,
        mamba_head_dim=4,
        mamba_num_heads=8,     # nheads
        mamba_num_groups=2,    # ngroups  (ngroups < nheads → F2 path)
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
    )

    # pg_collection: the mixer needs .tp (a process group with .size()). Try the
    # real ProcessGroupCollection, fall back to a namespace exposing the tp group.
    tp_group = parallel_state.get_tensor_model_parallel_group()
    try:
        from megatron.core.process_groups_config import ProcessGroupCollection
        pg = ProcessGroupCollection(tp=tp_group)
    except Exception as e:  # test scaffold only — surface, don't hide
        print(f"[test] ProcessGroupCollection ctor differs ({e}); using namespace shim")
        import types
        pg = types.SimpleNamespace(tp=tp_group)

    submodules = NoConvMambaMixerSubmodules(
        in_proj=ModuleSpec(module=ColumnParallelLinear),
        out_proj=ModuleSpec(module=RowParallelLinear),
    )

    mixer = Mamba3NoConvMixer(
        config=config,
        submodules=submodules,
        d_model=config.hidden_size,
        layer_number=1,
        pg_collection=pg,
    ).cuda().to(torch.bfloat16)
    mixer.train()
    print(f"[test] built mixer nheads={mixer.nheads} ngroups={mixer.ngroups} "
          f"d_state={mixer.d_state} headdim={mixer.headdim}")
    assert mixer.nheads == 8 and mixer.ngroups == 2

    # Trace the F2 point: grouped B/C -> per-head B/C.
    orig = mixer._preprocess_bc_mamba3

    def traced(B, C, *a, **k):
        print(f"[test] before F2: B={tuple(B.shape)} C={tuple(C.shape)}")
        Bo, Co = orig(B, C, *a, **k)
        print(f"[test] after  F2: B={tuple(Bo.shape)} C={tuple(Co.shape)}")
        assert Bo.shape[2] == mixer.nheads and Co.shape[2] == mixer.nheads, "F2 did not expand to per-head"
        return Bo, Co
    mixer._preprocess_bc_mamba3 = traced

    seq, batch = 32, 2
    hs = torch.randn(seq, batch, config.hidden_size, device="cuda",
                     dtype=torch.bfloat16, requires_grad=True)
    out = mixer(hs)
    if isinstance(out, tuple):
        y, bias = out
        out = y if bias is None else y + bias
    print(f"[test] out={tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    assert torch.isfinite(out).all(), "non-finite forward output"

    out.float().sum().backward()
    n_grad = sum(1 for _, p in mixer.named_parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all())
    print(f"[test] params with finite grad: {n_grad}")
    assert n_grad > 0, "no finite param grads"
    assert torch.isfinite(hs.grad).all(), "non-finite input grad"

    print("F2_CUDA_TEST_PASSED nheads=8 ngroups=2 per_head_BC_accepted_by_SSD_kernel")

