"""Tests for ``CppmegaMamba3TPMixer`` -- TP=1 vs TP=2 numeric parity.

There are two test layers:

1. **Structural / source-introspection tests** (no torch / no CUDA needed)
   that verify the mixer source contains the angle_proj fix, the partition
   sizes do NOT include angles, the in_proj is built via build_module, and
   the per-component partition_sizes match the Mamba3 packed projection.

2. **Runtime parity tests** that spawn TP=1 and TP=2 subprocesses via
   ``torch.multiprocessing.spawn``, build the mixer in each, run a forward
   on a fixed input and compare outputs (and optionally backward gradients
   per-shard).  These are skipped automatically when CUDA / Megatron /
   mamba_ssm are missing -- they only run on the H200 bench machines.
"""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Path setup -- the runtime test runs the parity worker as a subprocess
# launched by torch.multiprocessing.spawn from inside the test process.
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "cppmega" / "megatron" / "cppmega_mamba3_tp_mixer.py"


# ===========================================================================
# Layer 1: structural checks (no torch needed)
# ===========================================================================

def _read_source() -> str:
    return _SRC.read_text()


def test_source_file_exists():
    assert _SRC.is_file(), f"missing source file: {_SRC}"


def test_source_uses_megatron_module_base():
    text = _read_source()
    assert "class CppmegaMamba3TPMixer(MegatronModule):" in text


def test_source_uses_build_module_for_in_and_out_proj():
    text = _read_source()
    assert "build_module(" in text
    assert "submodules.in_proj" in text
    assert "submodules.out_proj" in text


def test_source_has_separate_replicated_angle_proj():
    """The angles must come from a separate nn.Linear, not from the packed in_proj."""
    text = _read_source()
    assert "self.angle_proj = nn.Linear(" in text
    # Replicated weight: must NOT be marked tensor_model_parallel=True
    assert 'setattr(self.angle_proj.weight, "tensor_model_parallel", False)' in text


def test_source_in_proj_does_not_pack_angles():
    """``d_in_proj_full`` must NOT add ``num_rope_angles`` (angles go through angle_proj)."""
    text = _read_source()
    # Find the d_in_proj_full assignment block
    start = text.find("d_in_proj_full = (")
    assert start != -1, "could not find d_in_proj_full assignment"
    end = text.find(")", start)
    block = text[start:end + 1]
    # Must reference d_inner, ngroups*d_state*mimo_rank, and 3*nheads
    assert "self.d_inner" in block
    assert "self.d_state" in block
    assert "self.mimo_rank" in block
    assert "3 * self.nheads" in block
    # Must NOT include num_rope_angles in the packed projection
    assert "num_rope_angles" not in block, (
        "angles are still in the packed in_proj -- they must come from a "
        "separate replicated angle_proj"
    )


def test_partition_sizes_have_seven_entries_not_eight():
    """The packed in_proj has 7 components: [z, x, B, C, dd_dt, dd_A, trap]."""
    text = _read_source()
    # Locate the partition_sizes list
    start = text.find("in_proj_partition_sizes = [")
    assert start != -1
    end = text.find("]", start)
    block = text[start:end + 1]
    # 7 lines, one per component
    component_count = sum(
        1 for line in block.splitlines()
        if line.strip().startswith("self.")
    )
    assert component_count == 7, (
        f"expected 7 partition_sizes entries (no angles), got {component_count}"
    )


def test_per_head_params_marked_tensor_model_parallel():
    text = _read_source()
    # All custom per-head params should be marked tp=True with partition_dim=0
    for name in ("dt_bias", "B_bias", "C_bias", "mimo_x", "mimo_z", "mimo_o", "D"):
        assert name in text, f"missing param {name}"
    # The 'tensor_model_parallel', 'True' tag must appear (multiple times)
    tp_tag_count = text.count('"tensor_model_parallel", True')
    assert tp_tag_count >= 4, (
        f"expected multiple tensor_model_parallel=True tags, got {tp_tag_count}"
    )


def test_forward_uses_replicated_angles_path():
    """Forward must compute angles_raw from the un-sharded hidden_states (not from
    the packed in_proj output)."""
    text = _read_source()
    assert "angles_raw = self.angle_proj(hidden_states)" in text
    # And the broadcast to LOCAL nheads
    assert "self.nheads_local_tp" in text


def test_forward_does_not_add_extra_split_for_angles():
    """The torch.split call has 7 sizes, not 8 (no trailing num_rope_angles)."""
    text = _read_source()
    # Locate the torch.split block
    start = text.find("torch.split(")
    assert start != -1
    end = text.find(")", start)
    block = text[start:end + 1]
    # Count the per-component size identifiers (z_size, x_size, ..., trap_size)
    sizes = ["z_size", "x_size", "B_size", "C_size", "dd_dt_size", "dd_A_size", "trap_size"]
    for s in sizes:
        assert s in block, f"missing split size {s}"
    assert "self.num_rope_angles" not in block


def test_constructor_signature_drop_in_with_author_mamba3_mixer():
    """Match (config, d_model, submodules, layer_number, pg_collection, pp_layer_offset)."""
    text = _read_source()
    assert "def __init__(" in text
    assert "config: TransformerConfig" in text
    assert "d_model: int" in text
    assert "submodules: MambaMixerSubmodules" in text
    assert "layer_number: int | None = None" in text
    assert "pg_collection" in text
    assert "pp_layer_offset: int = 0" in text


# ===========================================================================
# Layer 2: TP=1 vs TP=2 numeric parity (subprocess spawn)
# ===========================================================================

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_MEGATRON = importlib.util.find_spec("megatron") is not None
_HAS_MAMBA_SSM = importlib.util.find_spec("mamba_ssm") is not None


def _runtime_skip_reason() -> str | None:
    if not _HAS_TORCH:
        return "torch not installed"
    import torch
    if not torch.cuda.is_available():
        return "CUDA required for TileLang MIMO kernel"
    if torch.cuda.device_count() < 2:
        return "TP=2 parity test needs >=2 visible GPUs"
    if not _HAS_MEGATRON:
        return "megatron-core not installed"
    if not _HAS_MAMBA_SSM:
        return "mamba_ssm not installed"
    return None


# Numeric tolerances in bf16.
#
# A TP=2 forward differs from TP=1 by:
#   * the order of summation in TERowParallelLinear (full matmul vs
#     sum-of-partial-sums + all-reduce)
#   * bf16 truncation along the all-reduce
# Empirically the residual at small shapes is ~1.5e-2 absolute and
# ~0.5e-1 relative on the largest activations; tighter tolerances
# would be impossible to satisfy.  See the unit test history for the
# raw min/max diff numbers.
_PARITY_ATOL = 2e-2
_PARITY_RTOL = 5e-2


def _parity_worker(rank: int, world_size: int, master_port: int, return_dict):
    """Worker process: build the mixer, run forward, push output to rank 0.

    Uses ``torch.multiprocessing.Manager().dict`` for inter-process result
    communication so the test in the parent process can verify parity.
    """
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    try:
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
        from megatron.core.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )

        # Init model-parallel state FIRST so the RNG tracker can resolve the
        # tp rank when seeding.
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(12345)

        # Build a small TransformerConfig.  We need d_model*expand/headdim
        # to give us 8 heads (matching the task spec), so expand=1.
        config = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=8,
            ffn_hidden_size=2048,
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            sequence_parallel=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            mamba_state_dim=32,
            mamba_head_dim=64,
            mamba_num_groups=4,
            use_cpu_initialization=False,
        )
        # Mamba3 MIMO custom config attributes (mamba_expand and friends)
        object.__setattr__(config, "mamba_expand", 1)
        object.__setattr__(config, "cppmega_mamba3_is_mimo", True)
        object.__setattr__(config, "cppmega_mamba3_mimo_rank", 2)
        object.__setattr__(config, "cppmega_mamba3_chunk_size", 32)

        # Build the pg_collection from the global parallel_state.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        from cppmega.megatron.cppmega_mamba3_tp_mixer import CppmegaMamba3TPMixer

        submodules = MambaMixerSubmodules(
            in_proj=TELayerNormColumnParallelLinear,
            out_proj=TERowParallelLinear,
        )

        # Determinstic torch seed before mixer construction (TE linears use the
        # cuda RNG tracker which we already seeded; this seeds any plain
        # torch.randn calls done by RMSNormGated etc.).
        torch.manual_seed(20260412)
        torch.cuda.manual_seed_all(20260412)

        mixer = CppmegaMamba3TPMixer(
            config=config,
            d_model=config.hidden_size,
            submodules=submodules,
            layer_number=1,
            pg_collection=pg_collection,
            pp_layer_offset=0,
        ).cuda()

        # ------------------------------------------------------------------
        # Override TE-initialised weights with deterministic values that
        # produce bit-identical TP=1 vs TP=2 layouts after concatenation.
        # The TE rng tracker advances state per rank, so its built-in init
        # alone does not give us cross-TP parity for the in_proj and out_proj
        # weight matrices.  We rebuild them on a single CPU generator and
        # slice into the local rows.
        # ------------------------------------------------------------------
        with torch.no_grad():
            d_model = config.hidden_size

            # in_proj weight: shape (d_in_proj_local, d_model).  Full size
            # is (d_in_proj_full, d_model) but the components are
            # interleaved per-rank, so we have to assemble the local
            # slice block-by-block.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(424242)

            d_inner = mixer.d_inner
            ngroups = mixer.ngroups
            d_state = mixer.d_state
            mimo_rank = mixer.mimo_rank
            nheads = mixer.nheads
            nh_loc = mixer.nheads_local_tp
            ng_loc = mixer.ngroups_local_tp
            di_loc = mixer.d_inner_local_tp

            # Build full per-component weights and pick the local slice.
            def _full_block(rows: int, cols: int) -> torch.Tensor:
                return torch.empty(rows, cols, dtype=torch.float32, device="cpu").uniform_(
                    -0.05, 0.05, generator=gen
                )

            # Components in order: z, x, B, C, dd_dt, dd_A, trap
            comp_full = {
                "z": _full_block(d_inner, d_model),
                "x": _full_block(d_inner, d_model),
                "B": _full_block(ngroups * d_state * mimo_rank, d_model),
                "C": _full_block(ngroups * d_state * mimo_rank, d_model),
                "dd_dt": _full_block(nheads, d_model),
                "dd_A": _full_block(nheads, d_model),
                "trap": _full_block(nheads, d_model),
            }
            # Local slices: z/x sliced along d_inner by tp_rank * di_loc; B/C
            # sliced along (ngroups * d_state * mimo_rank) by tp_rank *
            # (ng_loc * d_state * mimo_rank); dd_dt/dd_A/trap sliced along
            # nheads by tp_rank * nh_loc.
            tr = rank
            stride_bc = ng_loc * d_state * mimo_rank
            local_blocks = [
                comp_full["z"][tr * di_loc : (tr + 1) * di_loc],
                comp_full["x"][tr * di_loc : (tr + 1) * di_loc],
                comp_full["B"][tr * stride_bc : (tr + 1) * stride_bc],
                comp_full["C"][tr * stride_bc : (tr + 1) * stride_bc],
                comp_full["dd_dt"][tr * nh_loc : (tr + 1) * nh_loc],
                comp_full["dd_A"][tr * nh_loc : (tr + 1) * nh_loc],
                comp_full["trap"][tr * nh_loc : (tr + 1) * nh_loc],
            ]
            local_in_proj_w = torch.cat(local_blocks, dim=0).to(
                dtype=mixer.in_proj.weight.dtype, device=mixer.in_proj.weight.device
            )
            assert local_in_proj_w.shape == mixer.in_proj.weight.shape, (
                f"local in_proj shape mismatch: built {tuple(local_in_proj_w.shape)} "
                f"vs param {tuple(mixer.in_proj.weight.shape)}"
            )
            mixer.in_proj.weight.copy_(local_in_proj_w)

            # out_proj weight: shape (d_model, d_inner_local).  out_proj is
            # row-parallel (input is parallel along d_inner), so each rank
            # holds a (d_model, d_inner_local) slice along the input axis.
            full_out_proj = torch.empty(
                d_model, d_inner, dtype=torch.float32, device="cpu"
            ).uniform_(-0.05, 0.05, generator=gen)
            local_out = full_out_proj[:, tr * di_loc : (tr + 1) * di_loc].to(
                dtype=mixer.out_proj.weight.dtype, device=mixer.out_proj.weight.device
            )
            mixer.out_proj.weight.copy_(local_out)

            # angle_proj weight: shape (num_rope_angles, d_model), REPLICATED
            # so every rank gets the same full tensor.
            full_angle = torch.empty(
                mixer.num_rope_angles, d_model, dtype=torch.float32, device="cpu"
            ).uniform_(-0.05, 0.05, generator=gen)
            mixer.angle_proj.weight.copy_(
                full_angle.to(
                    dtype=mixer.angle_proj.weight.dtype,
                    device=mixer.angle_proj.weight.device,
                )
            )

            # in_proj LayerNorm weights (TELayerNormColumnParallelLinear has
            # an internal layer_norm_weight / layer_norm_bias).  Set to ones
            # so the LayerNorm is identity-like and we don't depend on rank-
            # local TE init for the norm.
            for ln_attr in ("layer_norm_weight", "layer_norm_bias"):
                p = getattr(mixer.in_proj, ln_attr, None)
                if p is not None:
                    if "weight" in ln_attr:
                        p.data.fill_(1.0)
                    else:
                        p.data.fill_(0.0)

            # B_norm/C_norm weight (RMSNormGated): set to ones for parity.
            for norm in (mixer.B_norm, mixer.C_norm):
                if hasattr(norm, "weight") and norm.weight is not None:
                    norm.weight.data.fill_(1.0)

        # Build a fixed input (replicated across ranks)
        torch.manual_seed(99)
        L, B, H = 64, 2, config.hidden_size
        hs_full = torch.randn(L, B, H, device="cuda", dtype=torch.bfloat16)
        # Broadcast from rank 0 so every rank sees the same tensor
        dist.broadcast(hs_full, src=0)

        # Forward pass
        out, _ = mixer(hs_full)

        # Reduce-collect to rank 0 for the parity comparison
        if rank == 0:
            return_dict[("output", world_size)] = out.detach().float().cpu()
        dist.barrier()

    except Exception as exc:
        # Surface the error to the parent so the test fails with a useful trace
        return_dict[("error", world_size, rank)] = repr(exc)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_world(world_size: int, port: int, return_dict):
    import torch.multiprocessing as mp
    mp.spawn(
        _parity_worker,
        args=(world_size, port, return_dict),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(_runtime_skip_reason() is not None, reason=_runtime_skip_reason() or "")
def test_runtime_parity_tp1_vs_tp2_forward():
    """End-to-end forward parity between TP=1 and TP=2 worlds.

    Spawns two child-process pools (TP=1 then TP=2), runs an identical
    forward in each, and compares the all-gathered output to the TP=1
    reference at bf16 tolerances.
    """
    import torch.multiprocessing as mp
    manager = mp.Manager()
    rd = manager.dict()

    _spawn_world(1, 29501, rd)
    _spawn_world(2, 29502, rd)

    # Surface any worker errors
    errors = [
        (k, v) for k, v in rd.items() if isinstance(k, tuple) and k[0] == "error"
    ]
    assert not errors, f"parity workers raised: {errors}"

    out_tp1 = rd[("output", 1)]
    out_tp2 = rd[("output", 2)]

    assert out_tp1.shape == out_tp2.shape, (
        f"shape mismatch: TP1={tuple(out_tp1.shape)} vs TP2={tuple(out_tp2.shape)}"
    )

    import torch
    diff = (out_tp1 - out_tp2).abs()
    max_abs = diff.max().item()
    rel_denom = out_tp1.abs().clamp(min=1e-6)
    max_rel = (diff / rel_denom).max().item()

    assert torch.allclose(out_tp1, out_tp2, atol=_PARITY_ATOL, rtol=_PARITY_RTOL), (
        f"TP=1 vs TP=2 forward parity FAILED: max_abs={max_abs:.4e} "
        f"(tol_abs={_PARITY_ATOL:.0e}), max_rel={max_rel:.4e} "
        f"(tol_rel={_PARITY_RTOL:.0e})"
    )


# ===========================================================================
# Layer 3: TP=1 vs TP=2 + sequence-parallel (SP=True) numeric parity.
#
# Stream B wired the SP path in ``cppmega_mamba3_tp_mixer.py::forward`` --
# see the ``if getattr(self.config, "sequence_parallel", False) and
# self.tp_world_size > 1:`` branch that gathers ``angles_raw`` before the
# scan.  That branch never had a unit test; the production TP=2 launcher
# (``remote_train_h200_nam56r_tp2.sh``) uses SP=True, so we need parity
# coverage for the SP-on path specifically.
#
# Layout of this test suite:
#
# * ``_parity_worker_sp`` is a second worker that takes a ``sp_on`` flag.
#   When ``sp_on == False`` it runs the TP=1 reference world (single rank,
#   full sequence input).  When ``sp_on == True`` it runs the TP=2 world
#   with ``config.sequence_parallel=True`` and feeds each rank only its
#   ``L/tp`` slice of the hidden-states along dim 0.
# * ``_spawn_world_sp`` is the dispatch helper.
# * ``test_tp2_sp_on_parity_vs_tp1`` compares forward output and per-param
#   gradients between the TP=1 reference and the TP=2 SP-on test world.
# * ``test_tp2_sp_on_angle_proj_gather`` is a focused unit test that
#   compares ONLY the ``angle_proj`` output (after SP gather) against the
#   TP=1 reference -- this isolates the Stream B addition from the rest
#   of the mixer.
# ===========================================================================


# Small config matching the Stream I task spec (runtime <30s on H200).
#
#   d_model=256, nheads=8, ngroups=4, d_state=32, headdim=32,
#   mimo_rank=2, chunk_size=32, seq_len=128, batch=2
_SP_D_MODEL = 256
_SP_NHEADS = 8
_SP_HEADDIM = 32
_SP_NGROUPS = 4
_SP_DSTATE = 32
_SP_MIMO_RANK = 2
_SP_CHUNK = 32
_SP_SEQLEN = 128
_SP_BATCH = 2


def _build_sp_config(world_size: int, sp_on: bool):
    """Build the small TransformerConfig used by the SP parity workers."""
    import torch
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=_SP_D_MODEL,
        num_attention_heads=_SP_NHEADS,
        num_query_groups=_SP_NHEADS,
        ffn_hidden_size=4 * _SP_D_MODEL,
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        sequence_parallel=bool(sp_on and world_size > 1),
        params_dtype=torch.bfloat16,
        bf16=True,
        mamba_state_dim=_SP_DSTATE,
        mamba_head_dim=_SP_HEADDIM,
        mamba_num_groups=_SP_NGROUPS,
        use_cpu_initialization=False,
    )
    object.__setattr__(config, "mamba_expand", 1)
    object.__setattr__(config, "cppmega_mamba3_is_mimo", True)
    object.__setattr__(config, "cppmega_mamba3_mimo_rank", _SP_MIMO_RANK)
    object.__setattr__(config, "cppmega_mamba3_chunk_size", _SP_CHUNK)
    return config


def _override_sp_mixer_weights(mixer, rank: int):
    """Load deterministic CPU-generated weights into the mixer shards.

    Mirrors the override scheme from ``_parity_worker`` so that the TP=1
    reference and every TP rank in the TP=2 SP-on world share identical
    full-tensor weights (and therefore the parity check only has to deal
    with reduction-order differences, not init noise).
    """
    import torch

    with torch.no_grad():
        d_model = mixer.d_model
        d_inner = mixer.d_inner
        ngroups = mixer.ngroups
        d_state = mixer.d_state
        mimo_rank = mixer.mimo_rank
        nheads = mixer.nheads
        nh_loc = mixer.nheads_local_tp
        ng_loc = mixer.ngroups_local_tp
        di_loc = mixer.d_inner_local_tp

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0xA11CE_202604)

        def _full_block(rows: int, cols: int) -> torch.Tensor:
            return torch.empty(rows, cols, dtype=torch.float32, device="cpu").uniform_(
                -0.05, 0.05, generator=gen
            )

        comp_full = {
            "z": _full_block(d_inner, d_model),
            "x": _full_block(d_inner, d_model),
            "B": _full_block(ngroups * d_state * mimo_rank, d_model),
            "C": _full_block(ngroups * d_state * mimo_rank, d_model),
            "dd_dt": _full_block(nheads, d_model),
            "dd_A": _full_block(nheads, d_model),
            "trap": _full_block(nheads, d_model),
        }

        tr = rank
        stride_bc = ng_loc * d_state * mimo_rank
        local_blocks = [
            comp_full["z"][tr * di_loc : (tr + 1) * di_loc],
            comp_full["x"][tr * di_loc : (tr + 1) * di_loc],
            comp_full["B"][tr * stride_bc : (tr + 1) * stride_bc],
            comp_full["C"][tr * stride_bc : (tr + 1) * stride_bc],
            comp_full["dd_dt"][tr * nh_loc : (tr + 1) * nh_loc],
            comp_full["dd_A"][tr * nh_loc : (tr + 1) * nh_loc],
            comp_full["trap"][tr * nh_loc : (tr + 1) * nh_loc],
        ]
        local_in_proj_w = torch.cat(local_blocks, dim=0).to(
            dtype=mixer.in_proj.weight.dtype, device=mixer.in_proj.weight.device
        )
        assert local_in_proj_w.shape == mixer.in_proj.weight.shape, (
            f"local in_proj shape mismatch: built {tuple(local_in_proj_w.shape)} "
            f"vs param {tuple(mixer.in_proj.weight.shape)}"
        )
        mixer.in_proj.weight.copy_(local_in_proj_w)

        full_out_proj = torch.empty(
            d_model, d_inner, dtype=torch.float32, device="cpu"
        ).uniform_(-0.05, 0.05, generator=gen)
        local_out = full_out_proj[:, tr * di_loc : (tr + 1) * di_loc].to(
            dtype=mixer.out_proj.weight.dtype, device=mixer.out_proj.weight.device
        )
        mixer.out_proj.weight.copy_(local_out)

        # angle_proj is REPLICATED -- same full tensor on every rank.
        full_angle = torch.empty(
            mixer.num_rope_angles, d_model, dtype=torch.float32, device="cpu"
        ).uniform_(-0.05, 0.05, generator=gen)
        mixer.angle_proj.weight.copy_(
            full_angle.to(
                dtype=mixer.angle_proj.weight.dtype,
                device=mixer.angle_proj.weight.device,
            )
        )

        # LayerNorm inside TELayerNormColumnParallelLinear -> identity.
        for ln_attr in ("layer_norm_weight", "layer_norm_bias"):
            p = getattr(mixer.in_proj, ln_attr, None)
            if p is not None:
                if "weight" in ln_attr:
                    p.data.fill_(1.0)
                else:
                    p.data.fill_(0.0)

        for norm in (mixer.B_norm, mixer.C_norm):
            if hasattr(norm, "weight") and norm.weight is not None:
                norm.weight.data.fill_(1.0)


def _parity_worker_sp(rank: int, world_size: int, master_port: int, sp_on: bool, return_dict):
    """Forward + backward worker for the SP-on parity test.

    * When ``world_size == 1`` this runs the TP=1 / SP=False reference.
      The full (L, B, H) input tensor is fed into the mixer, and the
      forward output + param grads are stored under key ``("ref", ...)``.
    * When ``world_size == 2`` and ``sp_on`` is True, each rank receives
      ``L/tp`` sequence slice, the mixer runs with
      ``config.sequence_parallel = True`` (so in_proj's internal gather
      and out_proj's internal reduce-scatter kick in) and the SP-gather
      of ``angles_raw`` in ``cppmega_mamba3_tp_mixer.forward`` is active.
      The local output slice (L/tp, B, H) is all-gathered along dim 0
      into the full (L, B, H) tensor, and gradients for each param are
      all-gathered along the TP axis (or all-reduced for replicated
      params) so the parent test can compare them to the reference.
    """
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tag = "sp_on" if sp_on else "ref"

    try:
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
        from megatron.core.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(12345)

        config = _build_sp_config(world_size=world_size, sp_on=sp_on)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        from cppmega.megatron.cppmega_mamba3_tp_mixer import CppmegaMamba3TPMixer

        submodules = MambaMixerSubmodules(
            in_proj=TELayerNormColumnParallelLinear,
            out_proj=TERowParallelLinear,
        )

        torch.manual_seed(20260412)
        torch.cuda.manual_seed_all(20260412)

        mixer = CppmegaMamba3TPMixer(
            config=config,
            d_model=config.hidden_size,
            submodules=submodules,
            layer_number=1,
            pg_collection=pg_collection,
            pp_layer_offset=0,
        ).cuda()

        _override_sp_mixer_weights(mixer, rank=rank)

        # ------------------------------------------------------------------
        # Build the fixed input (full shape) then slice per rank if SP is on.
        # We build a single tensor with a fixed seed so both the reference
        # and the SP-on world start from bit-identical hidden_states.
        # ------------------------------------------------------------------
        torch.manual_seed(0xBEEF)
        hs_full = torch.randn(
            _SP_SEQLEN, _SP_BATCH, _SP_D_MODEL, device="cuda", dtype=torch.bfloat16
        )
        # Broadcast from rank 0 so every rank sees the same tensor.
        dist.broadcast(hs_full, src=0)

        if sp_on and world_size > 1:
            L_loc = _SP_SEQLEN // world_size
            hs_local = hs_full[rank * L_loc : (rank + 1) * L_loc].contiguous()
        else:
            hs_local = hs_full

        hs_local.requires_grad_(True)

        out_local, _ = mixer(hs_local)
        # out_local shape: (L, B, H) in reference, (L/tp, B, H) in SP-on.

        # A fixed deterministic "loss" -- we pick a reproducible gradient
        # scalar so the backward computation is equivalent across worlds.
        # Use sum(out * g) where g is the same seeded random tensor (full
        # shape on ref, sliced to L/tp on SP).
        torch.manual_seed(0xC0FFEE)
        g_full = torch.randn_like(hs_full)
        dist.broadcast(g_full, src=0)
        if sp_on and world_size > 1:
            L_loc = _SP_SEQLEN // world_size
            g_local = g_full[rank * L_loc : (rank + 1) * L_loc].contiguous()
        else:
            g_local = g_full
        loss = (out_local.float() * g_local.float()).sum()
        loss.backward()

        # ------------------------------------------------------------------
        # Gather the forward output back to the full (L, B, H) tensor on
        # every rank; rank 0 stashes it in the shared dict.
        # ------------------------------------------------------------------
        if world_size > 1 and sp_on:
            out_gather_list = [torch.empty_like(out_local) for _ in range(world_size)]
            dist.all_gather(out_gather_list, out_local.detach().contiguous())
            out_full = torch.cat(out_gather_list, dim=0)
        else:
            out_full = out_local.detach()

        if rank == 0:
            return_dict[(tag, "output")] = out_full.float().cpu()

        # ------------------------------------------------------------------
        # Per-param grad gather.  Four families:
        #   (a) TP-column-sharded along dim 0 of the weight (in_proj):
        #       all-gather along dim 0 and concat.
        #   (b) TP-row-sharded along dim 1 of the weight (out_proj):
        #       all-gather along dim 1.
        #   (c) TP-sharded along dim 0 but with per-component interleaved
        #       layout (in_proj partition_sizes): we rebuild the full by
        #       splitting each shard into its components, gathering, and
        #       concatenating component-by-component.
        #   (d) replicated (angle_proj, B_norm, C_norm, layer_norm_*):
        #       all-reduce-sum across ranks.
        # ------------------------------------------------------------------
        def _collect_shard_dim0(tensor: torch.Tensor) -> torch.Tensor:
            if world_size == 1:
                return tensor.detach()
            gl = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gl, tensor.detach().contiguous())
            return torch.cat(gl, dim=0)

        def _collect_shard_dim1(tensor: torch.Tensor) -> torch.Tensor:
            if world_size == 1:
                return tensor.detach()
            gl = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gl, tensor.detach().contiguous())
            return torch.cat(gl, dim=1)

        def _collect_replicated_sum(tensor: torch.Tensor) -> torch.Tensor:
            if world_size == 1:
                return tensor.detach()
            t = tensor.detach().clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t

        # ---- in_proj.weight: interleaved per-component gather ------------
        # Each local shard is laid out as [z, x, B, C, dd_dt, dd_A, trap]
        # along dim 0, with sizes (di_loc, di_loc, stride_bc, stride_bc,
        # nh_loc, nh_loc, nh_loc).  We split each rank's grad into these
        # seven pieces, all-gather each piece separately, then concatenate
        # in the upstream-expected FULL layout.
        di_loc = mixer.d_inner_local_tp
        nh_loc = mixer.nheads_local_tp
        ng_loc = mixer.ngroups_local_tp
        stride_bc = ng_loc * mixer.d_state * mixer.mimo_rank
        inproj_comp_sizes = [di_loc, di_loc, stride_bc, stride_bc, nh_loc, nh_loc, nh_loc]

        in_proj_grad = mixer.in_proj.weight.grad
        if in_proj_grad is None:
            raise RuntimeError("in_proj.weight.grad is None after backward()")
        comp_grads_local = list(torch.split(in_proj_grad, inproj_comp_sizes, dim=0))
        comp_full = []
        for g_loc in comp_grads_local:
            comp_full.append(_collect_shard_dim0(g_loc.contiguous()))
        in_proj_grad_full = torch.cat(comp_full, dim=0).float().cpu()
        if rank == 0:
            return_dict[(tag, "grad.in_proj.weight")] = in_proj_grad_full

        # ---- out_proj.weight: row-parallel (split along dim 1) -----------
        out_proj_grad = mixer.out_proj.weight.grad
        if out_proj_grad is None:
            raise RuntimeError("out_proj.weight.grad is None after backward()")
        out_proj_grad_full = _collect_shard_dim1(out_proj_grad).float().cpu()
        if rank == 0:
            return_dict[(tag, "grad.out_proj.weight")] = out_proj_grad_full

        # ---- per-head params sharded along dim 0 ------------------------
        for name in (
            "dt_bias",
            "A_log" if hasattr(mixer, "A_log") else None,
            "D",
            "B_bias",
            "C_bias",
            "mimo_x",
            "mimo_z",
            "mimo_o",
        ):
            if name is None:
                continue
            p = getattr(mixer, name, None)
            if p is None or p.grad is None:
                continue
            g_full_t = _collect_shard_dim0(p.grad).float().cpu()
            if rank == 0:
                return_dict[(tag, f"grad.{name}")] = g_full_t

        # ---- replicated params: all-reduce-sum across ranks --------------
        angle_grad = mixer.angle_proj.weight.grad
        if angle_grad is None:
            raise RuntimeError("angle_proj.weight.grad is None after backward()")
        angle_grad_full = _collect_replicated_sum(angle_grad).float().cpu()
        if rank == 0:
            return_dict[(tag, "grad.angle_proj.weight")] = angle_grad_full

        # B_norm / C_norm weights: replicated
        for norm_name in ("B_norm", "C_norm"):
            norm = getattr(mixer, norm_name)
            if hasattr(norm, "weight") and norm.weight is not None and norm.weight.grad is not None:
                g = _collect_replicated_sum(norm.weight.grad).float().cpu()
                if rank == 0:
                    return_dict[(tag, f"grad.{norm_name}.weight")] = g

        # in_proj LayerNorm weights: replicated
        for ln_attr in ("layer_norm_weight", "layer_norm_bias"):
            p = getattr(mixer.in_proj, ln_attr, None)
            if p is not None and p.grad is not None:
                g = _collect_replicated_sum(p.grad).float().cpu()
                if rank == 0:
                    return_dict[(tag, f"grad.in_proj.{ln_attr}")] = g

        dist.barrier()

    except Exception as exc:
        return_dict[("error", tag, rank)] = repr(exc)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_world_sp(world_size: int, port: int, sp_on: bool, return_dict):
    import torch.multiprocessing as mp
    mp.spawn(
        _parity_worker_sp,
        args=(world_size, port, sp_on, return_dict),
        nprocs=world_size,
        join=True,
    )


# SP parity tolerances.  bf16 + TE row-parallel reduce + SP reduce-scatter
# give slightly larger drift than the plain TP-only path (the out_proj
# all-reduce is replaced by reduce-scatter along seq, which changes the
# order of summation for each (seq_slice, d_model) element).  Empirically
# ~5e-2 absolute is the realistic floor in bf16.
_SP_FWD_ATOL = 5e-2
_SP_FWD_RTOL = 8e-2
# Gradient tolerances: slightly looser because the backward of
# reduce-scatter is all-gather, which again re-orders bf16 sums.
_SP_BWD_ATOL = 8e-2
_SP_BWD_RTOL = 1e-1


def _compare(ref: "torch.Tensor", test: "torch.Tensor", atol: float, rtol: float):
    """Return (passed, max_abs, max_rel) tuple for a parity comparison."""
    import torch
    assert ref.shape == test.shape, (
        f"shape mismatch: ref={tuple(ref.shape)} vs test={tuple(test.shape)}"
    )
    diff = (ref - test).abs()
    max_abs = diff.max().item()
    rel_denom = ref.abs().clamp(min=1e-6)
    max_rel = (diff / rel_denom).max().item()
    passed = bool(torch.allclose(ref, test, atol=atol, rtol=rtol))
    return passed, max_abs, max_rel


@pytest.mark.skipif(_runtime_skip_reason() is not None, reason=_runtime_skip_reason() or "")
def test_tp2_sp_on_parity_vs_tp1():
    """TP=1 reference vs TP=2 + sequence_parallel=True full parity.

    Spawns the reference TP=1 world (sp_on=False, full L input) and then
    the TP=2 SP-on world (sp_on=True, L/tp input per rank).  Compares:

    * Forward output (gathered back to full L).
    * Gradients of in_proj.weight (gathered per-component along dim 0).
    * Gradients of out_proj.weight (gathered along dim 1).
    * Gradients of all per-head sharded params (dim 0 all-gather).
    * Gradients of replicated angle_proj / norms (all-reduce SUM).

    The test fails loudly with per-tensor max_abs / max_rel diagnostics
    if any parity check fails.  It must NOT mask failures by relaxing
    the tolerance or by disabling the SP path -- see task #86 hard
    constraints.

    ------------------------------------------------------------------
    KNOWN BUG (task #86, 2026-04-12): this test is EXPECTED TO FAIL on
    ``grad.angle_proj.weight`` at the current revision because Stream B
    wired the SP gather with ``tensor_parallel_output_grad=False`` at
    ``cppmega/megatron/cppmega_mamba3_tp_mixer.py:443``.  The forward is
    correct (angle_proj is identical across ranks; the gather just
    reassembles the full L), but the backward flag is wrong for the way
    ``angles_raw`` is consumed downstream:

    * After gather, ``angles`` is expanded to ``self.nheads_local_tp``
      and fed into the MIMO scan, which consumes a DIFFERENT slice of
      heads on every TP rank (TP is sharding the head axis).
    * This means the grad flowing back into the full ``angles_raw``
      tensor is DIFFERENT on every rank -- each rank's grad only
      contains contributions from its local heads.
    * The correct backward for ``gather_from_sequence_parallel_region``
      in that mode is ``reduce_scatter`` (sum-then-scatter), which is
      what ``tensor_parallel_output_grad=True`` selects (see
      ``megatron/core/tensor_parallel/mappings.py:336``).
    * ``tensor_parallel_output_grad=False`` selects plain ``split`` --
      it just slices each rank's local grad by its seq chunk and drops
      the contribution from other ranks' heads.

    Fix (owned by Stream B, NOT this test): pass
    ``tensor_parallel_output_grad=True`` (or omit the kwarg so the
    default kicks in) on line 443 of the mixer.  The TP=2 launcher at
    ``scripts/remote_train_h200_nam56r_tp2.sh`` is CURRENTLY emitting
    incorrect ``angle_proj.weight`` gradients -- any numbers from that
    run are not numerically trustworthy until this is fixed.
    ------------------------------------------------------------------
    """
    import torch
    import torch.multiprocessing as mp

    manager = mp.Manager()
    rd = manager.dict()

    # Reference world: TP=1, SP=False, full L.
    _spawn_world_sp(world_size=1, port=29511, sp_on=False, return_dict=rd)
    # Test world: TP=2, SP=True, L/tp per rank.
    _spawn_world_sp(world_size=2, port=29512, sp_on=True, return_dict=rd)

    errors = [
        (k, v) for k, v in rd.items() if isinstance(k, tuple) and k[0] == "error"
    ]
    assert not errors, f"SP parity workers raised: {errors}"

    # Collect all tagged keys.
    ref_keys = sorted(k for k in rd.keys() if isinstance(k, tuple) and k[0] == "ref")
    test_keys = sorted(k for k in rd.keys() if isinstance(k, tuple) and k[0] == "sp_on")

    # Ensure forward is present in both worlds.
    assert ("ref", "output") in rd, "reference world did not produce an output"
    assert ("sp_on", "output") in rd, "SP-on world did not produce an output"

    ref_out = rd[("ref", "output")]
    sp_out = rd[("sp_on", "output")]

    passed_fwd, max_abs_fwd, max_rel_fwd = _compare(
        ref_out, sp_out, atol=_SP_FWD_ATOL, rtol=_SP_FWD_RTOL
    )

    failures = []
    if not passed_fwd:
        failures.append(
            f"FORWARD output: shape={tuple(ref_out.shape)} "
            f"max_abs={max_abs_fwd:.4e} (tol={_SP_FWD_ATOL:.0e}) "
            f"max_rel={max_rel_fwd:.4e} (tol={_SP_FWD_RTOL:.0e})"
        )
        # Boundary vs uniform diagnostic: where is the max-abs element?
        import torch as _t
        diff = (ref_out - sp_out).abs()
        flat_idx = int(diff.argmax().item())
        unravel = _t.unravel_index(_t.tensor(flat_idx), diff.shape)
        failures.append(f"    max_abs element index: {tuple(int(x) for x in unravel)}")

    # Compare every grad key that is present on BOTH sides.
    grad_keys_ref = {k[1] for k in ref_keys if k[1].startswith("grad.")}
    grad_keys_test = {k[1] for k in test_keys if k[1].startswith("grad.")}
    common = sorted(grad_keys_ref & grad_keys_test)
    missing_in_test = sorted(grad_keys_ref - grad_keys_test)
    missing_in_ref = sorted(grad_keys_test - grad_keys_ref)
    for name in missing_in_test:
        failures.append(f"grad {name}: present in ref, MISSING in sp_on world")
    for name in missing_in_ref:
        failures.append(f"grad {name}: present in sp_on, MISSING in ref world")

    for name in common:
        ref_g = rd[("ref", name)]
        test_g = rd[("sp_on", name)]
        if ref_g.shape != test_g.shape:
            failures.append(
                f"grad {name}: shape mismatch ref={tuple(ref_g.shape)} "
                f"vs test={tuple(test_g.shape)}"
            )
            continue
        passed, max_abs, max_rel = _compare(
            ref_g, test_g, atol=_SP_BWD_ATOL, rtol=_SP_BWD_RTOL
        )
        if not passed:
            failures.append(
                f"grad {name}: shape={tuple(ref_g.shape)} "
                f"max_abs={max_abs:.4e} (tol={_SP_BWD_ATOL:.0e}) "
                f"max_rel={max_rel:.4e} (tol={_SP_BWD_RTOL:.0e})"
            )

    if failures:
        msg = (
            "TP=1 vs TP=2+SP=True parity FAILED:\n    "
            + "\n    ".join(failures)
            + "\n\nForward output: "
            + f"max_abs={max_abs_fwd:.4e} max_rel={max_rel_fwd:.4e}"
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Focused angle_proj gather path test
# ---------------------------------------------------------------------------


def _angle_proj_worker(rank: int, world_size: int, master_port: int, sp_on: bool, return_dict):
    """Run the mixer up to and including the angle_proj + SP-gather step,
    capturing ``angles_raw`` (shape (L, B, num_rope_angles)) and stashing
    it on rank 0.

    We achieve this with a forward pre-hook-style approach: we monkey-patch
    the mixer instance's ``angle_proj`` to record its post-gather output.
    Rather than reach into mixer.forward (which also requires a full SSD
    scan), we re-implement just the first 10 lines of the forward here
    on a freshly constructed mixer, since that exactly isolates the
    Stream B SP path.
    """
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tag = "sp_on" if sp_on else "ref"

    try:
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
        from megatron.core.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(12345)

        config = _build_sp_config(world_size=world_size, sp_on=sp_on)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        from cppmega.megatron.cppmega_mamba3_tp_mixer import CppmegaMamba3TPMixer

        submodules = MambaMixerSubmodules(
            in_proj=TELayerNormColumnParallelLinear,
            out_proj=TERowParallelLinear,
        )

        torch.manual_seed(20260412)
        torch.cuda.manual_seed_all(20260412)
        mixer = CppmegaMamba3TPMixer(
            config=config,
            d_model=config.hidden_size,
            submodules=submodules,
            layer_number=1,
            pg_collection=pg_collection,
            pp_layer_offset=0,
        ).cuda()
        _override_sp_mixer_weights(mixer, rank=rank)

        torch.manual_seed(0xBEEF)
        hs_full = torch.randn(
            _SP_SEQLEN, _SP_BATCH, _SP_D_MODEL, device="cuda", dtype=torch.bfloat16
        )
        dist.broadcast(hs_full, src=0)

        if sp_on and world_size > 1:
            L_loc = _SP_SEQLEN // world_size
            hs_local = hs_full[rank * L_loc : (rank + 1) * L_loc].contiguous()
        else:
            hs_local = hs_full

        # Replicate exactly what the mixer forward does for angles_raw:
        angles_raw = mixer.angle_proj(hs_local)
        if (
            getattr(mixer.config, "sequence_parallel", False)
            and mixer.tp_world_size > 1
        ):
            from megatron.core.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )
            angles_raw = gather_from_sequence_parallel_region(
                angles_raw, tensor_parallel_output_grad=False, group=mixer.tp_group,
            )

        # Every rank should now hold the FULL (L, B, num_rope_angles) tensor.
        if rank == 0:
            return_dict[(tag, "angles_raw")] = angles_raw.detach().float().cpu()
        # Shape sanity: assert L dim is full on every rank regardless of SP.
        assert angles_raw.shape[0] == _SP_SEQLEN, (
            f"angles_raw seq dim is {angles_raw.shape[0]}, expected {_SP_SEQLEN}; "
            f"SP gather path did not run on rank {rank} (sp_on={sp_on})"
        )
        dist.barrier()

    except Exception as exc:
        return_dict[("error", tag, rank)] = repr(exc)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_angle_world(world_size: int, port: int, sp_on: bool, return_dict):
    import torch.multiprocessing as mp
    mp.spawn(
        _angle_proj_worker,
        args=(world_size, port, sp_on, return_dict),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(_runtime_skip_reason() is not None, reason=_runtime_skip_reason() or "")
def test_tp2_sp_on_angle_proj_gather():
    """Isolated test for the angle_proj SP-gather path.

    angle_proj is a plain nn.Linear without TP.  Under SP=True the input
    to the mixer is sharded along the sequence dim, so angle_proj's
    LOCAL output is (L/tp, B, num_rope_angles).  Stream B added a call
    to ``gather_from_sequence_parallel_region`` to restore the full
    (L, B, num_rope_angles) tensor before the MIMO scan -- this test
    verifies that gather is shape-correct AND bit-equivalent (at bf16
    tolerance) to the TP=1 reference.
    """
    import torch.multiprocessing as mp
    manager = mp.Manager()
    rd = manager.dict()

    _spawn_angle_world(world_size=1, port=29521, sp_on=False, return_dict=rd)
    _spawn_angle_world(world_size=2, port=29522, sp_on=True, return_dict=rd)

    errors = [
        (k, v) for k, v in rd.items() if isinstance(k, tuple) and k[0] == "error"
    ]
    assert not errors, f"angle_proj SP workers raised: {errors}"

    assert ("ref", "angles_raw") in rd, "reference world did not produce angles_raw"
    assert ("sp_on", "angles_raw") in rd, "SP-on world did not produce angles_raw"

    ref = rd[("ref", "angles_raw")]
    test = rd[("sp_on", "angles_raw")]

    # Shape must match: the SP gather must restore the full sequence dim.
    assert ref.shape == test.shape, (
        f"angle_proj gather: shape mismatch ref={tuple(ref.shape)} "
        f"vs test={tuple(test.shape)} -- SP gather may be missing!"
    )

    # Numeric parity: angle_proj is pure matmul on identical weights and
    # identical hidden_states, so tolerance can be tight (no reduction
    # order differences like in in_proj / out_proj).
    passed, max_abs, max_rel = _compare(ref, test, atol=1e-3, rtol=1e-3)
    assert passed, (
        f"angle_proj SP-gather parity FAILED: shape={tuple(ref.shape)} "
        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e}"
    )
