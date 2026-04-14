"""Reproducer: MambaModel.output_layer is plain ColumnParallelLinear,
GPTModel.output_layer is LinearCrossEntropyModule — parity regression.

Context:
  - PR #3226 (merged to `dev` 2026-02-04 01:47 UTC) wired
    LinearCrossEntropyModule into BOTH gpt_model.py and mamba_model.py.
  - PR #3207 (merged to `dev` 2026-02-04 22:40 UTC, "Reapply MTP for
    hybrid models") was rebased on a pre-#3226 snapshot and silently
    clobbered the Mamba side, leaving plain ColumnParallelLinear.
  - Result: hybrid Mamba models cannot use
    `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear`
    even when the flag is set — forcing the non-fused path that
    materialises [s, b, V] logits (~12 GiB at NAM56R MBS=12).

What this script does:
  1. Constructs a minimal TransformerConfig + ProcessGroupCollection
     (single-rank, no actual distributed init).
  2. Builds GPTModel(post_process=True) and MambaModel(post_process=True).
  3. Prints type(output_layer) for each and asserts:
        isinstance(gpt.output_layer,   LinearCrossEntropyModule)  -- passes
        isinstance(mamba.output_layer, LinearCrossEntropyModule)  -- FAILS
  4. Applies the proposed fix as a monkey-patch (identical to
     cppmega/megatron/apply_linear_ce_patch.py): reassign
     `mamba.output_layer.__class__ = LinearCrossEntropyModule`.
  5. Re-asserts both — now both pass.

Exit code:
    0 — both assertions match expectations (i.e. GPT passes, Mamba fails
        without fix, Mamba passes after fix); bug IS present as expected.
    1 — unexpected state (no bug found, or fix didn't stick).
    2 — environment error (missing deps, cannot construct model).

Requires:
    torch>=2.12
    megatron-core with both GPTModel and MambaModel available
    (tested against commit 9d71cb1 on NVIDIA/Megatron-LM@dev)
"""
from __future__ import annotations

import os
import sys
import traceback


def _init_single_rank_dist() -> None:
    """Initialise a 1-process torch.distributed group + Megatron model-parallel
    state so the ProcessGroupCollection auto-builder finds every sub-group
    (tp, pp, dp, cp, ep, embd, ...)."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    # Megatron VocabParallelEmbedding.__init__ forks a "model-parallel-rng"
    # state during weight init — register it.
    from megatron.core.tensor_parallel.random import (
        model_parallel_cuda_manual_seed,
    )
    model_parallel_cuda_manual_seed(0)


def _build_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Minimal config: just enough fields for output_layer + language head.
    return TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        kv_channels=16,
        # Mamba mixer: nheads must be divisible by ngroups. Default mamba_mixer
        # picks nheads = hidden/head_dim = 128/64 = 2, ngroups = 1 → OK.
        mamba_head_dim=64,
        mamba_num_groups=1,
        # Linear CE fusion flags — the Mamba path should honour these but
        # silently doesn't, because output_layer is the wrong class.
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=__import__("torch").bfloat16,
    )


def _build_pg_collection():
    """Build a full ProcessGroupCollection from the initialised MPU state.

    `use_mpu_process_groups()` snapshots every sub-group that
    `parallel_state.initialize_model_parallel` created (tp, pp, dp, cp,
    ep, embd, pos_embd, ...), which is the contract LanguageModule
    expects (asserts `hasattr(pgc, "embd")`).
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    return ProcessGroupCollection.use_mpu_process_groups()


def _build_gpt():
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )

    cfg = _build_config()
    spec = get_gpt_layer_local_spec()
    return GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=256,
        max_sequence_length=64,
        pre_process=True,
        post_process=True,
        pg_collection=_build_pg_collection(),
    )


def _build_mamba():
    from megatron.core.models.mamba.mamba_model import MambaModel
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec

    cfg = _build_config()
    return MambaModel(
        config=cfg,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=256,
        max_sequence_length=64,
        hybrid_layer_pattern="M-",  # 1 Mamba + 1 attention for a minimal stack
        pre_process=True,
        post_process=True,
        pg_collection=_build_pg_collection(),
    )


def _apply_fix(mamba_model) -> None:
    """Reassign mamba.output_layer.__class__ -> LinearCrossEntropyModule.

    LinearCrossEntropyModule is a pure subclass of ColumnParallelLinear
    (only forward() differs; no extra state). __class__ reassignment is
    the minimal runtime representation of the proposed source fix.
    """
    from megatron.core.tensor_parallel import ColumnParallelLinear
    from megatron.core.transformer.linear_cross_entropy import (
        LinearCrossEntropyModule,
    )

    assert hasattr(mamba_model, "output_layer"), \
        "model has no output_layer — run with post_process=True"
    assert isinstance(mamba_model.output_layer, ColumnParallelLinear), \
        f"output_layer is unexpectedly {type(mamba_model.output_layer)}"
    mamba_model.output_layer.__class__ = LinearCrossEntropyModule
    # Mirror the attribute the fused forward() branch inspects.
    mamba_model.fuse_linear_cross_entropy = True


def main() -> int:
    # --- Imports -----------------------------------------------------------
    try:
        import torch  # noqa: F401
        from megatron.core.transformer.linear_cross_entropy import (
            LinearCrossEntropyModule,
        )
        from megatron.core.tensor_parallel import ColumnParallelLinear
    except Exception as exc:
        print(f"[env] import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    # --- Distributed stub --------------------------------------------------
    try:
        _init_single_rank_dist()
    except Exception as exc:
        print(f"[env] torch.distributed init failed: {exc}", file=sys.stderr)
        return 2

    # --- Build models ------------------------------------------------------
    try:
        gpt = _build_gpt()
    except Exception as exc:
        print(f"[env] GPTModel construction failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        return 2
    try:
        mamba = _build_mamba()
    except Exception as exc:
        print(f"[env] MambaModel construction failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        traceback.print_exc()
        return 2

    # --- Inspect output_layer class ---------------------------------------
    gpt_cls = type(gpt.output_layer)
    mamba_cls = type(mamba.output_layer)
    print("=" * 72)
    print("Output-layer class check (before fix)")
    print("-" * 72)
    print(f"  GPTModel.output_layer    = {gpt_cls.__module__}.{gpt_cls.__name__}")
    print(f"  MambaModel.output_layer  = {mamba_cls.__module__}.{mamba_cls.__name__}")
    print()

    gpt_ok = isinstance(gpt.output_layer, LinearCrossEntropyModule)
    mamba_bug = (
        isinstance(mamba.output_layer, ColumnParallelLinear)
        and not isinstance(mamba.output_layer, LinearCrossEntropyModule)
    )

    print(f"  assert isinstance(gpt,   LinearCrossEntropyModule)  -> {gpt_ok}")
    print(f"  assert isinstance(mamba, LinearCrossEntropyModule)  -> "
          f"{isinstance(mamba.output_layer, LinearCrossEntropyModule)} "
          f"({'BUG' if mamba_bug else 'no bug'})")

    if not gpt_ok:
        print("\n[unexpected] GPTModel is NOT using LinearCrossEntropyModule. "
              "Is PR #3226 in your tree? (expected on NVIDIA/Megatron-LM@dev.)",
              file=sys.stderr)
        return 1
    if not mamba_bug:
        print("\n[unexpected] MambaModel already uses LinearCrossEntropyModule. "
              "Has PR #3207's regression been fixed upstream?", file=sys.stderr)
        return 1

    # --- Apply proposed fix -----------------------------------------------
    print("\n" + "=" * 72)
    print("Applying proposed fix (class-swap, equivalent to restoring PR #3226)")
    print("-" * 72)
    _apply_fix(mamba)

    mamba_cls_after = type(mamba.output_layer)
    print(f"  MambaModel.output_layer  = "
          f"{mamba_cls_after.__module__}.{mamba_cls_after.__name__}")
    mamba_ok = isinstance(mamba.output_layer, LinearCrossEntropyModule)
    print(f"  assert isinstance(mamba, LinearCrossEntropyModule)  -> {mamba_ok}")
    if not mamba_ok:
        print("\n[fail] monkey-patch did not stick — investigate.", file=sys.stderr)
        return 1

    # --- Verdict -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT: regression CONFIRMED and fix VALIDATED.")
    print("  - GPTModel.output_layer is LinearCrossEntropyModule (PR #3226)")
    print("  - MambaModel.output_layer is plain ColumnParallelLinear (PR #3207 regression)")
    print("  - Class-swap (or restoring PR #3226's mamba_model.py diff) fixes it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
