"""Typed run profiles for cppmega training launchers.

The shell launchers still have to export a few values because the current
runtime patches are imported as ordinary Python modules and read ``os.environ``
at import time.  This module is the source of truth for those values: a launcher
selects one named profile, renders it, and then passes the rendered arguments to
Megatron.  Core model/training choices should live here instead of being spread
across shell snippets.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from typing import Literal

from cppmega.recipes.nam56r_launch import (
    build_nam56r_lite_main_pattern,
    build_nam56r_megatron_native_args,
)
from cppmega.recipes.nam56r_megatron import build_nam56r_feature_plan

MtpCEKernel = Literal["native", "liger", "off"]
Fp8Recipe = Literal["off", "tensorwise", "mxfp8"]
Mxfp8BackwardBackend = Literal["te_tn_adapter", "cutlass_native"]
Mxfp8TransposeEmitBackend = Literal["auto", "te", "off"]
CutlassMxfp8ScaleBackend = Literal["compact", "prepack"]
ParamStorage = Literal["auto", "bf16", "mxfp8"]
SparseMlaMode = Literal["tilelang", "gather_scatter", "pytorch"]
MoeDispatcher = Literal["flex", "alltoall", "allgather"]


@dataclass
class ModelProfile:
    """Logical model shape and feature layout for one launch profile."""

    # Nanochat/Nemotron-style source pattern.  ``A`` becomes attention/MLA/DSA,
    # ``E`` becomes MoE, ``M`` and ``R`` become Mamba-family blocks in the local
    # no-conv stack.  The translated Megatron pattern is derived, not hand-written.
    pattern: str = "AEMEAEMEAEMR"
    # Local GB10 uses the 13-layer quarter-depth stack at full NAM56R width.
    # H200 production uses the same dataclass with depth=52.
    depth: int = 13
    # MTP predictor depth.  This must drive both the hybrid-layer-pattern suffix
    # and ``--mtp-num-layers``; keeping those separate caused false MTP=2 tests.
    mtp_depths: int = 2
    # Width/head settings for the NAM56R-quarter debug lane.
    hidden_size: int = 3584
    ffn_hidden_size: int = 18_944
    num_attention_heads: int = 28
    # DSA ranks are counted in A-layer order, not absolute layer ids.
    dsa_a_layer_ranks: str = "1,2,3"
    # Remote PP/VPP launchers need the main hybrid pattern split by "|".
    # Keeping that here avoids shell-side pattern surgery.
    pipe_hybrid_layer_pattern: bool = False
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = 1
    # Native MoE/DSA args.  Remote scripts used to mutate these with sed.
    moe_expert_model_parallel_size: int = 1
    moe_token_dispatcher_type: MoeDispatcher = "flex"
    moe_router_dtype: str | None = "fp32"
    dsa_indexer_loss_coeff: float = 0.001


@dataclass
class TrainingProfile:
    """Batching, data, and schedule knobs that affect the token stream."""

    # Real clang semantic 4k Megatron indexed data.  Keep the weight prefix here
    # so the launcher cannot accidentally mix in mock data.
    data_path: str = "1.0 /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train"
    tokenizer_model: str = "/home/dave/cppmega-root/cpp_tokenizer_hf"
    vocab_size: int = 65_536
    seq_length: int = 4096
    micro_batch_size: int = 4
    global_batch_size: int = 4
    train_iters: int = 10
    lr: str = "1e-4"
    min_lr: str = "1e-5"

    @property
    def tokens_per_step(self) -> int:
        return self.seq_length * self.global_batch_size


@dataclass
class PrecisionProfile:
    """Precision and kernel-route choices for the launch."""

    # Tensorwise FP8 is the accepted local GB10 correctness lane.  MXFP8 is still
    # probe-only because backward needs the TN adapter/CUTLASS workarounds.
    fp8_recipe: Fp8Recipe = "tensorwise"
    sparse_mla_fp8_quant: str = "te_tensorwise"
    # Flash attention is mandatory when available for full attention/MLA blocks.
    use_flash_attention: bool = True
    # These fields are consumed only when fp8_recipe == "mxfp8" or a TE precision
    # config requests block-scaled TE.  They are kept here so MXFP8 probes are
    # reproducible and do not reintroduce hidden BF16 backward bridges.
    allow_te_mxfp8_sm12: bool = True
    pad_mamba_in_proj_for_mxfp8: bool = True
    mxfp8_bwd_tn_adapter: bool = True
    mxfp8_bwd_backend: Mxfp8BackwardBackend = "te_tn_adapter"
    mxfp8_transpose_emit_backend: Mxfp8TransposeEmitBackend = "te"
    mxfp8_transpose_emit_swizzled: bool = True
    mxfp8_transpose_emit_strict: bool = True
    cutlass_mxfp8_scale_backend: CutlassMxfp8ScaleBackend = "compact"
    # Megatron's MXFP8 param-gather path requires distributed optimizer/FSDP.
    # Keep it as an explicit profile choice so single-GB10 local runs do not
    # accidentally enable an incompatible distributed-param contract.
    fp8_param_gather: bool = False
    reuse_grad_buf_for_mxfp8_param_ag: bool = True
    mxfp8_bwd_allow_bf16_fallback: bool = False
    mxfp8_dgrad_bf16: bool = False
    mxfp8_wgrad_bf16: bool = False


@dataclass
class OptimizerProfile:
    """Memory-first optimizer choices for the local no-master Muon lane."""

    optimizer: str = "muon"
    # ``auto`` means BF16 model params for tensorwise/BF16 runs and primary
    # MXFP8 model params for MXFP8 runs.  This is the optimizer/param-storage
    # contract: the optimizer must update the model storage itself, not a hidden
    # FP32/BF16 master copy.
    param_storage: ParamStorage = "auto"
    muon_momentum: str = "0.95"
    muon_scale_mode: str = "spectral"
    muon_num_ns_steps: int = 5
    muon_tp_mode: str = "blockwise"
    muon_scalar_optimizer: str = "adam8bit"
    muon_quantized_momentum: bool = True
    muon_quantized_momentum_dtype: str = "int8"
    muon_quantized_momentum_block_size: int = 256
    use_bf16_no_master_emerging_optimizer: bool = True
    use_bf16_no_master_emerging_fallback_optimizer: bool = True
    grad_reduce_in_bf16: bool = True
    use_distributed_optimizer: bool = False
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: str = "bf16"
    main_params_dtype: str = "fp16"
    exp_avg_dtype: str = "fp8"
    exp_avg_sq_dtype: str = "fp8"
    local_ddp_disable_contiguous_grad_buffer: bool = True


@dataclass
class RuntimePatchProfile:
    """Import-time runtime patch choices still read by cppmega shim modules."""

    mamba3_mimo: bool = True
    mamba_num_groups: int = 8
    mamba_recompute: bool = True
    dsa_sparse_mode: SparseMlaMode = "tilelang"
    dsa_indexer_loss_coeff: str = "0"
    dsa_skip_indexer_loss: bool = True
    ngram_hash_enabled: bool = True
    structure_enabled: bool = True
    structure_components: str = "core"
    mtp_ce_kernel: MtpCEKernel = "native"
    acknowledge_liger_mtp_ce_deprecated: bool = False


@dataclass
class ProfilingProfile:
    """Optional profiling hooks for a run profile."""

    memory_debug: bool = False
    mem_profile: bool = False
    mem_profile_steps: int = 2
    torch_profile: bool = False
    torch_profile_steps: int = 2
    nsys_profile: bool = False


@dataclass
class RunProfile:
    """Complete launch contract for one reproducible cppmega run."""

    name: str
    description: str
    model: ModelProfile = field(default_factory=ModelProfile)
    training: TrainingProfile = field(default_factory=TrainingProfile)
    precision: PrecisionProfile = field(default_factory=PrecisionProfile)
    optimizer: OptimizerProfile = field(default_factory=OptimizerProfile)
    runtime: RuntimePatchProfile = field(default_factory=RuntimePatchProfile)
    profiling: ProfilingProfile = field(default_factory=ProfilingProfile)
    spec_module: str = "cppmega.megatron.nam56r_noconv_spec"
    spec_function: str = "build_cppmega_nam56r_noconv_stack_spec"

    def resolved_param_storage(self) -> Literal["bf16", "mxfp8"]:
        """Return the concrete model-parameter storage dtype for this profile."""

        if self.optimizer.param_storage == "auto":
            return "mxfp8" if self.precision.fp8_recipe == "mxfp8" else "bf16"
        return self.optimizer.param_storage

    def hybrid_layer_pattern(self) -> str:
        """Return the Megatron hybrid-layer-pattern derived from model fields."""

        pattern = build_nam56r_lite_main_pattern(
            pattern=self.model.pattern,
            depth=self.model.depth,
            mtp_depths=self.model.mtp_depths,
        )
        if not self.model.pipe_hybrid_layer_pattern:
            return pattern
        if "/" in pattern:
            main, mtp_part = pattern.split("/", 1)
        else:
            main, mtp_part = pattern, ""
        n_chunks = self.model.pipeline_model_parallel_size * max(
            self.model.virtual_pipeline_model_parallel_size, 1
        )
        if n_chunks > 1:
            total = len(main)
            if total % n_chunks != 0:
                raise ValueError(
                    f"cannot split {total}-layer main pattern into {n_chunks} PP/VPP chunks"
                )
            per_chunk = total // n_chunks
            main = "|".join(
                main[i * per_chunk : (i + 1) * per_chunk] for i in range(n_chunks)
            )
        return main + (("/" + mtp_part) if mtp_part else "")

    def native_args_fragment(self) -> str:
        """Return the Megatron-native feature fragment derived from this profile."""

        plan = build_nam56r_feature_plan(
            pattern=self.model.pattern,
            depth=self.model.depth,
            mtp_depths=self.model.mtp_depths,
        )
        bundle = build_nam56r_megatron_native_args(
            plan=plan,
            enable_mla=True,
            enable_mtp=self.model.mtp_depths > 0,
            mtp_mode="hybrid",
            mtp_num_predictors=self.model.mtp_depths,
            enable_moe=True,
            moe_expert_model_parallel_size=self.model.moe_expert_model_parallel_size,
            moe_token_dispatcher_type=self.model.moe_token_dispatcher_type,
            moe_router_dtype=self.model.moe_router_dtype,
            enable_dsa=True,
            dsa_indexer_loss_coeff=self.model.dsa_indexer_loss_coeff,
        )
        return bundle.to_shell_fragment()


def _bool(value: bool) -> str:
    return "1" if value else "0"


def set_local_gb10_quarter_profile(profile: RunProfile | None = None) -> RunProfile:
    """Fill the local GB10 NAM56R-quarter profile.

    This is the default correctness lane used on the single-GB10 box: full
    NAM56R width, quarter depth, real 4k clang data, MTP=2, Liger MTP CE, Flash
    Attention, tensorwise FP8, no-master Muon, q8 Muon momentum, and disabled
    contiguous local DDP grad buffer.  It intentionally favors debuggability and
    memory pressure over production H200 throughput.
    """

    if profile is None:
        profile = RunProfile(
            name="local_gb10_quarter",
            description="Single-GB10 NAM56R-quarter correctness/profiling lane",
        )
    profile.model = ModelProfile()
    profile.training = TrainingProfile()
    profile.precision = PrecisionProfile()
    profile.optimizer = OptimizerProfile()
    profile.runtime = RuntimePatchProfile(
        mtp_ce_kernel="liger",
        acknowledge_liger_mtp_ce_deprecated=True,
    )
    profile.profiling = ProfilingProfile()
    return profile


def set_h200_dsa_9_4_m_profile(profile: RunProfile | None = None) -> RunProfile:
    """Fill the H200 full-depth NAM56R DSA 9+4 profile skeleton.

    This profile documents the production target in typed form, but the remote
    H200 launchers still own machine-specific topology details such as PP/VPP/EP
    and tmux orchestration.  Keeping it here prevents local GB10 exceptions
    from being mistaken for the production contract.
    """

    if profile is None:
        profile = RunProfile(
            name="h200_dsa_9_4_m",
            description="Full-depth H200 NAM56R DSA 9+4 production target skeleton",
        )
    profile.model = ModelProfile(
        depth=52,
        mtp_depths=2,
        hidden_size=4096,
        ffn_hidden_size=21_504,
        num_attention_heads=32,
        dsa_a_layer_ranks="1,2,3,5,6,7,9,10,11",
        pipe_hybrid_layer_pattern=True,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    profile.training = TrainingProfile(micro_batch_size=4, global_batch_size=64)
    profile.precision = PrecisionProfile(fp8_recipe="tensorwise")
    profile.optimizer = OptimizerProfile(local_ddp_disable_contiguous_grad_buffer=False)
    profile.runtime = RuntimePatchProfile(
        mtp_ce_kernel="liger",
        acknowledge_liger_mtp_ce_deprecated=True,
    )
    profile.profiling = ProfilingProfile()
    return profile


PROFILE_SETTERS = {
    "local_gb10_quarter": set_local_gb10_quarter_profile,
    "h200_dsa_9_4_m": set_h200_dsa_9_4_m_profile,
}


def get_run_profile(name: str) -> RunProfile:
    """Build a named profile through its setter function."""

    try:
        setter = PROFILE_SETTERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROFILE_SETTERS))
        raise ValueError(f"unknown cppmega run profile {name!r}; supported: {supported}") from exc
    return setter()


def profile_shell_assignments(profile: RunProfile) -> dict[str, str]:
    """Render the profile to shell assignments consumed by legacy launchers."""

    env: dict[str, str] = {
        "CPPMEGA_RUN_PROFILE": profile.name,
        "CPPMEGA_NEM_PATTERN": profile.model.pattern,
        "CPPMEGA_LAYER_DEPTH": str(profile.model.depth),
        "MTP_DEPTHS": str(profile.model.mtp_depths),
        "CPPMEGA_DSA_A_LAYER_RANKS": profile.model.dsa_a_layer_ranks,
        "CPPMEGA_PP_SIZE": str(profile.model.pipeline_model_parallel_size),
        "CPPMEGA_VPP_SIZE": str(profile.model.virtual_pipeline_model_parallel_size),
        "CPPMEGA_EP_SIZE": str(profile.model.moe_expert_model_parallel_size),
        "CPPMEGA_SEQ_LENGTH": str(profile.training.seq_length),
        "CPPMEGA_MAX_POSITION_EMBEDDINGS": str(profile.training.seq_length),
        "CPPMEGA_MICRO_BATCH_SIZE": str(profile.training.micro_batch_size),
        "CPPMEGA_GLOBAL_BATCH_SIZE": str(profile.training.global_batch_size),
        "CPPMEGA_TRAIN_ITERS": str(profile.training.train_iters),
        "CPPMEGA_LR": profile.training.lr,
        "CPPMEGA_MIN_LR": profile.training.min_lr,
        "CPPMEGA_DATA_PATH": profile.training.data_path,
        "CPPMEGA_TOKENIZER_MODEL": profile.training.tokenizer_model,
        "CPPMEGA_VOCAB_SIZE": str(profile.training.vocab_size),
        "CPPMEGA_HIDDEN_SIZE": str(profile.model.hidden_size),
        "CPPMEGA_FFN_HIDDEN_SIZE": str(profile.model.ffn_hidden_size),
        "CPPMEGA_NUM_ATTN_HEADS": str(profile.model.num_attention_heads),
        "CPPMEGA_SPEC_MODULE": profile.spec_module,
        "CPPMEGA_SPEC_FUNCTION": profile.spec_function,
        "CPPMEGA_FP8_RECIPE": profile.precision.fp8_recipe,
        "CPPMEGA_SPARSE_MLA_FP8_QUANT": profile.precision.sparse_mla_fp8_quant,
        "CPPMEGA_OPTIMIZER": profile.optimizer.optimizer,
        "CPPMEGA_PARAM_STORAGE": profile.resolved_param_storage(),
        "CPPMEGA_MUON_MOMENTUM": profile.optimizer.muon_momentum,
        "CPPMEGA_MUON_SCALE_MODE": profile.optimizer.muon_scale_mode,
        "CPPMEGA_MUON_NUM_NS_STEPS": str(profile.optimizer.muon_num_ns_steps),
        "CPPMEGA_MUON_TP_MODE": profile.optimizer.muon_tp_mode,
        "CPPMEGA_MUON_SCALAR_OPTIMIZER": profile.optimizer.muon_scalar_optimizer,
        "CPPMEGA_MUON_QUANTIZED_MOMENTUM": _bool(profile.optimizer.muon_quantized_momentum),
        "CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE": (
            profile.optimizer.muon_quantized_momentum_dtype
        ),
        "CPPMEGA_MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE": str(
            profile.optimizer.muon_quantized_momentum_block_size
        ),
        "CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER": _bool(
            profile.optimizer.use_bf16_no_master_emerging_optimizer
        ),
        "CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER": _bool(
            profile.optimizer.use_bf16_no_master_emerging_fallback_optimizer
        ),
        "CPPMEGA_GRAD_REDUCE_IN_BF16": _bool(profile.optimizer.grad_reduce_in_bf16),
        "CPPMEGA_USE_DISTRIBUTED_OPTIMIZER": _bool(
            profile.optimizer.use_distributed_optimizer
        ),
        "CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER": _bool(
            profile.optimizer.use_precision_aware_optimizer
        ),
        "CPPMEGA_MAIN_GRADS_DTYPE": profile.optimizer.main_grads_dtype,
        "CPPMEGA_MAIN_PARAMS_DTYPE": profile.optimizer.main_params_dtype,
        "CPPMEGA_EXP_AVG_DTYPE": profile.optimizer.exp_avg_dtype,
        "CPPMEGA_EXP_AVG_SQ_DTYPE": profile.optimizer.exp_avg_sq_dtype,
        "CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER": _bool(
            profile.optimizer.local_ddp_disable_contiguous_grad_buffer
        ),
        "CPPMEGA_MAMBA3_MIMO": _bool(profile.runtime.mamba3_mimo),
        "CPPMEGA_MAMBA_NUM_GROUPS": str(profile.runtime.mamba_num_groups),
        "CPPMEGA_MAMBA_RECOMPUTE": _bool(profile.runtime.mamba_recompute),
        "CPPMEGA_DSA_SPARSE_MODE": profile.runtime.dsa_sparse_mode,
        "CPPMEGA_DSA_INDEXER_LOSS_COEFF": profile.runtime.dsa_indexer_loss_coeff,
        "CPPMEGA_DSA_SKIP_INDEXER_LOSS": _bool(profile.runtime.dsa_skip_indexer_loss),
        "CPPMEGA_NGRAM_HASH_ENABLED": _bool(profile.runtime.ngram_hash_enabled),
        "CPPMEGA_STRUCTURE_ENABLED": _bool(profile.runtime.structure_enabled),
        "CPPMEGA_STRUCTURE_COMPONENTS": profile.runtime.structure_components,
        "CPPMEGA_MTP_CE_KERNEL": profile.runtime.mtp_ce_kernel,
        "CPPMEGA_MEMORY_DEBUG": _bool(profile.profiling.memory_debug),
        "CPPMEGA_MEM_PROFILE": _bool(profile.profiling.mem_profile),
        "CPPMEGA_MEM_PROFILE_STEPS": str(profile.profiling.mem_profile_steps),
        "CPPMEGA_TORCH_PROFILE": _bool(profile.profiling.torch_profile),
        "CPPMEGA_TORCH_PROFILE_STEPS": str(profile.profiling.torch_profile_steps),
        "CPPMEGA_NSYS_PROFILE": _bool(profile.profiling.nsys_profile),
        "HYBRID_LAYER_PATTERN": profile.hybrid_layer_pattern(),
        "HYBRID_PATTERN": profile.hybrid_layer_pattern(),
        "NATIVE_ARGS": profile.native_args_fragment(),
    }
    if profile.runtime.mtp_ce_kernel == "liger" and profile.runtime.acknowledge_liger_mtp_ce_deprecated:
        env["CPPMEGA_I_UNDERSTAND_MTP_LIGER_CE_IS_DEPRECATED"] = "1"
    if profile.precision.fp8_recipe == "mxfp8":
        env.update(
            {
                "CPPMEGA_ALLOW_TE_MXFP8_SM12": _bool(profile.precision.allow_te_mxfp8_sm12),
                "CPPMEGA_PAD_MAMBA_IN_PROJ_FOR_MXFP8": _bool(
                    profile.precision.pad_mamba_in_proj_for_mxfp8
                ),
                "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": _bool(
                    profile.precision.mxfp8_bwd_tn_adapter
                ),
                "CPPMEGA_TE_MXFP8_BWD_BACKEND": profile.precision.mxfp8_bwd_backend,
                "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND": (
                    profile.precision.mxfp8_transpose_emit_backend
                ),
                "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_SWIZZLED": _bool(
                    profile.precision.mxfp8_transpose_emit_swizzled
                ),
                "CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_STRICT": _bool(
                    profile.precision.mxfp8_transpose_emit_strict
                ),
                "CPPMEGA_CUTLASS_MXFP8_SCALE_BACKEND": (
                    profile.precision.cutlass_mxfp8_scale_backend
                ),
                "CPPMEGA_FP8_PARAM_GATHER": _bool(profile.precision.fp8_param_gather),
                "CPPMEGA_REUSE_GRAD_BUF_FOR_MXFP8_PARAM_AG": _bool(
                    profile.precision.reuse_grad_buf_for_mxfp8_param_ag
                ),
                "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": _bool(
                    profile.precision.mxfp8_bwd_allow_bf16_fallback
                ),
                "CPPMEGA_TE_MXFP8_DGRAD_BF16": _bool(profile.precision.mxfp8_dgrad_bf16),
                "CPPMEGA_TE_MXFP8_WGRAD_BF16": _bool(profile.precision.mxfp8_wgrad_bf16),
                "NVTE_BACKWARD_OVERRIDE": "none",
            }
        )
    return env


def render_shell(profile: RunProfile) -> str:
    """Render profile assignments as POSIX shell-safe ``export`` statements."""

    lines = [
        f"# cppmega run profile: {profile.name}",
        f"# {profile.description}",
    ]
    for key, value in profile_shell_assignments(profile).items():
        lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines)


def apply_cli_overrides(profile: RunProfile, args: argparse.Namespace) -> RunProfile:
    """Apply explicit CLI parameters to a profile after its setter ran."""

    if args.mtp_depths is not None:
        profile.model.mtp_depths = args.mtp_depths
    if args.pipeline_model_parallel_size is not None:
        profile.model.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    if args.virtual_pipeline_model_parallel_size is not None:
        profile.model.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    if args.pipe_hybrid_layer_pattern is not None:
        profile.model.pipe_hybrid_layer_pattern = args.pipe_hybrid_layer_pattern
    if args.expert_model_parallel_size is not None:
        profile.model.moe_expert_model_parallel_size = args.expert_model_parallel_size
    if args.moe_token_dispatcher_type is not None:
        profile.model.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    if args.moe_router_dtype is not None:
        profile.model.moe_router_dtype = None if args.moe_router_dtype == "none" else args.moe_router_dtype
    if args.dsa_indexer_loss_coeff is not None:
        profile.model.dsa_indexer_loss_coeff = args.dsa_indexer_loss_coeff
    if args.mtp_ce_kernel is not None:
        profile.runtime.mtp_ce_kernel = args.mtp_ce_kernel
        profile.runtime.acknowledge_liger_mtp_ce_deprecated = args.mtp_ce_kernel == "liger"
    if args.fp8_recipe is not None:
        profile.precision.fp8_recipe = args.fp8_recipe
    if args.mxfp8_bwd_backend is not None:
        profile.precision.mxfp8_bwd_backend = args.mxfp8_bwd_backend
    if args.mxfp8_transpose_emit_backend is not None:
        profile.precision.mxfp8_transpose_emit_backend = args.mxfp8_transpose_emit_backend
    if args.mxfp8_transpose_emit_swizzled is not None:
        profile.precision.mxfp8_transpose_emit_swizzled = args.mxfp8_transpose_emit_swizzled
    if args.mxfp8_transpose_emit_strict is not None:
        profile.precision.mxfp8_transpose_emit_strict = args.mxfp8_transpose_emit_strict
    if args.cutlass_mxfp8_scale_backend is not None:
        profile.precision.cutlass_mxfp8_scale_backend = args.cutlass_mxfp8_scale_backend
    if args.fp8_param_gather is not None:
        profile.precision.fp8_param_gather = args.fp8_param_gather
    if args.reuse_grad_buf_for_mxfp8_param_ag is not None:
        profile.precision.reuse_grad_buf_for_mxfp8_param_ag = (
            args.reuse_grad_buf_for_mxfp8_param_ag
        )
    if args.optimizer is not None:
        profile.optimizer.optimizer = args.optimizer
    if args.param_storage is not None:
        profile.optimizer.param_storage = args.param_storage
    if args.seq_length is not None:
        profile.training.seq_length = args.seq_length
    if args.micro_batch_size is not None:
        profile.training.micro_batch_size = args.micro_batch_size
    if args.global_batch_size is not None:
        profile.training.global_batch_size = args.global_batch_size
    if args.train_iters is not None:
        profile.training.train_iters = args.train_iters
    if args.mem_profile:
        profile.profiling.mem_profile = True
    if args.mem_profile_steps is not None:
        profile.profiling.mem_profile = True
        profile.profiling.mem_profile_steps = args.mem_profile_steps
    if args.torch_profile:
        profile.profiling.torch_profile = True
    if args.nsys_profile:
        profile.profiling.nsys_profile = True
    return profile


def _add_common_profile_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mtp-depths", type=int, default=None)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=None)
    parser.add_argument("--virtual-pipeline-model-parallel-size", type=int, default=None)
    pipe_group = parser.add_mutually_exclusive_group()
    pipe_group.add_argument(
        "--pipe-hybrid-layer-pattern",
        action="store_true",
        default=None,
        dest="pipe_hybrid_layer_pattern",
    )
    pipe_group.add_argument(
        "--no-pipe-hybrid-layer-pattern",
        action="store_false",
        default=None,
        dest="pipe_hybrid_layer_pattern",
    )
    parser.add_argument("--expert-model-parallel-size", type=int, default=None)
    parser.add_argument(
        "--moe-token-dispatcher-type",
        choices=("flex", "alltoall", "allgather"),
        default=None,
    )
    parser.add_argument(
        "--moe-router-dtype",
        choices=("fp32", "none"),
        default=None,
    )
    parser.add_argument("--dsa-indexer-loss-coeff", type=float, default=None)
    parser.add_argument("--mtp-ce-kernel", choices=("native", "liger", "off"), default=None)
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise", "mxfp8"), default=None)
    parser.add_argument(
        "--mxfp8-bwd-backend",
        choices=("te_tn_adapter", "cutlass_native"),
        default=None,
    )
    parser.add_argument(
        "--mxfp8-transpose-emit-backend",
        choices=("auto", "te", "off"),
        default=None,
    )
    parser.add_argument(
        "--cutlass-mxfp8-scale-backend",
        choices=("compact", "prepack"),
        default=None,
    )
    fp8_param_gather = parser.add_mutually_exclusive_group()
    fp8_param_gather.add_argument(
        "--fp8-param-gather",
        action="store_true",
        default=None,
        dest="fp8_param_gather",
    )
    fp8_param_gather.add_argument(
        "--no-fp8-param-gather",
        action="store_false",
        default=None,
        dest="fp8_param_gather",
    )
    reuse_mxfp8_ag = parser.add_mutually_exclusive_group()
    reuse_mxfp8_ag.add_argument(
        "--reuse-grad-buf-for-mxfp8-param-ag",
        action="store_true",
        default=None,
        dest="reuse_grad_buf_for_mxfp8_param_ag",
    )
    reuse_mxfp8_ag.add_argument(
        "--no-reuse-grad-buf-for-mxfp8-param-ag",
        action="store_false",
        default=None,
        dest="reuse_grad_buf_for_mxfp8_param_ag",
    )
    emit_swizzled = parser.add_mutually_exclusive_group()
    emit_swizzled.add_argument(
        "--mxfp8-transpose-emit-swizzled",
        action="store_true",
        default=None,
        dest="mxfp8_transpose_emit_swizzled",
    )
    emit_swizzled.add_argument(
        "--no-mxfp8-transpose-emit-swizzled",
        action="store_false",
        default=None,
        dest="mxfp8_transpose_emit_swizzled",
    )
    emit_strict = parser.add_mutually_exclusive_group()
    emit_strict.add_argument(
        "--mxfp8-transpose-emit-strict",
        action="store_true",
        default=None,
        dest="mxfp8_transpose_emit_strict",
    )
    emit_strict.add_argument(
        "--no-mxfp8-transpose-emit-strict",
        action="store_false",
        default=None,
        dest="mxfp8_transpose_emit_strict",
    )
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--param-storage", choices=("auto", "bf16", "mxfp8"), default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--mem-profile", action="store_true")
    parser.add_argument("--mem-profile-steps", type=int, default=None)
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--nsys-profile", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render cppmega typed run profiles")
    sub = parser.add_subparsers(dest="command", required=True)
    shell = sub.add_parser("shell", help="render shell exports for a named profile")
    shell.add_argument("profile", choices=sorted(PROFILE_SETTERS), help="profile name")
    _add_common_profile_overrides(shell)
    describe = sub.add_parser("describe", help="print the derived profile summary")
    describe.add_argument("profile", choices=sorted(PROFILE_SETTERS), help="profile name")
    _add_common_profile_overrides(describe)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    profile = apply_cli_overrides(get_run_profile(args.profile), args)
    if args.command == "shell":
        print(render_shell(profile))
    elif args.command == "describe":
        print(f"name={profile.name}")
        print(f"description={profile.description}")
        print(f"hybrid_layer_pattern={profile.hybrid_layer_pattern()}")
        print(f"native_args={profile.native_args_fragment()}")
        print(f"tokens_per_step={profile.training.tokens_per_step}")
    else:  # pragma: no cover - argparse enforces this.
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
