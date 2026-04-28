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

MtpCEKernel = Literal["native", "cce", "liger", "off"]
Fp8Format = Literal["hybrid", "e4m3"]
Fp8Recipe = Literal["off", "tensorwise", "mxfp8"]
Mxfp8BackwardBackend = Literal["te_tn_adapter", "flashinfer_cutlass", "cutlass_native"]
Mxfp8TransposeEmitBackend = Literal["auto", "te", "off"]
Mxfp8FlashinferRunner = Literal["mm_mxfp8", "direct_tactic"]
ParamStorage = Literal["auto", "bf16", "mxfp8"]
SparseMlaMode = Literal["tilelang", "gather_scatter", "pytorch"]
MoeDispatcher = Literal["flex", "alltoall", "allgather"]
MoeFlexBackend = Literal["deepep", "hybridep"]
NsysCaptureMode = Literal["full", "delay", "cudaProfilerApi"]
AttentionBackend = Literal["auto", "flash", "fused", "unfused"]


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
    moe_token_dispatcher_type: MoeDispatcher = "alltoall"
    moe_flex_dispatcher_backend: MoeFlexBackend = "deepep"
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

    # Tensorwise FP8 is the conservative GB10 lane.  Dense MXFP8 Linear GEMMs
    # route clean GEMM calls through TE payload + FlashInfer/CUTLASS
    # layout_128x4 by default; the old compact direct CUTLASS loader remains
    # an explicit experiment.  This is not the attention backend: attention is
    # controlled separately by ``attention_backend`` and local GB10 pins it to
    # patched FA4.
    # ``auto`` would hide the important contract here, so the resolved profile
    # owns the exact TE fp8 format: tensorwise keeps HYBRID, while MXFP8 uses
    # E4M3 because FlashInfer/CUTLASS ``mm_mxfp8`` accepts E4M3 payloads only.
    fp8_format: Fp8Format = "hybrid"
    fp8_recipe: Fp8Recipe = "tensorwise"
    sparse_mla_fp8_quant: str = "te_tensorwise"
    # Keep --use-flash-attn enabled for accelerated TE attention paths, but let
    # the launcher choose the concrete Megatron backend explicitly.  Local GB10
    # pins this to flash through the patched FA4 SM120 source tree.
    use_flash_attention: bool = True
    attention_backend: AttentionBackend = "flash"
    # These fields are consumed only when fp8_recipe == "mxfp8" or a TE precision
    # config requests block-scaled TE.  They are kept here so MXFP8 probes are
    # reproducible and do not reintroduce hidden BF16 backward bridges.
    allow_te_mxfp8_sm12: bool = True
    pad_mamba_in_proj_for_mxfp8: bool = True
    mxfp8_bwd_tn_adapter: bool = True
    mxfp8_bwd_backend: Mxfp8BackwardBackend = "flashinfer_cutlass"
    mxfp8_transpose_emit_backend: Mxfp8TransposeEmitBackend = "te"
    mxfp8_transpose_emit_swizzled: bool = True
    mxfp8_transpose_emit_strict: bool = True
    # Experimental dense Linear backward mode: TE saves original compact
    # columnwise MXFP8 operands and lets the cppmega compact-direct backend
    # read them directly.  This removes the dense rowwise-transpose copies, but
    # the current SM120 direct loader is slower than the TE-transpose TN path on
    # full-model GB10 runs, so keep it opt-in until the loader/mainloop is fixed.
    mxfp8_compact_columnwise_backward: bool = False
    # FlashInfer's public mm_mxfp8 path owns autotuning. direct_tactic bypasses
    # that layer and is only for shape/tactic probes when nsys shows overhead.
    mxfp8_flashinfer_runner: Mxfp8FlashinferRunner = "mm_mxfp8"
    mxfp8_flashinfer_tactic: int = 0
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
    noconv_mamba_chunk_size: int | None = None
    dsa_sparse_mode: SparseMlaMode = "tilelang"
    dsa_fp8_attention: bool = False
    # NOTE: same name as ModelProfile.dsa_indexer_loss_coeff but semantically
    # distinct — this is the env-var string passed to import-time shim patches,
    # while ModelProfile's is the float fed to the --dsa-indexer-loss-coeff
    # Megatron launcher arg.  Not the same value; be deliberate about which you
    # assign.
    dsa_indexer_loss_coeff: str = "0"
    dsa_skip_indexer_loss: bool = True
    ngram_hash_enabled: bool = True
    structure_enabled: bool = True
    structure_components: str = "core"
    mtp_ce_kernel: MtpCEKernel = "native"
    # Experimental CCE launch fusion for main+MTP heads. Real GB10 A/B on
    # 2026-04-28 was finite but slower, so this stays off unless explicitly
    # requested by a profile/CLI override.
    cce_fuse_main_mtp_ce: bool = False
    acknowledge_liger_mtp_ce_deprecated: bool = False
    # Local source overrides that must be part of the typed launch contract.
    # The GB10 FA4/TE fixes are source-tree patches, not installed PyPI wheels;
    # without these roots first on PYTHONPATH the process can import a mixed
    # flash-attn package and silently miss the SM120 guard/fallback fixes.
    transformer_engine_source: str | None = "/home/dave/TransformerEngine"
    flash_attention_source: str | None = "/home/dave/flash-attention-fa4"


@dataclass
class ProfilingProfile:
    """Optional profiling hooks for a run profile."""

    memory_debug: bool = False
    mem_profile: bool = False
    mem_profile_steps: int = 2
    torch_profile: bool = False
    torch_profile_steps: int = 2
    nsys_profile: bool = False
    # Nsight Systems CUDA-profiler capture ranges are unreliable on the local
    # CUDA 13.2/GB10 stack: reports contain CUDA API ranges but no kernel
    # activity records.  Hardware CUDA tracing has the same failure mode here,
    # so default to software CUDA tracing and a full-process capture.
    nsys_capture_mode: NsysCaptureMode = "full"
    nsys_trace: str = "cuda-sw,nvtx,osrt"
    nsys_delay: int = 0
    # ``0`` means "collect from delay until normal process exit".  On the
    # local GB10 stack, nsys --duration can terminate the wrapped torchrun child
    # with SIGTERM after report generation, which makes an otherwise successful
    # profile look failed to torch.distributed.run.
    nsys_duration: int = 0
    # Megatron cudaProfilerStart/Stop range used by external profilers such as
    # Nsight Compute.  Keep it separate from torch_profile/nsys_profile because
    # CUPTI permits only one active subscriber.
    cuda_profile: bool = False
    cuda_profile_step_start: int = 3
    cuda_profile_step_end: int = 4


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

    def resolved_fp8_format(self) -> Fp8Format:
        """Return the TE FP8 format required by the selected FP8 recipe."""

        if self.precision.fp8_recipe == "mxfp8":
            return "e4m3"
        return self.precision.fp8_format

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

        if (
            self.model.moe_token_dispatcher_type == "flex"
            and self.model.moe_expert_model_parallel_size <= 1
        ):
            raise ValueError(
                "moe_token_dispatcher_type=flex requires "
                "moe_expert_model_parallel_size > 1 in cppmega run profiles"
            )
        if (
            self.model.moe_token_dispatcher_type == "flex"
            and self.model.moe_router_dtype != "fp32"
        ):
            raise ValueError(
                "moe_token_dispatcher_type=flex requires moe_router_dtype=fp32 "
                "because DeepEP/HybridEP weighted dispatch consumes fp32 probabilities"
            )
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
            moe_flex_dispatcher_backend=self.model.moe_flex_dispatcher_backend,
            moe_router_dtype=self.model.moe_router_dtype,
            enable_dsa=True,
            dsa_indexer_loss_coeff=self.model.dsa_indexer_loss_coeff,
        )
        return bundle.to_shell_fragment()


def _bool(value: bool) -> str:
    return "1" if value else "0"


def _validate_noconv_mamba_chunk_size(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError("--noconv-mamba-chunk-size must be a positive power of two")
    return value


def set_local_gb10_quarter_profile(profile: RunProfile | None = None) -> RunProfile:
    """Fill the local GB10 NAM56R-quarter profile.

    This is the default correctness lane used on the single-GB10 box: full
    NAM56R width, quarter depth, real 4k clang data, MTP=2, CCE MTP CE, TE
    patched FA4 SM120 routing, tensorwise FP8, no-master Muon, q8 Muon momentum,
    and disabled contiguous local DDP grad buffer.  It intentionally favors
    debuggability and memory pressure over production H200 throughput.
    """

    if profile is None:
        profile = RunProfile(
            name="local_gb10_quarter",
            description="Single-GB10 NAM56R-quarter correctness/profiling lane",
        )
    # Single-GB10 has TP=EP=1, so Megatron's Flex dispatcher is invalid here.
    # Keep this in the typed profile instead of relying on a shell fallback.
    profile.model.moe_token_dispatcher_type = "alltoall"
    profile.precision.attention_backend = "flash"
    # The remaining BF16 GEMM hotspot on GB10 is Muon's Newton-Schulz loop.
    # nanochat's comparable performance presets use 3 iterations; keep this as
    # a typed profile default so runs can restore 5 with --muon-num-ns-steps 5.
    profile.optimizer.muon_num_ns_steps = 3
    profile.runtime = RuntimePatchProfile(
        mtp_ce_kernel="cce",
        acknowledge_liger_mtp_ce_deprecated=False,
        noconv_mamba_chunk_size=256,
    )
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
        "CPPMEGA_MOE_TOKEN_DISPATCHER_TYPE": profile.model.moe_token_dispatcher_type,
        "CPPMEGA_MOE_FLEX_DISPATCHER_BACKEND": profile.model.moe_flex_dispatcher_backend,
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
        "CPPMEGA_FP8_FORMAT": profile.resolved_fp8_format(),
        "CPPMEGA_FP8_RECIPE": profile.precision.fp8_recipe,
        "CPPMEGA_SPARSE_MLA_FP8_QUANT": profile.precision.sparse_mla_fp8_quant,
        "CPPMEGA_USE_FLASH_ATTN": _bool(profile.precision.use_flash_attention),
        "CPPMEGA_ATTN_BACKEND": profile.precision.attention_backend,
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
        "CPPMEGA_DSA_FP8_ATTENTION": _bool(profile.runtime.dsa_fp8_attention),
        "CPPMEGA_DSA_INDEXER_LOSS_COEFF": profile.runtime.dsa_indexer_loss_coeff,
        "CPPMEGA_DSA_SKIP_INDEXER_LOSS": _bool(profile.runtime.dsa_skip_indexer_loss),
        "CPPMEGA_NGRAM_HASH_ENABLED": _bool(profile.runtime.ngram_hash_enabled),
        "CPPMEGA_STRUCTURE_ENABLED": _bool(profile.runtime.structure_enabled),
        "CPPMEGA_STRUCTURE_COMPONENTS": profile.runtime.structure_components,
        "CPPMEGA_MTP_CE_KERNEL": profile.runtime.mtp_ce_kernel,
        "CPPMEGA_CCE_FUSE_MAIN_MTP_CE": _bool(profile.runtime.cce_fuse_main_mtp_ce),
        "CPPMEGA_MEMORY_DEBUG": _bool(profile.profiling.memory_debug),
        "CPPMEGA_MEM_PROFILE": _bool(profile.profiling.mem_profile),
        "CPPMEGA_MEM_PROFILE_STEPS": str(profile.profiling.mem_profile_steps),
        "CPPMEGA_TORCH_PROFILE": _bool(profile.profiling.torch_profile),
        "CPPMEGA_TORCH_PROFILE_STEPS": str(profile.profiling.torch_profile_steps),
        "CPPMEGA_NSYS_PROFILE": _bool(profile.profiling.nsys_profile),
        "CPPMEGA_NSYS_CAPTURE_MODE": profile.profiling.nsys_capture_mode,
        "CPPMEGA_NSYS_TRACE": profile.profiling.nsys_trace,
        "CPPMEGA_NSYS_DELAY": str(profile.profiling.nsys_delay),
        "CPPMEGA_NSYS_DURATION": str(profile.profiling.nsys_duration),
        "CPPMEGA_CUDA_PROFILE": _bool(profile.profiling.cuda_profile),
        "CPPMEGA_CUDA_PROFILE_STEP_START": str(
            profile.profiling.cuda_profile_step_start
        ),
        "CPPMEGA_CUDA_PROFILE_STEP_END": str(profile.profiling.cuda_profile_step_end),
        "HYBRID_LAYER_PATTERN": profile.hybrid_layer_pattern(),
        "NATIVE_ARGS": profile.native_args_fragment(),
    }
    extra_pythonpath = [
        path
        for path in (
            profile.runtime.flash_attention_source,
            profile.runtime.transformer_engine_source,
        )
        if path
    ]
    if extra_pythonpath:
        env["CPPMEGA_EXTRA_PYTHONPATH"] = ":".join(extra_pythonpath)
        env["CPPMEGA_FLASH_ATTN_SOURCE_ROOT"] = (
            profile.runtime.flash_attention_source or ""
        )
        env["CPPMEGA_TRANSFORMER_ENGINE_SOURCE_ROOT"] = (
            profile.runtime.transformer_engine_source or ""
        )
    if profile.runtime.noconv_mamba_chunk_size is not None:
        env["CPPMEGA_NOCONV_MAMBA_CHUNK_SIZE"] = str(
            _validate_noconv_mamba_chunk_size(profile.runtime.noconv_mamba_chunk_size)
        )
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
                "CPPMEGA_TE_MXFP8_COMPACT_COLUMNWISE_BACKWARD": _bool(
                    profile.precision.mxfp8_compact_columnwise_backward
                ),
                "CPPMEGA_FLASHINFER_MXFP8_RUNNER": (
                    profile.precision.mxfp8_flashinfer_runner
                ),
                "CPPMEGA_FLASHINFER_MXFP8_TACTIC": str(
                    profile.precision.mxfp8_flashinfer_tactic
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
    if args.moe_flex_dispatcher_backend is not None:
        profile.model.moe_flex_dispatcher_backend = args.moe_flex_dispatcher_backend
    if args.moe_router_dtype is not None:
        profile.model.moe_router_dtype = None if args.moe_router_dtype == "none" else args.moe_router_dtype
    if args.dsa_indexer_loss_coeff is not None:
        profile.model.dsa_indexer_loss_coeff = args.dsa_indexer_loss_coeff
    if args.mtp_ce_kernel is not None:
        profile.runtime.mtp_ce_kernel = args.mtp_ce_kernel
        profile.runtime.acknowledge_liger_mtp_ce_deprecated = args.mtp_ce_kernel == "liger"
    if args.cce_fuse_main_mtp_ce is not None:
        profile.runtime.cce_fuse_main_mtp_ce = args.cce_fuse_main_mtp_ce
    if args.noconv_mamba_chunk_size is not None:
        profile.runtime.noconv_mamba_chunk_size = _validate_noconv_mamba_chunk_size(
            args.noconv_mamba_chunk_size
        )
    if args.fp8_recipe is not None:
        profile.precision.fp8_recipe = args.fp8_recipe
        if args.fp8_recipe == "mxfp8" and args.fp8_format is None:
            profile.precision.fp8_format = "e4m3"
    if args.fp8_format is not None:
        profile.precision.fp8_format = args.fp8_format
    if args.attention_backend is not None:
        profile.precision.attention_backend = args.attention_backend
    if args.mxfp8_bwd_backend is not None:
        profile.precision.mxfp8_bwd_backend = args.mxfp8_bwd_backend
    if args.mxfp8_transpose_emit_backend is not None:
        profile.precision.mxfp8_transpose_emit_backend = args.mxfp8_transpose_emit_backend
    if args.mxfp8_transpose_emit_swizzled is not None:
        profile.precision.mxfp8_transpose_emit_swizzled = args.mxfp8_transpose_emit_swizzled
    if args.mxfp8_transpose_emit_strict is not None:
        profile.precision.mxfp8_transpose_emit_strict = args.mxfp8_transpose_emit_strict
    if args.mxfp8_compact_columnwise_backward is not None:
        profile.precision.mxfp8_compact_columnwise_backward = (
            args.mxfp8_compact_columnwise_backward
        )
    if args.mxfp8_flashinfer_runner is not None:
        profile.precision.mxfp8_flashinfer_runner = args.mxfp8_flashinfer_runner
    if args.mxfp8_flashinfer_tactic is not None:
        if args.mxfp8_flashinfer_tactic < 0:
            raise ValueError("--mxfp8-flashinfer-tactic must be non-negative")
        profile.precision.mxfp8_flashinfer_tactic = args.mxfp8_flashinfer_tactic
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
    if args.muon_num_ns_steps is not None:
        if args.muon_num_ns_steps < 1:
            raise ValueError("--muon-num-ns-steps must be >= 1")
        profile.optimizer.muon_num_ns_steps = args.muon_num_ns_steps
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
    if args.nsys_capture_mode is not None:
        profile.profiling.nsys_profile = True
        profile.profiling.nsys_capture_mode = args.nsys_capture_mode
    if args.nsys_trace is not None:
        profile.profiling.nsys_profile = True
        profile.profiling.nsys_trace = args.nsys_trace
    if args.nsys_delay is not None:
        profile.profiling.nsys_profile = True
        profile.profiling.nsys_delay = args.nsys_delay
    if args.nsys_duration is not None:
        profile.profiling.nsys_profile = True
        profile.profiling.nsys_duration = args.nsys_duration
    if profile.profiling.nsys_capture_mode != "delay":
        profile.profiling.nsys_delay = 0
        profile.profiling.nsys_duration = 0
    elif profile.profiling.nsys_duration < 0:
        raise ValueError("nsys_capture_mode=delay requires --nsys-duration >= 0")
    if args.cuda_profile:
        profile.profiling.cuda_profile = True
    if args.cuda_profile_step_start is not None:
        profile.profiling.cuda_profile = True
        profile.profiling.cuda_profile_step_start = args.cuda_profile_step_start
    if args.cuda_profile_step_end is not None:
        profile.profiling.cuda_profile = True
        profile.profiling.cuda_profile_step_end = args.cuda_profile_step_end
    if profile.profiling.cuda_profile:
        start = profile.profiling.cuda_profile_step_start
        end = profile.profiling.cuda_profile_step_end
        if start >= end:
            raise ValueError(
                f"cuda_profile_step_start ({start}) must be < cuda_profile_step_end ({end})"
            )
    if profile.precision.fp8_recipe == "mxfp8" and profile.precision.fp8_format != "e4m3":
        raise ValueError(
            "fp8_recipe=mxfp8 requires fp8_format=e4m3 because the "
            "FlashInfer/CUTLASS MXFP8 path accepts E4M3 payloads only"
        )
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
        "--moe-flex-dispatcher-backend",
        choices=("deepep", "hybridep"),
        default=None,
    )
    parser.add_argument(
        "--moe-router-dtype",
        choices=("fp32", "none"),
        default=None,
    )
    parser.add_argument("--dsa-indexer-loss-coeff", type=float, default=None)
    parser.add_argument("--mtp-ce-kernel", choices=("native", "cce", "liger", "off"), default=None)
    cce_fuse = parser.add_mutually_exclusive_group()
    cce_fuse.add_argument(
        "--cce-fuse-main-mtp-ce",
        action="store_true",
        default=None,
        dest="cce_fuse_main_mtp_ce",
    )
    cce_fuse.add_argument(
        "--no-cce-fuse-main-mtp-ce",
        action="store_false",
        default=None,
        dest="cce_fuse_main_mtp_ce",
    )
    parser.add_argument(
        "--noconv-mamba-chunk-size",
        type=int,
        default=None,
        help="Override the no-conv Mamba SSD scan chunk size for cppmega specs.",
    )
    parser.add_argument("--fp8-format", choices=("hybrid", "e4m3"), default=None)
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise", "mxfp8"), default=None)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "flash", "fused", "unfused"),
        default=None,
    )
    parser.add_argument(
        "--mxfp8-bwd-backend",
        choices=("te_tn_adapter", "flashinfer_cutlass", "cutlass_native"),
        default=None,
    )
    parser.add_argument(
        "--mxfp8-transpose-emit-backend",
        choices=("auto", "te", "off"),
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
    compact_columnwise_backward = parser.add_mutually_exclusive_group()
    compact_columnwise_backward.add_argument(
        "--mxfp8-compact-columnwise-backward",
        action="store_true",
        default=None,
        dest="mxfp8_compact_columnwise_backward",
        help=(
            "Experimental: save dense TE Linear backward operands in original "
            "compact-columnwise MXFP8 form and route to cppmega direct backend."
        ),
    )
    compact_columnwise_backward.add_argument(
        "--no-mxfp8-compact-columnwise-backward",
        action="store_false",
        default=None,
        dest="mxfp8_compact_columnwise_backward",
    )
    parser.add_argument(
        "--mxfp8-flashinfer-runner",
        choices=("mm_mxfp8", "direct_tactic"),
        default=None,
        help="Select FlashInfer MXFP8 runner mode through the typed profile.",
    )
    parser.add_argument(
        "--mxfp8-flashinfer-tactic",
        type=int,
        default=None,
        help="CUTLASS direct runner tactic for explicit MXFP8 shape probes.",
    )
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--param-storage", choices=("auto", "bf16", "mxfp8"), default=None)
    parser.add_argument(
        "--muon-num-ns-steps",
        type=int,
        default=None,
        help="Override Newton-Schulz iterations for Muon in typed profiles.",
    )
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--mem-profile", action="store_true")
    parser.add_argument("--mem-profile-steps", type=int, default=None)
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--nsys-profile", action="store_true")
    parser.add_argument(
        "--nsys-capture-mode",
        choices=("full", "delay", "cudaProfilerApi"),
        default=None,
        help=(
            "Nsight Systems capture mode. Prefer full or delay; cudaProfilerApi "
            "is kept only for external repros because it drops kernel records on GB10."
        ),
    )
    parser.add_argument("--nsys-trace", default=None)
    parser.add_argument("--nsys-delay", type=int, default=None)
    parser.add_argument("--nsys-duration", type=int, default=None)
    parser.add_argument("--cuda-profile", action="store_true")
    parser.add_argument("--cuda-profile-step-start", type=int, default=None)
    parser.add_argument("--cuda-profile-step-end", type=int, default=None)


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
