"""NeMo 3 Nano-style recipe for NAM56R on 8×H200.

Produces complete Megatron CLI arg vectors following Nemotron Nano v2 training
patterns: kernel fusions enabled, distributed optimizer, gradient overlap,
proper batch sizing.

Two parallelism modes:
  - ``nemo_native``: TP=2, SP=True.  Uses Megatron built-in Mamba mixer.
    Maximum throughput, but loses Author Mamba3 / M²RNN custom features.
  - ``author_dp``: TP=1, PP=1, DP=8.  Uses Author Mamba3 / M²RNN via the
    cppmega selective mixer.  Slightly lower comm-overlap potential, but
    model fits easily on a single H200 (141 GiB).

All configs importable on macOS (no torch/megatron at import time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cppmega.recipes.nam56r_megatron import parse_nem_pattern

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAM56R_PATTERN = "AEMEAEMEAEMR"
NAM56R_DEPTH = 52
NAM56R_HIDDEN = 3584
NAM56R_FFN_HIDDEN = 18944
NAM56R_ATTN_HEADS = 56  # query heads; GQA 7:1 vs 8 kv-heads, head_dim=3584/56=64
NAM56R_KV_HEADS = 8
NAM56R_VOCAB = 65536
NAM56R_SEQ_LEN = 4096
NAM56R_ROPE_THETA = 500_000

# MoE defaults (from nanochat NAM56R spec)
MOE_NUM_EXPERTS = 16
MOE_TOPK = 4
MOE_FFN_HIDDEN = 896
MOE_SHARED_EXPERT_SIZE = 1024

# MLA defaults
MLA_Q_LORA_RANK = 64
MLA_KV_LORA_RANK = 64
MLA_QK_HEAD_DIM = 64
MLA_QK_POS_EMB_HEAD_DIM = 32
MLA_V_HEAD_DIM = 64

# Mamba defaults (Nemotron Nano-aligned)
# Megatron MambaMixer: nheads = hidden_size / head_dim = 3584/64 = 56 (expand=1)
# Author Mamba3: nheads = hidden * expand / head_dim = 3584*2/64 = 112 (expand=2)
MAMBA_NUM_HEADS = 56  # NAM56R spec; Author Mamba3 mode overrides to 112
MAMBA_STATE_DIM = 64
MAMBA_HEAD_DIM = 64
MAMBA_NUM_GROUPS = 8

# Optimal batch for 8×H200 at 4K seq
DEFAULT_MICRO_BATCH = 4
DEFAULT_GLOBAL_BATCH = 64


# ---------------------------------------------------------------------------
# Pattern builder
# ---------------------------------------------------------------------------

def build_nemo_hybrid_pattern(
    *,
    pattern: str = NAM56R_PATTERN,
    depth: int = NAM56R_DEPTH,
    mtp_depths: int = 0,
    use_moe: bool = True,
) -> str:
    """Build Megatron hybrid_layer_pattern from NAM56R symbols.

    Mapping:
      A → ``*`` (TransformerLayer: attention + MLP)
      E → ``E`` (MoE layer) when ``use_moe=True``, else ``-`` (MLP-only)
      M → ``M`` (Mamba layer)
      R → ``M`` (Mamba layer; runtime selection to M²RNN via selective mixer)
    """
    symbol_map = {
        "A": "*",
        "E": "E" if use_moe else "-",
        "M": "M",
        "R": "M",
        "D": "G",
        "G": "G",
    }
    mapped = []
    for sym in parse_nem_pattern(pattern, depth):
        out = symbol_map.get(sym)
        if out is None:
            raise ValueError(f"unsupported NAM56R symbol: {sym!r}")
        mapped.append(out)

    result = "".join(mapped)
    if mtp_depths > 0:
        result += "/" + "/".join("*-" for _ in range(mtp_depths))
    return result


# ---------------------------------------------------------------------------
# Recipe dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NAM56RNeMoRecipe:
    """Complete Megatron CLI arg set for NAM56R following NeMo Nano patterns."""

    # --- parallelism mode ---
    mode: Literal["nemo_native", "author_dp"] = "author_dp"

    # --- model dims ---
    hidden_size: int = NAM56R_HIDDEN
    ffn_hidden_size: int = NAM56R_FFN_HIDDEN
    num_attention_heads: int = NAM56R_ATTN_HEADS
    num_query_groups: int = NAM56R_KV_HEADS
    num_layers: int = NAM56R_DEPTH
    seq_length: int = NAM56R_SEQ_LEN
    max_position_embeddings: int = NAM56R_SEQ_LEN
    vocab_size: int = NAM56R_VOCAB

    # --- Mamba SSM dims ---
    mamba_num_heads: int = MAMBA_NUM_HEADS
    mamba_state_dim: int = MAMBA_STATE_DIM
    mamba_head_dim: int = MAMBA_HEAD_DIM
    mamba_num_groups: int = MAMBA_NUM_GROUPS

    # --- pattern ---
    pattern: str = NAM56R_PATTERN
    mtp_depths: int = 0

    # --- MoE ---
    use_moe: bool = True
    moe_num_experts: int = MOE_NUM_EXPERTS
    moe_router_topk: int = MOE_TOPK
    moe_ffn_hidden_size: int = MOE_FFN_HIDDEN
    moe_shared_expert_size: int = MOE_SHARED_EXPERT_SIZE

    # --- MLA ---
    use_mla: bool = True
    q_lora_rank: int = MLA_Q_LORA_RANK
    kv_lora_rank: int = MLA_KV_LORA_RANK
    qk_head_dim: int = MLA_QK_HEAD_DIM
    qk_pos_emb_head_dim: int = MLA_QK_POS_EMB_HEAD_DIM
    v_head_dim: int = MLA_V_HEAD_DIM

    # --- training ---
    micro_batch_size: int = DEFAULT_MICRO_BATCH
    global_batch_size: int = DEFAULT_GLOBAL_BATCH
    train_iters: int = 100
    lr: float = 3e-4
    min_lr: float = 3e-5
    lr_warmup_iters: int = 10
    lr_decay_style: str = "cosine"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    clip_grad: float = 1.0

    # --- precision ---
    precision: Literal["bf16", "fp8"] = "bf16"

    # --- data ---
    # NEVER use mock_data=True or NullTokenizer for benchmarks — produces
    # fake throughput numbers that don't match real training.
    mock_data: bool = False
    data_path: str = ""
    tokenizer_type: str = "HuggingFaceTokenizer"
    tokenizer_model: str = ""

    # --- checkpoint ---
    save_dir: str = ""
    load_dir: str = ""
    save_interval: int = 500
    log_interval: int = 1

    # --- transformer engine ---
    transformer_impl: Literal["transformer_engine", "local"] = "transformer_engine"

    # --- spec module (for cppmega custom layers) ---
    spec_module: str = ""
    spec_name: str = ""

    # --- NeMo throughput features ---
    use_cuda_graphs: bool = False
    use_optimizer_cuda_graph: bool = False
    use_full_moe_cuda_graph: bool = False  # full MoE graph needs drop-and-pad
    moe_expert_capacity_factor: float = 0.0  # 0 = dropless; >0 = drop-and-pad
    use_selective_recompute: bool = True  # core_attn recompute saves memory
    attention_backend: str = "auto"  # auto lets TE pick best (flash/fused/unfused)

    # --- custom embedding env flags ---
    ngram_hash_enabled: bool = False
    structure_enabled: bool = False

    # --- TP-aware Mamba3 mixer toggle ---
    # When True, swap AuthorMamba3Mixer for CppmegaMamba3TPMixer in
    # nam56r_full_spec via the CPPMEGA_MAMBA3_TP_MIXER=1 env var.  Required
    # for tensor-model-parallel-size > 1.
    use_tp_mamba3_mixer: bool = False

    def _tp(self) -> int:
        return 2 if self.mode == "nemo_native" else 1

    def _dp(self) -> int:
        return 8 // self._tp()

    def build_hybrid_pattern(self) -> str:
        return build_nemo_hybrid_pattern(
            pattern=self.pattern,
            depth=self.num_layers,
            mtp_depths=self.mtp_depths,
            use_moe=self.use_moe,
        )

    def to_args(self) -> list[str]:
        """Produce the complete Megatron CLI arg list."""
        tp = self._tp()
        args: list[str] = []

        # --- parallelism ---
        args.extend([
            "--tensor-model-parallel-size", str(tp),
            "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1",
        ])
        if tp > 1:
            args.append("--sequence-parallel")

        # --- distributed optimizer (NeMo standard) ---
        args.append("--use-distributed-optimizer")
        args.append("--overlap-grad-reduce")
        # overlap-param-gather works across DP dimension (not just TP),
        # overlaps param all-gather with forward compute in distributed optimizer
        args.append("--overlap-param-gather")

        # --- model architecture ---
        hybrid_pattern = self.build_hybrid_pattern()
        args.extend([
            "--hybrid-layer-pattern", hybrid_pattern,
            "--hidden-size", str(self.hidden_size),
            "--ffn-hidden-size", str(self.ffn_hidden_size),
            "--num-attention-heads", str(self.num_attention_heads),
            "--num-query-groups", str(self.num_query_groups),
            "--num-layers", str(self.num_layers),
            "--seq-length", str(self.seq_length),
            "--max-position-embeddings", str(self.max_position_embeddings),
            "--make-vocab-size-divisible-by", "128",
        ])

        # --- Mamba SSM ---
        args.extend([
            "--mamba-num-heads", str(self.mamba_num_heads),
            "--mamba-state-dim", str(self.mamba_state_dim),
            "--mamba-head-dim", str(self.mamba_head_dim),
            "--mamba-num-groups", str(self.mamba_num_groups),
        ])

        # --- position embeddings, norm, etc. ---
        args.extend([
            "--position-embedding-type", "rope",
            "--rotary-base", str(NAM56R_ROPE_THETA),
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--untie-embeddings-and-output-weights",
        ])

        # Gradient accumulation fusion requires APEX. Disable when APEX is
        # not available (our H200 env uses TE but not APEX).
        args.append("--no-gradient-accumulation-fusion")

        # --- NeMo 3 Nano throughput optimizations ---
        args.extend([
            "--cross-entropy-loss-fusion",
            "--attention-backend", self.attention_backend,
        ])
        # nam56r_full_spec uses WrappedTorchNorm which doesn't support
        # persist_layer_norm. The mamba3_te_stack_spec uses upstream TE norms.
        if self.spec_module and "nam56r_full_spec" in self.spec_module:
            args.append("--no-persist-layer-norm")
        else:
            args.append("--first-last-layers-bf16")

        # --- selective recomputation (saves ~40% activation memory) ---
        # Cannot use core_attn recompute with CUDA-graphed attention
        if self.use_selective_recompute and not self.use_cuda_graphs:
            args.extend([
                "--recompute-granularity", "selective",
            ])

        # --- CUDA graphs (TE-scoped, ~20% throughput boost) ---
        if self.use_cuda_graphs:
            args.extend(["--cuda-graph-impl", "transformer_engine"])
            if self.use_full_moe_cuda_graph:
                # Full MoE graph: captures entire MoE layer (needs drop-and-pad)
                args.extend(["--cuda-graph-scope", "attn", "mamba", "moe"])
            else:
                # Partial MoE graph: only router/preprocess, expert runs eagerly
                args.extend(["--cuda-graph-scope", "attn", "mamba", "moe_router", "moe_preprocess"])
            args.extend(["--cuda-graph-warmup-steps", "3"])
        if self.use_optimizer_cuda_graph:
            args.append("--optimizer-cuda-graph")

        # --- MLA ---
        if self.use_mla:
            args.extend([
                "--multi-latent-attention",
                "--q-lora-rank", str(self.q_lora_rank),
                "--kv-lora-rank", str(self.kv_lora_rank),
                "--qk-head-dim", str(self.qk_head_dim),
                "--qk-pos-emb-head-dim", str(self.qk_pos_emb_head_dim),
                "--v-head-dim", str(self.v_head_dim),
            ])

        # --- MoE ---
        if self.use_moe:
            args.extend([
                "--expert-model-parallel-size", "1",
                "--num-experts", str(self.moe_num_experts),
                "--moe-router-topk", str(self.moe_router_topk),
                "--moe-ffn-hidden-size", str(self.moe_ffn_hidden_size),
                "--moe-shared-expert-intermediate-size", str(self.moe_shared_expert_size),
                "--moe-grouped-gemm",
                "--moe-aux-loss-coeff", "0.0001",
                "--moe-router-score-function", "sigmoid",
                "--moe-router-enable-expert-bias",
                "--moe-router-dtype", "fp32",
                "--moe-token-dispatcher-type", "alltoall",
                "--moe-permute-fusion",
                "--moe-router-fusion",  # TE 2.7+ fused TopK routing
                "--moe-shared-expert-overlap",
            ])
            # Drop-and-pad mode (required for full MoE CUDA graph)
            if self.moe_expert_capacity_factor > 0:
                args.extend([
                    "--moe-expert-capacity-factor", str(self.moe_expert_capacity_factor),
                    "--moe-pad-expert-input-to-capacity",
                ])

        # --- MTP ---
        if self.mtp_depths > 0:
            args.extend(["--mtp-num-layers", str(self.mtp_depths)])

        # --- precision ---
        if self.precision == "bf16":
            args.append("--bf16")
        elif self.precision == "fp8":
            args.extend([
                "--bf16",
                "--fp8-format", "hybrid",
                "--fp8-recipe", "tensorwise",  # per-tensor current scaling (NeMo Nano v2 style)
            ])

        # --- training hyperparameters ---
        args.extend([
            "--micro-batch-size", str(self.micro_batch_size),
            "--global-batch-size", str(self.global_batch_size),
            "--train-iters", str(self.train_iters),
            "--lr", str(self.lr),
            "--min-lr", str(self.min_lr),
            "--lr-decay-style", self.lr_decay_style,
            "--lr-warmup-iters", str(min(self.lr_warmup_iters, max(1, self.train_iters - 1))),
            "--lr-decay-iters", str(self.train_iters),
            "--weight-decay", str(self.weight_decay),
            "--adam-beta1", str(self.adam_beta1),
            "--adam-beta2", str(self.adam_beta2),
            "--clip-grad", str(self.clip_grad),
        ])

        # --- Megatron core ---
        args.extend([
            "--use-mcore-models",
            "--transformer-impl", self.transformer_impl,
            "--is-hybrid-model",
            "--attention-softmax-in-fp32",
        ])
        if self.transformer_impl == "local":
            args.append("--no-masked-softmax-fusion")

        # --- data ---
        if self.mock_data:
            raise ValueError(
                "mock_data=True is FORBIDDEN. Use real data with "
                "HuggingFaceTokenizer. Set data_path and tokenizer_model."
            )
        if self.data_path:
            # data_path may contain multiple space-separated tokens
            # (e.g. "0.3 /path/a 0.7 /path/b"), Megatron expects separate args
            args.append("--data-path")
            args.extend(self.data_path.split())
            # All data to training (eval via separate eval run)
            args.extend(["--split", "100,0,0"])
        args.extend([
            "--tokenizer-type", self.tokenizer_type,
            "--vocab-size", str(self.vocab_size),
        ])
        if self.tokenizer_model:
            args.extend(["--tokenizer-model", self.tokenizer_model])

        # --- spec (custom cppmega layers) ---
        if self.spec_module and self.spec_name:
            args.extend(["--spec", self.spec_module, self.spec_name])

        # --- checkpoint ---
        if self.save_dir:
            args.extend(["--save", self.save_dir, "--save-interval", str(self.save_interval)])
        if self.load_dir:
            args.extend(["--load", self.load_dir])

        # --- logging ---
        args.extend([
            "--log-interval", str(self.log_interval),
            "--eval-interval", "50000000",
            "--eval-iters", "0",
        ])

        return args

    def to_shell_fragment(self) -> str:
        return " \\\n  ".join(self.to_args())

    def to_env_dict(self) -> dict[str, str]:
        """Environment variables for cppmega custom features and NeMo perf."""
        env: dict[str, str] = {
            # CUDA / NCCL
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_IB_SL": "1",
            "NCCL_AVOID_RECORD_STREAMS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_GRAPH_REGISTER": "0",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            # TE throughput (LayerNorm SM margins for comm overlap)
            "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
            "NVTE_NORM_FWD_USE_CUDNN": "1",
            "NVTE_NORM_BWD_USE_CUDNN": "1",
            # cppmega pattern
            "CPPMEGA_NEM_PATTERN": self.pattern,
            "CPPMEGA_LAYER_DEPTH": str(self.num_layers),
        }
        if self.ngram_hash_enabled:
            env["CPPMEGA_NGRAM_HASH_ENABLED"] = "1"
            env["CPPMEGA_NGRAM_HASH_ORDERS"] = "2,3"
            env["CPPMEGA_NGRAM_HASH_HEADS"] = "8"
            env["CPPMEGA_NGRAM_HASH_TABLE_SIZE"] = "500000"
            env["CPPMEGA_NGRAM_HASH_EMBED_DIM"] = "16"
        if self.structure_enabled:
            env["CPPMEGA_STRUCTURE_ENABLED"] = "1"
            env["CPPMEGA_STRUCTURE_COMPONENTS"] = "core"
        if self.use_tp_mamba3_mixer:
            env["CPPMEGA_MAMBA3_TP_MIXER"] = "1"
        return env


# ---------------------------------------------------------------------------
# Pre-built recipes
# ---------------------------------------------------------------------------

def nam56r_nemo_native_pretrain() -> NAM56RNeMoRecipe:
    """NeMo-native: Megatron built-in Mamba, TP=1, DP=8.

    NAM56R at 4.73B (3.03B active) fits on a single H200 (141 GiB).
    TP=1 eliminates tensor-parallel AllReduce overhead → higher throughput.
    Uses Megatron's standard mamba_stack_spec (required by mamba_builder).
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, DP=8
        use_moe=True,
        use_mla=False,  # MLA needs custom spec wiring
        spec_module="megatron.core.models.mamba.mamba_layer_specs",
        spec_name="mamba_stack_spec",
        micro_batch_size=4,
        global_batch_size=32,
        use_cuda_graphs=True,
    )


def nam56r_nemo_native_max_throughput() -> NAM56RNeMoRecipe:
    """Max throughput: FP8 + full MoE CUDA graph + drop-and-pad, GBS=128.

    Uses full MoE CUDA graph scope (captures entire MoE layer including expert
    computation) with drop-and-pad mode.  FP8 tensorwise + gradient accumulation
    (MBS=4, GBS=128 = 4x accum) pushes throughput past 200k tok/sec.

    FP8 requires nheads % 16 == 0, so use nheads=64 (vs 56 in BF16 recipe).
    Achieves 211k tok/sec / 50.1% MFU on 8×H200.
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, DP=8
        use_moe=True,
        use_mla=False,
        spec_module="megatron.core.models.mamba.mamba_layer_specs",
        spec_name="mamba_stack_spec",
        mamba_num_heads=64,  # FP8-aligned (multiple of 16)
        micro_batch_size=5,
        global_batch_size=320,  # 8x grad accum: 8 GPUs * MBS 5 * 8 accum
        precision="fp8",
        use_cuda_graphs=True,
        use_full_moe_cuda_graph=True,  # capture entire MoE in graph
        moe_expert_capacity_factor=1.5,  # drop-and-pad for full MoE graph
        use_selective_recompute=False,  # no recompute for max throughput
    )


def nam56r_author_dp_pretrain() -> NAM56RNeMoRecipe:
    """Author-Mamba3 mode: TP=1, DP=8, TE-optimized NAM56R stack spec.

    Uses Author Mamba3 + M²RNN via selective mixer, but keeps ALL TE
    submodules (norms, attention, MoE) from upstream for maximum throughput.
    Requires TP=1. Author Mamba3 uses expand=2, so mamba_num_heads=112.
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",
        use_moe=True,
        use_mla=False,  # TE spec uses upstream attention (not MLA)
        spec_module="cppmega.megatron.nam56r_te_spec",
        spec_name="build_cppmega_nam56r_te_stack_spec",
        mamba_num_heads=112,  # Author Mamba3: hidden*expand/headdim = 3584*2/64
        micro_batch_size=4,
        global_batch_size=32,
    )


def nam56r_mamba3_te_pretrain() -> NAM56RNeMoRecipe:
    """Mamba-3 TE mode: TP=1, DP=8, CppMegaMamba3Mixer + upstream TE submodules.

    Uses CppMegaMamba3Mixer (QK-Norm, learnable B/C bias) as the Mamba mixer,
    while keeping ALL other TE-optimized submodules from upstream.

    DEPRECATED: Use nam56r_mamba3_native_pretrain() instead, which uses the
    correct nheads=56 (same as nemo_native baseline) for apples-to-apples
    comparison. This recipe used nheads=112 (Author expand=2 convention).
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",
        use_moe=True,
        use_mla=False,
        spec_module="cppmega.megatron.mamba3_te_stack_spec",
        spec_name="cppmega_mamba3_te_stack_spec",
        mamba_num_heads=112,
        micro_batch_size=4,
        global_batch_size=32,
        use_cuda_graphs=True,
    )


def nam56r_mamba3_native_pretrain() -> NAM56RNeMoRecipe:
    """Mamba-3 native mode: CppMegaMamba3Mixer with native SSD kernel.

    Same architecture as nemo_native (nheads=56, d_inner=3584) but with
    Mamba3 features: QK-Norm on B/C, learnable B/C bias.  Uses
    mamba_chunk_scan_combined (not Author kernels) for CUDA graph compat.

    Env vars to control features:
      CPPMEGA_MAMBA3_QKNORM=1  (default on)
      CPPMEGA_MAMBA3_BIAS=1    (default on)
      CPPMEGA_MAMBA3_DATA_DEP_A=0  (default off)
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, DP=8
        use_moe=True,
        use_mla=False,
        spec_module="cppmega.megatron.mamba3_te_stack_spec",
        spec_name="cppmega_mamba3_te_stack_spec",
        mamba_num_heads=56,  # same as nemo_native baseline
        micro_batch_size=4,
        global_batch_size=32,
        use_cuda_graphs=True,
    )


def nam56r_mamba3_native_max_throughput() -> NAM56RNeMoRecipe:
    """Mamba3 max throughput: FP8 + CUDA graphs + MBS=5 + GBS=320.

    Same optimization stack as nam56r_nemo_native_max_throughput() but with
    CppMegaMamba3Mixer (QK-Norm, B/C bias).  Target: match 211k tok/sec.

    Uses nheads=64 for FP8 alignment (multiple of 16).
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, DP=8
        use_moe=True,
        use_mla=False,
        spec_module="cppmega.megatron.mamba3_te_stack_spec",
        spec_name="cppmega_mamba3_te_stack_spec",
        mamba_num_heads=64,  # FP8-aligned
        micro_batch_size=5,
        global_batch_size=320,
        precision="fp8",
        use_cuda_graphs=True,
        use_full_moe_cuda_graph=True,
        moe_expert_capacity_factor=1.5,
        use_selective_recompute=False,
    )


def nam56r_noconv_max_throughput() -> NAM56RNeMoRecipe:
    """Branch B max throughput: NoConvMamba3BCMixer + fused Triton M²RNN + FP8.

    Uses vanilla ``mamba_chunk_scan_combined`` with QK-Norm + B/C bias
    preprocessing (NoConvMamba3BCMixer) on M-layers plus the fused Triton
    M²RNN kernel on R-layer positions.  This is the full NAM56R architecture
    with all architectural features enabled at production speed.

    Same optimization stack as ``nam56r_nemo_native_max_throughput()`` (FP8,
    full MoE CUDA graph, drop-and-pad, GBS=320, MBS=5) but with Mamba3
    features + M²RNN.  Target: 250k tok/sec on 8×H200 (>211k baseline).
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, PP=1, DP=8
        use_moe=True,
        use_mla=True,  # MLA with cuDNN 9.20 fused attention
        spec_module="cppmega.megatron.nam56r_noconv_spec",
        spec_name="build_cppmega_nam56r_noconv_stack_spec",
        mamba_num_heads=64,  # FP8-aligned (multiple of 16)
        micro_batch_size=5,
        global_batch_size=320,  # 8x grad accum: 8 GPUs × MBS 5 × 8 accum
        precision="fp8",
        use_cuda_graphs=True,
        use_full_moe_cuda_graph=True,  # capture entire MoE in graph
        moe_expert_capacity_factor=1.5,  # drop-and-pad for full MoE graph
        use_selective_recompute=False,  # no recompute for max throughput
    )


def nam56r_noconv_pretrain() -> NAM56RNeMoRecipe:
    """Branch B BF16 pretrain: NoConvMamba3BCMixer + fused Triton M²RNN.

    Conservative config for convergence testing with real data before
    enabling FP8.  MBS=4 GBS=32, BF16, te_attn CUDA graphs.
    """
    return NAM56RNeMoRecipe(
        mode="author_dp",  # TP=1, PP=1, DP=8
        use_moe=True,
        use_mla=True,
        spec_module="cppmega.megatron.nam56r_noconv_spec",
        spec_name="build_cppmega_nam56r_noconv_stack_spec",
        mamba_num_heads=56,  # BF16 — nheads=56 matches baseline
        micro_batch_size=4,
        global_batch_size=32,
        precision="bf16",
        use_cuda_graphs=True,
    )


def nam56r_smoke_test() -> NAM56RNeMoRecipe:
    """Minimal smoke test: small model, 2 iterations."""
    return NAM56RNeMoRecipe(
        mode="author_dp",
        hidden_size=256,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        num_query_groups=4,
        num_layers=NAM56R_DEPTH,
        seq_length=128,
        max_position_embeddings=128,
        mamba_num_heads=4,
        mamba_state_dim=16,
        mamba_head_dim=64,
        mamba_num_groups=2,
        use_moe=False,
        use_mla=False,
        micro_batch_size=1,
        global_batch_size=8,
        train_iters=2,
        precision="bf16",
        spec_module="cppmega.megatron.nam56r_full_spec",
        spec_name="build_cppmega_nam56r_full_stack_spec",
    )
