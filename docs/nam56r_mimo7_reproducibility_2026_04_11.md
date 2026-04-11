# NAM56R MIMO 7/7 Training Reproducibility Guide

**Date:** 2026-04-11
**Author:** David Gornshtein
**Status:** Two configs validated (56k + 112k tok/sec on 8x H200)

---

## Quick start: reproduce 112k tok/sec on a fresh H200x8

```bash
# 1. SSH into the europe H200x8 machine
gcloud compute ssh h200_1 --zone LOCATION_2

# 2. Activate venv + set paths
source /home/dave/cppmega-root/cppmega-venv/bin/activate
export PYTHONPATH="/home/dave/cppmega-root/cppmega:/home/dave/cppmega-root/megatron-lm"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# 3. Checkout exact commit
cd /home/dave/cppmega-root/cppmega && git checkout f21329278f5672f0c584e63275bd7d8f14ceb46e

# 4. Set MIMO + shim env vars
export CPPMEGA_MAMBA3_MIMO=1
export CPPMEGA_MAMBA_NUM_GROUPS=8
export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"
export CPPMEGA_LAYER_DEPTH=52
export CPPMEGA_R_LAYER_INDICES="12,24,36,48"
export TILELANG_EXECUTION_BACKEND=cython
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TRITON_CACHE_DIR="/home/dave/cppmega-root/.triton-cache"

# 5. Build launch workdir, install shim, run VPP PP=2 config
WORKDIR=$(mktemp -d /tmp/nam56r-mimo-repro.XXXXXX)
cp /home/dave/cppmega-root/megatron-lm/pretrain_mamba.py "${WORKDIR}/"
cp /home/dave/cppmega-root/cppmega/scripts/cppmega_fp8_shim.py "${WORKDIR}/"
# Create mamba_builders.py and model_provider.py (see Section 4B below)
# Then run the torchrun command from Section 4B
```

---

## 1. Git state

```
Commit: f21329278f5672f0c584e63275bd7d8f14ceb46e
Branch: main
Date:   2026-04-11
```

Recent history:

```
f213292 feat: implement CuTe DSL MIMO support, add FP8 shim utilities, and update Fast Hadamard Transform research and build scripts.
9f367db feat: implement Mamba3 MIMO cuTile parity and performance validation suite on B200
bbf5942 feat: introduce nam56r_noconv_spec and NoConvMamba3BCMixer to support selective Mamba-2 SSM training
651816b feat: add Mamba3 TE modules, M2RNN chunking, noconv mixer, and supporting tests and docs
cd9b226 feat: implement NAM56R full stack architecture with selective MLA/DSA and Mamba/M2RNN layer support
33c5538 refactor: remove legacy remote patch scripts and tests, and add new ngram hash smoke test
0fa1982 feat: implement modular structure and ngram-hash embedding patches for Megatron with supporting test suites and launch scripts
630b26a feat: add configuration modules and test coverage for Megatron feature plans and remote H200 setup scripts
c7a3bd3 feat: implement Engram and NgramHash configuration and recipe surface in cppmega.features.engram
35b4f46 chore: initialize virtual environment dependencies and site-packages for Python 3.13
```

---

## 2. Stack versions

All versions verified on the H200x8 bench machines as of 2026-04-10/11.

| Component | Version | Notes |
|---|---|---|
| **Python** | 3.13.12 | uv-managed CPython |
| **torch** | 2.12.0.dev20260409+cu132 | Nightly, cu132 |
| **CUDA toolkit** | 13.2 | Driver 595.58.03 |
| **NVIDIA driver** | 595.58.03 | |
| **transformer_engine** | 2.13.0 | `_cu13` + `_torch` packages |
| **mamba-ssm** | 2.3.1 | From state-spaces/mamba commit `31f3d7ba`, PR #909 patch applied |
| **megatron-core** | 0.18rc0 | Commit `e40feed4a` |
| **flash-attn** | 2.8.3 | |
| **tilelang** | 0.1.8 | CRITICAL: requires `apache-tvm-ffi<0.1.10` |
| **apache-tvm-ffi** | 0.1.9 | NOT 0.1.10 (breaks TileLang via `_ObjectSlotsMeta`) |
| **fast-hadamard-transform** | from GitHub HEAD | PyPI sdist 1.0.4 is broken (missing csrc) |
| **cuDNN** | 9.20 (venv wheel) | System has 9.10.2; venv must take priority via `LD_LIBRARY_PATH` |
| **cppmega** | 0.1.0 | This repo |
| **NeMo** | 2.7.2 | bench3 only; europe does not have NeMo |

### Critical version pins

1. **tvm-ffi < 0.1.10** -- Version 0.1.10 introduced `__slots__=()` enforcement via `_ObjectSlotsMeta` that breaks TileLang's `TVMDerivedObject`. TileLang main pinned this on 2026-04-08.

2. **cuDNN venv override** -- TE 2.13 calls `ctypes.CDLL("libcudnn.so")` unversioned. System resolves to cuDNN 9.10.2, which lacks the `(d_qk=96, d_v=64, sm_90)` MLA engine config. Fix requires BOTH:
   ```bash
   # One-time symlink
   ln -sf libcudnn.so.9 ${VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so
   # Per-launch
   export LD_LIBRARY_PATH="${VENV}/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
   ```

3. **fast-hadamard-transform** -- Install from GitHub, not PyPI:
   ```bash
   pip install --no-build-isolation 'git+https://github.com/Dao-AILab/fast-hadamard-transform.git'
   ```

### Bench machine locations

| Machine | Zone | Venv | Data |
|---|---|---|---|
| **h200_1** (primary) | LOCATION_1 | `/mnt/data/venv` | `/mnt/data/data/megatron/` |
| **h200_1** (europe) | LOCATION_2 | `/home/dave/cppmega-root/cppmega-venv` | `/home/dave/cppmega-root/data/megatron/` |

---

## 3. Model architecture

### Overview

| Dimension | Value |
|---|---|
| **Total parameters** | ~4.73B |
| **Active parameters** | ~3.03B (excluding inactive MoE experts) |
| **Layers** | 52 main + 1 MTP |
| **Pattern** | `AEMEAEMEAEMR` tiled to depth 52 |
| **Hidden size** | 3584 |
| **FFN hidden size** | 18944 |
| **Vocab size** | 65536 (training) / 131072 (padded, divisible by 128) |
| **Sequence length** | 4096 |
| **Precision** | BF16 (scan kernels fp32 internally) |

### Layer composition (pattern `AEMEAEMEAEMR` x 52)

| Symbol | Count | Component |
|---|---|---|
| **A** (Attention) | 13 | MLA on most; DSA on A-ranks 0, 4, 8 (layers 1, 17, 33) |
| **E** (Expert/MoE) | 22 | MoE FFN layers |
| **M** (Mamba) | 13 | AuthorMamba3Mixer with MIMO R=4 |
| **R** (Recurrent) | 4 | CppMegaM2RNNMixer (fused Triton kernel) |

Total Mamba-like layers (M + R): 17

### MLA configuration

| Parameter | Value |
|---|---|
| `--multi-latent-attention` | enabled |
| `--q-lora-rank` | 64 |
| `--kv-lora-rank` | 64 |
| `--qk-head-dim` | 64 |
| `--qk-pos-emb-head-dim` | 32 |
| `--v-head-dim` | 64 |
| `--num-attention-heads` | 28 |
| `--num-query-groups` | 8 (GQA) |

### DSA (DeepSeek Sparse Attention) configuration

- Enabled on A-layer ranks 0, 4, 8 (3 of 13 A-layers)
- Controlled by `CPPMEGA_DSA_A_LAYER_RANKS` env var or `dsa_a_layer_ranks` param
- Uses upstream Megatron `get_dsa_module_spec_for_backend`
- DSA indexer: `n_heads=8, head_dim=64, topk=16, loss_coeff=0.0`
- Requires `fast-hadamard-transform` (installed from GitHub, not PyPI)

### Mamba3 configuration (M-layers)

| Parameter | Value |
|---|---|
| Mixer class | `AuthorMamba3Mixer` (wraps `mamba_ssm.modules.mamba3.Mamba3`) |
| `d_model` | 3584 |
| `d_state` | 128 |
| `expand` | 2 |
| `headdim` | 64 |
| `ngroups` | 8 |
| `d_inner` | 7168 (= 3584 * 2) |
| `num_heads` | 112 (= 7168 / 64) |
| `rope_fraction` | 0.5 |
| **MIMO** | enabled (`cppmega_mamba3_is_mimo=True`) |
| **MIMO rank** | 4 |
| **chunk_size** | 16 (constraint: `chunk_size * mimo_rank <= 64`) |
| `is_outproj_norm` | False |
| TileLang backend | `TILELANG_EXECUTION_BACKEND=cython` (NOT nvrtc on cu13.2) |

The MIMO configuration is injected at runtime via `cppmega_fp8_shim.py` patch (2), which monkey-patches `TransformerConfig.__post_init__` to set `cppmega_mamba3_is_mimo=True`, `cppmega_mamba3_mimo_rank=4`, and `cppmega_mamba3_chunk_size=16`.

The `AuthorMamba3Mixer` includes an explicit `nn.RMSNorm` before `Mamba3.in_proj` to compensate for the `IdentityOp` norm in `MambaLayer` (standard in `mamba_stack_spec`). Without this norm, residual magnitudes grow unbounded through 52 layers, producing NaN grad_norm from iteration 1.

### MoE configuration (E-layers)

| Parameter | Value |
|---|---|
| `--num-experts` | 16 |
| `--moe-router-topk` | 4 |
| `--moe-ffn-hidden-size` | 896 |
| `--moe-shared-expert-intermediate-size` | 1024 |
| `--moe-grouped-gemm` | enabled |
| `--expert-model-parallel-size` | 1 |
| Routing | dropless top-4 |

### MTP (Multi-Token Prediction) configuration

| Parameter | Value |
|---|---|
| `--mtp-num-layers` | 1 |
| MTP mode | hybrid |
| Pattern suffix | `/*-` (appended to main 52-layer pattern) |

### M2RNN configuration (R-layers)

| Parameter | Value |
|---|---|
| Mixer class | `CppMegaM2RNNMixer` |
| R-layer indices (1-indexed) | 12, 24, 36, 48 |
| `k_head_dim` | 64 |
| `v_head_dim` | 16 |
| `conv_kernel` | 4 |
| `use_residual` | True |
| Kernel | Fused Triton (`m2rnn_scan_triton`) with inline PTX `tanh.approx.f32` SFU op |
| Fallback | `CPPMEGA_M2RNN_KERNEL=torch` for debug only (460x slower) |

The Triton kernel keeps the hidden state `(K=64, V=16)` in registers for the entire 4096-token sequence scan. Backward uses checkpoint recomputation by default (`CPPMEGA_M2RNN_SAVE_HNEW=0`) to save ~1.4 GB/layer.

### Hybrid layer pattern (Megatron format)

The NEM pattern `AEMEAEMEAEMR` at depth 52 with 1 MTP layer translates to Megatron's `--hybrid-layer-pattern` as:

```
*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME/*-
```

Where `*` = attention, `E` = MoE, `M` = Mamba-like (both M and R symbols map to M in Megatron format; the spec internally dispatches M vs R via `CppMegaSelectiveMambaMixer`).

---

## 4. Validated training configurations

### 4A. Config A -- PP=1 baseline (56,280 tok/sec)

**Machine:** h200_1 (LOCATION_1, 8x H200)

**Result:** 56,280 tok/sec, 1164 ms/iter (median iters 10-28), 30/30 iters clean, loss 11.75 -> 4.82

#### Environment variables

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export TRITON_CACHE_DIR="/mnt/data/.triton-cache"
export CPPMEGA_NEM_PATTERN="AEMEAEMEAEMR"
export CPPMEGA_LAYER_DEPTH=52
export CPPMEGA_R_LAYER_INDICES="12,24,36,48"
export CPPMEGA_MAMBA3_MIMO=1
export CPPMEGA_MAMBA_NUM_GROUPS=8
export TILELANG_EXECUTION_BACKEND=cython
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH="/mnt/data/venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
```

#### Megatron command

```bash
python -m torch.distributed.run --nproc_per_node=8 pretrain_mamba.py \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /mnt/data/tokenizer \
  --data-path /mnt/data/data/megatron/clang_semantic_4k_v10_train_text_document \
  --vocab-size 65536 \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --hybrid-layer-pattern "*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME/*-" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --train-iters 30 \
  --eval-interval 50000000 \
  --eval-iters 0 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --spec cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec \
  --multi-latent-attention \
  --q-lora-rank 64 \
  --kv-lora-rank 64 \
  --qk-head-dim 64 \
  --qk-pos-emb-head-dim 32 \
  --v-head-dim 64 \
  --mtp-num-layers 1 \
  --expert-model-parallel-size 1 \
  --num-experts 16 \
  --moe-router-topk 4 \
  --moe-ffn-hidden-size 896 \
  --moe-shared-expert-intermediate-size 1024 \
  --moe-grouped-gemm \
  --log-interval 1 \
  --save-interval 1
```

#### Required workdir files

The launch creates a temporary working directory with:

1. **`pretrain_mamba.py`** -- copied from `${MEGATRON_LM}/pretrain_mamba.py`
2. **`cppmega_fp8_shim.py`** -- copied from `scripts/cppmega_fp8_shim.py` and imported before training starts
3. **`mamba_builders.py`** -- one-liner:
   ```python
   from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder
   ```
4. **`model_provider.py`** -- standard Megatron model provider that delegates to `mamba_builder`

#### Memory

Peak GPU memory per rank: 103 GB / 140 GB

#### Iter timings

| Iter | Time (ms) | Note |
|---|---|---|
| 1 | 96,171 | TileLang JIT first compile |
| 2 | 48,854 | TileLang bwd kernel JIT |
| 3-4 | 1207 / 1197 | Warm-up |
| 5+ | ~1160 | Steady-state |

---

### 4B. Config B -- VPP PP=2 (112,152 tok/sec)

**Machine:** h200_1 (LOCATION_2, 8x H200)

**Result:** 112,152 tok/sec, 2,337 ms/iter (mean), 30/30 iters clean, loss 11.92 -> 2.73

#### Key differences from Config A

| Dimension | Config A | Config B |
|---|---|---|
| PP | 1 | **2** |
| VPP | 1 | **2** (13 layers per virtual stage) |
| DP | 8 | **4** |
| MBS | 2 | **4** |
| GBS | 16 | **64** |
| `--no-rope-fusion` | not needed | **required** (fused MLA RoPE crashes at PP>1) |
| Layers per GPU | 52 | 26 |

#### Pipeline pattern (VPP)

```
*EME*EME*EMM*|EME*EME*EMM*E|ME*EME*EMM*EM|E*EME*EMM*EME/*-
```

4 virtual pipeline chunks (PP=2 x VPP=2 = 4 segments of 13 layers each) + 1 MTP depth layer.

#### Environment variables

Same as Config A, plus:

```bash
# VPP-specific
export CPPMEGA_PP_SIZE=2
export CPPMEGA_MICRO_BATCH_SIZE=4
export CPPMEGA_GLOBAL_BATCH_SIZE=64
```

#### Megatron command

```bash
python -m torch.distributed.run --nproc_per_node=8 pretrain_mamba.py \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /home/dave/cppmega-root/data/tokenizer \
  --data-path /home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train_text_document \
  --vocab-size 65536 \
  --make-vocab-size-divisible-by 128 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --num-layers-per-virtual-pipeline-stage 13 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --use-distributed-optimizer \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --no-masked-softmax-fusion \
  --no-rope-fusion \
  --hybrid-layer-pattern "*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME*EME*EMM*EME/*-" \
  --hidden-size 3584 \
  --ffn-hidden-size 18944 \
  --num-attention-heads 28 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size 4 \
  --global-batch-size 64 \
  --train-iters 30 \
  --eval-interval 50000000 \
  --eval-iters 0 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style constant \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --bf16 \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --spec cppmega.megatron.nam56r_full_spec build_cppmega_nam56r_full_stack_spec \
  --multi-latent-attention \
  --q-lora-rank 64 \
  --kv-lora-rank 64 \
  --qk-head-dim 64 \
  --qk-pos-emb-head-dim 32 \
  --v-head-dim 64 \
  --mtp-num-layers 1 \
  --expert-model-parallel-size 1 \
  --num-experts 16 \
  --moe-router-topk 4 \
  --moe-ffn-hidden-size 896 \
  --moe-shared-expert-intermediate-size 1024 \
  --moe-grouped-gemm \
  --log-interval 1 \
  --save-interval 1
```

#### Memory

| Rank | Role | Peak allocated |
|---|---|---|
| 0 | PP stage 0 (embedding) | 118 GB |
| 4 | PP stage 1 (output head) | 94 GB |

MBS=8 OOMs at 132/140 GB. MBS=4 is the PP=2 sweet spot.

---

## 5. MFU calculation

### Formula

```
MFU = model_flops_per_iter / (peak_flops * iter_time)
model_flops_per_iter = 6 * N * tokens_per_iter
tokens_per_iter = GBS * seq_length
peak_flops = num_gpus * H200_bf16_peak
```

H200 SXM BF16 peak = 989 TFLOPS per GPU.
Cluster peak (8 GPUs) = 7,912 TFLOPS = 7.912 PFLOPS.

Note: The `6N` formula is an approximation for standard transformers. NAM56R is a hybrid Mamba-Transformer; Mamba layers have different FLOP accounting (scan is O(S*D^2) vs attention O(S^2*D)), but `6N` serves as a lower-bound estimate.

### Config A (PP=1, 56,280 tok/sec)

| Metric | Total params (4.73B) | Active params (3.03B) |
|---|---|---|
| tokens_per_iter | 65,536 | 65,536 |
| model_flops_per_iter | 1.860 PFLOP | 1.191 PFLOP |
| iter_time | 1.164 s | 1.164 s |
| **MFU** | **20.2%** | **12.9%** |

### Config B (VPP PP=2, 112,152 tok/sec)

| Metric | Total params (4.73B) | Active params (3.03B) |
|---|---|---|
| tokens_per_iter | 262,144 | 262,144 |
| model_flops_per_iter | 7.440 PFLOP | 4.766 PFLOP |
| iter_time | 2.337 s | 2.337 s |
| **MFU** | **40.2%** | **25.8%** |

The 2x throughput jump from PP=1 to VPP PP=2 comes from 4x more tokens per iter (GBS 16->64) with only 2x longer iter time. VPP reduces pipeline bubble from ~50% (PP=2 naive) to ~6%.

---

## 6. Data

| Dimension | Value |
|---|---|
| **Dataset** | `clang_semantic_4k_v10_train` (Megatron `.bin`/`.idx` format) |
| **Path (bench3)** | `/mnt/data/data/megatron/clang_semantic_4k_v10_train_text_document` |
| **Path (europe)** | `/home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train_text_document` |
| **Tokenizer type** | `HuggingFaceTokenizer` |
| **Tokenizer path (bench3)** | `/mnt/data/tokenizer/` (HF dir with `tokenizer.json` + `tokenizer_config.json`) |
| **Tokenizer path (europe)** | `/home/dave/cppmega-root/data/tokenizer/` |
| **Vocab size** | 65536 (padded to 131072 with `--make-vocab-size-divisible-by 128` for TP efficiency) |
| **Sequence length** | 4096 |

For mock-data validation runs, replace data/tokenizer flags with:
```
--mock-data --tokenizer-type NullTokenizer --vocab-size 131072
```

---

## 7. Known blockers and workarounds (shim patches 1-5)

The canonical shim lives at `scripts/cppmega_fp8_shim.py`. It must be imported before `pretrain_mamba.py` runs. It installs 5 monkey-patches:

### Patch (1): `deprecate_inference_params` compatibility alias

**Bug:** `cppmega.megatron.mamba3_te_mixer` imports `deprecate_inference_params` from `megatron.core.inference.contexts.static_context`, but in megatron-core 0.18rc0 the function lives in `megatron.core.utils`.

**Fix:** Alias the function into `static_context` module at import time.

### Patch (2): MIMO config injection via `TransformerConfig.__post_init__`

**Bug:** Megatron's `TransformerConfig` has no fields for `cppmega_mamba3_is_mimo`, `cppmega_mamba3_mimo_rank`, or `cppmega_mamba3_chunk_size`. These must be set at config init time.

**Fix:** Patches `__post_init__` to set `is_mimo=True`, `mimo_rank=4`, `chunk_size=16` when `CPPMEGA_MAMBA3_MIMO=1` env var is set. Also optionally overrides `mamba_num_groups` from `CPPMEGA_MAMBA_NUM_GROUPS`.

### Patch (3): `TransformerConfig.__getattr__` for `cppmega_mamba3_*` attributes

**Bug:** `getattr(config, "cppmega_mamba3_rope_fraction", 0.5)` returns `None` instead of the default 0.5 because the upstream `TransformerConfig.__getattr__` returns `None` for unknown attributes instead of raising `AttributeError`.

**Fix:** Installs a `__getattr__` that raises `AttributeError` for any `cppmega_mamba3_*` name, allowing `getattr()` defaults to work correctly.

### Patch (4): `Float16Module` one-shot Mamba3 fp32 param restore

**Bug:** `Float16Module.__init__` casts ALL parameters to bf16 indiscriminately. Mamba3 initializes `dt_bias`, `D`, `B_bias`, `C_bias`, `mimo_x`, `mimo_z`, `mimo_o` in fp32 deliberately because the TileLang `mamba_mimo_fwd_kernel` requires them in fp32. The blanket bf16 cast breaks the kernel contract, producing NaN or dtype-mismatch crashes.

**Fix:** Patches `Float16Module.__init__` to walk all `Mamba3` submodules after the blanket cast and restore the 7 fp32 parameters back to fp32. This is a one-shot patch at model init (replaces a deprecated per-forward pre-hook that cost 305 ms/iter = 25.7% of baseline iter time).

### Patch (5): CUDA graph compatibility -- bypass `_broadcast_cu_seqlens` at TP=1

**Bug:** When CUDA graphs capture `forward_step`, `_broadcast_cu_seqlens` does `torch.tensor(n, dtype=int64, device=cuda)` from a Python int, which requires pinned source memory under graph capture (not set up). At TP=1 the broadcast is a no-op anyway.

**Fix:** Bypasses `_broadcast_cu_seqlens` entirely when `get_tensor_model_parallel_world_size() == 1`.

### Additional blockers (not in shim)

These are environment/dependency issues, not runtime patches:

| Blocker | Fix |
|---|---|
| `fast-hadamard-transform` PyPI sdist broken (1.0.4) | Install from GitHub: `pip install --no-build-isolation 'git+https://github.com/Dao-AILab/fast-hadamard-transform.git'` |
| TileLang `nvrtc` backend broken on CUDA 13.2 | Set `TILELANG_EXECUTION_BACKEND=cython` |
| Fused MLA RoPE + PP>1 = `cudaErrorIllegalAddress` | Set `--no-rope-fusion` (Config B only) |
| cuDNN version collision (system 9.10.2 vs venv 9.20) | Symlink + `LD_LIBRARY_PATH` override (see Section 2) |

---

## 8. What didn't work

### FP8 ineffective at H=3584

FP8 was tested on all 4 Mamba3 paths (A/B/C/D). While FP8 training converges correctly, the throughput gain is marginal at hidden_size=3584 because FP8 WGMMA benefits scale with matrix size. The 251 ms BF16 WGMMA GEMM bucket in the baseline would roughly halve under FP8, but at H=3584 the matrices are not large enough for the FP8 casting overhead to be fully amortized. FP8 is projected to help more at TP=2 where per-GPU matrix dimensions are larger.

### Per-forward fp32 repair hook (deprecated, replaced by Patch 4)

The original workaround for the `Float16Module` bf16-cast bug was a `register_forward_pre_hook` that called `.data.float()` on 7 Mamba3 parameters every forward pass. Per nsys profile, this resulted in ~400 D2D copies per iter (7 params x ~16 Mamba layers x fwd+bwd+optimizer) = **305 ms/iter (25.7% of baseline)**. The one-shot `Float16Module.__init__` patch (Patch 4) eliminates this entirely.

### TF32 optimizer regression

The nsys profile showed 142 ms/iter (12%) spent in `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_128x128x8` -- Ampere-era fp32 fallback GEMMs from `nvte_multi_tensor_gemm` in the optimizer. These should use TF32 (`torch.backends.cuda.matmul.allow_tf32=True`) or H200 sm_90 tensor-core paths, but the optimizer code is hardcoded to pure fp32. This wastes ~3% of H200 peak utilization on Ampere FFMA instead of Hopper WGMMA.

### MBS scaling non-linear for Mamba3

MBS=8 with PP=2 OOMs (132/140 GB peak). MBS=4 is the practical ceiling at PP=2, reaching 118 GB on rank 0. The Mamba3 MIMO R=4 scan state tensors scale linearly with MBS (4x the state compute vs SISO), creating a higher memory floor than vanilla transformers.

### CUDA graphs + MIMO: previously blocked, now fixed

CUDA graph capture was initially blocked by `_broadcast_cu_seqlens` creating tensors from Python ints inside the captured region. Fixed by Patch (5) in the shim. CUDA graphs with `--cuda-graph-impl local` now work for MIMO 7/7 (confirmed with Author SISO; MIMO with `TILELANG_EXECUTION_BACKEND=nvrtc` also works but `cython` backend is required on cu13.2).

---

## 9. Files manifest

Every file that constitutes the "MIMO 7/7 training" surface:

### Core spec files

| File | Purpose |
|---|---|
| `cppmega/megatron/nam56r_full_spec.py` | Main 7/7 MIMO spec: `build_cppmega_nam56r_full_stack_spec`, `CppMegaSelectiveMambaMixer`, `CppMegaSelectiveAttentionLayer` |
| `cppmega/megatron/nam56r_layout.py` | Layer pattern (`AEMEAEMEAEMR` x 52), R-layer indices, A-layer routing |
| `cppmega/megatron/author_mamba3_spec.py` | `AuthorMamba3Mixer` -- wraps upstream `mamba_ssm.modules.mamba3.Mamba3` with explicit RMSNorm |
| `cppmega/megatron/m2rnn_spec.py` | `CppMegaM2RNNMixer` -- Megatron-style M2RNN mixer with Triton kernel dispatch |
| `cppmega/megatron/m2rnn_triton.py` | Fused Triton M2RNN scan kernel with inline PTX `tanh.approx.f32` |
| `cppmega/megatron/mla_shared.py` | MLA layer spec builder: `CppMegaMLASelfAttentionAdapter`, `CppMegaFusedMLASelfAttentionAdapter` |
| `cppmega/megatron/dsa_local_spec.py` | DSA layer spec (thin wrapper around upstream `get_dsa_module_spec_for_backend`) |
| `cppmega/megatron/mamba_builder.py` | `cppmega_mamba_builder` -- Megatron model builder entry point |
| `cppmega/megatron/custom_mamba_model.py` | `CppMegaMambaModel` -- MambaModel subclass with polymorphic embedding |

### Config files

| File | Purpose |
|---|---|
| `cppmega/features/mamba3/config.py` | `AuthorMamba3Config`, `build_author_mamba3_config` -- maps Megatron config surface to Mamba3 constructor args |
| `cppmega/features/m2rnn/config.py` | `CppMegaM2RNNConfig`, `build_cppmega_m2rnn_config` |

### Recipe / launcher files

| File | Purpose |
|---|---|
| `cppmega/recipes/nam56r_launch.py` | `build_nam56r_megatron_native_args`, `build_nam56r_lite_main_pattern`, `get_custom_layer_indices` |
| `cppmega/recipes/nam56r_megatron.py` | `MegatronHybridPlan`, `parse_nem_pattern`, `translate_nanochat_pattern_to_megatron`, `build_nam56r_feature_plan` |
| `cppmega/recipes/megatron_args.py` | `MegatronArgsBundle`, `build_megatron_args_bundle` -- emits Megatron CLI flags from plan |

### Scripts

| File | Purpose |
|---|---|
| `scripts/cppmega_fp8_shim.py` | Canonical shim: 5 runtime monkey-patches (see Section 7) |
| `scripts/remote_train_h200_nam56r_full.sh` | PP=1/PP=4 launch template (bench3) |
| `scripts/remote_smoke_h200_fp8_mamba3_matrix.sh` | FP8 matrix test template (4 paths x FP8 on/off) |

### Documentation

| File | Purpose |
|---|---|
| `docs/nam56r_mimo7_baseline_2026_04_11.md` | PP=1 baseline (56k tok/sec) full writeup |
| `docs/nam56r_mimo7_vpp_112k_2026_04_11.md` | VPP PP=2 config (112k tok/sec) full writeup |
| `docs/nam56r_mimo7_nsys_profile_2026_04_11.md` | nsys profile analysis: top 10 kernels, roofline, optimization plan |
| `docs/upstream_bugs.md` | Upstream bug tracker (4 entries) |
| `docs/fp8_path_status.md` | FP8 smoke matrix results per Mamba3 path |

---

## 10. Upstream issues to track

### 1. state-spaces/mamba PR #909 -- Cache `ctx.saved_tensors` in Mamba3 backward

- **PR:** https://github.com/state-spaces/mamba/pull/909
- **Status:** Open (as of 2026-04-10)
- **Impact:** Critical for FSDP activation checkpointing with Mamba3 scan kernels. `_Mamba3Function.backward()` accesses `ctx.saved_tensors` twice; activation checkpointing only allows single access. Fix: cache in local variable.
- **cppmega impact:** Low -- production uses `AuthorMamba3Mixer` which goes through `mamba_ssm.modules.mamba3.Mamba3`, not directly through `_Mamba3Function`. Applied locally on bench machines.

### 2. fast-hadamard-transform PyPI sdist broken (1.0.4)

- **Package:** `fast-hadamard-transform==1.0.4` on PyPI
- **Impact:** Blocker for DSA. PyPI tarball missing `csrc/fast_hadamard_transform.cpp`.
- **Fix:** Install from GitHub HEAD.

### 3. TileLang nvrtc backend broken on CUDA 13.2

- **Package:** tilelang (all versions through 0.1.8)
- **Impact:** Blocker for MIMO when using `TILELANG_EXECUTION_BACKEND=nvrtc` on cu13.2. Bundled cutlass `cute/container/array.hpp` conflicts with system CCCL 13.2.27.
- **Workaround:** Use `TILELANG_EXECUTION_BACKEND=cython` (NVCC subprocess).

### 4. Megatron Float16Module blanket bf16 cast

- **Package:** megatron-core (all versions through 0.18rc0)
- **Impact:** Silent correctness bug for any module with fp32-required parameters (Mamba3 bias/D/dt). Causes NaN or dtype mismatch.
- **Workaround:** Shim patch (4).
- **Clean fix needed:** Teach `Float16Module` to honor per-parameter `_keep_fp32` annotation.

### 5. Fused MLA RoPE Triton kernel crashes at PP>1

- **Package:** megatron-core (fused_mla_yarn_rope_apply.py)
- **Impact:** `cudaErrorIllegalAddress` in `rotary_fwd_kv_kernel` during Triton autotuner at PP>1. RoPE cos/sin cache is device-global and incompatible across pipeline stages.
- **Workaround:** `--no-rope-fusion` (cost: ~10-30 ms/iter).
- **Action:** Report upstream.

### 6. TileLang MIMO backward -- `ValueError: G value of 8 is not currently supported!`

- **Package:** mamba-ssm 2.3.1 (`mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py:1314`)
- **Impact:** Blocked MIMO R=4 backward when `1 < ngroups < nheads`.
- **Fix applied locally:** Added `elif G < H: dq = dq_tilelang.view(B, S, R, G, H//G, N).sum(dim=4)` for partial-group contraction. Applied on bench machines.

---

## Appendix A: Loss trajectories

### Config A (PP=1, GBS=16)

| Iter | lm loss | grad norm |
|---|---|---|
| 1 | 11.7463 | 79.67 |
| 5 | 10.5335 | 34.57 |
| 10 | 8.2498 | 19.41 |
| 20 | 5.6597 | 7.23 |
| 30 | 4.8173 | 5.57 |

### Config B (VPP PP=2, GBS=64)

| Iter | lm loss | grad norm |
|---|---|---|
| 1 | 11.92 | -- |
| 5 | 7.34 | 8.2 |
| 10 | 4.68 | 4.5 |
| 20 | 3.12 | 3.1 |
| 30 | 2.73 | 2.8 |

Config B converges faster per-step because GBS=64 provides 4x more gradient signal per step. Not directly comparable for convergence quality without matching total tokens seen.

---

## Appendix B: nsys profile summary (PP=1 baseline)

Top 5 time buckets from the steady-state profile (iters 6-10, 1186 ms/iter):

| Bucket | ms/iter | % | Key kernels |
|---|---|---|---|
| Elementwise + copy | 305 | 25.7% | `CUDAFunctor_add<float>`, `MulFunctor<float>` (fp32 shim overhead, now fixed by Patch 4) |
| TE Hopper BF16 GEMM (WGMMA) | 251 | 21.2% | `nvjet_sm90_tst_*_h_*_coopA_NNT` -- MLA/MoE/dense, confirmed WGMMA |
| FP32 cuBLAS sm80 GEMM | 142 | 12.0% | `sm80_xmma_gemm_f32f32` -- Ampere fallback in optimizer |
| TileLang MIMO bwd_bwd | 120 | 10.1% | `mamba_mimo_bwd_bwd_kernel`, 7.68 ms avg |
| NCCL RS+AG | 87 | 7.3% | Mostly overlapped with compute |

TileLang MIMO total: 208 ms/iter (17.5%). M2RNN Triton total: 73 ms/iter (6.2%).

---

## Appendix C: Projected throughput progression

From the VPP PP=2 baseline (112k tok/sec):

| Lever | Expected gain | Projected tok/sec | Status |
|---|---|---|---|
| VPP PP=2 baseline | -- | 112k | done |
| + FP8 on MLA+MoE | +15-20% | ~130-135k | pending |
| + Restore rope fusion | +2-3% | ~135-140k | needs upstream fix |
| + MBS=5 or 6 | +10-15% | ~150-160k | memory-constrained |
| + PP=4 VPP=4 finer pipeline | +10-20% | ~180-200k | medium effort |
| + TP=2 PP=2 VPP=2 | +15-25% | ~200-230k | medium effort |
| **Aggressive best case** | -- | **~200-230k** | -- |
| **Target** | -- | **250,000** | requires TP+PP+FP8+MBS |

---

## Late session updates

### MTP Super flags experiment (europe, 3 variants)

| Variant | Config | Iter ms | Tok/sec | LM loss@30 | MTP loss@30 |
|---|---|---|---|---|---|
| V1 baseline | untied, MTP on | 2348.5 | 111,621 | 2.70 | 2.66 |
| V2 `mtp_use_repeated_layer=True` depth=1 | Super flags | 2355.8 | 111,275 | 2.81 | 2.54 |
| V3 `mtp_use_repeated_layer=True` depth=2 | Super flags | 2688.9 | 97,530 | 2.75 | 2.66/2.49 |

**Conclusion:** `mtp_use_repeated_layer=True` works for Mamba hybrid but gives 0% speedup at depth=1 (no-op when only 1 layer). At depth=2, -12.6% (expected -- shared weights save params not FLOPs). MTP overhead is forward+backward FLOPs, not param count.

### Tied embeddings experiment (europe)

| Config | Iter ms | Tok/sec | Delta |
|---|---|---|---|
| Untied (baseline) | 2348.5 | 111,621 | 0% |
| Tied | 2349.4 | 111,623 | -0.04% |

**Conclusion:** 0% effect on PP=2 hybrid. Megatron can't fuse embedding/output head across PP ranks.

### 8-variant MTP sweep (europe + bench3)

| # | Name | Iter ms | Tok/sec | Status |
|---|---|---|---|---|
| 1 | Control untied | 2348.5 | 111,666 | baseline |
| 2 | Tied only | 2349.4 | 111,623 | 0% gain |
| 3 | Standalone VPP | -- | -- | BLOCKED (Megatron hybrid) |
| 4 | Tied + standalone | -- | -- | BLOCKED |
| 5 | Tied MBS=5 | -- | -- | OOM (~142/140 GB) |
| 7 | PP=4 VPP=1 | 3258.3 | 80,490 | -28% regression |
| 8 | NoMTP control | 1981.2 | 132,438 | +18.6% (confirms 133k) |

### Liger fused CE for MTP (bench3)

| Metric | Standard CE | Liger fused | Delta |
|---|---|---|---|
| MTP time (4 depths fwd+bwd) | 178.8 ms | 483.2 ms | 2.7x SLOWER |
| Peak memory | 27.36 GB | 5.49 GB | -82% |

**Conclusion:** Liger saves 82% memory but 2.7x slower on H200 (poor tensor core utilization on chunked small-M GEMMs). DO NOT enable on H200. Valuable for memory-constrained hardware only.

### CUDA graphs experiments (bench3)

| Scope | Status | Tok/sec | Delta |
|---|---|---|---|
| Baseline (no graphs) | -- | 68,844 | -- |
| `--cuda-graph-scope attn` | PASS | 69,822 | +1.4% |
| `--cuda-graph-scope full_iteration` | FAIL | -- | MoE `.item()` blocker |
| `transformer_engine` | FAIL | -- | Same MoE blocker |

**Key discovery:** agent used WRONG scope. cppmega already has working per-module scope: `--cuda-graph-scope attn mamba moe_router moe_preprocess` (in `nam56r_nemo_recipe.py:287-296`). Full MoE graph needs `--moe-pad-expert-input-to-capacity`. **211k tok/sec production recipe used this exact config.** Fix agent launched.

### CuTe DSL Phase 4 completion (europe)

Kernel: 553 LOC, 11 GEMMs (original 10 + GEMM 6' for transpose). ALL 14 outputs correct vs TileLang (rtol=1e-2). Architecture: WGMMA GEMMs in CuTe DSL kernel + PyTorch epilogue for 9 non-GEMM outputs. Second agent launched for fully-fused version (all inside kernel, no torch).

| Shape | Phase 4 (11G) | TileLang | Speedup |
|---|---|---|---|
| Smoke | 113 us | 175 us | 1.55x |
| Production | 2515 us | 3135 us | 1.25x |

GEMM 7/9 semantic errors FIXED. Dstates loop-carried accumulator CORRECT (rel 0.004).

### CuTe DSL Phase 4v2 (bench3)

New file `fused_bwd_bwd_sm90_p4v2.py` (410 LOC) with GEMM 7/9 fixed differently (gmem scratch for transpose). All 5 kernel outputs PASS (max rel 0.0038).

### GB10 correctness (196 tests)

| Suite | Pass | Fail | Notes |
|---|---|---|---|
| M2RNN Triton | 10 | 0 | All shapes correct |
| TileLang MIMO autograd | 5 | 1 known | bwd_bwd 140KB smem (fix: `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True`) |
| cuTile TileGym 3-phase | 11 | 0 | All 11 gradients PASS |
| CuTe DSL (6 tests) | 6 | 0 | All on sm_121a |
| FLA cutile (161 tests) | 161 | 0 | 100% pass |
| cppmega unit | 10 | 0 | All pass |

**No regressions.** GB10 bwd_bwd smem fix exists (`TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True`), correctness agent just didn't set the env var.

### Nemotron 3 Super/Nano MTP research

- Super uses `MambaModel` + PP=1 + `mtp_use_repeated_layer=True` + `mtp_num_layers=2`
- Nano has NO MTP
- Standalone MTP explicitly NOT supported for hybrid model (`mamba_model.py:195-199`, PR #3377 confirms)
- NVIDIA sidesteps by using PP=1

### cuDNN LD_LIBRARY_PATH fix (bench3)

Root cause of ALL bench3 training failures in late session: system cuDNN 9.10.2 loaded before venv cuDNN 9.20.0. Fix: `export LD_LIBRARY_PATH=.../nvidia/cudnn/lib:$LD_LIBRARY_PATH` in `~/.bashrc`. Applied and verified on both machines.

### ThunderKittens on GB10

TK compiles and runs on sm_121a with `KITTENS_AMPERE` flag: 64x64 BF16 GEMM = 4.11 us (1.84x faster than torch.mm). HMMA.16816 tensor core confirmed via cuobjdump. Warp-level MMA path only (no WGMMA/tcgen05).

### Megatron CUDA graph blocker patches (bench3)

1. MoE `.cpu()` at `token_dispatcher.py:295` -- keep on GPU (v2 preserves `.item()` for eager path)
2. `compute_dacs_segsum_triton` autotune -- fixed single config (8 autotune blocks collapsed)
3. `_broadcast_cu_seqlens` -- TP=1 bypass in `megatron/training/utils.py:566` (direct patch, not shim -- local function)
4. `tokens_per_expert.sum().item()` at line 306 -- additional D2H sync (identified, needs v3 fix OR per-module scope bypass)
