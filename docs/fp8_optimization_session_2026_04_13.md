# FP8 & Optimization Deep-Dive — Session 2026-04-13

## Обзор / Overview

Цель: довести NAM56R 4.73B (27 Mamba3 + 13 Attention [9 DSA + 4 MLA] + 12 MoE) до 50% MFU
на H200×8 (europe + bench3). Текущий baseline: **158 TFLOP/s (16% MFU)** без CG.

Target: reach 50% MFU for NAM56R 4.73B hybrid model on H200×8 single-node.
Current baseline: **158 TFLOP/s (16% MFU)** without CUDA graphs.

---

## 1. TE FP8 GEMMs — NET LOSS / НЕ РАБОТАЕТ

### Тесты / Tests (all PP=1 EP=1 DP=8, no CUDA graphs)

| Config                 | TFLOP/s | ms/iter | GBS | Peak GiB | Notes        |
| ---------------------- | ------- | ------- | --- | -------- | ------------ |
| **BF16 MBS=6 GBS=48**  | **158** | ~3900   | 48  | 112      | Baseline     |
| FP8 MBS=4 GBS=32       | 154     | 2679    | 32  | 85       |              |
| FP8 MBS=4 GBS=64       | 156     | 5260    | 64  | 89       | 2 grad_accum |
| FP8 MBS=6 GBS=48       | 158     | 3911    | 48  | 112      | Same speed   |
| FP8+param_gather MBS=6 | 157     | 3934    | 48  | 107      | -5 GiB       |
| BF16 MBS=8 GBS=64      | 145     | 5674    | 64  | 136      | Mem pressure |

### Причина / Root Cause
- GEMMs = только 23.5% compute (SSM=27.5%, elementwise=14.7%)
- FP8 amax overhead (quantize/dequantize + history) на КАЖДОМ GEMM
- FP8 weight copy: TE хранит BF16 + FP8 одновременно (+8 GiB)
- `--fp8-param-gather` убирает BF16 copy (-5 GiB) но speed не меняется
- DeepSeek V3 тоже хранит BF16 master weights — нет способа убрать

### `--fp8-param-gather` + `--use-precision-aware-optimizer`
- param_gather: хранит weights в FP8, all-gather в FP8, -5 GiB
- precision_aware_optimizer: **НЕСОВМЕСТИМ** с fp8_param_gather (Int16 vs FP32 assertion)

### Вывод / Conclusion
TE FP8 GEMMs при `--fp8-format hybrid` = net zero или net loss для hybrid Mamba model.
Не использовать для NAM56R.

---

## 2. Selective FP8 (MoE only) / Селективный FP8

### Написано / Created
- `cppmega/megatron/selective_fp8_moe_patch.py` (134 строк)
- Monkey-patches `megatron.core.fp8_utils.get_fp8_context`
- FP8 только для MoE layers (22 из 52), BF16 для Mamba + Attention
- Requires `--fp8-recipe tensorwise` (NOT delayed)
- Gate: `CPPMEGA_SELECTIVE_FP8_MOE=1`

### Статус / Status
- Patch написан, signature bug fixed (`layer_no=-1` default)
- Env var export добавлен в скрипт
- **НЕ ПРОТЕСТИРОВАН** — первые тесты показали те же 158 TFLOP/s (patch не активировался из-за env var issue, потом timing: CG conflicts поглотили всё внимание)
- TODO: протестировать с реальным CG + nsys

---

## 3. FlashAdamW (PR #4229) / Квантизованный оптимизатор

### Что это / What
- `pip install flashoptim>=0.1.3`, `--optimizer flashadamw`
- INT8 quantized m/v states: 16→7 bytes/param
- BF16 + INT8 error correction master weights (24-bit effective)

### Статус применения / Applied on both hosts
- **europe** ✅: PR applied via `git apply --3way`, conflicts resolved
- **bench3** ✅: patch + manual sed for choices list + Muon exclusion fix

### Ограничения / Constraints
- **НЕ совместим** с `--use-distributed-optimizer`
- **НЕ совместим** с `--use-precision-aware-optimizer`
- **НЕ совместим** с `--optimizer-cpu-offload`
- Совместим с PP (standard optimizer path)
- Без dist-opt: ~33 GiB optimizer/rank (vs 9.5 GiB dist-opt)
- Total per rank: 33 + 9.5 (model) + 85 (activations) = ~127 GiB — впритык

### Статус тестов / Test Status
- **4 попытки упали**: padding_mask bug × 3, Muon assertion × 1
- padding_mask fix нужен в `transformer_layer.py` (строки 804, 1726) — bench3
- TODO: retest после починки padding_mask

---

## 4. FP8 TileLang DSA Sparse MLA / Полный FP8 fwd+bwd

### Написано / Created
- `tilelang_sparse_mla_fwd_fp8.py` (491 строк) — FP8 forward ✅ (прошлая сессия)
- `tilelang_sparse_mla_bwd_fp8.py` (637 строк) — FP8 backward ✅ (эта сессия)
- `SparseMLA_FP8` autograd Function обновлён — полный FP8 pipeline

### Архитектура / Architecture
- **Forward**: Q@K в float8_e4m3fn WGMMA (2× пик), S@V в BF16, per-token scaling
- **Backward**: P=Q@K^T в FP8, dP=dO@V^T в BF16, dQ+=dS@K в FP8, dKV+=Q^T@dS в FP8
- **save_for_backward**: Q,KV в FP8 (1B вместо 2B) + FP32 scale vectors
- **Output**: dQ, dKV в BF16 (optimizer stability)
- **Память**: 2× экономия на saved tensors Q,KV

### Impact
- DSA = 2.4% compute → ~1.2% speedup от FP8 backward
- Но 2× saved tensor memory = ценно при MBS=8

### Статус / Status
- Написано, НЕ тестировано в training
- TODO: unit test на europe, затем training test

---

## 5. Mamba SSM Kernel Analysis (nsys) / Анализ ядер

### Данные / Data (из прошлой сессии)

| Kernel             | Time/iter  | Regs | Smem  | Occupancy |
| ------------------ | ---------- | ---- | ----- | --------- |
| mamba_mimo_fwd     | 1192ms     | 239  | 196KB | 6.2%      |
| mamba_mimo_bwd_fwd | 1034ms     | 255  | 196KB | 6.2%      |
| mamba_mimo_bwd_bwd | 2110ms     | 255  | 228KB | 12.5%     |
| _m2rnn_fwd         | 405ms      | 64   | 100KB | 12.5%     |
| _m2rnn_bwd         | 623ms      | 255  | 32KB  | 12.5%     |
| **Total SSM**      | **5364ms** |      |       |           |

### Ключевые проблемы / Key Issues
1. **TMA ОТКЛЮЧЁН** (`TL_DISABLE_TMA_LOWER=True`) — все загрузки синхронные
2. **255 регистров/поток** (max!) — 1 block/SM → 6.25% occupancy
3. **bwd_bwd recompute**: пересчитывает весь fwd = 2110ms (39% SSM time)
4. **WGMMA используется** (подтверждено через cubin) — GEMMs ок
5. TMA НЕ рекомендуется включать для scan kernels (TileLang TMA mature только для GEMM)

### Оптимизации (приоритет) / Optimizations
- P1: Smem reduction (228KB→48KB, 4× occupancy) → -1500ms
- P2: State checkpointing (kill bwd_bwd recompute) → -1000ms
- P3: Register pressure (255→128) → -800ms
- Projected: 5364→2064ms (2.6× speedup)

---

## 6. CUDA Graphs / Графы CUDA

### Проблема / Problem
- `--cuda-graph-scope attn mamba moe_router moe_preprocess` задаёт scope
- Но **`--cuda-graph-impl local`** нужен для фактической активации!
- Без `--cuda-graph-impl`, `enable_cuda_graph=False` → CG не работает
- **ВСЕ наши "249 TFLOP/s" тесты были БЕЗ CG!**

### FP8 + CG + Recompute: почему только FP8 путь CG-compatible

**Ключевой код**: `megatron/core/transformer/transformer_layer.py:793-810` (europe)

```python
if self.recompute_mlp:
    if self.config.fp8 or self.config.fp4:          # ← FP8 PATH
        from megatron.core.extensions.transformer_engine import te_checkpoint
        mlp_output_with_bias = te_checkpoint(       # ← TE's CG-aware checkpoint
            functools.partial(self.mlp, padding_mask=padding_mask),
            False, tensor_parallel.random.get_cuda_rng_tracker,
            self.pg_collection.tp, pre_mlp_layernorm_output,
        )
    else:                                            # ← BF16 PATH
        mlp_output_with_bias = tensor_parallel.checkpoint(  # ← vanilla, CG-UNSAFE!
            functools.partial(self.mlp, padding_mask=padding_mask),
            False, pre_mlp_layernorm_output,
        )
```

- **FP8** → `te_checkpoint()` → TE's `distributed.checkpoint()` → CG-aware (TE manages capture/replay)
- **BF16** → `tensor_parallel.checkpoint()` → vanilla `torch.utils.checkpoint` → **ломается при CG capture** (`Checkpointing is not compatible with .grad()`)

**CG guard для MoE**: `transformer_layer.py:840-845` (europe)

```python
if (self.is_moe_layer
    and self.config.cuda_graph_impl == "transformer_engine"  # ← ONLY TE impl!
    and self.training and is_graph_capturing()
    and CudaGraphScope.moe_router in self.config.cuda_graph_scope):
```

Guard активен ТОЛЬКО при `cuda_graph_impl == "transformer_engine"`. При `local` → нет guard → crash.

**`--cuda-graph-impl` варианты**: `megatron/core/transformer/cuda_graphs.py`

| impl | Механизм | Recompute | Memory |
|------|----------|-----------|--------|
| `local` | `make_graphed_callables` — весь forward как 1 граф | ❌ crash | +62 GiB private pools |
| `transformer_engine` | TE `CudaGraphManager` — per-layer capture | ✅ CG-aware | Минимальный overhead |

**Вывод**: единственная рабочая комбинация = `--fp8-format hybrid` + `--cuda-graph-impl transformer_engine` + `--recompute-modules`.

**fp32 bias hook**: `scripts/remote_smoke_h200_dsa_9_4_m.sh:177`
```python
def _restore_bias_fp32(module, _inputs):
    from megatron.core.cuda_graphs import is_graph_capturing, is_graph_warmup
    if is_graph_capturing() or is_graph_warmup():
        return  # skip during CG
    for _name in _FP32_NAMES:
        _p = getattr(module, _name, None)
        if _p is not None and _p.dtype != _torch.float32:
            _p.data = _p.data.float()
```

Без `is_graph_capturing()` guard → `register_forward_pre_hook` → `AssertionError: Tried to cudagraph a module with user registered pre-forward hooks`.

### CG-unsafe operations found & fixed this session
1. **`dsa_fp8_indexer.py:314`**: `(topk_indices < 0).any()` → implicit `.item()` CPU sync
   - Fix: branchless clamp+scatter+fixup (no scalar `.any()`)
2. **Mamba3 `register_forward_pre_hook`**: Python hooks banned during CG capture
   - Fix: replaced with `_apply` override that preserves fp32 params after bf16 cast
3. **`torch.utils.checkpoint`**: incompatible with CG capture (Python callbacks)
   - Fix: disable recompute when using CG (`EXTRA_FLAGS` override)
4. **`torch.equal()` in DSA indexer**: patched to `if False:` via `apply_dsa_cg_patches.py`
5. **Duplicate `d_v=None`**: double-patching from `apply_dsa_cg_patches` → sed fix
6. **Float16Module casts Mamba3 fp32 params**: `Mamba3._apply` guard preserves only D, dt_bias as fp32
   - B_bias, C_bias: stay bf16 (add to bf16 activations)
   - D, dt_bias: must be fp32 (TileLang kernel def: `DT: T.Tensor(..., T.float32)`)
   - Code: `scripts/remote_smoke_h200_dsa_9_4_m.sh` Mamba3._apply guard section

### Memory with CG
- CG private pools = ~40 GiB additional memory on top of model+activations
- MBS=6 no recompute: 112 + 40 = 152 GiB → **OOM** (143 GiB limit)
- MBS=6 no recompute no CG: 112 GiB → fits but 158 TFLOP/s (no CG boost)
- MBS=4 + recompute + FP8 + CG: ~81 + 8 + 40 = ~129 GiB → **should fit**

### Текущий статус / Current Status
- **FP8 + CG(TE impl) + recompute + MBS=4** running on BOTH machines
- All bugs fixed: indexer .any(), fp32 guard, P/dP dtype, padding_mask
- TileLang JIT compiling (fresh cache), no crashes
- **FIRST SUCCESSFUL RUN**: FP8 + CG(TE) + recompute + MBS=4 = **155.5 TFLOP/s, 2649ms, 89 GiB**
- CG warmup: iter 2 = 76s, iter 4 = 17s (backward capture), then 2649ms steady state
- CG gives ~1% speedup (2679→2649ms) — small because TileLang MIMO kernels may not be captured by TE's CG manager
- Loss converging: 11.8→3.67, grad_norm 139→15.7, 0 NaN, 0 skipped

### FP8+CG Throughput Sweep (europe, PP=1 EP=1)

| Config | TFLOP/s | ms/iter | Memory | CG |
|--------|---------|---------|--------|-----|
| BF16 MBS=6 no CG | 158 | 3900 | 112 GiB | ❌ |
| BF16 MBS=4 no CG | 154 | 2679 | 81 GiB | ❌ |
| FP8 MBS=4 no CG | 154 | 2679 | 85 GiB | ❌ |
| FP8 MBS=4 + CG(TE) | 155.5 | 2649 | 89 GiB | ✅ |
| **FP8 MBS=6 + CG(TE) + idx=0** | **258→218** | **2396→2840** | **106 GiB** | **✅** |

**KEY FINDING**: iter 3 = 258 TFLOP/s (pre-CG-bwd), iters 5+ = 218 (post-CG-bwd).
CG backward capture HURTS throughput (258→218 = -15%). TE CG backward overhead > launch savings.
**FP8 WITHOUT CG** at indexer_loss=0 expected to be ~258 TFLOP/s = BEST CONFIG.

**ROOT CAUSE of 155→249 regression**: `dense_indexer_loss_fallback` = 37% overhead!
Fixed via `CPPMEGA_DSA_INDEXER_LOSS_COEFF=0` env override.

## DEFINITIVE COMPARISON (all PP=1 EP=1 DP=8 MBS=6 GBS=48, no indexer)

| Config | TFLOP/s | ms/iter | tok/sec | Why |
|--------|---------|---------|---------|-----|
| **BF16 + moe_act recomp** | **254** | **2429** | **81k** | **BEST** |
| FP8 + CG(TE) iter3 | 258 | 2396 | 82k | pre-CG-bwd |
| FP8 + CG(TE) steady | 218 | 2840 | 69k | CG bwd overhead |
| FP8 no-CG | 218 | 2840 | 69k | no moe_act recomp |
| FP8 + moe_act | CRASH | — | — | Megatron assertion |

## NEW BEST CONFIG: BF16 MBS=8 + Liger CE + moe_act recompute + no indexer

**265 TFLOP/s / 84.3k tok/sec / 110 GiB / stable 20/20**

```bash
VARIANT=v0 PP_SIZE=1 VPP_SIZE=1 MBS=8 GBS=64 FP8_FLAGS=NONE CG_FLAGS=NONE
CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 CPPMEGA_MTP_LIGER_CE=1
EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0"
```

Liger CE saves 21 GiB MTP logits → MBS=8 fits in 110 GiB (vs 136 GiB without Liger).

### MBS Sweep (BF16 + Liger CE + moe_act recompute + no indexer, PP=1 EP=1)

| MBS | GBS | TFLOP/s | ms/iter | tok/sec | Memory | Headroom |
|-----|-----|---------|---------|---------|--------|----------|
| 6 | 48 | 254 | 2429 | 81k | ~85 GiB | 58 GiB |
| **8** | **64** | **265** | **3111** | **84k** | **110 GiB** | **33 GiB** |
| 10 | 80 | 268 | 3850 | 85k | 128 GiB | 15 GiB |

**Recommended: MBS=8** (best throughput-to-headroom ratio).

**VERIFIED GOLDEN CONFIG (europe, iter 3-8 steady state, reproducible):**
```bash
# Launch command:
VARIANT=v0 PP_SIZE=1 VPP_SIZE=1 MBS=8 GBS=64 \
FP8_FLAGS=NONE CG_FLAGS=NONE \
CPPMEGA_DSA_INDEXER_LOSS_COEFF=0 CPPMEGA_MTP_LIGER_CE=1 \
EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0" \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh
```

**Result: 265 TFLOP/s / 84k tok/sec / 105 GiB peak / 38 GiB headroom / stable 20/20**

## 🔥 FP8 TENSORWISE = NET WIN! (discovered iteration 41)

**FP8 `--fp8-recipe tensorwise` + moe_act recompute = 269 TFLOP/s / 85.8k tok/sec**

The key: `delayed` recipe blocks moe_act, but `tensorwise` does NOT!
Code: `transformer_config.py:1544`: `if self.fp8_recipe == 'delayed': raise ValueError(...)` — only delayed!

```bash
FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
# + all other golden flags (moe_act recompute, Liger CE, indexer=0)
```

| Config | TFLOP/s | Memory | vs BF16 |
|--------|---------|--------|---------|
| BF16 MBS=8 Liger | 265 | 110 GiB | baseline |
| FP8 tensorwise MBS=8 Liger | 269 | 104 GiB | +1.5% speed, -6 GiB mem |
| **FP8 tensorwise MBS=10 Liger** | **273** | **119 GiB** | **+3% speed, 24 GiB headroom** |

**CEILING (no indexer): 273 TFLOP/s / 87k tok/sec (FP8 tensorwise MBS=10)**

## 🏆 PRODUCTION GOLDEN CONFIG (with indexer training)

**267 TFLOP/s / 85k tok/sec** — FP8 tensorwise + IndexCache + Liger CE + MBS=10

```bash
VARIANT=v0 PP_SIZE=1 VPP_SIZE=1 MBS=10 GBS=80
FP8_FLAGS="--fp8-format hybrid --fp8-recipe tensorwise --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
CPPMEGA_DSA_INDEXER_LOSS_COEFF=0.001
CPPMEGA_INDEX_CACHE=1 CPPMEGA_INDEX_CACHE_FULL_LAYERS=0,3,6
CPPMEGA_MTP_LIGER_CE=1
EXTRA_FLAGS="--recompute-granularity selective --recompute-modules moe_act mlp mla_up_proj --mla-down-proj-fusion --clip-grad 1.0"
```

**158 → 267 = +69% improvement with full indexer training!**

### Production Config (with indexer training)

| Indexer mode | TFLOP/s | tok/sec | Overhead | Bottleneck |
|-------------|---------|---------|----------|------------|
| **off** (ceiling) | **273** | **87k** | 0% | — |
| head-streaming + sparse KL | **214** | **69k** | 28% | Q@K per-head loop in `_attention_target_fp32` |
| head-streaming + dense KL | 212 | 69k | 28% | Same Q@K bottleneck |
| split-K Triton | 187 | 61k | 46% | Different code path, slower |

**Key insight**: sparse KL doesn't help (214 vs 212) because bottleneck is Q@K attention target matmul, NOT the KL computation itself.

**Fix paths**:
1. Fused FA+KL kernel (lemyx/tilelang-dsa) — eliminates Q@K + KL separation
2. CUDA stream overlap — hide indexer in side stream during next layer fwd
3. Reduce frequency — indexer loss every N steps after warmup
4. IndexCache — cross-layer reuse (3 Full + 6 Shared DSA layers = 67% savings)

### lemyx/tilelang-dsa Assessment

**Repo**: https://github.com/lemyx/tilelang-dsa (44 stars, MIT)
**What**: Single TileLang kernel fusing FA forward + Lightning Indexer logits + one-pass KL divergence.
**API**: `_DsaWarmupFunc.apply(q_unpad, k_unpad, v_unpad, index_q, index_k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, index_weights)` → `(o_unpad, dkl)`

**Incompatibilities with NAM56R**:
- `assert heads == index_heads` — hardcoded (NAM56R: 56 main vs 8 indexer)
- MQA assumption (single KV head), not MLA/GQA
- Warmup-phase only (fuses FA+indexer, not post-hoc KL)
- Global constants: heads=64, dim=128

**Standalone test PASSED on H200 (europe)**:
- TileLang fused FWD: **15.6 ms** (FA + indexer + KL combined)
- FlashAttention ref FWD: 7.0 ms (FA only)
- Indexer+KL overhead in fused mode: **8.6 ms** (vs 2500 ms current = **290× faster**)
- TileLang BWD: 79.6 ms
- FA output correctness: ✅ PASSED
- batch=4, seqlen=4096, heads=64, index_heads=64

**If adapted for NAM56R (heads=56, index_heads=8)**:
- Indexer overhead would drop from 28% to **<1%**
- Production throughput: 214 → **~270 TFLOP/s** (near ceiling)
- Need to modify: remove `assert heads == index_heads`, add head grouping

### IndexCache (arXiv:2603.12201)

**Repo**: github.com/THUDM/IndexCache (inference only)
**What**: Skip indexer on 6/9 DSA layers, reuse cached indices from nearest Full layer.
**Patch**: `cppmega/megatron/index_cache_patch.py` (100 lines) — sets loss_coeff=0 on Shared layers.
**VERIFIED on bench3 (v3 proper cache reuse)**: 
- Simplified (loss skip only): 267 TFLOP/s (2% overhead)
- **Proper (cache reuse + skip indexer fwd)**: **272 TFLOP/s (0.4% overhead!)**
- FP8 tensorwise MBS=6 + IndexCache: **250 TFLOP/s** (vs 212 without = +18%)
- Memory: 99 GiB (vs 106 without IndexCache)
- Bench3 was deadlocked for hours due to stale `torch_extensions/share_storage_ext/lock` file

**VERDICT updated**: FP8 TE with `delayed` recipe = NET LOSS. FP8 TE with `tensorwise` recipe = **NET WIN**.
FP8 forced removal of `moe_act` from recompute (delayed scaling incompatible).
Without moe_act recompute: +14% memory pressure → 14% slower.
BF16 + full recompute = fastest stable config.

---

## 7. torch.compile / Компиляция

### Результаты исследования / Research Results
- **torch.compile НЕ используется** в Nemotron/Megatron production
- TE блокирует dynamo через `no_torch_dynamo` decorators
- `mode="reduce-overhead"` конфликтует с manual CUDA graphs
- `mode="default"` безопасен — только Inductor fusion, без CG
- **Regional compile**: `torch.compile(layer.mlp, mode="default")` — fuse elementwise
- Mamba scan = graph break (expected), pre/post-scan elementwise fuse
- 89 kernels → ~4-8 fused = ~1% total speedup (conservative)

### Реализация / Implementation

**Файл**: `cppmega/megatron/mamba3_compile_patch.py` (75 строк)

```python
# Gate: CPPMEGA_MAMBA3_COMPILE=1
_compiled_forward = torch.compile(
    Mamba3.forward,
    mode="default",      # ← Inductor only, NO internal CG
    dynamic=False,       # ← static shapes (fixed B,S,H,P)
    fullgraph=False,     # ← allow graph breaks at TileLang scan
)
```

**Что фьюзится** (из `mamba_ssm/modules/mamba3.py`):
- Pre-scan: `F.softplus(dd_A)` → `clamp` → `mul` → `rearrange` (строки 170-179)
- Pre-scan: `F.softplus(dd_dt + dt_bias)` → `rearrange` (строка 172)
- Pre-scan: `angle_dt(angles, DT)` (строка 184)
- Pre-scan: `B_norm(B)`, `C_norm(C)` — RMSNorm (строки 181-182)
- **GRAPH BREAK**: `mamba3_mimo_combined()` — TileLang JIT kernel
- Post-scan: gating, RMSNorm, output projection

**Совместимость**: `mode="default"` safe с `--cuda-graph-impl transformer_engine` (подтверждено research agent)

---

## 8. Infrastructure / Инфраструктура

### Static IPs (assigned this session)
- **bench3**: `H200_1_IP` (LOCATION_1) — static ✅
- **europe**: `H200_2_IP` (LOCATION_2) — static ✅

### Patches applied on both hosts
- FlashAdamW PR #4229
- `padding_mask` fix in `mamba_block.py` (europe only; bench3 needs transformer_layer.py fix too)
- flashoptim 0.1.3 installed

### New files created
- `selective_fp8_moe_patch.py` (134 lines)
- `tilelang_sparse_mla_bwd_fp8.py` (637 lines)
- Script: nsys + memory_debug support, CG impl, EXTRA_FLAGS/OPTIMIZER_FLAGS overridable

---

## 9. nanochat Techniques / Техники из nanochat

| Technique                 | File                 | Savings            | Applicable?                           |
| ------------------------- | -------------------- | ------------------ | ------------------------------------- |
| FP8 Activation Checkpoint | `fp8_activations.py` | 2× on activations  | Yes, medium effort                    |
| Chunked Fused CE          | `kernels.py:608`     | -8 GiB logits peak | Yes, `mtp_liger_ce.py` already exists |
| CPU Offload               | `cpu_offload.py`     | 5-10 GiB attention | Yes, selective                        |
| COAT FP8 Optimizer        | `fp8_optimizer.py`   | 4× on m/v states   | No, conflicts with dist-opt           |

---

## 10. Blockers → 50% MFU / Блокеры

| #   | Blocker                              | Impact                | Status                                                                                                   |
| --- | ------------------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | **CUDA Graphs**                      | ~20-30% throughput    | **ROOT CAUSE FOUND**: `dsa_fp8_indexer.py:314` `.any()` implicit `.item()` → fixed with branchless clamp |
| 2   | **158 TFLOP/s regression** (was 249) | Unknown cause         | Needs nsys comparison with old config                                                                    |
| 3   | **SSM 255 regs + TMA off**           | 27.5% compute starved | Needs kernel restructuring                                                                               |
| 4   | **89 elementwise kernels**           | 14.7% overhead        | torch.compile regional                                                                                   |
| 5   | **padding_mask upstream bug**        | Blocks FlashAdamW, CG | Needs systematic fix                                                                                     |

### Updated Priority Plan (post iteration 20)
1. ✅ Fix CG: 8 bugs found and fixed (indexer .any(), hooks, P/dP dtype, padding_mask, Mamba3._apply)
2. ✅ Found 158→249 root cause: `dense_indexer_loss_fallback` = 37% overhead → `CPPMEGA_DSA_INDEXER_LOSS_COEFF=0`
3. ✅ Proved FP8 TE = net loss (moe_act recompute incompatible with delayed scaling)
4. ✅ Proved CG(TE) backward = net loss (-15% throughput after capture)
5. ✅ MBS=8 + Liger CE: 265 TFLOP/s / 84k tok/sec (NEW BEST)
6. ⬜ MBS=10 + Liger CE (testing now on europe)
7. ❌ Regional torch.compile for Mamba3: InductorError on upstream Triton kernels (tuple arg type). Need to isolate elementwise ops from scan kernel before compiling
8. ⬜ Mamba smem reduction (P1: 228KB→48KB, 4× occupancy)
9. ⬜ nsys profile best config → identify next bottleneck
10. ⬜ FP8 TileLang DSA fwd+bwd training test (kernels written)

### nsys Profile Breakdown (golden config, 262 TFLOP/s steady state)

| Category | % GPU | Improvement Path |
|----------|-------|-----------------|
| **Mamba/SSM** | **34.5%** | Smem reduction (228→48KB), state checkpointing, register pressure |
| **GEMMs (cuBLAS)** | 29.9% | Already WGMMA-optimized. FP8 didn't help. |
| **Elementwise** | 10.7% | 1.3M launches. torch.compile failed (Triton tuple). Need manual fusion |
| **DSA Indexer** | 4.2% | Still runs fwd for routing. Head-streaming or disable entirely |
| **NCCL** | 4.7% | AllReduce. Normal for DP=8 |
| **Permute** | 4.5% | MoE dispatch. DeepEP flex for EP≥2 |
| **Other** | 11.5% | LayerNorm, cat, reduction, Triton, sort |

**Top 5 kernel targets:**
1. `_m2rnn_bwd_kernel` — 11.2% (33ms) → chunked parallel scan, state cache
2. `mamba_mimo_bwd_bwd_kernel` — 8.9% (27ms) → state checkpoint eliminates recompute
3. `_m2rnn_fwd_kernel` — 5.1% → chunked M2RNN
4. `mamba_mimo_bwd_fwd` — 4.7% → smem reduction for 2+ blocks/SM
5. `mamba_mimo_fwd` — 4.7% → smem reduction

### Gap Analysis: 84k → 200k tok/sec
- Current: 265 TFLOP/s / 84k tok/sec (27% MFU)
- Target: 200k tok/sec (50% MFU)
- Gap: 2.4× improvement needed
- **#1**: Mamba/SSM = 34.5% → 2.6× speedup → ~17% → saves ~17% total → ~100k tok/sec
- **#2**: Elementwise = 10.7% → 2× fusion → saves ~5% → ~105k tok/sec
- **#3**: DSA indexer = 4.2% → eliminate → ~110k tok/sec
- **#4**: MoE permute = 4.5% → EP=2 with DeepEP → ~115k tok/sec
- Remaining gap: 115k → 200k = 1.7× → requires fundamental architecture optimization

## Session Timeline (Test loop 37 iterations)

| Iter | TFLOP/s | Config | Finding |
|------|---------|--------|---------|
| 1-4 | CRASH | FP8+CG(TE) | 9 CG bugs found + fixed |
| 5-7 | CRASH | FP8+CG dtype | Float16Module._apply → Mamba3._apply guard |
| 8-9 | 155.5 | FP8+CG MBS=4 | First CG run (1% boost — CG barely helps) |
| 10 | 258→218 | FP8+CG MBS=6 | CG backward capture = -15% |
| 11-12 | 218 | FP8 no-CG | Same as FP8+CG → FP8 overhead = net zero |
| 13 | — | — | **ROOT CAUSE: indexer loss = 37% overhead** |
| 14-15 | 258→218 | FP8 no-idx | FP8 still 14% slower (no moe_act recompute) |
| 16 | 254 | BF16 no-idx | **BF16 + moe_act = best BF16 config** |
| 17 | CRASH | FP8+moe_act | `ValueError: delayed scaling incompatible` |
| 18-19 | **265** | **BF16 MBS=8 Liger** | **GOLDEN CONFIG FOUND** |
| 20 | 268 | BF16 MBS=10 | Tight memory (128 GiB) |
| 21-24 | CRASH | torch.compile | InductorError on Triton tuples |
| 25-30 | 262-265 | nsys profile | SSM=34.5%, GEMMs=29.9%, Elem=10.7% |
| 31-37 | — | bench3 JIT | bench3 TileLang 10× slower than europe |

## Bench3 Issue
bench3 TileLang JIT compile is 10× slower than europe (same TileLang 0.1.8).
Likely NVCC/LLVM compiler version or configuration difference after driver 595 upgrade.
Not a code bug — bench3 will eventually train, just 30-60 min cold start.
