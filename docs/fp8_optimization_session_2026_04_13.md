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

### CG-unsafe operations found & fixed this session
1. **`dsa_fp8_indexer.py:314`**: `(topk_indices < 0).any()` → implicit `.item()` CPU sync
   - Fix: branchless clamp+scatter+fixup (no scalar `.any()`)
2. **Mamba3 `register_forward_pre_hook`**: Python hooks banned during CG capture
   - Fix: replaced with `_apply` override that preserves fp32 params after bf16 cast
3. **`torch.utils.checkpoint`**: incompatible with CG capture (Python callbacks)
   - Fix: disable recompute when using CG (`EXTRA_FLAGS` override)
4. **`torch.equal()` in DSA indexer**: patched to `if False:` via `apply_dsa_cg_patches.py`
5. **Duplicate `d_v=None`**: double-patching from `apply_dsa_cg_patches` → sed fix

### Текущий статус / Current Status
- CG v3 running on BOTH machines with all fixes (no recompute, no hooks, branchless indexer)
- Waiting for steady-state throughput → expecting significant boost over 158 TFLOP/s baseline

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

### Рекомендация / Recommendation
```python
# Compile MoE MLP (elementwise-heavy, not CG-captured)
for layer in model.decoder.layers:
    if hasattr(layer, 'mlp') and not is_cuda_graphed(layer.mlp):
        layer.mlp = torch.compile(layer.mlp, mode="default", dynamic=False)
```

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

### Приоритетный план / Priority Plan
1. ✅ Fix CG: debug `cudaErrorStreamCaptureUnsupported`, add guards
2. ⬜ nsys profile BF16 with CG vs without → understand 158 vs 249 gap
3. ⬜ Regional torch.compile for elementwise
4. ⬜ Selective FP8 MoE (with working CG)
5. ⬜ Mamba smem reduction (P1 from kernel analysis)
6. ⬜ FP8 TileLang DSA training test
7. ⬜ FlashAdamW test (after padding_mask fix)
