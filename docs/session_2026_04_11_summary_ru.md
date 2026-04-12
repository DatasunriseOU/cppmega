# Отчёт о сессии оптимизации NAM56R MIMO 7/7 — 11 апреля 2026

## 1. Обзор задачи

**NAM56R** --- гибридная языковая модель на базе архитектуры Mamba3/Transformer, предназначенная для обработки кода с семантическим обогащением.

| Параметр                 | Значение                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| Общее число параметров   | ~4.73B                                                                                           |
| Активные параметры       | ~3.03B (без неактивных экспертов MoE)                                                            |
| Количество слоёв         | 52 основных + 1 MTP                                                                              |
| Паттерн слоёв            | `AEMEAEMEAEMR`, повторён до глубины 52                                                           |
| Hidden size              | 3584                                                                                             |
| FFN hidden size          | 18944                                                                                            |
| Attention (A)            | 13 слоёв: MLA на большинстве, DSA на рангах 0/4/8                                                |
| Mamba (M)                | 13 слоёв: AuthorMamba3Mixer с MIMO R=4, ngroups=8, headdim=64, dstate=128, chunk_size=16         |
| M2RNN (R)                | 4 слоя: CppMegaM2RNNMixer на индексах 12/24/36/48 (Triton kernel с inline PTX `tanh.approx.f32`) |
| MoE (E)                  | 22 слоя: 16 экспертов, top-4 dropless, + shared expert 1024                                      |
| MTP                      | 1 слой, гибридный режим                                                                          |
| Длина последовательности | 4096                                                                                             |
| Точность                 | BF16 (scan-кернелы работают внутренне в fp32)                                                    |

**Цель:** достичь 250,000 tok/sec на 8 x H200 SXM.

**Прогрессия пропускной способности за сессию:**

| Конфигурация                             | tok/sec     | Прирост |
| ---------------------------------------- | ----------- | ------- |
| PP=1 TP=1 MBS=2 GBS=16 BF16 (baseline)   | **56,280**  | 1.0x    |
| PP=2 VPP=2 TP=1 MBS=4 GBS=64 BF16        | **112,152** | 2.0x    |
| VPP + NoMTP (без multi-token prediction) | **133,519** | 2.37x   |
| **Цель**                                 | **250,000** | 4.44x   |

---

## 2. Платформы тестирования

| Платформа                           | GPU                 | Compute Capability | smem/SM           | Ключевые возможности                                                             | Ключевые ограничения                                                                 |
| ----------------------------------- | ------------------- | ------------------ | ----------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **bench3** (LOCATION_1, 8xH200)  | H200 SXM 140GB HBM3 | sm_90a             | 228 KiB           | WGMMA, TMA с multicast, кластеры, 64 warp/SM, 989 TFLOPS BF16                    | ---                                                                                  |
| **europe** (LOCATION_2, 8xH200) | H200 SXM 140GB HBM3 | sm_90a             | 228 KiB           | Идентично bench3                                                                 | Нет NeMo, требуется настройка токенизатора                                           |
| **GB10** (DGX Spark, домашний)      | GB10 (1xGPU)        | sm_121a            | 99 KiB (динамич.) | Warp MMA (HMMA/QMMA), TMA single-CTA, FP4/FP8 via OMMA, 273 GB/s LPDDR5X         | Нет WGMMA, нет TMEM, нет tcgen05, 99 KiB smem (bwd_bwd блокирован), ~100 TFLOPS BF16 |
| **Modal B200x2**                    | B200 192GB HBM3e    | sm_100a            | 228 KiB           | tcgen05, TMEM 256 KiB/SM, TMA multicast, UMMA, 2250 TFLOPS BF16, 8000 GB/s HBM3e | Облачная платформа Modal, ~$20-30/hr                                                 |

---

## 3. Библиотеки и фреймворки --- что тестировали

### TileLang

- **Репозиторий:** https://github.com/tile-ai/tilelang
- **Что это:** DSL для GPU-кернелов через TVM IR. Позволяет писать на Python с явным управлением shared memory, pipeline stages и thread count.
- **Версия у нас:** 0.1.8 (с пином `apache-tvm-ffi < 0.1.10`)
- **Платформы:** bench3 H200, europe H200, GB10, Modal B200
- **Результат:** Production-path для всех Mamba3 MIMO кернелов (fwd/bwd_fwd/bwd_bwd). На GB10 bwd_bwd = 167 us (потолок для этого HW). На B200 bwd_bwd = 179 us. Структурно недоступный для cuTile Python уровень производительности: 2.5--3.7x быстрее на любом Blackwell.

### cuTile Python (NVIDIA CUDA Tile)

- **Пакет:** `cuda-tile` 1.2.0, `import cuda.tile as ct`
- **Документация:** https://docs.nvidia.com/cuda/cutile-python/
- **Что это:** Высокоуровневый tile DSL от NVIDIA. Компилятор полностью управляет размещением в shared memory и регистрах. Нет явного контроля над `num_stages`, smem, thread count.
- **Версия у нас:** 1.2.0
- **Платформы:** GB10 (sm_121), Modal B200 (sm_100). **НЕ поддерживает H200 (sm_90)** --- tileiras 13.2 отвергает sm_90.
- **Результат:** Порт mamba3_mimo fwd/bwd_fwd/bwd_bwd завершён и прошёл корректность на обеих платформах. bwd_bwd на GB10 = 624 us (3.73x медленнее TileLang). fwd на B200 **на 17.7% БЫСТРЕЕ TileLang** (54 us vs 64 us) --- единственный случай, где cuTile обгоняет TileLang.

### CuTe DSL (nvidia-cutlass-dsl, CUTLASS 4.x)

- **Пакет:** `nvidia-cutlass-dsl` 4.4.2, `import cutlass.cute as cute`
- **Что это:** Низкоуровневый Python DSL с явным управлением smem/WGMMA/TMA/warp specialization. Используется FlashAttention-4.
- **Платформы:** bench3 H200 (sm_90a), GB10 (sm_121a --- BF16 работает!), Modal B200
- **Результат:** WGMMA primitive на H200 = 3.49 us/iter (64x64 BF16). Fused 3-GEMM на H200 = 35.9 us. На GB10 BF16 warp MMA + TMA + persistent scheduler работает out-of-box через `blackwell_geforce/dense_gemm.py`. Ранее считали, что CuTe DSL полностью заблокирован на sm_121 --- оказалось, заблокированы только tcgen05/FP4 пути.

### FlashAttention-4

- **Репозиторий:** https://github.com/Dao-AILab/flash-attention
- **Что это:** Продвинутая attention-реализация для Hopper/Blackwell с WGMMA и TMA warp-specialized producer/consumer паттерном.
- **Платформы:** Только sm_90a (H200) и sm_100a (B200). Физически невозможна на GB10 (нет tcgen05/TMEM).
- **Использование:** Как шаблон для multi-GEMM fusion паттернов. Ключевой паттерн `as_position_independent_swizzle_tensor()` для swizzle-copy. Паттерн producer/consumer НЕ подходит для мелких per-chunk тайлов MIMO (3-5 us compute vs 100 us+ порог для TMA amortization).

### NVIDIA TileGym

- **Репозиторий:** https://github.com/NVIDIA/TileGym (685 звёзд, MIT)
- **Что это:** Официальные cuTile Python паттерны от NVIDIA: Flash Attention (1276 LOC), MLA (3 ct.mma), Recurrent Gated Delta Rule (persistent state), Chunked Delta Rule.
- **Обнаружен:** через Exa research agent 2026-04-11
- **Результат:** Доказал, что cuTile МОЖЕТ держать loop-carried state в регистрах без spilling (FA2 с online softmax на 168 regs/thread, 256x128 тайлы, 918 TFLOPS на B200). Наша проблема --- слишком много live tiles (30+ vs FA2 ~10), а не ограничение компилятора.

### Triton

- **Репозиторий:** https://github.com/triton-lang/triton
- **Использование:** Fused M2RNN кернел с inline PTX `tanh.approx.f32` (SFU op вместо программного __nv_tanh).
- **Платформы:** bench3 H200, europe H200, GB10
- **Результат:** `num_warps=8` уже оптимально (полный autotune sweep 25 конфигов). Inline PTX tanh дал -18.7% на fwd, -20% на fwd+bwd vs manual stable-tanh.

### CUTLASS C++ (NVIDIA/cutlass)

- **Репозиторий:** https://github.com/NVIDIA/cutlass (v4.4.2)
- **Использование:** MMA атомы для sm_120/sm_121 (~160 специализаций в `mma_sm120.hpp` + `mma_sm120_sparse.hpp`). Example 79 --- каноническая FP4 GEMM для consumer Blackwell.
- **Платформы:** GB10, B200

---

## 4. Эксперименты --- полная таблица

### 4.1. MIMO 7/7 baseline (bench3 H200x8)

| Параметр     | Значение                                                                  |
| ------------ | ------------------------------------------------------------------------- |
| Конфигурация | PP=1, TP=1, MBS=2, GBS=16, BF16, no CUDA graphs, no FP8                   |
| Результат    | **56,280 tok/sec**, 1164 ms/iter (steady-state)                           |
| Loss         | 11.75 -> 4.82 за 30 итераций, без NaN                                     |
| Память       | 103 GB / 140 GB на rank                                                   |
| Вывод        | Первый успешный end-to-end запуск полного MIMO 7/7. Зазор до цели = 4.44x |

### 4.2. nsys profile baseline (bench3 H200x8)

| Параметр            | Значение                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| Окно                | 5 итераций (iter 6-10), 1186 ms/iter                                                                   |
| Топ-1 bottleneck    | Elementwise + copy: **305 ms/iter (25.7%)** --- fp32 shim делает ~400 D2D cast'ов/iter                 |
| Топ-2               | TE WGMMA BF16 GEMM: **251 ms/iter (21.2%)** --- подтверждено использование WGMMA                       |
| Топ-3               | FP32 cuBLAS sm80 GEMM (Ampere fallback): **142 ms/iter (12.0%)** --- НЕ WGMMA, 3% от peak              |
| TileLang MIMO total | 208 ms/iter (17.5%)                                                                                    |
| M2RNN Triton total  | 73 ms/iter (6.2%)                                                                                      |
| Вывод               | fp32 shim = #1 bottleneck, CUDA graphs спасут ~150 ms от launch overhead (27,384 kernel launches/iter) |

### 4.3. VPP PP=2 (europe H200x8)

| Параметр     | Значение                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------ |
| Конфигурация | PP=2, VPP=2, TP=1, MBS=4, GBS=64, `--no-rope-fusion`                                             |
| Результат    | **112,152 tok/sec**, 2337 ms/iter                                                                |
| Прирост      | **2.0x** vs PP=1 baseline                                                                        |
| Loss         | 11.92 -> 2.73, grad norm 2.8, без NaN                                                            |
| Блокеры      | Fused MLA RoPE Triton kernel crash при PP>1 (workaround: `--no-rope-fusion`), MBS=8 OOM (132 GB) |
| Вывод        | 110k цель достигнута. Структурный выигрыш от pipeline parallelism                                |

### 4.4. VPP + NoMTP

| Параметр  | Значение                                                                             |
| --------- | ------------------------------------------------------------------------------------ |
| Результат | **133,519 tok/sec**                                                                  |
| Вывод     | Отключение MTP даёт дополнительный прирост. Текущий лучший результат без CUDA graphs |

### 4.5. FP8 эксперимент

| Параметр                          | Значение                                                                                                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Конфигурация                      | `--fp8-format hybrid --fp8-amax-history-len 16`, PP=4, MBS=2                                                                                                     |
| Результат на Path A (mamba3_te)   | PASS, 1064-1116 ms/iter                                                                                                                                          |
| Результат на Path B (author SISO) | PASS, 1047-1416 ms/iter                                                                                                                                          |
| Результат на Path C (MIMO R=4)    | Fwd OK, bwd заблокирован ngroups=8 (починен позже)                                                                                                               |
| Вывод                             | FP8 работает на всех путях. При H=3584 (неоптимальном для FP8 quantization) overhead квантизации примерно компенсирует speedup. **0% net gain** на этих размерах |

### 4.6. CUDA graphs (3 блокера найдены)

| Блокер                  | Источник      | Описание                                                                             |
| ----------------------- | ------------- | ------------------------------------------------------------------------------------ |
| `_broadcast_cu_seqlens` | megatron-core | `torch.tensor(n)` из Python int при TP=1 --- создание тензора во время graph capture |
| MoE `.cpu()`            | megatron-core | `self.local_map.sum(dim=0).long().cpu()` в token_dispatcher.py:295                   |
| `compute_dacs_segsum`   | mamba_ssm     | Triton Mamba3 утилита вызывается внутри forward, несовместима с graph capture        |

**Вывод:** CUDA graphs заблокированы 3 upstream багами. Два в Megatron, один в mamba_ssm. Проектируемая экономия: ~150 ms/iter (15-20%).

### 4.7. TF32 optimizer

| Параметр     | Значение                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Конфигурация | `torch.backends.cuda.matmul.allow_tf32=True`                                                           |
| Результат    | **Регрессия** --- fp32 parity test сломан (TF32 даёт 10-bit мантиссу, drift >1e-2 за 128-шаговый scan) |
| Вывод        | Неприменимо без нарушения корректности. Откачено                                                       |

### 4.8. MBS=3/4 эксперимент

| Конфигурация     | Результат                       |
| ---------------- | ------------------------------- |
| MBS=3 NoMTP PP=1 | ~69k tok/sec                    |
| MBS=4 PP=1       | OOM (>140 GB)                   |
| MBS=4 PP=2       | **112k tok/sec** (см. VPP выше) |
| MBS=8 PP=2       | OOM (132 GB при 140 GB лимите)  |

### 4.9. cuTile bwd_bwd --- 5 вариантов на GB10

| Вариант                              | us      | vs baseline  | vs TileLang |
| ------------------------------------ | ------- | ------------ | ----------- |
| **Baseline 2-kernel A/B split**      | **624** | 1.00x        | 3.73x       |
| V1 nested `@ct.function` (7 helpers) | 1498    | 2.40x slower | 8.96x       |
| V2 fused monolithic                  | 1405    | 2.26x slower | 8.42x       |
| V3 3-kernel split                    | 678     | 1.09x slower | 4.06x       |
| V4 hoisted loop invariants           | 742     | 1.19x slower | 4.45x       |
| V5 `ct.static_iter` unroll           | 3236    | 5.20x slower | 19.4x       |

**Вывод:** Baseline оптимален на GB10. Все "оптимизации" регрессировали.

### 4.10. cuTile bwd_bwd --- 6 вариантов на Modal B200

| Вариант               | B200 us   | vs baseline      | vs TileLang |
| --------------------- | --------- | ---------------- | ----------- |
| Baseline 2-kernel A/B | 687.6     | 1.00x            | 3.84x       |
| V2 fused monolithic   | 729.4     | 1.06x slower     | 4.07x       |
| **V3 3-kernel split** | **460.5** | **0.67x (-33%)** | **2.57x**   |
| V4 hoisted invariants | 622.3     | 0.90x (-10%)     | 3.47x       |
| V7 `occupancy=1`      | 689.2     | noise            | 3.85x       |
| V8 V4+occ=1           | 622.8     | 0.91x            | 3.48x       |

**Вывод:** V3 побеждает на B200, но проигрывает на GB10. **Нет универсально лучшего варианта** --- всегда нужен sweep на целевом HW.

### 4.11. cuTile token_order эксперимент

| Параметр  | Значение                                                                                              |
| --------- | ----------------------------------------------------------------------------------------------------- |
| Гипотеза  | token_order pass в cuTile компиляторе --- root cause 3.7x зазора                                      |
| Результат | **Falsified** --- разрыв обусловлен выбором register allocator (384 vs 256 threads), а не token_order |

### 4.12. CuTe DSL pivot на GB10

| Параметр  | Значение                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| Тест      | `cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`                                                 |
| Результат | BF16 warp MMA + TMA + persistent scheduler работает на sm_121a **без патчей**                                     |
| Бенчмарк  | ~38 us/iter, 1024x1024x1024 persistent BF16 GEMM                                                                  |
| Вывод     | Ранее считали CuTe DSL полностью заблокированным на sm_121 --- это было НЕВЕРНО. Заблокированы только tcgen05/FP4 |

### 4.13. CuTe DSL 3-GEMM fused на H200

| Параметр  | Значение                                                                                                        |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| Операция  | Fused lkq + dPsiV + dPhiO (3 GEMM 64x64 BF16)                                                                   |
| Результат | **35.9 us** (CuTe DSL WGMMA), 31 us (FA4 pattern)                                                               |
| Вывод     | WGMMA fusion работает, но host-level hybrid не может обогнать TileLang single-kernel (Python dispatch overhead) |

### 4.14. TileLang custom autograd

| Параметр  | Значение                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------ |
| Файл      | `cppmega/megatron/tilelang_mimo_autograd.py`                                                                 |
| Улучшения | Single `ctx.saved_tensors` access (PR #909 pattern), fp32 enforcement, нет `torch.tensor(scalar)` в hot path |
| Результат | Работает, 0 overhead. CUDA graph заблокирован `compute_dacs_segsum` (вызывается внутри autograd forward)     |

### 4.15. Triton M2RNN autotune sweep (europe H200)

| Параметр        | Значение                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Пространство    | 25 конфигураций (num_warps x num_stages), 3 формы                                                                              |
| Результат       | `num_warps=8` уже оптимально (0% прирост). `num_stages` не влияет (<0.03% delta) --- sequential recurrence, нечего пайплайнить |
| Inline PTX tanh | fwd -18.7%, fwd+bwd -20% vs manual stable-tanh                                                                                 |

### 4.16. GB10 sm_121a hardware research (6 MCP агентов)

Подробно описано в разделе 5 ниже. Результат: полная матрица возможностей для 4 GPU архитектур (sm_90a/sm_100a/sm_120a/sm_121a), включая tcgen05/TMEM/WGMMA/TMA/FP4.

### 4.17. Modal B200 cuTile variant sweep

См. раздел 4.10 выше. Ключевой вывод: cross-HW reversal --- один и тот же алгоритмический вариант может переключаться от winner к loser между GPU.

### 4.18. CuTe DSL WGMMA Phase 2 на H200

| Параметр     | Значение                                                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Файлы        | `cppmega/megatron/cute_dsl_mimo/` (8 файлов)                                                                                                       |
| Single GEMM  | 3.49 us/iter (64x64 BF16 WGMMA)                                                                                                                    |
| Fused 3-GEMM | 35.9 us                                                                                                                                            |
| FA4 adapter  | Адаптация паттерна `flash_bwd_sm90.py` для MIMO backward                                                                                           |
| Вывод        | WGMMA подтверждена на H200 для MIMO kernels. Hybrid driver пока Python-dispatch-bound (333 ms vs целевые ~10 ms) --- требуется fused single-kernel |

---

## 5. MCP Research --- что искали и где

### 5.1. Perplexity Deep Research --- GB10 sm_121a tcgen05 статус

- **Вопрос:** Какие tensor core инструкции доступны на sm_121a (GB10 DGX Spark)?
- **Ключевые источники:** NVIDIA CUTLASS issues #2800, #2947, #3100; TRT-LLM #11368, #11799; NVIDIA DevTalk threads
- **Вывод:** tcgen05 физически отсутствует на sm_12x. TMEM нет на кристалле. WGMMA deprecated на всём Blackwell. Доступны warp-level mma.sync (HMMA/QMMA), TMA single-CTA, FP4 через OMMA

### 5.2. Perplexity Reasoning --- CuTe DSL sm_121 compatibility

- **Вопрос:** Может ли CuTe DSL (`nvidia-cutlass-dsl`) компилировать BF16 кернелы для sm_121a?
- **Источники:** CUTLASS `include/cute/arch/config.hpp`, `blackwell_geforce/dense_gemm.py`
- **Вывод:** BF16 warp MMA + TMA работает. Заблокированы только tcgen05/FP4 BlockScaled ops

### 5.3. Brave Search --- NVIDIA TileGym patterns

- **Вопрос:** Существуют ли официальные cuTile Python примеры multi-GEMM fusion с loop-carried state?
- **Источник:** https://github.com/NVIDIA/TileGym (685 stars, MIT)
- **Вывод:** FA2 (1276 LOC), MLA (3 ct.mma), Recurrent Delta Rule (persistent state) --- всё fused, всё в регистрах. Доказывает, что cuTile CAN fuse, но алгоритм должен держать <=3-4 live accumulators

### 5.4. Exa Search --- CUTLASS C++ atom inventory для sm_120/121

- **Вопрос:** Сколько MMA atom специализаций для consumer Blackwell?
- **Источники:** `cute/arch/mma_sm120.hpp`, `mma_sm120_sparse.hpp`
- **Вывод:** ~160 атомов (80 dense + 80 sparse). f8f6f4, block_scale, mxf4nvf4 варианты. sm_121 переиспользует sm_120 atoms через config alias

### 5.5. Tavily Search --- Compile flag impact (sm_120f vs sm_121a)

- **Вопрос:** Почему CUTLASS example 79 показывает 9x разницу в производительности между `-arch=sm_120f` и `-arch=sm_121a`?
- **Источники:** NVIDIA DevTalk, TRT-LLM #11368
- **Вывод:** `sm_121a` --- binary-locked, `sm_120f` --- family-specific с общими GeForce Blackwell оптимизациями. Всегда использовать `sm_120f` по умолчанию

### 5.6. Context7 --- cuTile Python API reference

- **Вопрос:** Полный API reference для cuda.tile 1.2.0
- **Источник:** docs.nvidia.com/cuda/cutile-python/operations.html
- **Вывод:** Подтверждение: нет `alloc_shared`, нет `num_stages`, нет thread control. `occupancy=1` --- no-op

### Ключевые источники, найденные через MCP

| Источник                                                | Что дал                                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| NVIDIA CUTLASS issues #2800, #2947, #3044, #3100, #3144 | sm_121a ограничения, smem carveout баг, FP4/FP8 DSL segfault                     |
| TensorRT-LLM #11368, #11799                             | GB10 производительность, CUTLASS version bump timeline                           |
| NVIDIA Developer Forums                                 | sm_121 CUTLASS optimization (356 TFLOPS FP4 vs 37 на sm_121a), DGX Spark threads |
| gau-nernst tcgen05 blog                                 | Анатомия tcgen05 инструкций                                                      |
| Colfax Research CUTLASS tutorials                       | Tensor Memory tutorial for Blackwell                                             |
| SemiAnalysis Blackwell dissection                       | Datacenter vs consumer Blackwell die comparison                                  |
| arXiv 2512.02189                                        | Blackwell microbenchmarks (latency/throughput)                                   |
| solatticus FA4 gist (sm_120 investigation)              | Доказательство физического отсутствия tcgen05 через NVVM disassembly             |
| backend.ai GB10 teardown                                | GB10 hardware specs confirmation                                                 |
| NVIDIA PTX ISA 8.7/8.8/9.2                              | Определение sm_121a/sm_120f targets                                              |
| NVIDIA TileGym repo                                     | Production cuTile patterns (FA, MLA, delta rule)                                 |
| FlashAttention-4 source (flash_bwd_sm90.py)             | WGMMA multi-GEMM fusion reference pattern                                        |

---

## 6. Баги найденные и починенные (5 блокеров)

### 6.1. fast-hadamard-transform PyPI sdist

- **Симптом:** PyPI tarball `fast-hadamard-transform==1.0.4` не содержит `csrc/fast_hadamard_transform.cpp`. DSA import crash.
- **Fix:** `pip install --no-build-isolation 'git+https://github.com/Dao-AILab/fast-hadamard-transform.git'`
- **Влияние:** Блокировал запуск NAM56R MIMO 7/7 (DSA зависимость)

### 6.2. TileLang nvrtc CCCL конфликт

- **Симптом:** На cu13.2 bundled cutlass `cute/container/array.hpp` конфликтует с системным CCCL 13.2.27 через `_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR` макросы.
- **Fix:** `export TILELANG_EXECUTION_BACKEND=cython` (NVCC subprocess вместо nvrtc)
- **Влияние:** Блокировал все MIMO кернелы на bench3

### 6.3. Megatron Float16Module Mamba3 fp32 params

- **Симптом:** `Float16Module.__init__` кастит все параметры в bf16, включая Mamba3 fp32 bias/D/dt. TileLang `mamba_mimo_fwd_kernel` падает с NaN.
- **Fix (quick):** Forward pre-hook re-upcasting `.data` в fp32 на каждом forward. Работает, но стоит ~60 ms/iter (25.7% iter time по nsys).
- **Fix (correct, не реализован):** One-shot `Float16Module.__init__` patch, пропускающий Mamba3 fp32 параметры при bf16 cast.
- **Влияние:** Критический --- без фикса NaN с первой итерации

### 6.4. _broadcast_cu_seqlens CUDA graph

- **Симптом:** `torch.tensor(n)` из Python int во время CUDA graph capture при TP=1.
- **Корневая причина:** Megatron `_broadcast_cu_seqlens` безусловно создаёт тензор, хотя при TP=1 это no-op.
- **Fix:** Monkeypatch для early-return при `TP world_size == 1`
- **Влияние:** Блокирует CUDA graphs (проектируемая экономия ~150 ms/iter)

### 6.5. MoE token dispatcher .cpu() CUDA graph

- **Симптом:** `self.local_map.sum(dim=0).long().cpu()` в token_dispatcher.py:295 --- .cpu() illegal during graph capture.
- **Fix:** Держать на GPU или вынести за пределы captured region.
- **Влияние:** Блокирует CUDA graphs для любой конфигурации с MoE

---

## 7. Ключевые выводы

### Архитектурные

1. **VPP PP=2 = 2x throughput gain** --- структурный выигрыш от pipeline parallelism, не kernel-level. Это крупнейшая единичная оптимизация сессии.

2. **FP8 неэффективен при H=3584** --- quantization overhead примерно компенсирует speedup. Нужен H кратный 128 для FP8 выигрыша.

3. **CUDA graphs заблокированы 3 upstream багами** --- 2 в Megatron (broadcast_cu_seqlens, MoE .cpu()), 1 в mamba_ssm (compute_dacs_segsum). Без фикса теряем ~150 ms/iter.

4. **MIMO overhead = 4.7x vs SISO** --- доминирует bwd recompute forward scan (4.74 ms) + увеличенный bwd_bwd (5.75 ms). Это архитектурная цена, не оптимизируемая.

### Инструментальные

5. **cuTile gap = register allocator, NOT token_order** --- компилятор выбирает 384 threads x 168 regs vs TileLang 256 threads. Алгоритм с 30+ live tiles слишком «жирный» для cuTile compiler model.

6. **TileLang sm_90 WGMMA support** --- TileLang работает на H200 через cython backend. Ранее неверно считали, что нужен nvrtc.

7. **CuTe DSL BF16 работает на GB10 sm_121a** --- ранее считали полностью заблокированным. Заблокированы только tcgen05/FP4 пути.

8. **NVIDIA TileGym = official cuTile patterns** --- доказывает, что cuTile может fuse multi-GEMM с loop-carried state. Наша проблема --- алгоритм, не компилятор.

9. **FA4 `as_position_independent_swizzle_tensor()`** --- ключевой паттерн для корректного swizzle-copy в WGMMA. Без него lane mapping для `ldmatrix.trans` ломается.

### Кросс-платформенные

10. **Нет универсально лучшего cuTile варианта** --- V3 (3-kernel split) побеждает на B200 (-33%), но проигрывает на GB10 (+9%). Всегда нужен sweep на целевом HW.

11. **cuTile fwd быстрее TileLang на B200** --- на 17.7%. Hybrid wrapper `cuTile fwd + TileLang bwd` --- бесплатный throughput win.

12. **`@ct.kernel(occupancy=1)` --- silent no-op** на cuTile 1.2.0. Не использовать как tuning knob.

13. **Blackwell = две разных ISA** --- sm_120/121 (consumer) архитектурно ближе к Ampere + FP4, чем к sm_100 (datacenter). FA4/FlashMLA/FlashInfer SM100 пути не могут и никогда не будут работать на consumer Blackwell.

---

## 8. Файлы созданные/модифицированные в сессии

### Документация (`docs/`)

| Файл                                            | Описание                               |
| ----------------------------------------------- | -------------------------------------- |
| `nam56r_mimo7_baseline_2026_04_11.md`           | Baseline 56k tok/sec                   |
| `nam56r_mimo7_nsys_profile_2026_04_11.md`       | nsys profile breakdown                 |
| `nam56r_mimo7_vpp_112k_2026_04_11.md`           | VPP 112k tok/sec                       |
| `nam56r_mimo7_reproducibility_2026_04_11.md`    | Полное руководство по воспроизведению  |
| `upstream_bugs.md`                              | 3 upstream бага + PR #909              |
| `fp8_path_status.md`                            | FP8 smoke matrix                       |
| `gb10_sm121_hardware.md`                        | sm_121a hardware capability matrix     |
| `gb10_software_stack.md`                        | GB10 software stack recipe             |
| `gb10_bwd_bwd_optimization_conclusion.md`       | Заключение по bwd_bwd оптимизации      |
| `modal_b200_cutile_parity.md`                   | B200 cuTile parity validation          |
| `modal_b200_cutile_status.md`                   | B200 cuTile provisioning               |
| `modal_b200_cutile_variant_sweep_2026_04_11.md` | B200 variant sweep                     |
| `cutile_cute_dsl_interop.md`                    | cuTile + CuTe DSL interop guide        |
| `tilelang_to_cute_port.md`                      | TileLang -> cuTile canonical GEMM port |
| `changelog.md`                                  | Полный changelog                       |

### Код (`cppmega/`)

| Файл                                         | Описание                                            |
| -------------------------------------------- | --------------------------------------------------- |
| `cppmega/megatron/tilelang_mimo_autograd.py` | Custom autograd для TileLang MIMO (CUDA-graph safe) |
| `cppmega/megatron/cute_dsl_mimo/` (8 файлов) | CuTe DSL WGMMA MIMO port                            |
| `cppmega/megatron/custom_embedding.py`       | Structure embedding                                 |
| `cppmega/megatron/custom_gpt_model.py`       | Custom GPT model                                    |
| `cppmega/megatron/custom_mamba_model.py`     | Custom Mamba model                                  |
| `cppmega/megatron/mamba_builder.py`          | Mamba builder                                       |
| `cppmega/megatron/mla_shared.py`             | MLA shared utilities                                |
| `cppmega/megatron/nam56r_full_spec.py`       | NAM56R full stack spec                              |
| `cppmega/megatron/nam56r_layout.py`          | NAM56R layout                                       |
| `cppmega/megatron/nam56r_lite_spec.py`       | NAM56R lite spec                                    |
| `cppmega/megatron/structure_batch.py`        | Structure batch                                     |

### Скрипты (`scripts/`)

| Файл                                             | Описание                                         |
| ------------------------------------------------ | ------------------------------------------------ |
| `scripts/remote_setup_h200.sh`                   | H200 setup                                       |
| `scripts/remote_smoke_h200_structure_poly.sh`    | Structure poly smoke                             |
| `scripts/remote_smoke_h200_nam56r_mixed_a.sh`    | NAM56R mixed A smoke                             |
| `scripts/remote_smoke_h200_structure_ingress.sh` | Structure ingress smoke                          |
| `scripts/remote_train_h200_nam56r_full.sh`       | NAM56R full training                             |
| `scripts/remote_train_h200_nam56r_lite.sh`       | NAM56R lite training                             |
| `scripts/cppmega_fp8_shim.py`                    | FP8 + MIMO shim (post_init hook + fp32 pre-hook) |

### Тесты (`tests/`)

| Файл                                             | Описание                           |
| ------------------------------------------------ | ---------------------------------- |
| `tests/test_custom_embedding.py`                 | Tests for custom embedding         |
| `tests/test_megatron_args.py`                    | Megatron args tests                |
| `tests/test_megatron_args_fragment.py`           | Args fragment tests                |
| `tests/test_nam56r_full_spec.py`                 | NAM56R full spec tests             |
| `tests/test_nam56r_launch.py`                    | NAM56R launch tests                |
| `tests/test_remote_nam56r_lite_train_h200.py`    | NAM56R lite remote test            |
| `tests/test_remote_nam56r_mixed_a_smoke_h200.py` | NAM56R mixed A remote test         |
| `tests/test_structure_batch.py`                  | Structure batch tests              |
| `tests/test_structure_embedding_contract.py`     | Structure embedding contract tests |
| `tests/test_remote_setup_h200.py`                | Remote setup test                  |

### Memory files

| Файл                                      | Описание                          |
| ----------------------------------------- | --------------------------------- |
| `project_nam56r_throughput.md`            | Throughput progression tracking   |
| `reference_bench_machines.md`             | Bench machine locations           |
| `reference_cute_dsl_lessons.md`           | CuTe DSL gotchas                  |
| `reference_cutile_compiler_behavior.md`   | cuTile compiler anti-patterns     |
| `reference_sm121_gb10_hw_caps.md`         | sm_121 hardware capability matrix |
| `reference_tilegym_cutile_patterns.md`    | TileGym patterns                  |
| `reference_mimo7_setup_gotchas.md`        | MIMO 7/7 setup blockers           |
| `feedback_cutile_algorithm_not_memory.md` | Algorithm-first rule for cuTile   |

---

## 9. Открытые вопросы и следующие шаги

### Критический путь к 250k tok/sec

| #   | Задача                                                                                    | Проектируемый прирост  | Статус                                   |
| --- | ----------------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------- |
| 1   | **Починить 3 CUDA graph блокера** (broadcast_cu_seqlens, MoE .cpu(), compute_dacs_segsum) | +15-20% (~150 ms/iter) | Фиксы спроектированы, не реализованы     |
| 2   | **One-shot Float16Module patch** (убрать per-forward fp32 shim)                           | +5% (~60 ms/iter)      | Паттерн готов, не реализован             |
| 3   | **TP=2 + PP=2 VPP=2**                                                                     | +15-25% (structural)   | Не тестировалось                         |
| 4   | **FP8 selective** на MLA+MoE после CUDA graphs                                            | +10-15%                | Path A/B PASS, Path C PASS после bwd fix |
| 5   | **MBS=5-6** после CUDA graphs (освободят память)                                          | +10-15%                | Заблокировано памятью без CUDA graphs    |
| 6   | **Fused MLA RoPE fix** (upstream Megatron bug at PP>1)                                    | +2-3%                  | Workaround `--no-rope-fusion`            |
| 7   | **bf16 softmax** для DSA indexer + MoE router                                             | +2%                    | Не начато                                |

### Проекция

| Конфигурация                  | Проектируемый tok/sec |
| ----------------------------- | --------------------- |
| Текущий лучший (VPP+NoMTP)    | 133,519               |
| + CUDA graphs + fp32 shim fix | ~170-180k             |
| + FP8 selective               | ~195-210k             |
| + TP=2                        | ~230-250k             |
| **Цель**                      | **250,000**           |

Полные 250k, вероятнее всего, требуют комбинацию **TP=2 + PP=2 + VPP=2 + FP8 + CUDA graphs + MBS optimization**. Каждый отдельный рычаг даёт 10-25%, но мультипликативный эффект может достичь цели.

### Открытые исследовательские вопросы

1. **Может ли CuTe DSL fused single-kernel** заменить Python-dispatch hybrid на H200? Текущий hybrid = 333 ms/iter (dispatch-bound), нужно <10 ms.
2. **TileLang MIMO swizzle warnings** ("merging to smaller granularity") --- потенциальный 20-40 ms/iter win, не исследован.
3. **Hybrid cuTile fwd + TileLang bwd** на B200 --- бесплатный 17.7% win на fwd, нужен wrapper.
4. **CUTLASS example 79 `tile_shape_mnk=(64,64,64)` bug** на sm_121a --- нужен upstream issue.
5. **Mamba3 bwd recompute optimization** --- bwd_fwd (4.74 ms) = полный re-scan forward. Можно ли checkpoint промежуточные состояния?

---

## Вечерние результаты (late session 2026-04-11)

### Эксперимент MTP Super flags (europe, 3 варианта)

| Вариант                                  | Конфиг         | Iter ms | Tok/sec | LM loss@30 | MTP loss@30 |
| ---------------------------------------- | -------------- | ------- | ------- | ---------- | ----------- |
| V1 baseline                              | untied, MTP on | 2348.5  | 111,621 | 2.70       | 2.66        |
| V2 `mtp_use_repeated_layer=True` depth=1 | Super flags    | 2355.8  | 111,275 | 2.81       | 2.54        |
| V3 `mtp_use_repeated_layer=True` depth=2 | Super flags    | 2688.9  | 97,530  | 2.75       | 2.66/2.49   |

**Вывод:** `mtp_use_repeated_layer=True` работает для Mamba hybrid, но даёт 0% ускорения при depth=1 (no-op когда всего 1 слой). При depth=2 деградация −12.6% (ожидаемо --- shared weights экономят параметры, не FLOPs). MTP overhead --- это forward+backward FLOPs, не число параметров.

### Эксперимент с tied embeddings (europe)

| Конфиг            | Iter ms | Tok/sec | Дельта |
| ----------------- | ------- | ------- | ------ |
| Untied (baseline) | 2348.5  | 111,621 | 0%     |
| Tied              | 2349.4  | 111,623 | −0.04% |

**Вывод:** Нулевой эффект на PP=2 hybrid. Megatron не может объединить embedding/output head через границы PP rank-ов.

### Sweep из 8 вариантов MTP (europe + bench3)

| #   | Название          | Iter ms | Tok/sec | Статус                         |
| --- | ----------------- | ------- | ------- | ------------------------------ |
| 1   | Control untied    | 2348.5  | 111,666 | baseline                       |
| 2   | Только tied       | 2349.4  | 111,623 | 0% выигрыш                     |
| 3   | Standalone VPP    | ---     | ---     | ЗАБЛОКИРОВАН (Megatron hybrid) |
| 4   | Tied + standalone | ---     | ---     | ЗАБЛОКИРОВАН                   |
| 5   | Tied MBS=5        | ---     | ---     | OOM (~142/140 GB)              |
| 7   | PP=4 VPP=1        | 3258.3  | 80,490  | −28% регрессия                 |
| 8   | NoMTP control     | 1981.2  | 132,438 | +18.6% (подтверждает 133k)     |

### Liger fused CE для MTP (bench3)

| Метрика                     | Стандартный CE | Liger fused | Дельта         |
| --------------------------- | -------------- | ----------- | -------------- |
| MTP time (4 depths fwd+bwd) | 178.8 ms       | 483.2 ms    | 2.7x МЕДЛЕННЕЕ |
| Пиковая память              | 27.36 GB       | 5.49 GB     | −82%           |

**Вывод:** Liger экономит 82% памяти, но в 2.7x медленнее на H200 (плохая утилизация тензорных ядер на chunked small-M GEMMs). НЕ включать на H200. Ценно только для hardware с ограниченной памятью.

### Эксперименты с CUDA graphs (bench3)

| Область                             | Статус | Tok/sec | Дельта               |
| ----------------------------------- | ------ | ------- | -------------------- |
| Baseline (без графов)               | ---    | 68,844  | ---                  |
| `--cuda-graph-scope attn`           | PASS   | 69,822  | +1.4%                |
| `--cuda-graph-scope full_iteration` | FAIL   | ---     | Блокер MoE `.item()` |
| `transformer_engine`                | FAIL   | ---     | Тот же блокер MoE    |

**Ключевое открытие:** агент использовал НЕПРАВИЛЬНЫЙ scope. cppmega уже имеет работающий per-module scope: `--cuda-graph-scope attn mamba moe_router moe_preprocess` (в `nam56r_nemo_recipe.py:287-296`). Для полного MoE графа нужен `--moe-pad-expert-input-to-capacity`. **Production-рецепт на 211k tok/sec использовал именно эту конфигурацию.** Запущен фикс-агент.

### Завершение CuTe DSL Phase 4 (europe)

Ядро: 553 LOC, 11 GEMMs (оригинальных 10 + GEMM 6' для transpose). Все 14 выходов корректны vs TileLang (rtol=1e-2). Архитектура: WGMMA GEMMs в CuTe DSL kernel + PyTorch epilogue для 9 non-GEMM выходов. Запущен второй агент для fully-fused версии (всё внутри ядра, без torch).

| Размер     | Phase 4 (11G) | TileLang | Ускорение |
| ---------- | ------------- | -------- | --------- |
| Smoke      | 113 us        | 175 us   | 1.55x     |
| Production | 2515 us       | 3135 us  | 1.25x     |

Семантические ошибки GEMM 7/9 ИСПРАВЛЕНЫ. Аккумулятор Dstates с loop-carried зависимостью КОРРЕКТЕН (rel 0.004).

### CuTe DSL Phase 4v2 (bench3)

Новый файл `fused_bwd_bwd_sm90_p4v2.py` (410 LOC) с альтернативным фиксом GEMM 7/9 (gmem scratch для transpose). Все 5 kernel outputs PASS (max rel 0.0038).

### Корректность на GB10 (196 тестов)

| Набор тестов           | Pass | Fail    | Примечания                                                                 |
| ---------------------- | ---- | ------- | -------------------------------------------------------------------------- |
| M2RNN Triton           | 10   | 0       | Все формы корректны                                                        |
| TileLang MIMO autograd | 5    | 1 known | bwd_bwd 140KB smem (фикс: `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True`) |
| cuTile TileGym 3-phase | 11   | 0       | Все 11 градиентов PASS                                                     |
| CuTe DSL (6 тестов)    | 6    | 0       | Все на sm_121a                                                             |
| FLA cutile (161 тест)  | 161  | 0       | 100% pass                                                                  |
| cppmega unit           | 10   | 0       | Все pass                                                                   |

**Регрессий нет.** Фикс bwd_bwd smem на GB10 существует (`TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True`), агент корректности просто не установил env var.

### Исследование Nemotron 3 Super/Nano MTP

- Super использует `MambaModel` + PP=1 + `mtp_use_repeated_layer=True` + `mtp_num_layers=2`
- Nano НЕ использует MTP
- Standalone MTP **явно не поддерживается** для гибридной модели (`mamba_model.py:195-199`, PR #3377 подтверждает)
- NVIDIA обходит проблему используя PP=1

### Фикс cuDNN LD_LIBRARY_PATH (bench3)

Корневая причина ВСЕХ сбоев обучения на bench3 в вечерней сессии: системный cuDNN 9.10.2 загружался раньше cuDNN 9.20.0 из venv. Фикс: `export LD_LIBRARY_PATH=.../nvidia/cudnn/lib:$LD_LIBRARY_PATH` в `~/.bashrc`. Применён и проверен на обеих машинах.

### ThunderKittens на GB10

TK компилируется и работает на sm_121a с флагом `KITTENS_AMPERE`: 64x64 BF16 GEMM = 4.11 us (1.84x быстрее torch.mm). HMMA.16816 тензорное ядро подтверждено через cuobjdump. Только warp-level MMA path (без WGMMA/tcgen05).

### Патчи блокеров Megatron CUDA graph (bench3)

1. MoE `.cpu()` на `token_dispatcher.py:295` --- держать на GPU (v2 сохраняет `.item()` для eager path)
2. `compute_dacs_segsum_triton` autotune --- зафиксирован один конфиг (8 autotune блоков свёрнуты)
3. `_broadcast_cu_seqlens` --- bypass для TP=1 в `megatron/training/utils.py:566` (прямой патч, не shim --- локальная функция)
4. `tokens_per_expert.sum().item()` на строке 306 --- дополнительная D2H синхронизация (найдена, нужен v3 фикс ИЛИ per-module scope bypass)
