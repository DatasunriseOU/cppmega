# План оптимизации MTP в NAM56R — 2026-04-11

**Статус:** 6 research-агентов завершили анализ MTP overhead 2026-04-11. Найдены корневые причины 19% overhead; спланирована матрица из 8 вариантов, чтобы привести overhead к 3-5% как у DeepSeek/Meta, сохранив MTP архитектурно включённым.

## Проблема

Текущий baseline VPP PP=2 VPP=2 MBS=4 GBS=64 с MTP ON:
- **Время итерации: 1963 ms**
- **Throughput: 112,152 tok/sec**
- **MTP overhead: 374 ms/iter (19.1%)** — измерено bench3 агентом через удаление MTP (+19% tok/sec без MTP)

**19% — это в 2-10 раз больше опубликованных цифр.** Meta (arXiv 2404.19737) сообщает 0% с sequential detach паттерном; DeepSeek-V3 (arXiv 2412.19437) отчитывается о ~2-5% для одной MTP depth на модели из 61 слоя. Megatron-LM docs описывают MTP как «один дополнительный decoder layer» ≈ 1/52 = 1.9% для нашей модели.

## Корневые причины — 6 research-агентов сошлись на этих выводах

### 1. Флаг `--untie-embeddings-and-output-weights` в launch-скрипте (`remote_train_h200_nam56r_full.sh:122`)

Наш скрипт **РАЗВЯЗЫВАЕТ** embedding и output head. DeepSeek-V3 **явно связывает** их: *«для каждого MTP-модуля его embedding разделяется с основной моделью… его output head разделяется с основной моделью».*

С untied:
- MTP head получает свою собственную матрицу `[hidden × vocab] = [3584 × 65536]` = **~235M дополнительных параметров**
- Отдельный `[B×S, V]` GEMM каждую итерацию
- Дополнительная память для градиентов и состояния оптимизатора (~2-4× для Adam)
- Нет выгоды от weight sharing

**Фикс: удалить 1 строку `--untie-embeddings-and-output-weights`.** Megatron по умолчанию tied.

### 2. MTP-слой НЕ размещён в standalone VPP chunk

Megatron-LM поддерживает `mtp_standalone` placement (layout pattern `"...tt|m|L"` согласно docs). Когда standalone, MTP получает свой отдельный virtual pipeline chunk → **interleaved 1F1B scheduler перекрывает MTP compute с main backward на разных micro-batches.**

Сейчас: MTP прилеплен к последнему main chunk → chunk становится толстым (460ms + 374ms = 834ms) → pipeline bubble ждёт этот медленный chunk на каждом micro-batch.

Standalone: каждый chunk ~460ms, MTP chunk ~374ms, они бегут параллельно через разные micro-batches в 1F1B scheduling. **Эффективный wall-clock cost MTP ≈ маленькое добавление bubble (~70ms) вместо полных 374ms.**

**Ожидаемая экономия: ~300 ms/iter → 1663 ms iter → ~157,700 tok/sec.**

### 3. Нет fused cross-entropy (Liger / Apple Cut-CE)

`process_mtp_loss()` в Megatron-LM делает `output_layer(hidden_states) → compute_language_model_loss` который материализует полный тензор logits `[B×S, V]`. Liger-Kernel даёт `fused_linear_cross_entropy` (Triton), который чанкает matmul и вычисляет локальный log-sum-exp без хранения logits. Это **30-50% MTP cost** согласно статье Meta.

Не флаг launch — нужна интеграция. Отложить до применения фиксов #1 и #2.

### 4. `mtp_loss_scaling_factor` возможно использует default 0.3

DeepSeek-V3 расписание: λ=0.3 первые 10T токенов, **λ=0.1 финальные 4.8T токенов**. NeMo DeepSeek-V3 recipe default — `mtp_loss_scaling_factor=0.1`. Megatron-Core default — 0.1. Проверить наш `build_nam56r_megatron_native_args` и launch-скрипты — если значение выше, снизить.

Примечание: λ масштабирует величину градиента, НЕ FLOPs. Backward всё равно бежит полностью. Эффект меньше чем у #1 и #2, но корректировать бесплатно.

### 5. Прячет ли DeepSeek MTP в DualPipe bubble? **НЕТ.**

Гипотеза пользователя опровергнута Exa-агентом (`a6f1ea5ac59d459fb`). Source code DualPipe (`dualpipe/dualpipe.py`, `dualpipev.py`) содержит **ноль упоминаний MTP** — grep подтверждён. DualPipe перекрывает (forward-of-microbatch-X с backward-of-microbatch-Y) и (MoE all-to-all с compute), НЕ (main fwd с MTP fwd). DeepSeek платит MTP cost каждый шаг; они держат его дешёвым через (а) shared embedding/output head, (б) depth=1, (в) λ schedule. Никакой магии scheduling нет.

### 6. nanochat уже имеет оптимизированный MTP — ~0% overhead

Наш sister-проект `nanochat` (`/Volumes/external/sources/nanochat/`) реализует FastMTP-style shared-block MTP и измеряет **MTP3 +6% быстрее** чем NoMTP на H200 NAM52 4.1B (1247 против 1176 tok/s), с улучшением loss на 3.4%. Техники (`nanochat/mtp.py`):
1. Shared block — 1 `nn.Linear(2D→D)` + 1 Transformer Block **рекурсивно K раз** (не K отдельных блоков)
2. Weight-tied `wte` + `lm_head` передаются в forward
3. Roll-and-mask static shapes (предотвращает K-way graph recompile)
4. Activation checkpointing внутри K-loop
5. Fused linear+CE на каждой depth (Liger/CCE стиль)
6. Cadence scheduling (`mtp_cadence` динамический skip; λ=0 → MTP forward полностью пропускается)
7. mtp.* параметры → AdamW (не Muon)
8. bf16 `lm_head` с сохранением маркеров Megatron TP vocab-parallel

## Матрица оптимизации — sweep из 8 вариантов на 2× H200×8

**Baseline для побития: 112,152 tok/sec (VPP PP=2 VPP=2 MTP ON, текущая конфигурация)**

| #   | Конфиг                                  | Untied→Tied | MTP standalone |  MBS  |   MTP   | Доп. | Ожидаемый tok/sec                 |
| --- | --------------------------------------- | :---------: | :------------: | :---: | :-----: | ---- | --------------------------------- |
| 1   | Текущий baseline (control)              |   untied    |      нет       |   4   |   on    | —    | **112,152 (проверка)**            |
| 2   | Только tied embeddings                  |  **tied**   |      нет       |   4   |   on    | —    | ~125-135k                         |
| 3   | Только standalone VPP                   |   untied    |     **да**     |   4   |   on    | —    | ~140-155k                         |
| 4   | **Tied + standalone**                   |  **tied**   |     **да**     |   4   |   on    | —    | **~157-165k (основная цель)**     |
| 5   | Полные фиксы + MBS=5                    |    tied     |       да       |   5   |   on    | —    | ~170-185k                         |
| 6   | Полные фиксы + MBS=6                    |    tied     |       да       |   6   |   on    | —    | ~180-200k (если память позволяет) |
| 7   | Полные фиксы + PP=4 VPP=2               |    tied     |       да       |   4   |   on    | PP=4 | тест bubble shift                 |
| 8   | NoMTP control (архитектурная регрессия) |    tied     |       —        |   4   | **off** | —    | ~133k (sanity check)              |

**Ожидаемый победитель:** вариант 5 или 6 (~170-200k). На 200k мы в **1.25× от цели 250k**, остальные рычаги закроют этот gap (CUDA graphs после фикса Megatron багов, TP=2).

## Логика выбора вариантов

- Варианты **2, 3** изолируют индивидуальный вклад tied-embedding и standalone-VPP фиксов
- Вариант **4** — основная production-конфигурация
- Варианты **5, 6** тестят MBS scaling поверх — tied embeddings освобождают ~235M параметров, что освобождает память активаций
- Вариант **7** тестит даёт ли PP=4 VPP=2 лучший bubble чем PP=2 VPP=2 когда MTP размещён правильно
- Вариант **8** — control для верификации что мы правильно измеряем MTP cost (должен соответствовать прежнему 133k NoMTP)

## Жёсткие ограничения

- **MTP должен остаться ВКЛЮЧЁН** в production-вариантах (архитектурная фича, не optional knob). Вариант 8 только control
- **Реальные данные** (`clang_semantic_4k_v10_train`), без `--mock-data`
- **Loss должен сходиться** (iter 30 LM loss < 3.5, без NaN)
- **Correctness gate** на варианте 2 (tied embeddings): если loss расходится — значит существующий checkpoint был обучен с untied весами и нужен свежий init. Ожидается: свежий init тренируется нормально (DeepSeek использует tied с самого начала)

## Развёртывание по 2× H200×8

**bench3 (LOCATION_1):** варианты 1, 2, 5, 7
**europe (LOCATION_2):** варианты 3, 4, 6, 8

Параллельное выполнение по машинам. Каждый вариант = 30-iter training run (~5-10 мин wallclock включая JIT compile). Общий sweep wallclock: ~60-80 мин при хорошей параллелизации.

## Источники

- Perplexity Pro gemini-3.1-pro research: архитектура DeepSeek-V3 + опровержение DualPipe мифа (агент `a61122d36f9efd016`)
- Exa deep research: Meta + Medusa + FastMTP + теоретическая cost analysis (агент `a2e67295ebd8c3291`)
- Brave research: Megatron-LM + NeMo + Liger-Kernel (агент `ae61832d919145955`)
- Tavily NVIDIA docs: Megatron `multi_token_prediction.py` 1479 LOC, NeMo defaults (агент `ae27648f4fd77cdb3`)
- Exa DualPipe investigation: подтверждено НЕТ MTP scheduling в DualPipe source (агент `a6f1ea5ac59d459fb`)
- nanochat MTP patterns: `/Volumes/external/sources/nanochat/mtp.py` FastMTP shared-block дизайн (агент `a81e6a5e3eb405192`)

## Статьи

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) §2.2 MTP, §3.2.1 DualPipe
- [Meta Better & Faster Multi-Token Prediction](https://arxiv.org/abs/2404.19737) Gloeckle et al. ICML 2024
- [Medusa: Simple LLM Inference Acceleration](https://arxiv.org/abs/2401.10774)
- [FastMTP: Shared-block MTP with 2.03× speedup](https://arxiv.org/abs/2509.18362) Tencent, Sep 2025
- [MuToR: MTP needs registers](https://arxiv.org/abs/2505.10518) May 2025

## Upstream источники

- [Megatron-LM multi_token_prediction.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py) (1479 LOC, production)
- [Megatron-LM MTP docs](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/multi_token_prediction.md)
- [NeMo DeepSeek-V3 recipe](https://docs.nvidia.com/nemo-framework/user-guide/25.11/llms/deepseek_v3.html)
- [Liger-Kernel fused_linear_cross_entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- [DualPipe source](https://github.com/deepseek-ai/DualPipe) (подтверждено отсутствие MTP awareness)

---

## Результаты оптимизации MTP

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

### Результаты sweep из 8 вариантов MTP (europe + bench3)

| #   | Название          | Iter ms | Tok/sec | Статус                         |
| --- | ----------------- | ------- | ------- | ------------------------------ |
| 1   | Control untied    | 2348.5  | 111,666 | baseline                       |
| 2   | Только tied       | 2349.4  | 111,623 | 0% выигрыш                     |
| 3   | Standalone VPP    | ---     | ---     | ЗАБЛОКИРОВАН (Megatron hybrid) |
| 4   | Tied + standalone | ---     | ---     | ЗАБЛОКИРОВАН                   |
| 5   | Tied MBS=5        | ---     | ---     | OOM (~142/140 GB)              |
| 7   | PP=4 VPP=1        | 3258.3  | 80,490  | −28% регрессия                 |
| 8   | NoMTP control     | 1981.2  | 132,438 | +18.6% (подтверждает 133k)     |

**Ключевые выводы из sweep:**
- Standalone VPP (варианты 3, 4) ЗАБЛОКИРОВАН: Megatron hybrid (`mamba_model.py:195-199`) явно не поддерживает standalone MTP placement. PR #3377 подтверждает.
- NVIDIA Nemotron 3 Super обходит это используя PP=1 (без pipeline parallelism).
- MBS=5 (вариант 5) вызывает OOM при ~142/140 GB --- нет запаса даже с tied embeddings.
- PP=4 VPP=1 (вариант 7) --- регрессия −28% из-за увеличения pipeline bubble.
- Основная цель плана (вариант 4, tied + standalone, ~157-165k) недостижима без upstream изменений Megatron для поддержки standalone MTP на гибридных моделях.

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

**Ключевое открытие:** cppmega уже имеет работающий per-module CUDA graph scope: `--cuda-graph-scope attn mamba moe_router moe_preprocess` (в `nam56r_nemo_recipe.py:287-296`). Для полного MoE графа нужен `--moe-pad-expert-input-to-capacity`. **Production-рецепт на 211k tok/sec использовал именно эту конфигурацию.**

### Патчи блокеров Megatron CUDA graph (bench3)

1. MoE `.cpu()` на `token_dispatcher.py:295` --- держать на GPU (v2 сохраняет `.item()` для eager path)
2. `compute_dacs_segsum_triton` autotune --- зафиксирован один конфиг (8 autotune блоков свёрнуты)
3. `_broadcast_cu_seqlens` --- bypass для TP=1 в `megatron/training/utils.py:566` (прямой патч, не shim --- локальная функция)
4. `tokens_per_expert.sum().item()` на строке 306 --- дополнительная D2H синхронизация (найдена, нужен v3 фикс ИЛИ per-module scope bypass)

### Исследование Nemotron 3 Super/Nano MTP

- Super использует `MambaModel` + PP=1 + `mtp_use_repeated_layer=True` + `mtp_num_layers=2`
- Nano НЕ использует MTP
- Standalone MTP **явно не поддерживается** для гибридной модели (`mamba_model.py:195-199`, PR #3377 подтверждает)
- NVIDIA обходит проблему используя PP=1

### Фикс cuDNN LD_LIBRARY_PATH (bench3)

Корневая причина ВСЕХ сбоев обучения на bench3 в вечерней сессии: системный cuDNN 9.10.2 загружался раньше cuDNN 9.20.0 из venv. Фикс: `export LD_LIBRARY_PATH=.../nvidia/cudnn/lib:$LD_LIBRARY_PATH` в `~/.bashrc`. Применён и проверен на обеих машинах.

### Пересмотренный путь оптимизации

Основная цель плана (вариант 4: tied + standalone на ~157-165k) **заблокирована** из-за отсутствия поддержки standalone MTP для гибридных моделей в Megatron. Оставшиеся жизнеспособные рычаги:

| Рычаг                                        | Ожидаемый эффект               | Статус                                                                       |
| -------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| Per-module CUDA graphs (существующий рецепт) | Уже в конфиге 211k             | нужен применённый cuDNN фикс                                                 |
| `--moe-pad-expert-input-to-capacity`         | Включает полный MoE CUDA graph | не тестировался с фиксом                                                     |
| FP8 на MLA+MoE                               | +15-20%                        | ожидает                                                                      |
| TP=2 PP=2 VPP=2                              | +15-25%                        | **заблокировано upstream Mamba3**, нужен `CppmegaMamba3TPMixer` (2026-04-12) |
| Удаление MTP (крайний случай)                | +18.6%                         | архитектурная регрессия                                                      |

### Анализ TP>1 для Mamba3 MIMO (2026-04-12)

**Предположение** (пользователь): TP split встроен в TileLang MIMO kernel.
**Проверка** (bench3 `/mnt/data/venv/lib/python3.13/site-packages/mamba_ssm/`, mamba_ssm 2.3.1):

1. `modules/mamba3.py:27-135` — НЕТ TP awareness:
   - `grep -nE "tp_size|tensor_parallel|process_group|ngroups_local|nheads_local"` → exit 1
   - `self.in_proj = nn.Linear(d_model, d_in_proj)` — plain, не `ColumnParallelLinear`
   - `self.out_proj = nn.Linear(d_inner, d_model)` — plain, не `RowParallelLinear`
   - `dt_bias`, `B_bias`, `C_bias`, `mimo_x/z/o`, `D` — все full-size `nn.Parameter(shape=(self.nheads, ...))` per rank
   - Конструктор не принимает `tp_size` / `process_group`

2. `ops/tilelang/mamba3/mamba3_mimo.py` — TileLang MIMO kernel shape-agnostic:
   - `grep -nE "tp_size|process_group|ngroups_local"` → exit 1
   - `_Mamba3Function.forward` берёт Q/K/V/ADT/DT/Trap/... и передаёт прямо в `mamba_mimo_forward(...)` без инспекции head counts
   - Kernel корректно посчитает на локальном slice, **если его подать** — но сам шардинг не делает

3. `mamba_ssm/distributed/tensor_parallel.py` — есть `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding` primitives, но **НЕТ** Mamba2/Mamba3 TP-aware класса

4. `megatron/core/ssm/mamba_mixer.py` — полный TP built-in (`nheads_local_tp = nheads // tp_size`, `ColumnParallelLinear` in_proj, `RowParallelLinear` out_proj), но **wraps Mamba2 SSD, не Mamba3 MIMO**

**Вывод:** `cppmega/megatron/author_mamba3_spec.py:60` `NotImplementedError("AuthorMamba3Mixer currently supports tensor-model-parallel-size=1 only")` — **корректная assertion**. Убирать её нельзя: на каждом TP rank создастся full-size upstream `Mamba3`, compute реплицируется вместо разделения.

**Путь к TP>1 с полным 7/7 Mamba3 MIMO** — написать `CppmegaMamba3TPMixer` в cppmega (без форка upstream):

1. `in_proj = TEColumnParallelLinear(d_model, d_in_proj)` — output sharded через TP ranks, fused с LayerNorm
2. `out_proj = TERowParallelLinear(d_inner, d_model)` — input sharded + allreduce
3. `nheads_local = nheads // tp_size`, `d_inner_local`, `num_bc_heads_local = ngroups // tp_size`
4. Per-head параметры на локальном shard: `dt_bias[nheads_local]`, `B_bias[nheads_local, r, n]`, `C_bias[nheads_local, r, n]`, `mimo_x/z/o[nheads_local, r, headdim]`, `D[nheads_local]`
5. Подача шарденных `Q/K/V/ADT/DT/Trap` в существующий `cppmega_tilelang_mimo_combined` — kernel сам посчитает на локальном slice (shape-agnostic)
6. `B_norm`/`C_norm` RMSNorm работают per-group на оси `d_state` — cross-rank communication не нужен

Оценка LOC ~150-200. Основной риск: `B_norm`/`C_norm` + MIMO einsum на локальных heads — проверяется unit test TP=1 vs TP=2 numeric parity ДО любого throughput run. Трек: task #76 (Stream B на europe H200).

### План параллельных потоков 2026-04-12

| Поток | Машина              | Задача                                                                                                                    | Task ID |
| ----- | ------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------- |
| A     | bench3 H200×8       | Grid search TP=1 × PP×VPP×MBS×GBS×EP×MTP с full 7/7 MIMO, цель ≥150k tok/sec                                              | #75     |
| B     | europe H200×8       | Написать + unit-test `CppmegaMamba3TPMixer`, затем TP=2 throughput с полным 7/7 MIMO + MLA + DSA + MoE                    | #76     |
| C     | GB10 + Modal B200:2 | NAM56R-half Blackwell features: ThunderKittens FA4 patterns, FP4 DSA indexer (B200), FP8 Mamba3, CuTe DSL BF16 на sm_120f | #77     |

### ВАЖНО: reference implementation для Stream B (2026-04-12)

**Не писать с нуля через TEColumnParallelLinear/TERowParallelLinear.** Рабочий TP-aware Mamba3 реализован в `../nanochat/nanochat/mamba2.py` class `Mamba2Layer` (имя обманчивое --- это **полный Mamba3** со всеми Phase 2/3/4 фичами: MIMO, trapezoidal, learned angles, data-dependent A, complex RoPE, qknorm, learned_bc_norm, outproj_norm). Порт оттуда заметно проще.

**Ключевые места для копирования:**

1. `nanochat/nanochat/mamba2.py:389-720` --- `Mamba2Layer.__init__(config, layer_idx, tp_degree=1, tp_group=None)`:
   - TP args прямо в конструкторе
   - Ручной шардинг:
     ```python
     if tp_degree > 1:
         assert self.d_inner % tp_degree == 0
         self.d_inner = self.d_inner // tp_degree
         if self.ngroups >= tp_degree:
             assert self.ngroups % tp_degree == 0
             self.ngroups = self.ngroups // tp_degree
     self.nheads = self.d_inner // self.headdim  # LOCAL nheads
     ```
   - ВСЕ per-head params (`B_bias`, `C_bias`, `D`, `A_log`, `dt_bias`, `mimo_x/z/o`) создаются на LOCAL shape `(nheads_local,...)` / `(ngroups_local, mimo_rank, d_state)`
   - `in_proj = nn.Linear(d_model, d_in_proj_LOCAL)` --- **plain Linear**, не `ColumnParallelLinear`; `d_in_proj_local` считается из уже-шарденных `d_inner/ngroups/nheads`
   - `out_proj = nn.Linear(d_inner_LOCAL, d_model)` --- **plain Linear**, full output dim
   - Опционально TE `LayerNormLinear(tp_size=1, sequence_parallel=False)` как drop-in для in_proj (линии 509-550) --- TP скашивается через локальные dims, зато fusion LayerNorm+GEMM даёт один cuDNN kernel

2. `nanochat/nanochat/megatron_tp.py:1027-1133` --- внешняя обёртка:
   ```python
   class _AllReduceLinear(nn.Module):
       def __init__(self, linear: nn.Linear, tp_group=None):
           super().__init__()
           self.linear = linear
           self.tp_group = tp_group
       def forward(self, x):
           output = self.linear(x)
           torch.distributed.all_reduce(output, group=self.tp_group)
           return output
       def _load_from_state_dict(...):
           # manually shards full_weight columns [:, start:end] per TP rank
   ```
   Плюс `_replace_mamba_layers(mamba_module, tp_group)` --- post-construction step: `mamba_module.out_proj = _AllReduceLinear(mamba_module.out_proj, tp_group)`. Вызывается из `apply_megatron_tp` после создания модели.

3. `nanochat/nanochat/mamba2.py:1467-1479` --- `_tp_output_rms_norm` с ручным `all_reduce(local_ss, group=tp_group)` для глобальной RMS статистики, если сетка какой-то reduction внутри ssm block делает.

**Почему это проще чем TEColumnParallelLinear/TERowParallelLinear:**

Явный commentary из `megatron_tp.py:1036-1037`:
> "Unlike Megatron's RowParallelLinear, this does NOT further divide the weight (which would cause double-division since Mamba2Layer already divided it)."

Input в `in_proj`: уже replicated across TP ranks (upstream TP region guarantees). Веса `in_proj` на каждом rank РАЗНЫЕ (sharded columns --- state_dict loader вручную нарезает). Output: local-sized, zero коммуникации на forward. Внутри scan работает только на локальных heads. `out_proj`: plain Linear с local input, full output → каждый rank partial sum, финальный `all_reduce(output)` складывает. Эквивалент RowParallel без двойного деления.

**Важное отличие от nanochat:**

Nanochat `Mamba2Layer` использует свой custom `_run_mimo_scan` (pure torch einsum + `_ssd_scan_ref_mimo_shared`, линии 1238-1368). Это reference/fallback, медленнее TileLang. **Для cppmega TP-машинка ортогональна выбору kernel** --- взять nanochat sharding pattern и скормить локальные Q/K/V/ADT/DT/Trap в существующий `cppmega_tilelang_mimo_combined` (он shape-agnostic, подтверждено 2026-04-12).

**План реализации `cppmega/megatron/cppmega_mamba3_tp_mixer.py`:**

1. Скопировать TP-шардинг и per-head param init из `Mamba2Layer.__init__` (nanochat lines 396-647) --- но обрезать до того набора фич, который используется в нашем `AuthorMamba3Mixer` (MIMO + qknorm + bias_per_head + learned_bc_norm + trapezoidal + outproj_norm)
2. Plain `nn.Linear` для in_proj/out_proj с local dims
3. Forward: вызвать `cppmega_tilelang_mimo_combined(Q=C_local, K=B_local, V=x_local, ..., Q_bias=C_bias_local, K_bias=B_bias_local, MIMO_V=mimo_x_local, MIMO_Z=mimo_z_local, MIMO_Out=mimo_o_local, D=D_local, ...)` --- все local
4. Скопировать `_AllReduceLinear` класс в cppmega (новый файл `cppmega/megatron/tp_all_reduce_linear.py` или inline в mixer)
5. В `AuthorMamba3Mixer.__init__` если `tp_world_size > 1`: после конструкции Mamba3-like модуля сделать `self.out_proj = _AllReduceLinear(self.out_proj, pg_collection.tp.group)`
6. Убрать `NotImplementedError` для `tp_world_size != 1` и переименовать класс либо создать новый `CppmegaMamba3TPMixer` рядом
7. Unit test: TP=1 reference vs TP=2 numeric parity с детерминированным параметром init (seed на `tp_rank + layer_number`), tol `abs < 1e-3, rel < 1e-3` в bf16
8. Throughput test europe H200 TP=2 PP=1/2 VPP=1/2 и сравнение vs TP=1 baseline 112k tok/sec

## DSA indexer FP8 port (2026-04-12, Stream E)

**Что портировалось:** функция `fp8_index` из `deepseek-ai/DeepSeek-V3.2-Exp/inference/kernel.py` (lines 199-274). Это TileLang kernel (`@tilelang.jit`), не pure torch и не CUDA --- `fp8_gemm(k_fp8, q_fp8, logits) -> relu * q_scale -> reduce_sum(h) -> * k_scale -> write`. Shape `q=[b,m,h,d] k=[b,n,d] o=[b,m,n]`. Использует `act_quant` с `block_size=128` (rowwise absmax → `e4m3fn` + fp32 scale). На Megatron-стороне текущая реализация `_compute_index_scores` в `megatron/core/transformer/experimental_attention_variant/dsa.py` (lines 255-295) --- это BF16 `torch.einsum('sbhd,tbd->sbht') -> relu -> * weights -> sum(h) -> transpose`, тот же alg в fp32 upcast.

**Подход к порту:** не тащим TileLang зависимость в Megatron. Вместо TileLang kernel используем `torch._scaled_mm` (torch 2.12.0.dev20260410+cu132 на bench3, WGMMA lowering на sm_90a H200) с rowwise fp32 scales. Весь `d <= 128` в наших recipes, поэтому DeepSeek-овский `block_size=128` per-group scaling collapses в per-row scaling --- **эквивалентный numerics**. Path: `q bf16 [sq,b,h,d] -> rowwise quantize -> q_fp8 [sq*h,d] + q_scale [sq*h]`, то же для `k`, per-batch `_scaled_mm(a, b.T, scale_a, scale_b) -> fp32 [sq*h, sk]`, reshape, relu, weighted sum over heads, collect в `[b,sq,sk] fp32`.

**Артефакты:**
- `cppmega/megatron/dsa_fp8_indexer.py` --- `compute_index_scores_fp8`, `quantize_rowwise_fp8`, `compute_index_scores_bf16_reference` (локальный clone BF16 референса для unit-тестов).
- `cppmega/megatron/dsa_fp8_patch.py` --- `apply_dsa_fp8_patch()` monkey-patches `megatron.core.transformer.experimental_attention_variant.dsa._compute_index_scores` на FP8 вариант, `resolve_indexer_dtype(config)` читает `config.dsa_indexer_dtype` → env `CPPMEGA_DSA_INDEXER_DTYPE` → default `bf16`. Идемпотентно (sentinel `__cppmega_dsa_fp8_patched__`). `add_dsa_indexer_dtype_arg(parser)` регистрирует argparse флаг `--dsa-indexer-dtype {bf16,fp8}` в группе `experimental_attention_variant`.
- `cppmega/recipes/megatron_args.py` --- новый kwarg `dsa_indexer_dtype: str = "bf16"` в `build_megatron_args_bundle`, эмитит `--dsa-indexer-dtype <value>` когда `use_dsa=True`, валидирует `{bf16,fp8}`, при `fp8` добавляет в `custom_notes` напоминание про `apply_dsa_fp8_patch()`.
- `cppmega/recipes/nam56r_launch.py` --- propagate `dsa_indexer_dtype` через `build_nam56r_megatron_native_args` + CLI `--dsa-indexer-dtype`.
- `scripts/remote_smoke_h200_dsa_fp8_indexer.sh` --- полный NAM56R 4.73B AEMEAEMEAEMR depth=52 + MLA + MoE + MTP hybrid + DSA с FP8 indexer. Wrapper вокруг `pretrain_mamba.py` который ставит `CPPMEGA_DSA_INDEXER_DTYPE=fp8` и вызывает `apply_dsa_fp8_patch()` **до** exec upstream pretrain_mamba.py body. Наследует три DSA rope-fusion / spec routing / layernorm metainfo idempotent patches из BF16 lane.
- `tests/test_dsa_fp8_indexer.py` (9 тестов, все зелёные на H200 bench3): quantize roundtrip + absmax invariance, BF16 reference vs explicit triple loop (tolerance 2e-3), FP8 vs BF16 frac_within(0.5 abs / 0.2 rel) >= 0.9, **topk16 overlap >= 0.85** на shape `[sq=64,b=2,h=4,d=32,sk=128]`, `resolve_indexer_dtype` env/config precedence, argparse idempotency.
- `tests/test_megatron_args.py` (3 новых asserts) и `tests/test_remote_dsa_fp8_smoke_h200.py` (7 structural tests).

**Unit test результат (bench3 H200 2026-04-12):** `9 passed in 4.96s`. Frac-within на `atol=0.5, rtol=0.2` = 100%, topk16 overlap на синтетических gaussian inputs sq=64/sk=128 = 85-87%. На реалистичном sq=4096 topk16 overlap = **94.3-94.4%** (см. микро-bench ниже).

**Per-head fused rewrite (critical for memory):** после первого прохода микро-бенча (unfused `[sq*h, d] @ [sk, d]^T` в один gemm) стало ясно что основной выигрыш FP8 должен быть не в latency, а в **memory** --- Stream D документирует в `docs/nam56r_grid_search_2026_04_12.md:80-94` что BF16 `_compute_index_scores` аллоцирует `index_scores [sq, b, h, sk] fp32` живой через `sum(dim=2)`, что даёт ~3.2 GB live per indexer call на sq=4096 b=2 h=8, и именно это вызывает OOM на D1/D2 configs 9+4 DSA layout (134-135 GB PyTorch на stage 0). Unfused FP8 вариант имел ту же проблему потому что материализовал `logits [sq, h, sk] fp32`. Переписан на **per-head fused accumulation**: внешний loop по `bi`, внутренний по `hi`, каждая пара даёт `[sq, sk] fp32` блок который сразу relu'ется, умножается на per-token weight `[sq, 1]` и добавляется in-place в `[b, sq, sk]` аккумулятор. Никогда не аллоцируется `[sq, h, sk]`.

**Throughput + memory микро-bench (bench3 H200, 2026-04-12, synthetic):**

|   sq |    b |    h |    d | BF16 peak_delta | FP8 peak_delta |   memory ratio | BF16 ms | FP8 ms | latency ratio | topk16 overlap |
| ---: | ---: | ---: | ---: | --------------: | -------------: | -------------: | ------: | -----: | ------------: | -------------: |
| 4096 |    2 |    8 |   64 |       3254.8 MB |   **340.6 MB** |  **9.6x less** |    3.22 |   3.04 |         1.06x |          0.944 |
| 4096 |    4 |    8 |   64 |       6442.5 MB |   **479.8 MB** | **13.4x less** |    6.39 |   6.03 |         1.06x |          0.943 |
| 4096 |    2 |    8 |  128 |       3221.2 MB |   **345.3 MB** |  **9.3x less** |    3.91 |   3.08 |         1.27x |          0.944 |

**Интерпретация:**
- **Memory win** (9-13x меньше peak delta per indexer call) --- это прямое решение Stream D OOM-а. На full NAM56R DSA 9+4 layout: 9 DSA layers × (3254 - 340) MB saved = **26 GB saved on pipeline stage 0** вcarry forward'е, и ещё столько же в backward recompute (приблизительно). Это должно дать достаточно headroom'а чтобы D1 config (PP=2 VPP=2 MBS=4 GBS=64 MTP=2 CG BF16) сел внутрь 140 GB H200 без активационной рекомпьюты.
- **Latency win** скромный (1.06-1.27x) --- per-head loop serialization съедает часть FP8 MM преимущества. На fp8 vs bf16 gemm уровне ожидаемый win 2x на H200 (WGMMA fp8 peaks ~2x bf16), но h loop в python накладывает overhead. Будущая оптимизация: batched fp8 gemm через один call `_scaled_mm([sq*h, d], [sk, d].T, ...)` с **output-staging** --- записывать relu'нутые fp8 блоки сразу в `[b, sq, sk]` через `torch.scatter_add` или custom CUDA kernel.
- Topk overlap 94.3-94.4% на production shape --- DSA selection contract сохранён.

**Loss behaviour:** полный end-to-end throughput run (`scripts/remote_smoke_h200_dsa_fp8_indexer.sh`) **не запускался** в рамках Stream E потому что Stream D ещё не опубликовал BF16 baseline на 2026-04-12 --- без baseline сравнение FP8 vs BF16 delta будет несопоставимо. Скрипт готов и **будет запущен после** выхода Stream D'шного baseline числа (поллинг `/private/tmp/.../ae563d02ba8e316f9.output`, пока видно только debug-сообщения про NCCL init, iter 1 ещё не достигнут).

**Blockers / риски:**
1. **Backward path.** `FusedDSAIndexerLoss.bwd_fused_indexer_loss_naive` вызывает `_compute_index_scores` один раз для recompute, и напрямую `torch.einsum('sbhd,tbd->sbht', q.float(), k.float())` ещё раз (line 482) для получения `scores` без relu для `relu_mask`. Наш monkey-patch покрывает первый вызов, но **второй einsum в backward остаётся BF16**. Это не ломает корректность, но означает что FP8 speedup видно только на forward (примерно половина от теоретического). Будущая оптимизация: portировать вторую einsum тоже, либо фьюзнуть recompute с первым вызовом (pass `return_scores=True` через monkey-patch).
2. **FP8 scales на каждый step live recomputed** --- у нас нет global scaling tracker как в TransformerEngine DelayedScaling. Per-call absmax → divide accurate но noisy; в реалистичном training loop это может давать разброс loss на 2-3% по шагам. DS reference хранит scales в kvcache и reuses between decoding steps --- для training mode (где cache invalidated every step) такой кеш не применим.
3. **Megatron `transformer_config.py` не содержит `dsa_indexer_dtype` field** на bench3. Наш launcher эмитит `--dsa-indexer-dtype fp8` но Megatron argparse имеет `ignore_unknown_args=True` → silently ignored. Поэтому мы полагаемся на env var `CPPMEGA_DSA_INDEXER_DTYPE` + `apply_dsa_fp8_patch()` из wrapper `pretrain_mamba.py`. Добавить real config field = отдельный idempotent patch в remote script (сейчас deferred --- env var route работает без upstream Megatron mods).
4. **DSA full NAM56R smoke ещё не прошёл end-to-end** (зависит от Stream D). Без baseline числа невозможно сказать, сходится ли FP8 loss одинаково с BF16 через первые 20 iter. Это основной gap в task completion.

### Stream G backward FP8 cleanup (2026-04-12, task #84)

**Что портировалось:** `bwd_fused_indexer_loss_naive` из `megatron/core/transformer/experimental_attention_variant/dsa.py` (bench3 ref, lines ~346-500). Upstream BF16 backward делает три heavy GEMM-семейства:
1. `_compute_index_scores(q, weights, k)` recompute `[B, Sq, Sk]` --- уже FP8 через Stream E patch.
2. `torch.bmm(query.float(), key.float()) → attention_scores [b*np, sq, sk] fp32` (main-attention Q@K^T для KL target).
3. `torch.einsum('sbhd,tbd->sbht', q.float(), k.float()) → scores [sq, b, h, sk] fp32` для relu_mask + grad_scores; следом `grad_q = einsum('sbht,tbd->sbhd', ...)` и `grad_k = einsum('sbht,sbhd->tbd', ...)`.

**FP8 перенос (что портировано / что осталось BF16):**

| Тензор                                                              | Upstream dtype        | Stream G dtype                         | Причина решения                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------- | --------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_compute_index_scores` recompute `[b, sq, sk]`                     | FP32                  | **FP8**                                | Stream E patch уже активен через `_compute_index_scores` monkey-patch.                                                                                                                                                                                                                                      |
| `scores [sq, b, h, sk]` (raw q@k^T для relu_mask)                   | FP32 (2.15 GB @ prod) | **FP8 + per-(b,h) fused**              | Никогда не материализуется; логits `[sq, sk]` живут только в working buffer внутри `hi` loop, сразу потребляются для relu_mask + grad_weights + grad_scores.                                                                                                                                                |
| `scores_after_relu [sq, b, h, sk]`                                  | FP32 (2.15 GB)        | **Устранено**                          | Результат `torch.relu(logits)` внутри рабочего буфера `[sq, sk]`, моментально schannel'ится в `grad_weights_f32[:, bi, hi]` через `sum(dim=-1)`.                                                                                                                                                            |
| `grad_scores_after_relu [sq, b, h, sk]`                             | FP32 (2.15 GB)        | **Устранено**                          | Grad_scores живут в `[sq, sk]` per-head буфере.                                                                                                                                                                                                                                                             |
| `grad_scores [sq, b, h, sk]`                                        | FP32 (2.15 GB)        | **Устранено**                          | То же.                                                                                                                                                                                                                                                                                                      |
| `grad_q [sq, b, h, d]`                                              | FP32 (33 MB)          | FP32 accumulate + final `.to(q.dtype)` | Target grad tensor, писать in-place через `grad_q_f32[:, bi, hi, :].add_(dq)` от FP8 GEMM `_scaled_mm(grad_scores, k.t())`.                                                                                                                                                                                 |
| `grad_k [sk, b, d]`                                                 | FP32 (4 MB)           | FP32 accumulate + final `.to(k.dtype)` | То же через `_scaled_mm(grad_scores.t(), q.t())`.                                                                                                                                                                                                                                                           |
| `grad_weights [sq, b, h]`                                           | FP32 (tiny)           | FP32 reduction                         | Остаётся FP32 pointwise `(g_il * scores_after_relu).sum(dim=-1)`.                                                                                                                                                                                                                                           |
| Main-attention `torch.bmm(Q, K)` `[b, np, sq, sk]` (4.29 GB @ prod) | FP32                  | **BF16/FP32 оставлен как upstream**    | Output structurally feeds non-linear `softmax → sum(dim=1) → L1 normalize` по `np`; softmax non-linear across `sk`, должен произойти до суммы голов, значит full `[b, np, sq, sk]` обязан быть live. Per-head streaming НЕ применим. Документировано в design note в `cppmega/megatron/dsa_fp8_indexer.py`. |
| `attention_scores_softmax [b, np, sq, sk]`                          | FP32 (4.29 GB)        | FP32 оставлен                          | Требуется для KL target; не затронут.                                                                                                                                                                                                                                                                       |
| `grad_index_scores_logits [b, sq, sk]`                              | FP32 (256 MB)         | FP32 оставлен                          | Малый размер; intermediate между softmax-bwd и per-head fused loop.                                                                                                                                                                                                                                         |

**Артефакты:**
- `cppmega/megatron/dsa_fp8_indexer.py` --- новые функции `bwd_fused_indexer_loss_fp8` (+ helpers `_attention_target_fp32`, `_index_scores_softmax_fp32`), `bwd_fused_indexer_loss_bf16_reference` (byte-for-byte clone upstream для unit-тестов без Megatron checkout), расширен `__all__`.
- `cppmega/megatron/dsa_fp8_patch.py` --- `apply_dsa_fp8_patch()` теперь патчит и `_compute_index_scores` (Stream E), и `bwd_fused_indexer_loss_naive` (Stream G). Отдельный sentinel `__cppmega_dsa_fp8_bwd_patched__` чтобы фьюзед patches оставались идемпотентными independently. Verify: `FusedDSAIndexerLoss.backward` staticmethod делает plain `bwd_fused_indexer_loss_naive(...)` lookup через module globals, поэтому module-level rebind достаточен (закоментировано в коде).
- `tests/test_dsa_fp8_indexer.py` --- 2 новых теста: `test_backward_parity_bf16_vs_fp8` (shape `[sq=sk=128, b=2, h=4, d=32, np=4, hn=32, topk=16]`, `sparse_loss=True`, tolerance contract `frac_within(abs<=0.5 OR rel<=0.2) >= 0.9` per grad_q/grad_weights/grad_k, плюс sanity check forward indexer parity) и `test_backward_fp8_no_sparse_loss_parity` (shape sq=sk=64, `sparse_loss=False`, same tolerance).
- `scripts/measure_dsa_fp8_bwd_memory.py` --- standalone CUDA benchmark, shape hardcoded на NAM56R production (`batch=4, seqlen=4096, h=8, d=64, np=16, hn=128, topk=2048`), измеряет `torch.cuda.max_memory_allocated` для full-path BF16 vs full-path FP8, плюс isolation-pass с `np=1, hn=16` для изоляции именно indexer-side savings.

**Unit test результат (bench3 H200, 2026-04-12):** `11 passed in 23.30s`. `test_backward_parity_bf16_vs_fp8`: frac_within на grad_q, grad_weights, grad_k все >= 0.9 (tolerance `atol=0.5, rtol=0.2`). Forward index_scores sanity check passes. `test_backward_fp8_no_sparse_loss_parity`: идентичный контракт, проходит.

**Memory delta (bench3 H200 sm_90, `scripts/measure_dsa_fp8_bwd_memory.py`, GPU 4 isolated):**

| Variant                              | Shape                                  | BF16 peak_delta | FP8 peak_delta |                   Savings |
| ------------------------------------ | -------------------------------------- | --------------: | -------------: | ------------------------: |
| Full path (main-attention + indexer) | `b=4 sq=sk=4096 h=8 d=64 np=16 hn=128` |       9202.3 MB |      9138.3 MB |                      0.7% |
| Indexer-only isolation               | `b=4 sq=sk=4096 h=8 d=64 np=1 hn=16`   |       7366.0 MB |      2245.5 MB | **69.5% (5.12 GB saved)** |

**Trade-off / интерпретация (критично):** full-path peak memory НЕ меняется заметно, потому что main-attention `[b, np, sq, sk] fp32 bmm output = 4.29 GB + softmax fresh output 4.29 GB = 8.58 GB` доминирует peak в *обоих* путях. FP8 backward элиминирует the indexer-side peak (`[sq, b, h, sk] fp32 × 3` = ~6.5 GB peak) через per-(b, h) fused loop, но indexer peak и main-attention peak НЕ overlap во времени, так что `max(main_attn, indexer)` остаётся на main_attn. В isolation run (где main attention тривиальный) видно fp8 savings напрямую: 7366 → 2246 MB, **5.12 GB delta, ratio 3.3x**.

**Для Stream D v2 это значит:**
1. **Per-call peak на single DSA layer на stage 0 остаётся 9-10 GB** --- main-attention bmm floor не сдвигается. Stream E's 26 GB forward savings + этот 0.7% backward win = practically та же общая heap picture как после Stream E alone.
2. **PyTorch caching allocator reuse** между DSA-слоями всё же улучшается: после indexer работы FP8 путь освобождает ~5 GB блоков раньше, чем BF16 путь, что даёт allocator шанс переиспользовать их на следующем DSA слое или на mamba/MoE слое после него. Эффект не виден в single-call peak; будет виден только на полном PP-staged throughput run.
3. **Net ожидаемое улучшение Stream D v2 headroom: skeptical 1-3 GB максимум** на stage 0, не 3-5 GB как надеялись в задаче. Main-attention bmm --- это реальный bottleneck backward memory; его FP8'ить без inline softmax rewrite невозможно.

**Blockers / follow-ups:**
1. **End-to-end NAM56R smoke с FP8 backward не запущен** --- тот же gap как у Stream E (Stream D BF16 baseline ещё не опубликован). Готово: patch идемпотентный, monkey-patches BOTH forward и backward через один `apply_dsa_fp8_patch()` call из `scripts/remote_smoke_h200_dsa_fp8_indexer.sh` wrapper --- никаких изменений в launcher не требуется.
2. **Loss convergence через backward FP8 не валидирован** на реальных NAM56R sample'ах. Unit-test parity (`frac_within >= 0.9`) достаточен для "корректно реализовано", но не даёт ответа на "сойдётся ли loss за 10k iter". Предполагаемый риск 2-3% per-step loss jitter (как Stream E forward).
3. **Main-attention FP8 путь остался неинверстированным**. Если Stream D v2 упрётся в memory именно на backward peak, единственный реальный рычаг --- переписать `_attention_target_fp32` с streaming per-head softmax → L1 normalize (non-trivial, non-linear across sk). Cosmetic rewrite сейчас отложен; текущий backward FP8 не уменьшит full peak, но и не ломает никаких контрактов.
4. **Дополнительные FP8 GEMM'ы на dq / dk path** требуют rowwise re-quantization `k` и `q` в `[d, sk]` / `[d, sq]` layout внутри per-(b, h) loop --- мы quantize один раз за batch (`kb_for_dq_fp8` hoisted outside `hi` loop), но `q_for_dk_fp8` re-quantise каждый head. Дёшево (`d * sq` элементов, few MB), но можно вынести наружу позже. Не влияет на correctness.

**Следующий action:** Stream D v2 скорее всего получит "measurable but modest" speedup от backward FP8 (сам FP8 GEMM в 2x быстрее BF16 einsum на WGMMA, но per-(b,h) loop serialization съедает часть). Forward FP8 savings (26 GB) остаются основным wins; backward FP8 в большей степени cosmetic для memory peak и minor-to-moderate для latency. Рекомендация: включать backward FP8 одновременно с forward FP8 через единственный `CPPMEGA_DSA_INDEXER_DTYPE=fp8` env var (уже работает через обновлённый `apply_dsa_fp8_patch()`), не тратить отдельное время на main-attention rewrite пока Stream D v2 не покажет что главный блокер именно в backward peak.

## CppmegaMamba3TPMixer реализация (2026-04-12)

**Цель:** TP-aware Mamba3 MIMO-mixer с full 7/7 features, чтобы снять
ограничение `tensor-model-parallel-size=1` `AuthorMamba3Mixer` и дать
возможность горизонтального шардинга Mamba3 слоя через TE
ColumnParallel/RowParallel linear-ы.

### Что сделано

1. `cppmega/megatron/cppmega_mamba3_tp_mixer.py` — новый mixer (589 LOC)
   по Megatron-native `MambaMixer` паттерну. Ключевые отличия от
   `AuthorMamba3Mixer`:
   - `in_proj` строится через `build_module(submodules.in_proj, ...)` →
     `TELayerNormColumnParallelLinear` c `partition_sizes=[z, x, B, C, dd_dt,
     dd_A, trap]` (7 компонент, LOCAL размеры per-rank).
   - `out_proj` через `TERowParallelLinear` (all-reduce по TP).
   - **Angle-projection вынесен в отдельный REPLICATED `nn.Linear
     angle_proj`** — потому что mamba3 броадкастит `angles` на все nheads,
     и если шардить по TP каждый rank увидит разные angles — parity сломан.
     `angle_proj.weight` помечен `tensor_model_parallel=False`, инициализируется
     из детерминированного CPU generator с seed=`7919 + layer_number`.
   - **Linear-axis layout B/C изменен с `(r, g, n)` на `(g, r, n)`** чтобы
     TP-шардинг резал по оси групп чисто, а не перемешивал slices разных `r`
     рангов (это был скрытый баг в `mamba3_te_mixer.py`, который никогда не
     запускался в TP>1 и поэтому не ловил этот bug).
   - Per-head параметры (`dt_bias`, `B_bias`, `C_bias`, `mimo_x/z/o`, `D`)
     инициализируются на FULL nheads через `torch.Generator(device="cpu")
     .manual_seed(1337 + 100003 * layer_number)`, затем срез
     `[tp_rank*nh_loc:(tp_rank+1)*nh_loc]` кладётся как `nn.Parameter`. Гарантирует
     bit-exact parity с TP=1 reference после concat по head-оси.
   - Поддержка `--sequence-parallel`: `angle_proj(hidden_states)` даёт
     `(L/tp, B, num_rope_angles)`, после чего делается
     `gather_from_sequence_parallel_region` чтобы получить full-L для
     броадкаста на `nheads_local_tp`.

2. `cppmega/megatron/nam56r_full_spec.py` — добавлен env-var toggle
   `CPPMEGA_MAMBA3_TP_MIXER=1` (функция `_select_mamba3_mixer_cls()`)
   который подменяет `AuthorMamba3Mixer` на `CppmegaMamba3TPMixer` внутри
   `CppMegaSelectiveMambaMixer`. Также `build_cppmega_nam56r_full_stack_spec`
   теперь выбирает `TENorm` вместо `WrappedTorchNorm` когда
   `tensor_model_parallel_size > 1`, потому что `WrappedTorchNorm` не
   поддерживает sequence-parallel.

3. `cppmega/recipes/nam56r_nemo_recipe.py` — новое поле
   `use_tp_mamba3_mixer: bool` в рецепт, которое эмитит
   `CPPMEGA_MAMBA3_TP_MIXER=1` через `to_env_dict()`.

4. `tests/test_cppmega_mamba3_tp_mixer.py` — 10 структурных + 1 runtime
   parity test. Runtime test спавнит TP=1 и TP=2 миры через
   `torch.multiprocessing.spawn`, детерминированно перезаписывает
   веса in_proj/out_proj/angle_proj из общего CPU generator, прогоняет
   один forward на фиксированном inpute, сравнивает выходы на rank 0.

5. `scripts/remote_train_h200_nam56r_tp2.sh` — европейский launcher для
   100-iter throughput прогона с `TP=2 PP=1 VPP=1 MBS=4 GBS=64 MTP=2`,
   bench3-style shim + parametrized knobs. Использует system cuDNN
   (`/usr/local/cuda-13.2/lib64`) а не venv cuDNN (venv версия ломается в
   TE fused_attn для MLA: `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`).

### Parity test — PASS

Запуск на LOCATION_2 (2×H200, `CUDA_VISIBLE_DEVICES=0,1`):

```
tests/test_cppmega_mamba3_tp_mixer.py::test_runtime_parity_tp1_vs_tp2_forward PASSED
======================== 11 passed, 1 warning in 22.81s ========================
```

Максимальная абсолютная разница между TP=1 и TP=2 forward: **1.5625e-2** в
bf16 (atol=2e-2, rtol=5e-2). Это ожидаемое порядок bf16 floating-point
шума от `TERowParallelLinear` all-reduce (order of summation отличается
от full matmul).

Ключевой момент parity-отладки: первые попытки давали max_abs~1.1 (десятки
percent разошлось). Корень проблемы — тот самый `(r, g, n)` ordering B/C:
при слайсе linear axis ранг 0 получал "первую половину r для всех g"
вместо "первой половины g для всех r". Фикс layout на `(g, r, n)`
привёл max_abs к 1.5e-2.

### TP=2 throughput на europe H200×8

Конфигурация: `--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1
--sequence-parallel --mamba-num-groups 2 --mbs 4 --gbs 64 --mtp 2
--cuda-graph per_module`, real data
(`clang_semantic_4k_v10_train` + HuggingFaceTokenizer).

**Результаты (iters 50–100, 100-iter run, `nam56r_tp2_100iter_*.log`):**

| Метрика                                  | Значение        |
| ---------------------------------------- | --------------- |
| iter-ms (steady)                         | ~7136 ms        |
| TFLOP/s/GPU                              | 112.9           |
| tok/sec (tokens/iter / iter-s)           | ~36,727 tok/sec |
| MFU per GPU (H200 BF16 989 TFLOP/s peak) | ~11.4%          |
| Validation loss @ iter 100               | 2.41 (PPL 11.2) |

**Без CUDA graphs (CUDA_GRAPH_MODE=off, 30-iter run):** `7305 ms/iter` →
**~35,900 tok/sec**. Разница с CUDA graphs per-module минимальна (~2%).

### Дельта vs TP=1 baseline (112k tok/sec)

**TP=2 медленнее в ~3 раза (−67%):** 36.7k tok/sec vs 112k tok/sec baseline.

**Честная оценка причины:**

1. **Backward kernel G-branch** — `mamba_mimo_bwd_combined` в upstream
   принимает только `G == 1` или `G == H`. С `mamba_num_groups=2` и
   nheads=112 → ранее `G=2, H=112`, без поддержки; пришлось ставить
   `ngroups=2` чтобы `G_local = 1` после шардинга (т.е. SISO-mode на Mamba3
   при TP=2), что отбрасывает gain от SSM group-attention.

2. **Sequence-parallel all-gather на angle_proj** — каждый mamba layer теперь
   делает extra `all_gather along dim 0` для angles (`L/tp × B × num_rope_angles`
   bytes per layer). На 32 mamba-layer NAM56R это добавляет заметную
   communication overhead.

3. **TERowParallelLinear all-reduce на out_proj каждого mamba layer** —
   стандартный TP-overhead но на 32 слоя это много. На H200 NVLink это
   стоит ~200-400 μs per call, что на 32 слоях даёт 6-12 ms per iter сверху.

4. **MLA attention тоже делает TP=2 allreduce** — так работало и до TP=2
   на mamba но тогда только на 8 A-layers, а теперь везде добавляется
   reduce. Плюс DSA/MLA имеет свои TE allreduce-и.

5. **MoE + sequence-parallel + TP=2** — `AllToAll` dispatch теперь работает
   на sharded `L/tp` sequence а потом expert gemm на `L/ep`, результат
   all-reduce-scatter обратно. По сравнению с TP=1 DP=8 (где каждый GPU
   имеет свою полную копию всех activations), TP=2 DP=4 делит вычисление
   но добавляет коммуникацию.

**Вердикт:** TP=2 при текущей структуре NAM56R НЕ выигрышна по throughput.
Baseline TP=1 PP=2 VPP=2 с PP + MTP overlap даёт 3× больше tok/sec, чем TP=2
PP=1. Это ожидаемо для H200 на одном ноде: TP-шардинг работает только
когда compute dominate communication; в NAM56R compute на mamba layer
(TileLang MIMO SSD scan) UM-bound, и дополнительная allreduce-latency
поверх него просто складывается в wall-clock.

### Что НЕ в этой задаче (отложено)

- **ngram-hash embedding + sequence-parallel breakage.** При `--sequence-parallel`
  стандартный parallel embedding возвращает `[L/tp, B, H]`, а cppmega
  `custom_embedding.py:117` суммирует с `ngram_embeddings` размера `[L, B, H]`
  — shape mismatch. В нашем TP=2 launcher-е установлено
  `CPPMEGA_NGRAM_HASH_ENABLED=0` и `CPPMEGA_STRUCTURE_ENABLED=0` чтобы
  обойти это. Это pre-existing bug в custom_embedding который надо
  фиксить отдельно (дополнить gather/scatter для SP границы), не связан
  с CppmegaMamba3TPMixer. Записать в отдельную задачу.

- **MIMO backward kernel расширение на general G**. Upstream
  `mamba_mimo_bwd_combined` поддерживает только G=1 и G=H. Для полноценной
  TP>1 с `mamba_num_groups=8, nheads=112` (native NAM56R config) нужно
  чтобы был вариант kernel для general `G | H`. Это upstream issue в
  Goombalab state-spaces/mamba.

- **BP-aware parity**. Наш runtime parity тест покрывает только forward.
  Full backward parity требует идентичных optimizer steps на обоих TP
  мирах, что выходит за рамки unit-теста. Можно добавить в CI как
  отдельный long-running тест.

### Stream B v2 re-measurement (2026-04-12): claim restored, RCA corrected

Stream B v1 была права в главном verdict-е и неправа в объяснении.
Фраза «TP=2 тут 3× медленнее из-за `mamba_num_groups=2` / kernel
constraint `G==1 || G==H`» больше не должна использоваться как root cause.
После прогона TP-лаунчера с `mamba_num_groups=8` throughput **не
восстановился**: текущий single-node H200 TP=2 lane всё равно остаётся
жёстким net loss для NAM56R Mamba-3 MIMO.

#### Pre-check

`pytest tests/test_cppmega_mamba3_tp_mixer.py -xvs`:
**11 passed, 1 warning in 22.61s**, включая
`test_runtime_parity_tp1_vs_tp2_forward`. То есть TP infrastructure /
forward-path не сломаны; вопрос здесь в wall-clock, а не в банальной
инфраструктурной поломке.

#### Launch result

`RUN_ID=nam56r_tp2_v2 TP_SIZE=2 PP_SIZE=1 MBS=4 GBS=64 MTP_DEPTHS=2
CUDA_GRAPH_MODE=per_module FP8_MODE=off TRAIN_ITERS=100` на
europe H200×8, log:
`/home/dave/cppmega-root/cppmega/cppmega_nam56r_tp2_v2.log`.

Конфиг (эффективный после patch):
- `tensor_model_parallel_size=2, pipeline_model_parallel_size=1`
- `sequence_parallel=True, context_parallel_size=1`
- `mamba_num_groups=8` (дефолт Megatron, без `--mamba-num-groups`)
- `num_heads=112 → nheads_local=56, ngroups_local=4, heads/group=14`
- `mbs=4, gbs=64, seq=4096, mtp=2` (same as v1)
- DSA: **не включалась** (TP=1 baseline тоже без DSA → apples-to-apples)
- cuda_graph_scope=attn mamba moe_router moe_preprocess (per_module)

Запуск завершился штатно **100/100 iterations**, checkpoint сохранён.
Loss @ iter 1 = 11.78, iter 15 = 4.44, iter 50 = 2.65, iter 100 = 2.28.

#### Steady-state throughput (iters 50–100 window, n=51)

| Метрика                          | Stream B v2 (мера)              |
| -------------------------------- | ------------------------------- |
| iter-ms avg / median / stdev     | **7560.74 / 7560.60 / 2.24 ms** |
| TFLOP/s/GPU                      | **109.0** (steady)              |
| tokens per iter (gbs × seq)      | 64 × 4096 = 262144              |
| tok/sec (iters 50–100)           | **34,671.7 tok/sec** (34.67k)   |
| MFU per GPU (H200 BF16 peak 989) | ~11.0%                          |
| train loss @ iter 100            | 2.284                           |
| train loss @ iter 50             | 2.648                           |

Одна аномалия — **iter 4 spike**: 13,328 ms vs steady ~7560 ms,
вероятно TE CUDA graph перекомпиляция. Изолированный, не повторяется.
Iters 5–100 идеально стабильны (stdev 2.24 ms в окне 50–100).

#### Дельта vs Stream B v1 (ngroups=2, тот же iter window 50–100)

Apples-to-apples сравнение (v1 лог `nam56r_tp2_100iter_023921.log`,
iters 50–100, n=51):

| метрика         | v1 (ngroups=2) | v2 (ngroups=8 + GQA patch) | delta         |
| --------------- | -------------- | -------------------------- | ------------- |
| iter-ms avg     | 7136.91        | 7560.74                    | +5.9%         |
| iter-ms stdev   | 1.51           | 2.24                       | —             |
| TFLOP/s/GPU     | 112.9          | 109.0                      | −3.5%         |
| tok/sec         | 36,730.7       | 34,671.7                   | **−5.6%**     |
| loss @ iter 100 | 2.438          | 2.284                      | −6.3% (лучше) |

**v2 на 5.6% МЕДЛЕННЕЕ чем v1**, НЕ быстрее как предсказывала гипотеза
задачи. Loss чуть лучше (ниже), но это не перевешивает throughput
регресс.

#### Дельта vs TP=1 baseline (112,152 tok/sec, TP=1 PP=2 VPP=2, без DSA)

**v2:** 34,671 / 112,152 → **−69.1% throughput (3.24× slower)**.
**v1:** 36,730 / 112,152 → −67.3% (3.05× slower).

Оба случая теряют ~2/3 throughput. Следовательно, claim
**«TP=2 is net loss for NAM56R Mamba-3 MIMO on single-node H200»**
нужно восстановить, но с **другим** root cause analysis.

#### Honest verdict

Гипотеза «главная причина была в `ngroups=2` degradation, а
`ngroups=8` должен был вернуть throughput» **фальсифицирована данными**:

1. `ngroups=8` GQA-path на TP=2 со всеми правильно настроенными
   шардами всё равно выдаёт **~34.7k tok/sec vs 112k TP=1 baseline** —
   тот же порядок потери, что v1 репортила.
2. `ngroups=8` даже чуть медленнее `ngroups=2` (~5.5%), видимо из-за
   небольшого дополнительного generic-GQA reduce; это может объяснить
   **вторичный** 5%-уровневый delta v2 vs v1, но не 3.2× gap к baseline.
3. **Первичный bottleneck — collectives в текущем TP/SP lane.**
   Для каждого Mamba-слоя платится `TERowParallelLinear` / out-proj
   all-reduce; `angle_proj` под sequence parallel требует full-sequence
   gather перед SSM scan; поверх этого hybrid stack добавляет MLA TP
   reductions и MoE router/dispatch collectives. Эти коммуникации
   выходят прямо в wall-clock.
4. **Второй первичный фактор — compute profile самого NAM56R Mamba-3
   MIMO на H200.** Local compute under TP=2 shrink'ается недостаточно:
   scan остаётся bandwidth/latency-bound, поэтому TP экономит память
   сильнее, чем время. Это и видно по steady-state `~109 TFLOP/s/GPU`
   и `~11% MFU` даже после возврата к `ngroups=8`.
5. **Сравнение topology-specific, не универсальное.** Текущий TP lane —
   это `TP=2, PP=1, VPP=1, SP=on`; лучший baseline — `TP=1, PP=2,
   VPP=2`. Поэтому честная формулировка такая: **на текущем
   single-node H200 stack/launcher TP=2 — net loss**. Это не доказательство,
   что TP вообще никогда не выиграет при другой PP/VPP разбивке.

**CppmegaMamba3TPMixer как производственный путь для NAM56R НЕ
работает.** Тот факт что mixer корректен (11/11 unit tests pass, forward
parity) и kernel теперь поддерживает general GQA **не меняет wall-clock
верикт**: на H200×8 с TP=2 PP=1 NAM56R теряет 3×+ throughput
относительно лучшего TP=1 PP=2 VPP=2 baseline из-за
**communication overhead + weak compute shrink**, а не из-за выдуманного
`ngroups` kernel constraint. В текущем виде TP=2 имеет смысл только как
memory scale-up tool (если модель не влезает), а не как throughput
optimization. Если хотим честно переоценить TP, следующий правильный
эксперимент — держать ту же feature surface и мерить TP=2 на PP=2/VPP=2,
а не только на нынешнем `PP=1/VPP=1` lane.

#### Real blockers / caveats that actually matter

1. **Это не apples-to-apples launcher.**
   `scripts/remote_train_h200_nam56r_tp2.sh` по умолчанию идёт как
   `TP=2, PP=1, VPP=1`, с `--sequence-parallel`, и ещё отключает
   `ngram/structure` из-за отдельного SP bug в custom embeddings.
   Поэтому report должен говорить про **текущий TP lane**, а не про
   абстрактный "TP в вакууме".
2. **Возможен небольшой вторичный cost generic GQA reduce.**
   Если хотим его убрать, надо optimizе-ить/fuse-ить reduce в backward
   kernel. Но это уже борьба за проценты поверх 3× gap, а не root cause.
3. **Iter 4 spike** (13.3s vs 7.5s steady) похож на единичную
   TE/CUDA-graph recompile warmup anomaly. После iter 5 не повторяется
   и verdict о steady-state не меняет.
4. **Math-correctness и throughput здесь нужно разделять.**
   Ниже в этом документе отдельно зафиксирован SP-on backward bug по
   `angle_proj.weight`. Он блокирует loss/convergence comparisons, но не
   меняет сам wall-clock verdict: even при корректном `ngroups` текущий
   TP lane остаётся throughput net loss.

### Stream I SP-on parity test (2026-04-12)

**Что добавлено.** Два новых тест-кейса в `tests/test_cppmega_mamba3_tp_mixer.py`
для закрытия дыры в юнит-покрытии `CppmegaMamba3TPMixer` под
`sequence_parallel=True`:

1. **`test_tp2_sp_on_parity_vs_tp1`** — полный forward+backward parity тест
   между TP=1/SP=False reference и TP=2/SP=True test world. Мелкая
   конфигурация (`d_model=256, nheads=8, ngroups=4, d_state=32,
   headdim=32, mimo_rank=2, chunk_size=32, L=128, B=2`), ~30 сек на
   H200. Сравнивает: forward output (gather along seq), grads
   `in_proj.weight` (per-component gather по `[z,x,B,C,dd_dt,dd_A,trap]`),
   `out_proj.weight` (gather по dim 1), per-head `dt_bias/D/B_bias/
   C_bias/mimo_x/mimo_z/mimo_o` (dim 0 gather), `angle_proj.weight` +
   LayerNorm weights (all-reduce sum как для replicated параметров).

2. **`test_tp2_sp_on_angle_proj_gather`** — изолированный тест только
   Stream B-шного добавления: `angle_proj(local_hs) →
   gather_from_sequence_parallel_region → full (L, B, num_rope_angles)`
   на каждом rank. Проверяет shape и numeric parity до MIMO scan.

**Результат прогона на bench3 (H200, 2026-04-12):**

| Тест                                                                                   | Результат | max_abs / max_rel                   |
| -------------------------------------------------------------------------------------- | --------- | ----------------------------------- |
| 10 структурных (source-introspection)                                                  | PASS      | --                                  |
| `test_runtime_parity_tp1_vs_tp2_forward` (SP=False, существующий)                      | PASS      | регрессий нет                       |
| `test_tp2_sp_on_angle_proj_gather` (new)                                               | **PASS**  | `1e-3 / 1e-3` чистый                |
| `test_tp2_sp_on_parity_vs_tp1` forward (new)                                           | **PASS**  | `3.9e-3 / 6.06` (под `5e-2 / 8e-2`) |
| `test_tp2_sp_on_parity_vs_tp1` backward `in_proj`                                      | PASS      | под `8e-2 / 1e-1`                   |
| `test_tp2_sp_on_parity_vs_tp1` backward `out_proj`                                     | PASS      | под `8e-2 / 1e-1`                   |
| `test_tp2_sp_on_parity_vs_tp1` backward `dt_bias/D/B_bias/C_bias/mimo_x/mimo_z/mimo_o` | PASS      | под `8e-2 / 1e-1`                   |
| `test_tp2_sp_on_parity_vs_tp1` backward `B_norm/C_norm/LayerNorm`                      | PASS      | под `8e-2 / 1e-1`                   |
| `test_tp2_sp_on_parity_vs_tp1` backward **`angle_proj.weight`**                        | **FAIL**  | **`3.76 / 447.8`**                  |

Итого: **12 passed / 1 failed** на `1 failed, 12 passed, 3 warnings in 76.34s`.

**Root cause бага (подтверждено чтением megatron-core).**

Файл: `cppmega/megatron/cppmega_mamba3_tp_mixer.py:443`.

```python
angles_raw = gather_from_sequence_parallel_region(
    angles_raw, tensor_parallel_output_grad=False, group=self.tp_group,
)
```

Флаг `tensor_parallel_output_grad=False` неправильный для того, как
`angles_raw` используется дальше. В forward after-gather:

```python
angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1)
angles = angle_dt(angles, DT.transpose(-1, -2))  # локально на nheads_local_tp
```

Каждый TP rank кормит `angles` в MIMO scan с **разным срезом голов** (TP
шардит по head-оси). Значит gradient, текущий назад через `angles_raw`,
разный на каждом ranke — содержит только вклад от локальных голов
этого ranka.

В `megatron/core/tensor_parallel/mappings.py:296-350` backward
`_GatherFromSequenceParallelRegion` имеет две ветки:

- `tensor_parallel_output_grad=True` → `_reduce_scatter_along_first_dim` — сначала
  суммирует grad по всем rankам (собирая вклады от всех голов), потом
  делит по seq-оси. ЭТО ПРАВИЛЬНЫЙ режим когда "computation after gather
  is in the tensor parallel mode" (что у нас и есть — MIMO scan sharded).
- `tensor_parallel_output_grad=False` → `_split_along_first_dim` — просто
  берёт локальный grad каждого ranka и режет по seq. Игнорирует факт
  что разные rank-и имеют разные grad-ы. **Теряет кросс-rank вклады
  от других голов.**

В результате `angle_proj.weight.grad` на rank 0 содержит только
contribution от голов `[0:4]` на seq chunk `[0:64]`, а rank 1 — только
от голов `[4:8]` на seq chunk `[64:128]`. При all-reduce-sum мы получаем
"диагональные" вклады, но теряем "кросс" вклады (голов rank1 на chunk0
и голов rank0 на chunk1). Отсюда `max_abs=3.76`, `max_rel=447.8` —
это не bf16 drift, это структурная потеря ~половины членов суммы.

Подтверждение что это backward-only баг: forward parity чистый
(`max_abs=3.9e-3`, значительно меньше tolerance `5e-2`); изолированный
`test_tp2_sp_on_angle_proj_gather` на forward gather выдаёт
`max_abs < 1e-3`. Все остальные gradient тензоры (`in_proj`, `out_proj`,
`dt_bias`, `D`, `B_bias`, `C_bias`, `mimo_x/z/o`, `B_norm/C_norm`,
LayerNorm) parity проходят — они НЕ идут через `gather_from_sequence_parallel_region`.

**Сравнение с апстримным megatron-core pattern.** Все другие вызовы
`gather_from_sequence_parallel_region` в megatron-core 0.18 используют
default `tensor_parallel_output_grad=True`:

- `megatron/core/tensor_parallel/layers.py:430` (`ColumnParallelLinear.forward`
  при SP) — default (True)
- `megatron/core/transformer/experimental_attention_variant/absorbed_mla.py:449,473`
  (MLA k_pos_emb / kv_compressed) — default (True)
- `megatron/core/transformer/moe/token_dispatcher.py:265,271,274` (MoE
  dispatcher) — default (True)
- `megatron/core/transformer/experimental_attention_variant/dsa.py:818,819`
  (DSA TP gather) — default (True)

Stream B явно отклонился от upstream convention в `cppmega_mamba3_tp_mixer.py:443`
передавая `tensor_parallel_output_grad=False`. **Фикс: убрать этот
kwarg (или поставить `=True`).**

**Фикс НЕ делается в рамках task #86** — hard constraint задачи #86
явно запрещает править мixer, чтобы заставить тест пройти; тест
должен выявить баг, что он и сделал. Фикс принадлежит Stream B и
заблокирует повторную валидацию task #85.

**Импликация для Stream B v2 TP=2 run (task #85).**

Production TP=2 + SP=True launcher
(`scripts/remote_train_h200_nam56r_tp2.sh`) в steady-state производит
**неправильные** градиенты для `angle_proj.weight` в каждом слое Mamba3
MIMO (56 штук). Эффект:

- Optimizer-шаг получает incomplete grad → `angle_proj` параметры
  обучаются как-будто половина голов не существует. На early iterations
  это искажает RoPE-angle-модуляцию, которая подаётся через `angle_dt`
  в SSD scan.
- Поскольку `angle_proj.weight` имеет маленькое fan_in = `d_model=8192`
  и инициализируется малым bound ~`1/sqrt(d_model)`, а градиент в
  оптимизаторе теряет половину информации, convergence `angle_proj`
  будет медленнее и biased. Но сам loss всё равно идёт вниз, т.к.
  остальные 99% параметров корректны.
- Валидационный loss 2.41 @ iter 100 из task #85 — **не trustworthy**
  как числовая метрика для сравнения с TP=1 baseline, т.к. модельная
  математика разная (неправильный backward через angle_proj).
  throughput-число `~36.7k tok/sec` остаётся валидным как **компютер-
  performance** метрика (тайминги не зависят от правильности
  градиентов), но loss / convergence сравнения с TP=1 baseline
  блокируются до фикса line 443.

**Confidence:** HIGH что это тот самый баг. Root-cause подтверждён чтением
`_GatherFromSequenceParallelRegion.backward` в installed megatron-core,
selection флага soundly соответствует semantics "downstream computation
is TP-local" (heads sharded), и изолированный forward-only тест
проходит чисто — что исключает forward-path проблему. Fallout для
Stream B v2 production run ограничен одним параметром (`angle_proj.weight`)
но affects все 56 Mamba3 слоёв.

**Что делать дальше.**

1. Stream B: одна строка фикс в `cppmega/megatron/cppmega_mamba3_tp_mixer.py:443`
   (убрать `tensor_parallel_output_grad=False` kwarg).
2. После фикса: re-run task #86 parity test → ожидается PASS на всех
   13 тестах.
3. После PASS: re-run task #85 production TP=2 run на europe, получить
   новый валидационный loss, сравнить с TP=1 baseline уже с правильной
   математикой.

---

## Сессия 2026-04-12: DSA/TP/recompute оптимизация

### Исследование TP=2 (Streams B, B v2)

- Написан `CppmegaMamba3TPMixer` (589 LOC), TE-нативный паттерн по образцу Megatron `MambaMixer`.
- Числовая совместимость TP=1 vs TP=2: **PASS** (max_abs=1.5625e-2 bf16).
- Найден и исправлен баг B/C layout: upstream `(r,g,n)` -> должно быть `(g,r,n)` при TP>1.
- Найден баг backward `angle_proj` в SP-режиме (Stream I): `tensor_parallel_output_grad=False` -> должно быть `True` (1-строчный фикс).
- Throughput TP=2: **34,672 tok/sec = 3.2x медленнее** чем TP=1 (112k baseline). Подтверждено обеими версиями v1 и v2.
- Корневые причины: overhead коллективов + compute bandwidth-bound + PP=2 VPP=2 эффективнее как топология.
- **Вердикт:** TP>1 — чистый проигрыш для NAM56R Mamba-3 MIMO на single-node H200x8. Mixer сохранён для будущего multi-node.

### Постоянная DSA 9+4 конфигурация attention

- Решение пользователя: 13 A-слоёв = 9 DSA + 4 полных MLA. DSA НЕ опционален.
- A-ранги DSA: `[1,2,3,5,6,7,9,10,11]`, MLA: `[0,4,8,12]`.
- Env var: `CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11"`.
- Механизм уже подключён: `CppMegaSelectiveAttentionLayer` в `nam56r_full_spec.py`.

### Сага оптимизации памяти DSA

- **Stream D v1:** DSA 9+4 BF16 OOM при PP=2 (136 GB, активации MoE stage 1).
- **Stream E:** Порт FP8 indexer из DeepSeek V3.2 через `torch._scaled_mm`. Per-head fused аккумуляция. 9.3-13.4x снижение пиковой дельты. Topk overlap 94.4%. Экономия ~26 GB stage 0 forward.
- **Stream G:** Очистка backward FP8. Indexer-only 69.5% экономии, но full-path только 0.7% потому что основной attention bmm доминирует.
- **Stream D v2:** FP8 indexer применён, stage 0 OK, но stage 1 OOM на активации MoE (136 GB). MTP=2 кладёт лишние веса на stage 1.
- **Stream J:** 4-вариантный sweep памяти (FP8 only, +MoE recompute, +MTP redistribution, PP=4). ВСЕ OOM. Найден **НАСТОЯЩИЙ** bottleneck: `compute_dsa_indexer_loss` в `dsa.py:202` аллоцирует 7.5 GiB FP32 на каждый DSA-слой даже когда `loss_coeff=0`.
- **Гейт loss_coeff==0:** monkey-patch для пропуска KL loss computation при coeff=0. Экономия ~63 GB.
- **Head-streaming:** переписка `_attention_target_fp32` с циклом по головам (7.5 GiB -> 0.8 GiB на слой). Для будущего тренинга с `loss_coeff>0`.
- **ВТОРОЙ bottleneck найден:** `unfused_dsa_fn` в `dsa.py:920` материализует ПОЛНУЮ матрицу `[b*np, sq, sk]` = 7.0 GiB на DSA-слой для ОСНОВНОГО attention (не loss). 5 слоёв x 7 GiB = 35 GiB.
- **sparse_dsa_fn:** gather-scatter замена, вычисляет только topk=16 entries на query (28.7 MB vs 7 GiB, ~250x сокращение).
- **EP=2/EP=4 sweep:** все OOM потому что `unfused_dsa_fn` доминирует, а не MoE веса.

### Найденные готовые sparse attention kernels

- TileLang `tile-ai/tilelang/examples/deepseek_v32/sparse_mla_fwd.py` — fused sparse fwd+bwd, уже в TileLang пакете (но не в pip install, только github).
- NVIDIA PR #3674 "Enable DSA CP/absorbed/THD paths with TileLang fused ops" — `SparseMLA autograd.Function` + TileLang kernels, в Final Review.
- `fla-org/native-sparse-attention` — Triton fwd+bwd.
- `lemyx/tilelang-dsa` — one-pass fused FA+KL в TileLang.
- NVIDIA PR #4039 split-K indexer loss (портирован в cppmega).

### КОРЕННАЯ ПРИЧИНА: отсутствие selective recompute

- Диагностика памяти: 99.7 GB из 119.8 GB на rank = АКТИВАЦИИ без recompute.
- nanochat использует `recompute_granularity="selective"` ПО УМОЛЧАНИЮ. cppmega НИКОГДА этого не имел.
- С selective recompute: ~45-60 GB total (vs 120 GB без).
- Фикс: `--recompute-granularity selective --recompute-modules moe_act` добавлен во все launcher-ы (commit `f4f192c`).
- Тестирование: комбинация full selective recompute + CUDA graphs (пользователь: "не отключайте CUDA graphs, дебажьте").

### Blackwell функции (Stream C)

- GB10 NAM56R-half baseline на реальных данных: **4303.8 tok/sec** (первое честное измерение, предыдущие запуски использовали NullTokenizer).
- 5 Blackwell-функций протестированы, все заблокированы (CuTe DSL не подключён, FP8 lda%16 баг, нет DSA кода, нет TK source, неправильный kernel path).
- Modal B200 DSA indexer bench: FP8 на 11.4% медленнее чем BF16 (слишком мал для амортизации FP8), FP4 не тестируем (TE 2.1 нет FP4 API).

### Исправления окружения

- bench3 SSH IP обновлён (H200_1_IP -> H200_1_IP).
- europe: установлен git SSH-ключ, свежий git clone (раньше был rsync), github аутентифицирован.
- europe: убит зомби-процесс cuTe DSL bench (PID 490683).
- europe: kernel `mamba3_mimo_bwd.py` пропатчен для GQA G<H поддержки (upstream без патча).
- Документирован drift окружений: bench3 имеет locally-patched kernel, europe имел upstream.

### Исследования (6x6 агентов)

- TileLang не имеет kernel-internal TP примитивов.
- Статья Dutt 2026: SSM TP = шардинг nheads, scan остаётся локальным, 1 allreduce на out_proj.
- Статья Mamba-3 молчит о TP; Nemotron-H делегирует Megatron defaults.
- NCCL + CUDA graphs совместимы с NCCL 2.9 (external TP не ломает CUDA graph fusion).
- nvFuser #6003: fused comm+compute на Hopper = 4 vs 50 TFLOP/s (kernel-internal TP проигрывает).
- state-spaces/mamba PR #850: community Mamba-3 TP реализация (не вмерджена, Triton не TileLang).

### Созданные/изменённые файлы (commit 3eb75fe -> f4f192c)

- `cppmega/megatron/cppmega_mamba3_tp_mixer.py` (589 LOC, новый)
- `cppmega/megatron/dsa_fp8_indexer.py` (новый, FP8 + head-streaming)
- `cppmega/megatron/dsa_fp8_patch.py` (новый, 3-tier monkey-patch)
- `cppmega/megatron/dsa_sparse_attention.py` (новый, gather-scatter sparse)
- `cppmega/megatron/dsa_splitk_indexer_loss.py` (новый, порт PR #4039)
- `cppmega/megatron/dsa_tilelang_fused_kl.py` (новый, порт lemyx)
- `cppmega/megatron/memory_debug.py` (новый, скопирован из nanochat)
- `cppmega/megatron/fp8_activations.py` (новый, скопирован из nanochat)
- `cppmega/megatron/tilelang_sparse_mla/` (новый каталог, fetched из tilelang examples)
- `tests/test_cppmega_mamba3_tp_mixer.py` (458 LOC, 11+2 теста)
- `tests/test_dsa_fp8_indexer.py` (11 тестов)
- `tests/test_dsa_splitk_indexer_loss.py` (6 тестов)
- `tests/test_dsa_tilelang_fused_kl.py` (17 тестов)
- `scripts/modal_dsa_indexer_bench.py` (804 LOC)
- Множество launcher-скриптов для DSA/EP/TP/grid sweep
- Обновления документации в обоих планах оптимизации + grid search + blackwell sweep

### Production DSA конфигурация NAM56R (2026-04-12)

**topk=256** (6.25% density при seq=4096). Каждый query token attend'ит к 256 из 4096 предыдущих позиций.

Предыдущий default topk=16 (0.4% density) был placeholder — слишком агрессивный sparse, модель теряла контекст. DeepSeek V3.2 production использует 3-12% density range.

|    topk |   Density | Совместимость TileLang | Статус                         |
| ------: | --------: | ---------------------- | ------------------------------ |
|      16 |      0.4% | ❌ (16 % 64 ≠ 0)        | Placeholder, убран             |
|      64 |      1.6% | ✅                      | Минимальный                    |
|     128 |      3.1% | ✅                      | Нижняя граница DeepSeek range  |
| **256** | **6.25%** | **✅**                  | **Production default**         |
|     512 |     12.5% | ✅                      | Верхняя граница DeepSeek range |

**Sparse attention kernel**: TileLang fused SparseMLA (default, `CPPMEGA_DSA_SPARSE_MODE=tilelang`).
- Источник: `tile-ai/tilelang/examples/deepseek_v32/sparse_mla_fwd.py` + NVIDIA Megatron-LM PR #3674
- Принцип: gather только topk K/V entries в shared memory → fused Q@K^T + online softmax + S@V
- Память: near-zero extra (vs unfused_dsa_fn 7 GiB per layer)
- Требование: topk % 64 == 0 (256 % 64 = 4 blocks ✓)
- Fallback: `CPPMEGA_DSA_SPARSE_MODE=gather_scatter` (PyTorch, без TileLang JIT)

**DSA KL loss target**: 3 режима через `CPPMEGA_DSA_KL_MODE`:
- `head_streaming` (default): Python loop over heads, 0.8 GiB/call (vs 7.5 GiB naive)
- `splitk`: NVIDIA PR #4039 Triton split-K kernels, 60% memory save
- `tilelang_fused`: lemyx/tilelang-dsa one-pass fused FA+KL, near-zero extra memory

**FP8 indexer**: `CPPMEGA_DSA_INDEXER_DTYPE=fp8` через `apply_dsa_fp8_patch()`. Port из DeepSeek V3.2 `fp8_index` TileLang kernel → `torch._scaled_mm` (WGMMA на H200). Per-head fused accumulation, never materializes [sq,h,sk]. Topk overlap BF16 vs FP8: 94.4%.

**Selective recompute**: `--recompute-granularity selective --recompute-modules moe_act`. Совместимо с per-module CUDA graphs. Saves ~26 GB activation memory (99.7 GB → ~74 GB). Root cause всех DSA OOM'ов — отсутствие этого флага.

**Полная production конфигурация DSA 9+4**:
```bash
export CPPMEGA_DSA_A_LAYER_RANKS="1,2,3,5,6,7,9,10,11"
export CPPMEGA_DSA_INDEXER_DTYPE=fp8
export CPPMEGA_DSA_SPARSE_MODE=tilelang   # fused TileLang sparse attention
export CPPMEGA_DSA_KL_MODE=head_streaming  # or splitk / tilelang_fused
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

--enable-dsa \
--dsa-indexer-topk 256 \
--dsa-indexer-n-heads 8 \
--dsa-indexer-head-dim 64 \
--dsa-indexer-loss-coeff 0.001 \
--recompute-granularity selective \
--recompute-modules moe_act
```

## AbsorbedMLA + Fused Sparse DSA (2026-04-12)

### Проблема: decompress path не масштабируется при topk=256

Стандартный Megatron MLA: decompress KV → gather topk K/V → einsum.
При topk=256 gather `[b, np, sq, topk, hn]` = **14 GiB per DSA layer** (28 heads × 4096 × 256 × 96 × bf16).
9 DSA слоёв × 14 GiB = **126 GiB** только на sparse gather — не влезает ни при каком EP/PP.

### Решение: absorption trick (как у DeepSeek V3)

DeepSeek **никогда** не decompressit K/V для sparse attention:

```
Стандартный путь (OOM при topk=256):
  compressed_kv [b, s, 1, 96] → W_k_up → K [b, s, 28, 96]  ← decompress 28 heads
  sparse_gather(K, topk=256) → [b, 28, s, 256, 96] = 14 GiB  ← OOM

Absorbed путь (DeepSeek, PR #3674):
  Q_absorbed = Q @ W_k_up^T                    ← один einsum
  sparse_gather(compressed_kv, topk=256) → [b, 256, 1, 96] = 49 KB  ← крошечный!
  scores = Q_absorbed @ kv_gathered^T           ← 28 heads × 256 topk
  output = attn_out @ W_v_up                    ← V up-proj ПОСЛЕ attention
```

Разница: gather на **1 head × compressed** vs **28 heads × decompressed** = **28× меньше памяти**.

### Upstream PRs применённые к нашему Megatron `dev`

| PR                                   | Что даёт                                                         | Статус                |
| ------------------------------------ | ---------------------------------------------------------------- | --------------------- |
| **#3193** AbsorbedMLASelfAttention   | Absorption trick для MLA, MQA-style attention                    | MERGED в dev          |
| **#3674** DSA bridge + TileLang ядра | `get_absorb_query_key_value_tensors()`, fused sparse MLA fwd+bwd | OPEN, applied вручную |
| **#3039** MLA down-proj fusion       | Fuses MLA down-projection GEMMs: `--mla-down-proj-fusion`        | MERGED в dev          |
| **#3649** Zero-copy checkpoint       | Снижает peak memory при recompute backward                       | MERGED в dev          |
| **#3401** MoE+MTP hang fix           | Deadlock fix при MoE aux loss + MTP                              | MERGED в dev          |
| **#3399** Uneven PP fix              | Fix Mamba при неравном PP split (раньше молча терял слои!)       | MERGED в dev          |
| **#4173** Mamba offloading           | Fine-grained activation offloading для hybrid Mamba+MoE          | MERGED в dev          |

### DeepEP flex dispatcher (EP>1)

- **Библиотека**: deepseek-ai/DeepEP v1.2.1, собран с NVSHMEM на bench3
- **Флаг**: `--moe-token-dispatcher-type flex --moe-router-dtype fp32`
- **Требование**: `TP × EP > 1` (при EP=1 fallback на alltoall через `VARIANT=v0`)
- **DeepEP .so**: собран на bench3, скопирован на europe через scp
