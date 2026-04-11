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

| # | Конфиг | Untied→Tied | MTP standalone | MBS | MTP | Доп. | Ожидаемый tok/sec |
|---|---|:-:|:-:|:-:|:-:|---|---|
| 1 | Текущий baseline (control) | untied | нет | 4 | on | — | **112,152 (проверка)** |
| 2 | Только tied embeddings | **tied** | нет | 4 | on | — | ~125-135k |
| 3 | Только standalone VPP | untied | **да** | 4 | on | — | ~140-155k |
| 4 | **Tied + standalone** | **tied** | **да** | 4 | on | — | **~157-165k (основная цель)** |
| 5 | Полные фиксы + MBS=5 | tied | да | 5 | on | — | ~170-185k |
| 6 | Полные фиксы + MBS=6 | tied | да | 6 | on | — | ~180-200k (если память позволяет) |
| 7 | Полные фиксы + PP=4 VPP=2 | tied | да | 4 | on | PP=4 | тест bubble shift |
| 8 | NoMTP control (архитектурная регрессия) | tied | — | 4 | **off** | — | ~133k (sanity check) |

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

| Вариант | Конфиг | Iter ms | Tok/sec | LM loss@30 | MTP loss@30 |
|---|---|---|---|---|---|
| V1 baseline | untied, MTP on | 2348.5 | 111,621 | 2.70 | 2.66 |
| V2 `mtp_use_repeated_layer=True` depth=1 | Super flags | 2355.8 | 111,275 | 2.81 | 2.54 |
| V3 `mtp_use_repeated_layer=True` depth=2 | Super flags | 2688.9 | 97,530 | 2.75 | 2.66/2.49 |

**Вывод:** `mtp_use_repeated_layer=True` работает для Mamba hybrid, но даёт 0% ускорения при depth=1 (no-op когда всего 1 слой). При depth=2 деградация −12.6% (ожидаемо --- shared weights экономят параметры, не FLOPs). MTP overhead --- это forward+backward FLOPs, не число параметров.

### Эксперимент с tied embeddings (europe)

| Конфиг | Iter ms | Tok/sec | Дельта |
|---|---|---|---|
| Untied (baseline) | 2348.5 | 111,621 | 0% |
| Tied | 2349.4 | 111,623 | −0.04% |

**Вывод:** Нулевой эффект на PP=2 hybrid. Megatron не может объединить embedding/output head через границы PP rank-ов.

### Результаты sweep из 8 вариантов MTP (europe + bench3)

| # | Название | Iter ms | Tok/sec | Статус |
|---|---|---|---|---|
| 1 | Control untied | 2348.5 | 111,666 | baseline |
| 2 | Только tied | 2349.4 | 111,623 | 0% выигрыш |
| 3 | Standalone VPP | --- | --- | ЗАБЛОКИРОВАН (Megatron hybrid) |
| 4 | Tied + standalone | --- | --- | ЗАБЛОКИРОВАН |
| 5 | Tied MBS=5 | --- | --- | OOM (~142/140 GB) |
| 7 | PP=4 VPP=1 | 3258.3 | 80,490 | −28% регрессия |
| 8 | NoMTP control | 1981.2 | 132,438 | +18.6% (подтверждает 133k) |

**Ключевые выводы из sweep:**
- Standalone VPP (варианты 3, 4) ЗАБЛОКИРОВАН: Megatron hybrid (`mamba_model.py:195-199`) явно не поддерживает standalone MTP placement. PR #3377 подтверждает.
- NVIDIA Nemotron 3 Super обходит это используя PP=1 (без pipeline parallelism).
- MBS=5 (вариант 5) вызывает OOM при ~142/140 GB --- нет запаса даже с tied embeddings.
- PP=4 VPP=1 (вариант 7) --- регрессия −28% из-за увеличения pipeline bubble.
- Основная цель плана (вариант 4, tied + standalone, ~157-165k) недостижима без upstream изменений Megatron для поддержки standalone MTP на гибридных моделях.

### Liger fused CE для MTP (bench3)

| Метрика | Стандартный CE | Liger fused | Дельта |
|---|---|---|---|
| MTP time (4 depths fwd+bwd) | 178.8 ms | 483.2 ms | 2.7x МЕДЛЕННЕЕ |
| Пиковая память | 27.36 GB | 5.49 GB | −82% |

**Вывод:** Liger экономит 82% памяти, но в 2.7x медленнее на H200 (плохая утилизация тензорных ядер на chunked small-M GEMMs). НЕ включать на H200. Ценно только для hardware с ограниченной памятью.

### Эксперименты с CUDA graphs (bench3)

| Область | Статус | Tok/sec | Дельта |
|---|---|---|---|
| Baseline (без графов) | --- | 68,844 | --- |
| `--cuda-graph-scope attn` | PASS | 69,822 | +1.4% |
| `--cuda-graph-scope full_iteration` | FAIL | --- | Блокер MoE `.item()` |
| `transformer_engine` | FAIL | --- | Тот же блокер MoE |

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

| Рычаг | Ожидаемый эффект | Статус |
|---|---|---|
| Per-module CUDA graphs (существующий рецепт) | Уже в конфиге 211k | нужен применённый cuDNN фикс |
| `--moe-pad-expert-input-to-capacity` | Включает полный MoE CUDA graph | не тестировался с фиксом |
| FP8 на MLA+MoE | +15-20% | ожидает |
| TP=2 PP=2 VPP=2 | +15-25% | средняя трудоёмкость |
| Удаление MTP (крайний случай) | +18.6% | архитектурная регрессия |
