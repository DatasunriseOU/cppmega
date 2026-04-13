# NAM56R -- Сессия оптимизации 2026-04-13

**Модель**: NAM56R 4.73B гибридная (27 Mamba3 + 9 DSA + 4 полных MLA + 12 MoE), heads=32, hidden=4096, 52 слоя
**Оборудование**: H200x8 однонодовый (bench3 LOCATION_1, europe LOCATION_2), GB10 (для бенчмарков компиляции)
**Ветка**: `main` (коммит `b28a9f1`)
**Предыдущий baseline**: 240 TFLOP/s (PP=2 MBS=4, h=3584), 267 TFLOP/s золотая конфигурация (FP8 tensorwise MBS=10)

---

## 1. Завершено и проверено

### 1.1. Исправление сохранения чекпоинтов

| Пункт | Подробности |
|-------|-------------|
| **Корневая причина** | Guard в `Mamba3._apply` удерживал D/dt_bias в fp32, в то время как Float16Module приводил остальное к bf16. Распределённый оптимизатор создавал перекрёстные ссылки optimizer states между bf16/fp32 буферами при сохранении |
| **Исправление** | Полностью удалён guard в `_apply`. D/dt_bias теперь bf16 после Float16Module. Forward использует `.float()` приведение inline |
| **Верификация** | save@step=10 и save@step=20 -- оба `successfully saved` на PP=1 и PP=2 |
| **Влияние** | Разблокированы все зависящие от чекпоинтов процессы (resume, evaluation, export) |

### 1.2. Очистка кода

Удалены fallback-пути и устаревшие файлы для обеспечения единого пути DSA (lemyx + IndexCache):

| Действие | Файл |
|----------|------|
| Удалены fallback-и | `cppmega/megatron/index_cache_patch.py` |
| Удалены fallback-и | `cppmega/megatron/lemyx_dsa_warmup.py` |
| Удалён | `dsa_fp8_patch.py` |
| Удалён | `dsa_fp8_indexer.py` |
| Удалён | `dsa_tilelang_fused_kl.py` |
| Удалены env var gates | `CPPMEGA_INDEX_CACHE`, `CPPMEGA_LEMYX_DSA` -- теперь всегда включены |

### 1.3. Изменение архитектуры: heads=32, hidden=4096

Изменено с heads=28 на heads=32 для одновременного удовлетворения трёх ограничений:

| Ограничение | Требование | heads=28 | heads=32 |
|-------------|------------|----------|----------|
| FP8 tensorwise | `heads % 8 == 0` | 28%8=4 ОШИБКА | 32%8=0 OK |
| Тайлинг WGMMA | `head_dim` выровнен по тайлу | Не выровнен | Выровнен |
| Ядро lemyx | `heads == index_heads` | Несовпадение | Совпадение |

52 слоя подтверждены (`NAM56R_DEPTH=52`). Делится нацело на 4 (VPP=4) и на 2 (VPP=2).

### 1.4. NVIDIA Apex установлен

Apex собран из исходников с CUDA-расширениями на обоих H200-машинах:
- bench3 (LOCATION_1, `/mnt/data/venv`)
- europe (LOCATION_2, `/home/dave/cppmega-root/cppmega-venv`)

### 1.5. Ребейз Megatron

Ребейз на последний upstream dev. Cherry-pick PR-ов:

| PR | Описание | Статус |
|----|----------|--------|
| #3674 | DSA absorbed MLA | Применён |
| #4268 | Отложенный wgrad overlap с P2P backward | Применён |

Синхронизировано на: bench3, GB10, GCS.

### 1.6. Автономный тест DualPipeV

Пакет `deepseek-ai/DualPipe` установлен на всех 3 машинах (europe, bench3, GB10).

| Тест | Результат |
|------|-----------|
| Маппинг стадий PP=2 | 4 стадии: rank 0 = стадии (0,3), rank 1 = (1,2) |
| Корректность loss | Совпадает с эталоном |
| Корректность градиентов | Совпадает с эталоном |

Код интеграции: `cppmega/megatron/dualpipev_schedule.py` (897 строк). Реализует:
- Разбиение стадий: 52 слоя на 4x13 виртуальных стадии
- Обёртка функции loss
- Замена training step через monkey-patch

### 1.7. Результаты региональной компиляции torch.compile (GB10)

Тесты отдельных подмодулей с `torch.compile` на GB10:

| Подмодуль | Ускорение | Вердикт |
|-----------|-----------|---------|
| Вычисление data-dependent A | **5.93x** | КОМПИЛИРОВАТЬ |
| Пре-обработка Mamba3 | **2.66x** | КОМПИЛИРОВАТЬ |
| Пост-обработка Mamba3 | **1.84x** | КОМПИЛИРОВАТЬ |
| SiLU + gate multiply | **1.35x** | КОМПИЛИРОВАТЬ |
| RMSNorm | 0.41x | НЕ компилировать |
| RMSNormGated | 0.47x | НЕ компилировать |
| MoE Router | 0.97x | НЕ компилировать |
| MLA проекции | 1.01x | НЕ компилировать |

Файл патча: `cppmega/megatron/mamba3_compile_patch.py` (423 строки).

### 1.8. Результаты пропускной способности (конфигурация h=4096 heads=32)

| Конфигурация | TFLOP/s | Примечания |
|--------------|---------|------------|
| PP=2 VPP=2 EP=4, без compile | 193-194 | Save работает, готово к production |
| PP=1 EP=4 MBS=8, без compile | 193 | Нет pipeline bubble, больше модели на GPU |
| PP=2 MBS=4 (старый h=3584) | 240 | Baseline до изменения архитектуры |

Конфигурация h=4096 медленнее h=3584 на ~20% из-за большего размера модели. Закрытие этого разрыва -- цель DualPipeV, региональной компиляции и EP overlap.

### 1.9. Исследование combined_1f1b

Флаг: `--overlap-moe-expert-parallel-comm`. Требует `build_schedule_plan()` на объекте модели.

| Находка | Подробности |
|---------|-------------|
| GPTModel | Имеет `build_schedule_plan()` |
| MambaModel | НЕ имеет |
| Upstream PR | Не существует для поддержки MambaModel |
| Подтверждено | 6 независимых поисковых агентов |

Блокер: необходимо написать `hybrid_schedule_plan.py` для добавления `build_schedule_plan()` в гибридную MambaModel (MoE-слои переиспользуют GPT `TransformerLayerNode`, Mamba-слои используют единый opaque node). Оценка: ~150-200 строк.

---

## 2. В работе

### 2.1. Интеграция региональной компиляции

Файл: `cppmega/megatron/mamba3_compile_patch.py` (423 строки, переписан).

**Текущий блокер**: Патч `CppMegaMamba3TE.forward` имеет конфликт kwarg `padding_mask` с `te_checkpoint` + `torch.compile`. Значение по умолчанию `padding_mask=None` в патченной сигнатуре не пробрасывается через обёртку checkpoint TE.

**Исправление в процессе**: Добавить явный `padding_mask=None` в сигнатуру скомпилированной функции, обходя проброс kwargs через checkpoint TE.

### 2.2. build_schedule_plan для гибридной модели

Написание `hybrid_schedule_plan.py`:
- MoE-слои: переиспользование GPT `TransformerLayerNode` (имеет аннотации A2A comm)
- Mamba-слои: единый opaque `ScheduleNode` (нет перекрываемых коммуникаций)
- Удаление assert `isinstance(GPTModel)` в построителе расписаний Megatron
- Цель: ~150-200 строк

### 2.3. Тест VPP=4

Конфигурация: PP=2 VPP=4, 52/4=13 слоёв на виртуальную стадию. Тестируется на bench3. Ожидается уменьшение pipeline bubble с ~24% до ~12%.

### 2.4. Ядро lemyx для heads=32

Ядро обновлено под новое количество голов. Автономный тест выдал `cudaErrorLaunchFailure`. Корневая причина: кэш TileLang JIT содержит устаревшее скомпилированное ядро для heads=28. Требуется полная перекомпиляция с новым количеством голов.

---

## 3. План (следующие шаги)

### 3.1. combined_1f1b EP Overlap для гибридной модели

**Предусловие**: `hybrid_schedule_plan.py` (раздел 2.2).
**Флаг**: `--overlap-moe-expert-parallel-comm --delay-wgrad-compute`
**Ожидание**: Скрыть EP AlltoAll за вычислениями Mamba. AlltoAll сейчас ~8% от времени шага при EP=4.

### 3.2. Полный тренировочный тест DualPipeV

Подключить `cppmega/megatron/dualpipev_schedule.py` к точке входа тренировки. Тест PP=2 с расписанием DualPipeV.
**Ожидание**: Практически нулевой pipeline bubble (DualPipeV перекрывает forward/backward разных микробатчей между стадиями пайплайна).

### 3.3. Региональная компиляция на H200

Исправить проблему `padding_mask`, замерить реальный прирост пропускной способности на H200.
На GB10 показано **5.93x** на data-dependent A. Ожидается ~8% общего прироста пропускной способности модели от компиляции 4 выигрышных подмодулей.

### 3.4. PR #3116 (Seq1F1B)

Изменено 30 файлов, конфликтует с нашей кодовой базой. Требуется ручной merge. Включает последовательно-уровневый pipeline parallelism для длинного контекста.

### 3.5. Селективный FP8 для MoE

Файл: `cppmega/megatron/selective_fp8_moe_patch.py`.
Требуется отладка (проброс env var через менеджер контекста FP8 Megatron). MoE GEMMs составляют ~15% вычислений.

### 3.6. Оптимизация ядра Mamba SSM

SSM = 34.5% времени GPU, 255 регистров, 6.25% occupancy.

| Приоритет | Оптимизация | Ожидаемый эффект |
|-----------|-------------|------------------|
| P1 | TMA загрузки (замена ручных gmem->smem) | +15-20% пропускной способности |
| P2 | State checkpoint (уменьшение перевычислений в bwd) | -20% времени SSM bwd |
| P3 | Снижение давления регистров (цель <128 рег.) | 2x occupancy |
| P4 | Слияние поэлементных операций в ядре SSM | -5% от общего |
| P5 | Многочанковый пайплайнинг | +10% перекрытие |

---

## 4. Архитектурные решения

| Решение | Обоснование |
|---------|-------------|
| **heads=32, hidden=4096** | Production-конфигурация. Совместим с FP8/WGMMA/lemyx |
| **52 слоя** (NAM56R_DEPTH) | Делится на 4 (VPP=4) и на 2 (VPP=2) |
| **Единый путь DSA** | lemyx (разогрев) + IndexCache (production). Без fallback-ов |
| **Без guard в _apply** | D/dt_bias остаются bf16 после Float16Module. `.float()` в forward |
| **DualPipeV предпочтителен** | Вместо стандартного 1F1B PP для pipeline parallelism |
| **combined_1f1b для EP overlap** | При PP=1, когда build_schedule_plan будет готов |

---

## 5. Ключевые ссылки

| Ссылка | Описание |
|--------|----------|
| `cppmega/megatron/dualpipev_schedule.py` | Интеграция DualPipeV (897 строк) |
| `cppmega/megatron/mamba3_compile_patch.py` | Патч региональной компиляции (423 строки) |
| `cppmega/megatron/index_cache_patch.py` | Интеграция IndexCache (очищен) |
| `cppmega/megatron/lemyx_dsa_warmup.py` | Разогрев lemyx DSA (очищен) |
| `cppmega/megatron/selective_fp8_moe_patch.py` | Селективный FP8 для MoE-слоёв |
| `scripts/remote_smoke_h200_dsa_9_4_m.sh` | Скрипт smoke-теста H200 (564 строки) |
| Megatron PR #4268 | Отложенный wgrad overlap (cherry-picked) |
| Megatron PR #4099 | Переименование MambaModel в HybridModel (открыт, не применён) |
| Megatron PR #3116 | Seq1F1B (30 файлов, не применён) |
| Megatron Issue #1810 | Deadlock с A2A overlap на неравных стадиях пайплайна |
| `deepseek-ai/DualPipe` | Эталонная реализация DualPipeV |
| DeepSeek-V3 GB200 guide | Справочник по production EP тюнингу |

---

## 6. Статус машин

| Машина | Расположение | Путь env | Статус |
|--------|-------------|----------|--------|
| bench3 (H200x8) | LOCATION_1 | `/mnt/data/venv` | Основная тестовая машина. Apex установлен. Megatron ребейзнут |
| europe (H200x8) | LOCATION_2 | `/home/dave/cppmega-root/cppmega-venv` | Вторичная. Apex установлен. Megatron ребейзнут |
| GB10 | локально | N/A | Только для бенчмарков компиляции. DualPipe установлен |

Все машины: torch 2.12+cu132, mamba_ssm 2.3.1, TE 2.13, megatron-core 0.18.
