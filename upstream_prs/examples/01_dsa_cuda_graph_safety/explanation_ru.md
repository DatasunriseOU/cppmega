# Подробное объяснение: безопасность CUDA Graph для DSA

## Что это вообще и зачем

**DSA** (DeepSeek Sparse Attention) — это разновидность attention механизма, в котором вместо того чтобы каждый query-токен смотрел на все key-токены (это дорого по памяти и компьюту), сначала работает специальный модуль-индексатор (**indexer**). Он быстро прикидывает: какие из key-токенов вообще стоит рассматривать. После этого внимание считается только по выбранным позициям. Получается дёшево и почти так же качественно.

Внутри индексатора есть операция **scatter_topk** — буквально "разложить топ-K выбранных индексов по нужным позициям маски". Маска — это матрица размера `[batch, query_len, key_len]`, где `0.0` означает "сюда смотрим", а `-inf` — "запрещено". Индексатор выдаёт список вида "для query 7 выбраны keys [3, 17, 42, ...]", и `scatter_topk` ставит нули в нужные клетки.

Иногда индексатор не находит нужное количество кандидатов и в списке появляется специальное значение `-1` (sentinel — "пустота, игнорируй"). Эти `-1` нельзя просто так использовать как индексы — иначе scatter попадёт в позицию -1 (которая в Python превратится в "последнюю", и мы испортим маску).

**CUDA Graph capture** — это режим работы PyTorch, в котором вместо того чтобы каждый шаг заново отправлять сотни мелких CUDA-команд на GPU (а это дорого по CPU overhead), мы один раз "записываем" всю последовательность команд в граф и потом просто проигрываем его. Это даёт ощутимый прирост скорости. НО: пока идёт запись, нельзя ничего читать обратно с GPU на CPU — потому что граф должен быть полностью независим от значений.

## Как это ДОЛЖНО было работать

1. На каждом шаге обучения включается режим CUDA Graph capture (`--cuda-graph-impl transformer_engine` в нашем стеке).
2. PyTorch начинает записывать в граф все операции внутри forward и backward.
3. DSA-индексатор выдаёт список топ-K индексов с возможными `-1`-сентинелами.
4. `_scatter_topk_into_index_mask` раскладывает их в маску и пропускает sentinel-позиции.
5. Граф успешно записан, дальше каждый шаг просто replay'ится — экономим CPU-время.

## Что и почему НЕ работает

**Симптом**: тренировка падает при включении CUDA Graph с ошибкой типа `cudaErrorStreamCaptureInvalidated` (на bench3 с torch 2.12 nightly именно такая формулировка; в более старых документах встречается `cudaErrorStreamCaptureUnsupported` — это устаревшее наименование, по сути та же ошибка).

**Почему**: в `dsa.py` есть несколько мест, которые делают неявную синхронизацию GPU и CPU:

1. **Validation checks через `torch.equal()`**:
   - `torch.equal(finite, expected)` — проверяет что indexer выдал ожидаемое.
   - `torch.equal(key_positions, expected_key_pos)` — проверяет позиции.
   - `torch.equal(mask[bi], ref_mask)` — проверяет консистентность маски.
   
   Под капотом `torch.equal` — это "сравни два тензора на GPU, потом верни одно булево значение в Python". Чтобы вернуть это значение, PyTorch вынужден дёрнуть `.item()`, а это `cudaStreamSynchronize` — стоп-машина для capture.

2. **Условие `if torch.any(idx_chunk < 0)`** в `_scatter_topk_into_index_mask`:
   - Сначала считается на GPU "есть ли где-то -1".
   - Потом результат вытягивается на CPU чтобы сделать `if`.
   - Внутри ещё одно `if valid_topk.any():`. Та же история — две CPU-синхронизации в одной функции.

Корень проблемы: код рассчитан на eager mode (где синхронизации стоят дёшево), а под capture любая синхронизация ломает запись графа целиком.

## Как мы предлагаем чинить

### 1. Validation checks отключаем под capture

Эти проверки нужны были для дебага в раннем DSA — они верифицируют инварианты, которые на самом деле всегда выполнены by construction (они обязаны быть истинными по логике индексатора). На production это мёртвый код, который только мешает.

Минимальный фикс — обернуть в `if False:` или удалить. Чище — гейтнуть через capture-чек:

```python
if not torch.cuda.is_current_stream_capturing():
    assert torch.equal(finite, expected), "..."
```

Так в обычных eager-прогонах проверка работает (полезно для дебага), а в capture-mode тихо пропускается.

### 2. Branchless-версия scatter с обработкой sentinel'ов

Вместо `if`, который требует значение на CPU, делаем всё сразу — без ветвления:

```python
# Шаг 1: запоминаем, где были -1, в виде булевого тензора (на GPU).
sentinel = idx_chunk < 0

# Шаг 2: clamp заменяет все -1 на 0. Теперь scatter не упадёт.
safe_chunk = idx_chunk.clamp(min=0)

# Шаг 3: scatter — раскладываем индексы. Где был sentinel, попадёт в позицию 0.
index_mask[:, s0:s1].scatter_(-1, safe_chunk, 0.0)

# Шаг 4: чиним позицию 0 для тех строк, где был sentinel но НЕ было реального 0.
has_sent = sentinel.any(dim=-1)            # есть ли sentinel в строке
has_real0 = ((idx_chunk == 0) & ~sentinel).any(dim=-1)  # был ли там настоящий 0
fixup = has_sent & ~has_real0              # надо вернуть -inf в позицию 0
index_mask[:, s0:s1, 0].masked_fill_(fixup, float("-inf"))
```

Ключевой момент: `any(dim=-1)` тут НЕ скалярная редукция — мы редуцируем по последней оси и получаем тензор формы `[b, chunk_len]`, а не одно булево значение. CPU-синхронизация не нужна, capture проходит.

### Edge cases

- **Строка содержала и -1, и реальный 0**: оба попадают в `safe_chunk[..., 0]`. После scatter позиция 0 будет равна `0.0` (правильно — там был настоящий 0). `has_real0` = True, `fixup` = False. Не трогаем. OK.
- **Строка содержала только -1, без реального 0**: после scatter позиция 0 ошибочно равна `0.0`. `has_sent` = True, `has_real0` = False, `fixup` = True. Возвращаем `-inf`. OK.
- **Строка не содержала -1**: clamp ничего не делает (все индексы уже ≥ 0). `has_sent` = False, fixup не срабатывает. OK.

### Альтернативы

- **Можно полностью переписать на CUDA-кернел** — но это огромный over-engineering для двух десятков строк PyTorch.
- **Можно просто всегда заменять -1 на 0 и не делать fixup** — но тогда получим лишние "разрешённые" позиции в маске, что испортит softmax.

Branchless clamp+scatter+fixup — минимальный честный фикс.

## Где это в коде и как смотреть

- **Upstream**: `megatron/core/transformer/experimental_attention_variant/dsa.py`
  - Строки ~645 — `torch.equal(finite, expected)` и соседние validation checks.
  - Функция `_scatter_topk_into_index_mask` — ветка `if torch.any(idx_chunk < 0)`.
- **Наш production-патч**: `cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py` (Patch 1 и Patch 8).
- **Reproducer**: `/Volumes/external/sources/cppmega/upstream_prs/examples/01_dsa_cuda_graph_safety/reproducer.py`
- **PR template**: `/Volumes/external/sources/cppmega/upstream_prs/01_dsa_cuda_graph_safety.md`

Запуск reproducer:
```bash
cd upstream_prs/examples/01_dsa_cuda_graph_safety
pip install -r requirements.txt
python reproducer.py
```

В нашем стеке этот фикс активируется автоматически после запуска `apply_dsa_cg_patches.py` (см. `feedback_mandatory_patches.md`).

## Проверка что фикс работает

**До фикса** (unpatched):
```
(A) UNPATCHED — expecting CUDA graph capture to FAIL
  capture raised (as expected): RuntimeError: ... stream is capturing ...
  BUG_REPRODUCED
```
Реальный текст ошибки на bench3 / torch 2.12 nightly: `cudaErrorStreamCaptureInvalidated` (а не `Unsupported`, как в старом тексте README) — это нормально, оба сообщения относятся к одной и той же категории "запрещённая операция под capture".

**После фикса** (patched):
```
(B) PATCHED — expecting CUDA graph capture to SUCCEED
  capture OK, replay matches eager reference exactly
  FIX_VALIDATED

VERDICT: bug reproduced on unpatched + fix validated on patched.
```
Exit code: `0`.

В реальной тренировке: запускаем NAM56R с `--cuda-graph-impl transformer_engine --cuda-graph-scope attn mamba moe_router moe_preprocess` на 8xH200 — шаги проходят без `cudaErrorStreamCaptureUnsupported`, loss сходится идентично non-CG baseline.
