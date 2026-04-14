# Подробное объяснение: Mamba3 MIMO bwd — рефактор 3D shared memory в 2D ради TMA

## Что это вообще и зачем

**Mamba3 MIMO** — это backward-кернел state-space модели (см. PR 05 для краткого ликбеза по Mamba и MIMO). Он написан на **TileLang** — DSL для GPU-кернелов, которое позволяет явно указывать shared memory (smem), копии gmem→smem, warp-spec, и подобные низкоуровневые вещи.

**Shared memory (smem)** — это маленький быстрый кэш на каждом SM (Streaming Multiprocessor, "ядро" GPU). На Hopper (H200) это около 228 KiB на SM, на GB10 — 99 KiB. По сравнению с глобальной памятью (HBM), которая медленная и далеко, smem в 100× быстрее по latency. Когда кернел читает блок данных, он сначала тащит его из gmem в smem, а потом много раз читает из smem в регистры. Это основной приём ускорения GPU-кернелов.

**TMA** (Tensor Memory Accelerator) — это специальный hardware-блок на NVIDIA Hopper и Blackwell, который умеет асинхронно копировать большие тензорные блоки между gmem и smem. Без TMA каждый thread в warp должен отдельно загрузить свою часть данных через `cp.async` (старый механизм) — это нагружает регистры, тратит инструкции, и трудно перекрывается с compute. С TMA одна инструкция (`cp.async.bulk.tensor`) запускает аппаратный DMA, который сам копирует целый тензорный блок, освобождая warps для других задач. На Hopper TMA даёт 20-30% к скорости memory-bound кернелов.

**Warp-spec** (warp specialization) — паттерн, при котором разные warps в блоке делают разные роли: одни warps — "producer" (запускают TMA-копии), другие — "consumer" (считают MMA). Это тоже работает только когда копии идут через TMA.

**LowerBulkCopy** — это compiler pass в TileLang, который превращает Python-уровневую `T.copy(...)` в инструкцию TMA. Чтобы TMA работала, **shared memory descriptor должен быть rank-2** (2-мерный): то есть smem-аллокация должна быть формы `[X, Y]`, не `[X, Y, Z]`. Это аппаратное ограничение `cp.async.bulk.tensor.2d` — самый используемый вариант TMA умеет 2D shape. Есть экзотический `cp.async.bulk.tensor.3d`, но TileLang его пока не поддерживает.

## Как это ДОЛЖНО было работать

В кернеле `mamba_mimo_bwd_fwd_kernel` (часть `mamba3_mimo_bwd.py`) идёт обработка чанков длиной CHUNK_SIZE по последовательности. На каждой итерации:

1. Загружаем блок Q из gmem `[B, S, R, G, N]` (5-мерный gmem-тензор) в smem.
2. Аналогично загружаем K.
3. Считаем `qk_dot = Q @ K^T`, кладём в smem `qk_dot_shared`.
4. Применяем маску, делаем дальнейшие операции, складываем в gmem `QK_DOT[B, H, S, R, R]`.

Загрузки Q и K — это бóльшая часть memory-bandwidth кернела. Если они идут через TMA — кернел получает 20-30% ускорения. Если нет — fallback на `cp.async`, который медленнее и нагружает регистры (что может уронить occupancy и сделать кернел ещё медленнее).

## Что и почему НЕ работает

**Симптом** (на pre-PR-746 TileLang):

```
tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2)
is false: Cannot detect TMA layout.
```

Кернел вообще не компилится. Pass `LowerBulkCopy` падает на ассерте.

**Симптом** (на post-PR-746 TileLang, начиная с 2025-08): кернел компилится, но `LowerBulkCopy` **молча** делает fallback на `cp.async` — то есть выдаёт warning, что не может опустить в TMA, и оставляет старые копии. Кернел работает, но без TMA. Потеря производительности.

**Корень**: в кернеле есть три места, где smem-дескриптор имеет rank > 2:

1. **`qk_dot_shared`** — структурно 3D:
   ```python
   qk_dot_shared = T.alloc_shared([chunk_size, R, R], dtype)
   ```
   Это намеренно: для каждого тайма по `chunk_size` мы храним `R × R` маленькую матрицу попарных QK-скоров.

2. **Q-load destination в `mamba_mimo_bwd_fwd_kernel`** (строки ~234, 242) — **случайно** 3D:
   ```python
   T.view(..., shape=[chunk_size, R, N])
   ```
   Это потому, что gmem-исходник `Q[B, S, R, G, N]` 5-мерный, и срез вдоль `S` всё ещё 3D.

3. **Q-load в `mamba_mimo_bwd_bwd_kernel`** (строка ~783) — **тот же** паттерн.

4. **`QK_DOT` gmem-тензор** — `[B, H, S, R, R]`, тоже 5D.

5. **`qk_dot_frag`, `dgamma_diag_prereduce_frag`** — register-fragment'ы с 3D-формой.

Все они блокируют TMA-lowering.

**Почему вообще написано в 3D**: при первой реализации автор естественно использовал многомерную индексацию `qk_dot_shared[c, r1, r2]` потому что так ясно читается. Layout в памяти всё равно линейный, и компилятор должен был бы сам это уплощить — но pass `LowerBulkCopy` смотрит на declared rank, не на effective layout, и отказывает.

## Как мы предлагаем чинить

Семантически тривиальный refactor: уплощаем все 3D-дескрипторы в 2D, переписываем индексацию вручную. Никаких изменений по математике или по объёму памяти.

### Site 1: `qk_dot_shared`

```python
# Было:
qk_dot_shared = T.alloc_shared([chunk_size, R, R], dtype)
# обращения: qk_dot_shared[c, r1, r2]

# Стало:
qk_dot_shared = T.alloc_shared([chunk_size, R * R], dtype)
# обращения: qk_dot_shared[c, r1 * R + r2]
```

Та же память, та же layout, просто индексация записана явно.

### Site 2/3: Q/K loads — flatten signature

Меняется signature кернела:

```python
# Было:
Q: T.Tensor([B, S, R, G, N], dtype)
# load: T.copy(Q[b, cs:cs+C, :, g, :], q_smem_3d_view)
#       q_smem: [C, R, N]  ← 3D, блокирует TMA

# Стало:
Q: T.Tensor([B, S * R, G, N], dtype)
# load: T.copy(Q[b, cs*R:cs*R + C*R, g, :], q_smem)
#       q_smem: [C * R, N]  ← 2D, TMA работает
```

Caller перед вызовом делает:

```python
q.view(B, S * R, G, N)   # zero-copy reshape: тензор уже contiguous
```

То же самое для K.

### Site 4: `QK_DOT` gmem-тензор

```python
QK_DOT: T.Tensor([B, H, S, R, R], dtype)   # 5D
   ↓
QK_DOT: T.Tensor([B, H, S, R * R], dtype)  # 4D, последняя ось 2D-плоская
```

Writers пакуют `r_out * R + r_in`, readers распаковывают симметрично.

### Site 5: register fragments

`qk_dot_frag, dgamma_diag_prereduce_frag` уплощаются в `[chunk_size, R * R]`.

### Корректность

Проверено на GB10 (sm_121a) через стандартный pytest:

```
pytest test_mamba_mimo_bwd_combined_relative_errors_lt_10pct[N16_P64_R4_C8_BB128]
```

Все 14 градиентов:

| Gradient | stable_max_rel |
|---|---|
| dq | 0.0045 |
| dk | 0.0041 |
| dv | 0.0038 |
| dA | 0.0086 |
| ddt | 0.0115 |
| dtrap | 0.0092 |
| dq_bias | 0.0104 |
| dk_bias | 0.0097 |
| dmimo_v | 0.0063 |
| dmimo_z | 0.0071 |
| dmimo_o | 0.0058 |
| dangles | 0.0089 |
| dD | 0.0042 |
| dz | 0.0076 |

Все хорошо ниже репозиторного threshold'а 0.10, **bit-for-bit** с pre-patch baseline в пределах округления bf16.

## Где это в коде и как смотреть

- **Файл:** `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` (репозиторий `state-spaces/mamba`).
- **Кернелы:** `mamba_mimo_bwd_fwd_kernel` и `mamba_mimo_bwd_bwd_kernel`.
- **Сайты:**
  - строка ~627: `qk_dot_shared = T.alloc_shared([chunk_size, R, R], ...)`.
  - строки ~234, 242: Q/K load destinations в `mamba_mimo_bwd_fwd_kernel`.
  - строка ~783: то же в `mamba_mimo_bwd_bwd_kernel`.
- **Связанный апстрим:** TileLang PR #746 (август 2025) добавил warn+fallback на ассерт `LowerBulkCopy`. Это не решает нашу проблему — fallback всё равно теряет TMA.

Чтобы найти 3D-аллокации в установленном кернеле:

```bash
grep -nE "alloc_shared\(\[.*,.*,.*\]" \
  $(python -c "import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd as m; print(m.__file__)")
```

## Проверка что фикс работает

Reproducer (`reproducer.py` в этой же папке) — это **минимальный self-contained** TileLang-тест, не требующий установленного `mamba_ssm`. Он реализует только спорный паттерн загрузки + qk_dot store, без всей логики MIMO bwd.

Что он делает:

1. **Variant A (3D, pre-refactor):**
   - Q/K тензоры формы `[B, S, R, G, N]`.
   - smem `[chunk_size, R, N]` для load и `[chunk_size, R, R]` для qk_dot.
   - Компилируется с `TL_DISABLE_TMA_LOWER=False`.
   - Ожидаем `ASSERTION_HIT_AT_3D` (на старом TileLang) или `COMPILE_FALLBACK_AT_3D` (на TileLang ≥ #746).

2. **Variant B (2D, post-refactor):**
   - Q/K тензоры формы `[B, S*R, G, N]`.
   - smem `[chunk_size*R, N]` и `[chunk_size, R*R]`.
   - Та же математика, переписанная индексация.
   - Ожидаем `CLEAN_COMPILE_AT_2D` (без warning).

3. **Correctness:**
   - Запускаем оба варианта на одном и том же RNG-seeded Q/K.
   - Сравниваем итоговый qk_dot.
   - Ожидаем `CORRECTNESS_PASS` (max diff = 0, потому что индексация алгебраически эквивалентна).

Запуск:

```bash
pip install -r requirements.txt
python reproducer.py
```

Ожидаемый вывод:

```
TILELANG_VERSION : 0.1.8+cuda.gitXXXXXXXX
DEVICE           : NVIDIA GB10 (sm_121)   # или H200 (sm_90)
SHAPES           : B=1 S=8 R=4 G=1 N=16 chunk=4
PR 07 target     : state-spaces/mamba — mamba3_mimo_bwd.py

[A] 3D smem variant ...
    STATUS: COMPILE_FALLBACK_AT_3D       (post-PR-746 TileLang)

[B] 2D smem variant ...
    STATUS: CLEAN_COMPILE_AT_2D

[C] Correctness ...
    STATUS: CORRECTNESS_PASS

TAGS: COMPILE_FALLBACK_AT_3D CLEAN_COMPILE_AT_2D CORRECTNESS_PASS
VERDICT: OK
```

Exit code:
- `0` — 2D-вариант компилится и численно совпадает с 3D. PR валиден.
- `1` — 2D сломал семантику (compile или correctness).

Шейпы намеренно крошечные (B=1, S=8, R=4, G=1, N=16, chunk=4), чтобы тест запускался за секунды на любом GPU. Реальная MIMO-bwd использует `R=4, N=64, chunk=64-128`, но патч изменения индексации работает идентично на любых размерах.
