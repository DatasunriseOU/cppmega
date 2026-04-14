# Подробное объяснение: divide-by-zero в `LayoutInference` при свёртке `FloorMod`

## Что это вообще и зачем

**TileLang** — Python DSL для CUDA-ядер; компилируется в TVM TIR (промежуточное представление от Apache TVM), затем в CUDA-исходник, затем в бинарь через NVRTC. Для Mamba3 и других linear-attention архитектур это сейчас основной инструмент написания быстрых ядер.

**TVM passes** — последовательные преобразования TIR. Каждый pass получает программу в одной форме и возвращает в другой. Релевантные в этой баге:
- **`LayoutInference`** — выясняет, как буферы (тензоры) разложены по варпам, регистрам и shared-memory; решает, кто из потоков читает какой элемент.
- **`Substitute`** — внутренний механизм замены символов (например, замена параметра `R` на конкретное число `4`).

**Const-folding (constant folding)** — оптимизация, при которой выражение из констант вычисляется в compile-time. Например, `2 + 3` сворачивается в `5`, а `8 % 4` — в `0`. Делает это семейство шаблонов `TryConstFold<Op>` в `src/arith/const_fold.h`. Для `FloorMod<a, b>` (т.е. `a % b`) свёртка проверяет: если оба операнда — целочисленные литералы, то можно вернуть готовый ответ. Но если делитель `b == 0`, это математически некорректно — отсюда защитный `ICHECK(pb->value != 0)`.

**`T.Parallel(...)`** — это TileLang-конструкция для параллельного цикла, который компилятор раскладывает на потоки/варпы. Тело цикла — вычисления по индексам. Когда внутри тела есть выражения вида `csr % R` или `csr // R` (где `csr` — индекс цикла, `R` — параметр), LayoutInference должен построить **iter-map** — формальное описание соответствия между координатами цикла и адресами буфера. Этот iter-map потом нормализуется через `NormalizeIterMapToExpr`.

**`FloorMod`** в TIR — это узел AST, представляющий операцию `a % b` (точнее, floor-modulo, чтобы корректно работать с отрицательными числами). Он отличается от `truncmod` тем, что результат всегда имеет тот же знак, что и делитель.

## Как это ДОЛЖНО было работать

Возьмём упрощённый код из Mamba3 `bwd_bwd` ядра:

```python
fused_chunk_size = chunk_size * R   # Python int, R = 4

for csr, n in T.Parallel(fused_chunk_size, N):
    q_frag[csr, n] += q_bias_frag[csr % R, n]
```

Здесь `R = 4` — обычная Python-константа, замкнутая в outer-функцию с `@tilelang.jit`. Когда TileLang компилирует это, должно произойти следующее:

1. Pass `LayoutInference` проходит по всем буферам, находит `q_frag` (фрагмент в регистрах) и `q_bias_frag`.
2. Для `q_bias_frag[csr % R, n]` он вызывает `infer_fragment_index`, который строит iter-map по индексу `csr % R`.
3. Iter-map нормализуется через `NormalizeIterMapToExpr` → `IterMapToExprNormalizer::ConvertIterSplitExpr` → `tvm::floormod(csr, 4)`.
4. На каком-то этапе аналайзер подставляет реальное значение `R = 4` и выражение становится конкретным.
5. Const-fold спокойно обрабатывает `csr % 4`, потому что делитель ≠ 0.
6. LayoutInference успешно завершается, ядро компилируется дальше.

## Что и почему НЕ работает

**Симптом:** При компиляции Mamba3 MIMO `bwd_bwd` ядра с включёнными TMA + warp-specialization (`TL_DISABLE_TMA_LOWER=False`, `TL_DISABLE_WARP_SPECIALIZED=False`) компилятор падает:

```
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
```

**Полный backtrace** (взят из reproducer'а):

```
tvm::tl::LayoutInferencer::Substitute
  → tvm::tl::BufferUseDefCollector::Run
  → tvm::tl::BufferUseDefCollector::FinishInferQueue
  → tvm::tl::BufferUseDefCollector::RunInferStep
  → tvm::tl::ParallelOpNode::InferLayout
  → tvm::tl::ParallelOpNode::CompleteBufferFragment
  → tvm::tl::Fragment::Fragment
  → tvm::tl::FragmentNode::FragmentNode
  → tvm::tl::infer_fragment_index
  → tvm::tl::MakeFlattenedExpression
  → tvm::arith::NormalizeIterMapToExpr
  → tvm::arith::IterMapToExprNormalizer::VisitExpr
  → tvm::arith::IterMapToExprNormalizer::ConvertIterSumExpr
  → tvm::arith::IterMapToExprNormalizer::ConvertIterSplitExpr
  → tvm::floormod(PrimExpr, PrimExpr, Span)
  → tvm::arith::TryConstFold<tvm::tir::FloorMod>
  → tvm::runtime::detail::LogFatal::Entry::Finalize
```

**Корень по шагам:**

1. Pass `LayoutInferencer::Substitute` начинает обходить ядро, заменяя символьные параметры на конкретные значения.
2. В `ParallelOpNode::CompleteBufferFragment` он строит фрагмент (расположение тензора по регистрам) для буфера `q_bias_frag`.
3. `infer_fragment_index` хочет получить «плоский» индекс в регистровый фрагмент, для этого вызывает `MakeFlattenedExpression`, которая нормализует iter-map.
4. Внутри нормализации `IterMapToExprNormalizer::ConvertIterSplitExpr` встречает выражение, требующее построить `floormod(some_expr, divisor_expr)`.
5. **Здесь происходит беда:** `divisor_expr` на этом промежуточном шаге уже подставлен — но не в `4` (реальное значение `R`), а в **`0`**. Это zero-init / транзитное состояние: символ ещё не до конца разрешён, но узел уже создан с placeholder-значением.
6. `tvm::floormod(...)` вызывает `TryConstFold<FloorMod>`, который видит `b->value == 0`.
7. ICHECK в `TryConstFold<FloorMod>` срабатывает: `Check failed: pb->value != 0 (0 vs. 0) : Divide by zero` → `LogFatal` → процесс падает.

То есть: компилятор сам себе подставил 0 на промежуточном этапе нормализации, не дал substitution-машинерии завершить работу, и упал на жёстком ICHECK вместо того, чтобы вернуть «пока нельзя свернуть, оставь символьно».

**Важные уточнения:**
- Минимальный standalone-пример с `csr % R` внутри `T.Parallel` НЕ воспроизводит баг. Нужен полный buffer/fragment-граф `bwd_bwd` ядра — только при достаточной структурной сложности iter-map normalization идёт по той ветке, где транзитный 0 успевает попасть в `TryConstFold`.
- Алгебраический workaround `csr - (csr // R) * R` НЕ работает. `RewriteSimplifier` (другой TVM-pass) каноникализирует это обратно в `FloorMod` ещё до того, как `LayoutInference` начнёт работу.
- CUDA не нужна для воспроизведения. `LayoutInference` — это host-side pass, который запускается ещё до codegen. Достаточно `tilelang` + `mamba_ssm`.

## Как мы предлагаем чинить

Три кандидата (от наименее до наиболее инвазивного):

**(1) Guard `TryConstFold<FloorMod>` на zero modulus.** В файле `src/arith/const_fold.h` поменять поведение: если делитель — литеральный 0, возвращать `NullOpt` (т.е. «не могу свернуть, оставь как есть») вместо `LogFatal`. Это самое локальное изменение. Логика: если на текущем этапе делитель транзитно равен 0, значит substitution ещё не закончилась; следующий pass подставит реальное значение и свёртка случится потом.

**(2) Сохранять символический FloorMod до фиксации делителя.** В `src/arith/iter_affine_map.cc`, в методе `IterMapToExprNormalizer::ConvertIterSplitExpr`, не строить узел `floormod(a, b)` пока аналайзер не закрепил `b` за конкретным ненулевым PrimExpr. Архитектурно более правильное решение, но требует понимания, как iter-map относится к остальным normalize-шагам.

**(3) Soft-fallback в `ParallelOpNode::InferLayout`.** Если внутри fragment-индекса встречается FloorMod с символическим-но-сейчас-нулевым делителем, не упасть, а откатиться: использовать shared-memory layout вместо register-fragment layout (с warning'ом). Это safety net, независимый от арифметического фикса.

Опция (1) — самая быстрая для merge, опция (2) — самая правильная архитектурно, опция (3) — резервный план. PR #1458 в TileLang (merged 2025-12) фиксил **другой** FloorMod site в Z3-prover'е и нашу проблему не покрывает.

## Где это в коде и как смотреть

- Описание issue: `/Volumes/external/sources/cppmega/upstream_prs/13_tilelang_floormod_layout_inference_dbz.md`
- Reproducer: `/Volumes/external/sources/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz/reproducer.py`
- README: `/Volumes/external/sources/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz/README.md`
- Patch для Mamba3: `/Volumes/external/sources/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_layout_fix.patch` — flатит rank-3 smem-операнды до rank-2, чтобы соседняя бага из PR #08 (`LowerBulkCopy InputDim==2`) не выстрелила первой.
- Файлы в TileLang/TVM, где живёт корень:
  - `src/arith/const_fold.h` — шаблон `TryConstFold<FloorMod>`, тот самый ICHECK на `pb->value != 0`.
  - `src/arith/iter_affine_map.cc` — `IterMapToExprNormalizer::ConvertIterSplitExpr`, который вызывает `tvm::floormod`.
  - `src/transform/layout_inference.cc` — `ParallelOpNode::InferLayout`, верхушка стека.
- Внутренние заметки: `reference_tma_layout_fix_broken_h200.md` (memory-note про этот crash) и `reference_p1_blocked_tilelang_tma_layout.md` (как это блокирует Mamba3 P1-оптимизацию).

Reproducer работает так:
1. Находит установленный `mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd`.
2. Копирует исходник в `/tmp/`, применяет `mamba3_bwd_layout_fix.patch` (это снимает блокер 08).
3. В скопированном файле меняет `TL_DISABLE_TMA_LOWER: True → False` и `TL_DISABLE_WARP_SPECIALIZED: True → False`.
4. Импортирует пропатченный модуль и зовёт `mamba_mimo_bwd_bwd(B=1, S=64, H=4, G=1, N=64, P=64, R=4, chunk_size=16, ...)` — крошечная NAM56R-совместимая форма.
5. TileLang начинает компилировать → доходит до LayoutInference → падает с InternalError → reproducer ловит исключение, печатает backtrace и `TILELANG_BUG_REPRODUCED`.

## Проверка что фикс работает

Запуск (на любой машине с `tilelang` + `mamba_ssm`, CUDA не обязательна):

```bash
cd /Volumes/external/sources/cppmega/upstream_prs/examples/13_tilelang_floormod_dbz
pip install -r requirements.txt
python reproducer.py
```

**Текущее поведение (баг есть)** — exit code `1`:

```
Compiling mamba_mimo_bwd_bwd at B=1 S=64 H=4 G=1 N=64 P=64 R=4 (TMA+WS = ON)...
CRASH: InternalError
Check failed: pb->value != 0 (0 vs. 0) : Divide by zero

Traceback (most recent call last):
  File "<unknown>", line 0, in tvm::tl::LayoutInferencer::Substitute(...)
  File "<unknown>", line 0, in tvm::tl::ParallelOpNode::InferLayout(...)
  ...
  File "<unknown>", line 0, in tvm::arith::TryConstFold<tvm::tir::FloorMod>(...)
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero

TILELANG_BUG_REPRODUCED: LayoutInference FloorMod divide-by-zero
```

**После фикса** — exit code `0`:

```
OK: compiled cleanly (CUDA source NNNNN chars).
TILELANG_BUG_NOT_REPRODUCED
```

Reproducer верифицирован 2026-04-14 на bench3 (H200 SXM, sm_90a) против `tilelang 0.1.8+cuda.gitf309d814` (upstream main commit `f309d814`). Если фикс попадёт в любую из трёх предложенных точек (`TryConstFold<FloorMod>`, `ConvertIterSplitExpr`, или `ParallelOpNode::InferLayout`), reproducer перестанет ловить crash и напечатает `TILELANG_BUG_NOT_REPRODUCED` — это и есть критерий приёмки PR.

Текущий workaround для production: держать `TL_DISABLE_TMA_LOWER: True` на всех bwd-ядрах. Это стоит ~20% throughput на H200, но позволяет ядру компилироваться. После фикса этот флаг можно будет снять.
