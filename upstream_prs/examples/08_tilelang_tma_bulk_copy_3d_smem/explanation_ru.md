# Подробное объяснение: регрессионный тест на 3D shared-memory в TileLang `LowerBulkCopy`

## Что это вообще и зачем

**TileLang** — это Python DSL (язык, встроенный в Python) для написания CUDA-ядер высокого уровня. Вместо того чтобы вручную писать `__global__ void kernel(...)` на C++/CUDA, разработчик пишет Python-функцию с декоратором `@tilelang.jit`, а компилятор TileLang переводит её сначала в TVM TIR (промежуточное представление от Apache TVM), а затем в CUDA-код, который собирается через NVRTC/NVCC. TileLang используется для написания быстрых ядер для механизма внимания, Mamba и других linear-attention архитектур.

**TMA (Tensor Memory Accelerator)** — это аппаратный блок, появившийся в NVIDIA Hopper (H100/H200). Он умеет асинхронно копировать большие многомерные тайлы из глобальной памяти GPU (gmem) в shared memory (smem — быстрая память внутри SM). На уровне PTX-инструкций это `cp.async.bulk.tensor.{1,2,3,4,5}d`. Без TMA копирование делается через `cp.async`, которое нужно дробить на много параллельных потоков и ждать загрузки руками — TMA снимает с CUDA-cores эту работу и обычно даёт +20-30% к производительности.

**Compiler pipeline в TileLang** — это последовательность **passes** (преобразований). Каждый pass — это функция, которая берёт одну форму TIR и возвращает другую. Один из таких passes — `LowerBulkCopy` (находится в `src/op/copy.cc`). Его задача: взять высокоуровневый `T.copy(global_tensor, shared_tensor)` и решить, можно ли сгенерировать TMA-инструкцию, или нужно откатиться на обычный `cp.async`.

**LayoutInference** — другой pass, который выясняет, в каком виде каждый буфер живёт по потокам/варпам/тайлам. Здесь он не главный, но упоминается, потому что 08 и 13 — соседи и оба триггерятся на одном и том же ядре Mamba3.

**ICHECK vs LOG(WARNING)** — два уровня строгости в C++-логике компилятора:
- `ICHECK(condition)` — это аналог `assert`. Если условие ложно, процесс падает с `InternalError` и стеком вызовов. Никакого fallback.
- `LOG(WARNING)` — мягкое логирование. Печатает строку в stderr и продолжает работу. Часто после него идёт переход на медленный, но рабочий путь.

В этой багофиксе мы как раз и обсуждаем переход с `ICHECK` на `LOG(WARNING) + fallback`.

## Как это ДОЛЖНО было работать

Когда в TileLang-коде встречается `T.copy(X[0:16, :, :], xs)`, где `xs` — это `T.alloc_shared([16, 4, 64], "bfloat16")` (трёхмерный shared-memory тайл), компилятор должен:

1. Зайти в pass `LowerBulkCopy`.
2. Посмотреть на **shared layout descriptor** — структуру, описывающую, как тайл разложен в памяти. Метод `InputDim()` возвращает количество логических измерений smem-буфера (для нашего примера = 3).
3. Решить, может ли он эмитить настоящую TMA-инструкцию для данного количества измерений.
4. Если да — сгенерировать `cp.async.bulk.tensor.3d`. Если нет — мягко откатиться на `LowerNormalCopy`, который выдаст обычный `cp.async`. Главное: **компиляция должна продолжиться**.

## Что и почему НЕ работает

В версиях TileLang `≤ 0.1.7` (до коммита `5c11d245` от 2025-08-21) внутри `LowerBulkCopy` стояла такая проверка:

```cpp
ICHECK(shared_layout->InputDim() == 2) << "Cannot detect TMA layout.";
```

Это означало: TMA-эмиттер умеет работать **только** с двумерными smem-дескрипторами. Если кто-то аллоцирует трёхмерный (или больше) тайл, компиляция падает с `tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2) is false: Cannot detect TMA layout`.

Симптом → корень по шагам:
1. Mamba3 MIMO backward kernel аллоцирует `qk_dot_shared` как `[chunk_size, R, R]` — это **rank-3** smem.
2. Q/K также грузятся в `[chunk_size, R, N]` — снова **rank-3**.
3. Pass `LowerBulkCopy` доходит до этих копий, проверяет `InputDim() == 2`, видит `3`.
4. ICHECK срабатывает, компилятор валится с `InternalError`. Ядро вообще не собирается.
5. Пользователь не может включить TMA на bwd-ядрах Mamba3 → теряет 20-30% throughput на Hopper.

Корневая причина — слишком жёсткая инвариантная проверка: 2D-only TMA emitter существует, но 3D-emitter ещё не написан, а вместо корректного fallback стоит ICHECK, который убивает весь компилятор.

## Как мы предлагаем чинить

**Багу уже починили upstream.** TileLang PR [#746](https://github.com/tile-ai/tilelang/pull/746) (merged 2025-08-21, коммит `5c11d245`) заменил жёсткий ICHECK на мягкую проверку:

```cpp
if (shared_layout->InputDim() < 2) {
    LOG(WARNING) << "TMA bulk copy cannot support shared layout "
                 << "with input dimension N, fallback to normal copy.";
    return LowerNormalCopy(T, analyzer);
}
```

То есть для rank-3+ теперь:
- Печатается WARNING в stderr.
- Управление возвращается в `LowerNormalCopy`, который генерирует обычный (не TMA) `cp.async`.
- Компиляция продолжается. Ядро строится. TMA-ускорения на этой конкретной копии нет, но всё работает.

Полная поддержка 3D/4D/5D TMA-дескрипторов (где smem-формат сам становится `cp.async.bulk.tensor.3d`) — отдельная feature request, ещё не реализована. Но критическая блокирующая бага закрыта.

Поскольку фикс уже в upstream, наш PR — это **internal regression guard** (страховочный регрессионный тест). Если кто-нибудь когда-нибудь вернёт жёсткий ICHECK обратно (например, при рефакторинге `LowerBulkCopy`), наш reproducer мгновенно поймает это в CI.

## Где это в коде и как смотреть

- Upstream issue/тест template: `/Volumes/external/sources/cppmega/upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md`
- Reproducer: `/Volumes/external/sources/cppmega/upstream_prs/examples/08_tilelang_tma_bulk_copy_3d_smem/reproducer.py`
- README с инструкциями: `/Volumes/external/sources/cppmega/upstream_prs/examples/08_tilelang_tma_bulk_copy_3d_smem/README.md`
- Файл компилятора, где живёт исправление: `src/op/copy.cc` в репозитории `tile-ai/tilelang`, метод `CopyNode::LowerBulkCopy`.
- Связанные upstream PRs: PR #761 (1D TMA support, merged 2025-08-26), PR #2005 (1D TMA regression test от самого upstream, merged 2026-04-01).

Reproducer гоняет три кейса:
- **[A]** rank-3 smem `[16, 4, 64]` bfloat16 + `TL_DISABLE_TMA_LOWER=False`. Это бывший bug path. Должен компилироваться (с WARNING про fallback).
- **[B]** тот же rank-3 smem, но `TL_DISABLE_TMA_LOWER=True`. Sanity-check: путь без TMA всегда работал.
- **[C]** rank-2 smem `[16, 256]` + `TL_DISABLE_TMA_LOWER=False`. Это TMA fast-path, чтобы убедиться, что мы не сломали то, что и должно работать.

Чтобы перехватить C++-овый stderr (куда `LOG(WARNING)` пишет напрямую через C-runtime, минуя Python), reproducer делает trick через `os.dup2(2, ...)` — перенаправляет файловый дескриптор 2 на временный файл.

## Проверка что фикс работает

Запуск:

```bash
cd /Volumes/external/sources/cppmega/upstream_prs/examples/08_tilelang_tma_bulk_copy_3d_smem
pip install -r requirements.txt
python reproducer.py
```

Ожидаемый вывод (на TileLang `main` после PR #746):

```
[A] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False
    STATUS: OK — compile succeeded (warp-spec skipped (confirms 3D copy took non-TMA path))
[B] rank-3 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=True  (baseline)
    STATUS: OK
[C] rank-2 alloc_shared + T.copy, TL_DISABLE_TMA_LOWER=False (fast-path)
    STATUS: OK — rank-2 TMA fast-path compiled cleanly
========================================================================
VERDICT: OK. PR #746 warn+fallback behavior intact on tilelang 0.1.8+...
```

Exit code = `0`. Все три кейса собрались, регрессии нет.

Если регрессия случится (кто-то вернёт ICHECK), кейс [A] напечатает:

```
STATUS: REGRESSION — hard-assert on 'Cannot detect TMA layout'
```

и скрипт вернёт exit code `1`. На H200 в логе кейса [A] видна также строка `[WS] skipped: no TMA copies in pipeline loop` — это подтверждает, что TileLang действительно ушёл по fallback-пути и не сгенерировал TMA-копию (warp-specialization для pipeline-loop'а пропущена именно потому, что на rank-3 копии TMA не доступно).

Reproducer провалидирован 2026-04-14 на двух машинах: bench3 (H200 SXM, sm_90a) и GB10 (sm_121a). Оба вернули `OK` на трёх кейсах.
