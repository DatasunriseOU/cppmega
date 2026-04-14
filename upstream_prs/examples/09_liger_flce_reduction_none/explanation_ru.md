# Подробное объяснение: Liger FLCE с reduction="none" портит градиенты в backward

## Что это вообще и зачем

В конце любой языковой модели стоит "голова" (LM head), которая превращает скрытое
состояние в распределение вероятностей по словарю (vocab). Чтобы натренировать
модель, мы сравниваем это распределение с правильным токеном и считаем число —
"насколько модель ошиблась". Это число и называется **cross-entropy loss** (CE).
Чем ближе предсказание к истине, тем меньше loss.

**Fused linear CE** (FLCE) — это оптимизация: вместо двух шагов "сначала
посчитать огромную матрицу логитов `[batch * seq, vocab]`, потом из неё
посчитать loss" мы делаем один kernel, который считает loss напрямую и
никогда не материализует огромную матрицу логитов в памяти. Экономия — гигабайты.

Liger — библиотека от LinkedIn, в которой такой fused linear CE написан на
Triton. У её функции есть параметр `reduction`: `"mean"` (среднее по всем
токенам), `"sum"` (сумма) или `"none"` (вернуть отдельный loss на каждый
токен в виде вектора `[batch * seq]`). Megatron в своей "main head" обвязке
использует именно `reduction="none"`, потому что хочет сначала получить
per-token loss, потом помножить на маску (например, чтобы выкинуть padding
или служебные токены), и только потом просуммировать.

## Как это ДОЛЖНО было работать

1. На forward Liger получает hidden states `[BT, H]` и матрицу весов
   `[V, H]`, где `BT = batch * seq`, `H` — размер скрытого, `V` — размер
   словаря.
2. Внутри kernel логиты считаются по чанкам, чтобы не выделять память
   под полную матрицу `[BT, V]`.
3. При `reduction="none"` kernel должен вернуть тензор `[BT]` — отдельный
   loss на каждую позицию.
4. В backward пользователь делает `(loss * loss_mask).sum().backward()`.
   PyTorch autograd даёт kernel'у обратный градиент `grad_output` — это
   тоже тензор `[BT]`, в котором на каждой позиции стоит то значение,
   которое было в `loss_mask` (то есть разные числа для разных токенов).
5. Backward должен умножить заранее посчитанные градиенты по входу
   и весам на этот `grad_output` **поэлементно по строкам**: для строки
   i весь grad_input[i, :] и соответствующий вклад в grad_weight
   домножаются на `grad_output[i]`.

## Что и почему НЕ работает

1. **Симптом, который видим в продакшене**: на первой же итерации Megatron
   с Liger main-head выдаёт `grad_norm = NaN`. Optimizer step делает шаг
   на NaN-градиентах и портит веса. Вторая итерация падает с
   `CUDA illegal memory access` потому что в весах теперь NaN/Inf.

2. **Что показывает reproducer**: при `(loss * loss_mask).sum().backward()`
   расхождение grad_input с эталонной PyTorch реализацией —
   `4.7e-2`, а grad_weight — `2.5e-1`. Это огромная разница (для bf16
   нормальный шум — `5e-3`, то есть на два порядка меньше).

3. **Причина в коде**: в `fused_linear_cross_entropy_backward` стоит
   функция `element_mul_kernel`, которая написана в предположении что
   `grad_output` — это **скаляр** (одно число). Она читает только
   `grad_output[0]` (первый элемент!) и умножает на это число
   ВСЕ строки grad_input.

4. **Почему это не заметили раньше**: при обычном `loss.sum().backward()`
   autograd передаёт `grad_output = [1, 1, 1, ..., 1]` — все единицы.
   Прочитать первую единицу или любую другую — результат одинаковый.
   Тест "вроде прошёл", баг не виден.

5. **Когда баг вылезает**: как только маска неоднородная (например, у
   половины токенов вес 1.0, а у другой половины 0.5), `element_mul_kernel`
   читает `grad_output[0] = 1.0` и применяет это число ко всем строкам.
   В итоге половина строк, у которых маска должна была быть 0.5, получает
   неправильный градиент. Дальше chunked-аккумулятор для grad_weight
   получает мусор и довольно быстро переполняется в NaN.

6. **Корень**: Liger в forward уже посчитал и сохранил grad_input/grad_weight
   "как будто" пользователь потом разделит на `n_non_ignore` (как делают
   `reduction="mean"`). А в backward тупо домножает на якобы-скаляр
   `grad_output`. Для `reduction="none"` эта схема в принципе не работает —
   надо было либо хранить per-chunk grad_logits и пересобирать, либо
   запретить эту ветку. Liger не сделал ни того, ни другого.

## Как мы предлагаем чинить

Есть три варианта от простого к правильному:

**Вариант (a) — быстрая защита**: смержить открытый PR #1126. Он не чинит
функциональность, но добавляет `assert` в начало backward, который кричит
"reduction='none' backward not supported". Лучше громко упасть, чем
тихо испортить веса.

**Вариант (b) — настоящая починка**: переписать `element_mul_kernel`
так, чтобы он умножал grad_input построчно (`grad_input[i, :] *= grad_output[i]`),
а для grad_weight либо хранить per-chunk grad_logits в forward и пересобирать
в backward, либо честно повторить chunked matmul. Это +1 chunked GEMM по
времени в backward.

**Наш workaround в продакшене**: всегда зовём Liger с `reduction="mean"`,
потом домножаем результат на `n_valid` (количество не-игнорируемых токенов).
Математика: `sum_i CE_i == mean_i CE_i * n_valid`. Backward через
`mean_loss * n_valid` сокращает Liger'овский внутренний `1/n_valid` точно,
и градиент становится **бит-в-бит** равным эталонному
`F.cross_entropy(reduction="sum")`. Если caller хочет per-token тензор
`[b, s]`, мы делаем `loss_scalar.expand(b, s).contiguous()` — сумма
такого тензора с маской равна правильной сумме, когда маска совпадает
с `(target != ignore_index)` (это true в Megatron pretraining).

Edge-кейс: если маска внутри валидных токенов сама неоднородная (не просто
"игнор/не-игнор", а с разными весами), то наш broadcast — это уже
приближение, а не точная сумма. Для нашего пайплайна это не важно, для
других может быть.

## Где это в коде и как смотреть

- Сломанный код Liger:
  `src/liger_kernel/ops/fused_linear_cross_entropy.py`, функции
  `fused_linear_cross_entropy_forward` и `fused_linear_cross_entropy_backward`.
- Reproducer:
  `/Volumes/external/sources/cppmega/upstream_prs/examples/09_liger_flce_reduction_none/reproducer.py`.
- Наш production workaround:
  `cppmega/megatron/apply_linear_ce_patch.py`, функция `_install_liger_compute`.

Запуск reproducer:
```bash
cd /Volumes/external/sources/cppmega/upstream_prs/examples/09_liger_flce_reduction_none
pip install -r requirements.txt
python reproducer.py
```

Требует CUDA (Triton kernels работают только на GPU).

## Проверка что фикс работает

Reproducer печатает четыре строки:

1. `[OK] reduction="mean"` — должно быть OK всегда, это контрольная ветка.
2. `[OK] reduction="none" + .sum().backward()` — пройдёт случайно
   (uniform grad_output из единиц).
3. `[FAIL] reduction="none" + (loss*mask).sum()` — это и есть баг,
   `max|Δgrad_hidden|` около `4.7e-2`, `max|Δgrad_weight|` около `2.5e-1`.
4. `[OK] workaround: mean * n_valid` — наш workaround, должен дать
   расхождение в пределах bf16 шума `~5e-3`.

Внизу также печатается `grad_weight.norm()` — для сломанной ветки там
будет либо NaN, либо неадекватно большое число; для эталона и workaround —
нормальное значение порядка `1e1...1e2`.

Exit code:
- `0` — баг ИСПРАВЛЕН (все четыре ветки OK).
- `1` — баг есть (текущее состояние Liger 0.7.0).

После настоящей починки (вариант b) ветка 3 должна стать OK. Если применят
только assert (PR #1126), ветка 3 будет давать AssertionError, exit code
останется 1, и наш workaround всё равно нужен.
