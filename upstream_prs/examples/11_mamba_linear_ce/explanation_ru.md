# Подробное объяснение: MambaModel забыл LinearCrossEntropyModule (rebase-регрессия)

## Что это вообще и зачем

В Megatron-LM есть две основные модели: **GPTModel** (классический трансформер)
и **MambaModel** (гибридный, с Mamba state-space mixer'ами + attention слоями).
У обеих в конце стоит `output_layer` — линейный слой, который превращает
скрытое состояние в логиты по словарю.

Тут есть два класса с почти одинаковым интерфейсом:

- **`ColumnParallelLinear`** — обычный линейный слой, разрезанный по
  столбцам между TP-рангами. Считает логиты и возвращает их.
- **`LinearCrossEntropyModule`** — это **подкласс** `ColumnParallelLinear`,
  который умеет дополнительно: если ему передать `output_cross_entropy_loss=True`
  и `labels`, он вызовет fused linear CE kernel и вернёт сразу loss,
  ни разу не материализовав огромную матрицу `[seq, batch, vocab]` логитов.
  Никакого нового состояния (новых параметров) у него нет, отличается только
  `forward()`.

Зачем это нужно: на NAM56R с MBS=12 матрица логитов весит около 3.6 GiB,
плюс ещё столько же на её градиент в backward. Итого ~7 GiB лишней пиковой
памяти на каждый микробатч в pipeline slot. Это ровно та разница, которая
позволяет MBS=12 поместиться в 141 GiB H200 (или нет — будет OOM).

## Как это ДОЛЖНО было работать

1. У GPTModel в `__init__` стоит:
   `self.output_layer = LinearCrossEntropyModule(...)` — он использует
   правильный класс.
2. У MambaModel в `__init__` должно быть **то же самое**: создаётся
   `LinearCrossEntropyModule`, и в `forward()` есть условный вызов:
   если фьюзия включена — позвать `self.output_layer(output_cross_entropy_loss=True, ...)`,
   иначе — позвать `self.output_layer(...)` для логитов и потом
   обычный CE.
3. Когда пользователь включает `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear`,
   обе модели одинаково идут через fused путь.

## Что и почему НЕ работает

1. **Симптом**: на NAM56R MBS=12 на bench3 тренировка падает с OOM
   несмотря на флаг `--cross-entropy-loss-fusion`. На MBS=10 не падает,
   но через профайлер видно, что аллоцируется ~7 GiB на логиты — то есть
   фьюзия фактически не работает, флаг "молча" игнорируется для Mamba моделей.

2. **Что показывает reproducer**: строит минимальные `GPTModel` и
   `MambaModel`, печатает `type(output_layer)`:
   - `GPTModel.output_layer = LinearCrossEntropyModule` — правильно.
   - `MambaModel.output_layer = ColumnParallelLinear` — НЕ
     `LinearCrossEntropyModule`. Силент-баг.

3. **История регрессии** (любопытная и важная):
   - **PR #3226** "[DEV] Reapply fix Linear CE Fusion" смержен в `dev`
     2026-02-04 в 01:47 UTC. Этот PR правильно прописал
     `LinearCrossEntropyModule` в **обе** модели — и GPT, и Mamba,
     и добавил атрибут `self.fuse_linear_cross_entropy` в обе.
   - **PR #3207** "Reapply Add MTP support for hybrid models" смержен
     в `dev` 2026-02-04 в 22:40 UTC, через **21 час** после #3226.
     Этот PR делал rebase своих изменений MTP (Multi-Token Prediction)
     поверх **снепшота `dev` ДО** #3226. То есть в его diff
     `mamba_model.py` шёл "от старого", и при merge git тихо вернул
     `tensor_parallel.ColumnParallelLinear` обратно. GPT-сторону он
     не трогал, поэтому она осталась правильной.

4. **Почему никто не заметил**: тесты Megatron не проверяют
   `isinstance(mamba.output_layer, LinearCrossEntropyModule)`. Нет
   функционального теста на то, что Mamba действительно использует
   фьюзию. Тренировка просто молча идёт по нефьюженному пути и
   выглядит "рабочей".

5. **Корень**: rebase-конфликт между двумя PR, который слили без
   человеческого ревью на состояние Mamba. Никаких открытых PR,
   которые это исправляют, нет (проверено 2026-04-14 через
   `gh api repos/NVIDIA/Megatron-LM/pulls?state=open`).

## Как мы предлагаем чинить

**Upstream-фикс**: восстановить тот diff, который PR #3226 уже когда-то
приземлял. То есть в `megatron/core/models/mamba/mamba_model.py`:

1. Добавить `from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule`.
2. В `__init__` добавить:
   ```python
   self.fuse_linear_cross_entropy = (
       self.config.cross_entropy_loss_fusion
       and self.config.cross_entropy_fusion_impl == "linear"
   )
   ```
3. Заменить `tensor_parallel.ColumnParallelLinear(...)` на
   `LinearCrossEntropyModule(...)` при создании `self.output_layer`.
4. В `forward()` сделать как в `gpt_model.py`: если
   `self.fuse_linear_cross_entropy` — позвать fused путь, иначе старый.

**Наш runtime-workaround**: monkey-patch `MambaModel.__init__`, который
после создания модели делает:
```python
mamba.output_layer.__class__ = LinearCrossEntropyModule
mamba.fuse_linear_cross_entropy = True
```
Это работает потому, что `LinearCrossEntropyModule` — **чистый подкласс**
`ColumnParallelLinear`, без новых полей. Подмена `__class__` меняет
только то, какой `forward()` будет вызван. Никакой миграции состояния
не нужно.

Гейт: переменная окружения `CPPMEGA_MAIN_HEAD_LINEAR_CE=1`. Реализация
лежит в `cppmega/megatron/apply_linear_ce_patch.py`.

Edge-кейсы:
- Если upstream когда-то починит это сами, наш monkey-patch заметит,
  что класс уже правильный, и ничего не сделает (не сломается).
- MTP-голова (`self.mtp_process` ветка) тоже создаётся как
  `ColumnParallelLinear` — её аналогично нужно подменить, иначе
  MTP loss идёт по нефьюженному пути.

## Где это в коде и как смотреть

- Сломанный код:
  `megatron/core/models/mamba/mamba_model.py`, `__init__`, строка ~264:
  ```python
  self.output_layer = tensor_parallel.ColumnParallelLinear(
      config.hidden_size,
      self.vocab_size,
      ...
  )
  ```
- Правильный паттерн (для сравнения):
  `megatron/core/models/gpt/gpt_model.py`, строка ~251:
  ```python
  self.output_layer = LinearCrossEntropyModule(
      config.hidden_size,
      self.vocab_size,
      ...
  )
  ```
- Reproducer:
  `/Volumes/external/sources/cppmega/upstream_prs/examples/11_mamba_linear_ce/reproducer.py`.
- Наш monkey-patch:
  `/Volumes/external/sources/cppmega/cppmega/megatron/apply_linear_ce_patch.py`.

Запуск reproducer:
```bash
cd /Volumes/external/sources/cppmega/upstream_prs/examples/11_mamba_linear_ce
pip install -r requirements.txt
python reproducer.py
```

Reproducer инициализирует одно-процессную `torch.distributed` группу
(gloo на CPU, nccl на CUDA), строит обе модели, проверяет типы, применяет
monkey-patch и проверяет ещё раз.

## Проверка что фикс работает

**До фикса (текущий dev HEAD):**
```
Output-layer class check (before fix)
  GPTModel.output_layer    = ...LinearCrossEntropyModule
  MambaModel.output_layer  = ...ColumnParallelLinear

  assert isinstance(gpt,   LinearCrossEntropyModule)  -> True
  assert isinstance(mamba, LinearCrossEntropyModule)  -> False (BUG)

Applying proposed fix (class-swap, equivalent to restoring PR #3226)
  MambaModel.output_layer  = ...LinearCrossEntropyModule
  assert isinstance(mamba, LinearCrossEntropyModule)  -> True

VERDICT: regression CONFIRMED and fix VALIDATED.
```
Exit code: **0** (баг подтверждён, monkey-patch его лечит).

**После того, как upstream починит:**
```
[unexpected] MambaModel already uses LinearCrossEntropyModule.
Has PR #3207's regression been fixed upstream?
```
Exit code: **1** — это намеренный сигнал "перепроверь, что у тебя в дереве".

**Как понять что фикс работает в реальной тренировке**: запустить
NAM56R с `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear`
и `CPPMEGA_MAIN_HEAD_LINEAR_CE=1`. Через профайлер посмотреть пиковую
память: должна упасть на ~7 GiB на pipeline slot. На bench3 это
позволяет MBS=10 держать **269 TFLOP/s** (наш записанный рекорд),
а MBS=12 перестать падать в OOM (по крайней мере, по этой причине).
