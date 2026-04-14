# Подробное объяснение: Megatron fused linear CE крашится на Hopper (H100/H200)

## Что это вообще и зачем

В Megatron-LM есть встроенный **fused linear cross-entropy** kernel — это
объединённая операция "матричное умножение hidden × output_weight + cross-entropy
loss" в одном Triton/CuTe kernel. Зачем нужна такая фьюзия: финальная матрица
логитов имеет форму `[seq * batch, vocab_size]`. У NAM56R это
`12288 * 151552 * 2 байта ≈ 3.6 GiB` на один микробатч. Плюс ещё столько же
на её градиент в backward. Итого ~7 GiB лишней пиковой памяти, которая
может стать разницей между "помещается MBS=12" и "OOM".

Fused kernel считает loss напрямую, никогда не выделяя `[BT, V]` тензор.
Включается флагами `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear`.

**Compute capability (cc)** — это версия архитектуры NVIDIA GPU. Hopper
(H100/H200) — это `cc = (9, 0)`. Blackwell (B200) — это `cc = (10, 0)`.
GB10 — `cc = (12, 1)`. A100 — `cc = (8, 0)`. Это просто номера поколений
железа; разные kernels компилируются под разные cc.

## Как это ДОЛЖНО было работать

1. Пользователь включает в конфиге `cross_entropy_loss_fusion=True` и
   `cross_entropy_fusion_impl="linear"`.
2. Megatron при первом forward создаёт singleton `Platform`, который
   определяет cc устройства и подгружает kernel под эту cc.
3. На любом современном GPU (H100/H200, B200, GB10) kernel должен
   подгрузиться или хотя бы аккуратно деградировать в нефьюженную
   реализацию с предупреждением.
4. Forward возвращает loss без материализации логитов; backward
   считает градиенты через тот же kernel; пользователь экономит ~7 GiB.
   Но для этого pack'а нельзя ссылаться на старый bench3 `269.4 TFLOP/s`
   как на текущее доказательство: repo source-of-truth уже пометил его как
   superseded, а filing-ready retained H200 receipt для pack 10 пока не приложен.

## Что и почему НЕ работает

1. **Симптом**: на H200 с включённой фьюзией первый же шаг тренировки
   падает с `ValueError: Unsupported architecture: 9`.

2. **Где именно**: файл `megatron/core/fusions/fused_linear_cross_entropy.py`,
   класс `Platform`, метод `__init__` (примерно строки 28-43, ветка `dev`):

   ```python
   if cc[0] == 10:
       from .linear_cross_entropy.blackwell import entry as gpu_entry
       self.forward_func = gpu_entry.forward
       self.backward_func = gpu_entry.backward
   else:
       raise ValueError(f"Unsupported architecture: {cc[0]}")
   ```

3. **Почему так сделано**: первая реализация (PR #2206) была написана
   только под Blackwell (cc=10), и автор просто залил её с явным
   `else: raise`. То есть **любое** другое железо (H100, H200, A100, L40,
   GB10) идёт в `else` и крашится. Никакого fallback на нефьюженный
   путь нет — флаг "включить фьюзию" работает как мина.

4. **Почему ещё не починили**: открытый PR #3345 (`feat/hopper-kernels`,
   автор JungHoyoun) добавляет ветку `cc[0] == 9` с настоящими CuTe-DSL
   WGMMA kernels под Hopper. PR mergeable, ревью идёт, но ещё не
   слит на момент 2026-04-14. Никаких других открытых PR, которые
   адресуют другие cc, нет.

5. **Корень**: разработчики native Blackwell kernel сделали "Blackwell
   only без оговорок" вместо корректной деградации. PR #3345 чинит
   только Hopper, а не общий принцип "если нет нативного kernel,
   откатись на унифицированный fallback".

## Как мы предлагаем чинить

Двухэтапный подход:

**Tier A — смержить PR #3345 как есть.** Это сразу разблокирует H100/H200
(подавляющее большинство пользователей NeMo/Megatron). Локально мы уже
живём на дереве, где #3345 cherry-pick'нут, так что текущая bench3/europe
проверка подтверждает только то, что **patched Hopper path загружается и
работает**. Это полезная post-fix валидация, но не retained pre-fix receipt,
и не повод тащить в этот pack старые `269.4 TFLOP/s` claims как будто они
являются его актуальным filing evidence.

**Tier B — добавить мягкий fallback для всех остальных cc.** Минимальный
патч: вместо `raise ValueError` в `else` ветке выдать `RuntimeWarning`
и подгрузить уже существующий нефьюженный путь
`fused_cross_entropy.fused_vocab_parallel_cross_entropy`. То есть пользователь
получает варнинг "под твоё железо нет нативного kernel, использую
референсный путь", но тренировка не падает. Это ~40 строк кода и
правильно по контракту флага: "включить фьюзию" должно делать всё что
можно, а не превращаться в landmine.

**Наш local workaround** до того, как любой из tier'ов пройдёт upstream:
файл `cppmega/megatron/apply_linear_ce_patch.py` пробует вызвать
`_get_platform()`, ловит `ValueError`, и подменяет `LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss`
на путь через Liger kernel (или Apple CCE). Это даёт нам экономию памяти
и почти то же ускорение на H200 без зависимости от upstream.

Edge-кейсы:
- На GB10 (cc=12) даже PR #3345 не помогает, нужен либо Tier B, либо
  наш Liger reroute.
- Tier B fallback материализует логиты — это медленнее и требует памяти,
  но **корректно**, что лучше чем падение.

## Где это в коде и как смотреть

- Сломанный диспатчер upstream:
  `megatron/core/fusions/fused_linear_cross_entropy.py`, класс `Platform`.
- Открытый upstream fix: PR #3345 на `NVIDIA/Megatron-LM`.
- Reproducer:
  `/Volumes/external/sources/cppmega/upstream_prs/examples/10_megatron_flce_hopper/reproducer.py`.
- Наш local workaround:
  `/Volumes/external/sources/cppmega/cppmega/megatron/apply_linear_ce_patch.py`.

Чтобы запустить reproducer на H200 (например, bench3):
```bash
source /mnt/data/venv/bin/activate
cd /mnt/data/cppmega-root/cppmega/upstream_prs/examples/10_megatron_flce_hopper
python reproducer.py
```

Если у тебя только macOS, скрипт всё равно запустится — он напечатает
окружение и выйдет с кодом 77 ("SKIP: no CUDA"). Это нормально.

## Проверка что фикс работает

Reproducer ведёт себя по-разному в зависимости от того, что в дереве:

**До фикса (текущее состояние dev branch на H200):**
```
[env] cuda.cc=(9, 0)  device='NVIDIA H200'
[reproducer] calling _get_platform() on cc=(9, 0) ...
[reproducer] caught ValueError: 'Unsupported architecture: 9'
[ok] BUG REPRODUCED on cc=(9, 0).
```
Exit code: **0** (баг воспроизведён как ожидалось — это успех reproducer'а,
не успех самого Megatron).

**После того, как PR #3345 смерджен:**
```
[reproducer] calling _get_platform() on cc=(9, 0) ...
[ok] cc=(9, 0) (Hopper) — native Hopper entry loaded.
This means PR #3345 has been applied to this tree.
```
Exit code: **0**.

**На Blackwell B200 (контроль):**
```
[ok] cc=(10, 0) (Blackwell) — native Blackwell entry loaded (expected).
```
Exit code: **0**.

**На macOS / без CUDA:**
```
[skip] no CUDA device; dispatcher asserts CUDA availability first.
```
Exit code: **77**.

Главный признак того, что фикс заработал: после PR #3345 на H200
твоя реальная NAM56R тренировка с `--cross-entropy-loss-fusion
--cross-entropy-fusion-impl linear` запускается без `ValueError`.

Но для этого конкретного pack'а filing-ready proof сильнее: нужен retained
H200 log, который явно показывает либо pre-fix `Unsupported architecture: 9`,
либо post-fix successful load/run в корректно описанном patched tree. Пока
такого retained receipt нет, pack 10 остаётся `Ready: N`.
