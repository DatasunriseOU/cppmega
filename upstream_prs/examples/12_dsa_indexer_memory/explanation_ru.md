# Подробное объяснение: память DSA-индексатора в `_compute_index_scores`

## Что это вообще и зачем

**DSA** (DeepSeek Sparse Attention) — это вариант attention, придуманный для DeepSeek-V3.2. Идея простая: классический attention заставляет каждый query-токен смотреть на каждый key-токен. При длине последовательности 4096 это 4096×4096 = 16 миллионов пар на каждую голову. Очень дорого. DSA вместо этого вставляет лёгкий **indexer** — маленький "предварительный отборщик токенов", который для каждого query быстро выбирает только топ-K самых релевантных key-токенов. И уже основной attention считается только по ним.

Чтобы indexer мог решить "кого выбрать", ему нужны **scores** (оценки релевантности): для каждой пары (query, key) — число "насколько они связаны". По этим scores потом берётся `topk` (top-K самых высоких) и возвращается список индексов.

Scores нужно считать в **fp32** (32-битная плавающая точка), а не в bf16. Причина — стабильность softmax: маленькая разница между близкими по релевантности токенами в bf16 округлится до нуля, и индексатор просто наугад будет выбирать "первого попавшегося". Fp32 даёт нужное разрешение.

И вот тут начинается проблема. Indexer работает с многоголовыми (multi-head) тензорами: query имеет форму `[seqlen_q, batch, heads, dim]`, key — `[seqlen_k, batch, dim]`. Scores нужны на выходе формы `[batch, seqlen_q, seqlen_k]` — без размерности голов, потому что веса голов уже свёрнуты внутри. Промежуточно же приходится хранить `[seqlen_q, batch, heads, seqlen_k]` — отдельный слой для каждой головы.

## Как это ДОЛЖНО было работать

В upstream Megatron-LM функция `_compute_index_scores` написана через один большой `einsum`:

1. Берём query `q.float()` формы `[sq, b, h, d]` и key `k.float()` формы `[sk, b, d]`.
2. Делаем `torch.einsum('sbhd,tbd->sbht', q.float(), k.float())` — получаем тензор оценок `[sq, b, h, sk]` в fp32. Это матричное умножение query на key для каждой головы.
3. Применяем `relu` (отрицательные оценки обнуляем).
4. Умножаем на веса голов `weights.unsqueeze(-1)` формы `[sq, b, h, 1]`.
5. Суммируем по оси голов (`sum(dim=2)`) — получаем `[sq, b, sk]`.
6. Транспонируем в `[b, sq, sk]` — это и есть финальный score-тензор.

Логически всё чисто: одна команда — и готово.

## Что и почему НЕ работает

**Симптом**: на NAM56R (наша production-конфигурация) при `--micro-batch-size 10` тренировка падает с OOM (out of memory) на 8xH200, хотя в других местах шага свободно >40 GiB HBM. MBS=8 еле проходит. Видно, что что-то конкретно в DSA-индексаторе сжирает безумное количество памяти.

**Корень**: тот самый промежуточный тензор `[sq, b, h, sk]` в fp32. Считаем размер:

| Конфиг                | sq=sk | b  | h  | Размер промежутки |
| --------------------- | ----- | -- | -- | ----------------- |
| DeepSeek-V3.2 ref     | 4096  | 1  | 64 | ~4.0 GiB          |
| NAM56R DSA 9+4 MBS=8  | 4096  | 8  | 32 | **16.0 GiB**      |
| NAM56R MBS=10         | 4096  | 10 | 32 | **20.0 GiB**      |

То есть upstream-код выделяет 16 GiB просто чтобы немедленно свернуть это в 256 MiB суммой по головам. Этот тензор:

- Аллоцируется огромным куском.
- Используется ровно один раз (для редукции по головам).
- Сразу выкидывается.

Это классическая "**fuse-reduction opportunity**" — мы платим 16 GiB за то, что должно стоить ~268 MiB. И это блокирующий фактор для всего MBS=10 на H200, хотя другого свободного компьюта в шаге полно.

Дополнительно: `_compute_index_scores` дёргается из двух горячих мест — `fused_qk_topk_naive` (forward topk) и `fwd_fused_indexer_loss_naive` (forward indexer loss), плюс из backward recompute через `_LemyxFusedDSAIndexerLoss` и `IndexCache`. То есть память жрётся каждый forward + каждый recompute.

## Как мы предлагаем чинить

**Идея**: не материализовать полный `[sq, b, h, sk]`. Аккумулировать сразу в финальный `[b, sq, sk]` буфер, по одной голове за раз через `torch.bmm` (batched matrix multiply).

Пошагово:

1. Создаём output-буфер `index_scores` формы `[b, sq, sk]` в fp32 — это 268 MiB, которые мы и так должны будем держать как финальный результат.
2. Один раз готовим key в правильной форме: `k_bds = k.float().permute(1, 2, 0).contiguous()` — `[b, d, sk]`. Этот тензор переиспользуется для всех голов (~4 MiB на production shape).
3. В цикле по головам `for hi in range(h):`:
   - Берём query одной головы: `q_h = q[:, :, hi, :].float().permute(1, 0, 2).contiguous()` — `[b, sq, d]`.
   - Считаем `logits_h = torch.bmm(q_h, k_bds)` — `[b, sq, sk]` в fp32. Это один cuBLAS GEMM на голову.
   - `relu`.
   - Умножаем на вес головы и **прибавляем in-place** в аккумулятор: `index_scores.add_(logits_h * w_h)`.
4. Возвращаем `index_scores`.

Промежуточный `[sq, b, h, sk]` не существует никогда. Промежуточный `logits_h` живёт ровно одну итерацию цикла и переиспользует ту же память.

**Математика идентична** оригиналу с точностью до порядка ассоциативного сложения по головам в fp32 — измеренная относительная ошибка `max|a-b|/max(|a|,eps) = 1.9e-7` на production shape (на GB10), что на много порядков ниже любого разумного порога стабильности topk (например, 0.02). На малой shape с float64 — `rel_err = 0` бит-в-бит.

**FLOP count не меняется**: тот же объём арифметики, только разбит на `h` отдельных GEMM'ов вместо одного фьюзнутого einsum. Каждый GEMM ложится на cuBLAS, арифметическая интенсивность та же.

### Альтернативы

- **NVIDIA PR #4039** (Split-K Indexer Kernels) — пишет кастомные CUDA-ядра. Сложнее, требует build-system, в WIP. Наш патч работает сегодня без билда.
- **NVIDIA PR #2869** (Fused Indexer Loss Kernel) — целит в loss recompute, не трогает сам `_compute_index_scores`. Комплементарно, не пересекается.
- **TensorRT-LLM PR #12198** — сделал аналогичную оптимизацию для **inference**-стороны (другой модуль). Доказывает, что NVIDIA уже признаёт проблему. Наш PR закрывает training-сторону.

### Edge cases

- **`torch.compile` / capture-совместимость**: цикл `for hi in range(h)` Python-уровневый, разворачивается на трассировке. С CUDA Graph совместим (нет CPU-синхронизаций).
- **Backward**: gradcheck в float64 показывает `dq, dw, dk rel_err = 0` или `2.7e-16`. Полная autograd-парность с upstream.
- **`weights` от пользователя**: ожидается `[sq, b, h]` bf16 — конвертируется per-head через `weights[:, :, hi].float()`.

## Где это в коде и как смотреть

- **Upstream**: `megatron/core/transformer/experimental_attention_variant/dsa.py`, функция `_compute_index_scores`, строки 255–295. Конкретный einsum на строке 278 (по состоянию на main @ 2026-04-14).
- **Наш production-патч**: `cppmega/megatron/dsa_indexer_fused_patch.py::compute_index_scores_fused_bf16`. Применяется через `apply_dsa_indexer_fused_patch` (idempotent monkey-patch).
- **Env-kill-switch**: `CPPMEGA_DSA_INDEXER_FUSED=0` восстанавливает upstream-вариант.
- **Reproducer**: `/Volumes/external/sources/cppmega/upstream_prs/examples/12_dsa_indexer_memory/reproducer.py`
- **PR template**: `/Volumes/external/sources/cppmega/upstream_prs/12_megatron_dsa_compute_index_scores_memory.md`

Запуск reproducer:
```bash
cd upstream_prs/examples/12_dsa_indexer_memory
pip install -r requirements.txt
python reproducer.py               # auto: small + prod (+ full если HBM ≥ 40 GiB)
python reproducer.py --shapes prod # только production-shape
python reproducer.py --cpu         # CPU smoke (только small)
```

## Проверка что фикс работает

**Реальный вывод reproducer на GB10** (из `run_gb10.log`, torch 2.12 nightly):

Shape `small` (b=2, sq=256, sk=256, h=4, d=32):
```
expected upstream [sq,b,h,sk] fp32 intermediate: 2.0 MiB
upstream         peak_alloc=     36.2 MiB (delta vs inputs +36.0 MiB)
fused            peak_alloc=     34.3 MiB (delta vs inputs +1.6 MiB)
correctness: max rel_err = 0.000e+00
memory:      upstream 36.0 MiB -> fused 1.6 MiB   (saved 34.4 MiB, 22.1x)
PASS: correctness within 1e-4
```

Shape `prod` (b=4, sq=4096, sk=4096, h=8, d=128):
```
expected upstream [sq,b,h,sk] fp32 intermediate: 2048.0 MiB
upstream         peak_alloc=   4164.2 MiB (delta vs inputs +4096.0 MiB)
fused            peak_alloc=   1108.3 MiB (delta vs inputs +784.1 MiB)
correctness: max rel_err = 1.640e-07
memory:      upstream 4096.0 MiB -> fused 784.1 MiB   (saved 3311.9 MiB, 5.2x)
PASS: correctness within 1e-4
```

Gradcheck (float64, small):
```
upstream gradcheck: PASS
fused    gradcheck: PASS
fwd parity (double): rel_err = 0.000e+00
bwd parity: dq rel_err=0.000e+00  dw rel_err=0.000e+00  dk rel_err=2.706e-16
gradcheck overall: PASS
```

Ключевые цифры: на `prod` shape сэкономили 3.3 GiB (в 5.2 раза). На полной NAM56R MBS=8 (b=8, h=32) экономия растёт пропорционально — с ~16 GiB до ~600 MiB (≈26x).

**В реальной тренировке** на 8xH200: после применения патча `--micro-batch-size 10` для NAM56R DSA 9+4 запускается без OOM (до патча падало именно на этом einsum). End-to-end это даёт ~+5% throughput за счёт большего batch.
