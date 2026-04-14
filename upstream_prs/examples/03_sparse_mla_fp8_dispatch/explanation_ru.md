# Подробное объяснение: FP8-диспатч для SparseMLA при FP8-обучении

## Что это вообще и зачем

**FP8** (Float 8-bit) — это формат чисел в 1 байт вместо стандартных 2 байт BF16 (bfloat16, "урезанный" 16-битный float от Google) или 4 байт FP32. На современных GPU (Hopper H100/H200, Blackwell B200) есть аппаратная поддержка FP8 в tensor cores: matrix multiply (GEMM) в FP8 идёт **в 2 раза быстрее** чем в BF16, и тратит **в 2 раза меньше shared memory** (smem — быстрая память внутри SM-блока GPU). Поэтому хочется обучать модели в FP8 чтобы сэкономить время и память.

Проблема FP8 в том, что у него очень узкий диапазон значений (E4M3 формат держит примерно ±448, E5M2 — больше но с меньшей точностью). Чтобы тензор bf16 с произвольным диапазоном превратить в FP8 без потери осмысленности, нужно его **отскейлить**: найти максимум `amax = max|x|`, поделить весь тензор на `amax/448`, сохранить результат в uint8 (8 бит), и где-то рядом запомнить scale. При обратной операции (dequantize) умножаем обратно. Без scale FP8 не имеет смысла.

**Float8Tensor / QuantizedTensor** — это специальная Python-обёртка от NVIDIA Transformer Engine (TE), которая держит вместе:
* `_data` — реальные FP8-байты (хранятся как `torch.uint8`, потому что PyTorch не имеет нативного fp8-dtype во всех версиях),
* `_scale_inv` / `_amax` — метаданные для квантизации,
* и поверх всего этого притворяется обычным `torch.Tensor` с `dtype=bfloat16` (логический dtype который видит пользовательский код).

Это нужно потому что Megatron-LM передаёт между слоями обычные тензоры; если бы TE возвращал raw uint8, ничего бы не работало (forward-цепочка ожидает bf16-подобные тензоры). Float8Tensor — это удобный wrapper для прозрачной интеграции FP8 в существующий код.

**Raw CUDA kernel** (например наш TileLang SparseMLA) не знает про этот wrapper. Он просто берёт `tensor.data_ptr()` (адрес байтов в global memory — gmem, основная видеопамять GPU) и читает оттуда. Если ему передать Float8Tensor, начинаются проблемы.

## Как это ДОЛЖНО было работать

При FP8-обучении (`--fp8-format hybrid --fp8-recipe tensorwise`) Transformer Engine оборачивает выходы линейных слоёв в `Float8Tensor`. Эти тензоры попадают на вход attention. Идеальный сценарий:

1. Dispatcher SparseMLA (`_fused_sparse_mla_absorbed` в `dsa.py`) проверяет: `isinstance(query, QuantizedTensor)`?
2. Если да — вызывает FP8-вариант ядра (`SparseMLA_FP8`), которое внутри использует `T.float8_e4m3fn` dtype в GEMM-ах TileLang. Это даёт 2× throughput на Hopper WGMMA (Warp Group Matrix Multiply Accumulate — Hopper-инструкция для матричного умножения целым варп-группой).
3. Если нет — вызывает BF16-вариант (`SparseMLA`) как обычно.
4. Внутри FP8-ядра данные читаются из `Float8Tensor._data` (uint8) с использованием metadata-scale, GEMM идёт нативно в FP8, результат конвертируется обратно в bf16 для следующего слоя.

Тогда пользователь получает то что ожидал: попросил FP8 — получил 2× speedup и 50% экономию shared memory.

## Что и почему НЕ работает

Upstream `_fused_sparse_mla_absorbed` НЕ проверяет isinstance, а сразу передаёт `query`/`key` в ядро. Float8Tensor имеет несколько неприятных свойств:

* **`.data_ptr()` возвращает 0 (NULL)** — реальный указатель на байты лежит в `._data.data_ptr()`. Это потому что Float8Tensor — это Python-обёртка без собственного storage; данные физически в `_data`.
* **`.dtype` врёт** — возвращает `torch.bfloat16` (логический dtype который видит внешний код), хотя реальное хранение `_data.dtype == torch.uint8`.
* **`.contiguous()`, `.to()`, `.reshape()` НЕ разворачивают wrapper** — они возвращают тот же Float8Tensor. Это контринтуитивно: `.to(torch.bfloat16)` НЕ конвертирует тензор!
* **Только `.dequantize()`, `.float()`, `.permute()`, `.unsqueeze()` реально разворачивают** wrapper в обычный `torch.Tensor`.

**Headline-симптом из upstream-док:** `RuntimeError: kernel main input Q data pointer expected non-NULL, but got NULL`. Ядро пытается прочитать `q.data_ptr()`, получает 0, валится.

**ВАЖНЫЙ хедж про TE 2.13:** в новых версиях Transformer Engine добавлен `__torch_dispatch__` hook — это механизм PyTorch который позволяет перехватывать ВСЕ операции с тензором. TE использует его чтобы автоматически вызывать `.dequantize()` когда Float8Tensor попадает в чужой CUDA kernel (на границе torch.Tensor → C++ extension). На текущем стеке (TE 2.13) headline-крэш с NULL-указателем **НЕ срабатывает** — вместо краша происходит тихая (silent) auto-dequantize до bf16, и raw-ядро спокойно работает на bf16-тензоре.

Это даже хуже чем явный краш: пользователь попросил FP8 (заплатил за подготовку scale, метаданные и т.д.), но реально получает bf16-исполнение. Расход памяти удваивается, FP8-speedup теряется, и **никаких ошибок в логе нет**. Reproducer показывает это явно: `BF16 kernel on Float8Tensor -> finite=True (silent auto-dequant; lost FP8 speedup)`.

**Корень:** dispatcher не различает `Float8Tensor` от `bfloat16`-тензора, потому что `.dtype` врёт. Без явной `isinstance(QuantizedTensor)` проверки невозможно знать что внутри FP8-данные.

## Как мы предлагаем чинить

**Главный фикс — explicit dispatch:**

```python
_use_fp8_mla = False
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor
    if isinstance(query, QuantizedTensor) or isinstance(key, QuantizedTensor):
        _use_fp8_mla = True
except ImportError:
    pass

if _use_fp8_mla:
    _mla_fn = SparseMLA_FP8   # FP8-вариант (2x throughput на Hopper WGMMA)
else:
    _mla_fn = SparseMLA       # BF16-вариант
```

`SparseMLA_FP8` — это наша версия autograd-функции, которая внутри использует `T.float8_e4m3fn` dtype в TileLang GEMM-ах.

**Идеальная zero-copy реализация:** хотим взять `Float8Tensor._data` (который uint8) и интерпретировать его как FP8 без копирования. В нашем коде есть helper `_extract_fp8_data` который пытается это сделать:

```python
def _extract_fp8_data(t):
    if hasattr(t, "_data") and t._data.dtype == torch.float8_e4m3fn:
        return t._data, t._scale_inv
    return None
```

**Хедж про TE 2.13:** в этой версии `Float8Tensor._data.dtype == torch.uint8`, НЕ `torch.float8_e4m3fn`. Поэтому наш `_extract_fp8_data` возвращает `None`, и SparseMLA_FP8 падает в fallback-путь: `dequantize → ядро квантизует обратно внутри своих GEMM-ов`. Это корректно (получаем правильный численный результат), но **НЕ zero-copy** — теряем потенциальный speedup от прямого чтения uint8-байтов как FP8. Это отдельная оптимизация которую можно сделать в будущем когда TE начнёт хранить `_data.dtype == torch.float8_e4m3fn`.

**Альтернативы которые мы НЕ выбрали:**

* **Простой `dequantize()` в dispatcher** (`if isinstance(q, QuantizedTensor): q = q.dequantize()`) — работает и даёт корректность, но полностью убивает FP8-speedup. Это запасной вариант для моделей у которых нет `SparseMLA_FP8`.
* **Полагаться на `__torch_dispatch__` auto-dequant** — то что происходит сейчас в TE 2.13 без нашего фикса. Корректно, но silent loss of FP8 speedup и расход 2× memory bandwidth (память между gmem и smem — главное узкое место Hopper).

## Где это в коде и как смотреть

Файлы:

* `/Volumes/external/sources/cppmega/cppmega/megatron/sparse_mla_ops/sparse_mla.py` — содержит `_unwrap_quantized` helper и `SparseMLA_FP8` autograd-функцию.
* `/Volumes/external/sources/cppmega/cppmega/megatron/upstream_patches/apply_dsa_cg_patches.py` — Patch 9 инжектит `isinstance(query, QuantizedTensor)`-диспатч в upstream `dsa.py` во время bootstrap. Patch 9b убирает лишний `.dequantize()` round-trip который остался от предыдущего drift.
* `megatron/core/transformer/experimental_attention_variant/dsa.py` — функция `_fused_sparse_mla_absorbed` (upstream-target).

Reproducer показывает три сценария бок о бок:

* `/Volumes/external/sources/cppmega/upstream_prs/examples/03_sparse_mla_fp8_dispatch/reproducer.py`

## Проверка что фикс работает

Запустить reproducer на машине с Hopper (H100/H200), Ada (sm_89), Blackwell (B200) или GB10 (sm_121a):

```bash
cd /path/to/cppmega
CUDA_VISIBLE_DEVICES=0 python upstream_prs/examples/03_sparse_mla_fp8_dispatch/reproducer.py
```

Ожидаемый вывод на bench3 (H200, TE 2.13):

```
=== Scenario A: upstream raw-dispatch with Float8Tensor (expect BUG) ===
  Float8Tensor.data_ptr() = q:0  kv:0  (0 == NULL)
  Float8Tensor.dtype reports: torch.bfloat16 (lies about storage)
  Float8Tensor._data.dtype  = torch.uint8 (real storage)
  dispatch hazard: looks-like-bf16=True actually-fp8=True
                   => isinstance(QuantizedTensor) check is MANDATORY
  BF16 kernel on Float8Tensor -> finite=True
                   (silent auto-dequant; lost FP8 speedup)
  [BUG_REPRODUCED] Float8Tensor hazards confirmed

=== Scenario B: dequantize() preprocess fix (expect FIX_VALIDATED) ===
  max|fix_out - bf16_ref| = 2.441e-03  (tol=5e-02)
  [FIX_VALIDATED] dequantize() preprocess path

=== Scenario C: SparseMLA_FP8 dispatch fix (expect FIX_VALIDATED) ===
  max|fp8_out - bf16_ref|  = 3.174e-03  (tol=1e-01)
  [FIX_VALIDATED] SparseMLA_FP8 dispatch path
```

Реальные числовые расхождения:
* **Scenario B** (dequantize → BF16-ядро): `2.441e-03` — близко к pure-BF16 reference, дельта только от FP8 round-trip ошибки квантизации (E4M3 имеет ~7-битную мантиссу).
* **Scenario C** (SparseMLA_FP8 dispatch): `3.174e-03` — чуть больше из-за дополнительной FP8 ошибки внутри Q@K GEMM, но всё ещё в пределах толерантности `1e-1`. Это нормально для FP8.

Exit-коды:
* `0` — все три сценария ведут себя как ожидается (баг воспроизводится, оба фикса работают).
* `1` — какой-то сценарий упал.
* `2` — нет CUDA или FP8-железа.

Production-проверка: запустить training smoke с `--fp8-format hybrid --fp8-recipe tensorwise`, проверить что в логах нет `data pointer expected non-NULL` и что MFU/throughput действительно выше чем на чистом BF16.
