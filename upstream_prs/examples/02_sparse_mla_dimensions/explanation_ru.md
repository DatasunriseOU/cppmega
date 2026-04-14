# Подробное объяснение: Параметризация размерностей в TileLang-ядре SparseMLA

## Что это вообще и зачем

**Attention** — это базовая операция трансформера: каждый токен "смотрит" на все остальные токены через матрицу `softmax(Q @ K^T) @ V`. Для длинных последовательностей это очень дорого по памяти и compute, поэтому DeepSeek придумали **MLA** (Multi-head Latent Attention) — вариант, где ключи и значения сжимаются в маленький "латентный" вектор размерности `kv_lora_rank`, что резко снижает память на KV-cache.

**SparseMLA** — это дальнейшее улучшение: вместо того чтобы каждый query-токен смотрел на ВСЕ key-токены, мы заранее выбираем для него только `topk` (например, 64) самых релевантных через отдельный "indexer". То есть `Q` смотрит не на `seq_len` ключей, а только на `topk` из них. Это даёт квадратичную экономию (`O(seq * topk)` вместо `O(seq * seq)`).

**TileLang** — это DSL (язык), в котором пишут CUDA-ядра под GPU. Ядро (kernel) — это функция, выполняющаяся параллельно тысячами потоков на GPU. **Tile** — это маленький блок данных (например, 64x64 элементов), который ядро обрабатывает за раз внутри shared memory (smem) GPU. Хорошо написанное TileLang-ядро для SparseMLA в разы быстрее наивной реализации на PyTorch потому, что (1) сливает несколько GEMM-ов и softmax в один проход и (2) не материализует полную матрицу `Q @ K^T`.

В нашем проекте NAM56R используется конфигурация MLA с размерностями `kv_lora_rank=64` и `qk_pos_emb_head_dim=64`, что даёт `d_total = 64 + 64 = 128` и `v_channels = 64`. Это НЕ те размерности, под которые писалось upstream-ядро (DeepSeek-V3.2 имеет `d_total = 576`, `v_channels = 512`).

## Как это ДОЛЖНО было работать

Идеально TileLang-ядро SparseMLA должно быть параметрическим: ты передаёшь ему любые корректные размерности (`d_total`, `d_v`, `head_count`, и т.д.) и оно компилируется и работает. Вот пошагово что должно происходить:

1. Пользователь вызывает `_fused_sparse_mla_absorbed(query, key, ...)` из upstream Megatron-LM.
2. Эта функция должна посмотреть на форму тензоров (`query.shape[-1]` = `d_total`, `value.shape[-1]` = `d_v`) и передать эти числа в ядро.
3. Ядро должно скомпилироваться под конкретные размерности (TileLang делает это лениво — компиляция при первом вызове, потом кэшируется).
4. Forward-проход (fwd) считает выход attention; backward-проход (bwd) при обучении вычисляет градиенты по `Q`, `KV`.

На практике upstream-код жёстко прошит под DeepSeek-V3.2: он принципиально работает только если `d_total == 576` и `d_v == 512`. Любая другая модель отправляется в "медленный" путь.

## Что и почему НЕ работает

В upstream есть **четыре** отдельных хардкода размерностей:

**(а) `assert dim == next_power_of_2(dim)` в forward-ядре.** Эта проверка требует, чтобы `dim` была степенью двойки (32, 64, 128, 256, 512, 1024). На самом деле для warp-операций GPU достаточно, чтобы `dim` была кратна 16. Это излишне строгая проверка.

**(б) `assert dim_plus_tail_dim == 576` в forward-интерфейсе.** Тут уже жёстко прошита сумма `kv_lora_rank + qk_pos_emb_head_dim` = 576, что соответствует DeepSeek-V3.2. У нас 128. Эта assertion срабатывает первой и просто валит вызов.

**(в) `D = 512` в backward-ядре.** В bwd жёстко прописан `D = 512` — это `v_channels` от DeepSeek. У нас 64. Бэкворд работал бы с неправильной размерностью даже если бы fwd пропустил.

**(г) `dim_guard != 576` в `dsa.py`.** Сам dispatcher (выбиралка какое ядро вызвать) в upstream Megatron-LM имеет ранний `if query.size(-1) != 576 or v_channels != 512: return None`, который просто отказывается использовать TileLang-ядро для нестандартных размерностей.

**Симптом для пользователя:** при попытке использовать SparseMLA на NAM56R fused-путь возвращает `None`, и dispatcher отправляет в **unfused** реализацию — она материализует полную матрицу attention `(seq_len, seq_len, num_heads)` в global memory (gmem — основная память GPU). Для длинных последовательностей это OOM (out of memory), а на коротких — кратно медленнее.

**Корень:** ядро upstream писалось под одну конкретную модель (DeepSeek-V3.2) без выноса размерностей в параметры.

## Как мы предлагаем чинить

В нашем форке есть **две копии** этого кода, и одна из них уже починена:

* `cppmega/megatron/tilelang_sparse_mla/` — точная копия upstream, всё ещё сломана. Это цель будущего PR в `tile-ai/tilelang`.
* `cppmega/megatron/sparse_mla_ops/` — наша уже параметрическая версия, которой пользуется production NAM56R.

Конкретно мы изменили:

1. **Добавили параметр `d_v`** в `SparseMLA.forward/backward` autograd-функцию и пробросили его до интерфейса ядра.
2. **Заменили `assert dim == next_power_of_2(dim)`** на `assert dim % 16 == 0` (работает для любой кратной 16 размерности — для warp-операций больше не требуется).
3. **Сняли assertion `dim_plus_tail_dim == 576`** — само ядро уже параметризовано над `dim` и `tail_dim`, проверка была лишней.
4. **Заменили `D = 512`** на `D = d_v if d_v is not None else o.shape[-1]` — выводим V-размерность из формы выходного тензора.
5. **Бонус:** для численной стабильности dKV-градиента поменяли dtype `P_shared_cast` и `dP_shared_cast` с `dtype` (bf16 — 16-битный bfloat) на `accum_dtype` (fp32 — 32-битный float). Это критично потому что промежуточные суммы при backward могут терять точность в bf16.

**Альтернативы которые мы НЕ выбрали:**

* Можно было вообще выпилить TileLang-ядро для SparseMLA и оставить unfused-путь — но это OOM на длинных контекстах и кратное замедление.
* Можно было бы сделать обёртку которая копирует тензоры в DeepSeek-формы (паддинг до 576/512) — это работало бы, но тратило бы кучу памяти и compute впустую.

**Бонус-находка которая НЕ блокирует прод:** в `sparse_mla_ops` backward-ядре есть отдельный баг при `H=8` (`block_H=16` индексирование `Lse[..., bz*block_H + h_i]` читает за пределами буфера и даёт NaN). В production NAM56R используется больше голов на `kv_group`, поэтому этот путь не триггерится. Стоит зафиксить отдельно.

## Где это в коде и как смотреть

Файлы которые мы поменяли:

* `/Volumes/external/sources/cppmega/cppmega/megatron/sparse_mla_ops/sparse_mla.py` — добавлен параметр `d_v` в `SparseMLA.forward/backward`.
* `/Volumes/external/sources/cppmega/cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py` — снят `assert dim_plus_tail_dim == 576`, заменена next_power_of_2-проверка.
* `/Volumes/external/sources/cppmega/cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd.py` — `D = o.shape[-1]` вместо хардкода 512.

Сломанная копия (target upstream PR):

* `/Volumes/external/sources/cppmega/cppmega/megatron/tilelang_sparse_mla/sparse_mla_fwd.py`
* `/Volumes/external/sources/cppmega/cppmega/megatron/tilelang_sparse_mla/sparse_mla_bwd.py`

Dispatcher с гардом (тоже надо чинить upstream):

* `megatron/core/transformer/experimental_attention_variant/dsa.py` — функция `_fused_sparse_mla_absorbed`.

Reproducer демонстрирующий обе версии бок о бок:

* `/Volumes/external/sources/cppmega/upstream_prs/examples/02_sparse_mla_dimensions/reproducer.py`

## Проверка что фикс работает

Запустить reproducer на любой H200/Hopper-машине с активным cppmega-окружением:

```bash
cd /Volumes/external/sources/cppmega/upstream_prs/examples/02_sparse_mla_dimensions
python reproducer.py
```

Ожидаемый вывод (зафиксирован на `h200_1`, torch 2.12+cu132, tilelang 0.1.8):

```
(A) OLD copy (cppmega/megatron/tilelang_sparse_mla/) — upstream PR target
  error> FWD AssertionError: you should assign dim otherwise
  error> BWD AssertionError: warp_row_tiles must be greater than 16, got 8
  fwd_ok = False    bwd_ok = False
  BUG_REPRODUCED: the old copy refuses non-DeepSeek dims

(B) NEW copy (cppmega/megatron/sparse_mla_ops/) — fix already lives here
  imported = True
  fwd_ok   = True   out_shape = (1, 128, 8, 64)
  bwd_ok   = True   finite(dq,dkv) = (False, False)
  FIX_VALIDATED: parametric d_v path runs fwd + bwd at d_total=128,d_v=64
```

Exit-коды:
* `1` — баг воспроизводится И фикс работает (ожидаемое состояние сегодня).
* `0` — старая копия больше не валится (значит upstream приняли наш PR).
* `2` — окружение не готово (нет CUDA, нет tilelang).

Также для production-проверки можно запустить полноценный train smoke-скрипт NAM56R и убедиться что в логах нет `_fused_sparse_mla_absorbed returned None` сообщений и attention реально идёт через fused-путь.
