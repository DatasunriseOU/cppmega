# Подробное объяснение: Mamba3 — fp32 dt и поддержка GQA в MIMO backward

## Что это вообще и зачем

**Mamba3** — третье поколение state-space моделей от Tri Dao / Albert Gu (state-spaces/mamba). Это альтернатива attention: вместо того чтобы каждый токен смотрел на все прошлые, поддерживается компактное "состояние" (state), которое обновляется на каждом шаге через специальный SSM-рекуррентный апдейт.

**MIMO** = Multi-Input Multi-Output. В отличие от SISO (одна голова — один канал), MIMO Mamba3 имеет внутреннюю размерность `R` (`mimo_rank`, обычно 4): каждая голова держит не один, а R-мерный low-rank state. Это сильно увеличивает выразительность блока без квадратичного раздувания памяти. В нашем NAM56R используется именно MIMO с R=4.

**dt** (delta-time, "шаг по времени") — это специальный per-token, per-head скаляр, который контролирует, насколько сильно SSM-state обновляется на данном токене. Маленький dt = состояние почти не двигается (модель "помнит" далеко); большой dt = состояние быстро забывает старое. Это центральная динамическая величина SSM, и её точность влияет на стабильность всего обучения. Поэтому dt держат в **fp32** (а не в bf16): exp(-A·dt) очень чувствителен к ошибкам округления, и потеря точности тут даёт NaN'ы или дрифт.

**GQA** (Grouped-Query Attention) — это компромисс между MHA (Multi-Head, каждая Q-голова имеет свои K/V) и MQA (Multi-Query, все Q делят одну K/V): головы группируются в `ngroups` групп, внутри группы все Q делят одну K/V. Это экономит KV-cache и compute. В Mamba3 MIMO та же концепция: `nheads_qk` (число K-/Q-групп, обычно обозначается `G`) меньше, чем `nheads` (число V-голов, `H`), причём `H % G == 0`.

**TileLang** — это DSL для написания GPU-кернелов, более низкоуровневый чем Triton, с явным управлением shared memory, TMA, warp-spec. Mamba3 MIMO forward/backward написаны на TileLang.

**Float16Module** — это обёртка из Megatron-LM, которая ковертит все параметры модели в bf16 (или fp16) для смешанной точности обучения. Обходит модель и для каждого `nn.Parameter` делает `.to(bf16)`.

## Как это ДОЛЖНО было работать

Сценарий: NAM56R (наша 56B-MoE модель) с Mamba3 MIMO-блоками, ngroups=8, nheads=128 (то есть 16 голов на группу). Запускаем под Megatron, который заворачивает модель в Float16Module для bf16-обучения.

1. Forward: `mamba3.py` собирает входы, вычисляет `dt = softplus(dd_dt + dt_bias)`, и зовёт TileLang fwd-кернел `mamba_mimo_fwd_kernel`. Кернел получает `Q, K, V, DT, ADT, ...` — DT и ADT обязаны быть fp32.
2. Backward: автоград зовёт `mamba_mimo_bwd_combined`, который раскладывает градиенты обратно в каноничные формы. Внутри нужно отредуцировать `dq` и `dk` по головам внутри группы (потому что в forward все головы группы шарили один K и Q).

## Что и почему НЕ работает

Здесь **два независимых бага**, и они оба триггерятся в одном и том же сценарии (NAM56R + Megatron).

### Баг 1: dt становится bf16

**Симптом:**

```
RuntimeError: kernel mamba_mimo_fwd_kernel input DA_CS dtype expected float32, but got bfloat16
```

(или аналогичное про `DT`, в зависимости от того, какой аргумент TileLang проверяет первым.)

**Корень**: в `mamba_ssm/modules/mamba3.py` есть строка:

```python
DT = F.softplus(dd_dt + self.dt_bias)
```

Когда `dt_bias` — это обычный fp32-параметр, всё работает. Но Megatron Float16Module прошёлся по модели и сделал `dt_bias.data = dt_bias.data.to(bf16)`. Теперь `dd_dt + self.dt_bias` — это **bf16** (PyTorch промоутит fp32+bf16 в bf16, чтобы не аллоцировать лишний fp32-тензор), и `softplus` возвращает bf16. TileLang fwd-кернел его отвергает.

Альтернативный сценарий хуже — кернел **молча принимает** bf16 и считает с потерей точности. На длинных последовательностях это даёт дрифт и NaN'ы через несколько сот шагов, который потом ищется днями.

**Это не тот же баг, что B/C layout `(g,r,n)` vs `(r,g,n)`** — там переставлены оси, и баг триггерится только при TP>1 с ngroups>1. Здесь — про dtype, и триггер совершенно другой.

### Баг 2: GQA-ветка отсутствует в backward

**Симптом:**

```
ValueError: G value of 2 is not currently supported!
```

**Корень**: в `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` функция `mamba_mimo_bwd_combined` после вызова TileLang-кернела должна отредуцировать `dq_tilelang` и `dk_tilelang` (которые имеют форму `[B, S, R, H, N]`) в каноничные `dq, dk` формы `[B, S, R, G, N]`.

Upstream-код умеет только два случая:

```python
if G == 1:                     # MHA: одна группа на все головы → sum по всем H
    ...
elif G == H:                   # per-head: каждая голова — своя группа → identity
    ...
else:
    raise ValueError(f"G value of {G} is not currently supported!")
```

Промежуточный GQA-случай (`1 < G < H`, `H % G == 0`, например `G=2, H=16` или `G=8, H=128`) просто не реализован. Видимо, автор тестировал MHA и per-head на маленьких конфигах, а GQA планировал на потом.

GB10-машина не воспроизводит этот баг через reproducer — там TileLang падает раньше, на этапе компиляции `mamba_mimo_bwd_bwd` из-за `LayoutInference FloorMod` (это отдельный issue, отслеживается в PR 13). На bench3 (H200, sm_90a) оба бага воспроизводятся стабильно.

## Как мы предлагаем чинить

### Фикс 1: явный fp32 cast перед softplus

В `mamba3.py` обе строки с `softplus(dd_dt + self.dt_bias)`:

```python
# Было:
DT = F.softplus(dd_dt + self.dt_bias)

# Стало:
DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))
```

Применить нужно в **двух** местах: строки 169 (основной forward путь) и 255 (`_preprocess` для второго scan-пути). Стоимость — один лишний `.to(fp32)` на токен, что копейки на фоне остального compute.

### Фикс 2: добавить GQA-ветку в backward

В `mamba3_mimo_bwd.py` (и в `mamba3_mimo_bwd_varlen.py`):

```python
elif H % G == 0:
    # GQA-style: 1 < G < H, H делится на G. Суммируем по heads_per_group.
    hpg = H // G
    # bias-grads СНАЧАЛА: они нужны в форме [H, R, N]
    dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
    dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
    # потом редуцируем dq/dk по heads-per-group
    dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
    dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
    dmimo_v = dmimo_v.sum(dim=0)
    dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
    dD = dD.sum(dim=0) if dD is not None else None
```

Важный нюанс: bias-градиенты считаются **до** редукции `dq/dk`, потому что `Q_bias` имеет форму `[H, R, N]` (не редуцированную по группам). Если редуцировать `dq` сначала, а потом пытаться извлечь bias — потеряем информацию.

В `_varlen` версии та же логика, только `dmimo_v/dmimo_z/dD` суммируются по `(0, 2)` вместо `0` — потому что в варлен-режиме батч и время свёрнуты в одну ось.

## Где это в коде и как смотреть

- **Файл 1:** `mamba_ssm/modules/mamba3.py`, строки 169 и 255.
- **Файл 2:** `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`, около строки 1310 (функция `mamba_mimo_bwd_combined`).
- **Файл 3:** `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py`, около строки 1374 (функция `mamba_mimo_bwd_combined_varlen`).
- **Локальный convenience patch:** `upstream_prs/05_mamba3_dt_fp32_gqa_bwd.patch`.

Важно: этот `.patch` сейчас объединяет **два разных upstream-fix'а**,
потому что они ко-триггерятся в одном Megatron + Mamba3 training lane.
Для upstream filing их надо разделять:

- `bf16` / `fp32` стадии reproducer'а — это **PR 16** против `NVIDIA/Megatron-LM`
  (Float16Module silently casts Mamba3 fp32-contract params).
- `gqa_unpatched` / `gqa_patched` стадии — это **PR 05** против
  `state-spaces/mamba` (missing intermediate GQA branch in MIMO backward).

Чтобы посмотреть на bench3:

```bash
grep -n "softplus(dd_dt" /mnt/data/venv/lib/python3.13/site-packages/mamba_ssm/modules/mamba3.py
grep -n "G value of" /mnt/data/venv/lib/python3.13/site-packages/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
```

## Проверка что фикс работает

Reproducer (`reproducer.py`) запускает **четыре стадии** в отдельных subprocess'ах. Subprocess'ы нужны потому, что когда TileLang-кернел падает на bf16, он оставляет CUDA-контекст в "плохом" состоянии: любой следующий kernel launch в том же процессе возвращает `cudaErrorMisalignedAddress` (это воспроизводится и на GB10, и на H200). Чистый процесс на каждую стадию = независимая проверка.

Стадии:

1. **`bf16`** — собираем входы с `dt_dtype=bfloat16` (имитирует Float16Module после прохода). Запускаем fwd. Ожидаем `RuntimeError: ... expected float32, but got bfloat16`. Это **подтверждает Баг 1**.
2. **`fp32`** — те же входы, но с явным `.to(fp32)` перед softplus (имитация Фикса 1). Запускаем fwd+bwd, проверяем что все 9 градиентов конечны и ненулевые. Это **валидирует Фикс 1**.
3. **`gqa_unpatched`** — временно подменяем установленный `mamba3_mimo_bwd.py` на оригинал (без GQA-ветки), запускаем bwd с `G=2, H=16`. Ожидаем `ValueError: G value of 2 is not currently supported!`. Это **подтверждает Баг 2**.
4. **`gqa_patched`** — восстанавливаем патченную версию, запускаем тот же bwd. Ожидаем что все градиенты конечны и численно совпадают со стадией 2 (fp32-ok). Это **валидирует Фикс 2**.

Запуск:

```bash
source /mnt/data/venv/bin/activate    # bench3
cd upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd
python reproducer.py
```

Ожидаемый суммарный вывод:

```
Summary:
  stage bf16               -> bf16_refused
  stage fp32               -> fp32_ok
  stage gqa_unpatched      -> unpatched_raised
  stage gqa_patched        -> patched_ok

Problem 1: BUG_REPRODUCED ; FIX_VALIDATED
Problem 2: BUG_REPRODUCED ; FIX_VALIDATED
```

Exit code: **0**.

Реальные значения градиентов на bench3 (для контроля):

```
max|grad|: Q=3.359e+00, K=5.438e+00, V=2.088e+01,
           DT=1.962e+02, ADT=4.068e+01,
           Q_bias=1.862e+01, K_bias=1.688e+01,
           MIMO_V=2.064e+01, D=2.002e+01
```

Стадии 2 и 4 должны давать одинаковые числа до округления.

**На GB10 reproducer не работает целиком** — там `bwd_bwd` падает в TileLang компиляции из-за `FloorMod` constant-folding (отдельный PR 13). Однако стадия `bf16` (Баг 1) на GB10 воспроизводится — там падение случается в forward, до bwd-компиляции.
