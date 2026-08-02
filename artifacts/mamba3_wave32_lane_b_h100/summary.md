# Wave32 lane_b (bwd_bwd vectorized diag) — verdict: не включать, lane закрыт

Backlog: P089 (`docs/backlog_plan_2026_08_01.md`).
Harness: `scripts/modal_mamba3_wave32_lane_b_h100.py`
(GPU теперь параметризуется `CPPMEGA_LANE_B_GPU`, default `H100`).

## История прогонов

| run | дата | GPU | образ | итог |
| --- | --- | --- | --- | --- |
| r5 | 2026-04-30 | H100 | `ghcr.io/datasunriseou/cppmega:785c3fd` | bench rc=0, но `applier_gated` rc=1: applier отказался мутировать installed mamba_ssm без `MAMBA3_BWD_BWD_VECTORIZED_DIAG_ALLOW_FILE_MUTATION=1` → выводы невалидны |
| r6 | 2026-04-30 | H100 | тот же | умер сразу после `bench_start`, вывода нет — невалиден |
| r7 | 2026-08-02 | H200 | `ghcr.io/datasunriseou/cppmega:459a574` | gated+rollback rc=0 (флаг работает), но bench rc=1: образ 459a574 уже несёт stage2 force-nonTMA в installed mamba_ssm, сырой `.patch` (`patch -p4`) падает 21/24 hunk при сборке варианта `stage2_current` — image drift, `datasunriseou/cppmega:785c3fd` к этому моменту удалён из GHCR |
| **r8** | 2026-08-02 | **H200** | `ghcr.io/jewelmusicee/cppmega:785c3fd` (тот же бит-в-бит образ, что у r5) | **полный успех**: `applier_noop` rc=0, `applier_gated` rc=0 («DONE bwd_bwd vectorized diag patch applied»), `applier_rollback` rc=0 («DONE restored backup»), bench rc=0 |

Фикс харнесса для r7/r8: gated-прогон applier'а получает
`MAMBA3_BWD_BWD_VECTORIZED_DIAG_ALLOW_FILE_MUTATION=1`, после него добавлен
явный rollback-прогон (`CPPMEGA_MAMBA3_BWD_BWD_VECTORIZED_DIAG_ROLLBACK=1`),
чтобы baseline в bench собирался из нетронутого installed source. Контроль
целостности: в report.json r8 у варианта `baseline` маркер
`wave32_vectorized_diag=false` — контаминации baseline не было.
Артефакты: `wave32_lane_b_h200_vectorized_20260802_r8/report.json`
(r7 — `wave32_lane_b_h200_vectorized_20260802_r7/`, fail на bench по image drift).

## Численность (r8, обе формы)

- `stage2_current` ≡ `baseline` точно (max_abs = 0.0 по всем выходам).
- `wave32_vectorized_diag` vs baseline: max_abs ≤ 1.82e-06 (smoke) и
  ≤ 2.57e-06 (representative) — уровень bf16-шума. Численно кандидат корректен.

## Производительность (bwd_bwd kernel, representative shape)

| run | GPU | baseline | stage2_current | wave32_vectorized_diag | Δ wave32 vs stage2 |
| --- | --- | ---: | ---: | ---: | ---: |
| r5 | H100 | 0.3169 ms | 0.3127 ms | 0.3623 ms | **+15.9 % (медленнее)** |
| r8 | H200 | 0.3158 ms | 0.3150 ms | 0.3646 ms | **+15.7 % (медленнее)** |

`bwd_fwd` не изменяется (0.133–0.137 ms у всех вариантов). Память идентична
(peak 33.0 MiB у всех). Chain-замеры шумные (JIT/clock-эффекты: r5 stage2
chain 2.26 ms против r8 0.67 ms при том же коде) и в решение не берутся;
стабильный per-kernel сигнал воспроизвёлся на двух GPU и двух независимых
прогонах.

## Вердикт

**Откатить / не включать.** Кандидат численно верен, но целевой kernel
`bwd_bwd` стабильно на ~13–16 % медленнее stage2_current при нулевом выигрыше
по памяти. В репозитории откатывать нечего: патч применялся только внутри
одноразовых контейнеров и там же откачен; applier
(`cppmega/megatron/upstream_patches/apply_mamba3_bwd_bwd_vectorized_patches.py`)
остаётся env-gated и по умолчанию выключен. Lane_b закрыт.

## Воспроизведение r8

```bash
GHCR_REPO=ghcr.io/jewelmusicee/cppmega GHCR_TAG=785c3fd CPPMEGA_LANE_B_GPU=H200 \
  uvx --from modal --with torch modal run scripts/modal_mamba3_wave32_lane_b_h100.py \
  --run-id wave32_lane_b_h200_vectorized_20260802_r8
```

Замечание для будущих прогонов: на новых образах (>= `459a574`), где stage2
force-nonTMA уже вшит в installed mamba_ssm, bench-driver харнесса падает на
`patch -p4` — перед таким прогоном нужно научить `_make_variant` принимать
pre-patched installed source как `stage2_current` (или заново собрать
pristine baseline из wheel'а).
