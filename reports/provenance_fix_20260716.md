# Исправление provenance для staged commit pipeline

**Дата:** 2026-07-16 (CEST)  
**Ветка:** `codex/canonical-complete-20260715`  
**Коммиты:** `cppmega.mlx` [`7ca3f71`](https://github.com/DatasunriseOU/cppmega_mlx/commit/7ca3f71), `cppmega` [`71786b4`](https://github.com/DatasunriseOU/cppmega/commit/71786b4)

## Итог

Исправлен training-blocking дефект CASE5 v7: staged checkout имел путь `_src`, а `extract_git_history.py` сохранял `repo="_src"`. Строгий `process_commits.py` закономерно отклонял все commit ranges как невалидные project identities.

Теперь canonical identity передаётся явно по всей цепочке:

```text
repo_list.json
  -> streaming_conveyor / streaming_reindex_commits
  -> extract_git_history --repo-name <canonical-project>
  -> process_commits --project-id <canonical-project>
  -> PR lookup
  -> enriched JSONL
  -> tokenized parquet / stable IDs
```

Legacy cache не пересобирается молча. При валидном completed publication он принимается только после проверки старого checkpoint fingerprint, а receipt получает `hit_legacy_identity_override`. Каждая legacy record нормализуется и считается в `legacy_identity_overrides`.

## Root cause

В conveyor исходники commit-ветки складываются в synthetic directory:

```text
<repo>/_src
```

У Blender в старом cache записи содержали:

```json
{
  "repo": "_src",
  "repo_stable_id": "eb72682fb1c12f83",
  "filepath_stable_id": "07ac4af6b6d1a3f1"
}
```

После введения строгого [symbol identity contract](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/symbol_identity.py:122) такое значение должно быть отклонено. Ошибка была не в strict validator, а в том, что producer не передавал canonical identity в extractor и consumer.

## Что изменено

### Extractor

В [extract_git_history.py](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/extract_git_history.py:730) добавлен `--repo-name`. Для explicit identity проверяются:

- ровно один canonical slash;
- отсутствие конфликта с `remote.origin.url`;
- stable IDs создаются уже от canonical project.

### Commit consumer

В [process_commits.py](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:2407) добавлена нормализация записи перед PR lookup и clang processing.

- bare legacy repo (`_src`) заменяется на переданный `--project-id`;
- `repo_stable_id` пересчитывается;
- `filepath_stable_id` пересчитывается от `project_id + filepath`;
- другой уже-valid canonical project не переписывается, а вызывает fail-loud `SymbolIdentityError`;
- в итоговой статистике виден `legacy_identity_overrides`.

### Wiring

- [streaming_reindex.py](/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex.py:968) передаёт `--project-id` в commit indexer;
- [streaming_reindex_commits.py](/Volumes/external/sources/cppmega.mlx/scripts/streaming_reindex_commits.py:297) передаёт `--repo-name` extractor-у и identity range-у;
- [streaming_conveyor.py](/Volumes/external/sources/cppmega.mlx/scripts/streaming_conveyor.py:1199) умеет валидировать canonical и старый legacy checkpoint fingerprint и пишет explicit legacy receipt.

В root CUDA репозитории применён тот же production fix: [cppmega process_commits.py](/Volumes/external/sources/cppmega/tools/clang_indexer/process_commits.py:2554), [cppmega conveyor](/Volumes/external/sources/cppmega/scripts/streaming_conveyor.py:1198), [root regression suite](/Volumes/external/sources/cppmega/tests/test_commit_project_identity.py:1).

## Verification

### Test-first regression coverage

Красные тесты сначала воспроизвели:

- неизвестный `--repo-name`;
- отсутствие `project_id` в `process_jsonl_file`;
- отсутствие identity forwarding в stage wrappers.

После реализации:

| Проверка | Результат |
|---|---:|
| MLX focused data/conveyor/identity suite | **122 passed** |
| CUDA root identity suite | **4 passed** |
| Python compile для пяти production scripts в root | **pass** |
| Python compile для пяти production scripts в MLX | **pass** |

Полные локальные suites также запускались. В MLX до остановки на unrelated environment/research failures было `1453 passed, 13 skipped`; root дал `1353 passed, 152 skipped`. Оставшиеся failures относятся к TileLang/Metal, отсутствующему Megatron environment и stale test-surface contracts, а не к этому provenance diff.

### Exact 100-record Blender probe

Вход: `/tmp/cppmega_v8_blender_r0_probe/blender_r0.jsonl` (legacy records с `_src`).

Команда использовала `--project-id blender/blender` и тот же `process_commits.py`, что и production stage.

Receipt:

```text
records_read:              100
documents_written:         140
parse_errors:                0
legacy_identity_overrides: 100
output repos:                blender/blender (140/140)
bad repo stable IDs:         0
bad filepath stable IDs:     0
elapsed:                    67.2 s
```

### Production v8 cache/retry

Detached runtime:

`/Volumes/external/sources/cppmega_mlx_case5_v8_runtime`

Run:

`/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v8_retry_blender_20260716_0045`

Launcher:

`/Volumes/external/sources/cppmega.mlx/outputs/launch_case5_v8_retry_blender.sh`

Проверено production path:

```text
external cache:  HIT, publication validated
records:         175,111
cache status:    hit_legacy_identity_override
re-extraction:   0
first parquet:   canonical repo + stable IDs verified
unit_failed:     0 at latest receipt
unit_done:       16 at latest receipt snapshot
valid tokens:    5,676,156 at latest receipt snapshot
```

Repo-list использует для Blender lossless forge identity `projects.blender.org/blender%2Fblender`; это intentional canonical identity, а не bare directory и не silent `blender` fallback.

Live logs:

- [v8 launcher](/Volumes/external/sources/cppmega.mlx/outputs/launch_case5_v8_retry_blender.sh:1)
- [v8 log](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v8_retry_blender_20260716_0045.log:1)
- [v8 progress JSONL](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v8_retry_blender_20260716_0045/progress_commits.jsonl:1)

## Current live processes

| Process | State |
|---|---|
| `cppmega_case5_v7_code` | still running; latest progress receipt `1,152,296,883` valid code tokens |
| `cppmega_case5_v7_commits` | stopped after the probe proved the fix; old Blender failure storm no longer consumes workers |
| `cppmega_case5_v8_blender` | running with the canonical fix and legacy cache receipt |

## 4070 check

The explicitly authorized host `davidgor@10.0.0.16` is reachable with the configured runner key and reports an RTX 4070 Laptop GPU, 8188 MiB, driver `575.57.08`. The remote checkout currently has no `torch`, `megatron`, or `transformer_engine`, so no CUDA/Megatron training smoke was started without installing a new dependency stack.

## Remaining risks

1. v8 Blender retry is still running; `16/1752` ranges were complete at the latest snapshot, not the final corpus verdict.
2. Legacy cache records are normalized at consumption time. A future cache rewrite can remove the per-record override cost, but must preserve the canonical source fingerprint contract.
3. Full suites retain unrelated environment failures; they were not folded into this fix.
4. PR discussion hit rate for Blender remains `0` in the 100-record probe; this is a PR-store coverage/key issue, separate from identity corruption.
