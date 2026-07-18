# Hardening cppmega: окружение, CI и локальная принадлежность исходников

Дата: 2026-07-16
Репозиторий: [DatasunriseOU/cppmega](https://github.com/DatasunriseOU/cppmega/tree/codex/canonical-complete-20260715)

## Что именно исправлено

### 1. Dedicated Megatron environment вместо случайного Python

- Корневой `conftest.py` теперь загружает только реальный Megatron-LM из явно
  выбранного source root.
- Источник должен быть подтверждён dedicated receipt либо парой
  `MEGATRON_LM_REPO` + `CPPMEGA_MEGATRON_COMMIT`.
- Проверяются точный Git commit, чистота source tree, происхождение
  `megatron.core` и отсутствие test stubs в `sys.modules`.
- Явный source root, не совпадающий с root из receipt, больше не может
  воспользоваться чужим receipt: для него требуется собственный exact commit.
- Portable-data профиль проверяется после pytest collection. Node IDs и значения
  опций, например `--basetemp`, больше не ошибочно принимаются за test files.

Исходники:

- [conftest.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/conftest.py)
- [scripts/env/cppmega_env.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/env/cppmega_env.py)
- [docs/environment.md](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/docs/environment.md)

### 2. Self-hosted CI больше не зависит от `cppmega.mlx/.venv`

- macOS host теперь использует
  `/Volumes/external/sources/.venvs/cppmega.source/bin/python`.
- Runner очищает `PYTHONPATH`, `PYTHONHOME` и `VIRTUAL_ENV` и включает
  `PYTHONNOUSERSITE=1`/`PYTHONSAFEPATH=1` для каждого lane subprocess.
- `portable-data` перенесён из hardcoded Python mapping в декларативное поле
  `test_profile` в lane JSON.
- Child processes получают профиль через явный environment override.

Исходники:

- [configs/ci/hosts.json](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/configs/ci/hosts.json)
- [configs/ci/lanes.json](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/configs/ci/lanes.json)
- [scripts/ci/repository_runner.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/ci/repository_runner.py)
- [.github/workflows/ci-self-hosted.yml](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/.github/workflows/ci-self-hosted.yml)

### 3. cppmega снова самостоятельно владеет C/C++ prompt graph path

- Prompt graph loader требует `tools/clang_indexer/index_project.py` из того же
  checkout и восстанавливает исходный объект `sys.path`, а не только его значения.
- H200 generation eval по умолчанию использует текущий `cppmega` checkout, а не
  соседний `cppmega.mlx`.
- Cross-checkout смешивание indexer implementation теперь завершается ошибкой.
- `clang-format` ищется через обычный `PATH`, затем `xcrun`, затем известные LLVM
  Homebrew paths. Ошибка/timeout `xcrun` не превращается в traceback и не
  отключает format gate.
- Restore Megatron bundle CLI импортирует sibling module через repository package,
  поэтому прямой запуск файла работает при `PYTHONSAFEPATH=1`.

Исходники:

- [cppmega/prompt_graph_index.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/cppmega/prompt_graph_index.py)
- [cppmega/data/prompt_graph_index.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/cppmega/data/prompt_graph_index.py)
- [scripts/nebius_h200_megatron_cpp_generation_eval.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/nebius_h200_megatron_cpp_generation_eval.py)
- [scripts/cpp_generation_compile_eval.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/cpp_generation_compile_eval.py)
- [scripts/data/restore_megatron_bundle_from_nebius_s3.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/data/restore_megatron_bundle_from_nebius_s3.py)

### 4. Memory guard проверяет текущий RSS, а не исторический пик

- Admission guard больше не использует монотонный `ru_maxrss` как текущую память.
- Последовательно пробуются Linux procfs, `psutil` и Unix `ps`.
- Если все probes недоступны, guard завершается fail-closed.
- Watchdog и синхронная проверка используют одну семантику текущего RSS.

Исходники:

- [scripts/data/memory_guard.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/data/memory_guard.py)
- [scripts/nanochat_data/memory_guard.py](https://github.com/DatasunriseOU/cppmega/blob/codex/canonical-complete-20260715/scripts/nanochat_data/memory_guard.py)

### 5. Package и environment metadata согласованы

- Tracked `cppmega.egg-info` пересобран из текущего `pyproject.toml`: Python
  floor `>=3.11`, optional `data` с `datasketch==1.10.0`.
- Setuptools package discovery ограничен `cppmega`/`cppmega.*`; namespace
  `cppmega_mlx` не попадает в wheel.
- Runtime contracts `data/domain_schema_v1.json`, `tokenizer/tokenizer.json` и
  `tokenizer/tokenizer_contract_v1.json` явно включены в package data.
- Wheel smoke это проверяет: contracts и `cppmega` присутствуют,
  `cppmega_mlx` отсутствует.
- Cold timeout environment probe теперь возвращает структурированный FAIL,
  а не необработанный `TimeoutExpired`.

## Что сознательно не включено в production commit

| Dirty file/group | Причина |
|---|---|
| `.gitignore` report patterns | Слишком широкое подавление review artifacts, не относится к hardening |
| `tests/test_m2rnn_path_b_engine.py` | Cross-repo MLX contract без локального runtime ownership; остаётся receipt-gated |
| `tests/test_mamba3_path_c_engine.py` | Cross-repo Metal/MLX gate, не является локальной CUDA/Megatron проверкой |
| `docs/superpowers/plans/2026-07-15-...md` | Рабочий plan artifact, не production documentation |
| `tests/test_pr_export_batches.py` | Cross-repo MLX namespace smoke; не является CUDA/Megatron production gate |

## Проверка

| Gate | Результат |
|---|---|
| Полный pytest в dedicated Megatron env | `1385 passed, 159 skipped, 24 warnings` за `50.87s` |
| Selected environment/CI/regression suite | `104 passed` |
| Environment verification | Exact clean Megatron commit `ba7b5ebce12af60627a80985792a1449ce45f46c` |
| cppmega wheel smoke | PASS; runtime JSON contracts есть, namespace только `cppmega` |
| Changed-file Ruff | `All checks passed` (`ruff 0.15.22`) |
| Python compileall | PASS для `cppmega scripts tests tools` |
| Tracked shell `bash -n` | PASS |
| `git diff --check` | PASS |
| HTML changelog desktop/mobile | PASS: 1440px и 390px без page-level horizontal overflow |

## Dependabot review: 17 июля 2026

Alert `GHSA-rrmf-rvhw-rf47` был закрыт заменой vulnerable reproducer pin
`torch==2.12.0.dev20260410+cu132` на стабильный CUDA-релиз
`torch==2.13.0+cu132`. Тот же стабильный pin теперь используется в
`STACK.lock`, CUDA Docker и self-hosted wheel workflow; активные H200/B200
Modal-образы больше не разрешают nightly. `cppmega.mlx` уже имел стабильный
`torch 2.13.0` в `uv.lock`, а optional Path C теперь явно требует именно этот
релиз. Исторические отчёты о старых nightly-прогонах сохранены как история,
но текущий production/reproducer contract больше не зависит от dev-сборки.

Команда полного теста:

```bash
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  MEGATRON_LM_REPO=/Volumes/external/sources/Megatron-LM-core_v0.18.0 \
  /Volumes/external/sources/.venvs/cppmega.source/bin/python -m pytest -q
```

## Что осталось

- CUDA/H200 runtime в этом проходе локально не запускался. Workflow и unit
  contracts проверены, но реальный GPU receipt должен быть получен на self-hosted
  CUDA runner.
- Предупреждения Megatron о Transformer Engine/Apex ожидаемы для локального Mac
  environment и не считаются GPU proof.
- Полный repo-wide Ruff остаётся legacy debt (`202` нарушения в нетронутых
  файлах); все Python-файлы production diff проходят pinned changed-file Ruff.
- Cross-repo MLX tests оставлены вне commit, чтобы `cppmega` не выдавал отсутствие
  локального MLX runtime за успешную CUDA проверку.
- Dependabot alert по reproducer Torch остаётся отдельным operational risk и не
  должен считаться закрытым production fix.
