# Backlog Plan 2026-08-01 — что не доделано и не в работе (100 шагов)

Status: audit snapshot; каждый пункт требует live-revalidation перед исполнением
Canonical: текущие code/git/CI receipts и issue tracker, не этот snapshot
Date: 2026-08-01
Scope: аудит от 2026-07-31/08-01 по обоим репозиториям; только то, что НЕ
находится в активном фронте работ.

Snapshot cut: список был собран до посадки `cppmega` commits `27f77825` и
`50cb1ebe` и `cppmega.mlx` commit `6ba53352`. Поэтому P073 и P074 уже закрыты,
P019/P020/P075 частично выполнены, а предпосылки остальных пунктов могли
измениться. Файл сохраняет найденный backlog, но не заменяет проверку
актуального дерева и receipts.

## Границы

**Активный фронт на момент snapshot (уже приземлён после cut):**
document isolation CP/SP (`cppmega/megatron/document_isolation.py` и связанные),
FA4 `mask_mod` (`fa4_score_mod_adapter.py`, `fa4_graph_attention.py`), пин
`05_mamba3_dt_fp32_gqa_bwd.patch` (`STACK.lock`, `build-wheels.yml`), новые
тесты (`test_document_isolation_cp.py`, `test_fa4_document_isolation.py`,
`test_fa4_h200_parity.py`) и все H200/Modal/Nebius-прогоны фронта.

**Репозитории:** `cppmega` = /Volumes/external/sources/cppmega,
`mlx` = /Volumes/external/sources/cppmega.mlx.
**Общее окружение:** оба репо используют `.venv` →
`/Volumes/external/sources/.venvs/cppmega.mlx` (Python 3.13.12, torch 2.13.0,
mlx 0.32.0). Манифест cppmega: `<venv>/cppmega-environment.json`
(megatron_root = `Megatron-LM-test-e40feed4`, commit `e40feed4a...`).

**Формулы проверки:**
- cppmega: `cd /Volumes/external/sources/cppmega && .venv/bin/python -m pytest <files> -q`
  (env-переменные не нужны — манифест в venv).
- cppmega portable-data: `CPPMEGA_TEST_PROFILE=portable-data .venv/bin/python -m pytest <files> -q`
- mlx: `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest <files> -q`
- bd: все шаги заведены в `cppmega.mlx/.beads` с меткой `backlog-20260801`;
  номер шага = префикс `[P0NN]` в заголовке issue.

**Приоритеты:** P1 — разблокирует релиз/данные/фронт; P2 — обычная работа;
P3 — стратегическое/отложенное.

---

# Фаза A. cppmega — код и гигиена, без железа (P001–P014)

## [P001] Восстановить окружение nanochat после инцидента uv sync — DONE
- Репо: external | Приоритет: P1 | Тип: chore | Зависит от: —
- **Где:** `/Volumes/external/sources/nanochat/pyproject.toml`, `uv.lock`,
  `.venvs/music-jax-mps`.
- **Что сделано:** `jax[cpu]>=0.4.30` возвращён в зависимости nanochat,
  `uv.lock` пересобран (`uv sync`).
  Конфликта с cppmega/cppmega.mlx больше нет — они переключены на выделенный venv.
- **Проверка:**
  - `/Volumes/external/sources/.venvs/music-jax-mps/bin/python -c "import jax; print(jax.__version__)"` → `0.11.0`.
  - `readlink cppmega.mlx/.venv` → `../.venvs/cppmega.mlx`.
  - nanochat commit `42803d87` запушен в `origin/main`.

## [P002] Верификация wheel-сборки после починки packaging-бага — DONE
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** `cppmega/pyproject.toml:31-33`, новые
  `cppmega/features/{mhc,mod,structure}/__init__.py`.
- **Что сделано:** packaging fix `include = ["cppmega", "cppmega.*"]` оставлен;
  локально собран и изолированно проверен wheel
  `cppmega-0.1.0-py3-none-any.whl`
  (`sha256:2f0895a921d7fd015ff3daa344c5440c83e1ef8288e42c2f9c4df02dc5a9824d`);
  внутри присутствуют `cppmega/features/{mhc,mod,structure}/` (7 файлов).
  Deprecated-модули сохранены: верификация packaging не требует удаления
  runtime-кода.
- **Проверка:**
  - `PYTHONPATH= .venv/bin/python3 -m build --wheel --no-isolation`
  - `unzip -l dist/*.whl | grep -c "features/\(mhc\|mod\|structure\)/"` → 7
  - wheel распакован во временный каталог; оба импорта
    `cppmega.recipes.nam56r_megatron` и
    `cppmega.megatron.custom_embedding` разрешились именно из распакованного
    wheel, не из source tree.

## [P003] Почистить исторические ссылки на удалённые модули
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: P002
- **Где:** `README.md:460`, `docs/changelog.md:938`,
  `docs/dsa_ep2_tilelang_sweep_2026_04_12.md`,
  `docs/nam56r_mtp_optimization_plan_2026_04_11_ru.md`, `reports/review_20260713/`.
- **Что делать:** не менять, пока runtime-модули сохранены. Исторические ссылки
  остаются корректными; cleanup нужен только после отдельного доказанного
  решения об удалении.
- **Проверка:** оба файла существуют и остаются покрыты текущими тестами/import
  checks; исторические docs не заявляют ложное удаление.

## [P004] Логирование вместо молчаливого except: hybrid_schedule_plan.py — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `cppmega/megatron/hybrid_schedule_plan.py:647`.
- **Что сделано:** единственный `except Exception` уже логируется через
  `log.warning(..., exc_info=True)`; поведение не изменено.
- **Проверка:** `rg -n "except Exception" cppmega/megatron/hybrid_schedule_plan.py`
  — у handler есть log; `pytest tests/ -k hybrid_schedule -q`.

## [P005] Логирование вместо молчаливого except: memory_debug.py — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `cppmega/megatron/memory_debug.py:45,49,208,233,304,305`.
- **Что сделано:** все `except Exception`/`except Exception as e` уже логируются
  (`log.debug(..., exc_info=True)` или `print` + `traceback.print_exc()`);
  поведение не изменено.
- **Проверка:** `pytest tests/ -k memory_debug -q`; ручной вызов с
  `logging.basicConfig(level=DEBUG)` — сообщения видны.

## [P006] Логирование вместо молчаливого except: grouped_mxfp8_gemm.py — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `cppmega/megatron/grouped_mxfp8_gemm.py:597,600,639`.
- **Что сделано:** `except Exception` в reshape-пути уже логируется через
  `log.debug(..., exc_info=True)`; build-failure handler сохраняет ошибку и
  re-raise; поведение не изменено.
- **Проверка:** `pytest tests/test_grouped_mxfp8_direct_routing.py -q`.

## [P007] Логирование вместо молчаливого except: mxfp8_sidecar_refs.py — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `cppmega/megatron/mxfp8_sidecar_refs.py:34,38`.
- **Что сделано:** оба `except Exception` в очистке sidecar-ссылок уже
  логируются через `log.debug(..., exc_info=True)`; поведение не изменено.
- **Проверка:** `pytest tests/test_mxfp8_sidecar_lifecycle.py -q`.

## [P008] Логирование в upstream_patches/apply_*.py (цепочки return False) — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `cppmega/megatron/upstream_patches/apply_mamba3_mimo_p1_patches.py:398`.
- **Что сделано:** последний «глухой» `except OSError: pass` заменён на
  `log.debug(..., exc_info=True)` при чтении lockfile в ожидании sentinel;
  остальные `except` в apply_* уже логировались. Поведение не изменено.
- **Проверка:** `pytest tests/test_mamba3_mimo_p1_patches.py -q` → 3 passed;
  `rg -n "except.*:\\s*pass" cppmega/megatron/upstream_patches/apply_*.py` —
  пусто.

## [P009] Deprecation-заголовок для mamba3_author_spec.py — DONE
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: — (НЕ трогать
  `author_mamba3_spec.py` — он в активном фронте)
- **Где:** `cppmega/megatron/mamba3_author_spec.py:1-6`.
- **Что сделано:** docstring модуля уже содержит явную пометку
  «Deprecated 2026-08-01: replaced by ``author_mamba3_spec.py``; retained for
  historical reference and scheduled for removal» и упоминание env-гейта
  `deprecated_paths.require_deprecated_ack`.
- **Проверка:** `pytest tests/ -k deprecated_paths -q` → 1 passed, 18 skipped;
  `head -20 cppmega/megatron/mamba3_author_spec.py` содержит пометку.

## [P010] Решение по PsiV cache scaffold (Phase A замер или архив) — DONE
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: —
- **Где:** `docs/mamba3_mimo_p2_psiv_cache_design.md` §14 + addendum,
  `cppmega/megatron/mamba3_psiv_cache.py` (остаётся как deprecated scaffold),
  `tests/test_mamba3_psiv_cache.py`.
- **Что сделано:** принято решение об архивации: Phase A требует GPU-замера,
  локального GPU нет, работа не в активной hardware-очереди. В дизайн-док
  добавлен addendum «archived 2026-08-01» с условиями воскрешения
  (≥0.5% TFLOP/s win на H200/GB10). Модуль оставлен как deprecated scaffolding
  с gate `CPPMEGA_MAMBA3_P2_PSIV_CACHE` OFF; entrypoints по-прежнему
  `NotImplementedError` при включении gate.
- **Проверка:** `pytest tests/test_mamba3_psiv_cache.py -q` → 3 passed, 4 skipped;
  дизайн-док содержит addendum с обоснованием archived-статуса.

## [P011] Triage 22 warnings последнего локального прогона — DONE
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: —
- **Где:** вывод `pytest tests/test_document_isolation_cp.py
  tests/test_fa4_document_isolation.py tests/test_mxfp8_transpose_emit_ext.py`
  (22 warnings).
- **Что сделано:** классифицированы:
  - 2 `DeprecationWarning` — чужие (torch.jit._script, megatron.core.inference.contexts);
  - остальные `UserWarning` — Megatron об отсутствии optional deps (TE, Apex, absl-py).
  Ни один warning не исходит из cppmega. Задокументировано в
  `docs/upstream_bugs.md` (раздел «Local pytest DeprecationWarning triage»).
- **Проверка:**
  ```bash
  .venv/bin/python3 -m pytest tests/test_document_isolation_cp.py \
    tests/test_fa4_document_isolation.py tests/test_mxfp8_transpose_emit_ext.py -q \
    -W error::DeprecationWarning \
    -W "ignore::DeprecationWarning:torch.jit._script" \
    -W "ignore::DeprecationWarning:megatron.core.inference.contexts"
  ```
  → 10 passed, 6 skipped, 7 оставшихся UserWarning (все от зависимостей).

## [P012] Линт и типы для файлов, изменённых аудитом 2026-07-31/08-01 — DONE
- Репо: cppmega | Приоритет: P2 | Тип: chore | Зависит от: P004–P008
- **Где:** `cppmega/features/{mhc,mod,structure}/__init__.py`,
  `scripts/ci/repository_runner.py`, `cppmega/megatron/{structure_dataset_patch,
  moe_dispatcher_patch,selective_fp8_moe_patch,flashinfer_mxfp8_gemm}.py`,
  `scripts/cppmega_fp8_shim.py`, `tests/test_mxfp8_transpose_emit_ext.py`.
- **Что сделано:**
  - `ruff check <paths>` — чисто (4 E402 в `cppmega_fp8_shim.py` убраны путём
    переноса module-level imports в начало файла).
  - `mypy --ignore-missing-imports --follow-imports=skip <paths>` — чисто в 10
    target-файлах; исправлены типовые замечания в `repository_runner.py`,
    `structure_dataset_patch.py`, `moe_dispatcher_patch.py`,
    `flashinfer_mxfp8_gemm.py` и `cppmega_fp8_shim.py`.
  - Исправлена ошибочная проверка мёртвых weakref в финализаторе identity-cache:
    запись теперь удаляется только тем же объектом weakref.
- **Остаток вне P012:** без `--follow-imports=skip` mypy рекурсивно проверяет
  импортированный граф и на текущем baseline находит 73 ошибки в 14 других
  файлах; это не выдаётся за зелёный repo-wide mypy.
- **Проверка:**
  - `ruff check cppmega/features/mhc/__init__.py cppmega/features/mod/__init__.py cppmega/features/structure/__init__.py scripts/ci/repository_runner.py cppmega/megatron/structure_dataset_patch.py cppmega/megatron/moe_dispatcher_patch.py cppmega/megatron/selective_fp8_moe_patch.py cppmega/megatron/flashinfer_mxfp8_gemm.py scripts/cppmega_fp8_shim.py tests/test_mxfp8_transpose_emit_ext.py`
  - `mypy --ignore-missing-imports --follow-imports=skip <same paths>`
  - `pytest tests/test_mxfp8_transpose_emit_ext.py tests/test_moe_dispatcher_patch.py tests/test_structure_dataset_patch_source.py tests/test_structure_dataset_patch_bridge.py -q` → 30 passed, 5 skipped.

## [P013] Полный локальный прогон cppmega suite в общем venv (baseline) — DONE
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P012
- **Где:** весь `tests/` кроме GPU-gated.
- **Что сделано:** первоначальный красный baseline перепроверен в
  repository-owned venv `/Volumes/external/sources/.venvs/cppmega.source` с
  exact-pinned Megatron `ba7b5ebc` и cppmega.mlx `65b6889f`.
  Две настоящие причины ложного red baseline исправлены в `312f4749`:
  cross-repo identity checks больше не используют исчезнувший hard-coded
  checkout, а M2RNN import test корректно пропускается интерпретатором без MLX
  и не использует pytest runtime patching.
  Результаты зафиксированы в
  `outputs/reports/local_suite_baseline_2026_08_01.json`:
  - tested commit `312f4749d38061254b418c40ccb8007142623d43`;
    tested tree `61ca7b77c64b4621a9f9d0052b80e10298cacd06`;
  - 2760 items, 2584 passed, 0 failed, 176 skipped, 429.25s;
  - последующие коммиты до фиксации receipt меняли только этот backlog,
    проверенное дерево исходного кода и тестов не менялось.
- **Проверка:**
  - `jq '.status, .passed, .failed, .skipped' outputs/reports/local_suite_baseline_2026_08_01.json`
    → `"passed"`, 2584, 0, 176.
  - `python -m pytest tests/ -q -rf -p no:cacheprovider`
    в зафиксированном environment contract → 2584 passed, 176 skipped.
  - direct compile evaluations jobs 1, 3, 5 и 8 → 5/5 passed каждый;
    прежний 60s `clang++` timeout был конкуренцией полного прогона, а не
    основанием увеличивать timeout.

## [P014] Обновить docs/environment.md под shared venv + manifest — DONE
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** `docs/environment.md:1-230` (shared venv receipt, manifest, test formulas).
- **Что сделано:** раздел «Shared venv Receipt (2026-08-01)» уже описывает общий
  `.venvs/cppmega.mlx`, манифест `cppmega-environment.json` с
  `megatron_root=/Volumes/external/sources/Megatron-LM-test-e40feed4` и
  `megatron_commit=e40feed4a060a84cd4cd1e5096316cc487014c87`, причину выбора
  `test-e40feed4` вместо `core_v0.18.0` (ba7b5eb тянет Triton, недоступен на
  macOS), и формулы запуска тестов из обоих checkout.
- **Проверка:** свежая shell-сессия:
  `cd /Volumes/external/sources/cppmega && .venv/bin/python3 -m pytest tests/test_mxfp8_transpose_emit_ext.py -q`
  → 4 skipped, 20 warnings (все от зависимостей), без env-переменных.

---

# Фаза B. cppmega — CI и инфраструктура (P015–P024)

## [P015] Failure-receipt для subcommand `run` (оркестратор) — DONE
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: —
- **Где:** `scripts/ci/repository_runner.py:2047-2132` (`_write_early_failure_receipt`),
  вызовы в `main()` на `OSError/RepositoryCIError/subprocess.SubprocessError`
  (`:2124`) и на fallback `Exception` (`:2132`).
- **Что сделано:** `run`/orchestrate уже покрыт общим `_write_early_failure_receipt`,
  который пишет receipt в `--receipt-dir` с run_id, subcommand, exit_code и
  redacted-сообщением. Добавлены/расширены тесты:
  - `test_run_orchestrator_writes_failure_receipt_for_unknown_lane` (`tests/ci/test_repository_ci_runner.py:865`),
  - `test_run_early_failure_receipt_generates_run_id_when_missing` (`:908`),
  - `test_run_early_failure_receipt_does_not_clobber_an_existing_receipt` (`:931`),
  - `test_run_early_failure_receipt_redacts_project_token_formats` (`:1005`).
- **Проверка:** `CPPMEGA_TEST_PROFILE=portable-data .venv/bin/python -m pytest tests/ci -q` → 34 passed.

## [P016] Trap в ci-self-hosted.yml для pre-python падений — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: P073 (policy-тест
  `test_workflow_runner_policy.py` в активном фронте — править после его коммита)
- **Где:** `.github/workflows/ci-self-hosted.yml` (shell-преамбула: проверки
  `test -x python_bin`, git, verify_tokenizer_contract).
- **Что сделано:** преамбула обёрнута в `trap 'on_pre_python_error' ERR` для
  обеих job (`mac-contracts` и `linux-portable`). Trap пишет минимальный
  `workflow-preamble` receipt в `--receipt-dir` до вызова python и
  разоружается (`trap - ERR`) перед `run_repository_ci.py lane`. Добавлен
  policy-тест `test_linux_lane_writes_a_pre_python_failure_receipt`.
- **Проверка:**
  - `CPPMEGA_TEST_PROFILE=portable-data .venv/bin/python -m pytest tests/test_workflow_runner_policy.py -q` → 20 passed.
  - Exact self-hosted run `30704454462` на `3ad84ca4` завершил обе lanes
    успешно; наведённое live-падение preamble ещё не выполнялось.

## [P017] Job summary с receipt.json при падении lane — PARTIAL
- Репо: cppmega | Приоритет: P3 | Тип: task | Зависит от: P016
- **Где:** `.github/workflows/ci-self-hosted.yml:82-97` (macOS) и `:152-167` (Linux).
- **Что сделано:** добавлены шаги `Surface macOS/Linux lane failure receipt in job summary`
  с `if: failure()`, которые пишут `receipt.json` в `$GITHUB_STEP_SUMMARY`.
- **Что осталось:** выполнить наведённое падение и сохранить ссылку на run,
  где JSON действительно виден в GitHub Summary. Зелёный run этот failure-path
  не доказывает.
- **Проверка:**
  - `grep -n "Surface.*lane failure receipt" .github/workflows/ci-self-hosted.yml` → 2 строки.
  - Live failure receipt/summary — ещё нет.

## [P018] Свежие CI-диагностики: outputs/ci_diagnostics устарели — DONE
- Репо: cppmega | Приоритет: P2 | Тип: chore | Зависит от: —
- **Где:** `outputs/ci_diagnostics/`, `outputs/ci_diagnostics/README.md`,
  `scripts/fetch_ci_diagnostics.py`, `scripts/fetch_ci_logs.py`.
- **Что сделано:** оба источника диагностик обновлены 2026-08-01:
  - Upstream repo diagnostics: 15 `.jsonl` файлов, 7 из них непустые
    (`cgal`, `ChibiOS`, `clamav`, `Crow`, `dpdk`, `flash-attention`, `geant4`).
  - Lane receipts: `lane_receipts/macos-30676130389-1/` и
    `lane_receipts/linux-30673683695-1/` с валидными `receipt.json` и логами.
  - `README.md` задокументирован с датами refresh, командами `gh run download`
    и retention 14d.
- **Проверка:**
  - `find outputs/ci_diagnostics -maxdepth 1 -name '*.jsonl' -newer docs/backlog_plan_2026_08_01.md | wc -l` → 15.
  - `find outputs/ci_diagnostics/lane_receipts -type f | wc -l` → 12.
  - `python -c "import json; json.load(open('outputs/ci_diagnostics/lane_receipts/macos-30676130389-1/receipt.json'))"` — OK.
  - `python -c "import json; [json.loads(l) for l in open('outputs/ci_diagnostics/cgal.jsonl')]"` — OK.

## [P019] Контрольная сборка mamba_ssm wheel с verify-шагом — DONE
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P073 (build-wheels.yml
  в активном фронте)
- **Где:** `.github/workflows/build-wheels.yml` (matrix.patch +
  «Verify Mamba wheel contains the pinned GQA backward patch»).
- **Что сделано:** verify-шаг расширен до 6 маркеров (`cppmega@45763efa`);
  job `91366705130` run `30698977374` собрал `mamba_ssm` и прошёл встроенный
  verify. Артефакт скачан; все 6 маркеров независимо найдены ровно по одному.
  Ненужный хвост full-matrix run отменён после сохранения артефакта.
- **Проверка:**
  - wheel SHA256:
    `3fade64fc70c08a6f0a7fdad822f90666f579cfe16e8c027d2d8121ff35dbbac`;
  - receipt:
    `/Volumes/external/cppmega_data/receipts/github-ci/mamba-45763efa-run-30698977374/binding.json`
    (SHA256
    `f885ae806c174bc772d07e1405c86d45e02acb0d6f4af2e01dd52f024c17aead`).

## [P020] Новые isolation-тесты в CI lanes — DONE
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P073
- **Где:** `configs/ci/lanes.json`, lanes `macos-contracts` и `linux-contracts`.
- **Что сделано:** `tests/test_document_isolation_cp.py` и
  `tests/test_fa4_document_isolation.py` уже присутствуют в обеих lanes
  (macOS: focused-contracts, linux: portable-contracts). NCCL-кейсы
  остаются skipped на macOS — это нормально.
- **Проверка:**
  - `grep -E 'test_document_isolation_cp|test_fa4_document_isolation' configs/ci/lanes.json` → 3 строки.
  - Exact self-hosted run `30704454462` на `3ad84ca4`: macOS
    `1317 passed, 4 skipped`; Linux portable
    `946 passed, 18 skipped`.

## [P021] CUDA-покрытие isolation-тестов в linux-cuda lane — PARTIAL
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P020
- **Где:** `configs/ci/lanes.json`, lane `linux-cuda` (`commands.cuda-contracts.argv`).
- **Что сделано:** в lane `linux-cuda` добавлены `tests/test_document_isolation_cp.py`
  (NCCL / 2-GPU кейсы) и `tests/test_fa4_h200_parity.py` (GPU FA4 parity).
- **Что осталось:** live CUDA-runner receipt, доказывающий, что оба файла
  выполнялись на GPU и не были skipped. Наличие argv в конфиге этого не
  доказывает.
- **Проверка:**
  - `grep -n "test_document_isolation_cp\|test_fa4_h200_parity" configs/ci/lanes.json` → строки в `linux-cuda`.
  - CUDA-runner receipt — ещё нет.

## [P022] Policy-тест verify-шага: убрать хрупкий string-count — DONE
- Репо: cppmega | Приоритет: P3 | Тип: task | Зависит от: P073
- **Где:** `tests/test_workflow_runner_policy.py:349-364`
  (`test_mamba_wheel_build_applies_the_pinned_gqa_backward_patch`).
- **Что сделано:** вместо хрупкого `workflow.count(...)` используется парсинг
  verify-шага через `re.search` и проверка маркеров `mamba3_mimo_bwd.py` /
  `mamba3_mimo_bwd_varlen.py` в теле шага.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega && .venv/bin/python -m pytest tests/test_workflow_runner_policy.py::test_mamba_wheel_build_applies_the_pinned_gqa_backward_patch -q` → 1 passed.
  - Переформатирование YAML в verify-шаге без смены смысла не ломает тест.

## [P023] Failure-receipt: секреты-редакция под тестом на реальных паттернах — DONE
- Репо: cppmega | Приоритет: P3 | Тип: task | Зависит от: P015
- **Где:** `scripts/ci/repository_runner.py:496-527` (`_Redactor`),
  `tests/ci/test_repository_ci_runner.py:963-1038`.
- **Что сделано:** тесты на реальные форматы токенов уже добавлены:
  - `test_step_logs_redact_project_token_formats` (`:970`) — проверяет, что
    `run_step` не протаскивает в лог GitHub classic PAT (`ghp_...`),
    fine-grained PAT (`github_pat_...`), Nebius API key и Modal token id/secret.
  - `test_run_early_failure_receipt_redacts_project_token_formats` (`:1005`) —
    проверяет, что `_write_early_failure_receipt` редуцирует те же форматы в
    `orchestration.json` и `orchestrator-failure.log`.
- **Проверка:**
  - `CPPMEGA_TEST_PROFILE=portable-data .venv/bin/python -m pytest tests/ci/test_repository_ci_runner.py::test_step_logs_redact_project_token_formats tests/ci/test_repository_ci_runner.py::test_run_early_failure_receipt_redacts_project_token_formats -q` → 2 passed.
  - `grep -R "nebius-unit-test-key\|ak-unitTEST\|as-unitTEST\|github_pat_11ABCDEFGH0" outputs/ci_diagnostics/lane_receipts/` → пусто (тестовые токены не остались в receipt'ах).

## [P024] Документирование CI-архитектуры: configs/ci/README.md — DONE
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: P016, P020
- **Где:** `configs/ci/README.md`, `.github/workflows/ci-self-hosted.yml`.
- **Что сделано:** README существует и обновлён: раздел «Failure receipts»
  теперь описывает обе workflow job (macOS и Linux), failure-receipt для
  pre-python преамбулы, добавление теста в lane и формулы локального
  воспроизведения.
- **Проверка:** `cat configs/ci/README.md | grep -c "Linux lane"` ≥ 1;
  `configs/ci/README.md` содержит разделы «Receipts», «Failure receipts»,
  «Adding a test to a lane», «Local reproduction».

---

# Фаза C. Данные — блокеры релиза (P025–P040)

Канонический статус: `docs/status/training_data_inventory.md` (оба репо).
Живые артефакты: `mlx/outputs/training_data_status/current.json` + `changelog.jsonl`
(watcher `scripts/report_training_data_status.py` уже бежит).

## [P025] DirectXTK case-fold double-count (-453k токенов) — DONE
- Репо: mlx | Приоритет: P1 | Тип: bug | Зависит от: —
- **Где:** `scripts/streaming_conveyor.py:2819-2889`
  (`claim_code_project_identity`, `claim_code_repo_for_submission`),
  `tests/test_streaming_conveyor_progress.py`.
- **Что сделано:**
  - Bare-name aliases сначала разрешаются через frozen repo-list в канонический
    `project_identity`, затем один project может быть принят только один раз;
    остальные aliases получают явный receipt/event
    `duplicate_project_identity_skipped`.
  - В repo-list v14 `DirectXTK` и `directxtk` оба связаны с
    `microsoft/DirectXTK`. `filesystem_repo_key` намеренно сохраняет раздельные
    collision-safe пути: casefold имени файла здесь создал бы риск
    перезаписи commit/code outputs вместо канонической дедупликации.
  - Кодовый фикс существует с `cppmega.mlx@500932ae`; текущий source conveyor
    на revision `73372dcc` его содержит.
- **Что осталось:** P026 — дождаться физического нового receipt и доказать
  ожидаемое уменьшение на 453,368 токенов / 215 строк. Это не часть code-fix
  P025 и ещё не завершено.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_streaming_conveyor_progress.py -q` → 59 passed.
  - `jq -r '.by_bare_name.DirectXTK, .by_bare_name.directxtk' repo_list_v14...`
    → две одинаковые строки `microsoft/DirectXTK`.

## [P026] Пересчёт receipt'ов затронутых доменов после P025
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P025
- **Где:** `outputs/training_data_status/` (через watcher), скрипты пересчёта
  в `scripts/data/`.
- **Что делать:** пересчитать receipt'ы всех доменов с case-вариациями путей
  (не только DirectXTK); записать дельту в `changelog.jsonl`.
- **Проверка:** `current.json` обновлён; суммарный valid-токены согласован с
  поэлементным пересчётом; запись в changelog.jsonl содержит причину.

## [P027] Failed units в source conveyor
- Репо: mlx | Приоритет: P1 | Тип: bug | Зависит от: —
- **Где:** `outputs/training_data_status/current.json` (список failed units),
  conveyor-код в `cppmega_mlx/data/`.
- **Что делать:** выгрузить failed units, классифицировать по классам ошибок,
  починить топ-классы; для нечинимых — явный quarantine с причиной.
- **Проверка:** failed count → 0 или каждый failed unit имеет quarantine-receipt
  с причиной; отчёт в `current.json`.

## [P028] PR store материализация до eligible Parquet
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P027
- **Где:** PR store (`outputs/pr_ingest/` в cppmega, PR-датасет в mlx),
  `docs/status/training_data_inventory.md` (PR staged, 0 eligible).
- **Что делать:** возобновить отменённую материализацию: export в eligible
  Parquet по существующей схеме пяти бакетов.
- **Проверка:** в `current.json` у PR-датасета eligible > 0; Parquet читается
  `cppmega_mlx/data` ingress'ом (smoke-тест чтения).

## [P029] Мониторинг завершения exhaustive CI fetch
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** два бегущих `scripts/ci_stream_fetch.py` (PID 82218, 45427),
  вывод в `outputs/ci_stream_prod_v4_20260422_20260721/` и
  `outputs/ci_stream_prod_v4_20260721_20260730/`.
- **Что делать:** добавить/проверить completion-маркер и receipt по окончании
  fetch (fetch не exhaustive — блокер из inventory). Не убивать процессы.
- **Проверка:** после завершения процессов существует receipt с финальными
  счётчиками; `current.json` отражает exhaustive=true для CI-датасета.

## [P030] CI cross-store global dedup
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P029
- **Где:** `cppmega_mlx/data/` (dedup-логика), CI store outputs.
- **Что делать:** реализовать глобальную дедупликацию между CI-store'ами
  (блокер «нет cross-store global dedup»).
- **Проверка:** dedup-статистика в `current.json`; тест на синтетических
  пересекающихся сторах.

## [P031] CI five-bucket Parquet export
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P030
- **Где:** `cppmega_mlx/data/` (export), `docs/status/training_data_inventory.md`
  (блокер «не запущен five-bucket export»).
- **Что делать:** запустить экспорт CI-датасета в пять бакетов Parquet.
- **Проверка:** бакеты на диске, счётчики сходятся с `current.json`; smoke-чтение
  через Megatron-indexed ingress.

## [P032] Запечатывание live-набора (3.525B) в sealed bundle v2
- Репо: both | Приоритет: P1 | Тип: task | Зависит от: P026, P028, P031
- **Где:** процедура запечатывания из `macro_routes_v1_20260713`
  (4.133B — единственный production-ready на сегодня), `scripts/data/`.
- **Что делать:** после закрытия блокеров запечатать live-набор: bundle +
  sidecar'ы + receipt; опубликовать в оба репо.
- **Проверка:** bundle проходит ту же валидацию, что v1 (контракт sidecar'ов,
  счётчики, SHA); запись в `docs/status/training_data_inventory.md` обоих репо.

## [P033] Пересборка global_symbols.sqlite (0 байт) — A1 DONE / A2 PENDING
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `outputs/crossrepo/global_symbols.sqlite`,
  `scripts/crossrepo/build_global_symbol_index.py:_validate_symbol_identity_claims`
  и `tools/clang_indexer/index_project.py:canonical_symbol_identity`.
- **Что сделано:**
  - Все девять A1-проходов (`abseil,boost,folly,openssl,boringssl,protobuf,
    eigen,fmt,glib`) завершены: `9/9 done`, `0 errors`; schema-v5 SQLite
    занимает `580 MiB`, содержит `213777 symbols` и `197435`
    identity-записей.
  - Исправлена потеря declaration `column` между clang records и SQLite;
    schema поднята с v4 до v5, reader и reopen-валидация требуют тот же столбец.
  - External-provider key при reopen теперь проверяется с сохранёнными
    `provider`/`include_provenance`; добавлен реальный SQLite round-trip,
    включая fail-closed проверку несовпавшего column.
  - Reopen-валидация и настоящий base16k smoke прошли; временный `482 MiB`
    extraction cache после проверки удалён.
- **Что делать:** выполнить A2 (`std,libc`) с `--per-lib-pass` / ограничением
  файлов.
- **Проверка:**
  - `sqlite3 outputs/crossrepo/global_symbols.sqlite "select lib, done, n_funcs, n_types from symbol_index_libs order by lib;"` — все A1 libs `done=1`.
  - Закрыть/повторно открыть SQLite и проверить schema v5 без identity drift.
  - `scripts/crossrepo/export_base16k_sampler.py` больше не падает fail-closed.

## [P034] base16k sampler smoke после P033 — DONE
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P033
- **Где:** `scripts/crossrepo/export_base16k_sampler.py`.
- **Что сделано:** исправлен общий контракт `_base_doc` в `cppmega` и
  `cppmega.mlx`: каждый документ теперь несёт symbol identity schema v3 и
  канонический key/ID функции в collision registry и плотном symbol sidecar.
- **Проверка:** end-to-end smoke (`global_symbols.sqlite → tokenizer →
  materialize → best-fit pack → ZSTD`) дал `12 docs`, `5 rows`,
  `69942 valid tokens`; все 25 плотных sidecar-колонок выровнены на `16384`,
  в каждой row есть ненулевые `token_symbol_ids`,
  dedup содержит `4 keys × 3 claims`. Parquet:
  `/Volumes/external/cppmega_data/receipts/global-symbol-index-v5-3704d0aa/base16k-smoke-v4/parquet/16384/base16k_smoke_v4.parquet`,
  SHA256 `844eb792cff7f8015945843b6e76945aba2a215ba7af3e204e9c8218b7bcb9d2`.

## [P035] Staleness-алерт для training_data_status watcher — DONE
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: —
- **Где:** `scripts/report_training_data_status.py:1204-1387` (`_live_upstream_inputs`,
  `collect_freshness`, `check_heartbeat`, `build_status`),
  `configs/launchd/ai.cppmega.training-data-status.plist`.
- **Что сделано:** freshness-секция уже пишется в `current.json`:
  - `freshness.upstreams` — возраст каждого live upstream input и флаг `stale`;
  - `freshness.heartbeat` — PID, `recorded_at`, `age_seconds`, флаг `stale`;
  - `freshness.stale` — список stale upstream'ов.
  Главный цикл (`main`:1660-1749) при наличии stale upstream'ов или stale
  heartbeat пишет warning в `changelog.jsonl`.
- **Проверка:**
  - `python3 -c "import json; d=json.load(open('cppmega.mlx/outputs/training_data_status/current.json')); print(d['freshness']['stale'], d['freshness']['heartbeat']['stale'])"`.
  - `ps aux | grep report_training_data_status` — watcher запущен (PID меняется).
  - `tail cppmega.mlx/outputs/training_data_status/changelog.jsonl` — при stale
    условиях появляются записи с `warning: stale_upstreams` или
    `warning: stale_watcher_heartbeat`.

## [P036] Паритет token_* колонок между репо (mirror_mlx_parquet)
- Репо: both | Приоритет: P2 | Тип: task | Зависит от: P026
- **Где:** `scripts/data/mirror_mlx_parquet.py` (зеркалит mlx token_* колонки
  в cppmega-parquet).
- **Что делать:** прогнать зеркалирование после пересчётов; добавить
  checksum-тест паритета колонок.
- **Проверка:** checksum-тест зелёный; расхождения (если есть) задокументированы.

## [P037] Синхронизация training_data_inventory.md в обоих репо
- Репо: both | Приоритет: P2 | Тип: chore | Зависит от: P032
- **Где:** `cppmega/docs/status/training_data_inventory.md`,
  `mlx/docs/status/training_data_inventory.md`.
- **Что делать:** после закрытия блокеров обновить оба файла (блокеры → closed,
  новые sealed-счётчики); устранить кросс-репо указатель на живые артефакты
  (хрупкость из аудита) — заменить на версированный receipt.
- **Проверка:** `rg "outputs/training_data_status" cppmega/docs` — нет прямых
  ссылок на чужой checkout; даты и счётчики совпадают в обоих файлах.

## [P038] Регрессионный тест Megatron .bin/.idx ingress на новом bundle
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P032
- **Где:** `cppmega_mlx/data/megatron_indexed.py`,
  `docs/megatron_indexed_ingress.md`.
- **Что делать:** тест чтения sealed bundle v2 (не только синтетика): первые/
  последние токены, sidecar-колонки, document IDs.
- **Проверка:** тест зелёный в mlx suite; падает при подмене .idx (fail-closed).

## [P039] Консистентность prompt graph index/provenance с новыми данными
- Репо: cppmega | Приоритет: P3 | Тип: task | Зависит от: P032
- **Где:** `cppmega/prompt_graph_index.py`, `cppmega/prompt_graph_provenance.py`.
- **Что делать:** прогнать связанные тесты на новом наборе; проверить, что
  provenance-ссылки указывают на существующие receipt'ы.
- **Проверка:** `pytest tests/ -k "prompt_graph" -q` зелёный; 0 висячих ссылок.

## [P040] Data release checklist (воспроизводимый статус) — DONE
- Репо: both | Приоритет: P2 | Тип: chore | Зависит от: P025
- **Где:** `docs/data_release_checklist.md` + ссылки в
  `docs/status/README.md` и `docs/status/training_data_inventory.md` обоих
  репозиториев.
- **Что сделано:** чеклист из 5 блокеров inventory с командами проверки каждого
  и combined single-command gate. Текущий прогон честно показывает 4 открытых
  блокера (Python aux mixed, PR not eligible, CI not ready, sealed bundle not
  ready); DirectXTK case-fold collision resolved.
- **Проверка:**
  - Выполнить пять `jq -e` gate-команд и combined gate из
    `docs/data_release_checklist.md`; текущий результат: gate 1 проходит,
    gates 2–5 не проходят, существующий sealed-bundle gate проходит.

---

# Фаза D. cppmega.mlx — код (P041–P060)

## [P041] TileLang Phase 4.1: strict parity для mamba3 Path C
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** `tests/test_tilelang_path_c_vs_b_parity.py:296-322` (strict-xfail
  маркеры), `cppmega_mlx/nn/_tilelang/mamba3_path_c*`.
- **Что делать:** приземлить недостающие Path C applies для mamba3; снять
  xfail по одному, добиваясь strict-паритета с Path B.
- **Проверка:** xfail-маркеры mamba3 удалены; файл parity-тестов зелёный
  без xfail для mamba3.

## [P042] TileLang Phase 4.2: strict parity для sparse_mla Path C
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P041
- **Где:** `cppmega_mlx/nn/_tilelang/sparse_mla_path_c*`, тот же parity-файл.
- **Что делать/Проверка:** аналогично P041 для sparse_mla.

## [P043] TileLang Phase 4.3: strict parity для fp8_vecmat Path C
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P041
- **Где:** `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c*`.
- **Что делать/Проверка:** аналогично P041 для fp8_vecmat.

## [P044] TileLang Phase 4.4: strict parity topk_selector и остатки
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: P041
- **Где:** `cppmega_mlx/nn/_tilelang/` (topk_selector и любые оставшиеся
  xfail в parity-файле).
- **Что делать/Проверка:** аналогично P041; цель — файл
  `test_tilelang_path_c_vs_b_parity.py` без единого xfail.

## [P045] TileLang Phase 4.5: удаление Path B адаптера
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P044
- **Где:** `cppmega_mlx/nn/_tilelang/_mlx_runtime.py:628-876` (Path B raw-MSL
  адаптер).
- **Что делать:** удалить Path B и auto-routing на него; обновить routing-тесты
  на Path C only; bench-receipts перегенерировать.
- **Проверка:** `pytest tests/test_lint_mlx.py tests/test_kernel_policy.py
  tests/ -k tilelang -q` зелёный; `rg "path_b" cppmega_mlx/` пусто.

## [P046] TileLang Phase 4.6: удаление _msl_transform.py
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P045
- **Где:** `cppmega_mlx/nn/_tilelang/_msl_transform.py` (862 LOC).
- **Что делать:** удалить файл и все ссылки; обновить `__init__`/exports.
- **Проверка:** полный nn-suite (291 тест) зелёный; `rg "_msl_transform"` пусто.

## [P047] Successor-документ миграции TileLang
- Репо: mlx | Приоритет: P2 | Тип: chore | Зависит от: P046
- **Где:** новый `docs/tilelang_unified_pipeline_status.md`; `MIGRATION_PLAN.md`
  уже помечен superseded аудитом.
- **Что делать:** canonical-док: архитектура unified pipeline, что приземлено
  (Phase 1–4 с коммитами), форк `DatasunriseOU/tilelang` как anchor, как
  добавлять новое ядро.
- **Проверка:** `docs/status/README.md` ссылается на новый canonical-док;
  в MIGRATION_PLAN.md superseded-блок указывает на него.

## [P048] ATEN SDPA wiring (фича)
- Репо: mlx | Приоритет: P2 | Тип: feature | Зависит от: —
- **Где:** регистрация `_scaled_dot_product_flash_attention_for_cpu` поверх
  TileLang FA-ядра из `/Volumes/external/sources/tilelang/poc/torch_dynamo/_kernels/flash_attention.py`.
- **Что делать:** кросс-репо фича: `torch.library` регистрация, fallback на
  math-backend при неподдерживаемых формах; parity-тесты против SDPA math.
- **Проверка:** новые тесты зелёные; `torch.nn.functional.scaled_dot_product_attention`
  на CPU использует ядро (профиль/probe), fallback покрыт тестом.

## [P049] reduce_prod C++ pass bug (upstream TVM-форк)
- Репо: mlx | Приоритет: P2 | Тип: bug | Зависит от: — (ждать, пока сабмодуль
  `tilelang/3rdparty/tvm` станет чистым — там активная работа)
- **Где:** `/Volumes/external/sources/tilelang/3rdparty/tvm/src/tirx/transform/vectorize_loop.cc`,
  `storage_rewrite.cc`; гейт `poc/triton_frontend/op_mapping.py::_USE_LOGEXP_PROD`.
- **Что делать:** минимальный repro → фикс в форке (или issue с repro);
  снять xfail в харнессе tilelang-репо.
- **Проверка:** нативный reduce_prod проходит per-backend validation; xfail снят.

## [P050] xfail triage: test_mamba3_chunked_backward_b0b1b2 — DONE
- Репо: mlx | Приоритет: P2 | Тип: bug | Зависит от: —
- **Где:** `tests/test_mamba3_chunked_backward_b0b1b2.py:344,371`.
- **Что сделано:** torch-MPS positional-buffer stage harness стабильно падает на
  не-production конфиге `G=1,H=2,N=16` из-за command-buffer ordering hazard,
  который искажает чтение `dA_cumsum` (ошибка в `dx` до ~0.6). `xfail` заменён
  на `strict=True`; причина и симптомы детально описаны в reason. Production MLX
  route (`mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd`) остаётся зелёным
  в `tests/test_mamba3_path_c_chunked_vs_path_b.py`.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_mamba3_chunked_backward_b0b1b2.py -q` → `11 passed, 2 xfailed`.
  - `rg "strict=True" tests/test_mamba3_chunked_backward_b0b1b2.py` — найдено.

## [P051] xfail triage: test_galcov_stage_d (12 штук) — DONE
- Репо: mlx | Приоритет: P3 | Тип: task | Зависит от: —
- **Где:** `tests/v4/test_galcov_stage_d.py:71-167` (gallery fixture),
  `:198-206` (`test_gallery_fixture_known_gaps_are_zero`).
- **Что сделано:** все 5 прежних gaps (GPT-2 XL, Tiny Aya, xLSTM 7B, Gemma 4
  E2B/E4B) замаплены на presets в V7-Q04; условные `pytest.xfail` в
  `:245-246`, `:262-263`, `:285-286` стали мёртвым кодом (preset всегда не
  None). Добавлен regression-тест `test_gallery_fixture_known_gaps_are_zero`,
  который fail-loud при появлении новых unmapped entries.
- **Проверка:** `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/v4/test_galcov_stage_d.py -q` → 218 passed; `rg 'xfail.*strict=False' tests/v4/test_galcov_stage_d.py` — пусто.

## [P052] Starlette/httpx testclient deprecation в v4-тестах — DONE
- Репо: mlx | Приоритет: P3 | Тип: chore | Зависит от: —
- **Где:** `tests/v4/`.
- **Что сделано:** в текущем дереве v4-suite не выдаёт Starlette/httpx deprecation
  warnings; оставшиеся 14 warnings исходят от `torch.jit.script_method` в
  `tests/v4/test_engram_v4.py` (upstream torch, не относится к этому пункту).
- **Проверка:** `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/v4 -q` → 3158 passed, 16 skipped, 14 warnings (все от torch).

## [P053] Paged attention compatibility path в serving (фича) — PARTIAL
- Репо: mlx | Приоритет: P2 | Тип: feature | Зависит от: —
- **Где:**
  - `cppmega_mlx/inference/serving.py:525-850` — `gather_paged_kv`,
    `scatter_paged_kv`, `_pad_or_trim_kv`; `require_model_integrated_paged_attention`
    переведён в `DeprecationWarning` (compatibility path доступен).
  - `cppmega_mlx/nn/attention.py:1076-1220` — `CausalSelfAttention.__call__`
    принимает `paged_kv_manager`, `paged_block_table`, `paged_seq_lengths`,
    `paged_layer_idx` и вызывает `_apply_paged_kv_compatibility_path`.
- **Что сделано:** реализован correctness-first scatter/gather paged KV →
  contiguous K/V через `mx.take` / per-block `mx.slice_update`, интегрирован в
  `CausalSelfAttention` для prefill-only GQA/MLA/full режимов. Follow-up
  `cppmega.mlx@cdbdacd4` сохраняет KV последовательностей вне текущего batch,
  маскирует mixed-length rows независимо и валидирует длины fail-closed.
  Decode закрыт в `cppmega.mlx@0286d667`: `scatter_paged_kv_offsets` пишет
  новые токены по абсолютным per-sequence offsets, `_project_qkv` получил
  per-sequence RoPE offsets (`rope_offsets`), paged-путь принимает
  `paged_seq_lengths` как тоталы после шага и строит per-row decode mask;
  смешанные batch (prefill+decode в одном forward) поддерживаются. Follow-up
  `cppmega.mlx@62a3e269` до любой записи fail-closed проверяет отсутствующие и
  aliased physical blocks во всём live prefix.
- **Что осталось:** Sparse-MLA fp8 Path B/C в paged-пути (не роутятся),
  нативный paged-attention kernel и in-place scatter ABI (убрать pool copy).
  До них это совместимый prefill+decode path, а не завершённый production
  paged attention.
  Dense-fallback для `mode='dsa'` разрешён в `cppmega.mlx@7a0378a5` (parity с
  contiguous prefill и decode покрыт тестами); `cppmega.mlx@eab00e33`
  fail-closed запрещает этот fallback при явно выбранном Sparse-MLA Path B/C.
  Native TileLang decode-scatter path приземлён в
  `cppmega.mlx@e2e0d1fe`: opt-in `backend='tilelang'` в
  `scatter_paged_kv_offsets` (pool copy + dst scatter через tvm_ffi,
  caller-owned outputs), bit-exact parity с host loop, fail-closed на
  dtype/backend; receipt
  `outputs/reports/paged_kv_offsets_scatter_bench_2026_08_01.json`
  (1.25–1.31x best, ~2x mean против host loop). Follow-up
  `cppmega.mlx@84a4b416` исправляет source-layout `[B,H,S,D]` при
  token-major dispatch и проверяет distinct K/V для каждого
  `(batch,head,token,dim)` настоящим Metal parity-тестом.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_inference_serving.py tests/test_attention.py -q` — 67 passed.
  - `pytest tests/test_hybrid_lm.py tests/test_dense_cpp_lm.py tests/test_dense_cpp_lm_grad_checkpoint.py -q` — 76 passed (регрессия).
  - `pytest tests/test_attention.py::test_paged_kv_compatibility_path_decode_matches_contiguous_cache tests/test_attention.py::test_paged_kv_compatibility_path_decode_mixed_batch -q` — parity с contiguous cache decode.
  - `pytest tests/test_inference_serving.py -k scatter_paged_kv_offsets -q` — offsets-scatter round-trip + fail-closed кейсы.

## [P054] DenseCppLM rope_only режим — DONE
- Репо: mlx | Приоритет: P1 | Тип: feature | Зависит от: —
- **Где:** `cppmega_mlx/models/dense_cpp_lm.py:103-106,502-506,582-588`
  (`DenseCppLMConfig.rope_only`, `DenseCppLM.position_embedding`, `embed()`),
  `scripts/convert_megatron_dense500m_torchdist_to_mlx.py:1497` (`rope_only: true`
  в `model.json`).
- **Что сделано:** добавлен конфиг-флаг `rope_only`. Когда он `True`,
  `DenseCppLM` не создаёт `position_embedding` и не применяет learned position
  table; конвертер выставляет `rope_only=True` и не эмитит
  `position_embedding.weight`. Добавлены parity-тесты: `rope_only` модель
  совпадает с обычной моделью с занулённой position table; `model.json`
  содержит `rope_only`; веса не содержат `position_embedding.weight`.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_dense_cpp_lm.py tests/test_convert_megatron_dense500m_torchdist_to_mlx.py -q` → зелёный.
  - `rg "rope_only" cppmega_mlx/models/dense_cpp_lm.py scripts/convert_megatron_dense500m_torchdist_to_mlx.py` — флаг проброшен.

## [P055] Совместимость generic generation API с DenseCppLM — DONE
- Репо: mlx | Приоритет: P2 | Тип: bug | Зависит от: P054
- **Где:** `cppmega_mlx/models/dense_cpp_lm.py:1056` (tuple output `(logits, None)`)
  vs `cppmega_mlx/inference/generation.py:105-113`/`804-827` (`next_token_logits` /
  `_standard_generation_logits` адаптер); обходной путь `logits, _loss = model(...)`
  в `scripts/cpp_jsonl_generation_compile_eval.py:1340`.
- **Что сделано:**
  - Адаптер в `generation.py` разворачивает `(logits, None)`; проверено, что
    `generate_tokens(DenseCppLM, ...)` работает.
  - Обходной путь в eval-скрипте заменён на вызов адаптера:
    `next_token_logits(model(...), input_ids)`; `_sample_next` теперь получает
    `(B, V)` logits напрямую.
  - Добавлен regression-тест `test_dense_cpp_lm_works_with_generic_generation_api`
    в `tests/test_dense_cpp_lm.py`.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_dense_cpp_lm.py::test_dense_cpp_lm_works_with_generic_generation_api -q` → 1 passed.
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_dense_cpp_lm.py tests/test_inference_generation.py tests/test_cpp_jsonl_generation_compile_eval.py -q` → 147 passed.

## [P056] Wave-Next 1: side-channel preservation в Megatron-конвертере — DONE
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: —
- **Где:** `scripts/convert_megatron_dense500m_torchdist_to_mlx.py` и
  `cppmega_mlx/data/batch.py` (mapping `NGRAM_SOURCE_TO_TARGET` /
  `STRUCTURE_SOURCE_TO_TARGET`); `conversion_runtime_requirements` теперь
  содержит `side_channels` receipt.
- **Что сделано:** конвертер сохраняет side-channel колонки (ngram/structure)
  при переносе чекпоинта; добавлены тесты.
- **Проверка:** `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_convert_megatron_dense500m_torchdist_to_mlx.py tests/test_inference_serving.py -q` — 35/35 passed (20 конвертер + 15 serving).

## [P057] Wave-Next 2: archived JSON baseline для эвалов
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P067
- **Где:** `docs/porting_plan.md:449-482`; эвалы `outputs/evals/`.
- **Что делать:** зафиксировать baseline-эвалы как версионируемые JSON
  (в git или versioned store), с которыми сравниваются новые прогоны.
- **Проверка:** новый эвал-прогон выдаёт diff против baseline; тест схемы
  baseline-файла.

## [P058] Wave-Next 3: matched GB10 runner для паритета
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: —
- **Где:** `docs/porting_plan.md:449-482`.
- **Что делать:** единый runner-скрипт, гоняющий одинаковый workload на
  M4 Max и GB10 с идентичными seed/shape; receipt с обеих сторон.
- **Проверка:** два receipt с совпадающей конфигурацией; числовой diff
  зафиксирован (входит в P095).

## [P059] Wave-Next 4: checkpoint sharding для больших моделей — DONE
- Репо: mlx | Приоритет: P3 | Тип: feature | Зависит от: —
- **Где:** `cppmega_mlx/training/checkpoint.py`, `tests/test_checkpoint.py`.
- **Что сделано:** прототип из `wip/checkpoint-sharding-20260801@647b8036`
  повторно применён на main в `cppmega.mlx@acd857f2` и закрыты все три
  trust-boundary/data-loss сценария из аудита:
  - переключение sharded↔single теперь транзакционно: stale-layout файлы
    удаляются только после записи нового metadata.json, а shard index
    публикуется атомарно через tmp + `os.replace`; краш в любой точке
    оставляет прежний layout загружаемым;
  - `_load_shard_index` fail-closed: каждый `weight_map` value обязан
    соответствовать схеме `model-NNNNN-of-NNNNN.safetensors`; абсолютные
    пути, `..`, подкаталоги и чужие имена отклоняются до обращения к файлам;
  - `_remove_stale_sharded_weights` удаляет shard-файлы первыми, index —
    строго последним, и только при наличии index (чужие каталоги не трогаем).
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_checkpoint.py -q` → 100 passed (3 roundtrip-теста прототипа + 6 новых на cleanup ordering, unsafe index paths и крах до metadata-write в обе стороны).
  - `pytest tests/test_checkpoint.py tests/test_package_exports.py tests/test_tiny_train.py -q` → 116 passed.

## [P060] Полный mlx suite baseline (~7900 тестов) — DONE
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P052
- **Где:** `tests/` + `tests/v4/` (всё).
- **Что сделано:** полный прогон в общем venv
  `/Volumes/external/sources/.venvs/cppmega.mlx`.
  Результаты зафиксированы в
  `outputs/reports/mlx_full_suite_baseline_2026_08_01_rerun.json`:
  - tested commit `0ebf49a10b028ba37c4339f9a491057280eaca57`;
    tested tree `deb221aff245de651ad1adf9b52d3b486fa82c0d`;
  - 7860 passed, 70 skipped, 2 xfailed, 0 failed, 1790.17s.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/ -q --durations=50`
    → 7860 passed, 70 skipped, 2 xfailed, 0 failed.
  - `jq '.summary.passed, .summary.failed, .summary.skipped' outputs/reports/mlx_full_suite_baseline_2026_08_01_rerun.json`
    → 7860, 0, 70.

---

# Фаза E. Мост CUDA→MLX (P061–P072)

Контракт: `mlx/docs/mtr005_megatron_dcp_to_mlx.md`; конвертер
`mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py` (1515 строк).

## [P061] Атомарная публикация + SHA-256 в конвертере — DONE
- Репо: mlx | Приоритет: P1 | Тип: bug | Зависит от: —
- **Где:** `mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py:355-369`
  (`_sha256_receipt`), `:1495-1511` (atomic `os.replace` pair),
  `:1513-1551` (`published_checkpoint_status`).
- **Что сделано:**
  - `model.json` пишется как staging-файл рядом с весами, затем оба файла
    атомарно публикуются через `os.replace`.
  - `model.json["publish"]` содержит `completion_marker` и `weights_sha256`.
  - `published_checkpoint_status` fail-closed: отсутствие маркера,
    отсутствие `weights_sha256` или несовпадение SHA возвращает `complete: False`
    с причиной.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_convert_megatron_dense500m_torchdist_to_mlx.py -q` → 20 passed.
  - Тесты `test_publish_receipt_records_weights_sha256_and_completion`,
    `test_published_checkpoint_status_rejects_weights_without_completion_marker`,
    `test_published_checkpoint_status_detects_weights_sha_mismatch` — зелёные.

## [P062] Перегенерация 6 mlx_converted чекпоинтов
- Репо: both | Приоритет: P1 | Тип: task | Зависит от: P061, P054
- **Где:** `cppmega/outputs/checkpoints/mlx_converted/` (6 чекпоинтов от
  2026-06-30/07-01 — произведены ДО parity gate 2026-07-14, без
  `logit_parity` и SHA; все MLX-эвалы 30 июня опираются на них).
- **Что делать:** перегенерировать все 6 текущим конвертером из исходных
  torch_dist чекпоинтов (они на месте в `outputs/checkpoints/`).
- **Проверка:** каждый model.json содержит `logit_parity` receipt и SHA-256;
  parity atol=4e-3 подтверждён; старые чекпоинты удалены/помечены superseded.

## [P063] CI-гейт конвертера на реальном H200-чекпоинте
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P062
- **Где:** `scripts/run_self_hosted_ci.py:45` (тест конвертера уже в CI, но
  все фикстуры синтетические на tmp_path).
- **Что делать:** CI-джоба (или staged-процедура) с реальным срезом
  H200-чекпоинта: конверсия + parity gate.
- **Проверка:** CI run зелёный с реальным чекпоинтом; receipt в artifacts.

## [P064] Graph-route parity: реальные sidecar в MLX-эвале
- Репо: both | Приоритет: P1 | Тип: bug | Зависит от: P062
- **Где:** эвалы `cppmega/outputs/evals/local_mlx_*` (гоняются с
  `require_graph_routes=False`, нулевыми sidecar, без `block_bias` —
  «graph-route» по факту не проверяется; lane-04 review High #2/#3).
- **Что делать:** подать реальные graph sidecar в MLX-эвал, включить
  `block_bias`, сверить logits с CUDA graph-bias на тех же промптах.
- **Проверка:** parity receipt с max-abs diff; `require_graph_routes=True`
  в конфиге эвала.

## [P065] Решение по конверсии DSA/mamba3/MoE чекпоинтов — DONE
- Репо: both | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** `mlx/docs/status/mamba3_dsa_mlx_conversion_decision.md`,
  `mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py:104-110`.
- **Что сделано:** decision-док зафиксирован 2026-08-01: **Option C — staged
  partial port** в порядке mamba3 → DSA → MoE → MTP. Обоснование: CUDA-трек
  меняется, big-bang неустойчив; MLX reference-блоки уже есть, каждый этап —
  преимущественно key-mapping + parity gate; MTP откладывается, так как нет
  inference consumer. Финальный go/no-go rests with the owner; при «go»
  создать bd epic `cppmega-mlx-c30.1` с подзадачами.
- **Проверка:**
  - `ls cppmega.mlx/docs/status/mamba3_dsa_mlx_conversion_decision.md` — файл существует.
  - `grep -n "Option C" cppmega.mlx/docs/status/mamba3_dsa_mlx_conversion_decision.md` — строка 98.
  - `grep '"title":"\[P065\]' cppmega.mlx/.beads/issues.jsonl` — status `closed`.

## [P066] Packed-document n-gram parity для сконвертированных моделей
- Репо: both | Приоритет: P2 | Тип: task | Зависит от: P062
- **Где:** mtr005-док и конвертер `:1390` («not claimed»).
- **Что делать:** дизайн + тесты packed-document n-gram parity (связано с
  фронтом document isolation на CUDA-стороне — сверить контракты после P073).
- **Проверка:** тест parity зелёный на сконвертированном чекпоинте; mtr005-док
  обновлён (пункт «not claimed» снят).

## [P067] Перезапуск локальных MLX-эвалов на проверенных весах
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P062
- **Где:** `cppmega/outputs/evals/local_mlx_*` (~25 прогонов 2026-06-30/07-01,
  последний `local_mlx_fixed_stage05_current_graph_routes_docstring_clang_greedy`,
  compile 0/4).
- **Что делать:** перегнать эталонный набор эвалов на переконвертированных
  чекпоинтах; сравнить compile_pass с прежними результатами.
- **Проверка:** сводный отчёт «старые веса vs новые» в `outputs/reports/`;
  расхождения объяснены.

## [P068] Канонизация tokenizer.json (1-байтное расхождение) — DONE
- Репо: both | Приоритет: P2 | Тип: chore | Зависит от: —
- **Где:** `cppmega/data/tokenizer_v2/tokenizer.json` (канон) и
  `mlx/cppmega_mlx/tokenizer/tokenizer.json` (vendored copy).
- **Что сделано:** файлы байт-идентичны; SHA-256
  `e6a8ad2d2cc74af05c75ed4338b6649a301122d942683e0a8cf44393a185296a`.
  Процедура синка задокументирована в
  `mlx/tests/test_tokenizer_contract.py:314-318`: канон — в `cppmega`,
  при drift перекопировать канон в `cppmega.mlx`.
- **Проверка:**
  - `sha256sum cppmega/data/tokenizer_v2/tokenizer.json cppmega.mlx/cppmega_mlx/tokenizer/tokenizer.json` — совпадают.
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_tokenizer_contract.py::test_cppmega_tokenizer_v2_matches_vendored_artifact_when_available -q` → 1 passed.

## [P069] Обновление outputs/megatron_ready под sealed bundle v2
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P032
- **Где:** `outputs/megatron_ready/` (.bin/.idx + 24 token_* sidecar от 2026-06-29).
- **Что делать:** перегенерировать Megatron indexed dataset из sealed v2.
- **Проверка:** H200-тренировка smoke читает новый датасет; sidecar-контракт
  валиден (тесты `test_domain_megatron_sidecars.py` и др.).

## [P070] MTP/STP objective parity CUDA↔MLX
- Репо: both | Приоритет: P2 | Тип: task | Зависит от: P062
- **Где:** `cppmega_mlx/training/objectives` (MTP/STP/indexer losses) vs
  CUDA-тренер (Megatron-сторона).
- **Что делать:** числовой parity-тест objective'ов на одинаковом батче
  (NumPy-референс), зафиксировать допуски.
- **Проверка:** parity-тест зелёный в mlx suite; допуски задокументированы.

## [P071] Обновление mtr005-документа до текущего состояния — DONE
- Репо: mlx | Приоритет: P3 | Тип: chore | Зависит от: P061–P064
- **Где:** `docs/mtr005_megatron_dcp_to_mlx.md`.
- **Что сделано:** документ дополнен разделами:
  - Atomic publication (staging + `os.replace` + `published_checkpoint_status`).
  - RoPE-only position encoding (`rope_only=True`, отсутствие
    `position_embedding.weight`).
  - Graph-route parity (real sidecars, NumPy reference gate, single-document
    scope).
  - Side-channel preservation (`runtime_requirements.side_channels`).
  - Beyond dense GQA — ссылка на staged partial port decision (P065).
- **Проверка:**
  - `rg -n "Atomic publication|RoPE-only|Graph-route parity|Side-channel preservation|staged partial port" docs/mtr005_megatron_dcp_to_mlx.md` — разделы присутствуют.
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_convert_megatron_dense500m_torchdist_to_mlx.py -q` — 20 passed.

## [P072] Машиночитаемый индекс сконвертированных чекпоинтов — PARTIAL
- Репо: both | Приоритет: P3 | Тип: task | Зависит от: P062
- **Где:** `cppmega.mlx/scripts/generate_mlx_converted_index.py`,
  `cppmega.mlx/tests/test_mlx_converted_index.py`,
  `cppmega/outputs/checkpoints/mlx_converted/index.json`.
- **Что сделано:** добавлен генератор индекса, который сканирует
  `outputs/checkpoints/mlx_converted/`, читает каждый `model.json`, записывает
  `id → path, source_checkpoint, schema, dtype, weights_sha256, has_logit_parity,
  has_publish_receipt, rope_only` и fail-closed статус `ready` / `superseded`.
  Publish receipt принимается только при совпадении SHA с физическими весами;
  запись индекса атомарна. Текущий индекс v2 содержит 7 записей: ready 0,
  superseded 7; SHA-256 индекса
  `20e0180d679a87d9b5b348e3330779e64431eec5638f1542bf1a79b2d3e4cf7f`.
- **Что осталось:** выполнить P062 и научить eval-скрипты выбирать checkpoint
  по id из индекса. До этого исходный контракт P072 не закрыт.
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_mlx_converted_index.py -q` → 2 passed.
  - `jq '{schema,count,ready,superseded}' /Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted/index.json`
    → schema v2, count 7, ready 0, superseded 7.

---

# Фаза F. После приземления активного фронта (P073–P084)

Всё здесь зависит от P073 — коммита текущего грязного дерева (его делает
владелец фронта после H200-валидации).

## [P073] Разбить working tree фронта на коммиты — DONE
- Репо: cppmega | Приоритет: P1 | Тип: chore | Зависит от: — (железная
  валидация владельцем фронта)
- **Где:** грязное дерево: `document_isolation.py`, `fa4_*`, три миксера,
  `STACK.lock`, `build-wheels.yml`, 4 новых файла + параллельные правки
  (`docker/Dockerfile*`, `tests/test_cppmega_mamba3_tp_mixer.py`,
  `tests/test_document_isolation.py`).
- **Что сделано:** фронт приземлён в main; рабочее дерево больше не содержит
  uncommitted изменений по document isolation/FA4/mamba patch. Текущие
  незакоммиченные изменения относятся только к закрываемым в этом run
  пунктам P016/P020/P024/P033.
- **Проверка:**
  - `git status --short` в `cppmega` не показывает файлы фронта
    (`document_isolation.py`, `fa4_*`, `STACK.lock`, `build-wheels.yml`).
  - `git log --oneline -10` содержит коммиты фронта, включая `P065` и
    `P061`.

## [P074] Два пропущенных GPU-теста в Modal -k фильтр — DONE
- Репо: cppmega | Приоритет: P1 | Тип: bug | Зависит от: P073
- **Где:** `tests/test_fa4_h200_parity.py` (~строка 1400, Modal `-k` фильтр):
  вне фильтра `test_document_mask_rectangular_unaligned_decode_forward_backward_parity`
  (:783) и `test_graph_route_aux_multi_document_forward_backward_parity` (:1035).
- **Что сделано:** тест переименован, `_DEFAULT_H200_TEST_FILTER` обновлён в
  `cppmega@6da483b7`; `pytest --collect-only` находит 4 GPU-теста.
- **Проверка:** `cd /Volumes/external/sources/cppmega && .venv/bin/python -m pytest tests/test_fa4_h200_parity.py --collect-only -q | grep test_` — 4 GPU-теста;
  следующий Modal-прогон должен выполнить их все (мониторить квитанцию
  `fa4_parity.json`).

## [P075] Гейт §8 beta23: зафиксировать доказательства
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P073
- **Где:** `docs/fa4_beta23_upgrade_plan.md` (§8), `docs/fa4_beta23_score_mod_poc.md`,
  `docs/changelog.md`, заголовок плана (пометка «гейт не пройден» из аудита).
- **Что делать:** после H200-прогонов (API-проба, fwd/bwd-эквивалентность,
  20-step smoke) записать результаты; снять пометку «не пройден».
- **Проверка:** в poc-доке есть числа и квитанции; changelog обновлён.

## [P076] Обновить production_status.md
- Репо: cppmega | Приоритет: P1 | Тип: chore | Зависит от: P075
- **Где:** `docs/production_status.md` (апрельский канон: bench3/europe цифры).
- **Что делать:** добавить beta23-стек, document isolation, актуальные
  throughput-числа после H200-прогонов.
- **Проверка:** README docs/status по-прежнему называет его canonical — и это
  правда по содержимому.

## [P077] Guard сигнатур приватного API Megatron — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: P073
- **Где:** `cppmega/megatron/document_isolation.py:407-478`
  (`_assert_mamba_cp_signatures`), `tests/test_document_isolation.py:547-562`.
- **Что сделано:** guard уже реализован: проверяет сигнатуры
  `_redo_attention_load_balancing` и `_undo_attention_load_balancing` через
  `inspect.signature`, включая имена параметров, типы и return annotation.
  Тесты покрывают happy path и подмену сигнатуры.
- **Проверка:**
  - `.venv/bin/python -m pytest tests/test_document_isolation.py::test_mamba_cp_signature_guard_accepts_pinned_api tests/test_document_isolation.py::test_mamba_cp_signature_guard_rejects_changed_api -q` → 2 passed.
  - `pytest tests/test_document_isolation_cp.py -q` зелёный.

## [P078] Задокументировать контракт id(tensor) в _received_document_ids — DONE
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: P073
- **Где:** `cppmega/megatron/document_isolation.py:16-26`, `:1254`, `:1296-1301`
  (`_received_document_ids` и lookup в `isolated_set_input_tensor`).
- **Что сделано:** контракт уже задокументирован: docstring/комментарии
  объясняют, что ключ — exact tensor object identity, и любой `.clone()`,
  `.detach()` или не-contiguous `.contiguous()` копия меняет `id()` и ломает
  lookup. Поведение покрыто тестами document isolation.
- **Проверка:**
  - `rg -n "exact tensor object identity\|id(tensor)" cppmega/megatron/document_isolation.py` → есть совпадения.
  - `pytest tests/test_document_isolation_cp.py tests/test_document_isolation.py -q` → зелёный.

## [P079] Nebius DEFAULT_DOCKER_IMAGE bump на beta23
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P075
- **Где:** `scripts/nebius_h200_megatron_cpp_world_sweep.py:53` (старый digest
  `sha256:08c5db...`), план beta23 §9 (opt-in шаг).
- **Что делать:** отдельный PR: новый digest образа + smoke-прогон на Nebius.
- **Проверка:** smoke на Nebius с новым образом зелёный; digest в git.

## [P080] Повтор case6 по runbook
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: P079
- **Где:** `docs/case6_nebius_h200_runbook.md` (placeholders `REPLACE_WITH_...`).
- **Что делать:** заполнить runbook реальными значениями, выполнить case6,
  архивировать квитанции.
- **Проверка:** runbook без placeholders; квитанции в `outputs/nebius/`.

## [P081] Inference-cache для миксеров (фича)
- Репо: cppmega | Приоритет: P2 | Тип: feature | Зависит от: P073
- **Где:** `NotImplementedError` в `author_mamba3_spec.py:151,153,179`,
  `m2rnn_spec.py:269-335`, `cppmega_mamba3_tp_mixer.py:456-460,628`.
- **Что делать:** дизайн cache-контракта миксеров (state per layer, seqlen=1
  шаг), реализация, тесты parity vs полного прохода.
- **Проверка:** decode-шаг с cache совпадает с full-forward (численный тест);
  NotImplementedError сняты.

## [P082] varlen (cu_seqlens) для document isolation
- Репо: cppmega | Приоритет: P3 | Тип: feature | Зависит от: P073
- **Где:** `cppmega/megatron/document_isolation.py:195` (осознанный ponytail
  в `map_sequence_by_document`).
- **Что делать:** дизайн varlen-пути (cu_seqlens вместо mask_mod); оценить
  выигрыш против текущего mask-пути на реальных батчах.
- **Проверка:** дизайн-док + численный parity-тест прототипа (если реализуем).

## [P083] SWA window_size plumbing
- Репо: cppmega | Приоритет: P3 | Тип: feature | Зависит от: P075
- **Где:** `docs/long_context_roadmap.md` (~5 LOC в `mla_shared.py`, ждёт
  seq>8k фазы).
- **Что делать:** при старте seq>8k — провести `window_size` через конфиг в
  attention; тест на окне < seq.
- **Проверка:** тест: attention с window != full совпадает с референсом на
  усечённом окне.

## [P084] CP-порт для 128k-фазы (дизайн)
- Репо: cppmega | Приоритет: P3 | Тип: task | Зависит от: P083
- **Где:** `docs/long_context_roadmap.md` (CP отложен до 128k).
- **Что делать:** дизайн-док CP для 128k (у части механизма уже есть основа в
  document isolation CP — переиспользовать).
- **Проверка:** дизайн-док в `docs/`; оценка памяти/связки с Megatron CP API.

---

# Фаза G. Железозависимое (P085–P096)

Требует H200 (Modal/Nebius) или GB10. Координировать с владельцем фронта,
чтобы не конкурировать за железо.

## [P085] TE-форк: проверить quantize_rowwise_transpose на 3D и reshape shared-expert fc1
- Репо: external (TE-форк на GB10) | Приоритет: P1 | Тип: bug | Зависит от: —
- **Где:** форк TransformerEngine на GB10-хосте: `MXFP8Quantizer.quantize`/
  `update_quantized` (eager-emit ветка), reshape выхода `Linear.forward` для
  fc1 shared-эксперта (`save_original_input=True`, Megatron `shared_experts.py:74-84`).
- **Что делать:** проверить поведение на 3D-входе и reshape 2D/3D выхода —
  корень краша MoE `(4096,4,3584)` vs `(16384,3584)` из RESULTS.md. Фикс в форке.
- **Проверка:** юнит в форке на 3D-вход; затем P086.

## [P086] Перепрогон mxfp8_legacy на GB10
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P085
- **Где:** `runs/mxfp8_profile_compare/run_compare.sh` (legacy-профиль),
  `runs/mxfp8_profile_compare/RESULTS.md:48-82`.
- **Что делать:** прогнать legacy-контракт после фикса форка (и shim-фикса
  flatten из аудита); обновить RESULTS.md — закрыть или перефиксировать оба
  открытых бага (shape mismatch, misaligned address при dense_saved_operands=off).
- **Проверка:** 20/20 итераций без краша; RESULTS.md обновлён с датой и коммитом
  форка.

## [P087] Изоляция процессов для back-to-back mxfp8 прогонов
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: P086
- **Где:** `runs/mxfp8_profile_compare/RESULTS.md:209-216` (cuBLAS internal
  error при back-to-back в одном процессе).
- **Что делать:** раннер compare — subprocess per config.
- **Проверка:** полный compare (bf16+mxfp8, b4+b16) в одном вызове раннера без
  cuBLAS ошибок.

## [P088] fp8_amax CUDA-путь: контрольный прогон
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: — (CUDA-хост)
- **Где:** `cppmega/tests/test_fp8_amax_tilelang.py` (18 тестов + Triton-parity
  skip на macOS), `mlx/cppmega_mlx/nn/_tilelang/fp8_amax.py`.
- **Что делать:** прогнать на CUDA-хосте: Metal-обход (двухстадийный amax,
  uint8 encode) не должен затрагивать CUDA-ветку (atomic path).
- **Проверка:** все тесты зелёные включая Triton-parity; receipt приложен к
  issue.

## [P089] wave32 lane_b: перезапуск с env-флагом
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: — (H100/H200)
- **Где:** `artifacts/mamba3_wave32_lane_b_h100/` (r5: applier отказался
  мутировать без `MAMBA3_BWD_BWD_VECTORIZED_DIAG_ALLOW_FILE_MUTATION=1`;
  r6 умер на bench — выводы невалидны).
- **Что делать:** перезапустить с флагом, довести r6 до конца.
- **Проверка:** `report.json` с применённым gated-вариантом; вердикт по
  lane_b (оставить/откатить) зафиксирован в summary.md.

## [P090] grouped_head_reduce: оптимизировать или закрыть
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: — (H100/H200)
- **Где:** `artifacts/mamba3_wave32_grouped_head_reduce_h100/` (численно ок,
  Triton в 10–160× медленнее torch).
- **Что делать:** решение: CUDA-ядро (оптимизация) или закрыть как тупик с
  вердикт-доком (численность уже доказана).
- **Проверка:** либо bench в пределах 2× от torch, либо summary.md с вердиктом
  «closed as dead end» и ссылкой на замену.

## [P091] GHCR-образ: добавить fa3/fa4
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P073 (образ собирается
  из дерева с beta23-пинами)
- **Где:** `docker/Dockerfile*`, `artifacts/mamba3_wave32_h200_20step_gate/backend_probe.json`
  (`fa3_usable: false`, `fa4_usable: false`; только flash_attn 2.8.3).
- **Что делать:** собрать образ с fa3/fa4, probe-шаг в CI/сборке.
- **Проверка:** backend_probe в новом образе: `fa3_usable: true`,
  `fa4_usable: true`.

## [P092] wave32 20-step gate: перепрогон
- Репо: cppmega | Приоритет: P1 | Тип: task | Зависит от: P091
- **Где:** `artifacts/mamba3_wave32_h200_20step_gate/` (summary пустой,
  fallback: 1 шаг ok, loss 11.41, grad norm = NaN).
- **Что делать:** перепрогнать gate на новом образе; разобрать NaN grad norm
  в fallback-диагностике.
- **Проверка:** непустой summary; 20 шагов без NaN; квитанция в artifacts.

## [P093] Gate-harness: fail-loud при пустом summary
- Репо: cppmega | Приоритет: P2 | Тип: bug | Зависит от: P092
- **Где:** gate-харнесс `fa3_prod_gate_v2` (пустой summary «прошёл» молча).
- **Что делать:** харнесс падает с ошибкой, если summary пуст/нет строк.
- **Проверка:** наведённый пустой прогон → exit != 0 с понятным сообщением.

## [P094] Nebius curriculum: памятные конверты стадий
- Репо: cppmega | Приоритет: P2 | Тип: task | Зависит от: — (H200)
- **Где:** `outputs/nebius/cppmega-h200-graphroutes-1782831200/` (stage3 bs48
  и stage4 bs20/16 OOM; пик 141 GiB).
- **Что делать:** профилирование пика по стадиям; задокументировать безопасные
  конверты (bs × seq) в runbook.
- **Проверка:** таблица конвертов в runbook; повторный прогон stage3/4 на
  граничных bs без OOM.

## [P095] M4 Max vs GB10 parity
- Репо: mlx | Приоритет: P2 | Тип: task | Зависит от: P058
- **Где:** `docs/porting_plan.md:484-544` (не доказано).
- **Что делать:** прогнать matched runner на обеих машинах; зафиксировать
  числовой diff и производительность.
- **Проверка:** parity receipt с обеих сторон; вывод «parity proven/fails»
  в porting_plan.

## [P096] Distributed MLX training smoke (2 Mac)
- Репо: mlx | Приоритет: P3 | Тип: task | Зависит от: — (2 Mac с MLX)
- **Где:** `docs/distributed_zero1_smoke_procedure.md`; эпик Stream F
  (`cppmega-mlx-qq0`) — этот шаг его разблокирует/закрывает частично.
- **Что делать:** выполнить smoke-процедуру zero1 на двух машинах.
- **Проверка:** receipt процедуры; чекпоинт resume после distributed-шага.

---

# Фаза H. Стратегические решения (P097–P100)

## [P097] V9 go/no-go — DONE
- Репо: mlx | Приоритет: P1 | Тип: task | Зависит от: —
- **Где:** `VisualBuilderPlan-v9.md:4-6`, `:253-277`.
- **Что сделано:** решение зафиксировано 2026-08-01: V9 = U01 (Draft tabs strip)
  + U02 (Floating canvas toolbar); U03–U10 deferred to V10. Обоснование —
  August release blockers потребляют весь P1/P2 capacity, а U01/U02 закрывают
  gallery→train round trip и визуальную поверхность.
- **Проверка:**
  - `grep -n "V9 scope decision" cppmega.mlx/VisualBuilderPlan-v9.md` → строка 253.
  - `grep -n "U01\|U02" cppmega.mlx/VisualBuilderPlan-v9.md | head -10` — U01/U02 в
    scope, U03–U10 в deferral-таблице.

## [P098] NAM56R full readiness в mlx: оценка разрыва — DONE
- Репо: mlx | Приоритет: P3 | Тип: task | Зависит от: P060
- **Где:** `cppmega.mlx/docs/nam56r_readiness_assessment.md`,
  `cppmega.mlx/tests/test_nam56r_readiness_assessment.py`,
  `docs/porting_plan.md:484-544`.
- **Что сделано:** assessment зафиксирован: pattern/routing/config — landed;
  attention, mamba3, m2rnn, MoE, checkpoint conversion, training — partial;
  FP8/CP/distributed — missing. Вердикт: NAM56R в `cppmega_mlx` ещё не
  полностью ready. bd issue `cppmega-mlx-d5so` closed.
- **Проверка:**
  - `ls cppmega.mlx/docs/nam56r_readiness_assessment.md` — файл существует.
  - `cd /Volumes/external/sources/cppmega.mlx && .venv/bin/python -m pytest tests/test_nam56r_readiness_assessment.py -q` — зелёный.
  - `grep '"title":"\[P098\]' cppmega.mlx/.beads/issues.jsonl` — status `closed`.

## [P099] Обновить long_context_roadmap.md — DONE
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: P075
- **Где:** `docs/long_context_roadmap.md` (2026-04-14: пороги 4k→16k→128k,
  SWA/CP отложены).
- **Что сделано:** документ приведён в соответствие с beta23-реальностью и
  document isolation; добавлен тест `tests/test_long_context_roadmap_current.py`
  (`2441145e`).
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega && .venv/bin/python -m pytest tests/test_long_context_roadmap_current.py -q` → 1 passed.
  - `grep -n "document isolation\|beta23\|128k" docs/long_context_roadmap.md` — актуальные статусы.

## [P100] Квартальный docs sweep по политике retention — DONE
- Репо: cppmega | Приоритет: P3 | Тип: chore | Зависит от: P076
- **Где:** `docs/status/README.md` (политика: canonical/active/evidence/
  superseded/archived), `docs/sessions/README.md`.
- **Что сделано:** выполнен квартальный sweep 2026-08-01: добавлен
  `docs/quarterly_docs_sweep_2026_08_01.md`, обновлены `docs/status/README.md`
  и `docs/sessions/README.md`, добавлен тест
  `tests/test_docs_retention_sweep_2026_08_01.py` (`3525b315`).
- **Проверка:**
  - `cd /Volumes/external/sources/cppmega && .venv/bin/python -m pytest tests/test_docs_retention_sweep_2026_08_01.py -q` → 3 passed.
  - `ls docs/quarterly_docs_sweep_2026_08_01.md` — файл существует.

---

## Сводка по фазам

| Фаза | Шаги | P1 | Тема |
| --- | --- | --- | --- |
| A | P001–P014 | 3 | cppmega код/гигиена, без железа |
| B | P015–P024 | 2 | CI/инфраструктура cppmega |
| C | P025–P040 | 9 | данные, блокеры релиза |
| D | P041–P060 | 5 | cppmega.mlx код |
| E | P061–P072 | 5 | мост CUDA→MLX |
| F | P073–P084 | 5 | после приземления фронта |
| G | P085–P096 | 4 | железозависимое |
| H | P097–P100 | 1 | стратегические решения |

Критический путь: P025→P026→(P027→P028),(P029→P030→P031)→P032 (данные);
P073→P074/P075/P076→P079 (фронт); P061→P062→P064/P067 (мост);
P041→…→P046 (TileLang Phase 4).
