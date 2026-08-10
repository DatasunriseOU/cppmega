# Распределённая подготовка корпуса cppmega: замысел, подтверждённый прогресс и план завершения — 250 пунктов

Дата среза evidence: `2026-08-10`; последний полный deployed watchdog cycle с GCP CASE5 audit: `2026-08-10T12:58:44Z`; immutable GCP source audit: `2026-08-10T13:23:34Z`; последний live training-status refresh, использованный в отчёте: `2026-08-10T13:31:23Z`; terminal GCP stop-result readback: `2026-08-10T14:45:28Z`

Рабочая ветка документа: `docs/distributed-prep-150-runbook-20260808`

Целевой результат: lossless Parquet и Megatron для `code`, GitHub PR, GitLab MR и CI на длинах `1024, 2048, 4096, 8192, 16384, 32768, 65536`, с immutable receipts, независимым GCS/Nebius readback и без оставленных дорогих transient VM.

## Как читать отчёт

- `W001–W100` восстанавливают исходный замысел: что система должна была уметь и каким доказательством это предполагалось закрыть.
- `D001–D050` перечисляют только уже существующие commits, файлы, receipts, manifests, тестовые результаты или readback. Завершённый промежуточный слой не приравнивается к готовому датасету.
- `N001–N100` — исполнимый дальнейший план. Каждый пункт называет код или инфраструктурный объект, способ проверки и критерий завершения.
- `training_ready=true` запрещён для нового общего корпуса до финального `N099`; staged CAS, verified SQLite, partial Parquet, работающий worker и отправленное assignment не считаются training-ready.
- Автоматический retry допустим только для immutable attempt с подтверждённым HTTP 429/exit `75`. Exit `2`, drift contract, 401/403 и parser defect требуют нового revision/run, а не повтора in-place.

## Оперативный срез на момент factual refresh

- Новый общий four-lane release ещё не готов к обучению: `training_ready=false`. Исторический source-only bundle 1K–16K не переносит свой legacy `release_ready` на новый scope.
- GitHub `main` на момент последней проверки — merge `eee7d7214ee29658958ff467f7928a352c979e65` PR `#144`; локальный source residual writer продолжает работу из pin `00373dfc`, поэтому его detached worktree и output root нельзя переключать или чистить. Более новый `main` не даёт права менять revision уже активного writer.
- Локальный source residual жив с одним supervisor PID `68027`: в mutable ledger на `2026-08-10T09:59:57Z` было `7 done` и `8 failed`; после `xbox_leak_may_2020` writer перешёл к `unix-history`, `threadx`, `cmake`, затем запустил `linux`. Текущий failed set: `apple-security`, `cmake`, `intel-llvm-dpcpp`, `src`, `threadx`, `webkit`, `windows_10_shared_source_kit`, `xbox_leak_may_2020`. Terminal conveyor/root receipts отсутствуют, поэтому это residual progress, а не закрытые 501/501; mutable `_done.json` не является terminal receipt.
- Два CI fetcher физически продвигаются в раздельных stores. На live status-срезе `2026-08-10T13:31:23Z` old store имел `143,493,243,863`, new store — `249,235,618,522` exact unique payload tokens; их арифметическая сумма `392,728,862,385` остаётся только store-local upper bound и не является global union, packed valid/trained или training tokens. В очередях оставались соответственно `16,073` и `106,581` pending attempts, по `8` processing и `discovery_eof=false`; frozen CASE5 snapshot остаётся `61,311,228,208` exact store-local tokens и `training_ready=false`.
- Два последовательных production cycle deployed CASE5-monitor прошли: `12:19:25Z–12:28:37Z` и `12:49:25Z–12:58:44Z`. Для `.003`: state `running_without_heartbeat`, `6` partial output objects, `509,496,346` bytes, terminal receipts отсутствуют. Для `.004`: state `running`, fresh heartbeat `sequence=256`, `14` partial output objects, `1,319,208,652` bytes, terminal receipts отсутствуют. Для обеих `cleanup_authorized=false`, `retry_eligible=false`, `training_ready=false`; partial objects не являются completion evidence и VM пока уничтожать нельзя.
- `scripts/gcp_cloud_lane_run_monitor.py` landed через PR `#141`: implementation commit `f167b32284b4726091ea9ef086b0283b76fb3e5b`, merge `b9df4b0e61f715929a8560cb69cc144bd13d2df8`. Проверки: `61 passed`, `py_compile` green, Pyright `0 errors/warnings`, `git diff --check` green и GitHub checks `6/6` SUCCESS. Stable detached runtime `/Volumes/external/cppmega_data/worktrees/cppmega-cloud-lane-monitor-runtime-b9df4b0e-20260810` имеет monitor SHA-256 `98c07f13fb9f4205b616eefb0947eefeebcb919e28e118bfe638b870bd0bc0d2`; 30-минутный loop с SHA-256 `b7e9a17085e624b9cc4210362fc3ec72f2d35162e4cabf8d8e7dd0981784aaa5` работает в tmux одним PID `65497` и включает `.003/.004` через overlap lock. Immutable health receipt: `/Volumes/external/cppmega_data/gcp_cloud_lane_run_monitor_20260810/evidence/watchdog-deployment-health-20260810T131821Z.receipt.json`, file SHA-256 `2545d7a78dca7b05bfa6fcbc7fb35c8cfcbfc76bcca753d9b1e391a18920ad20`.
- GitHub PR store остаётся verified: `2,794,562` PR из `460` repos, `33,388` gaps закрыты, unresolved `0`; production training Parquet всё ещё `0`, потому что exact primary membership/materialization не завершены. GitLab v2 сохранил inventory sidecars и `2` primary Eigen records, после чего deterministic остановился на Mesa discussions HTTP `401`; completion receipt и training Parquet отсутствуют, automatic retry запрещён до host-scoped `read_api` repair.
- Live training-status updater восстановлен: LaunchAgent работает с PID `93951` из durable runtime `/Volumes/external/cppmega_data/worktrees/cppmega-training-status-runtime-a9145509-20260810`, exact expected revision `a91455098f2a1a82f863ce64e98d237b9e930177` и cadence `300` секунд. Использованный срез `current.md/current.json` сгенерирован в `2026-08-10T13:31:23Z`, internal status SHA-256 `f49d3322d80115012a9f2e0cf369a10a714c57536a8dda300791eb30be845e2c`; source completion остаётся честно stale. Fix commit `e35be00f7b4da4550b126d03d881e96252d6e825` landed через PR `#139`, merge `02fc1e57a0b3b60ab9daa0c7dd64ad69583e6d0a`, GitHub checks `6/6` SUCCESS. Live runtime pin сохраняется до отдельного deployment retirement/readback.
- CASE5 heartbeat/429 hardening уже landed в `main`: implementation commit `306179407570f97d4d7a571a447e95e0e9d05aed`, merge PR `#133` — `a65db87ca4e95de7b8a3eb1949786dac813d8d83`. Focused suite `25 passed`, related CASE5/cloud suite `116 passed`, Pyright `0 errors`; `py_compile`, `bash -n`, `shellcheck`, Terraform fmt/validate/tests, CodeQL и GitHub CI прошли. Следующий runtime обязан pin exact landed revision; merge сам по себе не завершает текущие `.003/.004`.
- Source-worker taxonomy «retry только при чисто подтверждённом HTTP 429» завершена: implementation `9d8a2a78168987d42d5c658e3a02bdc013c51616`, PR `#135`, merge `52472d831a6a2191f36b6f363f7d4057e844c4aa`, входит в текущий `origin/main`. Focused suite `27 passed`, related source/scheduler/cloud suite `189 passed`, changed-test Pyright `0 errors`; production `source_worker.py` всё ещё имеет `17` ранее существовавших Pyright errors и не объявляется полностью Pyright-clean. Pure 429 может дать bounded retry/exit `75`; mixed 429+deterministic, 401/403, 408/5xx, parser/contract/network defects дают exit `2` без automatic retry.
- Source repair продвинут без вмешательства в старый writer: ThreadX fix merged через PR `#143`; CMake NUL-fixture fix commit `91f7948664c71c796647615ef9945906d8cbeee4` прошёл `76` focused tests и `6/6` GitHub checks, затем merged через PR `#144` в `eee7d7214ee29658958ff467f7928a352c979e65`. Эти fixes применяются только в следующем pinned residual run; живой PID `68027` остаётся на `00373dfc`.
- GCP source audit на `13:23:34Z` даёт any-run union `432/482`, uncovered `50`, preferred new+repair complete `407`, residual `75`; new `.001` имеет `408 completed / 34 terminal / 28 current / 12 unclaimed`, repair `.001` — `19 completed / 30 terminal / 5 current`. Terminal slot/run receipts всё ещё отсутствуют. После state-aware cost audit workers `01/05/15` были `SUSPENDED` с шестью Local SSD `PRESERVED`. Три exact `stop --discard-local-ssd=false` operations для idle workers `03/11/12` завершились `DONE`, но все с deterministic `RESOURCE_ERROR/FAILED_WITH_INSTANCE_RUNNING`; VM вернулись в `RUNNING`, шесть Local SSD имеют `DISK_SAVED_STATE_UNSPECIFIED`. Failure-result receipt SHA-256 `5d03568ef8b14df69158f660080607c7313f40865dbae8b5c3a4876e89ed11b0`; retry/reset/destroy/Terraform запрещены до новой отдельной authorization.
- Terraform/bootstrap проверен на clean revision `b9df4b0e` без чтения production state и без plan/apply: recursive fmt, foundation validate/`1 passed`, workers validate/`8 passed`, provider Google `7.42.0`. Receipt `/Volumes/external/cppmega_data/gcp_terraform_bootstrap_verification_20260810/evidence/terraform-bootstrap-verification-20260810T133138Z.receipt.json`, SHA-256 `fdaba6ae7fd3de0af68e83e245650a7448ba675e028ef3124ee8d5c830f2ded5`; это static/bootstrap proof, не production conformance и не teardown authorization.

## Авторитетные контуры и идентичности

| Назначение | Контур | Ограничение |
|---|---|---|
| Исходный код | `/Volumes/external/sources/cppmega` | Runtime запускается только из pinned clean worktree. |
| Сборщики и live status | `/Volumes/external/sources/cppmega.mlx` | Stores и outputs не смешиваются с Git worktree. |
| Тяжёлые данные | `/Volumes/external/cppmega_data` | Partial и immutable roots физически разделяются. |
| GCP project | `natural-bison-491019-t9` | Run ID обязателен во всех backend/GCS/resource names. |
| GCS | `gs://natural-bison-491019-t9-cppmega-corpus` | Публикация create-only, затем exact-generation readback. |
| Terraform operator | `nanochat-automation@natural-bison-491019-t9.iam.gserviceaccount.com` | Использовать scoped credential override, не менять глобальный account. |
| GCE worker | `cppmega-corpus-worker@natural-bison-491019-t9.iam.gserviceaccount.com` | Цель — object viewer/creator только в exact run prefix; текущий foundation condition шире и закрывается N097. |
| Human Git/GCP | `david@jewelmusic.art` | Не делать unattended production зависимым от интерактивной reauth. |
| GitHub | token ledger/env в `cppmega.mlx` | В receipts хранить имя источника и fingerprint, не PAT. |
| GitLab | production policy: отдельный host-scoped `read_api` token для каждого из трёх hosts | Runtime допускает ровно один mode на host (`--token-env` или `--public-host`); production выбирает token, 401 не является пустым успехом. |
| Nebius | существующий S3 profile/env | Секреты не сериализуются; endpoint/bucket/key/hash фиксируются. |

---

## Часть I. Что изначально хотели построить — 100 пунктов

### Архитектура, истинность и границы

### W001 — Разделить корпус на четыре независимые data lanes

- **Замысел:** обрабатывать `code`, GitHub PR, GitLab MR и CI отдельно, чтобы сбой одного producer не загрязнял три других.
- **Где/чем:** `scripts/distributed_data_prep/`, lane-specific roots в `/Volumes/external/cppmega_data`, отдельные GCS prefixes.
- **Изначальный критерий:** четыре terminal lane manifests с непересекающимися artifact sets и общим final seal только поверх immutable ссылок.

### W002 — Ввести строгую лестницу состояний готовности

- **Замысел:** различать fetched, staged, verified store, packed, sealed и release-ready вместо одного расплывчатого флага.
- **Где/чем:** `configs/training_data_status.json`, `scripts/report_training_data_status.py`, completion/seal receipts.
- **Изначальный критерий:** status никогда не прибавляет overlapping snapshots и не выставляет release-ready без lossless seal.

### W003 — Зафиксировать точный corpus scope

- **Замысел:** связать каждый run с canonical списком repositories, commit/tree и digest, исключив «примерно 500» как критерий.
- **Где/чем:** source manifest, `outputs/pr_ingest/repo_list.json`, GitLab `repo_list.json`, CI inventory receipts.
- **Изначальный критерий:** accepted, excluded, active и unresolved образуют дизъюнктное полное разбиение bound scope.

### W004 — Сделать семь sequence lengths обязательным контрактом

- **Замысел:** выпускать реальные или verified-zero buckets для 1K, 2K, 4K, 8K, 16K, 32K и 64K.
- **Где/чем:** `scripts/distributed_data_prep/seal_outputs.py`, packers, Parquet manifests и Megatron prefixes.
- **Изначальный критерий:** для каждого из четырёх kinds существует ровно семь materialized/verified-zero состояний, всего 28.

### W005 — Закрепить один tokenizer contract

- **Замысел:** исключить расхождение token IDs между локальной упаковкой, GCP workers и Megatron.
- **Где/чем:** `scripts/nanochat_data/token_budget.py`, canonical tokenizer artifact и tokenizer receipts.
- **Изначальный критерий:** tokenizer digest одинаков у producer, auditor, converter и restore; token IDs лежат в `[0, 65536)`.

### W006 — Сохранить end-to-end provenance

- **Замысел:** провести repo, commit, file, PR/MR, CI run/job/step и document IDs от входа до training sample.
- **Где/чем:** provenance columns, sidecars, graph edges, `scripts/data/verify_provenance.py`.
- **Изначальный критерий:** любой Megatron sample трассируется до immutable source object и producer receipt без неоднозначного join.

### W007 — Сделать receipts первичным источником истины

- **Замысел:** не доверять process state, SQLite count или имени директории без bound terminal receipt.
- **Где/чем:** `cppmega/receipt_binding.py`, run-scoped JSON receipts, content hashes.
- **Изначальный критерий:** verifier отклоняет отсутствующий, mutable, stale, duplicate или не соответствующий artifact receipt.

### W008 — Развести transient и deterministic failures

- **Замысел:** автоматически повторять только подтверждённый HTTP 429 и не зацикливать parser defects, auth failures, contract drift, generic 408/5xx или необъяснённые network exceptions.
- **Где/чем:** exit taxonomy `75/2`, immutable assignment diagnostics, `scripts/distributed_data_prep/source_worker.py`, `scripts/distributed_data_prep/cloud_lane_pool_worker.py` и watchdog recovery policy.
- **Изначальный критерий:** exit `75` разрешён только при `confirmed_http_429=true`; retry создаёт новый attempt, а любой deterministic/неподтверждённый сбой требует нового pinned revision/residual run. Текущую более широкую source-worker transient classification нужно сузить и закрепить regression tests.

### W009 — Наблюдать pipeline каждые 30 минут

- **Замысел:** автоматически проверять local map/reducer/export, GCP receipts и упавшие управляющие агенты.
- **Где/чем:** `com.datasunrise.cppmega-pipeline-watchdog`, `com.codex.multi-429-watchdog`, immutable current reports.
- **Изначальный критерий:** два последовательных среза не старше 35 минут, last exit `0`, а recovery не создаёт duplicate writers.

### W010 — Встроить cost teardown в критерий завершения

- **Замысел:** не оставлять VM, Local SSD, static IP и placement policy после сохранения и readback результатов.
- **Где/чем:** `infra/gcp_corpus_pool/workers`, isolated Terraform backends, destroy receipts.
- **Изначальный критерий:** terminal readback предшествует destroy; пост-проверка показывает отсутствие только run-scoped ресурсов.

### Source scope, получение и parser conveyor

### W011 — Нормализовать число source repositories

- **Замысел:** объяснить различие между 501 локальным corpus scope, 482 cloud assignments и повторными attempt records.
- **Где/чем:** source list, `_done.json`, GCP source manifests и composition verifier.
- **Изначальный критерий:** один canonical identity ledger связывает scope, exclusions и attempts без сложения несопоставимых чисел.

### W012 — Поддержать прямое получение repository без обязательного tar.zst

- **Замысел:** на GCP клонировать следующий repo непосредственно из GitHub/GitLab, обрабатывать и освобождать scratch.
- **Где/чем:** source worker bootstrap, pinned remote URL/commit, Local SSD workspace.
- **Изначальный критерий:** clone receipt связывает remote, commit/tree и checkout hash; отсутствие tar не меняет corpus membership.

### W013 — Сохранить максимальную полезную git history

- **Замысел:** собирать не только HEAD code, но commit messages, diffs и source evolution в разрешённых границах.
- **Где/чем:** `scripts/nanochat_data/extract_git_history.py`, commit conveyor и source composition.
- **Изначальный критерий:** receipt отдельно фиксирует fetched refs/branches, annotated/lightweight tags, submodule/LFS policy, effective depth и server-side shallow fallback; неполный ref/depth contract не проходит как full-history result.

### W014 — Изолировать каждый parser checkout

- **Замысел:** не запускать долгий child process из worktree, который другой агент может удалить или переключить.
- **Где/чем:** pinned detached worktrees, launch receipt с cwd/tree digest, durable output root.
- **Изначальный критерий:** preflight и heartbeat проверяют существование cwd; resume восстанавливает тот же revision в новом immutable path.

### W015 — Классифицировать source files по обучающей роли

- **Замысел:** отдельно маршрутизировать C/C++ source, headers, build, shell, SQL, diagnostics и auxiliary Python.
- **Где/чем:** `scripts/nanochat_data/route_packed_source_docs.py`, route sidecars.
- **Изначальный критерий:** категории взаимно исключаются, totals сохраняются, auxiliary не смешивается с primary code без явного policy.

### W016 — Построить clang-based semantic index

- **Замысел:** извлекать declarations, references, diagnostics и graph entities поверх текста файлов.
- **Где/чем:** `tools/clang_indexer/index_project.py`, compiler database discovery, entity sidecars.
- **Изначальный критерий:** fixture tests подтверждают координаты, file identity и отсутствие cross-repo edges.

### W017 — Извлекать build-system контекст

- **Замысел:** сохранить CMake, Meson, Bazel, Make, Ninja и toolchain targets рядом с кодом.
- **Где/чем:** build classifiers, command/target sidecars, repository parser.
- **Изначальный критерий:** build documents имеют repo/commit provenance и не маскируются как C++ source.

### W018 — Материализовать commit lane отдельно от code lane

- **Замысел:** не смешивать history/diff documents с snapshot code при dedup и packing.
- **Где/чем:** commit conveyor, `cppmega/data/source_conveyor_composition.py`, separate route roots; `scripts/distributed_data_prep/seal_outputs.py` принимает top-level kind `source`, внутри которого обязательны route submanifests `code` и `commits`.
- **Изначальный критерий:** code и commit routes имеют distinct IDs, totals и по семь Megatron states; top-level `source` manifest агрегирует обе routes без превращения общей матрицы `4×7` в неявную `5×7`.

### W019 — Сохранить compiler/linker/sanitizer diagnostics

- **Замысел:** превратить реальные диагностические цепочки в структурированные training documents.
- **Где/чем:** diagnostic parsers, CI/source sidecars, occurrence records.
- **Изначальный критерий:** сообщение, location, toolchain и source binding восстанавливаются из sidecar без потери.

### W020 — Сохранить graph edges и chunk boundaries

- **Замысел:** удержать структуру AST/call/include/build graph и границы документов после packing.
- **Где/чем:** graph CSR sidecars, document/chunk offsets, `scripts/data/verify_side_channel_shapes.py`.
- **Изначальный критерий:** offsets монотонны, находятся в sample capacity и согласованы с `.bin/.idx`.

### W021 — Сделать source conveyor потоковым

- **Замысел:** не хранить сотни гигабайт промежуточных checkout/archive, а писать content-addressed outputs по мере обработки.
- **Где/чем:** `scripts/streaming_conveyor.py`, `scripts/source_conveyor_supervisor.py`, atomic staging.
- **Изначальный критерий:** crash-resume принимает только hash-verified done artifacts и не повторяет завершённые repositories.

### W022 — Ограничить память каждого source assignment

- **Замысел:** избежать системного OOM и BrokenProcessPool при крупных монорепозиториях.
- **Где/чем:** `scripts/data/memory_guard.py`, worker resource limits, per-repo profiles.
- **Изначальный критерий:** memory exit классифицирован и имеет bounded diagnostic; scheduler может направить heavy repo в отдельный class.

### W023 — Ввести exact quarantine вместо широких ignore rules

- **Замысел:** исключать только доказанно проблемные bytes/diagnostics, не целые extensions или directories.
- **Где/чем:** `configs/source_quarantine_manifest.json`, `tools/clang_indexer/source_quarantine.py`.
- **Изначальный критерий:** one-byte drift ломает quarantine match; negative fixtures не отбрасываются.

### W024 — Считать worker-local hashes без удаления candidates

- **Замысел:** экономить повторное хеширование/cache lookup внутри assignment, но сохранять все candidate documents и occurrences до глобального выбора winner.
- **Где/чем:** per-worker cache/content hashes и candidate receipts; `scripts/distributed_data_prep/source_manifest.py` фиксирует `dedup_applied_on_worker=false`, а `scripts/distributed_data_prep/source_reducer.py` выполняет exact/near dedup единственным writer.
- **Изначальный критерий:** worker не отбрасывает candidates; reducer доказывает winner/duplicate/occurrence conservation. Любой будущий worker-side removal требует новой schema и отдельного occurrence-conservation receipt.

### W025 — Реализовать crash-resume локального conveyor

- **Замысел:** продолжать после restart с последнего подтверждённого repository/shard.
- **Где/чем:** `_done.json`, progress log, launch/repair receipts и shared-base lock.
- **Изначальный критерий:** resume не мутирует accepted artifacts и отклоняет drift repo list/code revision.

### W026 — Классифицировать каждую source failure

- **Замысел:** разделить transport, parser defect, OOM/signal, corrupt input и operator interruption.
- **Где/чем:** bounded stderr digests, exit receipts, source failure inventory.
- **Изначальный критерий:** перед repair нет `unknown`, а каждый failure связан с repo/commit/failing file.

### W027 — Авторизовать bounded repair

- **Замысел:** чинить только unresolved set и никогда не перезапускать successful repos.
- **Где/чем:** repair contract, source composition inputs, pinned repair branch.
- **Изначальный критерий:** authorization содержит finite repo set и отвергается при изменении base receipt.

### W028 — Собрать логическую source composition

- **Замысел:** объединить base, local repair и GCP runs через receipts, не копированием output trees.
- **Где/чем:** `cppmega/data/source_conveyor_composition.py`, artifact-set bindings.
- **Изначальный критерий:** ровно один accepted producer на repo/document; unresolved и duplicates равны нулю.

### W029 — Закрыть source token conservation

- **Замысел:** доказать путь каждого документа через accepted, duplicate, quarantine, auxiliary и packed outcomes.
- **Где/чем:** reducer/route receipts, `scripts/report_training_data_status.py`.
- **Изначальный критерий:** valid/trained/pad equations сходятся без необъяснённой дельты.

### W030 — Выдать source-only handoff

- **Замысел:** передать PR/MR/CI lanes immutable code/commit composition, не выставляя общий флаг готовности.
- **Где/чем:** source lane manifest, GCS readback, A→B/C receipt.
- **Изначальный критерий:** handoff перечисляет семь buckets и остаётся `training_ready=false` через integration seal, remote restore и loader smoke — до N099.

### GCP foundation и распределённая source обработка

### W031 — Создать отдельный Terraform foundation

- **Замысел:** один раз подготовить project APIs, bucket, service accounts и базовую сеть отдельно от transient workers.
- **Где/чем:** `infra/gcp_corpus_pool/foundation`.
- **Изначальный критерий:** `terraform validate/test` зелёные, foundation state не уничтожается вместе с run.

### W032 — Создать run-scoped Terraform workers module

- **Замысел:** разворачивать несколько одинаковых workers с изолированным backend prefix.
- **Где/чем:** `infra/gcp_corpus_pool/workers/backend.tf.example`, `infra/gcp_corpus_pool/workers/lane-workers.backend.hcl.example`, `infra/gcp_corpus_pool/workers/workers.tftest.hcl`; реальный run-scoped backend HCL генерирует `scripts/gcp_isolated_worker_rollout.py`. Общий example prefix `terraform/lane-workers` не годится для concurrent production runs.
- **Изначальный критерий:** backend prefix включает validated run ID, а два run IDs не могут адресовать одинаковые VM/IP/state objects.

### W033 — Обеспечить быстрый сетевой путь

- **Замысел:** выбирать machine/network topology, способную использовать Google backbone и высокую egress/PD throughput.
- **Где/чем:** worker machine type, gVNIC/network tier, region/zone variables, `scripts/estimate_distributed_data_prep.py` и immutable `network-throughput.receipt.json`.
- **Изначальный критерий:** bootstrap receipt фиксирует effective NIC/network tier; три measured GCS/local-SSD trials фиксируют p50/p95 bytes/s, поэтому «10 Gbit» не объявляется без фактического результата.

### W034 — Назначить статические run-scoped addresses

- **Замысел:** сделать SSH/control endpoints стабильными во время run и удаляемыми после него.
- **Где/чем:** `google_compute_address.worker`, outputs и destroy plan.
- **Изначальный критерий:** address name содержит run ID и отсутствует после verified destroy.

### W035 — Использовать CPU workers порядка 16 vCPU/64 GB

- **Замысел:** начать с `n2-standard-16` и масштабировать число VM, а heavy repos выделять отдельно.
- **Где/чем:** worker tfvars, scheduler resource classes и `scripts/estimate_distributed_data_prep.py`.
- **Изначальный критерий:** measured baseline даёт ETA/confidence interval и cost reforecast для 1/4/16 workers, связывая repo throughput, CPU utilization и memory/I/O high-water marks.

### W036 — Использовать transient Local SSD как scratch

- **Замысел:** быстро клонировать/распаковывать/indexировать без накопления долговечных локальных сотен гигабайт.
- **Где/чем:** Local SSD/NVMe mounts в `infra/gcp_corpus_pool/workers/startup.sh.tftpl`.
- **Изначальный критерий:** mount/read-write benchmark проходит; accepted artifacts загружены до destroy VM.

### W037 — Сделать boot disk воспроизводимым

- **Замысел:** хранить только OS/bootstrap/cache, а не уникальные training outputs.
- **Где/чем:** resolved exact Compute image self-link вместо одной mutable family `debian-12`, disk type/size variables и bootstrap payload receipt.
- **Изначальный критерий:** plan/apply receipt связывает exact image identity; новый worker из того же image+payload воспроизводит tool versions и source hashes.

### W038 — Развести operator и worker IAM

- **Замысел:** Terraform operator создаёт ресурсы, worker читает/создаёт только объекты разрешённого run prefix.
- **Где/чем:** operator SA, per-run worker identity либо exact run-bound IAM condition и no-secret metadata; общий `${gcs_prefix}/` недостаточно узок.
- **Изначальный критерий:** negative IAM tests запрещают worker читать/писать sibling run, Terraform state и foundation objects.

### W039 — Использовать GCS как control/data plane

- **Замысел:** хранить immutable manifests, claims, heartbeats, outcomes, candidates и reducer artifacts вне VM.
- **Где/чем:** `gs://natural-bison-491019-t9-cppmega-corpus/runs/<RUN_ID>`.
- **Изначальный критерий:** object creation использует generation precondition и readback inventory hash.

### W040 — Раздать source assignments детерминированно

- **Замысел:** каждый repo получает assignment digest, home shard и явный resource class; текущая source manifest schema имеет identity/shard, но ещё не resource envelope.
- **Где/чем:** `scripts/distributed_data_prep/source_manifest.py`, `scripts/distributed_data_prep/source_work_queue.py` и schema migration для heavy/resource class.
- **Изначальный критерий:** assignment set digest равен manifest; unknown/duplicate assignment и heavy class без matching worker envelope отвергаются.

### W041 — Добавить dynamic work stealing

- **Замысел:** не держать VM пустой, если её home shard закончен, пока в очереди есть runnable work.
- **Где/чем:** `scripts/distributed_data_prep/source_slot_scheduler.py`, claims и leases в GCS.
- **Изначальный критерий:** lease исключает двойного winner; stale claim может быть заменён только по contract.

### W042 — Публиковать heartbeat на active assignment

- **Замысел:** отличать долгий large-repo parser от зависшего worker.
- **Где/чем:** attempt-specific heartbeat objects и monitor report.
- **Изначальный критерий:** heartbeat содержит assignment/attempt/lease digests и не может завершить assignment.

### W043 — Разделить assignment и worker completion

- **Замысел:** не считать repo готовым по dispatch/claim или завершению одного slot process.
- **Где/чем:** assignment receipts, assignment completions, slot receipts, worker terminal receipts.
- **Изначальный критерий:** scheduler закрывает run только при полном expected set и verified artifacts.

### W044 — Запустить single-writer cloud reducer

- **Замысел:** детерминированно выбрать winners и выполнить global dedup после map phase.
- **Где/чем:** `scripts/distributed_data_prep/source_reducer.py`, reducer receipts.
- **Изначальный критерий:** повторный reducer на том же input inventory даёт byte-identical manifest.

### W045 — Удалить source pool после readback

- **Замысел:** остановить расходы сразу после composition, artifact upload и независимой проверки.
- **Где/чем:** isolated Terraform destroy plan для каждого source run.
- **Изначальный критерий:** VM/IP/policy absent, GCS inventory и source handoff продолжают верифицироваться.

### GitHub PR и GitLab MR

### W046 — Зафиксировать GitHub repository scope

- **Замысел:** сканировать PR только для canonical GitHub repos и связать список с source scope.
- **Где/чем:** `scripts/pr_ingest/build_repo_list.py`, `outputs/pr_ingest/repo_list.json`.
- **Изначальный критерий:** repo count/set digest и immutable observation boundary (`as_of`, terminal cursors/pages) совпадают во scan, store и completion receipt; изменения во время pagination имеют explicit follow-up set.

### W047 — Выполнить exhaustive GitHub GraphQL stream

- **Замысел:** получить PR metadata, reviews, comments, links и diffs с resumable pagination.
- **Где/чем:** `scripts/pr_ingest/graphql_pr_stream.py`, token ledger, PR store.
- **Изначальный критерий:** terminal completion связывает все cursors/pages и не содержит unverified production rows.

### W048 — Закрыть truncated GraphQL gaps

- **Замысел:** отдельно дособрать PR, где основной stream достиг API truncation/лимитов.
- **Где/чем:** `scripts/pr_ingest/github_graphql_fallback.py`, gap target manifest.
- **Изначальный критерий:** target=completed, unresolved/skipped/missed равны нулю, record hashes входят в exact scan.

### W049 — Сделать GitHub store проверяемым, но не training-ready

- **Замысел:** хранить exact records в SQLite как source-of-materialization, не как готовый training artifact.
- **Где/чем:** `scripts/pr_ingest/pr_store.py`, `prs.sqlite`, completion verifier.
- **Изначальный критерий:** integrity/count/hash green; eligible Parquet остаётся нулём до membership/export.

### W050 — Вычислить exact GitHub primary membership

- **Замысел:** выбирать PR только по allowlisted commit/source provenance, а не по наличию строки в store.
- **Где/чем:** `cppmega/data/pr_primary_membership.py`, source composition и PR completion.
- **Изначальный критерий:** portable membership Parquet имеет unique `(repo, pr_number)` и bound source/store hashes.

### W051 — Материализовать GitHub PR documents losslessly

- **Замысел:** экспортировать только exact primary members, сохранив discussion, diff, review и provenance.
- **Где/чем:** `scripts/pr_ingest/export_pr_parquet.py`, membership receipt, isolated output root; текущий default `ZSTD_LEVELS` 1K–16K должен быть расширен и протестирован для 32K/64K.
- **Изначальный критерий:** rendered document count совпадает с membership, nonmembers физически не входят в shards, а seven-length contract проверен positive/verified-zero fixtures.

### W052 — Рендерить discussion детерминированно

- **Замысел:** получить стабильный текст PR без зависимости от SQLite row order или текущего API ответа.
- **Где/чем:** `scripts/pr_ingest/render_discussion.py`, canonical ordering rules.
- **Изначальный критерий:** повторный render одинакового record даёт тот же byte hash и role boundaries.

### W053 — Сохранить PR sidecars

- **Замысел:** вынести actors, timestamps, review states, linked issues, files и commit links в проверяемые side channels.
- **Где/чем:** новый/расширенный PR sidecar producer для actors/reviews/files/links и `scripts/audit_sidecar_parquet.py`; auditor не считается producer.
- **Изначальный критерий:** row/document IDs и sidecar shapes совпадают для всех семи buckets.

### W054 — Ограничить GitLab scope Eigen/Mesa/Tor

- **Замысел:** добавить три важных C/C++ проекта, которые живут на разных GitLab hosts.
- **Где/чем:** `gitlab.com/libeigen/Eigen`, `gitlab.freedesktop.org/mesa/mesa`, `gitlab.torproject.org/tpo/core/tor`.
- **Изначальный критерий:** canonical repo list содержит ровно три host/project identities и собственный digest.

### W055 — Использовать отдельную auth policy для каждого GitLab host

- **Замысел:** не подменять токен одного инстанса другим и не считать 401 пустым inventory.
- **Где/чем:** host→auth-mode mapping в `/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/gitlab_mr_stream.py`; каждый host задаётся ровно одним `--token-env` или `--public-host`, production policy требует token.
- **Изначальный критерий:** preflight 200 для каждого host; receipt хранит fingerprint, но не значение token.

### W056 — Сначала построить GitLab inventory

- **Замысел:** отделить дешёвый список MR и candidate classification от тяжёлого detail fetch.
- **Где/чем:** GitLab scanner manifest, pagination receipts и deterministic gzip sidecars.
- **Изначальный критерий:** declared=candidate+noncandidate, host coverage полная, `as_of`/terminal cursors/pages связаны receipt, а MR, изменившиеся во время pagination, входят в explicit follow-up set.

### W057 — Физически разделить primary и ancillary GitLab stores

- **Замысел:** не допустить попадания ancillary/error/terminal records в primary training membership.
- **Где/чем:** distinct `primary.sqlite`, `ancillary.sqlite` и sidecar root.
- **Изначальный критерий:** store files имеют разные paths/hashes; route conservation и store counts проверены.

### W058 — Выполнить authenticated smoke до authoritative detail fetch

- **Замысел:** доказать доступ к candidate details на каждом host ограниченным run до authoritative production detail fetch. Уже выполненный public production-labelled inventory на 46,978 MR этого gate не закрывает.
- **Где/чем:** отдельные smoke roots и host-scoped tokens.
- **Изначальный критерий:** smoke completion восстанавливается из store/manifest; 401/403/429 классифицированы правильно.

### W059 — Вычислить exact GitLab primary MR membership

- **Замысел:** связать MR с accepted source/commit evidence тем же fail-closed принципом, что и GitHub.
- **Где/чем:** `/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/build_gitlab_primary_membership.py`, membership bridge и отдельный GitLab membership→document exporter contract.
- **Изначальный критерий:** membership receipt unique, input-bound и не принимает legacy inventory без current contract.

### W060 — Запечатать GitHub PR и GitLab MR раздельно

- **Замысел:** не смешивать платформы до общего release manifest и разрешить local/GCP producers только через deterministic winner/composition receipt.
- **Где/чем:** два lane manifests, два GCS prefixes, cloud-lane assignments при GCP исполнении и отдельные composition/conservation/readback receipts.
- **Изначальный критерий:** каждый lane покрывает семь lengths, имеет ровно одного winner на membership key и остаётся `training_ready=false` до remote restore/loader/final audit.

### CI fetch, CASE5 и sidecars

### W061 — Собирать CI по всему bound repository scope

- **Замысел:** получить workflow/job logs и metadata для всех поддерживаемых GitHub repositories, а не sample.
- **Где/чем:** `scripts/ci_stream_fetch.py`, repo partitions и token ledger.
- **Изначальный критерий:** terminal fetch inventory покрывает exact repo set и все pages/runs до freeze boundary.

### W062 — Параллелить CI fetch безопасными shards

- **Замысел:** использовать несколько fetchers без общей mutable cursor/state гонки.
- **Где/чем:** shard-specific fetch state, `scripts/merge_ci_stream_shards.py`.
- **Изначальный критерий:** shard sets непересекаются либо dedupятся по canonical key; merge receipt закрывает union.

### W063 — Сохранить hierarchy repo/run/workflow/job/step

- **Замысел:** не превращать CI log в текст без структуры исполнения.
- **Где/чем:** CI metadata tables, `scripts/ci_log_sidecars.py`.
- **Изначальный критерий:** parent-child keys валидны и каждый log chunk связан с job/step.

### W064 — Сохранить actor и runner context

- **Замысел:** различать автора, triggering actor, runner type/labels и execution environment.
- **Где/чем:** actor/runner sidecars.
- **Изначальный критерий:** nullable/unknown values явны, а не заменены выдуманными defaults.

### W065 — Извлечь язык, платформу и toolchain

- **Замысел:** классифицировать compiler, OS, architecture, runtime и build system каждого CI occurrence.
- **Где/чем:** `scripts/ci_log_sidecars.py` и command classifiers; `ci_source_sidecars.py` отвечает за source-blob binding, а не эту классификацию.
- **Изначальный критерий:** classifier version и evidence spans записаны; totals консервативны.

### W066 — Извлечь команды и targets

- **Замысел:** сохранить реальные configure/build/test invocations и target names.
- **Где/чем:** shell parser, command/target sidecars.
- **Изначальный критерий:** redaction убирает secrets, но сохраняет executable/args/working-directory semantics.

### W067 — Извлечь test outcomes

- **Замысел:** представить pass/fail/skip/flaky и framework details структурированно.
- **Где/чем:** `scripts/ci_log_sidecars.py` и test-result parsers; `ci_source_binding_projection.py` остаётся source-binding projection.
- **Изначальный критерий:** result связан с exact log byte range и не выводится только из exit code при неоднозначности.

### W068 — Извлечь compiler/linker/sanitizer diagnostics из CI

- **Замысел:** собрать occurrence records с source binding и context lines.
- **Где/чем:** production CI canonicalization/export path и CASE5 occurrence schema; `scripts/fetch_ci_diagnostics.py` используется только как bounded/sample fetcher.
- **Изначальный критерий:** oversized metadata bounded, duplicate occurrence keys детерминированно объединены.

### W069 — Построить CI entities и graph edges

- **Замысел:** связать commands, targets, files, diagnostics и tests в обучающий graph.
- **Где/чем:** entity/edge sidecars, canonical ID namespace.
- **Изначальный критерий:** edges не ссылаются на отсутствующие nodes и проходят shape verifier.

### W070 — Сохранить chunk boundaries CI logs

- **Замысел:** делить большие logs без потери строк, diagnostics или document isolation.
- **Где/чем:** chunk sidecars, line/byte offsets и tokenizer spans.
- **Изначальный критерий:** concatenation/conservation возвращает canonical normalized log content.

### W071 — Использовать content-addressed CI store

- **Замысел:** дедуплицировать одинаковые payloads при сохранении всех occurrence bindings.
- **Где/чем:** `scripts/ci_content_store.py`, CAS blobs и occurrence DB.
- **Изначальный критерий:** blob hash проверен, occurrence count не уменьшается из-за payload dedup.

### W072 — Заморозить CASE5 threshold snapshot

- **Замысел:** получить immutable срез достаточно большого CI corpus для воспроизводимого export.
- **Где/чем:** `scripts/distributed_data_prep/ci_case5_snapshot.py`, frozen receipt.
- **Изначальный критерий:** отдельная frozen SQLite/CAS copy не содержит WAL/SHM, её hashes стабильны и она не складывается с overlapping live upper bound; live fetchers могут продолжать работу в других stores.

### W073 — Перенести CASE5 inputs в GCS

- **Замысел:** не требовать сотни гигабайт свободного места на Mac для cloud materialization.
- **Где/чем:** GCS run inputs prefix, publication receipt.
- **Изначальный критерий:** exact-generation readback подтверждает каждый input before worker launch.

### W074 — Материализовать CASE5 на GCP

- **Замысел:** использовать CPU/Local SSD workers для reducer/export/packing рядом с GCS.
- **Где/чем:** cloud lane manifests, `scripts/distributed_data_prep/cloud_lane_pool_worker.py`, isolated Terraform run.
- **Изначальный критерий:** smoke создаёт assignment, output и terminal receipts; production масштабируется только после smoke.

### W075 — Выдать CI lane seal

- **Замысел:** завершить global union/dedup, lossless Parquet и семь Megatron prefixes до интеграции.
- **Где/чем:** CI export receipt, sidecar audits, CI lane manifest.
- **Изначальный критерий:** frozen input conservation и artifact readback green, `training_ready=false` через four-kind seal — до remote restore/loader/final audit.

### Parquet, sidecars и Megatron

### W076 — Зафиксировать canonical Parquet schema

- **Замысел:** унифицировать tokens, labels, document IDs, provenance и objective columns между lanes.
- **Где/чем:** консолидировать training-row schema из `cppmega/data/nanochat_pipeline/packed_rows_schema.py` и `scripts/nanochat_data/pack_enriched_rows.py` в один versioned contract с schema digest; `scripts/canonical_parquet_ledger.py` остаётся отдельной ledger schema (`sequence_index`, `record_json`, `record_sha256`).
- **Изначальный критерий:** exporter, auditor и Megatron converter проверяют один schema version/digest и одинаковые dtypes/semantics; training rows нельзя принять по digest ledger schema.

### W077 — Использовать ZSTD и атомарную публикацию Parquet

- **Замысел:** минимизировать storage/network без partial files в authoritative root.
- **Где/чем:** `scripts/data/atomic_publish.py`, Parquet writer options.
- **Изначальный критерий:** regular file, footer readable, codec ZSTD, final hash стабилен после rename/fsync.

### W078 — Доказать document/token losslessness

- **Замысел:** объяснить каждый исходный token как trained, masked, padded, duplicate или quarantined.
- **Где/чем:** export/reducer/conservation receipts.
- **Изначальный критерий:** equations сходятся по lane, bucket и global union без silent truncation.

### W079 — Аудировать все sidecars

- **Замысел:** не допустить рассинхронизации token rows и graph/provenance/objective data.
- **Где/чем:** `scripts/audit_sidecar_parquet.py`, `scripts/data/verify_side_channel_shapes.py`.
- **Изначальный критерий:** counts, offsets, IDs и hashes согласованы на каждом shard.

### W080 — Выполнить global dedup один раз

- **Замысел:** применять deterministic exact/near dedup один раз внутри каждого lane после canonical composition; отдельная release policy решает intentional cross-lane duplicates.
- **Где/чем:** source reducer + `scripts/data/verify_global_dedup_store.py`, lane-specific PR/MR/CI reducers и cross-lane policy audit.
- **Изначальный критерий:** scope/winner policy/version/input order bound; per-lane rerun даёт тот же set digest, а N088 отдельно проверяет cross-lane collisions.

### W081 — Материализовать 1K bucket

- **Замысел:** оптимизировать короткие документы, сохраняя document isolation.
- **Где/чем:** packer, Parquet route `1024`, Megatron prefix.
- **Изначальный критерий:** capacity/labels/sidecars audited и `.bin/.idx` readback green.

### W082 — Материализовать 2K bucket

- **Замысел:** поддержать средние snippets и discussions отдельной geometry.
- **Где/чем:** route `2048`, same canonical packer/converter.
- **Изначальный критерий:** единый route-by-fit contract делает семь membership sets дизъюнктными; conservation и MMIDIDX integrity green. Полное семикратное repacking одного документа запрещено.

### W083 — Материализовать 4K bucket

- **Замысел:** сохранить более длинные code/CI contexts без неоправданного split.
- **Где/чем:** route `4096`.
- **Изначальный критерий:** exact capacity, trained mask и sidecar coordinates доказаны.

### W084 — Материализовать 8K bucket

- **Замысел:** покрыть длинные files, PR discussions и logs.
- **Где/чем:** route `8192`.
- **Изначальный критерий:** split/overflow receipt lossless и Megatron files nonempty либо verified-zero.

### W085 — Материализовать 16K bucket

- **Замысел:** сохранить крупные semantic/diagnostic documents.
- **Где/чем:** route `16384`.
- **Изначальный критерий:** long-context packing tests и readback проходят.

### W086 — Материализовать 32K bucket

- **Замысел:** убрать историческое ограничение пяти buckets и реально выпускать 32K.
- **Где/чем:** route `32768`, bundle builder, long-context verifier.
- **Изначальный критерий:** positive data или exact verified-zero receipt; manifest-only запись запрещена.

### W087 — Материализовать 64K bucket

- **Замысел:** выпускать максимальный training context без скрытого truncation.
- **Где/чем:** route `65536`, packer/converter/readback.
- **Изначальный критерий:** lossless split, valid token range и physical `.bin/.idx` либо verified-zero.

### W088 — Конвертировать Parquet в Megatron MMIDIDX

- **Замысел:** получить train-consumable `.bin/.idx` плюс полный sidecar profile.
- **Где/чем:** `scripts/data_prep_parquet_to_megatron.py`.
- **Изначальный критерий:** sequence/document counts и byte sizes вычисляются из manifest и совпадают при reread.

### W089 — Выполнить независимый Megatron readback

- **Замысел:** не считать успешный write достаточным доказательством.
- **Где/чем:** `scripts/data/verify_dataset_megacpp.py` для базового `.bin/.idx` чтения плюс отдельные artifact-hash и sidecar-alignment auditors в empty restore directory.
- **Изначальный критерий:** offsets monotonic, data bounds valid, sidecars align и manifest-bound artifact hashes совпадают; узкий Megatron verifier один не закрывает весь критерий.

### W090 — Зафиксировать batch geometry

- **Замысел:** считать full batches/remainder для каждой длины без изменения training tokens и без выдачи одного исторического `batch_size=192` за универсальный training contract 1K–64K.
- **Где/чем:** per-length/configurable training geometry manifest и training status; исторический status batch size используется только как информационный расчёт.
- **Изначальный критерий:** для каждого выбранного `(lane, length, batch_size)` выполняется `rows=full_batches×batch_size+remainder`; padding не считается valid/trained, а sealing не зависит от незафиксированного default batch size.

### Publication, release и эксплуатация

### W091 — Собрать четыре lane manifests

- **Замысел:** дать integration sealer content-addressed inventories вместо mutable paths.
- **Где/чем:** lane-specific manifest builders; `scripts/distributed_data_prep/seal_outputs.py` только принимает и проверяет четыре уже построенных manifests.
- **Изначальный критерий:** каждый producer manifest перечисляет Parquet, Megatron, sidecars, audits и lineage до вызова sealer.

### W092 — Выполнить four-kind integration seal

- **Замысел:** проверить полную матрицу `4×7`, global tokenizer и provenance contracts.
- **Где/чем:** distributed sealer и release manifest.
- **Изначальный критерий:** один immutable pre-publication seal receipt с 28 materialized/verified-zero states и без mutable dependency; он остаётся `training_ready=false` до N093/N094/N099.

### W093 — Опубликовать artifacts в GCS create-only

- **Замысел:** сделать GCS главным облачным источником для workers/reducers и долговременных receipts.
- **Где/чем:** production four-lane uploader, run-scoped object prefixes и generation preconditions; `seal_outputs.py` выдаёт plan, но не загружает bytes.
- **Изначальный критерий:** existing mismatch никогда не перезаписывается; publication receipt хранит generation/hash/size.

### W094 — Выполнить exact-generation GCS readback

- **Замысел:** проверить именно загруженную generation, а не latest object с тем же key.
- **Где/чем:** GCS readback verifier и sorted inventory digest.
- **Изначальный критерий:** server/local checksums, sizes и generations совпадают для каждого artifact.

### W095 — Опубликовать финальный bundle в Nebius

- **Замысел:** передать сжатые проверенные Parquet/Megatron outputs туда, где будет обучение.
- **Где/чем:** новый distributed-release bundle adapter поверх `scripts/data/publish_megatron_bundle_to_nebius_s3.py`; текущий publisher принимает Megatron bundle v3/v4, а не distributed seal v1 и не полный Parquet set.
- **Изначальный критерий:** adapter включает Parquet+Megatron+sidecars, а multipart/object receipt связан с release manifest, endpoint/bucket/key и checksum.

### W096 — Восстановить bundle из Nebius в пустой root

- **Замысел:** доказать, что remote copy самодостаточна и не зависит от uploader cache.
- **Где/чем:** matching distributed-release restore adapter поверх `scripts/data/restore_megatron_bundle_from_nebius_s3.py --require-empty-output-root`.
- **Изначальный критерий:** safe member set, hashes, PyArrow Parquet footers, MMIDIDX, sidecars и schema compatibility проходят повторно; затем N094 открывает результат настоящим training loader.

### W097 — Публиковать честный training-data status

- **Замысел:** показывать live/staged/sealed отдельно и не суммировать overlapping snapshots.
- **Где/чем:** `/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.md` и status SHA.
- **Изначальный критерий:** counters выводятся только из receipts; blockers перечислены явно.

### W098 — Исключить secrets из control plane

- **Замысел:** не записывать PAT, GitLab token, OAuth access token и Authorization headers в metadata/logs/receipts.
- **Где/чем:** wrappers, secret scans, service-account identity.
- **Изначальный критерий:** receipts содержат только env names/fingerprints; redaction tests и repository scan green.

### W099 — Выполнить полный cost/resource cleanup

- **Замысел:** удалить transient source/CI/reducer VM, disks, IP и placement policies после remote readback.
- **Где/чем:** run-specific Terraform destroy receipts и GCP inventory.
- **Изначальный критерий:** foundation/bucket сохранены, expensive run resources отсутствуют, data readback остаётся green.

### W100 — Выдать финальный release handoff

- **Замысел:** передать training owner один immutable manifest вместо набора локальных директорий.
- **Где/чем:** release receipt, GCS/Nebius pointers, tokenizer/Megatron commit bindings.
- **Изначальный критерий:** только post-GCS/Nebius-restore/loader/completion-audit receipt имеет `training_ready=true`; текущую pre-publication семантику `seal_outputs.py`, где flag становится true раньше upload/readback, необходимо изменить в N090.

---

## Часть II. Что уже сделано и подтверждено — 50 пунктов

Live срез `/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.md`, использованный для factual refresh, сгенерирован `2026-08-10T13:31:23Z`: physical Markdown SHA-256 `467c72d874db05293c55467604030187ce184cc518f97c96126fae5845f63c6b`, physical JSON SHA-256 `59e86545d35a7805e84daa546f99552eccbb205abe8816853838330fe27e8a54`, internal status SHA-256 `f49d3322d80115012a9f2e0cf369a10a714c57536a8dda300791eb30be845e2c`. Это главный источник текущих physical/token totals, но live counters считаются timestamp-bound evidence, а не terminal receipts. Для GitHub/GitLab/GCP используются отдельные receipts, названные в пунктах ниже.

### Код распределённого runtime и infrastructure-as-code

### D001 — Source/GCP repair stack интегрирован в основную историю

- **Сделано:** ветка integration landed через commit `ac41718b` и последующие merges; `origin/main` на evidence-срезе содержит source repair/runtime stack.
- **Доказательство:** `git log --all`, merge `10066e88`, parser/tokenizer repair `b045e547`; commits `ac41718b`, `10066e88`, `b045e547`, прежний срез `88f4d8e7`, CASE5 merge `a65db87c`, source retry merge `52472d83` и source recovery merge `a87d2fe0` находятся в текущей основной истории.
- **Честная граница:** наличие кода в main не означает, что все source assignments завершены.

### D002 — Terraform foundation оформлен отдельным модулем

- **Сделано:** существуют versioned `main.tf`, `variables.tf`, `outputs.tf`, lock file, examples и `foundation.tftest.hcl`.
- **Доказательство:** `infra/gcp_corpus_pool/foundation/` в `origin/main`.
- **Честная граница:** этот факт доказывает IaC-контракт, но не current account permissions; worker IAM condition сейчас охватывает общий configured GCS prefix, а не автоматически exact `runs/<RUN_ID>`.

### D003 — Terraform workers поддерживает run-scoped resource names

- **Сделано:** workers module содержит backend, startup template, outputs, tfvars examples и `workers.tftest.hcl`.
- **Доказательство:** `infra/gcp_corpus_pool/workers/`.
- **Честная граница:** resource names принимают run ID, но `lane-workers.backend.hcl.example` всё ещё имеет общий `terraform/lane-workers`; production обязан генерировать unique backend prefix и доказывать отсутствие старых VM отдельным readback/destroy receipt.

### D004 — Receipt-bound GCP source/cloud monitors реализованы

- **Сделано:** source monitor считает claims, heartbeats, outcomes и assignment receipts; commit `ba417173` добавил attempt-scoped failure receipts, UUID/URI binding и совместимость с legacy receipts. Отдельный cloud-lane monitor проверяет topology, exact object generations, heartbeat/completion/failure publications, output inventory, cleanup gate и pure-429 retry contract.
- **Доказательство:** source branch `fix/gcp-monitor-attempt-scoped-failures-20260809`, commit `ba417173`; live source-monitor SHA-256 `c5e38447ffc321965380170e0cdfd30b98191c2956bcf00afa35ae31c319a064`. Cloud-lane monitor commit `f167b32284b4726091ea9ef086b0283b76fb3e5b` landed через PR `#141`, merge/main `b9df4b0e61f715929a8560cb69cc144bd13d2df8`; `61 passed`, `py_compile` green, Pyright `0 errors/warnings`, `git diff --check` green и GitHub `6/6` SUCCESS. Deployed monitor/loop SHA-256: `98c07f13fb9f4205b616eefb0947eefeebcb919e28e118bfe638b870bd0bc0d2`/`b7e9a17085e624b9cc4210362fc3ec72f2d35162e4cabf8d8e7dd0981784aaa5`.
- **Честная граница:** immutable health receipt с двумя последовательными cycles уже выпущен, но он доказывает здоровье controller, а не completion `.003/.004`. Deployed monitor сам parser defects не исправляет и не делает partial outputs training-ready.

### D005 — Dynamic source work stealing реализован

- **Сделано:** scheduler поддерживает claim queue/leases вместо жёсткого ожидания home shard.
- **Доказательство:** commit `23d864de` и live `scheduler_mode: dynamic_claim_queue` в repair report.
- **Честная граница:** work stealing повышает utilization, но не делает deterministic exit успешным.

### D006 — Receipt-bound cloud lane materializer добавлен

- **Сделано:** codebase содержит подготовку payload, lane worker и materializer с manifest binding.
- **Доказательство:** commits `f5a37ea9`/`aef67a72`, `scripts/distributed_data_prep/lane_materializer.py`.
- **Честная граница:** production output должен быть доказан отдельным terminal receipt.

### D007 — CASE5 cloud snapshot adapter добавлен

- **Сделано:** CI frozen inputs могут быть описаны cloud-lane snapshot manifest.
- **Доказательство:** commit `259f793f`, `scripts/distributed_data_prep/ci_case5_snapshot.py`.
- **Честная граница:** adapter не равен завершённому CASE5 Parquet export.

### D008 — GitHub PR cloud lane существует только как локальный prototype

- **Сделано:** локальный prototype умеет представить membership-bound PR materialization как receipt-bound cloud lane.
- **Доказательство:** локальный commit object `3a5f7309` содержит связанные distributed scripts/tests; `git merge-base --is-ancestor 3a5f7309 origin/main` возвращает `1`, а refs, содержащие commit, отсутствуют.
- **Честная граница:** prototype не pushed и не landed, поэтому не считается доступной production-возможностью; текущий training status по-прежнему показывает GitHub PR Parquet `0`.

### D009 — Pool execution и immutable heartbeat для CASE5 workers реализованы

- **Сделано:** pooled worker path дополнен immutable self-digested heartbeat, create-only publication с exact-generation readback, единым fail-closed 429/deterministic classifier, idempotent acceptance completed assignment и bounded cooperative shutdown.
- **Доказательство:** исходный pool commit `8ba86ec5`; hardening commit `306179407570f97d4d7a571a447e95e0e9d05aed`, merged PR `#133`/`a65db87ca4e95de7b8a3eb1949786dac813d8d83`; focused `25 passed`, related CASE5/cloud `116 passed`, Pyright `0 errors`, shell/Python/Terraform/CI/CodeQL проверки green.
- **Честная граница:** `ci-case5-smoke-20260809-002` завершился deterministic exit `2` без assignment/output completion; активные `.003`/`.004` запущены на более ранних pinned revisions. Landed hardening защищает следующие attempts, но не переписывает и не завершает уже запущенные VM.

### D010 — CASE5 smoke topology изолирована от production

- **Сделано:** smoke использует отдельный run ID/backend/resource set и не мутирует production state.
- **Доказательство:** commit `7942cd81`, isolated plan/apply receipts для `ci-case5-smoke-20260808-003`.
- **Честная граница:** изоляция устраняет collision, но не доказывает throughput или output receipt.

### D011 — Resource topology CASE5 привязана к lane contract

- **Сделано:** worker topology входит в manifest/receipt binding и не может тихо измениться между plan и run.
- **Доказательство:** commit `2d6d1cfd`.
- **Честная граница:** bound topology ещё нужно benchmark/validate на реальном production volume.

### D012 — Source failure receipts сделаны attempt-unique

- **Сделано:** разные attempts deterministic failure больше не конфликтуют по одному object key; source transport retry дополнительно сужен до pure confirmed HTTP 429.
- **Доказательство:** attempt-unique commit `b3e15f9c`; retry implementation `9d8a2a78168987d42d5c658e3a02bdc013c51616`, PR `#135`, merge `52472d831a6a2191f36b6f363f7d4057e844c4aa`; focused `27 passed`, related `189 passed`, changed-test Pyright `0 errors`.
- **Честная граница:** уникальность receipt и корректная taxonomy не исправляют первопричину deterministic failure. Production module сохраняет `17` pre-existing Pyright errors и не считается полностью type-clean.

### D013 — CASE5 contract допускает корректный prefix ladder

- **Сделано:** verifier различает поддерживаемую семибакетную лестницу и конкретный materialized prefix/subset.
- **Доказательство:** commits `de666288`, `2595f5d4`, `0485fb3c`.
- **Честная граница:** 32K/64K всё равно должны быть физически materialized или verified-zero перед final seal.

### D014 — Tokenizer resolution больше не зависит от process cwd

- **Сделано:** canonical tokenizer находится через стабильный path contract, а не только текущую директорию.
- **Доказательство:** commit `2890ce5f`.
- **Честная граница:** удалённый cwd всё ещё ломает сам process spawn; этот fix закрывает только tokenizer lookup.

### D015 — Добавлены round-trip и sidecar alignment tests tokenizer

- **Сделано:** тесты проверяют packed source tokenizer round-trip и целостность sidecar alignment.
- **Доказательство:** commit `b045e547`, новые `tests/test_packed_source_tokenizer_sidecar_integrity.py` и `tests/test_token_budget_tokenizer_path.py`.
- **Честная граница:** эти unit/regression tests не заменяют production artifact conservation audit.

### Локальный source corpus и существующий sealed bundle

### D016 — Честный machine-readable training status выпускается

- **Сделано:** status раздельно показывает live source, sealed Megatron, GitHub PR, GitLab MR и CI; updater снова работает каждые 300 секунд из durable exact runtime и не требует существования удалённого temporary worktree.
- **Доказательство:** LaunchAgent command pin `a91455098f2a1a82f863ce64e98d237b9e930177`, runtime `/Volumes/external/cppmega_data/worktrees/cppmega-training-status-runtime-a9145509-20260810`, PID `93951`; срез `2026-08-10T13:31:23Z` имеет physical Markdown SHA-256 `467c72d874db05293c55467604030187ce184cc518f97c96126fae5845f63c6b`, physical JSON SHA-256 `59e86545d35a7805e84daa546f99552eccbb205abe8816853838330fe27e8a54`, internal status SHA-256 `f49d3322d80115012a9f2e0cf369a10a714c57536a8dda300791eb30be845e2c`. Fix commit `e35be00f7b4da4550b126d03d881e96252d6e825` landed через PR `#139`, merge `02fc1e57a0b3b60ab9daa0c7dd64ad69583e6d0a`; GitHub `6/6` checks SUCCESS.
- **Честная граница:** updater честно помечает source completion stale и сохраняет `release_ready=false` для новых live/PR/MR/CI lanes. Исторический sealed source-only bundle не переносит свой `release_ready=true` на новый four-lane scope.

### D017 — Live source Parquet 1K физически существует

- **Сделано:** на status-срезе `2026-08-10T10:15:37Z` учтено 444 файла, 3,208,207 rows, 3,175,983,339 valid и 3,159,298,982 trained tokens.
- **Доказательство:** physical inventory root `/Volumes/external/cppmega_data/source_full501_7f55ff0c12d88bb835fea9a68b8ba9d90522ddd5/reindexed`, inventory SHA-256 `2e8bc64399edc2f9efa3805e338afaf6b2db9e993695cf59486417f9f56c80cd`.
- **Честная граница:** bucket `packed_unsealed`, а completion totals расходятся с physical Parquet.

### D018 — Live source Parquet 2K физически существует

- **Сделано:** учтено 443 файла, 470,086 rows, 669,822,296 valid и 669,352,210 trained tokens.
- **Доказательство:** тот же physical inventory/root с SHA-256 `2e8bc643…c80cd`.
- **Честная граница:** данные входят в незавершённый overlapping live snapshot.

### D019 — Live source Parquet 4K физически существует

- **Сделано:** учтено 440 файлов, 231,836 rows, 654,826,938 valid и 654,595,102 trained tokens.
- **Доказательство:** тот же physical inventory/root с SHA-256 `2e8bc643…c80cd`.
- **Честная граница:** global composition/dedup/seal ещё не закрыты.

### D020 — Live source Parquet 8K физически существует

- **Сделано:** учтено 441 файл, 102,556 rows, 578,489,841 valid и 578,387,285 trained tokens.
- **Доказательство:** тот же physical inventory/root с SHA-256 `2e8bc643…c80cd`.
- **Честная граница:** наличие rows не доказывает terminal repository coverage.

### D021 — Live source Parquet 16K физически существует

- **Сделано:** учтено 432 файла, 105,934 rows, 1,502,391,092 valid и 1,502,285,158 trained tokens.
- **Доказательство:** тот же physical inventory/root с SHA-256 `2e8bc643…c80cd`.
- **Честная граница:** 32K/64K в этом live root ещё не материализованы.

### D022 — Live source token accounting посчитан

- **Сделано:** physical inventory фиксирует 2,200 файлов, 4,118,619 rows, 17,594,769 source documents, 6,581,513,506 valid и 6,563,918,737 trained tokens без сложения с overlapping sealed snapshot.
- **Доказательство:** сумма пяти physical bucket tables; inventory SHA-256 `2e8bc64399edc2f9efa3805e338afaf6b2db9e993695cf59486417f9f56c80cd`.
- **Честная граница:** totals относятся к `packed_unsealed`, не к release-ready corpus.

### D023 — Source documents разложены по девяти категориям

- **Сделано:** посчитаны bash, build, headers, C/C++ source, diagnostics, exact excluded-other, Python auxiliary, shell other и SQL.
- **Доказательство:** physical category inventory: bash `15,113`, build `128,364`, headers `13,577,801`, C/C++ source `3,723,475`, diagnostics `11,870`, excluded-other `2,665`, Python auxiliary `106,518`, shell other `18,434`, SQL `10,529`.
- **Честная граница:** status отмечает, что Python auxiliary пока смешан с main rows.

### D024 — Source blockers не скрыты

- **Сделано:** current status явно сообщает mismatch completion/physical totals, failed units и auxiliary mixing.
- **Доказательство:** секция `Blockers` live source.
- **Честная граница:** это завершённая диагностика/прозрачность, а не устранение blockers.

### D025 — Существующий sealed Megatron manifest и remote archive receipt сохранены

- **Сделано:** manifest `/Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/macro_routes_v1_20260713/manifest.json` и долговременный Nebius archive receipt сохраняют отдельный historical snapshot.
- **Доказательство:** manifest SHA `1e44fcd8ff9192a15e25b62c18ac71569ca7627e3f2221b29d249d5c2fb93391`, artifact-set SHA `8c550514dd52ddc99ecad91cfd5e4b0355c194c5d3e38d6686550cf4fd2d088b`, archive SHA `5ac3e095289b28d4924c3547ebae62ee79f002c4ad0c81467fda59b03a1c93a7`, size `2,823,467,371`, status `uploaded_verified`.
- **Честная граница:** локально отсутствуют 230 из 236 manifest artifacts (`local_snapshot_retained=false`). Legacy `release_ready=true` относится только к historical source-only 1K–16K bundle: manifest не имеет современного `training_ready` gate/loader-smoke receipt и не является новым four-lane release.

### D026 — Sealed Megatron 1K зафиксирован manifest/audit

- **Сделано:** manifest связывает 2,366 files, 1,804,082 rows, 1,712,299,227 valid и 1,704,343,213 trained tokens.
- **Доказательство:** `Sealed Megatron bundle` table в current status и remote archive binding D025.
- **Честная граница:** это manifest/audit total исторического source-only bundle, а не утверждение о локальном наличии каждого файла.

### D027 — Sealed Megatron 2K зафиксирован manifest/audit

- **Сделано:** manifest связывает 2,367 files, 441,140 rows, 629,504,765 valid и 629,063,625 trained tokens.
- **Доказательство:** sealed bundle table и remote archive binding D025.
- **Честная граница:** локальное наличие всех artifacts не доказано; bundle не содержит новые PR/MR/CI lanes.

### D028 — Sealed Megatron 4K зафиксирован manifest/audit

- **Сделано:** manifest связывает 2,363 files, 230,004 rows, 653,341,457 valid и 653,111,453 trained tokens.
- **Доказательство:** sealed bundle table и remote archive binding D025.
- **Честная граница:** это не доказательство локального physical set или 32K/64K поддержки.

### D029 — Sealed Megatron 8K зафиксирован manifest/audit

- **Сделано:** manifest связывает 2,358 files, 110,040 rows, 621,715,449 valid и 621,605,409 trained tokens.
- **Доказательство:** sealed bundle table и remote archive binding D025.
- **Честная граница:** локальное наличие всех artifacts не доказано; bucket нельзя складывать с live 8K из-за overlap.

### D030 — Sealed Megatron 16K зафиксирован manifest/audit

- **Сделано:** manifest связывает 2,317 files, 46,197 rows, 515,886,337 valid и 515,840,140 trained tokens.
- **Доказательство:** sealed bundle table и remote archive binding D025.
- **Честная граница:** локальное наличие всех artifacts не доказано; historical manifest не содержит 32K/64K states.

### D031 — Totals sealed bundle посчитаны без overlap

- **Сделано:** status фиксирует 4,132,747,235 valid и 4,123,963,840 trained tokens.
- **Доказательство:** summary current status и сумма D026–D030.
- **Честная граница:** эти totals не прибавляются к 6.29B live source.

### GitHub PR и GitLab MR evidence

### D032 — GitHub exact PR store полностью verified

- **Сделано:** completion receipt содержит declared=stored=`2,794,562`, unverified=`0`, status `verified`.
- **Доказательство:** `/Volumes/external/sources/cppmega.mlx/outputs/pr_ingest/exact_29c6869/pr_completion.verify7.json`, scan ID `1c103e…d1901`.
- **Честная граница:** verified SQLite store не является Parquet и даёт eligible packed tokens `0`.

### D033 — GitHub scope из 460 repositories связан digest

- **Сделано:** completion receipt фиксирует `expected_repo_count=460` и set SHA `d0720b…17ced`.
- **Доказательство:** тот же absolute `pr_completion.verify7.json` плюс bound repo list/store hashes.
- **Честная граница:** 460 — GitHub PR scope, а не число всех local source attempts.

### D034 — Все 33,388 GraphQL gaps закрыты

- **Сделано:** gap completion имеет target count `33,388`; каждый completed record хранит repo/PR/hash.
- **Доказательство:** `/Volumes/external/sources/cppmega.mlx/outputs/pr_ingest/exact_29c6869/graphql_gap_completion.verify6.json`, SHA bound из PR completion.
- **Честная граница:** gap closure завершает fetch verification, но не primary membership.

### D035 — GitHub unresolved/unverified production rows равны нулю

- **Сделано:** exact completion не содержит unverified production PR; gap verification не оставляет unresolved targets.
- **Доказательство:** `unverified_store_pr_count=0` и terminal gap receipt.
- **Честная граница:** это не означает, что все 2.79M records должны войти в training set.

### D036 — 227 вне-scan GitHub rows помещены в recoverable quarantine

- **Сделано:** rows вынесены в ZSTD Parquet с SHA `f8fe3916…a59d1`; production store после операции имеет unverified `0`.
- **Доказательство:** `unverified_store_quarantine_227.receipt.json`, status `complete`.
- **Честная граница:** quarantine audit-only и не является training membership.

### D037 — GitLab three-host inventory завершён по legacy contract

- **Сделано:** verified receipt фиксирует 46,978 declared MR по Eigen/Mesa/Tor и три expected hosts/repositories.
- **Доказательство:** `/Volumes/external/cppmega_data/gitlab_mr_stream_prod_v1_20260804/completion_receipt.json`.
- **Честная граница:** contract SHA legacy; current production scan по hardened contract ещё нужен.

### D038 — GitLab candidate classification завершена

- **Сделано:** из 46,978 inventory records выделен 821 candidate, 46,157 noncandidate.
- **Доказательство:** legacy completion receipt и route conservation.
- **Честная граница:** candidate не равен primary training member.

### D039 — GitLab primary/ancillary stores физически разделены

- **Сделано:** receipt связывает distinct `primary.sqlite` и `ancillary.sqlite`; validation `primary_ancillary_physical_separation=true`.
- **Доказательство:** legacy completion receipt, primary/ancillary counts `0/11`.
- **Честная граница:** legacy primary `0` показывает незавершённость auth/membership, а не отсутствие MR у проектов.

### D040 — GitLab deterministic sidecars материализованы

- **Сделано:** сохранено 47,799 canonical-json-gzip files, `gzip_mtime=0`, с logical/physical set hashes.
- **Доказательство:** legacy completion receipt, physical size 91,503,709 bytes.
- **Честная граница:** sidecars являются verified inventory evidence, но eligible Parquet tokens остаются `0`.

### D041 — Authenticated Eigen smoke нашёл primary MR

- **Сделано:** отдельный one-host smoke verified 2,736 declared records, 2 candidates и 2 stored primary MR.
- **Доказательство:** `/Volumes/external/cppmega_data/gitlab_mr_stream_eigen_auth_20260804_001/completion_receipt.json`.
- **Честная граница:** smoke не покрывает Mesa/Tor и использует legacy contract SHA.

### D042 — GitLab primary-membership bridge закоммичен и pushed

- **Сделано:** branch `feat/gitlab-primary-membership-bridge-20260808` существует локально и на `origin` с commit `4ec96cf5`.
- **Доказательство:** `git branch -a --contains 4ec96cf5`.
- **Честная граница:** bridge ожидает terminal source composition и current GitLab production inputs.

### D043 — Focused bridge regression suite воспроизводимо проходит

- **Сделано:** `tests/test_gitlab_primary_membership_bridge.py` повторно запущен из detached clean worktree commit `4ec96cf5` в project venv с explicit portable profile; результат `5 passed`.
- **Доказательство:** команда использовала `/Volumes/external/sources/cppmega.mlx/.venv/bin/python`, `CPPMEGA_TEST_PROFILE=portable-data` и временный worktree, удалённый после проверки.
- **Честная граница:** точное historical duration не запечатано отдельным receipt; независимый rerun дал `5 passed in 1.46s`. Это focused bridge suite; current GitLab scanner/exporter и production bytes требуют собственных N047–N056 тестов/receipts.

### CI, watchdog и GCP operational evidence

### D044 — CI live CAS имеет измеренный store-local upper bound

- **Сделано:** timestamp-bound training-status slice фиксирует `392,728,862,385` exact unique payload tokens как сумму store-local counters двух live stores: `143,493,243,863` old и `249,235,618,522` new.
- **Доказательство:** `/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.md`, generated `2026-08-10T13:31:23Z`, physical Markdown SHA-256 `467c72d874db05293c55467604030187ce184cc518f97c96126fae5845f63c6b`, internal status SHA-256 `f49d3322d80115012a9f2e0cf369a10a714c57536a8dda300791eb30be845e2c`; оба progress receipts на момент среза были fresh.
- **Честная граница:** это store-local upper bound, не global union/dedup и не training tokens.

### D045 — Frozen CASE5 threshold snapshot сохранён

- **Сделано:** frozen receipt фиксирует 61,311,228,208 exact unique payload tokens.
- **Доказательство:** current status и GCS run `ci-case5-threshold-20260804-001`.
- **Честная граница:** eligible packed Parquet valid/trained tokens остаются `0/0`.

### D046 — Legacy CI sample Parquet существует для 1K–16K

- **Сделано:** status учитывает sample из 1,855 jobs с physical rows во всех пяти historical buckets.
- **Доказательство:** таблица `Legacy 1,855-job sample`.
- **Честная граница:** sample не является production CASE5, 32K/64K отсутствуют и release-ready не выставлен.

### D047 — 30-минутный pipeline watchdog установлен и исполняется

- **Сделано:** loop выполняет все GCP source scans и CASE5 `.003/.004`, считает 30-минутную cadence от начала цикла, защищён `shlock` от overlap и запрещает retry без self-digested `failed_confirmed_429`/exit `75`. Он работает одним процессом в tmux session `cppmega_pipeline_watchdog`.
- **Доказательство:** live loop PID `65497`, SHA-256 `b7e9a17085e624b9cc4210362fc3ec72f2d35162e4cabf8d8e7dd0981784aaa5`; два последовательных production cycles `2026-08-10T12:19:25Z–12:28:37Z` и `2026-08-10T12:49:25Z–12:58:44Z` дали monitor exit `0`, overlap lock удалён. Immutable health receipt `/Volumes/external/cppmega_data/gcp_cloud_lane_run_monitor_20260810/evidence/watchdog-deployment-health-20260810T131821Z.receipt.json` имеет file SHA-256 `2545d7a78dca7b05bfa6fcbc7fb35c8cfcbfc76bcca753d9b1e391a18920ad20`.
- **Честная граница:** health receipt доказывает cadence, pin и controller state, но mutable partial CASE5 objects не являются pipeline completion receipt; `.003/.004` сохраняют `cleanup_authorized=false` и `training_ready=false`.

### D048 — Codex 429 watchdog установлен

- **Сделано:** отдельный launchd label использует `StartInterval=1800`, `RunAtLoad=true`; на последней проверке было 72 runs и last exit `0`; последний подтверждённый scan не давал права повторять deterministic failures.
- **Доказательство:** `com.codex.multi-429-watchdog.plist`, `launchctl print`, script SHA-256 `6a6f03a2621283b95ba59a19d755512324d49993aa03efd569e7c3f7a4ae5c2d`.
- **Честная граница:** watchdog повторяет controller session; он не даёт права retry deterministic parser assignment.

### D049 — CASE5 runs `ci-case5-smoke-20260808-002` и `ci-case5-smoke-20260809-002` уничтожены scoped и evidence сохранено

- **Сделано:** для обоих разных `.002` run exact isolated Terraform destroy удалил только run VM/static address/placement policy; managed states пусты.
- **Доказательство:** historical destroy receipt `/Volumes/external/cppmega_data/ci_case5_cloud_materialize_20260808_002/isolated-terraform/ci-case5-smoke-20260808-002.destroy-receipt.json`; новый evidence root `/Volumes/external/cppmega_data/ci_case5_cloud_materialize_20260809_002/terminal-evidence`, bundle SHA/generation `47db8f7ac3c0bb399845d0ab2fe3c116fdd7be209f8c34f2790d7991c64cefae`/`1786289496842413`, terminal receipt SHA/generation `ae5c7fe30c36474962e10912d0492c6bba5339d3d3f4b9e5e45a627d6b06ea5a`/`1786289614633242`, destroy receipt SHA/generation `64fda071550b184f9f579cf29ddd582a5de118f904f93267f258acae8679e392`/`1786290047242620`, plan `0 add / 0 change / 3 destroy`.
- **Честная граница:** новый run на code revision `b05e0bd0` доказал HTTP `200`, `curl_exit=23` при записи большого `content-store-index.sqlite3`, diagnostic SHA `d7023d35dc95e6369998d9fe1194e95d5d9bf333d10dd969cff39b08a494a704`, deterministic exit `2`, `confirmed_http_429=false`, без assignment/output completion. Это diagnostic+cleanup, не data completion и не основание для retry.

### D050 — Четыре GCP source run сведены immutable historical read-only audit

- **Сделано:** fresh generation-pinned audit связал `.004=111 success/371 unclaimed/0 VM`, old `.005=410 success/54 terminal/18 current/2 fresh/16 VM`, new `.001=408 success/34 terminal/28 current/26 fresh/12 unclaimed/16 VM`, repair `.001=19 success/30 terminal/5 current/4 fresh/4 VM`; any-run union покрывает `432` проекта, `50` пока не покрыты ни одним run, preferred new+repair composition закрывает `407` и оставляет residual `75`.
- **Доказательство:** `/Volumes/external/cppmega_data/gcp_source_multi_run_audit_20260809/evidence/read-only-audit-20260810T132334Z/audit.md`; `audit.json` SHA-256 `bf0c98e8c3744e96dd8284e74fcde09ac9236c7bc20e839a09ae4063cb6005b0`, evidence receipt SHA-256 `ecf0a3a8b721999892b1dded0d9e41e6b289f94c0942cd0bbccba657be7ffcd4`.
- **Честная граница:** это точный read-only срез `2026-08-10T13:23:34Z`, а не live count. Ни один run на этом срезе не имеет terminal slot/run receipts или reducer seal. Для idle `.005` workers `01/05/15` подтверждены `SUSPENDED` и Local SSD `PRESERVED`; exact stop operations для `03/11/12` завершились `DONE` с `RESOURCE_ERROR/FAILED_WITH_INSTANCE_RUNNING`, а instances снова `RUNNING` с Local SSD state `UNSPECIFIED`. Result receipt `/Volumes/external/cppmega_data/gcp_source_multi_run_audit_20260809/evidence/stop-preserve-local-ssd-result-20260810T144528Z/result.receipt.json` имеет SHA-256 `5d03568ef8b14df69158f660080607c7313f40865dbae8b5c3a4876e89ed11b0`; retry, reset, destroy и broad Terraform action не авторизованы.

---

## Часть III. Детальный дальнейший план — 100 пунктов

### Немедленное восстановление и завершение source lane (`code` + `commits` routes)

### N001 — Выпустить новый общий runtime checkpoint после landed updater

- **Действие:** hash текущих `_done.json`, launch/repair contracts, live status, local locks и GCP monitor reports без остановки writers; связать уже landed PR `#139`/merge `02fc1e57` с post-merge readback и сохранить работающий durable runtime до формального retirement.
- **Код/инфра/аккаунт:** source root в `/Volumes/external/cppmega_data`, `scripts/report_training_data_status.py`, branch `fix/training-status-updater-pinned-runtime-20260809`, read-only GCS через scoped operator identity.
- **Проверка:** stat-before/hash/stat-after; mutable file получает отдельную observed-version запись, а не ложный stable hash. Merge должен оставаться `02fc1e57…`, два status cycles после merge — сохранить exact runtime revision `a9145509…`, fresh heartbeat и один reporter PID.
- **Готово когда:** `N001-runtime-checkpoint.json` перечисляет local/GCP/PR/MR/CI evidence, связывает landed updater с post-merge runtime и везде сохраняет `training_ready=false`.

### N002 — Зафиксировать уже восстановленный durable runtime pin

- **Действие:** проверить, что активный local residual run продолжает использовать pin `00373dfc` и durable paths; не создавать второй supervisor и не переключать detached writer worktree.
- **Код/инфра/аккаунт:** `/Volumes/external/cppmega_data/source_full501_repair_main_00373dfc_20260809T073953Z`, process/launch receipts и `git worktree list`.
- **Проверка:** PID/cwd/code revision/tokenizer/config paths совпадают launch contract; output root не перемещён и не имеет второго writer.
- **Готово когда:** свежий runtime-binding receipt подтверждает единственный живой supervisor и помечает старый missing cwd только как historical cause.

### N003 — Проверить landed durable-cwd fix на active generation

- **Действие:** подтвердить, что supervisor выбирает durable run workspace, проверяет cwd до spawn и fail-closed обрабатывает исчезновение checkout; новый fix в active writer не подмешивать.
- **Код/инфра/аккаунт:** `scripts/source_conveyor_supervisor.py`, `scripts/streaming_conveyor.py`, worktree lifecycle helper.
- **Проверка:** regression удаляет исходный checkout после staging; resume обязан либо продолжить из durable cwd, либо fail deterministic до writer mutation.
- **Готово когда:** landed revision/fixture tests и свежие active-run records подтверждают отсутствие нового `FileNotFoundError: cwd`; при расхождении — отдельный новый revision/run после завершения current writer.

### N004 — Повторно подтвердить restart/lock semantics

- **Действие:** прогнать уже существующие regression tests единственного supervisor/shared-base repair lock/stale PID/restart и сохранить immutable test receipt exact revision.
- **Код/инфра/аккаунт:** `tests/test_source_conveyor_supervisor.py`, `tests/test_streaming_conveyor_revision.py`.
- **Проверка:** два concurrent writers отвергаются; stale lock снимается только с bound evidence; accepted shard hash не меняется.
- **Готово когда:** clean pinned worktree выдаёт pytest exit `0` и test receipt с source tree SHA.

### N005 — Дать активному локальному conveyor продолжить без перезапуска

- **Действие:** не останавливать живой writer; наблюдать forward progress и запускать новый resume generation только после terminal exit либо доказанного controller failure.
- **Код/инфра/аккаунт:** существующий supervisor CLI/run root, local operator; broad kill pattern запрещён.
- **Проверка:** process identity/lock/output generation стабильны; два 30-минутных watchdog среза показывают новые parser/materialization counters без duplicate writer.
- **Готово когда:** active generation получает terminal receipt или возникает immutable deterministic blocker, авторизующий новый pinned run.

### N006 — Нормализовать 501 scope против progress attempts

- **Действие:** построить canonical ledger repo→attempts→accepted/unresolved/excluded и объяснить raw `440 done / 66 failed`.
- **Код/инфра/аккаунт:** source list, `_done.json`, `cppmega/data/source_conveyor_composition.py`.
- **Проверка:** неизвестные repos, duplicate accepted producer и unresolved+accepted overlap блокируют ledger.
- **Готово когда:** дизъюнктное разбиение ровно равно bound source scope; 506 progress records не выдаются за 506 repositories.

### N007 — Пересобрать exact local failure inventory

- **Действие:** для каждого unresolved attempt сохранить exit, bounded stderr, repo/commit, failing file и classification; применять уже landed source-worker taxonomy из `52472d83`, не расширяя её обратно на generic transient failures.
- **Код/инфра/аккаунт:** supervisor logs, parser diagnostics, quarantine receipts, `scripts/distributed_data_prep/source_worker.py` и его tests.
- **Проверка:** только `confirmed_http_429=true` может дать exit `75`; generic 408/5xx/network error, OOM, deterministic parser, corrupt input и operator interruption дают bounded exit `2`/terminal diagnostics; `unknown=0` до authorization.
- **Готово когда:** inventory SHA связан с N006, finite repair contract построен, production residual run pin является descendant `52472d83`, а immutable outcomes подтверждают отсутствие automatic retry для всего, кроме pure confirmed 429.

### N008 — Довести KeyDB repair до production receipt

- **Действие:** взять уже исправленный code path, повторить 27 focused tests из clean worktree, commit/push и запустить только KeyDB residual assignment.
- **Код/инфра/аккаунт:** repair branch, KeyDB fixture, source quarantine/parser tests.
- **Проверка:** regression воспроизводит исходный failure до fix и accepted candidate после fix; unrelated repo hashes не меняются.
- **Готово когда:** immutable KeyDB completion/artifact receipt проходит composition verifier.

### N009 — Закрыть ImHex и Haiku deterministic failures

- **Действие:** разделить их симптомы, создать минимальные fixtures и применить parser/quarantine fix без project-wide ignore.
- **Код/инфра/аккаунт:** `tools/clang_indexer/`, exact quarantine manifest, per-repo repair branch.
- **Проверка:** positive fixture проходит, one-byte/diagnostic negative mutation не попадает под quarantine.
- **Готово когда:** два distinct terminal receipts приняты, а repair diff не расширяет scope молча.

### N010 — Закрыть MXNet и unix-history failures

- **Действие:** проверить gitlink/history/input-shape причины отдельно и исправить assembly/receipt binding.
- **Код/инфра/аккаунт:** commit/source conveyor, `f5dfe56c` lineage, repository fixtures.
- **Проверка:** missing gitlink, historical encoding и malformed archive cases имеют deterministic tests.
- **Готово когда:** MXNet и unix-history получают accepted или explicit source-scope exclusion receipts с policy approval.

### N011 — Закрыть AOSP native parser fallback

- **Действие:** закончить отдельный `fix/aosp-native-parser-fallback-20260809`, не подменяя native semantics plain-text fallback без маркировки.
- **Код/инфра/аккаунт:** worktree `/Volumes/external/cppmega_data/worktrees/cppmega-aosp-native-parser-fallback-20260809`.
- **Проверка:** fixture сравнивает native/fallback provenance, diagnostics и graph availability; fallback reason попадает в sidecar.
- **Готово когда:** branch committed/pushed, tests green, residual assignment bound exact commit.

### N012 — Выделить heavy repository resource class

- **Действие:** добавить versioned resource-class/envelope в source manifest schema и профилировать Boost, ClickHouse, GCC, FreeBSD, ReactOS, Filament по RAM/CPU/time.
- **Код/инфра/аккаунт:** `scripts/distributed_data_prep/source_manifest.py`, scheduler, GCP `n2-standard-16` baseline и отдельный heavy tfvars.
- **Проверка:** schema отвергает неизвестный class; large fixture получает matching worker; изменение class не меняет content digest, но входит в assignment envelope digest.
- **Готово когда:** heavy jobs не блокируют обычную очередь, а assignments имеют bound resource class и measured cost/time estimate.

### N013 — Довести local base+repair до terminal coverage

- **Действие:** дождаться/возобновить все runnable assignments, применить только авторизованные fixes N008–N011.
- **Код/инфра/аккаунт:** local conveyor base root и repair generations.
- **Проверка:** accepted+excluded равно local scope; active/unresolved/duplicate accepted равны нулю.
- **Готово когда:** terminal local composition input receipt опубликован atomically.

### N014 — Сохранить доказанное здоровье cloud-lane monitor до terminal CASE5

- **Действие:** принять уже выпущенный two-cycle health receipt для landed PR `#141`/merge `b9df4b0e`, не перезапускать stable detached runtime и продолжать наблюдать CASE5 `.003/.004`; после любого изменения monitor/config/controller заново требовать два полных cycle и новый immutable health receipt.
- **Код/инфра/аккаунт:** worktree `/Volumes/external/cppmega_data/worktrees/cppmega-case5-gcp-watchdog-20260809`, `scripts/gcp_cloud_lane_run_monitor.py`, `/Users/dave/Library/Application Support/CppMega/pipeline-watchdog/pipeline-watchdog-loop.sh`, configs/reports в `/Volumes/external/cppmega_data/gcp_cloud_lane_run_monitor_20260810`, source-monitor pinned worktree, два plists; GCP вызовы только через `nanochat-automation@natural-bison-491019-t9.iam.gserviceaccount.com` command-scoped override.
- **Проверка:** existing receipt SHA `2545d7a78dca7b05bfa6fcbc7fb35c8cfcbfc76bcca753d9b1e391a18920ad20`, exact code/config hashes, один controller PID и каждый последующий report без overlap. Exit `75` разрешает retry только вместе с `report.retry_eligible=true`; exit `2`, stale/mixed/partial failure не запускают command. Monitor никогда автоматически не уничтожает VM.
- **Готово когда:** `.003/.004` имеют terminal completion/failure readback, последний health report связан с этими terminal generations, а cleanup остаётся запрещён до отдельного exact teardown gate.

### N015 — Вывести три оставшиеся `.005` VM из тарифицируемого RUNNING безопасным новым gate

- **Действие:** принять terminal failure receipt старого stop attempt и не повторять его. Выполнить новый state-aware read-only audit для workers `03/11/12`, подтвердить отсутствие owned assignment и отдельно выбрать минимальную авторизованную операцию: guest remediation с последующим suspend/stop-preserve либо discard/delete только после exact diagnostic/artifact readback.
- **Код/инфра/аккаунт:** project `natural-bison-491019-t9`, command-scoped `nanochat-automation@natural-bison-491019-t9.iam.gserviceaccount.com`, failure result `/Volumes/external/cppmega_data/gcp_source_multi_run_audit_20260809/evidence/stop-preserve-local-ssd-result-20260810T144528Z/result.receipt.json`; новая mutation требует нового exact-target authorization, broad Terraform action остаётся запрещён.
- **Проверка:** новый audit доказывает current assignment ownership, instance/operation state, Local SSD diagnostic value и GCS readback; proposed command связан с exact instance IDs и не использует retry classification, потому что failure не был HTTP 429/exit `75`.
- **Готово когда:** workers `03/11/12` подтверждены non-billable `SUSPENDED/TERMINATED` либо отсутствующими, судьба Local SSD и artifacts доказана immutable receipt, а никакие active/runnable assignments не потеряны.

### N016 — Дать `.001` завершить только уже активную полезную работу

- **Действие:** не retry ни один deterministic exit-2 outcome in-place; сохранить outputs всех fresh/runnable assignments, взяв их количество из нового immutable inventory, а не из устаревающего числа в runbook.
- **Код/инфра/аккаунт:** GCS run `source-repair-20260808-001`, dynamic claim queue.
- **Проверка:** heartbeat freshness, assignment digest, terminal candidate/hash; no automatic replacement flag.
- **Готово когда:** все не-deterministic runnable assignments имеют terminal accepted/outcome receipt, а active=0.

### N017 — Заморозить terminal `.001` composition inventory

- **Действие:** построить sorted set claims/outcomes/candidates/receipts и закрыть run для новых claims.
- **Код/инфра/аккаунт:** source monitor, GCS object generations, repair contract.
- **Проверка:** expected set reconciliation; duplicates, missing outcome diagnostics и mutable objects блокируют freeze.
- **Готово когда:** immutable `.001-terminal-inventory.json` помечает exact accepted и deterministic residual sets.

### N018 — Создать новый pinned repair revision для residual set

- **Действие:** объединить только проверенные fixes N008–N011 в clean branch без runtime/doc noise.
- **Код/инфра/аккаунт:** Git branch `fix/source-residual-<date>`, source parser/quarantine code.
- **Проверка:** focused + full source tests, `git diff --check`, commit/tree hash, no secret scan matches.
- **Готово когда:** commit pushed и referenced immutable residual authorization.

### N019 — Опубликовать новый GCP residual run `.002`

- **Действие:** создать новый run ID/backend/payload только для unresolved set N017; старый `.001` не переиспользовать.
- **Код/инфра/аккаунт:** `scripts/prepare_gcp_source_pilot.py`, `infra/gcp_corpus_pool/workers`, generated backend HCL с run ID и GCS create-only prefix.
- **Проверка:** plan receipt связывает code, source manifest, assignments, topology, exact image, service accounts и backend prefix; общий `terraform/lane-workers` либо collision fail exit `2`.
- **Готово когда:** apply/readiness receipts подтверждают exact worker count и no unexpected resources.

### N020 — Довести residual `.002` до terminal outcomes

- **Действие:** monitor каждые 30 минут, retry только immutable attempt с exit `75`, классифицировать новый exit `2` в следующий revision.
- **Код/инфра/аккаунт:** source monitor/watchdog, GCS receipts и worker diagnostics.
- **Проверка:** assignment set полностью покрыт completion/outcome; accepted artifacts проходят hash/readback.
- **Готово когда:** unresolved/active/missing artifacts равны нулю или существует approved explicit exclusion receipt.

### N021 — Собрать canonical local+GCP source composition

- **Действие:** логически объединить local base/repair и все полезные accepted generations `.001`–`.005`, включая поздние `.004/.005` outputs; superseded/duplicate producers оставить evidence, но не winners. Сохранить distinct `code`/`commits` route manifests внутри top-level `source` lane.
- **Код/инфра/аккаунт:** `cppmega/data/source_conveyor_composition.py` и composition tests.
- **Проверка:** один primary producer на repo/artifact; tokenizer/repo-list/revision drift и duplicate winners отвергаются.
- **Готово когда:** receipt покрывает exact scope и выдаёт route submanifests, которые final four-kind sealer принимает как один top-level `source` lane без потери `code`/`commits` totals.

### N022 — Выполнить single-writer global source dedup/reducer

- **Действие:** feed composition candidates в canonical order, применить exact/near dedup один раз.
- **Код/инфра/аккаунт:** `scripts/distributed_data_prep/source_reducer.py`, `verify_global_dedup_store.py`.
- **Проверка:** deterministic rerun, winner/duplicate/quarantine conservation, portable dedup receipt.
- **Готово когда:** reducer manifest hash стабилен и не зависит от worker-local DB ordering.

### N023 — Физически отделить Python auxiliary и route classes

- **Действие:** перепаковать source documents по policy, вынеся auxiliary из primary rows.
- **Код/инфра/аккаунт:** `route_packed_source_docs.py`, category sidecars и training status.
- **Проверка:** category totals до/после сходятся; code/commit/aux roots distinct; no lost document IDs.
- **Готово когда:** blocker `Python auxiliary documents are still mixed into main rows` исчезает по receipt, не ручной правке status.

### N024 — Материализовать source Parquet на всех семи lengths

- **Действие:** перепаковать canonical reducer output по единому route-by-fit contract в дизъюнктные 1K–64K sets с real 32K/64K или exact verified-zero.
- **Код/инфра/аккаунт:** packer, `build_macro_routes_megatron_bundle.py`, atomic Parquet publication.
- **Проверка:** footer/versioned training schema/ZSTD, no partial/symlink, каждый document ID ровно в одном fit bucket до lossless split, token range valid; sevenfold repacking запрещён.
- **Готово когда:** seven-bucket inventory имеет physical hashes и не содержит manifest-only 32K/64K.

### N025 — Закрыть source Parquet/sidecar conservation

- **Действие:** аудировать каждый shard и equations documents/valid/trained/pad/duplicates/quarantine.
- **Код/инфра/аккаунт:** `scripts/audit_sidecar_parquet.py`, provenance/side-shape verifiers.
- **Проверка:** graph offsets, document isolation, labels, file/row/token totals; один bad shard блокирует lane.
- **Готово когда:** aggregate conservation receipt объясняет нулевую дельту для code, commit и auxiliary routes.

### N026 — Построить source/commit Megatron prefixes 1K–64K

- **Действие:** конвертировать каждый nonempty audited bucket через canonical tokenizer с полным sidecar profile.
- **Код/инфра/аккаунт:** `scripts/data_prep_parquet_to_megatron.py`, distinct code/commit output generations.
- **Проверка:** MMIDIDX header/offset/data bytes, sequence/document counts и sidecar shapes reread из файлов.
- **Готово когда:** fourteen code/commit bucket states physical или verified-zero и связаны N024/N025 hashes.

### N027 — Опубликовать source lane в GCS create-only

- **Действие:** загрузить composition, Parquet, Megatron, sidecars и audits в content-addressed run prefix.
- **Код/инфра/аккаунт:** GCS bucket, scoped operator upload identity, publication helper.
- **Проверка:** `ifGenerationMatch=0`, server checksum/size/generation и sorted inventory digest.
- **Готово когда:** независимый exact-generation readback воспроизводит source artifact-set SHA.

### N028 — Выполнить source bundle Nebius publish/restore

- **Действие:** построить source release-bundle adapter с Parquet+Megatron+sidecars, опубликовать и восстановить в новый пустой root до удаления workers.
- **Код/инфра/аккаунт:** адаптер к `publish_megatron_bundle_to_nebius_s3.py`/`restore_megatron_bundle_from_nebius_s3.py`, которые сейчас знают Megatron bundle v3/v4, и Nebius S3 profile.
- **Проверка:** no secrets, safe archive paths, per-member hashes, Parquet/MMIDIDX/sidecars reread.
- **Готово когда:** remote restore artifact-set hash совпадает с N027 manifest.

### N029 — Удалить все source transient resources

- **Действие:** после N027/N028 удалять source workers по одному run/worker только после Local SSD diagnostics и exact-generation artifact readback; broad combined destroy запрещён.
- **Код/инфра/аккаунт:** exact isolated Terraform backend lineage/serial каждого run; foundation/bucket и соседние runs вне destroy scope.
- **Проверка:** новый instance-only plan адресует только проверенные VM/attached Local SSD; addresses/policies удаляются отдельным поздним plan после inventory reconciliation. Известный combined plan на `22 destroy`, расширяющий scope до всех 16 VM, применять нельзя.
- **Готово когда:** каждый worker/run имеет собственный `destroyed_verified` receipt, Compute readback absent, remote data всё ещё verifiable и source transient cost inventory равен нулю.

### N030 — Выдать immutable source A→B/C handoff

- **Действие:** связать composition, seven-bucket Parquet/Megatron, GCS/Nebius readback и destroy receipts.
- **Код/инфра/аккаунт:** source lane manifest и handoff schema.
- **Проверка:** consumer загружает только receipt paths/generations, не «latest» directories.
- **Готово когда:** B/C verifier принимает handoff; global `training_ready` остаётся false.

### GitHub PR lane

### N031 — Принять source composition как единственный membership input

- **Действие:** валидировать N030 и извлечь code/commit allowlists без ручного выбора shards.
- **Код/инфра/аккаунт:** `cppmega/data/pr_primary_membership.py`, source composition loader.
- **Проверка:** missing/changed bucket, dedup or artifact-set receipt fail closed.
- **Готово когда:** `github-source-handoff.accepted.json` связывает все семь source buckets.

### N032 — Повторно верифицировать GitHub completion bytes

- **Действие:** открыть store read-only и перепроверить exact completion/gap/quarantine bindings, terminal cursors/pages и immutable `as_of` boundary перед export.
- **Код/инфра/аккаунт:** `verify_pr_completion.py`, `pr_completion.verify7.json`, `prs.sqlite`.
- **Проверка:** stored=declared=2,794,562, repos=460, unverified=0, input hashes unchanged.
- **Готово когда:** receipt не мутирует WAL/store, совпадает со scan ID и имеет explicit follow-up set для records, изменённых после boundary.

### N033 — Выполнить bounded SQLite health check

- **Действие:** `mode=ro`, WAL checkpoint state, `quick_check` и bounded integrity verification без vacuum/reindex.
- **Код/инфра/аккаунт:** `scripts/pr_ingest/pr_store.py`, 49 GB PR store.
- **Проверка:** file size/hash до/после одинаковы; scan count matches N032.
- **Готово когда:** store-health receipt green, но явно `training_ready=false`.

### N034 — Повторно доказать GraphQL gap/quarantine boundaries

- **Действие:** сверить 33,388 target record hashes и физическое исключение 227 quarantined rows.
- **Код/инфра/аккаунт:** gap verify6 и quarantine receipt/Parquet.
- **Проверка:** target=completed, unresolved/skipped/missed=0; quarantined keys absent in exact scan.
- **Готово когда:** boundary receipt связывает оба set digests с N032.

### N035 — Вычислить exact GitHub primary membership

- **Действие:** построить unique `(repo, pr_number)` из accepted commit provenance N031 и exact store N032.
- **Код/инфра/аккаунт:** `build_primary_pr_membership`, clean bridge worktree.
- **Проверка:** every key exists in exact scan; unmatched counters preserved; repo sums equal global unique count.
- **Готово когда:** deterministic membership summary >0 и ≤2,794,562 с input digests.

### N036 — Опубликовать portable GitHub membership Parquet

- **Действие:** atomically записать ZSTD membership и canonical receipt в новый immutable root.
- **Код/инфра/аккаунт:** `publish_primary_pr_membership_inputs`, B run root.
- **Проверка:** schema, unique/sorted key set, file SHA/size и revalidation после close.
- **Готово когда:** membership receipt проходит portable verifier и не ссылается на mutable temp path.

### N037 — Закрыть exporter long-context/zero semantics

- **Действие:** расширить `export_pr_parquet.py` с текущих default 1K–16K до 1K–64K и реализовать PR sidecar producer для actors/reviews/files/links; true empty выдаёт verified-zero.
- **Код/инфра/аккаунт:** exporter, packer, новый sidecar producer, `tests/test_pr_export_batches.py`.
- **Проверка:** positive 32K/64K, true/forged zero, oversized doc, split-conservation и sidecar foreign-key/shape fixtures.
- **Готово когда:** producer, а не только auditor, выпускает seven-length Parquet+sidecars; tests green и revision pinned.

### N038 — Запустить bounded GitHub export canary

- **Действие:** materialize 100 membership-bound PR во всех семи target lengths в отдельный canary root, явно передав `1024,2048,4096,8192,16384,32768,65536`.
- **Код/инфра/аккаунт:** revision N037, `scripts/pr_ingest/export_pr_parquet.py --limit 100 --target-lengths 1024,2048,4096,8192,16384,32768,65536`.
- **Проверка:** only members read, rendered discussion deterministic, provenance/sidecars align, 32K/64K и verified-zero contract реально exercised.
- **Готово когда:** canary conservation/audit receipt green; canary artifacts не копируются в production root.

### N039 — Выполнить полный resumable GitHub export

- **Действие:** обработать N036 keys батчами локально и/или GCP, не читая nonmembers; каждый membership key получает deterministic winner across producers.
- **Код/инфра/аккаунт:** production B root, cloud-lane assignments для GCP, batch manifest, exact-run worker SA и composition reducer.
- **Проверка:** resume принимает только verified done artifacts; input hashes stable; local/GCP duplicates не дают двух winners.
- **Готово когда:** completed unique keys=membership count, duplicate/missing=0 и immutable local+GCP composition receipt существует.

### N040 — Мониторить GitHub attempts по retry contract

- **Действие:** каждые 30 минут фиксировать progress/heartbeat/exit; новый attempt только при exit `75` с diagnostics.
- **Код/инфра/аккаунт:** pipeline watchdog, export `_done.json`, attempt receipts.
- **Проверка:** exit `2`, OOM без 429 и artifact drift не retry in-place.
- **Готово когда:** complete timeline покрывает все attempts и ни один accepted shard не перезаписан.

### N041 — Аудировать GitHub Parquet и sidecars

- **Действие:** проверять footer/schema/ZSTD/token ranges/provenance/graph boundaries каждого production shard.
- **Код/инфра/аккаунт:** `scripts/audit_sidecar_parquet.py`, independent auditor process.
- **Проверка:** no partial/symlink, stable SHA, document isolation и sidecar coordinate agreement.
- **Готово когда:** per-shard receipts и seven-bucket aggregate inventory green.

### N042 — Закрыть GitHub conservation

- **Действие:** сверить membership keys, rendered docs, packed docs, valid/trained/pad и routed bucket IDs.
- **Код/инфра/аккаунт:** export terminal manifest и conservation reducer.
- **Проверка:** rendered count=membership; bucket sets disjoint; no unexplained token/document delta.
- **Готово когда:** GitHub terminal export receipt имеет status complete и seven-state matrix.

### N043 — Построить GitHub Megatron 1K–64K

- **Действие:** конвертировать audited PR Parquet с полным objective/provenance sidecar profile.
- **Код/инфра/аккаунт:** canonical converter и GitHub-specific prefixes.
- **Проверка:** MMIDIDX reread, sample/document counts, long-context receipts.
- **Готово когда:** seven physical/verified-zero prefix states связаны N042.

### N044 — Опубликовать и прочитать GitHub lane из GCS

- **Действие:** create-only upload membership, Parquet, Megatron, sidecars и audits.
- **Код/инфра/аккаунт:** `gs://…/runs/<B_RUN_ID>/github-pr`, scoped upload SA.
- **Проверка:** exact-generation remote readback and artifact-set digest.
- **Готово когда:** GitHub lane manifest `lane_ready=true`, global training false.

### N045 — Выдать GitHub B→C handoff

- **Действие:** опубликовать immutable handoff на N036/N042/N043/N044.
- **Код/инфра/аккаунт:** B lane handoff schema.
- **Проверка:** C verifier не требует SQLite или local absolute path.
- **Готово когда:** consumer принимает portable receipts и rehashes remote artifacts.

### GitLab MR lane

### N046 — Заморозить legacy GitLab roots как superseded evidence

- **Действие:** запретить запись в `gitlab_mr_stream_prod_v1_20260804` и Eigen smoke; сохранить hashes/reasons.
- **Код/инфра/аккаунт:** два completion receipts и stores read-only.
- **Проверка:** legacy contract SHA `915700…6459` отличается от current version contract; no writer locks.
- **Готово когда:** supersession receipt объясняет primary 0/ancillary 11 и Eigen primary 2 без merge stores.

### N047 — Pin current GitLab scanner/exporter revision

- **Действие:** создать отдельный новый detached clean worktree из hardened `cppmega.mlx` checkpoint с `GITLAB_CONTRACT_VERSION=3`; существующий `/Volumes/external/sources/cppmega.mlx` является live runtime tree и не объявляется новым worktree.
- **Код/инфра/аккаунт:** новый path под `/Volumes/external/cppmega_data/worktrees/`, runtime source `/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/gitlab_mr_stream.py`, exporter/store code.
- **Проверка:** source subtree clean, commit/tree/script hashes recorded.
- **Готово когда:** code-binding receipt становится input всех новых smoke/production runs.

### N048 — Повторить полный focused GitLab suite

- **Действие:** прогнать auth, pagination, deterministic gzip, store separation, receipt rebuild и exit taxonomy tests.
- **Код/инфра/аккаунт:** `/Volumes/external/sources/cppmega.mlx/tests/test_gitlab_mr_stream.py` и membership bridge tests.
- **Проверка:** 401/403→2, exhausted 429→75, contract drift→2; no network fixtures deterministic.
- **Готово когда:** test receipt exit `0` связан N047 source hashes.

### N049 — Выпустить canonical three-host repo scope

- **Действие:** создать новый repo list только для Eigen/Mesa/Tor с exact host/project URLs.
- **Код/инфра/аккаунт:** new GitLab B root, repo-list builder.
- **Проверка:** count=3, duplicates/unknown/unresolved=0, sorted digest stable.
- **Готово когда:** scope receipt используется preflight и production manifest.

### N050 — Проверить host-scoped credential mapping

- **Действие:** проверить runtime invariant «ровно один auth mode на host» (`--token-env` xor `--public-host`) и применить production policy: три token env names без public fallback.
- **Код/инфра/аккаунт:** GitLab auth resolver, три `read_api` credentials; values не логировать.
- **Проверка:** missing/extra/overlap empty; secret scan wrappers/logs/receipts.
- **Готово когда:** auth-scope receipt хранит host→env name/fingerprint и no secret values.

### N051 — Выполнить bounded API preflight на трёх GitLab hosts

- **Действие:** по одному inventory и candidate-detail probe на host до полного fetch.
- **Код/инфра/аккаунт:** `_preflight_project_access`, scoped credentials N050.
- **Проверка:** exact HTTPS host/project, status 200, bounded body/decompression; 401/403/429 taxonomy.
- **Готово когда:** три redacted preflight receipts green либо deterministic blocker явно закрывает production launch.

### N052 — Выполнить три независимых authenticated smoke

- **Действие:** отдельный one-repo root для Eigen, Mesa и Tor, чтобы fault одного host не маскировал другие.
- **Код/инфра/аккаунт:** GitLab scanner, three smoke run IDs/stores.
- **Проверка:** completion receipt rebuild из manifest/store/sidecars; no cross-root SQLite copy.
- **Готово когда:** каждый host имеет terminal verified smoke или исправленный deterministic defect с новым smoke generation.

### N053 — Зарезервировать новый GitLab production root

- **Действие:** создать collision-free local/GCS roots, не дописывая legacy production.
- **Код/инфра/аккаунт:** `/Volumes/external/cppmega_data/gitlab-mr-<RUN_ID>` и GCS B prefix.
- **Проверка:** roots absent/empty, create-only policy, run ID bound N047/N049/N050.
- **Готово когда:** output allocation receipt принят до первого API page.

### N054 — Выполнить full authenticated three-host scan

- **Действие:** inventory/detail fetch для всех N049 projects с resumable per-host pagination и immutable `as_of` boundary.
- **Код/инфра/аккаунт:** `gitlab_mr_stream.py`, token env mappings, primary/ancillary stores.
- **Проверка:** page cursors/ETags, terminal pages, rate-limit receipts, deterministic gzip, route conservation и changed-after-boundary follow-up set.
- **Готово когда:** terminal manifest имеет full host coverage, no unresolved page/candidate и bound observation window.

### N055 — Перестроить GitLab completion receipt из bytes

- **Действие:** после close stores пересчитать counts/hashes/sidecar sets независимо от writer memory.
- **Код/инфра/аккаунт:** completion verifier, N053 root.
- **Проверка:** quick/integrity checks, inventory=candidate+noncandidate, primary+ancillary+excluded+terminal conservation.
- **Готово когда:** current-contract completion status verified, но membership gate всё ещё false.

### N056 — Построить exact GitLab primary membership

- **Действие:** применить bridge commit `4ec96cf5` к N030 source composition и N055 production receipt.
- **Код/инфра/аккаунт:** `build_gitlab_primary_membership.py`, clean bridge worktree.
- **Проверка:** source/store/contract bindings, unique host/project/IID keys, unmatched counters.
- **Готово когда:** portable membership receipt проходит полный текущий focused suite и production verifier; count тестов берётся из pinned revision, а не исторического handoff.

### N057 — Запустить GitLab export canary

- **Действие:** реализовать/зафиксировать отдельный GitLab membership→document exporter contract и materialize bounded members всех hosts в seven-bucket canary.
- **Код/инфра/аккаунт:** GitLab exporter/adapter, N056 membership; GitHub exporter не считается совместимым без schema tests.
- **Проверка:** only primary store reads, ancillary key exclusion, 32K/64K/verified-zero semantics, sidecars.
- **Готово когда:** canary audit/conservation green и production root остаётся untouched.

### N058 — Выполнить полный GitLab MR export

- **Действие:** resumable batches по N056 membership в local/GCP immutable roots с deterministic cross-producer winner.
- **Код/инфра/аккаунт:** GitLab exporter, cloud-lane assignments, composition reducer и 30-minute watchdog.
- **Проверка:** completed keys=membership; no legacy/smoke accepted; local/GCP duplicates collapse по bound policy; attempts follow 75/2.
- **Готово когда:** terminal composition/export receipt покрывает семь buckets и одного winner на MR key.

### N059 — Аудировать и конвертировать GitLab lane

- **Действие:** Parquet/sidecar conservation, затем Megatron 1K–64K.
- **Код/инфра/аккаунт:** sidecar auditor, canonical converter, GitLab prefixes.
- **Проверка:** footer/schema/ZSTD, document/token equations, MMIDIDX reread и long-context receipts.
- **Готово когда:** GitLab lane manifest имеет seven physical/verified-zero states.

### N060 — Опубликовать GitLab lane и B→C handoff

- **Действие:** create-only GCS upload, exact-generation readback и portable handoff.
- **Код/инфра/аккаунт:** `gs://…/<B_RUN_ID>/gitlab-mr`, scoped operator SA.
- **Проверка:** artifact-set hash and remote generations; C не требует local SQLite.
- **Готово когда:** GitLab handoff accepted, global training false.

### CI CASE5 lane

### N061 — Заморозить inventory живых CI stores/fetchers

- **Действие:** снять per-store sizes/hashes/cursors/process-safe checkpoints и immutable `as_of` boundary, не суммируя overlapping stores.
- **Код/инфра/аккаунт:** `scripts/ci_stream_inventory.py`, fetch state/receipts, local data roots.
- **Проверка:** checkpoint after writer flush/close либо explicitly live upper-bound; repo partitions identified.
- **Готово когда:** exact input ledger отделяет frozen threshold, live shards и legacy sample.

### N062 — Довести local CI fetch shards до terminal freeze

- **Действие:** продолжать существующие fetchers, retry только confirmed 429, затем закрыть cursors/WAL и publish terminal receipts.
- **Код/инфра/аккаунт:** `scripts/ci_stream_fetch.py`, token ledger, shard states.
- **Проверка:** exact repo coverage, page/run boundary, no duplicate active writer, no secret logs.
- **Готово когда:** каждый shard terminal verified или finite residual inventory создан.

### N063 — Обработать CI fetch residuals

- **Действие:** новый residual run для missing repos/pages/jobs, не broad refetch successful shards.
- **Код/инфра/аккаунт:** fetch state migration/resume scripts, immutable authorization.
- **Проверка:** residual set digest, 75/2 taxonomy, existing shard hashes unchanged.
- **Готово когда:** repo/run/workflow/job inventory scope закрыт без unresolved records.

### N064 — Слить CI shards логически

- **Действие:** построить canonical union manifests/CAS occurrence ledger и deterministic winner across local/GCP producers без копирования mutable SQLite trees.
- **Код/инфра/аккаунт:** `scripts/merge_ci_stream_shards.py`, `scripts/clone_ci_stream_union_for_resume.py` только по contract.
- **Проверка:** key uniqueness, shard coverage, duplicate payload/occurrence distinction, input hashes.
- **Готово когда:** terminal union receipt воспроизводим и read-only.

### N065 — Проверить CI CAS и preserved archives

- **Действие:** hash blobs, validate zlib/gzip, recover only receipt-authorized preserved archives.
- **Код/инфра/аккаунт:** `scripts/ci_content_store.py`, `scripts/ci_zlib_evidence.py`, `scripts/recover_ci_preserved_archives.py`.
- **Проверка:** CAS blob hash, decompression bounds, orphan/set-based receipt checks.
- **Готово когда:** corrupt/missing/quarantined sets explicit, unexplained missing payload=0.

### N066 — Выполнить global CI payload/occurrence dedup

- **Действие:** вычислить exact global union, сохранив все repo/run/job/step occurrence bindings.
- **Код/инфра/аккаунт:** CI reducer, canonical manifest order, portable dedup store.
- **Проверка:** store-local 295.7B upper bound заменён exact global totals; payload dedup не теряет occurrences.
- **Готово когда:** deterministic dedup receipt имеет winner/duplicate/conservation equations.

### N067 — Опубликовать exact CASE5 input в GCS

- **Действие:** создать manifest-bound frozen SQLite/CAS/occurrence inputs из N064–N066 и publish chunks напрямую в GCS, не требуя локальных 500 GiB.
- **Код/инфра/аккаунт:** `scripts/prepare_gcp_cloud_lane_payload.py`, `scripts/distributed_data_prep/ci_case5_snapshot.py`, GCS inputs prefix.
- **Проверка:** frozen copy без WAL/SHM, create-only upload, chunk/archive SHA, exact-generation readback; historical `61,311,228,208` threshold отдельно верифицируется и не складывается с union.
- **Готово когда:** production workers могут стартовать только с immutable GCS inputs + manifest, accepted exact global total N066 и explicit threshold lineage.

### N068 — Довести CASE5 smokes `.003` и `.004` до terminal evidence

- **Действие:** не уничтожать `.003`, пока process/CPU/read counters показывают полезную materialization; для `.004` дождаться assignment completion/physical output либо immutable deterministic failure. Не считать sustained heartbeat completion.
- **Код/инфра/аккаунт:** `ci-case5-smoke-20260808-003`, `ci-case5-smoke-20260809-004`, scoped Compute/GCS identities и isolated backends.
- **Проверка:** systemd/PID/cgroup/Local SSD/SQLite phase evidence; claim/completion/output receipt; exact-generation GCS readback. Retry только при `confirmed_http_429=true`/exit `75`.
- **Готово когда:** каждый smoke имеет terminal+readback+scoped destroy receipt либо immutable supersession diagnostic; минимум один fixed-revision smoke имеет nonzero audited Parquet output.

### N069 — Привязать landed CASE5 hardening к следующему runtime attempt

- **Действие:** использовать landed CASE5 merge `a65db87c` и source retry merge `52472d83` либо их clean descendant при формировании следующего CASE5 payload; не перезапускать `.003/.004` только ради подмены revision и не смешивать их evidence с новым attempt.
- **Код/инфра/аккаунт:** `cloud_lane_pool_worker.py`, `cloud_lane_heartbeat.py`, runner template, payload manifest, run-specific GCS prefix и isolated Terraform backend.
- **Проверка:** payload/code revision/tree SHA совпадают с apply/claim/heartbeat/completion receipts; immutable heartbeat проходит self-digest и exact-generation readback; mixed 429+deterministic остаётся exit `2`, только pure confirmed 429 может дать `75`.
- **Готово когда:** clean landed commit/tree SHA входит в новый payload/apply receipt, regression receipt сохранён immutable, а новый worker публикует heartbeat и terminal evidence по hardened contract.

### N070 — Спланировать production CASE5 pool по measured smoke

- **Действие:** по N068 оценить machines/shards/Local SSD/runtime/cost, включая сценарий `4 × 16 vCPU / 64 GB`; theoretical «10 Gbit» не использовать как измерение.
- **Код/инфра/аккаунт:** CASE5-specific estimator/benchmark receipt, workers tfvars и Compute quotas; source-only estimator не считать CI network/SSD/cost model без расширения.
- **Проверка:** минимум три trials фиксируют CPU, RSS, Local SSD/GCS p50/p95 throughput, reducer bottleneck и scaling efficiency; quotas confirmed.
- **Готово когда:** plan receipt содержит measured baseline, ETA interval/cost для 1/4/16 workers, assignment geometry и teardown policy.

### N071 — Выполнить pre-apply Terraform/bootstrap gate

- **Действие:** до production apply прогнать fmt/init-backend=false/validate/test foundation/workers/pilot, generated backend collision tests и exact image resolution.
- **Код/инфра/аккаунт:** `infra/gcp_corpus_pool/foundation`, `infra/gcp_corpus_pool/workers`, `infra/gcp_corpus_pool/pilot`, pinned Terraform/providers.
- **Проверка:** no startup drift, no secrets in plans, distinct concurrent backend prefixes, Local SSD mount/download-on-target-filesystem и heartbeat service tests.
- **Готово когда:** immutable pre-apply test receipt exit `0` связан N069 commit и exact production plan inputs.

### N072 — Выполнить pre-apply IAM/network/cost gate

- **Действие:** до VM apply сузить worker access до exact run prefix и проверить operator/worker identities, firewall/gVNIC/network tier/static-IP necessity/quotas/budget.
- **Код/инфра/аккаунт:** scoped operator SA override, exact-run worker IAM condition, Cloud Billing/Compute read-only views.
- **Проверка:** sibling-run/state/foundation read/write denied; no secret metadata; measured network/SSD receipt и maximum-cost bound приложены.
- **Готово когда:** security/network/cost gate green и production apply не требует интерактивной human reauth.

### N073 — Развернуть production CASE5 workers

- **Действие:** только после N061–N072 apply generated run-ID backend с run-scoped VM/IP/policies, exact image и Local SSD topology.
- **Код/инфра/аккаунт:** `infra/gcp_corpus_pool/workers`, operator SA credential override.
- **Проверка:** plan/backend/state-serial/image/payload digests, ready heartbeats, expected instances only; GCS control-plane smoke and negative IAM probe.
- **Готово когда:** expected slots готовы, exact N067 input принят и assignment claims начались без unexpected resources.

### N074 — Материализовать production CASE5 Parquet

- **Действие:** запустить `scripts/export_ci_content_store_case5.py` поверх N067 exact union/dedup input в workers N073.
- **Код/инфра/аккаунт:** CASE5 exporter, GCS input/output prefixes, Local SSD scratch.
- **Проверка:** resumable immutable done receipts, no mutable store reads, exact membership/document/token conservation; partial generations не authoritative.
- **Готово когда:** terminal export receipt покрывает весь exact CASE5 membership и output прошёл independent footer/schema/hash readback.

### N075 — Материализовать полный CI sidecar set

- **Действие:** выпустить repo/run/workflow/job/step/actor/runner, language/platform/toolchain/build, commands/targets, tests, diagnostics, entities/edges и chunks.
- **Код/инфра/аккаунт:** `scripts/ci_log_sidecars.py` для classification/tests, `scripts/ci_source_sidecars.py` и `scripts/ci_source_binding_projection.py` для source bindings; `scripts/fetch_ci_diagnostics.py` только bounded evidence.
- **Проверка:** shape/foreign-key/offset/hash audits, occurrence conservation и secret redaction; nullable fields explicit.
- **Готово когда:** каждый Parquet document имеет согласованный sidecar binding либо schema-authorized absence.

### N076 — Закрыть CI seven-length ladder

- **Действие:** route/pack N074 documents по disjoint route-by-fit contract на 1K–64K; 32K/64K physical либо exact verified-zero.
- **Код/инфра/аккаунт:** canonical packer, versioned training schema и distributed sealing contract.
- **Проверка:** disjoint routed IDs, lossless oversized split, capacity/token range, no forged empty.
- **Готово когда:** CI Parquet inventory содержит семь audited states и exact totals.

### N077 — Построить CI Megatron 1K–64K

- **Действие:** конвертировать N076 с полными objective/graph/provenance sidecars.
- **Код/инфра/аккаунт:** `scripts/data_prep_parquet_to_megatron.py`, CI-specific prefixes.
- **Проверка:** MMIDIDX reread, `.bin/.idx` counts/bytes, document isolation, long-context receipts.
- **Готово когда:** seven physical/verified-zero prefixes связаны с N066/N074–N076 receipts.

### N078 — Запечатать и опубликовать CI lane

- **Действие:** собрать immutable lane manifest, create-only publish N074–N077 и выполнить exact-generation GCS readback.
- **Код/инфра/аккаунт:** CI lane manifest builder, GCS, scoped operator identity.
- **Проверка:** artifact-set hash remote=local; frozen input/dedup/Parquet/Megatron/sidecar lineage complete; `training_ready=false`.
- **Готово когда:** portable CI lane handoff принят independent verifier без local SQLite path.

### N079 — Удалить CI pool безопасно

- **Действие:** после N078 удалять run-scoped instances worker-by-worker; Local SSD diagnostics/output generations проверять до destroy, addresses/policy — отдельным поздним plan.
- **Код/инфра/аккаунт:** exact N073 backend lineage/serial и Compute readback.
- **Проверка:** instance-only plan не затрагивает sibling runs/foundation; broad expanded destroy запрещён; post-destroy GCS artifacts остаются green.
- **Готово когда:** VM/Local SSD/IP/policy отсутствуют, retained non-expensive foundation explicit и destroy receipts immutable.

### N080 — Выдать CI C-lane handoff

- **Действие:** связать N061–N079 inventory/freeze/dedup/export/sidecar/Megatron/readback/destroy receipts в один content-addressed handoff.
- **Код/инфра/аккаунт:** CI lane handoff schema и final GCS pointer.
- **Проверка:** consumer восстанавливает artifact-set SHA только по exact generations; store-local upper bounds/legacy sample не принимаются как production total.
- **Готово когда:** CI handoff portable, `lane_ready=true`, global `training_ready=false`, transient CI compute cost zero.

### Общий four-lane seal и публикация

### N081 — Принять source handoff fail-closed

- **Действие:** C загружает N030 и независимо revalidates composition, source artifacts/readbacks и source teardown.
- **Код/инфра/аккаунт:** integration worktree, `seal_outputs.py`, GCS read-only.
- **Проверка:** no local mutable paths, seven-state source matrix, artifact hashes/generations exact.
- **Готово когда:** source input acceptance receipt green.

### N082 — Принять GitHub handoff fail-closed

- **Действие:** revalidate N045 membership, conservation, Parquet/Megatron and GCS readback.
- **Код/инфра/аккаунт:** integration sealer GitHub lane input.
- **Проверка:** exact scan/source bindings and seven-state matrix; no SQLite dependency.
- **Готово когда:** GitHub input acceptance receipt green.

### N083 — Принять GitLab handoff fail-closed

- **Действие:** revalidate N060 current contract, three-host coverage, membership and artifacts.
- **Код/инфра/аккаунт:** integration sealer GitLab lane input.
- **Проверка:** legacy/smoke roots cannot satisfy production input; seven-state matrix complete.
- **Готово когда:** GitLab input acceptance receipt green.

### N084 — Принять CI handoff fail-closed

- **Действие:** revalidate N080 frozen input, global dedup, sidecars, Parquet/Megatron and destroy receipt.
- **Код/инфра/аккаунт:** integration sealer CI lane input.
- **Проверка:** no store-local upper-bound used as exact total; seven-state matrix complete.
- **Готово когда:** CI input acceptance receipt green.

### N085 — Собрать полную матрицу 4×7

- **Действие:** перечислить sealer kinds `source`, `github_pr`, `gitlab_mr`, `ci` × семь lengths; пользовательская code lane представлена top-level `source`, чьи cells агрегируют отдельно проверенные `code`/`commits` route submanifests N021/N026.
- **Код/инфра/аккаунт:** `scripts/distributed_data_prep/seal_outputs.py`.
- **Проверка:** exactly 28 unique top-level cells; source route totals/code+commit lineage сохраняются; missing, duplicate, manifest-only, forged-zero fail.
- **Готово когда:** matrix coverage receipt green и связан N081–N084.

### N086 — Повторно открыть каждый Parquet artifact

- **Действие:** independent integration auditor перечитывает footer/schema/row groups/compression и пересчитывает totals.
- **Код/инфра/аккаунт:** `audit_sidecar_parquet.py`, remote-restored files.
- **Проверка:** no symlink/partial, physical bytes/hash, versioned training-row schema (не ledger schema), valid/trained/pad/capacity and sidecar alignment.
- **Готово когда:** один aggregate physical Parquet audit покрывает все materialized cells N085.

### N087 — Проверить единый tokenizer/objective contract

- **Действие:** сравнить tokenizer, vocab, token range, label masking и objective artifact across four lanes.
- **Код/инфра/аккаунт:** tokenizer verifier, objective manifests, Megatron converter receipts.
- **Проверка:** mismatch hash/version/range/column semantics fail; no cwd-dependent lookup.
- **Готово когда:** global tokenizer/objective receipt green.

### N088 — Проверить global provenance и cross-lane dedup policy

- **Действие:** доказать per-lane deterministic dedup, уникальность document IDs/lineage и обработку intentional cross-lane duplicates отдельной release policy.
- **Код/инфра/аккаунт:** provenance verifier, source/PR/MR/CI dedup receipts и cross-lane policy; source-only verifier не является global proof.
- **Проверка:** sample→lane→source object traversal, no dangling graph/provenance keys, conservation preserved.
- **Готово когда:** global provenance/dedup receipt имеет zero unexplained collision/delta.

### N089 — Собрать content-addressed release manifest

- **Действие:** перечислить все Parquet, Megatron, sidecars, receipts, tokenizer и code/toolchain bindings.
- **Код/инфра/аккаунт:** release manifest builder, no mutable absolute paths.
- **Проверка:** sorted deterministic entries, safe relative paths, role/format/size/SHA/generation complete.
- **Готово когда:** manifest artifact-set SHA воспроизводим из N081–N088.

### N090 — Выполнить финальный distributed seal

- **Действие:** сначала изменить `seal_outputs.py`, чтобы pre-publication seal выдавал `sealed_not_restored`/`training_ready=false`, затем запустить его на N089.
- **Код/инфра/аккаунт:** `seal_outputs.py`; текущие `training_ready = not blocking_reasons` и `publication_authorized=training_ready` нужно разделить.
- **Проверка:** regression доказывает: 28 cells разрешают publication, но не training; только post-restore/loader N099 выставляет final flag.
- **Готово когда:** seal complete, `publication_authorized=true`, `training_ready=false`; преждевременный true отвергается тестом.

### N091 — Опубликовать release objects в финальный GCS prefix

- **Действие:** реализовать production four-lane uploader из plan-only handoff, затем create-only upload/reuse exact generations и publish pointer последним.
- **Код/инфра/аккаунт:** uploader/exact-generation verifier, final GCS prefix, operator SA; source reducer smoke publisher недостаточен.
- **Проверка:** generation/size/server checksum/local SHA каждого object; mismatched existing key fail.
- **Готово когда:** GCS publication/readback receipt artifact-set SHA=N089.

### N092 — Опубликовать финальный bundle в Nebius

- **Действие:** построить distributed-release bundle schema/adapter с Parquet+Megatron+sidecars+receipts, затем сжать и resumably upload.
- **Код/инфра/аккаунт:** adapter к `publish_megatron_bundle_to_nebius_s3.py`; принять distributed seal v1, не только Megatron bundle v3/v4.
- **Проверка:** dry-run, multipart checksums, object inventory, no credentials in logs/receipts.
- **Готово когда:** Nebius publication receipt связан N089/N090/N091.

### N093 — Выполнить независимый Nebius restore

- **Действие:** matching distributed-release adapter на пустом root скачивает release без uploader cache/original paths.
- **Код/инфра/аккаунт:** adapter к restore script `--require-empty-output-root`, exact release ID.
- **Проверка:** archive path safety, members/hash, PyArrow Parquet footer/schema, MMIDIDX, all sidecars and 28-state inventory.
- **Готово когда:** restore receipt artifact-set SHA=N089 и zero missing/corrupt members.

### N094 — Запустить bounded Megatron consumption smoke

- **Действие:** открыть каждый nonempty lane/length prefix через training data loader и прочитать bounded batches.
- **Код/инфра/аккаунт:** настоящий training data loader из pinned Megatron environment и restored Nebius root; `verify_dataset_megacpp.py` — дополнительная, не заменяющая проверка.
- **Проверка:** index/data/sidecars load, token range, batch geometry, no cross-document labels; no model training claim.
- **Готово когда:** 28 cells materialized/verified-zero и все materialized cells имеют green loader smoke.

### N095 — Удалить оставшиеся reducer/export transient resources

- **Действие:** после N093/N094 уничтожить final reducer/export VM, disks, IP, policies и temporary staging resources.
- **Код/инфра/аккаунт:** all remaining isolated Terraform backends and cloud inventories.
- **Проверка:** destroy plans scoped; Compute readback absent; GCS/Nebius readbacks remain green.
- **Готово когда:** expensive transient inventory=0 и per-run destroy receipts сохранены.

### Финальная эксплуатационная приёмка

### N096 — Повторить Terraform/bootstrap verification после всех cloud runs

- **Действие:** повторить pre-apply gate N071 на финальном release code, сверить actual apply payloads и destroy backends, чтобы исключить drift между тестом и production.
- **Код/инфра/аккаунт:** `infra/gcp_corpus_pool/foundation`, `infra/gcp_corpus_pool/workers`, `infra/gcp_corpus_pool/pilot`.
- **Проверка:** pinned Terraform/providers, distinct concurrent backend prefixes, exact image self-link, no startup drift и no secret material in plans.
- **Готово когда:** CI/local test receipt exit `0` связан release code commit.

### N097 — Провести итоговый IAM/network/cost audit

- **Действие:** повторить pre-apply gate N072 после teardown: проверить actual IAM/firewall/network/static addresses/quotas/billing и удалить временные grants/resources.
- **Код/инфра/аккаунт:** GCP scoped credentials, foundation outputs, Cloud Billing/Compute read-only views.
- **Проверка:** sibling-run/state access denied, p50/p95 network/SSD receipt, no public secrets/orphan IP/VM/SSD/policy, estimate-vs-actual per run.
- **Готово когда:** security/cost audit receipt green либо explicit non-expensive retained foundation list.

### N098 — Обновить watchdog и предварительный честный status после cleanup

- **Действие:** pipeline watchdog проверяет terminal receipts/readbacks вместо старых active runs; status пересобирается из seals, но до N099 остаётся `training_ready=false`.
- **Код/инфра/аккаунт:** launchd dispatchers, `report_training_data_status.py`, status config и `/Volumes/external/sources/cppmega.mlx/outputs/training_data_status/current.md`.
- **Проверка:** two fresh 30-minute reports, no retry of terminal/deterministic runs, no overlapping token addition.
- **Готово когда:** preliminary current status перечисляет exact valid/trained по четырём lanes, blockers resolved и явно ждёт final completion audit N099.

### N099 — Выполнить requirement-by-requirement completion audit

- **Действие:** сопоставить W001–W100/N001–N098 с authoritative receipts/tests/readbacks и отметить missing/contradicted evidence.
- **Код/инфра/аккаунт:** этот документ, release manifest, Git/GCP/GCS/Nebius/test evidence.
- **Проверка:** ни одно требование не закрывается intent, partial store, narrow verifier, pre-publication seal или отсутствием ошибки; transition N090→N099 проверяется отдельно.
- **Готово когда:** все explicit requirements доказаны и отдельный immutable final completion receipt получает `training_ready=true`; при любом missing/contradicted evidence флаг остаётся false.

### N100 — Передать training owner финальный immutable handoff

- **Действие:** после N099 заново сгенерировать final status из completion receipt, затем выдать название release, GCS/Nebius exact pointers, manifest/seal/restore/smoke/destroy receipts и restoration command contract.
- **Код/инфра/аккаунт:** `scripts/report_training_data_status.py`, signed/content-addressed handoff document и changelog в Git.
- **Проверка:** final status SHA связан N099; независимый consumer по одному handoff восстанавливает inventory и подтверждает artifact-set SHA без локальных Mac paths.
- **Готово когда:** post-audit status показывает `training_ready=true`, handoff accepted, docs/code commits pushed, worktrees clean и active corpus goal закрывается по полному evidence.

---

## Критический путь

```text
status-updater merge + cloud-monitor deploy -> durable local-runtime verification
  -> exact source residual -> source composition/dedup
  -> source 1K–64K seal -> exact GitHub/GitLab membership
  -> PR/MR 1K–64K seals -> CI terminal freeze/union/dedup/GCS input
  -> CASE5 smoke -> Terraform/IAM pre-apply gates -> CI production seal
  -> 4×7 integration -> GCS/Nebius readback -> loader smoke
  -> destroy transient compute -> final completion audit
  -> post-audit status -> immutable handoff
```

Главная неопределённость по времени — deterministic parser repair и CI CASE5 reducer/export, а не сам факт наличия дополнительных VM. Добавление workers ускоряет только runnable assignments; оно не превращает exit `2`, непроверенный membership или staged CAS в готовые training artifacts.
