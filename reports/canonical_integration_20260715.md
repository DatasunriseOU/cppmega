# Каноническая интеграция `cppmega` и `cppmega.mlx`

**Снимок фактов:** 2026-07-15 20:52 CEST (повторная проверка git и receipts)
**Каноническая ветка в обоих checkout:** `codex/canonical-complete-20260715`
**GitHub:** [`DatasunriseOU/cppmega`](https://github.com/DatasunriseOU/cppmega) · [`DatasunriseOU/cppmega_mlx`](https://github.com/DatasunriseOU/cppmega_mlx)
**Область этого документа:** интеграция кода, контрактов данных, локальных проверок и текущего производственного конвейера. CASE5 span-fixes не считаются принятыми без v7 rerun/receipt. Завершённый H200 gate описывается только в точном проверенном объёме и не экстраполируется на полноценный training run.

**Правило чтения snapshot:** числа `CASE5 v7` ниже относятся к зафиксированному снимку run (`14:40-14:54 CEST`, `77 done / 11 failed`). Файл `_done.json` является живым и после этого снимка продолжал меняться; его позднее состояние не подменяет приведённую историческую receipt-точку и не является финальным corpus verdict.

## Навигация

- [Короткий итог](#summary)
- [Зачем нужны два репозитория](#roles)
- [Что и откуда перенесено](#ported)
- [KSH, Python и reserved IDs 245-248](#contracts)
- [Независимый путь source -> parquet -> Megatron](#pipeline)
- [Роль MLX runtime и визуального инструментария](#mlx-role)
- [Точные коммиты](#commits)
- [Тесты и статические проверки](#tests)
- [Detours: локальный end-to-end receipt](#detours)
- [CASE5 v4: токены и полные шаги](#case5-v4)
- [CASE5 v7: живой статус](#case5-v7)
- [A2 std DB и идентичность символов](#a2)
- [FIM/AST-FIM и physical identities](#fim-ast-fim)
- [Незавершённые проблемы и риски](#risks)
- [Индекс доказательств](#evidence)
- [Изменённые файлы](#changed-files)

<a id="summary"></a>
## 1. Короткий итог

Интеграция собрала два самостоятельных продукта с общими форматами данных, но разными runtime-обязанностями:

1. `cppmega` теперь является каноническим Megatron-first/CUDA репозиторием. Он владеет модельным runtime, подготовкой данных, clang-индексацией, parquet, graph sidecars и конвертацией в Megatron `.bin/.idx`. Production-код не должен импортировать `cppmega_mlx` или `nanochat` и не должен искать их по `../...` путям.
2. `cppmega.mlx` остаётся самостоятельным Apple Silicon репозиторием: MLX/Metal runtime, локальные проверки и бенчмарки, Path C/TileLang исследования, визуальный конструктор и инспектор данных. Он хранит эквивалентные data-contracts в своём namespace и не является обязательной runtime-зависимостью root.
3. В обоих репозиториях KSH и Python расширяют существующий `v1` контракт на заранее зарезервированных ID `245-248`. Сам tokenizer artifact не перенумерован и не изменён.
4. Локально доказан реальный root-only проход `Detours source -> packed parquet -> fail-closed audit -> Megatron MMIDIDX + token/graph sidecars`.
5. CASE5 v4 содержит `1 411 702 081` trained tokens и **7 178** schedule-valid полных шагов. Значение `7 180` получается только при неверном для расписания предположении, что остатки разных бакетов взаимозаменяемы.
6. Зафиксированный CASE5 v7 snapshot даёт `153 090 795` trained tokens и `776` полных шагов по бакетам; на этой receipt-точке manifest показывал `77 done / 11 failed`. Run закреплён на старом `1ba36f7`, а не на текущем MLX HEAD `d5fdec0`, поэтому snapshot нельзя объявлять финальным или clean.
7. A2 DB реально содержит `15 485` строк `std` и `0` строк libiberty. При этом только `5 113` distinct qnames, что прямо доказывает: `qname` не является уникальным ключом.
8. Nebius H200 cycle завершён на exact root commit `2b5dd3d`: full `tests/test_dsa_splitk_indexer_loss.py` дал `7 passed`, финальный required CUDA/TE/Megatron gate - `11 passed`. Writer/reader receipt для Detours фиксирует `46` documents, `45 257` valid tokens, `45 063` trained tokens, `20` token-aligned sidecars и `11` graph CSR/ragged sidecars. VM `computeinstance-e00dezxd2mdw060cem` удалена.
9. Три обязательных unresolved-блока остаются открытыми: FIM/AST-FIM eligibility должна fail-close на отсутствующих или смешанных physical source identities; Xbox Live Source не имеет подтверждённой provenance identity; root environment packaging всё ещё зависит от внешнего venv и не объявляет runtime dependencies.

<a id="roles"></a>
## 2. Зачем нужны два репозитория

| Репозиторий | Основная задача | Что он обязан делать самостоятельно | Что не является его задачей |
|---|---|---|---|
| `cppmega` | Production training и data plane для NAM56R в Megatron/PyTorch/CUDA | Подготовить corpus, построить semantic/graph sidecars, сконвертировать parquet в Megatron, загрузить sidecars, применить DSA/graph supervision, запускать CUDA/TE на GPU | MLX UI, Apple Metal runtime, импорт кода из соседнего `cppmega.mlx` |
| `cppmega.mlx` | Локальный Apple Silicon runtime и исследовательско-визуальный контур | Проверять модельные/data contracts на macOS, выполнять MLX eager/compiled smokes, исследовать Metal/TileLang Path C, визуально инспектировать токены/графы/данные | Быть скрытым backend для root, подменять CUDA runtime, заявлять distributed H200 readiness без отдельного GPU receipt |

Коротко: общий язык между репозиториями - это **зафиксированные schemas, hashes и sidecar semantics**, а не Python-import через соседний checkout.

Основные локальные точки входа:

- root-модель и production назначение: [README.md](/Volumes/external/sources/cppmega/README.md:1);
- MLX назначение и ограничения: [README.md](/Volumes/external/sources/cppmega.mlx/README.md:1);
- root runtime-isolation contract: [test_megatron_root_runtime_independence.py](/Volumes/external/sources/cppmega/tests/test_megatron_root_runtime_independence.py:1).

<a id="ported"></a>
## 3. Что и откуда перенесено

### 3.1. `nanochat` -> оба канонических репозитория

Из `nanochat` взята не модельная зависимость, а проверенная механика data preparation:

- извлечение git history;
- преобразование clang-enriched JSONL в tokenized parquet;
- packing fixed-length rows;
- document identity, corpus statistics, platform vocabulary и packed-row schemas;
- token budgets, memory guards и atomic publication.

В `cppmega` это теперь root-owned код:

- [cppmega/data/nanochat_pipeline](/Volumes/external/sources/cppmega/cppmega/data/nanochat_pipeline/__init__.py:1);
- [scripts/nanochat_data/clang_enriched_to_parquet.py](/Volumes/external/sources/cppmega/scripts/nanochat_data/clang_enriched_to_parquet.py:1);
- [scripts/nanochat_data/pack_enriched_rows.py](/Volumes/external/sources/cppmega/scripts/nanochat_data/pack_enriched_rows.py:1);
- [scripts/streaming_conveyor.py](/Volumes/external/sources/cppmega/scripts/streaming_conveyor.py:1).

В `cppmega.mlx` эквивалент остаётся в собственном namespace:

- [cppmega_mlx/data/nanochat_pipeline](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/nanochat_pipeline/__init__.py:1);
- [scripts/nanochat_data/clang_enriched_to_parquet.py](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/clang_enriched_to_parquet.py:1);
- [scripts/nanochat_data/pack_enriched_rows.py](/Volumes/external/sources/cppmega.mlx/scripts/nanochat_data/pack_enriched_rows.py:1).

Важно: локальный smoke запускал дочерние процессы через внешний `nanochat/.venv`, потому что там есть PyArrow/libclang/tokenizer dependencies. Это ограничение локального environment, а не production import dependency.

### 3.2. `cppmega.mlx` -> канонический `cppmega`

Из MLX-ветки в root перенесён и переименован самостоятельный source-processing слой:

- domain schema, packet и ingestion;
- build parsers: CMake, Make, Ninja, Bazel, Autotools;
- shell parsers, включая KSH;
- diagnostic parsers;
- Python stdlib `ast`/`tokenize` parser;
- language/build context;
- prompt graph и prompt graph index;
- tokenizer contract, fingerprint и artifact;
- canonical project/source/symbol identities;
- clang project indexer, commit processor и dedup store;
- PR ingestion/export, cross-repo symbol DB, audit/repair/report tooling.

Ключевые root-файлы:

- [domain_ingestion.py](/Volumes/external/sources/cppmega/cppmega/data/domain_ingestion.py:1);
- [domain_schema.py](/Volumes/external/sources/cppmega/cppmega/data/domain_schema.py:1);
- [prompt_graph.py](/Volumes/external/sources/cppmega/cppmega/data/prompt_graph.py:1);
- [source_identity.py](/Volumes/external/sources/cppmega/cppmega/data/source_identity.py:1);
- [symbol_identity.py](/Volumes/external/sources/cppmega/cppmega/symbol_identity.py:1);
- [tools/clang_indexer/index_project.py](/Volumes/external/sources/cppmega/tools/clang_indexer/index_project.py:1);
- [scripts/pr_ingest/pr_store.py](/Volumes/external/sources/cppmega/scripts/pr_ingest/pr_store.py:1);
- [scripts/crossrepo/build_global_symbol_index.py](/Volumes/external/sources/cppmega/scripts/crossrepo/build_global_symbol_index.py:1).

Что сознательно **не** перенесено как root dependency: MLX tensors, Metal kernels, `vbgui`, MLX checkpoint runtime и скрытые `../cppmega.mlx` defaults. Root FP8 и DSA используют собственные CUDA/TE/Triton seams и fail-loud, если backend недоступен.

### 3.3. `full_integration` -> канонические ветки

Канонические ветки содержат полные integration heads как предков, а не как отдельные незавершённые worktree patches:

| Target | Унаследованный integration head | GitHub | Что пришло этим слоем |
|---|---|---|---|
| `cppmega` | `0d3098c6bc3975db034bc7a425d5527ae04d399d` | [commit](https://github.com/DatasunriseOU/cppmega/commit/0d3098c6bc3975db034bc7a425d5527ae04d399d) · [branch](https://github.com/DatasunriseOU/cppmega/tree/codex/full-review-integration) | objective contracts, clang/uint64 identity, CASE5 sidecars, prompt graph generation, graph gradients, immutable bundles/generations, fail-closed H200 launchers, CI hardening, удаление legacy inference/generation fallbacks |
| `cppmega.mlx` | `70b1f5d5199135647994cac246326f5a8f2b2678` | [commit](https://github.com/DatasunriseOU/cppmega_mlx/commit/70b1f5d5199135647994cac246326f5a8f2b2678) · [branch](https://github.com/DatasunriseOU/cppmega_mlx/tree/codex/full-review-integration) | MLX graph/objective training, data/conveyor/indexer hardening, Path C planner/receipt, Mamba3 F2 semantics, row-carry preservation, RoPE parity, native/Metal fail-closed contracts, isolated CI/runtime environments |

После переноса integration worktrees не содержат уникальных tracked production-изменений. В root integration worktree остаются только generated `cppmega.egg-info/*`; MLX integration worktree clean.

<a id="contracts"></a>
## 4. KSH, Python и reserved IDs 245-248

Оба репозитория согласованы по одному и тому же `v1` контракту:

| Сущность | Значение |
|---|---:|
| `DomainKind.KSH` | `24` |
| `DomainKind.PYTHON` | `31` |
| `KSH_START` | `245` |
| `KSH_END` | `246` |
| `PYTHON_START` | `247` |
| `PYTHON_END` | `248` |

Tokenizer artifact по-прежнему буквально хранит:

```text
245  <RESERVED_245>
246  <RESERVED_246>
247  <RESERVED_247>
248  <RESERVED_248>
```

Это и есть in-place совместимость:

1. ID были заранее зарезервированы, поэтому существующие token IDs не сдвигаются.
2. Меняется роль reserved IDs в `tokenizer_contract_v1.json`, но не сам tokenizer vocabulary.
3. Schema остаётся `cppmega_domain_sidecars_v1`, отдельный `v2` не вводится.
4. Старые complete v1 hash triples принимаются атомарно. Смешанный набор old/new hashes отвергается, чтобы loader не принял частично обновлённый shard.

Текущие hashes одинаковы в обоих репозиториях:

| Контракт | SHA-256 |
|---|---|
| delimiter contract | `09fe81e915ee713004a1148abe54fbca2cf9ccfa9445901299a395d2b9fe253b` |
| domain schema | `522bf7d664bfc01da647a275baa022ed9f9894b3962bd160e096dbc6642a7a2a` |
| tokenizer contract | `77e7c934622cfea4e43999c400eb2e210def289a47cccfbb37c4a45222bc38b8` |
| tokenizer artifact | `d3c4711161a452ee36d64222b6977845ddd58b1e723a7de54158c64c50d2a888` |

Реализация и тесты:

- root KSH: [ksh.py](/Volumes/external/sources/cppmega/cppmega/data/shell_parsers/ksh.py:1);
- root Python: [stdlib.py](/Volumes/external/sources/cppmega/cppmega/data/python_parsers/stdlib.py:1);
- MLX KSH: [ksh.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/ksh.py:1);
- MLX Python: [stdlib.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/python_parsers/stdlib.py:1);
- cross-repo parser regressions: [root tests](/Volumes/external/sources/cppmega/tests/test_ksh_python_domain_parsers.py:1) · [MLX tests](/Volumes/external/sources/cppmega.mlx/tests/test_ksh_python_domain_parsers.py:1);
- legacy triple compatibility: [test_contract_v1_backward_compatibility.py](/Volumes/external/sources/cppmega/tests/test_contract_v1_backward_compatibility.py:1).

Python parser использует stdlib `ast` и `tokenize`, строит роли, scopes, AST/call/def-use edges и сохраняет частичный результат при синтаксически неполном Python. KSH использует общий shell contract с отдельным domain kind и delimiter pair.

<a id="pipeline"></a>
## 5. Независимый путь source -> parquet -> Megatron

Канонический root-путь теперь выглядит так:

```text
explicit source roots
  -> scripts/data/source_conveyor.py
  -> scripts/streaming_conveyor.py
  -> tools/clang_indexer/index_project.py
  -> enriched JSONL
  -> scripts/nanochat_data/clang_enriched_to_parquet.py
  -> tokenized enriched parquet
  -> route_by_fit + scripts/nanochat_data/pack_enriched_rows.py
  -> fixed-length parquet + domain/source/graph sidecars
  -> scripts/audit_sidecar_parquet.py
  -> scripts/data_prep_parquet_to_megatron.py
  -> immutable Megatron generation: .bin/.idx + token sidecars + graph CSR sidecars
```

Root-independence обеспечивается не только импортами:

- source roots, repo list, tokenizer и output roots передаются явно;
- root entrypoints не содержат `MEGACPP_NANOCHAT_ROOT` или `cppmega.mlx` defaults;
- code revision guard фиксирует exact git revision и relevant dirty scope;
- bundle paths root-local или обязательны как explicit arguments;
- FP8/DSA backend errors fail-loud вместо автоматической подмены MLX/TileLang реализацией;
- публикация Megatron output использует generation directory и atomic current pointer.

Код и проверки:

- [source_conveyor.py](/Volumes/external/sources/cppmega/scripts/data/source_conveyor.py:1);
- [prepare_data.sh](/Volumes/external/sources/cppmega/scripts/data/prepare_data.sh:1);
- [data_prep_parquet_to_megatron.py](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:1);
- [test_root_data_entrypoints.py](/Volumes/external/sources/cppmega/tests/test_root_data_entrypoints.py:1);
- [test_portable_source_pipeline_imports.py](/Volumes/external/sources/cppmega/tests/test_portable_source_pipeline_imports.py:1);
- [test_megatron_root_runtime_independence.py](/Volumes/external/sources/cppmega/tests/test_megatron_root_runtime_independence.py:1).

<a id="mlx-role"></a>
## 6. Роль MLX runtime и визуального инструментария

`cppmega.mlx` не дублирует назначение root. Его ценность в трёх областях:

1. **Локальный runtime на Apple Silicon.** MLX eager/compiled train smokes, checkpoints, бенчмарки и локальная проверка model contracts без H200.
2. **Metal/TileLang исследовательский контур.** Path C planner, generated kernels, fusion receipts, Mamba3/MLA/DSA experiments и честные fail-closed границы там, где Metal lowering ещё не готов.
3. **Визуальный конструктор и инспектор.** `vbgui` показывает model graph, tokenizer spans, parquet rows, domain/graph metadata и debugging state. Это удобный human-facing слой над теми же schema/hash contracts, которые root использует для training.

Локальные точки:

- [DataInspector.tsx](/Volumes/external/sources/cppmega.mlx/vbgui/src/components/DataInspector.tsx:1);
- [Tokenizer visualizer RPC](/Volumes/external/sources/cppmega.mlx/cppmega_v4/jsonrpc/tokenizer_methods.py:1);
- [Path C schedules](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/runtime/path_c_fusion_schedules.py:1);
- [MLX train-step harness](/Volumes/external/sources/cppmega.mlx/scripts/m04_train_step.py:1);
- [sidecar example renderer](/Volumes/external/sources/cppmega.mlx/scripts/render_sidecar_example.py:1).

Ограничение остаётся принципиальным: локальный MLX pass не доказывает CUDA/TE, distributed Megatron или H200 performance. И наоборот, H200 component smoke не заменяет локальные macOS generation/eval/compile gates.

<a id="commits"></a>
## 7. Точные коммиты на снимке

### 7.1. `cppmega`

Branch: `codex/canonical-complete-20260715`
HEAD: `2b5dd3df263883a1c88a5051e5edabafa54845ed`

| Commit | Время CEST | Назначение |
|---|---|---|
| `2b5dd3df263883a1c88a5051e5edabafa54845ed` | 14:36 | `test(dsa): generate causal topk fixtures` |
| `dd82a55aa023453e37c642605b6d2f6ad7f5373e` | 14:16 | `fix(data): align parser spans with token materialization` |
| `7d3a590871ec0d15fd88be4908c6591b3a61df81` | 13:51 | `fix(megatron): expose case5 contract hashes to loader` |
| `e925acaf9d76d159ce0e7987d470be266e5ad7f6` | 13:29 | `feat(data): integrate independent corpus and megatron pipeline` |
| `0a6061ea267e32de9658d30e0e7318ced6d81098` | 12:42 | `feat(data): add canonical source processing core` |
| `166f6cb85fe0ac19e0c495628b852e94df177d38` | 12:03 | `fix(data): validate objective contracts before parquet import` |
| `0d3098c6bc3975db034bc7a425d5527ae04d399d` | 03:07 | published `full-review-integration` head |

Локальный commit object проверен через `git -C /Volumes/external/sources/cppmega show --no-patch 2b5dd3df263883a1c88a5051e5edabafa54845ed`. В `origin` ветка `codex/canonical-complete-20260715` отсутствует: `git ls-remote --heads origin codex/canonical-complete-20260715` вернул пустой результат.

Worktree caveat на финальной проверке: root status также содержит чужие dirty changes `M .gitignore`, `M scripts/data/build_dataset_manifest.py`, `M tools/clang_indexer/index_project.py`, незатрекованный `tests/test_clang_exception_specification.py` и plan-файл. Эти изменения находятся вне ownership отчёта и не откатывались. Два report-файла являются единственными файлами, которые редактирует этот pass.

### 7.2. `cppmega.mlx`

Branch: `codex/canonical-complete-20260715`
HEAD: `d5fdec0eded97a2dd0c17828d75d979d4029c8f9`

| Commit | Время CEST | Назначение |
|---|---|---|
| `d5fdec0eded97a2dd0c17828d75d979d4029c8f9` | 20:13 | `fix(training): preserve packed document sidecars` |
| `050be1155ac5bb9f7e2332acdd161e82a7e64a01` | 20:04 | `fix(data): keep golden pipeline repository local` |
| `189b38bd030b00cadf46b7d980eed7d1a4aaa49d` | 19:54 | `fix(eval): make mlx compile gate repository local` |
| `e25622982d4044a2ccaac3bc0539571b411b51b9` | 14:06 | `fix(data): align parser spans with token materialization` |
| `1ba36f7ddecdab91f3acc8ced5698c9ffadbe52b` | 13:06 | `fix(data): audit invalid domain files in corpus mode` |
| `8ea8f1ece4b3b81e590fc86e974aeed64726d787` | 12:52 | `fix(data): skip malformed generic text candidates` |
| `43804fafe22561f9d2c10e6f0718729c3a2bb58e` | 12:35 | `feat(data): add ksh and python domain parsers` |
| `9ff05c5f0acc40e4cbe4f661beb16618fc41164a` | 12:01 | `fix(data): recover stored non-github project identities` |
| `70b1f5d5199135647994cac246326f5a8f2b2678` | 04:50 | published `full-review-integration` head |

Локальный commit object проверен через `git -C /Volumes/external/sources/cppmega.mlx show --no-patch d5fdec0eded97a2dd0c17828d75d979d4029c8f9`. Во время этого documentation pass другие исполнители продолжили менять MLX worktree: status содержит dirty FIM/objective files (`cppmega_mlx/data/ast_fim.py`, `cppmega_mlx/training/{objective_data,objective_mixer,objectives,megatron_objectives}.py`), связанные tests и clang identity files. Они не входят в `d5fdec0`, не редактировались этим pass и не считаются принятым fix без отдельного commit/receipt. Ветка `codex/canonical-complete-20260715` отсутствует и в MLX remote, поэтому новые canonical hashes выше являются точными локальными commits, но не снабжаются вводящими в заблуждение ссылками на несуществующую remote branch.

<a id="tests"></a>
## 8. Тесты и статические проверки

Числа ниже являются отдельными receipts и частично перекрываются. Их нельзя суммировать в один искусственный total.

| Слой | Receipt | Статус/смысл |
|---|---:|---|
| Published `full_integration` baseline | `1659 passed, 6 skipped` | Непересекающиеся suites старшего integration pass: MLX Path C/native/data и root Megatron/data/CI |
| Root canonical focused integration | `166 passed, 2 skipped` | source core, contracts, converter, portable pipeline, PR/crossrepo/audit/runtime independence |
| Root KSH/Python + entrypoints | `22 passed` | parser discovery/materialization и root-local CLI wiring |
| Root graph/loader после `7d3a590` | `27 passed` | CASE5 receipt/hash exports и graph-route loader bridge |
| Root parser TDD на `dd82a55` | RED `5 failed, 1 passed`; GREEN `6 passed` | Исправление доказано red/green на f-string anchors, embedded spans, SPTAG false SQL и fail-loud empty span |
| Root expanded после parser fix | `49 passed, 1 skipped` | Расширенный parser/materializer/contract regression set |
| MLX focused rerun на `e256229` | `112 passed` | Receipt относится к `e256229`; текущий MLX HEAD `d5fdec0` содержит ещё три более поздних commit и этим числом автоматически не покрывается |
| MLX invalid-input corpus policy | `83 passed` | generic malformed text skip, explicit typed fail-loud API, audited conveyor mode |
| Nebius H200 initial gate на `b4e5261` | `11 passed, 27 warnings` | Первый hardware receipt до исправления larger-seq fixture |
| H200 full DSA split-K file на `2b5dd3d` | `7 passed, 35 warnings` | Полный [tests/test_dsa_splitk_indexer_loss.py](/Volumes/external/sources/cppmega/tests/test_dsa_splitk_indexer_loss.py:1) на exact root HEAD |
| H200 final required gate на exact `2b5dd3d` | `11 passed, 27 warnings in 8.00s` | Финальный CUDA/TE/Megatron component gate; writer/reader: `46` docs, `45 257` valid, `45 063` trained, `20` aligned sidecars, `11` graph CSR |
| Root/MLX Ruff | `PASS` | перенесённый Python слой чист |
| Root `compileall`, shell syntax, `git diff --check` | `PASS` | syntax/static integrity committed delta |

На локальном Mac real Megatron import/runtime по-прежнему недоступен из-за отсутствующего `megatron-core`; это не локальный pass. Hardware-specific path доказан отдельно на H200 и завершён для точного набора обязательных component tests.

### 8.1. H200 exact receipt и граница доказательства

| Поле | Live receipt |
|---|---|
| VM | `computeinstance-e00dezxd2mdw060cem` / `cppmega-h200-canonical-20260715-140105` |
| Container image | pinned digest prefix `sha256:08c5...` |
| Первый GHCR pull | `403`; token не имел `read:packages` |
| Исправление auth | Повторный login package-capable account; secret/token в отчёт не переносится |
| Preliminary source | `b4e526160fc5eeada5ad041a5f7109d558899c19` - только ранний gate до causal-fixture repair |
| Exact final root source | `2b5dd3df263883a1c88a5051e5edabafa54845ed` |
| Full DSA suite | `7 passed, 35 warnings` |
| Final required gate | `11 passed, 27 warnings in 8.00s` |
| Writer/reader receipt | `46` documents; `45 257` valid tokens; `45 063` trained tokens; `20` token-aligned sidecars; `11` graph CSR/ragged sidecars |
| Cleanup | VM удалена; direct `get` -> `NotFound`, project instance list -> empty |

Full split-K сначала обнаружил проблему не в production ordering, а в larger-seq test fixture: случайные `topk_indices` и немаскированные `index_scores` расходились с causal production path. Commit `2b5dd3d` causal-masks `index_scores` до top-k и передаёт в loss те же masked scores, то есть генерирует fixture в production-порядке. После этого full DSA file и финальный required gate прошли на exact canonical HEAD. Локальные source anchors: [causal DSA fixture](/Volumes/external/sources/cppmega/tests/test_dsa_splitk_indexer_loss.py:1), [H200 sweep tests](/Volumes/external/sources/cppmega/tests/test_nebius_h200_megatron_cpp_world_sweep.py:1), [H200 sweep launcher](/Volumes/external/sources/cppmega/scripts/nebius_h200_megatron_cpp_world_sweep.py:1). Это component acceptance, а не заявление о завершённом длительном distributed training run.

<a id="detours"></a>
## 9. Detours: локальный end-to-end receipt

Artifact root: `/tmp/cppmega-root-smoke.1CGnuc`

### 9.1. Conveyor

Запуск был привязан к чистому root commit `e925acaf9d76d159ce0e7987d470be266e5ad7f6` и exact relevant-scope hash. Основной entrypoint находился в `cppmega`, source path также был root-owned.

| Метрика | Значение |
|---|---:|
| Repo | `microsoft/Detours` |
| Bucket | `1024` |
| Rows / sequences | `46` |
| Capacity tokens | `47 104` |
| Valid tokens | `45 257` |
| Trained tokens | `45 063` |
| Padding | `1 847` (`3.921111%`) |
| Total conveyor time | `84.734 s` |
| Failed units | `0` |

Stage timings:

| Stage | Seconds |
|---|---:|
| clang index | `9.167591` |
| materialize | `58.590907` |
| route by fit | `11.273985` |
| pack | `5.160196` |
| recompress | `0.115098` |

Receipts:

- [progress.jsonl](/tmp/cppmega-root-smoke.1CGnuc/packed/.conveyor/progress.jsonl:1);
- [_done.json](/tmp/cppmega-root-smoke.1CGnuc/packed/.conveyor/_done.json:1);
- [launch/revision receipt](/tmp/cppmega-root-smoke.1CGnuc/packed/.conveyor/code_revision_guard/e925acaf9d76d159ce0e7987d470be266e5ad7f6-be678ff3aeec94fd/launch_receipt.json:1);
- [packed Detours parquet](/tmp/cppmega-root-smoke.1CGnuc/packed/1024/Detours.parquet).

### 9.2. Fail-closed audit

Audit обработал `1` parquet file и завершился `passed`:

- `bad files = 0`;
- `bad rows = 0`;
- все обязательные lengths/value contracts прошли;
- CASE5 success receipt записан.

Ссылки: [audit Markdown](/tmp/cppmega-root-smoke.1CGnuc/audit/sidecar_parquet_audit.md:1) · [audit JSON](/tmp/cppmega-root-smoke.1CGnuc/audit/sidecar_parquet_audit.json:1).

### 9.3. Megatron converter

Converter прочитал один shard и выпустил:

- `detours.bin` + `detours.idx` в MMIDIDX формате;
- `20` token-aligned sidecars;
- `11` graph CSR/ragged sidecars;
- SQLite source identity registry;
- immutable generation `generation-1784115486756288000-6g2sjm59`;
- atomic pointer `.detours.current` на эту generation.

Graph receipt содержит:

| Канал | Items |
|---|---:|
| call edges | `1` |
| type edges | `1` |
| domain edges | `542` |
| build edges | `154` |
| shell / diagnostic / cross-domain edges | `0 / 0 / 0` |
| chunk starts / ends / kinds / dep levels | `207` в каждом канале |

Source identity registry: `13` identities, `84` sequence references, `46` sequences, `uint64_be` IDs и SHA-256 canonical digest.

Главный локальный receipt: [detours.json](/tmp/cppmega-root-smoke.1CGnuc/megatron/detours.json:1) · [converter source](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:1). В самом JSON `writer_backend` обозначает MMIDIDX file format; H200 writer/reader acceptance приведён отдельно ниже.

### 9.4. Real Megatron writer+reader

H200 receipt не ограничился проверкой файлов на диске: он зафиксировал настоящий Megatron writer и последующее чтение созданного dataset через `IndexedDataset`.

| Проверка | Receipt |
|---|---:|
| Documents | `46` |
| Valid tokens | `45 257` |
| Trained tokens | `45 063` |
| Первый прочитанный document | length `1014`, dtype `int32` |
| Token-aligned sidecars | `20` |
| Graph CSR/ragged sidecars | `11` |

Это закрывает writer+reader interoperability для Detours fixture в exact H200 component scope. Параллельный локальный artifact сохраняет те же counts и sidecar inventory, но его manifest явно говорит `writer_backend=mmididx`; эти два уровня receipts не следует смешивать. H200 receipt не заменяет full distributed training/performance result и не превращает локальный внешний Python environment в standalone root packaging.

### 9.5. Environment caveat

В `/Volumes/external/sources/cppmega` нет `.venv`, а [pyproject.toml](/Volumes/external/sources/cppmega/pyproject.toml:1) содержит `dependencies = []`. Поэтому smoke был запущен через:

```text
/Volumes/external/sources/cppmega.mlx/.venv/bin/python
```

Этот executable фактически разрешается в `/Volumes/external/sources/nanochat/.venv`, где установлены PyArrow, NumPy, libclang/tokenizer dependencies. Это не нарушает root-independent **code path**, но показывает незавершённость root environment packaging: для воспроизводимого standalone запуска root нужен собственный declared environment.

<a id="case5-v4"></a>
## 10. CASE5 v4: стабильный inventory токенов и полных шагов

Источник: существующие parquet roots

```text
/Volumes/external/sources/cppmega.mlx/outputs/reindexed_case5_v4_20260714_093120_code
/Volumes/external/sources/cppmega.mlx/outputs/reindexed_case5_v4_20260714_093120_commits
```

Runtime revision: `63b0f587ce9504672bd58c964c6c4176fc10956d`.
Расчёт использует **trained tokens**, а не capacity/valid tokens.

Источник расчёта и проверяемый код: [report_training_steps.py](/Volumes/external/sources/cppmega.mlx/scripts/report_training_steps.py:1), [code root](/Volumes/external/sources/cppmega.mlx/outputs/reindexed_case5_v4_20260714_093120_code), [commit root](/Volumes/external/sources/cppmega.mlx/outputs/reindexed_case5_v4_20260714_093120_commits). Повторный запуск `report_training_steps.py` на этих двух roots дал те же `1 411 702 081` trained tokens и `7 178` schedule-valid steps; skipped files `0`.

| Bucket | Batch | Tokens/step | Code trained | Commit+PR trained | Combined trained | Полных шагов |
|---:|---:|---:|---:|---:|---:|---:|
| `1024` | `192` | `196 608` | `716 866 504` | `36 658 247` | `753 524 751` | `3 832` |
| `2048` | `96` | `196 608` | `134 702 573` | `43 259 845` | `177 962 418` | `905` |
| `4096` | `48` | `196 608` | `126 565 916` | `51 451 443` | `178 017 359` | `905` |
| `8192` | `24` | `196 608` | `109 386 520` | `56 598 061` | `165 984 581` | `844` |
| `16384` | `12` | `196 608` | `87 504 305` | `48 708 667` | `136 212 972` | `692` |
| **Итого** | - | - | **`1 175 025 818`** | **`236 676 263`** | **`1 411 702 081`** | **`7 178`** |

Почему итог `7 178`, а не `7 180`:

```text
per-bucket schedule: 3832 + 905 + 905 + 844 + 692 = 7178
fungible aggregate: floor(1 411 702 081 / 196 608) = 7180
```

Training schedule выбирает batch size отдельно для каждого sequence-length bucket. Остаток 1024-бакета нельзя перенести в 16384-бакет, поэтому корректна сумма полных шагов по бакетам: **7 178**.

Этот стабильный inventory описывает уже произведённые v4 shards, но не утверждает, что весь потенциальный corpus обработан без gaps: v4 commit lane ранее останавливалась на provenance mapping для `blender`, а набор источников не является доказательством полного corpus coverage.

<a id="case5-v7"></a>
## 11. CASE5 v7: живой статус

Run ID: `case5_v7_20260715_130725`
Runtime: `/Volumes/external/sources/cppmega_mlx_case5_v7_runtime`
Pinned revision: `1ba36f7ddecdab91f3acc8ced5698c9ffadbe52b`

Снимок запуска регистрировал tmux sessions:

```text
cppmega_case5_v7_code
cppmega_case5_v7_commits
```

Launchers: [code](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/launch_code.sh:1) · [commits](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/launch_commits.sh:1).

### 11.1. Trained-token snapshot на 14:40 CEST

Во всех бакетах tokens/step равен `196 608`; schedule-valid steps считаются отдельно по каждому бакету.

| Bucket | Combined trained tokens | Полных шагов |
|---:|---:|---:|
| `1024` | `98 863 686` | `502` |
| `2048` | `15 449 130` | `78` |
| `4096` | `15 938 281` | `81` |
| `8192` | `13 519 250` | `68` |
| `16384` | `9 320 448` | `47` |
| **Итого** | **`153 090 795`** | **`776`** |

Это partial trained-token snapshot, не final corpus. Его нельзя прибавлять к CASE5 v4: v7 повторно обрабатывает пересекающиеся источники и является replacement candidate, а не независимым additive corpus.

### 11.2. Текущая производительность

| Lane | Valid tokens | Elapsed | Throughput |
|---|---:|---:|---:|
| Code | `140 957 215` | `4 324.754 s` | `~117.3M tok/h` |
| Commits | `12 714 472` | `4 236.171 s` | `~10.8M tok/h` |
| Одновременная сумма lane rates | - | - | **`~128.1M tok/h`** |

Combined throughput здесь является суммой одновременно работающих lane rates, а не делением объединённых токенов на один общий wall-clock denominator.

### 11.3. Snapshot manifest: движущийся, не финальный

Требуемая receipt-точка v7 зафиксирована как `77 done / 11 failed`. Это именно snapshot, а не обещание, что run завершён:

| Поле | Зафиксированный snapshot |
|---|---:|
| Done units | `77` |
| Failed units | `11` |
| Runtime revision | `1ba36f7ddecdab91f3acc8ced5698c9ffadbe52b` |
| Snapshot window | `14:40-14:54 CEST` |
| Статус | live / partial / не final |

После этой точки `_done.json` продолжал изменяться. Поэтому файл по ссылке ниже используется как live artifact и anchor для provenance, но не как immutable источник числа `77/11`; позднее состояние намеренно не подставляется в этот исторический snapshot. Это также означает, что `153 090 795` tokens и `776` steps нельзя объявлять окончательным replacement corpus.

### 11.4. Rerun и provenance blockers

| Unit | Зафиксированный failure | Состояние |
|---|---|---|
| `LightGBM::repo` | domain-edge endpoint не попал в nonempty token span | Targeted rerun на `e256229` обязателен |
| `SDL::repo` | тот же parser/materializer span class | Targeted rerun на `e256229` обязателен |
| `SPTAG::repo` | embedded domain span `1966:2315` не сопоставился token spans | Targeted rerun на `e256229` обязателен |
| Другие parser/materialize units | `ITK`, `VTK`, `apache-tvm`, `apple-mlx`, `ardupilot`, `arm-compute-library`, `arm-trusted-firmware`, `arrow` | Два repair agents работают; нужен новый pinned rerun и manifest audit |
| `apple-libdispatch::repo` | clang binding: `Unknown template argument kind 9` | Отдельный clang enum compatibility repair и rerun |
| `php-src::commits` | Отдельный live failed commit-lane unit | Разобрать receipt и rerun; не смешивать со span class без доказательства |
| `Xbox Live Source::code` | отсутствует canonical project identity | Provenance не подтверждён; identity нельзя выдумывать |

Canonical span fix уже присутствует симметрично в `e256229` и `dd82a55`; MLX focused rerun даёт `112 passed`, root TDD даёт RED `5/1` -> GREEN `6 passed`, expanded root suite - `49 passed, 1 skipped`. Но v7 snapshot запущен с revision `1ba36f7`, а текущий MLX checkout уже на `d5fdec0`. [Launch receipt](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/code_revision_guard/1ba36f7ddecdab91f3acc8ced5698c9ffadbe52b-be678ff3aeec94fd/launch_receipt.json:1) также содержит marker `CPPMEGA_CODE_REVISION_DRIFT`. Только новый pinned runtime, targeted rerun LightGBM/SDL/SPTAG и обновлённый fail-closed manifest могут подтвердить fix; до этого CASE5 v7 нельзя объявлять clean. Финальный audit также должен сверить все другие failed units старого pre-fix manifest, а не молча считать их закрытыми.

Xbox receipt показывает источник `data-cpp_all.tar.zst`, а не нормальный forge remote: [source-cache receipt](</Volumes/external/sources/cppmega.mlx/outputs/source_cache/code/Xbox Live Source/.cppmega_source_cache_complete.json:1>). Корректные варианты только два: назначить явную archive provenance identity после проверки происхождения/лицензии или документированно исключить источник. GitHub identity фабриковать нельзя.

Live artifacts: [manifest](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/_done.json:1) · [code launcher](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/launch_code.sh:1) · [commit launcher](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/launch_commits.sh:1) · [code progress](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/progress_code.jsonl:1) · [commit progress](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/progress_commits.jsonl:1).

<a id="a2"></a>
## 12. A2 std DB и идентичность символов

DB: [global_symbols_case5_v3_20260714_063541.sqlite](/Volumes/external/sources/cppmega.mlx/outputs/crossrepo/global_symbols_case5_v3_20260714_063541.sqlite)

Прямая SQL-проверка:

| Метрика | Значение |
|---|---:|
| `base_lib='std'` rows | `15 485` |
| Functions | `10 439` |
| Types | `5 046` |
| Distinct std qnames | `5 113` |
| `std::*` prefixed rows | `15 485` |
| libiberty rows | `0` |

Старое загрязнение A2, где std-capacity уходила в GCC/libiberty internals, устранено. Но отношение `15 485 rows / 5 113 qnames` одновременно доказывает, что `qname` нельзя использовать как primary identity.

Примеры из текущей DB:

| qname | Candidates | Distinct signatures | Files |
|---|---:|---:|---:|
| `std::operator==` | `122` | `75` | `75` |
| `std::swap` | `112` | `92` | `62` |
| `std::operator<<` | `103` | `53` | `65` |
| `std::formatter` | `86` | `64` | `14` |

Canonical identity работает по приоритету:

1. Если clang дал стабильный USR, ключ строится как `USR + owning project`.
2. Если USR нет, используется canonical signature, symbol kind, normalized qname и scope. Для file-scoped symbols scope включает project/file/line.
3. Если нет и пригодной signature, разрешён только typed `repo_file_location` fallback: canonical project + repository-relative file + line/column + kind + qname.
4. Canonical key хешируется SHA-256 domain-separated схемой в unsigned 64-bit ID; collision registry fail-loud.

В SQLite primary key - `symbol_uid`, а `qname` имеет обычный lookup index. Reader сначала получает `qname -> candidates`, затем фильтрует по USR/signature/project/file/provider. Если после фильтра остаётся больше одного кандидата, lookup падает с `ambiguous global symbol lookup`, а не выбирает произвольную большую функцию.

Реализация: [canonical_symbol_identity](/Volumes/external/sources/cppmega/tools/clang_indexer/index_project.py:971) · [global candidate lookup](/Volumes/external/sources/cppmega/scripts/crossrepo/build_global_symbol_index.py:1088) · [A2 regressions](/Volumes/external/sources/cppmega/tests/test_crossrepo_std_namespace_index.py:1).

<a id="fim-ast-fim"></a>
## 13. FIM/AST-FIM: eligibility и physical identities

Это отдельный unresolved-блок, не закрытый общими parser receipts или H200 component gate.

На момент финальной проверки MLX worktree содержит параллельную незакоммиченную разработку physical-source-safe FIM middle. Она может изменить детали реализации, но пока не входит в HEAD `d5fdec0` и не имеет отдельного accepted receipt; поэтому отчёт не объявляет gap исправленным.

### Что уже есть в коде

- `EligibilityAwareTaskMixer` различает `fim`, `ast_fim` и `ifim`, требует поддерживаемый domain interior, минимальную длину, а для AST-FIM - непустые clang chunk metadata: [objective_mixer.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/objective_mixer.py:329).
- Для transformed document предусмотрена проверка token-aligned `source_identity_ids`: значения должны быть положительными `uint64`, а выбранный logical document должен ссылаться ровно на одну physical source identity: [physical identity guard](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/objective_mixer.py:269).
- Materializer умеет сохранить/remap provenance и в production-режиме требует identity sidecar: [Megatron objective materialization](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/megatron_objectives.py:891) и [transformed identity validation](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/megatron_objectives.py:1058). Положительные тесты для одно-physical и multi-document выбора существуют: [objective mixer tests](/Volumes/external/sources/cppmega.mlx/tests/test_production_objective_mixer.py:1078).

### Почему это ещё не acceptance

В `_transformed_document_physical_identity_reason()` и `_transformed_packet_physical_identity_reason()` отсутствующий `packet.source_identity_ids` сейчас возвращает `None`, то есть не делает FIM/AST-FIM candidate неeligible на стадии квоты. Позднее production materialization может fail-close из-за обязательного sidecar, но это уже после выбора objective и не является доказательством корректного eligibility accounting. Риск особенно важен для AST-FIM, где преобразование меняет порядок/границы токенов и physical provenance нельзя восстанавливать из одного `qname` или текста.

| Требование для закрытия | Нужное доказательство |
|---|---|
| FIM/AST-FIM candidate без token-aligned physical identity не попадает в quota | focused negative receipt на `None`, нулевые, malformed и mixed identity arrays |
| выбранный logical document имеет ровно одну physical identity | production parquet/registry FK audit на transformed rows |
| realized objective accounting не считает rejected candidate как FIM/AST-FIM | pinned v7 rerun с per-objective realized-token receipt |

До такого pinned rerun и receipt **FIM/AST-FIM physical-identity eligibility не считается исправленной**. Старый review фиксирует тот же contract boundary: [data/parquet/Megatron review](/Volumes/external/sources/cppmega.mlx/docs/reviews/2026-07-14-data-parquet-megatron-review.md:213) · [objective mixture contract](/Volumes/external/sources/cppmega/docs/objective_mixture_contract.md:32).

<a id="risks"></a>
## 14. Незавершённые проблемы и риски

| Риск | Что уже доказано | Что ещё требуется |
|---|---|---|
| CASE5 span contracts | `e256229`/`dd82a55` landed; MLX `112 passed`; root RED `5 failed, 1 passed` -> GREEN `6 passed`; expanded root `49 passed, 1 skipped` | Новый pinned runtime и targeted rerun LightGBM/SDL/SPTAG; затем полный audit старого failed manifest |
| FIM/AST-FIM physical identities | Guard и unit tests существуют, но missing identity может пройти mixer eligibility до production materialization | Fail-closed eligibility, registry/FK audit и realized-token receipt на pinned rerun |
| Xbox Live Source provenance | Известен archive source path; fake GitHub mapping отвергнут | Подтвердить происхождение и лицензию, затем explicit archive identity или exclusion |
| H200 exact acceptance | `2b5dd3d`: full DSA file `7 passed, 35 warnings`; final required gate `11 passed, 27 warnings in 8.00s`; Detours writer+reader real | Не экстраполировать component gate на длительный distributed training/performance result |
| Nebius cleanup | Image pin `sha256:08c5...`; GHCR 403 диагностирован и auth исправлен; VM удалена и API подтверждает отсутствие | Ничего для этого gate; новый VM нужен только для следующего отдельного GPU workflow |
| Root Python environment | Root code path независим; реальный Detours E2E прошёл | Собственный `.venv`/lock/declared runtime dependencies вместо внешнего nanochat-linked environment |
| Canonical publication | Exact local commits и ancestry проверены | Ветки `codex/canonical-complete-20260715` ещё не pushed на снимке |
| CASE5 v7 completeness | Зафиксированный snapshot: `77/11`; run pinned на старом `1ba36f7`, не на текущем `d5fdec0` | Завершить run/repairs/reruns, провести audit и только затем считать replacement corpus готовым |

Требуемый H200 CUDA/TE/Megatron component gate получен и VM удалена. Локальные model generation/eval/compile gates по-прежнему должны выполняться на macOS; этот GPU receipt не подменяет локальную оценку модели и не доказывает длительный training run.

<a id="evidence"></a>
## 15. Индекс доказательств

### GitHub

- [`DatasunriseOU/cppmega`](https://github.com/DatasunriseOU/cppmega)
- [`DatasunriseOU/cppmega_mlx`](https://github.com/DatasunriseOU/cppmega_mlx)
- [root full integration commit `0d3098c`](https://github.com/DatasunriseOU/cppmega/commit/0d3098c6bc3975db034bc7a425d5527ae04d399d)
- [MLX full integration commit `70b1f5d`](https://github.com/DatasunriseOU/cppmega_mlx/commit/70b1f5d5199135647994cac246326f5a8f2b2678)

### Root source и tests

- [root data package](/Volumes/external/sources/cppmega/cppmega/data/__init__.py:1)
- [root tokenizer contract](/Volumes/external/sources/cppmega/cppmega/data/tokenizer_contract.py:1)
- [root clang indexer](/Volumes/external/sources/cppmega/tools/clang_indexer/index_project.py:1)
- [root source conveyor](/Volumes/external/sources/cppmega/scripts/data/source_conveyor.py:1)
- [root parquet audit](/Volumes/external/sources/cppmega/scripts/audit_sidecar_parquet.py:1)
- [root Megatron converter](/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py:1)
- [root runtime independence tests](/Volumes/external/sources/cppmega/tests/test_megatron_root_runtime_independence.py:1)
- [root CASE5 span regressions](/Volumes/external/sources/cppmega/tests/test_case5_v7_parser_token_span_regressions.py:1)
- [root DSA causal fixture](/Volumes/external/sources/cppmega/tests/test_dsa_splitk_indexer_loss.py:1)
- [root H200 sweep tests](/Volumes/external/sources/cppmega/tests/test_nebius_h200_megatron_cpp_world_sweep.py:1)
- Local exact root commit: `2b5dd3df263883a1c88a5051e5edabafa54845ed` (`git -C /Volumes/external/sources/cppmega show --no-patch ...`)

### MLX source и visual layer

- [MLX domain ingestion](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/domain_ingestion.py:1)
- [MLX tokenizer contract](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/tokenizer_contract.py:1)
- [MLX Path C runtime](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/runtime/path_c_fusion_launcher.py:1)
- [MLX CASE5 span regressions](/Volumes/external/sources/cppmega.mlx/tests/test_case5_v7_parser_token_span_regressions.py:1)
- [MLX objective eligibility](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/objective_mixer.py:269)
- [MLX Megatron provenance materializer](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/megatron_objectives.py:891)
- [MLX physical-identity tests](/Volumes/external/sources/cppmega.mlx/tests/test_production_objective_mixer.py:1078)
- [Visual Builder DataInspector](/Volumes/external/sources/cppmega.mlx/vbgui/src/components/DataInspector.tsx:1)

### Runtime receipts

- [Detours conveyor progress](/tmp/cppmega-root-smoke.1CGnuc/packed/.conveyor/progress.jsonl:1)
- [Detours audit](/tmp/cppmega-root-smoke.1CGnuc/audit/sidecar_parquet_audit.md:1)
- [Detours audit JSON](/tmp/cppmega-root-smoke.1CGnuc/audit/sidecar_parquet_audit.json:1)
- [Detours Megatron receipt](/tmp/cppmega-root-smoke.1CGnuc/megatron/detours.json:1)
- [Detours immutable generation](/tmp/cppmega-root-smoke.1CGnuc/megatron/snapshot/megatron_generations/detours/generation-1784115486756288000-6g2sjm59/detours.json:1)
- [CASE5 v7 manifest](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/_done.json:1)
- [CASE5 v7 launch revision receipt](/Volumes/external/sources/cppmega.mlx/outputs/conveyor_case5_v7_20260715_130725/code_revision_guard/1ba36f7ddecdab91f3acc8ced5698c9ffadbe52b-be678ff3aeec94fd/launch_receipt.json:1)
- [Xbox source-cache receipt](/Volumes/external/sources/cppmega.mlx/outputs/source_cache/code/Xbox%20Live%20Source/.cppmega_source_cache_complete.json:1)
- [A2 global symbol DB](/Volumes/external/sources/cppmega.mlx/outputs/crossrepo/global_symbols_case5_v3_20260714_063541.sqlite)

<a id="changed-files"></a>
## 16. Изменённые файлы

Этот documentation pass создаёт только:

- `/Volumes/external/sources/cppmega/reports/canonical_integration_20260715.md`
- `/Volumes/external/sources/cppmega/reports/canonical_integration_20260715.html`

Код, tests, configs, manifests и commits этим pass не изменяются.
