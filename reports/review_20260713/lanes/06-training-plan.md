**Findings**

Critical: none confirmed.

High - [Confirmed bug] [train_stage1.py](/Volumes/external/sources/cppmega.mlx/scripts/train_stage1.py:270) trains `commit_diff` with `diff_token_ids=post`, and [line 261](/Volumes/external/sources/cppmega.mlx/scripts/train_stage1.py:261) uses a synthetic `[BOS, CODE_START]` commit message. Failure mode: reports can say commit-diff is exercised, but the smoke objective predicts edited post content, not a unified diff, and it is not PR-conditioned. Evidence: the real objective expects “Predict the unified diff from the commit message” at [objectives.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/objectives.py:257). Focused fix/test: make the fixture provide real unified-diff tokens and commit/PR text, then assert `commit_diff` examples include diff delimiters and non-synthetic commit text; otherwise rename this smoke path so it is not counted as commit-diff training.

High - [Design gap] [train_eval_stage1.py](/Volumes/external/sources/cppmega.mlx/scripts/train_eval_stage1.py:50) is the real local streaming runner, but it reads only `clang_semantic_4k_v10` token/side-channel columns and [line 402](/Volumes/external/sources/cppmega.mlx/scripts/train_eval_stage1.py:402) calls plain model CE. Failure mode: FIM/IFIM/commit/recovery objectives exist, but the longer runner does not train them. Evidence: `TaskMixer` stage-1 rates exist at [task_mixer.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/task_mixer.py:58), but `TaskMixer` usage is only smoke/tests. Focused fix/test: add a real objective-mixed runner over production shards and fail the run if realized per-objective token counts deviate from configured ratios.

High - [Design gap / contradiction] [stage_domain_routed_foundation.yaml](/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml:26) says token-only baseline is disallowed and graph sidecars are required, but [train_eval_stage1.py](/Volumes/external/sources/cppmega.mlx/scripts/train_eval_stage1.py:67) only reads token plus structural side-channel columns. Failure mode: graph-supervised routing/build diagnostics are plan/config enforced on paper but bypassed by current real local train. Focused fix/test: runner loads the YAML and fails closed when graph sidecars are absent unless an explicit `allow_token_only_baseline` override is set.

High - [Confirmed bug] [domain_graph_routes.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/domain_graph_routes.py:19) defaults `edge_weights` to `{}`, while [line 82](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/domain_graph_routes.py:82) defines diagnostic/linker weights and [line 39](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/domain_graph_routes.py:39) uses an empty default config. Failure mode: diagnostic edges silently get weight `1.0`, not intended `2.0`, unless every caller passes config manually. Focused fix/test: make default config use `DEFAULT_EDGE_WEIGHTS`, or wire YAML weights into callers; add a default-config test for `DIAG_PRIMARY_LOCATION == 2.0`.

Medium - [Confirmed bug] [ast_fim.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/ast_fim.py:309) labels IFIM AST fallback as `char_fim`, and [line 316](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/ast_fim.py:316) still applies `apply_ifim_permutation`. Failure mode: IFIM examples are mis-accounted as non-instruction FIM in metrics/curriculum accounting. Focused fix/test: force char fallback and assert metadata is `char_ifim` or carries `instruction_aware=True`.

Medium - [Design gap] PR enrichment is implemented in data plumbing, but not in the commit objective schema. Evidence: [process_commits.py](/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/process_commits.py:161) attaches `pr_discussion`, [export_pr_parquet.py](/Volumes/external/sources/cppmega.mlx/scripts/pr_ingest/export_pr_parquet.py:121) exports PR discussion text, but [commit_packet.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/commit_packet.py:72) has only pre/post/diff/msg fields and [code_packet_builder.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/code_packet_builder.py:468) returns a `CommitPacket` without PR fields. Failure mode: “commit pre/post/diff with PR enrichment” is present as corpus text, not as a supervised commit transform objective. Focused fix/test: extend `CommitPacket` or add `CommitPRPacket`; test merge-commit SHA lookup fills PR title/body/comments into the training prompt.

Medium - [Design gap] [task_mixer.py](/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/task_mixer.py:70) only defines `stage1`, while the domain config defines stage2 repair at [stage_domain_routed_foundation.yaml](/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml:60) and stage3 world trajectory at [line 72](/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml:72). Failure mode: repair/world-model curricula remain declarative, not runnable by the current mixer. Focused fix/test: add stage2/stage3 mixer presets and a dry-run report that prints realized token/objective ratios.

Low - [Stale doc] [2026-07-02-domain-routed-build-shell-diagnostics.md](/Volumes/external/sources/cppmega/docs/superpowers/plans/2026-07-02-domain-routed-build-shell-diagnostics.md:268) still marks graph-route runtime unchecked, [line 283](/Volumes/external/sources/cppmega/docs/superpowers/plans/2026-07-02-domain-routed-build-shell-diagnostics.md:283) marks domain objectives unchecked, and [line 297](/Volumes/external/sources/cppmega/docs/superpowers/plans/2026-07-02-domain-routed-build-shell-diagnostics.md:297) marks evals unchecked, although corresponding files/tests now exist. Failure mode: the plan is no longer a reliable status source. Focused fix: replace checkboxes with an evidence matrix: implemented, unit-tested, runner-wired, H200-exercised, compile-gated.

Low - [Contradiction] current “roughly 500M” wording conflicts with live local log: [train_eval_stage1.log](/Volumes/external/sources/cppmega.mlx/outputs/train_eval_stage1.log:1) reports `hidden=1280 depth=24` and `[params] 714.68M`. Focused fix: curriculum docs should call this the current dense ~715M run, or pin a separate true ~500M config.

**Implemented Vs Exercised**

Causal LM: implemented and exercised in the real local runner; compile probe at step 1 had `compile_pass_rate=0.000`.

Docstring/codegen eval: implemented and exercised by local/H200 receipts; sampled H200 and local graph-route docstring compile reports are both `0/4`.

FIM/AST-FIM: implemented and unit/smoke exercised; not wired into the long local runner.

IFIM: implemented and unit-tested; current smoke folds IFIM into AST-FIM because fixture lacks `source_text`.

Commit pre/post/diff: objective builders implemented and tests exist; current smoke uses synthetic/post-as-diff data, not real PR-enriched diff supervision.

Graph routing/build diagnostics: parsers/routes/evals exist, and H200 graph-route curriculum artifacts reached 16k after batch fallback; MLX real train does not fail-closed on graph sidecars yet.

Tool trajectories/world model/stable loop: implemented and unit/smoke exercised; not yet a large-stage curriculum run.

**Staged Curriculum**

Stage A, stabilize causal/code base at 1k/2k: 70% causal C++/headers, 10% build files, 10% AST-FIM, 5% IFIM docstring, 5% graph/indexer. Gate: val loss down, no tokenizer sidecar drift, compile smoke non-regressing.

Stage B, repair mix at 2k/4k: 35% causal C++, 20% AST-FIM/IFIM, 25% real commit pre/post/diff, 10% PR discussion/docstring conditioning, 10% build diagnostics. Gate: realized ratio report, unified-diff fixture test, compile/docstring eval improves over current 0/4.

Stage C, graph-diagnostic routing at 4k/8k: 30% code, 20% build/shell/diagnostics, 20% diagnostic-conditioned IFIM, 15% graph edge/indexer losses, 15% commit/PR repair. Gate: graph sidecar coverage thresholds, default diagnostic edge weights verified, token-only baseline disabled.

Stage D, world/tool trajectory at 8k/16k: 25% causal/code refresh, 25% commit/PR repair, 25% compile/test observation trajectories, 15% tool action/result prediction, 10% stable-loop/world-model reward/transition. Gate: trajectory packets from real tool outcomes only, reward labels from compile/test results, compile-fix eval and local macOS smoke pass before H200 continuation.

Changed files: none. Simplifications made: none. Verification was read-only source/artifact inspection; I did not run tests or touch processes. Attempting process enumeration was blocked by sandbox permissions, so I did not verify active data jobs beyond avoiding any mutating commands.

