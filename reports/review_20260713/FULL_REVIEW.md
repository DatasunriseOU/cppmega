# CppMega Full Data, Model, Training, and Codegen Review

Reviewed: 2026-07-13  
Repositories: `cppmega` and `cppmega.mlx`  
Method: eight independent `gpt-5.5`/`xhigh` review lanes, exhaustive manifest-backed parquet audit, focused fixes, regression tests, and a frozen five-bucket Megatron build.

## Verdict

The landed corpus is large enough for the next meaningful training cycle, but the old parquet-to-Megatron path and the earlier graph-route checkpoints are not acceptable as correctness evidence. The review found real training corruption in packed `code/1024`, stale/empty sidecars, wrong graph-coordinate handling, gradient-dead side embeddings, fail-open producer states, and eval gates that could misreport generated code.

The confirmed corruption and conversion-contract defects are now repaired in code and in the frozen build snapshot. The next training run must use only the new manifest-backed bundle and must not reuse the old `cppmega_*current_mix*` or `cppmega_reindexed_*` prefixes as data-contract evidence.

The project still lacks one production training path that jointly exercises the implemented FIM/IFIM, real commit/PR transforms, graph-supervised routing, diagnostics, and trajectory objectives. Those are implemented in pieces, but not yet a single trustworthy curriculum runner.

## Highest-Severity Findings

| Severity | Finding | Current status | Evidence / action |
|---|---|---|---|
| Critical | 810,921 `code/1024` rows trained across logical document boundaries; 3,719,382 targets leaked across documents. | Fixed in producer and frozen snapshot repair; exhaustive post-repair audit passed with zero bad files/rows before publication. | `cppmega.mlx/scripts/repair_packed_document_boundaries.py`; [landed parquet review](../lane2-landed-parquet-truth-REVIEW.md#cr-01--critical--blocker-810921-landed-rows-train-across-real-document-boundaries) |
| Critical | Conversion previously globbed failed/interrupted parquet files, including 39 non-manifest residuals. | Fixed by immutable manifest allowlist and hardlink snapshot. | `scripts/data/build_macro_routes_megatron_bundle.py`; [Lane 2](lanes/02-parquet-truth.md) |
| Critical | Packaged H200 tokenizer had IDs 46/47 as reserved tokens rather than `<SPACE>/<NL>`. | Fixed and verifier now checks exact strings in both artifacts. | `data/tokenizer_v2/tokenizer.json`; `scripts/data/verify_tokenizer_contract.py` |
| Critical | Megatron wrapper omitted loss, domain, role, confidence, and graph-route sidecars. | Fixed; wrapper reuses the converter's canonical contract. | `scripts/data/prepare_format_megacpp.py`; `scripts/data_prep_parquet_to_megatron.py` |
| Critical | Domain/structure residual embeddings initialized both factors to zero, so neither factor received gradients. | Fixed with zero residual but live gradient path; regression covered. | `cppmega/features/{domain,structure}/embedding.py`; `tests/test_*embedding.py` |
| Critical | Commit parse failures and unresolved manifests could still exit zero. | Fixed fail-loud in conveyor and standalone drivers; parse errors require explicit handling. | `cppmega.mlx/tools/clang_indexer/process_commits.py`; `scripts/streaming_{conveyor,reindex,reindex_commits}.py` |
| High | Call/type edges were chunk IDs in parquet but treated as token offsets in Megatron. | Fixed with graph schema v2 and explicit chunk-to-sample remap. | `cppmega/megatron/structure_dataset_patch.py`; converter manifest coordinate spaces |
| High | Token/source platform mirrors were universally zero despite real multi-label provenance. | Fake token mirrors removed; lossless per-source platform bags preserved as nested CSR tied to row-local `doc_ids`. | `cppmega_source_platform_v1` in converter/bundle manifests |
| High | The exhaustive auditor read each parquet file in one allocation and reached 6-9 GB RSS per worker on large repositories. | Fixed with additive row-group auditing; the real 6.2M-token `lammps` shard now peaks at 400 MB with identical counters. | `cppmega.mlx/scripts/audit_sidecar_parquet.py`; row-group-only regression |
| High | The local MLX MMIDIDX reader dropped every compact sequence shorter than the context bucket; only 220 of 441,140 rows were iterable at 2048. | Fixed: converter metadata now triggers in-memory restoration of the original fixed row with zero padding, without expanding the 162 GiB bundle. All five buckets expose exactly one local sample per source row and keep document-aligned graph routes. | `cppmega.mlx/cppmega_mlx/data/megatron_indexed.py`; `local_mlx_bundle_smoke.json` |
| High | The S3 publisher described bundle keys as immutable but could overwrite an existing object whose SHA differed. | Fixed: artifacts and bundle manifests now fail on any remote mismatch; overwrite is allowed only for the final `latest.json` pointer. | `scripts/data/publish_megatron_bundle_to_nebius_s3.py`; mismatch regression |
| High | Nebius returns user metadata as `Sha256`, while the publisher originally compared only lowercase `sha256`; correct uploads therefore failed their post-upload HEAD gate. | Fixed from a live Nebius receipt: metadata keys are normalized case-insensitively and the exact Nebius form is regression-tested. | `scripts/data/publish_megatron_bundle_to_nebius_s3.py`; remote HEAD proof in `archive_publish_receipt.json` |
| High | `qname` is no longer the global DB primary key, but local/global lookup and 31-bit semantic token IDs still collapse overloads to one qname identity. | Storage now preserves composite `symbol_uid`; routing remains incomplete and requires clang USR/signature identity plus candidate resolution. | `ProjectIndex.functions[qname]`; `_compute_symbol_id(qname)`; `GlobalSymbolReader.lookup(... LIMIT 1)` |
| High | Graph sidecars could be silently truncated and sparse DSA could gather sentinel indices. | Fixed fail-closed capacity checks and sentinel masking. | `structure_dataset_patch.py`; `dsa_sparse_attention.py` |
| High | FastMTP could use unsafe `reduction="none"` and ignore the real loss mask. | Fixed to a safe scalar masked loss; gradient tests covered. | `cppmega/megatron/fastmtp_layer.py` |
| High | Eval compile gate accepted semantically wrong but compilable programs, and completion trimming cut valid nested blocks. | Fixed: binary oracle executes when required; lexical brace trimming preserves nested blocks. | MLX and H200 eval scripts/tests |

## Remaining Training Blockers

1. **Production objective mixer is missing.** The long Megatron/H200 path is still primarily causal CE. `TaskMixer`, AST-FIM, IFIM, commit objectives, graph losses, tool/world packets, and stable loops exist, but are not wired into one production runner with realized-token accounting.
2. **MLX graph routes stop at the reader.** `MegatronIndexedDataset` can load graph v2 packets, but the normal MLX batch/model/loss path does not automatically turn them into `block_bias` or indexer supervision. A graph packet can therefore be loaded without changing logits.
3. **Prompt inference does not yet construct real repository graph routes.** H200 standalone generation now supplies an explicit empty graph packet, avoiding a crash and avoiding a false absent-graph state. Repo repair/code-with-dependencies eval still needs clang graph extraction and graph tensors for the prompt.
4. **Fused DSA caller masks remain unproven.** The fused score wrapper accepts upstream `mask=` but the local patch does not apply it itself. A masked high-score key must be excluded in an end-to-end upstream-version test before sparse long-context training.
5. **Commit/PR supervision is corpus text, not a typed objective.** The current `CommitPacket` has no PR title/body/comments fields, and the Stage-1 smoke incorrectly uses adjacent post rows as a fake unified diff. Real pre/post/diff/PR examples must be materialized and counted separately.
6. **Producer promotion is not fully transactional across all five buckets.** The frozen builder is safe because it selects only manifest-complete units. Future production should publish all five parquet outputs plus dedup claims as one unit transaction; sibling staged dedup claims can still race before global promotion.
7. **Overlong exclusions are incomplete provenance.** Code dropped 7,224 documents / 200,412,883 tokens above 16k without durable per-unit manifest accounting. Commit drops are recorded. The code path must record the same exclusion receipt.
8. **The frozen set is a certified completed subset, not a final corpus.** Its source manifest contains 16 failed units. They are excluded from this bundle, while live conveyors continue producing a later generation.
9. **Restart reservations are still PID-only.** A reused PID can keep a stale unit reserved because the claim has no process start time, command identity, or heartbeat expiry.
10. **Live parquet publication is not atomic at the final filename.** Manifest gating prevents incomplete files from entering this frozen bundle, but external readers can still observe a torn file before a producer marks the unit done. Future producers should write a sibling temp, validate it, then `os.replace`.
11. **Symbol identity is only half-fixed.** The global DB primary key correctly includes base lib, repo, file, source span, and body hash, so qname is not claimed unique in storage. However, the local index maps functions by qname, token semantic IDs hash only qname into 31 bits, and global lookup chooses one qname match by body size. Replace these with clang USR/signature IDs, keep `qname -> candidates`, resolve calls through referenced cursors, and regenerate affected semantic/call/type routes. Tests must cover overloads in one TU, same qname across files/repos, templates, and inline namespaces.
12. **Clang semantic extraction can silently return partial arrays.** `extract_semantic_metadata` catches every AST-walk exception and returns whatever was filled without a mandatory confidence downgrade. Restrict catches to known libclang failures, record parse diagnostics and completeness, and exclude partial rows from graph-supervised losses unless an explicit lossy policy is selected.

## Remediation Plan

| Priority | Workstream | Concrete changes | Proof / data impact |
|---|---|---|---|
| P0 | Symbol identity | Add clang USR/signature fields in `index_project.py` and `process_commits.py`; index local symbols by USR; store qname candidate lists; make global lookup return candidates and resolve referenced cursor/signature. | Overload/template/same-name fixtures; regenerate semantic IDs and call/type routes before graph-supervised training. |
| P0 | Production objective mixer | Wire `TaskMixer`, AST-FIM/IFIM, real unified diff/pre-post, recovery, and graph losses into the H200 runner; report realized loss-mask tokens per objective/domain/context. | 50M-token S0 receipt must match configured ratios and reject synthetic post-as-diff examples. |
| P0 | Graph path parity | Convert loaded graph v2 packets into dense bias and indexer targets in normal MLX batches; build real prompt graph packets on both MLX and H200; preserve generated suffix sidecar policy explicitly. | Identical prompt+graph Megatron/MLX logits test; graph-on changes logits/gradients; empty graph only for explicit standalone tasks. |
| P0 | Codegen eval contract | Separate source-prefix, docstring/signature-only, FIM, and IFIM prompts; add candidate IDs/pass@k, leakage hashes, clang-format/parse/compile/link/run/sanitizer outcomes. | Reference completions green locally; old 0/4 checkpoints remain baseline, not acceptance evidence. |
| P1 | Domain parsers/data | Add multiline CMake and distinct Make/Ninja/Bazel/Autotools parsers; dialect-specific shell parsers; GCC/Clang/MSVC/linker/build/test diagnostics; embedded SQL extraction and routes. | Parser golden corpus; delimiter IDs present; nonzero shell/diagnostic/cross-domain edge coverage; regenerate affected parquet. |
| P1 | Producer transactions | Publish five parquet buckets + dedup claims atomically; write final files via temp+replace; add reservation start-time/command/heartbeat; resolve sibling staged-dedup race. | Kill/restart and duplicate-race tests; no torn visible files or stranded claims. |
| P1 | Sparse DSA | Apply caller mask before top-k, define deterministic graph/neural candidate union, measure recall and dense parity; keep dense GQA at current contexts. | recall@256 >=95%, dense-vs-sparse NLL delta <=2%, then 32k/64k/128k memory/MFU sweep. |
| P2 | Stable loop/tool world model | Define typed trajectory packets and actions, stable loop depth schedule, fixed-point/deep-supervision losses, and verified build/test outcomes. | Bounded residual norms; real edit-build-test trajectories; reward labels trace to tool receipts. |

## Frozen Parquet Inventory

Only `_done.json`-backed files are in the immutable snapshot. Physical orphan files and live post-freeze output are excluded.

| Source | Bucket | Files | Rows | Valid tokens |
|---|---:|---:|---:|---:|
| code | 1024 | 293 | 1,509,110 | 1,495,779,442 |
| code | 2048 | 293 | 222,445 | 315,334,535 |
| code | 4096 | 289 | 100,355 | 282,151,520 |
| code | 8192 | 289 | 39,880 | 223,059,383 |
| code | 16384 | 277 | 14,863 | 165,555,100 |
| commits + integrated PR | 1024 | 2,073 | 294,972 | 216,519,785 |
| commits + integrated PR | 2048 | 2,074 | 218,695 | 314,170,230 |
| commits + integrated PR | 4096 | 2,074 | 129,649 | 371,189,937 |
| commits + integrated PR | 8192 | 2,069 | 70,160 | 398,656,066 |
| commits + integrated PR | 16384 | 2,040 | 31,334 | 350,331,237 |
| **Total** | | **11,771** | **2,631,463** | **4,132,747,235** |

Expected trained-token count after boundary repair is **4,123,963,840**. Capacity including padding is 5,351,270,400 tokens. Standalone PR parquet remains intentionally absent; PR text is integrated into commit documents. Future SHA-based PR joins now retain canonical PR number/title, but 14,041 already-frozen PR-bearing source documents still have null structured PR numbers while retaining their discussion text.

At the equal-capacity schedule (1024x192, 2048x96, 4096x48, 8192x24, 16384x12 = 196,608 capacity tokens/step), the bundle contains these trained-token-equivalent steps: **8,668 / 3,199 / 3,321 / 3,161 / 2,623** respectively. These are loss-mask token equivalents, not claims that every packed batch is 100% trained tokens.

Graph inventory is substantial but domain-skewed: **13,594,174 chunks**, **644,450 call edges**, **452,370 type edges**, and **17,949,252 token-coordinate edges**. The latter are 15,889,000 code-domain, 1,488,023 commit-domain, and 572,229 code-build edges. Shell, diagnostic, and cross-domain edge counts are zero in this frozen generation, so those mechanisms are contracts without training coverage yet.

The rebuilt global symbol DB now contains **14,199 `std` rows** (10,570 functions, 3,629 types), **14,276 total `std::`-prefixed rows**, and **zero libiberty rows**. Thus the original A2 contamination is fixed. It also makes the remaining identity defect measurable: those 14,199 std rows collapse to only 4,869 distinct qnames; operators such as `std::operator==` have more than 100 definitions/overloads, while the current reader selects one candidate.

## New Megatron Bundle Contract

Each of the five sequence buckets contains:

- token `.bin/.idx` in MMIDIDX v1;
- `loss_mask` and row-local `doc_ids`;
- domain, role, entity, scope, confidence, structure, AST, semantic, and temporal token sidecars;
- graph schema `cppmega_graph_routes_v2` with chunk-index call/type edges, token-index domain/build/shell/diagnostic/cross-domain edges, and explicit chunk spans;
- compact source-platform nested CSR (`cppmega_source_platform_v1`) instead of a fabricated scalar platform token channel;
- document/token/trained-token counts, dtypes, coordinate spaces, and SHA-256 artifact inventory;
- source manifest, repaired-snapshot manifest, repair receipt, and exhaustive audit receipt.

Local target: `/Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/macro_routes_v1_20260713`  
Build status: **COMPLETE; local five-bucket MLX smoke passed**  
Bundle ID: `macro_routes_v1_20260713-8c550514dd52ddc9`  
Artifacts: **236 files / 174,421,216,970 bytes**  
Local archive: `/Volumes/external/sources/cppmega.mlx/outputs/megatron_ready/macro_routes_v1_20260713.tar.zst`  
Archive: **2,823,467,371 bytes**, SHA-256 `5ac3e095289b28d4924c3547ebae62ee79f002c4ad0c81467fda59b03a1c93a7`, approximately **61.8x smaller** than the logical artifact bytes  
Nebius target: `s3://cppmega-sidecar-20260627/cppmega-megatron/macro-routes/transports/macro_routes_v1_20260713-8c550514dd52ddc9/`  
Publish status: **COMPLETE; archive, logical manifest, transport descriptor, and final latest pointer independently HEAD-verified**

The archive contains exactly 237 members: the 236 manifest artifacts plus the byte-identical logical `manifest.json`. Before upload, the publisher streamed all 174.42 GB through zstd/tar and recomputed every member size/SHA with bounded RSS under 83 MB. It then uploaded the immutable archive, uploaded an independently inspectable `logical_manifest.json`, wrote `transport.json`, and updated `latest_transport.json` last. All four remote objects were independently re-read or HEAD-checked after the publisher exited; no multipart uploads remain. The abandoned loose-object prefix was never committed and was removed.

Nebius restore is fail-closed and atomic:

```bash
python3 scripts/data/restore_megatron_bundle_from_nebius_s3.py \
  --output-root /data/cppmega_sidecar/bundles \
  --bundle-id macro_routes_v1_20260713-8c550514dd52ddc9
```

The restore path verifies the latest/transport/logical-manifest hashes, checks free space for the 174.42 GB expansion plus headroom, validates the archive SHA, extracts into a partial directory, rehashes all logical artifacts, and only then performs the final rename.

The MMIDIDX converter was also changed from per-row/per-sidecar writes to Arrow row-group extraction plus batched NumPy writes. On the same real `lammps` shard it converted 6,205,366 tokens in 3.7 seconds instead of 64 seconds, about a 17x improvement. Ragged graph and source-platform values retain the same per-row validation but flush once per row group.

The local MLX reader recognizes this converter's compact fixed-row contract from `source_capacity_token_count`, restores omitted padding in memory, and pads every token sidecar consistently. Real-bundle verification exposed **1,804,082 / 441,140 / 230,004 / 110,040 / 46,197** local samples across the five buckets, exactly matching the MMIDIDX document counts; graph packets loaded in every bucket.

## Target System

CppMega is a C/C++ software world model, not a general-purpose chat model. The intended system has five explicit contracts:

1. C/C++ source, comments/docstrings, macros/templates, and semantic clang metadata form the primary language stream.
2. CMake/Make/Ninja/Bazel/Autotools and Bash/Sh/Zsh/Tcsh, compiler/linker/build/test diagnostics, and later embedded SQL are distinct delimiter-bounded domains with domain-specific roles and parsers; they are not flattened into comments.
3. Dense GQA consumes the full context while deterministic AST/def-use/type/call/include/build/diagnostic routes bias information flow. A learned DSA indexer combines `I_neural + beta*S_graph` only after dense graph training proves retrieval quality, mainly for 32k-128k contexts.
4. Commit + integrated PR examples teach real pre/diff/post repair transitions. Stable looped blocks later model edit-build-diagnostic-test trajectories rather than only next-token text.
5. Generation emits code or typed edit/tool actions, then reparses, formats, compiles, links, executes, and feeds verified outcomes back into evaluation and trajectory training.

## Model and Mechanism Review

| Mechanism | Implemented | Production exercised | Remaining work |
|---|---|---|---|
| Dense GQA + graph bias | Yes | Earlier H200 curriculum reached 16k, but used the old faulty data/embedding contracts. | Rerun on this bundle; compare graph-on/off logits and graph-edge gradients. |
| DSA `I_neural + beta*S_graph` | Yes in Megatron patch | Unit/reference covered; no trustworthy long-context run yet. | Mask correctness, recall@k gate, dense-vs-sparse NLL parity, then enable near 128k rather than forcing DSA at 1k-16k. |
| MLX graph-route parity | Reader only | No | Wire graph packet through batch/model/loss, then compare Megatron vs MLX logits on identical prompt + graph. |
| Mamba3 / NAM routes | Route implementations exist | Smoke/older runs | Pass `trap/angles` in no-conv variant; validate TP/cache/decode and route-specific activations. |
| MTP / FastMTP | Yes | Smoke | Define intended backbone/head gradient ownership and add parity receipt. |
| Stable loop / world model | Reference implementation | Smoke only | Production architecture metadata, loop-depth curriculum, fixed-point diagnostics, trajectory runner. |
| Build/shell/diagnostic domains | Contracts/parsers/routes exist | Partial data paths | Add real ingestion and edge coverage; current frozen graph totals have zero shell, diagnostic, and cross-domain edges. |

Dense GQA still attends to all tokens. Graph routes do not remove context there; they add a deterministic structural prior. DSA becomes valuable when the context is long enough that sparse selection matters. The correct order is dense graph-biased training and graph-indexer supervision first, then sparse DSA after retrieval quality is demonstrated.

## Code Generation Plan

1. **Define prompt contracts, do not mix labels.** Keep `source-prefix-with-docstring` for body completion, add a genuinely separate docstring/signature-only contract, and add dedicated FIM and IFIM middle-generation contracts. Existing `prompt_mode=docstring` is currently an alias for full source prefix and must not be reported as docstring-only.
2. **Build prompt sidecars before decode.** Run the language/domain parser and clang adapter on the prompt. For repository repair, include dependency blocks plus real call/type/def-use/build/diagnostic routes. Standalone one-function tasks may explicitly carry an empty graph.
3. **Handle generated suffix honestly.** Zero sidecars are acceptable only for a minimal baseline. The production path should incrementally parse stable generated regions or periodically rebuild prompt + suffix sidecars, then always reparse the full final candidate.
4. **Format and validate.** Decode special whitespace exactly, reject metadata tokens, apply `clang-format`, parse with clang, compile, link, execute the test oracle, and classify diagnostics. Formatting is not a correctness gate by itself.
5. **Use a real eval matrix.** C and C++; source-prefix, docstring/signature, FIM, IFIM, commit repair, build repair; greedy and pass@1/5/10; zero vs parsed prompt sidecars; graph off/on; compile, link, run, timeout, sanitizer, repetition, and leakage status.
6. **Add leakage protection.** Every eval case needs repo/commit/file/function/content hashes checked against the frozen training manifest.
7. **Keep evals local by default.** Convert checkpoints on H200 if necessary, copy only checkpoints/receipts, and run generation/compile gates on local macOS unless a remote comparison is explicitly requested.

Current evidence remains `0/4` compile for both older native CUDA generation and converted MLX generation. That means the problem is not only checkpoint conversion. The likely causes are insufficient objective-aligned training, prompt mismatch, metadata-token drift in old runs, and missing real graph-conditioned inference. The old 5,000-step causal checkpoint's low teacher-forced loss/PPL is not evidence of usable code generation.

## Training Curriculum

The current dense model is approximately **714.68M parameters**, not 500M. Use real non-padding trained tokens for accounting.

| Stage | Cumulative tokens | Objectives | Context/domain mix | Promotion gate |
|---|---:|---|---|---|
| S0 contract smoke | 0.05B | causal CE + sidecar/graph gradient probes | 1k/2k | exact tokenizer/data hashes; graph-on changes logits; compile harness green on references |
| S1 C/C++ foundation | 5.0B | causal 65%, FIM 10%, AST-FIM 10%, IFIM 5%, symbol/type/callee recovery 10% | C/C++ 70%, build 10%, shell 5%, commits 10%, diagnostics 5%; 1k/2k/4k = 45/35/20 | objective-token ratios, no token-only fallback, held-out PPL + syntax/compile trend |
| S2 commit/repair midtrain | 8.5B | causal 30%, FIM 10%, AST-FIM 10%, IFIM 10%, real commit diff 20%, pre-to-post 15%, recovery 5% | commits + integrated PR about 45%; 2k/4k/8k | real unified-diff and PR packets, repair compile/run improvement |
| S3 graph/diagnostic routing | 10.5B | graph-positive examples 70%, dense graph bias, indexer KL/ranking, diagnostic-conditioned repair | 4k/8k/16k | graph edge coverage; indexer recall@256 >=95%; dense-vs-sparse NLL delta <=2% before sparse DSA |
| S4 tool/world trajectories | 13.2B | typed edit/tool actions, next diagnostic/test outcome, graph delta, reward/continuation | 8k/16k, later 32k-128k | real tool outcomes only; stable-loop residuals bounded; local compile/test trajectory eval |

Save milestone checkpoints at 50M, 5B, 8.5B, 10.5B, and 13.2B trained tokens, with rolling 250M-token checkpoints including optimizer, RNG, scheduler, and exact data cursor. Do not treat five context buckets as five unrelated models: they are one curriculum with a checkpoint after each stage/bucket boundary.

The frozen bundle provides 4.124B causal loss targets. Reaching the 5B S1 boundary is therefore about 1.21 effective passes when FIM/AST-FIM/IFIM transforms are included, not a claim of 5B unique source tokens. For the current ~714.7M model, the 13.2B final target is close to the usual ~20 tokens/parameter compute-optimal order of magnitude; live conveyor additions should supply fresh examples rather than repeatedly cycling this snapshot.

## Verification Completed

- Eight independent review reports: [`lanes/`](lanes/).
- Exhaustive landed parquet review: [`lane2-landed-parquet-truth-REVIEW.md`](../lane2-landed-parquet-truth-REVIEW.md).
- Focused root suites for converter, builder, graph routes, DSA, embeddings, FastMTP, tokenizer, curriculum/subsetting, H200 generation, and S3 publication.
- Focused MLX suites for packing, boundary audit/repair, conveyor restart/dedup/recompression, PR lookup, Megatron reader, generation, domain eval, AST-FIM, and graph weights.
- Current focused totals: `cppmega` **151 passed, 1 skipped**; `cppmega.mlx` **239 passed**. The MLX total is from the repository `.venv` containing `mlx`/`mlx_lm`; a discarded system-Python run failed only because those packages were absent.
- The HTML report was rendered in Chromium at 1440x1000 and through a true 390x844 DevTools mobile viewport. Both screenshots are stored under [`screenshots/`](screenshots/); the mobile receipt proves `innerWidth=390`, `htmlScrollWidth=390`, and no verdict overflow.
- A prior broad root run reached **836 passed, 149 skipped, 34 failed**. Two confirmed defects from that run (MoE identity-cache aliasing and Mamba mutation acknowledgement) were fixed and focused tests pass. The remaining failures are mostly optional runtime/import gaps (`mamba_ssm`, full Megatron, partial local `cppmega_mlx`) and stale recipe/wave expectations. A fresh generic `pytest -q` currently stops during collection because the optional `modal` package is imported unconditionally by `test_mamba3_mono_ab_modal_hygiene.py`; this is not counted as a green broad run.
- Local bundle receipt: [`local_mlx_bundle_smoke.json`](local_mlx_bundle_smoke.json); immutable artifact inventory: [`manifest.json`](../../../cppmega.mlx/outputs/megatron_ready/macro_routes_v1_20260713/manifest.json); S3 transport receipt: [`archive_publish_receipt.json`](../../../cppmega.mlx/outputs/megatron_ready/macro_routes_v1_20260713/archive_publish_receipt.json).

## Changed Areas

`cppmega`: converter and wrapper contracts, immutable bundle builder, Nebius object/archive publisher and atomic restore path, tokenizer contract, structure/domain embeddings, graph-v2 bridge, DSA/sparse attention, FastMTP, H200 curriculum/generation, subset tooling, and regression tests.

`cppmega.mlx`: row-local packing, boundary oracle/repair, bounded row-group auditing, fail-loud conveyor and commit parsing, readonly PR lookup, graph-v2 Megatron reader with compact fixed-row restoration, Dense generation, runtime eval gates, AST-IFIM accounting, graph weights, and regression tests.

## Remaining Risks

- The compressed S3 transport is complete and live-proven. Its restore script is unit/smoke-tested locally, but a full 174.42 GB extraction has not yet been exercised on a fresh Nebius training host; that is the first preflight step for the next H200 run.
- Current live producer output is newer than this frozen bundle and must become a new bundle ID rather than mutating this one.
- The old graph-route H200 checkpoints were trained before the boundary, edge-coordinate, tokenizer, and embedding fixes; use them only for debugging conversion mechanics.
- A trustworthy next H200 training run is blocked on the production objective mixer and graph-conditioned inference/eval wiring, not on the availability of causal tokens.
