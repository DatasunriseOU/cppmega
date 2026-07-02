# Domain-Routed Build, Shell, and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make C++ code, build systems, shell scripts, compiler/linker/build/test diagnostics, and tool outputs first-class domain-routed training documents with explicit token delimiters, typed sidecars, graph routes, and Megatron/MLX consumption.

**Architecture:** Keep one frozen tokenizer artifact and map new logical roles to existing `<RESERVED_N>` ids. Parse every domain into a typed `DomainPacket`, emit parquet sidecars and graph routes, then feed those routes into dense GQA bias and DSA indexer/top-k bias. Do not train domain streams as undifferentiated comments unless they are true PR/commit discussion text attached to a code/edit example.

**Tech Stack:** Python 3.13, pyarrow/parquet, tree-sitter where available, existing cppmega tokenizer contract, `cppmega.mlx` data pipeline, `cppmega` Megatron sidecars, MLX smoke tests, pytest.

---

## Non-Negotiable Contract

- The tokenizer stays frozen at 65,536 vocab. New logical delimiters must use ids from `cppmega_mlx/tokenizer/tokenizer_contract_v1.json`.
- Domain delimiters are inserted as token ids, not as literal strings like `<CMAKE_START>` unless the tokenizer artifact also resolves that literal to the same id.
- C++ graph routes are not optional for cppmega world/code models. If graph-route fields are requested but unavailable, loaders must fail closed unless the run is explicitly marked `--allow-token-only-baseline`.
- Build, shell, and diagnostics are separate domains with separate parsers and sidecars. Do not collapse CMake/Make/Ninja/Bazel/Autotools into one “build text” label.
- Diagnostics are not comments/docstrings by default. They are structured observation documents linked to code/build/edit documents. Commit/PR discussion may be rendered as docstrings only when it is intentionally human prose attached to a commit transformation.

## Existing Anchors

- Token role source of truth: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer_contract_v1.json`
- Domain role mirror: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/tokenizer_contract.py`
- Current build-file discovery/docs: `/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/index_project.py`
- Build context detector: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/nanochat_pipeline/build_context.py`
- Expert SFT extractor: `/Volumes/external/sources/cppmega.mlx/scripts/build_expert_sft_data.py`
- Megatron sidecar converter: `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py`
- Dense graph attention bias patch: `/Volumes/external/sources/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py`
- Existing route types: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/code_graph_routes.py`

## Target Document Types

| Domain | Start/end roles | Parser | Main sidecars | Route edges |
|---|---|---|---|---|
| C/C++ | `CPP_CODE_START/END` | Clang + tree-sitter fallback | AST, symbols, type refs, call refs, def-use, chunk deps | call, type, def-use, AST, chunk, include |
| CMake | `CMAKE_START/END` | CMake parser + CMake file API when configured | command ids, target ids, variable ids, generator expressions | target->source, target->lib, var-use->def, command->target |
| Make | `MAKE_START/END` | Make parser + recipe shell lexer | target ids, prerequisites, variables, recipe spans | target->prereq, var-use->def, recipe->command |
| Ninja | `NINJA_START/END` | Ninja manifest parser | rule ids, build edge ids, input/output ids, depfile paths | output->input, build->rule, rule->command |
| Bazel | `BAZEL_START/END` | Starlark/tree-sitter + bazel query/cquery/aquery when runnable | rule ids, label ids, attr ids, config ids | target->deps, action->input/output, attr->label |
| Autoconf/Automake | `MAKE_START/END` plus build kind sidecar `autoconf/automake` | m4/autoconf + Makefile.am parser | macro ids, generated-file ids, var ids | macro->expansion, target->source, var-use->def |
| Shell | `BASH/ZSH/SH/TCSH_START/END` | shell-specific parser, shlex fallback only for weak mode | command ids, argv spans, var ids, pipe stages, redirs | command->file, var-use->def, pipe, redir, env |
| Compiler diagnostics | `COMPILER_DIAGNOSTIC/ERROR_START/END` | Clang/GCC/MSVC diagnostic parser, SARIF/JSON preferred | severity, tool, file, line, col, option, fixit spans | diagnostic->code-span, note->primary, fixit->replacement |
| Build diagnostics | `BUILD_DIAGNOSTIC/ERROR_START/END` | CMake/Make/Ninja/Bazel log parser | failing target, command, exit code, log phase | error->target, error->command, target->build-file |
| Linker errors | `LINKER_ERROR_START/END` | ld/lld/MSVC link parser | undefined symbol, object/lib, archive, target | symbol->reference, symbol->candidate-def, object->target |
| Test output | `TEST_OUTPUT_START/END` | ctest/gtest/pytest/custom parsers | test name, failure file/line, assertion spans | test->code-under-test, failure->assertion |
| Tool output | `TOOL_OUTPUT_START/END` | typed tool-result parser | tool kind, exit code, stdout/stderr spans | tool-result->action, artifact->file |

## Representation Rule

For a training example involving a build failure, render:

```text
<CPP_CODE_START> ... changed or relevant code ... <CPP_CODE_END>
<CMAKE_START> ... relevant CMakeLists.txt slice ... <CMAKE_END>
<BUILD_ERROR_START> ... raw build log excerpt ... <BUILD_ERROR_END>
<COMPILER_ERROR_START> ... normalized compiler diagnostic ... <COMPILER_ERROR_END>
<FIM_INSTRUCTION> fix the compile error ...
<FIM_PREFIX> code before hole <FIM_SUFFIX> code after hole <FIM_MIDDLE> target patch
```

Do not render compiler errors as `/** @brief ... */` unless the source commit really introduced a human-written comment. Diagnostics are observations from the world, not source-code comments.

## Sidecar Schema Additions

Add these parquet columns to all variable-length sequence outputs that may contain non-C++ domains:

```python
DOMAIN_TOKEN_SIDECARS = {
    "token_domain_ids": "uint16",       # cpp, cmake, make, ninja, bazel, autoconf, bash, zsh, sh, diag...
    "token_role_ids": "uint16",         # keyword, target, variable, command, path, option, symbol, severity...
    "token_entity_ids": "uint32",       # stable per-doc entity id
    "token_scope_ids": "uint32",        # function, target, rule, shell block, diagnostic group
    "token_source_doc_ids": "uint32",   # original source document in packed row
    "token_confidence_ids": "uint8",    # exact, partial, heuristic, raw
}
```

Add these graph sidecars:

```python
DOMAIN_GRAPH_SIDECARS = {
    "token_domain_edges": ("edge_triples", "int32"),       # src, dst, edge_kind
    "token_build_edges": ("edge_triples", "int32"),        # target/prereq/rule/action
    "token_shell_edges": ("edge_triples", "int32"),        # pipeline/redir/env/file
    "token_diagnostic_edges": ("edge_triples", "int32"),   # diag->source/note/fixit
    "token_cross_domain_edges": ("edge_triples", "int32"), # diag/build/test/tool -> code/build files
}
```

Edge kinds must be a versioned enum:

```python
class DomainEdgeKind(IntEnum):
    AST_PARENT = 1
    CALL = 2
    TYPE = 3
    DEF_USE = 4
    INCLUDE = 5
    BUILD_TARGET_DEP = 20
    BUILD_TARGET_SOURCE = 21
    BUILD_RULE_COMMAND = 22
    BUILD_ACTION_INPUT = 23
    BUILD_ACTION_OUTPUT = 24
    SHELL_PIPE = 40
    SHELL_REDIR_IN = 41
    SHELL_REDIR_OUT = 42
    SHELL_VAR_DEF_USE = 43
    DIAG_PRIMARY_LOCATION = 60
    DIAG_NOTE = 61
    DIAG_FIXIT = 62
    LINK_UNDEFINED_SYMBOL = 70
    TEST_FAILURE_LOCATION = 80
```

## Model Consumption

### Dense GQA

Build one additive bias:

```python
attention_bias = beta_code * B_code_graph
               + beta_build * B_build_graph
               + beta_shell * B_shell_graph
               + beta_diag * B_diagnostic_graph
               + beta_cross * B_cross_domain
```

The bias is `[B, 1, S, S]` and is passed through the existing TE/GQA seam. Bias must be causal-masked after construction. For non-C++ text, local attention still works; graph bias only reinforces parsed relations.

### DSA / Sparse Top-K

Before fused top-k:

```python
I_final(q, k) = I_neural(q, k) + beta * S_graph(q, k)
selected = topk(I_final, k_learned) union forced_graph_blocks
```

`forced_graph_blocks` should be small and capped:

- C++: def/use, call target, type declaration, changed chunks.
- Build: current target, source files of target, linked libraries.
- Diagnostics: primary source span, notes, fix-it spans, failing command/build target.
- Shell: command inputs/outputs and pipeline neighbors.

### Side Embeddings

Add small embeddings:

```python
x = token_embedding
  + domain_embedding[token_domain_ids]
  + role_embedding[token_role_ids]
  + confidence_embedding[token_confidence_ids]
  + existing_structure_embeddings
```

This prevents “CMake `target`” and “C++ identifier `target`” from collapsing into one semantics.

## Task 1: Domain Contract and Tests

**Files:**
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/tokenizer_contract.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/tokenizer/tokenizer_contract_v1.json`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/domain_schema.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_domain_schema.py`

- [ ] Add `DomainKind`, `DomainRoleKind`, `DomainEdgeKind`, and `ParseConfidence` enums in `domain_schema.py`.
- [ ] Add a test that every `*_START` has matching `*_END`, all ids resolve to `<RESERVED_N>`, and `DomainKind` maps to a delimiter pair.
- [ ] Run:

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python -m pytest -q tests/test_tokenizer_contract.py tests/test_domain_schema.py
```

Expected: both pass.

## Task 2: Domain Packet Abstraction

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/domain_packet.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/code_packet_builder.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_domain_packet.py`

- [ ] Define `DomainPacket(tokens, domain_ids, role_ids, entity_ids, scope_ids, confidence_ids, graph_edges, metadata)`.
- [ ] Add `validate_lengths()` that fail-louds if any token sidecar length differs from `tokens`.
- [ ] Add `wrap_with_domain_tokens(packet, start_id, end_id)` that inserts delimiters and pads all token sidecars correctly.
- [ ] Test with CMake and compiler-error packets.

## Task 3: Build-System Parsers

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/base.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/cmake.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/make.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/ninja.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/bazel.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/build_parsers/autotools.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/tools/clang_indexer/index_project.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_build_domain_parsers.py`

- [ ] CMake parser extracts commands, targets, sources, libraries, variables, generator expressions.
- [ ] Make parser extracts targets, prerequisites, variables, recipe command spans.
- [ ] Ninja parser extracts rules, build edges, inputs, outputs, variables, commands.
- [ ] Bazel parser extracts labels, rules, attrs, deps, srcs, copts, linkopts; if Bazel is runnable later, enrich with query/cquery/aquery.
- [ ] Autotools parser extracts configure macros and Automake targets/sources.
- [ ] Replace current flat `build_build_doc` output with `DomainPacket`-backed output while preserving `doc_type='build'` and `build_kind`.

## Task 4: Shell Parsers

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/base.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/posix.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/bash.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/zsh.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/shell_parsers/tcsh.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/agent_trajectory.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_shell_domain_parsers.py`

- [ ] Use a real parser if available in the environment; otherwise use fail-marked lexical parse with `ParseConfidence.HEURISTIC`.
- [ ] Extract command, argv, env assignment, variable use, redirection, pipeline, subshell, glob, and path spans.
- [ ] Wrap shell docs with the exact shell delimiter from shebang/file extension/tool metadata.
- [ ] Do not label unknown shell as bash; use `SH_START/END` unless evidence says zsh/tcsh/bash.

## Task 5: Diagnostics Parsers

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/base.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/clang.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/gcc.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/msvc.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/cmake.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/diagnostic_parsers/linker.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_diagnostic_domain_parsers.py`

- [ ] Prefer structured input: Clang parseable fixits, GCC JSON/SARIF where available, MSVC SARIF where available.
- [ ] Text fallback extracts primary file:line:col, severity, message, option, caret span, notes, fix-it hints, undefined symbol.
- [ ] Emit `COMPILER_ERROR` for compiler failure, `LINKER_ERROR` for link failure, `BUILD_ERROR` for build-system failure, `TEST_OUTPUT` for test failure.
- [ ] Add cross-domain edges from diagnostics to code tokens when file/path/line maps are available.

## Task 6: Parquet Emission and Audit

**Files:**
- Modify: `/Volumes/external/sources/cppmega.mlx/scripts/audit_sidecar_parquet.py`
- Modify: `/Volumes/external/sources/cppmega/scripts/data_prep_parquet_to_megatron.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/scripts/upload_verified_sidecar_to_nebius_s3.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_domain_sidecar_parquet.py`
- Test: `/Volumes/external/sources/cppmega/tests/test_domain_megatron_sidecars.py`

- [ ] Add `DOMAIN_TOKEN_SIDECARS` and `DOMAIN_GRAPH_SIDECARS` to parquet conversion.
- [ ] Audit must fail if a row has domain delimiters in `token_ids` but missing matching sidecar domain ids.
- [ ] Audit must fail if diagnostics rows have no `token_diagnostic_edges` and no explicit `ParseConfidence.RAW`.
- [ ] Megatron `.bin/.idx` sidecar conversion must include domain sidecars, not token ids only.

## Task 7: Graph Route Runtime

**Files:**
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/code_graph_routes.py`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/domain_graph_routes.py`
- Modify: `/Volumes/external/sources/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py`
- Modify: `/Volumes/external/sources/cppmega/cppmega/megatron/dsa_indexer_fused_patch.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_domain_graph_routes.py`
- Test: `/Volumes/external/sources/cppmega/tests/test_graph_route_attention_bias_patch.py`

- [ ] Convert domain edge triples into per-block and per-token route bias.
- [ ] Add beta weights by edge family: code/build/shell/diagnostic/cross.
- [ ] Add forced block candidates with caps per edge family.
- [ ] Dense GQA path consumes `attention_bias`.
- [ ] DSA path consumes `I_neural + beta*S_graph` before fused top-k.
- [ ] Fail closed if `CPPMEGA_GRAPH_ROUTES_ENABLED=1` and requested sidecars are absent.

## Task 8: Training Mix and Objectives

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/configs/stage_domain_routed_foundation.yaml`
- Create: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/domain_objectives.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/training/indexer_losses.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_domain_objectives.py`

- [ ] Add domain classification auxiliary loss from hidden states at opener tokens.
- [ ] Add edge prediction losses for build/shell/diagnostic graph edges.
- [ ] Add cross-domain retrieval/indexer loss: diagnostic token should rank primary code span, failing target, and build rule above hard negatives.
- [ ] Stage 1 recipe: mostly C++ + build metadata, small diagnostics.
- [ ] Stage 2 recipe: commit/PR + build/diagnostics + FIM/IFIM repair.
- [ ] Stage 3 recipe: trajectory/world-model transitions with tool outputs and compile/test observations.

## Task 9: Eval Suite

**Files:**
- Create: `/Volumes/external/sources/cppmega.mlx/evals/domain_routed_prompts.jsonl`
- Create: `/Volumes/external/sources/cppmega.mlx/scripts/eval_domain_routed_codegen.py`
- Test: `/Volumes/external/sources/cppmega.mlx/tests/test_eval_domain_routed_codegen.py`

- [ ] Eval C++ docstring-to-code with and without sidecars.
- [ ] Eval FIM compile-fix with compiler diagnostic packet.
- [ ] Eval CMake fix from CMake error packet.
- [ ] Eval linker fix from undefined-symbol packet.
- [ ] Eval shell/build command understanding from build logs.
- [ ] All compile gates run locally on macOS by default unless explicitly overridden.

## Task 10: Regeneration Gate

**Files:**
- Modify: `/Volumes/external/sources/cppmega.mlx/scripts/build_verification_report.py`
- Modify: `/Volumes/external/sources/cppmega.mlx/scripts/audit_sidecar_parquet.py`
- Create: `/Volumes/external/sources/cppmega.mlx/scripts/verify_domain_routed_dataset.py`

- [ ] Verification report prints token counts by domain and by seqlen bucket.
- [ ] Report sidecar fill rates for each domain independently.
- [ ] Report edge-density by edge family and confidence.
- [ ] Fail if C++ rows have graph-route coverage below threshold.
- [ ] Fail if build/shell/error rows are present but delimiter ids are missing.
- [ ] Fail if `COMPILER_ERROR` text was placed inside C++ comment/docstring unless row type is explicitly `commit_discussion`.

## Research Notes To Apply

- GraphCodeBERT validates using data-flow edges and graph-guided masked attention rather than plain token sequence pretraining.
- AST-T5 validates AST-aware segmentation/span corruption for code generation.
- Constrained decoding work is useful for inference, but grammar constraints can distort model likelihoods; use it as decode-time guardrail, not as substitute for training graph routes.
- Graph2Diff-style build repair treats source, build files, and diagnostics as one graph for predicting diffs. That is the closest research shape to our build-error sidecars.
- Use official build tool APIs when possible: CMake file API, Bazel query/cquery/aquery, Ninja logs/deps, structured diagnostics via SARIF/JSON/fixits.

## Rollout Order

1. Implement schema/enums and packet wrapper.
2. Implement build parsers and parquet sidecars.
3. Implement diagnostic parsers and cross-domain edges.
4. Implement shell parsers for trajectory/tool data.
5. Wire parquet -> Megatron sidecars.
6. Wire MLX route bias smoke path.
7. Wire Megatron dense GQA + DSA graph routes.
8. Regenerate a small 1024/2048/4096 corpus slice.
9. Run local evals and H200 smoke training.
10. Only then restart full conveyor/regeneration.

## Verification Commands

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python -m pytest -q \
  tests/test_tokenizer_contract.py \
  tests/test_domain_schema.py \
  tests/test_domain_packet.py \
  tests/test_build_domain_parsers.py \
  tests/test_shell_domain_parsers.py \
  tests/test_diagnostic_domain_parsers.py \
  tests/test_domain_sidecar_parquet.py \
  tests/test_domain_graph_routes.py \
  tests/test_domain_objectives.py

cd /Volumes/external/sources/cppmega
python -m pytest -q \
  tests/test_domain_megatron_sidecars.py \
  tests/test_graph_route_attention_bias_patch.py
```

## Acceptance Criteria

- A detokenized sample containing C++ + CMake + compiler error clearly shows separate delimiter ids rendered back to their logical names by the debug renderer.
- Parquet audit shows nonzero `token_domain_ids`, `token_role_ids`, and `token_diagnostic_edges` for diagnostic rows.
- Build rows are split by kind: cmake/make/ninja/bazel/autoconf/automake/meson/gn/scons/xmake, not one generic bucket.
- Dense GQA smoke shows nonzero graph bias when sidecars exist.
- DSA smoke shows graph-positive blocks rank above hard negatives before top-k.
- Token-only run is impossible unless explicitly opted into.
