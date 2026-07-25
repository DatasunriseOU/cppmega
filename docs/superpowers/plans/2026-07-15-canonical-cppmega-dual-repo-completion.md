# Canonical cppmega Dual-Repository Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/Volumes/external/sources/cppmega` and `/Volumes/external/sources/cppmega.mlx` the complete, independently runnable canonical repositories, with native source-to-sidecar pipelines in both repos and no production dependency on sibling checkouts or temporary integration worktrees.

**Architecture:** `cppmega` owns a complete C/C++-first source ingestion, Clang indexing, domain parsing, tokenization, parquet packing, Megatron conversion, and CUDA training path. `cppmega.mlx` owns an independent but contract-compatible ingestion path plus its generic visual model constructor/debugger. The repositories share versioned, byte-identical JSON contracts and conformance fixtures, but neither imports code or reads default production paths from the other repository.

**Tech Stack:** Python 3.11+, libclang/Clang, Hugging Face `tokenizers` artifact compatibility, PyArrow parquet, Megatron Core/PyTorch/CUDA, MLX/Metal, pytest, Ruff, self-hosted CUDA runners.

---

## 1. Confirmed Current State

### Git destinations

The directory name does not determine the remote. A Git worktree commits to the branch checked out in the repository's shared `.git` database.

| Worktree | Repository remote | Checked-out branch | Remote branch |
|---|---|---|---|
| `/Volumes/external/sources/cppmega_full_integration` | `git@github.com:DatasunriseOU/cppmega.git` | `codex/full-review-integration` | `origin/codex/full-review-integration` |
| `/Volumes/external/sources/cppmega_mlx_full_integration` | `git@github.com:DatasunriseOU/cppmega_mlx.git` | `codex/full-review-integration` | `origin/codex/full-review-integration` |
| `/Volumes/external/sources/cppmega` | `git@github.com:DatasunriseOU/cppmega.git` | `codex/review-baseline-20260713` | `origin/codex/review-baseline-20260713` |
| `/Volumes/external/sources/cppmega.mlx` | `git@github.com:DatasunriseOU/cppmega_mlx.git` | `codex/review-baseline-20260713` | `origin/codex/review-baseline-20260713` |

Current ancestry is promotion-friendly:

```text
cppmega:     origin/main...codex/full-review-integration = 0 46
cppmega.mlx: origin/main...codex/full-review-integration = 0 124
```

`origin/main` is an ancestor of both integration branches. The integration commits are already pushed, but the canonical worktrees have not been advanced to them.

### Blocking autonomy defects in cppmega

The current root repository is not an independent source processor:

1. `scripts/data/prepare_tokenize_megacpp.py` requires `MEGACPP_NANOCHAT_ROOT` and dispatches to a different repository.
2. `tools/clang_indexer/` does not exist in `cppmega`.
3. `scripts/data/build_macro_routes_megatron_bundle.py` defaults to paths under `../cppmega.mlx`.
4. `scripts/data/mirror_mlx_parquet.py` treats MLX-generated parquet as the source.
5. `scripts/data/verify_tokenizer_contract.py` reaches into the sibling MLX source tree.
6. Several eval, recipe, DSA reference, and FP8 paths import or locate `cppmega_mlx`.
7. The frozen v1 domain contract has Bash/Zsh/Sh/Tcsh but no first-class Ksh or Python domains.

These are acceptance blockers. Promoting branches without fixing them would make the canonical directory newer, but not complete.

---

## 2. Target Repository Contract

### cppmega

The canonical CUDA repository must execute this path without `cppmega.mlx` or `nanochat` present:

```text
source archives / Git repositories
  -> repository discovery and provenance
  -> compile database and build-context detection
  -> local Clang semantic indexer
  -> local build/shell/Python/diagnostic parsers
  -> local tokenizer and token-coordinate materialization
  -> packed parquet with loss/document/graph isolation
  -> parquet audit and immutable publication
  -> Megatron .bin/.idx plus sidecar bundle
  -> dense GQA graph bias and DSA graph supervision
  -> CUDA training, checkpointing, generation and compile evaluation
```

### cppmega.mlx

The canonical MLX repository must execute independently:

```text
source or parquet
  -> selectable language/domain adapter
  -> token and graph sidecars
  -> visual graph/model inspection
  -> MLX model construction/training/debugging
  -> local macOS generation and compile evaluation
```

Its C/C++ profile uses Clang sidecars. Other profiles may replace the language parser while reusing the domain, graph, packing, visualization, and training contracts.

### Cross-repository rule

Allowed:

- identical JSON contracts and reserved-token role assignments;
- identical tiny conformance fixtures;
- explicit artifact transfer through parquet, Megatron bundles, checkpoints, or receipts;
- tests that compare independently produced artifacts.

Forbidden:

- production imports from the sibling package;
- default paths under `../cppmega` or `../cppmega.mlx`;
- requiring a sibling Git checkout;
- silently falling back from a selected native parser/kernel to a sibling implementation;
- assigning different meanings or IDs to the same reserved tokens in the two repositories.

---

### Task 1: Establish Promotion Branches and Protect Canonical Worktrees

**Files:**
- Create: `docs/repository_layout.md` in both repositories
- Create: `scripts/ci/check_repo_independence.py` in both repositories
- Test: `tests/test_repo_independence_contract.py` in both repositories

- [ ] **Step 1: Record worktree and dirty-state receipts**

Run:

```bash
git -C /Volumes/external/sources/cppmega status --short --branch
git -C /Volumes/external/sources/cppmega.mlx status --short --branch
git -C /Volumes/external/sources/cppmega_full_integration status --short --branch
git -C /Volumes/external/sources/cppmega_mlx_full_integration status --short --branch
git -C /Volumes/external/sources/cppmega worktree list --porcelain
git -C /Volumes/external/sources/cppmega.mlx worktree list --porcelain
```

Expected:

- preserve the existing modified `cppmega/.gitignore`;
- preserve all user reports and untracked data;
- classify `cppmega.egg-info/*`, `uv.lock` and `outputs/review/` in the root integration worktree before cleaning any of them;
- do not remove a worktree or reset a file in this step.

- [ ] **Step 2: Create explicit completion branches**

After preserving dirty state, create one branch per repository:

```bash
git -C /Volumes/external/sources/cppmega_full_integration switch -c codex/canonical-native-source-pipeline
git -C /Volumes/external/sources/cppmega_mlx_full_integration switch -c codex/canonical-domain-runtime
git -C /Volumes/external/sources/cppmega_full_integration push -u origin codex/canonical-native-source-pipeline
git -C /Volumes/external/sources/cppmega_mlx_full_integration push -u origin codex/canonical-domain-runtime
```

Expected:

- root commits push only to `DatasunriseOU/cppmega.git`;
- MLX commits push only to `DatasunriseOU/cppmega_mlx.git`;
- no further production commits are made directly on `codex/full-review-integration`.

- [ ] **Step 3: Add the independence checker**

The checker must scan production Python and shell entrypoints and reject:

```python
FORBIDDEN_PRODUCTION_PATTERNS = (
    "../cppmega.mlx",
    "../cppmega",
    "/Volumes/external/sources/cppmega.mlx",
    "/Volumes/external/sources/cppmega",
    "MEGACPP_NANOCHAT_ROOT",
    "from cppmega_mlx",
    "import cppmega_mlx",
)
```

It may allow explicit parity references under `tests/` and historical prose under `docs/archive/`, but it must reject sibling dependencies in `cppmega/`, active `scripts/data/`, active training launchers, and production evaluation code.

- [ ] **Step 4: Lock the checker with tests**

Test cases:

```python
def test_rejects_sibling_import_in_production(tmp_path):
    source = tmp_path / "cppmega" / "bad.py"
    source.parent.mkdir()
    source.write_text("from cppmega_mlx.data import domain_schema\n")
    assert scan_tree(tmp_path).violations


def test_allows_cross_repo_parity_fixture_in_tests(tmp_path):
    source = tmp_path / "tests" / "test_parity_note.py"
    source.parent.mkdir()
    source.write_text('REFERENCE_REPO = "cppmega.mlx"\n')
    assert not scan_tree(tmp_path).violations
```

- [ ] **Step 5: Run and commit**

```bash
python3 -m pytest tests/test_repo_independence_contract.py -q
python3 scripts/ci/check_repo_independence.py
git diff --check
git add docs/repository_layout.md scripts/ci/check_repo_independence.py tests/test_repo_independence_contract.py
git commit -m "ci: enforce independent canonical repositories"
```

Create equivalent commits in each repository.

---

### Task 2: Extend the Existing Domain and Tokenizer Contracts In Place

**Files in cppmega:**
- Modify: `data/domain_schema_v1.json`
- Modify: `data/tokenizer_v2/tokenizer_contract_v1.json`
- Modify: `cppmega/megatron/domain_route_contract.py`
- Modify: `scripts/data/verify_tokenizer_contract.py`
- Test: `tests/test_domain_contract.py`
- Test: `tests/test_tokenizer_contract.py`

**Files in cppmega.mlx:**
- Modify: `cppmega_mlx/data/domain_schema_v1.json`
- Modify: `cppmega_mlx/tokenizer/tokenizer_contract_v1.json`
- Modify: `cppmega_mlx/data/domain_schema.py`
- Modify: `cppmega_mlx/data/tokenizer_contract.py`
- Modify: `cppmega_mlx/tokenizer/cpp_tokenizer.py`
- Test: `tests/test_domain_contract.py`
- Test: `tests/test_tokenizer_contract.py`

- [ ] **Step 1: Prove the reserved slots are free**

Verify that tokenizer IDs 245-248 still map to `<RESERVED_245>` through `<RESERVED_248>` in both tokenizer artifacts and are absent from every current role assignment.

```python
def test_ksh_python_slots_are_reserved(tokenizer_vocab, role_assignments):
    for token_id in range(245, 249):
        assert tokenizer_vocab[token_id] == f"<RESERVED_{token_id}>"
        assert token_id not in role_assignments.values()
```

- [ ] **Step 2: Add Ksh and Python to the current contract**

Use these stable domain IDs:

```json
{
  "KSH": 24,
  "PYTHON": 31
}
```

Use the next currently unused reserved token IDs:

```json
{
  "KSH_START": 245,
  "KSH_END": 246,
  "PYTHON_START": 247,
  "PYTHON_END": 248
}
```

- [ ] **Step 3: Keep old tokenized data readable**

Existing rows do not contain IDs 245-248 with the new semantic roles, so their token sequences remain unchanged. Update contract hashes in newly produced manifests, while consumers continue to accept existing manifests and rows that use the old hash.

- [ ] **Step 4: Add exact cross-repo equality tests**

The current JSON files must be canonicalized and byte-identical after sorted-key serialization:

```python
def test_root_and_mlx_contracts_are_identical():
    assert canonical_json(ROOT_DOMAIN_CONTRACT) == canonical_json(MLX_DOMAIN_CONTRACT)
    assert canonical_json(ROOT_TOKENIZER_CONTRACT) == canonical_json(MLX_TOKENIZER_CONTRACT)
```

- [ ] **Step 5: Run and commit separately**

```bash
python3 -m pytest tests/test_domain_contract.py tests/test_tokenizer_contract.py -q
python3 scripts/data/verify_tokenizer_contract.py \
  --contract data/tokenizer_v2/tokenizer_contract_v1.json \
  --tokenizer data/tokenizer_v2/tokenizer.json \
  --domain-schema data/domain_schema_v1.json
git diff --check
git commit -m "feat(data): add ksh and python domain roles"
```

The MLX repository runs the equivalent tests with `./.venv/bin/python`.

---

### Task 3: Make cppmega Own Its C/C++ Indexer and Tokenizer

**Files to create in cppmega:**
- `tools/clang_indexer/__init__.py`
- `tools/clang_indexer/index_project.py`
- `tools/clang_indexer/process_commits.py`
- `tools/clang_indexer/dedup_store.py`
- `cppmega/data/__init__.py`
- `cppmega/data/build_context.py`
- `cppmega/data/language_info.py`
- `cppmega/data/source_identity.py`
- `cppmega/data/symbol_identity.py`
- `cppmega/data/tokenizer_contract.py`
- `cppmega/tokenizer/__init__.py`
- `cppmega/tokenizer/cpp_tokenizer.py`
- `scripts/data/atomic_publish.py`
- `scripts/data/memory_guard.py`

**Files to modify:**
- `scripts/data/prepare_tokenize_megacpp.py`
- `cppmega/prompt_graph_index.py`
- `pyproject.toml`

**Tests:**
- `tests/test_native_clang_indexer.py`
- `tests/test_native_cpp_tokenizer.py`
- `tests/test_cppmega_source_pipeline_no_sibling.py`

- [ ] **Step 1: Port the current proven indexer into the root namespace**

Port behavior from the MLX integration implementation, but rewrite imports:

```python
from cppmega.data.build_context import (
    detect_build_context,
    find_compile_commands_file,
    load_compile_commands_file,
)
from cppmega.data.symbol_identity import SymbolIdentityRegistry
from cppmega.data.source_identity import source_identity
from scripts.data.atomic_publish import atomic_output_file
from scripts.data.memory_guard import check_memory_limit, start_memory_guard
```

No root production file may import `cppmega_mlx`.

- [ ] **Step 2: Preserve C++ semantic behavior**

The root indexer must support:

- compile_commands-aware parsing;
- C++17/20/23/26 mode selection;
- headers containing templates and inline definitions;
- macro definitions, invocations, conditionals, redefinitions, and include-order edges;
- exact Clang USR identity when available;
- fallback identity containing repository, file, source range, signature, and kind;
- overloaded functions without qname collisions;
- token-level call, type, def-use, AST, include, and macro routes;
- provenance and parse-confidence fields;
- fail-loud parse errors with an explicit partial/heuristic classification.

- [ ] **Step 3: Port the tokenizer implementation**

The root tokenizer loads only artifacts under `data/tokenizer_v2/`. Its public interface is:

```python
class CppMegaTokenizer:
    contract_version: int
    vocab_size: int

    def encode(self, text: str, *, trusted_control_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def token_offsets(self, text: str) -> list[tuple[int, int]]: ...
```

Raw source containing literal control-token spellings must be escaped or rejected according to the current contract. Trusted pipeline code inserts control IDs directly.

- [ ] **Step 4: Lock line-comment EOL preservation**

Add the exact round-trip regression:

```python
def test_line_comment_keeps_next_statement_on_new_line(tokenizer):
    source = "// explanation\nint value = 7;\n"
    ids = tokenizer.encode(source)
    assert tokenizer.decode(ids) == source
    comment_end = ids.index(COMMENT_END_ID)
    assert ids[comment_end + 1] == NL_ID
```

- [ ] **Step 5: Replace external nanochat dispatch**

`scripts/data/prepare_tokenize_megacpp.py` must invoke the local indexer, local tokenizer, local parquet writer, and local packer. Delete the `MEGACPP_NANOCHAT_ROOT` requirement. Do not retain an automatic legacy fallback.

- [ ] **Step 6: Run an isolated checkout smoke**

Create a temporary archive containing only the root repository and run:

```bash
tmp_dir="$(mktemp -d)"
git archive HEAD | tar -x -C "$tmp_dir"
cd "$tmp_dir"
python3 scripts/data/prepare_tokenize_megacpp.py \
  --project-dir tests/fixtures/native_source_repo \
  --output-dir "$tmp_dir/out" \
  --buckets 1024
test -f "$tmp_dir/out/1024/part-00000.parquet"
```

The test environment must not contain a sibling `cppmega.mlx` or `nanochat` checkout.

- [ ] **Step 7: Commit**

```bash
python3 -m pytest \
  tests/test_native_clang_indexer.py \
  tests/test_native_cpp_tokenizer.py \
  tests/test_cppmega_source_pipeline_no_sibling.py -q
git diff --check
git commit -m "feat(data): make cppmega source indexing self-contained"
```

---

### Task 4: Add Native Build, Shell, Python, and Diagnostic Parsers to cppmega

**Files to create:**
- `cppmega/data/domain_schema.py`
- `cppmega/data/domain_packet.py`
- `cppmega/data/domain_ingestion.py`
- `cppmega/data/build_parsers/base.py`
- `cppmega/data/build_parsers/cmake.py`
- `cppmega/data/build_parsers/make.py`
- `cppmega/data/build_parsers/ninja.py`
- `cppmega/data/build_parsers/bazel.py`
- `cppmega/data/build_parsers/autotools.py`
- `cppmega/data/build_parsers/meson.py`
- `cppmega/data/build_parsers/gn.py`
- `cppmega/data/build_parsers/scons.py`
- `cppmega/data/build_parsers/xmake.py`
- `cppmega/data/shell_parsers/base.py`
- `cppmega/data/shell_parsers/bash.py`
- `cppmega/data/shell_parsers/posix.py`
- `cppmega/data/shell_parsers/zsh.py`
- `cppmega/data/shell_parsers/ksh.py`
- `cppmega/data/shell_parsers/tcsh.py`
- `cppmega/data/python_parser.py`
- `cppmega/data/diagnostic_parsers/base.py`
- `cppmega/data/diagnostic_parsers/gcc_clang.py`
- `cppmega/data/diagnostic_parsers/msvc.py`
- `cppmega/data/diagnostic_parsers/linker.py`
- `cppmega/data/diagnostic_parsers/build.py`
- `cppmega/data/diagnostic_parsers/runtime.py`

**Tests:**
- `tests/test_build_domain_parsers.py`
- `tests/test_shell_domain_parsers.py`
- `tests/test_python_domain_parser.py`
- `tests/test_diagnostic_domain_parsers.py`

- [ ] **Step 1: Use one typed parser result**

```python
@dataclass(frozen=True)
class ParsedDomainDocument:
    domain_kind: DomainKind
    text: str
    spans: tuple[DomainSpan, ...]
    edges: tuple[DomainEdge, ...]
    confidence: ConfidenceKind
    metadata: Mapping[str, JsonValue]
```

Every parser must return this type. Malformed input returns `RAW` or `PARTIAL` confidence with a reason; it must not pretend to be exact.

- [ ] **Step 2: Keep build systems distinct**

Dispatch independently by filename, extension, shebang, and explicit caller intent:

```python
PARSER_KEYS = {
    "CMakeLists.txt": "cmake",
    "Makefile": "make",
    "build.ninja": "ninja",
    "BUILD.bazel": "bazel",
    "configure.ac": "autoconf",
    "Makefile.am": "automake",
    "meson.build": "meson",
    "BUILD.gn": "gn",
    "SConstruct": "scons",
    "xmake.lua": "xmake",
}
```

Do not combine all build files under a generic `MAKE` domain.

- [ ] **Step 3: Keep shell dialects distinct**

The dispatch order for shebangs is `tcsh/csh`, `zsh`, `ksh`, `bash`, then POSIX `sh`. Ksh receives DomainKind `KSH` and its own START/END tokens.

- [ ] **Step 4: Parse Python with stdlib ast and tokenize**

Python sidecars include:

- AST parent edges;
- definition/use edges for local scopes;
- import edges;
- call-name candidate edges;
- decorator, async, class, function, and comprehension roles;
- docstring spans separated from ordinary string literals.

Syntax-invalid Python returns `PARTIAL` with tokenizer-level spans rather than silently using C++ or raw-shell parsing.

- [ ] **Step 5: Treat diagnostics as observations, not comments**

Diagnostics remain inside dedicated diagnostic delimiters. Emit:

- tool and platform;
- severity and diagnostic code;
- primary file/line/column;
- note and fix-it relationships;
- undefined linker symbol and candidate-definition relationships;
- build command and target relationships;
- sanitizer stack-frame and test-failure locations;
- cross-domain edges from diagnostics to source/build/symbol tokens when resolvable.

No diagnostic is encoded as a docstring or ordinary source comment.

- [ ] **Step 6: Run and commit**

```bash
python3 -m pytest \
  tests/test_build_domain_parsers.py \
  tests/test_shell_domain_parsers.py \
  tests/test_python_domain_parser.py \
  tests/test_diagnostic_domain_parsers.py -q
git diff --check
git commit -m "feat(data): add native domain and diagnostic parsers"
```

---

### Task 5: Build the Native cppmega Source-to-Parquet Conveyor

**Files to create or port:**
- `scripts/data/source_conveyor.py`
- `scripts/data/extract_git_history.py`
- `scripts/data/clang_enriched_to_parquet.py`
- `scripts/data/materialize_tokenized_enriched_parquet.py`
- `scripts/data/pack_enriched_rows.py`
- `scripts/data/audit_sidecar_parquet.py`
- `scripts/data/token_budget.py`
- `cppmega/data/packed_rows_schema.py`
- `cppmega/data/tokenized_enriched_schema.py`

**Files to modify:**
- `scripts/data/prepare_data.sh`
- `scripts/data/prepare_tokenize_megacpp.py`
- `scripts/data/build_macro_routes_megatron_bundle.py`
- `scripts/data/build_dataset_manifest.py`
- `scripts/data/mirror_mlx_parquet.py`

**Tests:**
- `tests/test_source_conveyor.py`
- `tests/test_packed_document_isolation.py`
- `tests/test_source_to_parquet_e2e.py`
- `tests/test_parquet_publication_atomicity.py`

- [ ] **Step 1: Define explicit stages**

```text
discover -> extract -> parse -> tokenize -> pack -> audit -> publish
```

Each stage writes a local checkpoint and immutable input fingerprint. A stage is marked complete only after its output has passed validation and been atomically promoted.

- [ ] **Step 2: Use commit-on-success dedup**

Workers write reservations and dedup claims to stage-local state. The parent process promotes claims to the global dedup database only after parquet append and validation succeed.

- [ ] **Step 3: Bound memory and submission queues**

The conveyor exposes:

```text
--repo-workers
--max-active-repos
--range-submit-window
--worker-rss-limit-gib
--parent-rss-limit-gib
```

Repository ranges are submitted through a bounded queue. No worker may accumulate an unbounded list of source documents or Arrow tables.

- [ ] **Step 4: Produce all configured buckets directly**

One source parse may feed independent packers for `1024`, `2048`, `4096`, `8192`, and `16384`. Each bucket receives distinct document selections according to the configured repetition cap; a function is not blindly copied into every sample.

- [ ] **Step 5: Preserve packed-document isolation**

For every packed row:

- `doc_ids` identifies each logical document;
- `loss_mask` is zero across document transitions;
- graph edges never cross documents unless explicitly represented as repository edges;
- token coordinates and sidecars remain length-aligned;
- trained token count equals the sum of the validated loss mask.

- [ ] **Step 6: Remove sibling defaults**

`build_macro_routes_megatron_bundle.py` defaults to paths under the root repository's configured data root. `mirror_mlx_parquet.py` becomes an explicit migration utility and is not called by `prepare_data.sh`.

- [ ] **Step 7: End-to-end fixture**

Build a fixture repository containing:

- C++ source and header;
- overloaded functions;
- a template defined in a header;
- macros and conditional compilation;
- CMake, Make, Ninja, Bazel and Autotools files;
- Bash, Ksh and Tcsh scripts;
- Python;
- compiler and linker logs;
- one commit and one PR discussion.

Run the complete pipeline and assert a valid parquet shard for each requested bucket.

- [ ] **Step 8: Commit**

```bash
python3 -m pytest \
  tests/test_source_conveyor.py \
  tests/test_packed_document_isolation.py \
  tests/test_source_to_parquet_e2e.py \
  tests/test_parquet_publication_atomicity.py -q
git diff --check
git commit -m "feat(data): add native source-to-parquet conveyor"
```

---

### Task 6: Bind Domain Sidecars to Megatron GQA and DSA

**Files to modify:**
- `cppmega/megatron/domain_route_contract.py`
- `cppmega/megatron/structure_dataset_patch.py`
- `cppmega/megatron/graph_route_attention_bias_patch.py`
- `cppmega/megatron/dsa_indexer_fused_patch.py`
- `cppmega/megatron/graph_objective_loss.py`
- `cppmega/megatron/graph_recipe.py`
- `cppmega/megatron/objective_contract.py`
- `scripts/data_prep_parquet_to_megatron.py`

**Tests:**
- `tests/test_domain_megatron_sidecars.py`
- `tests/test_graph_route_attention_bias_patch.py`
- `tests/test_dsa_indexer_fused_patch.py`
- `tests/test_stage1_graph_domain_production.py`

- [ ] **Step 1: Keep GQA dense**

The dense GQA path continues to attend to all causal tokens. Graph routes add an explicit bias:

```python
attention_logits = causal_attention_logits + beta_graph * graph_bias
```

The graph bias must not convert GQA to sparse attention and must preserve the causal/document masks.

- [ ] **Step 2: Add graph score before DSA top-k**

```python
index_score = neural_index_score + beta_graph * graph_candidate_score
topk_indices = fused_topk(index_score, k)
```

The graph score contains typed contributions from def-use, call, type, include, macro, build, shell, diagnostic, and cross-domain routes.

- [ ] **Step 3: Keep graph supervision independent**

Dense graph bias, DSA graph candidate scoring, and auxiliary graph losses are independently configurable. Disabling the auxiliary loss must not disable inference-time graph routing.

- [ ] **Step 4: Add domain embeddings and route-type embeddings**

Token-aligned `domain_id`, `role_id`, and `confidence_id` feed bounded embeddings. Edge-type IDs feed graph scoring, not the token vocabulary.

- [ ] **Step 5: Run and commit**

```bash
python3 -m pytest \
  tests/test_domain_megatron_sidecars.py \
  tests/test_graph_route_attention_bias_patch.py \
  tests/test_dsa_indexer_fused_patch.py \
  tests/test_stage1_graph_domain_production.py -q
git diff --check
git commit -m "feat(megatron): consume native graph routes"
```

---

### Task 7: Remove Remaining cppmega Runtime Dependencies on MLX

**Files to modify:**
- `cppmega/megatron/fp8_activations.py`
- `cppmega/megatron/dsa_splitk_indexer_loss.py`
- `cppmega/recipes/run_profiles.py`
- `scripts/nebius_h200_megatron_cpp_generation_eval.py`
- `scripts/nebius_h200_megatron_cpp_world_curriculum.py`
- `scripts/nebius_h200_megatron_cpp_world_sweep.py`
- `scripts/data/verify_tokenizer_contract.py`
- `scripts/data/build_dataset_manifest.py`

**Tests:**
- `tests/test_cppmega_no_mlx_runtime_import.py`
- `tests/test_native_cuda_reference_paths.py`

- [ ] **Step 1: Replace MLX kernel references**

CUDA production code uses local Torch/CUDA/TileLang implementations or an explicit root reference implementation. MLX imports may remain only in parity tests.

- [ ] **Step 2: Replace sibling paths with required arguments or root defaults**

All remote/eval launchers derive data, tokenizer, indexer, and bundle paths from:

```text
--data-root
--tokenizer-dir
--clang-indexer-root
--bundle-root
```

Defaults resolve inside `cppmega` or `CPPMEGA_DATA_ROOT`, never the sibling repository.

- [ ] **Step 3: Prove import isolation**

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /Volumes/external/sources/.venvs/cppmega.source/bin/python -I -c \
  "import cppmega; import cppmega.megatron.domain_route_contract"
python3 scripts/ci/check_repo_independence.py
```

- [ ] **Step 4: Commit**

```bash
python3 -m pytest \
  tests/test_cppmega_no_mlx_runtime_import.py \
  tests/test_native_cuda_reference_paths.py -q
git diff --check
git commit -m "refactor: remove cppmega runtime dependency on mlx"
```

---

### Task 8: Complete the MLX Domain Runtime and Visual Inspector

**Files to modify in cppmega.mlx:**
- `cppmega_mlx/data/domain_schema.py`
- `cppmega_mlx/data/domain_ingestion.py`
- `cppmega_mlx/data/shell_parsers/base.py`
- `cppmega_mlx/data/shell_parsers/__init__.py`
- `cppmega_mlx/inference/side_channels.py`
- `cppmega_mlx/nn/code_graph_routes.py`
- `cppmega_mlx/nn/domain_embedding.py`
- `cppmega_v4/jsonrpc/data_methods.py`
- `vbgui/src/components/DataInspector.tsx`
- Create: `vbgui/src/components/DomainRouteInspector.tsx`

**Files to create:**
- `cppmega_mlx/data/shell_parsers/ksh.py`
- `cppmega_mlx/data/python_parser.py`

**Tests:**
- `tests/test_python_domain_parser.py`
- `tests/test_shell_domain_parsers.py`
- `tests/test_domain_route_visual_payload.py`
- `vbgui/e2e/domain-route-inspector.spec.ts`

- [ ] **Step 1: Add the same Ksh and Python contracts**

Use DomainKind 24/31 and reserved IDs 245-248 exactly as in root.

- [ ] **Step 2: Preserve generic language extensibility**

Introduce a profile registry:

```python
@dataclass(frozen=True)
class LanguageProfile:
    name: str
    source_parser: SourceParser
    domain_adapters: Mapping[str, DomainParser]
    graph_builder: GraphBuilder
    formatter: Formatter | None
```

The C/C++ profile selects Clang and clang-format. Python selects the Python parser and optional formatter adapter. Other languages may provide their own source parser without changing model construction.

- [ ] **Step 3: Expose domain routes in JSON-RPC**

The data preview response includes:

```typescript
type DomainRoutePreview = {
  domainCounts: Record<string, number>;
  roleCounts: Record<string, number>;
  confidenceCounts: Record<string, number>;
  edgeCounts: Record<string, number>;
  spans: DomainSpanPreview[];
  edges: DomainEdgePreview[];
};
```

- [ ] **Step 4: Add the visual inspector**

The inspector must display:

- domain-colored token spans;
- edge-family filters;
- graph source/target highlighting;
- confidence and parser origin;
- document boundaries and loss-mask gaps;
- no nested decorative cards or overlapping labels.

- [ ] **Step 5: Verify locally on macOS**

```bash
./.venv/bin/python -m pytest \
  tests/test_python_domain_parser.py \
  tests/test_shell_domain_parsers.py \
  tests/test_domain_route_visual_payload.py -q
cd vbgui
npm test
npx playwright test e2e/domain-route-inspector.spec.ts
```

Capture desktop `1280x900` and mobile `390x844` screenshots and verify no horizontal overflow.

- [ ] **Step 6: Commit**

```bash
git diff --check
git commit -m "feat(gui): inspect versioned language and domain routes"
```

---

### Task 9: Add Cross-Repository Conformance Fixtures

**Files in both repositories:**
- Create: `tests/fixtures/domain_contract/`
- Create: `scripts/ci/export_domain_conformance_receipt.py`
- Create: `tests/test_cross_repo_domain_conformance.py`

- [ ] **Step 1: Use the same tiny fixture corpus**

The fixture contains one file per supported domain plus C/C++ code and diagnostics. Its expected output records:

- token IDs;
- domain, role, and confidence IDs;
- document IDs and loss mask;
- typed edge lists;
- contract hashes;
- source and symbol identities.

- [ ] **Step 2: Generate independently**

Run the root producer and MLX producer separately. Do not import either package into the other process.

- [ ] **Step 3: Compare canonical receipts**

```python
assert root_receipt["contract_hashes"] == mlx_receipt["contract_hashes"]
assert root_receipt["fixture_hash"] == mlx_receipt["fixture_hash"]
assert root_receipt["canonical_sidecars"] == mlx_receipt["canonical_sidecars"]
```

- [ ] **Step 4: Add self-hosted CI lanes**

The root repository lane runs without MLX installed. The MLX lane runs without root installed. A third conformance lane downloads only the two receipt JSON files and compares them.

- [ ] **Step 5: Commit**

```bash
python3 -m pytest tests/test_cross_repo_domain_conformance.py -q
git diff --check
git commit -m "test: prove cross-repo domain conformance"
```

---

### Task 10: Full Verification and Canonical Promotion

**Files:**
- Create: `outputs/review/canonical_promotion_receipt_20260715.json`
- Create: `docs/canonical_promotion_20260715.md`

- [ ] **Step 1: Root clean-checkout verification**

Run from an isolated checkout with no sibling repositories:

```bash
python3 -m pytest -q
python3 -m compileall -q cppmega scripts tools
python3 -m ruff check cppmega scripts tools tests
python3 scripts/ci/check_repo_independence.py
git diff --check
```

- [ ] **Step 2: Root real-data mini pipeline**

Process at least one real repository into `1024`, `2048`, and `4096` parquet, audit every row, build a Megatron bundle, restore it into a fresh directory, and compare all manifest hashes.

- [ ] **Step 3: Root CUDA proof**

On a repository-owned CUDA runner:

- run Megatron import and fused-extension preflight;
- load the newly generated sidecar bundle;
- execute a dense GQA + graph-bias smoke;
- execute DSA neural-plus-graph scoring before fused top-k;
- train at least 20 steps;
- save and restore a checkpoint;
- generate C/C++ completions and pass them through clang-format and the compile gate.

No MLX package or sibling checkout may be installed in this lane.

- [ ] **Step 4: MLX clean-checkout verification**

On local macOS:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check cppmega_mlx cppmega_v4 scripts tests
./.venv/bin/python scripts/run_self_hosted_ci.py --local
git diff --check
```

Run local checkpoint conversion, generation, sidecar inference, clang-format, and compile evaluation. MLX evals remain local macOS unless explicitly requested otherwise.

- [ ] **Step 5: Push completion branches**

```bash
git -C /Volumes/external/sources/cppmega_full_integration push origin codex/canonical-native-source-pipeline
git -C /Volumes/external/sources/cppmega_mlx_full_integration push origin codex/canonical-domain-runtime
```

- [ ] **Step 6: Promote into canonical branches**

After all receipts pass, advance each canonical repository independently. Do not copy files between repositories.

Root:

```bash
git -C /Volumes/external/sources/cppmega fetch origin
git -C /Volumes/external/sources/cppmega merge --ff-only origin/codex/canonical-native-source-pipeline
git -C /Volumes/external/sources/cppmega push origin HEAD
```

MLX:

```bash
git -C /Volumes/external/sources/cppmega.mlx fetch origin
git -C /Volumes/external/sources/cppmega.mlx merge --ff-only origin/codex/canonical-domain-runtime
git -C /Volumes/external/sources/cppmega.mlx push origin HEAD
```

If the canonical branches are not direct ancestors because user changes were committed meanwhile, perform a normal non-destructive merge and rerun the full verification.

- [ ] **Step 7: Retire temporary worktrees**

Only after canonical paths show the promoted commits and tests pass:

```bash
git -C /Volumes/external/sources/cppmega worktree remove /Volumes/external/sources/cppmega_full_integration
git -C /Volumes/external/sources/cppmega.mlx worktree remove /Volumes/external/sources/cppmega_mlx_full_integration
git -C /Volumes/external/sources/cppmega worktree prune
git -C /Volumes/external/sources/cppmega.mlx worktree prune
```

Do not delete remote integration branches until their commits are reachable from the promoted canonical branches and the promotion receipt records that fact.

---

## 3. Required Acceptance Criteria

The work is complete only when all conditions are true:

1. `cppmega` processes raw C/C++ repositories, commits, build files, shell files, Python, and diagnostics without `cppmega.mlx` or `nanochat`.
2. `cppmega.mlx` processes its supported profiles without `cppmega`.
3. Both repositories read existing tokenized data and use the extended current contract for newly generated data.
4. Ksh and Python have distinct domains and tokenizer delimiters.
5. Build systems and shell dialects are not collapsed into generic domains.
6. Diagnostics are separate observations with typed links to source/build tokens.
7. Packed rows preserve document boundaries, loss masking, and graph isolation.
8. Dense GQA sees all causal tokens while receiving graph bias.
9. DSA scores `I_neural + beta * S_graph` before fused top-k.
10. Root production code has no sibling MLX imports or paths.
11. MLX visual tooling exposes domain spans, graph edges, confidence, and loss boundaries.
12. Clean-checkout local tests, self-hosted CUDA smoke, local macOS MLX evals, and cross-repo conformance all pass.
13. The canonical paths contain the promoted commits.
14. The `*_full_integration` worktrees are no longer operational sources of truth.

---

## 4. Commit Sequence

Use small repository-specific commits:

```text
cppmega
  ci: enforce independent canonical repositories
  feat(data): add ksh and python domain roles
  feat(data): make cppmega source indexing self-contained
  feat(data): add native domain and diagnostic parsers
  feat(data): add native source-to-parquet conveyor
  feat(megatron): consume native graph routes
  refactor: remove cppmega runtime dependency on mlx
  test: prove cross-repo domain conformance
  docs: record canonical promotion receipt

cppmega.mlx
  ci: enforce independent canonical repositories
  feat(data): add ksh and python domain roles
  feat(data): add ksh and python domain adapters
  feat(gui): inspect versioned language and domain routes
  test: prove cross-repo domain conformance
  docs: record canonical promotion receipt
```

No commit may contain parquet data, checkpoints, `egg-info` output, unreviewed `uv.lock` churn, or generated review artifacts.
