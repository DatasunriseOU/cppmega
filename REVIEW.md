---
phase: CASE1-CASE6-adversarial
reviewed: 2026-07-18T21:00:08Z
depth: deep
files_reviewed: 12
files_reviewed_list:
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/training/objective_mixer.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/megatron/graph_route_attention_bias_patch.py
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/prompt_graph_index.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/data/prompt_graph_index.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/prompt_graph_provenance.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/tools/clang_indexer/index_project.py
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/symbol_identity.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/scripts/eval_domain_routed_codegen.py
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/eval_domain_routed_codegen.py
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/sidecar_manifest_contract.py
  - /Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/upload_verified_sidecar_to_nebius_s3.py
  - /Volumes/external/sources/cppmega_pr_integration_20260718/scripts/data/restore_megatron_bundle_from_nebius_s3.py
findings:
  critical: 11
  warning: 1
  info: 0
  total: 12
status: issues_found
---

# CASE1–CASE6: Independent adversarial review

**Compared:** `origin/main..HEAD` in `/Volumes/external/sources/cppmega_pr_integration_20260718` (`ac9e25e`) and `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718` (`fa2f4d9`)

**Status:** 11 BLOCKER, 1 WARNING.

Focused checks that did complete included the root portable domain suite (`37 passed`), MLX focused suites (`155 passed` and `54 passed`), and a direct root↔MLX identity-parity probe. Full root pytest collection is blocked by a stale/mismatched Megatron environment receipt. Native external-include probing is blocked by missing `libclang`; the same rejection path is established statically and through the available fixture probe. The archive `status=failed` runtime probe is blocked by missing `torch`. Nebius/H200 execution was not started because it was outside the requested local review scope. CASE2 packet flow was traced through route bias and indexer supervision; no separate graph-packet read-but-unused finding was proven.

## Critical Issues

### CR-01: CASE1 matcher can discard a valid graph-positive quota assignment

**Severity:** BLOCKER

**File:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/training/objective_mixer.py:785-875`

**Issue:** `forced_candidate_pairs()` binds every eligible packet/task pair to `first_slot_by_task`. With repeated quota slots, the matcher therefore explores only a narrow set of forced assignments, followed by one greedy bipartite match. A valid assignment can exist while every explored realization fails `required_realized_assignment`, causing a false `ObjectiveQuotaUnsatisfiedError` and dropping a graph-positive training window.

**Reproduction / evidence:** A direct probe using the `ppgp` source-pool pattern, FIM objective, `output_count=2`, and `seed=71` (also `seed=6`) raises `ObjectiveQuotaUnsatisfiedError`. Exhaustive/manual assignment of source indices `(2, 3)` satisfies the same quotas and contains a graph-positive `graph_eligibility["eligible"] == True` item. The aggregate eligibility precheck at lines 709–723 passes, so this is matcher incompleteness rather than an actually impossible pool.

**Suggested fix:** Search assignments over all quota slots, or make the graph predicate part of the matching constraint. Do not reduce repeated task slots to the first slot when enumerating forced candidates.

**Suggested regression test:** Parameterize the minimal `ppgp` fixture over seeds `6` and `71`; assert that exact quotas materialize without exception, that source indices `(2, 3)` remain an admissible solution, and that at least one returned assignment has `graph_eligibility["eligible"] is True`.

### CR-02: CASE2 malformed graph-bias flag silently disables the bias path

**Severity:** BLOCKER

**File:** `/Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/megatron/graph_route_attention_bias_patch.py:84-85,108-113`

**Issue:** `_env_flag()` maps every unrecognized value to `False`. When graph routes are enabled, a typo in `CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS` therefore turns off dense graph bias instead of rejecting the configuration. `_graph_attention_bias_for_layer()` then returns `None`, allowing graph-enabled training/inference to proceed without the required route bias.

**Reproduction / evidence:** With `CPPMEGA_GRAPH_ROUTES_ENABLED=1` and `CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=tru`, `graph_dense_bias_enabled()` returns `False`; no exception or failed preflight is emitted. The function docstring and default establish that dense bias is enabled whenever routes are enabled, while the runtime consumer treats `False` as a valid disabled path.

**Suggested fix:** Parse a closed set of true/false spellings after stripping whitespace and raise on any other value. Validate the same flag through the production preflight/config ingress.

**Suggested regression test:** For `tru`, `0x1`, whitespace-padded garbage, and an empty non-default value, assert `ValueError`; assert only documented false spellings disable the bias and documented true spellings enable it.

### CR-03: CASE3 root/MLX prompt-index integrity contracts drift

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/prompt_graph_index.py:985-1015`; `/Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/prompt_graph_provenance.py:216-217`

**Issue:** The MLX producer emits producer/schema/cache provenance but omits `index_integrity_version`. The root production validator requires that field to equal `"1"`. Artifacts produced by one checkout consequently cannot pass the other checkout's production-index contract even when the schema, project, and payload are otherwise valid.

**Reproduction / evidence:** Building the available MLX repository fixture yields `index.provenance.get("index_integrity_version") is None`. Loading that artifact through the root validator raises `ValueError: production repository index integrity version mismatch`. A separate direct identity-parity probe passes, but it does not validate or repair this producer-receipt mismatch; the two observations establish cross-repo contract drift rather than a fixture-content error.

**Suggested fix:** Define the integrity version in one shared contract and emit/validate it in both producers before cache publication; reject mixed contract versions explicitly.

**Suggested regression test:** Build an MLX fixture, serialize it, and load it through the root validator (and vice versa); assert both directions succeed with the same required provenance fields and fail on a deliberately changed integrity version.

### CR-04: CASE3 prompt-graph cache hash omits imported indexer dependencies

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/prompt_graph_index.py:515-524`; `/Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/data/prompt_graph_index.py:547-555`; `/Volumes/external/sources/cppmega_pr_integration_20260718/tools/clang_indexer/index_project.py:55-74`

**Issue:** The cache fingerprint hashes the repository's C/C++ inputs, compile arguments, `index_project.py` itself, and libclang, but not Python modules imported by `index_project.py` (including identity, source-identity, and build-context helpers). `repository_snapshot()` excludes those `.py` dependencies, so a semantic change in an imported helper can reuse an old index under the same cache key.

**Reproduction / evidence:** Build a cache entry, change/copy an imported helper without changing `index_project.py` or repository C/C++ inputs, and recompute the producer fingerprint: `cache_key` remains unchanged and the loader takes the cached path. The import list at `index_project.py:55-74` contains dependencies absent from both `dependency_manifest` and `indexer_sha256`.

**Suggested fix:** Hash a deterministic, resolved import closure (including each imported local module and its transitive local imports), or make the producer version/hash contract explicitly include all implementation dependencies.

**Suggested regression test:** In a temporary checkout, build once, modify one imported local dependency, build again, and assert `cache_hit is False`, the cache key changes, and the new provenance records the changed dependency hash.

### CR-05: CASE4 libc++/libstdc++ USRs can collapse distinct provider identities

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_pr_integration_20260718/tools/clang_indexer/index_project.py:992-1029,1323-1337`; `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/symbol_identity.py:446-499`

**Issue:** When a clang USR is present, `canonical_symbol_identity()` returns a key composed only of schema, optional project, and USR. `canonical_signature`, provider, and include provenance are ignored. The registry then treats two external records with the same USR as one canonical key/ID, so provider-specific overload/implementation provenance can alias semantic supervision instead of producing a collision.

**Reproduction / evidence:** Registering two external records with the same USR (`c:@N@std@FT@move`) but different providers (`libc++` versus `libstdc++`) produces the identical key `usr:schema=v3␟usr=c:@N@std@FT@move` and identical numeric ID; the registry retains one identity. The returned reference still carries provider fields at lines 1323–1337, but those fields cannot distinguish the already-collapsed key.

**Suggested fix:** Include trusted provider/include provenance (and, where required, a normalized signature) in the external identity namespace, or fail closed when the same USR arrives with conflicting provider/signature claims.

**Suggested regression test:** Feed the registry same-USR/different-provider and same-USR/different-signature records; require distinct canonical keys/IDs or an explicit collision error, and assert lookup/edge targets never alias.

### CR-06: CASE3 real-repository graph generation rejects external include references

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_pr_integration_20260718/cppmega/data/prompt_graph_index.py:96-140,789-795`; `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/prompt_graph_index.py:94-138,755-761`

**Issue:** `_identity_for_cursor()` requires every referenced cursor's `project` to equal the repository's `project_id`. A valid reference into a provider-owned header (or sibling repository) necessarily carries a different provider identity or an external marker, so the producer aborts instead of emitting a typed external target and edge.

**Reproduction / evidence:** The real-repository/provider fixture reaches the branch with `ValueError: CASE 4 v3 symbol reference provenance project does not match the repository project`. The same rejection is present in both checkouts. The native root rerun is additionally blocked by missing `libclang`, but the failing branch is deterministic and independently visible in the source and fixture probe.

**Suggested fix:** Permit only validated external/provider projects, preserve provider/include provenance in the target identity, and distinguish repository-local ownership from external references rather than equating all references with the root project.

**Suggested regression test:** Build a repository TU that includes a provider header and references one provider symbol; assert graph construction succeeds, the target carries provider provenance, and the corresponding `def_use`/call/type edge is emitted.

### CR-07: CASE3 MLX producer accepts a foreign indexer checkout

**Severity:** BLOCKER

**File:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/cppmega_mlx/data/prompt_graph_index.py:251-269`

**Issue:** MLX `_load_indexer()` accepts any existing `indexer_root` and dynamically imports it. There is no same-checkout guard before execution or cache use. The root producer has an explicit guard at its corresponding loader, so the two implementations disagree on the trust boundary.

**Reproduction / evidence:** Constructing the MLX producer with `indexer_root=/Volumes/external/sources/cppmega_pr_integration_20260718` and building an MLX fixture succeeds. The resulting provenance records the foreign `indexer_path`. The root implementation rejects the analogous input with `Cross-checkout cppmega/cppmega.mlx indexer mixing is unsupported`.

**Suggested fix:** Resolve and compare `indexer_root` with the package checkout before import; validate the resolved module origin and reject symlink/foreign roots consistently in both producers.

**Suggested regression test:** Parameterize both producers with each other's checkout and with a symlinked foreign root; assert rejection occurs before module execution, cache creation, or provenance publication.

### CR-08: CASE5 root sidecar oracle ignores generated completion content

**Severity:** BLOCKER

**File:** `/Volumes/external/sources/cppmega_pr_integration_20260718/scripts/eval_domain_routed_codegen.py:1137-1142`

**Issue:** `sidecar_structure_oracle()` deletes `completion` and validates only the frozen prompt sidecars. A model can emit unrelated or empty content and still receive a passing generation result.

**Reproduction / evidence:** For the shipped `ksh_build_sidecar_route` row, both `"garbage"` and `"The route is valid."` return `{"status": "sidecar_structure_passed"}`. The existing test at `tests/test_eval_domain_routed_codegen.py:656-668` explicitly asserts that the unrelated completion passes, locking in the false positive.

**Suggested fix:** Parse and validate generated typed blocks/relations and bind them to the declared shell/diagnostic contract; do not make a static sidecar receipt sufficient for a generation pass.

**Suggested regression test:** Mutate the completion of `ksh_build_sidecar_route` to unrelated text, missing one domain block, and a wrong relation; assert `sidecar_structure_failed` for each while the canonical generated completion passes.

### CR-09: CASE5 root shell/diagnostic checks are semantic under-approximations and lack a cross-domain generation oracle

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_pr_integration_20260718/scripts/eval_domain_routed_codegen.py:75-87,931-1015,1084-1134,1207-1230`

**Issue:** The root shell oracle checks parseability and shell `-n` syntax, not whether the completion answers the prompt's declared command/target. The diagnostic oracle checks that at least one typed edge exists, not the expected severity/message/location/metadata. `ORACLE_KINDS` and dispatch have no `cross_domain_structure` branch, so a root cross-domain case has no generated relation/content oracle and is effectively static-sidecar-only.

**Reproduction / evidence:** For the shipped `shell_build_command` prompt asking for `cmake --build build -j8`, completion `echo ok\n` returns `shell_syntax_passed`. A semantically wrong but parseable compiler diagnostic with a typed diagnostic edge returns `diagnostic_structure_passed` because only edge family/kind/count are enforced. A cross-domain completion cannot select a root `cross_domain_structure` oracle because that kind is absent from the registry and dispatch.

**Suggested fix:** Give shell and diagnostic prompts structured semantic contracts (required tokens/roles/edge kinds/metadata) and add a root cross-domain oracle that parses generated domain blocks and requires the declared relation.

**Suggested regression test:** Add negative completions for wrong shell command, wrong diagnostic severity/path/message, and missing/wrong cross-domain relation; assert all fail while the canonical completions pass. Add a parity test requiring root and MLX to expose the same oracle kinds.

### CR-10: CASE5 MLX frozen prompt text is not bound to frozen token IDs/sidecars

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/eval_domain_routed_codegen.py:1818-1854,1932-1969`

**Issue:** `build_prompt_graph_for_prompt()` performs the required rendered-text/token-ID/sidecar binding, but `sidecar_structure_oracle()` never calls it. The static oracle validates the supplied sidecars independently, and the shipped `ksh_build_sidecar_route` row has no `prompt_graph` specification, so frozen graph data can be paired with mutated prompt text or token IDs.

**Reproduction / evidence:** Retaining the shipped frozen sidecars and a structurally valid generated completion while replacing only `DomainEvalPrompt.prompt` (or token IDs) still allows the sidecar path to pass; the binding helper would reject the same mutation, but it is unreachable from this oracle. This is a distinct false-positive seam even though MLX does parse generated completion blocks.

**Suggested fix:** Require a graph spec for every frozen sidecar case and invoke `build_prompt_graph_for_prompt()` with the production tokenizer before accepting the oracle result; bind the graph artifact receipt to the eval row.

**Suggested regression test:** Mutate only prompt text, then only `prompt_token_ids`, and then one frozen sidecar column; assert the oracle fails each time and accepts the canonical row only after production graph reconstruction matches exactly.

### CR-11: CASE6 sidecar audit receipt is not bound to parquet bytes/content

**Severity:** BLOCKER

**Files:** `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/sidecar_manifest_contract.py:199-227`; `/Volumes/external/sources/cppmega_mlx_pr_integration_20260718/scripts/upload_verified_sidecar_to_nebius_s3.py:325-361`

**Issue:** `validate_audit_receipt()` verifies aggregate green counts and bucket coverage. `_write_manifest()` inventories the current directories but compares parquet selections to the receipt only by file count; it does not require per-file hashes/content from the audit receipt or rerun the semantic parquet audit. A stale green receipt can therefore certify a different byte set.

**Reproduction / evidence:** Replacing an audited `shard.parquet` with arbitrary bytes while preserving the selected bucket's file count, then calling `_write_manifest()` with the old green aggregate receipt, returns `ACCEPTED` and publishes `verified_valid_tokens=936`. The only exact hash check in this path binds the audit receipt file itself (lines 362–371), not each selected parquet file. The downloader then verifies the same inventory/receipt binding, so it does not repair the stale semantic certification.

**Suggested fix:** Make the audit receipt contain a canonical per-file path/size/SHA (and semantic schema/version) and compare it with the upload inventory, or rerun the audit over the exact inventory immediately before manifest publication.

**Suggested regression test:** Audit a directory, replace one selected parquet with same-count arbitrary bytes, and assert manifest creation and download verification both fail unless a fresh audit receipt hashes and semantically validates that exact file.

## Warnings

### WR-01: CASE6 restore reuses a receipt whose status is `failed`

**Severity:** WARNING

**File:** `/Volumes/external/sources/cppmega_pr_integration_20260718/scripts/data/restore_megatron_bundle_from_nebius_s3.py:206-239`

**Issue:** When an archive and receipt already exist, the code checks URI/size/SHA and optional binding fields but never checks `receipt["status"]`. Any matching receipt, including a prior `failed` receipt, is treated as proof and returned as `reused_verified`.

**Reproduction / evidence:** Static control flow shows that lines 211–229 validate only binding fields; line 239 unconditionally returns `reused_verified`. A runtime probe with a matching archive plus `status: "failed"` receipt was not completed because this checkout lacks `torch`, but no status guard exists on the exercised path.

**Suggested fix:** Accept only an explicit allow-list of successful receipt statuses (and validate receipt schema/version); otherwise discard the receipt and perform a fresh verified acquisition.

**Suggested regression test:** Pre-create a byte-matching archive and receipt with `status: "failed"`; assert `_acquire_archive()` rejects/re-verifies it and never returns `reused_verified` from that receipt.

---

_Reviewed: 2026-07-18T21:00:08Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
