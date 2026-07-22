# CASE3 Identity And Repository Graph Contract

This checkout treats a clang cursor identity as semantic data, not as a display
name.

## Symbol Routing

- A normal externally linkable declaration uses the project-bound clang USR.
- A file-local or explicitly file-scoped declaration uses the scoped USR form,
  which binds `project`, repository-relative `file`, and the canonical clang
  signature. This prevents identical internal-linkage USRs from aliasing across
  translation units.
- A cursor without a usable USR uses a signature fallback scoped by
  `project`, `file`, `line`, and `column`. The fallback is a repository-local
  witness; it is not a claim of global semantic uniqueness.
- Qualified names remain display and lookup metadata. An ambiguous qname never
  resolves to an arbitrary overload. Exact symbol references carry the USR or
  scoped fallback key.
- Canonical signature normalization preserves inline-namespace spelling for
  identity comparisons. Public-name normalization may still be used for
  compatibility lookup, but it is not an identity key.

## Repository Prompt Graphs

`PromptGraphContext.from_repository_prompt` accepts only a sealed production
index. The index must carry the production producer/contract, integrity hash,
repository and indexed dependency manifests, trusted identity adapter, and at
least one indexed symbol, chunk, and semantic edge (or a typed external
reference). Empty or synthetic indexes fail before token projection.

The full production validator additionally verifies the supplied repository
checkout and the exact indexer checkout. The index payload integrity hash covers
the repository manifest and indexer-closure receipt, so the artifact's existing
`index_sha256` binds those claims without duplicating them in the artifact
format. The indexer loader binds its in-process module name to the complete
AST-resolved Python dependency closure, rejects modules already loaded from
another checkout, and fails closed if that closure changes after import.

The regression coverage is in
`tests/test_case3_identity_repro_20260719.py`, with root/data parity checks in
the existing prompt-graph contract tests.
