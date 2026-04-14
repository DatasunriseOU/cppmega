# Grounding & Verification Methodology

## Rule

For every upstream PR / issue claim we make in an `upstream_prs/*.md` template
or in an outbound GitHub comment, the ground truth is:

```
gh api repos/{owner}/{repo}/pulls/{N}
gh api repos/{owner}/{repo}/issues/{N}
```

These endpoints are authoritative: PR number, title, author, state
(open/closed/merged), merge commit SHA, and base/head refs all come straight
from GitHub's database. Cite those fields, not paraphrases.

**Never cite MCP engine output (perplexity, exa, tavily, brave) directly to
upstream reviewers.** Always cross-check with `gh api` and cite the API
response instead. MCP search is for discovery only.

## Finding from MCP grounding session 2026-04-14

Full search log: `/Volumes/external/sources/cppmega/.tmp/mcp_grounding_pr_claims.md`

During a multi-engine sweep of the PR template claims, we observed:

- **perplexity_reasoning hallucinated Liger-Kernel "PR #680"** as the fix for
  the FLCE `reduction='none'` backward bug. PR #680 on linkedin/Liger-Kernel
  is unrelated; the real change set lives in issues/PRs we had to retrieve
  via `gh api`.
- **brave / tavily / exa returned no hits** for several newer upstream PRs
  and issues that do exist when queried via `gh api`: Liger-Kernel #968,
  Liger-Kernel #1126, Megatron-LM #3345, Megatron-LM #3226, Megatron-LM #3207,
  Megatron-LM issue #1738. Search-engine index lag for 2026-Q1 GitHub content
  is significant — a PR can be merged and still not surface on any
  general-purpose web search for weeks.
- **tavily rate-limited (432 plan usage)** after a single query; brave
  **rate-limited (429)** on its first four queries. MCP search cannot be
  relied on as a uniform signal.

## Claims with `NO_SIGNAL` from MCP, verified via `gh api`

These are claims where the MCP sweep returned nothing useful but the PR/issue
does exist on GitHub and was verified by direct API call:

- **C1** Liger-Kernel #968 (FLCE `reduction='none'` bug report) and
  #1126 (proposed fix).
- **C2** Megatron-LM #3345 (Hopper native fused linear CE, JungHoyoun,
  `feat/hopper-kernels`).
- **C3** Megatron-LM #3226 (wired LinearCrossEntropyModule to Mamba) and
  #3207 (silent revert).

## Claims VERIFIED via MCP cross-check (and still re-confirmed via `gh api`)

- **C4** TileLang #746 — merge commit, date, author matched between
  perplexity/exa and `gh api`.
- **C5** Apple CCE 25.9.x is Triton-based, not CuTe DSL — confirmed by
  repository source file names (`cce_lse_forward.py` is a Triton kernel).
- **C6** Mamba3 GQA dt bf16 cast bug — upstream issues state-spaces/mamba
  #868 and #886 confirmed by both MCP and `gh api`.

## Guideline for future sessions

MCP search (perplexity, exa, tavily, brave) is useful for **discovery**:
finding adjacent prior art, surfacing related issues, catching version-drift
alerts ("did someone already fix this?"). It is **not** useful for
ground-truth citation. Treat MCP output as a lead, not a source. Anything
that ends up in an `upstream_prs/*.md` file or in a comment posted to a
third-party repository must be verified via `gh api` and the `gh api`
response is what we cite.
