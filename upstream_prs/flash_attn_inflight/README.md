# flash-attn-4 in-flight upstream PRs (vendored for GB10/sm_121)

These four open Dao-AILab/flash-attention PRs together fix `flash-attn-4`
install + correctness on GB10 (sm_121a / DGX Spark). Until they merge upstream,
we vendor them as a patch series and apply on top of a pinned FA `main` SHA in
`STACK.lock` / `.github/workflows/build-wheels.yml`.

Pinned base: `Dao-AILab/flash-attention@b995b246` (master tip 2026-04-29).

## Apply order (numeric prefix is significant)

| # | Upstream PR | Author | Scope | Files |
|---|-------------|--------|-------|-------|
| 01 | [#2257](https://github.com/Dao-AILab/flash-attention/pull/2257) | community | `setup.py` adds `sm_121` + `compute_121f` for CUDA 13.0+ | `setup.py` |
| 02 | [#2475](https://github.com/Dao-AILab/flash-attention/pull/2475) | CuriousCaliBoi | tolerate both `nvvm.atomicrmw` Python bindings — fixes fresh `pip install flash-attn-4[cu13]` on GB10 | `flash_attn/cute/utils.py`, `tests/cute/test_utils.py` |
| 03 | [#2484](https://github.com/Dao-AILab/flash-attention/pull/2484) | blake-snc | `FlashAttentionForwardSm120` `__init__` override: pin `arch=sm_80`, disable `is_split_kv`, disable `pack_gqa`. Validated on **SM121a / DGX Spark GB10**. | `flash_attn/cute/flash_fwd_sm120.py` |
| 04 | [#2474](https://github.com/Dao-AILab/flash-attention/pull/2474) | CuriousCaliBoi | Spark fwd+bwd regressions: `softmax_scale_for_scoremod` plumbing, `dQ_single_wg` init, regression test. **Trimmed** — the original PR also touched `flash_fwd_sm120.py` with a smaller subset of the `__init__` override that #2484 supersedes; that hunk is dropped here. | `flash_attn/cute/flash_bwd.py`, `flash_attn/cute/interface.py`, `tests/cute/test_sm120.py` |

PRs 03 and 04 both add `__init__` overrides on `FlashAttentionForwardSm120`;
#2484 is a strict superset so the #2474 hunk on that file is intentionally
dropped (`04_*_trimmed`). When upstream merges either PR, the next FA pin
bump should drop the redundant patch from this series.

## Verification

Last verified (2026-04-29): all four patches `git apply` cleanly in order on
top of `Dao-AILab/flash-attention@b995b246`. Re-run with:

```
cd /tmp && rm -rf fa-test && git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git fa-test
cd fa-test
for p in <repo>/upstream_prs/flash_attn_inflight/0*.patch; do git apply --check "$p" && git apply "$p"; done
```

A whitespace warning on `02_*.patch` ("new blank line at EOF") is benign and
matches the upstream PR.

## Deferred (not auto-applied)

`deferred/` holds five additional sm_120 PRs that conflict with the
correctness set above and/or are stale against `main`. They each add
their own `FlashAttentionForwardSm120.__init__` (overlapping #2484) or
they patch `interface.py` against context that has since moved upstream.

| File | PR | Reason for deferral |
|------|----|--------------------|
| `deferred/pr_2336_split_kv_sm120.patch` | [#2336](https://github.com/Dao-AILab/flash-attention/pull/2336) | duplicates `__init__` arch override; needs hand-merge with #2484 |
| `deferred/pr_2348_paged_kv_sm120.patch` | [#2348](https://github.com/Dao-AILab/flash-attention/pull/2348) | same |
| `deferred/pr_2349_sm120_tma_fwd_ws.patch` | [#2349](https://github.com/Dao-AILab/flash-attention/pull/2349) | stale against `main` (`interface.py:36` rebase) |
| `deferred/pr_2389_block_sparse_sm120.patch` | [#2389](https://github.com/Dao-AILab/flash-attention/pull/2389) | duplicates `__init__` arch override |
| `deferred/pr_2406_sm120_tma_optimized.patch` | [#2406](https://github.com/Dao-AILab/flash-attention/pull/2406) | stale against `main` (`interface.py:54`) |

Pull these in case-by-case once correctness PRs merge and rebases land
upstream — most likely needed only if measured perf/feature gaps justify the
ongoing maintenance cost.

## Why not just wait for upstream merge?

GB10 is the production target right now and #2484/#2474/#2475 are unblocking
fixes for `flash-attn-4` on sm_121 — without them the wheel either fails to
install (#2475), crashes at first dispatch (#2484), or hits varlen layout
errors (#2474). Vendoring lets us measure FA4 vs TE-fused on NAM56R MLA shapes
on GB10 today rather than waiting on upstream review velocity.

When upstream merges any of the four, drop the corresponding numbered patch
from this directory and bump the FA `ref` in `STACK.lock`.
