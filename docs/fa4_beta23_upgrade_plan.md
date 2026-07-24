# FA4 beta23 Image Upgrade Plan

Status: planning only — no Dockerfile / STACK.lock changes in this commit.
Owners: cppmega runtime / image
Scope: `docker/Dockerfile`, `STACK.lock`, `.github/workflows/build-wheels.yml`,
`.github/workflows/build-image.yml`, Nebius H200 image pin.
Companion docs: `docs/fa4_beta23_score_mod_poc.md`, `docs/fa4_score_mod_design.md`,
`docs/changelog.md` (2026-04-10 TileLang/tvm-ffi entry).

---

## 1. Why this upgrade

The chunk-native graph-route attention work
(`cppmega/megatron/fa4_score_mod_adapter.py`,
`docs/fa4_beta23_score_mod_poc.md`) targets the **FA4 beta23** `score_mod`
API surface. The production image still pins **FA4 beta19**. beta23 is the
first pin where the `score_mod` / `aux_tensors` / `block_sparse_tensors`
contract the adapter relies on is stable, and it carries the kernel/compile
fixes the POC was designed against.

beta23 has a hard dependency floor that the current image does not meet:

- FA4 `4.0.0b23` requires `apache-tvm-ffi>=0.1.12,<0.2`.
- The image pins `apache-tvm-ffi==0.1.9`.

So this is not a one-line bump: it forces a coordinated move of the
`apache-tvm-ffi` pin, which in turn collides with the TileLang pin (see §4).

---

## 2. Current versions (as pinned today)

| Component | Pin | Where |
|-----------|-----|-------|
| `flash-attn-4[cu13]` | `4.0.0b19` | `docker/Dockerfile:67`, `STACK.lock:115` (`runtime_pypi.flash_attn_4`) |
| `apache-tvm-ffi` | `0.1.9` | `docker/Dockerfile:58` |
| TileLang (wheel) | `0.1.8+cuda.gitf309d814`, upstream `tile-ai/tilelang@f309d814` | `STACK.lock:88-96`, `build-wheels.yml` matrix `tilelang` |
| torch | `2.13.0+cu132` | `STACK.lock:11`, `Dockerfile:48` |
| CUDA base image | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` | `STACK.lock:5`, `Dockerfile:14` |
| TransformerEngine | `v2.16` (wheel) | `STACK.lock:17-27` |
| Megatron-LM | `core_v0.18.0` (editable clone) | `STACK.lock:119-124`, `Dockerfile:74-79` |
| Nebius image digest | `ghcr.io/datasunriseou/cppmega@sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983` | `scripts/nebius_h200_megatron_cpp_world_sweep.py:50-53` (and curriculum variant) |

Image build/publish path:

- `build-wheels.yml` builds the CUDA wheels (TE, flash_attn FA2, flash_attn_3
  FA3, mamba_ssm, causal_conv1d, fast_hadamard_transform, tilelang,
  qoptim_cuda) on the self-hosted runner and uploads them to a GH Release
  `wheels-<tag>`.
- `build-image.yml` downloads that release into `wheels/`, runs
  `docker/Dockerfile`, and pushes `ghcr.io/datasunriseou/cppmega` with tags
  `<long-sha>`, `<short-sha>`, `latest` (default branch), and the wheel tag.
- FA4 (`flash-attn-4`) is **not** a built wheel — it is installed from
  `pypi.nvidia.com` inside the Dockerfile (`--pre --extra-index-url`), along
  with its `nvidia-cutlass-dsl[cu13]` / `quack-kernels` dependency stack.
- There is **no** `docker/Dockerfile.h200` or any other Dockerfile variant —
  `docker/Dockerfile` is the single image source for all targets (H100/H200,
  B200, GB10). Arch coverage comes from `TORCH_CUDA_ARCH_LIST="9.0;10.0;12.1"`
  and FA4's runtime JIT, not from per-target Dockerfiles.

---

## 3. Target versions

| Component | Target | Constraint source |
|-----------|--------|-------------------|
| `flash-attn-4[cu13]` | `4.0.0b23` | POC API surface (`fa4_beta23_score_mod_poc.md`) |
| `apache-tvm-ffi` | `>=0.1.12,<0.2` (pick the newest patch that TileLang tolerates — see §4) | FA4 beta23 dependency floor |
| TileLang | **TBD — must be re-pinned** (see §4) | tvm-ffi compatibility |

Everything else (torch 2.13.0+cu132, CUDA 13.2.1, TE v2.16, Megatron
core_v0.18.0) is intended to stay fixed for this upgrade. The goal is to
isolate the change to the FA4 + tvm-ffi + TileLang triple.

---

## 4. The TileLang / tvm-ffi conflict (the real risk)

This is the crux of the upgrade and the reason it is not a trivial bump.

Documented history (`docs/changelog.md`, 2026-04-10 "TileLang MIMO Unblocked"):

- TileLang `0.1.8` **crashes on import** with `apache-tvm-ffi==0.1.10`:
  `AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'`.
- Root cause: `apache-tvm-ffi 0.1.10` (PR #480) enforces `__slots__=()` on all
  `Object` subclasses via `_ObjectSlotsMeta`. The fix lives in `apache/tvm`
  PR #18938 (`__slots__ = ("__dict__", "__weakref__")` on `TVMDerivedObject`),
  but TileLang's **vendored TVM fork is pinned to `882a774`, which does NOT
  contain the fix**.
- TileLang main pinned `apache-tvm-ffi<0.1.10` (tile-ai/tilelang#2020) as a
  workaround. cppmega followed: the image and Modal base pin `0.1.9`.

The conflict:

- FA4 beta23 needs `apache-tvm-ffi>=0.1.12`.
- TileLang `0.1.8@f309d814` needs `apache-tvm-ffi<0.1.10`.

These ranges are **disjoint**. We cannot satisfy both with the current
TileLang pin. Options, in order of preference:

1. **Re-pin TileLang to a commit that supports tvm-ffi >=0.1.12.**
   - Investigate whether upstream `tile-ai/tilelang` has bumped its vendored
     TVM fork past `882a774` (i.e. picked up apache/tvm#18938) and relaxed the
     `apache-tvm-ffi<0.1.10` cap from #2020. If a newer commit (or a release
     newer than 0.1.8) imports cleanly under tvm-ffi 0.1.12+, move
     `STACK.lock:88-96` and the `build-wheels.yml` `tilelang` matrix entry to
     that ref and rebuild the wheel.
   - This is the clean path: one tvm-ffi version satisfies both FA4 and
     TileLang.
   - **Risk:** TileLang is used by the MIMO kernels and the DSA TileLang fused
     sparse path (`docs/dsa_ep2_tilelang_sweep_2026_04_12.md`,
     `docs/mamba3_mimo_*`). A TileLang ref bump can change kernel numerics or
     codegen and must be re-validated, not just import-tested.

2. **Patch our TileLang fork to carry apache/tvm#18938 and drop the cap.**
   - If no upstream commit is usable, apply the `__slots__` fix to the
     vendored TVM at `f309d814` and remove the `<0.1.10` constraint, then
     build against tvm-ffi 0.1.12.
   - Higher maintenance cost; we would own a fork delta that must survive
     future TileLang bumps. Prefer option 1.

3. **Decouple: keep TileLang on old tvm-ffi in a separate environment.**
   - Not viable in a single runtime image — both FA4 (`flash_attn.cute`) and
     TileLang are imported in the same process by cppmega. Two tvm-ffi
     versions cannot coexist in one venv. Rejected.

**Decision for this plan:** pursue option 1 first; fall back to option 2 only
if no upstream TileLang ref imports cleanly under tvm-ffi 0.1.12+. The
TileLang ref choice is a prerequisite that must be resolved *before* the
Dockerfile/STACK.lock PR is opened.

---

## 5. Breaking / changed API surface b19 → beta23

These are the deltas the POC and existing FA4 code must be re-verified
against on the actual `4.0.0b23` wheel. Treat each as "assumed until probed":

- `flash_attn.cute.interface.flash_attn_func` keyword contract:
  `score_mod`, `score_mod_bwd`, `aux_tensors`, `aux_scalars`,
  `block_sparse_tensors`, `mask_mod`, `return_lse`. Confirm names, defaults,
  and ordering match what `fa4_score_mod_adapter.py` and
  `fa4_graph_attention.py` pass.
- `score_mod` / `score_mod_bwd` callable signatures (the
  `(score, batch, head, q_idx, kv_idx, seqlen_info, aux_tensors)` shape).
- `score_mod_bwd` **required** whenever `score_mod` is set (assumed unchanged).
- `softcap` and `score_mod` mutually exclusive (assumed unchanged).
- Custom `score_mod` is SM90+ only; SM8x raises `NotImplementedError`
  (assumed unchanged — relevant for any A100 fallback).
- `aux_tensors` compile-key hashing: FA4 hashes the callable plus aux tensor
  *metadata* (shape/dtype/device). Confirm the metadata set did not change in
  a way that would cause per-step recompiles.
- `score` arrives already scaled by `softmax_scale` when `score_mod` is
  present (`flash_attn/cute/utils.py::compute_softmax_scale_log2`). The host
  builders pre-multiply weights by `softmax_scale`; confirm beta23 preserves
  this so numerics stay identical.
- Attention dropout: the POC assumes the `score_mod` path still has **no**
  `dropout_p`. Confirm; the adapter raises on non-zero dropout fail-closed.
- `nvidia-cutlass-dsl[cu13]` / `quack-kernels` versions pulled by beta23 may
  move. The Dockerfile smoke test imports `cutlass`, `cutlass.cute`,
  `cutlass.utils.{LayoutEnum, SmemAllocator}`, and `quack` submodules — these
  are the tripwires for a CUTLASS/quack ABI drift.
- The `flash_attn.cute.block_sparsity` import in the Dockerfile smoke test
  (`Dockerfile:101`) must still resolve under beta23.

Existing in-repo FA4 consumers that must be re-checked against beta23:

- `cppmega/megatron/fa4_score_mod_adapter.py` (chunk-native POC; calls
  `flash_attn_func` with `score_mod`/`aux_tensors`).
- `cppmega/megatron/fa4_graph_attention.py` (CSR + block-sparse backend;
  `CPPMEGA_FA4_GRAPH_ATTENTION=1`).
- `cppmega/megatron/cute_dsl_mimo/fa4_bwd_adapter*.py`, `test_fa4_adapter.py`,
  `benchmark.py` (FA4 bwd adapter surface).
- `tests/test_fa4_score_mod_poc.py`, `tests/test_fa4_graph_attention.py`.

---

## 6. Risk assessment

| Risk | Severity | Notes / mitigation |
|------|----------|--------------------|
| TileLang import crash under tvm-ffi 0.1.12 (the `_NestedLoopCheckVisitor` failure) | High | Hard blocker. Resolved only by §4 option 1 or 2. Must import-test TileLang under the chosen tvm-ffi before image build. |
| TileLang ref bump changes MIMO / DSA kernel numerics | High | Re-run MIMO equivalence + DSA fused-sparse tests after any TileLang ref change (§8). |
| FA4 beta23 API drift vs POC assumptions | Medium | One-shot probe against the real wheel (§8 step 1) before code lands; update `fa4_beta23_score_mod_poc.md` on drift. |
| CUTLASS DSL / quack ABI drift pulled by beta23 | Medium | Dockerfile smoke test imports the full cutlass/quack surface; build fails fast on mismatch. Watch `cutlass.utils.{LayoutEnum,SmemAllocator}` and `quack.{sm90_utils,copy_utils,layout_utils,mx_utils}`. |
| Numerical change in FA4 fwd/bwd between b19 and b23 | Medium | Run fwd/bwd equivalence vs the b19 baseline and vs TE/cuDNN dense path (§8). |
| Nebius training breakage | High (business) | Mitigated by immutable digest pinning — see §9. The pinned digest does not move when `latest` moves. |
| `apache-tvm-ffi` floor `<0.2` excludes a future 0.2 | Low | Pin a specific 0.1.12+ patch, not a floating range, in the Dockerfile (mirror the existing exact-pin style). |
| Image size / layer cache churn | Low | FA4 is a PyPI layer above the wheel `COPY`; bumping it invalidates layers from that point down but reuses the expensive wheel layer via buildx registry cache. |

---

## 7. Step-by-step upgrade procedure

Prerequisite (before opening the PR):

1. Resolve the TileLang ref per §4. Confirm a TileLang commit/release that
   imports cleanly under `apache-tvm-ffi>=0.1.12`. Record the chosen ref and
   the tvm-ffi patch version.

PR contents (the actual upgrade — separate PR, requires H200 validation):

2. `STACK.lock`:
   - `runtime_pypi.flash_attn_4.package` → `flash-attn-4[cu13]==4.0.0b23`.
   - `wheels.tilelang.ref` / `.version` → the new TileLang ref from step 1.
3. `docker/Dockerfile`:
   - Line 58: `apache-tvm-ffi==0.1.9` → the chosen `0.1.12+` patch (exact pin).
   - Line 67: `flash-attn-4[cu13]==4.0.0b19` → `4.0.0b23`.
   - Keep the `--pre --extra-index-url https://pypi.nvidia.com` install form.
   - Leave the smoke-test import block as the build-time ABI tripwire; extend
     it only if beta23 adds a new required import.
4. `.github/workflows/build-wheels.yml`:
   - Update the `tilelang` matrix `ref` to match `STACK.lock` (only the
     TileLang wheel rebuilds; FA4 is not a built wheel).
   - No change needed for FA4 (PyPI-sourced).
5. Run `build-wheels.yml` (TileLang wheel rebuild) → then `build-image.yml`.
   The Dockerfile build's smoke import is the first hard gate.
6. Probe the built image per §8 before tagging/publishing beyond a candidate
   tag.

Ordering note: bump tvm-ffi and FA4 together in one image. There is no
intermediate state where the image has tvm-ffi 0.1.12 but FA4 b19 that we
want to ship — b19 is what forced 0.1.9 in the first place.

---

## 8. Validation plan (post-build, pre-publish)

Run inside the candidate image, on H200 (SM90) at minimum; Blackwell
(SM100/SM120) where the cute path is exercised:

1. **FA4 beta23 API probe (one-shot).** Against the real `4.0.0b23` wheel,
   confirm: `flash_attn_func` kwargs; `score_mod`/`score_mod_bwd` signatures;
   `score_mod_bwd`-required rule; `softcap`⊥`score_mod`; SM8x
   `NotImplementedError`; `aux_tensors` metadata hashing; post-scale `score`;
   absence of `dropout_p` on the `score_mod` path. Record results back into
   `fa4_beta23_score_mod_poc.md`.
2. **Import smoke (already in Dockerfile).** `torch`, `transformer_engine`,
   `flash_attn`, `flash_attn.cute.interface.flash_attn_func`,
   `flash_attn.cute.block_sparsity`, `flash_attn_3`, `mamba_ssm`,
   `causal_conv1d`, `fast_hadamard_transform`, `tilelang`, `qoptim_cuda`,
   `cutlass`/`cutlass.cute`/`cutlass.utils`, `quack` submodules.
3. **TileLang under tvm-ffi 0.1.12.** Import `tilelang` and run the MIMO
   kernel equivalence tests + DSA TileLang fused-sparse tests
   (`docs/dsa_ep2_tilelang_sweep_2026_04_12.md`). This is the regression the
   0.1.9 pin was protecting against.
4. **FA4 fwd/bwd equivalence.** Compare beta23 against the b19 baseline and
   against the TE/cuDNN dense path for `B ∈ {1,2,8}`, `Sq=Sk ∈ {128,512,1024}`,
   bf16 `atol/rtol = 2e-2` plus an fp32-reference `atol=1e-4`. Backward
   gradient equivalence on `q,k,v`.
5. **score_mod adapter tests.** `tests/test_fa4_score_mod_poc.py`,
   `tests/test_fa4_graph_attention.py`, plus the chunk-native equivalence /
   sentinel / fail-closed / compile-key-stability checks from
   `fa4_beta23_score_mod_poc.md` §8.
6. **Compile-key stability.** Two consecutive steps with different edge counts
   but identical high-water marks must not change FA4's compile cache hit
   count.
7. **Training smoke.** A short Megatron run (the existing H200 preflight /
   20-step gate pattern, e.g. `scripts/h200_megatron_preflight.py`,
   `scripts/modal_mamba3_wave32_h200_20step_gate.py`) to confirm end-to-end
   import + step + loss sanity.

Gate: do not promote the candidate to `latest` until 1–7 pass on H200.

---

## 9. Tagging strategy and Nebius impact

**Do not replace the pinned digest. Do not force Nebius onto the new image.**

- Nebius scripts pin an **immutable digest**, not a mutable tag:
  `scripts/nebius_h200_megatron_cpp_world_sweep.py:50-53` and the curriculum
  variant default to
  `ghcr.io/datasunriseou/cppmega@sha256:08c5db73...d983`, and
  `validate_docker_image_digest` (line 77) **rejects mutable tags** — only
  `repo@sha256:<64 hex>` is accepted. Existing Nebius training therefore keeps
  running on exactly the b19/0.1.9 image regardless of what we publish.
- Publishing a new image moves the `<sha>` tags and (on the default branch)
  `latest`. That is safe: nothing in the Nebius path resolves `latest`.

Recommended tagging:

1. Build the candidate and publish under its commit `<sha>` tags plus a
   distinct candidate tag (e.g. `fa4-beta23-candidate` / the wheel release
   tag). **Do not** retag or delete `sha256:08c5db...`.
2. Run §8 validation against the candidate digest.
3. Only after H200 validation passes, allow `latest` to advance (default
   behavior of `build-image.yml` on the default branch). Keep the old digest
   available for rollback.
4. Nebius opt-in is a **separate, explicit step**: bump
   `DEFAULT_DOCKER_IMAGE` in the Nebius scripts to the new validated digest in
   its own PR, after a curriculum/sweep run confirms parity. Until that PR
   lands, Nebius is untouched.

Rollback: because the old digest is immutable and never deleted, rollback is
"point the Nebius script back at `sha256:08c5db...`" — zero rebuild.

---

## 10. Out of scope / follow-ups

- The actual Dockerfile / STACK.lock / build-wheels.yml edits (separate PR,
  gated on §4 TileLang resolution and §8 H200 validation).
- Nebius `DEFAULT_DOCKER_IMAGE` bump (separate PR after validation).
- FA4 dropout support (upstream blocker A in
  `fa4_beta23_score_mod_poc.md` §7.1) — unchanged by this upgrade.
- TE `score_mod` pass-through (upstream blocker B) — unchanged.
- 128×128 `block_sparse_tensors` FLOP-skip phase 2 — independent of the pin
  bump.

---

## 11. References (local)

- `docker/Dockerfile` — FA4 (`:67`), `apache-tvm-ffi` (`:58`), smoke import
  block (`:95-108`).
- `STACK.lock` — `runtime_pypi.flash_attn_4` (`:109-116`), `wheels.tilelang`
  (`:88-96`).
- `.github/workflows/build-wheels.yml` — `tilelang` matrix entry, wheel
  release assembly.
- `.github/workflows/build-image.yml` — image tags (`<sha>`, `latest`, wheel
  tag), buildx registry cache.
- `scripts/nebius_h200_megatron_cpp_world_sweep.py` — immutable digest pin
  (`:50-53`), `validate_docker_image_digest` (`:77`).
- `docs/changelog.md` — 2026-04-10 TileLang/tvm-ffi `<0.1.10` root-cause
  analysis (the `_NestedLoopCheckVisitor` / apache/tvm#18938 / vendored TVM
  `882a774` chain).
- `docs/fa4_beta23_score_mod_poc.md`, `docs/fa4_score_mod_design.md` — FA4
  `score_mod` API surface and assumptions to re-verify on the beta23 wheel.
- `cppmega/megatron/fa4_score_mod_adapter.py`,
  `cppmega/megatron/fa4_graph_attention.py` — in-repo FA4 consumers.

---

## 12. Concrete implementation steps (beta23 + tvm-ffi 0.1.13rc1)

### 12.1 Exact pip install commands

Inside the Dockerfile (or a test venv with CUDA 13.2 + torch 2.13.0+cu132):

```bash
# Step 1: Upgrade apache-tvm-ffi from 0.1.9 to 0.1.13rc1
python -m pip install --no-cache-dir --pre "apache-tvm-ffi==0.1.13rc1"

# Step 2: Install FA4 beta23 (pulls nvidia-cutlass-dsl[cu13] + quack-kernels)
python -m pip install --no-cache-dir --pre \
    --extra-index-url https://pypi.nvidia.com \
    "flash-attn-4[cu13]==4.0.0b23"

# Step 3 (only if TileLang ref is updated — see §12.4):
# Rebuild TileLang wheel against tvm-ffi 0.1.13rc1, then:
python -m pip install --no-deps /wheels/tilelang-*.whl
```

Combined single-layer Dockerfile form (replaces current lines 58 + 65-67):

```dockerfile
RUN python -m pip install --no-cache-dir --pre \
        --extra-index-url https://pypi.nvidia.com \
        "apache-tvm-ffi==0.1.13rc1" \
        "flash-attn-4[cu13]==4.0.0b23"
```

### 12.2 Dockerfile lines to change

| Line | Current | New |
|------|---------|-----|
| 58 | `apache-tvm-ffi==0.1.9 \` | `apache-tvm-ffi==0.1.13rc1 \` |
| 67 | `"flash-attn-4[cu13]==4.0.0b19"` | `"flash-attn-4[cu13]==4.0.0b23"` |

Alternatively, merge lines 58 and 65-67 into a single `RUN` layer (see §12.1
combined form) so pip resolves the tvm-ffi + FA4 + cutlass-dsl dependency graph
atomically. If kept separate, tvm-ffi MUST be installed first (line 58) so that
FA4's `>=0.1.12,<0.2` constraint is already satisfied.

`STACK.lock` corresponding changes:

| Location | Current | New |
|----------|---------|-----|
| `STACK.lock:115` (`runtime_pypi.flash_attn_4.package`) | `"flash-attn-4[cu13]==4.0.0b19"` | `"flash-attn-4[cu13]==4.0.0b23"` |
| `STACK.lock:88-96` (`wheels.tilelang`) | ref `f309d814`, version `0.1.8+cuda.gitf309d814` | **TBD** — see §12.4 |

### 12.3 Post-rebuild test checklist

Run in order inside the candidate image on H200 (SM90):

1. **Import smoke** (already in Dockerfile lines 95-108 — build fails if any
   import breaks):
   ```bash
   python -c "import tilelang; import flash_attn; from flash_attn.cute.interface import flash_attn_func; import cutlass; import quack"
   ```
2. **TileLang + tvm-ffi 0.1.13rc1 import** (the critical regression):
   ```bash
   python -c "import tilelang; print(tilelang.__version__)"
   # Must NOT raise: AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
   ```
3. **FA4 score_mod adapter unit tests**:
   ```bash
   pytest tests/test_fa4_score_mod_poc.py tests/test_fa4_graph_attention.py -x -q
   ```
4. **FA4 CuTe JIT compile test**:
   ```bash
   pytest tests/test_fa4_cute_jit_compile.py -x -q
   ```
5. **FA4 H200 parity test** (fwd/bwd equivalence vs b19 baseline):
   ```bash
   pytest tests/test_fa4_h200_parity.py -x -q
   ```
6. **TileLang MIMO kernel equivalence** (if TileLang ref changed):
   ```bash
   pytest tests/ -k "mimo or tilelang" -x -q
   ```
7. **Training smoke** (20-step Megatron gate):
   ```bash
   python scripts/modal_megatron_train.py --steps 20 --preflight
   ```

### 12.4 TileLang compatibility with tvm-ffi 0.1.13rc1

**Current version** (STACK.lock lines 88-96):

```yaml
tilelang:
  repo: tile-ai/tilelang
  ref: f309d814
  version: 0.1.8+cuda.gitf309d814
  submodules: true
```

**Compatibility verdict: INCOMPATIBLE without a ref bump.**

- TileLang `0.1.8@f309d814` vendors a TVM fork pinned at commit `882a774`.
- That vendored fork does NOT contain apache/tvm#18938 (the `__slots__` fix
  for `_ObjectSlotsMeta` enforcement introduced in tvm-ffi 0.1.10+).
- TileLang upstream pinned `apache-tvm-ffi<0.1.10` (tile-ai/tilelang#2020)
  as a workaround.
- tvm-ffi `0.1.13rc1` is `>=0.1.10`, so the `_NestedLoopCheckVisitor`
  `AttributeError` crash WILL occur on import.

**What needs to change for beta23 support:**

1. Find a TileLang commit (or release >0.1.8) that has:
   - Bumped its vendored TVM fork past `882a774` (includes apache/tvm#18938), AND
   - Relaxed the `apache-tvm-ffi<0.1.10` cap from tile-ai/tilelang#2020.
2. Update `STACK.lock:88-96`:
   ```yaml
   tilelang:
     repo: tile-ai/tilelang
     ref: <NEW_COMPATIBLE_REF>
     version: <NEW_VERSION>
     submodules: true
   ```
3. Update `.github/workflows/build-wheels.yml` `tilelang` matrix entry to match.
4. Rebuild the TileLang wheel and re-run MIMO/DSA kernel equivalence tests.

**If no compatible upstream ref exists:** apply apache/tvm#18938
(`__slots__ = ("__dict__", "__weakref__")` on `TVMDerivedObject`) to the
vendored TVM at `f309d814`, remove the `<0.1.10` cap, and build against
tvm-ffi 0.1.13rc1 (plan §4 option 2).

### 12.5 Risk: TileLang 0.1.8+cuda.gitf309d814 vs tvm-ffi 0.1.13rc1

| Factor | Detail |
|--------|--------|
| **Conflict type** | Hard import-time crash (`AttributeError`), not a subtle numeric drift |
| **Root cause** | tvm-ffi 0.1.10+ enforces `__slots__=()` via `_ObjectSlotsMeta`; TileLang's vendored TVM at `882a774` lacks the `__slots__ = ("__dict__", "__weakref__")` fix from apache/tvm#18938 |
| **Trigger** | `import tilelang` — immediate, deterministic, no workaround short of patching |
| **Severity** | **Blocker** — TileLang is imported in the same process as FA4; cannot coexist with tvm-ffi 0.1.13rc1 at the current ref |
| **Mitigation** | Bump TileLang ref (preferred) or patch vendored TVM (fallback). Must be resolved BEFORE the Dockerfile PR. |
| **Secondary risk** | Even after fixing the import crash, a TileLang ref bump may change MIMO/DSA kernel codegen and numerics — requires full equivalence re-validation |
