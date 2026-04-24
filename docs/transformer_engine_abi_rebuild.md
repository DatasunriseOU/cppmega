# Transformer Engine ABI-rebuild recipe (GB10 + torch nightly)

How to keep Transformer Engine working after a `torch` nightly upgrade. Applies to any cppmega env on GB10 (aarch64) running PyTorch nightly + CUDA 13.x. Last verified 2026-04-24 with `torch 2.13.0.dev20260417+cu132` and TE 2.14.0.

## Why this is needed at all

NVIDIA publishes three TE artifacts per release, but **only two of them are prebuilt wheels** — the third is deliberately ship-as-sdist because it links against libtorch:

| package | artifact on PyPI | ABI-sensitive to torch? |
|---|---|---|
| `transformer_engine` | `py3-none-any` wheel | no (pure Python metapackage) |
| `transformer_engine_cu13` | `manylinux_2_28_<arch>` wheel (~258 MB) | no (links only CUDA runtime / cuDNN) |
| `transformer_engine_torch` | **sdist only** (`.tar.gz`, ~300 KB) | **yes** — C++ shim binding TE core to PyTorch C++ API |

`transformer_engine_torch` is therefore always built locally at install-time against whatever torch is currently installed. If you later upgrade torch and the C++ ABI of libtorch changes, the previously-compiled shim keeps referencing symbols that no longer exist → `ImportError: undefined symbol: ...`.

## Failure signature

```
ImportError: /.../transformer_engine/wheel_lib/transformer_engine_torch.cpython-*.so:
  undefined symbol: _ZNK3c1010TensorImpl15decref_pyobjectEv
```

The demangled symbol (`c10::TensorImpl::decref_pyobject() const`) and its `incref_pyobject` sibling are out-of-line in older torch snapshots and inlined in newer ones. That particular pair broke between torch nightlies of 2026-04-10 and 2026-04-17, but the failure class is generic — any torch C++ ABI change can trigger it. If you see any `undefined symbol: ...` from `transformer_engine_torch.*.so`, apply this recipe regardless of the specific symbol.

Quick diagnosis:

```bash
python -c "import transformer_engine.pytorch"
# If it fails, check which symbols the shim wants vs. libtorch provides:
SO=$(python -c "import transformer_engine, os; p=os.path.dirname(transformer_engine.__file__); \
  import glob; print(glob.glob(p+'/wheel_lib/transformer_engine_torch*.so')[0])")
nm -D --undefined-only "$SO" | grep TensorImpl
```

If any listed undefined symbol is absent from `libtorch_python.so` / `libc10.so` in the current venv, the shim is stale.

## The rebuild

```bash
# 1) Upgrade/align cu13 + metapackage. These are ABI-neutral prebuilts; fast.
pip install --upgrade --no-deps \
    "transformer_engine_cu13==2.14.0" \
    "transformer_engine==2.14.0"

# 2) Rebuild the torch shim from sdist against the currently-installed torch.
NVTE_PYTORCH_FORCE_BUILD=TRUE MAX_JOBS=16 \
pip install --upgrade --no-build-isolation --no-deps --no-cache-dir \
    "transformer_engine_torch==2.14.0"
```

Pin the three versions identically — `transformer_engine_torch`'s `setup.py` computes `install_requires += [f"transformer_engine_cu{major}=={__version__}"]` and will refuse a mismatched core.

### Why each flag is load-bearing

| flag | what it does | what breaks without it |
|---|---|---|
| `NVTE_PYTORCH_FORCE_BUILD=TRUE` | Short-circuits `CachedWheelsCommand.run` in `setup.py` which otherwise tries to download a prebuilt wheel from `github.com/NVIDIA/TransformerEngine/releases/download/v<ver>/...` matching the exact `(cu_major, torch_version, cxx11abi, py_version)` tuple | Always a 404 for torch nightlies → wasted roundtrip, sometimes silent fallback to a version-mismatched wheel |
| `--no-build-isolation` | Makes the build step use the venv's installed `torch`, `setuptools`, etc. | pip spins up an isolated build env, `pyproject.toml`'s `requires = ["torch>=2.1"]` pulls the **latest stable** torch from PyPI, the shim gets linked against *that* libtorch, and the resulting `.so` is still ABI-incompatible with the nightly in your venv — exact same symptom as before |
| `--no-deps` | Don't touch `einops`, `onnx`, `pydantic`, `transformers`, etc. | pip resolver may downgrade/reinstall adjacent heavy ML deps |
| `--no-cache-dir` | Force a fresh build, don't pull a wheel from `~/.cache/pip` that was built against a different torch | Silent "install" of a stale cached wheel, reproducing the ABI mismatch you just tried to fix |
| `MAX_JOBS=16` | Parallelism for the C++ compile | Slow build (serial by default) |

## Verification

```bash
python -c "
import torch, transformer_engine as te, transformer_engine.pytorch as tep
print('torch', torch.__version__, '| TE', te.__version__, '| cuda', torch.version.cuda)
m = tep.Linear(64, 64).cuda()
x = torch.randn(8, 64, device='cuda')
y = m(x); y.sum().backward()
print('OK')
"
```

Plus an ABI sanity check on the fresh `.so`:

```bash
SO=$(python -c "import transformer_engine, glob, os; \
  print(glob.glob(os.path.dirname(transformer_engine.__file__)+'/wheel_lib/transformer_engine_torch*.so')[0])")
stat -c '%y  %n' "$SO"                                # mtime should be "just now"
nm -D --undefined-only "$SO" | grep TensorImpl || echo "clean"
```

`ldd "$SO"` reporting `libc10.so => not found` is **not a problem** — torch C++ extensions resolve libtorch via RPATH at `import torch` time, not via the dynamic loader's default search path.

## What NOT to do

- **Don't try to `pip install` a prebuilt `transformer_engine_torch` wheel** — NVIDIA does not publish one for any version (2.5 → 2.14 confirmed). `pip install --only-binary=:all: transformer_engine_torch==X.Y.Z` will always fail.
- **Don't skip `--no-build-isolation`** — the silent fallback to stable torch in the build env produces a binary that imports successfully at build time but fails at runtime against the nightly. This is the same failure mode, one rebuild deep, and hard to diagnose.
- **Don't attempt to patch the old `.so`** (e.g. `patchelf --replace-needed`) — the missing symbols aren't in *any* torch library, they were inlined into headers. Only a recompile fixes it.
- **Don't pin `transformer_engine_torch` to a different version than the core lib.** The `setup.py` install_requires will refuse it, and even if forced, internal TE contracts between core and shim are version-locked.

## Stack baseline this was verified against

| component | version | notes |
|---|---|---|
| Host | Ubuntu 24.04, aarch64 (GB10 DGX Spark) | driver 595.58.03 |
| CUDA toolkit | 13.2.78 (`/usr/local/cuda` → `cuda-13.2`) | runtime CUDA 13.2 |
| GPU | NVIDIA GB10 (sm_121a / Blackwell consumer) | see `docs/gb10_sm121_hardware.md` |
| torch | 2.13.0.dev20260417+cu132 | nightly |
| TransformerEngine | 2.14.0 (all three packages) | |
| gcc / g++ | 15.2.0 | only matters for C++ shim build |
| cmake / ninja | 4.3.1 / 1.13.0 | |
| cuDNN | 9.20.0 (system `libcudnn9-dev-cuda-13` + `nvidia-cudnn` in venv) | |

The `transformer_engine_torch` shim is pure `.cpp` (no `.cu` — verified by `find csrc -name '*.cu'` on the sdist), so nvcc/GCC-version compatibility isn't on the hot path for *this* package. It is for `transformer_engine_cu13` if you ever need to rebuild *that* from source, but the prebuilt aarch64 wheel covers our case.

## When to expect to run this again

Every time torch nightly rolls through a C++ ABI change. Empirically on this tree: at least once per ~1–2 weeks of torch nightlies. Symptom is always the same `undefined symbol` from `transformer_engine_torch.*.so` at `import transformer_engine.pytorch`. No harm in running the recipe preemptively whenever you bump torch.
