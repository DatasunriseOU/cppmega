# CppMega Environment Contract

This checkout uses a source checkout of Megatron-LM. The repository-local
`.venv` is not an owned environment: on the current machine it is a symlink
to `/Volumes/external/sources/nanochat/.venv`. Do not install packages, run
`uv sync`, or change processes through that path.

The small stdlib-only tool in
[`scripts/env/cppmega_env.py`](../scripts/env/cppmega_env.py) provides two
operations:

* `bootstrap` creates a dedicated sibling venv, writes a source-only `.pth`
  file, and records the source commit in `cppmega-environment.json`.
* `verify` checks the target interpreter, source origin, exact Megatron commit,
  clean source state, package requirements, import origins, and `pip check`.

The tool never runs `pip install` or `uv sync`. Package provisioning is an
explicit operation against the dedicated target environment or the pinned
CUDA image; it is never inferred from the shared venv.

## Reproduced Drift

The following observations were made on 2026-07-16 without modifying either
the shared environment or the Megatron checkout:

| Check | Result |
| --- | --- |
| `.venv` target | symlink to `/Volumes/external/sources/nanochat/.venv` |
| `python` from the checkout shell | not found; only `.venv/bin/python` exists |
| shared interpreter | CPython 3.13.12, prefix `/Volumes/external/sources/nanochat/.venv` |
| installed `megatron-core` | missing |
| `import megatron` without source path | `ModuleNotFoundError` |
| source import with explicit `PYTHONPATH` | succeeds, but only as an uninstalled source import and emits optional-dependency fallbacks |
| shared `PYTHONPATH` | TileLang and MLX source directories are inherited |
| `pip check` | five conflicts: `datasets/fsspec`, `repowise/rich`, `modal/protobuf`, `litellm/click`, and `wandb/protobuf` |
| legacy Megatron checkout | `/Volumes/external/sources/Megatron-LM` may be dirty and is not the local source receipt |
| clean Megatron checkout | `/Volumes/external/sources/Megatron-LM-core_v0.18.0` at `ba7b5ebce12af60627a80985792a1449ce45f46c` |
| `STACK.lock` source ref | `core_v0.18.0` resolves to `ba7b5ebce12af60627a80985792a1449ce45f46c` |

The legacy source still reports a compatible version, so a version-only check
would miss commit and dirty-tree drift. The passing local receipt therefore
binds the exact clean commit, not only the package version.

Representative read-only reproductions:

```bash
.venv/bin/python -c 'import megatron'
# ModuleNotFoundError: No module named 'megatron'

PYTHONPATH=/Volumes/external/sources/Megatron-LM:/Volumes/external/sources/cppmega \
  .venv/bin/python -c 'import megatron.core.transformer.transformer_layer'
# imports the sibling source tree, but does not prove an isolated environment

.venv/bin/python -m pip check
# exits non-zero with the five conflicts listed above
```

The tool reproduces the intended refusal without touching `.venv`:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  .venv/bin/python scripts/env/cppmega_env.py verify \
  --env .venv \
  --megatron-root /Volumes/external/sources/Megatron-LM \
  --megatron-ref e40feed4a060a84cd4cd1e5096316cc487014c87 \
  --profile source
```

Expected result: non-zero exit, with `target isolation` reporting that the
path resolves through the shared `cppmega .venv` symlink. A non-empty inherited
`PYTHONPATH` is also a hard failure.

## Bootstrap Contract

Use a target outside the checkout. The default is `../cppmega-venv`; an
explicit target is clearer in automation:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /opt/homebrew/opt/python@3.13/bin/python3.13 \
  scripts/env/cppmega_env.py bootstrap \
  --env /Volumes/external/sources/.venvs/cppmega.source \
  --megatron-root /Volumes/external/sources/Megatron-LM-core_v0.18.0 \
  --megatron-ref <clean-megatron-commit> \
  --profile locked \
  --skip-verify
```

`bootstrap` fails before creating a target when the source ref is wrong or the
source tree is dirty. It also refuses a target that is a symlink, lives inside
the checkout, enables `include-system-site-packages`, or is already owned by a
different manifest. `--allow-dirty-source` is available only for explicit
local diagnosis; it records the dirty state and is not a reproducibility
receipt.

The bootstrap writes only to the dedicated target:

```text
/Volumes/external/sources/.venvs/cppmega.source/
  bin/python
  lib/python3.13/site-packages/00_cppmega_sources.pth
  cppmega-environment.json
```

The `.pth` contains exactly the cppmega checkout and the selected real
Megatron-LM source root. No editable install is performed, so the source
checkout is not modified and no stale `megatron-core` wheel is silently used.

After the target has been provisioned from an approved wheelhouse/image, run:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /Volumes/external/sources/.venvs/cppmega.source/bin/python \
  scripts/env/cppmega_env.py verify \
  --env /Volumes/external/sources/.venvs/cppmega.source
```

`verify` reads the commit and roots from the target manifest when they are not
provided on the command line. It launches the target interpreter from a
temporary working directory with `PYTHONPATH` and user-site imports disabled,
then checks that `cppmega`, `megatron.core`, and representative Megatron
modules resolve from the recorded source trees. Unexpected `sys.path` entries
from copied `.pth` files are rejected as well; only the target, the two
recorded source roots, and the target interpreter's base Python are allowed.
PEP 660 namespace placeholders are accepted only when the exact placeholder is
registered by a loaded editable finder. An external directory is rejected even
when its basename starts with `__editable__`.

The manifest's `repo_root` records the checkout used when the environment was
bootstrapped; it is provenance, not a runtime path lock. Repository CI extracts
the committed tree into a temporary checkout and places that checkout first on
`PYTHONPATH`, while pytest continues to validate the manifest's Megatron commit
and clean-source receipt. This keeps staged execution on the requested tree
without importing cppmega from the canonical checkout.

## Local Mac Source Receipt

The CUDA stack in `STACK.lock` is not a Mac-installable H200 environment. For
local import and contract checks, use a clean detached Megatron worktree at the
locked Core commit and a separate source-profile environment:

```bash
git -C /Volumes/external/sources/Megatron-LM worktree add --detach \
  /Volumes/external/sources/Megatron-LM-core_v0.18.0 \
  ba7b5ebce12af60627a80985792a1449ce45f46c

env -u PYTHONPATH uv pip install \
  --python /Volumes/external/sources/.venvs/cppmega.source/bin/python \
  'torch>=2.6.0' numpy 'packaging>=24.2' absl-py \
  'pytest>=9.0.3' 'einops>=0.8' 'libclang==18.1.1' 'pyarrow>=21' \
  'tokenizers>=0.22' 'datasketch==1.10.0'
```

Megatron Core imports Triton during its broad package probe. On this Mac the
working Triton receipt is an explicit editable source checkout, not a CUDA
wheel:

```bash
env -u PYTHONPATH uv pip install --no-deps -e \
  /Volumes/external/sources/triton-pr9701 \
  --python /Volumes/external/sources/.venvs/cppmega.source/bin/python
```

Then run `cppmega_env.py verify` with the clean worktree and `--profile source`.
This proves source wiring and importability only; it does not claim Triton
kernel execution, Transformer Engine, or H200 CUDA readiness. Those require
the pinned Linux image and self-hosted GPU runner.

Current local receipt:

```bash
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  /Volumes/external/sources/.venvs/cppmega.source/bin/python \
  scripts/env/cppmega_env.py verify \
  --profile source \
  --env /Volumes/external/sources/.venvs/cppmega.source \
  --megatron-root /Volumes/external/sources/Megatron-LM-core_v0.18.0 \
  --megatron-ref ba7b5ebce12af60627a80985792a1449ce45f46c
```

This returns `PASS`, including the exact source origin, dependency contract,
representative Megatron imports, and `pip check`. Pytest also reads the same
environment manifest before probing legacy sibling checkouts, so an omitted
`MEGATRON_LM_REPO` does not silently select the dirty source tree.

## Profiles

`locked` is the default and enforces the exact `torch` pin in `STACK.lock`, as
well as Python 3.13 and the base Megatron dependencies. This is the production
reproducibility profile and is intended for the CUDA/Linux image or a matching
provisioned host.

`source` keeps the source-tree and isolation checks but uses the dependency
constraints declared by the real Megatron `pyproject.toml` instead of the
CUDA-specific torch pin. It is useful for a deliberately provisioned local
source-import environment; it is not evidence that the CUDA training stack is
available.

Neither profile authorizes installation into the shared `.venv`.
Both `verify` and pytest require `source_dirty=false` for a usable receipt.
`--allow-dirty-source` permits diagnostic inspection and records the dirty
state, but verification remains non-zero and the receipt cannot bootstrap
pytest.

The direct repository runner preserves an explicitly supplied
`CPPMEGA_TEST_PROFILE`. Its `linux-contracts` lane sets `portable-data`
directly; the macOS and CUDA lanes clear that profile because they require the
real Megatron source contract.

## Regression Tests

The tests are intentionally colocated with the tool so they stay within the
environment ownership boundary and use only the Python standard library:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Volumes/external/sources/.venvs/cppmega.source/bin/python \
  -m unittest discover \
  -s scripts/env -p 'test_*.py' -v
```

They cover shared symlink rejection, system-site rejection, source-path and
manifest creation, staged-checkout manifest reuse, no-`pip-install` bootstrap
behavior, missing dependency failure, registered versus arbitrary editable
paths, profile propagation, and exact commit/dirty-source failures.

## Current Boundary

The repository-local `.venv` symlink and the legacy Megatron checkout are left
untouched because live conveyors and other work may still reference them. New
local source checks and repository-owned macOS CI use the dedicated
`/Volumes/external/sources/.venvs/cppmega.source` interpreter. Its manifest
selects the clean detached Megatron checkout at the locked commit.

The local Mac can validate source wiring and dependency contracts, but it
cannot by itself prove the CUDA-only Transformer Engine, fused wheels, or
distributed H200 runtime. Those remain separate Linux/CUDA receipts.
