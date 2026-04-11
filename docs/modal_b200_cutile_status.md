# Modal B200 cuTile Python status (2026-04-10)

Verified the cuTile Python + TileLang + mamba_ssm stack can be provisioned on
Modal B200:2 and that cuTile Python compiles + launches a real kernel on the
B200 (sm_100, datacenter Blackwell). This unblocks cuTile Python kernel
development while GB10 (sm_121 consumer Blackwell) is physically down.

## What works

- **Modal auth**: `jewelmusic` profile authenticated via `~/.modal.toml`.
- **Image build**: `modal.Image.debian_slim(python_version="3.13")` base +
  pip install of the stack (no Docker Hub pull, sidesteps rate limits).
- **GPU detected**: `torch.cuda.get_device_name(0) = "NVIDIA B200"`,
  capability `(10, 0)` i.e. **sm_100**. Count 2 (`B200:2`).
- **torch**: `2.12.0.dev20260410+cu132` — exact match to h200_1.
- **CUDA toolkit**: `nvcc 13.2.51` on PATH via `nvidia-cuda-nvcc` wheel.
- **cuda.tile**: `1.2.0` imports cleanly; auto-detects sm_100 via
  `cuda.tile._compile.get_sm_arch()`.
- **tileiras**: installed as the `nvidia-cuda-tileiras` wheel pulled in via the
  `cuda-tile[tileiras]` extra (NOT the bare `cuda-tile` — the compiler binary
  is in the optional extra, which was a hidden gotcha).
- **tilelang**: `0.1.8`
- **apache-tvm-ffi**: `0.1.9` (load-bearing pin, see memory/reference_stack_bench.md)

## Sanity kernel verified

Trivial vector-add kernel compiled with `@cuda.tile.kernel` and launched via
`cuda.tile.launch` on B200 sm_100:

- Compile path: Python → tileiras (from nvidia-cuda-tileiras wheel) → PTX → cubin
- Load + run: OK
- Numerical check: `max_abs_err = 0.000e+00` vs torch reference
- Wall clock: sub-second after the first JIT compile
- Detected arch: `sm_100` (printed from `cuda.tile._compile.get_sm_arch()`)

This is the confirmation we were missing: cuTile Python 1.2.0 DOES compile for
sm_100 datacenter Blackwell and the cubin loads on B200. This supplements the
H200 negative-result test matrix in `memory/reference_cutile_python.md`.

## mamba_ssm status

NOT in the sanity image. Installation from the prebuilt wheel fails against
torch 2.12 nightly (no matching manylinux wheel as of 2026-04-10) and the
source build fails under pip's build isolation because the isolated env sees
torch 2.11+cu130 not 2.12+cu132. Needs `--no-build-isolation` + `TORCH_CUDA_ARCH_LIST='10.0'`
+ `MAMBA_FORCE_BUILD=1`. A separate layered image (`_image_with_mamba()` in
`modal_cutile_b200.py`) handles this lazily and is only materialised when a
function that needs mamba_ssm is invoked — not the base sanity path.

## Gotchas encountered

1. **Docker Hub rate limit**: Modal's skopeo pulls from `docker.io/nvidia/cuda`
   hit the unauthenticated rate limit immediately. Switched to
   `modal.Image.debian_slim(python_version="3.13")` (Modal's own mirror) +
   PyPI-based CUDA toolkit. No rate limit, no authentication.
2. **`nvidia-cuda-*-cu13` deprecated**: The pip package names without the
   `-cu13` suffix are the canonical ones now (`nvidia-cuda-nvcc` not
   `nvidia-cuda-nvcc-cu13`).
3. **nvcc path**: `nvidia-cuda-nvcc` wheel drops nvcc at
   `nvidia/cu13/bin/nvcc` in site-packages, not `nvidia/cuda_nvcc/bin/`.
4. **tileiras is an extra**: The bare `cuda-tile` wheel installs fine but
   fails at the first `ct.launch()` with
   `FileNotFoundError: 'tileiras' compiler not found`. You MUST ask for
   `cuda-tile[tileiras]` which pulls `nvidia-cuda-tileiras==13.2.51`.
5. **`tileiras` CLI**: cuda-tile 1.2.0 no longer ships a user-facing
   `tileiras --help` CLI; the compiler is only invoked from the Python API.
   The sanity script was originally shelling out to `tileiras --help`; it
   now introspects `cuda.tile._compile.get_sm_arch()` and relies on the
   compile-and-launch round-trip to prove the target is real.
6. **Constant annotation globals**: `cuda.tile` resolves kernel annotations
   via `typing.get_type_hints(pyfunc, globalns=pyfunc.__globals__)`, so
   aliases like `ConstInt = ct.Constant[int]` MUST live in the module
   globals of the file where the kernel is defined — not inside a closure.
   Moved the vadd kernel to module scope in a try/except guarded import
   block so the file still parses on developer laptops without cuda-tile.

## Cost (best estimate)

Modal does not expose a `modal cost` CLI. Based on the Modal dashboard at
https://modal.com/apps/jewelmusic/main/, 8 app invocations were made between
00:55 and 01:12 MSK. Image build steps run on CPU-only builders (free-ish);
only the `@app.function(gpu="B200:2")` body runs on the B200:2 instance and
each body took under 30 seconds of actual GPU time. Rough upper bound:

- 8 runs x 30 s GPU time = 4 min B200:2 = 0.066 hr
- B200:2 list price ~$20-30/hr
- Estimated spend: **under $2.00**

Well under the $15 session cap. All apps show `state=stopped` in `modal app list`.

## Volume

Created `cppmega-cutile-mamba3` via `modal.Volume.from_name(..., create_if_missing=True)`.
Mounted at `/vol` inside the sanity function, wrote a `READY` sentinel file
and committed. Ready to receive the rsync of
`/home/dave/mamba3_mimo_cutile/` from GB10 once GB10 is physically back.

## Files created

- `scripts/modal_cutile_b200.py` — Modal app: image, sanity, inspect, main
- `scripts/modal_cutile_mamba_mimo.py` — stub for the full Mamba3 MIMO parity
  test once the GB10 port is available
- `docs/modal_b200_cutile_status.md` — this file

## Next step

1. **Bring GB10 back online** (physical reset at home).
2. `rsync -avz gb10:/home/dave/mamba3_mimo_cutile/ /Volumes/external/sources/cppmega/.tmp/mamba3_mimo_cutile/`
3. Flesh out `modal_cutile_mamba_mimo.py::run_parity` with the actual
   fwd/bwd_fwd/bwd_bwd test calls against the ported package.
4. Run `modal run scripts/modal_cutile_mamba_mimo.py::main` and bench vs
   the TileLang/mamba_ssm reference. The B200 (sm_100, 192 GiB HBM3e) has
   ~180 KB/SM smem budget per the Blackwell whitepaper, which should clear
   the GB10 bwd_bwd blocker documented in
   `memory/reference_gb10_bwd_bwd_blocker.md` (GB10 only has 100 KB/SM).
5. If B200 parity passes all three kernels, consider promoting the cuTile
   port and retiring the GB10-specific fallback.
