"""Modal checker for the Mamba3 monolithic chunk wave1 skeleton.

The default entrypoint only checks whether the prebuilt image has the CuTe DSL
stack.  The GPU smoke path is available for H200 follow-up but is not required
for local compile validation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import modal


GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200")
CPPMEGA_ROOT = "/opt/cppmega"
APP_NAME = "cppmega-mamba3-mono-chunk-wave1"


def _image() -> modal.Image:
    image: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    )
    image = image.add_local_dir("cppmega", f"{CPPMEGA_ROOT}/cppmega", copy=True)
    image = image.add_local_dir("tools", f"{CPPMEGA_ROOT}/tools", copy=True)
    image = image.env({"PYTHONPATH": CPPMEGA_ROOT, "CPPMEGA_IMAGE_REF": GHCR_REF})
    return image


app = modal.App(APP_NAME)


@app.function(image=_image(), timeout=300)
def check_cute_stack() -> dict[str, Any]:
    import importlib.metadata as md
    import importlib.util

    modules = [
        "cutlass",
        "cutlass.cute",
        "cuda.bindings.driver",
        "quack",
        "torch",
    ]
    specs = {}
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
            specs[module] = None if spec is None else spec.origin
        except Exception as exc:  # noqa: BLE001
            specs[module] = f"{type(exc).__name__}: {exc}"

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "image_ref": GHCR_REF,
        "specs": specs,
        "versions": {
            "nvidia-cutlass-dsl": version("nvidia-cutlass-dsl"),
            "cuda-python": version("cuda-python"),
            "torch": version("torch"),
            "quack": version("quack"),
        },
        "cute_viable": (
            specs.get("cutlass.cute") is not None
            and not str(specs.get("cutlass.cute")).startswith("ModuleNotFoundError")
        ),
    }


@app.function(image=_image(), gpu=GPU_SPEC, timeout=20 * 60)
def smoke_h200(shape: str = "--B 1 --S 16 --H 1 --P 64") -> dict[str, Any]:
    import shlex
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "tools/probes/mamba3_mono_chunk_smoke.py",
        *shlex.split(shape),
    ]
    proc = subprocess.run(
        cmd,
        cwd=CPPMEGA_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
        "returncode": proc.returncode,
        "output": proc.stdout[-12000:],
    }


@app.local_entrypoint()
def main(run_gpu_smoke: bool = False) -> None:
    result = smoke_h200.remote() if run_gpu_smoke else check_cute_stack.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
