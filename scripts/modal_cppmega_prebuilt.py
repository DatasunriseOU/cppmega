"""Modal entrypoint that pulls the prebuilt cppmega image from ghcr.io
instead of rebuilding the stack from declarative pip_install layers.

Image is built by .github/workflows/build-image.yml and lives at
ghcr.io/jewelmusicee/cppmega:<sha>. To pull a private image, Modal needs a
secret named `ghcr-pull` with REGISTRY_USERNAME and REGISTRY_PASSWORD
(use a fine-grained PAT scoped to read:packages on the jewelmusicee org):

    modal secret create ghcr-pull \
        REGISTRY_USERNAME=<gh-username> \
        REGISTRY_PASSWORD=<fine-grained-PAT>

Override the tag at runtime via env var GHCR_TAG (defaults to :latest).
"""
from __future__ import annotations

import os
from typing import Any

import modal

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/jewelmusicee/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "latest")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H100:2")


def cppmega_prebuilt_image() -> modal.Image:
    """Pull the prebuilt cppmega image from ghcr.io (no Modal-side build)."""
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,  # python 3.13 already baked in
    )
    return img


# A minimal app for smoke-testing the pulled image on real GPU hardware.
app = modal.App("cppmega-prebuilt-smoke")


@app.function(
    image=cppmega_prebuilt_image(),
    gpu=GPU_SPEC,
    timeout=600,
)
def smoke() -> dict:
    """Verify torch + TE + flash-attn + mamba_ssm + tilelang + qoptim_cuda
    all import cleanly and CUDA is visible."""
    import torch
    import transformer_engine
    import transformer_engine.pytorch  # noqa: F401
    import flash_attn  # noqa: F401
    import mamba_ssm  # noqa: F401
    import causal_conv1d  # noqa: F401
    import fast_hadamard_transform  # noqa: F401
    import tilelang
    import qoptim_cuda  # noqa: F401
    import cutlass
    import cutlass.cute as cute  # noqa: F401
    from cutlass.utils import LayoutEnum, SmemAllocator  # noqa: F401
    import quack  # noqa: F401
    from quack import sm90_utils, copy_utils, layout_utils, mx_utils  # noqa: F401

    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "te": transformer_engine.__version__,
        "tilelang": getattr(tilelang, "__version__", None),
        "tilelang_file": getattr(tilelang, "__file__", None),
        "cutlass": getattr(cutlass, "__version__", None),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_ref": GHCR_REF,
        "gpu_spec": GPU_SPEC,
    }


@app.function(
    image=cppmega_prebuilt_image(),
    timeout=300,
)
def versions() -> dict:
    """Report package/source versions from the GHCR image without allocating GPU."""
    import importlib.metadata as md
    import os

    import torch
    import transformer_engine
    import flash_attn
    import mamba_ssm
    import tilelang
    import cppmega

    def dist_version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "image_ref": GHCR_REF,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "te": transformer_engine.__version__,
        "flash_attn": getattr(flash_attn, "__version__", None),
        "mamba_ssm": getattr(mamba_ssm, "__version__", None),
        "tilelang": getattr(tilelang, "__version__", None),
        "tilelang_dist": dist_version("tilelang"),
        "tilelang_file": getattr(tilelang, "__file__", None),
        "cppmega_file": getattr(cppmega, "__file__", None),
        "cppmega_has_git": os.path.exists("/opt/cppmega/.git"),
    }


@app.local_entrypoint()
def main() -> None:
    if os.environ.get("CPPMEGA_MODAL_MODE") == "versions":
        result = versions.remote()
    else:
        result = smoke.remote()
    print(result)
