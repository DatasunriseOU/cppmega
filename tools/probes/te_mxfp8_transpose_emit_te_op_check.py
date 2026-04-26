#!/usr/bin/env python3
"""Check patched TE MXFP8 transpose emission against the local probe kernel."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch

_EMIT_EXT_PATH = Path(__file__).with_name("te_mxfp8_transpose_emit_ext.py")
_EMIT_EXT_SPEC = importlib.util.spec_from_file_location(
    "te_mxfp8_transpose_emit_ext_check", _EMIT_EXT_PATH
)
if _EMIT_EXT_SPEC is None or _EMIT_EXT_SPEC.loader is None:
    raise RuntimeError(f"could not load transpose-emission helper from {_EMIT_EXT_PATH}")
emit_ext = importlib.util.module_from_spec(_EMIT_EXT_SPEC)
_EMIT_EXT_SPEC.loader.exec_module(emit_ext)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _load_te_common(path: str | None) -> None:
    if path:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _tex_info() -> dict[str, Any]:
    try:
        import transformer_engine.common as te_common

        te_common.load_framework_extension("torch")
        tex = importlib.import_module("transformer_engine_torch")
    except ImportError as exc:
        return {"available": False, "error": str(exc).splitlines()[0]}
    return {
        "available": True,
        "file": getattr(tex, "__file__", None),
        "has_mxfp8_scaling_transpose_cast": hasattr(tex, "mxfp8_scaling_transpose_cast"),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows % 32 != 0:
        raise ValueError("--rows must be divisible by 32")
    if args.cols % 32 != 0:
        raise ValueError("--cols must be divisible by 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    _load_te_common(args.te_common_lib)

    torch.manual_seed(args.seed)
    source = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.bfloat16)
    scale_rows = _round_up(args.rows // 32, 4)
    scale_cols = _round_up(args.cols, 128)
    columnwise_scale_inv = torch.full(
        (scale_rows, scale_cols),
        args.scale_byte,
        device="cuda",
        dtype=torch.uint8,
    )

    os.environ["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] = "probe"
    probe_data, probe_scale = emit_ext.emit_transpose_from_bf16(source, columnwise_scale_inv)

    os.environ["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] = "te"
    te_data, te_scale = emit_ext.emit_transpose_from_bf16(source, columnwise_scale_inv)
    torch.cuda.synchronize()

    return {
        "shape": {"rows": args.rows, "cols": args.cols},
        "scale_shape": list(columnwise_scale_inv.shape),
        "tex": _tex_info(),
        "payload_equal": bool(torch.equal(probe_data, te_data)),
        "scale_equal": bool(torch.equal(probe_scale, te_scale)),
        "max_payload_abs_byte_delta": int(
            (probe_data.to(torch.int16) - te_data.to(torch.int16)).abs().max().item()
        ),
        "max_scale_abs_byte_delta": int(
            (probe_scale.to(torch.int16) - te_scale.to(torch.int16)).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scale-byte", type=int, default=127)
    parser.add_argument(
        "--te-common-lib",
        type=Path,
        default=None,
        help="Optional libtransformer_engine.so to preload with RTLD_GLOBAL.",
    )
    args = parser.parse_args()
    te_common_lib = str(args.te_common_lib) if args.te_common_lib is not None else None
    args.te_common_lib = te_common_lib
    print(json.dumps(_run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
