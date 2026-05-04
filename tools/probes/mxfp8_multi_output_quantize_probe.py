from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeMXFP8Tensor:
    def __init__(self, name: str):
        self.name = name
        self._rowwise_data = object()
        self._rowwise_scale_inv = object()
        self._columnwise_data = object()
        self._columnwise_scale_inv = object()
        self._with_gemm_swizzled_scales = False


class FakeMXFP8Quantizer:
    def quantize(self, tensor, *args, **kwargs):
        return FakeMXFP8Tensor("quantized")

    def update_quantized(self, src, dst, *args, **kwargs):
        return dst


def install_fake_te(*, expose_multi_output: bool):
    fake_tex = ModuleType("transformer_engine_torch")

    def split_quantize(*_args, **_kwargs):
        raise RuntimeError("single-output split_quantize was called")

    fake_tex.split_quantize = split_quantize

    if expose_multi_output:
        calls = []

        def mxfp8_split_quantize_with_rowwise_transpose(
            tensor,
            split_sections,
            quantizers,
            disable_bulk_allocation,
            transpose_scales_with_gemm_swizzled,
        ):
            calls.append(
                {
                    "splits": list(split_sections),
                    "quantizers": len(quantizers),
                    "disable_bulk_allocation": bool(disable_bulk_allocation),
                    "transpose_scales_with_gemm_swizzled": bool(
                        transpose_scales_with_gemm_swizzled
                    ),
                }
            )
            outputs = [FakeMXFP8Tensor(f"out{i}") for i, _ in enumerate(split_sections)]
            for i, out in enumerate(outputs):
                out._te_gemm_ready_rowwise_transpose_for_backward = FakeMXFP8Tensor(
                    f"ready{i}"
                )
            return outputs

        fake_tex.mxfp8_split_quantize_with_rowwise_transpose = (
            mxfp8_split_quantize_with_rowwise_transpose
        )
        fake_tex.cppmega_probe_calls = calls

    fake_recipe = ModuleType("transformer_engine.common.recipe")
    fake_recipe.MXFP8BlockScaling = type("MXFP8BlockScaling", (), {})
    fake_recipe.NVFP4BlockScaling = type("NVFP4BlockScaling", (), {})

    fake_tensor = ModuleType("transformer_engine.pytorch.tensor")
    fake_tensor.MXFP8Quantizer = FakeMXFP8Quantizer
    fake_tensor.MXFP8Tensor = FakeMXFP8Tensor
    fake_tensor.QuantizedTensor = type("QuantizedTensor", (), {})

    fake_quantization = ModuleType("transformer_engine.pytorch.quantization")

    class FakeFP8State:
        @staticmethod
        def get_fp8_recipe():
            return None

    fake_quantization.FP8GlobalStateManager = FakeFP8State
    fake_quantization.check_mxfp8_support = lambda: (True, "")
    fake_fp8 = ModuleType("transformer_engine.pytorch.fp8")
    fake_fp8.FP8GlobalStateManager = FakeFP8State

    fake_mxfp8_tensor = ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    fake_mxfp8_tensor.MXFP8Tensor = FakeMXFP8Tensor

    fake_linear = ModuleType("transformer_engine.pytorch.module.linear")
    fake_linear.general_gemm = lambda *args, **kwargs: None
    fake_linear.general_grouped_gemm = lambda *args, **kwargs: None

    fake_base = ModuleType("transformer_engine.pytorch.module.base")

    class FakeBaseModule:
        @staticmethod
        def grad_output_preprocess(ctx, grad_output, row_parallel_mode, quantizer):
            return grad_output, None

    fake_base.TransformerEngineBaseModule = FakeBaseModule

    sys.modules.update(
        {
            "transformer_engine_torch": fake_tex,
            "transformer_engine": ModuleType("transformer_engine"),
            "transformer_engine.common": ModuleType("transformer_engine.common"),
            "transformer_engine.common.recipe": fake_recipe,
            "transformer_engine.pytorch": ModuleType("transformer_engine.pytorch"),
            "transformer_engine.pytorch.fp8": fake_fp8,
            "transformer_engine.pytorch.quantization": fake_quantization,
            "transformer_engine.pytorch.tensor": fake_tensor,
            "transformer_engine.pytorch.tensor.mxfp8_tensor": fake_mxfp8_tensor,
            "transformer_engine.pytorch.module": ModuleType(
                "transformer_engine.pytorch.module"
            ),
            "transformer_engine.pytorch.module.base": fake_base,
            "transformer_engine.pytorch.module.linear": fake_linear,
        }
    )
    return fake_tex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-api", action="store_true")
    args = parser.parse_args()

    os.environ["CPPMEGA_TE_VERSION_STRICT"] = "0"
    os.environ["CPPMEGA_DSA_SPARSE_MODE"] = "gather_scatter"
    os.environ["CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW"] = "1"
    os.environ["CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER"] = "1"
    os.environ["CPPMEGA_TE_MXFP8_TRANSPOSE_EMIT_BACKEND"] = "te"
    os.environ["CPPMEGA_TE_MXFP8_GROUPED_QUANTIZE_PRODUCER"] = "multi_output"

    fake_tex = install_fake_te(expose_multi_output=not args.missing_api)
    sys.modules.pop("scripts.cppmega_fp8_shim", None)
    shim = importlib.import_module("scripts.cppmega_fp8_shim")

    status = "pass"
    error = None
    try:
        fake_tex.split_quantize(
            torch.empty((4, 2), dtype=torch.bfloat16),
            [2, 2],
            [FakeMXFP8Quantizer(), FakeMXFP8Quantizer()],
        )
    except Exception as exc:  # expected only for --missing-api
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    print(
        json.dumps(
            {
                "status": status,
                "error": error,
                "calls": getattr(fake_tex, "cppmega_probe_calls", []),
                "counters": shim._cppmega_te_bwd_stats_snapshot(),
            },
            sort_keys=True,
        )
    )
    return 0 if (args.missing_api or status == "pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
