# DEPRECATED: This script uses removed APIs. See modal_fa4_production_test.py for current GPU test.
"""Modal H200: validate the FA4 (flash-attn-4) score_mod path on a real GPU.

The cppmega GHCR image pins ``flash-attn-4[cu13]==4.0.0b19``.  The production
chunk-native score_mod lives in ``cppmega/megatron/fa4_score_mod_adapter.py``
and ships TWO callables:

  * ``graph_score_mod``          -- beta23 ABI, 7 args (score, b, h, seqlen_info, q, k, aux_tensors)
  * ``graph_score_mod_beta19``   -- beta19 ABI, 6 args (score, b, h, q, k, aux_tensors)

with ``_detect_fa4_score_mod_arity()`` choosing between them at first forward.
CPU mock tests cannot prove which ABI the installed wheel actually wants, nor
that the callback survives FlashAttention-4's CuTe-DSL JIT.  This script closes
that gap on a real H200.

It does, on the GPU:
  a) import ``flash_attn.cute`` and report the version
  b) detect the score_mod arity the installed wheel expects (both via the
     production detector and via direct source inspection)
  c) JIT-compile the production ``graph_score_mod`` / ``graph_score_mod_beta19``
     under ``flash_attn.cute.jit``
  d) run a small forward pass: Q,K,V [1, 128, 4, 64] with a simple score_mod
  e) report version, arity, compile success/failure, forward output shape

Usage:
    modal run scripts/modal_fa4_gpu_test.py

Validated findings (Modal H200, image flash-attn-4==4.0.0b19,
nvidia-cutlass-dsl==4.6.0.dev0, apache-tvm-ffi==0.1.9):

  1. The CuTe-DSL score_mod COMPILE PIPELINE IS BROKEN in this image,
     independent of cppmega code: even FA4's built-in ``softcap`` score_mod
     fails with ``make_kwargs_wrapper() got an unexpected keyword argument
     'map_dataclass_to_tuple'``. cutlass-dsl 4.6.0.dev0 passes
     ``map_dataclass_to_tuple=`` but the pinned apache-tvm-ffi 0.1.9 does not
     accept it. No score_mod (built-in or custom) can JIT-compile until the
     tvm-ffi / cutlass-dsl pins are reconciled.

  2. The real b19 score_mod ABI (from flash_attn/cute/softmax.py
     ``call_score_mod``) is 7-arg fwd / 8-arg bwd and ALWAYS passes
     ``seqlen_info``, invoked BY KEYWORD::

         score_mod(score, batch_idx, head_idx, q_idx=, kv_idx=,
                   seqlen_info=, aux_tensors=)
         score_mod_bwd(grad_out, score, batch_idx, head_idx, q_idx=, kv_idx=,
                       seqlen_info=, aux_tensors=)

     So ``_detect_fa4_score_mod_arity()`` returning 6 ("beta19 6-arg") is
     WRONG: b19 has no 6-arg ABI. The production callbacks also use parameter
     names ``b, h, q, k`` which do not match the keyword names
     ``batch_idx, head_idx, q_idx, kv_idx`` FA4 calls with -- they would raise
     ``TypeError: unexpected keyword argument`` even once #1 is fixed.

  3. Indexing an aux tensor as ``aux[batch_idx, q_idx, kv_idx]`` with DSL
     tensor-valued indices raises ``ValueError: Expected Coord, whose leaves
     are integers or None`` -- the FlexAttention aux-tensor indexing convention
     differs and must be re-derived from a working FA4 example.
"""
from __future__ import annotations

import sys

print(
    "DEPRECATED: scripts/modal_fa4_gpu_test.py uses removed cppmega APIs "
    "(_detect_fa4_score_mod_arity, _select_score_mod_fns, "
    "graph_score_mod_beta19, graph_score_mod_bwd_beta19). "
    "Use scripts/modal_fa4_production_test.py for the current FA4 GPU test.",
    flush=True,
)
sys.exit(0)

import json
import os
import pathlib
import traceback
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_DIGEST = os.environ.get(
    "GHCR_DIGEST",
    "sha256:08c5db7368d1037d930e0825281468927de9c85b12ba10373fe07e082150d983",
)
GHCR_REF = f"{GHCR_REPO}@{GHCR_DIGEST}"
GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "H200:1")

app = modal.App("cppmega-fa4-gpu-test")


def _image() -> modal.Image:
    img: Any = modal.Image.from_registry(
        GHCR_REF,
        secret=modal.Secret.from_name("ghcr-pull"),
        add_python=None,
    ).env(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "WANDB_MODE": "disabled",
        }
    )
    img = (
        img.add_local_dir(
            str(_REPO_ROOT / "cppmega"), remote_path="/opt/cppmega/cppmega", copy=True
        )
        .add_local_file(
            str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml"
        )
    )
    return img


def _section(title: str) -> None:
    print("\n" + "=" * 70, flush=True)
    print(title, flush=True)
    print("=" * 70, flush=True)


@app.function(image=_image(), gpu=GPU_SPEC, timeout=1800)
def fa4_gpu_test() -> dict[str, Any]:
    """Run the FA4 score_mod validation suite on a single H200."""
    import inspect

    result: dict[str, Any] = {
        "gpu": GPU_SPEC,
        "image": GHCR_REF,
        "flash_attn_version": None,
        "flash_attn_4_version": None,
        "cutlass_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "detected_arity": None,
        "detector_label": None,
        "interface_has_seqlen_info": None,
        "jit_compile": {},
        "forward": {},
    }

    # ------------------------------------------------------------------
    # (0) GPU / environment
    # ------------------------------------------------------------------
    _section("(0) Environment")
    import torch

    result["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        print(f"CUDA available: True  device: {result['gpu_name']}", flush=True)
        cap = torch.cuda.get_device_capability(0)
        print(f"compute capability: sm_{cap[0]}{cap[1]}", flush=True)
    else:
        print("CUDA available: False -- cannot proceed", flush=True)
        return result

    # ------------------------------------------------------------------
    # (a) Import flash_attn.cute and report versions
    # ------------------------------------------------------------------
    _section("(a) flash_attn.cute import + version")
    try:
        import flash_attn

        result["flash_attn_version"] = getattr(flash_attn, "__version__", None)
        print(f"flash_attn.__version__ = {result['flash_attn_version']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"flash_attn import FAILED: {exc}", flush=True)
        result["import_error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        import importlib.metadata as _md

        for pkg in ("flash_attn_4", "flash-attn-4", "flash_attn4"):
            try:
                result["flash_attn_4_version"] = _md.version(pkg)
                print(f"pip metadata {pkg} = {result['flash_attn_4_version']}", flush=True)
                break
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        print(f"metadata lookup error: {exc}", flush=True)

    try:
        import cutlass

        result["cutlass_version"] = getattr(cutlass, "__version__", None)
        print(f"cutlass.__version__ = {result['cutlass_version']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"cutlass import note: {exc}", flush=True)

    try:
        import flash_attn.cute as cute  # noqa: F401
        from flash_attn.cute.interface import flash_attn_func  # noqa: F401

        print("flash_attn.cute + flash_attn.cute.interface imported OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"flash_attn.cute import FAILED: {exc}", flush=True)
        traceback.print_exc()
        result["cute_import_error"] = f"{type(exc).__name__}: {exc}"
        return result

    # ------------------------------------------------------------------
    # (b) Detect the score_mod arity the wheel actually wants
    # ------------------------------------------------------------------
    _section("(b) score_mod arity detection")

    # (b.1) Direct source inspection of the installed interface.
    iface_has_seqlen = None
    try:
        from flash_attn.cute import interface as _fa4_iface

        for cand_name in ("_flash_attn_fwd", "flash_attn_func"):
            cand = getattr(_fa4_iface, cand_name, None)
            if cand is None:
                continue
            try:
                src = inspect.getsource(cand)
            except (OSError, TypeError):
                continue
            if "seqlen_info" in src:
                iface_has_seqlen = True
                print(f"{cand_name}: source mentions 'seqlen_info' -> beta23 (7-arg)", flush=True)
                break
            if "score_mod" in src:
                iface_has_seqlen = False
                print(f"{cand_name}: 'score_mod' present, no 'seqlen_info' -> beta19 (6-arg)", flush=True)
                break
    except Exception as exc:  # noqa: BLE001
        print(f"interface inspection error: {exc}", flush=True)
    result["interface_has_seqlen_info"] = iface_has_seqlen

    # (b.2) Production detector from the adapter.
    detected_arity = None
    detector_label = None
    try:
        from cppmega.megatron.fa4_score_mod_adapter import (
            _detect_fa4_score_mod_arity,
            _select_score_mod_fns,
        )

        detected_arity = _detect_fa4_score_mod_arity()
        _, _, detector_label = _select_score_mod_fns()
        print(f"_detect_fa4_score_mod_arity() = {detected_arity}", flush=True)
        print(f"_select_score_mod_fns() label = {detector_label}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"production detector error: {exc}", flush=True)
        traceback.print_exc()
    result["detected_arity"] = detected_arity
    result["detector_label"] = detector_label

    # ------------------------------------------------------------------
    # (c) score_mod call-contract introspection
    # ------------------------------------------------------------------
    # flash-attn-4 b19 does NOT expose a standalone ``flash_attn.cute.jit``
    # decorator: the score_mod is JIT-compiled *internally* by FA4 via
    # ``cutlass cute.compile`` during ``flash_attn_func`` (exercised in (d)).
    # Here we (1) confirm what compile entrypoints exist and (2) read back the
    # EXACT keyword signature FA4 uses when it invokes score_mod, so we know
    # the precise parameter names/arity the production callback must have.
    _section("(c) score_mod call-contract introspection")

    import flash_attn.cute as _cute_mod

    result["cute_has_jit"] = hasattr(_cute_mod, "jit")
    result["cute_has_compile"] = hasattr(_cute_mod, "compile")
    print(f"flash_attn.cute has 'jit'    : {result['cute_has_jit']}", flush=True)
    print(f"flash_attn.cute has 'compile': {result['cute_has_compile']}", flush=True)

    try:
        from flash_attn.cute import softmax as _fa4_softmax

        call_src = inspect.getsource(_fa4_softmax.call_score_mod)
        result["call_score_mod_source"] = call_src
        print("--- flash_attn.cute.softmax.call_score_mod source ---", flush=True)
        print(call_src, flush=True)
    except Exception as exc:  # noqa: BLE001
        result["call_score_mod_source_error"] = f"{type(exc).__name__}: {exc}"
        print(f"could not read call_score_mod source: {exc}", flush=True)

    from cppmega.megatron.fa4_score_mod_adapter import (
        graph_score_mod,
        graph_score_mod_beta19,
        graph_score_mod_bwd,
        graph_score_mod_bwd_beta19,
    )

    for label, fn in [
        ("graph_score_mod (beta23, 7-arg)", graph_score_mod),
        ("graph_score_mod_beta19 (6-arg)", graph_score_mod_beta19),
        ("graph_score_mod_bwd (beta23)", graph_score_mod_bwd),
        ("graph_score_mod_bwd_beta19", graph_score_mod_bwd_beta19),
    ]:
        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = []
        result["jit_compile"][label] = {"n_params": len(params), "params": params}
        print(f"[SIG] {label}: params={params}", flush=True)

    # ------------------------------------------------------------------
    # (d) Small forward pass with a simple score_mod
    # ------------------------------------------------------------------
    _section("(d) Forward pass Q,K,V [1,128,4,64] with simple score_mod")

    from flash_attn.cute.interface import flash_attn_func

    device = torch.device("cuda")
    batch, seqlen, heads, head_dim = 1, 128, 4, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    scale = head_dim ** -0.5

    # A simple additive score_mod backed by one aux bias tensor [B, Sq, Sk].
    # FA4 invokes score_mod BY KEYWORD. The b19 traceback shows the kw names:
    #   score_mod(score, batch_idx=, head_idx=, q_idx=, kv_idx=, aux_tensors=)
    # We try BOTH arities (with/without seqlen_info) using those exact names so
    # we learn definitively which one the installed wheel accepts.
    bias = torch.zeros(batch, seqlen, seqlen, device=device, dtype=torch.float32)

    def simple_score_mod_beta19(score, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        return score + aux_tensors[0][batch_idx, q_idx, kv_idx]

    def simple_score_mod_bwd_beta19(
        grad_out, score, batch_idx, head_idx, q_idx, kv_idx, aux_tensors
    ):
        return grad_out

    def simple_score_mod_beta23(
        score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        return score + aux_tensors[0][batch_idx, q_idx, kv_idx]

    def simple_score_mod_bwd_beta23(
        grad_out, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        return grad_out

    variants = [
        ("beta19 (6-arg, kw names)", simple_score_mod_beta19, simple_score_mod_bwd_beta19),
        ("beta23 (7-arg, seqlen_info)", simple_score_mod_beta23, simple_score_mod_bwd_beta23),
    ]

    accepted_variant = None
    for label, smod, smod_bwd in variants:
        entry: dict[str, Any] = {}
        try:
            out = flash_attn_func(
                q=q,
                k=k,
                v=v,
                softmax_scale=scale,
                causal=True,
                score_mod=smod,
                score_mod_bwd=smod_bwd,
                aux_tensors=[bias],
                block_sparse_tensors=None,
                mask_mod=None,
                return_lse=False,
            )
            if isinstance(out, tuple):
                out = out[0]
            entry["ok"] = True
            entry["out_shape"] = list(out.shape)
            entry["out_dtype"] = str(out.dtype)
            entry["finite"] = bool(torch.isfinite(out).all().item())
            print(
                f"[FWD] {label}: OK out_shape={tuple(out.shape)} "
                f"dtype={out.dtype} finite={entry['finite']}",
                flush=True,
            )
            if accepted_variant is None:
                accepted_variant = label
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()
            print(f"[FWD] {label}: FAILED: {type(exc).__name__}: {exc}", flush=True)
        result["forward"][label] = entry

    result["accepted_forward_variant"] = accepted_variant

    # ------------------------------------------------------------------
    # (d.2) Forward with the PRODUCTION score_mod selected by the detector
    # ------------------------------------------------------------------
    _section("(d.2) Forward pass with PRODUCTION score_mod (detector-selected)")
    try:
        from cppmega.megatron.fa4_score_mod_adapter import _select_score_mod_fns

        prod_smod, prod_smod_bwd, prod_label = _select_score_mod_fns()
        # Production aux layout: [token_to_chunk_q, token_to_chunk_k, chunk_bias,
        #                         rare_q, rare_k, rare_w]
        num_chunks = 4
        token_to_chunk_q = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)
        token_to_chunk_k = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)
        chunk_bias = torch.zeros(
            batch, num_chunks + 1, num_chunks + 1, dtype=torch.float32, device=device
        )
        rare_q = torch.zeros(batch, 1, dtype=torch.int32, device=device)
        rare_k = torch.full((batch, 1), -1, dtype=torch.int32, device=device)
        rare_w = torch.zeros(batch, 1, dtype=torch.float32, device=device)
        prod_aux = [token_to_chunk_q, token_to_chunk_k, chunk_bias, rare_q, rare_k, rare_w]

        out = flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=scale,
            causal=True,
            score_mod=prod_smod,
            score_mod_bwd=prod_smod_bwd,
            aux_tensors=prod_aux,
            block_sparse_tensors=None,
            mask_mod=None,
            return_lse=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        result["production_forward"] = {
            "label": prod_label,
            "ok": True,
            "out_shape": list(out.shape),
            "out_dtype": str(out.dtype),
            "finite": bool(torch.isfinite(out).all().item()),
        }
        print(
            f"[PROD-FWD] {prod_label}: OK out_shape={tuple(out.shape)} "
            f"finite={result['production_forward']['finite']}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        result["production_forward"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"[PROD-FWD] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    # ------------------------------------------------------------------
    # (e) Summary
    # ------------------------------------------------------------------
    _section("(e) SUMMARY")
    print(f"flash_attn version      : {result['flash_attn_version']}", flush=True)
    print(f"flash-attn-4 (pip)      : {result['flash_attn_4_version']}", flush=True)
    print(f"interface seqlen_info   : {result['interface_has_seqlen_info']}", flush=True)
    print(f"detected arity          : {result['detected_arity']} ({result['detector_label']})", flush=True)
    for label, entry in result["jit_compile"].items():
        print(f"  JIT {label}: ok={entry.get('jit_ok')} err={entry.get('error', '')}", flush=True)
    print(f"accepted forward variant: {result['accepted_forward_variant']}", flush=True)
    pf = result.get("production_forward", {})
    print(f"production forward      : ok={pf.get('ok')} shape={pf.get('out_shape')} err={pf.get('error', '')}", flush=True)

    return result


@app.local_entrypoint()
def main() -> None:
    print("FA4 score_mod GPU validation")
    print(f"  GPU: {GPU_SPEC}")
    print(f"  Image: {GHCR_REF}")
    result = fa4_gpu_test.remote()
    print("\n" + "=" * 70)
    print("RESULT JSON")
    print("=" * 70)
    print(json.dumps(result, indent=2))
