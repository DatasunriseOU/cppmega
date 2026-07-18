#!/usr/bin/env python3
"""Run the fail-closed cppmega production Megatron H200 preflight."""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega.recipes.run_profiles import (  # noqa: E402
    get_run_profile,
    profile_shell_assignments,
)
from scripts.data.publish_megatron_bundle_to_nebius_s3 import (  # noqa: E402
    _safe_prefix_file,
    _sha256,
    _validate_bundle,
    _validate_prefix_manifest_contract,
    _validate_tokenizer_directory,
)
from cppmega.receipt_binding import (  # noqa: E402
    build_receipt_binding,
    canonical_sha256,
    validate_binding_shape,
    validate_receipt_binding,
)
from cppmega.megatron.graph_objective_loss import (  # noqa: E402
    validate_runtime_graph_contract,
)

PENDING_CHECKPOINT_SHA256 = "0" * 64
_GRAPH_CHUNK_SIDECARS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
_GRAPH_EDGE_KINDS = frozenset({"edge_pairs", "edge_triples"})
STACK_LOCK_PATH = ROOT / "STACK.lock"
STACK_REQUIRED_IMPORTS = (
    "transformer_engine",
    "transformer_engine.pytorch",
    "flash_attn",
    "flash_attn_3",
    "flash_attn.cute",
    "mamba_ssm",
    "causal_conv1d",
    "fast_hadamard_transform",
    "tilelang",
    "qoptim_cuda",
    "cutlass",
    "quack",
    "megatron.core",
    "cppmega",
)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _csr_offsets_receipt(
    path: Path,
    *,
    document_count: int,
    item_count: int,
    offset_dtype: object,
) -> tuple[dict[str, object], array]:
    if offset_dtype != "int64":
        raise RuntimeError(f"graph CSR offsets must be int64: {path}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"graph CSR offset path must be a regular file: {path}")
    expected_entries = document_count + 1
    expected_bytes = expected_entries * 8
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"graph CSR offset size mismatch for {path}: "
            f"{path.stat().st_size} != {expected_bytes}"
        )
    offsets = array("q")
    with path.open("rb") as stream:
        offsets.fromfile(stream, expected_entries)
    if sys.byteorder != "little":
        offsets.byteswap()
    if len(offsets) != expected_entries or offsets[0] != 0:
        raise RuntimeError(f"invalid graph CSR offsets header: {path}")

    maximum = 0
    previous = 0
    for offset in offsets[1:]:
        if offset < previous:
            raise RuntimeError(f"graph CSR offsets are not monotonic: {path}")
        maximum = max(maximum, offset - previous)
        previous = offset
    if previous != item_count:
        raise RuntimeError(
            f"graph CSR item count mismatch for {path}: {previous} != {item_count}"
        )
    return (
        {
            "offsets_path": path.name,
            "offsets_sha256": _sha256(path),
            "item_count": item_count,
            "max_items_per_document": maximum,
        },
        offsets,
    )


def _derive_graph_capacity_from_manifest(
    data_prefix: Path,
    *,
    manifest: dict,
    sequence_length: int,
) -> dict[str, object]:
    """Derive exact graph tensor capacities from manifest-bound CSR offsets."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    data_prefix = data_prefix.resolve()
    document_count = int(manifest.get("document_count", 0))
    if document_count <= 0:
        raise RuntimeError("graph capacity derivation requires document_count > 0")
    expected_capacity_tokens = document_count * sequence_length
    if int(manifest.get("source_capacity_token_count", -1)) != expected_capacity_tokens:
        raise RuntimeError(
            "graph capacity derivation requires the fixed-row source capacity "
            f"contract: source_capacity_token_count={manifest.get('source_capacity_token_count')!r} "
            f"expected={expected_capacity_tokens}"
        )

    graph_paths = manifest.get("graph_sidecar_paths")
    if not isinstance(graph_paths, dict) or not graph_paths:
        raise RuntimeError("graph capacity derivation requires graph_sidecar_paths")

    sidecar_receipts: dict[str, dict[str, object]] = {}
    max_edges = 0
    chunk_offsets: array | None = None
    max_chunks = 0
    for name, raw_entry in sorted(graph_paths.items()):
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"graph sidecar entry must be an object: {name}")
        offsets_name = raw_entry.get("offsets_path")
        if not isinstance(offsets_name, str) or not offsets_name:
            raise RuntimeError(f"graph sidecar lacks offsets_path: {name}")
        item_count = int(raw_entry.get("item_count", -1))
        if item_count < 0:
            raise RuntimeError(f"graph sidecar lacks nonnegative item_count: {name}")
        offsets_path = _safe_prefix_file(data_prefix.parent, offsets_name)
        receipt, offsets = _csr_offsets_receipt(
            offsets_path,
            document_count=document_count,
            item_count=item_count,
            offset_dtype=raw_entry.get("offset_dtype"),
        )
        sidecar_receipts[name] = receipt
        maximum = int(receipt["max_items_per_document"])
        if raw_entry.get("kind") in _GRAPH_EDGE_KINDS:
            max_edges = max(max_edges, maximum)
        if name in _GRAPH_CHUNK_SIDECARS:
            if chunk_offsets is None:
                chunk_offsets = offsets
                max_chunks = maximum
            elif offsets != chunk_offsets:
                raise RuntimeError(
                    "chunk graph CSR offsets disagree across starts/ends/kinds/dep-levels"
                )

    missing_chunks = sorted(set(_GRAPH_CHUNK_SIDECARS) - set(sidecar_receipts))
    if missing_chunks:
        raise RuntimeError(f"graph capacity derivation lacks chunk sidecars: {missing_chunks}")
    if not any(
        isinstance(entry, dict) and entry.get("kind") in _GRAPH_EDGE_KINDS
        for entry in graph_paths.values()
    ):
        raise RuntimeError("graph capacity derivation lacks edge sidecars")

    return {
        "schema": "cppmega_graph_capacity_v1",
        "status": "verified",
        "data_prefix": str(data_prefix),
        "prefix_manifest_sha256": _sha256(data_prefix.with_suffix(".json")),
        "sequence_length": sequence_length,
        "document_count": document_count,
        "source_capacity_token_count": expected_capacity_tokens,
        "graph_max_edges": max(1, max_edges),
        "graph_max_chunks": max(1, max_chunks),
        "derivation": "max_per_fixed_capacity_document_from_csr_offsets_v1",
        "sidecars": sidecar_receipts,
    }


def derive_graph_capacity_receipt(
    data_prefix: Path,
    *,
    sequence_length: int,
) -> dict[str, object]:
    """Derive exact graph tensor capacities from a manifest-bound CSR prefix."""
    data_prefix = data_prefix.resolve()
    manifest, _referenced = _validate_prefix_manifest_contract(data_prefix)
    return _derive_graph_capacity_from_manifest(
        data_prefix,
        manifest=manifest,
        sequence_length=sequence_length,
    )


def write_graph_capacity_receipt(
    data_prefix: Path,
    *,
    sequence_length: int,
    output: Path,
) -> dict[str, object]:
    receipt = derive_graph_capacity_receipt(
        data_prefix,
        sequence_length=sequence_length,
    )
    _write_json_atomic(output, receipt)
    return receipt


def _write_wrappers(workdir: Path) -> Path:
    wrapper = workdir / "pretrain_mamba.py"
    wrapper.write_text(
        """from __future__ import annotations
import atexit
import os
import runpy
import sys

from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
from cppmega.megatron.checkpoint_restore_preflight import install_checkpoint_restore_preflight
from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch

apply_te_checkpoint_kwarg_patch()
if os.environ.get('CPPMEGA_DSA_PATCH_ENABLED', '0') == '1':
    from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
    apply_dsa_indexer_fused_patch()
apply_graph_route_attention_bias_patch()
install_checkpoint_restore_preflight()
import cppmega.megatron.structure_dataset_patch  # noqa: F401

@atexit.register
def _cppmega_distributed_shutdown():
    import torch
    import torch.distributed as dist
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@atexit.register
def _cppmega_peak_memory_report():
    import torch
    if torch.cuda.is_available():
        print(
            'CPPMEGA_CUDA_PEAK allocated_gib='
            f'{torch.cuda.max_memory_allocated() / 1024**3:.3f} '
            'reserved_gib='
            f'{torch.cuda.max_memory_reserved() / 1024**3:.3f}',
            flush=True,
        )

_inner = '/opt/megatron-lm/pretrain_mamba.py'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(_inner))
sys.argv[0] = _inner
runpy.run_path(_inner, run_name='__main__')
""",
        encoding="utf-8",
    )
    (workdir / "mamba_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as mamba_builder\n",
        encoding="utf-8",
    )
    (workdir / "hybrid_builders.py").write_text(
        "from cppmega.megatron.mamba_builder import cppmega_mamba_builder as hybrid_builder\n",
        encoding="utf-8",
    )
    return wrapper


def _profile_environment(
    *,
    sequence_length: int,
    micro_batch_size: int,
    fp8_recipe: str,
    graph_max_edges: int | None = None,
    graph_max_chunks: int | None = None,
    enable_dsa_patch: bool = True,
) -> dict[str, str]:
    if graph_max_edges is None or graph_max_chunks is None:
        raise ValueError(
            "H200 preflight requires graph capacities derived from CSR offsets"
        )
    if graph_max_edges <= 0 or graph_max_chunks <= 0:
        raise ValueError("derived graph capacities must be positive")
    profile = get_run_profile("h200_cpp_world_mini")
    profile.training.seq_length = sequence_length
    profile.training.micro_batch_size = micro_batch_size
    profile.training.global_batch_size = micro_batch_size
    profile.precision.fp8_recipe = fp8_recipe
    profile.model.dense = False
    profile.spec_module = "cppmega.megatron.nam56r_full_spec"
    profile.spec_function = "build_cppmega_nam56r_full_stack_spec"
    environment = os.environ.copy()
    environment.update(profile_shell_assignments(profile))
    environment.pop("CPPMEGA_DENSE_GQA", None)
    environment.update(
        {
            "CPPMEGA_STRUCTURE_ENABLED": "1",
            "CPPMEGA_GRAPH_ROUTES_ENABLED": "1",
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS": "0",
            "CPPMEGA_GRAPH_MAX_EDGES": str(graph_max_edges),
            "CPPMEGA_GRAPH_MAX_CHUNKS": str(graph_max_chunks),
            "CPPMEGA_DSA_PATCH_ENABLED": "1" if enable_dsa_patch else "0",
            "CPPMEGA_DSA_GRAPH_AUX_ENABLED": "1",
            "CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED": "1",
            "CPPMEGA_DSA_GRAPH_AUX_WEIGHT": "1",
            "CPPMEGA_DSA_INDEXER_LOSS_COEFF": "0.001",
            "CPPMEGA_DSA_SKIP_INDEXER_LOSS": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_GRAPH_REGISTER": "0",
        }
    )
    return environment


def build_megatron_command(
    *,
    wrapper: Path,
    data_prefix: Path,
    tokenizer_model: Path,
    checkpoint_root: Path,
    sequence_length: int,
    micro_batch_size: int,
    train_iters: int,
    environment: dict[str, str],
    load_checkpoint: bool,
) -> list[str]:
    native_args = shlex.split(environment["NATIVE_ARGS"])
    try:
        variant_index = native_args.index("--experimental-attention-variant")
    except ValueError as exc:
        raise ValueError(
            "production graph auxiliary preflight requires native DSA model args"
        ) from exc
    if native_args[variant_index + 1 : variant_index + 2] != ["dsa"]:
        raise ValueError(
            "production graph auxiliary preflight requires "
            "--experimental-attention-variant dsa"
        )
    coefficient_index = native_args.index("--dsa-indexer-loss-coeff")
    if native_args[coefficient_index + 1 : coefficient_index + 2] != [
        environment["CPPMEGA_DSA_INDEXER_LOSS_COEFF"]
    ]:
        raise ValueError(
            "native DSA indexer coefficient differs from graph auxiliary contract"
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        str(wrapper),
        "--data-path",
        "1.0",
        str(data_prefix),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(tokenizer_model),
        "--vocab-size",
        "65536",
        "--make-vocab-size-divisible-by",
        "128",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--hybrid-layer-pattern",
        environment["HYBRID_LAYER_PATTERN"],
        "--hidden-size",
        environment["CPPMEGA_HIDDEN_SIZE"],
        "--ffn-hidden-size",
        environment["CPPMEGA_FFN_HIDDEN_SIZE"],
        "--num-attention-heads",
        environment["CPPMEGA_NUM_ATTN_HEADS"],
        *native_args,
        "--swiglu",
        "--rotary-base",
        "10000",
        "--seq-length",
        str(sequence_length),
        "--max-position-embeddings",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(micro_batch_size),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        "1",
        "--eval-iters",
        "1",
        "--lr",
        environment["CPPMEGA_LR"],
        "--min-lr",
        environment["CPPMEGA_MIN_LR"],
        "--lr-decay-style",
        "constant",
        "--position-embedding-type",
        "rope",
        "--no-rope-fusion",
        "--normalization",
        "RMSNorm",
        "--disable-bias-linear",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        environment["CPPMEGA_ATTN_BACKEND"],
        "--spec",
        environment["CPPMEGA_SPEC_MODULE"],
        environment["CPPMEGA_SPEC_FUNCTION"],
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl",
        "te",
        "--recompute-granularity",
        "selective",
        "--recompute-modules",
        "mlp",
        "--clip-grad",
        "1.0",
        "--optimizer",
        environment["CPPMEGA_OPTIMIZER"],
        "--rerun-mode",
        "disabled",
        "--save",
        str(checkpoint_root),
        "--save-interval",
        "1",
        "--log-interval",
        "1",
    ]
    if environment.get("CPPMEGA_USE_FLASH_ATTN") == "1":
        command.append("--use-flash-attn")
    if environment.get("CPPMEGA_FP8_RECIPE") == "tensorwise":
        command.extend(
            [
                "--fp8-format",
                environment["CPPMEGA_FP8_FORMAT"],
                "--fp8-recipe",
                "tensorwise",
                "--fp8-amax-history-len",
                "16",
                "--fp8-amax-compute-algo",
                "max",
            ]
        )
    if load_checkpoint:
        command.extend(["--load", str(checkpoint_root)])
    return command


def _claimed_backend_modules(environment: dict[str, str]) -> tuple[str, ...]:
    if environment.get("CPPMEGA_DENSE_GQA", "0") == "1":
        return ()
    mode = environment.get("CPPMEGA_DSA_SPARSE_MODE", "").strip().lower()
    if mode == "tilelang":
        return ("tilelang",)
    if mode == "triton":
        return ("triton",)
    if mode == "tvm":
        return ("tvm",)
    return ()


def _validate_backend_dispatch_receipt(
    receipt: object, *, claims: tuple[str, ...]
) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise RuntimeError("backend dispatch receipt must be an object")
    if receipt.get("schema") != "cppmega_backend_dispatch_v1":
        raise RuntimeError("backend dispatch receipt schema mismatch")
    selected = receipt.get("selected_backend")
    if not isinstance(selected, str) or selected not in claims:
        raise RuntimeError(
            f"actual selected backend {selected!r} is not claimed by profile"
        )
    for phase in ("forward", "backward"):
        evidence = receipt.get(phase)
        if (
            not isinstance(evidence, dict)
            or evidence.get("status") != "passed"
            or evidence.get("finite") is not True
        ):
            raise RuntimeError(f"backend dispatch receipt lacks {phase} evidence")
    numerical = receipt.get("numerical")
    if (
        not isinstance(numerical, dict)
        or numerical.get("status") != "passed"
        or not math.isfinite(float(numerical.get("max_abs_error", math.nan)))
    ):
        raise RuntimeError("backend dispatch receipt lacks numerical evidence")
    return receipt


def _load_backend_dispatch_receipt(
    path: Path,
    *,
    claims: tuple[str, ...],
    receipt_binding: dict[str, object],
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required backend dispatch receipt was not written: {path}")
    receipt = _validate_backend_dispatch_receipt(
        json.loads(path.read_text(encoding="utf-8")), claims=claims
    )
    validate_receipt_binding(
        receipt.get("binding"),
        expected=receipt_binding,
        where=path.name,
    )
    return receipt


def _validate_graph_prior_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise RuntimeError("DSA graph prior receipt must be an object")
    if receipt.get("status") != "verified":
        raise RuntimeError("DSA graph prior receipt is not verified")
    if receipt.get("consumer") != "dsa_indexer":
        raise RuntimeError(
            "DSA graph prior receipt consumer must be exactly dsa_indexer"
        )
    prior = receipt.get("prior")
    if not isinstance(prior, dict) or int(prior.get("nonzero", 0)) <= 0:
        raise RuntimeError("DSA graph prior receipt lacks a nonzero prior")
    return receipt


def _dsa_graph_gradient_evidence(
    text: str,
    *,
    expected_coefficient: float,
) -> dict[str, object]:
    objective_records: list[dict[str, object]] = []
    gradient_records: list[dict[str, object]] = []
    for line in text.splitlines():
        if line.startswith("CPPMEGA_DSA_GRAPH_OBJECTIVE "):
            payload = json.loads(line.split(" ", 1)[1])
            objective_records.append(payload)
        elif line.startswith("CPPMEGA_DSA_INDEXER_GRAD "):
            payload = json.loads(line.split(" ", 1)[1])
            gradient_records.append(payload)

    if not objective_records or not gradient_records:
        raise RuntimeError(
            "H200 preflight log lacks actual DSA graph objective and indexer gradient evidence"
        )
    expected_module = (
        "megatron.core.transformer.experimental_attention_variant.dsa.DSAttention"
    )
    expected_indexer = (
        "megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer"
    )
    coefficients = []
    for record in objective_records:
        if record.get("actual_dsa_module") != expected_module:
            raise RuntimeError("DSA graph receipt does not identify actual DSAttention")
        layer_number = record.get("layer_number")
        if not isinstance(layer_number, int) or layer_number <= 0:
            raise RuntimeError("DSA graph receipt lacks a positive layer number")
        graph_loss = float(record.get("graph_loss", math.nan))
        coefficient = float(record.get("effective_coefficient", math.nan))
        if not math.isfinite(graph_loss) or graph_loss <= 0.0:
            raise RuntimeError("DSA graph receipt lacks a finite positive graph loss")
        if not math.isfinite(coefficient):
            raise RuntimeError("DSA graph receipt has a non-finite coefficient")
        if not math.isclose(coefficient, expected_coefficient, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                "DSA graph receipt coefficient differs from the configured indexer coefficient"
            )
        coefficients.append(coefficient)

    for record in gradient_records:
        if record.get("actual_indexer_module") != expected_indexer:
            raise RuntimeError("DSA gradient receipt does not identify actual DSAIndexer")
        layer_number = record.get("layer_number")
        if not isinstance(layer_number, int) or layer_number <= 0:
            raise RuntimeError("DSA gradient receipt lacks a positive layer number")
        grad_norm = float(record.get("grad_norm", math.nan))
        parameter_norms = record.get("parameter_grad_norms")
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError("DSA indexer receipt lacks a finite positive grad norm")
        if not isinstance(parameter_norms, dict) or not parameter_norms:
            raise RuntimeError("DSA indexer receipt lacks per-parameter grad norms")
        positive_parameter_gradient = False
        for value in parameter_norms.values():
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0.0:
                raise RuntimeError("DSA indexer receipt has an invalid parameter grad norm")
            positive_parameter_gradient |= parsed > 0.0
        if not positive_parameter_gradient:
            raise RuntimeError("DSA indexer receipt has no nonzero parameter gradient")

    objective_layers = [record.get("layer_number") for record in objective_records]
    gradient_layers = [record.get("layer_number") for record in gradient_records]
    if sorted(objective_layers) != sorted(gradient_layers):
        raise RuntimeError("DSA graph and indexer receipts do not cover the same layers")
    return {
        "actual_dsa_modules": sorted(
            {str(record["actual_dsa_module"]) for record in objective_records}
        ),
        "actual_indexer_modules": sorted(
            {str(record["actual_indexer_module"]) for record in gradient_records}
        ),
        "effective_coefficient": coefficients[-1],
        "graph_losses": [float(record["graph_loss"]) for record in objective_records],
        "per_indexer": gradient_records,
    }


def _load_stack_lock(path: Path = STACK_LOCK_PATH) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"STACK.lock must be a regular file: {path}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "H200 preflight requires PyYAML to parse the authoritative STACK.lock"
        ) from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("STACK.lock must decode to an object")
    return payload


def validate_stack_compatibility(
    stack_lock: dict[str, object],
    *,
    python_version: tuple[int, int],
    torch_version: object,
    cuda_runtime: object,
    transformer_engine_version: object,
    imported_modules: Iterable[str],
) -> dict[str, object]:
    base = stack_lock.get("base")
    wheels = stack_lock.get("wheels")
    if not isinstance(base, dict) or not isinstance(wheels, dict):
        raise RuntimeError("STACK.lock lacks base/wheels contracts")
    te = wheels.get("transformer_engine")
    if not isinstance(te, dict):
        raise RuntimeError("STACK.lock lacks transformer_engine contract")

    expected_python = str(base.get("python", ""))
    actual_python = f"{python_version[0]}.{python_version[1]}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"Python version mismatch: runtime={actual_python} STACK.lock={expected_python}"
        )
    expected_torch = str(base.get("torch", ""))
    if str(torch_version) != expected_torch:
        raise RuntimeError(
            f"torch version mismatch: runtime={torch_version!r} "
            f"STACK.lock={expected_torch!r}"
        )
    cuda_image = str(base.get("cuda_image", ""))
    match = re.search(r"cuda:(\d+\.\d+)", cuda_image)
    if match is None:
        raise RuntimeError("STACK.lock base.cuda_image does not encode a CUDA version")
    expected_cuda = match.group(1)
    if str(cuda_runtime) != expected_cuda:
        raise RuntimeError(
            f"CUDA runtime mismatch: torch={cuda_runtime!r} "
            f"STACK.lock={expected_cuda!r}"
        )
    expected_te = str(te.get("version", ""))
    actual_te = str(transformer_engine_version)
    if not (
        actual_te == expected_te
        or actual_te.startswith(expected_te + ".")
        or actual_te.startswith(expected_te + "+")
    ):
        raise RuntimeError(
            "Transformer Engine version mismatch: "
            f"runtime={actual_te!r} STACK.lock={expected_te!r}"
        )
    imported = set(imported_modules)
    missing = sorted(set(STACK_REQUIRED_IMPORTS) - imported)
    if missing:
        raise RuntimeError(f"required H200 extension imports are missing: {missing}")
    return {
        "status": "verified",
        "python": expected_python,
        "torch": expected_torch,
        "cuda_runtime": expected_cuda,
        "transformer_engine": expected_te,
        "required_imports": sorted(STACK_REQUIRED_IMPORTS),
    }


def _stack_report(environment: dict[str, str]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("H200 preflight requires CUDA")
    device = torch.cuda.get_device_properties(0)
    if "H200" not in device.name or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError(
            f"H200 preflight requires an H200 SM90 GPU, got {device.name!r} "
            f"capability={torch.cuda.get_device_capability(0)!r}"
        )
    modules = {}
    base_modules = ("torch", *STACK_REQUIRED_IMPORTS)
    backend_claims = _claimed_backend_modules(environment)
    for name in (*base_modules, *backend_claims):
        module = importlib.import_module(name)
        modules[name] = {
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    stack_lock = _load_stack_lock()
    compatibility = validate_stack_compatibility(
        stack_lock,
        python_version=(sys.version_info.major, sys.version_info.minor),
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        transformer_engine_version=modules["transformer_engine"]["version"],
        imported_modules=modules,
    )
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": sys.version,
        "modules": modules,
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": device.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(device.total_memory),
        },
        "nvidia_smi": nvidia_smi,
        "backend_claims": list(backend_claims),
        "stack_lock": {
            "path": str(STACK_LOCK_PATH),
            "sha256": _sha256(STACK_LOCK_PATH),
            "compatibility": compatibility,
        },
    }


def _iteration_evidence(text: str, *, expected_iteration: int) -> dict[str, object]:
    if not re.search(
        rf"iteration\s+{expected_iteration}/\s*{expected_iteration}", text
    ):
        raise RuntimeError(
            f"H200 preflight log lacks iteration {expected_iteration} completion"
        )

    def last_float(label: str, pattern: str) -> float:
        values = re.findall(pattern, text, flags=re.IGNORECASE)
        if not values:
            raise RuntimeError(f"H200 preflight log lacks {label}")
        try:
            return float(values[-1])
        except ValueError as error:
            raise RuntimeError(f"H200 preflight log has invalid {label}") from error

    loss = last_float("LM loss", r"\blm loss:\s*([^\s|]+)")
    grad_norm = last_float("grad norm", r"\bgrad norm:\s*([^\s|]+)")
    if not math.isfinite(loss) or loss <= 0:
        raise RuntimeError(
            f"H200 preflight requires finite positive LM loss, got {loss}"
        )
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(
            f"H200 preflight requires finite positive grad norm, got {grad_norm}"
        )
    skipped_values = re.findall(
        r"number of skipped iterations:\s*(\d+)", text, flags=re.IGNORECASE
    )
    nan_values = re.findall(
        r"number of nan iterations:\s*(\d+)", text, flags=re.IGNORECASE
    )
    if not skipped_values or int(skipped_values[-1]) != 0:
        raise RuntimeError("H200 preflight log reports skipped iterations")
    if not nan_values or int(nan_values[-1]) != 0:
        raise RuntimeError("H200 preflight log reports NaN iterations")
    return {
        "iteration": expected_iteration,
        "lm_loss": loss,
        "grad_norm": grad_norm,
        "skipped_iterations": 0,
        "nan_iterations": 0,
    }


def write_training_loss_receipt(
    log_path: Path,
    *,
    expected_iteration: int,
    output: Path,
    expected_dsa_coefficient: float | None = None,
) -> dict[str, object]:
    if expected_iteration <= 0:
        raise ValueError("expected_iteration must be positive")
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    evidence = _iteration_evidence(
        log_text,
        expected_iteration=expected_iteration,
    )
    receipt = {
        "schema": "cppmega_h200_training_loss_gate_v1",
        "status": "verified",
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "evidence": evidence,
    }
    if expected_dsa_coefficient is not None:
        receipt["dsa_graph_gradient"] = _dsa_graph_gradient_evidence(
            log_text,
            expected_coefficient=expected_dsa_coefficient,
        )
    _write_json_atomic(output, receipt)
    return receipt


def _checkpoint_load_evidence(
    text: str, *, expected_iteration: int
) -> dict[str, object]:
    pattern = re.compile(
        r"successfully loaded checkpoint from\s+.+?"
        r"(?:\[\s*t\s+\d+\s+p\s+\d+\s*\]\s*)?"
        rf"at iteration\s+{expected_iteration}\b",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            "H200 restore lacks explicit Megatron successful checkpoint load "
            f"at iteration {expected_iteration}"
        )
    return {
        "status": "verified",
        "iteration": expected_iteration,
        "log_evidence": match.group(0),
    }


def _validate_checkpoint_state_restore(
    saved: object, loaded: object
) -> dict[str, object]:
    if not isinstance(saved, dict) or not isinstance(loaded, dict):
        raise RuntimeError("checkpoint state receipts must be objects")
    for receipt, mode in ((saved, "save"), (loaded, "load")):
        if (
            receipt.get("schema") != "cppmega_h200_checkpoint_state_v1"
            or receipt.get("status") != "verified"
            or receipt.get("mode") != mode
            or receipt.get("iteration") != 1
        ):
            raise RuntimeError(f"invalid checkpoint {mode} state receipt")
    saved_fingerprints = saved.get("fingerprints")
    loaded_fingerprints = loaded.get("fingerprints")
    if not isinstance(saved_fingerprints, dict) or not isinstance(
        loaded_fingerprints, dict
    ):
        raise RuntimeError("checkpoint state fingerprints are missing")
    matched: list[str] = []
    for component in ("model", "optimizer", "rng"):
        expected = saved_fingerprints.get(component)
        actual = loaded_fingerprints.get(component)
        if (
            not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or actual != expected
        ):
            raise RuntimeError(
                f"checkpoint restored {component} fingerprint mismatch"
            )
        matched.append(component)
    return {"status": "verified", "matched": matched, "iteration": 1}


def _checkpoint_tree_sha256(checkpoint_root: Path, iteration: int = 1) -> str:
    iteration_root = checkpoint_root / f"iter_{iteration:07d}"
    if not iteration_root.is_dir():
        raise RuntimeError(
            f"checkpoint iteration directory is missing: {iteration_root}"
        )
    records = []
    for path in sorted(item for item in iteration_root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(iteration_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise RuntimeError(f"checkpoint iteration directory is empty: {iteration_root}")
    return canonical_sha256(records)


def _stage_cold_checkpoint(
    *,
    source: Path,
    destination: Path,
    receipt_path: Path,
    receipt_binding: dict[str, object],
) -> dict[str, object]:
    binding = validate_binding_shape(
        receipt_binding, where="cold checkpoint receipt"
    )
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise RuntimeError("cold checkpoint destination must differ from save root")
    if destination.exists():
        raise RuntimeError(f"refusing stale cold checkpoint: {destination}")
    if receipt_path.exists():
        raise RuntimeError(f"refusing stale cold checkpoint receipt: {receipt_path}")
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"checkpoint save root is not a regular directory: {source}")
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"checkpoint save root contains symlinks: {symlinks[0]}")
    checkpoint_sha256 = _checkpoint_tree_sha256(source, iteration=1)
    if checkpoint_sha256 != binding["checkpoint_sha256"]:
        raise RuntimeError("cold checkpoint source hash does not match receipt binding")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.partial-{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"refusing stale cold checkpoint staging tree: {staging}")
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2)
        latest = staging / "latest_checkpointed_iteration.txt"
        if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != "1":
            raise RuntimeError("cold checkpoint staging tree does not select iteration 1")
        staged_sha256 = _checkpoint_tree_sha256(staging, iteration=1)
        if staged_sha256 != checkpoint_sha256:
            raise RuntimeError("cold checkpoint staging hash mismatch")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    payload: dict[str, object] = {
        "schema": "cppmega_h200_cold_checkpoint_v1",
        "status": "verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "checkpoint_sha256": checkpoint_sha256,
        "binding": binding,
    }
    _write_json_atomic(receipt_path, payload)
    return payload


def _bundle_identity(
    bundle_root: Path, data_prefix: Path, *, hash_jobs: int
) -> tuple[dict[str, object], dict[str, str]]:
    root = bundle_root.resolve()
    manifest, _records = _validate_bundle(root, hash_jobs)
    selected = data_prefix.resolve()
    try:
        selected.relative_to(root)
    except ValueError as error:
        raise RuntimeError("H200 data prefix escapes explicit bundle root") from error
    prefix_hashes: dict[str, str] = {}
    found = False
    for result in manifest["bucket_results"]:
        prefix = (root / str(result["prefix"])).resolve()
        manifest_path = prefix.with_suffix(".json")
        relative = manifest_path.relative_to(root).as_posix()
        prefix_hashes[relative] = _sha256(manifest_path)
        if prefix == selected:
            found = True
    if not found:
        raise RuntimeError("H200 data prefix is not declared by bundle bucket_results")
    return manifest, dict(sorted(prefix_hashes.items()))


def _finalize_bound_receipt(
    path: Path, *, expected_pending: dict[str, object], final: dict[str, object]
) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required receipt was not written: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt_binding(
        payload.get("binding"),
        expected=expected_pending,
        where=path.name,
    )
    payload["binding"] = final
    _write_json_atomic(path, payload)
    return payload


def _run_phase(
    *,
    name: str,
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    batch_receipt: Path,
    graph_prior_receipt: Path,
    checkpoint_state_receipt: Path,
    backend_dispatch_receipt: Path,
    backend_claims: tuple[str, ...],
    expected_iteration: int,
    expected_loaded_iteration: int | None,
    checkpoint_root: Path,
    receipt_binding: dict[str, object],
) -> dict[str, object]:
    required_receipts = [batch_receipt, graph_prior_receipt, checkpoint_state_receipt]
    if backend_claims:
        required_receipts.append(backend_dispatch_receipt)
    for path in required_receipts:
        path.unlink(missing_ok=True)
    phase_environment = dict(environment)
    phase_environment["CPPMEGA_H200_BATCH_RECEIPT"] = str(batch_receipt)
    phase_environment["CPPMEGA_H200_GRAPH_PRIOR_RECEIPT"] = str(
        graph_prior_receipt
    )
    phase_environment["CPPMEGA_H200_CHECKPOINT_STATE_RECEIPT"] = str(
        checkpoint_state_receipt
    )
    phase_environment["CPPMEGA_H200_DSA_GRAPH_RECEIPTS"] = "1"
    if backend_claims:
        phase_environment["CPPMEGA_H200_BACKEND_DISPATCH_RECEIPT"] = str(
            backend_dispatch_receipt
        )
    phase_environment["CPPMEGA_H200_EXPECTED_LOAD_ITERATION"] = "1"
    phase_environment["CPPMEGA_H200_CHECKPOINT_PROOF_MODE"] = (
        "restore" if expected_loaded_iteration is not None else "save"
    )
    phase_environment["CPPMEGA_H200_RECEIPT_BINDING"] = json.dumps(
        receipt_binding, sort_keys=True
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            env=phase_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"H200 preflight {name} phase failed with exit {result.returncode}"
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iteration_evidence = _iteration_evidence(
        text, expected_iteration=expected_iteration
    )
    dsa_graph_gradient = _dsa_graph_gradient_evidence(
        text,
        expected_coefficient=float(environment["CPPMEGA_DSA_INDEXER_LOSS_COEFF"]),
    )
    load_evidence = (
        _checkpoint_load_evidence(
            text, expected_iteration=expected_loaded_iteration
        )
        if expected_loaded_iteration is not None
        else None
    )
    latest = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != str(
        expected_iteration
    ):
        raise RuntimeError(
            f"H200 preflight {name} checkpoint did not reach {expected_iteration}"
        )
    if not batch_receipt.is_file():
        raise RuntimeError(f"H200 preflight {name} did not record a production batch")
    batch = json.loads(batch_receipt.read_text(encoding="utf-8"))
    active_graph = batch.get("active_graph")
    source_provenance = batch.get("source_provenance")
    objective_mix = batch.get("objective_mix")
    if (
        batch.get("status") != "verified"
        or not isinstance(active_graph, dict)
        or int(active_graph.get("route_edge_count", 0)) <= 0
        or not isinstance(source_provenance, dict)
        or int(source_provenance.get("minimum_source_doc_id", 0)) <= 0
        or not isinstance(objective_mix, dict)
        or not objective_mix.get("observed_objective_ids")
    ):
        raise RuntimeError(
            f"H200 preflight {name} production batch is not verified or lacks "
            "objective mix accounting"
        )
    if not graph_prior_receipt.is_file():
        raise RuntimeError(
            f"H200 preflight {name} did not record graph-prior consumption"
        )
    graph_prior = json.loads(graph_prior_receipt.read_text(encoding="utf-8"))
    _validate_graph_prior_receipt(graph_prior)
    if not checkpoint_state_receipt.is_file():
        raise RuntimeError(
            f"H200 preflight {name} did not record checkpoint runtime state"
        )
    checkpoint_state = json.loads(
        checkpoint_state_receipt.read_text(encoding="utf-8")
    )
    backend_dispatch = None
    if backend_claims:
        backend_dispatch = _load_backend_dispatch_receipt(
            backend_dispatch_receipt,
            claims=backend_claims,
            receipt_binding=receipt_binding,
        )
    expected_mode = "load" if expected_loaded_iteration is not None else "save"
    if (
        checkpoint_state.get("status") != "verified"
        or checkpoint_state.get("mode") != expected_mode
        or checkpoint_state.get("iteration") != 1
    ):
        raise RuntimeError(
            f"H200 preflight {name} checkpoint state receipt is invalid"
        )
    checkpoint_sha256 = _checkpoint_tree_sha256(checkpoint_root, iteration=1)
    final_binding = dict(receipt_binding)
    final_binding["checkpoint_sha256"] = checkpoint_sha256
    if receipt_binding["checkpoint_sha256"] == PENDING_CHECKPOINT_SHA256:
        batch = _finalize_bound_receipt(
            batch_receipt,
            expected_pending=receipt_binding,
            final=final_binding,
        )
        graph_prior = _finalize_bound_receipt(
            graph_prior_receipt,
            expected_pending=receipt_binding,
            final=final_binding,
        )
        checkpoint_state = _finalize_bound_receipt(
            checkpoint_state_receipt,
            expected_pending=receipt_binding,
            final=final_binding,
        )
        if backend_dispatch is not None:
            backend_dispatch = _finalize_bound_receipt(
                backend_dispatch_receipt,
                expected_pending=receipt_binding,
                final=final_binding,
            )
    else:
        bound_receipts = [
            (batch_receipt, batch),
            (graph_prior_receipt, graph_prior),
            (checkpoint_state_receipt, checkpoint_state),
        ]
        if backend_dispatch is not None:
            bound_receipts.append((backend_dispatch_receipt, backend_dispatch))
        for path, payload in bound_receipts:
            validate_receipt_binding(
                payload.get("binding"),
                expected=final_binding,
                where=path.name,
            )
    peak = re.findall(
        r"CPPMEGA_CUDA_PEAK allocated_gib=([0-9.]+) reserved_gib=([0-9.]+)", text
    )
    if not peak:
        raise RuntimeError(f"H200 preflight {name} log lacks CUDA peak memory")
    return {
        "status": "passed",
        "command": command,
        "command_shell": shlex.join(command),
        "log": str(log_path),
        "batch_receipt": str(batch_receipt),
        "graph_prior_receipt": str(graph_prior_receipt),
        "checkpoint_state_receipt": str(checkpoint_state_receipt),
        "backend_dispatch_receipt": (
            str(backend_dispatch_receipt) if backend_dispatch is not None else None
        ),
        "backend_dispatch": backend_dispatch,
        "binding": final_binding,
        "checkpoint_sha256": checkpoint_sha256,
        "completed_iteration": expected_iteration,
        "iteration_evidence": iteration_evidence,
        "dsa_graph_gradient": dsa_graph_gradient,
        "checkpoint_load_evidence": load_evidence,
        "forward_backward_numerical": {
            "status": "passed",
            "finite_lm_loss": iteration_evidence["lm_loss"],
            "finite_grad_norm": iteration_evidence["grad_norm"],
        },
        "cuda_peak_allocated_gib": float(peak[-1][0]),
        "cuda_peak_reserved_gib": float(peak[-1][1]),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--data-prefix", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise"), default="off")
    parser.add_argument(
        "--enable-dsa-patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install the required fused DSA graph-objective patch.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/data/cppmega_h200_preflight_checkpoint"),
    )
    parser.add_argument("--cold-checkpoint-root", type=Path)
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = build_arg_parser().parse_args(raw_argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise ValueError("H200 preflight run_id is not a safe identifier")
    if args.sequence_length <= 0 or args.micro_batch_size <= 0 or args.hash_jobs <= 0:
        raise ValueError("sequence length, micro batch size, and hash jobs must be positive")

    bundle_root = args.bundle_root.resolve()
    data_prefix = args.data_prefix.resolve()
    tokenizer_model = args.tokenizer_model.resolve()
    manifest, prefix_manifest_hashes = _bundle_identity(
        bundle_root, data_prefix, hash_jobs=args.hash_jobs
    )
    tokenizer_descriptor = manifest["tokenizer"]
    expected_tokenizer = (bundle_root / str(tokenizer_descriptor["path"])).resolve()
    if tokenizer_model != expected_tokenizer:
        raise RuntimeError(
            "H200 tokenizer must be the descriptor-bound tokenizer inside bundle root"
        )
    _validate_tokenizer_directory(tokenizer_model)
    prefix_manifest, _referenced = _validate_prefix_manifest_contract(data_prefix)
    graph_capacity = derive_graph_capacity_receipt(
        data_prefix,
        sequence_length=args.sequence_length,
    )
    environment = _profile_environment(
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        fp8_recipe=args.fp8_recipe,
        graph_max_edges=int(graph_capacity["graph_max_edges"]),
        graph_max_chunks=int(graph_capacity["graph_max_chunks"]),
        enable_dsa_patch=args.enable_dsa_patch,
    )
    objective_descriptor = prefix_manifest.get("objective_contract")
    objective_payload = (
        objective_descriptor.get("payload")
        if isinstance(objective_descriptor, dict)
        else None
    )
    graph_contract = (
        objective_payload.get("graph_auxiliary")
        if isinstance(objective_payload, dict)
        else None
    )
    if not isinstance(graph_contract, dict):
        raise ValueError("H200 preflight requires graph_auxiliary objective contract")
    if environment["CPPMEGA_DSA_PATCH_ENABLED"] != "1":
        raise ValueError(
            "graph_auxiliary.included_in_total_loss requires the fused DSA patch "
            "before model construction"
        )
    validate_runtime_graph_contract(
        graph_contract,
        environment=environment,
        require_included_auxiliary=True,
    )
    backend_claims = _claimed_backend_modules(environment)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing stale H200 preflight receipt: {output}")
    graph_capacity_output = output.with_name(
        f"{output.stem}_graph_capacity{output.suffix or '.json'}"
    )
    if graph_capacity_output.exists():
        raise RuntimeError(
            f"refusing stale H200 graph capacity receipt: {graph_capacity_output}"
        )
    _write_json_atomic(graph_capacity_output, graph_capacity)
    checkpoint_root = args.checkpoint_root.resolve()
    cold_checkpoint_root = (
        args.cold_checkpoint_root.resolve()
        if args.cold_checkpoint_root is not None
        else checkpoint_root.with_name(f"{checkpoint_root.name}-cold")
    )
    if checkpoint_root.exists():
        raise RuntimeError(
            f"refusing stale H200 preflight checkpoint root: {checkpoint_root}"
        )
    if cold_checkpoint_root.exists():
        raise RuntimeError(
            f"refusing stale H200 cold checkpoint root: {cold_checkpoint_root}"
        )

    config = {
        "schema": "cppmega_h200_megatron_preflight_config_v1",
        "profile": "h200_cpp_world_mini",
        "bundle_root": str(bundle_root),
        "data_prefix": data_prefix.relative_to(bundle_root).as_posix(),
        "tokenizer": tokenizer_model.relative_to(bundle_root).as_posix(),
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "fp8_recipe": args.fp8_recipe,
        "enable_dsa_patch": args.enable_dsa_patch,
        "graph_max_edges": graph_capacity["graph_max_edges"],
        "graph_max_chunks": graph_capacity["graph_max_chunks"],
        "checkpoint_root": str(checkpoint_root),
        "cold_checkpoint_root": str(cold_checkpoint_root),
    }
    identity = {
        "bundle_id": str(manifest["bundle_id"]),
        "artifact_set_sha256": str(manifest["artifact_set_sha256"]),
        "prefix_manifest_sha256s": prefix_manifest_hashes,
    }

    with tempfile.TemporaryDirectory(prefix="cppmega-h200-preflight-") as raw_workdir:
        workdir = Path(raw_workdir)
        wrapper = _write_wrappers(workdir)
        save_command = build_megatron_command(
            wrapper=wrapper,
            data_prefix=data_prefix,
            tokenizer_model=tokenizer_model,
            checkpoint_root=checkpoint_root,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            train_iters=1,
            environment=environment,
            load_checkpoint=False,
        )
        restore_command = build_megatron_command(
            wrapper=wrapper,
            data_prefix=data_prefix,
            tokenizer_model=tokenizer_model,
            checkpoint_root=cold_checkpoint_root,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            train_iters=2,
            environment=environment,
            load_checkpoint=True,
        )
        invocation_command = [str(Path(__file__).resolve()), *raw_argv]
        save_binding = build_receipt_binding(
            **identity,
            checkpoint_sha256=PENDING_CHECKPOINT_SHA256,
            config=config,
            command=save_command,
            run_id=args.run_id,
        )
        restore_pending_binding = build_receipt_binding(
            **identity,
            checkpoint_sha256=PENDING_CHECKPOINT_SHA256,
            config=config,
            command=restore_command,
            run_id=args.run_id,
        )
        top_pending_binding = build_receipt_binding(
            **identity,
            checkpoint_sha256=PENDING_CHECKPOINT_SHA256,
            config=config,
            command=invocation_command,
            run_id=args.run_id,
        )
        base_receipt: dict[str, object] = {
            "schema": "cppmega_h200_megatron_preflight_v1",
            "status": "dry_run" if args.dry_run else "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "bundle": {"root": str(bundle_root), **identity},
            "binding": top_pending_binding,
            "phase_binding_templates": {
                "save": save_binding,
                "restore": restore_pending_binding,
            },
            "config": config,
            "data": {
                "prefix": str(data_prefix),
                "manifest": prefix_manifest,
                "sequence_length": args.sequence_length,
                "micro_batch_size": args.micro_batch_size,
                "graph_capacity": graph_capacity,
                "graph_capacity_receipt": str(graph_capacity_output),
            },
            "checkpoint": {
                "root": str(checkpoint_root),
                "cold_root": str(cold_checkpoint_root),
                "save_iteration": 1,
                "restored_from_iteration": 1,
                "post_restore_iteration": 2,
                "full_optimizer_and_rng_state": True,
            },
            "commands": {
                "invocation": invocation_command,
                "save": save_command,
                "restore": restore_command,
            },
            "backend_dispatch": {
                "status": "required" if backend_claims else "not_claimed",
                "claimed_backends": list(backend_claims),
            },
        }
        if args.dry_run:
            _write_json_atomic(output, base_receipt)
            print(json.dumps(base_receipt, indent=2, sort_keys=True))
            return 0

        stack = _stack_report(environment)
        try:
            save_state_receipt = output.parent / "h200_preflight_save_state.json"
            save = _run_phase(
                name="save",
                command=save_command,
                environment=environment,
                log_path=output.parent / "h200_preflight_save.log",
                batch_receipt=output.parent / "h200_preflight_save_batch.json",
                graph_prior_receipt=output.parent
                / "h200_preflight_save_graph_prior.json",
                checkpoint_state_receipt=save_state_receipt,
                backend_dispatch_receipt=output.parent
                / "h200_preflight_save_backend_dispatch.json",
                backend_claims=backend_claims,
                expected_iteration=1,
                expected_loaded_iteration=None,
                checkpoint_root=checkpoint_root,
                receipt_binding=save_binding,
            )
            checkpoint_sha256 = str(save["checkpoint_sha256"])
            cold_command = [
                "cppmega-stage-cold-checkpoint",
                str(checkpoint_root),
                str(cold_checkpoint_root),
            ]
            cold_binding = build_receipt_binding(
                **identity,
                checkpoint_sha256=checkpoint_sha256,
                config=config,
                command=cold_command,
                run_id=args.run_id,
            )
            cold = _stage_cold_checkpoint(
                source=checkpoint_root,
                destination=cold_checkpoint_root,
                receipt_path=output.parent / "h200_preflight_cold_checkpoint.json",
                receipt_binding=cold_binding,
            )
            restore_binding = build_receipt_binding(
                **identity,
                checkpoint_sha256=checkpoint_sha256,
                config=config,
                command=restore_command,
                run_id=args.run_id,
            )
            restore_state_receipt = output.parent / "h200_preflight_restore_state.json"
            restore = _run_phase(
                name="restore",
                command=restore_command,
                environment=environment,
                log_path=output.parent / "h200_preflight_restore.log",
                batch_receipt=output.parent / "h200_preflight_restore_batch.json",
                graph_prior_receipt=output.parent
                / "h200_preflight_restore_graph_prior.json",
                checkpoint_state_receipt=restore_state_receipt,
                backend_dispatch_receipt=output.parent
                / "h200_preflight_restore_backend_dispatch.json",
                backend_claims=backend_claims,
                expected_iteration=2,
                expected_loaded_iteration=1,
                checkpoint_root=cold_checkpoint_root,
                receipt_binding=restore_binding,
            )
            if restore["checkpoint_sha256"] != checkpoint_sha256:
                raise RuntimeError("cold-restored iteration-1 checkpoint hash drifted")
            restore_proof = _validate_checkpoint_state_restore(
                json.loads(save_state_receipt.read_text(encoding="utf-8")),
                json.loads(restore_state_receipt.read_text(encoding="utf-8")),
            )
            top_binding = build_receipt_binding(
                **identity,
                checkpoint_sha256=checkpoint_sha256,
                config=config,
                command=invocation_command,
                run_id=args.run_id,
            )
        except Exception as error:
            _write_json_atomic(
                output,
                {
                    **base_receipt,
                    "status": "failed",
                    "stack": stack,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        receipt = {
            **base_receipt,
            "status": "passed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "stack": stack,
            "binding": top_binding,
            "checkpoint": {
                **base_receipt["checkpoint"],
                "sha256": checkpoint_sha256,
                "restore_proof": restore_proof,
            },
            "commands": {
                **base_receipt["commands"],
                "cold_checkpoint": cold_command,
            },
            "phases": {"save": save, "cold_restore": cold, "restore": restore},
        }
        _write_json_atomic(output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
