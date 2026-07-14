#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate C/C++ eval completions from a Megatron checkpoint on one Nebius H200.

This runner is deliberately separate from the training sweep.  It uploads the
current cppmega overlay, tokenizer, eval prompts/cases, and a local Megatron
checkpoint root, then runs an inference-only Megatron process in the GHCR image.
The generated completions are copied back and evaluated locally with the
compile/run gate from ``scripts/cpp_generation_compile_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from scripts.nebius_h200_megatron_cpp_world_sweep import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_IMAGE_ID,
    DEFAULT_PARENT_ID,
    DEFAULT_SECURITY_GROUP_ID,
    DEFAULT_SUBNET_ID,
    DEFAULT_TOKENIZER_DIR,
    create_instance,
    default_ssh_key,
    make_ghcr_auth_tar,
    make_overlay_tar,
    run,
    ssh,
    ssh_base,
    stream_tar_to_remote,
    wait_for_ip,
    wait_for_ssh,
)
from cppmega.prompt_graph import (
    PromptProjectIndex,
    require_prompt_graph_project_id,
)
from cppmega.prompt_graph_index import ClangPromptProjectIndexProducer


ROOT = _SCRIPT_ROOT
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "cppmega-h200-megatron-1782697038"
    / "seq_1024_bs_192"
)
DEFAULT_CASES = ROOT / "evals" / "cpp_docstring_compile_cases.jsonl"
DEFAULT_PROMPTS = ROOT / "outputs" / "evals" / "cpp_docstring_compile_prompts.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def _contained_path(root: Path, raw: object, *, where: str) -> tuple[Path, Path]:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{where} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{where} must be a contained relative path, got {raw!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{where} escapes {root}: {raw!r}") from exc
    return path, relative


def effective_case_prompt_graph_mode(
    row: dict[str, object],
    global_mode: str,
) -> str:
    if global_mode not in {"repo", "off"}:
        raise ValueError(f"unsupported prompt graph mode {global_mode!r}")
    task_id = str(row.get("task_id", "<missing-task-id>"))
    case_mode = row.get("prompt_graph_mode")
    if case_mode not in {"repo", "off"}:
        raise ValueError(
            f"{task_id}.prompt_graph_mode must be explicitly 'repo' or 'off'"
        )
    return "off" if global_mode == "off" else str(case_mode)


@dataclass(frozen=True)
class PreparedEvalGraphCases:
    rows: tuple[dict[str, object], ...]
    assets: dict[Path, Path]


def _prepare_eval_graph_cases(
    cases: Path,
    *,
    prompt_graph_mode: str,
    prompt_index_cache_dir: Path,
    indexer_root: Path | None = None,
) -> PreparedEvalGraphCases:
    if prompt_graph_mode not in {"repo", "off"}:
        raise ValueError(f"unsupported prompt graph mode {prompt_graph_mode!r}")
    if not cases.is_file():
        raise FileNotFoundError(cases)

    root = cases.resolve().parent
    assets: dict[Path, Path] = {}
    staged_rows: list[dict[str, object]] = []
    for original_row in iter_jsonl(cases):
        row = dict(original_row)
        task_id = str(row.get("task_id", "<missing-task-id>"))
        case_mode = effective_case_prompt_graph_mode(row, prompt_graph_mode)
        row["prompt_graph_mode"] = case_mode
        if case_mode == "off":
            staged_rows.append(row)
            continue

        repo_path, _repo_relative = _contained_path(
            root,
            row.get("prompt_graph_repo"),
            where=f"{task_id}.prompt_graph_repo",
        )
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"{task_id}: prompt graph repository not found: {repo_path}"
            )
        project_id = require_prompt_graph_project_id(
            row.get("prompt_graph_project_id"),
            where=f"{task_id}.prompt_graph_project_id",
        )
        raw_index = row.get("prompt_graph_index")
        if isinstance(raw_index, str) and raw_index:
            index_path, _index_relative = _contained_path(
                root,
                raw_index,
                where=f"{task_id}.prompt_graph_index",
            )
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"{task_id}: prompt graph index not found: {index_path}"
                )
            project_index = PromptProjectIndex.from_json_path(index_path)
            if project_index.project_id != project_id:
                raise ValueError(
                    f"{task_id}: prompt graph project_id mismatch: "
                    f"case={project_id!r} index={project_index.project_id!r}"
                )
            project_index.verify_repository(repo_path)
            index_receipt = dict(project_index.provenance)
        else:
            built = ClangPromptProjectIndexProducer(
                cache_dir=prompt_index_cache_dir,
                indexer_root=indexer_root,
                strict_diagnostics=True,
            ).build(repo_path, project_id=project_id)
            project_index = built.index
            index_path = built.path
            index_receipt = dict(built.receipt)

        raw_source_path = row.get("prompt_source_path")
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ValueError(
                f"{task_id}.prompt_source_path must be a non-empty relative path"
            )
        source_document = project_index.document_for_path(raw_source_path)
        source_start = row.get("prompt_source_start")
        if isinstance(source_start, bool) or not isinstance(source_start, int):
            raise ValueError(f"{task_id}.prompt_source_start must be an integer")
        prompt = row.get("source_prefix")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{task_id}.source_prefix must be a non-empty string")
        prompt_end = source_start + len(prompt)
        if (
            source_start < 0
            or source_document.source[source_start:prompt_end] != prompt
        ):
            raise ValueError(
                f"{task_id}: source_prefix does not match indexed document "
                f"{source_document.source_path!r} at prompt_source_start={source_start}"
            )

        cache_key = str(index_receipt.get("cache_key") or project_index.index_sha256)
        stage_root = Path("prompt_graph") / cache_key
        index_relative = stage_root / "project_index.json"
        row["prompt_graph_repo"] = stage_root.as_posix()
        row["prompt_graph_index"] = index_relative.as_posix()
        row["prompt_graph_project_id"] = project_id
        row["prompt_graph_index_receipt"] = index_receipt
        candidates: list[tuple[Path, Path]] = [(index_relative, index_path)]
        manifest = index_receipt.get("repository_manifest")
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError(
                f"{task_id}: prompt graph index lacks repository_manifest provenance"
            )
        for source_relative in sorted(manifest):
            source_path, normalized = _contained_path(
                repo_path,
                source_relative,
                where=f"{task_id}.repository_manifest[{source_relative!r}]",
            )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            candidates.append((stage_root / normalized, source_path))
        for relative, asset in candidates:
            previous = assets.get(relative)
            if previous is not None and previous != asset:
                raise ValueError(
                    f"conflicting eval graph assets for {relative}: {previous} != {asset}"
                )
            assets[relative] = asset
        staged_rows.append(row)
    return PreparedEvalGraphCases(rows=tuple(staged_rows), assets=assets)


def eval_graph_assets(
    cases: Path,
    *,
    prompt_graph_mode: str,
    indexer_root: Path | None,
) -> dict[Path, Path]:
    cache_dir = Path(tempfile.gettempdir()) / "cppmega-prompt-index-cache"
    return _prepare_eval_graph_cases(
        cases,
        prompt_graph_mode=prompt_graph_mode,
        prompt_index_cache_dir=cache_dir,
        indexer_root=indexer_root,
    ).assets


def make_eval_tar(
    cases: Path,
    prompts: Path,
    path: Path,
    *,
    prompt_graph_mode: str,
    indexer_root: Path | None,
) -> None:
    for item in (cases, prompts):
        if not item.exists():
            raise FileNotFoundError(item)
    with tempfile.TemporaryDirectory(prefix="cppmega-eval-stage-") as stage_raw:
        stage = Path(stage_raw)
        eval_stage = stage / "cppmega_eval"
        eval_stage.mkdir()
        prepared = _prepare_eval_graph_cases(
            cases,
            prompt_graph_mode=prompt_graph_mode,
            prompt_index_cache_dir=stage / "prompt_index_cache",
            indexer_root=indexer_root,
        )
        (eval_stage / "cases.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in prepared.rows
            ),
            encoding="utf-8",
        )
        os.symlink(prompts.resolve(), eval_stage / "prompts.jsonl")
        for relative, source in sorted(
            prepared.assets.items(), key=lambda item: str(item[0])
        ):
            target = eval_stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source, target)
        cmd = ["tar", "-czhf", str(path), "-C", str(stage), "cppmega_eval"]
        env = {**os.environ, "GZIP": "-1", "COPYFILE_DISABLE": "1"}
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[nebius-generation] $ GZIP=-1 COPYFILE_DISABLE=1 {printable}", flush=True)
        subprocess.run(cmd, check=True, env=env)


def make_tokenizer_tar(tokenizer_dir: Path, path: Path) -> None:
    if not tokenizer_dir.exists():
        raise FileNotFoundError(tokenizer_dir)
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"{tokenizer_dir} missing tokenizer.json")
    with tempfile.TemporaryDirectory(prefix="cppmega-tokenizer-stage-") as stage_raw:
        stage = Path(stage_raw)
        tok_stage = stage / "cpp_tokenizer_hf"
        tok_stage.mkdir()
        for item in sorted(tokenizer_dir.iterdir()):
            if item.is_file():
                os.symlink(item.resolve(), tok_stage / item.name)
        cmd = ["tar", "-czhf", str(path), "-C", str(stage), "cpp_tokenizer_hf"]
        env = {**os.environ, "GZIP": "-1", "COPYFILE_DISABLE": "1"}
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[nebius-generation] $ GZIP=-1 COPYFILE_DISABLE=1 {printable}", flush=True)
        subprocess.run(cmd, check=True, env=env)


def make_checkpoint_plain_tar(checkpoint_dir: Path, path: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    latest_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} does not look like a Megatron checkpoint root: "
            "missing latest_checkpointed_iteration.txt"
        )
    latest_raw = latest_path.read_text(encoding="utf-8").strip()
    latest_dir = f"iter_{int(latest_raw):07d}" if latest_raw.isdigit() else f"iter_{latest_raw}"
    if not (checkpoint_dir / latest_dir).is_dir():
        raise FileNotFoundError(
            f"{checkpoint_dir} latest iteration {latest_raw!r} points to missing {latest_dir}"
        )
    cmd = [
        "tar",
        "-cf",
        str(path),
        "-C",
        str(checkpoint_dir),
        "latest_checkpointed_iteration.txt",
        latest_dir,
    ]
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[nebius-generation] $ COPYFILE_DISABLE=1 {printable}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def stream_plain_tar_to_remote(args: argparse.Namespace, ip: str, tar_path: Path, target: str) -> None:
    cmd = f"mkdir -p {shlex.quote(target)} && tar -xf - -C {shlex.quote(target)}"
    ssh_cmd = ssh_base(args, ip) + [cmd]
    printable = " ".join(shlex.quote(part) for part in ssh_cmd)
    print(f"[nebius-generation] streaming {tar_path.name} -> {target}: {printable}", flush=True)
    with tar_path.open("rb") as f:
        subprocess.run(ssh_cmd, stdin=f, check=True)


def generation_worker_source() -> str:
    return r'''
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch

from cppmega.megatron.dsa_indexer_fused_patch import apply_dsa_indexer_fused_patch
from cppmega.megatron.graph_route_attention_bias_patch import (
    PromptGraphInferenceState,
    apply_graph_route_attention_bias_patch,
    set_prompt_graph_inference_state,
)
from cppmega.megatron.structure_dataset_patch import _set_current_structure_batch
from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch
from cppmega.prompt_graph import (
    CppPromptTokenizerAdapter,
    GENERATED_QUERY_COUNT_KEY,
    GENERATED_QUERY_ROUTE_KEY,
    PAIR_ROUTE_KEYS,
    TOKEN_SIDECAR_NAMES,
    TRIPLE_ROUTE_KEYS,
    PromptGraphBuilder,
    PromptGraphContext,
    PromptProjectIndex,
    require_prompt_graph_project_id,
)


OPAQUE_SYMBOL_ID_SIDECARS = frozenset(
    {"symbol_ids", "call_targets", "type_refs"}
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            yield row


def contained_eval_path(root: Path, raw: object, *, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{where} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{where} must be a contained relative path, got {raw!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{where} escapes {root}: {raw!r}") from exc
    return path


def effective_case_graph_mode(case: dict[str, object], global_mode: str) -> str:
    case_mode = case.get("prompt_graph_mode")
    if case_mode not in {"repo", "off"}:
        raise ValueError(
            f"{case.get('task_id', '<missing>')}.prompt_graph_mode must be "
            "explicitly 'repo' or 'off'"
        )
    return "off" if global_mode == "off" else str(case_mode)


def load_tokenizer(tokenizer_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tok.eos_token_id is None:
        tok.eos_token_id = 3
    tokenizer_json = Path(tokenizer_dir) / "tokenizer.json"
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"{tokenizer_json}: missing model.vocab for cppmega decode")
    tok._cppmega_id_to_token = {int(token_id): token for token, token_id in vocab.items()}
    return tok


def decode(tokenizer, ids: list[int]) -> str:
    id_to_token = getattr(tokenizer, "_cppmega_id_to_token", None)
    if id_to_token is None:
        raise ValueError("cppmega tokenizer is missing id_to_token decode table")
    text = "".join(id_to_token.get(int(token_id), "") for token_id in ids)
    return (
        text.replace("<SPACE>", " ")
        .replace("<RESERVED_46>", " ")
        .replace("<NL>", "\n")
        .replace("<RESERVED_47>", "\n")
    )


def trim_body_completion(text: str) -> str:
    stripped = text.replace("\r\n", "\n")
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            first_newline = stripped.find("\n")
            if first_newline != -1 and stripped[:first_newline].strip().isidentifier():
                stripped = stripped[first_newline + 1 :]
    stop_markers = (
        "int main(",
        "#include ",
        "```",
        "<|endoftext|>",
        "<BOS>",
        "<EOS>",
        "<CODE_START>",
        "<CODE_END>",
        "<FIM_",
        "<RESERVED_",
    )
    for marker in stop_markers:
        pos = stripped.find(marker)
        if pos >= 0:
            stripped = stripped[:pos]
    stripped = _trim_at_function_closing_brace(stripped)
    body = stripped.strip()
    return body + ("\n" if body else "")


def _trim_at_function_closing_brace(text: str) -> str:
    """Drop only the brace that closes the prompt's already-open function."""
    depth = 1
    index = 0
    state = "code"
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                index += 2
            elif char == quote:
                state = "code"
                index += 1
            else:
                index += 1
            continue

        if char == "/" and following == "/":
            state = "line-comment"
            index += 2
            continue
        if char == "/" and following == "*":
            state = "block-comment"
            index += 2
            continue
        if char == "R" and following == '"':
            delimiter_end = text.find("(", index + 2, min(len(text), index + 20))
            if delimiter_end >= 0:
                delimiter = text[index + 2 : delimiter_end]
                if len(delimiter) <= 16 and not any(
                    item.isspace() or item in "()\\" for item in delimiter
                ):
                    terminator = ")" + delimiter + '"'
                    raw_end = text.find(terminator, delimiter_end + 1)
                    if raw_end < 0:
                        return text
                    index = raw_end + len(terminator)
                    continue
        if char == '"':
            state = "string"
            index += 1
            continue
        if char == "'":
            state = "character"
            index += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[:index]
        index += 1
    return text


def build_prompt_rows(
    cases_path: Path,
    prompts_path: Path,
    prompt_mode: str,
    prompt_graph_mode: str,
) -> list[dict[str, object]]:
    if prompt_graph_mode not in {"repo", "off"}:
        raise ValueError(f"unknown prompt graph mode {prompt_graph_mode!r}")
    cases = {str(row["task_id"]): row for row in iter_jsonl(cases_path)}
    prompts = {str(row["task_id"]): row for row in iter_jsonl(prompts_path)}
    if set(cases) != set(prompts):
        raise ValueError(
            "cases/prompts task_id mismatch: "
            f"cases_only={sorted(set(cases) - set(prompts))[:5]} "
            f"prompts_only={sorted(set(prompts) - set(cases))[:5]}"
        )
    rows: list[dict[str, object]] = []
    for task_id in sorted(cases):
        case = cases[task_id]
        case_graph_mode = effective_case_graph_mode(case, prompt_graph_mode)
        if case_graph_mode == "repo" and prompt_mode != "source-prefix":
            raise ValueError(
                f"{task_id}: repository prompt graph requires prompt_mode=source-prefix"
            )
        prompt_row = prompts[task_id]
        if prompt_mode == "source-prefix":
            prompt_text = str(case["source_prefix"])
        elif prompt_mode == "instruction":
            prompt_text = str(prompt_row["prompt"])
        else:
            raise ValueError(f"unknown prompt mode {prompt_mode!r}")
        row: dict[str, object] = {
            "task_id": task_id,
            "language": str(case.get("language", "cpp")),
            "prompt": prompt_text,
            "prompt_graph_mode": case_graph_mode,
        }
        if case_graph_mode == "repo":
            eval_root = cases_path.resolve().parent
            repo_path = contained_eval_path(
                eval_root,
                case.get("prompt_graph_repo"),
                where=f"{task_id}.prompt_graph_repo",
            )
            index_path = contained_eval_path(
                eval_root,
                case.get("prompt_graph_index"),
                where=f"{task_id}.prompt_graph_index",
            )
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"{task_id}: prompt graph index not found: {index_path}"
                )
            project_index = PromptProjectIndex.from_json_path(index_path)
            expected_project_id = require_prompt_graph_project_id(
                case.get("prompt_graph_project_id"),
                where=f"{task_id}.prompt_graph_project_id",
            )
            if project_index.project_id != expected_project_id:
                raise ValueError(
                    f"{task_id}: prompt graph project_id mismatch: "
                    f"case={expected_project_id!r} "
                    f"index={project_index.project_id!r}"
                )
            project_index.verify_repository(repo_path)
            source_path = case.get("prompt_source_path")
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(
                    f"{task_id}.prompt_source_path must be a non-empty relative path"
                )
            source_document = project_index.document_for_path(source_path)
            source_start = case.get("prompt_source_start")
            if isinstance(source_start, bool) or not isinstance(source_start, int):
                raise ValueError(f"{task_id}.prompt_source_start must be an integer")
            if source_start < 0:
                raise ValueError(f"{task_id}.prompt_source_start must be non-negative")
            source_end = source_start + len(prompt_text)
            if source_document.source[source_start:source_end] != prompt_text:
                raise ValueError(
                    f"{task_id}: prompt does not match indexed source document"
                )
            row["prompt_graph_index_path"] = str(index_path)
            row["prompt_graph_repository_path"] = str(repo_path)
            row["prompt_document_id"] = source_document.id
            row["prompt_source_path"] = source_document.source_path
            row["prompt_source_start"] = source_start
            row["prompt_graph_index_receipt"] = case.get(
                "prompt_graph_index_receipt"
            )
        rows.append(row)
    return rows


def build_megatron_argv(seq_length: int, checkpoint_dir: str, tokenizer_dir: str, fp8_recipe: str) -> list[str]:
    argv = [
        "cppmega_generate_worker.py",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_dir,
        "--vocab-size", "65536",
        "--make-vocab-size-divisible-by", "128",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--hybrid-layer-pattern", os.environ["HYBRID_LAYER_PATTERN"],
        "--hidden-size", os.environ["CPPMEGA_HIDDEN_SIZE"],
        "--ffn-hidden-size", os.environ["CPPMEGA_FFN_HIDDEN_SIZE"],
        "--num-attention-heads", os.environ["CPPMEGA_NUM_ATTN_HEADS"],
        "--seq-length", str(seq_length),
        "--max-position-embeddings", str(seq_length),
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--train-iters", "1",
        "--eval-interval", "50000000",
        "--eval-iters", "1",
        "--lr", os.environ["CPPMEGA_LR"],
        "--min-lr", os.environ["CPPMEGA_MIN_LR"],
        "--lr-decay-style", "constant",
        "--position-embedding-type", "rope",
        "--no-rope-fusion",
        "--normalization", "RMSNorm",
        "--disable-bias-linear",
        "--bf16",
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--spec", "cppmega.megatron.nam56r_noconv_spec", "build_cppmega_nam56r_noconv_stack_spec",
        "--attention-backend", os.environ["CPPMEGA_ATTN_BACKEND"],
        "--group-query-attention",
        "--num-query-groups", os.environ["CPPMEGA_NUM_QUERY_GROUPS"],
        "--kv-channels", os.environ["CPPMEGA_KV_CHANNELS"],
        "--swiglu",
        "--rotary-base", "10000",
        "--load", checkpoint_dir,
        "--no-load-optim",
        "--no-load-rng",
        "--ckpt-format", "torch_dist",
        "--no-check-for-nan-in-loss-and-grad",
        "--rerun-mode", "disabled",
        "--save-interval", "50000000",
        "--log-interval", "1",
    ]
    if os.environ.get("CPPMEGA_USE_FLASH_ATTN") == "1":
        argv.insert(argv.index("--attention-backend"), "--use-flash-attn")
    if fp8_recipe == "tensorwise":
        argv.extend(
            [
                "--fp8-format", os.environ["CPPMEGA_FP8_FORMAT"],
                "--fp8-recipe", "tensorwise",
                "--fp8-amax-history-len", "16",
                "--fp8-amax-compute-algo", "max",
            ]
        )
    elif fp8_recipe != "off":
        raise ValueError(f"unsupported fp8 recipe {fp8_recipe!r}")
    return argv


def initialize_megatron_compat() -> None:
    import inspect

    import megatron.training.initialize as init_mod

    source = inspect.getsource(init_mod.initialize_megatron)
    if "parse_args(" in source:
        init_mod.initialize_megatron()
        return

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import get_args, set_global_variables

    parsed = parse_args()
    parsed = validate_args(parsed, {})
    set_global_variables(parsed)
    print("CPPMEGA_INITIALIZE_COMPAT=preparsed_global_args", flush=True)
    init_mod.initialize_megatron()
    # Fail loud if the compatibility path did not leave global args live.
    get_args()


def stack_report(out_dir: Path) -> None:
    import importlib

    report = {}
    for name in (
        "torch",
        "transformer_engine",
        "transformer_engine.pytorch",
        "flash_attn",
        "mamba_ssm",
        "megatron.core",
        "cppmega",
    ):
        mod = importlib.import_module(name)
        report[name] = {
            "file": getattr(mod, "__file__", None),
            "version": getattr(mod, "__version__", None),
        }
    report["cuda"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }
    (out_dir / "stack_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("CPPMEGA_GENERATION_STACK_REPORT=" + json.dumps(report, sort_keys=True), flush=True)


def target_module(model):
    inner = getattr(model, "module", model)
    inner = getattr(inner, "module", inner)
    return inner


def build_prompt_graph_structure_inputs(
    graph_artifact,
    *,
    total_token_count: int,
    window_start: int,
    window_end: int,
    query_start: int,
    device: torch.device,
):
    graph_inputs = graph_artifact.model_inputs(
        total_token_count=total_token_count,
        window_start=window_start,
        window_end=window_end,
    )
    routes = graph_inputs.graph_routes
    query_offset = query_start - window_start
    query_length = window_end - query_start
    if query_offset < 0 or query_length <= 0:
        raise ValueError(
            f"invalid graph query [{query_start},{window_end}) for "
            f"window [{window_start},{window_end})"
        )
    structure_inputs = {
        name: torch.tensor(
            [
                graph_inputs.side_channels[name][
                    query_offset : query_offset + query_length
                ]
            ],
            dtype=(
                torch.uint64
                if name in OPAQUE_SYMBOL_ID_SIDECARS
                else torch.long
            ),
            device=device,
        )
        for name in TOKEN_SIDECAR_NAMES
    }
    for _relation, (route_key, count_key) in PAIR_ROUTE_KEYS.items():
        structure_inputs[route_key] = torch.tensor(
            routes[route_key], dtype=torch.long, device=device
        ).reshape(1, -1, 2)
        structure_inputs[count_key] = torch.tensor(
            routes[count_key], dtype=torch.long, device=device
        )
    structure_inputs[GENERATED_QUERY_ROUTE_KEY] = torch.tensor(
        routes[GENERATED_QUERY_ROUTE_KEY], dtype=torch.long, device=device
    ).reshape(1, -1, 2)
    structure_inputs[GENERATED_QUERY_COUNT_KEY] = torch.tensor(
        routes[GENERATED_QUERY_COUNT_KEY], dtype=torch.long, device=device
    )
    for _relation, (route_key, count_key) in TRIPLE_ROUTE_KEYS.items():
        structure_inputs[route_key] = torch.tensor(
            routes[route_key], dtype=torch.long, device=device
        ).reshape(1, -1, 3)
        structure_inputs[count_key] = torch.tensor(
            routes[count_key], dtype=torch.long, device=device
        )
    for name in (
        "graph_chunk_starts",
        "graph_chunk_ends",
        "graph_chunk_kinds",
        "graph_chunk_dep_levels",
    ):
        structure_inputs[name] = torch.tensor(
            [routes[name]], dtype=torch.long, device=device
        )
    structure_inputs["graph_chunk_counts"] = torch.tensor(
        routes["graph_chunk_counts"], dtype=torch.long, device=device
    )
    return structure_inputs, graph_inputs.receipt


def set_prompt_graph_structure_inputs(model, structure_inputs) -> None:
    module = target_module(model)
    setter = getattr(module, "set_cppmega_structure_inputs", None)
    if setter is None:
        raise AttributeError("CppMega model does not expose set_cppmega_structure_inputs")
    setter(structure_inputs)


def restore_checkpoint_strict(load_checkpoint, model_list):
    return load_checkpoint(model_list, None, None, strict=True)


def _prompt_graph_counters(structure_inputs) -> dict[str, int]:
    def count_value(value) -> int:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)
        if isinstance(value, (list, tuple)):
            return sum(
                int(item.item() if hasattr(item, "item") else item)
                for item in value
            )
        if hasattr(value, "numel") and int(value.numel()) > 1:
            return int(value.sum().item())
        return int(value.item() if hasattr(value, "item") else value)

    if structure_inputs is None:
        return {
            "chunks": 0,
            "graph_edges": 0,
            "generated_query_edges": 0,
        }
    edge_count_names = (
        "graph_call_edge_counts",
        "graph_type_edge_counts",
        "graph_domain_edge_counts",
        "graph_build_edge_counts",
        "graph_shell_edge_counts",
        "graph_diagnostic_edge_counts",
        "graph_cross_domain_edge_counts",
    )
    return {
        "chunks": count_value(structure_inputs.get("graph_chunk_counts", [0])),
        "graph_edges": sum(
            count_value(structure_inputs.get(name, [0]))
            for name in edge_count_names
        ),
        "generated_query_edges": count_value(
            structure_inputs.get("graph_generated_query_edge_counts", [0])
        ),
    }


def _require_nonempty_prompt_graph(
    prompt_graph_mode: str,
    counters: dict[str, int],
) -> None:
    if prompt_graph_mode == "repo" and (
        counters["chunks"] <= 0 or counters["graph_edges"] <= 0
    ):
        raise RuntimeError(
            "repository prompt graph mode requires nonempty graph tensors: "
            + str(sorted(counters.items()))
        )


def graph_conditioned_forward(
    model,
    input_ids,
    position_ids,
    *,
    prompt_graph_mode: str,
    structure_inputs,
    inference_context,
):
    counters = _prompt_graph_counters(structure_inputs)
    _require_nonempty_prompt_graph(prompt_graph_mode, counters)
    if prompt_graph_mode == "repo":
        set_prompt_graph_structure_inputs(model, structure_inputs)
        _set_current_structure_batch(structure_inputs)
    try:
        logits = model(
            input_ids,
            position_ids,
            None,
            runtime_gather_output=True,
            inference_context=inference_context,
        )
    finally:
        _set_current_structure_batch(None)
    return logits, counters


def prompt_graph_receipt_fields(
    prompt_graph_mode: str,
    artifact_receipt,
    window_receipt,
    counters: dict[str, int],
) -> dict[str, object]:
    _require_nonempty_prompt_graph(prompt_graph_mode, counters)
    return {
        "prompt_graph_mode": prompt_graph_mode,
        "prompt_graph_counters": dict(counters),
        "prompt_graph_receipt": artifact_receipt,
        "prompt_graph_window_receipt": window_receipt,
    }


def make_static_inference_context(seq_length: int):
    try:
        from megatron.core.inference.contexts import StaticInferenceContext
    except ImportError:
        from megatron.core.inference_params import InferenceParams as StaticInferenceContext
    try:
        return StaticInferenceContext(
            max_batch_size=1,
            max_sequence_length=seq_length,
        )
    except TypeError:
        return StaticInferenceContext(1, seq_length)


def advance_inference_context(inference_context, amount: int) -> None:
    increment = getattr(inference_context, "increment_sequence_len_offset", None)
    if callable(increment):
        increment(amount)
    else:
        inference_context.sequence_len_offset += amount


def cppmega_generation_model_provider(pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    from megatron.training import get_args
    from cppmega.megatron.mamba_builder import cppmega_mamba_builder

    return cppmega_mamba_builder(
        get_args(),
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
    )


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        filtered = sorted_probs * keep
        filtered = filtered / filtered.sum()
        idx = torch.multinomial(filtered, num_samples=1)
        return int(sorted_ids[idx].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def last_step_logits(logits: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected 3D logits, got {tuple(logits.shape)}")
    if logits.shape[0] == batch and logits.shape[1] == seq:
        return logits[:, -1, :]
    if logits.shape[0] == seq and logits.shape[1] == batch:
        return logits[-1, :, :]
    raise ValueError(
        "cannot identify logits layout; expected [batch, seq, vocab] or "
        f"[seq, batch, vocab], got {tuple(logits.shape)} for batch={batch} seq={seq}"
    )


def main() -> int:
    apply_te_checkpoint_kwarg_patch()
    apply_dsa_indexer_fused_patch()
    apply_graph_route_attention_bias_patch()
    random.seed(int(os.environ.get("CPPMEGA_GENERATION_SEED", "1234")))
    torch.manual_seed(int(os.environ.get("CPPMEGA_GENERATION_SEED", "1234")))

    out_dir = Path("/data/cppmega_h200_generation_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_report(out_dir)

    seq_length = int(os.environ["CPPMEGA_SEQ_LENGTH"])
    max_new_tokens = int(os.environ["CPPMEGA_MAX_NEW_TOKENS"])
    temperature = float(os.environ["CPPMEGA_TEMPERATURE"])
    top_p = float(os.environ["CPPMEGA_TOP_P"])
    prompt_mode = os.environ["CPPMEGA_PROMPT_MODE"]
    prompt_graph_mode = os.environ["CPPMEGA_PROMPT_GRAPH_MODE"]
    checkpoint_dir = os.environ["CPPMEGA_CHECKPOINT_DIR"]
    tokenizer_dir = os.environ["CPPMEGA_TOKENIZER_DIR"]
    fp8_recipe = os.environ["CPPMEGA_FP8_RECIPE"]
    if prompt_graph_mode not in {"repo", "off"}:
        raise ValueError(f"unknown prompt graph mode {prompt_graph_mode!r}")
    if prompt_graph_mode == "repo":
        required_flags = (
            "CPPMEGA_STRUCTURE_ENABLED",
            "CPPMEGA_GRAPH_ROUTES_ENABLED",
            "CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS",
        )
        disabled = [name for name in required_flags if os.environ.get(name) != "1"]
        if disabled:
            raise RuntimeError(
                "repository prompt graphs require enabled structure/route flags: "
                + ", ".join(disabled)
            )

    tokenizer = load_tokenizer(tokenizer_dir)
    prompt_tokenizer = CppPromptTokenizerAdapter(
        tokenizer,
        tokenizer_path=Path(tokenizer_dir) / "tokenizer.json",
    )
    graph_builder = (
        PromptGraphBuilder(
            prompt_tokenizer,
            cache_dir=Path(os.environ["CPPMEGA_PROMPT_GRAPH_CACHE_DIR"]),
        )
        if prompt_graph_mode == "repo"
        else None
    )
    prompt_rows = build_prompt_rows(
        Path("/data/cppmega_eval/cases.jsonl"),
        Path("/data/cppmega_eval/prompts.jsonl"),
        prompt_mode,
        prompt_graph_mode,
    )

    sys.argv = build_megatron_argv(seq_length, checkpoint_dir, tokenizer_dir, fp8_recipe)
    from megatron.core.enums import ModelType
    from megatron.training import get_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.training import get_model

    initialize_megatron_compat()
    args = get_args()
    print("CPPMEGA_GENERATION_ARGS=" + json.dumps({
        "seq_length": args.seq_length,
        "hidden_size": args.hidden_size,
        "hybrid_layer_pattern": args.hybrid_layer_pattern,
        "structure_enabled": os.environ.get("CPPMEGA_STRUCTURE_ENABLED"),
        "prompt_graph_mode": prompt_graph_mode,
        "fp8_recipe": fp8_recipe,
        "load": args.load,
    }, sort_keys=True), flush=True)

    model_list = get_model(cppmega_generation_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    iteration = restore_checkpoint_strict(load_checkpoint, model_list)
    model = model_list[0]
    model.eval()
    _set_current_structure_batch(None)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    completions_path = out_dir / "completions.jsonl"
    detail_path = out_dir / "generation_detail.jsonl"
    summary = {
        "checkpoint_iteration": int(iteration) if isinstance(iteration, int) else str(iteration),
        "prompt_mode": prompt_mode,
        "prompt_graph_mode": prompt_graph_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "total": len(prompt_rows),
        "items": [],
    }
    eos_id = int(getattr(tokenizer, "eos_token_id", 3) or 3)
    with completions_path.open("w", encoding="utf-8") as comp_fh, detail_path.open("w", encoding="utf-8") as detail_fh:
        for row in prompt_rows:
            task_id = str(row["task_id"])
            prompt = str(row["prompt"])
            case_graph_mode = str(row["prompt_graph_mode"])
            os.environ["CPPMEGA_GRAPH_ROUTES_ENABLED"] = (
                "1" if case_graph_mode == "repo" else "0"
            )
            os.environ["CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS"] = (
                "1" if case_graph_mode == "repo" else "0"
            )
            graph_artifact = None
            if case_graph_mode == "repo":
                if graph_builder is None:
                    raise RuntimeError("repository prompt graph builder was not initialized")
                project_index = PromptProjectIndex.from_json_path(
                    str(row["prompt_graph_index_path"])
                )
                graph_artifact = graph_builder.build(
                    project_index,
                    PromptGraphContext.from_repository_prompt(
                        project_index,
                        prompt,
                        document_id=int(row["prompt_document_id"]),
                        source_path=str(row["prompt_source_path"]),
                        source_start=int(row["prompt_source_start"]),
                        language=str(row["language"]),
                    ),
                )
                prompt_ids = list(graph_artifact.token_ids)
            else:
                prompt_ids, _prompt_offsets = prompt_tokenizer.encode_with_offsets(
                    prompt
                )
            if not prompt_ids:
                raise ValueError(f"{task_id}: prompt tokenized to empty ids")
            if len(prompt_ids) >= seq_length:
                raise ValueError(
                    f"{task_id}: prompt has {len(prompt_ids)} tokens, which does not "
                    f"fit seq_length={seq_length} without breaking graph alignment"
                )
            if graph_artifact is not None and len(prompt_ids) + max_new_tokens > seq_length:
                raise ValueError(
                    f"{task_id}: prompt plus max_new_tokens exceeds seq_length; "
                    "repository graph decode does not discard indexed prompt tokens"
                )
            ids = list(prompt_ids)
            generated: list[int] = []
            graph_window_receipt = None
            graph_counters = _prompt_graph_counters(None)
            inference_context = make_static_inference_context(seq_length)
            with torch.inference_mode():
                for decode_step in range(max_new_tokens):
                    query_start = 0 if decode_step == 0 else len(ids) - 1
                    query_ids = ids if decode_step == 0 else ids[-1:]
                    key_length = query_start + len(query_ids)
                    input_ids = torch.tensor(
                        [query_ids], dtype=torch.long, device=device
                    )
                    position_ids = torch.arange(
                        query_start,
                        key_length,
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                    structure_inputs = None
                    if graph_artifact is not None:
                        structure_inputs, graph_window_receipt = build_prompt_graph_structure_inputs(
                            graph_artifact,
                            total_token_count=len(ids),
                            window_start=0,
                            window_end=len(ids),
                            query_start=query_start,
                            device=device,
                        )
                        set_prompt_graph_inference_state(
                            inference_context,
                            PromptGraphInferenceState(
                                structure_batch=structure_inputs,
                                query_start=query_start,
                                key_length=key_length,
                            ),
                        )
                    logits, graph_counters = graph_conditioned_forward(
                        model,
                        input_ids,
                        position_ids,
                        prompt_graph_mode=case_graph_mode,
                        structure_inputs=structure_inputs,
                        inference_context=inference_context,
                    )
                    advance_inference_context(inference_context, len(query_ids))
                    next_id = sample_next(
                        last_step_logits(
                            logits,
                            batch=1,
                            seq=len(query_ids),
                        )[0],
                        temperature,
                        top_p,
                    )
                    ids.append(next_id)
                    generated.append(next_id)
                    if next_id == eos_id:
                        break
                    partial = decode(tokenizer, generated)
                    if "\n}" in partial or "\nint main(" in partial:
                        break
            raw_completion = decode(tokenizer, generated)
            completion = trim_body_completion(raw_completion)
            graph_receipt_fields = prompt_graph_receipt_fields(
                case_graph_mode,
                None if graph_artifact is None else graph_artifact.receipt,
                graph_window_receipt,
                graph_counters,
            )
            out_row = {
                "task_id": task_id,
                "completion": completion,
                "raw_completion": raw_completion,
                "generated_ids": generated,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
                **graph_receipt_fields,
            }
            completion_row = {
                "task_id": task_id,
                "completion": completion,
                "completion_source": "model_generation",
                **graph_receipt_fields,
            }
            comp_fh.write(json.dumps(completion_row, ensure_ascii=False) + "\n")
            detail_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            summary["items"].append(out_row)
            print("CPPMEGA_GENERATION_ITEM=" + json.dumps({
                "task_id": task_id,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
                "prompt_graph_mode": case_graph_mode,
                "prompt_graph_counters": graph_counters,
                "prompt_graph_cache_key": (
                    graph_artifact.receipt["cache_key"]
                    if graph_artifact is not None
                    else None
                ),
                "prompt_graph_edge_counts": (
                    graph_artifact.receipt["edge_counts"]
                    if graph_artifact is not None
                    else None
                ),
                "completion_preview": completion[:120],
            }, sort_keys=True), flush=True)

    summary["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    summary["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("CPPMEGA_GENERATION_SUMMARY=" + json.dumps({
        "total": summary["total"],
        "cuda_peak_allocated_gib": summary["cuda_peak_allocated_gib"],
        "cuda_peak_reserved_gib": summary["cuda_peak_reserved_gib"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.strip() + "\n"


def remote_generation_script(
    *,
    docker_image: str,
    seq_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_mode: str,
    prompt_graph_mode: str,
    fp8_recipe: str,
    disable_nvrtc: bool,
) -> str:
    if prompt_graph_mode not in {"repo", "off"}:
        raise ValueError(f"unsupported prompt graph mode {prompt_graph_mode!r}")
    graph_flag_default = 1 if prompt_graph_mode == "repo" else 0
    # Keep the generated Python here-doc aligned with the surrounding shell
    # template so textwrap.dedent can put HEREDOC delimiters at column 0.
    worker = textwrap.indent(generation_worker_source(), "        ")
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        sudo mkdir -p /data/cppmega_h200_generation_results /data/cppmega_overlay
        sudo chown -R "$USER":"$USER" /data

        if ! command -v docker >/dev/null 2>&1; then
          sudo apt-get update
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
          sudo systemctl enable --now docker
        fi
        sudo usermod -aG docker "$USER" || true

        if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
          if command -v nvidia-ctk >/dev/null 2>&1; then
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
          fi
        fi

        nvidia-smi
        if [[ -s /data/cppmega_auth/ghcr_token ]]; then
          sudo docker login ghcr.io \
            -u "$(cat /data/cppmega_auth/ghcr_user)" \
            --password-stdin < /data/cppmega_auth/ghcr_token
          rm -f /data/cppmega_auth/ghcr_token
        fi
        sudo docker pull {shlex.quote(docker_image)}

        cat >/data/cppmega_h200_generation_results/container_generate.sh <<'INNER'
        set -euo pipefail
        cp -a /overlay/. /opt/cppmega/
        export PYTHONPATH="/opt/cppmega:/opt/megatron-lm:${{PYTHONPATH:-}}"
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NCCL_GRAPH_REGISTER=0
        export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
        export TRITON_CACHE_DIR="/data/.triton-cache"
        export NVTE_DISABLE_NVRTC="{1 if disable_nvrtc else 0}"
        export CPPMEGA_STRUCTURE_ENABLED="${{CPPMEGA_STRUCTURE_ENABLED:-{graph_flag_default}}}"
        export CPPMEGA_GRAPH_ROUTES_ENABLED="${{CPPMEGA_GRAPH_ROUTES_ENABLED:-{graph_flag_default}}}"
        export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS="${{CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS:-{graph_flag_default}}}"
        export CPPMEGA_SEQ_LENGTH="{seq_length}"
        export CPPMEGA_MAX_NEW_TOKENS="{max_new_tokens}"
        export CPPMEGA_TEMPERATURE="{temperature}"
        export CPPMEGA_TOP_P="{top_p}"
        export CPPMEGA_PROMPT_MODE="{prompt_mode}"
        export CPPMEGA_PROMPT_GRAPH_MODE="{prompt_graph_mode}"
        export CPPMEGA_PROMPT_GRAPH_CACHE_DIR="/data/cppmega_h200_generation_results/prompt_graph_cache"
        export CPPMEGA_FP8_RECIPE="{fp8_recipe}"
        export CPPMEGA_CHECKPOINT_DIR="/data/cppmega_load_checkpoint"
        export CPPMEGA_TOKENIZER_DIR="/data/cpp_tokenizer_hf"
        mkdir -p "$TRITON_CACHE_DIR" "$CPPMEGA_PROMPT_GRAPH_CACHE_DIR"

        eval "$(python -m cppmega.recipes.run_profiles shell h200_cpp_world_mini \
          --seq-length {seq_length} \
          --micro-batch-size 1 \
          --global-batch-size 1 \
          --train-iters 1 \
          --fp8-recipe {fp8_recipe})"

        cat >/data/cppmega_h200_generation_results/generate_worker.py <<'PYGEN'
{worker}        PYGEN

        python -m torch.distributed.run --nproc_per_node=1 \
          /data/cppmega_h200_generation_results/generate_worker.py \
          2>&1 | tee /data/cppmega_h200_generation_results/generation.log
        INNER

        sudo docker run --gpus all --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
          -v /data:/data \
          -v /data/cppmega_overlay:/overlay:ro \
          {shlex.quote(docker_image)} \
          bash /data/cppmega_h200_generation_results/container_generate.sh
        """
    )


def run_compile_gate(cases: Path, completions: Path, report: Path, *, keep_workdir: bool) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cpp_generation_compile_eval.py"),
        "--cases",
        str(cases),
        "--completions",
        str(completions),
        "--out",
        str(report),
        "--json",
        "--fail-on-fail",
    ]
    if keep_workdir:
        cmd.append("--keep-workdir")
    proc = run(cmd, check=False)
    return proc.returncode


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id", default=os.environ.get("NEBIUS_PARENT_ID", DEFAULT_PARENT_ID))
    parser.add_argument("--subnet-id", default=os.environ.get("NEBIUS_SUBNET_ID", DEFAULT_SUBNET_ID))
    parser.add_argument(
        "--security-group-id",
        default=os.environ.get("NEBIUS_SECURITY_GROUP_ID", DEFAULT_SECURITY_GROUP_ID),
    )
    parser.add_argument("--image-id", default=os.environ.get("NEBIUS_IMAGE_ID", DEFAULT_IMAGE_ID))
    parser.add_argument("--platform", default="gpu-h200-sxm")
    parser.add_argument("--preset", default="1gpu-16vcpu-200gb")
    parser.add_argument("--disk-type", default="network_ssd")
    parser.add_argument("--disk-size-gib", type=int, default=512)
    parser.add_argument("--instance-name", default=f"cppmega-h200-generation-{int(time.time())}")
    parser.add_argument("--ssh-user", default="dave")
    parser.add_argument("--ssh-key", type=Path, default=default_ssh_key())
    parser.add_argument("--ssh-pubkey", type=Path, default=None)
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--ghcr-user", default=None)
    parser.add_argument("--ghcr-token-file", type=Path, default=None)
    parser.add_argument("--no-ghcr-auth", action="store_true")
    parser.add_argument("--checkpoint-local", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--clang-indexer-root",
        type=Path,
        default=(
            Path(os.environ["CPPMEGA_CLANG_INDEXER_ROOT"])
            if os.environ.get("CPPMEGA_CLANG_INDEXER_ROOT")
            else None
        ),
        help="Required for repo cases without a prebuilt prompt graph index.",
    )
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-mode", choices=("source-prefix", "instruction"), default="source-prefix")
    parser.add_argument("--prompt-graph-mode", choices=("repo", "off"), default="repo")
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise"), default="off")
    parser.add_argument("--disable-nvrtc", action="store_true")
    parser.add_argument("--remote-timeout-s", type=int, default=3600)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--allow-compile-fail",
        action="store_true",
        help="Diagnostic-only override; generation compile failures fail by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seq_length <= 0:
        raise ValueError("--seq-length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not (0.0 <= args.top_p <= 1.0):
        raise ValueError("--top-p must be in [0, 1]")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.prompt_graph_mode == "repo" and args.prompt_mode != "source-prefix":
        raise ValueError("--prompt-graph-mode=repo requires --prompt-mode=source-prefix")
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)
    eval_graph_assets(
        args.cases,
        prompt_graph_mode=args.prompt_graph_mode,
        indexer_root=args.clang_indexer_root,
    )

    pubkey_path = args.ssh_pubkey or Path(str(args.ssh_key) + ".pub")
    if not pubkey_path.exists():
        raise FileNotFoundError(f"ssh public key not found: {pubkey_path}")
    ssh_pubkey = pubkey_path.read_text().strip()

    script = remote_generation_script(
        docker_image=args.docker_image,
        seq_length=args.seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_mode=args.prompt_mode,
        prompt_graph_mode=args.prompt_graph_mode,
        fp8_recipe=args.fp8_recipe,
        disable_nvrtc=args.disable_nvrtc,
    )
    if args.dry_run:
        print(f"checkpoint_local={args.checkpoint_local}")
        print(f"tokenizer_dir={args.tokenizer_dir}")
        print(f"cases={args.cases}")
        print(f"prompts={args.prompts}")
        print(f"prompt_graph_mode={args.prompt_graph_mode}")
        print(script[:5000])
        return 0

    instance_id: str | None = None
    out_dir = ROOT / "outputs" / "nebius" / args.instance_name
    try:
        with tempfile.TemporaryDirectory(prefix="cppmega-h200-generation-") as tmp:
            tmp_path = Path(tmp)
            overlay_tar = tmp_path / "cppmega_overlay.tgz"
            tokenizer_tar = tmp_path / "cppmega_tokenizer.tgz"
            eval_tar = tmp_path / "cppmega_eval.tgz"
            checkpoint_tar = tmp_path / "cppmega_load_checkpoint.tar"
            ghcr_auth_tar = tmp_path / "cppmega_ghcr_auth.tgz"

            make_overlay_tar(overlay_tar)
            make_tokenizer_tar(args.tokenizer_dir, tokenizer_tar)
            make_eval_tar(
                args.cases,
                args.prompts,
                eval_tar,
                prompt_graph_mode=args.prompt_graph_mode,
                indexer_root=args.clang_indexer_root,
            )
            make_checkpoint_plain_tar(args.checkpoint_local, checkpoint_tar)
            has_ghcr_auth = make_ghcr_auth_tar(args, ghcr_auth_tar)
            if args.docker_image.startswith("ghcr.io/") and not has_ghcr_auth and not args.no_ghcr_auth:
                raise RuntimeError(
                    "GHCR image selected but no auth was found. Set GHCR_TOKEN/GITHUB_TOKEN, "
                    "pass --ghcr-token-file, or docker login ghcr.io locally."
                )

            instance_id = create_instance(args, ssh_pubkey)
            ip = wait_for_ip(instance_id)
            wait_for_ssh(args, ip)
            stream_tar_to_remote(args, ip, overlay_tar, "/data/cppmega_overlay")
            stream_tar_to_remote(args, ip, tokenizer_tar, "/data")
            stream_tar_to_remote(args, ip, eval_tar, "/data")
            stream_plain_tar_to_remote(args, ip, checkpoint_tar, "/data/cppmega_load_checkpoint")
            if has_ghcr_auth:
                stream_tar_to_remote(args, ip, ghcr_auth_tar, "/data")
            ssh(
                args,
                ip,
                f"cat > /data/run_cppmega_h200_generation.sh <<'EOF'\n{script}\nEOF\n"
                "chmod +x /data/run_cppmega_h200_generation.sh",
            )
            try:
                ssh(args, ip, "bash /data/run_cppmega_h200_generation.sh", timeout=args.remote_timeout_s)
            finally:
                out_dir.mkdir(parents=True, exist_ok=True)
                scp_cmd = [
                    "scp",
                    "-i",
                    str(args.ssh_key),
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=15",
                    "-r",
                    f"{args.ssh_user}@{ip}:/data/cppmega_h200_generation_results/.",
                    str(out_dir),
                ]
                run(scp_cmd, check=False)
    finally:
        if instance_id and not args.keep_instance:
            run(
                [
                    "nebius",
                    "compute",
                    "instance",
                    "delete",
                    instance_id,
                    "--format",
                    "json",
                    "--no-progress",
                    "--timeout",
                    "20m",
                ],
                check=False,
                timeout=1500,
            )

    completions = out_dir / "completions.jsonl"
    report = out_dir / "compile_report.json"
    if not completions.exists():
        raise FileNotFoundError(f"remote generation did not produce {completions}")
    compile_rc = run_compile_gate(args.cases, completions, report, keep_workdir=args.keep_workdir)
    if compile_rc != 0 and not args.allow_compile_fail:
        return compile_rc
    summary = json.loads(report.read_text())["summary"]
    print("CPPMEGA_LOCAL_COMPILE_EVAL=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
