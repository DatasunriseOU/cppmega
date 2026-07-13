#!/usr/bin/env python3
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
import tarfile
import tempfile
import textwrap
import time
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


def make_eval_tar(cases: Path, prompts: Path, path: Path) -> None:
    for item in (cases, prompts):
        if not item.exists():
            raise FileNotFoundError(item)
    with tempfile.TemporaryDirectory(prefix="cppmega-eval-stage-") as stage_raw:
        stage = Path(stage_raw)
        eval_stage = stage / "cppmega_eval"
        eval_stage.mkdir()
        os.symlink(cases.resolve(), eval_stage / "cases.jsonl")
        os.symlink(prompts.resolve(), eval_stage / "prompts.jsonl")
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
from cppmega.megatron.graph_route_attention_bias_patch import apply_graph_route_attention_bias_patch
from cppmega.megatron.te_checkpoint_kwarg_patch import apply_te_checkpoint_kwarg_patch


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


def encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


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


def build_prompt_rows(cases_path: Path, prompts_path: Path, prompt_mode: str) -> list[dict[str, str]]:
    cases = {str(row["task_id"]): row for row in iter_jsonl(cases_path)}
    prompts = {str(row["task_id"]): row for row in iter_jsonl(prompts_path)}
    if set(cases) != set(prompts):
        raise ValueError(
            "cases/prompts task_id mismatch: "
            f"cases_only={sorted(set(cases) - set(prompts))[:5]} "
            f"prompts_only={sorted(set(prompts) - set(cases))[:5]}"
        )
    rows: list[dict[str, str]] = []
    for task_id in sorted(cases):
        case = cases[task_id]
        prompt_row = prompts[task_id]
        if prompt_mode == "source-prefix":
            prompt_text = str(case["source_prefix"])
        elif prompt_mode == "instruction":
            prompt_text = str(prompt_row["prompt"])
        else:
            raise ValueError(f"unknown prompt mode {prompt_mode!r}")
        rows.append(
            {
                "task_id": task_id,
                "language": str(case.get("language", "cpp")),
                "prompt": prompt_text,
            }
        )
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


def set_default_structure_inputs(model, batch: int, seq: int, device: torch.device) -> None:
    module = target_module(model)
    setter = getattr(module, "set_cppmega_structure_inputs", None)
    if setter is None:
        raise AttributeError("CppMega model does not expose set_cppmega_structure_inputs")
    zeros = torch.zeros((batch, seq), dtype=torch.long, device=device)
    empty_counts = torch.zeros((batch,), dtype=torch.long, device=device)
    setter(
        {
            "structure_ids": zeros,
            "dep_levels": zeros,
            "ast_depth_ids": zeros,
            "sibling_index_ids": zeros,
            "node_type_ids": zeros,
            # Standalone function prompts have no repository graph. Preserve the
            # graph-routed checkpoint contract with an explicit empty graph; an
            # absent graph must fail closed instead of looking token-only by accident.
            "graph_call_edges": torch.zeros((batch, 0, 2), dtype=torch.long, device=device),
            "graph_call_edge_counts": empty_counts,
            "graph_chunk_starts": torch.zeros((batch, 0), dtype=torch.long, device=device),
            "graph_chunk_ends": torch.zeros((batch, 0), dtype=torch.long, device=device),
            "graph_chunk_counts": empty_counts,
        }
    )


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
    checkpoint_dir = os.environ["CPPMEGA_CHECKPOINT_DIR"]
    tokenizer_dir = os.environ["CPPMEGA_TOKENIZER_DIR"]
    fp8_recipe = os.environ["CPPMEGA_FP8_RECIPE"]

    tokenizer = load_tokenizer(tokenizer_dir)
    prompt_rows = build_prompt_rows(
        Path("/data/cppmega_eval/cases.jsonl"),
        Path("/data/cppmega_eval/prompts.jsonl"),
        prompt_mode,
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
        "fp8_recipe": fp8_recipe,
        "load": args.load,
    }, sort_keys=True), flush=True)

    model_list = get_model(cppmega_generation_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    iteration = load_checkpoint(model_list, None, None, strict=True)
    model = model_list[0]
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    completions_path = out_dir / "completions.jsonl"
    detail_path = out_dir / "generation_detail.jsonl"
    summary = {
        "checkpoint_iteration": int(iteration) if isinstance(iteration, int) else str(iteration),
        "prompt_mode": prompt_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "total": len(prompt_rows),
        "items": [],
    }
    eos_id = int(getattr(tokenizer, "eos_token_id", 3) or 3)
    with completions_path.open("w", encoding="utf-8") as comp_fh, detail_path.open("w", encoding="utf-8") as detail_fh:
        for row in prompt_rows:
            prompt_ids = encode(tokenizer, row["prompt"])
            if not prompt_ids:
                raise ValueError(f"{row['task_id']}: prompt tokenized to empty ids")
            if len(prompt_ids) >= seq_length:
                prompt_ids = prompt_ids[-(seq_length - 1) :]
            ids = list(prompt_ids)
            generated: list[int] = []
            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    ctx = ids[-seq_length:]
                    input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
                    position_ids = torch.arange(len(ctx), dtype=torch.long, device=device).unsqueeze(0)
                    if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") == "1":
                        set_default_structure_inputs(model, 1, len(ctx), device)
                    logits = model(
                        input_ids,
                        position_ids,
                        None,
                        runtime_gather_output=True,
                    )
                    next_id = sample_next(
                        last_step_logits(logits, batch=1, seq=len(ctx))[0],
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
            out_row = {
                "task_id": row["task_id"],
                "completion": completion,
                "raw_completion": raw_completion,
                "generated_ids": generated,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
            }
            comp_fh.write(json.dumps({"task_id": row["task_id"], "completion": completion}, ensure_ascii=False) + "\n")
            detail_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            summary["items"].append(out_row)
            print("CPPMEGA_GENERATION_ITEM=" + json.dumps({
                "task_id": row["task_id"],
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated),
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
    fp8_recipe: str,
    disable_nvrtc: bool,
) -> str:
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
        export CPPMEGA_STRUCTURE_ENABLED="${{CPPMEGA_STRUCTURE_ENABLED:-1}}"
        export CPPMEGA_GRAPH_ROUTES_ENABLED="${{CPPMEGA_GRAPH_ROUTES_ENABLED:-1}}"
        export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS="${{CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS:-1}}"
        export CPPMEGA_SEQ_LENGTH="{seq_length}"
        export CPPMEGA_MAX_NEW_TOKENS="{max_new_tokens}"
        export CPPMEGA_TEMPERATURE="{temperature}"
        export CPPMEGA_TOP_P="{top_p}"
        export CPPMEGA_PROMPT_MODE="{prompt_mode}"
        export CPPMEGA_FP8_RECIPE="{fp8_recipe}"
        export CPPMEGA_CHECKPOINT_DIR="/data/cppmega_load_checkpoint"
        export CPPMEGA_TOKENIZER_DIR="/data/cpp_tokenizer_hf"
        mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_generation_results

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
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-mode", choices=("source-prefix", "instruction"), default="source-prefix")
    parser.add_argument("--fp8-recipe", choices=("off", "tensorwise"), default="off")
    parser.add_argument("--disable-nvrtc", action="store_true")
    parser.add_argument("--remote-timeout-s", type=int, default=3600)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--fail-on-compile-fail", action="store_true")
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
        fp8_recipe=args.fp8_recipe,
        disable_nvrtc=args.disable_nvrtc,
    )
    if args.dry_run:
        print(f"checkpoint_local={args.checkpoint_local}")
        print(f"tokenizer_dir={args.tokenizer_dir}")
        print(f"cases={args.cases}")
        print(f"prompts={args.prompts}")
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
            make_eval_tar(args.cases, args.prompts, eval_tar)
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
    if args.fail_on_compile_fail and compile_rc != 0:
        return compile_rc
    summary = json.loads(report.read_text())["summary"]
    print("CPPMEGA_LOCAL_COMPILE_EVAL=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if compile_rc == 0 or not args.fail_on_compile_fail else compile_rc


if __name__ == "__main__":
    raise SystemExit(main())
