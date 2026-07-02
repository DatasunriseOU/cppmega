set -euo pipefail
cp -a /overlay/. /opt/cppmega/
export PYTHONPATH="/opt/cppmega:/opt/megatron-lm:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="/data/.triton-cache"
export NVTE_DISABLE_NVRTC="1"
export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"
export CPPMEGA_GRAPH_ROUTES_ENABLED="${CPPMEGA_GRAPH_ROUTES_ENABLED:-1}"
export CPPMEGA_SEQ_LENGTH="1024"
export CPPMEGA_MAX_NEW_TOKENS="128"
export CPPMEGA_TEMPERATURE="0.0"
export CPPMEGA_TOP_P="1.0"
export CPPMEGA_PROMPT_MODE="source-prefix"
export CPPMEGA_FP8_RECIPE="off"
export CPPMEGA_CHECKPOINT_DIR="/data/cppmega_load_checkpoint"
export CPPMEGA_TOKENIZER_DIR="/data/cpp_tokenizer_hf"
mkdir -p "$TRITON_CACHE_DIR" /data/cppmega_h200_generation_results

eval "$(python -m cppmega.recipes.run_profiles shell h200_cpp_world_mini           --seq-length 1024           --micro-batch-size 1           --global-batch-size 1           --train-iters 1           --fp8-recipe off)"

cat >/data/cppmega_h200_generation_results/generate_worker.py <<'PYGEN'
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import torch

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
    return text.replace("<SPACE>", " ").replace("<NL>", "\n")


def trim_body_completion(text: str) -> str:
    stripped = text.replace("\r\n", "\n")
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            first_newline = stripped.find("\n")
            if first_newline != -1 and stripped[:first_newline].strip().isidentifier():
                stripped = stripped[first_newline + 1 :]
    stop_markers = ("int main(", "#include ", "```", "<|endoftext|>")
    for marker in stop_markers:
        pos = stripped.find(marker)
        if pos >= 0:
            stripped = stripped[:pos]
    kept: list[str] = []
    for line in stripped.splitlines():
        if line.startswith("}"):
            break
        kept.append(line)
    body = "\n".join(kept).strip()
    return body + ("\n" if body else "")


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
    setter(
        {
            "structure_ids": zeros,
            "dep_levels": zeros,
            "ast_depth_ids": zeros,
            "sibling_index_ids": zeros,
            "node_type_ids": zeros,
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
PYGEN

python -m torch.distributed.run --nproc_per_node=1           /data/cppmega_h200_generation_results/generate_worker.py           2>&1 | tee /data/cppmega_h200_generation_results/generation.log
