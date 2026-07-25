"""Modal C++ code generation eval for CppMega mini checkpoint on H200.

Loads a trained nanochat checkpoint, generates completions for eval cases,
compiles and runs them, reports pass rates.

Usage:
    modal run scripts/modal_eval_cppmega_mini.py --step 2000
    modal run scripts/modal_eval_cppmega_mini.py --step 2000 --num-samples 5
"""

import modal
import os
import sys
from typing import Any, cast

app = modal.App("cppmega-mini-eval")

_TORCH_NIGHTLY = (
    "https://download.pytorch.org/whl/nightly/cu130/"
    "torch-2.12.0.dev20260304%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl"
)
_TRITON_NIGHTLY = (
    "https://download.pytorch.org/whl/nightly/"
    "triton-3.6.0%2Bgit9844da95-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
_CORE_DEPS = ["numpy>=1.26", "pyarrow>=17.0", "packaging", "einops"]
_TRAINING_DEPS = [
    "tokenizers>=0.20", "wandb", "tqdm", "datasets", "psutil", "tabulate",
    "scipy", "regex", "tiktoken", "cut-cross-entropy", "google-cloud-storage",
    "transformers", "flask",
    "flash-linear-attention @ git+https://github.com/sustcsonglin/flash-linear-attention",
]


def _ignore_modal_repo_path(p):
    parts = p.parts
    root = parts[0] if parts else ""
    excluded = {
        ".agent_fanout", ".beads", ".cache", ".claude", ".codex", ".coverage",
        ".git", ".gkb", ".omx", ".playwright-mcp", ".pytest_cache", ".repowise",
        ".tmp", ".tmp_pytest", ".venv", ".wheels", "beads", "data", "docs",
        "experiments", "reports", "tests", "tmp", "tools", "torch", "vertex_ai",
    }
    return (
        root in excluded
        or "__pycache__" in parts
        or ".mypy_cache" in parts
        or ".ruff_cache" in parts
        or p.suffix in (".log", ".pyc")
    )


image = (
    cast(Any, modal.Image).from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu24.04", add_python="3.13"
    )
    .apt_install("gcc", "g++", "git", "clang")
    .pip_install(_TORCH_NIGHTLY, _TRITON_NIGHTLY)
    .pip_install(*_CORE_DEPS)
    .run_commands(
        "apt-get update && apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0",
        "pip install psutil ninja packaging",
    )
    .pip_install(*_TRAINING_DEPS)
    .add_local_dir(
        "/Volumes/external/sources/nanochat",
        "/app/nanochat",
        copy=True,
        ignore=_ignore_modal_repo_path,
    )
)

checkpoint_vol = modal.Volume.from_name("nanochat-checkpoints", create_if_missing=True)
_GPU_SPEC = os.environ.get("NANOCHAT_MODAL_GPU", "H200")

EVAL_CASES = [
    {
        "task_id": "clamp_int",
        "source_prefix": '#include <algorithm>\n#include <cassert>\n\n/**\n * Return value clamped into inclusive [lo, hi]. If lo > hi, swap bounds first.\n */\nint clamp_int(int value, int lo, int hi) {\n',
        "source_suffix": '}\n\nint main() {\n    assert(clamp_int(5, 0, 10) == 5);\n    assert(clamp_int(-3, 0, 10) == 0);\n    assert(clamp_int(18, 0, 10) == 10);\n    assert(clamp_int(4, 10, 0) == 4);\n    assert(clamp_int(-1, 10, 0) == 0);\n    return 0;\n}\n',
        "compile_args": ["-std=c++20", "-O0"],
    },
    {
        "task_id": "is_prime",
        "source_prefix": '#include <cassert>\n\n/**\n * Return true if n is a prime integer.\n */\nbool is_prime(int n) {\n',
        "source_suffix": '}\n\nint main() {\n    assert(!is_prime(-7));\n    assert(!is_prime(0));\n    assert(!is_prime(1));\n    assert(is_prime(2));\n    assert(is_prime(3));\n    assert(!is_prime(4));\n    assert(is_prime(97));\n    assert(!is_prime(221));\n    return 0;\n}\n',
        "compile_args": ["-std=c++20", "-O0"],
    },
    {
        "task_id": "join_non_empty",
        "source_prefix": '#include <cassert>\n#include <string>\n#include <string_view>\n#include <vector>\n\n/**\n * Join non-empty strings with the separator, preserving order.\n */\nstd::string join_non_empty(const std::vector<std::string>& parts, std::string_view sep) {\n',
        "source_suffix": '}\n\nint main() {\n    assert(join_non_empty({"a", "", "b", "c"}, ":") == "a:b:c");\n    assert(join_non_empty({"", "x", ""}, ",") == "x");\n    assert(join_non_empty({}, ",") == "");\n    assert(join_non_empty({"", ""}, ",") == "");\n    return 0;\n}\n',
        "compile_args": ["-std=c++20", "-O0"],
    },
    {
        "task_id": "parse_uint",
        "source_prefix": '#include <cassert>\n#include <cstdint>\n#include <limits>\n#include <optional>\n#include <string_view>\n\n/**\n * Parse an unsigned 32-bit integer from an entire decimal string.\n */\nstd::optional<std::uint32_t> parse_uint(std::string_view text) {\n',
        "source_suffix": '}\n\nint main() {\n    assert(parse_uint("0") && *parse_uint("0") == 0u);\n    assert(parse_uint("42") && *parse_uint("42") == 42u);\n    assert(parse_uint("4294967295") && *parse_uint("4294967295") == std::numeric_limits<std::uint32_t>::max());\n    assert(!parse_uint(""));\n    assert(!parse_uint("-1"));\n    assert(!parse_uint("12x"));\n    assert(!parse_uint("4294967296"));\n    return 0;\n}\n',
        "compile_args": ["-std=c++20", "-O0"],
    },
]


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=3600,
    retries=0,
    volumes={"/checkpoints": checkpoint_vol},
)
def run_eval(
    run_name: str = "cppmega-mini-phase1",
    step: int = 2000,
    num_samples: int = 3,
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """Generate C++ completions and evaluate compilation + correctness."""
    import json
    import subprocess
    import tempfile
    import time

    os.environ["NANOCHAT_BASE_DIR"] = "/checkpoints"
    os.environ["TORCHINDUCTOR_TRITON_CUDAGRAPHS"] = "0"
    sys.path.insert(0, "/app/nanochat")

    import torch
    from nanochat.checkpoint_manager import build_model

    ckpt_dir = f"/checkpoints/base_checkpoints/{run_name}"
    print(f"Loading checkpoint from {ckpt_dir} step={step}")

    device = torch.device("cuda")
    model, tokenizer, meta = build_model(ckpt_dir, step, device, phase="eval")
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    print(f"Checkpoint step: {meta.get('step', '?')}")

    results = []
    total_compiled = 0
    total_passed = 0
    total_samples = 0

    for case in EVAL_CASES:
        task_id = case["task_id"]
        prompt = case["source_prefix"]
        suffix = case["source_suffix"]
        compile_args = case["compile_args"]

        case_compiled = 0
        case_passed = 0

        for sample_idx in range(num_samples):
            total_samples += 1
            prompt_ids = tokenizer.encode(prompt)

            with torch.no_grad():
                generated_ids = []
                for token in model.generate(
                    list(prompt_ids),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=50,
                    seed=42 + sample_idx,
                ):
                    generated_ids.append(token)
                    decoded = tokenizer.decode(generated_ids)
                    if "\nint main(" in decoded or "\nvoid main(" in decoded:
                        break
                    if decoded.count("}") >= 1 and len(decoded) > 30:
                        depth = 0
                        end_idx = len(decoded)
                        for i, ch in enumerate(decoded):
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                if depth == 0:
                                    end_idx = i
                                    break
                                depth -= 1
                        if end_idx < len(decoded):
                            break

            completion = tokenizer.decode(generated_ids)
            depth = 0
            end_idx = len(completion)
            for i, ch in enumerate(completion):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        end_idx = i
                        break
                    depth -= 1
            completion = completion[:end_idx]
            for marker in ("\nint main(", "\nvoid main("):
                idx = completion.find(marker)
                if idx != -1:
                    completion = completion[:idx]
            full_source = prompt + completion + suffix

            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, "solution.cpp")
                bin_path = os.path.join(tmpdir, "solution")
                with open(src_path, "w") as f:
                    f.write(full_source)

                try:
                    compile_proc = subprocess.run(
                        ["g++"] + compile_args + ["-o", bin_path, src_path],
                        capture_output=True, text=True, timeout=30,
                    )
                    compiled = compile_proc.returncode == 0
                except (subprocess.TimeoutExpired, OSError):
                    compiled = False

                passed = False
                if compiled:
                    case_compiled += 1
                    total_compiled += 1
                    try:
                        run_proc = subprocess.run(
                            [bin_path], capture_output=True, text=True, timeout=10,
                        )
                        passed = run_proc.returncode == 0
                    except (subprocess.TimeoutExpired, OSError):
                        pass

                if passed:
                    case_passed += 1
                    total_passed += 1

            status = "PASS" if passed else ("COMPILE" if compiled else "FAIL")
            print(f"  [{task_id}] sample {sample_idx}: {status}")
            if not compiled and sample_idx == 0:
                print(f"    Source preview: {full_source[:300]}...")

        results.append({
            "task_id": task_id,
            "compiled": case_compiled,
            "passed": case_passed,
            "samples": num_samples,
        })

    print(f"\n{'='*60}")
    print(f"EVAL RESULTS: {run_name} step={step}")
    print(f"{'='*60}")
    print(f"Compilation rate: {total_compiled}/{total_samples} = {total_compiled/max(total_samples,1):.1%}")
    print(f"Pass rate:        {total_passed}/{total_samples} = {total_passed/max(total_samples,1):.1%}")
    print(f"\nPer-task:")
    for r in results:
        print(f"  {r['task_id']:20s} compile={r['compiled']}/{r['samples']} pass={r['passed']}/{r['samples']}")

    checkpoint_vol.commit()
    return {
        "run_name": run_name,
        "step": step,
        "compilation_rate": total_compiled / max(total_samples, 1),
        "pass_rate": total_passed / max(total_samples, 1),
        "results": results,
    }


@app.local_entrypoint()
def main(
    run_name: str = "cppmega-mini-phase1",
    step: int = 2000,
    num_samples: int = 3,
    temperature: float = 0.2,
):
    """Run C++ code generation eval on a trained checkpoint."""
    print(f"CppMega mini eval: {run_name} step={step}")
    print(f"  GPU: {_GPU_SPEC}, samples={num_samples}, temp={temperature}")

    result = run_eval.remote(
        run_name=run_name,
        step=step,
        num_samples=num_samples,
        temperature=temperature,
    )
    print(f"\nFinal: compile={result['compilation_rate']:.1%} pass={result['pass_rate']:.1%}")
