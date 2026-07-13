import ast
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.nebius_h200_megatron_cpp_generation_eval import (
    generation_worker_source,
    main,
    make_checkpoint_plain_tar,
    make_eval_tar,
    make_tokenizer_tar,
    remote_generation_script,
    run_compile_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CASE3_FIXTURE = ROOT / "tests" / "fixtures" / "case3_prompt_repo"


class _OffsetTokenizer:
    name_or_path = "case3-offset-tokenizer"

    def encode_with_offsets(self, text: str):
        return [ord(ch) % 251 + 1 for ch in text], [
            (index, index + 1) for index in range(len(text))
        ]


def _case3_prompt() -> str:
    row = json.loads((CASE3_FIXTURE / "cases.jsonl").read_text().splitlines()[0])
    return row["source_prefix"]


def test_remote_generation_script_is_bash_parseable(tmp_path):
    script = remote_generation_script(
        docker_image="ghcr.io/datasunriseou/cppmega:latest",
        seq_length=1024,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        prompt_mode="source-prefix",
        prompt_graph_mode="repo",
        fp8_recipe="off",
        disable_nvrtc=True,
    )
    path = tmp_path / "run_generation.sh"
    path.write_text(script)

    assert script.startswith("#!/usr/bin/env bash\n")
    assert "container_generate.sh" in script
    assert "python -m torch.distributed.run --nproc_per_node=1" in script
    assert 'export CPPMEGA_STRUCTURE_ENABLED="${CPPMEGA_STRUCTURE_ENABLED:-1}"' in script
    assert 'export CPPMEGA_GRAPH_ROUTES_ENABLED="${CPPMEGA_GRAPH_ROUTES_ENABLED:-1}"' in script
    assert 'export CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS="${CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS:-1}"' in script
    assert "apply_dsa_indexer_fused_patch()" in script
    assert "apply_graph_route_attention_bias_patch()" in script
    assert 'export CPPMEGA_PROMPT_MODE="source-prefix"' in script
    assert 'export CPPMEGA_PROMPT_GRAPH_MODE="repo"' in script
    assert 'export CPPMEGA_PROMPT_GRAPH_CACHE_DIR="/data/cppmega_h200_generation_results/prompt_graph_cache"' in script
    assert 'export CPPMEGA_CHECKPOINT_DIR="/data/cppmega_load_checkpoint"' in script
    assert 'export NVTE_DISABLE_NVRTC="1"' in script
    assert "pretrain_mamba.py" not in script
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_generation_worker_builds_model_loads_checkpoint_and_threads_sidecars():
    worker = generation_worker_source()

    assert "def last_step_logits" in worker
    assert "last_step_logits(logits, batch=1, seq=len(ctx))[0]" in worker
    assert "logits[0, -1]" not in worker
    assert "_cppmega_id_to_token" in worker
    assert '".join(id_to_token.get(int(token_id), "") for token_id in ids)' in worker
    assert '.replace("<SPACE>", " ")' in worker
    assert '.replace("<NL>", "\\n")' in worker
    assert '.replace("<RESERVED_46>", " ")' in worker
    assert '.replace("<RESERVED_47>", "\\n")' in worker
    assert '"<BOS>"' in worker
    assert '"generated_ids": generated' in worker
    assert "def initialize_megatron_compat()" in worker
    assert "validate_args(parsed, {})" in worker
    assert "set_global_variables(parsed)" in worker
    assert "def cppmega_generation_model_provider" in worker
    assert "cppmega_mamba_builder(" in worker
    assert "get_model(cppmega_generation_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)" in worker
    assert "load_checkpoint(model_list, None, None, strict=True)" in worker
    assert "set_cppmega_structure_inputs" in worker
    assert "TOKEN_SIDECAR_NAMES" in worker
    assert "build_prompt_graph_structure_inputs" in worker
    assert "PromptGraphBuilder" in worker
    assert "CppPromptTokenizerAdapter" in worker
    assert "PromptProjectIndex.from_json_path" in worker
    assert "_set_current_structure_batch(structure_inputs)" in worker
    assert "_set_current_structure_batch(None)" in worker
    assert "finally:" in worker
    assert '"prompt_graph_receipt": graph_artifact.receipt' in worker
    assert '"graph_call_edges": torch.zeros((batch, 0, 2)' not in worker
    assert '"graph_call_edge_counts": empty_counts' not in worker
    assert '"graph_chunk_counts": empty_counts' not in worker
    assert "--load" in worker
    assert "--no-load-optim" in worker
    assert "--no-load-rng" in worker
    assert "pretrain_mamba" not in worker
    assert "apply_graph_route_attention_bias_patch()" in worker
    assert "apply_dsa_indexer_fused_patch()" in worker


def test_prompt_graph_builder_serializes_h200_structure_inputs(tmp_path):
    from cppmega.prompt_graph import (
        PromptGraphBuilder,
        PromptGraphContext,
        PromptProjectIndex,
        TOKEN_SIDECAR_DEFAULTS,
    )

    prompt = _case3_prompt()
    builder = PromptGraphBuilder(_OffsetTokenizer(), cache_dir=tmp_path)
    artifact = builder.build(
        PromptProjectIndex.from_json_path(CASE3_FIXTURE / "project_index.json"),
        PromptGraphContext.from_prompt(prompt, source_start=0),
    )

    assert artifact.edge_counts["call"] == 1
    assert artifact.edge_counts["type"] == 1
    assert artifact.edge_counts["def_use"] == 1
    assert artifact.graph_routes["graph_call_edges"] == [[2, 1]]
    assert artifact.graph_routes["graph_type_edges"] == [[3, 0]]
    assert artifact.graph_routes["graph_domain_edges"] == [
        [
            artifact.first_token_for_identity("call:warmup->clamp_to_zero"),
            artifact.first_token_for_identity("tiny::clamp_to_zero"),
            2,
        ]
    ]
    restored = artifact.__class__.from_dict(json.loads(artifact.to_json()))
    assert restored.receipt == artifact.receipt
    assert (tmp_path / f"{artifact.receipt['cache_key']}.json").is_file()
    model_inputs = artifact.model_inputs(
        total_token_count=artifact.token_count + 2,
        window_start=1,
        window_end=artifact.token_count + 2,
    )
    assert model_inputs.graph_routes["graph_call_edges"] == [[2, 1]]
    assert all(
        values[-2:] == [TOKEN_SIDECAR_DEFAULTS[name]] * 2
        for name, values in model_inputs.side_channels.items()
    )


def test_generation_worker_can_emit_tensorwise_fp8_args():
    worker = generation_worker_source()

    assert 'if fp8_recipe == "tensorwise":' in worker
    assert '"--fp8-recipe", "tensorwise"' in worker
    assert '"--fp8-amax-history-len", "16"' in worker


def test_trim_body_completion_preserves_nested_blocks_and_trailing_return():
    tree = ast.parse(generation_worker_source())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"trim_body_completion", "_trim_at_function_closing_brace"}
    ]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<worker>", "exec"), namespace)
    completion = """if (value < lo) {
    value = lo;
}
// Ignore this brace: }
return value;
}
int main() { return 0; }
"""

    assert namespace["trim_body_completion"](completion) == """if (value < lo) {
    value = lo;
}
// Ignore this brace: }
return value;
"""


def test_make_eval_tar_contains_cases_prompts_and_project_index(tmp_path):
    out = tmp_path / "eval.tgz"

    make_eval_tar(
        CASE3_FIXTURE / "cases.jsonl",
        CASE3_FIXTURE / "prompts.jsonl",
        out,
        prompt_graph_mode="repo",
    )

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "cppmega_eval/cases.jsonl" in names
    assert "cppmega_eval/prompts.jsonl" in names
    assert "cppmega_eval/project_index.json" in names
    assert "cppmega_eval/src/math_prompt.cpp" in names


def test_make_eval_tar_fails_closed_when_repo_graph_index_is_missing(tmp_path):
    cases = tmp_path / "cases.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    cases.write_text(
        '{"task_id":"x","source_prefix":"int f(){\\n","source_suffix":"}\\n"}\n'
    )
    prompts.write_text('{"task_id":"x","prompt":"p"}\n')

    with pytest.raises(ValueError, match="x.*prompt_graph_index"):
        make_eval_tar(
            cases,
            prompts,
            tmp_path / "eval.tgz",
            prompt_graph_mode="repo",
        )


def test_case3_fixture_passes_local_compile_gate(tmp_path):
    report = tmp_path / "compile_report.json"

    rc = run_compile_gate(
        CASE3_FIXTURE / "cases.jsonl",
        CASE3_FIXTURE / "completions.jsonl",
        report,
        keep_workdir=False,
    )

    assert rc == 0
    assert json.loads(report.read_text())["summary"]["passed"] == 1


def test_make_tokenizer_tar_requires_and_includes_tokenizer_json(tmp_path):
    tokenizer = tmp_path / "tok"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    out = tmp_path / "tok.tgz"

    make_tokenizer_tar(tokenizer, out)

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "cpp_tokenizer_hf/tokenizer.json" in names
    assert "cpp_tokenizer_hf/tokenizer_config.json" in names


def test_make_checkpoint_plain_tar_archives_checkpoint_without_gzip(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("7\n")
    old_dir = checkpoint / "iter_0000006"
    old_dir.mkdir()
    (old_dir / "metadata.json").write_text("{}")
    iter_dir = checkpoint / "iter_0000007"
    iter_dir.mkdir()
    (iter_dir / "metadata.json").write_text("{}")
    out = tmp_path / "checkpoint.tar"

    make_checkpoint_plain_tar(checkpoint, out)

    with tarfile.open(out, "r:") as tf:
        names = set(tf.getnames())
    assert "latest_checkpointed_iteration.txt" in names
    assert "iter_0000007/metadata.json" in names
    assert "iter_0000006/metadata.json" not in names


def test_dry_run_prints_remote_generation_script(tmp_path, capsys):
    pubkey = tmp_path / "id_ed25519.pub"
    pubkey.write_text("ssh-ed25519 TESTKEY codex\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n")
    tokenizer = tmp_path / "tok"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    rc = main(
        [
            "--dry-run",
            "--ssh-pubkey",
            str(pubkey),
            "--checkpoint-local",
            str(checkpoint),
            "--tokenizer-dir",
            str(tokenizer),
            "--cases",
            str(CASE3_FIXTURE / "cases.jsonl"),
            "--prompts",
            str(CASE3_FIXTURE / "prompts.jsonl"),
            "--max-new-tokens",
            "8",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "checkpoint_local=" in out
    assert "generate_worker.py" in out
    assert "cppmega_h200_generation_results" in out
    assert 'CPPMEGA_PROMPT_GRAPH_MODE="repo"' in out
