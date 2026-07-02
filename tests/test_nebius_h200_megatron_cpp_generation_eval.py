import subprocess
import tarfile

from scripts.nebius_h200_megatron_cpp_generation_eval import (
    generation_worker_source,
    main,
    make_checkpoint_plain_tar,
    make_eval_tar,
    make_tokenizer_tar,
    remote_generation_script,
)


def test_remote_generation_script_is_bash_parseable(tmp_path):
    script = remote_generation_script(
        docker_image="ghcr.io/datasunriseou/cppmega:latest",
        seq_length=1024,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        prompt_mode="source-prefix",
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
    assert '"structure_ids": zeros' in worker
    assert "--load" in worker
    assert "--no-load-optim" in worker
    assert "--no-load-rng" in worker
    assert "pretrain_mamba" not in worker
    assert "apply_graph_route_attention_bias_patch()" in worker
    assert "apply_dsa_indexer_fused_patch()" in worker


def test_generation_worker_can_emit_tensorwise_fp8_args():
    worker = generation_worker_source()

    assert 'if fp8_recipe == "tensorwise":' in worker
    assert '"--fp8-recipe", "tensorwise"' in worker
    assert '"--fp8-amax-history-len", "16"' in worker


def test_make_eval_tar_contains_cases_and_prompts(tmp_path):
    cases = tmp_path / "cases.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    cases.write_text('{"task_id":"x","prompt":"p","source_prefix":"int f(){\\n","source_suffix":"}\\n"}\n')
    prompts.write_text('{"task_id":"x","prompt":"p"}\n')
    out = tmp_path / "eval.tgz"

    make_eval_tar(cases, prompts, out)

    with tarfile.open(out, "r:gz") as tf:
        names = set(tf.getnames())
    assert "cppmega_eval/cases.jsonl" in names
    assert "cppmega_eval/prompts.jsonl" in names


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
    cases = tmp_path / "cases.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    cases.write_text('{"task_id":"x","prompt":"p","source_prefix":"int f(){\\n","source_suffix":"}\\n"}\n')
    prompts.write_text('{"task_id":"x","prompt":"p"}\n')

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
            str(cases),
            "--prompts",
            str(prompts),
            "--max-new-tokens",
            "8",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "checkpoint_local=" in out
    assert "generate_worker.py" in out
    assert "cppmega_h200_generation_results" in out
