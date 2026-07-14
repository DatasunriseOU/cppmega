# C++ Generation Evals

This project should not treat token loss as a real C/C++ generation eval. The
hard gate for generation is:

1. build a prompt from a docstring/comment, code prefix, or repo repair task,
2. generate a completion,
3. assemble real C/C++ source,
4. compile it,
5. run tests or assertions,
6. report pass/fail from the compiler/test process only.

LLM judges and text similarity can be recorded as advisory metadata, but they
must not mark a case passed.

## Local Smoke Suite

The first checked-in suite is intentionally small and deterministic:

- cases: `evals/cpp_docstring_compile_cases.jsonl`
- gold completions: `evals/cpp_docstring_compile_reference.jsonl`
- runner: `scripts/cpp_generation_compile_eval.py`

The gold rows only validate the compile/run oracle. They are not generation
E2E evidence. Generated rows identify themselves with
`completion_source: "model_generation"`.

Run:

```bash
python3 scripts/cpp_generation_compile_eval.py \
  --cases evals/cpp_docstring_compile_cases.jsonl \
  --completions evals/cpp_docstring_compile_reference.jsonl \
  --out outputs/evals/cpp_docstring_compile_reference_report.json \
  --prompts-out outputs/evals/cpp_docstring_compile_prompts.jsonl \
  --keep-workdir \
  --fail-on-fail
```

The prompts file is the bridge to model inference. A remote generation job can
read `outputs/evals/cpp_docstring_compile_prompts.jsonl`, write JSONL rows with
`task_id` and `completion`, then feed those completions back into the same hard
gate.

The runner also accepts `language: "c"` rows and compiles them with `--c-compiler`
(`clang` by default), so the same format covers both C and C++.

## Sidecar Contract

For standalone docstring-to-function tasks, prompt-side sidecars should include:

- `platform_ids`
- `token_structure_ids`
- `token_ast_depth`
- `token_ast_node_type`

For repo repair, dependency completion, or code with surrounding project
context, graph routes are required, not optional:

- `token_symbol_ids`
- `token_call_targets`
- `token_type_refs`
- `token_def_use`
- `token_call_edges`
- `token_type_edges`
- `token_chunk_starts`
- `token_chunk_ends`
- `token_chunk_kinds`
- `token_chunk_dep_levels`

Repository prompts prepend the transitive indexed definition chunks referenced
by the visible target prefix. Repository cases are compiled from an isolated
copy of the real checkout with all declared translation units, then linked and
run; they are not rebuilt as standalone temporary source files.

During token-by-token generation, the generated suffix can use zero/default
token sidecars until parser refresh. After a statement/function candidate is
complete, reparse the assembled source and run the compiler/test gate. The
future parser-in-loop path should refresh graph routes after each stable chunk.

## External Suites To Import

Recommended order:

1. MultiPL-E C++ HumanEval/MBPP: multilingual HumanEval/MBPP translated to many
   languages, including C++; good first public pass@k baseline.
   <https://github.com/nuprl/MultiPL-E>
2. HumanEval-X C++: 820 multilingual samples across Python, C++, Java,
   JavaScript, and Go with test cases; useful for direct C++ generation and
   translation.
   <https://github.com/THUDM/CodeGeeX/blob/main/codegeex/benchmark/README.md>
3. LiveCodeBench: live/updating benchmark with code generation, self-repair,
   code execution, and test-output prediction; useful once contamination and
   sandboxing are controlled.
   <https://github.com/livecodebench/livecodebench>
4. EvalPlus: rigorous extra-test philosophy for HumanEval+/MBPP+; mainly Python
   upstream today, but the test-amplification approach is the right standard for
   our own C++ holdout tasks.
   <https://github.com/evalplus/evalplus>

## H200 Megatron Wrapper

The Nebius H200 wrapper is:

```bash
python3 scripts/nebius_h200_megatron_cpp_generation_eval.py \
  --checkpoint-local outputs/checkpoints/cppmega-h200-megatron-1782697038/seq_1024_bs_192 \
  --cases evals/cpp_docstring_compile_cases.jsonl \
  --prompts outputs/evals/cpp_docstring_compile_prompts.jsonl \
  --clang-indexer-root ../cppmega.mlx \
  --max-new-tokens 128 \
  --disable-nvrtc \
  --keep-workdir
```

It uploads the current cppmega overlay, tokenizer, eval JSONL, and latest
Megatron `torch_dist` checkpoint iteration, runs one inference-only process in
`ghcr.io/datasunriseou/cppmega:latest`, copies completions back, deletes the
instance, then runs the local compile/run gate.

The clang producer checkout is an explicit dependency. Strict eval indexing
rejects clang error/fatal diagnostics, and a failed local compile gate makes the
wrapper fail by default. `--allow-compile-fail` is a diagnostic-only override.

The wrapper uses the cppmega tokenizer decode contract, not generic Hugging Face
`decode()`: token strings are concatenated and `<SPACE>`/`<RESERVED_46>` and
`<NL>`/`<RESERVED_47>` are mapped to whitespace. This is required because the
training tokenizer is not a normal natural-language byte-level decoder.

This generation eval must be measured separately from train/valid/test PPL.

## MLX Wrapper

The MLX-side JSONL wrapper lives in `../cppmega.mlx`:

```bash
cd ../cppmega.mlx
python3 scripts/cpp_jsonl_generation_compile_eval.py \
  --cases ../cppmega/evals/cpp_docstring_compile_cases.jsonl \
  --checkpoint outputs/stage1_ckpts/model_step005000.safetensors \
  --out-dir outputs/mlx_generation_eval
```

It supports native MLX `DenseCppLM` `.safetensors` or MLX checkpoint
directories. It intentionally refuses Megatron `.distcp` / `torch_dist`
checkpoints until an explicit converter exists.
