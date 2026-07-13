**Critical**
None confirmed.

**High**
1. [Confirmed bug] Generic MLX generation rejects `DenseCppLM`.
File/line: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/models/dense_cpp_lm.py:573`, `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/inference/generation.py:765`.
Failure mode: `DenseCppLM.__call__` returns `(logits, loss)`, but `generate_tokens`/`next_token_logits` reject any tuple as unsupported MTP output. Any converted DenseCppLM used through the generic inference API fails before decoding.
Evidence: bespoke eval unwraps `logits, _loss` at `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:403`, so JSONL eval works only through that side path.
Focused test/fix: add a tiny `DenseCppLM` generation test that calls `generate_tokens(..., max_new_tokens=1)`; accept `(logits, None)` or add a DenseCppLM generation adapter.

2. [Design gap] “current graph routes” MLX receipts are not graph-route parity.
File/line: `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:333`, `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:399`, `/Volumes/external/sources/cppmega.mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py:299`.
Failure mode: converted graph-route checkpoints are evaluated with `require_graph_routes=False`, no `block_bias`, and generated suffix sidecars set to zero. This is token/structure eval, not graph-route inference parity.
Evidence: manifest source is the 16k graph-routes stage at `/Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted/cppmega_h200_graphroutes_stage05_current_seq1024/model.json:28`; the local receipt uses that checkpoint and clang sidecars at `/Volumes/external/sources/cppmega/outputs/evals/local_mlx_fixed_stage05_current_graph_routes_docstring_clang_greedy/generation_summary.json:3`; compile result is 0/4 at `/Volumes/external/sources/cppmega/outputs/evals/local_mlx_fixed_stage05_current_graph_routes_docstring_clang_greedy/compile_report.json:192`.
Focused test/fix: build graph-route sidecars into local eval and pass a nonzero `block_bias`; add a fixture proving logits differ with/without graph routes, then compare against CUDA graph-bias logits for the same prompt.

3. [Design gap] CUDA/H200 generation receipts also do not exercise real graph-route sidecars.
File/line: `/Volumes/external/sources/cppmega/scripts/nebius_h200_megatron_cpp_generation_eval.py:391`, `/Volumes/external/sources/cppmega/scripts/nebius_h200_megatron_cpp_generation_eval.py:530`, `/Volumes/external/sources/cppmega/cppmega/megatron/graph_route_attention_bias_patch.py:233`.
Failure mode: generation sets five zero token sidecar tensors and never supplies graph edge tensors, while the graph-bias patch builds bias from the current structure batch. CUDA receipts therefore cannot prove graph-route parity either.
Evidence: tests lock the zero-sidecar worker behavior at `/Volumes/external/sources/cppmega/tests/test_nebius_h200_megatron_cpp_generation_eval.py:64`; CUDA redecoded compile result is also 0/4 at `/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-generation-1782753301/compile_report_redecoded.json:144`.
Focused test/fix: feed real `graph_*` sidecars through the generation worker or label receipts as token-only; add a worker test that asserts graph edge tensors reach `_get_current_structure_batch()`.

**Medium**
4. [Confirmed bug] MLX local decode does not stop/ban `<RESERVED_*>` tokens, unlike CUDA.
File/line: `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:186`, `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:246`, `/Volumes/external/sources/cppmega/scripts/nebius_h200_megatron_cpp_generation_eval.py:214`.
Failure mode: reserved metadata tokens can enter MLX candidates or survive trimming; CUDA trims `<RESERVED_` and maps IDs 46/47 to space/newline.
Evidence: CUDA raw generation contains `<RESERVED_46>`/`<RESERVED_47>` at `/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-generation-1782753301/generation_summary.json:7`.
Focused test/fix: force a scripted MLX model to emit IDs 46/47 and assert they are banned or trimmed; align MLX decode/trim with CUDA.

5. [Design gap] `prompt_mode="docstring"` is an alias for `source_prefix`.
File/line: `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:93`, `/Volumes/external/sources/cppmega.mlx/tests/test_cpp_jsonl_generation_compile_eval.py:25`.
Failure mode: “docstring” and “source-prefix” receipts are not independent prompt contracts, so prompt/objective conclusions are blurred.
Evidence: the test explicitly asserts both modes return `source_prefix` at `/Volumes/external/sources/cppmega.mlx/tests/test_cpp_jsonl_generation_compile_eval.py:32`.
Focused test/fix: use a fixture where `prompt` and `source_prefix` differ; either make docstring mode use the intended instruction/docstring field or rename the mode.

6. [Design gap] FIM/IFIM exists for training transforms, not local inference parity.
File/line: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/fim.py:164`, `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/data/ast_fim.py:246`, `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:93`.
Failure mode: local generation only supports body completion modes, while FIM special tokens are banned during body decode. There is no FIM/IFIM infill eval proving converted checkpoint behavior on those objectives.
Evidence: body decode bans FIM tokens at `/Volumes/external/sources/cppmega.mlx/scripts/cpp_jsonl_generation_compile_eval.py:186`.
Focused test/fix: add an infill eval path with prefix/suffix/instruction prompts and expected middle-only decode; keep it separate from body completion compile gates.

7. [Design gap] Converter has shape/QKV tests but no end-to-end logits parity test.
File/line: `/Volumes/external/sources/cppmega.mlx/tests/test_convert_megatron_dense500m_torchdist_to_mlx.py:81`, `/Volumes/external/sources/cppmega.mlx/tests/test_convert_megatron_dense500m_torchdist_to_mlx.py:179`.
Failure mode: row mapping coverage will not catch a full-model parity break in norm order, scale, dtype, tied head, sidecar defaults, or attention bias.
Evidence: current converter does fix grouped GQA rows at `/Volumes/external/sources/cppmega.mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py:74`; RoPE split-half sign is covered at `/Volumes/external/sources/cppmega.mlx/tests/test_attention_rope_megatron.py:8`, but no Megatron-vs-MLX logits receipt exists.
Focused test/fix: create a tiny deterministic Megatron DCP fixture, convert it, and compare layer activations plus final logits to a CUDA/torch reference.

**Low**
8. [Design gap] Stable/looped support is not wired into this conversion/inference lane.
File/line: `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/stable_loop.py:79`, `/Volumes/external/sources/cppmega.mlx/scripts/convert_megatron_dense500m_torchdist_to_mlx.py:276`.
Failure mode: stable loop inference exists as a separate reference core, but the converter always builds `DenseCppLM`; current converted receipts prove dense transformer behavior only.
Evidence: stable loop has its own fixed-point inference path at `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/stable_loop.py:230`.
Focused test/fix: either keep Lane 4 scoped to dense conversion or add explicit architecture metadata and reject looped/stable checkpoints unless a converter/eval path exists.

**Bad Generation Cause Split**
Confirmed parity/eval problems: generic generation API incompatibility, missing graph-route sidecars, reserved-token decode drift, docstring-mode aliasing, and missing FIM/IFIM infill eval.

Not confirmed as current parity bugs: grouped GQA/QKV and Megatron split-half RoPE. Current code gathers grouped Q/K/V rows and has targeted tests; RoPE sign is also tested.

Bad generation is not MLX-only. Current local MLX graph-route-named receipt is 0/4 compile pass at `/Volumes/external/sources/cppmega/outputs/evals/local_mlx_fixed_stage05_current_graph_routes_docstring_clang_greedy/compile_report.json:196`; CUDA/Megatron redecoded receipt is also 0/4 at `/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-generation-1782753301/compile_report_redecoded.json:146`. That points to model/objective/prompt-distribution issues in addition to parity gaps: native CUDA output already repeats invalid code and metadata tokens before MLX conversion is involved.

**Stale Docs**
No stale-doc finding promoted. I used current source files and live JSON artifacts; old reports were not used as truth.

**Verification**
Changed files: none. Commits: none. I did not run tests or evals because this was a read-only review and I avoided starting/stopping work that could disturb data processes. `ps` was blocked by sandbox permissions, so active process state could not be verified from here.

