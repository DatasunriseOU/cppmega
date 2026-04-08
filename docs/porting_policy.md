# Porting Policy

`cppmega` ports only what Megatron/Nemotron do not already provide.

## Rules

1. Ground each feature before porting it.
2. Prefer official NVIDIA docs, local Megatron-LM checkout, local NeMo AutoModel checkout, and the author `state-spaces/mamba` repo.
3. If Megatron already has a real equivalent, use it instead of copying `nanochat`.
4. If a `nanochat` feature only differs in naming or light glue, write a thin translation/helper layer instead of copying the implementation.
5. Fail closed on ambiguous mappings. Do not silently reinterpret unsupported pattern symbols or behavior.
6. Keep local macOS code importable without CUDA. All real runtime validation happens on the H200 box.
7. Do not mutate the shared nanochat H200 venv in place. Clone its package set into a dedicated `cppmega` remote venv before installing Megatron or author `mamba-ssm`.

## Current evidence sources

- local Megatron-LM checkout: `/private/tmp/megatron-lm`
- remote H200 Megatron-LM checkout: `/mnt/data/megatron-lm` (authoritative for live Mamba runtime contracts)
- local NeMo AutoModel checkout: `/private/tmp/nemo_automodel`
- local author Mamba checkout: `/private/tmp/state-spaces-mamba`
- local `nanochat` checkout: `/Volumes/external/sources/nanochat`
- remote CUDA target: `h200_legacy`

## Tool reality in this environment

- `exa`: working
- `tavily`: working
- `perplexity`: working
- `brave`: not wired in this workspace yet

That means the dependable grounding path for this repo is:

1. local reference checkouts
2. remote H200 Megatron checkout when the local checkout drifts from the live runtime lane
3. built-in web verification when needed
4. `exa`, `tavily`, and `perplexity` when external lookup adds value

## First remote validation order

1. clone `/mnt/data/venv` into a dedicated `cppmega` remote venv
2. install Megatron-LM in that dedicated env
3. run `pretrain_mamba.py --mock-data` with a minimal hybrid pattern
4. only then optionally build author `mamba-ssm` from source in the dedicated env
5. only then start wiring custom seams like author `Mamba3`

## Current verified H200 smoke frontier

The first remote Megatron smoke is now past bring-up and completes 2 training iterations on
`h200_legacy` with checkpoint saves.

The smoke currently depends on four explicit launcher choices that were verified one blocker at a time
against real H200 tracebacks:

1. `--spec cppmega.megatron.mamba_local_spec cppmega_mamba_stack_spec`
   because the upstream Mamba stack spec is TE-bound in this environment.
2. `--no-gradient-accumulation-fusion`
   because local non-TE `ColumnParallelLinear` otherwise requires Apex fused weight-grad CUDA extensions.
3. `--no-persist-layer-norm`
   because `WrappedTorchNorm` rejects `persist_layer_norm=True`.
4. `--no-masked-softmax-fusion`
   because the fused attention softmax path otherwise imports `scaled_masked_softmax_cuda`.
5. `--eval-interval 50000000 --eval-iters 0`
   because `pretrain_mamba.py`'s training path does not tolerate `eval_interval=None` in this smoke setup.

These switches are H200 smoke-lane compatibility facts, not proof that `cppmega` should reimplement those Megatron components.

## Current custom seam frontier

The next grounded custom seam is author `Mamba3`:

1. keep Megatron `MambaLayer` and `MambaStack`
2. replace only the `mixer` with the upstream author `Mamba3`
3. fail closed for TP>1, CP>1, packed-sequence training, and Megatron inference/cache paths until they are explicitly ported

That seam is now validated on `h200_legacy` for the current smoke lane:

1. `CPPMEGA_SPEC_MODULE=cppmega.megatron.author_mamba3_spec`
2. `CPPMEGA_SPEC_NAME=cppmega_author_mamba3_stack_spec`
3. 8 GPUs, 2 train iterations, checkpoint saves at iterations 1 and 2

So the next missing-feature frontier moves past author `Mamba3` and `M2RNN` and back to the remaining custom candidates such as DSA, Engram, mHC, ngram enrichment, and MoD-family features.

For `M2RNN`, the intended port shape is the same as `Mamba3`: keep the Megatron stack and layer shell, and copy only the matrix-recurrence mixer core plus minimal config glue.

That narrow seam now exists in `cppmega.megatron.m2rnn_spec` and has passed the current 8xH200, 2-iteration smoke lane with checkpoint saves.

DSA now requires a stricter rule than before: current Megatron Core exposes an official experimental DSA surface (`get_dsa_module_spec_for_backend`, `DSAttention`, `DSAIndexer`, and DSA-related `TransformerConfig` fields), so `cppmega` must first integrate and validate that upstream path before copying any `nanochat` DSA code. Only the residual `nanochat`-specific behavior with no Megatron equivalent should remain a custom seam.
