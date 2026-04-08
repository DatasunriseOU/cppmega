# Migration Matrix

This matrix is the contract for what `cppmega` reuses from Megatron/Nemotron and what remains a custom port candidate.

## Native Megatron reuse

| Surface                                                              | cppmega action                                                                                     | Grounding                                                                                                                                                         |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TP / PP / CP / EP / sequence parallel / FSDP / distributed optimizer | Reuse native Megatron flags and runtime                                                            | Megatron parallelism guide and `pretrain_mamba.py`                                                                                                                |
| Megatron hybrid stack / `--hybrid-layer-pattern`                     | Reuse native hybrid pattern mechanism                                                              | `examples/mamba/README.md`, `arguments.py`                                                                                                                        |
| H200 mock-data smoke lane                                            | Reuse `pretrain_mamba.py`, but keep the launcher no-extension-safe in no-TE / no-Apex environments | verified on `h200_legacy` with `--no-gradient-accumulation-fusion`, `--no-persist-layer-norm`, `--no-masked-softmax-fusion`, and explicit eval flags |
| MLA                                                                  | Reuse native Megatron MLA                                                                          | `docs/user-guide/features/multi_latent_attention.md`                                                                                                              |
| MTP                                                                  | Reuse native Megatron MTP                                                                          | `docs/user-guide/features/multi_token_prediction.md`, `arguments.py`                                                                                              |
| FIM                                                                  | Reuse native Megatron FIM dataset and flags                                                        | `megatron/training/datasets/fim_dataset.py`, `arguments.py`                                                                                                       |
| MoE substrate                                                        | Reuse native Megatron MoE flags and grouped GEMM path                                              | `arguments.py`, DeepSeek example script                                                                                                                           |
| QK layernorm / router bias / top-k scaling / shared expert sizing    | Reuse native Megatron flags where available                                                        | DeepSeek FSDP example and `arguments.py`                                                                                                                          |
| Mock-data smoke                                                      | Reuse `pretrain_mamba.py --mock-data`                                                              | `pretrain_mamba.py`, `arguments.py`                                                                                                                               |

## Custom port candidates

| Surface                      | Why not native yet                                                                                                              | Current cppmega stance                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Non-TE Mamba stack shim      | Upstream `mamba_stack_spec` is TE-bound on the current H200 environment                                                         | Keep the narrow `cppmega.megatron.mamba_local_spec` smoke shim and reuse it as the stack shell for alternate smoke specs                                                               |
| Author `Mamba3`              | Megatron main has Mamba support, but not author-pure `Mamba3`                                                                   | Narrow custom seam now lives in `cppmega.megatron.author_mamba3_spec`, with TP1/CP1 training-only fail-closed constraints; verified on H200 for the 2-iteration smoke lane             |
| `M2RNN` / `R` block          | `nanochat` uses `R` in `AEMEAEMEAEMR`; no direct Megatron equivalent found                                                      | Narrow custom seam now lives in `cppmega.megatron.m2rnn_spec`; verified on H200 for the same 2-iteration smoke lane, but still training-only / no packed-seq / no inference-cache path |
| DSA sparse attention         | Current Megatron Core now exposes an official DSA experimental surface, but coverage versus `nanochat` DSA is not yet validated | Prefer native Megatron DSA first; custom seam only for remaining `nanochat`-specific behavior after H200 validation                                                                    |
| Engram                       | No verified native equivalent found                                                                                             | Custom candidate; fail-closed config/recipe surface now exists in `cppmega.features.engram` and `cppmega.recipes.nam56r_megatron`                                                   |
| mHC                          | No verified native equivalent found                                                                                             | Custom candidate                                                                                                                                                                       |
| ngram hash input enrichment  | No verified native equivalent found                                                                                             | Custom candidate; fail-closed config/recipe surface now exists in `cppmega.features.engram` and `cppmega.recipes.nam56r_megatron`                                                   |
| MoD / Gamma-MoD / MoDA       | No verified native equivalent found                                                                                             | Custom candidate                                                                                                                                                                       |
| Enriched structure/data glue | Megatron covers generic data flows, not nanochat's enriched structure contract                                                  | Custom candidate                                                                                                                                                                       |

## Explicit non-goals

- Do not port nanochat's training loop.
- Do not port nanochat's Megatron emulation wrappers when native Megatron already exists.
- Do not port `nanochat/mamba2.py`.
- Do not add CPU, MPS, or TPU runtime support.

## Pattern translation note

`nanochat` Nemotron patterns are not 1:1 with Megatron hybrid syntax:

- `A` maps to Megatron attention `*`
- `M` maps to Megatron `M`; `cppmega` can now swap the mixer to author `Mamba3`, but only through the narrow TP1/CP1 training-only seam
- `D` would map to Megatron `G`
- `E` maps to Megatron `E`
- `R` has no Megatron-native equivalent and must not be silently remapped

For current NAM-style bring-up, `cppmega` uses a fail-closed translator in `cppmega/recipes/nam56r_megatron.py`.
