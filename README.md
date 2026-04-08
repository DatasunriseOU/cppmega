# cppmega

`cppmega` is a CUDA-only, Megatron-first rebuild of the `nanochat` training surface.

The project goal is narrow:

- reuse native NVIDIA Megatron Core and Megatron-LM training/runtime features wherever they already exist
- reuse NeMo/Nemotron guidance where it matches Megatron-native behavior
- port only the `nanochat` blocks and glue that do not have a real upstream equivalent
- keep `Mamba3` sourced from the upstream author implementation, not from `nanochat`

## Constraints

- local macOS is not a Megatron runtime target
- local `.venv` is the shared [`../nanochat/.venv`](/Volumes/external/sources/nanochat/.venv) for helper code only
- real Megatron and CUDA validation happens on the GCP H200 node `h200_legacy`
- `cppmega` should not install itself into the `nanochat` repo or reuse its worktree
- remote CUDA work must not mutate the shared `/mnt/data/venv`; `cppmega` uses a cloned remote env with the same starting package set

## Current shape

- `docs/migration_matrix.md` records what is reused from Megatron vs what remains a custom port candidate
- `docs/porting_policy.md` defines the fail-closed porting rules
- `cppmega/recipes/nam56r_megatron.py` contains the first translation/helper layer for `nanochat` NAM-style patterns into Megatron hybrid patterns
- `scripts/remote_setup_h200.sh` clones the nanochat H200 venv into a dedicated `cppmega` env, installs Megatron there, and can optionally build author `mamba-ssm`
- `scripts/remote_smoke_h200.sh` runs the first `pretrain_mamba.py --mock-data` smoke on H200 and can switch specs via `CPPMEGA_SPEC_MODULE` / `CPPMEGA_SPEC_NAME`
- `scripts/remote_smoke_h200_dsa.sh` is the native-first H200 smoke for official Megatron DSA (`pretrain_gpt.py` + MLA + `--experimental-attention-variant dsa`)
- `scripts/remote_smoke_h200_ngram_hash_poly.sh` is the canonical H200 smoke for ngram-hash enrichment through the `CppMegaGPTModel` / `CppMegaLanguageModelEmbedding` builder path
- `scripts/remote_smoke_h200_structure_poly.sh` is the canonical H200 smoke for structure enrichment through the same Megatron-style polymorphic builder path

## Verified bring-up status

- local macOS remains helper-only; Megatron runtime validation is still H200-only
- `scripts/remote_smoke_h200.sh` now completes a real 8-GPU H200 smoke with 2 training iterations and checkpoint saves on `h200_legacy`
- the current smoke uses `cppmega.megatron.mamba_local_spec.cppmega_mamba_stack_spec` instead of Megatron's default TE-bound `mamba_stack_spec`
- the smoke launcher now isolates checkpoints/logs per spec via `CPPMEGA_RUN_ID`, so alternate smoke specs do not try to load incompatible state
- `cppmega.megatron.author_mamba3_spec.cppmega_author_mamba3_stack_spec` now also completes the same 8-GPU H200 smoke with 2 training iterations and checkpoint saves on `h200_legacy`
- `scripts/remote_smoke_h200_ngram_hash_poly.sh` completes a real 8-GPU H200 smoke with 2 training iterations and checkpoint saves through the `cppmega.megatron.gpt_builder.cppmega_gpt_builder` route
- `scripts/remote_smoke_h200_structure_poly.sh` completes the same 8-GPU H200 smoke with 2 training iterations and checkpoint saves through the same builder route
- the launcher also applies the minimal no-extension flags required by the verified H200 environment:
  - `--no-gradient-accumulation-fusion`
  - `--no-persist-layer-norm`
  - `--no-masked-softmax-fusion`
  - `--eval-interval 50000000 --eval-iters 0`

These are smoke-lane compatibility settings for the current no-TE / no-Apex H200 environment. They are not a license to fork Megatron behavior in the actual ported training stack.

## Canonical custom-feature route

- custom feature enrichment now enters Megatron through `cppmega.megatron.gpt_builder.cppmega_gpt_builder`
- the builder owns `CppMegaGPTModel`, which swaps only the embedding surface to `CppMegaLanguageModelEmbedding`
- `CppMegaLanguageModelEmbedding.forward()` preserves upstream Megatron shape/scatter behavior and adds enrichment only in the local embedding layout before the upstream transpose path
- old remote file-rewrite patch helpers are legacy-only and are no longer the primary launch or test contract

## Grounded upstream baseline

- Megatron Core already covers the main training substrate: TP, PP, CP, EP, sequence parallel, FSDP, distributed optimizer, MLA, MTP, and FIM
- Megatron-LM already has hybrid Mamba models, `pretrain_mamba.py`, `MambaModel`, `--hybrid-layer-pattern`, and mock-data support
- Megatron main still targets Megatron's Mamba stack, not author-pure `Mamba3`
- upstream `state-spaces/mamba` is the authority for `Mamba3`
- grounding tools in this workspace: `exa`, `tavily`, and `perplexity` are available; `brave` is not wired here yet

## First custom seams

The first custom seams that remain plausible after grounding are:

- author `Mamba3` wrapped into a Megatron-style hybrid stack; `cppmega.megatron.author_mamba3_spec` is the new narrow TP1/CP1, training-only seam
- `M2RNN` / `R` block support, because `nanochat` uses it in `AEMEAEMEAEMR` and Megatron has no direct equivalent; the first local seam now exists as `cppmega.megatron.m2rnn_spec` and has passed the same 2-iteration 8xH200 smoke lane
- DSA sparse attention, but only after exhausting the new official Megatron DSA/experimental-attention surface first
- Engram; first fail-closed config/recipe port surface now exists in `cppmega.features.engram` and `cppmega.recipes.nam56r_megatron`
- mHC
- ngram hash enrichment; first fail-closed config/recipe port surface now exists in `cppmega.features.engram` and `cppmega.recipes.nam56r_megatron`
- MoD / Gamma-MoD / MoDA family
- enriched structure/data glue that is not already represented by Megatron datasets/configs

Everything else should default to Megatron-native implementation unless proven missing.
