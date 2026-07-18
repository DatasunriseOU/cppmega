# Production Objective Mixture Contract

The Megatron/H200 path consumes objectives pre-materialized by `cppmega.mlx`.
Megatron still sees causal token documents, but every document is an exact
shifted-LM representation of one realized objective and carries its objective
loss mask. The adjacent `cppmega_pre_materialized_objectives_v1` receipt binds
the schedule, token/loss accounting, typed sources, and graph-loss settings.

This is the production boundary. Megatron does not infer objectives from prompt
text, rendered comments, model internals, or token patterns.

## Materialize In cppmega.mlx

Run the materializer against tokenized-enriched parquet containing all columns
required by `OBJECTIVE_SOURCE_COLUMNS`:

```bash
python scripts/materialize_megatron_objectives.py \
  --data-glob '/data/tokenized_enriched/*.parquet' \
  --output-dir /data/objectives/train \
  --samples 600000 \
  --seq-len 4096 \
  --quota-window-samples 60 \
  --seed 17 \
  --graph-relations call,type,domain,build,shell,diagnostic,cross_domain \
  --graph-aux-weight 1.0 \
  --graph-bce-weight 0.10 \
  --graph-coverage-weight 0.05 \
  --graph-topk 256
```

Each quota window is assigned by the deterministic eligibility-aware Hamilton
mixer. The required production tasks are `causal_lm`, `fim`, `ast_fim`,
`ifim`, `commit_diff`, and `pre_to_post`; configured recovery tasks are included
as well. FIM, IFIM, and commit repair are selections of this same materialized
document mixer, not separate token-only loader paths. Every configured task must
receive a nonzero quota. Missing or empty typed fields make that source
ineligible, and an unsatisfiable window aborts.

Authoritative typed inputs are:

- IFIM: `ifim_instruction_token_ids`
- commit repair: `commit_msg_token_ids` plus `diff_token_ids`
- commit transduction: `commit_msg_token_ids`, `pre_token_ids`, and
  `post_token_ids`

In particular, `post_token_ids` is not accepted as a diff and rendered comment
wrappers are never parsed for an instruction or commit message.

The output directory contains `objectives_*.parquet`,
`objective_contract.json`, and the canonical
`objective_materialization.json`. The artifact binds the exact contract digest,
ordered parquet names, byte sizes and SHA-256 digests, full token/graph sidecar
profile, document count, and graph-mask semantics. A consumer rejects missing,
changed, or unlisted parquet instead of widening the input glob.

For every objective example, `input_ids` is
`[objective_input[0], *objective_targets]`; `loss_mask` is the objective mask
plus one zero sentinel. Materialization proves
`target_ids[:-1] == input_ids[1:]` before writing.

`doc_ids` and `token_source_doc_ids` have intentionally different meanings.
`doc_ids` is a positive row-local segment ID used for attention, loss, and graph
boundaries. It changes at every packed subdocument. `token_source_doc_ids` is a
positive stable logical source identity and may repeat across rows and shards.
Zero source IDs inside `valid_token_count` are invalid; padding may be zero.

## Convert In cppmega

On the machine with Megatron Core installed:

```bash
python scripts/data_prep_parquet_to_megatron.py \
  --objective-artifact /data/objectives/train/objective_materialization.json \
  --output-prefix /data/megatron/objectives_train \
  --writer-backend megatron
```

The converter verifies, before publishing its JSON manifest:

- exact configured rates, Hamilton quotas, and realized samples per objective;
- exact input-token and loss-token totals per objective and globally;
- the zero-sentinel shifted-LM representation;
- positive edge and eligible-sample counts for the configured graph relations;
- presence of the objective loss mask and configured graph sidecars.

It writes a document-aligned `uint8` objective-ID sidecar and embeds the
canonical contract plus SHA-256 digest in `<output-prefix>.json`. Conversion
fails if any parquet row drifts from the receipt.

## Megatron Runtime Gate

Production graph training requires:

```bash
export CPPMEGA_STRUCTURE_ENABLED=1
export CPPMEGA_GRAPH_ROUTES_ENABLED=1
export CPPMEGA_DSA_GRAPH_AUX_RELATIONS=call,type,domain,build,shell,diagnostic,cross_domain
export CPPMEGA_DSA_GRAPH_AUX_WEIGHT=1.0
export CPPMEGA_DSA_GRAPH_BCE_WEIGHT=0.10
export CPPMEGA_DSA_GRAPH_COVERAGE_WEIGHT=0.05
export CPPMEGA_DSA_GRAPH_AUX_TOPK=256
export CPPMEGA_GRAPH_BIAS_BETA=1
# Legacy aliases are accepted only when they exactly match the canonical beta.
# export CPPMEGA_DSA_GRAPH_BIAS_BETA=1
# export CPPMEGA_GRAPH_ATTENTION_BIAS_BETA=1
export CPPMEGA_DSA_GRAPH_POS_WEIGHT=1.0
export CPPMEGA_DSA_GRAPH_MARGIN=1.0
export CPPMEGA_OBJECTIVE_CONTRACT_REQUIRED=1
```

Relation order and every numeric value must match the receipt exactly. Model
construction also requires a positive `dsa_indexer_loss_coeff` and
`dsa_indexer_use_sparse_loss=False`; otherwise the graph objective cannot reach
the autograd-carried DSA loss and construction fails. The weighted graph BCE
and top-k coverage terms are added directly to Megatron's dense DSA indexer
loss. Non-finite scores on negative or otherwise masked pairs are excluded from
BCE, positive-edge normalization, and coverage. A positive graph target paired
with a non-finite score is an invalid batch and fails closed; it must never be
converted into zero positives or zero loss.

When graph routes are enabled, the fused DSA score passed to selector top-k is
exactly `I_neural + beta*S_graph`, with one resolved beta shared by DSA, dense
attention, and graph-loss subtraction. The canonical runtime name is
`CPPMEGA_GRAPH_BIAS_BETA`; the historical DSA and dense names are compatibility
aliases and a mismatch between any present names is rejected. Runtime graph
prior receipts carry the exact rational beta binding. The graph auxiliary loss
removes that fixed prior before supervising the neural score, so the selector
prior cannot silently become a token-only or post-top-k side path. A contract
that declares graph loss included in total loss is rejected unless the DSA graph
auxiliary flag is enabled.

Dataset ingress validates the embedded digest and requires the objective-ID
sidecar byte count to equal the indexed document count. Legacy indexed prefixes
without this contract are intentionally rejected when graph routes are enabled;
they must be rematerialized and reconverted rather than reused.

## CASE6 Bundle/H200 Handoff

CASE6 must treat `objective_materialization.json` as the only objective dataset
entry point. Its bundle builder must preserve that file plus every referenced
contract and parquet shard byte-for-byte, then invoke
`convert_parquet_to_megatron(..., objective_artifact_path=...)` or the CLI above.
It must not independently choose a split, infer a schedule, substitute raw
packed parquet, omit `doc_ids`/`token_source_doc_ids`, or attach graph routes to
a legacy prefix.

The indexed prefix admitted to H200 training must contain all of the following:

- the full token sidecar profile bound by the artifact, including `loss_mask`,
  row-local `doc_ids`, and positive stable `token_source_doc_ids`;
- all graph sidecars named by the artifact, with chunk edges expanded as the
  Cartesian product of source/destination token spans;
- an embedded `cppmega_pre_materialized_objectives_v1` payload and digest plus a
  document-aligned objective-ID sidecar;
- runtime graph settings exactly equal to the embedded relation order and
  rational weights.

`structure_dataset_patch` rejects graph-enabled prefixes without that embedded
contract. The runtime intersects graph pairs with causal order, equal positive
row-local document IDs, and Megatron's upstream `mask=` before adding the
weighted BCE and coverage terms to total DSA indexer loss. This is the required
fail-closed behavior until every CASE6 bundle path consumes the canonical
artifact directly.

The first real H200 batch receipt also contains `objective_mix` with
`input_tokens_by_objective`, `loss_tokens_by_objective`, and
`observed_objective_ids`. Missing objective IDs, unknown IDs, shape drift, or a
batch without this accounting fail preflight; a token-only batch is never
accepted as production evidence.
