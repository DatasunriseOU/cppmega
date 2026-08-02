# CASE6 Nebius H200 Restore and Preflight Runbook

This runbook covers the CASE6 Megatron bundle, restore path, and one-H200
world sweep in this checkout. Local steps are network-free unless explicitly
marked leader-only. Do not place credentials in a bundle, archive, receipt, or
worktree.

## 1. Bound environment

Use the receipt-bound source environment. Do not substitute a system Python or
an unrelated Megatron checkout.

```bash
set -euo pipefail

export CASE6_ROOT=/Volumes/external/sources/cppmega_case6_v3_adapt
export CASE6_PYTHON=/Volumes/external/sources/.venvs/cppmega.source/bin/python
export MEGATRON_LM_REPO=/Volumes/external/sources/Megatron-LM-core_v0.18.0
export CPPMEGA_MEGATRON_COMMIT=ba7b5ebce12af60627a80985792a1449ce45f46c
export CASE6_STAGE="$HOME/cppmega-case6-nebius-stage"
export CASE6_BUNDLE=/absolute/path/to/verified/cppmega_bundle
export CASE6_ARCHIVE=/absolute/path/to/verified/cppmega_bundle.tar.zst
export CASE6_PREFIX="$CASE6_BUNDLE/data/cppmega_1024_current_mix_graph_train"
export CASE6_TOKENIZER="$CASE6_BUNDLE/tokenizer"
export CASE6_BUCKET=cppmega-sidecar-20260627
export CASE6_PREFIX_S3=cppmega-megatron/case6
export CASE6_ENDPOINT=https://storage.eu-north1.nebius.cloud
export CPPMEGA_IMAGE='ghcr.io/datasunriseou/cppmega@sha256:REPLACE_WITH_64_LOWERCASE_HEX'
export NEBIUS_SSH_HOST_KEY_FILE=/absolute/path/to/trusted/nebius-host-ed25519.pub
export NEBIUS_SSH_HOST_KEY_FINGERPRINT='SHA256:REPLACE_WITH_OUT_OF_BAND_FINGERPRINT'
mkdir -p "$CASE6_STAGE"
cd "$CASE6_ROOT"
```

`CASE6_BUNDLE` must be a locally materialized bundle whose logical manifest,
prefix manifests, tokenizer, objective receipts, and graph sidecars refer to
the same artifact set. The launcher derives graph capacities from those
manifest-bound CSR offsets; hand-written `256/256` defaults are not accepted.

`NEBIUS_SSH_HOST_KEY_FILE` and `NEBIUS_SSH_HOST_KEY_FINGERPRINT` must come from
an out-of-band trusted host inventory and must describe the same `ssh-ed25519`
key. If no out-of-band trusted host key is available, stop. Do not use
`ssh-keyscan`, `StrictHostKeyChecking=no`, `accept-new`, or a key learned from
the instance under test to manufacture a trust decision.

## 2. Local verification

These checks do not create cloud resources. The test command is the exact
focused CASE6 gate requested for this adaptation.

```bash
"$CASE6_PYTHON" -m py_compile \
  cppmega/megatron/objective_contract.py \
  cppmega/megatron/patch_install.py \
  cppmega/megatron/structure_dataset_patch.py \
  scripts/data/publish_megatron_bundle_to_nebius_s3.py \
  scripts/data/restore_megatron_bundle_from_nebius_s3.py \
  scripts/h200_megatron_preflight.py \
  scripts/nebius_h200_megatron_cpp_world_sweep.py
git diff --check

MEGATRON_LM_REPO="$MEGATRON_LM_REPO" \
  "$CASE6_PYTHON" -m pytest -q \
  tests/test_case6_nebius_contracts.py \
  tests/test_patch_install.py \
  tests/test_publish_megatron_bundle_to_nebius_s3.py \
  tests/test_restore_megatron_bundle_from_nebius_s3.py \
  tests/test_nebius_h200_megatron_cpp_world_sweep.py
```

The focused gate must report the CASE6 file as collected and must not rely on
pytest environment mutation for the capacity regression. A system Python
failure to import the receipt-bound Megatron source is an environment failure,
not a product pass.

## 3. Network-free bundle and H200 plans

Validate a prepared bundle and archive without uploading them:

```bash
MEGATRON_LM_REPO="$MEGATRON_LM_REPO" \
  "$CASE6_PYTHON" scripts/data/publish_megatron_bundle_to_nebius_s3.py \
  --bundle "$CASE6_BUNDLE" \
  --archive "$CASE6_ARCHIVE" \
  --bucket "$CASE6_BUCKET" \
  --prefix "$CASE6_PREFIX_S3" \
  --endpoint-url "$CASE6_ENDPOINT" \
  --env-file "$CASE6_STAGE/no-such.env" \
  --dry-run
```

Generate the local H200 receipt plan. This validates nested manifests,
tokenizer/objective bindings, graph contracts, and CSR-derived capacities; it
does not start Megatron in `--dry-run` mode.

```bash
MEGATRON_LM_REPO="$MEGATRON_LM_REPO" \
  "$CASE6_PYTHON" scripts/h200_megatron_preflight.py \
  --bundle-root "$CASE6_BUNDLE" \
  --data-prefix "$CASE6_PREFIX" \
  --tokenizer-model "$CASE6_TOKENIZER" \
  --run-id case6-local-plan-001 \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --fp8-recipe off \
  --checkpoint-root "$CASE6_STAGE/preflight-checkpoint" \
  --cold-checkpoint-root "$CASE6_STAGE/preflight-checkpoint-cold" \
  --output "$CASE6_STAGE/h200_preflight.json" \
  --dry-run
```

The preflight writes a companion `h200_preflight_graph_capacity.json`. Review
its `prefix_manifest_sha256`, sidecar hashes, `graph_max_edges`, and
`graph_max_chunks` before using the values in any remote command.

Generate, but do not execute, the exact remote leader script:

```bash
MEGATRON_LM_REPO="$MEGATRON_LM_REPO" \
  "$CASE6_PYTHON" scripts/nebius_h200_megatron_cpp_world_sweep.py \
  --parent-id "${NEBIUS_PARENT_ID:?set a current parent ID}" \
  --subnet-id "${NEBIUS_SUBNET_ID:?set a current subnet ID}" \
  --security-group-id "${NEBIUS_SECURITY_GROUP_ID:?set a current security group ID}" \
  --image-id "${NEBIUS_IMAGE_ID:?set a current image ID}" \
  --ssh-key "$HOME/.ssh/id_ed25519" \
  --ssh-pubkey "$HOME/.ssh/id_ed25519.pub" \
  --ssh-host-key-file "$NEBIUS_SSH_HOST_KEY_FILE" \
  --ssh-host-key-fingerprint "$NEBIUS_SSH_HOST_KEY_FINGERPRINT" \
  --docker-image "$CPPMEGA_IMAGE" \
  --bundle-root "$CASE6_BUNDLE" \
  --sidecar-prefix "$CASE6_PREFIX" \
  --tokenizer-dir "$CASE6_TOKENIZER" \
  --batch-sizes 1,2,4 \
  --train-iters 3 \
  --fp8-recipe off \
  --dry-run \
  --plan-script "$CASE6_STAGE/h200_remote_leader.sh" \
  --no-ghcr-auth
test -x "$CASE6_STAGE/h200_remote_leader.sh"
```

The image must use a lower-case immutable digest. Mutable tags such as
`:latest` are rejected. The plan script is an atomic, executable artifact and
must be reviewed together with the preflight receipts. Its dry-run receipt must
show the exact objective contract SHA, objective IDs/rates/planned samples and
totals, the canonical graph recipe binding, and the graph bias-beta binding.

## 4. Leader-only remote sequence

Run this section only after the local receipts and plan script have been
reviewed. The leader requires `aws`, `zstd`, `ssh`, `scp`, Docker with the
NVIDIA runtime, and the Nebius CLI.

1. Export one complete S3 credential family. Prefer
   `NEBIUS_S3_ACCESS_KEY_ID` and `NEBIUS_S3_SECRET_ACCESS_KEY`; do not mix them
   with an unrelated AWS session token.
2. Restore the committed transport into a new, empty dedicated output root:

   ```bash
   "$CASE6_PYTHON" scripts/data/restore_megatron_bundle_from_nebius_s3.py \
     --output-root /data/cppmega_bundle \
     --run-id case6-restore-001 \
     --bucket "$CASE6_BUCKET" \
     --prefix "$CASE6_PREFIX_S3" \
     --endpoint-url "$CASE6_ENDPOINT" \
     --hash-jobs 4 \
     --free-space-headroom-gb 40 \
     --require-empty-output-root
   ```

   `--require-empty-output-root` checks the root before remote reads and again
   before promotion. It rejects a reused bundle, archive, lock, receipt, or
   other stale entry; do not remove a trusted host-key gate or reuse a dirty
   root to make the restore proceed.
   The restore lock is scoped to the bundle, not the run ID. The logical
   manifest is validated before archive acquisition; archive members, sizes,
   hashes, destination paths, and final artifact bindings are checked before
   promotion. A failed restore must not leave a promoted destination.
3. Run the H200 preflight in the pinned image or equivalent pinned runtime.
   It must prove stack compatibility, the first production batch with a
   nonzero graph route and objective mix, save/cold staging, full-state
   restore, finite positive loss and gradient norm for the expected iteration,
   zero skipped/NaN iterations, graph-prior consumption with the canonical
   recipe and beta binding, and bound receipts. The save and restore phase
   receipts must include the checkpoint hash, explicit load-at-iteration-1
   evidence, and matching model/optimizer/RNG fingerprints.
4. Execute only the reviewed digest-pinned leader script. Keep the same bundle,
   tokenizer, prefix, and run ID values used by the receipts.
5. Copy `/data/cppmega_h200_results`, all preflight and checkpoint receipts,
   `nvidia-smi` samples, and the stack report before declaring completion.
6. Resource deletion is receipt-gated. The sweep deletes the instance by
   default; `--keep-instance` is an explicit exception requiring review.

## 5. Evidence boundary

Local tests and dry runs prove contracts and command construction only. They do
not prove Nebius IAM, quota, network reachability, H200 SM90 identity, CUDA
kernel execution, Transformer Engine imports, distributed forward/backward
finiteness, archive transport, checkpoint restore, or resource cleanup. Those
claims require the corresponding remote receipts. A process that exists without
advancing receipts or logs is not evidence of a healthy run.

## 6. Curriculum memory envelopes (graph-routes stack)

Measured on one Nebius H200 SXM (143771 MiB visible) in run
`cppmega-h200-graphroutes-1782831200` with the production graph-routes
configuration: `CPPMEGA_STRUCTURE_ENABLED=1`, `CPPMEGA_GRAPH_ROUTES_ENABLED=1`,
`CPPMEGA_GRAPH_DENSE_ATTENTION_BIAS=1`, FP8 tensorwise,
`--recompute-granularity selective --recompute-modules mlp`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. These envelopes do not
transfer to the dense noconv lane: the same stage-3 shape (seq 4096, batch 48)
fit in 111056 MiB without the dense graph attention bias
(`cppmega-h200-curriculum-resume2-1782783207`) and OOMed with it.

Evidence per stage (`outputs/nebius/cppmega-h200-graphroutes-1782831200/`;
torch peaks from the `CPPMEGA_CUDA_PEAK` lines, sampled peaks from the 1 s
`nvidia-smi` CSVs, status from `summary.log`):

| stage | seq | global bs | micro bs | result | torch peak alloc (GiB) | torch peak reserved (GiB) | nvidia-smi peak (MiB) |
|-------|-----|-----------|----------|--------|------------------------|---------------------------|-----------------------|
| 1 | 1024 | 192 | 192 | OK | 113.090 | 115.152 | 120410 |
| 2 | 2048 | 96 | 96 | OK | 122.101 | 123.408 | 128866 |
| 3 | 4096 | 48 | 48 | FAIL (OOM) | 135.371 | 136.477 | 141016 |
| 3 | 4096 | 40 | 40 | OK | 125.114 | 126.924 | 132472 |
| 4 | 8192 | 20 | 20 | FAIL (OOM) | 126.741 | 127.787 | 129028 |
| 4 | 8192 | 16 | 16 | FAIL (OOM) | 116.964 | 117.672 | 122212 |
| 4 | 8192 | 16 | 4 | OK | 45.559 | 46.119 | 49722 |
| 5 | 16384 | 8 | 2 | OK | 68.155 | 68.662 | 72812 |

Near the boundary trust the torch peak and the stage result, not the sampled
`nvidia-smi` peak: both flat stage-4 OOMs show sampled peaks below the passing
stage-2 run, because the fatal transient (an `expandable_segments` mapping
failure with under 21 MiB free of 139.8 GiB mapped) falls between 1 s samples.
All three OOMs hit inside the first iteration, about 30 s into the stage.

Safe envelopes (validated end-to-end in the cited run):

| stage | seq | global bs | micro bs | torch peak reserved (GiB) | headroom to 140.4 GiB total |
|-------|-----|-----------|----------|---------------------------|-----------------------------|
| 1 | 1024 | 192 | 192 | 115.2 | ~25 GiB |
| 2 | 2048 | 96 | 96 | 123.4 | ~17 GiB |
| 3 | 4096 | 40 | 40 | 126.9 | ~13 GiB |
| 4 | 8192 | 16 | 4 | 46.1 | ~94 GiB |
| 5 | 16384 | 8 | 2 | 68.7 | ~72 GiB |

Rules of thumb:

- Keep torch peak allocated at or below ~127 GiB. The largest observed pass is
  126.9 GiB reserved (stage 3, batch 40); OOM manifests at 136.5 GiB reserved.
- Flat mode (global bs == micro bs) is proven only for seq <= 4096. At
  seq >= 8192 micro-batch so that micro tokens per step (micro bs x seq) stay
  at or below 32768; flat seq 8192 OOMs even at batch 16 (131072 tokens per
  step).
- At a fixed token budget per step the peak grows with sequence length under
  the dense graph attention bias (113.1 -> 122.1 -> 135.4 GiB allocated for
  196608 tokens per step at seq 1024/2048/4096). Never extrapolate envelopes
  across sequence lengths from token counts alone.

Reproducing the measurement: the curriculum launcher
`scripts/nebius_h200_megatron_cpp_world_curriculum.py` emits, per stage, a 1 s
`nvidia-smi` CSV (`stage_*_*.nvsmi.csv`), a `CPPMEGA_NVIDIA_SMI_PEAK` line, a
`CPPMEGA_CUDA_PEAK` torch allocated/reserved line in the stage log, and a
`CPPMEGA_CURRICULUM_STAGE_RESULT` status in `summary.log`. Validate the plan
with `--dry-run` first (section 3 pattern), then use the live leader path of
section 4. Explicit `--stage` prefixes must be declared in
`$CASE6_BUNDLE/manifest.json` under `bucket_results` with `bucket` equal to
the stage sequence length.

Boundary verification is PENDING; no GPU time has been spent on it. Because
every observed OOM occurs in the first iteration, a short-iteration probe is a
valid envelope test. Exact command for the two open boundaries (stage 3
between batch 40 and 48, stage 4 between micro 4 and 16), run from
`$CASE6_ROOT` with the section 1 environment:

```bash
PREFIX_4096="$CASE6_BUNDLE/$("$CASE6_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(next(r["prefix"] for r in m["bucket_results"] if int(r["bucket"])==4096))' "$CASE6_BUNDLE/manifest.json")"
PREFIX_8192="$CASE6_BUNDLE/$("$CASE6_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(next(r["prefix"] for r in m["bucket_results"] if int(r["bucket"])==8192))' "$CASE6_BUNDLE/manifest.json")"

MEGATRON_LM_REPO="$MEGATRON_LM_REPO" "$CASE6_PYTHON" \
  scripts/nebius_h200_megatron_cpp_world_curriculum.py \
  --bundle-root "$CASE6_BUNDLE" \
  --docker-image "$CPPMEGA_IMAGE" \
  --ssh-host-key-file "$NEBIUS_SSH_HOST_KEY_FILE" \
  --ssh-host-key-fingerprint "$NEBIUS_SSH_HOST_KEY_FINGERPRINT" \
  --no-ghcr-auth \
  --stage "4096=44=10=$PREFIX_4096" \
  --stage "8192=16=8=10=$PREFIX_8192"
```

Both probes run in one instance; the second stage warm-starts model weights
from the first stage checkpoint, which is acceptable for a memory probe. Ten
iterations per stage is enough: the OOM signature appears in iteration 1 and a
few extra iterations prove stability past warmup. Pass/fail closes the
envelope: stage 3 shrinks to batch 44 or stays at 40; stage 4 grows to micro 8
or stays at 4. An optional stage-5 headroom probe can be appended as
`--stage "16384=8=4=10=$PREFIX_16384"` with the corresponding manifest lookup.
