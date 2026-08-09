# GCP `.005` idle-worker cleanup gate

Evidence cutoff: `2026-08-09T09:41:42Z`.

## Verdict

`UNSAFE_TO_DELETE`. No VM, address, disk, object, or Terraform state was
deleted or changed. No saved plan is approved for apply.

The six receipt-level idle candidates are `01`, `03`, `05`, `11`, `12`, and
`15`. The current generation-bound monitor snapshot has zero current claims
and zero fresh heartbeats on each of them. That is necessary but not enough
for teardown: every VM still has two auto-deleting 375 GB Local SSDs, and all
six IAP probes failed before the read-only remote marker could execute.

## IAP and Local SSD gate

Each command used `david@jewelmusic.art` through a process-scoped
`CLOUDSDK_CORE_ACCOUNT` plus explicit `--account`, forced IAP, and an OpenSSH
15-second connection timeout. Every worker returned SSH exit `255` with a
banner-exchange timeout. The complete normalized combined stdout/stderr is in
`2026-08-09-gcp-source-005-idle-cleanup-gate-iap.raw.txt`, SHA-256
`4457d5abfb2408dfadf83a01931da146283847f5211de0ebdea35afb9606c0f1`.

Read-only serial output strengthens the failure diagnosis. Worker `01`
continuously reports `network is unreachable` to the metadata server,
including OS Login initialization. Workers `05` and `11` also show source
worker/GCS activity followed by metadata or identity connectivity failures.
The remaining idle candidates contain metadata-server timeouts. These logs do
not inventory `/mnt/cppmega-stage`; they therefore cannot prove that Local SSD
has no unuploaded useful state.

## Claims and completions

The canonical monitor snapshot at `2026-08-09T09:32:58Z` is bound by source
file SHA-256
`dea1f4992608f76e17482d5ae57d953cf9ac23035a5a80ba6f3a2523bf118d8c`.
It reports:

- 482 manifest assignments and 482 claimed assignments;
- 410 immutable completion receipts;
- 54 deterministic outcomes and zero transient outcomes;
- 18 current unresolved claims, four with fresh heartbeats;
- zero current claims on `01`, `03`, `05`, `11`, `12`, and `15`;
- zero slot completion receipts and zero completed workers;
- `training_ready=false` and run state `blocked_deterministic`.

Protected workers `10`, `13`, and `14` were neither IAP-probed nor targeted.
At the bound snapshot, `10` still had two current claims, while `13` and `14`
each had two current claims with fresh heartbeats.

## Exact Terraform state

The applied `.005` state is not `terraform/workers`. It is:

```text
bucket = natural-bison-491019-t9-cppmega-corpus
prefix = terraform/source-runs/source-prod-20260804-005
```

Canonical local inputs:

- backend config:
  `/Volumes/external/cppmega_data/gcp_source_prod_20260804_005/n2d-ab-missing5/isolated-terraform/source-prod-20260804-005.backend.hcl`;
- `TF_DATA_DIR`:
  `/Volumes/external/cppmega_data/gcp_source_prod_20260804_005/n2d-ab-missing5/isolated-terraform/terraform-data-source-prod-20260804-005`;
- var file:
  `/Volumes/external/cppmega_data/gcp_source_prod_20260804_005/source-prod-005-us-central1-ab-missing5.tfvars`.

The remote state is serial `11`, lineage
`0a0be625-4587-a098-79ad-bb67fb79dc89`, with 33 managed instances: 16 VMs,
16 addresses, and one compact placement policy. Two independent pulls of the
sorted identity projection produced SHA-256
`9abdd3d8991b3ecd07dd23e60a69090bd6aa6a11ed31d2f203245651c96e3545`.
The workers module tree exactly matched `origin/main` at plan time.

Compute/GCS/Provider access used only the explicit scoped service account
`nanochat-automation@natural-bison-491019-t9.iam.gserviceaccount.com`. The
global account ended restored to
`cloud-love-ci-observer@gen-lang-client-0223744829.iam.gserviceaccount.com`.

## Read-only targeted plans

Both plans used `-destroy -refresh=true -lock=false -input=false`, wrote only a
local binary preview, and were never applied.

The requested VM+address target set is unsafe. Terraform's conservative
dependency graph expands the six address targets to every VM:

```text
Plan: 0 to add, 0 to change, 22 to destroy.
```

It would delete all 16 instances plus the six requested addresses, including
protected workers. Its binary is retained only as negative evidence at:

`/Volumes/external/cppmega_data/gcp_source_prod_20260804_005/n2d-ab-missing5/isolated-terraform/source-prod-20260804-005.idle-010305111215.destroy-preview-20260809T0928Z.tfplan`

Binary SHA-256:
`064f05725d04f205f53507f69c8e91951fff7f8e50907401abd9522e740aa823`.
Plan JSON SHA-256:
`5ac066bba102c652547e2c5c38736d20ade86992aa80dd7eeac3ee4bc08bd0be`.

The instance-only preview is topologically exact:

```text
Plan: 0 to add, 0 to change, 6 to destroy.
```

Its only actions are deletion of VM resources `01`, `03`, `05`, `11`, `12`,
and `15`; it does not include addresses or protected instances. It is still
blocked because Local SSD readback is unproven. Its path is:

`/Volumes/external/cppmega_data/gcp_source_prod_20260804_005/n2d-ab-missing5/isolated-terraform/source-prod-20260804-005.idle-010305111215.instances-only.destroy-preview-20260809T0928Z.tfplan`

Binary SHA-256:
`d24f436bd5567cc9823aa80ca5a60553238681fdaa4b997bfa617785c00d80b2`.
Plan JSON SHA-256:
`f8e8654333c206d6157ae575669995460565cafa614332bf073c1ccc37b900c1`.

## Next safe action

Restore read-only guest access, or obtain an equivalent immutable inventory of
`/mnt/cppmega-stage`, then generation-pin and read back every useful local
candidate, receipt, and checkpoint. Only after that gate may the six-instance
preview be regenerated against the same state serial. Address release must be
a later Terraform stage after the six VMs are absent; the combined 22-delete
preview must never be applied.
