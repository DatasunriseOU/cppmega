# GCP corpus worker pool

This Terraform provisions four disposable `n2-standard-16` workers by
default. Each worker has 16 vCPUs, 64 GB RAM, a gVNIC interface, a reserved
external IPv4 address, and two 375 GiB NVMe Local SSD devices striped into a
750 GiB RAID0 staging filesystem. Cloud Storage is canonical; Local SSD and
boot disks are never authoritative.

The configuration is deliberately split into two independent Terraform
states:

- `foundation/` owns the private VPC, subnet, firewall rules, worker service
  account, IAM binding, and canonical regional Cloud Storage bucket.
- `workers/` owns only static addresses, placement policy, and disposable
  VMs. Destroying this state returns the compute pool without touching data.

## Why this machine shape

Google's N2 machine table lists `n2-standard-16` as 16 vCPUs, 64 GB RAM,
2/4/8/16/24 Local SSD devices, and up to 32 Gbps default egress. That exceeds
the requested 10 Gbps capability without paid Tier 1 networking. Tier 1 is
not available at this N2 size and is therefore not configured. gVNIC and
Premium Network Tier are explicit.

Two Local SSD devices are the smallest supported count for this shape. Each
device is 375 GiB. RAID0 is intentional because this storage is a cache for
one assigned shard; startup and completion receipts, compressed Parquet, and
Megatron files must be uploaded continuously to GCS.

Official constraints:

- [N2 machine types and bandwidth](https://cloud.google.com/compute/docs/general-purpose-machines#n2_machine_types)
- [Local SSD sizes, allowed counts, and data-loss semantics](https://cloud.google.com/compute/docs/disks/local-ssd)
- [Cloud Storage location guidance](https://cloud.google.com/storage/docs/locations)

## Provision

Copy the examples to untracked `terraform.tfvars` files and review both
plans. No cloud resource is created by validation.

```bash
cd infra/gcp_corpus_pool
cp foundation/terraform.tfvars.example foundation/terraform.tfvars
cp workers/terraform.tfvars.example workers/terraform.tfvars

terraform -chdir=foundation init
terraform -chdir=foundation plan -out=foundation.tfplan
terraform -chdir=foundation apply foundation.tfplan

# Migrate the bootstrap local state into the newly created protected bucket.
cp foundation/backend.tf.example foundation/backend.tf
terraform -chdir=foundation init -migrate-state \
  -backend-config="bucket=natural-bison-491019-t9-cppmega-corpus" \
  -backend-config="prefix=terraform/foundation"

cp workers/backend.tf.example workers/backend.tf
terraform -chdir=workers init \
  -backend-config="bucket=natural-bison-491019-t9-cppmega-corpus" \
  -backend-config="prefix=terraform/workers"
terraform -chdir=workers plan -out=workers.tfplan
terraform -chdir=workers apply workers.tfplan
```

Replace the example bucket in both backend commands if `bucket_name` was
overridden. Backend configuration is intentionally not committed and the
generated `backend.tf` is ignored. The short-lived local foundation state is
migrated immediately after bucket creation; the disposable worker state is
remote from its first apply, so another operator can reliably destroy it.

The default bucket name is `<project_id>-cppmega-corpus`. It is regional
`STANDARD` storage in `us-central1`, with public access prevention, uniform
bucket-level IAM, object versioning, seven-day soft delete, and Terraform
`prevent_destroy`. The worker service account can read existing objects and
create new objects only below the configured `runs/` prefix, but cannot read
Terraform state outside that prefix, overwrite/delete objects, delete the
bucket, or mutate infrastructure. Receipts and checkpoints therefore use
immutable names.

Bucket listing is intentionally denied to workers because IAM Conditions
cannot prefix-filter `storage.objects.list`. Assignment manifests must name
exact `gs://` input objects; workers may fetch those exact objects below
`runs/` without being able to enumerate the bucket or see Terraform state.

The pool has no open public SSH rule. IAP TCP forwarding can reach port 22;
the operator still needs the appropriate OS Login and IAP IAM permissions.
Use the generated `iap_ssh_commands` output.

## Worker contract

On first boot, the startup script:

1. installs a small Debian toolset and the Google Cloud CLI;
2. discovers Local SSDs through stable `/dev/disk/by-id/` names;
3. assembles them as RAID0 and mounts `/mnt/cppmega-stage`;
4. creates `input`, `work`, `output`, and `receipts` staging directories;
5. writes `/etc/cppmega/worker.json` with the exact run and shard identity;
6. uploads a ready receipt to
   `gs://<bucket>/<prefix>/<run_id>/control/ready/<worker>.<boot-id>.json`;
7. optionally downloads a SHA-256-pinned runner from GCS and executes it as
   the unprivileged `cppmega` user.

When a runner is configured, Terraform requires the complete five-part
binding: runner URI, runner SHA-256, bundle SHA-256, overlay SHA-256, and raw
manifest-file SHA-256. Startup checks the runner bytes and then checks that
the runner embeds the exact three input digests before executing it. This
prevents a correctly hashed but wrongly configured runner from silently
joining another source run.

Assignments should use the deterministic `worker_index` and `worker_count`
values. A runner must download only its assigned inputs, checkpoint reusable
state to GCS, upload output to a temporary object name, validate hashes and
row/token receipts, and only then publish its completion receipt. Never rely
on Local SSD surviving stop, deletion, host failure, or Spot preemption.

### Bounded slots per VM

`worker_count` remains the physical VM count. `slots_per_worker` derives the
logical manifest worker count (`worker_count * slots_per_worker`), and VM `v`
owns contiguous logical IDs `v * slots_per_worker .. v * slots_per_worker +
slots_per_worker - 1`. The default is one slot, preserving the original smoke
payload. Terraform caps the supported profile at two slots per VM and checks
aggregate parser and memory limits before apply.

Each slot gets a separate Git checkout, scratch tree, log, and receipt root.
The scheduler refuses a manifest whose logical worker list does not exactly
match the VM topology, and refuses aggregate resource overcommit. It publishes
an immutable slot receipt only after every source receipt has been read back;
the source worker also publishes an immutable assignment pointer, so a crash
can skip assignments already confirmed in GCS. No worker-local dedup database
is shared or enabled.

For `n2-standard-16` (16 vCPU, 64 GB), the recommended first production
profile is four VMs with `slots_per_worker = 2`,
`parse_workers_per_slot = 6`, `memory_limit_gb_per_slot = 24`,
`cpu_budget_vcpus = 16`, and `memory_budget_gb = 56`. Build the source payload
with the same physical count and `--slots-per-worker 2`; this creates eight
logical manifest workers. Keep the default one-slot profile for the existing
smoke run and use a new run ID and new content-addressed payload for a two-slot
run.

Recommended object layout:

```text
gs://BUCKET/runs/RUN_ID/
  manifests/                 immutable source and shard manifests
  bootstrap/                 digest-named runner, git bundle, overlay
  control/ready/             startup receipts
  checkpoints/worker-NN/     resumable state
  staging/worker-NN/         unpublished output objects
  parquet/{code,pr_mr,ci}/    validated Parquet plus sidecars
  megatron/{code,pr_mr,ci}/  sealed sequence variants plus receipts
  receipts/                  membership, hashes, counts, validation, sealing
```

## Verified source pilot v4

The values in `workers/terraform.tfvars.example` reproduce the immutable
input chain exercised on all four live pilot VMs under
`runs/source-pilot-20260803-002`:

They are evidence/recovery values for that existing pilot, not a new run ID.
Do not apply the example over running workers merely to update metadata; use
a new immutable `run_id` and a newly rendered content-addressed runner for a
new production run.

| Role | SHA-256 | GCS suffix |
| --- | --- | --- |
| runner v4 | `cf62ebff18126915061ad2505f3f4773bb4e1d5702d78e50e51291fea3f79cf4` | `bootstrap/<sha>.source-worker-runner` |
| cppmega bundle | `cf312c1216ce521eedd5412b619d6de238f9baced1e9933b549faf366cd2445d` | `bootstrap/<sha>.cppmega.bundle` |
| data-prep overlay | `c075c268145782c6956d4ac8d13b26a73e6ae7aedc815debfb4056b1e464d323` | `bootstrap/<sha>.distributed-data-prep.tar.zst` |
| source manifest file | `8b928428520138ab90700640af3f2811bc988e3de883442eed938f28eb054ae2` | `manifests/<sha>.source-manifest.json` |

The manifest's separate logical digest is
`83541fc49aedab353d1580695494618f8b7d344a69ef9424fa561586eda29b13`;
do not substitute it for the raw file digest used in the object name and
runner check. The manifest pins code revision
`a732d6a9ddc9bdcb60f3a609637f74a64a950f86`, four canonical worker IDs, and
the same run output prefix.

`pilot/source-worker-runner.sh.tmpl` is the source template for those runner
bytes. Rendering its three placeholders with the bundle, overlay, and
manifest digests above yields runner SHA-256 `cf62ebff...f79cf4`, matching the
live object. The verified v4 fix performs `git bundle verify` inside the
freshly cloned repository; v3 incorrectly tried to verify without repository
context. The initial generic startup also used `gcloud storage cp` with a file
destination and failed before processing. Startup now uses the exact-object
JSON download path already proven by v4. Do not reuse either older path.

## Return the pool

Destroy only the disposable state:

```bash
terraform -chdir=workers plan -destroy -out=workers-destroy.tfplan
terraform -chdir=workers apply workers-destroy.tfplan
```

This deletes VMs, auto-deleted boot disks, Local SSD contents, placement
policy, and reserved external addresses. It cannot address the foundation
state or bucket. Do not run `terraform destroy` in `foundation/`; the bucket
also has an explicit `prevent_destroy` guard.

For a temporary pause that keeps static addresses, use
`gcloud compute instances stop`; Local SSD data is discarded by default, so
the workload must already be checkpointed. Setting `worker_count = 0` and
applying the workers state is equivalent to returning the pool.

## Validation

```bash
terraform fmt -check -recursive infra/gcp_corpus_pool
terraform -chdir=infra/gcp_corpus_pool/foundation init -backend=false
terraform -chdir=infra/gcp_corpus_pool/foundation validate
terraform -chdir=infra/gcp_corpus_pool/foundation test
terraform -chdir=infra/gcp_corpus_pool/workers init -backend=false
terraform -chdir=infra/gcp_corpus_pool/workers validate
terraform -chdir=infra/gcp_corpus_pool/workers test
```

`terraform plan` additionally needs Google credentials and already-applied
foundation resources. Before applying, check regional CPU, Local SSD, static
IPv4, and instance quotas plus zonal capacity. Compact placement can be set
to `false` if a zone cannot place all four instances together; Google limits
one compact placement policy to 22 instances, and Terraform enforces that
limit.

For `natural-bison-491019-t9`, the read-only quota check on 2026-08-03 showed
`0/3000` N2 vCPUs, `11/175` static addresses, `5/6000` instances, and no
effective on-demand Local SSD cap. The default pool fits with wide margin.
`PREEMPTIBLE_LOCAL_SSD_GB` was `0/0`, so `use_spot=true` requires a quota
increase even though ordinary preemptible CPU quota is available. Zonal
capacity is never guaranteed by quota.

## Cost envelope

The default pool consumes 64 vCPUs, 256 GB RAM, 3,000 GiB Local SSD, four
50 GB boot disks, and four in-use static IPv4 addresses. Compute, RAM, Local
SSD, boot disk, and IPv4 are billed until the workers state is destroyed.
Same-region reads from the regional GCS bucket have no outbound data-transfer
charge; stored objects, operations, soft-deleted versions, and external or
cross-region export are billed separately.

At the published `us-central1` on-demand rates checked 2026-08-03, the active
four-worker pool is approximately **$3.484/hour** or **$83.61/day**, before GCS
storage/operations and internet egress:

| Resource | Calculation | Pool cost/hour |
| --- | ---: | ---: |
| N2 compute + RAM | `4 × $0.776944` | `$3.107776` |
| Local SSD | `3,000 GiB × $0.000109589` | `$0.328767` |
| 50 GiB balanced boot disks | `200 GiB × $0.000136986` | `$0.027397` |
| In-use external IPv4 | `4 × $0.005` | `$0.020000` |

See the current [VM](https://cloud.google.com/products/compute/pricing/general-purpose),
[disk](https://cloud.google.com/compute/disks-image-pricing), and
[external IP](https://cloud.google.com/vpc/network-pricing#ipaddress) prices
before apply; prices, credits, and discounts are account- and date-dependent.
On-demand workers are the safe default. Enable Spot only after
forced-preemption resume and receipt checks pass, because Local SSD is not a
checkpoint.
