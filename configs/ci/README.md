# Repository-owned CI

This path runs `cppmega` CI directly on repository-owned machines. It does not
require a GitHub Actions job, registration token, or GitHub-hosted runner.

## Commands

List the pinned inventory and lane commands without making network calls:

```bash
python3 scripts/ci/run_repository_ci.py list
python3 scripts/ci/run_repository_ci.py list --json
```

Run read-only identity, SSH, platform, dependency, and CUDA preflights. This
writes a machine-readable orchestration receipt but does not stage source or
run tests:

```bash
python3 scripts/ci/run_repository_ci.py run \
  --dry-run \
  --receipt-dir outputs/ci_diagnostics/repository-ci
```

Run one selected lane after its preflight passes:

```bash
python3 scripts/ci/run_repository_ci.py run \
  --host local-macos \
  --lane macos-contracts \
  --ref HEAD \
  --receipt-dir outputs/ci_diagnostics/repository-ci
```

Run the focused companion `cppmega.mlx` lane directly against its local
worktree while retaining the same before/after provenance and redacted logs:

```bash
python3 scripts/ci/run_repository_ci.py lane \
  --lanes-config configs/ci/cppmega_mlx_lanes.json \
  --lane macos-cppmega-mlx-contracts \
  --repo-root /path/to/cppmega_mlx \
  --python /path/to/cppmega_mlx/.venv/bin/python \
  --receipt-dir /path/to/cppmega_mlx/outputs/ci_diagnostics/repository-ci
```

The companion invocation does not install or update that worktree's
dependencies. It fails closed when the configured interpreter or modules are
missing.

The full default matrix adds `linux-contracts` and `linux-cuda` on
`10.0.0.16`. The runner stages the exact requested commit tree in a temporary
worktree, runs each command with a bounded timeout, and deletes only that
temporary worktree after collecting its receipts.

## Host identity and credentials

`10.0.0.11` and `10.0.0.16` have repository-pinned ED25519 public keys and
fingerprints. Every SSH invocation uses a dedicated generated `known_hosts`
file, `StrictHostKeyChecking=yes`, public-key-only batch authentication, no
agent forwarding, no forwarding rules, and bounded connection attempts.

`10.0.0.12` is quarantined. A dry-run may report its currently observed
fingerprint, but the runner never authenticates or dispatches to it until an
operator verifies the key out of band and updates `hosts.json`.

No credential belongs in either JSON config. The runner can use the caller's
SSH agent. To select a private key without storing its path in a receipt, set
`CPPMEGA_CI_SSH_IDENTITY_FILE` in the caller environment. Command processes
receive an allowlisted environment with credential-shaped variables removed,
and their logs redact common credential formats.

## Immutable environments

The lane config contains no dependency installation step. Each host must have
its configured Python interpreter and required modules pre-provisioned. A
missing module, tool, or CUDA runtime blocks the lane during preflight; the
runner does not repair the host with `pip`, `conda`, `apt`, `brew`, or another
mutable package manager.

## Receipts

Receipts are written under the requested base directory:

```text
<receipt-dir>/<run-id>/orchestration.json
<receipt-dir>/<run-id>/<host-id>/preflight.json
<receipt-dir>/<run-id>/<host-id>/<lane-id>/receipt.json
<receipt-dir>/<run-id>/<host-id>/<lane-id>/*.log
```

Lane receipts contain the requested source commit, source tree, archive
SHA-256, host details, bounded step results, and Git/worktree fingerprints
captured immediately before and after tests. A provenance change fails the
lane even when all test commands exit zero. The orchestration receipt applies
the same before/after check to the source worktree.

Exit code `0` means the selected dry-run or lanes passed, `1` means a lane or
provenance check failed, and `2` means configuration or required preflight
blocked dispatch.
