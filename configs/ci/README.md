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

## Lanes

`lanes.json` declares the lanes that `.github/workflows/ci-self-hosted.yml`
delegates to. The workflow jobs only verify checkout identity and the
tokenizer contract, then hand off to `run_repository_ci.py lane` with
`--expected-source-commit`/`--expected-source-tree` source binding.

- `macos-contracts` (darwin/arm64, no CUDA, no test profile): interpreter
  comes from `CPPMEGA_CI_PYTHON` or the pinned default venv, with
  `MEGATRON_LM_REPO` exported by the workflow. Commands:
  `focused-contracts` (focused pytest selection), `frozen-domain-eval`,
  `tokenizer-contract`, `source-whitespace`.
- `linux-contracts` (linux/x86_64, no CUDA, `portable-data` test profile):
  `actions/setup-python` 3.13 plus `pytest numpy pyarrow tokenizers boto3
  zstandard`. Commands: `portable-contracts` (allowlisted pytest selection),
  `frozen-domain-eval`, `tokenizer-contract`, `source-whitespace`.
- `linux-cuda` (linux/x86_64, CUDA required, modules `torch`, `triton`,
  `megatron.core`, `mamba_ssm`): `cuda-contracts` plus `source-whitespace`.

## Failure receipts

Every lane writes `<receipt-dir>/receipt.json` and per-command logs, and the
workflow uploads that directory with `if-no-files-found: error`. Three layers
keep the root cause visible when something fails before the lane finishes:

1. A fatal error inside the orchestrator (unknown lane id, unreadable lane
   config, unexpected exception) is caught by `_write_early_failure_receipt`
   in `scripts/ci/repository_runner.py`, which writes a minimal
   `orchestrator`-stage receipt and exits `2`.
2. The macOS workflow job wraps its pre-python preamble (interpreter check,
   checkout identity, tokenizer contract) in a bash `ERR` trap that writes a
   minimal `workflow-preamble` receipt into the same directory before
   exiting, so a broken interpreter or checkout no longer surfaces as a bare
   artifact-upload error (incident: run 30638778832). The trap is disarmed
   right before the orchestrator runs so it never overwrites lane receipts.
3. A dedicated `if: failure()` step appends `receipt.json` to
   `$GITHUB_STEP_SUMMARY`, so the failure reason is readable in the job
   summary without downloading the artifact.

## Adding a test to a lane

1. Add the test path to the lane's pytest `argv` in `configs/ci/lanes.json`
   (both `macos-contracts` and `linux-contracts` when the test is portable).
2. If the test must run under the `portable-data` profile, add it to
   `conftest._PORTABLE_TEST_ALLOWLIST`; the profile refuses anything outside
   that list.
3. Keep `tests/test_workflow_runner_policy.py` in sync — it pins the
   workflow/lane wiring and fails when the structure drifts.

## Local reproduction

Run the same pytest selection a lane runs, from the repository root:

```bash
.venv/bin/python -m pytest <test files> -q
```

For the `portable-data` profile (`linux-contracts`):

```bash
CPPMEGA_TEST_PROFILE=portable-data .venv/bin/python -m pytest <test files> -q
```

To exercise the full lane runner locally, including preflight and receipts:

```bash
.venv/bin/python scripts/ci/run_repository_ci.py lane \
  --lanes-config configs/ci/lanes.json \
  --lane macos-contracts \
  --repo-root "$PWD" \
  --python .venv/bin/python \
  --receipt-dir outputs/ci_diagnostics/repository-ci
```
