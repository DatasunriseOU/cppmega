# CI diagnostics

Local cache of CI diagnostics. Everything in this directory is gitignored
except this README.

## Lane receipts (`lane_receipts/`)

Receipts of the repository-owned CI lanes (`macos-contracts`,
`linux-contracts`) from the `CppMega self-hosted CI` workflow
(`.github/workflows/ci-self-hosted.yml`). Each lane run uploads its receipt
directory (`receipt.json` plus per-command logs) as a GitHub Actions artifact
named `cppmega-repository-ci-<lane>-<run_id>-<attempt>` with **14-day
retention**.

Refresh (requires `gh` logged in with `repo` + `workflow` scopes):

```bash
gh run list --repo DatasunriseOU/cppmega --workflow=ci-self-hosted.yml --limit 5
gh run download <run_id> --repo DatasunriseOU/cppmega \
  --name cppmega-repository-ci-macos-<run_id>-<attempt> \
  --dir outputs/ci_diagnostics/lane_receipts/macos-<run_id>-<attempt>
gh run download <run_id> --repo DatasunriseOU/cppmega \
  --name cppmega-repository-ci-linux-<run_id>-<attempt> \
  --dir outputs/ci_diagnostics/lane_receipts/linux-<run_id>-<attempt>
```

Last refresh: 2026-08-01.

- `macos-30676130389-1/` — run 30676130389, lane `macos-contracts`,
  status `passed`, completed 2026-08-01T00:45:07Z.
- `linux-30673683695-1/` — run 30673683695, lane `linux-contracts`,
  status `passed`, completed 2026-08-01T00:25:19Z (latest completed Linux
  lane at refresh time; run 30676130389's Linux job was still queued).

## Upstream repo diagnostics (`<repo>.jsonl`)

Per-repo compiler diagnostics extracted from GitHub Actions logs of upstream
C/C++ projects by `scripts/fetch_ci_diagnostics.py` (repo list in the
script; the sibling `fetch_ci_diagnostics.py` in the `cppmega.mlx` repo
covers the mlx-side repo set). Re-run with:

```bash
.venv/bin/python scripts/fetch_ci_diagnostics.py
```

Last refresh: 2026-08-01 (previous fetch was 2026-07-20).

`domain-routed-codegen.json` is written by the `frozen-domain-eval` step of
the CI lanes, not by the fetch scripts.
