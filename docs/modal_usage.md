# Modal CLI Usage Notes

## Key Commands

```bash
# Run a script (blocking, prints result to stdout)
modal run scripts/modal_cppmega_run_tests.py

# Run detached (returns immediately, app runs in cloud)
modal run --detach scripts/modal_cppmega_run_tests.py

# List apps
modal app list
modal app list --json

# Get logs from a completed/running app
modal app logs <app_id>

# Volume operations
modal volume get <volume_name> <remote_path> <local_dir>
modal volume ls <volume_name> <path>

# Secrets
modal secret list
modal secret create <name> KEY=VALUE
```

## Patterns (from nanochat)

### Standard App Structure
```python
import modal

app = modal.App("app-name")
vol = modal.Volume.from_name("results-vol", create_if_missing=True)

@app.function(image=img, gpu="H200:1", timeout=3600, volumes={"/results": vol})
def work() -> dict:
    result = do_work()
    # Persist to volume for later retrieval
    pathlib.Path("/results/latest.json").write_text(json.dumps(result))
    vol.commit()
    return result

@app.local_entrypoint()
def main():
    result = work.remote()  # blocking
    # or: work.spawn()  # non-blocking
```

### GPU Selection
GPU is set via env var at module load time (baked into decorator):
```bash
CPPMEGA_MODAL_GPU="H200:1" modal run scripts/modal_cppmega_run_tests.py
```

### Result Retrieval
1. **Volume (preferred for detached):** Write JSON to volume, read later with
   `modal volume get cppmega-test-results /results/latest.json /tmp/`
2. **Stdout (for blocking runs):** `local_entrypoint()` prints result
3. **App logs:** `modal app logs <app_id>` shows remote container stdout
   (NOT local_entrypoint output)

### Image Caching
- First run builds image layers (slow: 2-10 min)
- Subsequent runs reuse cached layers (fast: <5s deploy)
- `add_local_dir(..., copy=True)` overlays code without rebuilding base

### Detached + Polling Pattern (nanochat)
```python
# Launch
subprocess.run(["modal", "run", "--detach", "scripts/app.py"])
# Poll
result = subprocess.run(["modal", "app", "list", "--json"], capture_output=True)
apps = json.loads(result.stdout)
# Read results from volume
subprocess.run(["modal", "volume", "get", "vol-name", "/results/latest.json", "/tmp/"])
```

## Our cppmega Setup

- **Image:** `ghcr.io/datasunriseou/cppmega:latest` (private, needs `ghcr-pull` secret)
- **Secret:** `ghcr-pull` (REGISTRY_USERNAME + REGISTRY_PASSWORD)
- **Volume:** `cppmega-test-results` (test result JSONs)
- **Data Volume:** `nanochat-training-data` (parquet + sidecar data)
- **GPU:** H200:1 default, override with `CPPMEGA_MODAL_GPU`
- **Megatron:** /opt/megatron-lm at commit 7d095b98 (image HEAD)

## Gotchas

- `modal app logs` shows REMOTE container logs, not local_entrypoint print output
- Image rebuild triggered by any change to image definition (pip_install, env, etc)
- `--detach` still blocks during image build phase
- Volume writes need explicit `.commit()` call
- pytest-xdist must be installed in image for `-n auto`
