"""Modal H100 harness for the Wave31 dPsiV_D helper-boundary probe.

The lane is intentionally H100-only. It compiles a tiny CUDA extension for the
`dPsiV_D` bf16 cast/fuse boundary, records ptxas output when available, checks
correctness against the torch reference, and writes timing/memory artifacts to
the shared Modal benchmark volume.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any

import modal

_REPO_ROOT = pathlib.Path(__file__).parent.parent

APP_NAME = "cppmega-wave31-dpsiv-helper-h100"
RESULTS_VOL = "cppmega-mamba3-benchmarks"
BENCH_DIR = "/benchmarks/mamba3_wave31_dpsiv_helper_h100"
GHCR_REPO = os.environ.get("GHCR_REPO", "ghcr.io/datasunriseou/cppmega")
GHCR_TAG = os.environ.get("GHCR_TAG", "785c3fd")
GHCR_REF = f"{GHCR_REPO}:{GHCR_TAG}"
GPU_SPEC = "H100"


def _image() -> modal.Image:
    return (
        modal.Image.from_registry(
            GHCR_REF,
            secret=modal.Secret.from_name("ghcr-pull"),
            add_python=None,
        )
        .env({"PYTHONPATH": "/opt/cppmega:/opt/megatron-lm"})
        .add_local_dir(str(_REPO_ROOT / "tools"), remote_path="/opt/cppmega/tools")
        .add_local_file(str(_REPO_ROOT / "pyproject.toml"), remote_path="/opt/cppmega/pyproject.toml")
    )


app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=True)
cache_vol = modal.Volume.from_name("cppmega-modal-cache", create_if_missing=True)
image = _image()


def _safe_run_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._") or "run"


def _parse_ptxas_resources(text: str) -> dict[str, Any]:
    lines: list[str] = []
    keep_followup = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if "ptxas" in lowered:
            lines.append(line)
            keep_followup = 3 if "function properties" in lowered else 0
            continue
        if keep_followup > 0 and ("stack frame" in lowered or "spill" in lowered):
            lines.append(line)
            keep_followup -= 1
    resources: list[dict[str, Any]] = []
    for line in lines:
        entry: dict[str, Any] = {"line": line}
        match = re.search(r"Used\s+(\d+)\s+registers", line)
        if match:
            entry["registers"] = int(match.group(1))
        match = re.search(r"used\s+(\d+)\s+barriers", line)
        if match:
            entry["barriers"] = int(match.group(1))
        for bytes_key, pattern in (
            ("smem_bytes", r"(\d+)\s+bytes\s+smem"),
            ("stack_frame_bytes", r"(\d+)\s+bytes\s+stack\s+frame"),
            ("spill_store_bytes", r"(\d+)\s+bytes\s+spill\s+stores"),
            ("spill_load_bytes", r"(\d+)\s+bytes\s+spill\s+loads"),
        ):
            match = re.search(pattern, line)
            if match:
                entry[bytes_key] = int(match.group(1))
        cmem = {int(idx): int(size) for size, idx in re.findall(r"(\d+)\s+bytes\s+cmem\[(\d+)\]", line)}
        if cmem:
            entry["cmem_bytes"] = cmem
        if len(entry) > 1:
            resources.append(entry)
    return {
        "line_count": len(lines),
        "lines_tail": lines[-80:],
        "resources": resources,
    }


def _run_capture(argv: list[str], env: dict[str, str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False, timeout=timeout_s)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_stdout_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "candidate" in parsed:
            return parsed
    return None


def _summary(report: dict[str, Any]) -> str:
    probe = report.get("probe_result") or {}
    timing = probe.get("timing_ms") or {}
    memory = probe.get("memory") or {}
    correctness = probe.get("correctness") or {}
    ptxas = report.get("ptxas") or {}
    lines = [
        f"# Wave31 dPsiV_D H100 helper probe: {report['run_id']}",
        "",
        f"status: `{report['status']}`",
        f"returncode: `{report['returncode']}`",
        f"gpu: `{report.get('nvidia_smi', '').strip()}`",
        f"kernel_ms: `{timing.get('kernel')}`",
        f"torch_reference_ms: `{timing.get('torch_reference')}`",
        f"max_abs_vs_ref: `{correctness.get('max_abs_vs_torch_bf16_reference')}`",
        f"peak_allocated_delta_bytes: `{memory.get('peak_allocated_delta_bytes')}`",
        f"scratch_bytes: `{memory.get('scratch_bytes')}`",
        f"ptxas_line_count: `{ptxas.get('line_count')}`",
        "",
        "## ptxas resources",
        "",
    ]
    resources = ptxas.get("resources") or []
    if resources:
        lines.extend(f"- `{item}`" for item in resources)
    else:
        lines.append("- no parsed resource line")
    return "\n".join(lines) + "\n"


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=2400,
    volumes={"/vol": results_vol, "/cache": cache_vol},
)
def run_probe(
    run_id: str,
    tiles: int = 512,
    cs: int = 128,
    r: int = 2,
    p: int = 128,
    warmup: int = 20,
    iters: int = 100,
    timeout_s: int = 1800,
    verbose_build: bool = True,
) -> dict[str, Any]:
    safe_id = _safe_run_id(run_id)
    out_dir = pathlib.Path("/vol") / BENCH_DIR.lstrip("/") / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_json = out_dir / "probe.json"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "/opt/cppmega:/opt/megatron-lm",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "TORCH_EXTENSIONS_DIR": f"/cache/torch_extensions/mamba3_wave31_dpsiv_helper_h100/{safe_id}",
            "MAX_JOBS": "1",
        }
    )
    if pathlib.Path("/usr/local/cuda").exists():
        env.setdefault("CUDA_HOME", "/usr/local/cuda")

    nvidia = _run_capture(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
        env,
        timeout_s=60,
    )
    nvcc = _run_capture(["bash", "-lc", "nvcc --version"], env, timeout_s=60)
    torch_info = _run_capture(
        [
            "python",
            "-c",
            (
                "import torch, json;"
                "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
                "'device': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0)}))"
            ),
        ],
        env,
        timeout_s=60,
    )

    cmd = [
        "python",
        "/opt/cppmega/tools/probes/mamba3_wave31_dpsiv_d_boundary_probe.py",
        "--tiles",
        str(tiles),
        "--cs",
        str(cs),
        "--r",
        str(r),
        "--p",
        str(p),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--json",
        str(probe_json),
    ]
    if verbose_build:
        cmd.append("--verbose-build")

    started = time.time()
    proc = _run_capture(cmd, env, timeout_s=timeout_s)
    elapsed_s = time.time() - started
    combined_log = proc.stdout + "\n" + proc.stderr
    ptxas = _parse_ptxas_resources(combined_log)

    (out_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": cmd,
                "env": {
                    key: env[key]
                    for key in ("PYTHONPATH", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "MAX_JOBS")
                    if key in env
                },
                "ghcr_ref": GHCR_REF,
                "gpu": GPU_SPEC,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (out_dir / "nvidia_smi.txt").write_text(nvidia.stdout + nvidia.stderr, encoding="utf-8")
    (out_dir / "nvcc.txt").write_text(nvcc.stdout + nvcc.stderr, encoding="utf-8")
    (out_dir / "torch_info.txt").write_text(torch_info.stdout + torch_info.stderr, encoding="utf-8")

    probe_result: dict[str, Any] | None = None
    if probe_json.exists():
        try:
            probe_result = json.loads(probe_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            probe_result = {"status": "invalid_probe_json", "error": str(exc)}
    else:
        probe_result = _extract_stdout_json(proc.stdout)

    status = "ok" if proc.returncode == 0 and probe_result and probe_result.get("status") == "GO_COMPILED_H100" else "failed"
    report: dict[str, Any] = {
        "run_id": safe_id,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "shape": {"tiles": tiles, "cs": cs, "r": r, "p": p},
        "nvidia_smi": (nvidia.stdout + nvidia.stderr).strip(),
        "nvcc": (nvcc.stdout + nvcc.stderr).strip(),
        "torch_info": (torch_info.stdout + torch_info.stderr).strip(),
        "ptxas": ptxas,
        "probe_result": probe_result,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "artifacts": str(out_dir),
    }
    _write_json(out_dir / "report.json", report)
    (out_dir / "summary.md").write_text(_summary(report), encoding="utf-8")
    results_vol.commit()
    cache_vol.commit()
    return report


@app.local_entrypoint()
def launch_probe(
    run_id: str = "wave31_dpsiv_helper_h100",
    tiles: int = 512,
    cs: int = 128,
    r: int = 2,
    p: int = 128,
    warmup: int = 20,
    iters: int = 100,
    timeout_s: int = 1800,
    verbose_build: bool = True,
):
    result = run_probe.remote(run_id, tiles, cs, r, p, warmup, iters, timeout_s, verbose_build)
    print(json.dumps(result, indent=2, sort_keys=True))
