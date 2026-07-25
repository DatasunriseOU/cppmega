#!/usr/bin/env python3
"""Fetch CI logs from GitHub Actions and extract compiler diagnostics."""
import json
import os
import re
import subprocess
import sys
import time

REPOS = [
    "CGAL/cgal", "ChibiOS/ChibiOS", "Cisco-Talos/clamav", "ClickHouse/ClickHouse",
    "CrowCpp/Crow", "DPDK/dpdk", "Dao-AILab/flash-attention", "DaveGamble/cJSON",
    "DiligentGraphics/DiligentEngine", "FFTW/fftw3", "FreeCAD/FreeCAD",
    "FreeRTOS/FreeRTOS", "GNOME/libxml2", "Geant4/geant4", "HDFGroup/hdf5",
]

OUT_DIR = "/Volumes/external/sources/cppmega/outputs/ci_diagnostics"

# Diagnostic patterns
GCC_CLANG_RE = re.compile(
    r'^(?:\./)?([^\s:]+):(\d+):(\d+):\s*(error|warning|fatal error):\s*(.+)$', re.MULTILINE
)
MSVC_RE = re.compile(
    r'^([^\(]+)\((\d+)\):\s*(error|warning)\s+(C\d+):\s*(.+)$', re.MULTILINE
)
LINK_RE = re.compile(
    r'(undefined reference to\s+[`\'](.+?)[`\']|LNK2019:|LNK2001:)', re.MULTILINE
)
CMAKE_RE = re.compile(
    r'CMake Error at ([^\s:]+):(\d+)', re.MULTILINE
)

def run_gh(args, timeout=60):
    """Run gh command and return stdout or None on error."""
    try:
        result = subprocess.run(
            ["gh"] + args, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return None

def fetch_runs(repo):
    """Fetch last 10 completed runs."""
    url = f'repos/{repo}/actions/runs?per_page=10&status=completed'
    out = run_gh(["api", url, "--jq", ".workflow_runs[] | {id, name, conclusion, head_sha}"])
    if not out:
        return []
    runs = []
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs

def fetch_jobs(repo, run_id):
    """Fetch jobs for a run."""
    url = f'repos/{repo}/actions/runs/{run_id}/jobs?per_page=20'
    out = run_gh(["api", url, "--jq", ".jobs[] | {id, name, conclusion}"])
    if not out:
        return []
    jobs = []
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        try:
            jobs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return jobs

def fetch_annotations(repo, job_id):
    """Fetch check-run annotations."""
    url = f'repos/{repo}/check-runs/{job_id}/annotations'
    out = run_gh(["api", url])
    if not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []

def parse_log_diagnostics(log_text):
    """Parse diagnostics from log text."""
    diagnostics = []
    seen = set()

    # GCC/Clang style
    for m in GCC_CLANG_RE.finditer(log_text):
        path, line, col, severity, msg = m.groups()
        # Filter out noise
        if path.startswith("##[") or "node_modules" in path:
            continue
        key = (path, line, col, severity, msg[:80])
        if key in seen:
            continue
        seen.add(key)
        compiler = "clang" if "clang" in msg.lower() else "gcc"
        diagnostics.append({
            "file": path, "line": int(line), "col": int(col),
            "severity": severity, "message": msg.strip()[:200],
            "compiler": compiler
        })

    # MSVC style
    for m in MSVC_RE.finditer(log_text):
        path, line, severity, code, msg = m.groups()
        key = (path, line, severity, code, msg[:80])
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append({
            "file": path.strip(), "line": int(line), "col": 0,
            "severity": severity, "message": f"{code}: {msg.strip()[:180]}",
            "compiler": "msvc"
        })

    # Linker errors
    for m in LINK_RE.finditer(log_text):
        text = m.group(0)[:200]
        key = ("link", text[:80])
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append({
            "file": "", "line": 0, "col": 0,
            "severity": "error", "message": text,
            "compiler": "linker"
        })

    # CMake errors
    for m in CMAKE_RE.finditer(log_text):
        path, line = m.groups()
        key = ("cmake", path, line)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append({
            "file": path, "line": int(line), "col": 0,
            "severity": "error", "message": "CMake Error",
            "compiler": "cmake"
        })

    return diagnostics[:50]  # Cap at 50 per job

def fetch_job_log(repo, job_id):
    """Fetch job log, filtering for error/warning lines if large."""
    url = f'repos/{repo}/actions/jobs/{job_id}/logs'
    try:
        # Try direct fetch with timeout
        result = subprocess.run(
            ["gh", "api", url], capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0:
            return None
        log = result.stdout
        # If log is large, filter to relevant lines
        if len(log) > 5_000_000:
            lines = log.split("\n")
            filtered = [l for l in lines if re.search(
                r'(error|warning|undefined reference|LNK\d|CMake Error)', l, re.IGNORECASE
            )]
            log = "\n".join(filtered)
        return log
    except (subprocess.TimeoutExpired, Exception):
        return None

def process_repo(repo):
    """Process a single repo and return list of diagnostic records."""
    records = []
    print(f"  Processing {repo}...", flush=True)

    runs = fetch_runs(repo)
    time.sleep(1)

    if not runs:
        print(f"    No completed runs found or API error for {repo}", flush=True)
        return records

    # Find failed runs, or take up to 3 most recent
    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    target_runs = failed_runs if failed_runs else runs[:3]

    for run in target_runs[:5]:  # Cap at 5 runs per repo
        run_id = run["id"]
        head_sha = run.get("head_sha", "")
        conclusion = run.get("conclusion", "")

        jobs = fetch_jobs(repo, run_id)
        time.sleep(1)

        if not jobs:
            continue

        # Focus on failed jobs, or build-related jobs
        failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
        if not failed_jobs:
            # Look for build jobs
            failed_jobs = [j for j in jobs if any(
                kw in j.get("name", "").lower()
                for kw in ["build", "compile", "make", "cmake", "gcc", "clang", "msvc"]
            )][:2]

        for job in failed_jobs[:3]:  # Cap at 3 jobs per run
            job_id = job["id"]
            job_name = job.get("name", "")

            # Try annotations first
            annotations = fetch_annotations(repo, job_id)
            time.sleep(1)

            diagnostics = []
            if annotations:
                for ann in annotations:
                    if ann.get("annotation_level") in ("failure", "warning", "error"):
                        diagnostics.append({
                            "file": ann.get("path", ""),
                            "line": ann.get("start_line", 0),
                            "col": ann.get("start_column", 0),
                            "severity": "error" if ann.get("annotation_level") == "failure" else "warning",
                            "message": ann.get("message", "")[:200],
                            "compiler": "unknown"
                        })

            # Also try log parsing if annotations didn't yield much
            if len(diagnostics) < 3:
                log = fetch_job_log(repo, job_id)
                if log:
                    log_diags = parse_log_diagnostics(log)
                    # Merge, avoiding duplicates
                    existing_keys = {(d["file"], d["line"], d["message"][:50]) for d in diagnostics}
                    for d in log_diags:
                        key = (d["file"], d["line"], d["message"][:50])
                        if key not in existing_keys:
                            diagnostics.append(d)
                            existing_keys.add(key)
                time.sleep(1)

            if diagnostics:
                records.append({
                    "repo": repo,
                    "run_id": run_id,
                    "job_name": job_name,
                    "commit_sha": head_sha,
                    "conclusion": conclusion,
                    "platform": "unknown",
                    "diagnostics": diagnostics[:30],
                    "build_command": ""
                })

    return records

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total_diags = 0
    repos_processed = 0
    failures = []

    for repo in REPOS:
        try:
            records = process_repo(repo)
            repos_processed += 1

            # Write JSONL
            repo_name = repo.split("/")[1]
            out_path = os.path.join(OUT_DIR, f"{repo_name}.jsonl")
            with open(out_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            n_diags = sum(len(r["diagnostics"]) for r in records)
            total_diags += n_diags
            print(f"    {repo}: {len(records)} records, {n_diags} diagnostics -> {out_path}", flush=True)

        except Exception as e:
            failures.append(f"{repo}: {e}")
            print(f"    ERROR processing {repo}: {e}", flush=True)

        time.sleep(1)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Repos processed: {repos_processed}/{len(REPOS)}")
    print(f"Total diagnostics: {total_diags}")
    if failures:
        print(f"Failures: {failures}")

if __name__ == "__main__":
    main()
