#!/usr/bin/env python3
"""nanochat Training Dashboard — lightweight Flask web UI.

Shows all running training processes on the host, their parameters,
live metrics, and history. Talks to the REST API on port 8471+ exposed
by each base_train.py process.

Usage:
    pip install flask
    python tools/dashboard/app.py                  # http://localhost:8080
    python tools/dashboard/app.py --port 9000       # custom port
    python tools/dashboard/app.py --host 0.0.0.0    # listen on all interfaces
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, TypeAlias

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.secret_key = os.environ.get("DASHBOARD_SECRET", secrets.token_hex(32))

# Default credentials (override via env)
ADMIN_USER = os.environ.get("DASHBOARD_USER", "admin")
ADMIN_PASS = os.environ.get("DASHBOARD_PASS", "admin8421")

# Training API settings
API_BASE_PORT = 8471
API_PORT_RANGE = 10
API_TIMEOUT = 3.0

# History storage (in-memory, persisted to JSON)
_HISTORY_FILE = Path(__file__).parent / "run_history.json"
_history: list[dict] = []

NumericConfigValue: TypeAlias = int | float
ChartMetricSeries: TypeAlias = dict[str, list[float | None]]


# ── Auth ─────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "")
        passwd = request.form.get("password", "")
        if user == ADMIN_USER and passwd == ADMIN_PASS:
            session["logged_in"] = True
            session["username"] = user
            next_url = request.args.get("next", url_for("index"))
            return redirect(next_url)
        flash("Invalid credentials", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Process discovery ────────────────────────────────────────────────────────

@dataclass
class ProcessInfo:
    pid: int
    cmdline: str
    port: int = API_BASE_PORT
    run_name: str | None = None
    gpu_index: int | None = None
    start_time: str | None = None


def discover_processes() -> list[ProcessInfo]:
    """Find running base_train processes on this host."""
    procs = []
    try:
        result = subprocess.run(
            ["ps", "aux", "--sort=-start_time"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "base_train" not in line:
                continue
            if any(skip in line for skip in ["collect_training", "grep", "dashboard"]):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue

            cmdline = " ".join(parts[10:])

            port = API_BASE_PORT
            m = re.search(r"--api_port[= ](\d+)", cmdline)
            if m:
                port = int(m.group(1))

            run_name = None
            m = re.search(r"--run[= ](\S+)", cmdline)
            if m:
                run_name = m.group(1)

            gpu_index = None
            m = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", cmdline)
            if m:
                gpu_index = int(m.group(1))

            # Process start time from ps
            start_time = parts[8] if len(parts) > 8 else None

            procs.append(ProcessInfo(
                pid=pid, cmdline=cmdline, port=port,
                run_name=run_name, gpu_index=gpu_index,
                start_time=start_time,
            ))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return procs


# ── Training API queries ─────────────────────────────────────────────────────

def query_api(port: int, endpoint: str, method: str = "GET",
              body: dict | None = None, timeout: float = API_TIMEOUT) -> dict | None:
    """Query a training process REST API."""
    url = f"http://127.0.0.1:{port}{endpoint}"
    try:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return None


def scan_active_ports() -> list[int]:
    """Scan port range for responsive training APIs."""
    active = []
    for p in range(API_BASE_PORT, API_BASE_PORT + API_PORT_RANGE):
        status = query_api(p, "/status", timeout=1.0)
        if status is not None:
            active.append(p)
    return active


def get_full_run_info(port: int) -> dict[str, Any]:
    """Get combined status + config + memory for a port."""
    info: dict[str, Any] = {"port": port, "reachable": False}
    status = query_api(port, "/status")
    if status is None:
        return info
    info["reachable"] = True
    info["status"] = status
    info["config"] = query_api(port, "/config") or {}
    info["memory"] = query_api(port, "/memory") or {}
    return info


# ── History ──────────────────────────────────────────────────────────────────

def _load_history():
    global _history
    if _HISTORY_FILE.exists():
        try:
            _history = json.loads(_HISTORY_FILE.read_text())
        except (json.JSONDecodeError, Exception):
            _history = []


def _save_history():
    _HISTORY_FILE.write_text(json.dumps(_history, indent=2, default=str))


def _record_snapshot(runs: list[dict]):
    """Record a snapshot of current runs into history."""
    ts = datetime.now(timezone.utc).isoformat()
    for run in runs:
        if not run.get("reachable"):
            continue
        status = run.get("status", {})
        mem = run.get("memory", {})
        entry = {
            "timestamp": ts,
            "port": run["port"],
            "run_name": status.get("run_name", "unknown"),
            "step": status.get("step", 0),
            "loss": status.get("loss"),
            "loss_val": status.get("loss_val"),
            "val_bpb": status.get("val_bpb"),
            "lr": status.get("lr"),
            "tok_per_sec": status.get("tok_per_sec"),
            "mfu": status.get("mfu"),
            "eta_seconds": status.get("eta_seconds"),
            "grad_norm": status.get("grad_norm"),
            "mem_used_gb": mem.get("used_gb") or mem.get("allocated_gb"),
            "mem_peak_gb": mem.get("peak_gb") or mem.get("reserved_gb"),
        }
        # Deduplicate: skip if same step as last entry for this port
        port_history = [h for h in _history if h.get("port") == run["port"]]
        if port_history and port_history[-1].get("step") == entry["step"]:
            continue
        _history.append(entry)
    # Keep last 50K entries (more history for better analytics)
    if len(_history) > 50000:
        _history[:] = _history[-50000:]
    _save_history()


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple least-squares linear regression. Returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def compute_analytics(port_history: list[dict]) -> dict:
    """Compute trend analytics for a run's history."""
    if len(port_history) < 2:
        return {}

    # Filter entries with valid loss
    valid = [(h["step"], h["loss"]) for h in port_history
             if h.get("loss") is not None and h.get("step") is not None]
    if len(valid) < 2:
        return {}

    steps, losses = zip(*valid)

    # Recent window for trend (last 20% or at least 10 points)
    window = max(10, len(valid) // 5)
    recent_steps = list(steps[-window:])
    recent_losses = list(losses[-window:])

    slope, intercept = _linear_regression(
        [float(s) for s in recent_steps],
        [float(loss_value) for loss_value in recent_losses],
    )

    # Predict loss at current step + 1000, +5000, +10000
    current_step = steps[-1]
    predictions = {}
    for delta in [1000, 5000, 10000]:
        pred = slope * (current_step + delta) + intercept
        predictions[f"step_{current_step + delta}"] = round(pred, 4)

    # Loss improvement rate (loss drop per 1000 steps)
    loss_per_1k = slope * 1000

    # Convergence estimate (steps to reach target losses)
    convergence = {}
    current_loss = losses[-1]
    for target in [1.0, 0.5, 0.1]:
        if slope < 0 and current_loss > target:
            steps_to_target = (target - intercept) / slope - current_step
            if steps_to_target > 0:
                convergence[f"loss_{target}"] = int(steps_to_target)

    # Smoothed loss (EMA)
    alpha = 0.1
    ema = losses[0]
    smoothed: list[float | None] = []
    for loss_value in losses:
        ema = alpha * loss_value + (1 - alpha) * ema
        smoothed.append(round(ema, 4))

    return {
        "trend_slope": round(slope, 8),
        "trend_intercept": round(intercept, 4),
        "loss_per_1k_steps": round(loss_per_1k, 4),
        "predictions": predictions,
        "convergence": convergence,
        "window_size": window,
        "total_points": len(valid),
        "smoothed_loss_latest": smoothed[-1] if smoothed else None,
    }


# ── Background poller (records history even without browser visits) ──────────

_poller_running = False


def _background_poller():
    """Poll active training APIs every 15s and record history."""
    global _poller_running
    _poller_running = True
    while _poller_running:
        try:
            active_ports = scan_active_ports()
            runs = []
            for port in active_ports:
                info = get_full_run_info(port)
                runs.append(info)
            if runs:
                _record_snapshot(runs)
        except Exception:
            pass
        time.sleep(15)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    """Main dashboard — shows all running training processes."""
    processes = discover_processes()
    active_ports = scan_active_ports()

    runs = []
    for port in active_ports:
        info = get_full_run_info(port)
        # Match to discovered process
        proc = next((p for p in processes if p.port == port), None)
        if proc:
            info["process"] = {
                "pid": proc.pid,
                "cmdline": proc.cmdline,
                "gpu_index": proc.gpu_index,
                "start_time": proc.start_time,
            }
        runs.append(info)

    # Also show processes that are running but API unreachable
    api_ports = set(active_ports)
    for proc in processes:
        if proc.port not in api_ports:
            runs.append({
                "port": proc.port,
                "reachable": False,
                "process": {
                    "pid": proc.pid,
                    "cmdline": proc.cmdline,
                    "gpu_index": proc.gpu_index,
                    "start_time": proc.start_time,
                },
            })

    _record_snapshot(runs)

    return render_template("index.html", runs=runs, now=datetime.now(timezone.utc))


@app.route("/run/<int:port>")
@login_required
def run_detail(port: int):
    """Detailed view for a single training run."""
    info = get_full_run_info(port)
    proc = next((p for p in discover_processes() if p.port == port), None)
    if proc:
        info["process"] = {
            "pid": proc.pid,
            "cmdline": proc.cmdline,
            "gpu_index": proc.gpu_index,
            "start_time": proc.start_time,
        }

    # Get history for this port
    run_history = [h for h in _history if h.get("port") == port][-500:]
    analytics = compute_analytics(run_history)

    return render_template("run_detail.html", run=info, port=port,
                           history=run_history, analytics=analytics)


@app.route("/run/<int:port>/config", methods=["POST"])
@login_required
def update_config(port: int):
    """Update runtime config for a training run."""
    updates: dict[str, NumericConfigValue] = {}
    for key, value in request.form.items():
        if key.startswith("_"):
            continue
        try:
            # Try int first, then float
            updates[key] = int(value)
        except ValueError:
            try:
                updates[key] = float(value)
            except ValueError:
                flash(f"Invalid value for {key}: {value}", "error")
                return redirect(url_for("run_detail", port=port))

    if updates:
        result = query_api(port, "/config", method="POST", body=updates)
        if result and result.get("ok"):
            flash(f"Updated: {', '.join(updates.keys())}", "success")
        else:
            error = result.get("error", "Unknown error") if result else "API unreachable"
            flash(f"Update failed: {error}", "error")

    return redirect(url_for("run_detail", port=port))


@app.route("/run/<int:port>/checkpoint", methods=["POST"])
@login_required
def trigger_checkpoint(port: int):
    result = query_api(port, "/checkpoint", method="POST")
    if result and result.get("ok"):
        flash("Checkpoint requested", "success")
    else:
        flash("Checkpoint request failed", "error")
    return redirect(url_for("run_detail", port=port))


@app.route("/run/<int:port>/shutdown", methods=["POST"])
@login_required
def trigger_shutdown(port: int):
    result = query_api(port, "/shutdown", method="POST")
    if result and result.get("ok"):
        flash("Shutdown requested (saving checkpoint first)", "success")
    else:
        flash("Shutdown request failed", "error")
    return redirect(url_for("run_detail", port=port))


@app.route("/history")
@login_required
def history():
    """View run history."""
    # Group by run_name
    by_run: dict[str, list[dict]] = {}
    for entry in _history:
        name = entry.get("run_name", "unknown")
        by_run.setdefault(name, []).append(entry)
    return render_template("history.html", by_run=by_run)


@app.route("/api/runs")
@login_required
def api_runs():
    """JSON API for dashboard auto-refresh."""
    active_ports = scan_active_ports()
    runs = [get_full_run_info(p) for p in active_ports]
    return jsonify(runs)


@app.route("/api/chart/<int:port>")
@login_required
def api_chart_data(port: int):
    """JSON chart data for a run — loss, lr, tok/sec, mfu, val_bpb over steps.

    Returns arrays ready for Chart.js consumption, plus regression trendline
    and analytics (predictions, convergence estimates).
    """
    limit = request.args.get("limit", 2000, type=int)
    run_history = [h for h in _history if h.get("port") == port][-limit:]
    analytics = compute_analytics(run_history)

    # Build per-metric series (step → value)
    metrics: ChartMetricSeries = {
        "loss": [], "loss_val": [], "val_bpb": [], "lr": [],
        "tok_per_sec": [], "mfu": [], "grad_norm": [],
        "mem_used_gb": [],
    }
    steps: list[int] = []

    for h in run_history:
        step = h.get("step")
        if step is None:
            continue
        steps.append(step)
        for key in metrics:
            val = h.get(key)
            metrics[key].append(val)

    # Build trendline for loss
    trendline_steps = []
    trendline_values = []
    if analytics.get("trend_slope") is not None:
        slope = analytics["trend_slope"]
        intercept = analytics["trend_intercept"]
        valid_steps = [step for step, loss_value in zip(steps, metrics["loss"]) if loss_value is not None]
        if valid_steps:
            s0, s1 = valid_steps[0], valid_steps[-1]
            # Extend trendline 20% into future
            s_future = s1 + int((s1 - s0) * 0.2)
            for s in [s0, s1, s_future]:
                trendline_steps.append(s)
                trendline_values.append(round(slope * s + intercept, 4))

    # EMA smoothed loss
    smoothed: list[float | None] = []
    alpha = 0.05
    ema = None
    for loss_value in metrics["loss"]:
        if loss_value is not None:
            ema = loss_value if ema is None else alpha * loss_value + (1 - alpha) * ema
            smoothed.append(round(ema, 4))
        else:
            smoothed.append(None)

    return jsonify({
        "steps": steps,
        "metrics": metrics,
        "smoothed_loss": smoothed,
        "trendline": {"steps": trendline_steps, "values": trendline_values},
        "analytics": analytics,
        "run_name": run_history[-1].get("run_name") if run_history else None,
    })


@app.route("/system")
@login_required
def system():
    """System info — GPU/CPU/memory."""
    info = {}
    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": parts[0], "name": parts[1],
                    "mem_used": parts[2], "mem_total": parts[3],
                    "util": parts[4], "temp": parts[5],
                })
        info["gpus"] = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpus"] = []

    # CPU/memory
    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        info["ram_used_gb"] = round(mem.used / (1024**3), 1)
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_percent"] = mem.percent
    except ImportError:
        info["cpu_percent"] = None

    return render_template("system.html", info=info)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="nanochat Training Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _load_history()

    # Start background poller (records history every 15s without browser)
    poller = threading.Thread(target=_background_poller, daemon=True)
    poller.start()

    print(f"nanochat Dashboard: http://{args.host}:{args.port}")
    print(f"Login: {ADMIN_USER} / {'*' * len(ADMIN_PASS)}")
    print("Background poller: recording history every 15s")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
