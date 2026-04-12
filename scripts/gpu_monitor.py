#!/usr/bin/env python3
"""Lightweight GPU + training monitor REST API.

No dependencies beyond stdlib. Runs on the H200 machine, polls nvidia-smi
and optionally tails training logs.

    python scripts/gpu_monitor.py                    # http://0.0.0.0:8471
    python scripts/gpu_monitor.py --port 9000        # custom port
    python scripts/gpu_monitor.py --log /path/to.log # also serve log tail

Endpoints:
    GET /gpu          — GPU status (memory, util, temp) as JSON
    GET /health       — quick healthcheck
    GET /log?n=50     — last N lines of training log
    GET /status       — combined GPU + log + disk info
    GET /iterations   — parsed training iterations from log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


_LOG_PATH: str | None = None


def _gpu_info() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in r.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 7:
                gpus.append({
                    "index": int(p[0]), "name": p[1],
                    "mem_used_mib": int(p[2]), "mem_total_mib": int(p[3]),
                    "mem_pct": round(int(p[2]) / max(int(p[3]), 1) * 100, 1),
                    "util_pct": int(p[4]), "temp_c": int(p[5]),
                    "power_w": float(p[6]),
                })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]


def _disk_info() -> dict:
    info = {}
    for path, label in [("/", "root"), ("/mnt/data", "data"), ("/home/dave", "home")]:
        try:
            u = shutil.disk_usage(path)
            info[label] = {
                "total_gb": round(u.total / 1e9, 1),
                "used_gb": round(u.used / 1e9, 1),
                "free_gb": round(u.free / 1e9, 1),
                "pct": round(u.used / max(u.total, 1) * 100, 1),
            }
        except OSError:
            pass
    return info


def _log_tail(n: int = 50) -> list[str]:
    if not _LOG_PATH or not os.path.exists(_LOG_PATH):
        return []
    try:
        r = subprocess.run(["tail", f"-{n}", _LOG_PATH],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.splitlines()
    except Exception:
        return []


def _parse_iterations(lines: list[str]) -> list[dict]:
    iters = []
    pat = re.compile(
        r"iteration\s+(\d+)/\s*(\d+).*?"
        r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
        r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*?"
        r"lm loss:\s*([\d.eE+-]+).*?"
        r"grad norm:\s*([\d.eE+-]+|inf|nan)",
    )
    for line in lines:
        m = pat.search(line)
        if m:
            iters.append({
                "iter": int(m.group(1)), "total": int(m.group(2)),
                "ms": float(m.group(3)), "tflops": float(m.group(4)),
                "loss": float(m.group(5)),
                "grad_norm": m.group(6),
            })
    return iters


def _tmux_sessions() -> list[str]:
    try:
        r = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=3)
        return r.stdout.strip().splitlines()
    except Exception:
        return []


class Handler(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._json({"status": "ok"})

        elif path == "/gpu":
            self._json({"gpus": _gpu_info()})

        elif path == "/log":
            n = int(qs.get("n", ["50"])[0])
            self._json({"lines": _log_tail(n), "path": _LOG_PATH})

        elif path == "/iterations":
            lines = _log_tail(500)
            self._json({"iterations": _parse_iterations(lines)})

        elif path == "/status":
            gpus = _gpu_info()
            lines = _log_tail(100)
            iters = _parse_iterations(lines)
            self._json({
                "gpus": gpus,
                "disk": _disk_info(),
                "tmux": _tmux_sessions(),
                "log_path": _LOG_PATH,
                "last_iterations": iters[-5:] if iters else [],
                "total_iterations": len(iters),
            })

        elif path == "" or path == "/":
            self._json({
                "endpoints": ["/health", "/gpu", "/log?n=50", "/iterations", "/status"],
                "log_path": _LOG_PATH,
            })
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        pass  # suppress access logs


def main():
    global _LOG_PATH
    parser = argparse.ArgumentParser(description="GPU Monitor REST API")
    parser.add_argument("--port", type=int, default=8471)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log", help="Training log file to monitor")
    args = parser.parse_args()
    _LOG_PATH = args.log
    server = HTTPServer((args.host, args.port), Handler)
    print(f"GPU Monitor: http://{args.host}:{args.port}")
    print(f"  /gpu       — GPU status")
    print(f"  /status    — full status")
    print(f"  /iterations — parsed training metrics")
    if _LOG_PATH:
        print(f"  /log       — tail of {_LOG_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
