#!/usr/bin/env python3
"""Validate the accepted GB10 Mamba MXFP8 training path.

The accepted path is:

* MXFP8 backward uses the TN adapter.
* BF16/dequantized MXFP8 backward fallbacks stay disabled and unused.
* Deprecated BF16 bridge, DSA gather-scatter, and FP8 activation legacy packer
  fail closed unless their explicit ACK env vars are set.

This probe is intentionally orchestration-only.  It runs the existing
``te_blockscaled_backward_probe.py`` for the CUDA/TE MXFP8 path and contains
small parsers for the probe output and one-step training logs.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


MXFP8_BF16_ACK = "CPPMEGA_I_UNDERSTAND_MXFP8_BF16_BACKWARD_BRIDGE_IS_DEPRECATED_AND_SLOW"
DSA_GATHER_ACK = "CPPMEGA_I_UNDERSTAND_DSA_GATHER_SCATTER_IS_DEPRECATED_AND_SLOW"
FP8_ACT_LEGACY_ACK = (
    "CPPMEGA_I_UNDERSTAND_FP8_ACTIVATION_LEGACY_BACKEND_IS_DEPRECATED_AND_SYNCY"
)

FALLBACK_STAT_KEYS = ("bf16_fallback_dgrad", "bf16_fallback_wgrad")
ADAPTER_STAT_KEYS = ("mxfp8_tn_adapter_dgrad", "mxfp8_tn_adapter_wgrad")
PASSTHROUGH_STAT_KEYS = ("native_passthrough_dgrad", "native_passthrough_wgrad")
ALL_STAT_KEYS = ADAPTER_STAT_KEYS + FALLBACK_STAT_KEYS + PASSTHROUGH_STAT_KEYS

NUMBER_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    data: dict[str, Any] | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    root = repo_root()
    env = os.environ.copy()
    for key in (MXFP8_BF16_ACK, DSA_GATHER_ACK, FP8_ACT_LEGACY_ACK):
        env.pop(key, None)
    pythonpath = [
        str(root / "scripts"),
        str(root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
    return env


def _run_python(
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
    timeout_s: int = 30,
) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stdout}\n{proc.stderr}".strip()


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _expect_fail_closed(
    name: str,
    proc: subprocess.CompletedProcess[str],
    expected: tuple[str, ...],
) -> CheckResult:
    output = _combined_output(proc)
    if proc.returncode == 0:
        return CheckResult(
            name=name,
            status="fail",
            detail="deprecated path did not fail closed",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    missing = [needle for needle in expected if needle not in output]
    if missing:
        return CheckResult(
            name=name,
            status="fail",
            detail=f"deprecated path failed, but missing expected text: {missing}",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    return CheckResult(
        name=name,
        status="pass",
        detail="failed closed without ACK",
        data={"returncode": proc.returncode},
    )


def check_mxfp8_bf16_bridge_gate() -> CheckResult:
    proc = _run_python(
        "import cppmega_fp8_shim",
        extra_env={
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "1",
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "0",
            "CPPMEGA_TE_MXFP8_DGRAD_BF16": "0",
            "CPPMEGA_TE_MXFP8_WGRAD_BF16": "0",
            "NVTE_BACKWARD_OVERRIDE": "none",
        },
    )
    return _expect_fail_closed(
        "old_mxfp8_bf16_bridge_requires_ack",
        proc,
        ("Deprecated MXFP8 BF16 backward bridge requested", MXFP8_BF16_ACK),
    )


def check_dsa_gather_scatter_gate() -> CheckResult:
    # Stub the unrelated MTP native patch so this import reaches the DSA gate
    # even on minimal test environments without Megatron installed.
    proc = _run_python(
        "\n".join(
            [
                "import sys, types",
                "mod = types.ModuleType('cppmega.megatron.mtp_native_hopper_ce')",
                "mod.patch_mtp_native_hopper_ce = lambda: None",
                "sys.modules['cppmega.megatron.mtp_native_hopper_ce'] = mod",
                "import cppmega_fp8_shim",
            ]
        ),
        extra_env={
            "CPPMEGA_DSA_SPARSE_MODE": "gather_scatter",
            "CPPMEGA_MTP_CE_KERNEL": "native",
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "0",
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "0",
        },
    )
    return _expect_fail_closed(
        "dsa_gather_scatter_requires_ack",
        proc,
        ("CPPMEGA_DSA_SPARSE_MODE=gather_scatter is DEPRECATED", DSA_GATHER_ACK),
    )


def check_fp8_activation_legacy_gate() -> CheckResult:
    proc = _run_python(
        "\n".join(
            [
                "from cppmega.megatron import fp8_activations as fp8",
                "fp8._use_te_packer()",
            ]
        ),
        extra_env={"CPPMEGA_FP8_ACTIVATION_BACKEND": "legacy"},
    )
    return _expect_fail_closed(
        "fp8_activation_legacy_requires_ack",
        proc,
        ("CPPMEGA_FP8_ACTIVATION_BACKEND=legacy is DEPRECATED", FP8_ACT_LEGACY_ACK),
    )


def _probe_prereqs_available(timeout_s: int) -> tuple[bool, str]:
    proc = _run_python(
        "\n".join(
            [
                "import torch",
                "import transformer_engine",  # noqa: F401
                "import transformer_engine_torch",  # noqa: F401
                "print('cuda=' + str(torch.cuda.is_available()))",
            ]
        ),
        timeout_s=timeout_s,
    )
    if proc.returncode != 0:
        return False, _tail(_combined_output(proc), 1000)
    if "cuda=True" not in proc.stdout:
        return False, "torch.cuda.is_available() is false"
    return True, "CUDA and Transformer Engine are available"


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in probe output")


def validate_probe_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    stats = report.get("shim_stats")
    if not isinstance(stats, dict):
        return ["probe report is missing shim_stats"]

    for key in FALLBACK_STAT_KEYS:
        if int(stats.get(key, -1)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0")
    for key in ADAPTER_STAT_KEYS:
        if int(stats.get(key, 0)) <= 0:
            errors.append(f"{key}={stats.get(key)}; expected >0")
    for key in PASSTHROUGH_STAT_KEYS:
        if int(stats.get(key, -1)) != 0:
            errors.append(f"{key}={stats.get(key)}; expected 0")
    if stats.get("fallback_reasons", {}) not in ({}, None):
        errors.append(f"fallback_reasons={stats.get('fallback_reasons')!r}; expected empty")

    results = {
        row.get("name"): row
        for row in report.get("results", [])
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    for name in ("mxfp8_dgrad_shim_NN_to_TN", "mxfp8_wgrad_shim_NT_to_TN"):
        row = results.get(name)
        if row is None:
            errors.append(f"missing probe result {name}")
        elif row.get("status") != "pass":
            errors.append(f"{name} status={row.get('status')!r}; expected pass")
    return errors


def run_mxfp8_shim_probe(args: argparse.Namespace) -> CheckResult:
    if args.skip_mxfp8_probe:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="skip",
            detail="skipped by --skip-mxfp8-probe",
        )

    available, reason = _probe_prereqs_available(args.probe_timeout_s)
    if not available and not args.require_mxfp8_probe:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="skip",
            detail=f"CUDA/TE probe prerequisites unavailable: {reason}",
        )

    cmd = [
        sys.executable,
        str(repo_root() / "tools" / "probes" / "te_blockscaled_backward_probe.py"),
        "--format",
        "mxfp8",
        "--use-shim",
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
    ]
    env = _base_env()
    env.update(
        {
            "CPPMEGA_TE_MXFP8_BWD_TN_ADAPTER": "1",
            "CPPMEGA_TE_MXFP8_BWD_ALLOW_BF16_FALLBACK": "0",
            "CPPMEGA_TE_MXFP8_DGRAD_BF16": "0",
            "CPPMEGA_TE_MXFP8_WGRAD_BF16": "0",
            "NVTE_BACKWARD_OVERRIDE": "none",
            "CPPMEGA_TE_MXFP8_BWD_DEBUG": "1",
        }
    )
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.probe_timeout_s,
        check=False,
    )
    output = _combined_output(proc)
    if proc.returncode != 0:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail="te_blockscaled_backward_probe.py failed",
            data={"returncode": proc.returncode, "output_tail": _tail(output)},
        )
    try:
        report = extract_first_json_object(output)
        errors = validate_probe_report(report)
    except Exception as exc:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail=f"could not parse/validate probe output: {type(exc).__name__}: {exc}",
            data={"output_tail": _tail(output)},
        )
    if errors:
        return CheckResult(
            name="mxfp8_tn_adapter_probe",
            status="fail",
            detail="probe did not satisfy accepted-path counters",
            data={"errors": errors, "shim_stats": report.get("shim_stats")},
        )
    return CheckResult(
        name="mxfp8_tn_adapter_probe",
        status="pass",
        detail="shim routed MXFP8 backward through TN adapter with BF16 fallback counters at zero",
        data={"shim_stats": report.get("shim_stats")},
    )


def parse_training_log(text: str) -> dict[str, Any]:
    losses: dict[str, list[float]] = {"lm": [], "mtp_1": []}
    for key in list(losses):
        for pattern in (
            rf"\b{re.escape(key)}\s+loss\s*[:=]\s*({NUMBER_RE})",
            rf"\bloss:\s*.*?\b{re.escape(key)}\s+({NUMBER_RE})",
        ):
            losses[key].extend(float(match) for match in re.findall(pattern, text))
    losses = {key: values for key, values in losses.items() if values}

    counters: dict[str, int] = {}
    fallback_reasons: dict[str, Any] | None = None
    marker = "TE block-scaled backward stats:"
    for line in text.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        for key in ALL_STAT_KEYS:
            if key in parsed:
                counters[key] = int(parsed[key])
        reasons = parsed.get("fallback_reasons")
        if isinstance(reasons, dict):
            fallback_reasons = reasons

    for key in ALL_STAT_KEYS:
        matches = re.findall(rf"\b{re.escape(key)}\s*=\s*(\d+)\b", text)
        if matches:
            counters[key] = int(matches[-1])
    if fallback_reasons is None and re.search(r"\bfallback_reasons\s*=\s*\{\s*\}", text):
        fallback_reasons = {}

    return {"losses": losses, "counters": counters, "fallback_reasons": fallback_reasons}


def validate_training_log(path: Path) -> CheckResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_training_log(text)
    errors: list[str] = []
    if not parsed["losses"]:
        errors.append("no lm or mtp_1 loss found")
    counters = parsed["counters"]
    if not counters:
        errors.append("no TE block-scaled backward counters found")
    else:
        for key in FALLBACK_STAT_KEYS:
            if int(counters.get(key, -1)) != 0:
                errors.append(f"{key}={counters.get(key)}; expected 0")
        for key in ADAPTER_STAT_KEYS:
            if int(counters.get(key, 0)) <= 0:
                errors.append(f"{key}={counters.get(key)}; expected >0")
    reasons = parsed.get("fallback_reasons")
    if reasons not in ({}, None):
        errors.append(f"fallback_reasons={reasons!r}; expected empty")
    if errors:
        return CheckResult(
            name="one_step_log_parse",
            status="fail",
            detail="training log does not satisfy accepted-path evidence",
            data={"path": str(path), "errors": errors, **parsed},
        )
    return CheckResult(
        name="one_step_log_parse",
        status="pass",
        detail="parsed loss and zero-BF16-fallback MXFP8 counters from training log",
        data={"path": str(path), **parsed},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mxfp8-probe", action="store_true")
    parser.add_argument(
        "--require-mxfp8-probe",
        action="store_true",
        help="Fail instead of skipping when CUDA/Transformer Engine prerequisites are unavailable.",
    )
    parser.add_argument("--probe-timeout-s", type=int, default=180)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--train-log",
        type=Path,
        help="Optional one-step training log to parse for loss and MXFP8 counters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    checks = [
        check_mxfp8_bf16_bridge_gate(),
        check_dsa_gather_scatter_gate(),
        check_fp8_activation_legacy_gate(),
        run_mxfp8_shim_probe(args),
    ]
    if args.train_log is not None:
        checks.append(validate_training_log(args.train_log))

    failed = any(check.status == "fail" for check in checks)
    if args.require_mxfp8_probe:
        failed = failed or any(
            check.name == "mxfp8_tn_adapter_probe" and check.status == "skip"
            for check in checks
        )

    print(
        json.dumps(
            {
                "status": "fail" if failed else "pass",
                "checks": [asdict(check) for check in checks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
