from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_ROOT = _REPO_ROOT / "scripts"
_EXCLUDED_SCRIPT = _SCRIPT_ROOT / "nebius_h200_megatron_cpp_generation_eval.py"
_FORBIDDEN_PATTERNS = {
    "legacy static_context import": re.compile(
        r"(?:"
        r"from\s+megatron\.core\.inference\.contexts(?:\.static_context)?\s+"
        r"import\s+(?:static_context|deprecate_inference_params)"
        r"|import\s+megatron\.core\.inference\.contexts\.static_context\b"
        r")"
    ),
    "static_context alias assignment": re.compile(
        r"\.\s*deprecate_inference_params\s*="
    ),
    "dynamic alias assignment": re.compile(
        r"setattr\([^\n]*[\"']deprecate_inference_params[\"']\s*,"
    ),
    "static_context alias guard": re.compile(
        r"hasattr\([^\n]*deprecate_inference_params"
    ),
    "dynamic API fallback": re.compile(
        r"getattr\([^\n]*[\"']deprecate_inference_params[\"']\s*,"
    ),
    "canonical API fallback": re.compile(
        r"try:\s*\n\s*from\s+megatron\.core\.utils\s+import\s+"
        r"deprecate_inference_params(?:\s+as\s+\w+)?\s*\n"
        r"\s*except\s+(?:ImportError|Exception)\b",
        re.MULTILINE,
    ),
}


def _production_script_sources() -> list[Path]:
    return sorted(
        path
        for path in _SCRIPT_ROOT.rglob("*")
        if path.suffix in {".py", ".sh"}
        and path != _EXCLUDED_SCRIPT
    )


def test_production_scripts_do_not_inject_deprecated_inference_alias() -> None:
    violations: list[str] = []

    for path in _production_script_sources():
        source = path.read_text(encoding="utf-8")
        for label, pattern in _FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{line}: {label}")

    assert not violations, "Deprecated inference alias injection found:\n" + "\n".join(
        violations
    )
