"""Deterministic routing policy for CI chunks used as training data.

The CI parser intentionally preserves a broad, lossless view of GitHub job
output.  Acquisition breadth is not training eligibility: the primary corpus
contains only C/C++/CUDA, SQL, native build systems, and the tests or
diagnostics tied to those domains.  Python and JavaScript/TypeScript evidence
is retained as auxiliary routing metadata and never makes a chunk primary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any


TRAINING_SIDECAR_SCHEMA = "cppmega_ci_chunk_training_sidecars_v2"
TRAINING_SCOPE_POLICY_SCHEMA = "cppmega_ci_training_scope_policy_v1"
TRAINING_SCOPE_DECISION_SCHEMA = "cppmega_ci_training_scope_decision_v1"

PRIMARY_ROUTE = "primary_cpp_sql_build_test"
AUX_PYTHON_ROUTE = "aux_python"
AUX_JS_TS_ROUTE = "aux_js_ts"

_PRIMARY_LANGUAGES = frozenset(
    {
        "build",
        "c",
        "c/c++",
        "c++",
        "cuda",
        "module",
        "sql",
    }
)
_PYTHON_LANGUAGES = frozenset({"py", "python"})
_JS_TS_LANGUAGES = frozenset(
    {
        "javascript",
        "javascript/typescript",
        "js",
        "jsx",
        "ts",
        "tsx",
        "typescript",
    }
)

_PRIMARY_DOMAINS = frozenset(
    {
        "AUTOCONF",
        "BAZEL",
        "BUILD_DIAGNOSTIC",
        "CMAKE",
        "COMPILER_DIAGNOSTIC",
        "COMPILER_ERROR",
        "CONFIGURE",
        "CPP",
        "GN",
        "LINKER_ERROR",
        "MAKE",
        "MESON",
        "NINJA",
        "SANITIZER_OUTPUT",
        "SCONS",
        "SQL",
        "XMAKE",
    }
)
_PYTHON_DOMAINS = frozenset({"PYTHON"})
_JS_TS_DOMAINS = frozenset({"JAVASCRIPT", "JS", "TYPESCRIPT"})

_PRIMARY_BUILD_TOOLS = frozenset(
    {
        "ar",
        "autotools",
        "bazel",
        "bazelisk",
        "clang",
        "clang++",
        "clang-cl",
        "cmake",
        "g++",
        "gcc",
        "gn",
        "intel",
        "ld",
        "ld.lld",
        "lib",
        "link",
        "lld-link",
        "llvm-ar",
        "make",
        "meson",
        "msbuild",
        "msvc",
        "ninja",
        "nvcc",
        "scons",
        "xmake",
    }
)
_PRIMARY_BUILD_ACTION_KINDS = frozenset(
    {"archive", "build", "compile", "configure", "link"}
)
_PRIMARY_TEST_FRAMEWORKS = frozenset({"ctest", "gtest"})
_PYTHON_TEST_FRAMEWORKS = frozenset({"pytest"})
_JS_TS_TEST_FRAMEWORKS = frozenset(
    {"ava", "jest", "mocha", "node:test", "playwright", "vitest"}
)
_PRIMARY_DIAGNOSTIC_CATEGORIES = frozenset(
    {"compiler", "linker", "sanitizer"}
)
_PRIMARY_COMPILER_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cppm",
        ".cu",
        ".cuh",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".inc",
        ".inl",
        ".ipp",
        ".ixx",
        ".mpp",
        ".tcc",
    }
)
_PRIMARY_BUILD_DIAGNOSTIC_TOOLS = frozenset(
    {"build_system", "cmake", "msbuild", "ninja", "make"}
)


class CITrainingScopeError(ValueError):
    """A parser sidecar cannot be classified without guessing."""


def _normalized(value: object) -> str:
    return str(value).strip().casefold()


def _path_suffix(value: object) -> str:
    normalized = str(value or "").strip().replace("\\", "/").casefold()
    basename = normalized.rsplit("/", 1)[-1]
    dot = basename.rfind(".")
    return basename[dot:] if dot >= 0 else ""


def _mapping(value: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CITrainingScopeError(f"{where} must be an object")
    return value


def _records(value: object, *, where: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise CITrainingScopeError(f"{where} must be a list")
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        result.append(_mapping(raw, where=f"{where}[{index}]"))
    return result


def _policy_payload() -> dict[str, object]:
    return {
        "schema": TRAINING_SCOPE_POLICY_SCHEMA,
        "semantics": {
            "primary": (
                "C/C++/CUDA/SQL and native build/test/diagnostic evidence"
            ),
            "auxiliary": (
                "Python and JavaScript/TypeScript evidence is retained outside "
                "the primary token target"
            ),
            "shell_only": "not eligible without a routed domain signal",
            "locality": (
                "this decision is chunk-local; the exporter may propagate a "
                "route only within the exact receipt-bound CI step"
            ),
        },
        "primary_languages": sorted(_PRIMARY_LANGUAGES),
        "python_languages": sorted(_PYTHON_LANGUAGES),
        "js_ts_languages": sorted(_JS_TS_LANGUAGES),
        "primary_domains": sorted(_PRIMARY_DOMAINS),
        "python_domains": sorted(_PYTHON_DOMAINS),
        "js_ts_domains": sorted(_JS_TS_DOMAINS),
        "primary_build_tools": sorted(_PRIMARY_BUILD_TOOLS),
        "primary_build_action_kinds": sorted(_PRIMARY_BUILD_ACTION_KINDS),
        "primary_test_frameworks": sorted(_PRIMARY_TEST_FRAMEWORKS),
        "python_test_frameworks": sorted(_PYTHON_TEST_FRAMEWORKS),
        "js_ts_test_frameworks": sorted(_JS_TS_TEST_FRAMEWORKS),
        "primary_diagnostic_categories": sorted(
            _PRIMARY_DIAGNOSTIC_CATEGORIES
        ),
        "primary_compiler_source_suffixes": sorted(
            _PRIMARY_COMPILER_SOURCE_SUFFIXES
        ),
        "primary_build_diagnostic_tools": sorted(
            _PRIMARY_BUILD_DIAGNOSTIC_TOOLS
        ),
    }


def training_scope_policy() -> dict[str, object]:
    """Return the canonical JSON-serializable routing policy and its identity."""

    payload = _policy_payload()
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


@dataclass(frozen=True)
class CITrainingScopeDecision:
    """Chunk-local evidence before exact-step propagation."""

    primary: bool
    aux_python: bool
    aux_js_ts: bool
    reasons: tuple[str, ...]
    evidence_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        routes: list[str] = []
        if self.primary:
            routes.append(PRIMARY_ROUTE)
        else:
            if self.aux_python:
                routes.append(AUX_PYTHON_ROUTE)
            if self.aux_js_ts:
                routes.append(AUX_JS_TS_ROUTE)
        return {
            "schema": TRAINING_SCOPE_DECISION_SCHEMA,
            "local_primary": self.primary,
            "local_aux_python": self.aux_python,
            "local_aux_js_ts": self.aux_js_ts,
            "local_routes": routes,
            "reasons": list(self.reasons),
            "evidence_counts": dict(self.evidence_counts),
        }


def classify_ci_training_sidecars(
    training_sidecars: Mapping[str, Any],
) -> CITrainingScopeDecision:
    """Classify one exact parser chunk without inspecting its free-form text."""

    training = _mapping(training_sidecars, where="training_sidecars")
    if training.get("schema") != TRAINING_SIDECAR_SCHEMA:
        raise CITrainingScopeError(
            "training_sidecars has an unsupported schema: "
            f"{training.get('schema')!r}"
        )

    entities = _records(training.get("entities"), where="entities")
    build_actions = _records(
        training.get("build_actions"), where="build_actions"
    )
    tests = _records(training.get("tests"), where="tests")
    diagnostics = _records(
        training.get("diagnostics"), where="diagnostics"
    )
    # Commands are required even though an unqualified shell command is never a
    # training signal.  Requiring the field prevents a truncated sidecar from
    # being mistaken for an irrelevant but valid chunk.
    _records(training.get("commands"), where="commands")

    evidence: Counter[str] = Counter()
    primary = False
    aux_python = False
    aux_js_ts = False

    for entity in entities:
        domain = str(entity.get("domain") or "").strip().upper()
        if domain in _PRIMARY_DOMAINS:
            primary = True
            evidence[f"primary_domain:{domain}"] += 1
        elif domain in _PYTHON_DOMAINS:
            aux_python = True
            evidence[f"aux_python_domain:{domain}"] += 1
        elif domain in _JS_TS_DOMAINS:
            aux_js_ts = True
            evidence[f"aux_js_ts_domain:{domain}"] += 1

        attributes = entity.get("attributes")
        if attributes is None:
            continue
        attributes = _mapping(attributes, where="entity.attributes")
        for field in ("language", "likely_language"):
            language = _normalized(attributes.get(field) or "")
            if not language:
                continue
            if language in _PRIMARY_LANGUAGES:
                primary = True
                evidence[f"primary_language:{language}"] += 1
            elif language in _PYTHON_LANGUAGES:
                aux_python = True
                evidence[f"aux_python_language:{language}"] += 1
            elif language in _JS_TS_LANGUAGES:
                aux_js_ts = True
                evidence[f"aux_js_ts_language:{language}"] += 1

    for action in build_actions:
        tool = _normalized(action.get("tool") or "").removesuffix(".exe")
        kind = _normalized(action.get("kind") or action.get("action_kind") or "")
        if tool in _PRIMARY_BUILD_TOOLS and kind in _PRIMARY_BUILD_ACTION_KINDS:
            primary = True
            evidence[f"primary_build_action:{tool}:{kind}"] += 1

    for test in tests:
        framework = _normalized(test.get("framework") or "")
        if framework in _PRIMARY_TEST_FRAMEWORKS:
            primary = True
            evidence[f"primary_test:{framework}"] += 1
        elif framework in _PYTHON_TEST_FRAMEWORKS:
            aux_python = True
            evidence[f"aux_python_test:{framework}"] += 1
        elif framework in _JS_TS_TEST_FRAMEWORKS:
            aux_js_ts = True
            evidence[f"aux_js_ts_test:{framework}"] += 1

    for diagnostic in diagnostics:
        category = _normalized(diagnostic.get("category") or "")
        tool = _normalized(diagnostic.get("tool") or "").removesuffix(".exe")
        if (
            category == "compiler"
            and _path_suffix(diagnostic.get("file"))
            in _PRIMARY_COMPILER_SOURCE_SUFFIXES
        ):
            primary = True
            evidence[f"primary_diagnostic:{category}:{tool or 'unknown'}"] += 1
        elif category in {"linker", "sanitizer"}:
            primary = True
            evidence[f"primary_diagnostic:{category}:{tool or 'unknown'}"] += 1
        elif (
            category == "build"
            and tool in _PRIMARY_BUILD_DIAGNOSTIC_TOOLS
        ):
            primary = True
            evidence[f"primary_build_diagnostic:{tool}"] += 1

    # Primary has physical routing priority.  Auxiliary evidence remains in the
    # decision for audit, but auxiliary exporters must not duplicate a primary
    # chunk.
    reasons = tuple(sorted(evidence))
    return CITrainingScopeDecision(
        primary=primary,
        aux_python=aux_python,
        aux_js_ts=aux_js_ts,
        reasons=reasons,
        evidence_counts=tuple(sorted(evidence.items())),
    )
