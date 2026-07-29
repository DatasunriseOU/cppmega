from __future__ import annotations

import pytest

from cppmega.data.ci_training_scope import (
    AUX_JS_TS_ROUTE,
    AUX_PYTHON_ROUTE,
    CITrainingScopeError,
    PRIMARY_ROUTE,
    TRAINING_SCOPE_POLICY_SCHEMA,
    classify_ci_training_sidecars,
    training_scope_policy,
)


def _training(
    *,
    entities: list[dict] | None = None,
    build_actions: list[dict] | None = None,
    tests: list[dict] | None = None,
    diagnostics: list[dict] | None = None,
    commands: list[dict] | None = None,
) -> dict:
    return {
        "schema": "cppmega_ci_chunk_training_sidecars_v2",
        "entities": entities or [],
        "build_actions": build_actions or [],
        "tests": tests or [],
        "diagnostics": diagnostics or [],
        "commands": commands or [],
    }


@pytest.mark.parametrize(
    "training,reason",
    [
        (
            _training(
                entities=[
                    {
                        "domain": "CPP",
                        "attributes": {"likely_language": "C++"},
                    }
                ]
            ),
            "primary_domain:CPP",
        ),
        (
            _training(
                build_actions=[
                    {"tool": "cmake", "kind": "configure"}
                ]
            ),
            "primary_build_action:cmake:configure",
        ),
        (
            _training(tests=[{"framework": "ctest"}]),
            "primary_test:ctest",
        ),
        (
            _training(
                diagnostics=[
                    {
                        "category": "compiler",
                        "tool": "clang",
                        "file": "src/main.cpp",
                    }
                ]
            ),
            "primary_diagnostic:compiler:clang",
        ),
        (
            _training(
                entities=[
                    {
                        "domain": "SQL",
                        "attributes": {"likely_language": "SQL"},
                    }
                ]
            ),
            "primary_domain:SQL",
        ),
    ],
)
def test_primary_evidence_is_routed_without_free_text(
    training: dict,
    reason: str,
) -> None:
    decision = classify_ci_training_sidecars(training)

    assert decision.primary is True
    assert PRIMARY_ROUTE in decision.as_dict()["local_routes"]
    assert reason in decision.reasons


def test_shell_only_and_generic_workflow_errors_are_not_primary() -> None:
    decision = classify_ci_training_sidecars(
        _training(
            entities=[{"domain": "BASH", "attributes": {}}],
            commands=[{"shell_dialect": "bash"}],
            diagnostics=[{"category": "workflow", "tool": "github_actions"}],
        )
    )

    assert decision.primary is False
    assert decision.as_dict()["local_routes"] == []


def test_python_and_js_are_auxiliary_and_do_not_raise_primary_count() -> None:
    decision = classify_ci_training_sidecars(
        _training(
            entities=[
                {
                    "domain": "PYTHON",
                    "attributes": {"likely_language": "Python"},
                },
                {
                    "domain": "TYPESCRIPT",
                    "attributes": {"likely_language": "TypeScript"},
                },
            ],
            tests=[
                {"framework": "pytest"},
                {"framework": "vitest"},
            ],
        )
    )

    assert decision.primary is False
    assert decision.aux_python is True
    assert decision.aux_js_ts is True
    assert decision.as_dict()["local_routes"] == [
        AUX_PYTHON_ROUTE,
        AUX_JS_TS_ROUTE,
    ]


@pytest.mark.parametrize("path", ["src/check.py", "src/check.ts"])
def test_non_native_compiler_shaped_diagnostic_is_not_primary(path: str) -> None:
    decision = classify_ci_training_sidecars(
        _training(
            diagnostics=[
                {
                    "category": "compiler",
                    "tool": "gcc_or_clang",
                    "file": path,
                }
            ]
        )
    )

    assert decision.primary is False


def test_primary_route_has_priority_over_auxiliary_evidence() -> None:
    decision = classify_ci_training_sidecars(
        _training(
            entities=[
                {
                    "domain": "CPP",
                    "attributes": {"likely_language": "C++"},
                },
                {
                    "domain": "PYTHON",
                    "attributes": {"likely_language": "Python"},
                },
            ]
        )
    )

    assert decision.primary is True
    assert decision.aux_python is True
    assert decision.as_dict()["local_routes"] == [PRIMARY_ROUTE]


def test_policy_identity_is_deterministic_and_explicit() -> None:
    first = training_scope_policy()
    second = training_scope_policy()

    assert first == second
    assert first["schema"] == TRAINING_SCOPE_POLICY_SCHEMA
    assert len(first["sha256"]) == 64
    assert first["semantics"]["shell_only"].startswith("not eligible")


def test_missing_or_legacy_sidecars_fail_closed() -> None:
    with pytest.raises(CITrainingScopeError, match="unsupported schema"):
        classify_ci_training_sidecars(
            {"schema": "cppmega_ci_chunk_training_sidecars_v1"}
        )
    with pytest.raises(CITrainingScopeError, match="commands must be a list"):
        classify_ci_training_sidecars(
            {
                "schema": "cppmega_ci_chunk_training_sidecars_v2",
                "entities": [],
                "build_actions": [],
                "tests": [],
                "diagnostics": [],
            }
        )
