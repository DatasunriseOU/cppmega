from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ci_log_sidecars import (  # noqa: E402
    CANONICALIZATION_SCHEMA,
    DEDUPLICATION_SCHEMA,
    EDGE_IDS,
    SIDECAR_SCHEMA,
    TRAINING_SIDECAR_SCHEMA,
    _is_package_version,
    canonicalize_ci_log,
    extract_ci_log_sidecar,
    stable_json_dumps,
)


def _timestamped(*payloads: str, crlf: bool = False) -> bytes:
    terminator = "\r\n" if crlf else "\n"
    return "".join(
        f"2026-07-26T04:35:{index:02d}.{index:07d}Z {payload}{terminator}"
        for index, payload in enumerate(payloads)
    ).encode()


def _assert_conserved(result: dict) -> None:
    canonical = result["canonical_text"]
    dedup = result["dedup_text"]
    sections = result["sections"]
    chunks = result["chunks"]
    sidecar = result["sidecar"]

    assert "".join(section["text"] for section in sections) == canonical
    assert "".join(section["dedup_text"] for section in sections) == dedup
    assert "".join(chunk["canonical_text"] for chunk in chunks) == canonical
    assert "".join(chunk["text"] for chunk in chunks) == dedup
    assert all(
        chunk["char_start"] == (0 if index == 0 else chunks[index - 1]["char_end"])
        for index, chunk in enumerate(chunks)
    )
    assert not chunks or chunks[-1]["char_end"] == len(canonical)
    assert all(
        canonical[chunk["char_start"] : chunk["char_end"]]
        == chunk["canonical_text"]
        for chunk in chunks
    )
    assert all(
        hashlib.sha256(chunk["text"].encode()).hexdigest() == chunk["sha256"]
        for chunk in chunks
    )
    assert all(sidecar["conservation"][key] for key in (
        "section_canonical_chars_cover_exactly_once",
        "section_dedup_chars_cover_exactly_once",
        "chunk_canonical_chars_cover_exactly_once",
        "chunk_dedup_chars_cover_exactly_once",
        "chunk_spans_contiguous",
        "chunk_semantic_rle_covers_every_char_once",
    ))
    for chunk in chunks:
        chunk_length = chunk["char_end"] - chunk["char_start"]
        for field in ("role_spans", "domain_spans"):
            spans = chunk[field]
            assert spans[0]["start_char"] == 0
            assert spans[-1]["end_char"] == chunk_length
            assert all(
                left["end_char"] == right["start_char"]
                for left, right in zip(spans, spans[1:])
            )


def test_cmake_ninja_gcc_cuda_success_sections_entities_and_edges() -> None:
    raw = _timestamped(
        "Current runner version: '2.335.1'",
        "##[group]Operating System",
        "Ubuntu",
        "24.04.4",
        "LTS",
        "##[endgroup]",
        "##[group]Runner Image",
        "Image: ubuntu-24.04",
        "Version: 20260720.247.2",
        "##[endgroup]",
        "##[group]Run cmake -S . -B build -DCMAKE_CUDA_COMPILER=nvcc",
        "  shell: /usr/bin/bash --noprofile --norc -e {0}",
        "[command]cmake -S . -B build -DCMAKE_CXX_COMPILER=g++",
        "##[group]Run cmake --build build --target cuda_app",
        "  shell: /usr/bin/bash -e {0}",
        "[command]ninja -C build cuda_app src/kernel.cu -o build/cuda_app",
        "[100%] Built target cuda_app",
    )
    metadata = {
        "repository": {"full_name": "owner/project"},
        "repository_requested": "old-owner/project",
        "repository_id": 101,
        "source_repository": "contributor/project",
        "source_repository_id": 202,
        "workflow": {"name": "CI", "path": ".github/workflows/ci.yml"},
        "run": {"id": 41, "attempt": 2},
        "run_number": 19,
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-07-26T04:34:00Z",
        "updated_at": "2026-07-26T04:36:00Z",
        "run_started_at": "2026-07-26T04:34:05Z",
        "display_title": "CUDA build",
        "event_name": "pull_request",
        "head_sha": "a" * 40,
        "head_commit": {
            "id": "a" * 40,
            "message": "exercise CUDA build",
            "author": {"name": "Contributor", "email": "author@example.test"},
            "committer": {
                "name": "Contributor",
                "email": "author@example.test",
            },
        },
        "head_branch": "feature/cuda",
        "actor": {"login": "octocat"},
        "triggering_actor": {"login": "maintainer"},
        "job": {
            "id": 9,
            "name": "build (ubuntu-24.04, gcc)",
            "conclusion": "success",
            "started_at": "2026-07-26T04:34:05Z",
            "completed_at": "2026-07-26T04:36:00Z",
            "runner_name": "GitHub Actions 1001",
            "runner_group_id": 2,
            "runner_group_name": "GitHub Actions",
            "labels": ["ubuntu-24.04", "x64"],
            "matrix": {"compiler": "gcc", "os": "ubuntu-24.04"},
            "steps": [
                {"name": "cmake -S . -B build -DCMAKE_CUDA_COMPILER=nvcc"},
                {"name": "cmake --build build --target cuda_app"},
            ],
        },
        "runner": {
            "labels": ["ubuntu-latest", "x64"],
            "os": "Linux",
            "arch": "X64",
        },
    }

    result = canonicalize_ci_log(raw, metadata, max_chunk_chars=180)
    sidecar = result["sidecar"]
    classes = sidecar["classifications"]

    assert sidecar["schema"] == SIDECAR_SCHEMA
    assert sidecar["canonicalization_schema"] == CANONICALIZATION_SCHEMA
    assert sidecar["deduplication_schema"] == DEDUPLICATION_SCHEMA
    assert sidecar["raw"]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert sidecar["raw"]["raw_byte_count"] == len(raw)
    assert sidecar["raw"]["status"] == "valid"
    assert sidecar["canonicalization"]["timestamp_prefixes"]["count"] == 17
    assert len(
        sidecar["canonicalization"]["timestamp_prefixes"]["sequence_samples"]
    ) == 8
    assert sidecar["canonicalization"]["timestamp_prefixes"][
        "sequence_omitted_count"
    ] == 9
    assert sidecar["canonicalization"]["timestamp_prefixes"]["delta_ns"][
        "count"
    ] == 16
    assert sidecar["canonicalization"]["accounting"][
        "character_count_conserved"
    ]
    assert sidecar["provenance"]["repository"] == "owner/project"
    assert sidecar["provenance"]["repository_requested"] == (
        "old-owner/project"
    )
    assert sidecar["provenance"]["repository_alias_changed"] is True
    assert sidecar["provenance"]["source_repository"] == (
        "contributor/project"
    )
    assert sidecar["provenance"]["run"] == {
        "id": 41,
        "attempt": 2,
        "number": 19,
        "status": "completed",
        "conclusion": "success",
        "created_at": "2026-07-26T04:34:00Z",
        "updated_at": "2026-07-26T04:36:00Z",
        "started_at": "2026-07-26T04:34:05Z",
        "display_title": "CUDA build",
        "event": "pull_request",
        "head_sha": "a" * 40,
        "head_commit": metadata["head_commit"],
        "branch": "feature/cuda",
    }
    assert sidecar["provenance"]["job"]["started_at"] == (
        "2026-07-26T04:34:05Z"
    )
    assert sidecar["provenance"]["job"]["completed_at"] == (
        "2026-07-26T04:36:00Z"
    )
    assert sidecar["provenance"]["runner"]["name"] == (
        "GitHub Actions 1001"
    )
    assert sidecar["provenance"]["runner"]["group_id"] == 2
    assert sidecar["provenance"]["runner"]["labels"] == [
        "ubuntu-24.04",
        "x64",
    ]

    assert [section["kind"] for section in result["sections"]] == [
        "job_preamble",
        "step",
        "step",
    ]
    assert result["sections"][1]["metadata_correlation_confidence"]["score"] == 1.0
    assert result["sections"][2]["metadata_step_index"] == 1
    assert all(
        chunk["section_ordinal"] in {0, 1, 2}
        and chunk["job_ordinal"] == 0
        for chunk in result["chunks"]
    )

    assert {item["name"] for item in classes["build_systems"]} == {
        "cmake",
        "ninja",
    }
    assert {"g++", "nvcc"} <= {item["name"] for item in classes["toolchains"]}
    assert {item["name"] for item in classes["build_targets"]} == {"cuda_app"}
    assert classes["build_actions"]
    ninja_action = next(
        action for action in classes["build_actions"] if action["tool"] == "ninja"
    )
    assert ninja_action["target"] == "cuda_app"
    assert ninja_action["source_inputs"] == ["src/kernel.cu"]
    assert ninja_action["outputs"] == ["build/cuda_app"]
    assert ninja_action["repository_source_bindings"] == [
        {
            "repository": "owner/project",
            "head_sha": "a" * 40,
            "source_path": "src/kernel.cu",
            "confidence": {
                "score": 0.95,
                "level": "high",
                "source": "relative_source_path_v1",
            },
        }
    ]
    training = next(
        chunk["training_sidecars"]
        for chunk in result["chunks"]
        if any(
            action["tool"] == "ninja"
            for action in chunk["training_sidecars"]["build_actions"]
        )
    )
    assert training["schema"] == TRAINING_SIDECAR_SCHEMA
    training_ninja_action = next(
        action
        for action in training["build_actions"]
        if action["tool"] == "ninja"
    )
    assert training_ninja_action["action_entity_id"].startswith("entity:")
    assert not any(
        key.endswith("entityid")
        for record_group in (
            training["commands"],
            training["build_actions"],
            training["tests"],
            training["diagnostics"],
        )
        for record in record_group
        for key in record
    )
    assert {
        edge["kind_id"] for edge in training["edges"]
    } >= {
        EDGE_IDS["BUILD_ACTION_INPUT"],
        EDGE_IDS["BUILD_ACTION_OUTPUT"],
        EDGE_IDS["BUILD_COMMAND_TARGET"],
    }
    assert all(
        0 <= record["start_char"] < record["end_char"]
        <= training["chunk_char_count"]
        for records in (
            training["entities"],
            training["commands"],
            training["build_actions"],
        )
        for record in records
    )
    assert all(
        "training_sidecars" not in chunk
        for chunk in sidecar["chunk_index"]
    )
    outbound_cross_chunk_edges = [
        edge
        for chunk in result["chunks"]
        for edge in chunk["training_sidecars"]["cross_chunk_edges"]
    ]
    training_receipt = sidecar["evidence_accounting"]["training_sidecars"]
    assert training_receipt["cross_chunk_edge_count"] == len(
        outbound_cross_chunk_edges
    )
    assert all(
        0
        <= edge["from_char"]
        < chunk["training_sidecars"]["chunk_char_count"]
        and 0 <= edge["to_member_char"] < len(result["canonical_text"])
        for chunk in result["chunks"]
        for edge in chunk["training_sidecars"]["cross_chunk_edges"]
    )
    assert "CUDA" in {item["name"] for item in classes["languages"]}
    assert classes["platform"]["os"]["value"] == "Linux"
    assert classes["platform"]["os_version"]["value"] == "24.04.4"
    assert classes["platform"]["runner_image"]["value"] == "ubuntu-24.04"
    assert classes["platform"]["matrix"]["value"] == {
        "compiler": "gcc",
        "os": "ubuntu-24.04",
    }

    target_edges = [
        edge
        for edge in sidecar["edges"]
        if edge["kind"] == "BUILD_COMMAND_TARGET"
    ]
    assert target_edges
    assert all(edge["kind_id"] == EDGE_IDS["BUILD_COMMAND_TARGET"] for edge in target_edges)
    for entity in sidecar["entities"]:
        assert result["canonical_text"][
            entity["start_char"] : entity["end_char"]
        ] == entity["text"]
    _assert_conserved(result)


def test_package_inventory_rows_are_not_build_actions() -> None:
    raw = _timestamped(
        "cmake                     4.3.2  h8cb302d_0  conda-forge  18MB",
        "make                      4.4.1  hc9fafa5_2  conda-forge  Cached",
        "ninja                    1.13.2  h49c215f_0  conda-forge  Cached",
        "bazel                     8.3.0",
        "cmake/jammy-updates,now 3.22.1-1ubuntu1.22.04.2 amd64 [installed]",
        "[command]cmake -S . -B build",
        "[command]cmake --build build --target app",
        "[command]ninja -C build app",
    )

    result = canonicalize_ci_log(raw)
    actions = result["sidecar"]["classifications"]["build_actions"]

    assert [action["command"] for action in actions] == [
        "cmake -S . -B build",
        "cmake --build build --target app",
        "ninja -C build app",
    ]
    assert all(
        action["confidence"]["source"] == "build_system_command_v1"
        for action in actions
    )
    _assert_conserved(result)


def test_package_version_parser_is_linear_and_rejects_ambiguous_suffixes() -> None:
    assert _is_package_version("v4.3.2")
    assert _is_package_version("3.22.1-1ubuntu1.22.04.2")
    assert _is_package_version("1.2rc1")
    assert not _is_package_version("1")
    assert not _is_package_version("1.2-")
    assert not _is_package_version("1.2--post")
    assert not _is_package_version("0.0" + ("0" * 100_000) + "!")


def test_build_action_graph_links_non_output_path_inputs() -> None:
    result = canonicalize_ci_log(
        _timestamped(
            "[command]msbuild.exe project/app.sln /t:Build "
            "/p:Configuration=Release",
        )
    )
    training = result["chunks"][0]["training_sidecars"]
    action = training["build_actions"][0]
    input_entity = next(
        entity
        for entity in training["entities"]
        if entity["kind"] == "path"
        and result["chunks"][0]["text"][
            entity["start_char"] : entity["end_char"]
        ]
        == "project/app.sln"
    )

    assert action["action_entity_id"].startswith("entity:")
    assert any(
        edge["kind"] == "BUILD_ACTION_INPUT"
        and edge["source"] == action["action_entity_id"]
        and edge["target"] == input_entity["entity_id"]
        for edge in training["edges"]
    )


def test_bazel_pytest_gtest_ctest_failure_diagnostics_and_sanitizer() -> None:
    raw = _timestamped(
        "##[group]Run bazel test //lib:all_tests",
        "  shell: /usr/bin/bash -e {0}",
        "[command]bazel test //lib:all_tests",
        "##[group]Run pytest -q",
        "[command]pytest -q tests/test_math.py",
        "tests/test_math.py::test_add PASSED [ 50%]",
        "tests/test_math.py::test_divide FAILED [100%]",
        "1 passed, 1 failed in 0.25s",
        "[ RUN      ] Math.Add",
        "[       OK ] Math.Add (2 ms)",
        "[  FAILED  ] Math.Divide (3 ms)",
        "[==========] 2 tests from 1 test suite ran. (5 ms total)",
        "1/2 Test #1: configure .................... Passed 0.10 sec",
        "2/2 Test #2: compile ...................... ***Failed 0.20 sec",
        "50% tests passed, 1 tests failed out of 2",
        "src/math.cpp:17:9: error: division by zero",
        "/usr/bin/ld: obj/math.o: undefined reference to `divide(int, int)'",
        "==312==ERROR: AddressSanitizer: heap-use-after-free",
        "ninja: build stopped: subcommand failed.",
    )

    result = canonicalize_ci_log(
        raw,
        {
            "repository": "owner/project",
            "workflow_name": "test",
            "run_id": 3,
            "job_id": 8,
            "job_name": "tests",
            "conclusion": "failure",
            "steps": [{"name": "bazel test //lib:all_tests"}, {"name": "pytest -q"}],
        },
    )
    classes = result["sidecar"]["classifications"]

    assert "bazel" in {item["name"] for item in classes["build_systems"]}
    assert "//lib:all_tests" in {
        item["name"] for item in classes["build_targets"]
    }
    assert {"pytest", "gtest", "ctest"} <= {
        item["framework"] for item in classes["tests"]
    }
    assert any(
        item["framework"] == "pytest"
        and item["case"] == "tests/test_math.py::test_divide"
        and item["result"] == "failed"
        for item in classes["tests"]
    )
    assert any(
        item["framework"] == "ctest"
        and item["case"] == "compile"
        and item["duration_ms"] == 200.0
        for item in classes["tests"]
    )
    assert {item["framework"] for item in classes["test_summaries"]} == {
        "pytest",
        "gtest",
        "ctest",
    }
    assert {"compiler", "linker", "sanitizer", "build"} <= {
        item["category"] for item in classes["diagnostics"]
    }
    assert any(
        edge["kind"] == "DIAG_PRIMARY_LOCATION"
        for edge in result["sidecar"]["edges"]
    )
    assert any(
        edge["kind"] == "LINK_UNDEFINED_SYMBOL"
        for edge in result["sidecar"]["edges"]
    )
    _assert_conserved(result)


def test_windows_powershell_msvc_msbuild_matrix_metadata() -> None:
    raw = _timestamped(
        "##[group]Run msbuild.exe app.sln /m /t:Build",
        r"  shell: C:\Program Files\PowerShell\7\pwsh.EXE -command \". '{0}'\"",
        r"[command]C:\Program Files\Microsoft Visual Studio\MSBuild\Current\Bin\MSBuild.exe app.sln /t:Build",
        r"[command]cl.exe /c src\main.cpp /Fo:build\main.obj",
        r"D:\a\repo\src\main.cpp(23,11): warning C4100: 'argc': unreferenced formal parameter",
        r"D:\a\repo\src\main.cpp(24): error C2065: 'missing': undeclared identifier",
        "LINK : fatal error LNK1120: 1 unresolved externals",
    )
    result = canonicalize_ci_log(
        raw,
        {
            "repository": "owner/windows-project",
            "workflow_name": "Windows CI",
            "run_id": 77,
            "run_attempt": 3,
            "event": "workflow_dispatch",
            "head_sha": "b" * 40,
            "branch": "main",
            "job": {
                "id": 88,
                "name": "build (windows-2022, x64, Release)",
                "matrix": {
                    "os": "windows-2022",
                    "arch": "x64",
                    "configuration": "Release",
                },
                "steps": [{"name": "msbuild.exe app.sln /m /t:Build"}],
            },
            "runner": {
                "os": "Windows",
                "arch": "X64",
                "image": "windows-2022",
                "labels": ["windows-2022"],
            },
        },
    )
    classes = result["sidecar"]["classifications"]

    assert {item["name"] for item in classes["shell_dialects"]} == {"powershell"}
    assert "msbuild" in {item["name"] for item in classes["build_systems"]}
    assert "msvc" in {item["name"] for item in classes["toolchains"]}
    assert "Build" in {item["name"] for item in classes["build_targets"]}
    assert classes["platform"]["os"]["value"] == "Windows"
    assert classes["platform"]["runner_image"]["value"] == "windows-2022"
    assert classes["platform"]["architecture"]["value"] == "X64"
    assert classes["platform"]["matrix"]["confidence"]["score"] == 1.0
    msvc_diagnostics = [
        item for item in classes["diagnostics"] if item["tool"] == "msvc"
    ]
    assert {item["code"] for item in msvc_diagnostics} == {"C4100", "C2065"}
    assert all(item["file"].startswith("D:\\a\\repo") for item in msvc_diagnostics)
    assert any(
        path["category"] == "output" and path["value"].endswith("main.obj")
        for path in classes["paths"]
    )
    assert any(
        action["tool"] == "msvc"
        and action["kind"] == "compile"
        and any(value.endswith("main.cpp") for value in action["source_inputs"])
        and any(value.endswith("main.obj") for value in action["outputs"])
        for action in classes["build_actions"]
    )
    _assert_conserved(result)


def test_invalid_utf8_ansi_timestamps_bom_secrets_and_dedup_are_accounted() -> None:
    github_token = b"ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    aws_key = b"AKIAABCDEFGHIJKLMNOP"
    bearer = b"eyJhbGciOiJIUzI1NiJ9.payload.signature"
    raw = (
        b"\xef\xbb\xbf"
        b"2026-07-26T04:35:42.8060000Z Requested labels: ubuntu-latest\n"
        b"2026-07-26T04:35:42.8060000Z Job defined at: owner/repo/.github/workflows/ci.yml@refs/heads/main\n"
        b"2026-07-26T04:35:49.3320000Z Job is about to start running on the hosted runner: GitHub Actions 1001484384"
        b"\xef\xbb\xbf"
        b"2026-07-26T04:35:49.3280000Z Worker ID: {c3ce764e-18fd-4506-b00f-e389a8dfd076}\r\n"
        b"2026-07-26T04:35:50.0000000Z \x1b[31mERROR\x1b[0m token="
        + github_token
        + b"\n"
        b"2026-07-26T04:35:51.0000000Z AWS="
        + aws_key
        + b" Authorization: Bearer "
        + bearer
        + b"\n"
        b"2026-07-26T04:35:52.0000000Z bad utf8: \xff\xfe\n"
        b"2026-07-26T04:35:53.0000000Z AUTHORIZATION: basic ***\n"
        b"2026-07-26T04:35:54.0000000Z temp=/home/runner/work/_temp/3745df50-471d-45d6-ba05-fe1fb0f60dc6\n"
        b"2026-07-26T04:35:55.0000000Z process id: 98765\n"
    )

    result = canonicalize_ci_log(
        raw,
        {
            "repository": "owner/repo",
            "run_id": 1,
            "job_name": "system",
            "actor": {"login": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"},
        },
        max_chunk_chars=95,
    )
    sidecar = result["sidecar"]
    serialized = stable_json_dumps(result)

    assert sidecar["raw"]["status"] == "invalid_replaced"
    assert sidecar["raw"]["invalid_sequence_count"] == 2
    assert sidecar["raw"]["raw_byte_count"] == len(raw)
    assert sidecar["raw"]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "\ufffd\ufffd" in result["canonical_text"]
    assert "\x1b" not in result["canonical_text"]
    assert sidecar["canonicalization"]["ansi"]["sequence_count"] == 2
    assert sidecar["canonicalization"]["ansi"]["removed_char_count"] == 9
    assert sidecar["canonicalization"]["record_boundary_anomalies"] == [
        {
            "decoded_char_offset": 0,
            "kind": "leading_utf8_bom",
            "action": "removed",
        },
        {
            "decoded_char_offset": 265,
            "kind": "midstream_bom_before_github_timestamp",
            "action": "replaced_with_missing_line_boundary",
        },
    ]
    assert sidecar["canonicalization"]["timestamp_prefixes"]["count"] == 10
    assert sidecar["canonicalization"]["secrets"]["redaction_count"] == 3
    assert "***" in result["canonical_text"]
    for secret in (
        github_token.decode(),
        aws_key.decode(),
        bearer.decode(),
    ):
        assert secret not in result["canonical_text"]
        assert secret not in serialized
    assert sidecar["security"]["metadata_secret_redaction_count"] == 1

    substitutions = sidecar["deduplication"]["substitutions"]
    assert {
        item["rule_id"] for item in substitutions
    } == {
        "hosted_runner_numeric_instance",
        "worker_id_uuid",
        "posix_temp_uuid",
        "ephemeral_pid",
    }
    assert len(result["dedup_text"]) == len(result["canonical_text"])
    reconstructed = list(result["dedup_text"])
    for item in substitutions:
        reconstructed[item["canonical_char_start"] : item["canonical_char_end"]] = (
            item["original"]
        )
    assert "".join(reconstructed) == result["canonical_text"]
    assert sidecar["canonicalization"]["accounting"]["character_count_conserved"]
    assert result["sections"][0]["kind"] == "system"
    assert sidecar["classifications"]["platform"]["runner_labels"]["value"] == [
        "ubuntu-latest"
    ]
    _assert_conserved(result)


def test_no_line_loss_deterministic_hashes_and_single_oversized_line() -> None:
    raw = _timestamped(
        "preamble",
        "##[group]Run echo first",
        "  shell: /bin/bash -e {0}",
        "[command]echo first",
        "x" * 180,
        "",
        "##[group]Run echo second",
        "  shell: /bin/bash -e {0}",
        "[command]echo second",
        "done",
    )
    metadata = {
        "repository": "owner/repo",
        "run_id": 5,
        "run_attempt": 1,
        "job": {
            "id": 6,
            "name": "build",
            "steps": [{"name": "echo first"}, {"name": "echo second"}],
        },
    }

    first = canonicalize_ci_log(raw, metadata, max_chunk_chars=80)
    second = canonicalize_ci_log(raw, dict(reversed(list(metadata.items()))), max_chunk_chars=80)

    assert first == second
    assert first["sidecar"]["sidecar_sha256"] == second["sidecar"]["sidecar_sha256"]
    assert len(first["canonical_text"].splitlines()) == 10
    assert any(chunk["oversized_single_line"] for chunk in first["chunks"])
    assert all(
        chunk["line_end"] > chunk["line_start"] for chunk in first["chunks"]
    )
    assert [section["step_ordinal"] for section in first["sections"]] == [
        None,
        0,
        1,
    ]
    assert json.loads(stable_json_dumps(first)) == first
    _assert_conserved(first)


def test_bazel_and_api_validation_and_sidecar_only_wrapper() -> None:
    result = canonicalize_ci_log(
        "##[group]Run bazel build //app:binary\n",
        {"run_id": 1},
    )
    assert extract_ci_log_sidecar(
        "##[group]Run bazel build //app:binary\n",
        {"run_id": 1},
    ) == result["sidecar"]
    assert result["sidecar"]["provenance"]["run"]["id"] == 1
    assert result["sidecar"]["provenance"]["run"]["attempt"] is None
    assert result["sidecar"]["provenance"]["field_confidence"]["run_id"][
        "score"
    ] == 1.0
    assert result["sidecar"]["provenance"]["field_confidence"]["run_attempt"][
        "score"
    ] == 0.0

    with pytest.raises(TypeError, match="raw_log"):
        canonicalize_ci_log(123, {})
    with pytest.raises(TypeError, match="metadata"):
        canonicalize_ci_log("", [])
    with pytest.raises(TypeError, match="integer"):
        canonicalize_ci_log("", {}, max_chunk_chars=True)
    with pytest.raises(ValueError, match="positive"):
        canonicalize_ci_log("", {}, max_chunk_chars=0)


def test_high_cardinality_evidence_is_bounded_and_fully_digested() -> None:
    raw = _timestamped(
        *[
            f"[command]clang++ -c src/file_{index:03d}.cpp -o build/file_{index:03d}.o"
            for index in range(80)
        ]
    )
    result = canonicalize_ci_log(raw, {"run_id": 1})
    sidecar = result["sidecar"]
    graph_receipt = sidecar["evidence_accounting"]["graph"]
    path_receipt = sidecar["evidence_accounting"]["classifications"]["paths"]

    assert graph_receipt["entity_occurrence_count"] > 100
    assert graph_receipt["entity_group_count"] > 24
    assert graph_receipt["retained_entity_group_count"] == 24
    assert graph_receipt["omitted_entity_group_count"] > 0
    assert len(graph_receipt["all_entity_groups_sha256"]) == 64
    assert len(sidecar["entities"]) == 24
    assert path_receipt["group_count"] > 16
    assert path_receipt["retained_group_count"] == 16
    assert path_receipt["omitted_group_count"] > 0
    assert len(path_receipt["all_groups_sha256"]) == 64
    assert len(sidecar["classifications"]["paths"]) == 16
    last_source_offset = result["canonical_text"].index("src/file_079.cpp")
    last_source_chunk = next(
        chunk
        for chunk in result["chunks"]
        if chunk["char_start"] <= last_source_offset < chunk["char_end"]
    )
    local_offset = last_source_offset - last_source_chunk["char_start"]
    assert any(
        span["start_char"] <= local_offset < span["end_char"]
        and span["role_id"] == 13
        for span in last_source_chunk["role_spans"]
    )
    assert any(
        span["start_char"] <= local_offset < span["end_char"]
        and span["domain_id"] == 1
        for span in last_source_chunk["domain_spans"]
    )
    exhaustive_entities = last_source_chunk["training_sidecars"]["entities"]
    assert any(
        entity["role_id"] == 13
        and last_source_chunk["text"][
            entity["start_char"] : entity["end_char"]
        ]
        == "src/file_079.cpp"
        for entity in exhaustive_entities
    )
    assert sidecar["evidence_accounting"]["training_sidecars"][
        "entity_span_count"
    ] > len(sidecar["entities"])
    _assert_conserved(result)
