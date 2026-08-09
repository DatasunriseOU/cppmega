from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from pathlib import Path
from threading import Timer

import pytest

_VALGRIND_AUTOCONF_ARGS = [
    "-x",
    "c++",
    "-std=c++1y",
    "-m32",
    "-DVGPV_<arch>_<os>_<variant>",
    "-DVGP_<arch>_<os>",
    "-DVGP_arm_linux",
    "-DVGPV_arm_linux_vanilla",
    "-DVGP_arm_linux",
    "-DVGPV_arm_linux_android",
    "-std=c++1y",
    "-std=c++0x",
    "-march=@<:@^",
    "-march=mips64r2",
    "-march=mips64r2",
    "-m32",
    "-m64",
    "-march=octeon",
    "-march=octeon]",
    "-march=octeon2",
    "-march=octeon2]",
    "-march=mips64r2",
    "-m32",
    "-std=gnu99.",
    "-std=gnu99",
    "-D__user=",
    "-m32",
    "-m64",
    "-fsyntax-only",
    "-Wno-everything",
]


_GCC_LIMITS_CASELABELS_SOURCE = (
    b"#define LIM1(x) x##0: x##1: x##2: x##3: x##4: x##5: x##6: x##7: "
    b"x##8: x##9: \n"
    b"#define LIM2(x) LIM1(x##0) LIM1(x##1) LIM1(x##2) LIM1(x##3) LIM1(x##4) "
    b"\\\n\t\tLIM1(x##5) LIM1(x##6) LIM1(x##7) LIM1(x##8) LIM1(x##9)\n"
    b"#define LIM3(x) LIM2(x##0) LIM2(x##1) LIM2(x##2) LIM2(x##3) LIM2(x##4) "
    b"\\\n\t\tLIM2(x##5) LIM2(x##6) LIM2(x##7) LIM2(x##8) LIM2(x##9)\n"
    b"#define LIM4(x) LIM3(x##0) LIM3(x##1) LIM3(x##2) LIM3(x##3) LIM3(x##4) "
    b"\\\n\t\tLIM3(x##5) LIM3(x##6) LIM3(x##7) LIM3(x##8) LIM3(x##9)\n"
    b"#define LIM5(x) LIM4(x##0) LIM4(x##1) LIM4(x##2) LIM4(x##3) LIM4(x##4) "
    b"\\\n\t\tLIM4(x##5) LIM4(x##6) LIM4(x##7) LIM4(x##8) LIM4(x##9)\n"
    b"#define LIM6(x) LIM5(x##0) LIM5(x##1) LIM5(x##2) LIM5(x##3) LIM5(x##4) "
    b"\\\n\t\tLIM5(x##5) LIM5(x##6) LIM5(x##7) LIM5(x##8) LIM5(x##9)\n"
    b"#define LIM7(x) LIM6(x##0) LIM6(x##1) LIM6(x##2) LIM6(x##3) LIM6(x##4) "
    b"\\\n\t\tLIM6(x##5) LIM6(x##6) LIM6(x##7) LIM6(x##8) LIM6(x##9)\n"
    b"\nvoid q19_func (long i)\n{\n  switch (i) {\n    LIM5 (case 1)\n"
    b"      break;\n  }\n}\n"
)


def _write_valgrind_style_configure(project_dir: Path) -> None:
    detected_flags = _VALGRIND_AUTOCONF_ARGS[3:-2]
    (project_dir / "configure.ac").write_text(
        "AC_PROG_CXX\n"
        f"CXXFLAGS='{' '.join(detected_flags)}'\n",
        encoding="utf-8",
    )


def _write_autoconf_flags(
    project_dir: Path,
    *,
    compiler_macro: str,
    variable: str,
    flags: str,
) -> None:
    (project_dir / "configure.ac").write_text(
        f"{compiler_macro}\n{variable}='{flags}'\n",
        encoding="utf-8",
    )


def _load_indexer():
    try:
        from tools.clang_indexer import index_project

        index_project._configure_libclang()
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"libclang unavailable: {exc}")
    return index_project


def test_parse_file_batch_fails_loud_with_file_and_cause(tmp_path: Path) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-x", "definitely-not-a-language"],
                str(tmp_path),
                "fixture/parse-failure",
            )
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message


def test_sequential_project_parse_fails_loud_instead_of_publishing(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        "-x",
                        "definitely-not-a-language",
                        str(source),
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as raised:
        index_project.process_project(
            str(tmp_path),
            enriched=True,
            project_id="fixture/parse-failure",
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message


def test_sane_translation_unit_load_error_uses_bound_lossless_lexical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "native_crash.cpp"
    raw_source = (
        "// libclang native-crash regression: π\n"
        "int preserved_answer() { return 42; }\n"
    ).encode()
    source.write_bytes(raw_source)

    class TranslationUnitLoadError(Exception):
        pass

    def crash_translation_unit(*_args, **_kwargs):
        raise TranslationUnitLoadError("native parser crashed")

    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        crash_translation_unit,
    )
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/native-clang-crash",
        )
    )

    assert parsed_count == 1
    assert payload["functions"] == []
    assert payload["typedefs"] == []
    assert payload["lexical_fallback_files"] == ["native_crash.cpp"]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "native_crash.cpp",
            "trigger": "translation_unit_load_error",
            "status": "lexical_fallback",
            "fallback_mode": "lossless_cpp_lexical_v1",
            "fallback_reason": "translation_unit_load_error",
            "compile_args_status": "sane",
            "compile_arg_count": 3,
            "compile_args_sha256": payload["parse_recovery_records"][0][
                "compile_args_sha256"
            ],
            "source_size_bytes": len(raw_source),
            "source_char_count": len(raw_source.decode("utf-8")),
            "source_sha256": hashlib.sha256(raw_source).hexdigest(),
            "source_encoding": "utf-8",
        }
    ]
    summary = index_project._parse_recovery_summary(
        payload["parse_recovery_records"]
    )
    assert summary["status"] == "complete"
    assert summary["recovered_file_count"] == 1
    assert summary["semantic_recovered_file_count"] == 0
    assert summary["lexical_fallback_file_count"] == 1
    assert summary["lexical_fallback_source_bytes"] == len(raw_source)
    assert summary["unresolved_file_count"] == 0


def test_gcc_case_label_ast_recursion_uses_bound_lossless_lexical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    assert len(_GCC_LIMITS_CASELABELS_SOURCE) == 935
    assert hashlib.sha256(_GCC_LIMITS_CASELABELS_SOURCE).hexdigest() == (
        "cd4aaab81ac06ab6265f81567297015b364c8c6e026e757fac87cc38faae868e"
    )
    source = tmp_path / "limits-caselabels.c"
    source.write_bytes(_GCC_LIMITS_CASELABELS_SOURCE)

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    visitor_namespace: dict[str, object] = {}
    exec(
        compile(
            "def _visit():\n"
            "    raise RecursionError('maximum recursion depth exceeded')\n",
            index_project.__file__,
            "exec",
        ),
        visitor_namespace,
    )
    visitor = visitor_namespace["_visit"]
    assert callable(visitor)
    monkeypatch.setattr(
        index_project,
        "parse_translation_unit",
        lambda *_args, **_kwargs: visitor(),
    )
    compile_args = ["-std=c11", "-fsyntax-only", "-Wno-everything"]

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            compile_args,
            str(tmp_path),
            "gcc-mirror/gcc",
        )
    )

    assert parsed_count == 1
    assert payload["functions"] == []
    assert payload["typedefs"] == []
    assert payload["lexical_fallback_files"] == ["limits-caselabels.c"]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "limits-caselabels.c",
            "trigger": "ast_recursion_error",
            "status": "lexical_fallback",
            "fallback_mode": "lossless_cpp_lexical_v1",
            "fallback_reason": "ast_recursion_error",
            "compile_args_status": "sane",
            "compile_arg_count": 5,
            "compile_args_sha256": payload["parse_recovery_records"][0][
                "compile_args_sha256"
            ],
            "source_size_bytes": len(_GCC_LIMITS_CASELABELS_SOURCE),
            "source_char_count": len(_GCC_LIMITS_CASELABELS_SOURCE),
            "source_sha256": hashlib.sha256(
                _GCC_LIMITS_CASELABELS_SOURCE
            ).hexdigest(),
            "source_encoding": "utf-8",
        }
    ]

    documents = index_project.emit_cpp_lexical_fallback_documents(
        payload["lexical_fallback_files"],
        index=index_project.ProjectIndex(),
        project_dir=str(tmp_path),
        project_id="gcc-mirror/gcc",
        compile_db={},
        default_args=compile_args,
        default_build_info=None,
        parse_recovery_records=payload["parse_recovery_records"],
        enriched=True,
    )

    assert len(documents) == 1
    assert documents[0]["text"].encode("utf-8") == _GCC_LIMITS_CASELABELS_SOURCE
    assert documents[0]["cpp_parse_fallback"]["reason"] == (
        "ast_recursion_error"
    )
    assert documents[0]["domain_parse_info"]["fallback_reason"] == (
        "ast_recursion_error"
    )


def test_unrelated_recursion_error_still_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "unrelated-recursion.cpp"
    source.write_text("int value = 1;\n", encoding="utf-8")

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    monkeypatch.setattr(
        index_project,
        "parse_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RecursionError("unrelated recursion bug")
        ),
    )

    with pytest.raises(RuntimeError, match="RecursionError"):
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
                str(tmp_path),
                "fixture/unrelated-recursion",
            )
        )


def test_non_translation_unit_error_still_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "unexpected_failure.cpp"
    source.write_text("int preserved = 1;\n")
    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("not a libclang TU load error")
        ),
    )

    with pytest.raises(RuntimeError, match="ValueError"):
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
                str(tmp_path),
                "fixture/non-tu-error",
            )
        )


def test_sane_translation_unit_load_error_emits_heuristic_source_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega.data.domain_schema import ParseConfidence
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "driver.cpp"
    raw_source = (
        b"// Useful driver implementation must not be quarantined.\n"
        b"int driver_value(int input) { return input + 7; }\n"
    )
    source.write_bytes(raw_source)

    class TranslationUnitLoadError(Exception):
        pass

    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TranslationUnitLoadError("native parser crashed")
        ),
    )
    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/lossless-lexical-fallback",
    )

    assert len(documents) == 1
    document = documents[0]
    assert document["doc_type"] == "code"
    assert document["filepath"] == "driver.cpp"
    assert document["text"].encode("utf-8") == raw_source
    assert document["cpp_parse_fallback"] == {
        "schema": "cppmega.cpp_parse_fallback_v1",
        "mode": "lossless_cpp_lexical_v1",
        "reason": "translation_unit_load_error",
        "compile_args_status": "sane",
        "source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "source_encoding": "utf-8",
        "source_span": document["source_span"],
    }
    assert document["source_span"] == {
        "chunk_index": 0,
        "byte_start": 0,
        "byte_end": len(raw_source),
        "char_start": 0,
        "char_end": len(raw_source.decode("utf-8")),
        "source_size_bytes": len(raw_source),
        "chunk_limit_bytes": index_project.CPP_LEXICAL_FALLBACK_CHUNK_BYTES,
        "split_reason": "eof",
        "source_encoding": "utf-8",
    }
    text_len = len(document["text"])
    assert document["domain_confidence_ids"] == [
        int(ParseConfidence.HEURISTIC)
    ] * text_len
    for field in (
        "call_edges",
        "type_edges",
        "build_edges",
        "shell_edges",
        "diagnostic_edges",
    ):
        assert document[field] == []
    for field in (
        "structure_ids",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        "domain_scope_ids",
    ):
        assert document[field] == [0] * text_len
    for field in (
        "domain_ids",
        "domain_role_ids",
        "domain_entity_ids",
        "domain_source_doc_ids",
        "domain_source_identity_ids",
    ):
        assert len(document[field]) == text_len


def test_cpp_lexical_fallback_chunks_preserve_utf8_bytes_and_spans(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "utf8.cpp"
    raw_source = (
        "// αβγδεζηθ\n"
        "int first = 1;\n"
        "int second = 2;\n"
    ).encode()
    source.write_bytes(raw_source)

    chunks = list(
        index_project._iter_cpp_lexical_fallback_chunks(
            str(source),
            max_chunk_bytes=17,
        )
    )

    assert b"".join(text.encode("utf-8") for text, _span in chunks) == raw_source
    assert [span["chunk_index"] for _text, span in chunks] == list(
        range(len(chunks))
    )
    assert [span["byte_start"] for _text, span in chunks] == [
        0,
        *[span["byte_end"] for _text, span in chunks[:-1]],
    ]
    assert chunks[-1][1]["byte_end"] == len(raw_source)


def test_cpp_lexical_fallback_rejects_project_root_escape(tmp_path: Path) -> None:
    from tools.clang_indexer import index_project

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    outside_source = tmp_path / "outside.cpp"
    outside_source.write_text("int outside = 1;\n")

    class TranslationUnitLoadError(Exception):
        pass

    with pytest.raises(RuntimeError, match="escapes the project root"):
        index_project._record_cpp_lexical_fallback(
            str(outside_source),
            ["-x", "c++", "-std=c++17"],
            str(project_dir),
            TranslationUnitLoadError("native parser crashed"),
            [],
        )


def test_cpp_lexical_fallback_rechecks_compile_args_digest(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "changed_context.cpp"
    source.write_text("int changed_context = 1;\n")

    class TranslationUnitLoadError(Exception):
        pass

    parse_args = index_project._resolve_file_args(
        str(source),
        {},
        ["-std=c++17"],
    )
    records: list[dict[str, object]] = [
        {
            "relative_path": "changed_context.cpp",
            "trigger": "missing_include_diagnostic",
            "status": "unresolved",
        }
    ]
    relative_path = index_project._record_cpp_lexical_fallback(
        str(source),
        parse_args,
        str(tmp_path),
        TranslationUnitLoadError("native parser crashed"),
        records,
    )

    assert relative_path == "changed_context.cpp"
    assert records[0]["trigger"] == "translation_unit_load_error"
    with pytest.raises(RuntimeError, match="compile args changed"):
        index_project.emit_cpp_lexical_fallback_documents(
            [relative_path],
            index=index_project.ProjectIndex(),
            project_dir=str(tmp_path),
            project_id="fixture/compile-args-drift",
            compile_db=None,
            default_args=["-std=c++20"],
            default_build_info=None,
            parse_recovery_records=records,
            enriched=True,
        )


def test_cp1252_source_round_trips_without_clang_token_utf8_decode(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "legacy.cpp"
    raw_type = b"struct /* compiler\x92s type */ LegacyType { int value; };\r\n"
    raw_function = (
        b"int /* compiler\x92s invalid candidate */ "
        b"legacy_answer() { return 42; }\r\n"
    )
    raw_source = raw_type + raw_function
    source.write_bytes(raw_source)

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/cp1252-source",
        )
    )

    assert parsed_count == 1
    function = next(
        item for item in payload["functions"]
        if item["name"] == "legacy_answer"
    )
    type_definition = next(
        item for item in payload["typedefs"]
        if item["name"] == "LegacyType"
    )
    assert function["text"] == raw_function[:-2].decode("cp1252")
    assert "compiler’s invalid candidate" in function["text"]
    assert type_definition["text"] == raw_type[:-3].decode("cp1252")
    assert "compiler’s type" in type_definition["text"]
    for field in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "semantic_symbol_ids",
        "semantic_call_targets",
        "semantic_type_refs",
        "semantic_def_use",
    ):
        assert len(function[field]) == len(function["text"])


def test_nested_and_reopened_anonymous_namespaces_parse_with_scoped_usr(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "anonymous_namespaces.cpp"
    source.write_text(
        """
namespace outer {
namespace {
namespace {
int nested_answer() { return 42; }
}
}
namespace {
int reopened_answer() { return 43; }
}
}
""".lstrip(),
        encoding="utf-8",
    )

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/anonymous-namespaces",
        )
    )

    assert parsed_count == 1
    assert {item["name"] for item in payload["functions"]} == {
        "nested_answer",
        "reopened_answer",
    }


def test_macro_expanded_call_without_explicit_callee_token_is_not_annotated(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "macro_call.cpp"
    source.write_text(
        "#define DO_CALL() target()\n"
        "int target() { return 7; }\n"
        "int macro_wrapper() { return DO_CALL(); }\n"
        "int direct_wrapper() { return target(); }\n",
        encoding="utf-8",
    )

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/macro-call",
        )
    )

    assert parsed_count == 1
    functions = {item["name"]: item for item in payload["functions"]}
    macro_wrapper = functions["macro_wrapper"]
    direct_wrapper = functions["direct_wrapper"]
    assert not any(macro_wrapper["semantic_call_targets"])
    target_start = direct_wrapper["text"].index("target")
    target_end = target_start + len("target")
    assert all(
        direct_wrapper["semantic_call_targets"][target_start:target_end]
    )


def test_no_linkage_parameter_in_spaced_repository_path_parses(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = (
        tmp_path
        / "third_party/libsdl2/Xcode-iOS/Template"
        / "SDL iOS Application"
        / "main.c"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "int randomInt(const int min, const int max) {\n"
        "    return min < max ? min : max;\n"
        "}\n",
        encoding="utf-8",
    )

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c11", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/spaced-repository-path",
        )
    )

    assert parsed_count == 1
    assert [item["name"] for item in payload["functions"]] == ["randomInt"]


@pytest.mark.parametrize(
    "compile_args",
    (
        [
            "-x",
            "c",
            "-std=iso9899:199x",
            "-D",
            "FEATURE_LEVEL=2",
            "-U",
            "LEGACY_FEATURE",
            "-I",
            "include",
            "-isystem",
            "sysinc",
            "-iquote",
            "quoted",
            "-include",
            "config.h",
            "--target=x86_64-unknown-linux-gnu",
            "-march=x86-64",
        ],
        [
            "-xc++",
            "--std=c++20",
            "-DFEATURE_LEVEL=2",
            "-ULEGACY_FEATURE",
            "-m64",
        ],
        ["-x", "c", "-std=iso9899:199409"],
        ["-x", "c", "-std=iso9899:201x"],
        ["-x", "c++", "-std=c++2c"],
        ["-x", "cl", "-std=cl3.0"],
        ["-x", "cl", "-cl-std=CL3.0"],
        [
            "-xc++",
            "-isystemsysinc",
            "-iquotequoted",
            "-includeconfig.h",
            "--sysroot=/sdk",
            "-isysroot=/sdk2",
            "-resource-dir=/res",
        ],
        [
            "-xc++",
            "-march=armv8-a",
            "-march=armv8-a",
            "-mcpu=cortex-a76",
            "-mcpu=cortex-a76",
        ],
    ),
)
def test_sane_compile_arg_matrix_is_accepted(
    compile_args: list[str],
) -> None:
    from tools.clang_indexer import index_project

    assert index_project._is_sane_compile_args(compile_args)


@pytest.mark.parametrize(
    "compile_args",
    (
        [],
        ["-march=@<:@^"],
        ["-DVGPV_<arch>_<os>_<variant>"],
        ["-DVERSION=@VERSION@"],
        ["-DSEPARATOR=@S|@"],
        ["-std=cbanana"],
        ["-std=iso9899:1990:123"],
        ["-std=iso9899:2024"],
        ["-std=gnu99."],
        ["-std="],
        ["-std"],
        ["--std=gnu99."],
        ["-D"],
        ["-U"],
        ["-x"],
        ["-D", "-ULEGACY_FEATURE"],
        ["-x", "not-a-clang-language"],
        ["--target="],
        ["--target=x86/64"],
        ["-march="],
        ["-march=bad]"],
        ["-mcpu="],
        ["-mcpu=bad]"],
        ["-march=armv8-a", "-march=armv8.2-a+fp16"],
        ["-mcpu=cortex-a55", "-mcpu=cortex-a76"],
        ["-target", "x86_64-linux-gnu", "--target=aarch64-linux-gnu"],
        ["-m32", "-m64"],
        ["-x", "c++", "-std=iso9899:199x"],
        ["-x", "c++", "-std=c++17", "-std=c++20"],
        ["-x", "c++", "-std=c++20", "-std=c++17"],
        ["-x", "c++", "-std=c++20", "-std=c11"],
        ["-cl-std=CL3.0"],
        ["-x", "c", "-cl-std=CL3.0"],
        ["-x", "c++", "-cl-std=CL3.0"],
    ),
)
def test_unusable_detected_compile_args_are_rejected_atomically(
    compile_args: list[str],
) -> None:
    from tools.clang_indexer import index_project

    assert not index_project._is_sane_compile_args(compile_args)


def test_empty_detected_args_use_fallback_without_false_provenance(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            {
                "build_system": "autoconf",
                "source": "build_files",
                "compiler": "gcc",
                "standard": "c11",
            },
            [],
        )
    )

    assert default_args[:2] == ["-fsyntax-only", "-Wno-everything"]
    assert default_build_info == {
        "build_system": "autoconf",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }


def test_valgrind_autoconf_args_fall_back_atomically(tmp_path: Path) -> None:
    from cppmega.data.build_context import detect_build_context
    from tools.clang_indexer import index_project

    _write_valgrind_style_configure(tmp_path)

    platform_info, detected_args, compile_index = detect_build_context(
        str(tmp_path)
    )
    assert compile_index is None
    assert detected_args == _VALGRIND_AUTOCONF_ARGS
    assert len(detected_args) == 30
    assert platform_info["compiler"] == "g++"
    assert platform_info["standard"] == "c++1y"

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            platform_info,
            detected_args,
        )
    )

    assert default_args[:2] == ["-fsyntax-only", "-Wno-everything"]
    assert not any(
        arg.startswith(("-march=", "-std=", "-DVGP"))
        or arg in {"-m32", "-m64"}
        for arg in default_args
    )
    assert default_build_info == {
        "build_system": "autoconf",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }
    assert index_project.get_default_compile_args(str(tmp_path)) == default_args


def test_compute_library_conflicting_architectures_fall_back_atomically(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    detected_args = [
        "-x",
        "c++",
        "-march=armv8-a",
        "-march=armv8.2-a+fp16",
        "-march=armv8.6-a+sve2+fp16+dotprod",
        "-march=armv8.2-a+sve+fp16+dotprod",
        "-fsyntax-only",
        "-Wno-everything",
    ]

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            {
                "build_system": "bazel",
                "source": "build_files",
                "compiler": "g++",
            },
            detected_args,
        )
    )

    assert default_args[:2] == ["-fsyntax-only", "-Wno-everything"]
    assert not any(arg.startswith("-march=") for arg in default_args)
    assert default_build_info == {
        "build_system": "bazel",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }


@pytest.mark.parametrize("quoted_standard", ['"23"', "23"])
def test_cmake_quoted_cxx_standard_drives_truthful_parser_dialect(
    tmp_path: Path,
    quoted_standard: str,
) -> None:
    from cppmega.data.build_context import detect_build_context

    index_project = _load_indexer()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(quoted_standard LANGUAGES CXX)\n"
        f"set(CMAKE_CXX_STANDARD {quoted_standard} CACHE INTERNAL \"\")\n",
        encoding="utf-8",
    )
    source = tmp_path / "quoted_standard.cpp"
    source.write_text(
        "constexpr int quoted_standard(bool value) {\n"
        "    if consteval { return 23; }\n"
        "    return value ? 1 : 0;\n"
        "}\n",
        encoding="utf-8",
    )

    platform_info, detected_args, compile_index = detect_build_context(
        str(tmp_path)
    )

    assert compile_index is None
    assert platform_info["build_system"] == "cmake"
    assert platform_info["standard"] == "c++23"
    assert "-std=c++23" in detected_args
    file_args = index_project._resolve_file_args(
        str(source),
        {},
        index_project.get_default_compile_args(str(tmp_path)),
    )
    assert "-std=c++23" in file_args
    translation_unit = index_project._load_translation_unit(
        str(source),
        index_project.Index.create(),
        file_args,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]


@pytest.mark.parametrize(
    ("iso_alias", "canonical_standard"),
    (
        ("iso9899:199x", "c99"),
        ("iso9899:201x", "c11"),
    ),
)
def test_iso_c_alias_has_truthful_emitted_sidecars(
    tmp_path: Path,
    iso_alias: str,
    canonical_standard: str,
) -> None:
    from cppmega.data.build_context import detect_build_context

    index_project = _load_indexer()
    _write_autoconf_flags(
        tmp_path,
        compiler_macro="AC_PROG_CC",
        variable="CFLAGS",
        flags=f"-std={iso_alias}",
    )
    source = tmp_path / "src" / "iso_alias.c"
    source.parent.mkdir(parents=True)
    source.write_text(
        "int iso_alias_sum(int first, int second, int third) {\n"
        "    int first_pair = first + second;\n"
        "    int bounded_third = third < 0 ? 0 : third;\n"
        "    return first_pair + bounded_third;\n"
        "}\n",
        encoding="utf-8",
    )

    platform_info, detected_args, _compile_index = detect_build_context(
        str(tmp_path)
    )
    assert platform_info["standard"] == canonical_standard

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            platform_info,
            detected_args,
        )
    )
    assert default_args[:2] == ["-x", "c"]
    assert f"-std={iso_alias}" in default_args
    assert default_build_info["compiler"] == "gcc"
    assert default_build_info["standard"] == canonical_standard
    assert "compile_args_status" not in default_build_info

    file_args = index_project._resolve_file_args(
        str(source),
        {},
        default_args,
    )
    translation_unit = index_project._load_translation_unit(
        str(source),
        index_project.Index.create(),
        file_args,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]

    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/iso-c-sidecar",
    )
    code_document = next(
        document
        for document in documents
        if document["doc_type"] == "code"
        and document["filepath"] == "src/iso_alias.c"
    )
    assert code_document["build_info"]["compiler"] == "gcc"
    assert code_document["build_info"]["standard"] == canonical_standard
    assert code_document["language_info"]["primary_language"] == "c"
    assert (
        code_document["language_info"]["primary_standard"]
        == canonical_standard
    )


@pytest.mark.parametrize(
    (
        "compiler_macro",
        "variable",
        "flags",
        "case_label",
        "suffix",
        "expected_compiler",
        "expected_language",
        "expected_standard",
    ),
    (
        (
            "AC_PROG_CC",
            "CFLAGS",
            "-DNAME=1",
            "c-flags",
            ".c",
            "gcc",
            "c",
            "c11",
        ),
        (
            "AC_PROG_CC",
            "CFLAGS",
            None,
            "plain-c",
            ".c",
            "gcc",
            "c",
            "c11",
        ),
        (
            "AC_PROG_CXX",
            "CXXFLAGS",
            "-DNAME=1",
            "cpp-flags",
            ".cpp",
            "g++",
            "c++",
            None,
        ),
    ),
)
def test_autoconf_compiler_without_standard_has_truthful_parser_sidecars(
    tmp_path: Path,
    compiler_macro: str,
    variable: str,
    flags: str | None,
    case_label: str,
    suffix: str,
    expected_compiler: str,
    expected_language: str,
    expected_standard: str | None,
) -> None:
    from cppmega.data.build_context import detect_build_context

    index_project = _load_indexer()
    if flags is None:
        (tmp_path / "configure.ac").write_text(
            f"{compiler_macro}\n",
            encoding="utf-8",
        )
    else:
        _write_autoconf_flags(
            tmp_path,
            compiler_macro=compiler_macro,
            variable=variable,
            flags=flags,
        )
    source = tmp_path / "src" / f"configured_answer{suffix}"
    source.parent.mkdir(parents=True)
    source.write_text(
        "int configured_answer(int first, int second, int third) {\n"
        "    int first_pair = first + second;\n"
        "    int bounded_third = third < 0 ? 0 : third;\n"
        "    return first_pair + bounded_third + 1;\n"
        "}\n",
        encoding="utf-8",
    )

    platform_info, detected_args, _compile_index = detect_build_context(
        str(tmp_path)
    )
    assert "standard" not in platform_info
    assert detected_args[:2] == ["-x", expected_language]

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            platform_info,
            detected_args,
        )
    )
    assert default_build_info["compiler"] == expected_compiler
    assert "standard" not in default_build_info
    file_args = index_project._resolve_file_args(
        str(source),
        {},
        default_args,
    )
    if expected_standard is None:
        assert not any(arg.startswith("-std=") for arg in file_args)
    else:
        assert f"-std={expected_standard}" in file_args

    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id=f"fixture/autoconf-{case_label}-no-standard",
    )
    code_document = next(
        document
        for document in documents
        if document["doc_type"] == "code"
        and document["filepath"] == f"src/configured_answer{suffix}"
    )
    assert code_document["build_info"]["compiler"] == expected_compiler
    if expected_standard is None:
        assert "standard" not in code_document["build_info"]
    else:
        assert (
            code_document["build_info"]["standard"]
            == expected_standard
        )
    assert (
        code_document["language_info"]["primary_language"]
        == expected_language
    )
    assert (
        code_document["language_info"]["primary_standard"]
        == expected_standard
    )


def test_mixed_c_and_cpp_sidecars_match_each_files_adapted_parser_args(
    tmp_path: Path,
) -> None:
    from cppmega.data.build_context import detect_build_context

    index_project = _load_indexer()
    _write_autoconf_flags(
        tmp_path,
        compiler_macro="AC_PROG_CXX",
        variable="CXXFLAGS",
        flags="-std=c++20",
    )
    c_source = tmp_path / "src" / "mixed.c"
    cpp_source = tmp_path / "src" / "mixed.cpp"
    c_source.parent.mkdir(parents=True)
    c_source.write_text(
        "int mixed_c_answer(int first, int second, int third) {\n"
        "    int first_pair = first + second;\n"
        "    int bounded_third = third < 0 ? 0 : third;\n"
        "    return first_pair + bounded_third + 20;\n"
        "}\n",
        encoding="utf-8",
    )
    cpp_source.write_text(
        "int mixed_cpp_answer(int first, int second, int third) {\n"
        "    int first_pair = first + second;\n"
        "    int bounded_third = third < 0 ? 0 : third;\n"
        "    return first_pair + bounded_third + 21;\n"
        "}\n",
        encoding="utf-8",
    )

    platform_info, detected_args, _compile_index = detect_build_context(
        str(tmp_path)
    )
    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            platform_info,
            detected_args,
        )
    )
    assert default_build_info["standard"] == "c++20"

    c_args = index_project._resolve_file_args(
        str(c_source),
        {},
        default_args,
    )
    cpp_args = index_project._resolve_file_args(
        str(cpp_source),
        {},
        default_args,
    )
    assert c_args[:3] == ["-x", "c", "-std=c11"]
    assert not any(arg.startswith("-std=c++") for arg in c_args)
    assert cpp_args[:2] == ["-x", "c++"]
    assert "-std=c++20" in cpp_args

    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/autoconf-mixed-dialects",
    )
    code_documents = {
        document["filepath"]: document
        for document in documents
        if document["doc_type"] == "code"
    }
    c_document = code_documents["src/mixed.c"]
    cpp_document = code_documents["src/mixed.cpp"]

    assert c_document["build_info"]["standard"] == "c11"
    assert c_document["language_info"]["primary_language"] == "c"
    assert c_document["language_info"]["primary_standard"] == "c11"
    assert (
        c_document["language_info"]["provenance"]["standard_flag"]
        == "-std=c11"
    )

    assert cpp_document["build_info"]["standard"] == "c++20"
    assert cpp_document["language_info"]["primary_language"] == "c++"
    assert cpp_document["language_info"]["primary_standard"] == "c++20"
    assert (
        cpp_document["language_info"]["provenance"]["standard_flag"]
        == "-std=c++20"
    )


@pytest.mark.parametrize(
    ("flags", "misleading_detected_standard"),
    (
        ("-std=c++17 -std=c++20", "c++17"),
        ("-std=c++20 -std=c++17", "c++20"),
        ("-std=c++20 -std=c11", "c++20"),
    ),
)
def test_distinct_detected_standards_fall_back_without_false_sidecars(
    tmp_path: Path,
    flags: str,
    misleading_detected_standard: str,
) -> None:
    from cppmega.data.build_context import detect_build_context
    from tools.clang_indexer import index_project

    _write_autoconf_flags(
        tmp_path,
        compiler_macro="AC_PROG_CXX",
        variable="CXXFLAGS",
        flags=flags,
    )
    platform_info, detected_args, _compile_index = detect_build_context(
        str(tmp_path)
    )
    assert platform_info["standard"] == misleading_detected_standard

    default_args, default_build_info = (
        index_project._resolve_default_compile_context(
            str(tmp_path),
            platform_info,
            detected_args,
        )
    )
    assert default_args[:2] == ["-fsyntax-only", "-Wno-everything"]
    assert not any(arg.startswith(("-std=", "--std=")) for arg in default_args)
    assert default_build_info == {
        "build_system": "autoconf",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }


@pytest.mark.parametrize(
    ("language", "standard", "suffix", "source_text", "expected_name"),
    (
        (
            "c",
            "-std=iso9899:199x",
            ".c",
            "int exact_iso_c_answer(void) { return 42; }\n",
            "exact_iso_c_answer",
        ),
        (
            "c",
            "-std=iso9899:201x",
            ".c",
            "int exact_iso_c11_answer(void) { return 44; }\n",
            "exact_iso_c11_answer",
        ),
        (
            "c++",
            "-std=c++20",
            ".cpp",
            "constexpr int exact_cpp_answer() { return 43; }\n",
            "exact_cpp_answer",
        ),
    ),
)
def test_valid_standard_context_loads_cleanly_and_extracts(
    tmp_path: Path,
    language: str,
    standard: str,
    suffix: str,
    source_text: str,
    expected_name: str,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / f"standard_probe{suffix}"
    source.write_text(source_text, encoding="utf-8")
    compile_args = [
        "-x",
        language,
        standard,
        "-fsyntax-only",
        "-Wno-everything",
    ]

    assert index_project._is_sane_compile_args(compile_args)
    translation_unit = index_project._load_translation_unit(
        str(source),
        index_project.Index.create(),
        compile_args,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            compile_args,
            str(tmp_path),
            "fixture/valid-standard-load",
        )
    )
    assert parsed_count == 1
    assert expected_name in {
        function["name"] for function in payload["functions"]
    }


def test_valgrind_style_header_loads_with_sane_fallback_args(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    _write_valgrind_style_configure(tmp_path)
    dependency = tmp_path / "VEX" / "pub" / "libvex_basictypes.h"
    dependency.parent.mkdir(parents=True)
    dependency.write_text(
        "typedef unsigned long UWord;\n",
        encoding="utf-8",
    )
    header = tmp_path / "drd" / "drd_clientobj.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        '#include "libvex_basictypes.h"\n'
        "typedef struct {\n"
        "    UWord start;\n"
        "    UWord end;\n"
        "} DrdClientObject;\n"
        "static inline UWord drd_client_object_span(\n"
        "    const DrdClientObject *object) {\n"
        "    return object->end - object->start;\n"
        "}\n",
        encoding="utf-8",
    )

    default_args = index_project.get_default_compile_args(str(tmp_path))
    header_args = index_project._adapt_args_for_file(
        default_args,
        str(header),
    )

    assert header_args[:2] == ["-x", "c++-header"]
    assert not any(
        arg.startswith(("-march=", "-std=", "-DVGP"))
        or arg in {"-m32", "-m64"}
        for arg in header_args
    )
    recovery_records: list[dict[str, object]] = []
    translation_unit = index_project._load_translation_unit_with_include_recovery(
        str(header),
        index_project.Index.create(),
        header_args,
        str(tmp_path),
        allow_include_recovery=True,
        parse_recovery_records=recovery_records,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]
    assert recovery_records[0]["status"] == "recovered"
    assert recovery_records[0]["added_include_dir_examples"] == ["VEX/pub"]
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(header)],
            {},
            default_args,
            str(tmp_path),
            "fixture/valgrind-header-load",
        )
    )
    assert parsed_count == 1
    assert "DrdClientObject" in {
        type_definition["name"]
        for type_definition in payload["typedefs"]
    }
    assert "drd_client_object_span" in {
        function["name"] for function in payload["functions"]
    }


def test_gnu_c_standard_header_keeps_consistent_language_family(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    header = tmp_path / "KeychainSyncAccountUpdater.h"
    header.write_text(
        "#import <UAUPlugin/UAUSession.h>\n\n"
        "@interface KeychainSyncAccountUpdater : NSObject "
        "<UserAccountUpdaterProtocol>\n\n"
        "@end\n",
        encoding="utf-8",
    )

    adapted = index_project._adapt_args_for_file(
        ["-std=gnu2x", "-fblocks", "-fsyntax-only", "-Wno-everything"],
        str(header),
    )

    assert adapted[:3] == ["-x", "c-header", "-std=gnu2x"]
    assert index_project._is_sane_compile_args(adapted)
    translation_unit = index_project._load_translation_unit(
        str(header),
        index_project.Index.create(),
        adapted,
    )
    assert translation_unit.spelling == str(header)


@pytest.mark.parametrize(
    ("standard_args", "expected_language", "expected_standard"),
    [
        (["--std=c11", "--std=gnu2x"], "c-header", "--std=gnu2x"),
        (["-cl-std=CL1.2", "-cl-std=CL3.0"], "cl", "-cl-std=CL3.0"),
    ],
)
def test_header_adaptation_keeps_only_last_standard_alias(
    tmp_path: Path,
    standard_args: list[str],
    expected_language: str,
    expected_standard: str,
) -> None:
    index_project = _load_indexer()
    header = tmp_path / "dialect.h"
    header.write_text("int dialect_fixture;\n", encoding="utf-8")

    adapted = index_project._adapt_args_for_file(
        [*standard_args, "-fsyntax-only", "-Wno-everything"],
        str(header),
    )

    standard_flags = [
        arg
        for arg in adapted
        if arg.startswith(("-std=", "--std=", "-cl-std="))
    ]
    assert adapted[:2] == ["-x", expected_language]
    assert standard_flags == [expected_standard]
    assert index_project._is_sane_compile_args(adapted)


def test_mixed_case_cpp_suffix_forces_cpp_language(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "Outgoing.Cpp"
    source.write_text(
        "namespace tapi {\n"
        "class Outgoing {\n"
        "public:\n"
        "    int call() const { return 3; }\n"
        "};\n"
        "}\n",
        encoding="utf-8",
    )

    adapted = index_project._adapt_args_for_file(
        ["-std=c++17", "-Wno-everything"],
        str(source),
    )

    assert adapted[:2] == ["-x", "c++"]
    assert "-std=c++17" in adapted
    translation_unit = index_project.Index.create().parse(
        str(source),
        args=adapted,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]


def test_mixed_case_cpp_suffix_preserves_explicit_language(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "kernel.Cpp"
    explicit_args = ["-x", "cuda", "--cuda-host-only"]

    assert index_project._adapt_args_for_file(
        explicit_args,
        str(source),
    ) == explicit_args
    assert index_project._adapt_args_for_file(
        ["-xdefinitely-invalid"],
        str(source),
    ) == ["-xdefinitely-invalid"]


def test_lowercase_cpp_suffix_does_not_rewrite_explicit_language(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "kernel.cpp"
    explicit_args = ["-x", "objective-c++", "-fobjc-arc"]

    assert index_project._adapt_args_for_file(
        explicit_args,
        str(source),
    ) == explicit_args


def test_open_watcom_plusplus_c_suffix_uses_cpp_without_source_loss(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "bld" / "plusplus" / "bugs" / "zcc02.c"
    source.parent.mkdir(parents=True)
    source.write_text(
        "char *foo( void )\n"
        "{\n"
        "    return( ::new char[10] ( 'a', 'b', 'c', '\\0' ) );\n"
        "}\n",
        encoding="utf-8",
    )

    adapted = index_project._adapt_args_for_file(
        ["-std=c11", "-std=c++20", "-fsyntax-only", "-Wno-everything"],
        str(source),
    )

    assert adapted[:3] == ["-x", "c++", "-std=c++20"]
    assert "-std=c11" not in adapted
    translation_unit = index_project._load_translation_unit(
        str(source),
        index_project.Index.create(),
        adapted,
    )
    assert translation_unit.spelling == str(source)

    ordinary_c = tmp_path / "src" / "ordinary.c"
    ordinary_c.parent.mkdir()
    ordinary_c.write_text("int ordinary(void) { return 0; }\n", encoding="utf-8")
    assert index_project._adapt_args_for_file(
        ["-std=c++20", "-Wno-everything"],
        str(ordinary_c),
    )[:3] == ["-x", "c", "-std=c11"]


def test_valgrind_fallback_status_is_emitted_in_source_sidecars(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    _write_valgrind_style_configure(tmp_path)
    source = tmp_path / "drd" / "client_registry.cpp"
    source.parent.mkdir(parents=True)
    source.write_text(
        "int register_client_object(int start, int end, int generation) {\n"
        "    int bounded_start = start < 0 ? 0 : start;\n"
        "    int bounded_end = end < bounded_start ? bounded_start : end;\n"
        "    return bounded_end - bounded_start + generation;\n"
        "}\n",
        encoding="utf-8",
    )

    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/valgrind-fallback-sidecar",
    )

    code_document = next(
        document
        for document in documents
        if document["doc_type"] == "code"
        and document["filepath"] == "drd/client_registry.cpp"
    )
    assert code_document["build_info"] == {
        "build_system": "autoconf",
        "source": "build_files",
        "compile_args_status": "fallback_unusable_detected_args",
    }


def test_parse_batch_recovers_nested_legacy_include_context(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source_dir = tmp_path / "legacy" / "src"
    include_dir = tmp_path / "legacy" / "vendor" / "include"
    nested_include_dir = tmp_path / "legacy" / "vendor" / "inc.next"
    decoy_include_dir = tmp_path / "other" / "include"
    source_dir.mkdir(parents=True)
    include_dir.mkdir(parents=True)
    nested_include_dir.mkdir(parents=True)
    decoy_include_dir.mkdir(parents=True)
    source = source_dir / "main.cpp"
    source.write_text(
        "int legacy_answer() { return LEGACY_ANSWER; }\n",
        encoding="utf-8",
    )
    (include_dir / "legacy_prelude.h").write_text(
        '#include "legacy_value.h"\n',
        encoding="utf-8",
    )
    (nested_include_dir / "legacy_value.h").write_text(
        "#define LEGACY_ANSWER 42\n",
        encoding="utf-8",
    )
    (decoy_include_dir / "legacy_prelude.h").write_text(
        "#define LEGACY_ANSWER 13\n",
        encoding="utf-8",
    )

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-include", "legacy_prelude.h"],
            str(tmp_path),
            "fixture/nested-include-recovery",
        )
    )

    assert parsed_count == 1
    assert [item["name"] for item in payload["functions"]] == [
        "legacy_answer"
    ]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "legacy/src/main.cpp",
            "trigger": "missing_include_diagnostic",
            "added_include_dir_examples": [
                "legacy/vendor/include",
                "legacy/vendor/inc.next",
            ],
            "added_include_dir_count": 2,
            "added_include_dirs_sha256": (
                "5a4a956e5ff09b51e6b2a395d66cb09a9407c0414f8b6b8c563"
                "f0433660d1704"
            ),
            "added_include_dir_examples_truncated": False,
            "requested_include_name_count": 2,
            "requested_include_names_sha256": (
                "9a69b537257773714a90564239bc0d5722c84333c6e26c2ac851"
                "8af299ac3241"
            ),
            "requested_include_name_examples": [
                "legacy_prelude.h",
                "legacy_value.h",
            ],
            "requested_include_name_examples_truncated": False,
            "unresolved_include_name_count": 0,
            "retry_round_count": 2,
            "initial_missing_include_count": 1,
            "status": "recovered",
            "retry_missing_include_count": 0,
        }
    ]


def test_source_with_nul_byte_is_rejected() -> None:
    index_project = _load_indexer()

    with pytest.raises(ValueError, match="source contains NUL byte"):
        index_project._decode_source_bytes(b"int value = 0;\0\n", "binary.cpp")


@pytest.mark.parametrize(
    ("bom", "disk_encoding"),
    (
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
    ),
)
def test_bom_marked_wide_cpp_is_transcoded_losslessly_for_libclang(
    tmp_path: Path,
    bom: bytes,
    disk_encoding: str,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "wide.cpp"
    text = (
        "// BOM-marked source\r\n"
        "int wide_answer(int value) { return value + 42; }\r\n"
    )
    source.write_bytes(bom + text.encode(disk_encoding))

    decoded, detected_encoding = index_project._decode_source_bytes(
        source.read_bytes(),
        str(source),
    )
    parser_text, parser_bytes, parser_encoding = (
        index_project._read_source_file(str(source))
    )
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/wide-source",
        )
    )

    assert detected_encoding == disk_encoding
    assert decoded.encode(disk_encoding) == source.read_bytes()
    assert parser_encoding == "utf-8"
    assert parser_bytes == parser_text.encode("utf-8")
    assert parsed_count == 1
    assert "wide_answer" in {
        function["name"] for function in payload["functions"]
    }


def test_malformed_bom_marked_wide_source_fails_closed() -> None:
    index_project = _load_indexer()

    with pytest.raises(UnicodeDecodeError):
        index_project._decode_source_bytes(
            b"\xff\xfe\x00",
            "malformed-wide.cpp",
        )


def test_textual_nul_inside_comment_or_literal_round_trips() -> None:
    index_project = _load_indexer()
    source = (
        b'char normal[] = "value\\0";\n'
        b'char embedded[] = "value\x00";\n'
        b'char raw[] = R"tag(value\x00)tag";\n'
        b"/* glyph \x81 '\x00' */\n"
    )

    text, encoding = index_project._decode_source_bytes(source, "fixture.cpp")

    assert encoding == "latin-1"
    assert text.encode(encoding) == source


def test_mixed_legacy_source_uses_byte_exact_latin1_fallback() -> None:
    from tools.clang_indexer import index_project

    cp1252_source = b"// compiler\x92s type\n"
    cp1252_text, cp1252_encoding = index_project._decode_source_bytes(
        cp1252_source,
        "cp1252.h",
    )
    assert cp1252_encoding == "cp1252"
    assert cp1252_text.encode(cp1252_encoding) == cp1252_source

    mixed_source = b'// "\x8d\xc5\x8f\x89\x82\xcc\x8ds: \xb1\xb2\xb3"\n'
    mixed_text, mixed_encoding = index_project._decode_source_bytes(
        mixed_source,
        "mixed-shift-jis-font-table.h",
    )
    assert mixed_encoding == "latin-1"
    assert mixed_text.encode(mixed_encoding) == mixed_source


def test_header_macro_emission_preserves_mixed_legacy_bytes(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    header = tmp_path / "mixed.h"
    raw = b'#define MIXED "\x8d"\n'
    header.write_bytes(raw)
    docs: list[dict] = []

    stats = index_project.emit_header_documents(
        index=index_project.ProjectIndex(),
        header_files=[str(header)],
        project_dir=str(tmp_path),
        project_id="fixture/mixed-header",
        compile_db=None,
        default_args=[],
        default_build_info=None,
        max_tokens=4096,
        enriched=True,
        chunk_claims=None,
        emit_doc=docs.append,
    )

    assert stats["header_macro"] == 1
    assert len(docs) == 1
    assert docs[0]["text"].encode("latin-1") == raw
    assert "\ufffd" not in docs[0]["text"]


def test_macro_registry_compacts_shared_definitions_without_losing_root_views(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    (tmp_path / "shared.h").write_text(
        "#define VALUE 1\n"
        "#undef VALUE\n"
        "#define VALUE 2\n",
        encoding="utf-8",
    )
    roots = []
    for index, stem in enumerate(("alpha", "beta")):
        root = tmp_path / f"{stem}.cpp"
        root.write_text(
            ("// beta prefix one\n// beta prefix two\n" if index else "")
            + '#include "shared.h"\n'
            + f"int {stem}() {{ return VALUE; }}\n",
            encoding="utf-8",
        )
        roots.append(str(root))

    index = index_project.ProjectIndex()
    stats = index_project.register_header_macros(
        index,
        roots,
        project_dir=str(tmp_path),
        project_id="fixture/shared-macro-views",
        macro_usage_texts_by_file={
            "alpha.cpp": [("VALUE", 2)],
            "beta.cpp": [("VALUE", 4)],
        },
        max_retained_macros=2,
    )

    assert stats["registered_macros"] == 4
    assert stats["canonical_macro_definitions"] == 2
    assert stats["macro_visibility_records"] == 4
    assert stats["compacted_macro_occurrences"] == 2
    assert 0 < stats["macro_visibility_bytes"]
    assert (
        stats["macro_visibility_bytes"]
        <= stats["macro_visibility_byte_limit"]
    )
    assert len(index.macro_definitions) == 2

    alpha = index_project._select_visible_macro(
        index,
        "VALUE",
        target_file="alpha.cpp",
        max_line=2,
    )
    beta = index_project._select_visible_macro(
        index,
        "VALUE",
        target_file="beta.cpp",
        max_line=4,
    )
    assert alpha is not None
    assert beta is not None
    assert alpha.text == beta.text == "#undef VALUE\n#define VALUE 2\n"
    assert alpha.visible_in_file == "alpha.cpp"
    assert beta.visible_in_file == "beta.cpp"
    assert alpha.visible_line == 1
    assert beta.visible_line == 3
    assert alpha.sequence == 1
    assert beta.sequence == 3
    assert alpha.previous is not None
    assert beta.previous is not None
    assert alpha.previous.text == beta.previous.text == "#define VALUE 1\n"
    assert alpha.previous.visible_in_file == "alpha.cpp"
    assert beta.previous.visible_in_file == "beta.cpp"
    assert alpha.previous.sequence == 0
    assert beta.previous.sequence == 2
    assert [
        (macro.visible_in_file, macro.sequence)
        for macro in index_project._used_macro_defs(
            index,
            [("return VALUE;", 4)],
            target_file="beta.cpp",
            max_line=4,
        )
    ] == [("beta.cpp", 2), ("beta.cpp", 3)]


def test_macro_registry_bound_counts_distinct_source_definitions(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    roots = []
    for stem in ("one", "two"):
        (tmp_path / f"{stem}.h").write_text(
            "#define VALUE 1\n",
            encoding="utf-8",
        )
        root = tmp_path / f"{stem}.cpp"
        root.write_text(
            f'#include "{stem}.h"\n'
            f"int {stem}() {{ return VALUE; }}\n",
            encoding="utf-8",
        )
        roots.append(str(root))

    with pytest.raises(
        MemoryError,
        match=r"canonical definition registry bound: .*limit=1",
    ):
        index_project.register_header_macros(
            index_project.ProjectIndex(),
            roots,
            project_dir=str(tmp_path),
            project_id="fixture/distinct-macro-definitions",
            macro_usage_texts_by_file={
                "one.cpp": [("VALUE", 2)],
                "two.cpp": [("VALUE", 2)],
            },
            max_retained_macros=1,
        )


def test_macro_registry_visibility_byte_bound_fails_loud(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    root = tmp_path / "root.cpp"
    root.write_text(
        "#define VALUE 1\n"
        "int root() { return VALUE; }\n",
        encoding="utf-8",
    )

    with pytest.raises(
        MemoryError,
        match=r"root visibility byte bound: .*limit=1",
    ):
        index_project.register_header_macros(
            index_project.ProjectIndex(),
            [str(root)],
            project_dir=str(tmp_path),
            project_id="fixture/macro-visibility-byte-bound",
            macro_usage_texts_by_file={"root.cpp": [("VALUE", 2)]},
            max_retained_macros=2,
            max_macro_visibility_bytes=1,
        )


def test_parse_pool_emits_heartbeat_while_a_batch_is_still_running(
    capsys,
) -> None:
    from tools.clang_indexer import index_project

    class DelayedExecutor:
        def __init__(self) -> None:
            self.future = Future()
            self.timer = Timer(
                0.05,
                self.future.set_result,
                args=[("batch-result", 1)],
            )

        def submit(self, _fn, _batch):
            self.timer.start()
            return self.future

    executor = DelayedExecutor()
    results = list(
        index_project._iter_parse_batch_results(
            executor,
            ["slow-batch"],
            max_in_flight=1,
            heartbeat_interval_s=0.01,
        )
    )
    executor.timer.join()

    assert results == [("batch-result", 1)]
    assert "Parse pool heartbeat:" in capsys.readouterr().err
