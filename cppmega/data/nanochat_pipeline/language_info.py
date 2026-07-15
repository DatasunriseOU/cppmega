"""Language / dialect metadata helpers for enriched code documents."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


CPP_FEATURE_STD_HINTS: tuple[tuple[str, str], ...] = (
    ("__cpp_variadic_templates", "c++11"),
    ("__cpp_generic_lambdas", "c++14"),
    ("__cpp_if_constexpr", "c++17"),
    ("__cpp_inline_variables", "c++17"),
    ("__cpp_concepts", "c++20"),
    ("__cpp_modules", "c++20"),
    ("__cpp_coroutines", "c++20"),
    ("__cpp_consteval", "c++20"),
    ("__cpp_constexpr_dynamic_alloc", "c++20"),
    ("__cpp_lib_ranges", "c++20"),
    ("__cpp_lib_format", "c++20"),
    ("__cpp_if_consteval", "c++23"),
    ("__cpp_static_call_operator", "c++23"),
    ("__cpp_multidimensional_subscript", "c++23"),
    ("__cpp_deducing_this", "c++23"),
    ("__cpp_lib_expected", "c++23"),
    ("__cpp_lib_print", "c++23"),
)

CPP_STANDARD_HEADERS: dict[str, str] = {
    "thread": "c++11",
    "mutex": "c++11",
    "atomic": "c++11",
    "condition_variable": "c++11",
    "future": "c++11",
    "chrono": "c++11",
    "regex": "c++11",
    "random": "c++11",
    "array": "c++11",
    "unordered_map": "c++11",
    "unordered_set": "c++11",
    "tuple": "c++11",
    "type_traits": "c++11",
    "initializer_list": "c++11",
    "filesystem": "c++17",
    "optional": "c++17",
    "variant": "c++17",
    "any": "c++17",
    "string_view": "c++17",
    "charconv": "c++17",
    "execution": "c++17",
    "memory_resource": "c++17",
    "ranges": "c++20",
    "concepts": "c++20",
    "coroutine": "c++20",
    "span": "c++20",
    "bit": "c++20",
    "numbers": "c++20",
    "barrier": "c++20",
    "latch": "c++20",
    "semaphore": "c++20",
    "source_location": "c++20",
    "syncstream": "c++20",
    "stop_token": "c++20",
    "jthread": "c++20",
    "format": "c++20",
    "print": "c++23",
    "expected": "c++23",
    "stacktrace": "c++23",
    "generator": "c++23",
    "mdspan": "c++23",
    "flat_map": "c++23",
    "flat_set": "c++23",
}

STD_RANK: dict[str, int] = {
    "c89": 1,
    "c90": 1,
    "c95": 2,
    "c99": 3,
    "c11": 4,
    "c17": 5,
    "c23": 6,
    "c++98": 1,
    "c++11": 2,
    "c++14": 3,
    "c++17": 4,
    "c++20": 5,
    "c++23": 6,
    "c++26": 7,
    "cl1.0": 1,
    "cl1.1": 2,
    "cl1.2": 3,
    "cl2.0": 4,
    "cl3.0": 5,
    "clc++2021": 6,
}

CPP_DRAFT_STD_MAP: dict[str, str] = {
    "98": "c++98",
    "03": "c++98",
    "11": "c++11",
    "14": "c++14",
    "17": "c++17",
    "20": "c++20",
    "23": "c++23",
    "26": "c++26",
    "0x": "c++11",
    "1y": "c++14",
    "1z": "c++17",
    "2a": "c++20",
    "2b": "c++23",
    "2c": "c++26",
    "23preview": "c++23",
    "preview": "c++23",
    "latest": "c++26",
}

C_DRAFT_STD_MAP: dict[str, str] = {
    "89": "c89",
    "90": "c90",
    "95": "c95",
    "99": "c99",
    "9x": "c99",
    "11": "c11",
    "17": "c17",
    "18": "c17",
    "23": "c23",
    "2x": "c23",
    "2y": "c23",
    "latest": "c23",
}

OPENCL_STD_MAP: dict[str, str] = {
    "cl1.0": "cl1.0",
    "cl1.1": "cl1.1",
    "cl1.2": "cl1.2",
    "cl2.0": "cl2.0",
    "cl3.0": "cl3.0",
    "clc++2021": "clc++2021",
}

SQL_DIALECT_HINTS: tuple[tuple[str, str], ...] = (
    ("sqlite3_", "sqlite"),
    ("mysql_", "mysql"),
    ("mysql_real_query", "mysql"),
    ("mariadb", "mysql"),
    ("postgresql", "postgresql"),
    ("postgres", "postgresql"),
    ("pgsql", "postgresql"),
    ("libpq", "postgresql"),
    ("pqexec", "postgresql"),
    ("sp_executesql", "tsql"),
    ("nvarchar", "tsql"),
    ("sqlstate", "db2"),
    ("varchar_format", "db2"),
    ("dbms_", "plsql"),
    ("elsif", "plsql"),
)

SQL_KEYWORDS: tuple[str, ...] = (
    "select ",
    "insert ",
    "update ",
    "delete ",
    "create table",
    "drop table",
    "alter table",
    "from ",
    "where ",
    "join ",
    "group by",
    "order by",
    "having ",
    "union ",
    "with recursive",
)

EMBEDDED_SQL_HINTS: tuple[str, ...] = (
    "sqlpp::",
    "sqlpp11",
    "soci::",
    "odb::",
    "pqxx::",
    "sqlite3_",
    "mysql_",
    "pqexec",
    "exec sql",
    'sql"',
    'r"sql(',
    'r"sql[',
)


def _unique_sorted(values):
    return sorted(set(values))


def _best_std_label(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if _std_family(candidate) != _std_family(current):
        return current
    if STD_RANK.get(candidate, 0) > STD_RANK.get(current, 0):
        return candidate
    return current


def _trim_leading_noise(text: str) -> str:
    s = text.lstrip()
    while s:
        stripped = s.lstrip()
        if stripped.startswith("/*"):
            end = stripped.find("*/")
            if end != -1:
                s = stripped[end + 2 :]
                continue
        if stripped.startswith("//"):
            lines = stripped.splitlines()
            idx = 0
            while idx < len(lines):
                line = lines[idx].lstrip()
                if line.startswith("//") or not line:
                    idx += 1
                    continue
                break
            s = "\n".join(lines[idx:])
            continue
        break
    return s.strip() or text


def _std_family(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("c++"):
        return "c++"
    if value.startswith("cl"):
        return "opencl"
    if value.startswith("c"):
        return "c"
    return None


def _normalize_cpp_standard(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().strip()
    if normalized.startswith("gnu++"):
        suffix = normalized.removeprefix("gnu++")
        return CPP_DRAFT_STD_MAP.get(suffix)
    if normalized.startswith("c++"):
        suffix = normalized.removeprefix("c++")
        return CPP_DRAFT_STD_MAP.get(suffix)
    return None


def _normalize_c_standard(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().strip()
    if normalized.startswith("gnu"):
        suffix = normalized.removeprefix("gnu")
        return C_DRAFT_STD_MAP.get(suffix)
    if normalized.startswith("c"):
        suffix = normalized.removeprefix("c")
        return C_DRAFT_STD_MAP.get(suffix)
    if normalized.startswith("iso9899:"):
        iso_value = normalized.removeprefix("iso9899:")
        mapping = {
            "1990": "c90",
            "199409": "c95",
            "1999": "c99",
            "199x": "c99",
            "2011": "c11",
            "2017": "c17",
            "2018": "c17",
            "2024": "c23",
        }
        return mapping.get(iso_value)
    return None


def _normalize_opencl_standard(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().strip()
    if normalized in OPENCL_STD_MAP:
        return OPENCL_STD_MAP[normalized]
    if normalized.startswith("opencl"):
        suffix = normalized.removeprefix("opencl")
        if suffix and suffix[0].isdigit():
            return OPENCL_STD_MAP.get(f"cl{suffix}")
    return None


def _normalize_standard_value(value: str | None) -> tuple[str | None, str | None]:
    cpp_std = _normalize_cpp_standard(value)
    if cpp_std:
        return "c++", cpp_std
    c_std = _normalize_c_standard(value)
    if c_std:
        return "c", c_std
    opencl_std = _normalize_opencl_standard(value)
    if opencl_std:
        return "opencl", opencl_std
    return None, None


def _normalize_compile_language(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().strip()
    if normalized in {"c", "c-header", "cpp-output"}:
        return "c"
    if normalized in {
        "c++",
        "c++-header",
        "c++-cpp-output",
        "c++-module",
        "c++-system-header",
    }:
        return "c++"
    if "cuda" in normalized:
        return "cuda"
    if "hip" in normalized:
        return "hip"
    if normalized in {"cl", "opencl", "opencl-c", "clcpp", "openclcpp"}:
        return "opencl"
    return None


def _extract_compile_context(
    compile_args: Sequence[str] | None,
    build_info: dict | None,
):
    if not compile_args and not build_info:
        return None, None, [], [], None

    primary_language = None
    primary_standard = None
    signals: list[str] = []
    detector_sources: list[str] = []
    provenance: dict[str, str] = {}

    build_system = None
    compiler = None
    authoritative_build = False
    if build_info:
        raw_build_system = build_info.get("build_system")
        if isinstance(raw_build_system, str) and raw_build_system:
            build_system = raw_build_system
            authoritative_build = raw_build_system != "default"
            if authoritative_build:
                provenance["build_system"] = raw_build_system
                detector_sources.append("build_system")
                signals.append(f"build_system:{raw_build_system}")
        raw_compiler = build_info.get("compiler")
        if authoritative_build and isinstance(raw_compiler, str) and raw_compiler:
            compiler = Path(raw_compiler).name.lower()
            provenance["compiler"] = compiler
            signals.append(f"compiler:{compiler}")
            detector_sources.append("build_system")
            if compiler == "nvcc":
                primary_language = "cuda"
            elif compiler == "hipcc":
                primary_language = "hip"
        raw_standard = build_info.get("standard")
        if authoritative_build and isinstance(raw_standard, str):
            build_lang, build_std = _normalize_standard_value(raw_standard)
            if build_std:
                primary_language = primary_language or build_lang
                primary_standard = build_std
                provenance["build_standard"] = raw_standard
                detector_sources.append("build_system")
                signals.append(f"build_standard:{raw_standard.lower()}")

    args = [str(arg) for arg in compile_args or []]
    i = 0
    while i < len(args):
        arg = args[i]
        lower = arg.lower()

        x_value = None
        x_flag = None
        if arg == "-x" and i + 1 < len(args):
            x_value = args[i + 1]
            x_flag = f"-x {x_value}"
            i += 1
        elif lower.startswith("-x") and len(arg) > 2:
            x_value = arg[2:]
            x_flag = arg
        if x_value is not None:
            normalized_lang = _normalize_compile_language(x_value)
            if normalized_lang:
                primary_language = normalized_lang
                provenance["language_flag"] = x_flag or arg
                detector_sources.append("compile_args")
                signals.append(f"compile_lang:{x_value.lower()}")

        std_value = None
        if lower.startswith("-std="):
            std_value = arg.split("=", 1)[1]
        elif lower.startswith("/std:"):
            std_value = arg.split(":", 1)[1]
        elif lower.startswith("-cl-std="):
            std_value = arg.split("=", 1)[1]
        if std_value is not None:
            std_lang, normalized_std = _normalize_standard_value(std_value)
            if normalized_std:
                primary_language = primary_language or std_lang
                primary_standard = _best_std_label(primary_standard, normalized_std)
                provenance["standard_flag"] = arg
                detector_sources.append("compile_args")
                signals.append(f"compile_std:{std_value.lower()}")
                if _std_family(normalized_std) == "opencl":
                    primary_language = "opencl"

        if lower in {"/tc", "/tc".lower()}:
            primary_language = "c"
            provenance["language_flag"] = arg
            detector_sources.append("compile_args")
            signals.append("compile_lang:msvc_c")
        elif lower in {"/tp", "/tp".lower()}:
            primary_language = "c++"
            provenance["language_flag"] = arg
            detector_sources.append("compile_args")
            signals.append("compile_lang:msvc_cpp")
        elif lower.startswith("--cuda-gpu-arch") or lower.startswith("--generate-code") or lower.startswith("-gencode"):
            primary_language = primary_language or "cuda"
            detector_sources.append("compile_args")
            signals.append("compile_cuda_flag")
        elif lower == "--hip-link" or lower.startswith("--hip-path") or lower.startswith("--rocm-path"):
            primary_language = primary_language or "hip"
            detector_sources.append("compile_args")
            signals.append("compile_hip_flag")
        elif lower.startswith("--offload-arch") and compiler == "hipcc":
            primary_language = primary_language or "hip"
            detector_sources.append("compile_args")
            signals.append("compile_offload_arch")

        i += 1

    source_name = None
    if authoritative_build and build_system:
        source_name = build_system
    elif args:
        source_name = "compile_args"
    if source_name:
        provenance["source"] = source_name

    return (
        primary_language,
        primary_standard,
        _unique_sorted(signals),
        _unique_sorted(detector_sources),
        provenance or None,
    )


def _detect_primary_from_extension(filepath: str | None):
    if not filepath:
        return None
    ext = Path(filepath).suffix.lower()
    if not ext:
        return None
    mapping = {
        ".c": ("c", None),
        ".h": ("c_or_cpp_header", None),
        ".cpp": ("c++", None),
        ".cc": ("c++", None),
        ".cxx": ("c++", None),
        ".c++": ("c++", None),
        ".cp": ("c++", None),
        ".hpp": ("c++", None),
        ".hh": ("c++", None),
        ".hxx": ("c++", None),
        ".h++": ("c++", None),
        ".inl": ("c++", None),
        ".ipp": ("c++", None),
        ".tpp": ("c++", None),
        ".cu": ("cuda", None),
        ".cuh": ("cuda", None),
        ".hip": ("hip", None),
        ".cl": ("opencl", None),
        ".sql": ("sql", None),
        ".ddl": ("sql", None),
        ".dml": ("sql", None),
        ".psql": ("sql", None),
    }
    result = mapping.get(ext)
    if result is None:
        return None
    language, standard = result
    return language, standard, [f"path_ext:{ext}"]


def _detect_sql_dialect(source_lower: str):
    signals = []
    dialect = None
    for needle, label in SQL_DIALECT_HINTS:
        if needle in source_lower:
            signals.append(f"sql_hint:{needle}")
            dialect = dialect or label
    keyword_hits = sum(1 for needle in SQL_KEYWORDS if needle in source_lower)
    if keyword_hits >= 2:
        signals.append(f"sql_keywords:{keyword_hits}")
    return dialect, signals


def _detect_embedded_languages(source_lower: str, platform_info: dict | None):
    embedded = []
    if any(needle in source_lower for needle in EMBEDDED_SQL_HINTS):
        embedded.append("sql")
    if platform_info:
        for label in ("cuda", "hip", "opencl", "sycl"):
            if label in platform_info.get("gpu", []):
                embedded.append(label)
    return _unique_sorted(embedded)


def _detect_c_standard(source_lower: str):
    if "__stdc_version__" not in source_lower:
        return None
    mapping = (
        ("202311l", "c23"),
        ("201710l", "c17"),
        ("201112l", "c11"),
        ("199901l", "c99"),
        ("199409l", "c95"),
    )
    tail = source_lower[source_lower.find("__stdc_version__") :]
    for needle, label in mapping:
        if needle in tail:
            return label
    return None


def _extract_cpp_standard_from_platform(platform_info: dict | None):
    if not platform_info:
        return None
    value = platform_info.get("cpp_std") or platform_info.get("standard")
    return _normalize_cpp_standard(value)


def _extract_include_name(include_line: str) -> str | None:
    if "<" in include_line and ">" in include_line:
        return include_line[include_line.find("<") + 1 : include_line.rfind(">")]
    if '"' in include_line:
        left = include_line.find('"')
        right = include_line.rfind('"')
        if right > left:
            return include_line[left + 1 : right]
    return None


def _detect_cpp_standard(source_lower: str, platform_info: dict | None):
    detected: str | None = _extract_cpp_standard_from_platform(platform_info)
    signals = []

    for macro, feature_label in CPP_FEATURE_STD_HINTS:
        if macro in source_lower:
            next_detected = _best_std_label(detected, feature_label)
            if next_detected != detected:
                signals.append(f"cpp_feature_macro:{macro}")
                detected = next_detected

    for line in source_lower.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#include"):
            continue
        include_name = _extract_include_name(stripped)
        if not include_name:
            continue
        base = include_name.split("/")[-1]
        base_no_ext = base.replace(".h", "").replace(".hpp", "")
        header_label = CPP_STANDARD_HEADERS.get(base_no_ext)
        if header_label:
            next_detected = _best_std_label(detected, header_label)
            if next_detected != detected:
                signals.append(f"cpp_header:{base_no_ext}")
                detected = next_detected

    value_map = (
        (202612, "c++26"),
        (202302, "c++23"),
        (202002, "c++20"),
        (201703, "c++17"),
        (201402, "c++14"),
        (201103, "c++11"),
        (199711, "c++98"),
    )
    for macro_name in ("__cplusplus", "_msvc_lang"):
        pos = source_lower.find(macro_name)
        if pos == -1:
            continue
        tail = source_lower[pos : pos + 160]
        digits = "".join(ch if ch.isdigit() else " " for ch in tail).split()
        for token in digits:
            try:
                numeric = int(token)
            except ValueError:
                continue
            for threshold, std_label in value_map:
                if numeric >= threshold:
                    next_detected = _best_std_label(detected, std_label)
                    if next_detected != detected:
                        signals.append(f"cpp_std_macro:{macro_name}")
                        detected = next_detected
                    break
            break

    return detected, _unique_sorted(signals)


def _looks_like_c_family_code(source_lower: str):
    signals = 0
    for needle in (
        "#include",
        "#define",
        "typedef ",
        "struct ",
        "enum ",
        "static ",
        "extern ",
        "inline ",
    ):
        if needle in source_lower:
            signals += 1
    if "{" in source_lower and "}" in source_lower and ";" in source_lower:
        signals += 1
    if "(" in source_lower and ")" in source_lower and ";" in source_lower:
        signals += 1
    return signals >= 2


def _is_sql_primary_candidate(source_lower: str):
    keyword_hits = sum(1 for needle in SQL_KEYWORDS if needle in source_lower)
    if keyword_hits < 4 or _looks_like_c_family_code(source_lower):
        return False
    if not any(
        marker in source_lower
        for marker in (
            "select ",
            "insert ",
            "update ",
            "delete ",
            "create table",
            "alter table",
            "drop table",
            "exec sql",
            "begin transaction",
        )
    ):
        return False
    return True


def detect_language_info(
    text: str,
    filepath: str | None = None,
    platform_info: dict | None = None,
    compile_args: Sequence[str] | None = None,
    build_info: dict | None = None,
):
    if not text or not text.strip():
        return None

    analysis_text = _trim_leading_noise(text)
    source_lower = analysis_text.lower()
    signals = []
    detector_sources = []
    cpp_standard, cpp_standard_signals = _detect_cpp_standard(source_lower, platform_info)
    (
        compile_language,
        compile_standard,
        compile_signals,
        compile_sources,
        provenance,
    ) = _extract_compile_context(compile_args, build_info)
    signals.extend(compile_signals)
    detector_sources.extend(compile_sources)

    ext_result = _detect_primary_from_extension(filepath)
    if ext_result is not None:
        primary_language, primary_standard, ext_signals = ext_result
        detector_sources.append("path")
        signals.extend(ext_signals)
    else:
        primary_language = "unknown"
        primary_standard = None

    if compile_language and (
        primary_language in {"unknown", "c_or_cpp_header", "c_or_cpp"}
        or primary_language != compile_language
    ):
        primary_language = compile_language
        detector_sources.append("compile_args")
        signals.append("compile_lang_authoritative")

    if compile_standard and primary_standard is None:
        primary_standard = compile_standard
        detector_sources.append("compile_args")
        signals.append("compile_std_authoritative")

    if (
        primary_language == "unknown"
        or primary_language == "c_or_cpp_header"
    ):
        if (
            "__global__" in source_lower
            or "__device__" in source_lower
            or "cudamalloc" in source_lower
            or "cuda_runtime.h" in source_lower
            or "__cudacc__" in source_lower
            or "__cuda_arch__" in source_lower
            or "__cudacc_ver_major__" in source_lower
        ):
            primary_language = "cuda"
            signals.append("cuda_marker")
        elif (
            "hip/hip_runtime.h" in source_lower
            or "hip/hip_runtime_api.h" in source_lower
            or "hiplaunchkernelggl" in source_lower
            or "hipmalloc" in source_lower
            or "__hipcc__" in source_lower
            or "__hip__" in source_lower
            or "__hip_platform_amd__" in source_lower
            or "__hip_platform_nvidia__" in source_lower
            or "__hip_device_compile__" in source_lower
        ):
            primary_language = "hip"
            signals.append("hip_marker")
        elif (
            "__kernel" in source_lower
            or "get_global_id(" in source_lower
            or "cl_mem" in source_lower
            or "__opencl_c_version__" in source_lower
            or "cl_version" in source_lower
            or "#pragma opencl" in source_lower
            or "__global " in source_lower
            or "__local " in source_lower
            or "__constant " in source_lower
        ):
            primary_language = "opencl"
            signals.append("opencl_marker")
        elif (
            cpp_standard is not None
            or "__cplusplus" in source_lower
            or "_msvc_lang" in source_lower
            or "template" in source_lower
            or "namespace " in source_lower
            or "std::" in source_lower
            or "class " in source_lower
            or "constexpr" in source_lower
            or "typename " in source_lower
            or "using " in source_lower
        ):
            primary_language = "c++"
            signals.append("cpp_constructs")
        elif (
            (
                "__stdc_version__" in source_lower
                and "__cplusplus" not in source_lower
                and "_msvc_lang" not in source_lower
            )
            or (
                filepath
                and Path(filepath).suffix.lower() == ".c"
                and (
                    "#include <stdio.h>" in source_lower
                    or "#include <stdlib.h>" in source_lower
                )
            )
            or "typedef struct" in source_lower
        ):
            primary_language = "c"
            signals.append("c_constructs")
        elif _is_sql_primary_candidate(source_lower):
            primary_language = "sql"
            signals.append("sql_constructs")
        detector_sources.append("source")

    sql_dialect, sql_signals = _detect_sql_dialect(source_lower)
    signals.extend(sql_signals)
    embedded_languages = _detect_embedded_languages(source_lower, platform_info)
    if sql_dialect and primary_language != "sql" and "sql" not in embedded_languages:
        embedded_languages.append("sql")
        embedded_languages = _unique_sorted(embedded_languages)

    primary_dialect = sql_dialect if primary_language == "sql" else None

    if primary_language == "c" and primary_standard is None:
        primary_standard = _detect_c_standard(source_lower)
        if primary_standard:
            detector_sources.append("stdc_version")
            signals.append("c_std_macro")

    if primary_language in {"c++", "cuda", "hip"} and primary_standard is None and platform_info:
        primary_standard = _extract_cpp_standard_from_platform(platform_info)
        if primary_standard:
            detector_sources.append("platform")
            signals.append("cpp_std_platform")

    if primary_language in {"c++", "cuda", "hip"}:
        if cpp_standard and primary_standard is None:
            primary_standard = cpp_standard
        if cpp_standard_signals:
            detector_sources.append("cpp_std_heuristic")
            signals.extend(cpp_standard_signals)

    if primary_language == "unknown" and _looks_like_c_family_code(source_lower):
        primary_language = "c_or_cpp"
        detector_sources.append("fallback")
        signals.append("c_family_shape")

    if primary_language == "unknown":
        return None

    confidence = None
    if any(source in {"compile_args", "build_system"} for source in detector_sources):
        confidence = "high"
    elif any(s.startswith("path_ext:") for s in signals) and any(
        "marker" in s or "constructs" in s for s in signals
    ):
        confidence = "high"
    elif signals:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "primary_language": primary_language,
        "primary_standard": primary_standard,
        "primary_dialect": primary_dialect,
        "embedded_languages": embedded_languages,
        "signals": _unique_sorted(signals),
        "detector_sources": _unique_sorted(detector_sources),
        "confidence": confidence,
        **({"provenance": provenance} if provenance else {}),
    }


def language_info_to_prefix(language_info: dict | None) -> str:
    if not language_info:
        return ""

    parts = []
    primary = language_info.get("primary_language")
    if primary:
        parts.append(f"primary={primary}")
    standard = language_info.get("primary_standard")
    if standard:
        parts.append(f"standard={standard}")
    dialect = language_info.get("primary_dialect")
    if dialect:
        parts.append(f"dialect={dialect}")
    embedded = language_info.get("embedded_languages") or []
    if embedded:
        parts.append(f"embedded={','.join(sorted(embedded))}")
    confidence = language_info.get("confidence")
    if confidence:
        parts.append(f"confidence={confidence}")

    if not parts:
        return ""
    return f"// language: {' '.join(parts)}\n"
