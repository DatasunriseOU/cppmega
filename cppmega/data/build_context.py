"""Helpers for build-system-aware compile context detection."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shlex
from typing import Callable


CPP_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cp",
        ".cpp",
        ".cxx",
        ".c++",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".h++",
        ".inl",
        ".ipp",
        ".tpp",
        ".cu",
        ".cuh",
        ".hip",
        ".cl",
    }
)

DEFAULT_PLATFORM = {
    "platform": "x86_64-linux-gnu",
    "compiler": "g++",
    "standard": "c++17",
    "arch": "x86_64",
    "mode": "user",
}

_STD_FLAG_RE = re.compile(r"(?P<flag>(?:-std=|--std=|/std:|-cl-std=)(?P<value>[^\s\"']+))")
_TARGET_FLAG_RE = re.compile(r"(?P<flag>(?:--target=|-march=|-mcpu=)(?P<value>[^\s\"']+))")
_LANG_TOKEN_RE = re.compile(r"^(?:-x(?P<joined>.+)|-x)$")
_VAR_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Z_][A-Z0-9_]*)\s*(?:[:+?]?=)\s*(?P<value>.*?)\s*$")
_STRING_LITERAL_RE = re.compile(r"['\"]([^'\"]+)['\"]", re.S)
_CMAKE_STD_RE = re.compile(
    r"CMAKE_(?P<lang>C|CXX)_STANDARD\s+"
    r'(?P<quote>"?)(?P<version>\d+)(?P=quote)(?=[\s)])',
    re.IGNORECASE,
)
_CMAKE_COMPILER_RE = re.compile(
    r"CMAKE_(?P<lang>C|CXX)_COMPILER\s+\"?([^\"\s)]+)\"?",
    re.IGNORECASE,
)
_CMAKE_PROCESSOR_RE = re.compile(
    r"CMAKE_SYSTEM_PROCESSOR\s+\"?([^\"\s)]+)\"?",
    re.IGNORECASE,
)
_MESON_STD_RE = re.compile(r"(?P<kind>cpp|c)_std\s*[:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_MAKEFILE_NAMES = ("GNUmakefile", "Makefile", "makefile")
_BAZEL_BUILD_NAMES = ("BUILD.bazel", "BUILD", "WORKSPACE.bazel", "WORKSPACE", "MODULE.bazel", ".bazelrc")
_GN_BUILD_NAMES = ("BUILD.gn", "args.gn", ".gn")
_SCONS_BUILD_NAMES = ("SConstruct", "SConscript")
_XMAKE_BUILD_NAMES = ("xmake.lua",)
_FLAG_PREFIXES = ("-I", "-D", "-U", "-std=", "--std=", "/std:", "-cl-std=", "--target=", "-march=", "-mcpu=")
_VERBATIM_FLAGS = {"-m32", "-m64", "--hip-link"}
_PATH_VALUE_FLAGS = {
    "-I",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-F",
    "-iframework",
    "-include",
    "-imacros",
    "-isysroot",
    "--sysroot",
    "-resource-dir",
}
_PATH_JOINED_PREFIXES = (
    "-I",
    "-isystem",
    "-iquote",
    "-idirafter",
    "-F",
    "-iframework",
    "-include",
    "-imacros",
)
_PATH_EQ_PREFIXES = ("--sysroot=", "-isysroot=", "-resource-dir=")
_SEPARATE_FLAG_VALUE_OPTIONS = _PATH_VALUE_FLAGS | {
    "-D",
    "-U",
    "-x",
    "-target",
    "--target",
}
_COMPILE_COMMANDS_COMMON_DIRS = (
    "build",
    "build-debug",
    "build-release",
    "cmake-build-debug",
    "cmake-build-release",
    "out",
    "out/build",
)
_COMPILE_COMMANDS_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "third_party",
    "3rdparty",
    "vendor",
    "external",
    "deps",
}
_XCCONFIG_SKIP_DIRS = _COMPILE_COMMANDS_SKIP_DIRS | {
    ".idea",
    ".vscode",
    "__pycache__",
}
_XCCONFIG_RELEVANT_SETTINGS = frozenset(
    {
        "SDKROOT",
        "SUPPORTED_PLATFORMS",
        "GCC_C_LANGUAGE_STANDARD",
        "HEADER_SEARCH_PATHS",
        "USER_HEADER_SEARCH_PATHS",
        "SYSTEM_HEADER_SEARCH_PATHS",
    }
)
_XCCONFIG_HEADER_SETTINGS = frozenset(
    {
        "HEADER_SEARCH_PATHS",
        "USER_HEADER_SEARCH_PATHS",
        "SYSTEM_HEADER_SEARCH_PATHS",
    }
)
_XCCONFIG_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<conditions>(?:\[[^\]\r\n]+\])*)\s*"
    r"(?P<operator>\+=|\?=|=)\s*(?P<value>.*?)\s*$"
)
_XCCONFIG_INCLUDE_RE = re.compile(
    r'^\s*#\s*include(?P<optional>\?)?\s+["<](?P<path>[^">]+)[">]\s*$'
)
_XCCONFIG_VARIABLE_RE = re.compile(
    r"\$\([^)]+\)|\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*"
)
_XCCONFIG_IGNORED_BUILD_PATH_VARIABLES = (
    "$(BUILT_PRODUCTS_DIR)",
    "${BUILT_PRODUCTS_DIR}",
    "$(DERIVED_FILE_DIR)",
    "${DERIVED_FILE_DIR}",
    "$(DERIVED_SOURCES_DIR)",
    "${DERIVED_SOURCES_DIR}",
)
_MAX_XCCONFIG_FILES = 256
_MAX_XCCONFIG_SCAN_DIRS = 4096
_MAX_XCCONFIG_FILE_BYTES = 512 * 1024
_MAX_XCCONFIG_TOTAL_BYTES = 8 * 1024 * 1024
_MAX_XCCONFIG_LOGICAL_LINE_BYTES = 64 * 1024
_MAX_MACOS_SDK_SETTINGS_BYTES = 1024 * 1024


def is_cpp_path(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in CPP_EXTENSIONS


def split_shell_command(command: str | None) -> list[str]:
    if not command:
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


@dataclass(slots=True)
class CompileCommandEntry:
    filepath: str
    compile_args: list[str]
    build_info: dict
    candidates: set[str] = field(default_factory=set)


@dataclass(slots=True)
class CompileCommandsIndex:
    entries: list[CompileCommandEntry]
    by_key: dict[str, CompileCommandEntry]

    def lookup(self, file_path: str) -> tuple[list[str] | None, dict | None]:
        normalized = os.path.normpath(file_path)
        for key in _path_candidates(normalized):
            entry = self.by_key.get(key)
            if entry is not None:
                return list(entry.compile_args), dict(entry.build_info)
        for entry in self.entries:
            if normalized == entry.filepath or normalized.endswith(entry.filepath):
                return list(entry.compile_args), dict(entry.build_info)
            for candidate in entry.candidates:
                if normalized.endswith(candidate):
                    return list(entry.compile_args), dict(entry.build_info)
        return None, None


@dataclass(slots=True)
class BuildDetection:
    build_system: str
    source: str
    compiler: str | None = None
    standard: str | None = None
    language: str | None = None
    flags: list[str] = field(default_factory=list)
    arch: str | None = None


class BuildContextEvidenceError(RuntimeError):
    """Raised when build evidence exists but cannot be bound safely."""


@dataclass(frozen=True, slots=True)
class MacOSSDKBinding:
    path: str
    settings_path: str
    settings_sha256: str
    canonical_name: str


@dataclass(frozen=True, slots=True)
class MacOSXCConfigContext:
    sdkroot: str
    c_language_standard: str
    header_search_paths: tuple[str, ...]
    evidence_files: tuple[str, ...]
    sdk: MacOSSDKBinding
    compile_args: tuple[str, ...]

    def platform_info(self) -> dict[str, object]:
        return {
            "platform": "macosx",
            "compiler": "clang",
            "standard": "c23",
            "mode": "user",
            "build_system": "xcconfig",
            "source": "xcconfig",
            "sdkroot": self.sdkroot,
            "macos_sdk_path": self.sdk.path,
            "macos_sdk_settings_sha256": self.sdk.settings_sha256,
            "xcconfig_evidence_files": list(self.evidence_files),
            "xcconfig_header_search_paths": list(self.header_search_paths),
        }


@dataclass(frozen=True, slots=True)
class _XCConfigAssignment:
    name: str
    conditions: str
    value: str
    source: Path


def _path_candidates(path: str, directory: str | None = None) -> set[str]:
    candidates: set[str] = set()
    if not path:
        return candidates
    normalized = os.path.normpath(path)
    candidates.add(normalized)
    candidates.add(os.path.basename(normalized))
    if directory and not os.path.isabs(path):
        joined = os.path.normpath(os.path.join(directory, path))
        candidates.add(joined)
        candidates.add(os.path.basename(joined))
    return candidates


def _make_absolute_if_relative(path_text: str, directory: str | None) -> str:
    if not path_text or directory is None or os.path.isabs(path_text):
        return path_text
    return os.path.normpath(os.path.join(directory, path_text))


def _absolutize_compile_arg_paths(flags: list[str], directory: str | None) -> list[str]:
    if directory is None:
        return flags
    normalized_dir = os.path.normpath(directory)
    result: list[str] = []
    i = 0
    while i < len(flags):
        arg = flags[i]
        if arg in _PATH_VALUE_FLAGS and i + 1 < len(flags):
            result.append(arg)
            result.append(_make_absolute_if_relative(flags[i + 1], normalized_dir))
            i += 2
            continue
        eq_prefix = next((prefix for prefix in _PATH_EQ_PREFIXES if arg.startswith(prefix)), None)
        if eq_prefix is not None:
            result.append(eq_prefix + _make_absolute_if_relative(arg[len(eq_prefix) :], normalized_dir))
            i += 1
            continue
        joined_prefix = next(
            (
                prefix
                for prefix in _PATH_JOINED_PREFIXES
                if arg.startswith(prefix) and arg != prefix and arg[len(prefix) :]
            ),
            None,
        )
        if joined_prefix is not None:
            result.append(
                joined_prefix
                + _make_absolute_if_relative(arg[len(joined_prefix) :], normalized_dir)
            )
            i += 1
            continue
        result.append(arg)
        i += 1
    return result


def _sanitize_compile_args(
    argv: list[str],
    filepath: str,
    directory: str | None = None,
) -> tuple[list[str], str | None]:
    compiler = argv[0] if argv else None
    normalized_file = os.path.normpath(filepath)
    flags: list[str] = []
    skip_next = False
    for arg in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-o", "-MF", "-MQ", "-MT", "/Fo", "/Fd", "/Fi"}:
            skip_next = True
            continue
        if arg in {"-c", "-S"}:
            continue
        normalized_arg = os.path.normpath(arg)
        arg_candidates = {normalized_arg}
        if directory and not os.path.isabs(arg):
            arg_candidates.add(os.path.normpath(os.path.join(directory, arg)))
        if (
            normalized_file in arg_candidates
            or normalized_file.endswith(os.sep + normalized_arg)
        ):
            continue
        if arg.endswith((".o", ".obj", ".pcm")):
            continue
        flags.append(arg)
    return _absolutize_compile_arg_paths(flags, directory), compiler


def parse_compile_commands_entries(entries: list[dict]) -> CompileCommandsIndex:
    normalized_entries: list[CompileCommandEntry] = []
    by_key: dict[str, CompileCommandEntry] = {}

    for raw in entries:
        filepath = raw.get("file", "")
        directory = raw.get("directory", "") or None
        args = raw.get("arguments")
        if isinstance(args, list) and args:
            argv = [str(arg) for arg in args]
        else:
            argv = split_shell_command(raw.get("command", ""))
        if not filepath or not argv:
            continue
        candidates = _path_candidates(filepath, directory)
        preferred = os.path.normpath(os.path.join(directory, filepath)) if directory and not os.path.isabs(filepath) else os.path.normpath(filepath)
        compile_args, compiler = _sanitize_compile_args(argv, preferred, directory)
        entry = CompileCommandEntry(
            filepath=preferred,
            compile_args=compile_args,
            build_info={
                "build_system": "compile_commands",
                "source": "compile_commands",
                **({"compiler": compiler} if compiler else {}),
            },
            candidates=candidates,
        )
        normalized_entries.append(entry)
        for candidate in candidates:
            by_key.setdefault(candidate, entry)

    return CompileCommandsIndex(entries=normalized_entries, by_key=by_key)


def load_compile_commands_text(text: str | None) -> CompileCommandsIndex | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    index = parse_compile_commands_entries(data)
    return index if index.entries else None


def load_compile_commands_file(path: str) -> CompileCommandsIndex | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return load_compile_commands_text(f.read())


def _score_compile_commands_file(path: str) -> int:
    index = load_compile_commands_file(path)
    return len(index.entries) if index is not None else 0


def find_compile_commands_file(
    repo_dir: str,
    explicit_path: str | None = None,
) -> str | None:
    """Find the best compile_commands.json for a repo checkout.

    CMake usually writes the compilation database under a build directory, not
    the source root.  Prefer explicit/root files, then common build dirs, then a
    shallow repo scan.  The chosen file must parse and contain at least one
    entry.
    """
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("NANOCHAT_COMPILE_COMMANDS") or os.environ.get(
        "CPPMEGA_COMPILE_COMMANDS"
    )
    if env_path:
        candidates.append(env_path)

    root = os.path.abspath(repo_dir)
    candidates.append(os.path.join(root, "compile_commands.json"))
    for dirname in _COMPILE_COMMANDS_COMMON_DIRS:
        candidates.append(os.path.join(root, dirname, "compile_commands.json"))
    for child in os.listdir(root) if os.path.isdir(root) else []:
        if child.startswith(("build", "cmake-build", "out")):
            candidates.append(os.path.join(root, child, "compile_commands.json"))

    seen: set[str] = set()
    valid: list[tuple[int, str]] = []
    for raw in candidates:
        path = os.path.abspath(raw)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        score = _score_compile_commands_file(path)
        if score > 0:
            if path == os.path.join(root, "compile_commands.json") or raw == explicit_path:
                return path
            valid.append((score, path))

    if valid:
        valid.sort(key=lambda item: item[0], reverse=True)
        return valid[0][1]

    best_score = 0
    best_path: str | None = None
    for current_root, dirs, files in os.walk(root):
        rel = os.path.relpath(current_root, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= 4:
            dirs.clear()
        else:
            dirs[:] = [d for d in dirs if d not in _COMPILE_COMMANDS_SKIP_DIRS]
        if "compile_commands.json" not in files:
            continue
        path = os.path.join(current_root, "compile_commands.json")
        if path in seen:
            continue
        seen.add(path)
        score = _score_compile_commands_file(path)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _extract_string_literals(text: str) -> list[str]:
    return [match.group(1) for match in _STRING_LITERAL_RE.finditer(text)]


def _extract_flag_tokens(tokens: list[str]) -> list[str]:
    selected: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        stripped = token.strip().rstrip(",)")
        if stripped in _SEPARATE_FLAG_VALUE_OPTIONS:
            selected.append(stripped)
            if i + 1 < len(tokens):
                operand = tokens[i + 1].strip().rstrip(",)")
                if operand:
                    selected.append(operand)
                    i += 1
        elif (
            stripped in _VERBATIM_FLAGS
            or stripped.startswith(_FLAG_PREFIXES)
            or stripped.startswith(_PATH_JOINED_PREFIXES)
            or stripped.startswith(_PATH_EQ_PREFIXES)
        ):
            selected.append(stripped)
        elif stripped.startswith("-x") and len(stripped) > 2:
            selected.append(stripped)
        i += 1
    return selected


def _normalize_standard(raw_value: str | None) -> str | None:
    if not isinstance(raw_value, str):
        return None
    value = raw_value.strip().lower()
    if value.startswith("iso9899:"):
        return {
            "iso9899:1990": "c90",
            "iso9899:199409": "c95",
            "iso9899:1999": "c99",
            "iso9899:199x": "c99",
            "iso9899:2011": "c11",
            "iso9899:201x": "c11",
            "iso9899:2017": "c17",
            "iso9899:2018": "c17",
        }.get(value)
    if value.startswith("gnu++"):
        return f"c++{value.removeprefix('gnu++')}"
    if value.startswith("c++"):
        return value
    if value.startswith("gnu"):
        return f"c{value.removeprefix('gnu')}"
    if value.startswith("c"):
        return value
    if value.startswith("cl"):
        return value
    if value.startswith("cxx"):
        return f"c++{value.removeprefix('cxx')}"
    return None


def _extract_standard_from_flags(flags: list[str]) -> tuple[str | None, str | None]:
    for flag in flags:
        match = _STD_FLAG_RE.match(flag)
        if match:
            return _normalize_standard(match.group("value")), match.group("flag")
    return None, None


def _infer_arch_from_flags(flags: list[str]) -> str | None:
    for flag in flags:
        if flag == "-m32":
            return "i686"
        if flag == "-m64" or flag == "-march=x86-64":
            return "x86_64"
        lowered = flag.lower()
        if "aarch64" in lowered or "arm64" in lowered:
            return "aarch64"
        if "riscv64" in lowered:
            return "riscv64"
    return None


def _arch_to_platform(arch: str | None) -> str | None:
    if arch == "i686":
        return "i686-linux-gnu"
    if arch == "x86_64":
        return "x86_64-linux-gnu"
    if arch == "aarch64":
        return "aarch64-linux-gnu"
    if arch == "riscv64":
        return "riscv64-linux-gnu"
    return None


def _infer_language(standard: str | None, compiler: str | None, flags: list[str], default: str | None = None) -> str | None:
    compiler_name = Path(compiler or "").name.lower()
    if compiler_name == "nvcc":
        return "cuda"
    if compiler_name == "hipcc":
        return "hip"
    for idx, flag in enumerate(flags):
        lowered = flag.lower()
        match = _LANG_TOKEN_RE.match(lowered)
        if match:
            joined = match.group("joined")
            if joined is not None:
                if joined == "cuda":
                    return "cuda"
                if joined == "hip":
                    return "hip"
                if joined in {"cl", "opencl", "opencl-c"}:
                    return "opencl"
                if joined in {"c", "c-header"}:
                    return "c"
                if joined.startswith("c++"):
                    return "c++"
        if lowered == "-x" and idx + 1 < len(flags):
            next_flag = flags[idx + 1].lower()
            if next_flag in {"cuda", "hip", "cl", "opencl", "opencl-c", "c"}:
                return {"cl": "opencl", "opencl": "opencl", "opencl-c": "opencl"}.get(next_flag, next_flag)
            if next_flag.startswith("c++"):
                return "c++"
        if lowered in {"/tc"}:
            return "c"
        if lowered in {"/tp"}:
            return "c++"
        if lowered.startswith("--cuda-gpu-arch") or lowered.startswith("--generate-code") or lowered.startswith("-gencode"):
            return "cuda"
        if lowered.startswith("--offload-arch") or lowered.startswith("--hip-path"):
            return "hip"
    if standard:
        if standard.startswith("c++"):
            return "c++"
        if standard.startswith("cl"):
            return "opencl"
        if standard.startswith("c"):
            return "c"
    return default


def _fallback_compile_args(detection: BuildDetection) -> list[str]:
    args: list[str] = []
    flags = _extract_flag_tokens(detection.flags)
    language = _infer_language(detection.standard, detection.compiler, flags, default=detection.language) or "c++"
    if not any(flag == "-x" or flag.startswith("-x") or flag.lower() in {"/tc", "/tp"} for flag in flags):
        if language == "opencl":
            args.extend(["-x", "cl"])
        else:
            args.extend(["-x", language])
    std_flag = _extract_standard_from_flags(flags)[1]
    if std_flag is None and detection.standard:
        normalized = detection.standard
        if normalized.startswith("c++"):
            std_flag = f"-std={normalized}"
        elif normalized.startswith("c"):
            std_flag = f"-std={normalized}"
        elif normalized.startswith("cl"):
            std_flag = f"-cl-std={normalized}"
    if std_flag:
        args.append(std_flag)
    args.extend(flags)
    args.extend(["-fsyntax-only", "-Wno-everything"])
    return args


def _detection_to_context(detection: BuildDetection) -> tuple[dict, list[str]]:
    result = dict(DEFAULT_PLATFORM)
    compiler = detection.compiler
    if compiler:
        compiler_name = Path(compiler).name
        result["compiler"] = compiler_name
    standard = detection.standard or _extract_standard_from_flags(_extract_flag_tokens(detection.flags))[0]
    if standard:
        result["standard"] = standard
    else:
        # DEFAULT_PLATFORM describes the no-build-files fallback only.  Once a
        # real build system is detected, do not claim its dialect is C++17
        # unless the project supplied evidence for that standard.
        result.pop("standard", None)
    arch = detection.arch or _infer_arch_from_flags(_extract_flag_tokens(detection.flags))
    if arch:
        result["arch"] = arch
        platform = _arch_to_platform(arch)
        if platform:
            result["platform"] = platform
    result["build_system"] = detection.build_system
    result["source"] = detection.source
    compile_args = _fallback_compile_args(detection)
    return result, compile_args


def _build_detection_from_compile_commands(text: str | None) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    index = load_compile_commands_text(text)
    if index is None or not index.entries:
        return {}, [], None
    first = index.entries[0]
    flags = list(first.compile_args)
    standard = _extract_standard_from_flags(flags)[0]
    context, compile_args = _detection_to_context(
        BuildDetection(
            build_system="compile_commands",
            source="compile_commands",
            compiler=first.build_info.get("compiler"),
            standard=standard,
            flags=flags,
        )
    )
    return context, compile_args or flags, index


def _parse_cmake(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    compiler = None
    cxx_standard = None
    c_standard = None
    for match in _CMAKE_STD_RE.finditer(text):
        lang = match.group("lang").upper()
        normalized = _normalize_standard(
            f"{'c++' if lang == 'CXX' else 'c'}{match.group('version')}"
        )
        if lang == "CXX" and normalized:
            cxx_standard = normalized
        elif lang == "C" and normalized:
            c_standard = normalized
    for match in _CMAKE_COMPILER_RE.finditer(text):
        lang = match.group("lang").upper()
        candidate = match.group(2)
        if lang == "CXX":
            compiler = candidate
        elif compiler is None:
            compiler = candidate
    arch = None
    processor_match = _CMAKE_PROCESSOR_RE.search(text)
    if processor_match:
        proc = processor_match.group(1).lower()
        if "arm" in proc or "aarch64" in proc:
            arch = "aarch64"
        elif "riscv" in proc:
            arch = "riscv64"
        elif "86" in proc:
            arch = "x86_64"
    standard = cxx_standard or c_standard
    language = "c++" if cxx_standard or compiler and "++" in compiler else ("c" if c_standard else None)
    if not (flags or compiler or standard or arch):
        return None
    return BuildDetection(
        build_system="cmake",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
        arch=arch,
    )


def _parse_autoconf(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(split_shell_command(" ".join(_extract_string_literals(text))))
    compiler = None
    if "ac_prog_cxx" in text.lower():
        compiler = "g++"
    elif "ac_prog_cc" in text.lower():
        compiler = "gcc"
    standard = _extract_standard_from_flags(flags)[0]
    if standard and standard.startswith("c++"):
        language = "c++"
    elif standard and standard.startswith("cl"):
        language = "opencl"
    elif standard and standard.startswith("c"):
        language = "c"
    elif compiler == "g++":
        language = "c++"
    elif compiler == "gcc":
        language = "c"
    else:
        language = None
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="autoconf",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_meson(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    cpp_standard = None
    c_standard = None
    for match in _MESON_STD_RE.finditer(text):
        kind = match.group("kind").lower()
        normalized = _normalize_standard(match.group(2))
        if kind == "cpp" and normalized:
            cpp_standard = normalized
        elif kind == "c" and normalized:
            c_standard = normalized
    standard = cpp_standard or c_standard
    language = "c++" if cpp_standard else ("c" if c_standard else None)
    if not (flags or standard):
        return None
    return BuildDetection(
        build_system="meson",
        source="build_files",
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_make(text: str) -> BuildDetection | None:
    variables: dict[str, list[str]] = {}
    compiler = None
    language: str | None = None
    recipe_flags: list[str] = []
    for line in text.splitlines():
        match = _VAR_ASSIGN_RE.match(line)
        if match:
            name = match.group("name")
            value = match.group("value")
            if name in {"CC", "CXX"}:
                parts = split_shell_command(value)
                if parts:
                    compiler = parts[0]
            elif name in {"CFLAGS", "CXXFLAGS", "CPPFLAGS"}:
                variables.setdefault(name, []).extend(split_shell_command(value))
            continue
        stripped = line.lstrip()
        if stripped.startswith(("gcc ", "g++ ", "clang ", "clang++ ", "cc ", "c++ ", "nvcc ", "hipcc ")):
            recipe_flags.extend(split_shell_command(stripped))
    flags: list[str] = []
    flags.extend(variables.get("CPPFLAGS", []))
    if variables.get("CXXFLAGS"):
        flags.extend(variables["CXXFLAGS"])
        language = "c++"
    else:
        flags.extend(variables.get("CFLAGS", []))
        language = "c" if variables.get("CFLAGS") else None
    flags.extend(recipe_flags)
    flags = _extract_flag_tokens(flags)
    standard = _extract_standard_from_flags(flags)[0]
    if not (compiler or flags or standard):
        return None
    if compiler is None and any("c++" in flag or standard and standard.startswith("c++") for flag in flags):
        compiler = "g++"
    return BuildDetection(
        build_system="make",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_bazel(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for name, text in texts.items():
        literals = _extract_string_literals(text)
        if name == ".bazelrc":
            for token in literals + text.split():
                if token.startswith("--cxxopt="):
                    flags.append(token.removeprefix("--cxxopt="))
                elif token.startswith("--conlyopt="):
                    flags.append(token.removeprefix("--conlyopt="))
        else:
            flags.extend(_extract_flag_tokens(literals))
    if "nvcc" in " ".join(flags).lower():
        compiler = "nvcc"
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or standard):
        return None
    return BuildDetection(
        build_system="bazel",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language="c++",
        flags=flags,
    )


def _parse_gn(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for text in texts.values():
        literals = _extract_string_literals(text)
        flags.extend(_extract_flag_tokens(literals))
        if "is_clang = true" in text.lower():
            compiler = "clang++"
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="gn",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language="c++",
        flags=flags,
    )


def _parse_scons(texts: dict[str, str]) -> BuildDetection | None:
    flags: list[str] = []
    compiler = None
    for text in texts.values():
        for line in text.splitlines():
            match = _VAR_ASSIGN_RE.match(line)
            if match and match.group("name") in {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CCFLAGS", "CPPFLAGS"}:
                name = match.group("name")
                value = split_shell_command(match.group("value"))
                if name in {"CC", "CXX"} and value:
                    compiler = value[0]
                else:
                    flags.extend(value)
        flags.extend(_extract_flag_tokens(_extract_string_literals(text)))
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    if not (flags or compiler or standard):
        return None
    language = "c++" if compiler and ("++" in compiler or compiler == "clang++") else None
    if language is None and standard:
        language = "c++" if standard.startswith("c++") else "c"
    return BuildDetection(
        build_system="scons",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _parse_xmake(text: str) -> BuildDetection | None:
    flags = _extract_flag_tokens(_extract_string_literals(text))
    compiler = None
    for literal in _extract_string_literals(text):
        lowered = literal.lower()
        if lowered in {"clang", "clang++", "gcc", "g++", "nvcc", "hipcc"} and compiler is None:
            compiler = literal
        if lowered.startswith("cxx"):
            flags.append(f"-std=c++{lowered.removeprefix('cxx')}")
        elif lowered.startswith("c") and lowered[1:].isdigit():
            flags.append(f"-std=c{lowered.removeprefix('c')}")
    standard = _extract_standard_from_flags(_extract_flag_tokens(flags))[0]
    language = "c++" if standard and standard.startswith("c++") else None
    if not (flags or compiler or standard):
        return None
    return BuildDetection(
        build_system="xmake",
        source="build_files",
        compiler=compiler,
        standard=standard,
        language=language,
        flags=flags,
    )


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _read_bounded_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise BuildContextEvidenceError(f"cannot stat {label}: {path}") from exc
    if not path.is_file():
        raise BuildContextEvidenceError(f"{label} is not a regular file: {path}")
    if stat_result.st_size > max_bytes:
        raise BuildContextEvidenceError(
            f"{label} exceeds the {max_bytes}-byte trust bound: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BuildContextEvidenceError(f"cannot read {label}: {path}") from exc
    if len(payload) > max_bytes:
        raise BuildContextEvidenceError(
            f"{label} exceeds the {max_bytes}-byte trust bound: {path}"
        )
    return payload


def validate_macos_sdk_path(macos_sdk_path: str | os.PathLike[str]) -> MacOSSDKBinding:
    """Validate an explicitly supplied SDK without consulting host discovery."""

    raw_path = os.fspath(macos_sdk_path)
    if not raw_path or not os.path.isabs(raw_path):
        raise BuildContextEvidenceError("macOS SDK path must be explicit and absolute")
    absolute_path = Path(os.path.abspath(raw_path))
    try:
        resolved_path = absolute_path.resolve(strict=True)
    except OSError as exc:
        raise BuildContextEvidenceError(
            f"macOS SDK path cannot be resolved: {absolute_path}"
        ) from exc
    if absolute_path != resolved_path:
        raise BuildContextEvidenceError(
            f"macOS SDK path must not contain symlink components: {absolute_path}"
        )
    if not resolved_path.is_dir():
        raise BuildContextEvidenceError(
            f"macOS SDK path is not a directory: {resolved_path}"
        )

    json_marker = resolved_path / "SDKSettings.json"
    if json_marker.exists():
        try:
            resolved_marker = json_marker.resolve(strict=True)
        except OSError as exc:
            raise BuildContextEvidenceError(
                f"cannot resolve SDKSettings.json: {exc}"
            ) from exc
        if json_marker != resolved_marker or not _is_path_within(
            resolved_marker, resolved_path
        ):
            raise BuildContextEvidenceError(
                "SDKSettings.json must be a regular in-SDK file"
            )
        payload = _read_bounded_file(
            json_marker,
            max_bytes=_MAX_MACOS_SDK_SETTINGS_BYTES,
            label="macOS SDK SDKSettings.json",
        )
        try:
            settings = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BuildContextEvidenceError(
                f"invalid SDKSettings.json: {exc}"
            ) from exc
        if not isinstance(settings, dict):
            raise BuildContextEvidenceError(
                "invalid SDKSettings.json: root is not a mapping"
            )
        canonical_name = settings.get("CanonicalName")
        default_variant = settings.get("DefaultVariant")
        if not isinstance(canonical_name, str) or not canonical_name.lower().startswith(
            "macosx"
        ):
            raise BuildContextEvidenceError(
                "invalid SDKSettings.json: CanonicalName is not macosx"
            )
        if default_variant is not None and (
            not isinstance(default_variant, str)
            or default_variant.lower() != "macos"
        ):
            raise BuildContextEvidenceError(
                "invalid SDKSettings.json: DefaultVariant is not macos"
            )
        return MacOSSDKBinding(
            path=str(resolved_path),
            settings_path=str(json_marker),
            settings_sha256=hashlib.sha256(payload).hexdigest(),
            canonical_name=canonical_name,
        )

    plist_marker = resolved_path / "SDKSettings.plist"
    if plist_marker.exists():
        try:
            resolved_marker = plist_marker.resolve(strict=True)
        except OSError as exc:
            raise BuildContextEvidenceError(
                f"cannot resolve SDKSettings.plist: {exc}"
            ) from exc
        if plist_marker != resolved_marker or not _is_path_within(
            resolved_marker, resolved_path
        ):
            raise BuildContextEvidenceError(
                "SDKSettings.plist must be a regular in-SDK file"
            )
        payload = _read_bounded_file(
            plist_marker,
            max_bytes=_MAX_MACOS_SDK_SETTINGS_BYTES,
            label="macOS SDK SDKSettings.plist",
        )
        # The marker is bound by its bounded bytes/hash.  Some older SDK
        # layouts ship a plist-like marker that is not parseable by Python's
        # plist reader, so content decoding is advisory for the fallback.
        canonical_name = "macosx"
        try:
            settings = plistlib.loads(payload)
        except (plistlib.InvalidFileException, ValueError, TypeError):
            settings = None
        if isinstance(settings, dict):
            candidate_name = settings.get("CanonicalName")
            if isinstance(candidate_name, str) and candidate_name.lower().startswith(
                "macosx"
            ):
                canonical_name = candidate_name
        return MacOSSDKBinding(
            path=str(resolved_path),
            settings_path=str(plist_marker),
            settings_sha256=hashlib.sha256(payload).hexdigest(),
            canonical_name=canonical_name,
        )

    raise BuildContextEvidenceError(
        "macOS SDK has no valid SDKSettings.json or SDKSettings.plist"
    )


def _discover_xcconfig_files(root: Path) -> list[Path]:
    files: list[Path] = []
    scanned_dirs = 0
    # Source archives can contain very large vendored trees.  Xcode
    # configuration is conventionally kept at the root or in one of these
    # small Apple/config directories, so avoid walking the entire checkout.
    scan_roots: list[Path] = [root]
    for relative in (
        "OSX",
        "OSX/config",
        "macOS",
        "macOS/config",
        "config",
        "configs",
        "xcconfig",
        "Xcode",
    ):
        candidate = root / relative
        if candidate.is_dir() and not candidate.is_symlink():
            scan_roots.append(candidate)
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if (
            child.is_dir()
            and not child.is_symlink()
            and child.name.lower() in {"osx", "macos", "xcode", "config", "configs"}
        ):
            scan_roots.append(child)

    visited_roots: set[Path] = set()
    for scan_root in scan_roots:
        scan_root = scan_root.resolve(strict=True)
        if scan_root in visited_roots:
            continue
        visited_roots.add(scan_root)
        for current_root, dirs, names in os.walk(scan_root, followlinks=False):
            scanned_dirs += 1
            if scanned_dirs > _MAX_XCCONFIG_SCAN_DIRS:
                raise BuildContextEvidenceError(
                    "xcconfig discovery exceeded the directory trust bound"
                )
            if scan_root == root:
                # Root-level xcconfig files are authoritative candidates; the
                # recursive Apple/config roots above cover their subtrees.
                dirs[:] = []
            else:
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if name not in _XCCONFIG_SKIP_DIRS
                    and not Path(current_root, name).is_symlink()
                )
            for name in sorted(names):
                if not name.lower().endswith(".xcconfig"):
                    continue
                path = Path(current_root, name)
                if path.is_symlink():
                    raise BuildContextEvidenceError(
                        f"xcconfig evidence file must not be a symlink: {path}"
                    )
                if path not in files:
                    files.append(path)
                    if len(files) > _MAX_XCCONFIG_FILES:
                        raise BuildContextEvidenceError(
                            "xcconfig discovery exceeded the file trust bound"
                        )
    return files


def _strip_xcconfig_line_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {'"', "'"}:
            quote = None if quote == char else (char if quote is None else quote)
            continue
        if quote is None and char == "/" and line[index : index + 2] == "//":
            return line[:index]
    return line


def _xcconfig_logical_lines(text: str, source: Path) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1]
        pending += line
        if len(pending.encode("utf-8")) > _MAX_XCCONFIG_LOGICAL_LINE_BYTES:
            raise BuildContextEvidenceError(
                f"xcconfig logical line exceeds trust bound: {source}"
            )
        if continued:
            pending += " "
            continue
        lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


def _resolve_xcconfig_include(
    raw_path: str,
    *,
    source: Path,
    root: Path,
) -> tuple[Path | None, str | None]:
    if _XCCONFIG_VARIABLE_RE.search(raw_path) is not None:
        return None, f"xcconfig include has an unresolved variable: {source}: {raw_path}"
    if "\x00" in raw_path:
        return None, f"xcconfig include has a NUL byte: {source}"
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [source.parent / raw, root / raw]
    resolved_candidates: list[Path] = []
    outside_project = False
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if not _is_path_within(resolved, root):
            outside_project = True
            continue
        if resolved.is_file() and resolved not in resolved_candidates:
            resolved_candidates.append(resolved)
    if outside_project:
        return None, f"xcconfig include escapes project root: {source}: {raw_path}"
    if len(resolved_candidates) > 1:
        return None, f"xcconfig include is ambiguous: {source}: {raw_path}"
    return (resolved_candidates[0] if resolved_candidates else None), None


def _load_xcconfig_assignments(
    root: Path,
    initial_files: list[Path],
) -> tuple[list[_XCConfigAssignment], list[str]]:
    assignments: list[_XCConfigAssignment] = []
    issues: list[str] = []
    pending = list(initial_files)
    visited: set[Path] = set()
    total_bytes = 0
    while pending:
        source = pending.pop(0)
        if source in visited:
            continue
        visited.add(source)
        if len(visited) > _MAX_XCCONFIG_FILES:
            raise BuildContextEvidenceError(
                "xcconfig include graph exceeded the file trust bound"
            )
        payload = _read_bounded_file(
            source,
            max_bytes=_MAX_XCCONFIG_FILE_BYTES,
            label="xcconfig evidence file",
        )
        total_bytes += len(payload)
        if total_bytes > _MAX_XCCONFIG_TOTAL_BYTES:
            raise BuildContextEvidenceError(
                "xcconfig evidence exceeded the total-byte trust bound"
            )
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            issues.append(f"xcconfig is not UTF-8: {source}: {exc}")
            continue
        for raw_line in _xcconfig_logical_lines(text, source):
            line = _strip_xcconfig_line_comment(raw_line).strip()
            if not line:
                continue
            include_match = _XCCONFIG_INCLUDE_RE.match(line)
            if include_match is not None:
                include_path, issue = _resolve_xcconfig_include(
                    include_match.group("path"),
                    source=source,
                    root=root,
                )
                if issue is not None:
                    issues.append(issue)
                elif include_path is not None and include_path not in visited:
                    pending.append(include_path)
                continue
            assignment_match = _XCCONFIG_ASSIGNMENT_RE.match(line)
            if assignment_match is None:
                continue
            name = assignment_match.group("name")
            if name not in _XCCONFIG_RELEVANT_SETTINGS:
                continue
            assignments.append(
                _XCConfigAssignment(
                    name=name,
                    conditions=assignment_match.group("conditions"),
                    value=assignment_match.group("value").rstrip(";").strip(),
                    source=source,
                )
            )
    return assignments, issues


def _split_xcconfig_value(assignment: _XCConfigAssignment) -> list[str]:
    try:
        return [token.rstrip(";") for token in shlex.split(assignment.value)]
    except ValueError as exc:
        raise BuildContextEvidenceError(
            f"cannot parse {assignment.name} in {assignment.source}: {exc}"
        ) from exc


def _setting_values(
    assignments: list[_XCConfigAssignment],
    setting: str,
) -> tuple[set[str], list[_XCConfigAssignment]]:
    selected = [assignment for assignment in assignments if assignment.name == setting]
    values: set[str] = set()
    for assignment in selected:
        if assignment.conditions:
            raise BuildContextEvidenceError(
                f"conditional {setting} evidence is ambiguous: {assignment.source}"
            )
        for token in _split_xcconfig_value(assignment):
            lowered = token.lower()
            if lowered in {"$(inherited)", "${inherited}"}:
                continue
            if _XCCONFIG_VARIABLE_RE.search(token) is not None:
                raise BuildContextEvidenceError(
                    f"unresolved variable in {setting}: {assignment.source}: {token}"
                )
            if token:
                values.add(lowered)
    return values, selected


def _resolve_xcconfig_header_path(
    token: str,
    *,
    source: Path,
    root: Path,
) -> Path | None:
    if token.lower() in {"$(inherited)", "${inherited}"}:
        return None
    if token.startswith(_XCCONFIG_IGNORED_BUILD_PATH_VARIABLES):
        return None
    if any(char in token for char in "*?["):
        raise BuildContextEvidenceError(
            f"wildcard header search path is ambiguous: {source}: {token}"
        )

    base = source.parent
    expanded = token
    source_root_prefixes = ("$(SRCROOT)", "${SRCROOT}", "$(SOURCE_ROOT)", "${SOURCE_ROOT}")
    project_dir_prefixes = ("$(PROJECT_DIR)", "${PROJECT_DIR}")
    prefix = next((item for item in project_dir_prefixes if token.startswith(item)), None)
    if prefix is not None:
        suffix = token[len(prefix) :]
        if suffix and not suffix.startswith("/"):
            raise BuildContextEvidenceError(
                f"ambiguous PROJECT_DIR header search path: {source}: {token}"
            )
        expanded = str(source.parent) + suffix
    else:
        prefix = next((item for item in source_root_prefixes if token.startswith(item)), None)
        if prefix is not None:
            suffix = token[len(prefix) :]
            if suffix and not suffix.startswith("/"):
                raise BuildContextEvidenceError(
                    f"ambiguous SRCROOT header search path: {source}: {token}"
                )
            expanded = str(root) + suffix
            base = root

    if _XCCONFIG_VARIABLE_RE.search(expanded) is not None or "$" in expanded:
        raise BuildContextEvidenceError(
            f"unresolved variable in header search path: {source}: {token}"
        )
    raw_path = Path(expanded)
    candidate = raw_path if raw_path.is_absolute() else base / raw_path
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise BuildContextEvidenceError(
            f"cannot resolve header search path: {source}: {token}"
        ) from exc
    if not _is_path_within(resolved, root):
        raise BuildContextEvidenceError(
            f"header search path escapes project root: {source}: {token}"
        )
    if not resolved.exists():
        return None
    if not resolved.is_dir():
        raise BuildContextEvidenceError(
            f"header search path is not a directory: {source}: {token}"
        )
    return resolved


def detect_macos_xcconfig_context(
    repo_dir: str | os.PathLike[str],
    macos_sdk_path: str | os.PathLike[str] | None = None,
) -> MacOSXCConfigContext | None:
    """Return a bounded macOS context only for complete xcconfig evidence."""

    try:
        root = Path(repo_dir).resolve(strict=True)
    except OSError as exc:
        raise BuildContextEvidenceError(
            f"project root does not resolve: {repo_dir}"
        ) from exc
    if not root.is_dir():
        raise BuildContextEvidenceError(f"project root is not a directory: {root}")
    xcconfig_files = _discover_xcconfig_files(root)
    if not xcconfig_files:
        return None
    assignments, issues = _load_xcconfig_assignments(root, xcconfig_files)

    macos_hint = any(
        assignment.name in {"SDKROOT", "SUPPORTED_PLATFORMS"}
        and re.search(r"(?<![A-Za-z0-9_])macosx(?:\.internal)?(?![A-Za-z0-9_])", assignment.value.lower())
        is not None
        for assignment in assignments
    )
    if not macos_hint:
        return None
    if issues:
        raise BuildContextEvidenceError(issues[0])

    sdkroots, sdkroot_assignments = _setting_values(assignments, "SDKROOT")
    platforms, platform_assignments = _setting_values(
        assignments, "SUPPORTED_PLATFORMS"
    )
    standards, standard_assignments = _setting_values(
        assignments, "GCC_C_LANGUAGE_STANDARD"
    )
    if not sdkroot_assignments or not sdkroots:
        raise BuildContextEvidenceError("macOS xcconfig evidence is missing SDKROOT")
    if len(sdkroots) != 1 or not sdkroots <= {"macosx", "macosx.internal"}:
        raise BuildContextEvidenceError(
            f"macOS xcconfig SDKROOT evidence is ambiguous: {sorted(sdkroots)}"
        )
    if not platform_assignments or platforms != {"macosx"}:
        raise BuildContextEvidenceError(
            "macOS xcconfig SUPPORTED_PLATFORMS evidence must be exactly macosx"
        )
    if not standard_assignments or standards != {"gnu2x"}:
        raise BuildContextEvidenceError(
            "macOS xcconfig GCC_C_LANGUAGE_STANDARD evidence must be exactly gnu2x"
        )

    header_paths: set[Path] = set()
    header_assignments = [
        assignment
        for assignment in assignments
        if assignment.name in _XCCONFIG_HEADER_SETTINGS
    ]
    for assignment in header_assignments:
        if assignment.conditions:
            raise BuildContextEvidenceError(
                f"conditional header search evidence is ambiguous: {assignment.source}"
            )
        for token in _split_xcconfig_value(assignment):
            resolved = _resolve_xcconfig_header_path(
                token,
                source=assignment.source,
                root=root,
            )
            if resolved is not None:
                header_paths.add(resolved)
    if not header_assignments or not header_paths:
        raise BuildContextEvidenceError(
            "macOS xcconfig evidence has no usable project-local header search paths"
        )
    if macos_sdk_path is None:
        raise BuildContextEvidenceError(
            "macOS xcconfig evidence requires an explicit macOS SDK path"
        )
    sdk = validate_macos_sdk_path(macos_sdk_path)

    include_paths = {root, *header_paths}
    for dirname in ("include", "src", "lib", "source"):
        candidate = (root / dirname).resolve(strict=False)
        if candidate.is_dir() and _is_path_within(candidate, root):
            include_paths.add(candidate)
    ordered_include_paths = [root, *sorted(include_paths - {root}, key=str)]
    compile_args = (
        "-std=gnu2x",
        "-fblocks",
        *(f"-I{path}" for path in ordered_include_paths),
        "-isysroot",
        sdk.path,
        "-fsyntax-only",
        "-Wno-everything",
    )
    evidence_assignments = (
        sdkroot_assignments
        + platform_assignments
        + standard_assignments
        + header_assignments
    )
    evidence_files = tuple(
        sorted(
            {
                assignment.source.relative_to(root).as_posix()
                for assignment in evidence_assignments
            }
        )
    )
    return MacOSXCConfigContext(
        sdkroot=next(iter(sdkroots)),
        c_language_standard="gnu2x",
        header_search_paths=tuple(str(path) for path in ordered_include_paths),
        evidence_files=evidence_files,
        sdk=sdk,
        compile_args=compile_args,
    )


def detect_build_context_from_loader(read_text: Callable[[str], str | None]) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    compile_commands = _build_detection_from_compile_commands(read_text("compile_commands.json"))
    if compile_commands[2] is not None:
        return compile_commands

    cmake_text = read_text("CMakeLists.txt")
    if cmake_text:
        detection = _parse_cmake(cmake_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    configure_text = read_text("configure.ac")
    if configure_text:
        detection = _parse_autoconf(configure_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    meson_text = read_text("meson.build")
    if meson_text:
        detection = _parse_meson(meson_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    bazel_texts = {name: text for name in _BAZEL_BUILD_NAMES if (text := read_text(name)) is not None}
    if bazel_texts:
        detection = _parse_bazel(bazel_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    gn_texts = {name: text for name in _GN_BUILD_NAMES if (text := read_text(name)) is not None}
    if gn_texts:
        detection = _parse_gn(gn_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    scons_texts = {name: text for name in _SCONS_BUILD_NAMES if (text := read_text(name)) is not None}
    if scons_texts:
        detection = _parse_scons(scons_texts)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    xmake_text = next((read_text(name) for name in _XMAKE_BUILD_NAMES if read_text(name)), None)
    if xmake_text:
        detection = _parse_xmake(xmake_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    make_text = next((read_text(name) for name in _MAKEFILE_NAMES if read_text(name)), None)
    if make_text:
        detection = _parse_make(make_text)
        if detection is not None:
            context, compile_args = _detection_to_context(detection)
            return context, compile_args, None

    return {
        **DEFAULT_PLATFORM,
        "build_system": "default",
        "source": "default",
    }, ["-std=c++17", "-fsyntax-only", "-Wno-everything"], None


def detect_build_context(
    repo_dir: str,
    macos_sdk_path: str | os.PathLike[str] | None = None,
) -> tuple[dict, list[str], CompileCommandsIndex | None]:
    compile_commands_path = find_compile_commands_file(repo_dir)

    def read_text(name: str) -> str | None:
        if name == "compile_commands.json" and compile_commands_path:
            path = compile_commands_path
        else:
            path = os.path.join(repo_dir, name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # A valid compilation database is authoritative.  Do this before any
    # xcconfig discovery so a large or unrelated Xcode tree cannot replace
    # per-file compiler arguments.
    compile_context = _build_detection_from_compile_commands(
        read_text("compile_commands.json")
    )
    if compile_context[2] is not None:
        return compile_context

    macos_context = detect_macos_xcconfig_context(
        repo_dir,
        macos_sdk_path=macos_sdk_path,
    )
    if macos_context is not None:
        return (
            macos_context.platform_info(),
            list(macos_context.compile_args),
            None,
        )

    return detect_build_context_from_loader(read_text)
