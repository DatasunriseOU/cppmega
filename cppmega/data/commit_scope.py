"""Primary native-domain routing for commit and PR training data."""

from __future__ import annotations

from pathlib import Path

from cppmega.data.domain_ingestion import resolve_domain_parser


_ADAPTER_BUILD_KINDS = {
    "cpp-lexical": "cpp",
    "cmake": "cmake",
    "make": "make",
    "ninja": "ninja",
    "bazel-starlark": "bazel",
    "configure-shell": "configure",
    "autoconf": "autoconf",
    "automake": "automake",
    "meson": "meson",
    "gn-raw": "gn",
    "scons-raw": "scons",
    "xmake-raw": "xmake",
    "compile-commands-json": "compile_commands",
    "dockerfile": "dockerfile",
    "bash": "bash",
    "posix-sh": "sh",
    "zsh": "zsh",
    "tcsh": "tcsh",
    "ksh": "ksh",
    "powershell": "powershell",
    "cmd": "cmd",
    "sql-lexical": "sql",
}
_RAW_BUILD_NAMES = {
    "conanfile.txt": "conan",
    "conanfile.py": "conan",
    "vcpkg.json": "vcpkg",
    ".gn": "gn",
}
_RAW_BUILD_SUFFIXES = {
    ".vcxproj": "msvc",
    ".sln": "msvc",
}
_CPP_SUFFIX_OVERRIDES = {".tcc"}


def classify_primary_commit_path(path: str, text: str = "") -> str | None:
    """Return the deterministic primary-corpus kind, excluding Python/JS/logs."""

    path_obj = Path(path)
    explicit_kind = _RAW_BUILD_NAMES.get(path_obj.name)
    if explicit_kind is not None:
        return explicit_kind
    explicit_kind = _RAW_BUILD_SUFFIXES.get(path_obj.suffix.lower())
    if explicit_kind is not None:
        return explicit_kind
    if path_obj.suffix.lower() in _CPP_SUFFIX_OVERRIDES:
        return "cpp"
    return _ADAPTER_BUILD_KINDS.get(resolve_domain_parser(path_obj, text).name)
