"""Deterministic build-system domain parsers with lazy public exports."""

from importlib import import_module


_EXPORT_MODULES = {
    "parse_autoconf": "cppmega.data.build_parsers.autotools",
    "parse_automake": "cppmega.data.build_parsers.autotools",
    "parse_configure": "cppmega.data.build_parsers.autotools",
    "parse_bazel": "cppmega.data.build_parsers.bazel",
    "parse_cmake": "cppmega.data.build_parsers.cmake",
    "parse_make": "cppmega.data.build_parsers.make",
    "parse_ninja": "cppmega.data.build_parsers.ninja",
}

__all__ = [
    "parse_autoconf",
    "parse_automake",
    "parse_configure",
    "parse_bazel",
    "parse_cmake",
    "parse_make",
    "parse_ninja",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
