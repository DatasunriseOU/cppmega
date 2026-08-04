from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_sdk(tmp_path: Path) -> Path:
    sdk = tmp_path / "MacOSX.sdk"
    sdk.mkdir()
    (sdk / "SDKSettings.json").write_text(
        json.dumps(
            {
                "CanonicalName": "macosx99.1",
                "DefaultVariant": "macos",
                "DisplayName": "macOS 99.1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sdk.resolve()


def _write_verified_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    config = project / "config"
    config.mkdir(parents=True)
    for relative in (
        "include",
        "lib",
        "user_headers",
        "header_symlinks",
        "header_symlinks/macOS",
    ):
        (project / relative).mkdir()

    # The non-xcconfig suffix proves that this evidence is reached recursively,
    # not merely found by the initial project scan.
    (config / "base.settings").write_text(
        "SDKROOT = macosx.internal\n"
        "HEADER_SEARCH_PATHS = $(PROJECT_DIR)/../include \\\n"
        "    $(BUILT_PRODUCTS_DIR)/derived_src $(inherited)\n"
        "SYSTEM_HEADER_SEARCH_PATHS = $(PROJECT_DIR)/../header_symlinks/macOS "
        "$(PROJECT_DIR)/../header_symlinks\n",
        encoding="utf-8",
    )
    (config / "language.xcconfig").write_text(
        '#include "base.settings"\n'
        "GCC_C_LANGUAGE_STANDARD = gnu2x\n",
        encoding="utf-8",
    )
    (config / "target.xcconfig").write_text(
        '#include "language.xcconfig"\n'
        "SUPPORTED_PLATFORMS = macosx\n"
        "USER_HEADER_SEARCH_PATHS = $(inherited) "
        "$(PROJECT_DIR)/../user_headers\n",
        encoding="utf-8",
    )
    return project.resolve(), config / "target.xcconfig"


def test_verified_recursive_xcconfig_enriches_compile_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega.data.build_context import (
        detect_build_context,
        detect_macos_xcconfig_context,
    )

    project, _target = _write_verified_project(tmp_path)
    sdk = _write_sdk(tmp_path)
    monkeypatch.setenv("PATH", "")

    context = detect_macos_xcconfig_context(
        project,
        macos_sdk_path=sdk,
    )

    assert context is not None
    assert context.sdkroot == "macosx.internal"
    assert context.c_language_standard == "gnu2x"
    assert context.sdk.path == str(sdk)
    assert context.sdk.canonical_name == "macosx99.1"
    assert len(context.sdk.settings_sha256) == 64
    assert context.evidence_files == (
        "config/base.settings",
        "config/language.xcconfig",
        "config/target.xcconfig",
    )
    assert set(context.header_search_paths) >= {
        str(project),
        str(project / "include"),
        str(project / "lib"),
        str(project / "user_headers"),
        str(project / "header_symlinks"),
        str(project / "header_symlinks/macOS"),
    }
    assert context.compile_args[:2] == ("-std=gnu2x", "-fblocks")
    assert f"-I{project}" in context.compile_args
    assert f"-I{project / 'header_symlinks/macOS'}" in context.compile_args
    assert context.compile_args[-4:] == (
        "-isysroot",
        str(sdk),
        "-fsyntax-only",
        "-Wno-everything",
    )

    platform, compile_args, compile_index = detect_build_context(
        str(project),
        macos_sdk_path=str(sdk),
    )
    assert compile_index is None
    assert platform["platform"] == "macosx"
    assert platform["standard"] == "c23"
    assert platform["macos_sdk_settings_sha256"] == context.sdk.settings_sha256
    assert compile_args == list(context.compile_args)


def test_non_macos_xcconfig_leaves_normal_detection_unchanged(
    tmp_path: Path,
) -> None:
    from cppmega.data.build_context import detect_build_context

    (tmp_path / "mobile.xcconfig").write_text(
        "SDKROOT = iphoneos\n"
        "SUPPORTED_PLATFORMS = iphoneos\n"
        "HEADER_SEARCH_PATHS = $(UNBOUND)/headers\n",
        encoding="utf-8",
    )

    platform, compile_args, compile_index = detect_build_context(
        str(tmp_path),
        macos_sdk_path="/definitely/not/a/real/sdk",
    )

    assert compile_index is None
    assert platform["build_system"] == "default"
    assert compile_args == ["-std=c++17", "-fsyntax-only", "-Wno-everything"]


def test_compile_commands_take_precedence_over_macos_xcconfig(
    tmp_path: Path,
) -> None:
    from cppmega.data.build_context import detect_build_context

    project, _target = _write_verified_project(tmp_path)
    source = project / "main.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    (project / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(project),
                    "file": str(source),
                    "command": "clang -std=c11 -c main.c",
                }
            ]
        ),
        encoding="utf-8",
    )

    platform, compile_args, compile_index = detect_build_context(
        str(project),
        macos_sdk_path="/definitely/not/a/real/sdk",
    )

    assert compile_index is not None
    assert platform["build_system"] == "compile_commands"
    assert "-std=c11" in compile_args


def test_macos_evidence_requires_explicit_sdk(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_build_context,
    )

    project, _target = _write_verified_project(tmp_path)

    with pytest.raises(
        BuildContextEvidenceError,
        match="requires an explicit macOS SDK path",
    ):
        detect_build_context(str(project))


def test_macos_evidence_rejects_invalid_sdk_marker(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_macos_xcconfig_context,
    )

    project, _target = _write_verified_project(tmp_path)
    sdk = tmp_path / "not-macos.sdk"
    sdk.mkdir()
    (sdk / "SDKSettings.json").write_text(
        '{"CanonicalName":"iphoneos99.1","DefaultVariant":"iphoneos"}',
        encoding="utf-8",
    )

    with pytest.raises(BuildContextEvidenceError, match="CanonicalName is not macosx"):
        detect_macos_xcconfig_context(project, macos_sdk_path=sdk.resolve())


def test_sdk_binding_changes_when_settings_marker_changes(tmp_path: Path) -> None:
    from cppmega.data.build_context import validate_macos_sdk_path

    sdk = _write_sdk(tmp_path)
    first = validate_macos_sdk_path(sdk)
    (sdk / "SDKSettings.json").write_text(
        '{"CanonicalName":"macosx99.2","DefaultVariant":"macos"}',
        encoding="utf-8",
    )
    second = validate_macos_sdk_path(sdk)

    assert first.path == second.path
    assert first.canonical_name == "macosx99.1"
    assert second.canonical_name == "macosx99.2"
    assert first.settings_sha256 != second.settings_sha256


def test_sdk_path_rejects_symlink(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        validate_macos_sdk_path,
    )

    sdk = _write_sdk(tmp_path)
    alias = tmp_path / "sdk-alias"
    alias.symlink_to(sdk, target_is_directory=True)

    with pytest.raises(BuildContextEvidenceError, match="symlink components"):
        validate_macos_sdk_path(alias)


def test_macos_evidence_rejects_ambiguous_sdkroot(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_macos_xcconfig_context,
    )

    project, _target = _write_verified_project(tmp_path)
    sdk = _write_sdk(tmp_path)
    (project / "conflict.xcconfig").write_text(
        "SDKROOT = macosx\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildContextEvidenceError, match="SDKROOT evidence is ambiguous"):
        detect_macos_xcconfig_context(project, macos_sdk_path=sdk)


def test_macos_hint_rejects_missing_sdkroot_evidence(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_macos_xcconfig_context,
    )

    project = tmp_path / "project"
    (project / "include").mkdir(parents=True)
    (project / "target.xcconfig").write_text(
        "SUPPORTED_PLATFORMS = macosx\n"
        "GCC_C_LANGUAGE_STANDARD = gnu2x\n"
        "HEADER_SEARCH_PATHS = $(PROJECT_DIR)/include\n",
        encoding="utf-8",
    )
    sdk = _write_sdk(tmp_path)

    with pytest.raises(BuildContextEvidenceError, match="missing SDKROOT"):
        detect_macos_xcconfig_context(project, macos_sdk_path=sdk)


@pytest.mark.parametrize(
    "unsafe_header",
    (
        "$(PROJECT_DIR)/../../outside-project",
        "$(UNBOUND_HEADER_ROOT)/include",
        "$(PROJECT_DIR)/**",
    ),
)
def test_macos_evidence_rejects_unsafe_header_roots(
    tmp_path: Path,
    unsafe_header: str,
) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_macos_xcconfig_context,
    )

    project, target = _write_verified_project(tmp_path)
    sdk = _write_sdk(tmp_path)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(f"HEADER_SEARCH_PATHS += {unsafe_header}\n")

    with pytest.raises(BuildContextEvidenceError):
        detect_macos_xcconfig_context(project, macos_sdk_path=sdk)


def test_macos_evidence_rejects_include_traversal(tmp_path: Path) -> None:
    from cppmega.data.build_context import (
        BuildContextEvidenceError,
        detect_macos_xcconfig_context,
    )

    project, target = _write_verified_project(tmp_path)
    sdk = _write_sdk(tmp_path)
    with target.open("a", encoding="utf-8") as stream:
        stream.write('#include "../../outside-project.xcconfig"\n')

    with pytest.raises(BuildContextEvidenceError, match="include escapes project root"):
        detect_macos_xcconfig_context(project, macos_sdk_path=sdk)
