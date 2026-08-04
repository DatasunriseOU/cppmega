from __future__ import annotations

from pathlib import Path

from scripts import streaming_conveyor as conveyor
from scripts import streaming_reindex as reindex


def test_stage_index_source_passes_explicit_macos_sdk_to_indexer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_dir = tmp_path / "repo"
    work = tmp_path / "work"
    sdk = tmp_path / "MacOSX.sdk"
    repo_dir.mkdir()
    work.mkdir()
    sdk.mkdir()
    observed: dict[str, object] = {}

    def fake_run_checked(
        repo: str,
        stage: str,
        cmd: list[object],
        **_kwargs: object,
    ) -> None:
        observed.update(repo=repo, stage=stage, cmd=cmd)
        output = Path(str(cmd[cmd.index("--output") + 1]))
        output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(reindex, "run_checked", fake_run_checked)

    output = reindex.stage_index_source(
        "apple-security",
        "apple-oss-distributions/security",
        repo_dir,
        work,
        macos_sdk=sdk,
    )

    assert output.is_file()
    command = observed["cmd"]
    assert isinstance(command, list)
    sdk_index = command.index("--macos-sdk")
    assert command[sdk_index + 1] == str(sdk)


def test_conveyor_code_half_forwards_macos_sdk_to_reindex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sdk = tmp_path / "MacOSX.sdk"
    sdk.mkdir()
    observed: dict[str, object] = {}

    def fake_process_one_repo(*_args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"skipped": True}

    monkeypatch.setattr(conveyor.sr, "process_one_repo", fake_process_one_repo)

    result = conveyor.run_code_half(
        "apple-security",
        "apple-oss-distributions/security",
        tmp_path / "repo",
        (1024,),
        tmp_path / "work",
        None,
        True,
        macos_sdk=sdk,
    )

    assert result == {"skipped": True}
    assert observed["macos_sdk"] == sdk


def test_streaming_conveyor_accepts_explicit_macos_sdk_argument(
    tmp_path: Path,
) -> None:
    sdk = tmp_path / "MacOSX.sdk"
    sdk.mkdir()

    args = conveyor.parse_args(["--macos-sdk", str(sdk)])

    assert args.macos_sdk == str(sdk)
