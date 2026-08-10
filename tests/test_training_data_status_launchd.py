from __future__ import annotations

import plistlib
import re
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLIST_PATH = REPO_ROOT / "configs/launchd/ai.cppmega.training-data-status.plist"
EXPECTED_RUNTIME = Path(
    "/Volumes/external/cppmega_data/worktrees/"
    "cppmega-training-status-runtime-a9145509-20260810"
)
EXPECTED_REVISION = "a91455098f2a1a82f863ce64e98d237b9e930177"


def test_training_status_launchd_uses_exact_pinned_runtime() -> None:
    config = plistlib.loads(PLIST_PATH.read_bytes())
    assert config["Label"] == "ai.cppmega.training-data-status"
    assert config["RunAtLoad"] is True
    assert config["KeepAlive"] == {"SuccessfulExit": False}

    program_arguments = config["ProgramArguments"]
    assert program_arguments[:2] == ["/bin/zsh", "-c"]
    command = shlex.split(program_arguments[2])
    reporter = EXPECTED_RUNTIME / "scripts/report_training_data_status.py"
    status_config = EXPECTED_RUNTIME / "configs/training_data_status.json"
    assert command[:3] == [
        "exec",
        "/Volumes/external/sources/.venvs/cppmega.source/bin/python",
        str(reporter),
    ]
    assert command[command.index("--config") + 1] == str(status_config)
    assert command[command.index("--expected-code-revision") + 1] == (
        EXPECTED_REVISION
    )
    assert re.fullmatch(r"[0-9a-f]{40}", EXPECTED_REVISION)
    assert "active-parquet-temp" not in program_arguments[2]
