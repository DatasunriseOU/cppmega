from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTED_RUNNER = re.compile(
    r"^\s*runs-on:\s*.*(?:ubuntu|macos|windows)-latest\s*$",
    re.MULTILINE,
)


def test_workflows_do_not_use_github_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if HOSTED_RUNNER.search(workflow.read_text(encoding="utf-8")):
            violations.append(workflow.relative_to(REPO_ROOT).as_posix())

    assert not violations, f"GitHub-hosted runners are forbidden: {violations}"
