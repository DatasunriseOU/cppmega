from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from cppmega.megatron.graph_recipe import stage1_graph_recipe_binding
from cppmega.megatron.objective_contract import (
    load_objective_materialization_artifact,
)
from tests.test_megatron_objective_contract import _valid_contract


_EXPECTED_OBJECTIVE_ARTIFACT_FILE_SHA256 = (
    "8edfac1806082c7c17fdad73b3b26d9364ac9806a793efe6252b89fa4298d406"
)
_EXPECTED_OBJECTIVE_ARTIFACT_SET_SHA256 = (
    "ceb08c6a5a8f6dda3e703dd3535b092f0f66c9bcd065bdad6917facd1880a91b"
)


def _mlx_root() -> Path:
    configured = os.environ.get("CPPMEGA_RECIPE_PARITY_PEER_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(__file__).resolve().parents[2]
        / "cppmega_mlx_fix_recipe_parity_20260718"
    )
    candidates.append(Path(__file__).resolve().parents[2] / "cppmega.mlx")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("cross-repository artifact parity worktree is unavailable")


def _mlx_python(peer_root: Path) -> Path:
    """Select an interpreter that can actually import MLX.

    The root Megatron environment intentionally does not install MLX.  Using
    ``sys.executable`` here would turn a cross-repository parity test into an
    accidental environment test, so the peer interpreter is an explicit
    contract.  CI can bind it with ``CPPMEGA_RECIPE_PARITY_PYTHON``.
    """
    configured = os.environ.get("CPPMEGA_RECIPE_PARITY_PYTHON")
    if configured:
        candidates = [Path(configured)]
        explicit = True
    else:
        candidates = [
            peer_root / ".venv" / "bin" / "python",
            peer_root.parent / ".venvs" / "cppmega.mlx" / "bin" / "python",
            Path("/Volumes/external/sources/.venvs/cppmega.mlx/bin/python"),
        ]
        explicit = False

    for candidate in candidates:
        if not candidate.is_file():
            if explicit:
                pytest.fail(f"configured MLX parity interpreter is missing: {candidate}")
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import mlx.core"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
        if explicit:
            pytest.fail(
                "configured MLX parity interpreter cannot import mlx.core: "
                f"{candidate}\n{probe.stderr or probe.stdout}"
            )
    pytest.skip(
        "cross-repository artifact parity requires an interpreter that imports "
        "mlx.core; set CPPMEGA_RECIPE_PARITY_PYTHON in the root lane"
    )


def test_mlx_generated_objective_artifact_is_root_loadable_and_recipe_bound(
    tmp_path: Path,
) -> None:
    peer_root = _mlx_root()
    peer_python = _mlx_python(peer_root)

    contract = _valid_contract()
    contract_path = tmp_path / "input_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    shard = tmp_path / "objectives_00000.parquet"
    shard.write_bytes(b"cross-repository objective shard")

    peer_code = """
import json
import os
from pathlib import Path

from cppmega_mlx.training.megatron_objectives import (
    OBJECTIVE_ROUTE_RETENTION_SCHEMA,
    objective_route_mapping_contract,
    write_objective_materialization_artifact,
)

output_dir = Path(os.environ["CPPMEGA_PARITY_OUTPUT_DIR"])
contract = json.loads(Path(os.environ["CPPMEGA_PARITY_CONTRACT"]).read_text())
graph = contract["graph_auxiliary"]
graph["route_mapping"] = objective_route_mapping_contract()
graph["route_retention"] = {
    "schema": OBJECTIVE_ROUTE_RETENTION_SCHEMA,
    "by_objective": {"causal_lm": {"samples": 6}},
}
write_objective_materialization_artifact(
    output_dir,
    contract=contract,
    parquet_paths=[Path(os.environ["CPPMEGA_PARITY_SHARD"])],
)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "CPPMEGA_PARITY_OUTPUT_DIR": str(tmp_path),
            "CPPMEGA_PARITY_CONTRACT": str(contract_path),
            "CPPMEGA_PARITY_SHARD": str(shard),
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(peer_root), environment.get("PYTHONPATH", "")))
            ),
        }
    )
    completed = subprocess.run(
        [str(peer_python), "-c", peer_code],
        cwd=peer_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    artifact_path = tmp_path / "objective_materialization.json"
    artifact_bytes = artifact_path.read_bytes()
    raw = json.loads(artifact_bytes)

    assert set(raw) == {
        "schema",
        "graph_recipe",
        "documents",
        "objective_contract",
        "parquet_shards",
        "converter",
        "artifact_set_sha256",
    }
    assert raw["schema"] == "cppmega_objective_materialization_artifact_v2"
    assert raw["graph_recipe"] == stage1_graph_recipe_binding()
    assert raw["artifact_set_sha256"] == _EXPECTED_OBJECTIVE_ARTIFACT_SET_SHA256
    loaded = load_objective_materialization_artifact(artifact_path)
    assert loaded.payload == raw
    assert loaded.file_sha256 == hashlib.sha256(artifact_bytes).hexdigest()
    assert loaded.file_sha256 == _EXPECTED_OBJECTIVE_ARTIFACT_FILE_SHA256
    assert loaded.artifact_set_sha256 == raw["artifact_set_sha256"]
