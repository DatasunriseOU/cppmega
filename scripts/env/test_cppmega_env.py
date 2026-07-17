from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("cppmega_env.py")
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_source(root: Path) -> str:
    (root / "megatron" / "core").mkdir(parents=True)
    (root / "megatron" / "core" / "__init__.py").write_text("\n", encoding="utf-8")
    (root / "megatron" / "core" / "package_info.py").write_text(
        "__version__ = '0.18.0rc0'\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        """[project]\nname = 'megatron-core'\nversion = '0.18.0rc0'\nrequires-python = '>=3.12'\ndependencies = ['torch>=2.6.0', 'numpy', 'packaging>=24.2']\n""",
        encoding="utf-8",
    )
    (root / "megatron" / "core" / "transformer" / "moe").mkdir(parents=True)
    for module in (
        root / "megatron" / "core" / "transformer" / "__init__.py",
        root / "megatron" / "core" / "transformer" / "transformer_layer.py",
        root / "megatron" / "core" / "transformer" / "moe" / "__init__.py",
        root / "megatron" / "core" / "transformer" / "moe" / "moe_utils.py",
    ):
        module.write_text("\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "cppmega env tests")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


class CppmegaEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _install_minimal_distributions(checkout: Path, target: Path) -> None:
        import packaging

        packaging_root = Path(packaging.__file__).resolve().parent
        shutil.copytree(packaging_root, checkout / "packaging")
        site_packages = target / "lib" / "python3.13" / "site-packages"
        for name, version in (
            ("torch", "2.6.0"),
            ("numpy", "2.0.0"),
            ("packaging", "24.2"),
        ):
            dist_info = site_packages / f"{name}-{version}.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                encoding="utf-8",
            )

    def test_shared_symlink_is_rejected_before_bootstrap(self) -> None:
        from cppmega_env import EnvError, ensure_target_isolation

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = root / "nanochat" / ".venv"
            shared.mkdir(parents=True)
            checkout = root / "cppmega"
            checkout.mkdir()
            link = checkout / ".venv"
            link.symlink_to(shared, target_is_directory=True)

            with mock.patch.dict(os.environ, {"VIRTUAL_ENV": str(shared)}):
                with self.assertRaisesRegex(EnvError, "shared|symlink"):
                    ensure_target_isolation(checkout, link)

    def test_real_data_dependency_contract_includes_near_dedup_runtime(self) -> None:
        from cppmega_env import _dependency_contract, _project_data_dependencies, SourceInfo

        self.assertIn(
            "datasketch==1.10.0",
            _project_data_dependencies(SCRIPT.parents[2]),
        )

        source = SourceInfo(
            root=Path("/tmp/megatron"),
            head="head",
            expected_commit="head",
            expected_ref="head",
            dirty_entries=(),
            version="0.18.0rc0",
            requires_python=">=3.12",
            dependencies=(
                "demo~=1.4; platform_system == 'Darwin'",
                "demo~=1.4; platform_system == 'Darwin'",
            ),
        )
        contract = _dependency_contract(SCRIPT.parents[2], source, "source")
        self.assertEqual(len([item for item in contract if item["name"] == "demo"]), 1)
        self.assertEqual(
            contract[0]["requirement"],
            "demo~=1.4; platform_system == 'Darwin'",
        )

    def test_verify_rejects_conflicting_explicit_megatron_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.pop("VIRTUAL_ENV", None)
            environment["MEGATRON_LM_REPO"] = str(root / "source-a")
            environment["MEGATRON_ROOT"] = str(root / "source-b")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(SCRIPT.parents[2]),
                    "--env",
                    str(root / "environment"),
                ],
                cwd=SCRIPT.parents[2],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "conflicting explicit Megatron-LM source roots",
                result.stdout + result.stderr,
            )

    def test_pytest_rejects_repo_shared_venv_without_manifest(self) -> None:
        repo_root = SCRIPT.parents[2]
        shared_python = repo_root / ".venv" / "bin" / "python"
        if not shared_python.is_file():
            self.skipTest(f"shared repository venv is absent: {shared_python}")

        environment = os.environ.copy()
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "MEGATRON_LM_REPO",
            "MEGATRON_ROOT",
        ):
            environment.pop(name, None)
        result = subprocess.run(
            [
                str(shared_python),
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "tests/test_source_processing_core_imports.py",
            ],
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(
            result.stdout + result.stderr,
            r"shared.*venv|matching cppmega environment manifest",
        )

    def test_pytest_rejects_same_inode_alias_with_drifted_receipt(self) -> None:
        repo_root = SCRIPT.parents[2]
        first = repo_root.parent / "Megatron-LM"
        second = repo_root.parent / "megatron-lm"
        if not first.is_dir() or not second.is_dir():
            self.skipTest("case aliases for Megatron-LM are absent")
        if not os.path.samefile(first, second):
            self.skipTest("Megatron-LM paths are not aliases of one inode")

        environment = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            environment.pop(name, None)
        environment["MEGATRON_LM_REPO"] = str(first)
        environment["MEGATRON_ROOT"] = str(second)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "tests/test_source_processing_core_imports.py",
            ],
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("outside the environment receipt", result.stderr)
        self.assertIn("CPPMEGA_MEGATRON_COMMIT", result.stderr)

    def test_system_site_packages_is_rejected(self) -> None:
        from cppmega_env import EnvError, ensure_target_isolation

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "cppmega-venv"
            target.mkdir()
            (target / "pyvenv.cfg").write_text(
                "include-system-site-packages = true\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EnvError, "system-site-packages"):
                ensure_target_isolation(Path(temp) / "checkout", target)

            (target / "pyvenv.cfg").write_text(
                "home = /usr/bin/python3\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EnvError, "not proven isolated"):
                ensure_target_isolation(Path(temp) / "checkout", target)

    def test_bootstrap_writes_only_source_paths_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout = root / "cppmega"
            source = root / "Megatron-LM"
            checkout.mkdir()
            (checkout / "cppmega").mkdir()
            (checkout / "cppmega" / "__init__.py").write_text("\n", encoding="utf-8")
            (checkout / "STACK.lock").write_text(
                """base:\n  python: '3.13'\n  torch: '2.13.0+cu132'\nsources:\n  megatron_lm:\n    ref: fixture\n""",
                encoding="utf-8",
            )
            commit = _make_source(source)
            target = root / "cppmega-venv"
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.pop("VIRTUAL_ENV", None)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "bootstrap",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                    "--skip-verify",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads(
                (target / "cppmega-environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["megatron_commit"], commit)
            self.assertEqual(manifest["profile"], "source")
            source_path_file = (
                target
                / "lib"
                / "python3.13"
                / "site-packages"
                / "00_cppmega_sources.pth"
            )
            self.assertEqual(
                source_path_file.read_text(encoding="utf-8").splitlines(),
                [str(checkout.resolve()), str(source.resolve())],
            )
            self.assertEqual(_git(source, "status", "--porcelain"), "")

            verify_environment = environment.copy()
            verify_environment.pop("PYTHONPATH", None)
            verify_environment.pop("VIRTUAL_ENV", None)
            verify = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=verify_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(verify.returncode, 0)
            self.assertIn("missing distribution", verify.stdout)

            polluted_environment = verify_environment.copy()
            polluted_environment["PYTHONPATH"] = "/tmp/untrusted-source"
            polluted = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=polluted_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(
                polluted.returncode, 0, polluted.stdout + polluted.stderr
            )
            self.assertIn("shell PYTHONPATH", polluted.stdout)

    def test_verify_passes_for_clean_minimal_source_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkout = root / "cppmega"
            source = root / "Megatron-LM"
            checkout.mkdir()
            (checkout / "cppmega").mkdir()
            (checkout / "cppmega" / "__init__.py").write_text("\n", encoding="utf-8")
            (checkout / "STACK.lock").write_text(
                """base:\n  python: '3.13'\n  torch: '2.13.0+cu132'\nsources:\n  megatron_lm:\n    ref: fixture\n""",
                encoding="utf-8",
            )
            commit = _make_source(source)
            target = root / "cppmega-venv"
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment.pop("VIRTUAL_ENV", None)

            bootstrap = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "bootstrap",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                    "--skip-verify",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)
            self._install_minimal_distributions(checkout, target)

            verify = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verify.returncode, 0, verify.stdout + verify.stderr)
            self.assertIn("PASS cppmega environment verification", verify.stdout)

            manifest_path = target / "cppmega-environment.json"
            site_packages = target / "lib" / "python3.13" / "site-packages"
            (site_packages / "__editable___fixture_finder.py").write_text(
                "import sys\n"
                "PATH_PLACEHOLDER = '__editable__.fixture.finder.__path_hook__'\n"
                "if PATH_PLACEHOLDER not in sys.path:\n"
                "    sys.path.append(PATH_PLACEHOLDER)\n",
                encoding="utf-8",
            )
            (
                site_packages / "98_editable.pth"
            ).write_text(
                "import __editable___fixture_finder\n",
                encoding="utf-8",
            )
            editable_verify = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                editable_verify.returncode,
                0,
                editable_verify.stdout + editable_verify.stderr,
            )

            foreign = root / "foreign-source"
            editable_foreign = foreign / "__editable__.foreign.path.__path_hook__"
            editable_foreign.mkdir(parents=True)
            (site_packages / "__editable___fixture_finder.py").write_text(
                "import sys\n"
                f"PATH_PLACEHOLDER = {str(editable_foreign)!r}\n"
                "if PATH_PLACEHOLDER not in sys.path:\n"
                "    sys.path.append(PATH_PLACEHOLDER)\n",
                encoding="utf-8",
            )
            polluted = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            debug_path = subprocess.run(
                [str(target / "bin" / "python"), "-c", "import sys; print(sys.path)"],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(
                polluted.returncode,
                0,
                debug_path.stdout + polluted.stdout + polluted.stderr,
            )
            self.assertIn(str(editable_foreign), debug_path.stdout)
            self.assertIn("unapproved editable path", polluted.stdout)

            (site_packages / "__editable___fixture_finder.py").write_text(
                "import sys\n"
                "PATH_PLACEHOLDER = '__editable__.fixture.finder.__path_hook__'\n"
                "if PATH_PLACEHOLDER not in sys.path:\n"
                "    sys.path.append(PATH_PLACEHOLDER)\n",
                encoding="utf-8",
            )

            ordinary_foreign = root / "ordinary-foreign-source"
            ordinary_foreign.mkdir()
            (
                site_packages / "99_foreign.pth"
            ).write_text(f"{ordinary_foreign}\n", encoding="utf-8")
            ordinary_polluted = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(
                ordinary_polluted.returncode,
                0,
                ordinary_polluted.stdout + ordinary_polluted.stderr,
            )
            self.assertIn("unapproved sys.path entry", ordinary_polluted.stdout)

            (site_packages / "99_foreign.pth").unlink()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_dirty"] = 0
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            malformed_receipt = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(
                malformed_receipt.returncode,
                0,
                malformed_receipt.stdout + malformed_receipt.stderr,
            )
            self.assertIn("source_dirty", malformed_receipt.stdout)

            manifest["source_dirty"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            dirty_receipt = subprocess.run(
                [
                    str(target / "bin" / "python"),
                    str(SCRIPT),
                    "verify",
                    "--repo-root",
                    str(checkout),
                    "--megatron-root",
                    str(source),
                    "--megatron-ref",
                    commit,
                    "--env",
                    str(target),
                    "--profile",
                    "source",
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(
                dirty_receipt.returncode,
                0,
                dirty_receipt.stdout + dirty_receipt.stderr,
            )
            self.assertIn("source_dirty", dirty_receipt.stdout)

    def test_staged_checkout_accepts_canonical_environment_manifest(self) -> None:
        manifest_path = Path(sys.prefix) / "cppmega-environment.json"
        if not manifest_path.is_file():
            self.skipTest(f"dedicated environment manifest is absent: {manifest_path}")

        with tempfile.TemporaryDirectory() as temp:
            staged = Path(temp) / "checkout"
            staged.mkdir()
            shutil.copy2(SCRIPT.parents[2] / "conftest.py", staged / "conftest.py")
            environment = os.environ.copy()
            for name in (
                "PYTHONHOME",
                "VIRTUAL_ENV",
                "MEGATRON_LM_REPO",
                "MEGATRON_ROOT",
            ):
                environment.pop(name, None)
            environment["PYTHONPATH"] = str(staged)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import conftest; print(conftest.MEGATRON_SOURCE_ROOT)",
                ],
                cwd=staged,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Megatron-LM", result.stdout)

    def test_source_ref_and_dirty_state_are_fail_closed(self) -> None:
        from cppmega_env import EnvError, inspect_source

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "Megatron-LM"
            first_commit = _make_source(source)
            (source / "revision.txt").write_text("second\n", encoding="utf-8")
            _git(source, "add", "revision.txt")
            _git(source, "commit", "-qm", "second")
            second_commit = _git(source, "rev-parse", "HEAD")

            with self.assertRaisesRegex(EnvError, "does not match"):
                inspect_source(
                    Path(temp) / "cppmega",
                    source,
                    first_commit,
                    allow_dirty=True,
                )

            (source / "revision.txt").write_text("modified\n", encoding="utf-8")
            with self.assertRaisesRegex(EnvError, "dirty"):
                inspect_source(
                    Path(temp) / "cppmega",
                    source,
                    second_commit,
                    allow_dirty=False,
                )

    def test_bootstrap_does_not_run_pip_install(self) -> None:
        from cppmega_env import _run_bootstrap_venv

        calls: list[list[str]] = []

        def fake_run(
            command: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch("cppmega_env.subprocess.run", side_effect=fake_run):
            _run_bootstrap_venv(Path("/tmp/base-python"), Path("/tmp/new-venv"))

        self.assertEqual(
            calls, [["/tmp/base-python", "-m", "venv", "--copies", "/tmp/new-venv"]]
        )

    def test_target_probe_timeout_is_reported_fail_closed(self) -> None:
        from cppmega_env import SourceInfo, _probe_target

        source = SourceInfo(
            root=Path("/tmp/megatron"),
            head="head",
            expected_commit="head",
            expected_ref="head",
            dirty_entries=(),
            version="0.18.0rc0",
            requires_python=">=3.12",
            dependencies=(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "cppmega"
            env_dir = root / "environment"
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "python").write_text("", encoding="utf-8")

            with (
                mock.patch("cppmega_env._stack_value", return_value="3.13"),
                mock.patch("cppmega_env._dependency_contract", return_value=()),
                mock.patch(
                    "cppmega_env.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(
                        [str(env_dir / "bin" / "python"), "-c", "probe"],
                        timeout=120,
                    ),
                ),
            ):
                ok, detail = _probe_target(repo, env_dir, source, "source")

        self.assertFalse(ok)
        self.assertIn("timed out", detail)
        self.assertIn("120", detail)


if __name__ == "__main__":
    unittest.main()
