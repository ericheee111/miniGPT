from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from minigpt import __version__
from minigpt.project_doctor import (
    CheckStatus,
    DoctorMode,
    _check_release_artifacts,  # pyright: ignore[reportPrivateUsage]
    verify_project,
)
from minigpt.release_validation import ReleaseArtifactValidation, validate_release_artifacts

if TYPE_CHECKING:
    import pytest


def test_v1_version_has_one_authored_source_and_dynamic_metadata() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    version_source = Path("src/minigpt/_version.py").read_text(encoding="utf-8")

    assert importlib.metadata.version("minitrain-gpt") == __version__
    assert f'__version__ = "{__version__}"' in version_source
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "minigpt._version.__version__"}' in pyproject
    assert 'version = "1.0.0"' not in pyproject


def test_generated_egg_info_is_ignored_and_untracked() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "*.egg-info", "**/*.egg-info/**"],  # noqa: S607
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert "*.egg-info/" in ignored
    assert tracked.returncode == 0
    assert tracked.stdout.strip() == ""


def test_release_artifacts_build_and_fresh_install() -> None:
    result = validate_release_artifacts(Path.cwd())

    assert result.project_version == __version__
    assert result.wheel_built
    assert result.sdist_built
    assert result.required_modules_present
    assert result.fresh_install_passed
    assert result.pip_check_passed
    assert result.metadata_version_passed
    assert result.wheel_import_isolated
    assert result.module_entrypoint_passed
    assert result.module_help_passed
    assert result.console_script_passed
    assert result.console_help_passed
    assert result.quick_doctor_passed
    assert len(result.wheel_sha256) == 64
    assert len(result.sdist_sha256) == 64


def test_release_doctor_mode_invokes_artifact_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ReleaseArtifactValidation(
        project_version=__version__,
        wheel_sha256="a" * 64,
        sdist_sha256="b" * 64,
        wheel_built=True,
        sdist_built=True,
        required_modules_present=True,
        fresh_install_passed=True,
        pip_check_passed=True,
        metadata_version_passed=True,
        wheel_import_isolated=True,
        module_entrypoint_passed=True,
        module_help_passed=True,
        console_script_passed=True,
        console_help_passed=True,
        quick_doctor_passed=True,
    )

    def fake_validate(_root: Path) -> ReleaseArtifactValidation:
        return fake

    monkeypatch.setattr("minigpt.release_validation.validate_release_artifacts", fake_validate)

    result = _check_release_artifacts(Path.cwd())

    assert result.status is CheckStatus.PASS
    assert __version__ in result.detail
    assert "aaaaaaaaaaaa" in result.detail


def test_release_report_extends_ci_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ReleaseArtifactValidation(
        project_version=__version__,
        wheel_sha256="c" * 64,
        sdist_sha256="d" * 64,
        wheel_built=True,
        sdist_built=True,
        required_modules_present=True,
        fresh_install_passed=True,
        pip_check_passed=True,
        metadata_version_passed=True,
        wheel_import_isolated=True,
        module_entrypoint_passed=True,
        module_help_passed=True,
        console_script_passed=True,
        console_help_passed=True,
        quick_doctor_passed=True,
    )

    def fake_validate(_root: Path) -> ReleaseArtifactValidation:
        return fake

    monkeypatch.setattr("minigpt.release_validation.validate_release_artifacts", fake_validate)

    report = verify_project(Path.cwd(), mode=DoctorMode.RELEASE)

    assert report.passed
    names = {item.name for item in report.checks}
    assert {"runtime-smoke", "installed-cli", "release-artifacts"}.issubset(names)
