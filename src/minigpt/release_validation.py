"""Build and fresh-install miniGPT release artifacts under deterministic checks."""

from __future__ import annotations

import hashlib
import json
import os
import site
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from minigpt._version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ReleaseArtifactValidation:
    """Summarize a successful wheel/sdist build and fresh-install verification."""

    project_version: str
    wheel_sha256: str
    sdist_sha256: str
    wheel_built: bool
    sdist_built: bool
    required_modules_present: bool
    fresh_install_passed: bool
    pip_check_passed: bool
    metadata_version_passed: bool
    wheel_import_isolated: bool
    module_entrypoint_passed: bool
    module_help_passed: bool
    console_script_passed: bool
    console_help_passed: bool
    quick_doctor_passed: bool

    def to_document(self) -> dict[str, object]:
        """Return a stable report without temporary paths or timings."""
        return {
            "project_version": self.project_version,
            "wheel_sha256": self.wheel_sha256,
            "sdist_sha256": self.sdist_sha256,
            "wheel_built": self.wheel_built,
            "sdist_built": self.sdist_built,
            "required_modules_present": self.required_modules_present,
            "fresh_install_passed": self.fresh_install_passed,
            "pip_check_passed": self.pip_check_passed,
            "metadata_version_passed": self.metadata_version_passed,
            "wheel_import_isolated": self.wheel_import_isolated,
            "module_entrypoint_passed": self.module_entrypoint_passed,
            "module_help_passed": self.module_help_passed,
            "console_script_passed": self.console_script_passed,
            "console_help_passed": self.console_help_passed,
            "quick_doctor_passed": self.quick_doctor_passed,
        }


class ReleaseValidationError(RuntimeError):
    """Report a failed release artifact or fresh-install contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    pythonpath: Sequence[Path] = (),
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    _ = environment.pop("PYTHONPATH", None)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    if pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    return subprocess.run(  # noqa: S603 - fixed release-validation commands
        list(command),
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _doctor_failure_detail(stdout: str) -> str | None:
    if not stdout.strip().startswith("{"):
        return None
    try:
        raw = cast("object", json.loads(stdout))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    checks = cast("dict[str, object]", raw).get("checks")
    if not isinstance(checks, list):
        return None
    failures: list[str] = []
    for raw_check in cast("list[object]", checks):
        if not isinstance(raw_check, dict):
            continue
        check = cast("dict[str, object]", raw_check)
        if check.get("status") != "fail":
            continue
        name = check.get("name")
        detail = check.get("detail")
        if isinstance(name, str) and isinstance(detail, str):
            failures.append(f"{name}: {detail}")
    return "; ".join(failures) if failures else None


def _require_success(result: subprocess.CompletedProcess[str], context: str) -> None:
    if result.returncode != 0:
        structured = _doctor_failure_detail(result.stdout)
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = structured or (detail[-1] if detail else "no subprocess output")
        message = f"{context} failed: {suffix}"
        raise ReleaseValidationError(message)


def _venv_python(venv_root: Path) -> Path:
    return venv_root / "Scripts" / "python.exe" if os.name == "nt" else venv_root / "bin" / "python"


def _console_script(venv_root: Path) -> Path:
    return (
        venv_root / "Scripts" / "minigpt.exe" if os.name == "nt" else venv_root / "bin" / "minigpt"
    )


def _validate_fresh_wheel(
    wheel: Path,
    *,
    repository_root: Path,
    temp_root: Path,
) -> None:
    dependency_paths = tuple(Path(item).resolve() for item in site.getsitepackages())
    venv_root = temp_root / "venv"
    create = _run(
        (sys.executable, "-m", "venv", str(venv_root)),
        cwd=repository_root,
    )
    _require_success(create, "fresh virtual environment creation")
    python = _venv_python(venv_root)
    install = _run(
        (str(python), "-m", "pip", "install", "--no-deps", str(wheel)),
        cwd=repository_root,
    )
    _require_success(install, "fresh wheel installation")
    pip_check = _run(
        (str(python), "-m", "pip", "check"),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(pip_check, "fresh pip check")
    package_location = _run(
        (
            str(python),
            "-c",
            "import pathlib,minigpt; print(pathlib.Path(minigpt.__file__).resolve())",
        ),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(package_location, "fresh installed package location")
    installed_path = Path(package_location.stdout.strip()).resolve()
    if not installed_path.is_relative_to(venv_root.resolve()):
        reason = "fresh CLI imported minigpt outside the wheel installation environment"
        raise ReleaseValidationError(reason)
    metadata_version = _run(
        (
            str(python),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('minitrain-gpt'))",
        ),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(metadata_version, "fresh distribution metadata version")
    module_version = _run(
        (str(python), "-m", "minigpt", "--version"),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(module_version, "fresh module version")
    module_help = _run(
        (str(python), "-m", "minigpt", "--help"),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(module_help, "fresh module help")
    console = _console_script(venv_root)
    console_version = _run(
        (str(console), "--version"),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(console_version, "fresh console-script version")
    console_help = _run(
        (str(console), "--help"),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(console_help, "fresh console-script help")
    quick_doctor = _run(
        (
            str(python),
            "-m",
            "minigpt",
            "verify",
            "--root",
            str(repository_root),
            "--mode",
            "quick",
        ),
        cwd=temp_root,
        pythonpath=dependency_paths,
    )
    _require_success(quick_doctor, "fresh installed quick doctor")
    if (
        metadata_version.stdout.strip() != __version__
        or module_version.stdout.strip() != __version__
        or console_version.stdout.strip() != __version__
        or "verify" not in module_help.stdout
        or "verify" not in console_help.stdout
    ):
        reason = "fresh installed CLI version/help differs from release source"
        raise ReleaseValidationError(reason)


def validate_release_artifacts(root: Path) -> ReleaseArtifactValidation:
    """Build wheel/sdist, inspect contents, and verify a fresh installed release."""
    resolved = root.resolve()
    with tempfile.TemporaryDirectory(prefix="minigpt-release-") as raw_temp:
        temp_root = Path(raw_temp)
        dist_root = temp_root / "dist"
        build = _run(
            (
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--wheel",
                "--sdist",
                "--outdir",
                str(dist_root),
            ),
            cwd=resolved,
        )
        _require_success(build, "wheel/sdist build")
        wheels = sorted(dist_root.glob("*.whl"))
        sdists = sorted(dist_root.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            reason = "release build must produce exactly one wheel and one sdist"
            raise ReleaseValidationError(reason)
        wheel = wheels[0]
        sdist = sdists[0]
        with zipfile.ZipFile(wheel) as archive:
            members = set(archive.namelist())
        required_members = {
            "minigpt/__main__.py",
            "minigpt/_version.py",
            "minigpt/cli.py",
            "minigpt/evidence_registry.py",
            "minigpt/project_doctor.py",
            "minigpt/public_demo.py",
            "minigpt/prediction.py",
            "minigpt/release_validation.py",
            "minigpt/serving_runtime.py",
            "minigpt/story.py",
            "minigpt/story_data.py",
            "minigpt/story_evaluation.py",
            "minigpt/story_forge_evidence.py",
            "minigpt/story_forge_product.py",
            "minigpt/story_forge_product_evidence.py",
            "minigpt/story_forge_systems.py",
            "minigpt/tokenizer.py",
            "minigpt/stage19_evidence.py",
            "minigpt/stage20_evidence.py",
            "minigpt/stage21_evidence.py",
            "prepare_data.py",
            "prepare_stories.py",
            "evaluate_stories.py",
            "train.py",
            "generate.py",
            "simulate_serving.py",
            "serve.py",
        }
        missing = sorted(required_members - members)
        if missing:
            raise ReleaseValidationError("wheel is missing required modules: " + ", ".join(missing))
        _validate_fresh_wheel(wheel, repository_root=resolved, temp_root=temp_root)
        return ReleaseArtifactValidation(
            project_version=__version__,
            wheel_sha256=_sha256(wheel),
            sdist_sha256=_sha256(sdist),
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
