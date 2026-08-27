"""Generate and verify hash-bound Stage 20 unified-CLI/project-doctor evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt import __version__
from minigpt.data import JsonValue
from minigpt.evidence_registry import evidence_registry
from minigpt.project_doctor import DoctorMode, verify_project

if TYPE_CHECKING:
    from collections.abc import Sequence

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "20"
_SHA256_HEX_LENGTH = 64


@dataclass(slots=True)
class Stage20EvidenceVerificationError(ValueError):
    """Report invalid Stage 20 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 20 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage20EvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> EvidenceDocument:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    document = cast("dict[object, object]", raw)
    if any(not isinstance(key, str) for key in document):
        _invalid(f"{path} must contain string keys")
    return cast("EvidenceDocument", document)


def _write_json(path: Path, document: EvidenceDocument) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _verify_lifecycle_document(document: EvidenceDocument) -> None:
    if document.get("exit_code") != 0:
        _invalid("Stage 20 lifecycle tests did not pass")
    command = document.get("command")
    if isinstance(command, str):
        command_text = command
    elif isinstance(command, list) and command and all(isinstance(item, str) for item in command):
        command_text = " ".join(cast("list[str]", command))
    else:
        _invalid("Stage 20 lifecycle command is missing or malformed")
    command_text = command_text.replace("\\", "/")
    for required in (
        "tests/test_cli.py",
        "tests/test_project_doctor.py",
        "tests/test_serving_runtime.py",
        "tests/test_serve_subprocess.py",
        "tests/test_stage19_evidence.py",
        "tests/test_http_server.py",
        "tests/test_engine_runner.py",
    ):
        if required not in command_text:
            _invalid(f"Stage 20 lifecycle evidence omitted {required}")


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    resolved_environment = os.environ.copy()
    _ = resolved_environment.pop("PYTHONPATH", None)
    if environment is not None:
        resolved_environment.update(environment)
    resolved_environment["PYTHONUTF8"] = "1"
    resolved_environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(  # noqa: S603 - fixed interpreter/repository commands
        list(command),
        cwd=cwd,
        env=resolved_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def generate_stage20_cli_evidence(root: Path, output_path: Path) -> Path:
    """Record stable help/version behavior and prove the light path avoids HTTP imports."""
    version_result = _run((sys.executable, "-m", "minigpt", "--version"), cwd=root)
    help_result = _run((sys.executable, "-m", "minigpt", "--help"), cwd=root)
    import_probe = (
        "import json,sys; import minigpt.cli; "
        "code=minigpt.cli.main(['--version']); "
        "print(json.dumps(sorted(name for name in ('fastapi','httpx','uvicorn') "
        "if name in sys.modules))); raise SystemExit(code)"
    )
    import_result = _run((sys.executable, "-c", import_probe), cwd=root)
    import_lines = import_result.stdout.splitlines()
    expected_commands = ("prepare-data", "train", "generate", "simulate", "serve", "verify")
    passed = (
        version_result.returncode == 0
        and version_result.stdout.strip() == __version__
        and help_result.returncode == 0
        and all(item in help_result.stdout for item in expected_commands)
        and import_result.returncode == 0
        and import_lines[:2] == [__version__, "[]"]
    )
    if not passed:
        _invalid("unified CLI help/version/import contract did not pass")
    return _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "project_version": __version__,
                "module_entrypoint": True,
                "stable_root_help": True,
                "version_matches_source": True,
                "lazy_optional_imports": True,
                "http_modules_loaded_by_version": [],
                "commands": list(expected_commands),
            },
        ),
    )


def generate_stage20_doctor_evidence(root: Path, output_path: Path) -> Path:
    """Run quick and CI doctor modes against the reviewed Stage 7A-19 registry."""
    registry = tuple(item for item in evidence_registry() if item.stage != "20")
    quick = verify_project(root, mode=DoctorMode.QUICK, packages=registry)
    ci = verify_project(root, mode=DoctorMode.CI, packages=registry)
    if not quick.passed or not ci.passed:
        _invalid("project doctor quick/CI mode did not pass")
    if registry[0].stage != "7A" or registry[-1].stage != "19":
        _invalid("Stage 20 registry boundary is not Stage 7A-19")
    document: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "quick_passed": True,
        "ci_passed": True,
        "registry_first_stage": registry[0].stage,
        "registry_last_stage": registry[-1].stage,
        "registry_packages": len(registry),
        "runtime_smoke_passed": any(
            item.name == "runtime-smoke" and item.status.value == "pass" for item in ci.checks
        ),
        "installed_cli_passed": any(
            item.name == "installed-cli" and item.status.value == "pass" for item in ci.checks
        ),
        "quick_report": cast("JsonValue", quick.to_document()),
        "ci_report": cast("JsonValue", ci.to_document()),
    }
    return _write_json(output_path, document)


def _fresh_venv_python(venv_root: Path) -> Path:
    return venv_root / "Scripts" / "python.exe" if os.name == "nt" else venv_root / "bin" / "python"


def _fresh_console_script(venv_root: Path) -> Path:
    return (
        venv_root / "Scripts" / "minigpt.exe" if os.name == "nt" else venv_root / "bin" / "minigpt"
    )


def generate_stage20_packaging_evidence(root: Path, output_path: Path) -> Path:
    """Build wheel/sdist and verify a dependency-free fresh CLI installation."""
    with tempfile.TemporaryDirectory(prefix="minigpt-stage20-build-") as raw_temp:
        temp_root = Path(raw_temp)
        dist_root = temp_root / "dist"
        build_result = _run(
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
            cwd=root,
        )
        wheels = sorted(dist_root.glob("*.whl"))
        sdists = sorted(dist_root.glob("*.tar.gz"))
        if build_result.returncode != 0 or len(wheels) != 1 or len(sdists) != 1:
            _invalid("wheel/sdist build did not produce exactly one artifact each")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            wheel_members = set(archive.namelist())
        required_members = {
            "minigpt/cli.py",
            "minigpt/__main__.py",
            "minigpt/project_doctor.py",
            "prepare_data.py",
            "train.py",
            "generate.py",
            "simulate_serving.py",
            "serve.py",
        }
        if not required_members.issubset(wheel_members):
            missing = sorted(required_members - wheel_members)
            _invalid("wheel is missing command modules: " + ", ".join(missing))
        venv_root = temp_root / "venv"
        create_result = _run((sys.executable, "-m", "venv", str(venv_root)), cwd=root)
        if create_result.returncode != 0:
            _invalid("fresh packaging virtual environment creation failed")
        fresh_python = _fresh_venv_python(venv_root)
        install_result = _run(
            (str(fresh_python), "-m", "pip", "install", "--no-deps", str(wheel)),
            cwd=temp_root,
        )
        location_result = _run(
            (
                str(fresh_python),
                "-c",
                "import pathlib,minigpt; print(pathlib.Path(minigpt.__file__).resolve())",
            ),
            cwd=temp_root,
        )
        metadata_result = _run(
            (
                str(fresh_python),
                "-c",
                ("import importlib.metadata; print(importlib.metadata.version('minitrain-gpt'))"),
            ),
            cwd=temp_root,
        )
        version_result = _run((str(fresh_python), "-m", "minigpt", "--version"), cwd=temp_root)
        help_result = _run((str(fresh_python), "-m", "minigpt", "--help"), cwd=temp_root)
        console = _fresh_console_script(venv_root)
        console_result = _run((str(console), "--version"), cwd=temp_root)
        console_help_result = _run((str(console), "--help"), cwd=temp_root)
        location = Path(location_result.stdout.strip()).resolve()
        if not (
            install_result.returncode == 0
            and location_result.returncode == 0
            and location.is_relative_to(venv_root.resolve())
            and metadata_result.returncode == 0
            and metadata_result.stdout.strip() == __version__
            and version_result.returncode == 0
            and version_result.stdout.strip() == __version__
            and help_result.returncode == 0
            and "verify" in help_result.stdout
            and console_result.returncode == 0
            and console_result.stdout.strip() == __version__
            and console_help_result.returncode == 0
            and "verify" in console_help_result.stdout
        ):
            _invalid("fresh wheel CLI installation contract did not pass")
        document = cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "project_version": __version__,
                "wheel_built": True,
                "sdist_built": True,
                "fresh_install_passed": True,
                "wheel_import_isolated": True,
                "installed_metadata_version_matches": True,
                "module_entrypoint_passed": True,
                "module_help_passed": True,
                "console_script_passed": True,
                "console_help_passed": True,
                "root_help_passed": True,
                "wheel_sha256": _sha256(wheel),
                "sdist_sha256": _sha256(sdists[0]),
                "required_command_modules_present": True,
            },
        )
    return _write_json(output_path, document)


def generate_stage20_evidence(  # noqa: PLR0913
    *,
    cli_path: Path,
    doctor_path: Path,
    packaging_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the exact Stage 20 evidence package and verify it immediately."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "cli.json": cli_path,
        "doctor.json": doctor_path,
        "packaging.json": packaging_path,
        "lifecycle_tests.json": lifecycle_path,
    }
    for path in inputs.values():
        if not path.is_file():
            _invalid(f"evidence input does not exist: {path}")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        _ = shutil.copyfile(path, evidence_root / name)
    cli = _read_json(evidence_root / "cli.json")
    doctor = _read_json(evidence_root / "doctor.json")
    packaging = _read_json(evidence_root / "packaging.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    required = (
        cli.get("module_entrypoint"),
        cli.get("stable_root_help"),
        cli.get("version_matches_source"),
        cli.get("lazy_optional_imports"),
        doctor.get("quick_passed"),
        doctor.get("ci_passed"),
        doctor.get("runtime_smoke_passed"),
        doctor.get("installed_cli_passed"),
        packaging.get("wheel_built"),
        packaging.get("sdist_built"),
        packaging.get("fresh_install_passed"),
        packaging.get("wheel_import_isolated"),
        packaging.get("installed_metadata_version_matches"),
        packaging.get("module_help_passed"),
        packaging.get("console_script_passed"),
        packaging.get("console_help_passed"),
        packaging.get("required_command_modules_present"),
    )
    if not all(value is True for value in required):
        _invalid("Stage 20 evidence contracts did not all pass")
    _verify_lifecycle_document(lifecycle)
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "project_version": __version__,
        "unified_cli": True,
        "lazy_optional_imports": True,
        "project_doctor_quick": True,
        "project_doctor_ci": True,
        "evidence_registry_stage7a_through_19": True,
        "source_ancestry_verified": True,
        "canonical_runtime_smoke": True,
        "wheel_built": True,
        "sdist_built": True,
        "fresh_install_passed": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
    }
    _ = _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, doctor, packaging),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _ = _write_json(
        manifest_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "source_commit": source_commit,
                "artifacts": artifacts,
            },
        ),
    )
    _ = verify_stage20_evidence(package_root)
    return package_root


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[object, object]", raw)
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
        ):
            _invalid("manifest artifact entry fields are invalid")
        if relative in expected:
            _invalid(f"duplicate manifest path {relative}")
        expected[relative] = (size, digest)
    return expected


def verify_stage20_evidence(  # noqa: C901, PLR0912
    package_root: Path,
) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, contracts, and bounded claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 20")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        _invalid("manifest source_commit must be non-empty")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        _invalid("manifest artifacts must be a list")
    expected = _manifest_entries(cast("list[object]", entries))
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        _invalid("artifact membership differs from manifest")
    for relative, (size, digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    summary = _read_json(package_root / "summary.json")
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    true_keys = (
        "unified_cli",
        "lazy_optional_imports",
        "project_doctor_quick",
        "project_doctor_ci",
        "evidence_registry_stage7a_through_19",
        "source_ancestry_verified",
        "canonical_runtime_smoke",
        "wheel_built",
        "sdist_built",
        "fresh_install_passed",
        "lifecycle_passed",
    )
    if any(summary.get(key) is not True for key in true_keys):
        _invalid("summary required contract did not pass")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("Stage 20 benchmark verdict must remain descriptive_only")
    if summary.get("wall_clock_performance_improvement") is not False:
        _invalid("Stage 20 must not claim wall-clock improvement")
    doctor = _read_json(package_root / "evidence" / "doctor.json")
    if doctor.get("registry_first_stage") != "7A" or doctor.get("registry_last_stage") != "19":
        _invalid("doctor evidence registry boundary is invalid")
    packaging = _read_json(package_root / "evidence" / "packaging.json")
    if packaging.get("project_version") != summary.get("project_version"):
        _invalid("packaging project version differs from summary")
    packaging_contracts = (
        "wheel_import_isolated",
        "installed_metadata_version_matches",
        "module_help_passed",
        "console_help_passed",
    )
    if any(packaging.get(key) is not True for key in packaging_contracts):
        _invalid("packaging isolation/help contract did not pass")
    lifecycle = _read_json(package_root / "evidence" / "lifecycle_tests.json")
    _verify_lifecycle_document(lifecycle)
    return manifest


def _readme(
    summary: EvidenceDocument,
    doctor: EvidenceDocument,
    packaging: EvidenceDocument,
) -> str:
    return (
        "\n".join(
            (
                "# Stage 20 — Installable Unified CLI + Project Doctor",
                "",
                "Stage 20 provides one lazily imported installable command boundary and an",
                "explicit Stage 7A-19 project verification registry.",
                "",
                f"Project version: {summary['project_version']}.",
                f"Registered evidence packages: {doctor['registry_packages']}.",
                f"Quick doctor passed: {doctor['quick_passed']}.",
                f"CI doctor/runtime smoke passed: {doctor['ci_passed']}.",
                f"Fresh wheel install passed: {packaging['fresh_install_passed']}.",
                "",
                "The evidence verdict is descriptive_only. This release-engineering stage",
                "makes no wall-clock performance improvement claim.",
                "",
                f"Source commit: {summary['source_commit']}.",
            )
        )
        + "\n"
    )
