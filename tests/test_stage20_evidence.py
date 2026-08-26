from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from minigpt import __version__
from minigpt.stage20_evidence import (
    Stage20EvidenceVerificationError,
    generate_stage20_cli_evidence,
    generate_stage20_doctor_evidence,
    generate_stage20_evidence,
    generate_stage20_packaging_evidence,
    verify_stage20_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, document: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_stage20_cli_and_doctor_evidence_use_real_repository_contracts(tmp_path: Path) -> None:
    cli = generate_stage20_cli_evidence(tmp_path.cwd(), tmp_path / "cli.json")
    doctor = generate_stage20_doctor_evidence(tmp_path.cwd(), tmp_path / "doctor.json")
    cli_document = cast(
        "dict[str, object]",
        json.loads(cli.read_text(encoding="utf-8")),
    )
    doctor_document = cast(
        "dict[str, object]",
        json.loads(doctor.read_text(encoding="utf-8")),
    )

    assert cli_document["project_version"] == __version__
    assert cli_document["lazy_optional_imports"] is True
    assert doctor_document["quick_passed"] is True
    assert doctor_document["ci_passed"] is True
    assert doctor_document["registry_first_stage"] == "7A"
    assert doctor_document["registry_last_stage"] == "19"


def test_stage20_packaging_builds_and_fresh_installs_cli(tmp_path: Path) -> None:
    output = generate_stage20_packaging_evidence(tmp_path.cwd(), tmp_path / "packaging.json")
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    assert document["project_version"] == __version__
    assert document["wheel_built"] is True
    assert document["sdist_built"] is True
    assert document["fresh_install_passed"] is True
    assert document["console_script_passed"] is True


def test_stage20_package_binds_contracts_and_exact_hashes(tmp_path: Path) -> None:
    cli = _write(
        tmp_path / "cli.json",
        {
            "module_entrypoint": True,
            "stable_root_help": True,
            "version_matches_source": True,
            "lazy_optional_imports": True,
        },
    )
    doctor = _write(
        tmp_path / "doctor.json",
        {
            "quick_passed": True,
            "ci_passed": True,
            "runtime_smoke_passed": True,
            "installed_cli_passed": True,
            "registry_first_stage": "7A",
            "registry_last_stage": "19",
            "registry_packages": 15,
        },
    )
    packaging = _write(
        tmp_path / "packaging.json",
        {
            "project_version": __version__,
            "wheel_built": True,
            "sdist_built": True,
            "fresh_install_passed": True,
            "console_script_passed": True,
            "required_command_modules_present": True,
        },
    )
    lifecycle = _write(tmp_path / "lifecycle.json", {"exit_code": 0})

    package = generate_stage20_evidence(
        cli_path=cli,
        doctor_path=doctor,
        packaging_path=packaging,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage20-test-source",
    )
    manifest = verify_stage20_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    assert manifest["stage"] == "20"
    assert summary["source_commit"] == "stage20-test-source"
    assert summary["unified_cli"] is True
    assert summary["project_doctor_ci"] is True
    assert summary["fresh_install_passed"] is True
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert readme.endswith("\n")
    assert not readme.endswith("\n\n")

    cli_copy = package / "evidence" / "cli.json"
    _ = cli_copy.write_text(
        cli_copy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(Stage20EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage20_evidence(package)


def test_stage20_verifier_rejects_performance_claim(tmp_path: Path) -> None:
    cli = _write(
        tmp_path / "cli.json",
        {
            "module_entrypoint": True,
            "stable_root_help": True,
            "version_matches_source": True,
            "lazy_optional_imports": True,
        },
    )
    doctor = _write(
        tmp_path / "doctor.json",
        {
            "quick_passed": True,
            "ci_passed": True,
            "runtime_smoke_passed": True,
            "installed_cli_passed": True,
            "registry_first_stage": "7A",
            "registry_last_stage": "19",
            "registry_packages": 15,
        },
    )
    packaging = _write(
        tmp_path / "packaging.json",
        {
            "project_version": __version__,
            "wheel_built": True,
            "sdist_built": True,
            "fresh_install_passed": True,
            "console_script_passed": True,
            "required_command_modules_present": True,
        },
    )
    lifecycle = _write(tmp_path / "lifecycle.json", {"exit_code": 0})
    package = generate_stage20_evidence(
        cli_path=cli,
        doctor_path=doctor,
        packaging_path=packaging,
        lifecycle_path=lifecycle,
        package_root=tmp_path / "package",
        source_commit="stage20-test-source",
    )
    summary_path = package / "summary.json"
    summary = cast(
        "dict[str, object]",
        json.loads(summary_path.read_text(encoding="utf-8")),
    )
    summary["wall_clock_performance_improvement"] = True
    _ = summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(Stage20EvidenceVerificationError, match=r"hash mismatch|must not claim"):
        _ = verify_stage20_evidence(package)
