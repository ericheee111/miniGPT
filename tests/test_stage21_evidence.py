from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from minigpt import __version__
from minigpt.stage21_evidence import (
    Stage21EvidenceVerificationError,
    _verify_portable_external_document,  # pyright: ignore[reportPrivateUsage]
    generate_stage21_evidence,
    generate_stage21_registry_evidence,
    generate_stage21_runtime_evidence,
    generate_stage21_version_evidence,
    verify_stage21_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.data import JsonValue


def _write(path: Path, document: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


@pytest.mark.parametrize(
    "command",
    [
        [r"C:\\review-host\\venv\\Scripts\\python.exe", "-m", "pytest"],
        ["/opt/review-host/venv/bin/python", "-m", "pytest"],
    ],
)
def test_stage21_external_gate_evidence_rejects_absolute_command_paths(
    command: list[str],
) -> None:
    with pytest.raises(Stage21EvidenceVerificationError, match="absolute host path"):
        _verify_portable_external_document(cast("JsonValue", {"command": command}))


def test_stage21_version_registry_and_runtime_use_real_release_contracts(tmp_path: Path) -> None:
    root = tmp_path.cwd()
    version_path = generate_stage21_version_evidence(root, tmp_path / "version.json")
    registry_path = generate_stage21_registry_evidence(root, tmp_path / "registry.json")
    runtime_path = generate_stage21_runtime_evidence(root, tmp_path / "runtime.json")
    version = cast(
        "dict[str, object]",
        json.loads(version_path.read_text(encoding="utf-8")),
    )
    registry = cast(
        "dict[str, object]",
        json.loads(registry_path.read_text(encoding="utf-8")),
    )
    runtime = cast(
        "dict[str, object]",
        json.loads(runtime_path.read_text(encoding="utf-8")),
    )

    assert version["release_version"] == __version__ == "1.0.0"
    assert version["version_contract_passed"] is True
    assert registry["registry_first_stage"] == "7A"
    assert registry["registry_last_stage"] == "20"
    assert registry["all_packages_passed"] is True
    assert runtime["stage18_canonical_simulation_passed"] is True
    assert runtime["stage19_real_runtime_passed"] is True
    assert runtime["runtime_resources_released"] is True


def test_stage21_package_binds_capstone_contracts_and_exact_hashes(tmp_path: Path) -> None:
    version = _write(
        tmp_path / "version.json",
        {
            "release_version": "1.0.0",
            "version_contract_passed": True,
            "dynamic_version_metadata": True,
        },
    )
    registry = _write(
        tmp_path / "registry.json",
        {
            "registry_first_stage": "7A",
            "registry_last_stage": "20",
            "registry_packages": 16,
            "all_packages_passed": True,
            "all_modern_sources_are_ancestors": True,
        },
    )
    release = _write(
        tmp_path / "release.json",
        {
            "release_version": "1.0.0",
            "release_doctor_passed": True,
            "wheel_built": True,
            "sdist_built": True,
            "fresh_install_passed": True,
            "pip_check_passed": True,
            "fresh_quick_doctor_passed": True,
        },
    )
    runtime = _write(
        tmp_path / "runtime.json",
        {
            "stage18_canonical_simulation_passed": True,
            "stage19_real_runtime_passed": True,
            "runtime_resources_released": True,
            "runtime_lazy_kv_reservation": True,
            "simulation_completed_requests": 2,
        },
    )
    exact_resume = _write(tmp_path / "exact-resume.json", {"exit_code": 0})
    lifecycle = _write(tmp_path / "lifecycle.json", {"exit_code": 0})
    full_tests = _write(tmp_path / "full-tests.json", {"exit_code": 0})
    quality = _write(tmp_path / "quality.json", {"exit_code": 0})

    package = generate_stage21_evidence(
        version_path=version,
        registry_path=registry,
        release_path=release,
        runtime_path=runtime,
        exact_resume_path=exact_resume,
        lifecycle_path=lifecycle,
        full_tests_path=full_tests,
        quality_path=quality,
        package_root=tmp_path / "package",
        source_commit="stage21-test-source",
    )
    manifest = verify_stage21_evidence(package)
    summary = cast(
        "dict[str, object]",
        json.loads((package / "summary.json").read_text(encoding="utf-8")),
    )

    assert manifest["stage"] == "21"
    assert summary["source_commit"] == "stage21-test-source"
    assert summary["release_version"] == "1.0.0"
    assert summary["project_complete"] is True
    assert summary["checkpoint_v2_exact_resume_passed"] is True
    assert summary["full_test_suite_passed"] is True
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    assert summary["production_scale_performance_claim"] is False
    assert summary["gpu_parity_claim"] is False
    readme = (package / "README.md").read_text(encoding="utf-8")
    assert readme.endswith("\n")
    assert not readme.endswith("\n\n")

    registry_copy = package / "evidence" / "registry.json"
    _ = registry_copy.write_text(
        registry_copy.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(Stage21EvidenceVerificationError, match="hash mismatch"):
        _ = verify_stage21_evidence(package)


def test_stage21_verifier_rejects_unsupported_release_claim(tmp_path: Path) -> None:
    version = _write(
        tmp_path / "version.json",
        {
            "release_version": "1.0.0",
            "version_contract_passed": True,
            "dynamic_version_metadata": True,
        },
    )
    registry = _write(
        tmp_path / "registry.json",
        {
            "registry_first_stage": "7A",
            "registry_last_stage": "20",
            "registry_packages": 16,
            "all_packages_passed": True,
            "all_modern_sources_are_ancestors": True,
        },
    )
    release = _write(
        tmp_path / "release.json",
        {
            "release_version": "1.0.0",
            "release_doctor_passed": True,
            "wheel_built": True,
            "sdist_built": True,
            "fresh_install_passed": True,
            "pip_check_passed": True,
            "fresh_quick_doctor_passed": True,
        },
    )
    runtime = _write(
        tmp_path / "runtime.json",
        {
            "stage18_canonical_simulation_passed": True,
            "stage19_real_runtime_passed": True,
            "runtime_resources_released": True,
            "runtime_lazy_kv_reservation": True,
            "simulation_completed_requests": 2,
        },
    )
    exact_resume = _write(tmp_path / "exact-resume.json", {"exit_code": 0})
    lifecycle = _write(tmp_path / "lifecycle.json", {"exit_code": 0})
    full_tests = _write(tmp_path / "full-tests.json", {"exit_code": 0})
    quality = _write(tmp_path / "quality.json", {"exit_code": 0})
    package = generate_stage21_evidence(
        version_path=version,
        registry_path=registry,
        release_path=release,
        runtime_path=runtime,
        exact_resume_path=exact_resume,
        lifecycle_path=lifecycle,
        full_tests_path=full_tests,
        quality_path=quality,
        package_root=tmp_path / "package",
        source_commit="stage21-test-source",
    )
    summary_path = package / "summary.json"
    summary = cast(
        "dict[str, object]",
        json.loads(summary_path.read_text(encoding="utf-8")),
    )
    summary["production_scale_performance_claim"] = True
    _ = summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(Stage21EvidenceVerificationError, match=r"hash mismatch|unsupported"):
        _ = verify_stage21_evidence(package)
