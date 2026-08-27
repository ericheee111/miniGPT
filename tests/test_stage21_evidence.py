from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from minigpt import __version__
from minigpt.stage21_evidence import (
    Stage21EvidenceVerificationError,
    _verify_full_test_file_coverage,  # pyright: ignore[reportPrivateUsage]
    _verify_portable_external_document,  # pyright: ignore[reportPrivateUsage]
    generate_stage21_evidence,
    generate_stage21_registry_evidence,
    generate_stage21_runtime_evidence,
    generate_stage21_version_evidence,
    verify_stage21_evidence,
)

if TYPE_CHECKING:
    from minigpt.data import JsonValue


def _write(path: Path, document: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _gate_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    exact_resume = _write(
        tmp_path / "exact-resume.json",
        {
            "exit_code": 0,
            "results": [
                {
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_checkpoint.py",
                        "tests/test_trainer.py",
                        "tests/test_training_components.py",
                    ],
                    "exit_code": 0,
                }
            ],
        },
    )
    lifecycle = _write(
        tmp_path / "lifecycle.json",
        {
            "exit_code": 0,
            "results": [
                {
                    "command": [
                        "python",
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_serving_runtime.py",
                        "tests/test_serve_subprocess.py",
                        "tests/test_cli.py",
                        "tests/test_project_doctor.py",
                        "tests/test_release_validation.py",
                        "tests/test_stage19_evidence.py",
                        "tests/test_stage20_evidence.py",
                        "tests/test_stage21_evidence.py",
                    ],
                    "exit_code": 0,
                }
            ],
        },
    )
    full_tests = _write(
        tmp_path / "full-tests.json",
        {
            "exit_code": 0,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "partitions": [
                {
                    "command": ["python", "-m", "pytest", "-q", "tests/test_model.py"],
                    "exit_code": 0,
                }
            ],
        },
    )
    quality = _write(
        tmp_path / "quality.json",
        {
            "exit_code": 0,
            "results": [
                {"command": ["python", "-m", "pip", "check"], "exit_code": 0},
                {
                    "command": ["python", "-m", "ruff", "format", "--check", "src", "tests"],
                    "exit_code": 0,
                },
                {
                    "command": ["python", "-m", "ruff", "check", "src", "tests"],
                    "exit_code": 0,
                },
                {"command": ["basedpyright"], "exit_code": 0},
            ],
        },
    )
    return exact_resume, lifecycle, full_tests, quality


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
            "metadata_version_passed": True,
            "wheel_import_isolated": True,
            "module_entrypoint_passed": True,
            "module_help_passed": True,
            "console_script_passed": True,
            "console_help_passed": True,
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
    exact_resume, lifecycle, full_tests, quality = _gate_inputs(tmp_path)

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

    with pytest.raises(Stage21EvidenceVerificationError, match="source commit is unavailable"):
        _ = verify_stage21_evidence(package, repository_root=Path.cwd())

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
            "metadata_version_passed": True,
            "wheel_import_isolated": True,
            "module_entrypoint_passed": True,
            "module_help_passed": True,
            "console_script_passed": True,
            "console_help_passed": True,
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
    exact_resume, lifecycle, full_tests, quality = _gate_inputs(tmp_path)
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


def test_stage21_verification_error_allows_traceback_assignment() -> None:
    error = Stage21EvidenceVerificationError("injected")

    error.__traceback__ = None

    assert error.__traceback__ is None


def test_committed_stage21_evidence_binds_source_ancestry_and_full_test_coverage() -> None:
    manifest = verify_stage21_evidence(
        Path("docs/results/v1-release"),
        repository_root=Path.cwd(),
    )

    assert manifest["stage"] == "21"


def test_stage21_repository_context_rejects_partial_full_test_coverage(tmp_path: Path) -> None:
    _exact_resume, _lifecycle, full_tests_path, _quality = _gate_inputs(tmp_path)
    full_tests = cast(
        "dict[str, object]",
        json.loads(full_tests_path.read_text(encoding="utf-8")),
    )

    with pytest.raises(Stage21EvidenceVerificationError, match="file coverage differs"):
        _verify_full_test_file_coverage(cast("dict[str, JsonValue]", full_tests), Path.cwd())
