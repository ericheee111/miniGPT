from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from minigpt import __version__, project_doctor
from minigpt.evidence_registry import (
    EvidencePackage,
    _verify_legacy_manifest,  # pyright: ignore[reportPrivateUsage]
    evidence_registry,
)
from minigpt.project_doctor import (
    CheckResult,
    CheckStatus,
    DoctorMode,
    DoctorReport,
    _check_ancestry,  # pyright: ignore[reportPrivateUsage]
    _check_clean,  # pyright: ignore[reportPrivateUsage]
    _check_version,  # pyright: ignore[reportPrivateUsage]
    main,
    verify_project,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _pre_stage20_registry() -> tuple[EvidencePackage, ...]:
    return tuple(item for item in evidence_registry() if item.stage != "20")


def test_registry_is_explicit_and_complete_through_stage20() -> None:
    registry = evidence_registry()

    assert tuple(item.stage for item in registry) == (
        "7A",
        "8",
        "9",
        "10",
        "11A",
        "11B",
        "12",
        "13A",
        "13B",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
    )
    assert len({item.slug for item in registry}) == len(registry)
    assert all(not item.relative_root.is_absolute() for item in registry)


def test_stage7_legacy_contract_allows_only_declared_external_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    package = root / "docs" / "results" / "reference-training"
    package.mkdir(parents=True)
    report = package / "README.md"
    _ = report.write_text("reference evidence\n", encoding="utf-8", newline="\n")
    _write_json(
        package / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "path": "docs/results/reference-training/README.md",
                    "bytes": report.stat().st_size,
                    "sha256": _sha256(report),
                }
            ],
            "sources": {
                "checkpoint": {
                    "path": "checkpoints/reference.pt",
                    "bytes": 123,
                    "sha256": "0" * 64,
                }
            },
        },
    )

    result = _verify_legacy_manifest(
        package,
        stage="7A",
        allow_external_checkpoint=True,
    )

    assert result["verified_artifacts"] == 1
    assert result["external_artifacts"] == 1


def test_stage7_legacy_manifest_rejects_traversal_external_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package = root / "docs" / "results" / "reference-training"
    package.mkdir(parents=True)
    report = package / "README.md"
    _ = report.write_text("reference evidence\n", encoding="utf-8", newline="\n")
    _write_json(
        package / "artifact_manifest.json",
        {
            "artifacts": [
                {
                    "path": "README.md",
                    "bytes": report.stat().st_size,
                    "sha256": _sha256(report),
                },
                {
                    "path": "../../escape.pt",
                    "external": True,
                    "bytes": 123,
                    "sha256": "0" * 64,
                },
            ]
        },
    )

    with pytest.raises(ValueError, match="must stay inside"):
        _ = _verify_legacy_manifest(
            package,
            stage="7A",
            allow_external_checkpoint=True,
        )


def test_legacy_manifest_rejects_tamper_and_unlisted_files(tmp_path: Path) -> None:
    package = tmp_path / "legacy"
    package.mkdir()
    artifact = package / "result.json"
    _ = artifact.write_text("{}\n", encoding="utf-8", newline="\n")
    _write_json(
        package / "artifact_manifest.json",
        {
            "files": [
                {
                    "relative_path": "result.json",
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                }
            ]
        },
    )

    first = _verify_legacy_manifest(package, stage="8", allow_external_checkpoint=False)
    assert first["verified_artifacts"] == 1

    _ = artifact.write_text('{"tampered": true}\n', encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=r"size mismatch|hash mismatch"):
        _ = _verify_legacy_manifest(package, stage="8", allow_external_checkpoint=False)

    _ = artifact.write_text("{}\n", encoding="utf-8", newline="\n")
    _ = (package / "unlisted.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="membership differs"):
        _ = _verify_legacy_manifest(package, stage="8", allow_external_checkpoint=False)

    _ = (package / "unlisted.txt").unlink()
    _write_json(
        package / "artifact_manifest.json",
        {
            "files": [
                {
                    "relative_path": "result.json",
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                },
                {
                    "relative_path": "result.json",
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="duplicate manifest artifact"):
        _ = _verify_legacy_manifest(package, stage="8", allow_external_checkpoint=False)


def test_ancestry_rejects_shallow_or_non_ancestor_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def shallow(_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        assert arguments == ("rev-parse", "--is-shallow-repository")
        return subprocess.CompletedProcess(["git"], 0, stdout="true\n", stderr="")

    monkeypatch.setattr("minigpt.project_doctor._run_git", shallow)
    assert "shallow" in cast("str", _check_ancestry(tmp_path, "a" * 40))

    def non_ancestor(_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "rev-parse":
            return subprocess.CompletedProcess(["git"], 0, stdout="false\n", stderr="")
        if arguments[0] == "cat-file":
            return subprocess.CompletedProcess(["git"], 0, stdout="", stderr="")
        return subprocess.CompletedProcess(["git"], 1, stdout="", stderr="")

    monkeypatch.setattr("minigpt.project_doctor._run_git", non_ancestor)
    assert "not an ancestor" in cast("str", _check_ancestry(tmp_path, "b" * 40))


def test_explicit_squash_merge_provenance_allows_review_source() -> None:
    source = "f55d4dda24cdb632b05ac4ee8bee75bb4378f6a3"
    merged = "6c46cf4ee0087333dad1e82ae7a388a8dabadfd7"

    assert (
        _check_ancestry(
            Path.cwd(),
            source,
            reviewed_source_commit=source,
            merged_commit=merged,
        )
        is None
    )


def test_squash_merge_provenance_rejects_unreviewed_source() -> None:
    reviewed = "f55d4dda24cdb632b05ac4ee8bee75bb4378f6a3"
    merged = "6c46cf4ee0087333dad1e82ae7a388a8dabadfd7"
    wrong_source = "598e90dcc405e1bdf66f37edd5da5e1838984c3a"

    result = _check_ancestry(
        Path.cwd(),
        wrong_source,
        reviewed_source_commit=reviewed,
        merged_commit=merged,
    )

    assert result is not None
    assert "does not match reviewed squash source" in result


def test_version_drift_is_a_hard_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("minigpt.project_doctor._installed_version", lambda: "999.0.0")

    result = _check_version()

    assert result.status is CheckStatus.FAIL
    assert "differs" in result.detail


def test_clean_requirement_reports_dirty_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def dirty(_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        assert arguments[:2] == ("status", "--porcelain")
        return subprocess.CompletedProcess(
            ["git"], 0, stdout=" M README.md\n?? extra.txt\n", stderr=""
        )

    monkeypatch.setattr("minigpt.project_doctor._run_git", dirty)

    result = _check_clean(tmp_path)

    assert result.status is CheckStatus.FAIL
    assert "2 changed path" in result.detail


def test_report_json_is_stable_and_path_independent() -> None:
    report = DoctorReport(
        mode=DoctorMode.QUICK,
        project_version=__version__,
        checks=(CheckResult("one", CheckStatus.PASS, "stable"),),
    )

    first = json.dumps(report.to_document(), sort_keys=True)
    second = json.dumps(report.to_document(), sort_keys=True)

    assert first == second
    assert str(Path.cwd()) not in first
    assert json.loads(first)["repository_root"] == "."


def test_actual_repository_quick_doctor_passes_stage7a_through_19() -> None:
    report = verify_project(
        Path.cwd(),
        mode=DoctorMode.QUICK,
        packages=_pre_stage20_registry(),
    )

    assert report.passed
    names = {item.name for item in report.checks}
    assert "evidence-stage-7A" in names
    assert "evidence-stage-19" in names
    assert "evidence-stage-20" not in names
    assert "canonical-configs" in names


def test_actual_repository_ci_doctor_runs_runtime_smoke() -> None:
    report = verify_project(
        Path.cwd(),
        mode=DoctorMode.CI,
        packages=_pre_stage20_registry(),
    )

    assert report.passed
    by_name = {item.name: item for item in report.checks}
    assert by_name["runtime-smoke"].status is CheckStatus.PASS
    assert by_name["installed-cli"].status is CheckStatus.PASS


def test_cli_writes_stable_failure_or_success_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(project_doctor, "evidence_registry", _pre_stage20_registry)

    exit_code = main(["--root", ".", "--mode", "quick", "--output", str(output)])
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    assert exit_code == 0
    assert document["passed"] is True
    assert document["repository_root"] == "."
    assert document["project_version"] == __version__


def test_custom_package_failure_is_reported_without_traceback(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package_root = root / "docs" / "results" / "broken"
    package_root.mkdir(parents=True)

    def fail_verifier(_path: Path) -> dict[str, object]:
        reason = "injected verifier failure"
        raise ValueError(reason)

    package = EvidencePackage("X", "broken", Path("docs/results/broken"), fail_verifier)

    result = project_doctor._check_evidence_package(  # pyright: ignore[reportPrivateUsage]
        root,
        package,
    )

    assert result.status is CheckStatus.FAIL
    assert "injected verifier failure" in result.detail
