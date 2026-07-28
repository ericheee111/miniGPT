"""Test guarded baseline and candidate comparison for Benchmark v2 evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from minigpt.benchmark_v2_compare import compare_runs, compare_step_times, write_comparison

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue


_FIXTURES = Path(__file__).parent / "fixtures" / "benchmark_v2"
_SUMMARY_FIELDS = (
    "case_identity",
    "case_name",
    "replicate_count",
    "success_count",
    "failure_count",
    "median_step_time_ms",
    "min_step_time_ms",
    "max_step_time_ms",
    "population_stddev_step_time_ms",
    "median_absolute_deviation_step_time_ms",
    "coefficient_of_variation_percent",
    "median_tokens_per_second",
    "median_final_rss_mib",
    "max_peak_rss_mib",
    "stability",
)
_CASE_IDENTITY = "a" * 64


def _hash_entry(path: Path) -> dict[str, JsonValue]:
    """Return one strict manifest binding for a small fixture artifact."""
    content = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _write_run_package(  # noqa: PLR0913
    parent: Path,
    name: str,
    *,
    status: str = "complete",
    environment_updates: dict[str, JsonValue] | None = None,
    summary_updates: dict[str, str] | None = None,
    config_sha256: str = "b" * 64,
    threshold_percent: float = 5.0,
    case_identity: str = _CASE_IDENTITY,
) -> Path:
    """Create a hand-checked, hash-bound synthetic package without real timing claims."""
    run_directory = parent / name
    run_directory.mkdir()
    run_environment: dict[str, JsonValue] = {
        "captured_before_first_worker": True,
        "git": {"commit_sha": "c" * 40, "branch": "main", "dirty": False},
        "platform": "test-platform",
        "machine": "x86_64",
        "cpu_name": "test-cpu",
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "python_version": "3.14.0",
        "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.0",
        "cuda_available": False,
        "parent_torch_num_threads": 1,
        "parent_torch_num_interop_threads": 1,
        "configured_torch_num_interop_threads": 1,
        "configured_cpu_affinity": None,
        "relevant_environment_variables": {"OMP_NUM_THREADS": "1"},
        "process_priority": "normal",
        "power_scheme": {"value": "balanced", "reason": None},
    }
    environment: dict[str, JsonValue] = {
        "schema_version": 2,
        "run_id": name,
        "run_status": status,
        "config_sha256": config_sha256,
        "run_environment": run_environment,
        "worker_environments": [],
    }
    if environment_updates is not None:
        run_environment.update(environment_updates)
    environment_path = run_directory / "environment.json"
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    config = {
        "schema_version": 2,
        "regression_threshold_percent": threshold_percent,
    }
    _ = (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n"
    )
    _ = (run_directory / "raw_replicates.jsonl").write_text("", encoding="utf-8", newline="\n")
    summary: dict[str, str] = {
        "case_identity": case_identity,
        "case_name": "display-name-can-change",
        "replicate_count": "3",
        "success_count": "3",
        "failure_count": "0",
        "median_step_time_ms": "100.0",
        "min_step_time_ms": "99.0",
        "max_step_time_ms": "101.0",
        "population_stddev_step_time_ms": "0.8",
        "median_absolute_deviation_step_time_ms": "1.0",
        "coefficient_of_variation_percent": "0.8",
        "median_tokens_per_second": "1000.0",
        "median_final_rss_mib": "100.0",
        "max_peak_rss_mib": "120.0",
        "stability": "stable",
    }
    if summary_updates is not None:
        summary.update(summary_updates)
    summary_path = run_directory / "summary.csv"
    _ = summary_path.write_text(
        ",".join(_SUMMARY_FIELDS)
        + "\n"
        + ",".join(summary[field] for field in _SUMMARY_FIELDS)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _ = (run_directory / "summary.md").write_text("# fixture\n", encoding="utf-8", newline="\n")
    _ = (run_directory / "execution_order.json").write_text("[]\n", encoding="utf-8", newline="\n")
    artifact_names = (
        "environment.json",
        "resolved_config.yaml",
        "raw_replicates.jsonl",
        "summary.csv",
        "summary.md",
        "execution_order.json",
    )
    manifest: dict[str, JsonValue] = {
        "schema_version": 2,
        "run_id": name,
        "status": status,
        "started_at_utc": "2026-07-28T01:02:03+00:00",
        "ended_at_utc": "2026-07-28T01:02:04+00:00",
        "config_sha256": config_sha256,
        "git": cast("dict[str, JsonValue]", run_environment["git"]),
        "expected_task_count": 3,
        "completed_task_count": 3,
        "successful_task_count": 3,
        "failed_task_count": 0,
        "case_identities": [
            {"case_name": "display-name-can-change", "case_identity": case_identity}
        ],
        "artifacts": [
            _hash_entry(run_directory / artifact_name) for artifact_name in artifact_names
        ],
    }
    manifest_path = run_directory / "run_manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest_path


def test_compare_runs_aligns_identity_not_display_name_and_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    """Compare aligned stable fixtures even when a human-facing case label changes."""
    # Given: two complete, hash-bound runs with one matching workload identity and different labels.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        summary_updates={
            "case_name": "renamed-case",
            "median_step_time_ms": "105.0",
            "median_tokens_per_second": "952.3809523809524",
        },
    )

    # When: the candidate is compared to the baseline and materialized beside its manifest.
    comparison = compare_runs(baseline, candidate)
    artifacts = write_comparison(comparison)

    # Then: equality passes and outputs are immutable-source-safe siblings of the candidate run.
    assert comparison.verdict == "pass"
    assert comparison.case_comparisons[0].case_identity == _CASE_IDENTITY
    assert comparison.case_comparisons[0].regressed is False
    assert artifacts.json_path == candidate.parent.parent / "candidate-comparison-baseline.json"
    assert artifacts.markdown_path == candidate.parent.parent / "candidate-comparison-baseline.md"
    assert '"verdict": "pass"' in artifacts.json_path.read_text(encoding="utf-8")
    assert "| renamed-case |" in artifacts.markdown_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [
        ("cpu_name", "different-cpu"),
        ("python_version", "3.13.0"),
        ("torch_version", "2.14.0+cpu"),
        ("numpy_version", "2.4.0"),
        ("power_scheme", {"value": "high performance", "reason": None}),
        ("relevant_environment_variables", {"OMP_NUM_THREADS": "2"}),
    ],
)
def test_compare_runs_reports_performance_environment_mismatches_but_keeps_deltas(
    tmp_path: Path, field: str, candidate_value: JsonValue
) -> None:
    """Refuse a verdict after any performance-relevant environment mismatch."""
    # Given: one otherwise aligned candidate whose supplied compatibility field differs.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        environment_updates={field: candidate_value},
        summary_updates={"median_step_time_ms": "110.0", "median_tokens_per_second": "909.0"},
    )

    # When: comparison checks the complete environment evidence.
    comparison = compare_runs(baseline, candidate)

    # Then: a named mismatch blocks pass/fail while retaining the observed descriptive time delta.
    assert comparison.verdict == "not_comparable"
    assert comparison.environment_mismatches[0].field == field
    assert comparison.case_comparisons[0].step_time_change_percent == pytest.approx(10.0)
    assert comparison.case_comparisons[0].regressed is None


def test_compare_runs_ignores_git_provenance_differences(tmp_path: Path) -> None:
    """Allow code-provenance differences when all performance compatibility fields match."""
    # Given: aligned runs that intentionally have different commit SHA, branch, and dirty evidence.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    environment_path = candidate.parent / "environment.json"
    environment = cast(
        "dict[str, object]", json.loads(environment_path.read_text(encoding="utf-8"))
    )
    run_environment = cast("dict[str, object]", environment["run_environment"])
    run_environment["git"] = {"commit_sha": "d" * 40, "branch": "feature", "dirty": True}
    _ = environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    manifest_path = candidate
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["git"] = run_environment["git"]
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_entry.update(_hash_entry(environment_path))
    _ = manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    # When: comparison processes the validated provenance variation.
    comparison = compare_runs(baseline, candidate)

    # Then: Git identity differences remain visible only as provenance, not a performance blocker.
    assert comparison.verdict == "pass"
    assert not comparison.environment_mismatches


@pytest.mark.parametrize(
    ("status", "summary_updates", "case_identity", "reason"),
    [
        ("partial", None, _CASE_IDENTITY, "candidate run status is partial"),
        ("complete", {"stability": "unstable"}, _CASE_IDENTITY, "candidate case is unstable"),
        (
            "complete",
            {"success_count": "1", "failure_count": "2", "stability": "insufficient_samples"},
            _CASE_IDENTITY,
            "candidate case has insufficient successful replicates",
        ),
        ("complete", None, "d" * 64, "case identity sets do not align"),
    ],
)
def test_compare_runs_refuses_regression_verdict_for_ineligible_evidence(
    tmp_path: Path,
    status: str,
    summary_updates: dict[str, str] | None,
    case_identity: str,
    reason: str,
) -> None:
    """Keep aligned descriptive information but never score incomplete or unsuitable evidence."""
    # Given: a valid baseline and a candidate with one comparison-eligibility failure.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        status=status,
        summary_updates=summary_updates,
        case_identity=case_identity,
    )

    # When: guarded comparison evaluates the synthetic evidence.
    comparison = compare_runs(baseline, candidate)

    # Then: it emits an explicit refusal reason rather than a pass/fail regression claim.
    assert comparison.verdict == "not_comparable"
    assert reason in comparison.reasons
    if case_identity == _CASE_IDENTITY:
        assert comparison.case_comparisons[0].step_time_change_percent == 0.0


def test_compare_runs_refuses_configuration_mismatch_but_reports_descriptive_delta(
    tmp_path: Path,
) -> None:
    """Treat a different resolved configuration identity as a methodology incompatibility."""
    # Given: equal case identities but distinct hash-bound resolved configuration identities.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(
        tmp_path,
        "candidate",
        config_sha256="e" * 64,
        summary_updates={"median_step_time_ms": "110.0", "median_tokens_per_second": "909.0"},
    )

    # When: comparison checks methodology/config identity before deciding regression status.
    comparison = compare_runs(baseline, candidate)

    # Then: the delta remains descriptive but no regression verdict is made.
    assert comparison.verdict == "not_comparable"
    assert "config_sha256 differs" in comparison.reasons
    assert comparison.case_comparisons[0].step_time_change_percent == pytest.approx(10.0)


def test_compare_step_times_uses_a_strict_regression_threshold() -> None:
    """Flag only changes strictly above the configured relative-step-time boundary."""
    # Given: hand-calculated equality and just-over-threshold candidate measurements.
    # When: the pure comparison helper applies the same threshold to both inputs.
    equality = compare_step_times(100.0, 105.0, threshold_percent=5.0)
    exceeded = compare_step_times(100.0, 105.01, threshold_percent=5.0)

    # Then: equality is not a regression, while a larger relative increase is.
    assert equality.regressed is False
    assert exceeded.regressed is True


def test_write_comparison_rejects_existing_sibling_artifacts(tmp_path: Path) -> None:
    """Refuse deterministic output-name collisions instead of overwriting comparison evidence."""
    # Given: an aligned comparison has already written its two immutable-run sibling artifacts.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    comparison = compare_runs(baseline, candidate)
    _ = write_comparison(comparison)

    # When/Then: the same deterministic comparison is requested again.
    with pytest.raises(FileExistsError, match="already exists"):
        _ = write_comparison(comparison)


def test_compare_cli_writes_sibling_artifacts_and_returns_nonzero_for_invalid_input(
    tmp_path: Path,
) -> None:
    """Run the typed public CLI against strict fixtures and a malformed manifest path."""
    # Given: a valid synthetic baseline/candidate pair and the root comparison entrypoint.
    baseline = _write_run_package(tmp_path, "baseline")
    candidate = _write_run_package(tmp_path, "candidate")
    script = Path(__file__).parents[1] / "compare_benchmarks.py"
    command = [
        sys.executable,
        str(script),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]

    # When: the public CLI receives valid evidence and then a missing candidate path.
    successful = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603
    invalid = subprocess.run(  # noqa: S603
        [*command[:-1], str(tmp_path / "missing" / "run_manifest.json")],
        capture_output=True,
        check=False,
        text=True,
    )

    # Then: valid comparison emits sibling paths, while invalid data is a nonzero process result.
    assert successful.returncode == 0
    assert "comparison_json=" in successful.stdout
    assert (tmp_path / "candidate-comparison-baseline.json").is_file()
    assert invalid.returncode != 0
    assert "comparison failed:" in invalid.stderr


def test_fixed_fixture_packages_are_hash_valid_and_have_no_real_timing_claims() -> None:
    """Compare compact committed fixtures that only exercise the artifact contract."""
    # Given: the fixed baseline/candidate packages committed for CLI and loader coverage.
    baseline = _FIXTURES / "baseline" / "run_manifest.json"
    candidate = _FIXTURES / "candidate" / "run_manifest.json"

    # When: their strict package contents are compared.
    comparison = compare_runs(baseline, candidate)

    # Then: their synthetic-only report keeps an explicitly neutral comparison result.
    assert comparison.verdict == "pass"
    assert comparison.case_comparisons[0].step_time_change_percent == 0.0
