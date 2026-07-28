"""Test Benchmark v2 statistics and durable report artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

from minigpt.benchmark_v2 import RawReplicate, expand_benchmark_tasks, run_benchmark_v2
from minigpt.benchmark_v2_report import load_run_manifest, write_run_artifacts
from minigpt.benchmark_v2_statistics import summarize_replicates
from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config, ProfileV2Settings

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.benchmark_v2 import BenchmarkTask
    from minigpt.benchmark_v2_config import JsonValue


def make_config(output_root: Path) -> BenchmarkV2Config:
    """Build a compact single-case report fixture configuration."""
    case = BenchmarkV2Case(
        name="tiny_t1_s32_b2",
        model_name="tiny",
        n_layer=1,
        n_head=1,
        n_embd=8,
        torch_num_threads=1,
        block_size=32,
        batch_size=2,
    )
    return BenchmarkV2Config(
        schema_version=2,
        experiment_name="report_test",
        benchmark_seed=1337,
        vocab_size=65,
        output_root=output_root,
        worker_timeout_seconds=30.0,
        warmup_steps=0,
        measurement_steps=1,
        replicates=2,
        torch_num_interop_threads=1,
        cpu_affinity=None,
        max_cv_percent=10.0,
        minimum_replicates=2,
        regression_threshold_percent=5.0,
        relevant_environment_variables=("OMP_NUM_THREADS",),
        cases=(case,),
        profile=ProfileV2Settings(
            enabled=False,
            case_name=case.name,
            warmup_steps=1,
            active_steps=1,
        ),
    )


def worker_success_document(task: BenchmarkTask, worker_pid: int) -> dict[str, JsonValue]:
    """Build one complete protocol-valid synthetic worker result."""
    return {
        "protocol_version": 1,
        "status": "ok",
        "worker_pid": worker_pid,
        "started_at_utc": "2026-07-28T01:02:03+00:00",
        "ended_at_utc": "2026-07-28T01:02:04+00:00",
        "case_identity": task.case_identity,
        "case_name": task.case.name,
        "replicate_index": task.replicate_index,
        "warmup_steps": task.warmup_steps,
        "measurement_steps": task.measurement_steps,
        "elapsed_seconds": 0.5,
        "step_time_ms": 500.0,
        "tokens_per_second": 128.0,
        "tokens_per_step": 64,
        "parameter_count": 123,
        "final_rss_mib": 128.0,
        "peak_rss_mib": 160.0,
        "peak_rss_method": "windows_peak_working_set",
        "peak_rss_sampling_interval_ms": None,
        "environment": {
            "platform": "test-platform",
            "python_version": "3.14.0",
            "torch_version": "test-torch",
            "torch_num_threads": task.case.torch_num_threads,
            "torch_num_interop_threads": task.torch_num_interop_threads,
            "logical_cpu_count": 8,
            "requested_cpu_affinity": None,
            "effective_cpu_affinity": [0, 1],
            "relevant_environment_variables": {"OMP_NUM_THREADS": None},
        },
    }


def successful_record(task: BenchmarkTask, value: float) -> RawReplicate:
    """Build one real-shaped successful raw record with a hand-chosen aggregate value."""
    response = worker_success_document(task, worker_pid=10_000 + task.replicate_index)
    response["step_time_ms"] = value
    response["tokens_per_second"] = 1_000.0 / value
    response["final_rss_mib"] = 100.0 + value
    response["peak_rss_mib"] = 120.0 + value
    return RawReplicate(
        status="ok",
        case_identity=task.case_identity,
        case_name=task.case.name,
        replicate_index=task.replicate_index,
        worker_pid=cast("int", response["worker_pid"]),
        started_at_utc=cast("str", response["started_at_utc"]),
        ended_at_utc=cast("str", response["ended_at_utc"]),
        return_code=0,
        error_type=None,
        message=None,
        stdout=json.dumps(response),
        stderr="",
        worker_response=response,
    )


def failed_record(task: BenchmarkTask) -> RawReplicate:
    """Build one failure record that deliberately has no numeric worker measurement."""
    return RawReplicate(
        status="error",
        case_identity=task.case_identity,
        case_name=task.case.name,
        replicate_index=task.replicate_index,
        worker_pid=None,
        started_at_utc=None,
        ended_at_utc=None,
        return_code=None,
        error_type="WorkerTimeout",
        message="timed out",
        stdout="",
        stderr="",
        worker_response=None,
    )


@pytest.mark.parametrize(
    ("minimum_replicates", "max_cv_percent", "expected_stability"),
    [
        (5, 7.5, "insufficient_samples"),
        (3, 7.0, "unstable"),
        (3, 7.5, "stable"),
    ],
)
def test_summarize_replicates_uses_successful_aggregates_without_dropping_failures(
    tmp_path: Path,
    minimum_replicates: int,
    max_cv_percent: float,
    expected_stability: str,
) -> None:
    """Calculate exact robust statistics while retaining failed replicate counts."""
    # Given: three hand-checked aggregate times and one raw worker failure for the same case.
    task = expand_benchmark_tasks(replace(make_config(tmp_path), replicates=3))[0]
    records = (
        successful_record(task, 10.0),
        successful_record(replace(task, replicate_index=1), 11.0),
        successful_record(replace(task, replicate_index=2), 12.0),
        failed_record(replace(task, replicate_index=3)),
    )

    # When: the report statistic layer summarizes the raw records.
    summary = summarize_replicates(
        records,
        minimum_replicates=minimum_replicates,
        max_cv_percent=max_cv_percent,
    )

    # Then: failures contribute only to raw counts, not numeric aggregate statistics.
    assert summary.replicate_count == 4
    assert summary.success_count == 3
    assert summary.failure_count == 1
    assert summary.median_step_time_ms == 11.0
    assert summary.min_step_time_ms == 10.0
    assert summary.max_step_time_ms == 12.0
    assert summary.population_stddev_step_time_ms == pytest.approx(0.81649658)
    assert summary.median_absolute_deviation_step_time_ms == 1.0
    assert summary.coefficient_of_variation_percent == pytest.approx(7.422696)
    assert summary.stability == expected_stability


def test_write_run_artifacts_binds_all_outputs_without_hashing_its_manifest(tmp_path: Path) -> None:
    """Write a complete synthetic run with run-relative, byte-verified artifact evidence."""
    # Given: a complete two-replicate synthetic run in one intentionally empty directory.
    config = replace(make_config(tmp_path), cases=(make_config(tmp_path).cases[0],))
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / "20260728T010203Z-abcdef12-0123456789ab"
    run_directory.mkdir()

    # When: final report artifacts are generated from the raw evidence.
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_directory.name,
        status="complete",
        tasks=tasks,
        raw_replicates=records,
        started_at_utc=datetime(2026, 7, 28, 1, 2, 3, tzinfo=UTC).isoformat(),
        ended_at_utc=datetime(2026, 7, 28, 1, 2, 4, tzinfo=UTC).isoformat(),
    )

    # Then: every expected artifact exists and the self-excluded manifest exactly binds its bytes.
    expected_names = {
        "run_manifest.json",
        "environment.json",
        "resolved_config.yaml",
        "raw_replicates.jsonl",
        "summary.csv",
        "summary.md",
        "execution_order.json",
    }
    assert {path.name for path in run_directory.iterdir()} == expected_names
    manifest = load_run_manifest(artifacts.run_manifest_path)
    assert manifest.status == "complete"
    assert {entry.path for entry in manifest.artifacts} == expected_names - {"run_manifest.json"}
    for entry in manifest.artifacts:
        artifact_path = run_directory / entry.path
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size == entry.size_bytes
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == entry.sha256
        assert not entry.path.startswith("/")
        assert ".." not in entry.path.split("/")


def test_partial_report_preserves_failure_and_makes_no_success_claim(tmp_path: Path) -> None:
    """A worker failure remains visible in the final evidence and prevents completion wording."""
    # Given: one success and one failure from an expected two-replicate run.
    config = replace(make_config(tmp_path), cases=(make_config(tmp_path).cases[0],))
    tasks = expand_benchmark_tasks(config)
    records = (successful_record(tasks[0], 10.0), failed_record(tasks[1]))
    run_directory = tmp_path / "partial-run"
    run_directory.mkdir()

    # When: the run is finalized as partial evidence.
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id="partial-run",
        status="partial",
        tasks=tasks,
        raw_replicates=records,
        started_at_utc="2026-07-28T01:02:03+00:00",
        ended_at_utc="2026-07-28T01:02:04+00:00",
    )

    # Then: both failure counts and the non-successful state are unambiguous to a report reader.
    manifest = load_run_manifest(artifacts.run_manifest_path)
    markdown = artifacts.summary_markdown_path.read_text(encoding="utf-8")
    assert manifest.status == "partial"
    assert manifest.failed_task_count == 1
    assert "This run is partial and does not claim benchmark success." in markdown
    assert "not a shared-runner performance gate" in markdown


def test_orchestrator_finalizes_unique_run_identity_and_complete_manifest(tmp_path: Path) -> None:
    """Finalization exposes a Git/config-bound UTC run ID and complete manifest to callers."""
    # Given: a tiny all-successful two-replicate orchestration run.
    config = replace(make_config(tmp_path), cases=(make_config(tmp_path).cases[0],))
    tasks = expand_benchmark_tasks(config)

    class SuccessfulLauncher:
        """Return successful protocol documents without starting a real worker."""

        def __init__(self) -> None:
            """Track task order while returning a real protocol-shaped result."""
            self._index: int = 0

        def __call__(
            self,
            command: list[str],
            request_json: str,
            timeout: float,
        ) -> subprocess.CompletedProcess[str]:
            """Return one minimal completed process with a unique worker PID."""
            task = tasks[self._index]
            self._index += 1
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(worker_success_document(task, 12_000 + self._index)),
                "",
            )

    # When: orchestration reaches normal finalization.
    artifacts = run_benchmark_v2(config, launcher=SuccessfulLauncher())

    # Then: the returned report path carries a complete manifest and run identity components.
    manifest = load_run_manifest(artifacts.run_manifest_path)
    run_id_parts = artifacts.run_directory.name.split("-")
    assert manifest.status == "complete"
    assert len(run_id_parts) == 3
    assert run_id_parts[0].endswith("Z")
    assert len(run_id_parts[1]) >= 7
    assert len(run_id_parts[2]) == 12
