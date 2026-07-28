"""Test Benchmark v2 statistics and durable report artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import minigpt.benchmark_v2 as benchmark_module
from minigpt.benchmark_v2 import RawReplicate, expand_benchmark_tasks, run_benchmark_v2
from minigpt.benchmark_v2_report import load_run_manifest, write_run_artifacts
from minigpt.benchmark_v2_statistics import summarize_replicates
from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config, ProfileV2Settings

if TYPE_CHECKING:
    from collections.abc import Callable

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
    environment = cast(
        "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
    )
    run_environment = cast("dict[str, object]", environment["run_environment"])
    assert manifest.status == "complete"
    assert {entry.path for entry in manifest.artifacts} == expected_names - {"run_manifest.json"}
    for entry in manifest.artifacts:
        artifact_path = run_directory / entry.path
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size == entry.size_bytes
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == entry.sha256
        assert not entry.path.startswith("/")
        assert ".." not in entry.path.split("/")
    assert {
        "captured_before_first_worker",
        "git",
        "platform",
        "machine",
        "cpu_name",
        "physical_cpu_count",
        "logical_cpu_count",
        "python_version",
        "torch_version",
        "numpy_version",
        "cuda_available",
        "configured_torch_num_interop_threads",
        "configured_cpu_affinity",
        "relevant_environment_variables",
        "process_priority",
        "power_scheme",
    } <= set(run_environment)
    assert run_environment["captured_before_first_worker"] is True


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


def test_complete_artifacts_require_exact_unique_task_identities(tmp_path: Path) -> None:
    """Reject a nominally successful set with a duplicate task replacing an expected task."""
    # Given: two expected task identities but two raw successes for only the first one.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    duplicate_records = (successful_record(tasks[0], 10.0), successful_record(tasks[0], 11.0))
    run_directory = tmp_path / "duplicate-complete"
    run_directory.mkdir()

    # When: the malformed set is claimed to be a complete run.
    # Then: finalization rejects the duplicate/missing task identity rather than writing success.
    with pytest.raises(ValueError, match="exactly one successful raw record"):
        _ = write_run_artifacts(
            config=config,
            run_directory=run_directory,
            run_id="duplicate-complete",
            status="complete",
            tasks=tasks,
            raw_replicates=duplicate_records,
            started_at_utc="2026-07-28T01:02:03+00:00",
            ended_at_utc="2026-07-28T01:02:04+00:00",
        )


@pytest.mark.parametrize("mutation", ["tamper", "missing", "duplicate", "bad_hash"])
def test_load_run_manifest_verifies_each_bound_artifact(tmp_path: Path, mutation: str) -> None:
    """Reject missing, altered, duplicate, or non-hex artifact bindings before loading."""
    # Given: a valid complete report package and one requested manifest/artifact mutation.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"manifest-{mutation}"
    run_directory.mkdir()
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_directory.name,
        status="complete",
        tasks=tasks,
        raw_replicates=records,
        started_at_utc="2026-07-28T01:02:03+00:00",
        ended_at_utc="2026-07-28T01:02:04+00:00",
    )
    if mutation == "tamper":
        _ = artifacts.summary_csv_path.write_text("tampered\n", encoding="utf-8")
    elif mutation == "missing":
        _ = artifacts.summary_csv_path.unlink()
    else:
        document = cast(
            "dict[str, object]",
            json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")),
        )
        entries = cast("list[dict[str, object]]", document["artifacts"])
        if mutation == "duplicate":
            entries.append(entries[0].copy())
        else:
            entries[0]["sha256"] = "z" * 64
        _ = artifacts.run_manifest_path.write_text(
            json.dumps(document), encoding="utf-8", newline="\n"
        )

    # When/Then: strict loading detects a broken bound artifact before returning metadata.
    with pytest.raises(ValueError, match=r"artifact|bound"):
        _ = load_run_manifest(artifacts.run_manifest_path)


def test_orchestrator_marks_all_failures_failed_and_removes_transitional_state(
    tmp_path: Path,
) -> None:
    """All worker failures produce failed evidence without a surviving run-state artifact."""
    # Given: every expected worker returns a typed protocol failure.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)

    class FailingLauncher:
        """Return a worker-declared failure for each requested task."""

        def __init__(self) -> None:
            """Start with the first expected task."""
            self._index: int = 0

        def __call__(
            self, command: list[str], request_json: str, timeout: float
        ) -> subprocess.CompletedProcess[str]:
            """Return one protocol-shaped failure response."""
            task = tasks[self._index]
            self._index += 1
            response = worker_success_document(task, 15_000 + self._index)
            failure = {
                key: response[key]
                for key in (
                    "protocol_version",
                    "worker_pid",
                    "started_at_utc",
                    "ended_at_utc",
                    "case_identity",
                    "case_name",
                    "replicate_index",
                )
            }
            failure.update({"status": "error", "error_type": "RuntimeError", "message": "boom"})
            return subprocess.CompletedProcess(command, 1, json.dumps(failure), "boom")

    # When: all launch results are collected normally.
    artifacts = run_benchmark_v2(config, launcher=FailingLauncher())

    # Then: no success is mislabeled as partial or complete, and final artifacts replace run state.
    assert artifacts.status == "failed"
    assert load_run_manifest(artifacts.run_manifest_path).status == "failed"
    assert not artifacts.run_state_path.exists()


def test_interruption_without_a_success_finalizes_failed_evidence_then_reraises(
    tmp_path: Path,
) -> None:
    """An interruption before success writes failed evidence while preserving KeyboardInterrupt."""
    # Given: a launcher interrupted before it can return the first result.
    config = make_config(tmp_path)

    def interrupt_launcher(
        command: list[str], request_json: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        _ = (command, request_json, timeout)
        raise KeyboardInterrupt

    # When: orchestration is interrupted at the first launch.
    with pytest.raises(KeyboardInterrupt):
        _ = run_benchmark_v2(config, launcher=interrupt_launcher)

    # Then: the run directory contains a failed package rather than a success claim.
    (run_directory,) = tuple(tmp_path.iterdir())
    manifest = load_run_manifest(run_directory / "run_manifest.json")
    assert manifest.status == "failed"
    assert not (run_directory / "run_state.json").exists()


def test_ordinary_launcher_exception_finalizes_partial_evidence_then_reraises(
    tmp_path: Path,
) -> None:
    """An ordinary parent exception retains evidence and does not replace the original error."""
    # Given: one successful worker followed by a launcher-owned operating-system error.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    calls = 0

    def exception_launcher(
        command: list[str], request_json: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        _ = (request_json, timeout)
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(worker_success_document(tasks[0], 16_000)), ""
            )
        msg = "launcher unavailable"
        raise OSError(msg)

    # When: the parent launcher raises after durable successful evidence exists.
    with pytest.raises(OSError, match="launcher unavailable"):
        _ = run_benchmark_v2(config, launcher=exception_launcher)

    # Then: partial evidence is finalized before the original exception propagates.
    (run_directory,) = tuple(tmp_path.iterdir())
    assert load_run_manifest(run_directory / "run_manifest.json").status == "partial"


def test_exception_finalization_never_masks_launcher_error_when_state_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort state persistence failure leaves the original launcher OSError intact."""
    # Given: initial writes work, then finalization state persistence fails after launcher failure.
    config = make_config(tmp_path)
    real_write_json = cast(
        "Callable[[Path, JsonValue], None]", benchmark_module.__dict__["_write_json"]
    )
    calls = 0

    def fail_final_state_write(path: Path, document: JsonValue) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            msg = "state write failed"
            raise RuntimeError(msg)
        real_write_json(path, document)

    def fail_launcher(
        command: list[str], request_json: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        _ = (command, request_json, timeout)
        msg = "original launcher error"
        raise OSError(msg)

    monkeypatch.setattr(benchmark_module, "_write_json", fail_final_state_write)

    # When/Then: cleanup failure cannot replace the original launcher exception.
    with pytest.raises(OSError, match="original launcher error"):
        _ = run_benchmark_v2(config, launcher=fail_launcher)


def test_exception_finalization_never_masks_launcher_error_when_state_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort final state cleanup failure leaves the original launcher OSError intact."""
    # Given: reporting works but its successful state-file cleanup cannot unlink the state.
    config = make_config(tmp_path)
    real_unlink = Path.unlink

    def fail_state_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name == "run_state.json":
            msg = "state unlink failed"
            raise RuntimeError(msg)
        real_unlink(path, missing_ok=missing_ok)

    def fail_launcher(
        command: list[str], request_json: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        _ = (command, request_json, timeout)
        msg = "original launcher error"
        raise OSError(msg)

    monkeypatch.setattr(Path, "unlink", fail_state_unlink)

    # When/Then: cleanup failure cannot replace the original launcher exception.
    with pytest.raises(OSError, match="original launcher error"):
        _ = run_benchmark_v2(config, launcher=fail_launcher)


def test_snapshot_without_git_commit_never_rereads_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An immutable snapshot with no commit produces a deterministic token without Git access."""
    # Given: a pre-captured Git snapshot that explicitly has no commit SHA.
    config = make_config(tmp_path)

    def unexpected_git_lookup(command: str) -> str:
        msg = f"unexpected Git lookup: {command}"
        raise AssertionError(msg)

    monkeypatch.setattr(shutil, "which", unexpected_git_lookup)

    # When: a run ID is derived from the supplied immutable snapshot.
    run_id = benchmark_module.create_run_id(
        config,
        created_at=datetime(2026, 7, 28, tzinfo=UTC),
        environment_snapshot={"git": {"commit_sha": None}},
    )

    # Then: no live repository query occurs and the unavailable identity token is deterministic.
    assert run_id.split("-")[1] == "unknown00000"


@pytest.mark.parametrize("mutation", ["drive", "missing", "extra", "run_id", "environment"])
def test_load_run_manifest_rejects_exact_set_and_environment_identity_violations(
    tmp_path: Path, mutation: str
) -> None:
    """Reject unsafe paths, non-exact artifact sets, and identity disagreement."""
    # Given: a valid report package and one binding mutation whose hashes are updated when needed.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"strict-{mutation}"
    run_directory.mkdir()
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_directory.name,
        status="complete",
        tasks=tasks,
        raw_replicates=records,
        started_at_utc="2026-07-28T01:02:03+00:00",
        ended_at_utc="2026-07-28T01:02:04+00:00",
    )
    document = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    entries = cast("list[dict[str, object]]", document["artifacts"])
    if mutation == "drive":
        entries[0]["path"] = "C:outside.txt"
    elif mutation == "missing":
        _ = entries.pop()
    elif mutation == "extra":
        extra_path = run_directory / "extra.txt"
        _ = extra_path.write_text("extra\n", encoding="utf-8")
        entries.append(
            {
                "path": "extra.txt",
                "size_bytes": extra_path.stat().st_size,
                "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
            }
        )
    elif mutation == "run_id":
        document["run_id"] = "different-run"
    else:
        environment = cast(
            "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
        )
        environment["run_id"] = "different-run"
        _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
        environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
        environment_entry["size_bytes"] = artifacts.environment_path.stat().st_size
        environment_entry["sha256"] = hashlib.sha256(
            artifacts.environment_path.read_bytes()
        ).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(document), encoding="utf-8")

    # When/Then: strict loading rejects every invalid package boundary before returning a manifest.
    with pytest.raises(ValueError, match=r"artifact|run_id|environment"):
        _ = load_run_manifest(artifacts.run_manifest_path)
