"""Test Benchmark v2 statistics and durable report artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

import minigpt.benchmark_v2 as benchmark_module
import minigpt.benchmark_v2_report as report_module
import minigpt.benchmark_workload_methodology as methodology_module
from minigpt.benchmark_v2 import RawReplicate, expand_benchmark_tasks, run_benchmark_v2
from minigpt.benchmark_v2_report import load_run_manifest, write_run_artifacts
from minigpt.benchmark_v2_statistics import summarize_replicates
from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config, ProfileV2Settings
from minigpt.model import expected_gpt_parameter_count
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TextIO

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
        "tokens_per_step": task.case.batch_size * task.case.block_size,
        "parameter_count": expected_gpt_parameter_count(
            GPTConfig(
                vocab_size=task.vocab_size,
                block_size=task.case.block_size,
                n_layer=task.case.n_layer,
                n_head=task.case.n_head,
                n_embd=task.case.n_embd,
                dropout=methodology_module.MODEL_DROPOUT,
                bias=methodology_module.MODEL_BIAS,
            )
        ),
        "final_rss_mib": 128.0,
        "peak_rss_mib": 160.0,
        "peak_rss_method": "windows_peak_working_set",
        "peak_rss_scope": "worker_lifetime",
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


class RollbackFailingStream:
    """Inject independent rollback-operation failures after one interrupted raw write."""

    def __init__(self, interruption: KeyboardInterrupt) -> None:
        """Retain the exact interruption that a write operation raises."""
        self.interruption: KeyboardInterrupt = interruption
        self.calls: list[str] = []

    def tell(self) -> int:
        """Report a stable pre-write offset."""
        return 0

    def write(self, content: str) -> int:
        """Interrupt the raw write before any bytes become durable."""
        _ = content
        raise self.interruption

    def seek(self, offset: int) -> int:
        """Fail the first rollback action after recording its attempted execution."""
        _ = offset
        self.calls.append("seek")
        msg = "seek rollback failed"
        raise RuntimeError(msg)

    def truncate(self, size: int | None = None) -> int:
        """Fail the second rollback action after recording its attempted execution."""
        _ = size
        self.calls.append("truncate")
        msg = "truncate rollback failed"
        raise RuntimeError(msg)

    def flush(self) -> None:
        """Fail the third rollback action after recording its attempted execution."""
        self.calls.append("flush")
        msg = "flush rollback failed"
        raise RuntimeError(msg)

    def fileno(self) -> int:
        """Supply a descriptor so the injected fsync can fail independently."""
        self.calls.append("fileno")
        return 123


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
    assert summary.peak_rss_scope == "worker_lifetime"
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
    assert manifest.peak_rss_scope == "worker_lifetime"
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
    summary_header = artifacts.summary_csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "peak_rss_scope" in summary_header
    assert (
        "worker lifetime peak RSS"
        in artifacts.summary_markdown_path.read_text(encoding="utf-8")
    )


def test_load_run_manifest_rejects_non_worker_lifetime_peak_rss_scope(tmp_path: Path) -> None:
    # Given: one valid package whose self-excluded manifest scope is then forged.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0) for task in tasks)
    run_directory = tmp_path / "scope-test"
    run_directory.mkdir()
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
    manifest = cast(
        "dict[str, object]",
        json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")),
    )
    manifest["peak_rss_scope"] = "measurement_only"
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the strict loader refuses a scope not produced by Benchmark v2.
    with pytest.raises(ValueError, match="peak_rss_scope must be worker_lifetime"):
        _ = load_run_manifest(artifacts.run_manifest_path)


def test_cpu_name_uses_platform_identity_before_linux_cpuinfo_fallback() -> None:
    """Platform-provided CPU identity wins over an available Linux fallback document."""
    # Given: platform APIs provide a processor name and Linux text supplies a conflicting fallback.
    cpuinfo = "model name\t: Fallback CPU\n"

    # When: the pure identity helper resolves the available evidence in priority order.
    identity = report_module.resolve_cpu_name(
        platform_processor="Platform CPU",
        uname_processor="Uname CPU",
        system="Linux",
        linux_cpuinfo_text=cpuinfo,
        windows_processor_identifier=None,
    )

    # Then: existing platform evidence remains unchanged instead of reading a fallback.
    assert identity == "Platform CPU"


def test_cpu_name_parses_linux_cpuinfo_with_deterministic_key_priority() -> None:
    """Linux fallback selects its first supported brand key by priority, not line order."""
    # Given: injected Linux cpuinfo has supported keys in an order unlike the selection priority.
    cpuinfo = "Processor\t: Generic Processor\nHardware\t: Board CPU\nmodel name\t: Model CPU\n"

    # When: no platform API exposes a CPU name and the pure fallback parser is used.
    identity = report_module.resolve_cpu_name(
        platform_processor="",
        uname_processor="",
        system="Linux",
        linux_cpuinfo_text=cpuinfo,
        windows_processor_identifier=None,
    )

    # Then: the most specific documented model-name key wins deterministically.
    assert identity == "Model CPU"


def test_load_run_manifest_recomputes_config_and_case_identities_from_snapshotted_yaml(
    tmp_path: Path,
) -> None:
    """Reject a self-consistent manifest/environment pair that lies about its resolved workload."""
    # Given: a writer-produced complete package whose copied identity tokens are forged.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    run_directory = tmp_path / "forged-identities"
    run_directory.mkdir()
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_directory.name,
        status="complete",
        tasks=tasks,
        raw_replicates=tuple(successful_record(task, 1.0) for task in tasks),
        started_at_utc=datetime(2026, 7, 28, 1, 2, 3, tzinfo=UTC).isoformat(),
        ended_at_utc=datetime(2026, 7, 28, 1, 2, 4, tzinfo=UTC).isoformat(),
    )
    forged_sha256 = "d" * 64
    environment = cast(
        "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
    )
    environment["config_sha256"] = forged_sha256
    _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
    manifest = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    manifest["config_sha256"] = forged_sha256
    manifest["case_identities"] = [{"case_name": config.cases[0].name, "case_identity": "e" * 64}]
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_content = artifacts.environment_path.read_bytes()
    environment_entry["size_bytes"] = len(environment_content)
    environment_entry["sha256"] = hashlib.sha256(environment_content).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: copies agreeing with one another cannot override the snapshotted config bytes.
    with pytest.raises(ValueError, match="config_sha256"):
        _ = load_run_manifest(artifacts.run_manifest_path)


def test_load_run_manifest_rejects_forged_case_identity_with_a_real_config_hash(
    tmp_path: Path,
) -> None:
    """Reject a self-declared case identity even when config hashes are otherwise authentic."""
    # Given: a complete writer package with only its manifest's case identity forged.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    run_directory = tmp_path / "forged-case-identity"
    run_directory.mkdir()
    artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_directory.name,
        status="complete",
        tasks=tasks,
        raw_replicates=tuple(successful_record(task, 1.0) for task in tasks),
        started_at_utc=datetime(2026, 7, 28, 1, 2, 3, tzinfo=UTC).isoformat(),
        ended_at_utc=datetime(2026, 7, 28, 1, 2, 4, tzinfo=UTC).isoformat(),
    )
    manifest = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    manifest["case_identities"] = [{"case_name": config.cases[0].name, "case_identity": "e" * 64}]
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the manifest cannot override the case identity derived from snapshotted YAML.
    with pytest.raises(ValueError, match="case identities"):
        _ = load_run_manifest(artifacts.run_manifest_path)


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
    with pytest.raises(ValueError, match=r"artifact|bound|filesystem"):
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


@pytest.mark.parametrize(
    "mutation", ["drive", "missing", "extra", "run_id", "environment", "status"]
)
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
    elif mutation == "environment":
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
    else:
        environment = cast(
            "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
        )
        environment["run_status"] = "failed"
        _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
        environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
        environment_entry["size_bytes"] = artifacts.environment_path.stat().st_size
        environment_entry["sha256"] = hashlib.sha256(
            artifacts.environment_path.read_bytes()
        ).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(document), encoding="utf-8")

    # When/Then: strict loading rejects every invalid package boundary before returning a manifest.
    with pytest.raises(ValueError, match=r"artifact|run_id|environment|filesystem"):
        _ = load_run_manifest(artifacts.run_manifest_path)


def test_raw_rollback_preserves_original_interrupt_and_finalizes_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted raw append survives every rollback failure and still reaches finalization."""
    # Given: raw rollback seeks, truncates, flushes, and fsyncs all fail after one exact interrupt.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    interruption = KeyboardInterrupt("raw write interrupted")
    rollback_stream = RollbackFailingStream(interruption)
    append_record = cast(
        "Callable[[TextIO, RawReplicate], None]",
        benchmark_module.__dict__["_append_durable_raw_record"],
    )
    real_fsync = os.fsync

    def fail_fsync(descriptor: int) -> None:
        if descriptor != 123:
            real_fsync(descriptor)
            return
        rollback_stream.calls.append("fsync")
        msg = "fsync rollback failed"
        raise RuntimeError(msg)

    def interrupt_raw_append(raw_stream: TextIO, record: RawReplicate) -> None:
        _ = raw_stream
        append_record(cast("TextIO", cast("object", rollback_stream)), record)

    def successful_launcher(
        command: list[str], request_json: str, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        _ = (request_json, timeout)
        return subprocess.CompletedProcess(
            command, 0, json.dumps(worker_success_document(tasks[0], 12_345)), ""
        )

    monkeypatch.setattr(os, "fsync", fail_fsync)
    monkeypatch.setattr(benchmark_module, "_append_durable_raw_record", interrupt_raw_append)

    # When: the first real worker record encounters the interrupted, rollback-failing append.
    with pytest.raises(KeyboardInterrupt) as raised:
        _ = run_benchmark_v2(config, launcher=successful_launcher)

    # Then: the exact original interrupt survives and the outer lifecycle finalizes failed evidence.
    assert raised.value is interruption
    assert str(raised.value) == "raw write interrupted"
    assert rollback_stream.calls == ["seek", "truncate", "flush", "fileno", "fsync"]
    (run_directory,) = tuple(tmp_path.iterdir())
    assert load_run_manifest(run_directory / "run_manifest.json").status == "failed"
    assert not (run_directory / "run_state.json").exists()


@pytest.mark.parametrize("unexpected_entry", ["extra.txt", "run_state.json", "unfinished"])
def test_load_run_manifest_rejects_unbound_finalized_run_entries(
    tmp_path: Path, unexpected_entry: str
) -> None:
    """A finalized package admits only its six bound artifacts and self-excluded manifest."""
    # Given: a valid package gains one unbound file, lifecycle state, or directory entry.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"unbound-{unexpected_entry}"
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
    unexpected_path = run_directory / unexpected_entry
    if unexpected_entry == "unfinished":
        unexpected_path.mkdir()
    else:
        _ = unexpected_path.write_text("unexpected\n", encoding="utf-8")

    # When/Then: exact finalized-package membership rejects every extra filesystem entry.
    with pytest.raises(ValueError, match="filesystem entries"):
        _ = load_run_manifest(artifacts.run_manifest_path)


@pytest.mark.parametrize(
    "git_identity",
    [
        {"commit_sha": "a" * 39, "branch": None, "dirty": None},
        {"commit_sha": None, "branch": 42, "dirty": False},
        {"commit_sha": None, "branch": None, "dirty": None, "extra": "forbidden"},
    ],
)
def test_load_run_manifest_rejects_matching_malformed_git_identities(
    tmp_path: Path, git_identity: dict[str, object]
) -> None:
    """Matching manifest and environment Git mappings must still conform to the writer schema."""
    # Given: both identity copies agree on a malformed commit, type, or forbidden key.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / "malformed-git"
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
    manifest = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    environment = cast(
        "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
    )
    manifest["git"] = git_identity
    run_environment = cast("dict[str, object]", environment["run_environment"])
    run_environment["git"] = git_identity
    _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_entry["size_bytes"] = artifacts.environment_path.stat().st_size
    environment_entry["sha256"] = hashlib.sha256(
        artifacts.environment_path.read_bytes()
    ).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: matching values cannot bypass the strict Git identity schema.
    with pytest.raises(ValueError, match="git"):
        _ = load_run_manifest(artifacts.run_manifest_path)


def test_load_run_manifest_rejects_matching_non_sha_config_identities(tmp_path: Path) -> None:
    """Matching manifest and environment config identities must be lowercase SHA-256 digests."""
    # Given: both identity copies agree on a non-SHA config token and the hash binding is updated.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / "malformed-config-sha"
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
    manifest = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    environment = cast(
        "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
    )
    non_sha = "a" * 63
    manifest["config_sha256"] = non_sha
    environment["config_sha256"] = non_sha
    _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_entry["size_bytes"] = artifacts.environment_path.stat().st_size
    environment_entry["sha256"] = hashlib.sha256(
        artifacts.environment_path.read_bytes()
    ).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: matching values cannot bypass the SHA-256 format requirement.
    with pytest.raises(ValueError, match="config_sha256"):
        _ = load_run_manifest(artifacts.run_manifest_path)


@pytest.mark.parametrize("commit_sha", ["a" * 40, "b" * 64])
def test_load_run_manifest_accepts_writer_git_object_id_lengths(
    tmp_path: Path, commit_sha: str
) -> None:
    """Git SHA-1 and SHA-256 object IDs emitted by the writer remain loadable evidence."""
    # Given: the immutable writer snapshot holds a lowercase Git SHA-1 or SHA-256 object ID.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"valid-git-{len(commit_sha)}"
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
        environment_snapshot={"git": {"commit_sha": commit_sha, "branch": "main", "dirty": False}},
    )

    # When: the strict loader reads writer-produced Git object identity evidence.
    manifest = load_run_manifest(artifacts.run_manifest_path)

    # Then: both supported object ID lengths retain their exact identity.
    assert manifest.git["commit_sha"] == commit_sha


@pytest.mark.parametrize("commit_sha", ["c" * 41, "d" * 65, "e" * 39, "F" * 40])
def test_load_run_manifest_rejects_unsupported_git_object_ids(
    tmp_path: Path, commit_sha: str
) -> None:
    """Only lowercase SHA-1 and SHA-256 Git object IDs qualify as durable Git identity."""
    # Given: manifest and environment agree on an unsupported object ID with current hash bindings.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"invalid-git-{len(commit_sha)}"
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
    manifest = cast(
        "dict[str, object]", json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))
    )
    environment = cast(
        "dict[str, object]", json.loads(artifacts.environment_path.read_text(encoding="utf-8"))
    )
    git_identity = {"commit_sha": commit_sha, "branch": None, "dirty": None}
    manifest["git"] = git_identity
    run_environment = cast("dict[str, object]", environment["run_environment"])
    run_environment["git"] = git_identity
    _ = artifacts.environment_path.write_text(json.dumps(environment), encoding="utf-8")
    entries = cast("list[dict[str, object]]", manifest["artifacts"])
    environment_entry = next(entry for entry in entries if entry["path"] == "environment.json")
    environment_entry["size_bytes"] = artifacts.environment_path.stat().st_size
    environment_entry["sha256"] = hashlib.sha256(
        artifacts.environment_path.read_bytes()
    ).hexdigest()
    _ = artifacts.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: matching copies cannot make unsupported Git object IDs valid evidence.
    with pytest.raises(ValueError, match="commit_sha"):
        _ = load_run_manifest(artifacts.run_manifest_path)


@pytest.mark.parametrize("mutation", ["replace", "add"])
def test_load_run_manifest_rejects_filesystem_change_during_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """A replacement or added entry between initial enumeration and final check is rejected."""
    # Given: a valid package and an open seam that changes filesystem state during snapshotting.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"racy-{mutation}"
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
    real_open = cast("Callable[..., TextIO]", Path.open)
    original_summary = artifacts.summary_csv_path.read_bytes()
    changed = False

    def change_before_open(path: Path, *args: object, **kwargs: object) -> TextIO:
        nonlocal changed
        if not changed and path.name == "summary.csv":
            changed = True
            if mutation == "replace":
                replacement = path.with_name("summary-replacement.csv")
                _ = replacement.write_bytes(original_summary)
                _ = replacement.replace(path)
            else:
                _ = (path.parent / "late-extra.txt").write_text("late\n", encoding="utf-8")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", change_before_open)

    # When/Then: a loader that snapshots from stable handles rejects both deterministic changes.
    with pytest.raises(ValueError, match=r"changed|filesystem entries"):
        _ = load_run_manifest(artifacts.run_manifest_path)


@pytest.mark.parametrize("entry_name", ["summary.csv", "run_manifest.json"])
def test_load_run_manifest_rejects_symlinked_package_entries(
    tmp_path: Path, entry_name: str
) -> None:
    """A symlink cannot substitute for any regular finalized-package entry."""
    # Given: a valid package entry is replaced by a same-byte external symlink target.
    config = make_config(tmp_path)
    tasks = expand_benchmark_tasks(config)
    records = tuple(successful_record(task, 10.0 + task.replicate_index) for task in tasks)
    run_directory = tmp_path / f"symlink-{entry_name.replace('.', '-')}"
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
    entry_path = run_directory / entry_name
    external_target = tmp_path / f"external-{entry_name}"
    _ = external_target.write_bytes(entry_path.read_bytes())
    entry_path.unlink()
    try:
        entry_path.symlink_to(external_target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    # When/Then: strict loading rejects symlink substitution before trusting its target bytes.
    with pytest.raises(ValueError, match="regular"):
        _ = load_run_manifest(artifacts.run_manifest_path)
