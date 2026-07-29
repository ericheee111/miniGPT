"""Test randomized fresh-process Benchmark v2 orchestration and durable partial runs."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, cast

import psutil
import pytest

import minigpt.benchmark_v2 as benchmark_module
import minigpt.benchmark_workload_methodology as methodology_module
from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config, ProfileV2Settings
from minigpt.model import expected_gpt_parameter_count
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from minigpt.benchmark_v2 import BenchmarkTask, WorkerLauncher
    from minigpt.benchmark_v2_config import JsonValue


def make_config(output_root: Path) -> BenchmarkV2Config:
    """Build a small two-case orchestration configuration."""
    cases = (
        BenchmarkV2Case(
            name="tiny_t1_s32_b2",
            model_name="tiny",
            n_layer=1,
            n_head=1,
            n_embd=8,
            torch_num_threads=1,
            block_size=32,
            batch_size=2,
        ),
        BenchmarkV2Case(
            name="tiny_t2_s32_b2",
            model_name="tiny",
            n_layer=1,
            n_head=1,
            n_embd=8,
            torch_num_threads=2,
            block_size=32,
            batch_size=2,
        ),
    )
    return BenchmarkV2Config(
        schema_version=2,
        experiment_name="orchestration_test",
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
        cases=cases,
        profile=ProfileV2Settings(
            enabled=False,
            case_name=cases[0].name,
            warmup_steps=1,
            active_steps=1,
        ),
    )


def test_task_order_is_seeded_and_contains_every_replicate(tmp_path: Path) -> None:
    """Expand each case/replicate exactly once in deterministic shuffled order."""
    # Given: two cases with two replicates and a fixed benchmark seed.
    config = make_config(tmp_path)

    # When: the same config is expanded twice.
    first = benchmark_module.expand_benchmark_tasks(config)
    second = benchmark_module.expand_benchmark_tasks(config)

    # Then: order is repeatable and no expected task is lost or duplicated.
    assert first == second
    assert {(task.case.name, task.replicate_index) for task in first} == {
        ("tiny_t1_s32_b2", 0),
        ("tiny_t1_s32_b2", 1),
        ("tiny_t2_s32_b2", 0),
        ("tiny_t2_s32_b2", 1),
    }


def worker_success_document(task: BenchmarkTask, worker_pid: int) -> dict[str, JsonValue]:
    """Build a complete successful worker protocol response."""
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


def worker_failure_document(task: BenchmarkTask, worker_pid: int) -> dict[str, JsonValue]:
    """Build a complete worker-declared failure protocol response."""
    return {
        "protocol_version": 1,
        "status": "error",
        "worker_pid": worker_pid,
        "started_at_utc": "2026-07-28T01:02:03+00:00",
        "ended_at_utc": "2026-07-28T01:02:04+00:00",
        "case_identity": task.case_identity,
        "case_name": task.case.name,
        "replicate_index": task.replicate_index,
        "error_type": "RuntimeError",
        "message": "worker failed",
    }


def invalid_worker_document(
    task: BenchmarkTask,
    worker_pid: int,
    mutation: str,
) -> dict[str, JsonValue]:
    """Build one worker response with a single strict-schema violation."""
    if mutation == "failure_field_type":
        failure = worker_failure_document(task, worker_pid)
        failure["error_type"] = 7
        return failure
    response = worker_success_document(task, worker_pid)
    scalar_mutations: dict[str, tuple[str, JsonValue]] = {
        "string_metric": ("elapsed_seconds", "0.5"),
        "zero_positive_metric": ("step_time_ms", 0.0),
        "negative_memory": ("final_rss_mib", -1.0),
        "non_finite_metric": ("tokens_per_second", float("nan")),
        "overflow_metric": ("elapsed_seconds", 10**309),
        "integer_field_type": ("parameter_count", 123.0),
        "peak_method": ("peak_rss_method", "sampled"),
        "peak_interval": ("peak_rss_sampling_interval_ms", 1.0),
        "task_count": ("warmup_steps", task.warmup_steps + 1),
        "measurement_task_count": ("measurement_steps", task.measurement_steps + 1),
        "tokens_per_step": ("tokens_per_step", 1),
        "parameter_count": ("parameter_count", 1),
        "elapsed_relationship": ("elapsed_seconds", 0.25),
        "throughput_relationship": ("tokens_per_second", 127.0),
    }
    if mutation in scalar_mutations:
        field, value = scalar_mutations[mutation]
        response[field] = value
        return response
    environment = cast("dict[str, JsonValue]", response["environment"])
    if mutation == "environment_keys":
        del environment["platform"]
    else:
        environment_mutations: dict[str, tuple[str, JsonValue]] = {
            "environment_thread_type": ("torch_num_threads", True),
            "environment_affinity": ("effective_cpu_affinity", [0, 0]),
            "environment_variables": (
                "relevant_environment_variables",
                {"UNEXPECTED": 1},
            ),
        }
        field, value = environment_mutations[mutation]
        environment[field] = value
    return response


class ScriptedLauncher:
    """Return task-aware subprocess outcomes in a fixed sequence."""

    def __init__(self, outcomes: tuple[str, ...]) -> None:
        """Store outcome names and record the worker requests received."""
        self._outcomes: tuple[str, ...] = outcomes
        self.requests: list[dict[str, object]] = []

    def __call__(
        self,
        command: list[str],
        request_json: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Return the next response while enforcing the subprocess boundary contract."""
        assert command[-2:] == ["-m", "minigpt.benchmark_v2_worker"]
        assert timeout > 0
        request = cast("dict[str, object]", json.loads(request_json))
        self.requests.append(request)
        outcome = self._outcomes[len(self.requests) - 1]
        task = task_from_request(request)
        worker_pid = 9_000 + len(self.requests)
        if outcome == "success":
            stdout = json.dumps(worker_success_document(task, worker_pid))
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if outcome == "declared_failure":
            stdout = json.dumps(worker_failure_document(task, worker_pid))
            return subprocess.CompletedProcess(command, 1, stdout, "worker stderr")
        if outcome == "return_code_failure":
            stdout = json.dumps(worker_success_document(task, worker_pid))
            return subprocess.CompletedProcess(command, 1, stdout, "worker stderr")
        if outcome == "bare_return_code_failure":
            return subprocess.CompletedProcess(command, 7, "", "worker crashed")
        if outcome == "malformed":
            return subprocess.CompletedProcess(command, 0, "{not-json", "")
        if outcome.startswith("invalid_"):
            mutation = outcome.removeprefix("invalid_")
            stdout = json.dumps(
                invalid_worker_document(task, worker_pid, mutation),
                allow_nan=True,
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        raise subprocess.TimeoutExpired(command, timeout, output="partial stdout", stderr="timeout")


def task_from_request(request: dict[str, object]) -> BenchmarkTask:
    """Reconstruct task identity from a complete fake-launcher request."""
    case_document = cast("dict[str, object]", request["case"])
    raw_affinity = request["cpu_affinity"]
    raw_variables = cast("list[str]", request["relevant_environment_variables"])
    return benchmark_module.BenchmarkTask(
        case=BenchmarkV2Case(
            name=cast("str", case_document["name"]),
            model_name=cast("str", case_document["model_name"]),
            n_layer=cast("int", case_document["n_layer"]),
            n_head=cast("int", case_document["n_head"]),
            n_embd=cast("int", case_document["n_embd"]),
            torch_num_threads=cast("int", case_document["torch_num_threads"]),
            block_size=cast("int", case_document["block_size"]),
            batch_size=cast("int", case_document["batch_size"]),
        ),
        case_identity=cast("str", request["case_identity"]),
        replicate_index=cast("int", request["replicate_index"]),
        benchmark_seed=cast("int", request["benchmark_seed"]),
        vocab_size=cast("int", request["vocab_size"]),
        warmup_steps=cast("int", request["warmup_steps"]),
        measurement_steps=cast("int", request["measurement_steps"]),
        torch_num_interop_threads=cast("int", request["torch_num_interop_threads"]),
        cpu_affinity=tuple(cast("list[int]", raw_affinity)) if raw_affinity is not None else None,
        relevant_environment_variables=tuple(raw_variables),
    )


@pytest.mark.parametrize(
    ("outcome", "error_type", "expected_pid"),
    [
        ("declared_failure", "RuntimeError", 9_001),
        ("return_code_failure", "WorkerProcessError", 9_001),
        ("bare_return_code_failure", "WorkerProcessError", None),
        ("malformed", "InvalidWorkerResponse", None),
        ("timeout", "WorkerTimeout", None),
    ],
)
def test_execute_worker_converts_each_failure_boundary_to_a_raw_record(
    tmp_path: Path,
    outcome: str,
    error_type: str,
    expected_pid: int | None,
) -> None:
    """Preserve typed raw failure evidence instead of raising ordinary worker failures."""
    # Given: one expanded task and a launcher producing one boundary failure.
    task = benchmark_module.expand_benchmark_tasks(make_config(tmp_path))[0]
    launcher: WorkerLauncher = ScriptedLauncher((outcome,))

    # When: the orchestrator executes the worker boundary.
    record = benchmark_module.execute_worker(task, timeout_seconds=2.0, launcher=launcher)

    # Then: the task identity and available worker-owned PID survive in a typed failure.
    assert record.status == "error"
    assert record.case_identity == task.case_identity
    assert record.case_name == task.case.name
    assert record.replicate_index == task.replicate_index
    assert record.error_type == error_type
    assert record.worker_pid == expected_pid
    if outcome == "declared_failure":
        assert record.message == "worker failed"
        assert record.return_code == 1
        assert record.worker_response is not None


def test_partial_run_keeps_every_raw_record_in_execution_order(tmp_path: Path) -> None:
    """Continue after ordinary failures and durably retain their original task order."""
    # Given: four ordered task outcomes spanning success, exit, malformed JSON, and timeout.
    config = make_config(tmp_path)
    launcher: WorkerLauncher = ScriptedLauncher(
        ("success", "return_code_failure", "malformed", "timeout")
    )
    expected_tasks = benchmark_module.expand_benchmark_tasks(config)

    # When: the full benchmark run executes sequentially.
    artifacts = benchmark_module.run_benchmark_v2(config, launcher=launcher)

    # Then: every raw record remains ordered and any failure makes the run partial.
    assert artifacts.status == "partial"
    assert artifacts.tasks == expected_tasks
    assert [(record.case_name, record.replicate_index) for record in artifacts.raw_replicates] == [
        (task.case.name, task.replicate_index) for task in expected_tasks
    ]
    assert [record.status for record in artifacts.raw_replicates] == [
        "ok",
        "error",
        "error",
        "error",
    ]
    durable_records = [
        cast("dict[str, object]", json.loads(line))
        for line in artifacts.raw_replicates_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["case_name"], record["replicate_index"]) for record in durable_records] == [
        (task.case.name, task.replicate_index) for task in expected_tasks
    ]
    manifest = cast(
        "dict[str, object]",
        json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")),
    )
    assert manifest["status"] == "partial"
    assert manifest["completed_task_count"] == 4
    assert manifest["failed_task_count"] == 3
    assert not artifacts.run_state_path.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "string_metric",
        "zero_positive_metric",
        "negative_memory",
        "non_finite_metric",
        "overflow_metric",
        "integer_field_type",
        "peak_method",
        "peak_interval",
        "environment_keys",
        "environment_thread_type",
        "environment_affinity",
        "environment_variables",
        "failure_field_type",
        "task_count",
        "measurement_task_count",
        "tokens_per_step",
        "parameter_count",
        "elapsed_relationship",
        "throughput_relationship",
    ],
)
def test_invalid_complete_worker_response_is_a_failed_raw_failure(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Reject wrong response types, metrics, controls, and nested environment evidence."""
    # Given: one nominal worker response containing a single protocol violation.
    base_config = make_config(tmp_path)
    config = replace(
        base_config,
        cases=(base_config.cases[0],),
        replicates=1,
        minimum_replicates=1,
    )
    launcher: WorkerLauncher = ScriptedLauncher((f"invalid_{mutation}",))

    # When: the malformed response is run through normal durable orchestration.
    artifacts = benchmark_module.run_benchmark_v2(config, launcher=launcher)

    # Then: strict validation prevents success and zero successful measurements are failed.
    assert artifacts.status == "failed"
    assert len(artifacts.raw_replicates) == 1
    (record,) = artifacts.raw_replicates
    assert record.status == "error"
    assert record.error_type == "InvalidWorkerResponse"
    manifest = cast(
        "dict[str, object]",
        json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")),
    )
    assert manifest["status"] == "failed"
    assert manifest["failed_task_count"] == 1
    assert not artifacts.run_state_path.exists()


def test_keyboard_interrupt_finalizes_durable_partial_state_and_reraises(tmp_path: Path) -> None:
    """Flush completed records and partial status before propagating an interruption."""
    # Given: one successful worker followed by a parent-process interruption.
    config = make_config(tmp_path)
    launcher: WorkerLauncher = ScriptedLauncher(("success", "interrupt"))
    expected_tasks = benchmark_module.expand_benchmark_tasks(config)

    # When: orchestration is interrupted during the second launch.
    with pytest.raises(KeyboardInterrupt):
        _ = benchmark_module.run_benchmark_v2(config, launcher=launcher)

    # Then: initial order and the first record are durable, and run status is partial.
    run_directories = tuple(tmp_path.iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    execution_order = cast(
        "list[dict[str, object]]",
        json.loads((run_directory / "execution_order.json").read_text(encoding="utf-8")),
    )
    assert [(item["case_name"], item["replicate_index"]) for item in execution_order] == [
        (task.case.name, task.replicate_index) for task in expected_tasks
    ]
    raw_lines = (run_directory / "raw_replicates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    first_record = cast("dict[str, object]", json.loads(raw_lines[0]))
    assert (first_record["case_name"], first_record["replicate_index"]) == (
        expected_tasks[0].case.name,
        expected_tasks[0].replicate_index,
    )
    manifest = cast(
        "dict[str, object]",
        json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["status"] == "partial"
    assert manifest["completed_task_count"] == 1
    assert manifest["expected_task_count"] == 4
    assert not (run_directory / "run_state.json").exists()


def test_keyboard_interrupt_during_raw_fsync_rolls_back_non_durable_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count only fully durable JSONL records when persistence is interrupted."""
    # Given: the second raw-record fsync is interrupted after one durable line.
    config = make_config(tmp_path)
    launcher: WorkerLauncher = ScriptedLauncher(("success",) * 4)
    real_fsync = os.fsync
    fsync_calls = 0

    def interrupt_second_raw_fsync(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 4:
            raise KeyboardInterrupt
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", interrupt_second_raw_fsync)

    # When: persistence is interrupted after writing the second raw line.
    with pytest.raises(KeyboardInterrupt):
        _ = benchmark_module.run_benchmark_v2(config, launcher=launcher)

    # Then: the partial state counts exactly the one complete durable JSON line.
    (run_directory,) = tuple(tmp_path.iterdir())
    raw_text = (run_directory / "raw_replicates.jsonl").read_text(encoding="utf-8")
    raw_lines = raw_text.splitlines()
    assert len(raw_lines) == 1
    assert raw_text.endswith("\n")
    durable_record = cast("object", json.loads(raw_lines[0]))
    assert isinstance(durable_record, dict)
    manifest = cast(
        "dict[str, object]",
        json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["status"] == "partial"
    assert manifest["completed_task_count"] == len(raw_lines) == 1
    assert not (run_directory / "run_state.json").exists()


def test_real_tiny_workers_use_distinct_exited_processes(tmp_path: Path) -> None:
    """Run each replicate in a distinct child process that has exited before return."""
    # Given: two bounded replicates of one minimal real worker case.
    base_config = make_config(tmp_path)
    tiny_case = replace(
        base_config.cases[0],
        name="tiny_real",
        block_size=4,
        batch_size=1,
        torch_num_threads=1,
    )
    config = replace(
        base_config,
        cases=(tiny_case,),
        replicates=2,
        minimum_replicates=2,
        worker_timeout_seconds=60.0,
    )

    # When: production orchestration launches the real module worker.
    artifacts = benchmark_module.run_benchmark_v2(config)

    # Then: worker-owned lifecycle evidence names unique, already-reaped child processes.
    assert artifacts.status == "complete"
    assert all(record.status == "ok" for record in artifacts.raw_replicates)
    worker_pids = [record.worker_pid for record in artifacts.raw_replicates]
    assert None not in worker_pids
    concrete_pids = cast("list[int]", worker_pids)
    assert len(set(concrete_pids)) == 2
    assert os.getpid() not in concrete_pids
    pid_exists = cast("Callable[[int], bool]", psutil.__dict__["pid_exists"])
    assert all(not pid_exists(worker_pid) for worker_pid in concrete_pids)
    for record in artifacts.raw_replicates:
        assert record.started_at_utc is not None
        assert record.ended_at_utc is not None
        started_at = datetime.fromisoformat(record.started_at_utc)
        ended_at = datetime.fromisoformat(record.ended_at_utc)
        assert started_at.tzinfo is not None
        assert started_at <= ended_at
