"""Orchestrate randomized, fresh-process CPU Benchmark v2 replicates."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from typing_extensions import override

from minigpt.benchmark_v2_config import JsonValue, case_identity, resolved_config_sha256
from minigpt.benchmark_v2_worker import (
    WORKER_PROTOCOL_VERSION,
    WorkerRequest,
    worker_request_document,
)

if TYPE_CHECKING:
    from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config

_SUCCESS_RESPONSE_KEYS = frozenset(
    {
        "protocol_version",
        "status",
        "worker_pid",
        "started_at_utc",
        "ended_at_utc",
        "case_identity",
        "case_name",
        "replicate_index",
        "warmup_steps",
        "measurement_steps",
        "elapsed_seconds",
        "step_time_ms",
        "tokens_per_second",
        "tokens_per_step",
        "parameter_count",
        "final_rss_mib",
        "peak_rss_mib",
        "peak_rss_method",
        "peak_rss_sampling_interval_ms",
        "environment",
    }
)
_FAILURE_RESPONSE_KEYS = frozenset(
    {
        "protocol_version",
        "status",
        "worker_pid",
        "started_at_utc",
        "ended_at_utc",
        "case_identity",
        "case_name",
        "replicate_index",
        "error_type",
        "message",
    }
)


class WorkerLauncher(Protocol):
    """Launch one worker process with the exact subprocess boundary."""

    def __call__(
        self,
        command: list[str],
        request_json: str,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Return one completed worker process or raise its timeout."""
        ...


def _subprocess_launcher(
    command: list[str], request_json: str, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Launch the production worker with one bounded subprocess call."""
    return subprocess.run(  # noqa: S603 - command is the fixed current interpreter/module.
        command,
        input=request_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """Bind one case replicate to its complete worker execution context."""

    case: BenchmarkV2Case
    case_identity: str
    replicate_index: int
    benchmark_seed: int
    vocab_size: int
    warmup_steps: int
    measurement_steps: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawReplicate:
    """Preserve one ordered worker success or failure without dropping raw output."""

    status: Literal["ok", "error"]
    case_identity: str
    case_name: str
    replicate_index: int
    worker_pid: int | None
    started_at_utc: str | None
    ended_at_utc: str | None
    return_code: int | None
    error_type: str | None
    message: str | None
    stdout: str
    stderr: str
    worker_response: dict[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class BenchmarkV2Artifacts:
    """Expose the minimal durable Task 3 run seam for later report finalization."""

    run_directory: Path
    status: Literal["complete", "partial"]
    tasks: tuple[BenchmarkTask, ...]
    raw_replicates: tuple[RawReplicate, ...]
    execution_order_path: Path
    raw_replicates_path: Path
    run_state_path: Path


@dataclass(frozen=True, slots=True)
class InvalidWorkerResponseError(ValueError):
    """Report stdout that violates the versioned worker response protocol."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the worker response validation reason."""
        return f"invalid Benchmark v2 worker response: {self.reason}"


@dataclass(frozen=True, slots=True)
class _FailureEvidence:
    """Hold optional subprocess and worker-owned failure evidence."""

    return_code: int | None
    stdout: str
    stderr: str
    worker_pid: int | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    worker_response: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class _RunProgress:
    """Describe the current durable lifecycle counters."""

    status: Literal["running", "complete", "partial"]
    expected_task_count: int
    completed_task_count: int
    failed_task_count: int


def expand_benchmark_tasks(config: BenchmarkV2Config) -> tuple[BenchmarkTask, ...]:
    """Expand and deterministically shuffle every explicit case replicate."""
    tasks = [
        BenchmarkTask(
            case=case,
            case_identity=case_identity(config, case),
            replicate_index=replicate_index,
            benchmark_seed=config.benchmark_seed,
            vocab_size=config.vocab_size,
            warmup_steps=config.warmup_steps,
            measurement_steps=config.measurement_steps,
            torch_num_interop_threads=config.torch_num_interop_threads,
            cpu_affinity=config.cpu_affinity,
            relevant_environment_variables=config.relevant_environment_variables,
        )
        for case in config.cases
        for replicate_index in range(config.replicates)
    ]
    random.Random(config.benchmark_seed).shuffle(tasks)  # noqa: S311 - reproducible ordering.
    return tuple(tasks)


def _worker_request(task: BenchmarkTask) -> WorkerRequest:
    """Translate one expanded task into the versioned worker protocol."""
    return WorkerRequest(
        protocol_version=WORKER_PROTOCOL_VERSION,
        case_identity=task.case_identity,
        replicate_index=task.replicate_index,
        case=task.case,
        benchmark_seed=task.benchmark_seed,
        vocab_size=task.vocab_size,
        warmup_steps=task.warmup_steps,
        measurement_steps=task.measurement_steps,
        torch_num_interop_threads=task.torch_num_interop_threads,
        cpu_affinity=task.cpu_affinity,
        relevant_environment_variables=task.relevant_environment_variables,
    )


def _text_output(output: str | bytes | None) -> str:
    """Normalize subprocess timeout output without losing available bytes."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _failure_record(
    task: BenchmarkTask,
    error_type: str,
    message: str,
    evidence: _FailureEvidence,
) -> RawReplicate:
    """Build one orchestrator-owned raw failure while retaining worker evidence."""
    return RawReplicate(
        status="error",
        case_identity=task.case_identity,
        case_name=task.case.name,
        replicate_index=task.replicate_index,
        worker_pid=evidence.worker_pid,
        started_at_utc=evidence.started_at_utc,
        ended_at_utc=evidence.ended_at_utc,
        return_code=evidence.return_code,
        error_type=error_type,
        message=message,
        stdout=evidence.stdout,
        stderr=evidence.stderr,
        worker_response=evidence.worker_response,
    )


def _invalid_worker_response(reason: str) -> Never:
    """Raise one consistently typed worker response validation error."""
    raise InvalidWorkerResponseError(reason)


def _parse_timestamp(value: object, field: str) -> str:
    """Require a timezone-aware worker lifecycle timestamp."""
    if not isinstance(value, str):
        _invalid_worker_response(f"{field} must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        _invalid_worker_response(f"{field} must be timezone-aware")
    return value


def _worker_metadata(task: BenchmarkTask, response: dict[str, JsonValue]) -> tuple[int, str, str]:
    """Validate worker-owned process and lifecycle identity."""
    worker_pid = response["worker_pid"]
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
        _invalid_worker_response("worker_pid must be a positive integer")
    started_at_utc = _parse_timestamp(response["started_at_utc"], "started_at_utc")
    ended_at_utc = _parse_timestamp(response["ended_at_utc"], "ended_at_utc")
    if datetime.fromisoformat(started_at_utc) > datetime.fromisoformat(ended_at_utc):
        _invalid_worker_response("worker lifecycle ends before it starts")
    if response["case_identity"] != task.case_identity:
        _invalid_worker_response("case_identity does not match the task")
    if response["case_name"] != task.case.name:
        _invalid_worker_response("case_name does not match the task")
    if response["replicate_index"] != task.replicate_index:
        _invalid_worker_response("replicate_index does not match the task")
    return worker_pid, started_at_utc, ended_at_utc


def _parse_worker_response(
    task: BenchmarkTask, stdout: str
) -> tuple[dict[str, JsonValue], int, str, str]:
    """Parse an exact worker response and validate its task-owned identity."""
    raw_response = cast("object", json.loads(stdout))
    if not isinstance(raw_response, dict):
        _invalid_worker_response("worker response must be an object")
    raw_mapping = cast("dict[object, object]", raw_response)
    if any(not isinstance(key, str) for key in raw_mapping):
        _invalid_worker_response("worker response keys must be strings")
    response = cast("dict[str, JsonValue]", raw_mapping)
    status = response.get("status")
    expected_keys = _SUCCESS_RESPONSE_KEYS if status == "ok" else _FAILURE_RESPONSE_KEYS
    if frozenset(response) != expected_keys:
        _invalid_worker_response("worker response has an invalid field set")
    if response["protocol_version"] != WORKER_PROTOCOL_VERSION:
        _invalid_worker_response("worker response has an unsupported protocol version")
    if status not in {"ok", "error"}:
        _invalid_worker_response("worker response has an invalid status")
    worker_pid, started_at_utc, ended_at_utc = _worker_metadata(task, response)
    return response, worker_pid, started_at_utc, ended_at_utc


def _invalid_response_record(
    task: BenchmarkTask,
    completed: subprocess.CompletedProcess[str],
    error: Exception,
) -> RawReplicate:
    """Preserve a completed process whose stdout is not a valid worker response."""
    return _failure_record(
        task,
        error_type="InvalidWorkerResponse",
        message=str(error),
        evidence=_FailureEvidence(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ),
    )


def execute_worker(
    task: BenchmarkTask,
    timeout_seconds: float,
    *,
    launcher: WorkerLauncher = _subprocess_launcher,
) -> RawReplicate:
    """Run one task in a fresh subprocess and convert every ordinary failure to raw evidence."""
    request_json = json.dumps(
        worker_request_document(_worker_request(task)),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    command = [sys.executable, "-m", "minigpt.benchmark_v2_worker"]
    try:
        completed = launcher(command, request_json, timeout_seconds)
    except subprocess.TimeoutExpired as error:
        return _failure_record(
            task,
            error_type="WorkerTimeout",
            message=f"worker exceeded timeout of {timeout_seconds} seconds",
            evidence=_FailureEvidence(
                return_code=None,
                stdout=_text_output(error.stdout),
                stderr=_text_output(error.stderr),
            ),
        )

    try:
        response, worker_pid, started_at_utc, ended_at_utc = _parse_worker_response(
            task, completed.stdout
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if completed.returncode != 0:
            parse_failure = _failure_record(
                task,
                error_type="WorkerProcessError",
                message=f"worker exited with return code {completed.returncode}",
                evidence=_FailureEvidence(
                    return_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                ),
            )
        else:
            parse_failure = _invalid_response_record(task, completed, error)
        return parse_failure

    if completed.returncode != 0:
        return _failure_record(
            task,
            error_type="WorkerProcessError",
            message=f"worker exited with return code {completed.returncode}",
            evidence=_FailureEvidence(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                worker_pid=worker_pid,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                worker_response=response,
            ),
        )
    if response["status"] == "error":
        error_type = response["error_type"]
        message = response["message"]
        if not isinstance(error_type, str) or not isinstance(message, str):
            return _invalid_response_record(
                task,
                completed,
                ValueError("worker failure fields must be strings"),
            )
        return _failure_record(
            task,
            error_type=error_type,
            message=message,
            evidence=_FailureEvidence(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                worker_pid=worker_pid,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                worker_response=response,
            ),
        )
    return RawReplicate(
        status="ok",
        case_identity=task.case_identity,
        case_name=task.case.name,
        replicate_index=task.replicate_index,
        worker_pid=worker_pid,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        return_code=completed.returncode,
        error_type=None,
        message=None,
        stdout=completed.stdout,
        stderr=completed.stderr,
        worker_response=response,
    )


def raw_replicate_document(record: RawReplicate) -> dict[str, JsonValue]:
    """Serialize one typed raw replicate without discarding subprocess evidence."""
    return {
        "status": record.status,
        "case_identity": record.case_identity,
        "case_name": record.case_name,
        "replicate_index": record.replicate_index,
        "worker_pid": record.worker_pid,
        "started_at_utc": record.started_at_utc,
        "ended_at_utc": record.ended_at_utc,
        "return_code": record.return_code,
        "error_type": record.error_type,
        "message": record.message,
        "stdout": record.stdout,
        "stderr": record.stderr,
        "worker_response": record.worker_response,
    }


def _write_json(path: Path, document: JsonValue) -> None:
    """Atomically write one compact JSON document for durable run state."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            document,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        _ = stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _ = temporary_path.replace(path)


def _execution_order_document(tasks: tuple[BenchmarkTask, ...]) -> list[JsonValue]:
    """Serialize the fixed randomized order before any worker starts."""
    return [
        {
            "case_identity": task.case_identity,
            "case_name": task.case.name,
            "replicate_index": task.replicate_index,
        }
        for task in tasks
    ]


def _run_state_document(
    *,
    run_id: str,
    config_sha256: str,
    progress: _RunProgress,
) -> dict[str, JsonValue]:
    """Build the minimal Task 3 lifecycle document for later finalization."""
    return {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "status": progress.status,
        "expected_task_count": progress.expected_task_count,
        "completed_task_count": progress.completed_task_count,
        "failed_task_count": progress.failed_task_count,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }


def run_benchmark_v2(
    config: BenchmarkV2Config,
    *,
    launcher: WorkerLauncher = _subprocess_launcher,
) -> BenchmarkV2Artifacts:
    """Execute randomized tasks sequentially and preserve a durable partial run."""
    tasks = expand_benchmark_tasks(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_directory = Path(
        tempfile.mkdtemp(prefix=f".{config.experiment_name}-", dir=config.output_root)
    )
    run_id = run_directory.name
    config_sha256 = resolved_config_sha256(config)
    execution_order_path = run_directory / "execution_order.json"
    raw_replicates_path = run_directory / "raw_replicates.jsonl"
    run_state_path = run_directory / "run_state.json"
    _write_json(execution_order_path, _execution_order_document(tasks))
    _write_json(
        run_state_path,
        _run_state_document(
            run_id=run_id,
            config_sha256=config_sha256,
            progress=_RunProgress(
                status="running",
                expected_task_count=len(tasks),
                completed_task_count=0,
                failed_task_count=0,
            ),
        ),
    )
    records: list[RawReplicate] = []
    with raw_replicates_path.open("x", encoding="utf-8", newline="\n") as raw_stream:
        try:
            for task in tasks:
                record = execute_worker(
                    task,
                    timeout_seconds=config.worker_timeout_seconds,
                    launcher=launcher,
                )
                records.append(record)
                _ = raw_stream.write(
                    json.dumps(
                        raw_replicate_document(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                raw_stream.flush()
                os.fsync(raw_stream.fileno())
        except KeyboardInterrupt:
            _write_json(
                run_state_path,
                _run_state_document(
                    run_id=run_id,
                    config_sha256=config_sha256,
                    progress=_RunProgress(
                        status="partial",
                        expected_task_count=len(tasks),
                        completed_task_count=len(records),
                        failed_task_count=sum(record.status == "error" for record in records),
                    ),
                ),
            )
            raise

    status: Literal["complete", "partial"] = (
        "complete"
        if len(records) == len(tasks) and all(record.status == "ok" for record in records)
        else "partial"
    )
    _write_json(
        run_state_path,
        _run_state_document(
            run_id=run_id,
            config_sha256=config_sha256,
            progress=_RunProgress(
                status=status,
                expected_task_count=len(tasks),
                completed_task_count=len(records),
                failed_task_count=sum(record.status == "error" for record in records),
            ),
        ),
    )
    return BenchmarkV2Artifacts(
        run_directory=run_directory,
        status=status,
        tasks=tasks,
        raw_replicates=tuple(records),
        execution_order_path=execution_order_path,
        raw_replicates_path=raw_replicates_path,
        run_state_path=run_state_path,
    )
