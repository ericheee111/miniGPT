"""Orchestrate randomized, fresh-process CPU Benchmark v2 replicates."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from typing_extensions import override

from minigpt.benchmark_v2_config import JsonValue, case_identity, resolved_config_sha256
from minigpt.benchmark_v2_report import RunStatus, capture_run_environment, write_run_artifacts
from minigpt.benchmark_v2_worker import (
    WORKER_PROTOCOL_VERSION,
    WorkerRequest,
    worker_request_document,
)

_GIT_SHORT_SHA_LENGTH = 12

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TextIO

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
_ENVIRONMENT_RESPONSE_KEYS = frozenset(
    {
        "platform",
        "python_version",
        "torch_version",
        "torch_num_threads",
        "torch_num_interop_threads",
        "logical_cpu_count",
        "requested_cpu_affinity",
        "effective_cpu_affinity",
        "relevant_environment_variables",
    }
)
_PEAK_RSS_METHODS = frozenset(
    {
        "windows_peak_working_set",
        "linux_getrusage_ru_maxrss",
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
    """Expose the complete durable evidence package for one benchmark invocation."""

    run_directory: Path
    status: RunStatus
    run_id: str
    tasks: tuple[BenchmarkTask, ...]
    raw_replicates: tuple[RawReplicate, ...]
    execution_order_path: Path
    raw_replicates_path: Path
    run_state_path: Path
    run_manifest_path: Path
    environment_path: Path
    resolved_config_path: Path
    summary_csv_path: Path
    summary_markdown_path: Path


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

    status: Literal["running", "complete", "partial", "failed"]
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


def _string_keyed_mapping(value: object, context: str) -> dict[str, object]:
    """Require one JSON object whose keys are strings."""
    if not isinstance(value, dict):
        _invalid_worker_response(f"{context} must be an object")
    raw_mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw_mapping):
        _invalid_worker_response(f"{context} keys must be strings")
    return cast("dict[str, object]", raw_mapping)


def _require_exact_keys(
    document: dict[str, object],
    expected_keys: frozenset[str],
    context: str,
) -> None:
    """Require an exact worker-protocol key set."""
    if frozenset(document) != expected_keys:
        _invalid_worker_response(f"{context} has an invalid field set")


def _integer(
    value: object,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Require one strict JSON integer with an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid_worker_response(f"{field} must be an integer")
    if positive and value <= 0:
        _invalid_worker_response(f"{field} must be positive")
    if non_negative and value < 0:
        _invalid_worker_response(f"{field} must be non-negative")
    return value


def _positive_finite_number(value: object, field: str) -> float:
    """Require one positive finite JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid_worker_response(f"{field} must be a number")
    try:
        number = float(value)
    except OverflowError:
        _invalid_worker_response(f"{field} must be positive and finite")
    if not math.isfinite(number) or number <= 0.0:
        _invalid_worker_response(f"{field} must be positive and finite")
    return number


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    """Require one string, optionally allowing the empty worker message."""
    if not isinstance(value, str) or (not allow_empty and not value):
        _invalid_worker_response(f"{field} must be a valid string")
    return value


def _parse_timestamp(value: object, field: str) -> str:
    """Require a timezone-aware worker lifecycle timestamp."""
    if not isinstance(value, str):
        _invalid_worker_response(f"{field} must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        _invalid_worker_response(f"{field} must be timezone-aware")
    return value


def _affinity(value: object, field: str) -> tuple[int, ...] | None:
    """Require a null or non-empty unique list of logical CPU IDs."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        _invalid_worker_response(f"{field} must be a non-empty list or null")
    raw_values = cast("list[object]", value)
    values = tuple(_integer(item, field, non_negative=True) for item in raw_values)
    if len(values) != len(set(values)):
        _invalid_worker_response(f"{field} must not contain duplicates")
    return values


def _worker_metadata(task: BenchmarkTask, response: dict[str, object]) -> tuple[int, str, str]:
    """Validate worker-owned process and lifecycle identity."""
    worker_pid = _integer(response["worker_pid"], "worker_pid", positive=True)
    started_at_utc = _parse_timestamp(response["started_at_utc"], "started_at_utc")
    ended_at_utc = _parse_timestamp(response["ended_at_utc"], "ended_at_utc")
    if datetime.fromisoformat(started_at_utc) > datetime.fromisoformat(ended_at_utc):
        _invalid_worker_response("worker lifecycle ends before it starts")
    if _string(response["case_identity"], "case_identity") != task.case_identity:
        _invalid_worker_response("case_identity does not match the task")
    if _string(response["case_name"], "case_name") != task.case.name:
        _invalid_worker_response("case_name does not match the task")
    replicate_index = _integer(response["replicate_index"], "replicate_index", non_negative=True)
    if replicate_index != task.replicate_index:
        _invalid_worker_response("replicate_index does not match the task")
    return worker_pid, started_at_utc, ended_at_utc


def _validate_environment(task: BenchmarkTask, raw_environment: object) -> None:
    """Validate exact nested worker environment and applied CPU controls."""
    environment = _string_keyed_mapping(raw_environment, "environment")
    _require_exact_keys(environment, _ENVIRONMENT_RESPONSE_KEYS, "environment")
    _ = _string(environment["platform"], "environment.platform")
    _ = _string(environment["python_version"], "environment.python_version")
    _ = _string(environment["torch_version"], "environment.torch_version")
    torch_num_threads = _integer(
        environment["torch_num_threads"],
        "environment.torch_num_threads",
        positive=True,
    )
    if torch_num_threads != task.case.torch_num_threads:
        _invalid_worker_response("environment.torch_num_threads does not match the task")
    torch_num_interop_threads = _integer(
        environment["torch_num_interop_threads"],
        "environment.torch_num_interop_threads",
        positive=True,
    )
    if torch_num_interop_threads != task.torch_num_interop_threads:
        _invalid_worker_response("environment.torch_num_interop_threads does not match the task")
    logical_cpu_count = environment["logical_cpu_count"]
    if logical_cpu_count is not None:
        _ = _integer(
            logical_cpu_count,
            "environment.logical_cpu_count",
            positive=True,
        )
    requested_affinity = _affinity(
        environment["requested_cpu_affinity"],
        "environment.requested_cpu_affinity",
    )
    if requested_affinity != task.cpu_affinity:
        _invalid_worker_response("environment.requested_cpu_affinity does not match the task")
    _ = _affinity(
        environment["effective_cpu_affinity"],
        "environment.effective_cpu_affinity",
    )
    environment_variables = _string_keyed_mapping(
        environment["relevant_environment_variables"],
        "environment.relevant_environment_variables",
    )
    if frozenset(environment_variables) != frozenset(task.relevant_environment_variables):
        _invalid_worker_response(
            "environment.relevant_environment_variables has an invalid field set"
        )
    if any(
        value is not None and not isinstance(value, str) for value in environment_variables.values()
    ):
        _invalid_worker_response(
            "environment.relevant_environment_variables values must be strings or null"
        )


def _validate_success_response(task: BenchmarkTask, response: dict[str, object]) -> None:
    """Validate all success metrics, timer evidence, memory, and environment fields."""
    warmup_steps = _integer(response["warmup_steps"], "warmup_steps", non_negative=True)
    if warmup_steps != task.warmup_steps:
        _invalid_worker_response("warmup_steps does not match the task")
    measurement_steps = _integer(
        response["measurement_steps"],
        "measurement_steps",
        positive=True,
    )
    if measurement_steps != task.measurement_steps:
        _invalid_worker_response("measurement_steps does not match the task")
    for field in ("elapsed_seconds", "step_time_ms", "tokens_per_second"):
        _ = _positive_finite_number(response[field], field)
    _ = _integer(response["tokens_per_step"], "tokens_per_step", positive=True)
    _ = _integer(response["parameter_count"], "parameter_count", positive=True)
    final_rss_mib = _positive_finite_number(response["final_rss_mib"], "final_rss_mib")
    peak_rss_mib = _positive_finite_number(response["peak_rss_mib"], "peak_rss_mib")
    if peak_rss_mib < final_rss_mib:
        _invalid_worker_response("peak_rss_mib must not be below final_rss_mib")
    peak_rss_method = _string(response["peak_rss_method"], "peak_rss_method")
    if peak_rss_method not in _PEAK_RSS_METHODS:
        _invalid_worker_response("peak_rss_method is unsupported")
    if response["peak_rss_sampling_interval_ms"] is not None:
        _invalid_worker_response("peak_rss_sampling_interval_ms must be null")
    _validate_environment(task, response["environment"])


def _validate_failure_response(response: dict[str, object]) -> None:
    """Validate worker-declared error type and message fields."""
    _ = _string(response["error_type"], "error_type")
    _ = _string(response["message"], "message", allow_empty=True)


def _parse_worker_response(
    task: BenchmarkTask, stdout: str
) -> tuple[dict[str, JsonValue], int, str, str]:
    """Parse an exact worker response and validate its task-owned identity."""
    raw_response = cast("object", json.loads(stdout))
    response = _string_keyed_mapping(raw_response, "worker response")
    status = response.get("status")
    expected_keys = _SUCCESS_RESPONSE_KEYS if status == "ok" else _FAILURE_RESPONSE_KEYS
    _require_exact_keys(response, expected_keys, "worker response")
    protocol_version = _integer(response["protocol_version"], "protocol_version")
    if protocol_version != WORKER_PROTOCOL_VERSION:
        _invalid_worker_response("worker response has an unsupported protocol version")
    if status not in {"ok", "error"}:
        _invalid_worker_response("worker response has an invalid status")
    worker_pid, started_at_utc, ended_at_utc = _worker_metadata(task, response)
    if status == "ok":
        _validate_success_response(task, response)
    else:
        _validate_failure_response(response)
    return (
        cast("dict[str, JsonValue]", response),
        worker_pid,
        started_at_utc,
        ended_at_utc,
    )


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

    if response["status"] == "error":
        error_type = cast("str", response["error_type"])
        message = cast("str", response["message"])
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


def _git_short_sha(environment_snapshot: dict[str, JsonValue] | None = None) -> str:
    """Read the current Git short SHA without making run creation depend on Git availability."""
    if environment_snapshot is not None:
        git = environment_snapshot.get("git")
        if isinstance(git, dict):
            commit_sha = git.get("commit_sha")
            if isinstance(commit_sha, str) and len(commit_sha) >= _GIT_SHORT_SHA_LENGTH:
                return commit_sha[:_GIT_SHORT_SHA_LENGTH]
        return "unknown00000"
    executable = shutil.which("git")
    if executable is None:
        return "nogit0000000"
    completed = subprocess.run(  # noqa: S603 - command and arguments are fixed Git metadata reads.
        [executable, "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    short_sha = completed.stdout.strip()
    return short_sha if completed.returncode == 0 and short_sha else "nogit0000000"


def _best_effort(operation: Callable[[], None]) -> None:
    """Attempt exceptional-path cleanup without replacing the original active exception."""
    with suppress(BaseException):
        operation()


def create_run_id(
    config: BenchmarkV2Config,
    *,
    created_at: datetime | None = None,
    environment_snapshot: dict[str, JsonValue] | None = None,
) -> str:
    """Return a collision-checked UTC/Git/config identity for one fresh run directory."""
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        msg = "created_at must be timezone-aware"
        raise ValueError(msg)
    utc_timestamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{utc_timestamp}-{_git_short_sha(environment_snapshot)}-"
        f"{resolved_config_sha256(config)[:12]}"
    )


def _final_status(tasks: tuple[BenchmarkTask, ...], records: list[RawReplicate]) -> RunStatus:
    """Classify complete, partial, and failed evidence without a zero-success partial state."""
    success_count = sum(record.status == "ok" for record in records)
    if success_count == 0:
        return "failed"
    expected = {(task.case_identity, task.replicate_index) for task in tasks}
    observed = {(record.case_identity, record.replicate_index) for record in records}
    if (
        len(records) == len(tasks)
        and len(observed) == len(records)
        and observed == expected
        and success_count == len(tasks)
    ):
        return "complete"
    return "partial"


def _append_durable_raw_record(raw_stream: TextIO, record: RawReplicate) -> None:
    """Append one complete JSON line or roll it back before propagating interruption."""
    durable_offset = raw_stream.tell()
    try:
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
        _ = raw_stream.seek(durable_offset)
        _ = raw_stream.truncate()
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
        raise


def run_benchmark_v2(
    config: BenchmarkV2Config,
    *,
    launcher: WorkerLauncher = _subprocess_launcher,
) -> BenchmarkV2Artifacts:
    """Execute randomized tasks sequentially and preserve a durable partial run."""
    started_at = datetime.now(UTC)
    environment_snapshot = capture_run_environment(config)
    tasks = expand_benchmark_tasks(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_id = create_run_id(
        config,
        created_at=started_at,
        environment_snapshot=environment_snapshot,
    )
    run_directory = config.output_root / run_id
    run_directory.mkdir(exist_ok=False)
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
    raw_stream = raw_replicates_path.open("x", encoding="utf-8", newline="\n")
    try:
        for task in tasks:
            record = execute_worker(
                task,
                timeout_seconds=config.worker_timeout_seconds,
                launcher=launcher,
            )
            _append_durable_raw_record(raw_stream, record)
            records.append(record)
    except (KeyboardInterrupt, Exception):
        status = _final_status(tasks, records)
        final_state = _run_state_document(
            run_id=run_id,
            config_sha256=config_sha256,
            progress=_RunProgress(
                status=status,
                expected_task_count=len(tasks),
                completed_task_count=len(records),
                failed_task_count=sum(record.status == "error" for record in records),
            ),
        )
        _best_effort(lambda: _write_json(run_state_path, final_state))
        _best_effort(raw_stream.close)
        report_finalized = False
        try:
            _ = write_run_artifacts(
                config=config,
                run_directory=run_directory,
                run_id=run_id,
                status=status,
                tasks=tasks,
                raw_replicates=tuple(records),
                started_at_utc=started_at.isoformat(),
                ended_at_utc=datetime.now(UTC).isoformat(),
                environment_snapshot=environment_snapshot,
            )
        except BaseException:  # noqa: BLE001 - preserve the original orchestration exception.
            report_finalized = False
        else:
            report_finalized = True
        if report_finalized:
            _best_effort(run_state_path.unlink)
        raise
    else:
        raw_stream.close()

    status = _final_status(tasks, records)
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
    report_artifacts = write_run_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_id,
        status=status,
        tasks=tasks,
        raw_replicates=tuple(records),
        started_at_utc=started_at.isoformat(),
        ended_at_utc=datetime.now(UTC).isoformat(),
        environment_snapshot=environment_snapshot,
    )
    run_state_path.unlink()
    return BenchmarkV2Artifacts(
        run_directory=run_directory,
        status=status,
        run_id=run_id,
        tasks=tasks,
        raw_replicates=tuple(records),
        execution_order_path=report_artifacts.execution_order_path,
        raw_replicates_path=report_artifacts.raw_replicates_path,
        run_state_path=run_state_path,
        run_manifest_path=report_artifacts.run_manifest_path,
        environment_path=report_artifacts.environment_path,
        resolved_config_path=report_artifacts.resolved_config_path,
        summary_csv_path=report_artifacts.summary_csv_path,
        summary_markdown_path=report_artifacts.summary_markdown_path,
    )
