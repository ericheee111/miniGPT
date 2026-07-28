"""Compare hash-bound Benchmark v2 evidence without making unsafe regression claims."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml
from typing_extensions import override

from minigpt.benchmark_v2_report import load_run_manifest
from minigpt.benchmark_v2_statistics import BenchmarkV2Summary, summarize_replicates

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue
    from minigpt.benchmark_v2_report import RunManifest

ComparisonVerdict = Literal["pass", "fail", "not_comparable"]
_SUMMARY_FIELDS = tuple(BenchmarkV2Summary.__dataclass_fields__)
_STABILITY_VALUES = frozenset({"insufficient_samples", "unstable", "stable"})
_ENVIRONMENT_FIELDS = (
    "platform",
    "machine",
    "cpu_name",
    "physical_cpu_count",
    "logical_cpu_count",
    "python_version",
    "torch_version",
    "numpy_version",
    "cuda_available",
    "parent_torch_num_threads",
    "parent_torch_num_interop_threads",
    "configured_torch_num_interop_threads",
    "configured_cpu_affinity",
    "relevant_environment_variables",
    "process_priority",
    "power_scheme",
)
_REQUIRED_CPU_COMPATIBILITY_FIELDS = (
    "cpu_name",
    "physical_cpu_count",
    "logical_cpu_count",
)
_RUN_ENVIRONMENT_KEYS = frozenset({"captured_before_first_worker", "git", *_ENVIRONMENT_FIELDS})
_ENVIRONMENT_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_status",
        "config_sha256",
        "run_environment",
        "worker_environments",
    }
)
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_HEX_LENGTH = 64
_ENVIRONMENT_SCHEMA_VERSION = 2
_WORKER_ENVIRONMENT_KEYS = frozenset(
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
_WORKER_TEXT_FIELDS = ("platform", "python_version", "torch_version")
_RAW_REPLICATE_KEYS = frozenset(
    {
        "status",
        "case_identity",
        "case_name",
        "replicate_index",
        "worker_pid",
        "started_at_utc",
        "ended_at_utc",
        "return_code",
        "error_type",
        "message",
        "stdout",
        "stderr",
        "worker_response",
    }
)
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
_PEAK_RSS_METHODS = frozenset({"windows_peak_working_set", "linux_getrusage_ru_maxrss"})
_EXECUTION_ORDER_KEYS = frozenset(
    {
        "execution_index",
        "task_id",
        "case_name",
        "case_identity",
        "replicate_index",
        "worker_seed",
        "worker_pid",
        "status",
    }
)


@dataclass(slots=True)
class InvalidComparisonInputError(ValueError):
    """Describe an invalid or internally inconsistent input run package."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the artifact path and exact validation reason."""
        return f"invalid benchmark comparison input {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class EnvironmentMismatch:
    """Record one performance-relevant environment field that differs between runs."""

    field: str
    baseline: JsonValue
    candidate: JsonValue


@dataclass(frozen=True, slots=True)
class StepTimeComparison:
    """Describe a candidate step-time change relative to a baseline median."""

    step_time_change_percent: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class CaseComparison:
    """Retain descriptive per-case deltas and their guarded eligibility state."""

    case_identity: str
    baseline_case_name: str
    candidate_case_name: str
    baseline_median_step_time_ms: float | None
    candidate_median_step_time_ms: float | None
    baseline_median_tokens_per_second: float | None
    candidate_median_tokens_per_second: float | None
    step_time_change_percent: float | None
    throughput_change_percent: float | None
    reasons: tuple[str, ...]
    regressed: bool | None


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Represent all evidence, differences, and the only allowed regression verdict."""

    baseline_manifest_path: Path
    candidate_manifest_path: Path
    baseline_run_id: str
    candidate_run_id: str
    regression_threshold_percent: float
    environment_mismatches: tuple[EnvironmentMismatch, ...]
    missing_case_identities: tuple[str, ...]
    extra_case_identities: tuple[str, ...]
    reasons: tuple[str, ...]
    case_comparisons: tuple[CaseComparison, ...]
    verdict: ComparisonVerdict


@dataclass(frozen=True, slots=True)
class ComparisonArtifacts:
    """Expose the two new outputs written beside the candidate manifest."""

    json_path: Path
    markdown_path: Path


@dataclass(frozen=True, slots=True)
class _ComparisonInput:
    """Bind a validated manifest to its parsed environment, summaries, and threshold."""

    manifest_path: Path
    manifest: RunManifest
    run_environment: dict[str, JsonValue]
    worker_environment_signatures: tuple[tuple[object, ...], ...]
    case_worker_controls: tuple[tuple[str, frozenset[_WorkerControls]], ...]
    summaries: tuple[BenchmarkV2Summary, ...]
    regression_threshold_percent: float


@dataclass(frozen=True, slots=True)
class _WorkerControls:
    """Bind all worker-observed methodology controls that can affect timing comparison."""

    protocol_version: int
    peak_rss_method: str
    environment_signature: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _RawSummaryRecord:
    """Expose validated raw-success fields to the existing pure statistics implementation."""

    case_identity: str
    case_name: str
    replicate_index: int
    worker_pid: int | None
    worker_response: dict[str, JsonValue] | None
    status: Literal["ok", "error"]
    worker_controls: _WorkerControls | None


@dataclass(frozen=True, slots=True)
class _Methodology:
    """Bind all configuration controls needed to validate summary eligibility."""

    regression_threshold_percent: float
    minimum_replicates: int
    max_cv_percent: float


def _require_mapping(value: object, path: Path, context: str) -> dict[str, object]:
    """Require a JSON/YAML mapping with string keys before reading its contents."""
    if not isinstance(value, dict):
        raise InvalidComparisonInputError(path, f"{context} must be an object with string keys")
    mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        raise InvalidComparisonInputError(path, f"{context} must be an object with string keys")
    return cast("dict[str, object]", mapping)


def _reject_json_constant(value: str) -> object:
    """Refuse JavaScript-style non-finite constants that Python's permissive decoder accepts."""
    raise InvalidComparisonInputError(Path("<json>"), f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one decoded JSON object while rejecting duplicate keys at every nesting level."""
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise InvalidComparisonInputError(Path("<json>"), f"duplicate JSON object key {key!r}")
        document[key] = value
    return document


def _strict_json_document(content: bytes, path: Path, context: str) -> object:
    """Decode one UTF-8 JSON artifact with duplicate-key and non-finite-value rejection."""
    try:
        return cast(
            "object",
            json.loads(
                content.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except UnicodeDecodeError as error:
        raise InvalidComparisonInputError(path, f"{context} is not valid UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise InvalidComparisonInputError(path, f"{context} is not valid JSON") from error


def _require_sha256(value: object, path: Path, context: str) -> str:
    """Require the exact lowercase digest form used for case and config identities."""
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in _SHA256_HEX_DIGITS for character in value)
    ):
        raise InvalidComparisonInputError(path, f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_integer(value: str, path: Path, context: str) -> int:
    """Parse a non-negative integer CSV cell without accepting float spellings."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise InvalidComparisonInputError(path, f"{context} must be an integer") from error
    if parsed < 0 or str(parsed) != value:
        raise InvalidComparisonInputError(path, f"{context} must be a non-negative integer")
    return parsed


def _optional_worker_pid(value: object, path: Path, context: str) -> int | None:
    """Read an absent or positive worker PID without treating booleans as integers."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidComparisonInputError(path, f"{context} must be a positive integer or null")
    return value


def _integer_value(
    value: object, path: Path, context: str, *, positive: bool, non_negative: bool
) -> int:
    """Require one strict JSON integer with the requested lower-bound contract."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidComparisonInputError(path, f"{context} must be an integer")
    if positive and value <= 0:
        raise InvalidComparisonInputError(path, f"{context} must be positive")
    if non_negative and value < 0:
        raise InvalidComparisonInputError(path, f"{context} must be non-negative")
    return value


def _positive_finite_number(value: object, path: Path, context: str) -> float:
    """Require a finite positive JSON number while rejecting booleans and non-finite floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidComparisonInputError(path, f"{context} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise InvalidComparisonInputError(path, f"{context} must be a positive finite number")
    return number


def _nonempty_string(value: object, path: Path, context: str) -> str:
    """Require one protocol string that is not empty."""
    if not isinstance(value, str) or not value:
        raise InvalidComparisonInputError(path, f"{context} must be a non-empty string")
    return value


def _timestamp(value: object, path: Path, context: str) -> datetime:
    """Require one timezone-aware ISO timestamp and return it for lifecycle ordering checks."""
    if not isinstance(value, str):
        raise InvalidComparisonInputError(path, f"{context} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidComparisonInputError(path, f"{context} must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise InvalidComparisonInputError(path, f"{context} must be timezone-aware")
    return parsed


def _validate_raw_outer_fields(
    document: dict[str, object], manifest_path: Path, status: Literal["ok", "error"]
) -> int | None:
    """Validate every outer raw-record field before interpreting its worker response evidence."""
    worker_pid = _optional_worker_pid(document["worker_pid"], manifest_path, "raw worker_pid")
    started = document["started_at_utc"]
    ended = document["ended_at_utc"]
    if (started is None) != (ended is None):
        raise InvalidComparisonInputError(
            manifest_path, "raw worker timestamps must both be null or set"
        )
    if started is not None and _timestamp(
        started, manifest_path, "raw started_at_utc"
    ) > _timestamp(ended, manifest_path, "raw ended_at_utc"):
        raise InvalidComparisonInputError(
            manifest_path, "raw worker lifecycle ends before it starts"
        )
    if not isinstance(document["stdout"], str) or not isinstance(document["stderr"], str):
        raise InvalidComparisonInputError(manifest_path, "raw stdout and stderr must be strings")
    return_code = document["return_code"]
    if return_code is not None:
        _ = _integer_value(
            return_code, manifest_path, "raw return_code", positive=False, non_negative=False
        )
    error_type = document["error_type"]
    message = document["message"]
    if error_type is not None and not isinstance(error_type, str):
        raise InvalidComparisonInputError(manifest_path, "raw error_type must be a string or null")
    if message is not None and not isinstance(message, str):
        raise InvalidComparisonInputError(manifest_path, "raw message must be a string or null")
    if status == "ok" and (
        worker_pid is None
        or started is None
        or return_code != 0
        or error_type is not None
        or message is not None
    ):
        raise InvalidComparisonInputError(
            manifest_path, "successful raw record has invalid lifecycle fields"
        )
    if status == "error" and (
        not isinstance(error_type, str) or not error_type or not isinstance(message, str)
    ):
        raise InvalidComparisonInputError(
            manifest_path, "failed raw record has invalid lifecycle fields"
        )
    return worker_pid


def _require_number(value: str, path: Path, context: str, *, nullable: bool) -> float | None:
    """Parse a finite CSV number while preserving documented empty nullable metrics."""
    if nullable and value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise InvalidComparisonInputError(path, f"{context} must be a finite number") from error
    if not math.isfinite(parsed):
        raise InvalidComparisonInputError(path, f"{context} must be a finite number")
    return parsed


def _read_bound_artifact(manifest_path: Path, manifest: RunManifest, name: str) -> bytes:
    """Read and independently recheck an artifact required by the strict manifest loader."""
    entry = next((entry for entry in manifest.artifacts if entry.path == name), None)
    if entry is None:
        raise InvalidComparisonInputError(manifest_path, f"manifest does not bind {name}")
    path = manifest_path.parent / name
    try:
        content = path.read_bytes()
    except OSError as error:
        raise InvalidComparisonInputError(manifest_path, f"cannot read bound {name}") from error
    if len(content) != entry.size_bytes or hashlib.sha256(content).hexdigest() != entry.sha256:
        raise InvalidComparisonInputError(
            manifest_path, f"bound {name} changed after manifest validation"
        )
    return content


def _worker_integer(
    environment: dict[str, object], path: Path, field: str, *, nullable: bool
) -> int | None:
    """Normalize one positive worker control, preserving documented null CPU counts."""
    item = environment[field]
    if nullable and item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise InvalidComparisonInputError(path, f"worker environment {field} is invalid")
    return item


def _worker_affinity(
    environment: dict[str, object], path: Path, field: str
) -> tuple[int, ...] | None:
    """Normalize one actual/requested worker affinity without accepting duplicate CPUs."""
    item = environment[field]
    if item is None:
        return None
    if not isinstance(item, list) or not item:
        raise InvalidComparisonInputError(path, f"worker environment {field} is invalid")
    raw_values = cast("list[object]", item)
    if any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in raw_values):
        raise InvalidComparisonInputError(path, f"worker environment {field} is invalid")
    values = tuple(cast("int", cpu) for cpu in raw_values)
    if len(values) != len(set(values)):
        raise InvalidComparisonInputError(path, f"worker environment {field} has duplicate CPUs")
    return values


def _worker_variables(
    environment: dict[str, object], path: Path
) -> tuple[tuple[str, str | None], ...]:
    """Normalize the named environment controls that workers actually observed."""
    variables = _require_mapping(
        environment["relevant_environment_variables"], path, "worker environment variables"
    )
    if any(item is not None and not isinstance(item, str) for item in variables.values()):
        raise InvalidComparisonInputError(path, "worker environment variables are invalid")
    return tuple(sorted((name, cast("str | None", item)) for name, item in variables.items()))


def _worker_environment_signature(value: object, path: Path) -> tuple[object, ...]:
    """Validate and normalize the worker-applied controls relevant to performance comparison."""
    environment = _require_mapping(value, path, "worker environment")
    if frozenset(environment) != _WORKER_ENVIRONMENT_KEYS:
        raise InvalidComparisonInputError(path, "worker environment has an invalid field set")
    text_values = tuple(environment[field] for field in _WORKER_TEXT_FIELDS)
    if any(not isinstance(item, str) or not item for item in text_values):
        raise InvalidComparisonInputError(path, "worker environment text fields are invalid")
    return (
        *(cast("str", item) for item in text_values),
        _worker_integer(environment, path, "torch_num_threads", nullable=False),
        _worker_integer(environment, path, "torch_num_interop_threads", nullable=False),
        _worker_integer(environment, path, "logical_cpu_count", nullable=True),
        _worker_affinity(environment, path, "requested_cpu_affinity"),
        _worker_affinity(environment, path, "effective_cpu_affinity"),
        _worker_variables(environment, path),
    )


def _raw_record_identity(
    document: dict[str, object], manifest_path: Path, index: int
) -> tuple[Literal["ok", "error"], str, str, int]:
    """Validate one outer raw-record identity before reconciling task counts."""
    status = document["status"]
    identity = _require_sha256(
        document["case_identity"], manifest_path, "raw replicate case identity"
    )
    name = document["case_name"]
    replicate = document["replicate_index"]
    if (
        status not in {"ok", "error"}
        or not isinstance(name, str)
        or not name
        or isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or replicate < 0
    ):
        raise InvalidComparisonInputError(
            manifest_path, f"raw replicate line {index} has invalid identity"
        )
    return cast("Literal['ok', 'error']", status), identity, name, replicate


def _successful_raw_record(
    document: dict[str, object],
    manifest_path: Path,
    task: tuple[str, str, int, int | None],
) -> _RawSummaryRecord:
    """Validate the metrics and actual controls reported by one successful worker."""
    identity, name, replicate, worker_pid = task
    response = _require_mapping(
        document["worker_response"], manifest_path, "successful raw worker response"
    )
    if frozenset(response) != _SUCCESS_RESPONSE_KEYS:
        raise InvalidComparisonInputError(
            manifest_path, "successful raw worker response has an invalid field set"
        )
    if (
        response["protocol_version"] != 1
        or response["status"] != "ok"
        or response.get("worker_pid") != worker_pid
        or response.get("started_at_utc") != document["started_at_utc"]
        or response.get("ended_at_utc") != document["ended_at_utc"]
        or response.get("case_identity") != identity
        or response.get("case_name") != name
        or response.get("replicate_index") != replicate
    ):
        raise InvalidComparisonInputError(
            manifest_path, "successful raw worker response disagrees with raw record"
        )
    _ = _optional_worker_pid(response["worker_pid"], manifest_path, "worker response PID")
    started = _timestamp(response["started_at_utc"], manifest_path, "worker response start time")
    ended = _timestamp(response["ended_at_utc"], manifest_path, "worker response end time")
    if started > ended:
        raise InvalidComparisonInputError(
            manifest_path, "worker response lifecycle ends before it starts"
        )
    _ = _integer_value(
        response["replicate_index"],
        manifest_path,
        "worker response replicate_index",
        positive=False,
        non_negative=True,
    )
    _ = _integer_value(
        response["warmup_steps"],
        manifest_path,
        "worker response warmup_steps",
        positive=False,
        non_negative=True,
    )
    _ = _integer_value(
        response["measurement_steps"],
        manifest_path,
        "worker response measurement_steps",
        positive=True,
        non_negative=False,
    )
    for field in ("elapsed_seconds", "step_time_ms", "tokens_per_second"):
        _ = _positive_finite_number(response[field], manifest_path, f"worker response {field}")
    final_rss_mib = _positive_finite_number(
        response["final_rss_mib"], manifest_path, "worker response final_rss_mib"
    )
    peak_rss_mib = _positive_finite_number(
        response["peak_rss_mib"], manifest_path, "worker response peak_rss_mib"
    )
    _ = _integer_value(
        response["tokens_per_step"],
        manifest_path,
        "worker response tokens_per_step",
        positive=True,
        non_negative=False,
    )
    _ = _integer_value(
        response["parameter_count"],
        manifest_path,
        "worker response parameter_count",
        positive=True,
        non_negative=False,
    )
    if peak_rss_mib < final_rss_mib:
        raise InvalidComparisonInputError(
            manifest_path, "worker response peak_rss_mib must not be below final_rss_mib"
        )
    peak_method = _nonempty_string(
        response["peak_rss_method"], manifest_path, "worker response peak_rss_method"
    )
    if (
        peak_method not in _PEAK_RSS_METHODS
        or response["peak_rss_sampling_interval_ms"] is not None
    ):
        raise InvalidComparisonInputError(
            manifest_path, "worker response has invalid peak RSS evidence"
        )
    return _RawSummaryRecord(
        identity,
        name,
        replicate,
        worker_pid,
        cast("dict[str, JsonValue]", response),
        status="ok",
        worker_controls=_WorkerControls(
            protocol_version=1,
            peak_rss_method=peak_method,
            environment_signature=_worker_environment_signature(
                response.get("environment"), manifest_path
            ),
        ),
    )


def _failed_raw_record(
    document: dict[str, object],
    manifest_path: Path,
    task: tuple[str, str, int, int | None],
) -> _RawSummaryRecord:
    """Validate optional protocol failure evidence while retaining subprocess-only failures."""
    identity, name, replicate, worker_pid = task
    raw_response = document["worker_response"]
    if raw_response is not None:
        if worker_pid is None:
            raise InvalidComparisonInputError(
                manifest_path, "worker-declared failure requires a positive outer worker_pid"
            )
        started = document["started_at_utc"]
        ended = document["ended_at_utc"]
        if started is None or ended is None:
            raise InvalidComparisonInputError(
                manifest_path, "worker-declared failure requires outer lifecycle timestamps"
            )
        if _timestamp(started, manifest_path, "worker-declared failure start") > _timestamp(
            ended, manifest_path, "worker-declared failure end"
        ):
            raise InvalidComparisonInputError(
                manifest_path, "worker-declared failure lifecycle ends before it starts"
            )
        return_code = document["return_code"]
        if isinstance(return_code, bool) or not isinstance(return_code, int) or return_code == 0:
            raise InvalidComparisonInputError(
                manifest_path, "worker-declared failure requires a nonzero outer return_code"
            )
        response = _require_mapping(raw_response, manifest_path, "failed raw worker response")
        if frozenset(response) != _FAILURE_RESPONSE_KEYS:
            raise InvalidComparisonInputError(
                manifest_path, "failed raw worker response has an invalid field set"
            )
        _ = _nonempty_string(response["error_type"], manifest_path, "worker failure error_type")
        if not isinstance(response["message"], str):
            raise InvalidComparisonInputError(
                manifest_path, "worker failure message must be a string"
            )
        if (
            response["protocol_version"] != 1
            or response["status"] != "error"
            or response["worker_pid"] != worker_pid
            or response["case_identity"] != identity
            or response["case_name"] != name
            or response["replicate_index"] != replicate
            or response["started_at_utc"] != document["started_at_utc"]
            or response["ended_at_utc"] != document["ended_at_utc"]
            or response["error_type"] != document["error_type"]
            or response["message"] != document["message"]
        ):
            raise InvalidComparisonInputError(
                manifest_path, "failed raw worker response disagrees with raw record"
            )
    return _RawSummaryRecord(
        identity, name, replicate, worker_pid, None, status="error", worker_controls=None
    )


def _parse_raw_record(line: str, manifest_path: Path, index: int) -> _RawSummaryRecord:
    """Parse one exact raw JSONL record without collapsing worker failure evidence."""
    if not line:
        raise InvalidComparisonInputError(manifest_path, f"raw replicate line {index} is empty")
    document = _require_mapping(
        _strict_json_document(line.encode("utf-8"), manifest_path, f"raw replicate line {index}"),
        manifest_path,
        f"raw replicate line {index}",
    )
    if frozenset(document) != _RAW_REPLICATE_KEYS:
        raise InvalidComparisonInputError(
            manifest_path, f"raw replicate line {index} has an invalid field set"
        )
    status, identity, name, replicate = _raw_record_identity(document, manifest_path, index)
    worker_pid = _validate_raw_outer_fields(document, manifest_path, status)
    task = (identity, name, replicate, worker_pid)
    if status == "error":
        return _failed_raw_record(document, manifest_path, task)
    return _successful_raw_record(document, manifest_path, task)


def _load_raw_records(manifest_path: Path, manifest: RunManifest) -> tuple[_RawSummaryRecord, ...]:
    """Strictly reconcile bound raw records with manifest counts and recomputed case statistics."""
    content = _read_bound_artifact(manifest_path, manifest, "raw_replicates.jsonl")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InvalidComparisonInputError(
            manifest_path, "raw_replicates.jsonl is not valid UTF-8"
        ) from error
    records = tuple(
        _parse_raw_record(line, manifest_path, index) for index, line in enumerate(lines, 1)
    )
    tasks = {(record.case_identity, record.replicate_index) for record in records}
    if len(tasks) != len(records):
        raise InvalidComparisonInputError(
            manifest_path, "raw replicate task identities are not unique"
        )
    failures = sum(record.status == "error" for record in records)
    if (
        len(records) != manifest.completed_task_count
        or sum(record.status == "ok" for record in records) != manifest.successful_task_count
        or failures != manifest.failed_task_count
    ):
        raise InvalidComparisonInputError(
            manifest_path, "raw replicate counts disagree with manifest"
        )
    if manifest.status == "complete" and (
        len(records) != manifest.expected_task_count or failures != 0
    ):
        raise InvalidComparisonInputError(
            manifest_path, "complete manifest has incomplete or failed raw replicates"
        )
    return records


def _execution_entry(
    value: object, manifest_path: Path, index: int
) -> tuple[str, int, str, int | None, Literal["ok", "error", "pending"]]:
    """Validate one immutable execution-order entry before reconciling it with raw evidence."""
    entry = _require_mapping(value, manifest_path, f"execution_order entry {index}")
    if frozenset(entry) != _EXECUTION_ORDER_KEYS:
        raise InvalidComparisonInputError(
            manifest_path, f"execution_order entry {index} has an invalid field set"
        )
    identity = _require_sha256(
        entry["case_identity"], manifest_path, "execution-order case identity"
    )
    replicate = entry["replicate_index"]
    seed = entry["worker_seed"]
    name = entry["case_name"]
    status = entry["status"]
    if (
        entry["execution_index"] != index
        or not isinstance(name, str)
        or not name
        or isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or replicate < 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or entry["task_id"] != f"{identity}:{replicate}"
        or status not in {"ok", "error", "pending"}
    ):
        raise InvalidComparisonInputError(
            manifest_path, f"execution_order entry {index} is invalid"
        )
    return (
        identity,
        replicate,
        name,
        _optional_worker_pid(entry["worker_pid"], manifest_path, "execution-order worker_pid"),
        cast("Literal['ok', 'error', 'pending']", status),
    )


def _load_execution_order(
    manifest_path: Path, manifest: RunManifest, records: tuple[_RawSummaryRecord, ...]
) -> None:
    """Reconcile ordered tasks with finalized raw records, PID/status, and the case manifest."""
    document = _strict_json_document(
        _read_bound_artifact(manifest_path, manifest, "execution_order.json"),
        manifest_path,
        "execution_order.json",
    )
    if not isinstance(document, list):
        raise InvalidComparisonInputError(manifest_path, "execution_order.json must be a list")
    values = cast("list[object]", document)
    entries = tuple(
        _execution_entry(value, manifest_path, index) for index, value in enumerate(values)
    )
    task_keys = {(identity, replicate) for identity, replicate, _, _, _ in entries}
    raw_by_task = {(record.case_identity, record.replicate_index): record for record in records}
    manifest_cases = {case["case_identity"]: case["case_name"] for case in manifest.case_identities}
    if len(entries) != manifest.expected_task_count or len(task_keys) != len(entries):
        raise InvalidComparisonInputError(
            manifest_path, "execution_order.json task count is invalid"
        )
    if {identity for identity, _, _, _, _ in entries} != set(manifest_cases):
        raise InvalidComparisonInputError(
            manifest_path, "execution_order.json case set differs from manifest"
        )
    if set(raw_by_task) - task_keys:
        raise InvalidComparisonInputError(
            manifest_path, "raw replicate task is absent from execution_order.json"
        )
    for identity, replicate, name, worker_pid, status in entries:
        if name != manifest_cases[identity]:
            raise InvalidComparisonInputError(
                manifest_path, "execution_order.json case name differs from manifest"
            )
        record = raw_by_task.get((identity, replicate))
        if record is None:
            if status != "pending" or worker_pid is not None:
                raise InvalidComparisonInputError(
                    manifest_path, "execution_order.json missing raw task is not pending"
                )
            continue
        if (record.case_name, record.worker_pid, record.status) != (name, worker_pid, status):
            raise InvalidComparisonInputError(
                manifest_path, "execution_order.json disagrees with raw replicate evidence"
            )


def _validate_run_environment(environment: dict[str, object], manifest_path: Path) -> None:
    """Validate every run-level methodology field before compatibility comparison."""
    for field in ("platform", "machine", "python_version", "torch_version", "numpy_version"):
        _ = _nonempty_string(environment[field], manifest_path, f"run_environment {field}")
    cpu_name = environment["cpu_name"]
    if cpu_name is not None:
        _ = _nonempty_string(cpu_name, manifest_path, "run_environment cpu_name")
    for field in ("physical_cpu_count", "logical_cpu_count"):
        value = environment[field]
        if value is not None:
            _ = _integer_value(
                value,
                manifest_path,
                f"run_environment {field}",
                positive=True,
                non_negative=False,
            )
    if not isinstance(environment["cuda_available"], bool):
        raise InvalidComparisonInputError(
            manifest_path, "run_environment cuda_available must be boolean"
        )
    for field in (
        "parent_torch_num_threads",
        "parent_torch_num_interop_threads",
        "configured_torch_num_interop_threads",
    ):
        _ = _integer_value(
            environment[field],
            manifest_path,
            f"run_environment {field}",
            positive=True,
            non_negative=False,
        )
    _ = _worker_affinity(environment, manifest_path, "configured_cpu_affinity")
    _ = _worker_variables(environment, manifest_path)
    _ = _nonempty_string(environment["process_priority"], manifest_path, "run_environment priority")
    power_scheme = _require_mapping(environment["power_scheme"], manifest_path, "power scheme")
    if frozenset(power_scheme) != frozenset({"value", "reason"}):
        raise InvalidComparisonInputError(manifest_path, "power scheme has an invalid field set")
    for field in ("value", "reason"):
        if power_scheme[field] is not None and not isinstance(power_scheme[field], str):
            raise InvalidComparisonInputError(
                manifest_path, f"power scheme {field} must be text or null"
            )


def _load_run_environment(manifest_path: Path, manifest: RunManifest) -> dict[str, JsonValue]:
    """Load and validate the complete parent environment used for compatibility checks."""
    raw_document = _strict_json_document(
        _read_bound_artifact(manifest_path, manifest, "environment.json"),
        manifest_path,
        "environment.json",
    )
    document = _require_mapping(raw_document, manifest_path, "environment.json")
    if frozenset(document) != _ENVIRONMENT_DOCUMENT_KEYS:
        raise InvalidComparisonInputError(
            manifest_path, "environment.json has an invalid field set"
        )
    if document["run_id"] != manifest.run_id or document["run_status"] != manifest.status:
        raise InvalidComparisonInputError(
            manifest_path, "environment.json does not match manifest identity"
        )
    if document["config_sha256"] != manifest.config_sha256:
        raise InvalidComparisonInputError(
            manifest_path, "environment.json does not match manifest config"
        )
    if document["schema_version"] != _ENVIRONMENT_SCHEMA_VERSION or not isinstance(
        document["worker_environments"], list
    ):
        raise InvalidComparisonInputError(manifest_path, "environment.json has an invalid schema")
    environment = _require_mapping(document["run_environment"], manifest_path, "run_environment")
    if frozenset(environment) != _RUN_ENVIRONMENT_KEYS:
        raise InvalidComparisonInputError(manifest_path, "run_environment has an invalid field set")
    if environment["captured_before_first_worker"] is not True:
        raise InvalidComparisonInputError(
            manifest_path, "run_environment was not captured before workers"
        )
    if environment["git"] != manifest.git:
        raise InvalidComparisonInputError(
            manifest_path, "run_environment Git identity differs from manifest"
        )
    _validate_run_environment(environment, manifest_path)
    return cast("dict[str, JsonValue]", environment)


def _load_summaries(  # noqa: C901
    manifest_path: Path, manifest: RunManifest
) -> tuple[BenchmarkV2Summary, ...]:
    """Load exact summary rows before reconciling their identities against durable raw evidence."""
    try:
        text = _read_bound_artifact(manifest_path, manifest, "summary.csv").decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidComparisonInputError(
            manifest_path, "summary.csv is not valid UTF-8"
        ) from error
    reader = csv.DictReader(StringIO(text))
    if tuple(reader.fieldnames or ()) != _SUMMARY_FIELDS:
        raise InvalidComparisonInputError(manifest_path, "summary.csv has an invalid header")
    summaries: list[BenchmarkV2Summary] = []
    for index, row in enumerate(reader, start=2):
        if None in row or frozenset(row) != frozenset(_SUMMARY_FIELDS):
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} has an invalid field set"
            )
        values = cast("dict[str, str]", row)
        identity = _require_sha256(
            values["case_identity"], manifest_path, f"summary.csv row {index} identity"
        )
        name = values["case_name"]
        if not name:
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} has an empty case name"
            )
        replicate_count = _require_integer(
            values["replicate_count"], manifest_path, f"summary.csv row {index} replicate_count"
        )
        success_count = _require_integer(
            values["success_count"], manifest_path, f"summary.csv row {index} success_count"
        )
        failure_count = _require_integer(
            values["failure_count"], manifest_path, f"summary.csv row {index} failure_count"
        )
        if replicate_count != success_count + failure_count:
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} has inconsistent counts"
            )
        stability = values["stability"]
        if stability not in _STABILITY_VALUES:
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} has invalid stability"
            )
        numeric_fields = (
            "median_step_time_ms",
            "min_step_time_ms",
            "max_step_time_ms",
            "population_stddev_step_time_ms",
            "median_absolute_deviation_step_time_ms",
            "coefficient_of_variation_percent",
            "median_tokens_per_second",
            "median_final_rss_mib",
            "max_peak_rss_mib",
        )
        numbers = {
            field: _require_number(
                values[field], manifest_path, f"summary.csv row {index} {field}", nullable=True
            )
            for field in numeric_fields
        }
        if success_count == 0 and any(value is not None for value in numbers.values()):
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} has metrics without success"
            )
        if success_count > 0 and any(value is None for value in numbers.values()):
            raise InvalidComparisonInputError(
                manifest_path, f"summary.csv row {index} lacks successful metrics"
            )
        summaries.append(
            BenchmarkV2Summary(
                case_identity=identity,
                case_name=name,
                replicate_count=replicate_count,
                success_count=success_count,
                failure_count=failure_count,
                median_step_time_ms=numbers["median_step_time_ms"],
                min_step_time_ms=numbers["min_step_time_ms"],
                max_step_time_ms=numbers["max_step_time_ms"],
                population_stddev_step_time_ms=numbers["population_stddev_step_time_ms"],
                median_absolute_deviation_step_time_ms=numbers[
                    "median_absolute_deviation_step_time_ms"
                ],
                coefficient_of_variation_percent=numbers["coefficient_of_variation_percent"],
                median_tokens_per_second=numbers["median_tokens_per_second"],
                median_final_rss_mib=numbers["median_final_rss_mib"],
                max_peak_rss_mib=numbers["max_peak_rss_mib"],
                stability=cast("Literal['insufficient_samples', 'unstable', 'stable']", stability),
            )
        )
    observed = {summary.case_identity for summary in summaries}
    if len(summaries) != len(observed):
        raise InvalidComparisonInputError(
            manifest_path, "summary.csv case identities are not unique"
        )
    return tuple(sorted(summaries, key=lambda summary: summary.case_identity))


def _load_methodology(manifest_path: Path, manifest: RunManifest) -> _Methodology:
    """Read the candidate's bound strict regression threshold without trusting a CLI default."""
    try:
        document = yaml.safe_load(
            _read_bound_artifact(manifest_path, manifest, "resolved_config.yaml").decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise InvalidComparisonInputError(
            manifest_path, "resolved_config.yaml is invalid"
        ) from error
    config = _require_mapping(document, manifest_path, "resolved_config.yaml")

    def positive_number(field: str) -> float:
        value = config.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise InvalidComparisonInputError(
                manifest_path, f"config {field} is not positive and finite"
            )
        return float(value)

    minimum = config.get("minimum_replicates")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise InvalidComparisonInputError(manifest_path, "config minimum_replicates is invalid")
    return _Methodology(
        positive_number("regression_threshold_percent"), minimum, positive_number("max_cv_percent")
    )


def _validate_summaries_from_raw(
    manifest_path: Path,
    manifest: RunManifest,
    summaries: tuple[BenchmarkV2Summary, ...],
    records: tuple[_RawSummaryRecord, ...],
    methodology: _Methodology,
) -> None:
    """Recompute observed summaries and retain exact complete-run coverage requirements."""
    summary_identities = {summary.case_identity for summary in summaries}
    raw_identities = {record.case_identity for record in records}
    if raw_identities != summary_identities:
        raise InvalidComparisonInputError(
            manifest_path, "raw replicate case identities do not match summary.csv"
        )
    manifest_identities = {case["case_identity"] for case in manifest.case_identities}
    if manifest.status == "complete" and summary_identities != manifest_identities:
        raise InvalidComparisonInputError(
            manifest_path, "complete summary.csv case identities do not match manifest"
        )
    for summary in summaries:
        case_records = tuple(
            record for record in records if record.case_identity == summary.case_identity
        )
        if not case_records:
            raise InvalidComparisonInputError(manifest_path, "summary case has no raw replicates")
        recomputed = summarize_replicates(
            case_records,
            minimum_replicates=methodology.minimum_replicates,
            max_cv_percent=methodology.max_cv_percent,
        )
        if recomputed != summary:
            raise InvalidComparisonInputError(
                manifest_path, "summary disagrees with raw replicate statistics"
            )


def _case_worker_control_sets(
    records: tuple[_RawSummaryRecord, ...],
) -> tuple[tuple[str, frozenset[_WorkerControls]], ...]:
    """Normalize successful actual controls by case, excluding replicate-local process evidence."""
    controls_by_case: dict[str, set[_WorkerControls]] = {}
    for record in records:
        if record.worker_controls is not None:
            controls_by_case.setdefault(record.case_identity, set()).add(record.worker_controls)
    return tuple(
        (identity, frozenset(controls)) for identity, controls in sorted(controls_by_case.items())
    )


def _load_input(manifest_path: Path) -> _ComparisonInput:
    """Compose strict loader boundaries before any comparison calculation occurs."""
    try:
        manifest_content = manifest_path.read_bytes()
    except OSError as error:
        raise InvalidComparisonInputError(manifest_path, "cannot read run_manifest.json") from error
    _ = _strict_json_document(manifest_content, manifest_path, "run_manifest.json")
    manifest = load_run_manifest(manifest_path)
    methodology = _load_methodology(manifest_path, manifest)
    summaries = _load_summaries(manifest_path, manifest)
    records = _load_raw_records(manifest_path, manifest)
    _validate_summaries_from_raw(manifest_path, manifest, summaries, records, methodology)
    _load_execution_order(manifest_path, manifest, records)
    environment = _load_run_environment(manifest_path, manifest)
    raw_environment = _strict_json_document(
        _read_bound_artifact(manifest_path, manifest, "environment.json"),
        manifest_path,
        "environment.json",
    )
    worker_documents = _require_mapping(raw_environment, manifest_path, "environment.json")[
        "worker_environments"
    ]
    if not isinstance(worker_documents, list):
        raise InvalidComparisonInputError(manifest_path, "worker_environments must be a list")
    worker_values = cast("list[object]", worker_documents)
    reported_signatures = tuple(
        _worker_environment_signature(item, manifest_path) for item in worker_values
    )
    case_worker_controls = _case_worker_control_sets(records)
    raw_environment_signatures = tuple(
        record.worker_controls.environment_signature
        for record in records
        if record.worker_controls is not None
    )
    if Counter(raw_environment_signatures) != Counter(reported_signatures):
        raise InvalidComparisonInputError(
            manifest_path, "worker environments disagree with successful raw records"
        )
    return _ComparisonInput(
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        run_environment=environment,
        worker_environment_signatures=reported_signatures,
        case_worker_controls=case_worker_controls,
        summaries=summaries,
        regression_threshold_percent=methodology.regression_threshold_percent,
    )


def compare_step_times(
    baseline_step_time_ms: float, candidate_step_time_ms: float, *, threshold_percent: float
) -> StepTimeComparison:
    """Calculate one descriptive change and enforce a strictly-greater regression boundary."""
    values = (baseline_step_time_ms, candidate_step_time_ms, threshold_percent)
    if any(not math.isfinite(value) for value in values) or baseline_step_time_ms <= 0.0:
        msg = "step times and threshold must be finite, with a positive baseline step time"
        raise ValueError(msg)
    if threshold_percent <= 0.0:
        msg = "threshold_percent must be positive"
        raise ValueError(msg)
    change = (candidate_step_time_ms / baseline_step_time_ms - 1.0) * 100.0
    exact_change = (
        Decimal(str(candidate_step_time_ms)) / Decimal(str(baseline_step_time_ms)) - 1
    ) * 100
    exact_threshold = Decimal(str(threshold_percent))
    return StepTimeComparison(
        step_time_change_percent=change,
        regressed=exact_change > exact_threshold,
    )


def _descriptive_change(baseline: float | None, candidate: float | None) -> float | None:
    """Calculate a finite relative delta only when both medians are meaningful positive values."""
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def _environment_mismatches(
    baseline: _ComparisonInput, candidate: _ComparisonInput
) -> tuple[EnvironmentMismatch, ...]:
    """Compare documented performance fields while deliberately excluding Git provenance."""
    return tuple(
        EnvironmentMismatch(
            field=field,
            baseline=baseline.run_environment[field],
            candidate=candidate.run_environment[field],
        )
        for field in _ENVIRONMENT_FIELDS
        if baseline.run_environment[field] != candidate.run_environment[field]
    )


def _unavailable_cpu_compatibility_fields(environment: dict[str, JsonValue]) -> tuple[str, ...]:
    """Return required CPU fields that cannot establish hardware compatibility evidence."""
    unavailable: list[str] = []
    for field in _REQUIRED_CPU_COMPATIBILITY_FIELDS:
        value = environment[field]
        if field == "cpu_name":
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        if not valid:
            unavailable.append(field)
    return tuple(unavailable)


def _cpu_compatibility_reasons(
    baseline: _ComparisonInput, candidate: _ComparisonInput
) -> list[str]:
    """Describe missing CPU identity or topology evidence for each comparison input."""
    return [
        f"{label} environment lacks required CPU compatibility evidence: {field}"
        for label, comparison_input in (("baseline", baseline), ("candidate", candidate))
        for field in _unavailable_cpu_compatibility_fields(comparison_input.run_environment)
    ]


def _case_reasons(
    baseline: BenchmarkV2Summary,
    candidate: BenchmarkV2Summary,
    *,
    globally_comparable: bool,
) -> tuple[str, ...]:
    """List case-local blockers in a stable, reader-oriented order."""
    reasons: list[str] = []
    if not globally_comparable:
        reasons.append("run-level compatibility requirements are not met")
    if baseline.stability == "insufficient_samples":
        reasons.append("baseline case has insufficient successful replicates")
    if candidate.stability == "insufficient_samples":
        reasons.append("candidate case has insufficient successful replicates")
    if baseline.stability == "unstable":
        reasons.append("baseline case is unstable")
    if candidate.stability == "unstable":
        reasons.append("candidate case is unstable")
    if baseline.median_step_time_ms is None or candidate.median_step_time_ms is None:
        reasons.append("case has no comparable median step time")
    return tuple(reasons)


def _inconsistent_worker_control_reason(
    run_label: str, controls_by_case: dict[str, frozenset[_WorkerControls]]
) -> str | None:
    """Reject a complete case whose successful workers applied more than one control signature."""
    if any(len(controls) != 1 for controls in controls_by_case.values()):
        return f"{run_label} case has inconsistent applied worker controls"
    return None


def _run_reasons(
    baseline: _ComparisonInput,
    candidate: _ComparisonInput,
    missing: tuple[str, ...],
    extra: tuple[str, ...],
    environment_mismatches: tuple[EnvironmentMismatch, ...],
) -> list[str]:
    """Collect run-level eligibility blockers before evaluating any case-level result."""
    reasons: list[str] = []
    if baseline.manifest.status != "complete":
        reasons.append(f"baseline run status is {baseline.manifest.status}")
    if candidate.manifest.status != "complete":
        reasons.append(f"candidate run status is {candidate.manifest.status}")
    if missing or extra:
        reasons.append("case identity sets do not align")
    reasons.extend(_cpu_compatibility_reasons(baseline, candidate))
    reasons.extend(f"environment differs: {mismatch.field}" for mismatch in environment_mismatches)
    if baseline.manifest.status == "complete" and candidate.manifest.status == "complete":
        baseline_controls = dict(baseline.case_worker_controls)
        candidate_controls = dict(candidate.case_worker_controls)
        for label, controls in (
            ("baseline", baseline_controls),
            ("candidate", candidate_controls),
        ):
            inconsistency = _inconsistent_worker_control_reason(label, controls)
            if inconsistency is not None:
                reasons.append(inconsistency)
        if not any("inconsistent applied worker controls" in reason for reason in reasons):
            aligned = baseline_controls.keys() & candidate_controls.keys()
            if any(
                baseline_controls[identity] != candidate_controls[identity] for identity in aligned
            ):
                reasons.append("applied worker controls differ")
    return reasons


def _compare_case(
    identity: str,
    baseline: BenchmarkV2Summary,
    candidate: BenchmarkV2Summary,
    *,
    globally_comparable: bool,
    threshold_percent: float,
) -> CaseComparison:
    """Calculate descriptive and guarded verdict data for one aligned case identity."""
    reasons = _case_reasons(baseline, candidate, globally_comparable=globally_comparable)
    step_change = _descriptive_change(baseline.median_step_time_ms, candidate.median_step_time_ms)
    throughput_change = _descriptive_change(
        baseline.median_tokens_per_second, candidate.median_tokens_per_second
    )
    regressed = (
        compare_step_times(
            cast("float", baseline.median_step_time_ms),
            cast("float", candidate.median_step_time_ms),
            threshold_percent=threshold_percent,
        ).regressed
        if not reasons and step_change is not None
        else None
    )
    return CaseComparison(
        case_identity=identity,
        baseline_case_name=baseline.case_name,
        candidate_case_name=candidate.case_name,
        baseline_median_step_time_ms=baseline.median_step_time_ms,
        candidate_median_step_time_ms=candidate.median_step_time_ms,
        baseline_median_tokens_per_second=baseline.median_tokens_per_second,
        candidate_median_tokens_per_second=candidate.median_tokens_per_second,
        step_time_change_percent=step_change,
        throughput_change_percent=throughput_change,
        reasons=reasons,
        regressed=regressed,
    )


def _case_comparisons(
    baseline: _ComparisonInput, candidate: _ComparisonInput, *, globally_comparable: bool
) -> tuple[CaseComparison, ...]:
    """Align summaries by durable case identity and compare their common entries in order."""
    baseline_by_identity = {summary.case_identity: summary for summary in baseline.summaries}
    candidate_by_identity = {summary.case_identity: summary for summary in candidate.summaries}
    return tuple(
        _compare_case(
            identity,
            baseline_by_identity[identity],
            candidate_by_identity[identity],
            globally_comparable=globally_comparable,
            threshold_percent=candidate.regression_threshold_percent,
        )
        for identity in sorted(baseline_by_identity.keys() & candidate_by_identity.keys())
    )


def _append_case_blockers(reasons: list[str], cases: tuple[CaseComparison, ...]) -> None:
    """Promote local blockers into the run reason list once, preserving explanatory order."""
    for case in cases:
        for reason in case.reasons:
            if (
                reason != "run-level compatibility requirements are not met"
                and reason not in reasons
            ):
                reasons.append(reason)


def _verdict(reasons: list[str], cases: tuple[CaseComparison, ...]) -> ComparisonVerdict:
    """Return a pass/fail only when every aligned case has guarded regression eligibility."""
    if reasons or any(case.regressed is None for case in cases):
        return "not_comparable"
    if any(case.regressed for case in cases):
        return "fail"
    return "pass"


def compare_runs(baseline: Path, candidate: Path) -> BenchmarkComparison:
    """Compare two finalized runs with conservative eligibility and immutable source evidence."""
    baseline_input = _load_input(baseline)
    candidate_input = _load_input(candidate)
    environment_mismatches = _environment_mismatches(baseline_input, candidate_input)
    baseline_identities = {summary.case_identity for summary in baseline_input.summaries}
    candidate_identities = {summary.case_identity for summary in candidate_input.summaries}
    missing = tuple(sorted(baseline_identities - candidate_identities))
    extra = tuple(sorted(candidate_identities - baseline_identities))
    reasons = _run_reasons(baseline_input, candidate_input, missing, extra, environment_mismatches)
    cases = _case_comparisons(baseline_input, candidate_input, globally_comparable=not reasons)
    _append_case_blockers(reasons, cases)
    return BenchmarkComparison(
        baseline_manifest_path=baseline_input.manifest_path,
        candidate_manifest_path=candidate_input.manifest_path,
        baseline_run_id=baseline_input.manifest.run_id,
        candidate_run_id=candidate_input.manifest.run_id,
        regression_threshold_percent=candidate_input.regression_threshold_percent,
        environment_mismatches=environment_mismatches,
        missing_case_identities=missing,
        extra_case_identities=extra,
        reasons=tuple(reasons),
        case_comparisons=cases,
        verdict=_verdict(reasons, cases),
    )


def _comparison_document(comparison: BenchmarkComparison) -> dict[str, JsonValue]:
    """Render a deterministic JSON-safe comparison record without machine-local source mutations."""
    return {
        "baseline_manifest": comparison.baseline_manifest_path.as_posix(),
        "candidate_manifest": comparison.candidate_manifest_path.as_posix(),
        "baseline_run_id": comparison.baseline_run_id,
        "candidate_run_id": comparison.candidate_run_id,
        "regression_threshold_percent": comparison.regression_threshold_percent,
        "environment_mismatches": [
            asdict(mismatch) for mismatch in comparison.environment_mismatches
        ],
        "missing_case_identities": list(comparison.missing_case_identities),
        "extra_case_identities": list(comparison.extra_case_identities),
        "reasons": list(comparison.reasons),
        "case_comparisons": [
            {
                **asdict(case),
                "reasons": list(case.reasons),
            }
            for case in comparison.case_comparisons
        ],
        "verdict": comparison.verdict,
    }


def _markdown_cell(value: object) -> str:
    """Render one human-readable Markdown cell without allowing table delimiters to escape."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _environment_mismatch_row(mismatch: EnvironmentMismatch) -> str:
    """Render one environment mismatch as a compact escaped Markdown table row."""
    field = _markdown_cell(mismatch.field)
    baseline = _markdown_cell(mismatch.baseline)
    candidate = _markdown_cell(mismatch.candidate)
    return f"| {field} | {baseline} | {candidate} |"


def _comparison_markdown(comparison: BenchmarkComparison) -> str:
    """Explain verdict safeguards and retain all descriptive deltas in a deterministic report."""
    mismatch_rows = (
        "\n".join(
            _environment_mismatch_row(mismatch) for mismatch in comparison.environment_mismatches
        )
        or "| None | — | — |"
    )
    case_rows = (
        "\n".join(
            "| "
            + " | ".join(
                (
                    _markdown_cell(case.candidate_case_name),
                    case.case_identity,
                    _markdown_cell(case.baseline_median_step_time_ms),
                    _markdown_cell(case.candidate_median_step_time_ms),
                    _markdown_cell(case.step_time_change_percent),
                    _markdown_cell(case.throughput_change_percent),
                    _markdown_cell(case.regressed),
                    _markdown_cell("; ".join(case.reasons) or "—"),
                )
            )
            + " |"
            for case in comparison.case_comparisons
        )
        or "| None | — | — | — | — | — | — | no aligned cases |"
    )
    reasons = "\n".join(f"- {reason}" for reason in comparison.reasons) or "- None"
    return f"""# Benchmark v2 comparison

Verdict: **{comparison.verdict}**

Baseline: `{comparison.baseline_run_id}`
Candidate: `{comparison.candidate_run_id}`
Regression threshold: `{comparison.regression_threshold_percent}%` (strictly greater than)

## Verdict reasons

{reasons}

## Environment compatibility differences

| Field | Baseline | Candidate |
| --- | --- | --- |
{mismatch_rows}

## Aligned cases

| Case | Identity | Base ms | Candidate ms | Step % | Throughput % | Regressed | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{case_rows}

Descriptive deltas are retained even when this report is `not_comparable`. Git SHA, branch, dirty
state, and timestamps are provenance only; they do not independently block comparison. No outliers
are filtered by this comparison.
"""


def _atomic_write_new(path: Path, content: bytes) -> None:
    """Create one report atomically after rejecting a deterministic output-name collision."""
    if path.exists():
        msg = f"comparison artifact already exists: {path}"
        raise FileExistsError(msg)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_comparison(comparison: BenchmarkComparison) -> ComparisonArtifacts:
    """Write outputs beside, never inside, the immutable candidate run directory.

    Both paths live under the candidate run's parent, retain the baseline run ID suffix, and reject
    collisions rather than overwriting an earlier comparison artifact.
    """
    directory = comparison.candidate_manifest_path.parent.parent
    stem = f"{comparison.candidate_run_id}-comparison-{comparison.baseline_run_id}"
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    if json_path.exists() or markdown_path.exists():
        msg = f"comparison artifact already exists for {stem}"
        raise FileExistsError(msg)
    document = _comparison_document(comparison)
    _atomic_write_new(
        json_path,
        (
            json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )
    try:
        _atomic_write_new(markdown_path, _comparison_markdown(comparison).encode("utf-8"))
    except Exception:
        if json_path.exists():
            json_path.unlink()
        raise
    return ComparisonArtifacts(json_path=json_path, markdown_path=markdown_path)
