"""Orchestrate, summarize, and hash-bind isolated Stage 9 inference benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import psutil
import torch

from minigpt.inference_benchmark_config import (
    InferenceBenchmarkConfig,
    InferenceCase,
    JsonValue,
    expand_inference_cases,
    resolved_config_bytes,
    resolved_config_sha256,
)
from minigpt.inference_benchmark_worker import (
    InferenceWorkerRequest,
    worker_request_document,
)

if TYPE_CHECKING:
    from pathlib import Path

RunStatus: TypeAlias = Literal["complete", "partial", "failed"]
ComparisonVerdict: TypeAlias = Literal["pass", "not_comparable"]
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class InferenceReplicate:
    """Preserve one worker result or failure in execution order."""

    status: Literal["ok", "error"]
    case_name: str
    mode: str
    replicate_index: int
    worker_pid: int | None
    prefill_time_ms: float | None
    time_to_first_token_ms: float | None
    median_decode_time_ms: float | None
    generated_tokens_per_second: float | None
    end_to_end_time_ms: float | None
    peak_rss_mib: float | None
    kv_cache_bytes: int | None
    environment_signature: str | None
    worker_response: dict[str, JsonValue] | None
    error_type: str | None
    message: str | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class MetricStatistics:
    """Report raw-count median, MAD, and population CV for one metric."""

    count: int
    median: float | None
    median_absolute_deviation: float | None
    coefficient_of_variation_percent: float | None


@dataclass(frozen=True, slots=True)
class InferenceModeSummary:
    """Summarize all successful raw replicates for one case and mode."""

    case_name: str
    mode: str
    replicate_count: int
    success_count: int
    failure_count: int
    stability: Literal["stable", "unstable", "insufficient_samples"]
    environment_signatures: tuple[str, ...]
    prefill_time_ms: MetricStatistics
    time_to_first_token_ms: MetricStatistics
    median_decode_time_ms: MetricStatistics
    generated_tokens_per_second: MetricStatistics
    end_to_end_time_ms: MetricStatistics
    peak_rss_mib: MetricStatistics
    kv_cache_bytes: MetricStatistics


@dataclass(frozen=True, slots=True)
class InferenceComparison:
    """Guard descriptive cached/uncached deltas with strict eligibility reasons."""

    case_name: str
    verdict: ComparisonVerdict
    reasons: tuple[str, ...]
    end_to_end_change_percent: float | None
    decode_time_change_percent: float | None
    throughput_change_percent: float | None


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkArtifacts:
    """Expose the durable outputs from one inference benchmark invocation."""

    run_directory: Path
    run_manifest_path: Path
    summary_path: Path
    raw_replicates_path: Path
    status: RunStatus
    run_id: str


@dataclass(frozen=True, slots=True)
class _Task:
    """Identify one fresh worker launch."""

    case: InferenceCase
    mode: Literal["cached", "uncached"]
    replicate_index: int


def _metric(values: tuple[float, ...]) -> MetricStatistics:
    if not values:
        return MetricStatistics(0, None, None, None)
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean * 100.0 if mean != 0.0 else 0.0
    return MetricStatistics(len(values), median, mad, cv)


def _required_metric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"successful replicate has no numeric {field}"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        msg = f"successful replicate has invalid {field}"
        raise ValueError(msg)
    return number


def summarize_inference_replicates(
    raw_replicates: tuple[InferenceReplicate, ...],
    *,
    minimum_replicates: int,
    max_cv_percent: float,
) -> InferenceModeSummary:
    """Summarize one exact case/mode collection without filtering outliers."""
    if not raw_replicates:
        msg = "cannot summarize an empty replicate collection"
        raise ValueError(msg)
    identities = {(record.case_name, record.mode) for record in raw_replicates}
    if len(identities) != 1:
        msg = "replicates must contain exactly one case and mode"
        raise ValueError(msg)
    case_name, mode = identities.pop()
    successes = tuple(record for record in raw_replicates if record.status == "ok")
    metrics = {
        "prefill_time_ms": _metric(
            tuple(
                _required_metric(record.prefill_time_ms, "prefill_time_ms") for record in successes
            )
        ),
        "time_to_first_token_ms": _metric(
            tuple(
                _required_metric(record.time_to_first_token_ms, "time_to_first_token_ms")
                for record in successes
            )
        ),
        "median_decode_time_ms": _metric(
            tuple(
                _required_metric(record.median_decode_time_ms, "median_decode_time_ms")
                for record in successes
            )
        ),
        "generated_tokens_per_second": _metric(
            tuple(
                _required_metric(record.generated_tokens_per_second, "generated_tokens_per_second")
                for record in successes
            )
        ),
        "end_to_end_time_ms": _metric(
            tuple(
                _required_metric(record.end_to_end_time_ms, "end_to_end_time_ms")
                for record in successes
            )
        ),
        "peak_rss_mib": _metric(
            tuple(_required_metric(record.peak_rss_mib, "peak_rss_mib") for record in successes)
        ),
        "kv_cache_bytes": _metric(
            tuple(_required_metric(record.kv_cache_bytes, "kv_cache_bytes") for record in successes)
        ),
    }
    end_to_end_cv = metrics["end_to_end_time_ms"].coefficient_of_variation_percent
    decode_cv = metrics["median_decode_time_ms"].coefficient_of_variation_percent
    if len(successes) < minimum_replicates:
        stability: Literal["stable", "unstable", "insufficient_samples"] = "insufficient_samples"
    elif (
        end_to_end_cv is None
        or decode_cv is None
        or end_to_end_cv > max_cv_percent
        or decode_cv > max_cv_percent
    ):
        stability = "unstable"
    else:
        stability = "stable"
    signatures = tuple(
        sorted(
            {
                signature
                for record in successes
                if (signature := record.environment_signature) is not None
            }
        )
    )
    return InferenceModeSummary(
        case_name=case_name,
        mode=mode,
        replicate_count=len(raw_replicates),
        success_count=len(successes),
        failure_count=len(raw_replicates) - len(successes),
        stability=stability,
        environment_signatures=signatures,
        prefill_time_ms=metrics["prefill_time_ms"],
        time_to_first_token_ms=metrics["time_to_first_token_ms"],
        median_decode_time_ms=metrics["median_decode_time_ms"],
        generated_tokens_per_second=metrics["generated_tokens_per_second"],
        end_to_end_time_ms=metrics["end_to_end_time_ms"],
        peak_rss_mib=metrics["peak_rss_mib"],
        kv_cache_bytes=metrics["kv_cache_bytes"],
    )


def _change_percent(baseline: MetricStatistics, candidate: MetricStatistics) -> float | None:
    if baseline.median is None or candidate.median is None or baseline.median == 0.0:
        return None
    return (candidate.median / baseline.median - 1.0) * 100.0


def compare_inference_modes(
    uncached: InferenceModeSummary,
    cached: InferenceModeSummary,
) -> InferenceComparison:
    """Return pass only for complete, stable, equal, environment-compatible evidence."""
    if uncached.case_name != cached.case_name:
        msg = "mode summaries must describe the same case"
        raise ValueError(msg)
    reasons: list[str] = []
    if uncached.mode != "uncached" or cached.mode != "cached":
        reasons.append("summaries are not ordered as uncached then cached")
    if uncached.replicate_count != cached.replicate_count:
        reasons.append("replicate counts differ")
    if uncached.failure_count or cached.failure_count:
        reasons.append("one or more worker replicates failed")
    if uncached.stability != "stable" or cached.stability != "stable":
        reasons.append("end-to-end or decode CV exceeds the configured limit")
    if (
        len(uncached.environment_signatures) != 1
        or len(cached.environment_signatures) != 1
        or uncached.environment_signatures != cached.environment_signatures
    ):
        reasons.append("worker environment signatures are incompatible")
    return InferenceComparison(
        case_name=uncached.case_name,
        verdict="not_comparable" if reasons else "pass",
        reasons=tuple(reasons),
        end_to_end_change_percent=_change_percent(
            uncached.end_to_end_time_ms, cached.end_to_end_time_ms
        ),
        decode_time_change_percent=_change_percent(
            uncached.median_decode_time_ms, cached.median_decode_time_ms
        ),
        throughput_change_percent=_change_percent(
            uncached.generated_tokens_per_second, cached.generated_tokens_per_second
        ),
    )


def _tasks(config: InferenceBenchmarkConfig) -> tuple[_Task, ...]:
    tasks = [
        _Task(case, mode, replicate_index)
        for case in expand_inference_cases(config)
        for replicate_index in range(config.replicates)
        for mode in ("uncached", "cached")
    ]
    random.Random(config.benchmark_seed).shuffle(tasks)  # noqa: S311 - reproducible order.
    return tuple(tasks)


def _float(response: dict[str, object], field: str) -> float:
    value = response[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"worker response {field} must be numeric"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        msg = f"worker response {field} must be positive and finite"
        raise ValueError(msg)
    return number


def _success_record(
    task: _Task,
    completed: subprocess.CompletedProcess[str],
    response: dict[str, object],
) -> InferenceReplicate:
    if (
        response.get("status") != "ok"
        or response.get("case_name") != task.case.name
        or response.get("mode") != task.mode
        or response.get("replicate_index") != task.replicate_index
    ):
        msg = "worker response identity does not match task"
        raise ValueError(msg)
    worker_pid = response["worker_pid"]
    cache_bytes = response["kv_cache_bytes"]
    signature = response["environment_signature"]
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
        or isinstance(cache_bytes, bool)
        or not isinstance(cache_bytes, int)
        or cache_bytes < 0
        or not isinstance(signature, str)
        or len(signature) != _SHA256_LENGTH
    ):
        msg = "worker response has invalid identity or cache fields"
        raise ValueError(msg)
    return InferenceReplicate(
        status="ok",
        case_name=task.case.name,
        mode=task.mode,
        replicate_index=task.replicate_index,
        worker_pid=worker_pid,
        prefill_time_ms=_float(response, "prefill_time_ms"),
        time_to_first_token_ms=_float(response, "time_to_first_token_ms"),
        median_decode_time_ms=_float(response, "median_decode_time_ms"),
        generated_tokens_per_second=_float(response, "generated_tokens_per_second"),
        end_to_end_time_ms=_float(response, "end_to_end_time_ms"),
        peak_rss_mib=_float(response, "peak_rss_mib"),
        kv_cache_bytes=cache_bytes,
        environment_signature=signature,
        worker_response=cast("dict[str, JsonValue]", response),
        error_type=None,
        message=None,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _error_record(
    task: _Task,
    *,
    error_type: str,
    message: str,
    stdout: str,
    stderr: str,
) -> InferenceReplicate:
    """Build one complete orchestrator-owned failure record."""
    return InferenceReplicate(
        status="error",
        case_name=task.case.name,
        mode=task.mode,
        replicate_index=task.replicate_index,
        worker_pid=None,
        prefill_time_ms=None,
        time_to_first_token_ms=None,
        median_decode_time_ms=None,
        generated_tokens_per_second=None,
        end_to_end_time_ms=None,
        peak_rss_mib=None,
        kv_cache_bytes=None,
        environment_signature=None,
        worker_response=None,
        error_type=error_type,
        message=message,
        stdout=stdout,
        stderr=stderr,
    )


def _execute_task(config: InferenceBenchmarkConfig, task: _Task) -> InferenceReplicate:
    request = InferenceWorkerRequest.from_config(
        config,
        task.case,
        mode=task.mode,
        replicate_index=task.replicate_index,
    )
    request_json = json.dumps(
        worker_request_document(request),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "minigpt.inference_benchmark_worker"],
            input=request_json,
            capture_output=True,
            check=False,
            text=True,
            timeout=config.worker_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return _error_record(
            task,
            error_type="WorkerTimeout",
            message=f"worker exceeded {config.worker_timeout_seconds} seconds",
            stdout="" if error.stdout is None else str(error.stdout),
            stderr="" if error.stderr is None else str(error.stderr),
        )
    try:
        raw_response = cast("object", json.loads(completed.stdout))
    except json.JSONDecodeError as error:
        return _error_record(
            task,
            error_type=type(error).__name__,
            message=str(error),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    if not isinstance(raw_response, dict):
        return _error_record(
            task,
            error_type="InvalidWorkerResponse",
            message="worker response must be an object",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    response = cast("dict[str, object]", raw_response)
    if completed.returncode != 0:
        raw_error_type = response.get("error_type")
        raw_message = response.get("message")
        return _error_record(
            task,
            error_type=(raw_error_type if isinstance(raw_error_type, str) else "WorkerError"),
            message=(raw_message if isinstance(raw_message, str) else completed.stderr),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        return _success_record(task, completed, response)
    except (KeyError, TypeError, ValueError) as error:
        return _error_record(
            task,
            error_type=type(error).__name__,
            message=str(error),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _json_bytes(document: object) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    _ = path.write_bytes(content)


def _git_commit() -> str | None:
    executable = shutil.which("git")
    if executable is None:
        return None
    completed = subprocess.run(  # noqa: S603 - executable resolved by shutil.which.
        [executable, "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _parent_environment(config: InferenceBenchmarkConfig) -> dict[str, JsonValue]:
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "git_commit_sha": _git_commit(),
        "configured_cpu_affinity": (
            list(config.cpu_affinity) if config.cpu_affinity is not None else None
        ),
        "relevant_environment_variables": {
            name: os.environ.get(name) for name in config.relevant_environment_variables
        },
    }


def _summary_document(
    config: InferenceBenchmarkConfig,
    records: tuple[InferenceReplicate, ...],
    status: RunStatus,
) -> dict[str, JsonValue]:
    case_documents: list[JsonValue] = []
    comparisons: list[InferenceComparison] = []
    for case in expand_inference_cases(config):
        uncached = summarize_inference_replicates(
            tuple(
                record
                for record in records
                if record.case_name == case.name and record.mode == "uncached"
            ),
            minimum_replicates=config.minimum_replicates,
            max_cv_percent=config.max_cv_percent,
        )
        cached = summarize_inference_replicates(
            tuple(
                record
                for record in records
                if record.case_name == case.name and record.mode == "cached"
            ),
            minimum_replicates=config.minimum_replicates,
            max_cv_percent=config.max_cv_percent,
        )
        comparison = compare_inference_modes(uncached, cached)
        comparisons.append(comparison)
        case_documents.append(
            {
                "case": cast("dict[str, JsonValue]", asdict(case)),
                "uncached": cast("dict[str, JsonValue]", asdict(uncached)),
                "cached": cast("dict[str, JsonValue]", asdict(cached)),
                "comparison": cast("dict[str, JsonValue]", asdict(comparison)),
            }
        )
    strict_verdict: ComparisonVerdict = (
        "pass"
        if status == "complete" and all(item.verdict == "pass" for item in comparisons)
        else "not_comparable"
    )
    return {
        "schema_version": 1,
        "stage": "stage9-kv-cache-generation",
        "status": status,
        "strict_verdict": strict_verdict,
        "config_sha256": resolved_config_sha256(config),
        "replicate_count": len(records),
        "success_count": sum(record.status == "ok" for record in records),
        "failure_count": sum(record.status == "error" for record in records),
        "cases": case_documents,
    }


def _summary_markdown(summary: dict[str, JsonValue]) -> str:
    lines = [
        "# Stage 9 Inference Benchmark Run",
        "",
        f"Status: `{summary['status']}`. Strict verdict: `{summary['strict_verdict']}`.",
        "",
        "| Case | Uncached E2E ms | Cached E2E ms | Change | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    raw_cases = cast("list[object]", summary["cases"])
    for raw_case in raw_cases:
        case_document = cast("dict[str, object]", raw_case)
        case = cast("dict[str, object]", case_document["case"])
        uncached = cast("dict[str, object]", case_document["uncached"])
        cached = cast("dict[str, object]", case_document["cached"])
        comparison = cast("dict[str, object]", case_document["comparison"])
        uncached_e2e = cast("dict[str, object]", uncached["end_to_end_time_ms"])["median"]
        cached_e2e = cast("dict[str, object]", cached["end_to_end_time_ms"])["median"]
        change = comparison["end_to_end_change_percent"]
        row = f"| {case['name']} | {float(cast('float', uncached_e2e)):.6f} | "
        row += f"{float(cast('float', cached_e2e)):.6f} | "
        row += f"{float(cast('float', change)):+.2f}% | {comparison['verdict']} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def _run_status(records: tuple[InferenceReplicate, ...], expected: int) -> RunStatus:
    successes = sum(record.status == "ok" for record in records)
    if successes == expected:
        return "complete"
    return "failed" if successes == 0 else "partial"


def run_inference_benchmark(config: InferenceBenchmarkConfig) -> InferenceBenchmarkArtifacts:
    """Run every randomized task sequentially in a fresh subprocess and write evidence."""
    tasks = _tasks(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = _git_commit()
    run_id = f"{timestamp}-{(commit or 'nogit')[:12]}-{resolved_config_sha256(config)[:12]}"
    run_directory = config.output_root / run_id
    run_directory.mkdir(exist_ok=False)
    execution_order = [
        {
            "execution_index": index,
            "case_name": task.case.name,
            "mode": task.mode,
            "replicate_index": task.replicate_index,
        }
        for index, task in enumerate(tasks)
    ]
    records = tuple(_execute_task(config, task) for task in tasks)
    status = _run_status(records, len(tasks))
    raw_content = b"".join(
        json.dumps(
            asdict(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )
    worker_environments = {
        record.environment_signature: cast(
            "dict[str, JsonValue]", record.worker_response["environment"]
        )
        for record in records
        if record.status == "ok"
        and record.environment_signature is not None
        and record.worker_response is not None
    }
    summary = _summary_document(config, records, status)
    artifact_contents = {
        "resolved_config.yaml": resolved_config_bytes(config),
        "environment.json": _json_bytes(
            {
                "parent": _parent_environment(config),
                "worker_environments": worker_environments,
            }
        ),
        "execution_order.json": _json_bytes(execution_order),
        "raw_replicates.jsonl": raw_content,
        "summary.json": _json_bytes(summary),
        "summary.md": _summary_markdown(summary).encode("utf-8"),
    }
    for name, content in artifact_contents.items():
        _write(run_directory / name, content)
    artifacts = [
        {
            "path": name,
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in artifact_contents.items()
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "git_commit_sha": commit,
        "config_sha256": resolved_config_sha256(config),
        "artifacts": artifacts,
    }
    manifest_path = run_directory / "run_manifest.json"
    _write(manifest_path, _json_bytes(manifest))
    return InferenceBenchmarkArtifacts(
        run_directory=run_directory,
        run_manifest_path=manifest_path,
        summary_path=run_directory / "summary.json",
        raw_replicates_path=run_directory / "raw_replicates.jsonl",
        status=status,
        run_id=run_id,
    )


def verify_inference_run_manifest(path: Path) -> dict[str, JsonValue]:
    """Verify every bound artifact's exact byte size and SHA-256 digest."""
    raw_manifest = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw_manifest, dict):
        msg = "run manifest must be an object"
        raise TypeError(msg)
    manifest = cast("dict[str, JsonValue]", raw_manifest)
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        msg = "run manifest artifacts must be a list"
        raise TypeError(msg)
    for raw_entry in raw_artifacts:
        if not isinstance(raw_entry, dict):
            msg = "run manifest artifact must be an object"
            raise TypeError(msg)
        entry = cast("dict[str, object]", raw_entry)
        relative_path = entry.get("path")
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not isinstance(sha256, str)
        ):
            msg = "run manifest artifact fields are invalid"
            raise ValueError(msg)
        content = (path.parent / relative_path).read_bytes()
        if len(content) != size_bytes or hashlib.sha256(content).hexdigest() != sha256:
            msg = f"artifact {relative_path} failed size or SHA-256 verification"
            raise ValueError(msg)
    return manifest
