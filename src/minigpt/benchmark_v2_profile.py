"""Create separate CPU operator profiles bound to one Benchmark v2 case."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import torch
from torch.profiler import ProfilerAction, ProfilerActivity, profile
from typing_extensions import override

from minigpt.benchmark_v2_config import JsonValue, case_identity, resolved_config_sha256
from minigpt.benchmark_v2_environment import apply_cpu_affinity, capture_worker_environment
from minigpt.benchmark_v2_report import capture_run_environment
from minigpt.benchmark_v2_types import BenchmarkV2Case, BenchmarkV2Config
from minigpt.benchmark_workload import create_training_step_workload

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_PROFILE_PROTOCOL_VERSION = 1
_PROFILE_ARTIFACT_NAMES = ("top_operators.csv", "profile_report.md", "trace.json")
_REQUEST_KEYS = frozenset(
    {
        "protocol_version",
        "case",
        "benchmark_seed",
        "vocab_size",
        "warmup_steps",
        "active_steps",
        "torch_num_interop_threads",
        "cpu_affinity",
        "relevant_environment_variables",
        "output_directory",
    }
)
_CASE_KEYS = frozenset(
    {
        "name",
        "model_name",
        "n_layer",
        "n_head",
        "n_embd",
        "torch_num_threads",
        "block_size",
        "batch_size",
    }
)
JsonObject: TypeAlias = dict[str, JsonValue]


class _ProfilerEvent(Protocol):
    """Describe the profiler fields used to render stable, compact evidence."""

    key: str
    count: int
    self_cpu_time_total: float
    cpu_time_total: float
    self_cpu_memory_usage: int
    input_shapes: str


@dataclass(frozen=True, slots=True)
class ProfileConfigurationError(ValueError):
    """Report a v2 configuration that cannot select a profile workload."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the profile configuration reason."""
        return self.reason


@dataclass(frozen=True, slots=True)
class ProfileRunDirectoryCollisionError(FileExistsError):
    """Report an existing profile directory that must not be overwritten."""

    path: Path

    @override
    def __str__(self) -> str:
        """Render the protected existing evidence path."""
        return f"profile run directory already exists: {self.path}"


@dataclass(frozen=True, slots=True)
class InvalidProfileRunIdError(ValueError):
    """Report an optional profile identity that could escape the profile root."""

    run_id: str

    @override
    def __str__(self) -> str:
        """Explain the single-component containment requirement."""
        return "profile run ID must be one non-empty single path component"


@dataclass(frozen=True, slots=True)
class ProfileWorkerError(RuntimeError):
    """Report a dedicated profiler worker failure without making a benchmark claim."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render one concise worker failure reason."""
        return f"Benchmark v2 profile worker failed: {self.reason}"


@dataclass(frozen=True, slots=True)
class ProfileArtifactEntry:
    """Bind one profile-relative artifact to its exact bytes."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BenchmarkV2ProfileArtifacts:
    """Locate one complete separate profile package and its operator evidence."""

    profile_directory: Path
    profile_manifest_path: Path
    top_operators_csv_path: Path
    profile_markdown_path: Path
    trace_json_path: Path


@dataclass(frozen=True, slots=True)
class _ProfileWorkerRequest:
    """Define all state owned by the dedicated profiled-step subprocess."""

    protocol_version: int
    case: BenchmarkV2Case
    benchmark_seed: int
    vocab_size: int
    warmup_steps: int
    active_steps: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: tuple[str, ...]
    output_directory: Path


def _utc_now_iso() -> str:
    """Return one timezone-aware UTC timestamp for profile identity evidence."""
    return datetime.now(UTC).isoformat()


def _case_document(case: BenchmarkV2Case) -> JsonObject:
    """Serialize one explicit v2 case for the profiled worker protocol."""
    return {
        "name": case.name,
        "model_name": case.model_name,
        "n_layer": case.n_layer,
        "n_head": case.n_head,
        "n_embd": case.n_embd,
        "torch_num_threads": case.torch_num_threads,
        "block_size": case.block_size,
        "batch_size": case.batch_size,
    }


def _worker_request_document(request: _ProfileWorkerRequest) -> JsonObject:
    """Serialize a strict profiled-worker request without benchmark timer fields."""
    return {
        "protocol_version": request.protocol_version,
        "case": _case_document(request.case),
        "benchmark_seed": request.benchmark_seed,
        "vocab_size": request.vocab_size,
        "warmup_steps": request.warmup_steps,
        "active_steps": request.active_steps,
        "torch_num_interop_threads": request.torch_num_interop_threads,
        "cpu_affinity": list(request.cpu_affinity) if request.cpu_affinity is not None else None,
        "relevant_environment_variables": list(request.relevant_environment_variables),
        "output_directory": str(request.output_directory),
    }


def _require_mapping(
    value: object, expected_keys: frozenset[str], context: str
) -> dict[str, object]:
    """Require one string-keyed object with exactly the requested fields."""
    if not isinstance(value, dict):
        raise ProfileWorkerError(f"{context} must be an object")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        raise ProfileWorkerError(f"{context} keys must be strings")
    document = cast("dict[str, object]", raw)
    if frozenset(document) != expected_keys:
        raise ProfileWorkerError(f"{context} has an invalid field set")
    return document


def _integer(
    document: dict[str, object],
    key: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Read one strict integer, optionally requiring a positive value."""
    value = document[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (positive and value <= 0)
        or (non_negative and value < 0)
    ):
        suffix = " positive integer" if positive else " integer"
        raise ProfileWorkerError(f"{key} must be a{suffix}")
    return value


def _string(document: dict[str, object], key: str) -> str:
    """Read one required non-empty string protocol field."""
    value = document[key]
    if not isinstance(value, str) or not value:
        raise ProfileWorkerError(f"{key} must be a non-empty string")
    return value


def _profile_case(value: object) -> BenchmarkV2Case:
    """Parse one fully resolved explicit case for the profiled worker."""
    document = _require_mapping(value, _CASE_KEYS, "case")
    case = BenchmarkV2Case(
        name=_string(document, "name"),
        model_name=_string(document, "model_name"),
        n_layer=_integer(document, "n_layer", positive=True),
        n_head=_integer(document, "n_head", positive=True),
        n_embd=_integer(document, "n_embd", positive=True),
        torch_num_threads=_integer(document, "torch_num_threads", positive=True),
        block_size=_integer(document, "block_size", positive=True),
        batch_size=_integer(document, "batch_size", positive=True),
    )
    if case.n_embd % case.n_head != 0:
        raise ProfileWorkerError("n_embd must be divisible by n_head")
    return case


def _affinity(document: dict[str, object]) -> tuple[int, ...] | None:
    """Parse null or unique non-negative logical CPU IDs."""
    value = document["cpu_affinity"]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ProfileWorkerError("cpu_affinity must be a non-empty list or null")
    values = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise ProfileWorkerError("cpu_affinity must contain non-negative integers")
    result = cast("tuple[int, ...]", tuple(values))
    if len(result) != len(set(result)):
        raise ProfileWorkerError("cpu_affinity must not contain duplicates")
    return result


def _environment_variable_names(document: dict[str, object]) -> tuple[str, ...]:
    """Parse the configured relevant environment variable names."""
    value = document["relevant_environment_variables"]
    if not isinstance(value, list):
        raise ProfileWorkerError(
            "relevant_environment_variables must be a list of non-empty strings"
        )
    values = cast("list[object]", value)
    if any(not isinstance(item, str) or not item for item in values):
        raise ProfileWorkerError(
            "relevant_environment_variables must be a list of non-empty strings"
        )
    names = cast("tuple[str, ...]", tuple(values))
    if len(names) != len(set(names)):
        raise ProfileWorkerError("relevant_environment_variables must not contain duplicates")
    return names


def _parse_worker_request(raw: object) -> _ProfileWorkerRequest:
    """Validate one strict, timing-free profiled-worker request."""
    document = _require_mapping(raw, _REQUEST_KEYS, "profile request")
    protocol_version = _integer(document, "protocol_version", positive=True)
    if protocol_version != _PROFILE_PROTOCOL_VERSION:
        raise ProfileWorkerError(f"protocol_version must be {_PROFILE_PROTOCOL_VERSION}")
    output_directory = Path(_string(document, "output_directory"))
    if not output_directory.is_absolute():
        raise ProfileWorkerError("output_directory must be absolute")
    return _ProfileWorkerRequest(
        protocol_version=protocol_version,
        case=_profile_case(document["case"]),
        benchmark_seed=_integer(document, "benchmark_seed"),
        vocab_size=_integer(document, "vocab_size", positive=True),
        warmup_steps=_integer(document, "warmup_steps", non_negative=True),
        active_steps=_integer(document, "active_steps", positive=True),
        torch_num_interop_threads=_integer(document, "torch_num_interop_threads", positive=True),
        cpu_affinity=_affinity(document),
        relevant_environment_variables=_environment_variable_names(document),
        output_directory=output_directory,
    )


def _write_operator_csv(path: Path, events: Sequence[_ProfilerEvent]) -> None:
    """Write the top profiler operators without deriving any benchmark metric."""
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "rank",
            "operator",
            "count",
            "self_cpu_time_ms",
            "total_cpu_time_ms",
            "self_cpu_memory_bytes",
            "input_shapes",
        )
    )
    for rank, event in enumerate(events, start=1):
        writer.writerow(
            (
                rank,
                event.key,
                event.count,
                event.self_cpu_time_total / 1_000,
                event.cpu_time_total / 1_000,
                event.self_cpu_memory_usage,
                event.input_shapes,
            )
        )
    _atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _write_profile_report(path: Path, case: BenchmarkV2Case, active_steps: int) -> None:
    """Explain the profile scope while prohibiting benchmark-timing interpretation."""
    content = f"""# Benchmark v2 CPU operator profile

- Case: `{case.name}`
- Model: `{case.n_layer}` layers, `{case.n_head}` heads, `{case.n_embd}` embedding
- Active profiled steps: {active_steps}
- Activities: CPU only
- Schedule: one active CPU-only profiler window
- Shape recording: enabled
- Memory profiling: enabled

## High-level scopes

- `data_preparation`: random window sampling and tensor construction.
- `forward_backward`: model forward, cross entropy, backward, and gradient clipping.
- `optimizer_step`: AdamW parameter update.

## Interpretation boundary

Profiler overhead makes these timings unsuitable for benchmark comparisons.
They are not benchmark timings and never populate Benchmark v2 raw replicates, summaries, or
comparisons.

Open `trace.json` in `chrome://tracing` or Perfetto. See `top_operators.csv` for the selected
operators by self CPU time.
"""
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically create or replace one artifact using a unique sibling temporary file."""
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("xb") as stream:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _worker_environment_document(
    request: _ProfileWorkerRequest,
    *,
    effective_cpu_affinity: tuple[int, ...] | None,
) -> JsonObject:
    """Serialize post-profile worker CPU controls and runtime identity."""
    environment = capture_worker_environment(
        requested_cpu_affinity=request.cpu_affinity,
        effective_cpu_affinity=effective_cpu_affinity,
        relevant_environment_variables=request.relevant_environment_variables,
    )
    return {
        "platform": environment.platform,
        "python_version": environment.python_version,
        "torch_version": environment.torch_version,
        "torch_num_threads": environment.torch_num_threads,
        "torch_num_interop_threads": environment.torch_num_interop_threads,
        "logical_cpu_count": environment.logical_cpu_count,
        "requested_cpu_affinity": (
            list(environment.requested_cpu_affinity)
            if environment.requested_cpu_affinity is not None
            else None
        ),
        "effective_cpu_affinity": (
            list(environment.effective_cpu_affinity)
            if environment.effective_cpu_affinity is not None
            else None
        ),
        "relevant_environment_variables": cast(
            "dict[str, JsonValue]", environment.relevant_environment_variables
        ),
    }


def _run_profile_worker(request: _ProfileWorkerRequest) -> JsonObject:
    """Run only profiled steps in a fresh process; no canonical benchmark timer is present."""
    torch.set_num_threads(request.case.torch_num_threads)
    torch.set_num_interop_threads(request.torch_num_interop_threads)
    effective_cpu_affinity = apply_cpu_affinity(request.cpu_affinity)
    workload = create_training_step_workload(
        request.case,
        seed=request.benchmark_seed,
        vocab_size=request.vocab_size,
    )
    for _ in range(request.warmup_steps):
        workload.profiled_step()
    trace_path = request.output_directory / "trace.json"
    csv_path = request.output_directory / "top_operators.csv"
    markdown_path = request.output_directory / "profile_report.md"
    schedule_factory = cast(
        "Callable[..., Callable[[int], ProfilerAction]]",
        torch.profiler.schedule,
    )
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=schedule_factory(wait=0, warmup=0, active=request.active_steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
        acc_events=True,
    ) as profiler:
        for _ in range(request.active_steps):
            workload.profiled_step()
            profiler.step()
    profiler.export_chrome_trace(str(trace_path))
    averaged_events = cast("list[_ProfilerEvent]", profiler.key_averages(group_by_input_shape=True))
    ranked_events = sorted(
        averaged_events, key=lambda event: event.self_cpu_time_total, reverse=True
    )
    selected_events = list(ranked_events[:50])
    selected_names = {event.key for event in selected_events}
    selected_events.extend(
        event
        for event in ranked_events
        if event.key in {"data_preparation", "forward_backward", "optimizer_step"}
        and event.key not in selected_names
    )
    _write_operator_csv(csv_path, selected_events)
    _write_profile_report(markdown_path, request.case, request.active_steps)
    return {
        "status": "ok",
        "worker_pid": os.getpid(),
        "environment": _worker_environment_document(
            request,
            effective_cpu_affinity=effective_cpu_affinity,
        ),
    }


def profile_worker_main() -> int:
    """Run the strict dedicated worker protocol and emit one compact JSON response."""
    try:
        request = _parse_worker_request(cast("object", json.loads(sys.stdin.read())))
        response = _run_profile_worker(request)
        status = 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - ordinary worker failures must cross the JSON boundary.
        response = {"status": "error", "error_type": type(error).__name__, "message": str(error)}
        status = 1
    _ = sys.stdout.write(
        json.dumps(response, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return status


def _selected_case(config: BenchmarkV2Config) -> BenchmarkV2Case:
    """Select the single configured case and reject disabled profiling explicitly."""
    if not config.profile.enabled:
        raise ProfileConfigurationError("profile.enabled must be true")
    try:
        return next(case for case in config.cases if case.name == config.profile.case_name)
    except StopIteration as error:
        raise ProfileConfigurationError("profile.case_name is an unknown case") from error


def _profile_run_id(config: BenchmarkV2Config) -> str:
    """Create one readable profile-run identity that includes its source-config prefix."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"profile-{timestamp}-{resolved_config_sha256(config)[:12]}"


def _validate_profile_run_id(run_id: str) -> str:
    """Reject path syntax so a caller cannot place profile evidence outside its root."""
    candidate = Path(run_id)
    if (
        not run_id
        or candidate.is_absolute()
        or candidate.drive
        or len(candidate.parts) != 1
        or candidate.name != run_id
        or candidate.name in {".", ".."}
    ):
        raise InvalidProfileRunIdError(run_id)
    return run_id


def _artifact_entry(profile_directory: Path, path: Path) -> ProfileArtifactEntry:
    """Hash a finalized profile artifact using only its profile-relative path."""
    content = path.read_bytes()
    return ProfileArtifactEntry(
        path=path.relative_to(profile_directory).as_posix(),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _profile_manifest_document(  # noqa: PLR0913
    *,
    config: BenchmarkV2Config,
    source_config_path: Path,
    case: BenchmarkV2Case,
    profile_run_id: str,
    started_at_utc: str,
    ended_at_utc: str,
    parent_environment: JsonObject,
    worker_environment: JsonObject,
    artifacts: tuple[ProfileArtifactEntry, ...],
) -> JsonObject:
    """Bind a standalone profile to source configuration, case, method, and exact artifacts."""
    return {
        "schema_version": 1,
        "profile_run_id": profile_run_id,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "config_sha256": resolved_config_sha256(config),
        "case_name": case.name,
        "case_identity": case_identity(config, case),
        "source": {
            "config_path": source_config_path.resolve().as_posix(),
            "config_sha256": resolved_config_sha256(config),
            "case_name": case.name,
            "case_identity": case_identity(config, case),
        },
        "source_run": {
            "kind": "profiled_case_only",
            "benchmark_run_id": None,
            "benchmark_run_manifest": None,
            "config_sha256": resolved_config_sha256(config),
            "case_identity": case_identity(config, case),
        },
        "git": parent_environment["git"],
        "environment": {"parent": parent_environment, "worker": worker_environment},
        "method": {
            "worker_mode": "dedicated_profiled_step",
            "activities": ["CPU"],
            "schedule": {
                "wait": 0,
                "warmup": 0,
                "active": config.profile.active_steps,
                "repeat": 1,
            },
            "record_shapes": True,
            "profile_memory": True,
            "warmup_steps": config.profile.warmup_steps,
            "active_steps": config.profile.active_steps,
            "canonical_benchmark_timer_used": False,
        },
        "profiling_timings_are_not_benchmark_timings": True,
        "artifacts": [
            {"path": artifact.path, "size_bytes": artifact.size_bytes, "sha256": artifact.sha256}
            for artifact in artifacts
        ],
    }


def _launch_profile_worker(request: _ProfileWorkerRequest, timeout_seconds: float) -> JsonObject:
    """Run exactly one fresh subprocess that owns all profiled PyTorch state."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "minigpt.benchmark_v2_profile", "--worker"],
            input=json.dumps(
                _worker_request_document(request), allow_nan=False, separators=(",", ":")
            ),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        reason = f"worker timed out after {timeout_seconds} seconds"
        raise ProfileWorkerError(reason) from error
    try:
        raw_response = cast("object", json.loads(completed.stdout))
        response = _require_mapping(
            raw_response,
            frozenset({"status", "worker_pid", "environment"}),
            "profile worker response",
        )
    except (json.JSONDecodeError, ProfileWorkerError) as error:
        reason = (
            f"invalid worker response (exit {completed.returncode}): "
            f"{completed.stderr.strip() or error}"
        )
        raise ProfileWorkerError(reason) from error
    if completed.returncode != 0 or response["status"] != "ok":
        raise ProfileWorkerError(
            f"worker exited {completed.returncode}: {completed.stderr.strip()}"
        )
    environment = response["environment"]
    if not isinstance(environment, dict):
        raise ProfileWorkerError("worker response has no environment object")
    return cast("JsonObject", environment)


def run_benchmark_v2_profile(
    config: BenchmarkV2Config,
    source_config_path: Path,
    *,
    run_id: str | None = None,
) -> BenchmarkV2ProfileArtifacts:
    """Create a separate profile package without invoking any Benchmark v2 measurement path."""
    case = _selected_case(config)
    profile_run_id = _validate_profile_run_id(run_id or _profile_run_id(config))
    profile_directory = config.output_root / "profiles" / profile_run_id
    if profile_directory.exists():
        raise ProfileRunDirectoryCollisionError(profile_directory)
    profile_directory.mkdir(parents=True, exist_ok=False)
    started_at_utc = _utc_now_iso()
    parent_environment = capture_run_environment(config)
    request = _ProfileWorkerRequest(
        protocol_version=_PROFILE_PROTOCOL_VERSION,
        case=case,
        benchmark_seed=config.benchmark_seed,
        vocab_size=config.vocab_size,
        warmup_steps=config.profile.warmup_steps,
        active_steps=config.profile.active_steps,
        torch_num_interop_threads=config.torch_num_interop_threads,
        cpu_affinity=config.cpu_affinity,
        relevant_environment_variables=config.relevant_environment_variables,
        output_directory=profile_directory.resolve(),
    )
    worker_environment = _launch_profile_worker(request, config.worker_timeout_seconds)
    top_operators_csv_path = profile_directory / "top_operators.csv"
    profile_markdown_path = profile_directory / "profile_report.md"
    trace_json_path = profile_directory / "trace.json"
    if any(
        not path.is_file()
        for path in (top_operators_csv_path, profile_markdown_path, trace_json_path)
    ):
        raise ProfileWorkerError("worker did not produce the complete profile artifact set")
    artifacts = tuple(
        _artifact_entry(profile_directory, path)
        for path in (top_operators_csv_path, profile_markdown_path, trace_json_path)
    )
    profile_manifest_path = profile_directory / "profile_manifest.json"
    _atomic_write_bytes(
        profile_manifest_path,
        json.dumps(
            _profile_manifest_document(
                config=config,
                source_config_path=source_config_path,
                case=case,
                profile_run_id=profile_run_id,
                started_at_utc=started_at_utc,
                ended_at_utc=_utc_now_iso(),
                parent_environment=parent_environment,
                worker_environment=worker_environment,
                artifacts=artifacts,
            ),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )
    return BenchmarkV2ProfileArtifacts(
        profile_directory=profile_directory,
        profile_manifest_path=profile_manifest_path,
        top_operators_csv_path=top_operators_csv_path,
        profile_markdown_path=profile_markdown_path,
        trace_json_path=trace_json_path,
    )


def _module_main(arguments: Sequence[str] | None = None) -> int:
    """Dispatch only the internal worker mode when this module is executed directly."""
    parser = argparse.ArgumentParser(add_help=False)
    _ = parser.add_argument("--worker", action="store_true")
    parsed = parser.parse_args(arguments)
    worker = cast("bool", parsed.worker)
    if not worker:
        parser.error("only --worker is supported when executing this module directly")
    return profile_worker_main()


if __name__ == "__main__":
    raise SystemExit(_module_main())
