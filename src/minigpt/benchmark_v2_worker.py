"""Run one Benchmark v2 request in a fresh, versioned worker process."""

from __future__ import annotations

import gc
import json
import math
import string
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Never, TypeAlias, cast

import torch
from typing_extensions import override

from minigpt.benchmark_v2_environment import (
    PeakRssMethod,
    WorkerEnvironment,
    apply_cpu_affinity,
    capture_worker_environment,
    read_process_memory,
)
from minigpt.benchmark_v2_types import BenchmarkV2Case
from minigpt.benchmark_workload import create_training_step_workload

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

WORKER_PROTOCOL_VERSION = 1
_SHA256_HEX_LENGTH = 64
_REQUEST_KEYS = frozenset(
    {
        "protocol_version",
        "case_identity",
        "replicate_index",
        "case",
        "benchmark_seed",
        "vocab_size",
        "warmup_steps",
        "measurement_steps",
        "torch_num_interop_threads",
        "cpu_affinity",
        "relevant_environment_variables",
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


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """Describe exactly one isolated case replicate."""

    protocol_version: int
    case_identity: str
    replicate_index: int
    case: BenchmarkV2Case
    benchmark_seed: int
    vocab_size: int
    warmup_steps: int
    measurement_steps: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Report one successful uninstrumented measurement and its evidence."""

    protocol_version: int
    status: Literal["ok"]
    case_identity: str
    case_name: str
    replicate_index: int
    measurement_steps: int
    elapsed_seconds: float
    step_time_ms: float
    tokens_per_second: float
    tokens_per_step: int
    parameter_count: int
    final_rss_mib: float
    peak_rss_mib: float
    peak_rss_method: PeakRssMethod
    peak_rss_sampling_interval_ms: None
    environment: WorkerEnvironment


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Report an ordinary worker failure through the JSON protocol."""

    protocol_version: int
    status: Literal["error"]
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class InvalidWorkerRequestError(ValueError):
    """Report a request that violates the exact worker protocol schema."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the request validation reason."""
        return f"invalid Benchmark v2 worker request: {self.reason}"


def _invalid(reason: str) -> Never:
    """Raise one consistently typed strict request error."""
    raise InvalidWorkerRequestError(reason)


def _mapping(value: object, expected_keys: frozenset[str], context: str) -> dict[str, object]:
    """Require a string-keyed mapping with one exact key set."""
    if not isinstance(value, dict):
        _invalid(f"{context} must be an object")
    raw_mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw_mapping):
        _invalid(f"{context} keys must be strings")
    mapping = cast("dict[str, object]", raw_mapping)
    actual_keys = set(mapping)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing:
        _invalid(f"{context} missing key {min(missing)!r}")
    if unexpected:
        _invalid(f"{context} has unexpected key {min(unexpected)!r}")
    return mapping


def _string(document: dict[str, object], key: str) -> str:
    """Read one required non-empty string."""
    value = document[key]
    if not isinstance(value, str) or not value:
        _invalid(f"{key} must be a non-empty string")
    return value


def _integer(
    document: dict[str, object],
    key: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Read one required integer with an optional lower bound."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"{key} must be an integer")
    if positive and value <= 0:
        _invalid(f"{key} must be positive")
    if non_negative and value < 0:
        _invalid(f"{key} must be non-negative")
    return value


def _affinity(document: dict[str, object]) -> tuple[int, ...] | None:
    """Read a null or non-empty list of distinct logical CPU IDs."""
    value = document["cpu_affinity"]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        _invalid("cpu_affinity must be a non-empty list or null")
    raw_values = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw_values):
        _invalid("cpu_affinity items must be non-negative logical CPU IDs")
    values = cast("tuple[int, ...]", tuple(raw_values))
    if len(values) != len(set(values)):
        _invalid("cpu_affinity must not contain duplicates")
    return values


def _environment_variable_names(document: dict[str, object]) -> tuple[str, ...]:
    """Read a list of distinct non-empty environment-variable names."""
    value = document["relevant_environment_variables"]
    if not isinstance(value, list):
        _invalid("relevant_environment_variables must be a list")
    raw_values = cast("list[object]", value)
    if any(not isinstance(item, str) or not item for item in raw_values):
        _invalid("relevant_environment_variables must contain non-empty strings")
    values = cast("tuple[str, ...]", tuple(raw_values))
    if len(values) != len(set(values)):
        _invalid("relevant_environment_variables must not contain duplicates")
    return values


def _case(value: object) -> BenchmarkV2Case:
    """Parse and validate one explicit Benchmark v2 case."""
    document = _mapping(value, _CASE_KEYS, "case")
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
        _invalid("n_embd must be divisible by n_head")
    return case


def _parse_worker_request(raw: object) -> WorkerRequest:
    """Parse one strict JSON-safe worker request object."""
    document = _mapping(raw, _REQUEST_KEYS, "request")
    protocol_version = _integer(document, "protocol_version")
    if protocol_version != WORKER_PROTOCOL_VERSION:
        _invalid(f"protocol_version must be {WORKER_PROTOCOL_VERSION}")
    case_identity = _string(document, "case_identity")
    if len(case_identity) != _SHA256_HEX_LENGTH or any(
        character not in string.hexdigits for character in case_identity
    ):
        _invalid("case_identity must be a 64-character hexadecimal SHA-256")
    return WorkerRequest(
        protocol_version=protocol_version,
        case_identity=case_identity.lower(),
        replicate_index=_integer(document, "replicate_index", non_negative=True),
        case=_case(document["case"]),
        benchmark_seed=_integer(document, "benchmark_seed"),
        vocab_size=_integer(document, "vocab_size", positive=True),
        warmup_steps=_integer(document, "warmup_steps", positive=True),
        measurement_steps=_integer(document, "measurement_steps", positive=True),
        torch_num_interop_threads=_integer(
            document,
            "torch_num_interop_threads",
            positive=True,
        ),
        cpu_affinity=_affinity(document),
        relevant_environment_variables=_environment_variable_names(document),
    )


def worker_request_document(request: WorkerRequest) -> dict[str, JsonValue]:
    """Serialize a worker request for one fresh subprocess."""
    return {
        "protocol_version": request.protocol_version,
        "case_identity": request.case_identity,
        "replicate_index": request.replicate_index,
        "case": {
            "name": request.case.name,
            "model_name": request.case.model_name,
            "n_layer": request.case.n_layer,
            "n_head": request.case.n_head,
            "n_embd": request.case.n_embd,
            "torch_num_threads": request.case.torch_num_threads,
            "block_size": request.case.block_size,
            "batch_size": request.case.batch_size,
        },
        "benchmark_seed": request.benchmark_seed,
        "vocab_size": request.vocab_size,
        "warmup_steps": request.warmup_steps,
        "measurement_steps": request.measurement_steps,
        "torch_num_interop_threads": request.torch_num_interop_threads,
        "cpu_affinity": list(request.cpu_affinity) if request.cpu_affinity is not None else None,
        "relevant_environment_variables": list(request.relevant_environment_variables),
    }


def _environment_document(environment: WorkerEnvironment) -> dict[str, JsonValue]:
    """Serialize worker environment evidence without losing null values."""
    environment_variables = dict[str, JsonValue](environment.relevant_environment_variables)
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
        "relevant_environment_variables": environment_variables,
    }


def worker_response_document(response: WorkerResult | WorkerFailure) -> dict[str, JsonValue]:
    """Serialize one successful result or ordinary failure."""
    if isinstance(response, WorkerFailure):
        return {
            "protocol_version": response.protocol_version,
            "status": response.status,
            "error_type": response.error_type,
            "message": response.message,
        }
    return {
        "protocol_version": response.protocol_version,
        "status": response.status,
        "case_identity": response.case_identity,
        "case_name": response.case_name,
        "replicate_index": response.replicate_index,
        "measurement_steps": response.measurement_steps,
        "elapsed_seconds": response.elapsed_seconds,
        "step_time_ms": response.step_time_ms,
        "tokens_per_second": response.tokens_per_second,
        "tokens_per_step": response.tokens_per_step,
        "parameter_count": response.parameter_count,
        "final_rss_mib": response.final_rss_mib,
        "peak_rss_mib": response.peak_rss_mib,
        "peak_rss_method": response.peak_rss_method,
        "peak_rss_sampling_interval_ms": response.peak_rss_sampling_interval_ms,
        "environment": _environment_document(response.environment),
    }


def run_worker_request(request: WorkerRequest) -> WorkerResult:
    """Apply CPU controls and execute one canonical uninstrumented timer."""
    if request.protocol_version != WORKER_PROTOCOL_VERSION:
        _invalid(f"protocol_version must be {WORKER_PROTOCOL_VERSION}")
    torch.set_num_threads(request.case.torch_num_threads)
    torch.set_num_interop_threads(request.torch_num_interop_threads)
    effective_cpu_affinity = apply_cpu_affinity(request.cpu_affinity)
    workload = create_training_step_workload(
        request.case,
        seed=request.benchmark_seed,
        vocab_size=request.vocab_size,
    )
    for _ in range(request.warmup_steps):
        workload.step()

    _ = gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = perf_counter()
        for _ in range(request.measurement_steps):
            workload.step()
        elapsed_seconds = perf_counter() - started
    finally:
        if gc_was_enabled:
            gc.enable()

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        reason = "worker timer must produce a positive finite duration"
        raise RuntimeError(reason)
    memory = read_process_memory()
    environment = capture_worker_environment(
        requested_cpu_affinity=request.cpu_affinity,
        effective_cpu_affinity=effective_cpu_affinity,
        relevant_environment_variables=request.relevant_environment_variables,
    )
    step_time_ms = elapsed_seconds * 1_000 / request.measurement_steps
    return WorkerResult(
        protocol_version=WORKER_PROTOCOL_VERSION,
        status="ok",
        case_identity=request.case_identity,
        case_name=request.case.name,
        replicate_index=request.replicate_index,
        measurement_steps=request.measurement_steps,
        elapsed_seconds=elapsed_seconds,
        step_time_ms=step_time_ms,
        tokens_per_second=workload.tokens_per_step / (step_time_ms / 1_000),
        tokens_per_step=workload.tokens_per_step,
        parameter_count=workload.parameter_count,
        final_rss_mib=memory.final_rss_mib,
        peak_rss_mib=memory.peak_rss_mib,
        peak_rss_method=memory.peak_rss_method,
        peak_rss_sampling_interval_ms=memory.peak_rss_sampling_interval_ms,
        environment=environment,
    )


def worker_main() -> int:
    """Read one request from stdin and write one compact JSON response."""
    try:
        raw_request = cast("object", json.loads(sys.stdin.read()))
        response: WorkerResult | WorkerFailure = run_worker_request(
            _parse_worker_request(raw_request)
        )
        status = 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - protocol converts ordinary worker failures.
        response = WorkerFailure(
            protocol_version=WORKER_PROTOCOL_VERSION,
            status="error",
            error_type=type(error).__name__,
            message=str(error),
        )
        status = 1
    document = worker_response_document(response)
    _ = sys.stdout.write(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(worker_main())
