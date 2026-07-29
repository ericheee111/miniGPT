"""Run a strict fresh-process microbenchmark for the TokenBatcher data path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
import yaml
from typing_extensions import override

from minigpt.batching import TokenBatcher
from minigpt.benchmark_v2_environment import (
    PEAK_RSS_SCOPE,
    apply_cpu_affinity,
    read_process_memory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from minigpt.benchmark_config import ConfigMapping, ConfigValue
    from minigpt.benchmark_v2_config import JsonValue

_SCHEMA_VERSION: Final = 1
_METHODOLOGY_VERSION: Final = 1
_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "experiment_name",
        "output_root",
        "worker_timeout_seconds",
        "warmup_batches",
        "measurement_batches",
        "replicates",
        "minimum_replicates",
        "max_cv_percent",
        "corpus_tokens",
        "seed",
        "cpu_affinity",
        "cases",
    }
)
_CASE_KEYS: Final = frozenset({"name", "batch_size", "block_size"})
_WORKER_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "case",
        "case_identity",
        "corpus_path",
        "warmup_batches",
        "measurement_batches",
        "seed",
        "cpu_affinity",
        "replicate_index",
    }
)
_WORKER_RESULT_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "case_name",
        "case_identity",
        "replicate_index",
        "batch_time_ms",
        "tokens_per_second",
        "checksum",
        "final_rss_mib",
        "peak_rss_mib",
        "peak_rss_scope",
        "effective_cpu_affinity",
        "python_version",
        "numpy_version",
        "torch_version",
    }
)


@dataclass(frozen=True, slots=True)
class InvalidBatcherBenchmarkConfigError(ValueError):
    """Report a malformed batch-only benchmark configuration."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the source and strict validation failure."""
        return f"invalid batcher benchmark config {self.source}: {self.reason}"


@dataclass(frozen=True, slots=True)
class BatcherBenchmarkCase:
    """Define one resolved TokenBatcher workload."""

    name: str
    batch_size: int
    block_size: int


@dataclass(frozen=True, slots=True)
class BatcherBenchmarkConfig:
    """Hold one strict batch-only benchmark experiment."""

    experiment_name: str
    output_root: Path
    worker_timeout_seconds: float
    warmup_batches: int
    measurement_batches: int
    replicates: int
    minimum_replicates: int
    max_cv_percent: float
    corpus_tokens: int
    seed: int
    cpu_affinity: tuple[int, ...] | None
    cases: tuple[BatcherBenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class BatcherBenchmarkArtifacts:
    """Name the durable files created for one batch-only benchmark run."""

    status: Literal["complete", "partial"]
    run_directory: Path
    run_manifest_path: Path
    raw_replicates_path: Path
    summary_csv_path: Path
    summary_markdown_path: Path


@dataclass(frozen=True, slots=True)
class _Task:
    """Bind one case to one replicate index."""

    case: BatcherBenchmarkCase
    case_identity: str
    replicate_index: int


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Construct a mapping while rejecting duplicate and non-string keys."""
    if not isinstance(node, yaml.MappingNode):
        msg = "expected a YAML mapping node"
        raise yaml.YAMLError(msg)
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            msg = "YAML mapping keys must be strings"
            raise yaml.YAMLError(msg)
        if key in mapping:
            msg = f"duplicate YAML mapping key {key!r}"
            raise yaml.YAMLError(msg)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: ConfigValue, context: str, source: Path) -> ConfigMapping:
    """Require one mapping."""
    if not isinstance(value, dict):
        raise InvalidBatcherBenchmarkConfigError(source, f"{context} must be a mapping")
    return cast("ConfigMapping", value)


def _exact_keys(
    document: Mapping[str, object],
    expected: frozenset[str],
    context: str,
    source: Path,
) -> None:
    """Reject missing and unexpected keys."""
    actual = set(document)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise InvalidBatcherBenchmarkConfigError(
            source,
            f"{context} missing key {min(missing)!r}",
        )
    if unexpected:
        raise InvalidBatcherBenchmarkConfigError(
            source,
            f"{context} has unexpected key {min(unexpected)!r}",
        )


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    """Read one non-empty string."""
    value = document[key]
    if not isinstance(value, str) or not value:
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be a non-empty string")
    return value


def _integer(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    """Read one bounded integer."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be an integer")
    if positive and value <= 0:
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be positive")
    if non_negative and value < 0:
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be non-negative")
    return value


def _number(document: ConfigMapping, key: str, source: Path) -> float:
    """Read one finite positive number."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise InvalidBatcherBenchmarkConfigError(source, f"{key} must be positive and finite")
    return number


def _affinity(document: ConfigMapping, source: Path) -> tuple[int, ...] | None:
    """Read optional distinct logical CPU IDs."""
    value = document["cpu_affinity"]
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise InvalidBatcherBenchmarkConfigError(
            source,
            "cpu_affinity must be a non-empty list or null",
        )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise InvalidBatcherBenchmarkConfigError(
            source,
            "cpu_affinity items must be non-negative integers",
        )
    if len(value) != len(set(value)):
        raise InvalidBatcherBenchmarkConfigError(
            source,
            "cpu_affinity must not contain duplicates",
        )
    return cast("tuple[int, ...]", tuple(value))


def _cases(document: ConfigMapping, source: Path) -> tuple[BatcherBenchmarkCase, ...]:
    """Read non-empty unique cases."""
    value = document["cases"]
    if not isinstance(value, list) or not value:
        raise InvalidBatcherBenchmarkConfigError(source, "cases must be a non-empty list")
    cases: list[BatcherBenchmarkCase] = []
    names: set[str] = set()
    for index, raw_case in enumerate(value):
        case_document = _mapping(raw_case, f"cases[{index}]", source)
        _exact_keys(case_document, _CASE_KEYS, f"cases[{index}]", source)
        name = _string(case_document, "name", source)
        if name in names:
            raise InvalidBatcherBenchmarkConfigError(source, f"duplicate case name {name!r}")
        names.add(name)
        cases.append(
            BatcherBenchmarkCase(
                name=name,
                batch_size=_integer(case_document, "batch_size", source, positive=True),
                block_size=_integer(case_document, "block_size", source, positive=True),
            )
        )
    return tuple(cases)


def load_batcher_benchmark_config(path: Path) -> BatcherBenchmarkConfig:
    """Load one exact-schema YAML benchmark configuration."""
    try:
        document_value = cast(
            "ConfigValue",
            yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_DuplicateKeySafeLoader,  # noqa: S506
            ),
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise InvalidBatcherBenchmarkConfigError(path, str(error)) from error
    document = _mapping(document_value, "top-level YAML value", path)
    _exact_keys(document, _TOP_LEVEL_KEYS, "top-level config", path)
    if _integer(document, "schema_version", path, positive=True) != _SCHEMA_VERSION:
        raise InvalidBatcherBenchmarkConfigError(
            path,
            f"schema_version must equal {_SCHEMA_VERSION}",
        )
    config = BatcherBenchmarkConfig(
        experiment_name=_string(document, "experiment_name", path),
        output_root=Path(_string(document, "output_root", path)),
        worker_timeout_seconds=_number(document, "worker_timeout_seconds", path),
        warmup_batches=_integer(document, "warmup_batches", path, non_negative=True),
        measurement_batches=_integer(document, "measurement_batches", path, positive=True),
        replicates=_integer(document, "replicates", path, positive=True),
        minimum_replicates=_integer(document, "minimum_replicates", path, positive=True),
        max_cv_percent=_number(document, "max_cv_percent", path),
        corpus_tokens=_integer(document, "corpus_tokens", path, positive=True),
        seed=_integer(document, "seed", path, non_negative=True),
        cpu_affinity=_affinity(document, path),
        cases=_cases(document, path),
    )
    if config.minimum_replicates > config.replicates:
        raise InvalidBatcherBenchmarkConfigError(
            path,
            "minimum_replicates must not exceed replicates",
        )
    largest_block = max(case.block_size for case in config.cases)
    if config.corpus_tokens <= largest_block:
        raise InvalidBatcherBenchmarkConfigError(
            path,
            "corpus_tokens must exceed every case block_size",
        )
    return config


def batcher_case_identity(
    case: BatcherBenchmarkCase,
    *,
    corpus_tokens: int,
    seed: int,
) -> str:
    """Hash only resolved workload and fixed methodology controls."""
    document = {
        "identity_schema_version": 1,
        "methodology_version": _METHODOLOGY_VERSION,
        "batch_size": case.batch_size,
        "block_size": case.block_size,
        "corpus_tokens": corpus_tokens,
        "seed": seed,
        "source_dtype": "uint16",
        "output_dtype": "int64",
        "device": "cpu",
        "timed_scope": "TokenBatcher.next_batch",
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _config_document(config: BatcherBenchmarkConfig) -> dict[str, JsonValue]:
    """Serialize resolved config deterministically."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment_name": config.experiment_name,
        "output_root": str(config.output_root),
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "warmup_batches": config.warmup_batches,
        "measurement_batches": config.measurement_batches,
        "replicates": config.replicates,
        "minimum_replicates": config.minimum_replicates,
        "max_cv_percent": config.max_cv_percent,
        "corpus_tokens": config.corpus_tokens,
        "seed": config.seed,
        "cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity is not None else None,
        "cases": [cast("dict[str, JsonValue]", asdict(case)) for case in config.cases],
    }


def _config_sha256(config: BatcherBenchmarkConfig) -> str:
    """Hash a canonical resolved config."""
    canonical = json.dumps(
        _config_document(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _worker_request(
    task: _Task,
    config: BatcherBenchmarkConfig,
    corpus_path: Path,
) -> dict[str, JsonValue]:
    """Serialize one exact worker request."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "case": cast("dict[str, JsonValue]", asdict(task.case)),
        "case_identity": task.case_identity,
        "corpus_path": str(corpus_path),
        "warmup_batches": config.warmup_batches,
        "measurement_batches": config.measurement_batches,
        "seed": config.seed,
        "cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity is not None else None,
        "replicate_index": task.replicate_index,
    }


def _request_integer(document: Mapping[str, object], key: str, *, positive: bool = False) -> int:
    """Read an exact worker integer."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        msg = f"invalid worker request field {key}"
        raise ValueError(msg)
    return value


def _request_affinity(value: object) -> tuple[int, ...] | None:
    """Read worker affinity after the parent config was validated."""
    if value is None:
        return None
    if not isinstance(value, list):
        msg = "invalid worker request field cpu_affinity"
        raise TypeError(msg)
    items = cast("list[object]", value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in items):
        msg = "invalid worker request field cpu_affinity"
        raise ValueError(msg)
    return tuple(cast("list[int]", items))


def _execute_worker(request: Mapping[str, object]) -> dict[str, JsonValue]:
    """Execute one batch-only replicate inside its fresh worker."""
    _exact_worker_keys(request, _WORKER_REQUEST_KEYS, "worker request")
    if _request_integer(request, "schema_version", positive=True) != _SCHEMA_VERSION:
        msg = "worker schema version mismatch"
        raise ValueError(msg)
    raw_case = request["case"]
    if not isinstance(raw_case, dict):
        msg = "invalid worker request field case"
        raise TypeError(msg)
    case_mapping = cast("dict[str, object]", raw_case)
    _exact_worker_keys(case_mapping, _CASE_KEYS, "worker case")
    case_name = case_mapping["name"]
    case_identity = request["case_identity"]
    corpus_path = request["corpus_path"]
    if not isinstance(case_name, str) or not isinstance(case_identity, str):
        msg = "invalid worker case identity"
        raise TypeError(msg)
    if not isinstance(corpus_path, str):
        msg = "invalid worker corpus_path"
        raise TypeError(msg)
    case = BatcherBenchmarkCase(
        name=case_name,
        batch_size=_request_integer(case_mapping, "batch_size", positive=True),
        block_size=_request_integer(case_mapping, "block_size", positive=True),
    )
    warmup_batches = _request_integer(request, "warmup_batches")
    measurement_batches = _request_integer(request, "measurement_batches", positive=True)
    seed = _request_integer(request, "seed")
    replicate_index = _request_integer(request, "replicate_index")
    requested_affinity = _request_affinity(request["cpu_affinity"])
    effective_affinity = apply_cpu_affinity(requested_affinity)
    tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(Path(corpus_path), mmap_mode="r"),
    )
    batcher = TokenBatcher(
        tokens,
        batch_size=case.batch_size,
        block_size=case.block_size,
        seed=seed,
    )
    for _ in range(warmup_batches):
        _ = batcher.next_batch()
    start_ns = time.perf_counter_ns()
    last_x: torch.Tensor | None = None
    last_y: torch.Tensor | None = None
    for _ in range(measurement_batches):
        last_x, last_y = batcher.next_batch()
    elapsed_ns = time.perf_counter_ns() - start_ns
    if last_x is None or last_y is None:
        msg = "worker produced no measured batch"
        raise RuntimeError(msg)
    batch_time_ms = elapsed_ns / measurement_batches / 1_000_000
    memory = read_process_memory()
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "ok",
        "case_name": case.name,
        "case_identity": case_identity,
        "replicate_index": replicate_index,
        "batch_time_ms": batch_time_ms,
        "tokens_per_second": case.batch_size * case.block_size / (batch_time_ms / 1000),
        "checksum": int(last_x[0, 0]) + int(last_y[-1, -1]),
        "final_rss_mib": memory.final_rss_mib,
        "peak_rss_mib": memory.peak_rss_mib,
        "peak_rss_scope": PEAK_RSS_SCOPE,
        "effective_cpu_affinity": (
            list(effective_affinity) if effective_affinity is not None else None
        ),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


def _exact_worker_keys(
    document: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    """Reject a malformed internal worker protocol document."""
    if frozenset(document) != expected:
        msg = f"{context} keys do not match schema"
        raise ValueError(msg)


def _worker_main() -> int:
    """Read one worker request from stdin and emit one strict JSON response."""
    try:
        request_value = cast("object", json.loads(sys.stdin.read()))
        request = _json_object(request_value, "worker request")
        response = _execute_worker(request)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        _ = sys.stderr.write(f"batcher worker failed: {error}\n")
        return 1
    _ = sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
    return 0


def _json_object(value: object, context: str) -> dict[str, object]:
    """Require one decoded JSON object."""
    if not isinstance(value, dict):
        msg = f"{context} must be a JSON object"
        raise TypeError(msg)
    return cast("dict[str, object]", value)


def _validate_response(
    value: object,
    task: _Task,
) -> dict[str, JsonValue]:
    """Validate a worker response before accepting it as raw evidence."""
    if not isinstance(value, dict):
        msg = "worker response must be an object"
        raise TypeError(msg)
    document = cast("dict[str, object]", value)
    _exact_worker_keys(document, _WORKER_RESULT_KEYS, "worker response")
    if document["status"] != "ok":
        msg = "worker response status must be ok"
        raise ValueError(msg)
    if document["case_identity"] != task.case_identity:
        msg = "worker response case identity mismatch"
        raise ValueError(msg)
    if document["replicate_index"] != task.replicate_index:
        msg = "worker response replicate index mismatch"
        raise ValueError(msg)
    for field in (
        "batch_time_ms",
        "tokens_per_second",
        "final_rss_mib",
        "peak_rss_mib",
    ):
        metric = document[field]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            msg = f"worker response field {field} must be numeric"
            raise TypeError(msg)
        if not math.isfinite(float(metric)) or float(metric) <= 0.0:
            msg = f"worker response field {field} must be positive and finite"
            raise ValueError(msg)
    return cast("dict[str, JsonValue]", document)


def _git_value(arguments: list[str]) -> str:
    """Read one Git value without mutating the repository."""
    executable = shutil.which("git")
    if executable is None:
        return "unknown"
    completed = subprocess.run(  # noqa: S603 - fixed Git metadata command.
        [executable, *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _raw_record(task: _Task, response: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Bind accepted worker evidence to its task."""
    return {
        "status": "ok",
        "case_name": task.case.name,
        "case_identity": task.case_identity,
        "replicate_index": task.replicate_index,
        "worker_response": response,
    }


def _failure_record(
    task: _Task,
    completed: subprocess.CompletedProcess[str],
    reason: str,
) -> dict[str, JsonValue]:
    """Preserve a failed subprocess as raw evidence."""
    return {
        "status": "error",
        "case_name": task.case.name,
        "case_identity": task.case_identity,
        "replicate_index": task.replicate_index,
        "return_code": completed.returncode,
        "stderr": completed.stderr,
        "stdout": completed.stdout,
        "reason": reason,
    }


def _summaries(
    config: BatcherBenchmarkConfig,
    records: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Calculate unfiltered per-case descriptive statistics and stability."""
    summaries: list[dict[str, JsonValue]] = []
    for case in config.cases:
        case_records = [record for record in records if record["case_name"] == case.name]
        responses = [
            cast("dict[str, JsonValue]", record["worker_response"])
            for record in case_records
            if record["status"] == "ok"
        ]
        times = [float(cast("float", response["batch_time_ms"])) for response in responses]
        throughputs = [
            float(cast("float", response["tokens_per_second"])) for response in responses
        ]
        success_count = len(times)
        if success_count:
            mean = statistics.fmean(times)
            stddev = statistics.pstdev(times)
            cv_percent = stddev / mean * 100.0
            median = statistics.median(times)
            mad = statistics.median(abs(value - median) for value in times)
            stability = (
                "stable"
                if success_count >= config.minimum_replicates
                and cv_percent <= config.max_cv_percent
                else "unstable"
            )
            median_throughput: float | None = statistics.median(throughputs)
        else:
            cv_percent = None
            median = None
            mad = None
            stability = "unstable"
            median_throughput = None
        summaries.append(
            {
                "case_name": case.name,
                "case_identity": batcher_case_identity(
                    case,
                    corpus_tokens=config.corpus_tokens,
                    seed=config.seed,
                ),
                "replicate_count": len(case_records),
                "success_count": success_count,
                "failure_count": len(case_records) - success_count,
                "median_batch_time_ms": median,
                "median_absolute_deviation_batch_time_ms": mad,
                "coefficient_of_variation_percent": cv_percent,
                "median_tokens_per_second": median_throughput,
                "max_worker_lifetime_peak_rss_mib": (
                    max(float(cast("float", response["peak_rss_mib"])) for response in responses)
                    if responses
                    else None
                ),
                "peak_rss_scope": PEAK_RSS_SCOPE,
                "stability": stability,
            }
        )
    return summaries


def _write_csv(path: Path, summaries: list[dict[str, JsonValue]]) -> None:
    """Write stable summary columns."""
    fields = (
        "case_name",
        "case_identity",
        "replicate_count",
        "success_count",
        "failure_count",
        "median_batch_time_ms",
        "median_absolute_deviation_batch_time_ms",
        "coefficient_of_variation_percent",
        "median_tokens_per_second",
        "max_worker_lifetime_peak_rss_mib",
        "peak_rss_scope",
        "stability",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        write_row = cast("Callable[[object], object]", writer.writerow)
        _ = write_row(fields)
        for summary in summaries:
            _ = write_row([summary[field] for field in fields])


def _write_markdown(
    path: Path,
    config: BatcherBenchmarkConfig,
    summaries: list[dict[str, JsonValue]],
) -> None:
    """Write a compact human-readable batch-only summary."""
    lines = [
        f"# {config.experiment_name}",
        "",
        "This is an isolated `TokenBatcher.next_batch()` microbenchmark, not end-to-end training.",
        "",
        "| Case | Median ms/batch | Median tokens/s | CV % | MAD ms | Stability |",
        "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        [
            "| {case} | {time} | {throughput} | {cv} | {mad} | {stability} |".format(
                case=summary["case_name"],
                time=summary["median_batch_time_ms"],
                throughput=summary["median_tokens_per_second"],
                cv=summary["coefficient_of_variation_percent"],
                mad=summary["median_absolute_deviation_batch_time_ms"],
                stability=summary["stability"],
            )
            for summary in summaries
        ]
    )
    lines.extend(
        (
            "",
            "Each replicate ran in a fresh process. No outliers were removed.",
            "Peak RSS means worker lifetime peak RSS.",
            "",
        )
    )
    _ = path.write_text("\n".join(lines), encoding="utf-8")


def _save_corpus(path: Path, token_count: int) -> None:
    """Create a deterministic uint16 `.npy` corpus for read-only mmap workers."""
    tokens = np.arange(token_count, dtype=np.uint64)
    corpus = cast("npt.NDArray[np.uint16]", (tokens % 65).astype(np.uint16))
    save_tokens = cast(
        "Callable[[Path, npt.NDArray[np.uint16]], None]",
        np.save,
    )
    save_tokens(path, corpus)


def _run_id(config: BatcherBenchmarkConfig) -> str:
    """Build a sortable, commit-bound local run identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = _git_value(["rev-parse", "--short=12", "HEAD"])
    return f"{timestamp}-{commit}-{_config_sha256(config)[:12]}"


def _write_json(path: Path, document: Mapping[str, JsonValue]) -> None:
    """Write stable indented JSON."""
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_batcher_benchmark(config: BatcherBenchmarkConfig) -> BatcherBenchmarkArtifacts:
    """Run randomized fresh-process replicates and write their complete evidence."""
    run_directory = config.output_root / _run_id(config)
    run_directory.mkdir(parents=True, exist_ok=False)
    corpus_path = run_directory / "corpus.npy"
    raw_path = run_directory / "raw_replicates.jsonl"
    summary_csv_path = run_directory / "summary.csv"
    summary_markdown_path = run_directory / "summary.md"
    manifest_path = run_directory / "run_manifest.json"
    _save_corpus(corpus_path, config.corpus_tokens)
    tasks = [
        _Task(
            case=case,
            case_identity=batcher_case_identity(
                case,
                corpus_tokens=config.corpus_tokens,
                seed=config.seed,
            ),
            replicate_index=replicate_index,
        )
        for case in config.cases
        for replicate_index in range(config.replicates)
    ]
    random.Random(config.seed).shuffle(tasks)  # noqa: S311 - deterministic task ordering.
    records: list[dict[str, JsonValue]] = []
    command = [sys.executable, "-m", "minigpt.batcher_benchmark", "--worker"]
    for task in tasks:
        completed = subprocess.run(  # noqa: S603 - fixed current interpreter/module.
            command,
            input=json.dumps(_worker_request(task, config, corpus_path)),
            capture_output=True,
            check=False,
            text=True,
            timeout=config.worker_timeout_seconds,
        )
        if completed.returncode != 0:
            record = _failure_record(task, completed, "worker returned non-zero")
        else:
            try:
                response_value = cast("object", json.loads(completed.stdout))
                response = _validate_response(response_value, task)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                record = _failure_record(task, completed, str(error))
            else:
                record = _raw_record(task, response)
        records.append(record)
        with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
            _ = handle.write(json.dumps(record, sort_keys=True) + "\n")
    summaries = _summaries(config, records)
    _write_csv(summary_csv_path, summaries)
    _write_markdown(summary_markdown_path, config, summaries)
    complete = all(record["status"] == "ok" for record in records) and all(
        summary["stability"] == "stable" for summary in summaries
    )
    status: Literal["complete", "partial"] = "complete" if complete else "partial"
    manifest: dict[str, JsonValue] = {
        "schema_version": _SCHEMA_VERSION,
        "methodology_version": _METHODOLOGY_VERSION,
        "status": status,
        "experiment_name": config.experiment_name,
        "git_commit_sha": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--porcelain"])),
        "config_sha256": _config_sha256(config),
        "resolved_config": _config_document(config),
        "raw_replicate_count": len(records),
        "summaries": cast("list[JsonValue]", summaries),
        "artifacts": {
            "raw_replicates": raw_path.name,
            "summary_csv": summary_csv_path.name,
            "summary_markdown": summary_markdown_path.name,
        },
    }
    _write_json(manifest_path, manifest)
    return BatcherBenchmarkArtifacts(
        status=status,
        run_directory=run_directory,
        run_manifest_path=manifest_path,
        raw_replicates_path=raw_path,
        summary_csv_path=summary_csv_path,
        summary_markdown_path=summary_markdown_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the public/module CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    _ = parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a worker request or one configured batch-only benchmark."""
    arguments = _build_parser().parse_args(argv)
    if cast("bool", arguments.worker):
        return _worker_main()
    config_path = cast("Path | None", arguments.config)
    if config_path is None:
        _ = sys.stderr.write("--config is required\n")
        return 2
    try:
        config = load_batcher_benchmark_config(config_path)
        artifacts = run_batcher_benchmark(config)
    except (InvalidBatcherBenchmarkConfigError, OSError, subprocess.SubprocessError) as error:
        _ = sys.stderr.write(f"batcher benchmark failed: {error}\n")
        return 1
    _ = sys.stdout.write(f"status={artifacts.status}\nrun_manifest={artifacts.run_manifest_path}\n")
    return 0 if artifacts.status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
