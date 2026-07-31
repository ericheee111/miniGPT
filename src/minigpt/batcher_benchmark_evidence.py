"""Strictly load, recompute, and compare TokenBatcher benchmark evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from typing_extensions import override

from minigpt.batcher_benchmark import (
    BatcherBenchmarkConfig,
    batcher_case_identity,
    batcher_config_document,
    batcher_config_sha256,
    load_resolved_batcher_benchmark_config,
)
from minigpt.benchmark_v2_comparison_policy import ComparisonPolicy, comparison_policy_document

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "methodology_version",
        "run_id",
        "status",
        "started_at_utc",
        "ended_at_utc",
        "experiment_name",
        "git_commit_sha",
        "evidence_generator_git_commit_sha",
        "git_dirty",
        "config_sha256",
        "expected_task_count",
        "raw_replicate_count",
        "successful_task_count",
        "failed_task_count",
        "case_identities",
        "summaries",
        "artifacts",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "size_bytes", "sha256"})
_REQUIRED_ARTIFACTS = frozenset(
    {
        "environment.json",
        "execution_order.json",
        "raw_replicates.jsonl",
        "resolved_config.yaml",
        "summary.csv",
        "summary.md",
    }
)
_RAW_KEYS = frozenset(
    {
        "status",
        "case_name",
        "case_identity",
        "replicate_index",
        "worker_pid",
        "worker_response",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "worker_pid",
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
        "environment",
    }
)
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
_RUN_ENVIRONMENT_KEYS = frozenset(
    {
        "captured_before_first_worker",
        "platform",
        "machine",
        "cpu_name",
        "physical_cpu_count",
        "logical_cpu_count",
        "python_version",
        "numpy_version",
        "torch_version",
        "configured_cpu_affinity",
        "relevant_environment_variables",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_status",
        "config_sha256",
        "run_environment",
        "worker_environments",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "execution_index",
        "task_id",
        "case_name",
        "case_identity",
        "replicate_index",
        "worker_pid",
        "status",
    }
)
_SUMMARY_FIELDS = (
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
_COMPARISON_ENVIRONMENT_FIELDS = (
    "platform",
    "machine",
    "cpu_name",
    "physical_cpu_count",
    "logical_cpu_count",
    "python_version",
    "numpy_version",
    "torch_version",
    "configured_cpu_affinity",
    "relevant_environment_variables",
)
_LOWER_HEX = frozenset("0123456789abcdef")
_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40
_SCHEMA_VERSION = 2
_METHODOLOGY_VERSION = 2


@dataclass(slots=True)
class InvalidBatcherBenchmarkEvidenceError(ValueError):
    """Report one untrusted or internally inconsistent batch-only package."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the exact rejected artifact and reason."""
        return f"invalid batcher benchmark evidence {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class BatcherRawRecord:
    """Expose one validated raw replicate."""

    status: Literal["ok", "error"]
    case_name: str
    case_identity: str
    replicate_index: int
    worker_pid: int | None
    worker_response: dict[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class BatcherSummary:
    """Hold independently recomputable batch-only descriptive statistics."""

    case_name: str
    case_identity: str
    replicate_count: int
    success_count: int
    failure_count: int
    median_batch_time_ms: float | None
    median_absolute_deviation_batch_time_ms: float | None
    coefficient_of_variation_percent: float | None
    median_tokens_per_second: float | None
    max_worker_lifetime_peak_rss_mib: float | None
    peak_rss_scope: Literal["worker_lifetime"]
    stability: Literal["stable", "unstable"]


@dataclass(frozen=True, slots=True)
class LoadedBatcherBenchmarkRun:
    """Bind one strict run manifest to its raw-derived evidence."""

    manifest_path: Path
    run_id: str
    git_commit_sha: str
    evidence_generator_git_commit_sha: str
    config: BatcherBenchmarkConfig
    config_sha256: str
    run_environment: dict[str, JsonValue]
    records: tuple[BatcherRawRecord, ...]
    summaries: tuple[BatcherSummary, ...]


@dataclass(frozen=True, slots=True)
class BatcherCaseComparison:
    """Describe one policy-guarded candidate change."""

    case_identity: str
    case_name: str
    baseline_median_batch_time_ms: float | None
    candidate_median_batch_time_ms: float | None
    batch_time_change_percent: float | None
    throughput_change_percent: float | None
    peak_rss_change_mib: float | None
    baseline_cv_percent: float | None
    candidate_cv_percent: float | None
    baseline_stability: Literal["stable", "unstable"]
    candidate_stability: Literal["stable", "unstable"]
    reasons: tuple[str, ...]
    regressed: bool | None


@dataclass(frozen=True, slots=True)
class BatcherBenchmarkComparison:
    """Represent a machine-readable comparison under one shared policy."""

    baseline_manifest_path: Path
    candidate_manifest_path: Path
    baseline_run_id: str
    candidate_run_id: str
    baseline_git_commit_sha: str
    candidate_git_commit_sha: str
    policy_sha256: str
    comparison_policy: dict[str, JsonValue]
    environment_mismatches: tuple[str, ...]
    reasons: tuple[str, ...]
    cases: tuple[BatcherCaseComparison, ...]
    verdict: Literal["pass", "fail", "not_comparable"]


def _invalid(path: Path, reason: str) -> InvalidBatcherBenchmarkEvidenceError:
    """Build one consistently typed evidence failure."""
    return InvalidBatcherBenchmarkEvidenceError(path, reason)


def _mapping(value: object, path: Path, context: str) -> dict[str, object]:
    """Require one string-keyed object."""
    if not isinstance(value, dict):
        raise _invalid(path, f"{context} must be an object")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        raise _invalid(path, f"{context} keys must be strings")
    return cast("dict[str, object]", raw)


def _json(content: bytes, path: Path, context: str) -> object:
    """Decode strict UTF-8 JSON while rejecting non-finite constants."""

    def reject_constant(value: str) -> object:
        raise _invalid(path, f"{context} contains non-finite constant {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise _invalid(path, f"{context} contains duplicate key {key!r}")
            document[key] = value
        return document

    try:
        return cast(
            "object",
            json.loads(
                content.decode("utf-8"),
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _invalid(path, f"{context} is not valid UTF-8 JSON") from error


def _sha256(value: object, path: Path, context: str) -> str:
    """Require one lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise _invalid(path, f"{context} must be a lowercase SHA-256")
    return value


def _git_sha(value: object, path: Path) -> str:
    """Require one lowercase 40-character Git SHA."""
    if (
        not isinstance(value, str)
        or len(value) != _GIT_SHA_LENGTH
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise _invalid(path, "git_commit_sha must be 40 lowercase hexadecimal characters")
    return value


def _integer(value: object, path: Path, context: str, *, positive: bool = False) -> int:
    """Require one strict integer."""
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise _invalid(path, f"{context} must be {'positive ' if positive else ''}integer")
    return value


def _positive_number(value: object, path: Path, context: str) -> float:
    """Require one positive finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(path, f"{context} must be positive and finite")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise _invalid(path, f"{context} must be positive and finite")
    return number


def _artifact_contents(  # noqa: C901
    manifest_path: Path,
    document: dict[str, object],
) -> dict[str, bytes]:
    """Verify the exact self-excluded artifact set and return owned bytes."""
    raw_entries = document["artifacts"]
    if not isinstance(raw_entries, list):
        raise _invalid(manifest_path, "artifacts must be a list")
    entries: dict[str, tuple[int, str]] = {}
    for raw_entry in cast("list[object]", raw_entries):
        entry = _mapping(raw_entry, manifest_path, "artifact entry")
        if frozenset(entry) != _ARTIFACT_KEYS:
            raise _invalid(manifest_path, "artifact entry has an invalid field set")
        raw_path = entry["path"]
        size = entry["size_bytes"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or "\\" in raw_path
            or Path(raw_path).is_absolute()
            or ".." in Path(raw_path).parts
        ):
            raise _invalid(manifest_path, "artifact entry path is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise _invalid(manifest_path, "artifact size_bytes is invalid")
        entries[raw_path] = (size, _sha256(entry["sha256"], manifest_path, "artifact sha256"))
    if frozenset(entries) != _REQUIRED_ARTIFACTS:
        raise _invalid(manifest_path, "manifest does not bind the required artifact set")
    contents: dict[str, bytes] = {}
    for relative_path, (size, digest) in entries.items():
        path = manifest_path.parent / relative_path
        try:
            content = path.read_bytes()
        except OSError as error:
            raise _invalid(manifest_path, f"cannot read bound artifact {relative_path}") from error
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise _invalid(manifest_path, f"bound artifact hash or size mismatch: {relative_path}")
        contents[relative_path] = content
    expected_entries = _REQUIRED_ARTIFACTS | frozenset({"run_manifest.json"})
    try:
        actual_entries = frozenset(path.name for path in manifest_path.parent.iterdir())
    except OSError as error:
        raise _invalid(manifest_path, "cannot enumerate run package") from error
    if actual_entries != expected_entries:
        raise _invalid(manifest_path, "run package contains unexpected or missing entries")
    return contents


def _worker_environment(value: object, path: Path) -> dict[str, JsonValue]:
    """Validate one worker-observed environment document."""
    document = _mapping(value, path, "worker environment")
    if frozenset(document) != _WORKER_ENVIRONMENT_KEYS:
        raise _invalid(path, "worker environment has an invalid field set")
    for field in ("platform", "python_version", "torch_version"):
        if not isinstance(document[field], str) or not document[field]:
            raise _invalid(path, f"worker environment {field} is invalid")
    for field in ("torch_num_threads", "torch_num_interop_threads"):
        _ = _integer(document[field], path, f"worker environment {field}", positive=True)
    logical_count = document["logical_cpu_count"]
    if logical_count is not None:
        _ = _integer(logical_count, path, "worker environment logical_cpu_count", positive=True)
    for field in ("requested_cpu_affinity", "effective_cpu_affinity"):
        affinity = document[field]
        if affinity is not None and (
            not isinstance(affinity, list)
            or not affinity
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in cast("list[object]", affinity)
            )
        ):
            raise _invalid(path, f"worker environment {field} is invalid")
    variables = document["relevant_environment_variables"]
    if not isinstance(variables, dict) or any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in cast("dict[object, object]", variables).items()
    ):
        raise _invalid(path, "worker environment variables are invalid")
    return cast("dict[str, JsonValue]", document)


def _parse_raw_records(  # noqa: C901, PLR0912
    content: bytes,
    path: Path,
    config: BatcherBenchmarkConfig,
) -> tuple[BatcherRawRecord, ...]:
    """Validate every raw JSONL record and its worker-owned metrics."""
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise _invalid(path, "raw_replicates.jsonl is not UTF-8") from error
    expected_cases = {
        batcher_case_identity(case, corpus_tokens=config.corpus_tokens, seed=config.seed): case
        for case in config.cases
    }
    records: list[BatcherRawRecord] = []
    for index, line in enumerate(lines, 1):
        document = _mapping(
            _json(line.encode("utf-8"), path, f"raw line {index}"),
            path,
            f"raw line {index}",
        )
        if frozenset(document) != _RAW_KEYS:
            raise _invalid(path, f"raw line {index} has an invalid field set")
        status = document["status"]
        if status not in {"ok", "error"}:
            raise _invalid(path, f"raw line {index} has invalid status")
        name = document["case_name"]
        identity = _sha256(document["case_identity"], path, "raw case_identity")
        replicate = _integer(document["replicate_index"], path, "raw replicate_index")
        if replicate < 0 or not isinstance(name, str) or not name:
            raise _invalid(path, f"raw line {index} has invalid task identity")
        case = expected_cases.get(identity)
        if case is None or case.name != name or replicate >= config.replicates:
            raise _invalid(path, f"raw line {index} task is absent from resolved config")
        if status == "error":
            if document["worker_pid"] is not None or document["worker_response"] is not None:
                raise _invalid(path, f"raw line {index} has malformed failure evidence")
            records.append(BatcherRawRecord("error", name, identity, replicate, None, None))
            continue
        worker_pid = _integer(document["worker_pid"], path, "raw worker_pid", positive=True)
        response = _mapping(document["worker_response"], path, "worker response")
        if frozenset(response) != _RESPONSE_KEYS:
            raise _invalid(path, "worker response has an invalid field set")
        if (
            response["schema_version"] != _SCHEMA_VERSION
            or response["status"] != "ok"
            or response["worker_pid"] != worker_pid
            or response["case_name"] != name
            or response["case_identity"] != identity
            or response["replicate_index"] != replicate
            or response["peak_rss_scope"] != "worker_lifetime"
        ):
            raise _invalid(path, "worker response disagrees with raw task identity")
        batch_time = _positive_number(response["batch_time_ms"], path, "batch_time_ms")
        throughput = _positive_number(response["tokens_per_second"], path, "tokens_per_second")
        _ = _positive_number(response["final_rss_mib"], path, "final_rss_mib")
        peak_rss = _positive_number(response["peak_rss_mib"], path, "peak_rss_mib")
        if peak_rss < float(cast("float", response["final_rss_mib"])):
            raise _invalid(path, "peak_rss_mib is below final_rss_mib")
        expected_throughput = case.batch_size * case.block_size / (batch_time / 1_000)
        if not math.isclose(throughput, expected_throughput, rel_tol=1e-9):
            raise _invalid(path, "tokens_per_second disagrees with batch_time_ms")
        _ = _worker_environment(response["environment"], path)
        records.append(
            BatcherRawRecord(
                "ok",
                name,
                identity,
                replicate,
                worker_pid,
                cast("dict[str, JsonValue]", response),
            )
        )
    task_keys = {(record.case_identity, record.replicate_index) for record in records}
    if len(task_keys) != len(records):
        raise _invalid(path, "raw replicate task identities are not unique")
    return tuple(records)


def _summarize(
    config: BatcherBenchmarkConfig,
    records: tuple[BatcherRawRecord, ...],
    *,
    minimum_replicates: int | None = None,
    max_cv_percent: float | None = None,
) -> tuple[BatcherSummary, ...]:
    """Recompute median, MAD, population CV, RSS, and stability from raw records."""
    minimum = config.minimum_replicates if minimum_replicates is None else minimum_replicates
    max_cv = config.max_cv_percent if max_cv_percent is None else max_cv_percent
    summaries: list[BatcherSummary] = []
    for case in config.cases:
        identity = batcher_case_identity(
            case,
            corpus_tokens=config.corpus_tokens,
            seed=config.seed,
        )
        case_records = tuple(record for record in records if record.case_identity == identity)
        successful = tuple(
            record
            for record in case_records
            if record.status == "ok" and record.worker_response is not None
        )
        times = [
            float(
                cast("float", cast("dict[str, JsonValue]", record.worker_response)["batch_time_ms"])
            )
            for record in successful
        ]
        throughputs = [
            float(
                cast(
                    "float",
                    cast("dict[str, JsonValue]", record.worker_response)["tokens_per_second"],
                )
            )
            for record in successful
        ]
        peaks = [
            float(
                cast("float", cast("dict[str, JsonValue]", record.worker_response)["peak_rss_mib"])
            )
            for record in successful
        ]
        if times:
            median = statistics.median(times)
            cv = statistics.pstdev(times) / statistics.fmean(times) * 100.0
            mad = statistics.median(abs(value - median) for value in times)
            stability: Literal["stable", "unstable"] = (
                "stable" if len(times) >= minimum and cv <= max_cv else "unstable"
            )
            median_throughput: float | None = statistics.median(throughputs)
            max_peak: float | None = max(peaks)
        else:
            median = None
            cv = None
            mad = None
            median_throughput = None
            max_peak = None
            stability = "unstable"
        summaries.append(
            BatcherSummary(
                case.name,
                identity,
                len(case_records),
                len(successful),
                len(case_records) - len(successful),
                median,
                mad,
                cv,
                median_throughput,
                max_peak,
                "worker_lifetime",
                stability,
            )
        )
    return tuple(summaries)


def _parse_summary_csv(content: bytes, path: Path) -> tuple[BatcherSummary, ...]:
    """Parse the exact summary CSV schema without rounding."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _invalid(path, "summary.csv is not UTF-8") from error
    reader = csv.DictReader(StringIO(text))
    if tuple(reader.fieldnames or ()) != _SUMMARY_FIELDS:
        raise _invalid(path, "summary.csv has an invalid header")
    summaries: list[BatcherSummary] = []
    for index, row in enumerate(reader, 2):
        values = cast("dict[str, str]", row)
        try:
            nullable = {
                field: (None if values[field] == "" else float(values[field]))
                for field in (
                    "median_batch_time_ms",
                    "median_absolute_deviation_batch_time_ms",
                    "coefficient_of_variation_percent",
                    "median_tokens_per_second",
                    "max_worker_lifetime_peak_rss_mib",
                )
            }
            summary = BatcherSummary(
                values["case_name"],
                _sha256(values["case_identity"], path, f"summary row {index} identity"),
                int(values["replicate_count"]),
                int(values["success_count"]),
                int(values["failure_count"]),
                nullable["median_batch_time_ms"],
                nullable["median_absolute_deviation_batch_time_ms"],
                nullable["coefficient_of_variation_percent"],
                nullable["median_tokens_per_second"],
                nullable["max_worker_lifetime_peak_rss_mib"],
                cast("Literal['worker_lifetime']", values["peak_rss_scope"]),
                cast("Literal['stable', 'unstable']", values["stability"]),
            )
        except (KeyError, ValueError) as error:
            raise _invalid(path, f"summary.csv row {index} is invalid") from error
        if summary.peak_rss_scope != "worker_lifetime" or summary.stability not in {
            "stable",
            "unstable",
        }:
            raise _invalid(path, f"summary.csv row {index} has invalid classifications")
        summaries.append(summary)
    return tuple(summaries)


def _load_environment(  # noqa: PLR0913
    content: bytes,
    path: Path,
    *,
    run_id: str,
    status: str,
    config_sha256: str,
    records: tuple[BatcherRawRecord, ...],
) -> dict[str, JsonValue]:
    """Validate parent and worker environment evidence against raw records."""
    document = _mapping(_json(content, path, "environment.json"), path, "environment.json")
    if frozenset(document) != _ENVIRONMENT_KEYS:
        raise _invalid(path, "environment.json has an invalid field set")
    if (
        document["schema_version"] != 1
        or document["run_id"] != run_id
        or document["run_status"] != status
        or document["config_sha256"] != config_sha256
    ):
        raise _invalid(path, "environment.json disagrees with the manifest")
    run_environment = _mapping(document["run_environment"], path, "run_environment")
    if (
        frozenset(run_environment) != _RUN_ENVIRONMENT_KEYS
        or run_environment["captured_before_first_worker"] is not True
    ):
        raise _invalid(path, "run_environment has an invalid field set")
    raw_workers = [
        _worker_environment(record.worker_response["environment"], path)
        for record in records
        if record.status == "ok" and record.worker_response is not None
    ]
    reported = document["worker_environments"]
    if not isinstance(reported, list):
        raise _invalid(path, "worker_environments must be a list")
    reported_workers = [_worker_environment(item, path) for item in cast("list[object]", reported)]

    def normalize(value: dict[str, JsonValue]) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    if Counter(map(normalize, raw_workers)) != Counter(map(normalize, reported_workers)):
        raise _invalid(path, "worker environments disagree with raw replicates")
    return cast("dict[str, JsonValue]", run_environment)


def _validate_execution_order(
    content: bytes,
    path: Path,
    config: BatcherBenchmarkConfig,
    records: tuple[BatcherRawRecord, ...],
) -> None:
    """Validate exact task order, case identities, statuses, and worker PIDs."""
    raw_document = _json(content, path, "execution_order.json")
    if not isinstance(raw_document, list):
        raise _invalid(path, "execution_order.json must be a list")
    entries = cast("list[object]", raw_document)
    records_by_task = {(record.case_identity, record.replicate_index): record for record in records}
    expected_tasks = {
        (
            batcher_case_identity(
                case,
                corpus_tokens=config.corpus_tokens,
                seed=config.seed,
            ),
            replicate,
        )
        for case in config.cases
        for replicate in range(config.replicates)
    }
    observed: set[tuple[str, int]] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, path, f"execution entry {index}")
        if frozenset(entry) != _EXECUTION_KEYS or entry["execution_index"] != index:
            raise _invalid(path, f"execution entry {index} has an invalid field set")
        identity = _sha256(entry["case_identity"], path, "execution case_identity")
        replicate = _integer(entry["replicate_index"], path, "execution replicate_index")
        task = (identity, replicate)
        record = records_by_task.get(task)
        if (
            record is None
            or entry["task_id"] != f"{identity}:{replicate}"
            or entry["case_name"] != record.case_name
            or entry["worker_pid"] != record.worker_pid
            or entry["status"] != record.status
        ):
            raise _invalid(path, "execution_order.json disagrees with raw replicates")
        observed.add(task)
    if observed != expected_tasks or len(entries) != len(expected_tasks):
        raise _invalid(path, "execution_order.json task set differs from resolved config")


def load_batcher_benchmark_run(  # noqa: C901, PLR0912, PLR0915
    manifest_path: Path,
) -> LoadedBatcherBenchmarkRun:
    """Load a complete batch-only package and recompute every summary from committed raw data."""
    try:
        manifest_content = manifest_path.read_bytes()
    except OSError as error:
        raise _invalid(manifest_path, "cannot read run_manifest.json") from error
    document = _mapping(
        _json(manifest_content, manifest_path, "run_manifest.json"),
        manifest_path,
        "run_manifest.json",
    )
    if frozenset(document) != _MANIFEST_KEYS:
        raise _invalid(manifest_path, "run manifest has an invalid field set")
    if (
        document["schema_version"] != _SCHEMA_VERSION
        or document["methodology_version"] != _METHODOLOGY_VERSION
    ):
        raise _invalid(manifest_path, "run manifest has an unsupported schema")
    status = document["status"]
    run_id = document["run_id"]
    if status not in {"complete", "partial"} or not isinstance(run_id, str) or not run_id:
        raise _invalid(manifest_path, "run manifest identity is invalid")
    if manifest_path.parent.name != run_id:
        raise _invalid(manifest_path, "run_id does not match the package directory")
    contents = _artifact_contents(manifest_path, document)
    try:
        config = load_resolved_batcher_benchmark_config(
            contents["resolved_config.yaml"],
            manifest_path.parent / "resolved_config.yaml",
        )
    except ValueError as error:
        raise _invalid(manifest_path, "resolved_config.yaml is invalid") from error
    config_sha256 = _sha256(document["config_sha256"], manifest_path, "config_sha256")
    if batcher_config_sha256(config) != config_sha256:
        raise _invalid(manifest_path, "resolved config differs from config_sha256")
    if batcher_config_document(config)["experiment_name"] != document["experiment_name"]:
        raise _invalid(manifest_path, "experiment_name differs from resolved config")
    records = _parse_raw_records(contents["raw_replicates.jsonl"], manifest_path, config)
    expected_tasks = len(config.cases) * config.replicates
    successful = sum(record.status == "ok" for record in records)
    failed = len(records) - successful
    if (
        document["expected_task_count"] != expected_tasks
        or document["raw_replicate_count"] != len(records)
        or document["successful_task_count"] != successful
        or document["failed_task_count"] != failed
    ):
        raise _invalid(manifest_path, "manifest task counts disagree with raw replicates")
    if status == "complete" and (len(records) != expected_tasks or failed != 0):
        raise _invalid(manifest_path, "complete manifest has incomplete raw evidence")
    recomputed = _summarize(config, records)
    summary_csv = _parse_summary_csv(contents["summary.csv"], manifest_path)
    if summary_csv != recomputed:
        raise _invalid(manifest_path, "summary.csv disagrees with raw replicate statistics")
    raw_summaries = document["summaries"]
    if not isinstance(raw_summaries, list):
        raise _invalid(manifest_path, "manifest summaries must be a list")
    summary_documents = [
        _mapping(item, manifest_path, "manifest summary")
        for item in cast("list[object]", raw_summaries)
    ]
    if summary_documents != [asdict(summary) for summary in recomputed]:
        raise _invalid(manifest_path, "manifest summaries disagree with raw replicate statistics")
    expected_identities = {
        (
            case.name,
            batcher_case_identity(
                case,
                corpus_tokens=config.corpus_tokens,
                seed=config.seed,
            ),
        )
        for case in config.cases
    }
    raw_case_identities = document["case_identities"]
    if not isinstance(raw_case_identities, list):
        raise _invalid(manifest_path, "case_identities must be a list")
    observed_identities = {
        (
            _mapping(item, manifest_path, "case identity").get("case_name"),
            _mapping(item, manifest_path, "case identity").get("case_identity"),
        )
        for item in cast("list[object]", raw_case_identities)
    }
    if observed_identities != expected_identities:
        raise _invalid(manifest_path, "manifest case identities differ from resolved config")
    _validate_execution_order(contents["execution_order.json"], manifest_path, config, records)
    run_environment = _load_environment(
        contents["environment.json"],
        manifest_path,
        run_id=run_id,
        status=cast("str", status),
        config_sha256=config_sha256,
        records=records,
    )
    return LoadedBatcherBenchmarkRun(
        manifest_path.resolve(),
        run_id,
        _git_sha(document["git_commit_sha"], manifest_path),
        _git_sha(document["evidence_generator_git_commit_sha"], manifest_path),
        config,
        config_sha256,
        run_environment,
        records,
        recomputed,
    )


def _relative_change(baseline: float | None, candidate: float | None) -> float | None:
    """Calculate one finite descriptive relative change."""
    if baseline is None or candidate is None or baseline <= 0.0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def compare_batcher_benchmarks(  # noqa: C901
    baseline_manifest_path: Path,
    candidate_manifest_path: Path,
    policy: ComparisonPolicy,
) -> BatcherBenchmarkComparison:
    """Compare two strict runs after recomputing both under one authoritative policy."""
    baseline = load_batcher_benchmark_run(baseline_manifest_path)
    candidate = load_batcher_benchmark_run(candidate_manifest_path)
    baseline_summaries = {
        summary.case_identity: summary
        for summary in _summarize(
            baseline.config,
            baseline.records,
            minimum_replicates=policy.minimum_successful_replicates,
            max_cv_percent=policy.max_cv_percent,
        )
    }
    candidate_summaries = {
        summary.case_identity: summary
        for summary in _summarize(
            candidate.config,
            candidate.records,
            minimum_replicates=policy.minimum_successful_replicates,
            max_cv_percent=policy.max_cv_percent,
        )
    }
    environment_mismatches = tuple(
        field
        for field in _COMPARISON_ENVIRONMENT_FIELDS
        if baseline.run_environment[field] != candidate.run_environment[field]
    )
    reasons: list[str] = []
    if set(baseline_summaries) != set(candidate_summaries):
        reasons.append("baseline and candidate case identity sets differ")
    if environment_mismatches:
        reasons.append("baseline and candidate environments differ")
    cases: list[BatcherCaseComparison] = []
    any_regression = False
    for identity in sorted(set(baseline_summaries) & set(candidate_summaries)):
        baseline_summary = baseline_summaries[identity]
        candidate_summary = candidate_summaries[identity]
        case_reasons: list[str] = []
        if reasons:
            case_reasons.append("run-level compatibility requirements are not met")
        if baseline_summary.stability != "stable":
            case_reasons.append("baseline case is unstable")
        if candidate_summary.stability != "stable":
            case_reasons.append("candidate case is unstable")
        if (
            policy.require_equal_replicate_count
            and baseline_summary.replicate_count != candidate_summary.replicate_count
        ):
            case_reasons.append("baseline and candidate replicate counts differ")
        batch_change = _relative_change(
            baseline_summary.median_batch_time_ms,
            candidate_summary.median_batch_time_ms,
        )
        throughput_change = _relative_change(
            baseline_summary.median_tokens_per_second,
            candidate_summary.median_tokens_per_second,
        )
        if batch_change is None:
            case_reasons.append("case has no comparable median batch time")
        regressed = (
            None
            if case_reasons or batch_change is None
            else batch_change > policy.regression_threshold_percent
        )
        any_regression = any_regression or regressed is True
        peak_change = (
            None
            if baseline_summary.max_worker_lifetime_peak_rss_mib is None
            or candidate_summary.max_worker_lifetime_peak_rss_mib is None
            else candidate_summary.max_worker_lifetime_peak_rss_mib
            - baseline_summary.max_worker_lifetime_peak_rss_mib
        )
        cases.append(
            BatcherCaseComparison(
                identity,
                baseline_summary.case_name,
                baseline_summary.median_batch_time_ms,
                candidate_summary.median_batch_time_ms,
                batch_change,
                throughput_change,
                peak_change,
                baseline_summary.coefficient_of_variation_percent,
                candidate_summary.coefficient_of_variation_percent,
                baseline_summary.stability,
                candidate_summary.stability,
                tuple(case_reasons),
                regressed,
            )
        )
    verdict: Literal["pass", "fail", "not_comparable"]
    if reasons or any(case.reasons for case in cases):
        verdict = "not_comparable"
    elif any_regression:
        verdict = "fail"
    else:
        verdict = "pass"
    return BatcherBenchmarkComparison(
        baseline_manifest_path.resolve(),
        candidate_manifest_path.resolve(),
        baseline.run_id,
        candidate.run_id,
        baseline.git_commit_sha,
        candidate.git_commit_sha,
        policy.sha256,
        comparison_policy_document(policy),
        environment_mismatches,
        tuple(reasons),
        tuple(cases),
        verdict,
    )


def write_batcher_comparison(
    comparison: BatcherBenchmarkComparison,
    output_path: Path,
) -> None:
    """Write one deterministic machine-readable batch-only comparison."""
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "baseline_manifest_path": str(comparison.baseline_manifest_path),
        "candidate_manifest_path": str(comparison.candidate_manifest_path),
        "baseline_run_id": comparison.baseline_run_id,
        "candidate_run_id": comparison.candidate_run_id,
        "baseline_git_commit_sha": comparison.baseline_git_commit_sha,
        "candidate_git_commit_sha": comparison.candidate_git_commit_sha,
        "policy_sha256": comparison.policy_sha256,
        "comparison_policy": comparison.comparison_policy,
        "environment_mismatches": list(comparison.environment_mismatches),
        "reasons": list(comparison.reasons),
        "cases": [cast("dict[str, JsonValue]", asdict(case)) for case in comparison.cases],
        "verdict": comparison.verdict,
    }
    _ = output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
