"""Compare hash-bound Benchmark v2 evidence without making unsafe regression claims."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from io import StringIO
from typing import TYPE_CHECKING, Literal, cast

import yaml
from typing_extensions import override

from minigpt.benchmark_v2_report import load_run_manifest
from minigpt.benchmark_v2_statistics import BenchmarkV2Summary

if TYPE_CHECKING:
    from pathlib import Path

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
    summaries: tuple[BenchmarkV2Summary, ...]
    regression_threshold_percent: float


def _require_mapping(value: object, path: Path, context: str) -> dict[str, object]:
    """Require a JSON/YAML mapping with string keys before reading its contents."""
    if not isinstance(value, dict):
        raise InvalidComparisonInputError(path, f"{context} must be an object with string keys")
    mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        raise InvalidComparisonInputError(path, f"{context} must be an object with string keys")
    return cast("dict[str, object]", mapping)


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


def _load_run_environment(manifest_path: Path, manifest: RunManifest) -> dict[str, JsonValue]:
    """Load and validate the complete parent environment used for compatibility checks."""
    try:
        raw_document = cast(
            "object", json.loads(_read_bound_artifact(manifest_path, manifest, "environment.json"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidComparisonInputError(
            manifest_path, "environment.json is not valid UTF-8 JSON"
        ) from error
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
    return cast("dict[str, JsonValue]", environment)


def _load_summaries(  # noqa: C901
    manifest_path: Path, manifest: RunManifest
) -> tuple[BenchmarkV2Summary, ...]:
    """Load an exact summary schema and verify it covers the manifest case identity set once."""
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
    expected = {case["case_identity"] for case in manifest.case_identities}
    observed = {summary.case_identity for summary in summaries}
    if len(summaries) != len(observed) or observed != expected:
        raise InvalidComparisonInputError(
            manifest_path, "summary.csv case identities do not match manifest"
        )
    return tuple(sorted(summaries, key=lambda summary: summary.case_identity))


def _load_threshold(manifest_path: Path, manifest: RunManifest) -> float:
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
    threshold = config.get("regression_threshold_percent")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise InvalidComparisonInputError(
            manifest_path, "config has no numeric regression threshold"
        )
    value = float(threshold)
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidComparisonInputError(
            manifest_path, "config regression threshold is not positive and finite"
        )
    return value


def _load_input(manifest_path: Path) -> _ComparisonInput:
    """Compose strict loader boundaries before any comparison calculation occurs."""
    manifest = load_run_manifest(manifest_path)
    return _ComparisonInput(
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        run_environment=_load_run_environment(manifest_path, manifest),
        summaries=_load_summaries(manifest_path, manifest),
        regression_threshold_percent=_load_threshold(manifest_path, manifest),
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
    equal_to_threshold = math.isclose(change, threshold_percent, rel_tol=1e-12, abs_tol=1e-12)
    return StepTimeComparison(
        step_time_change_percent=change,
        regressed=change > threshold_percent and not equal_to_threshold,
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


def compare_runs(baseline: Path, candidate: Path) -> BenchmarkComparison:  # noqa: C901
    """Compare two finalized runs with conservative eligibility and immutable source evidence."""
    baseline_input = _load_input(baseline)
    candidate_input = _load_input(candidate)
    environment_mismatches = _environment_mismatches(baseline_input, candidate_input)
    baseline_identities = {summary.case_identity for summary in baseline_input.summaries}
    candidate_identities = {summary.case_identity for summary in candidate_input.summaries}
    missing = tuple(sorted(baseline_identities - candidate_identities))
    extra = tuple(sorted(candidate_identities - baseline_identities))
    reasons: list[str] = []
    if baseline_input.manifest.status != "complete":
        reasons.append(f"baseline run status is {baseline_input.manifest.status}")
    if candidate_input.manifest.status != "complete":
        reasons.append(f"candidate run status is {candidate_input.manifest.status}")
    if baseline_input.manifest.config_sha256 != candidate_input.manifest.config_sha256:
        reasons.append("config_sha256 differs")
    if missing or extra:
        reasons.append("case identity sets do not align")
    reasons.extend(f"environment differs: {mismatch.field}" for mismatch in environment_mismatches)
    globally_comparable = not reasons
    baseline_by_identity = {summary.case_identity: summary for summary in baseline_input.summaries}
    candidate_by_identity = {
        summary.case_identity: summary for summary in candidate_input.summaries
    }
    cases: list[CaseComparison] = []
    for identity in sorted(baseline_identities & candidate_identities):
        baseline_summary = baseline_by_identity[identity]
        candidate_summary = candidate_by_identity[identity]
        case_reasons = _case_reasons(
            baseline_summary, candidate_summary, globally_comparable=globally_comparable
        )
        step_change = _descriptive_change(
            baseline_summary.median_step_time_ms, candidate_summary.median_step_time_ms
        )
        throughput_change = _descriptive_change(
            baseline_summary.median_tokens_per_second, candidate_summary.median_tokens_per_second
        )
        regressed: bool | None = None
        if not case_reasons and step_change is not None:
            regressed = compare_step_times(
                cast("float", baseline_summary.median_step_time_ms),
                cast("float", candidate_summary.median_step_time_ms),
                threshold_percent=candidate_input.regression_threshold_percent,
            ).regressed
        cases.append(
            CaseComparison(
                case_identity=identity,
                baseline_case_name=baseline_summary.case_name,
                candidate_case_name=candidate_summary.case_name,
                baseline_median_step_time_ms=baseline_summary.median_step_time_ms,
                candidate_median_step_time_ms=candidate_summary.median_step_time_ms,
                baseline_median_tokens_per_second=baseline_summary.median_tokens_per_second,
                candidate_median_tokens_per_second=candidate_summary.median_tokens_per_second,
                step_time_change_percent=step_change,
                throughput_change_percent=throughput_change,
                reasons=case_reasons,
                regressed=regressed,
            )
        )
    for case in cases:
        for reason in case.reasons:
            if (
                reason != "run-level compatibility requirements are not met"
                and reason not in reasons
            ):
                reasons.append(reason)
    if reasons or any(case.regressed is None for case in cases):
        verdict: ComparisonVerdict = "not_comparable"
    elif any(case.regressed for case in cases):
        verdict = "fail"
    else:
        verdict = "pass"
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
        case_comparisons=tuple(cases),
        verdict=verdict,
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
    _atomic_write_new(markdown_path, _comparison_markdown(comparison).encode("utf-8"))
    return ComparisonArtifacts(json_path=json_path, markdown_path=markdown_path)
