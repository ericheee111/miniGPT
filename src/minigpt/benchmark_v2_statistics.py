"""Pure per-case statistics and stability classification for Benchmark v2."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from minigpt.benchmark_v2_environment import PEAK_RSS_SCOPE, PeakRssScope

if TYPE_CHECKING:
    from collections.abc import Iterable

    from minigpt.benchmark_v2_config import JsonValue

Stability = Literal["insufficient_samples", "unstable", "stable"]


class _RawReplicateLike(Protocol):
    """Describe the raw record fields used by pure statistics without orchestration coupling."""

    @property
    def status(self) -> Literal["ok", "error"]: ...

    @property
    def case_identity(self) -> str: ...

    @property
    def case_name(self) -> str: ...

    @property
    def worker_response(self) -> dict[str, JsonValue] | None: ...


@dataclass(frozen=True, slots=True)
class BenchmarkV2Summary:
    """Describe raw replicate counts, unfiltered statistics, and their stability."""

    case_identity: str
    case_name: str
    replicate_count: int
    success_count: int
    failure_count: int
    median_step_time_ms: float | None
    min_step_time_ms: float | None
    max_step_time_ms: float | None
    population_stddev_step_time_ms: float | None
    median_absolute_deviation_step_time_ms: float | None
    coefficient_of_variation_percent: float | None
    median_tokens_per_second: float | None
    median_final_rss_mib: float | None
    peak_rss_scope: PeakRssScope
    max_peak_rss_mib: float | None
    stability: Stability


def _finite_number(response: dict[str, JsonValue], field: str) -> float:
    """Read one worker metric that was already protocol-validated by orchestration."""
    value = response.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"successful worker response has no numeric {field}"
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"successful worker response has non-finite {field}"
        raise ValueError(msg)
    return number


def _successful_response(record: _RawReplicateLike) -> dict[str, JsonValue] | None:
    """Return the complete response only for a successful raw replicate."""
    if record.status != "ok":
        return None
    if record.worker_response is None:
        msg = "successful raw replicate has no worker response"
        raise ValueError(msg)
    return record.worker_response


def summarize_replicates(
    raw_replicates: Iterable[_RawReplicateLike],
    *,
    minimum_replicates: int,
    max_cv_percent: float,
) -> BenchmarkV2Summary:
    """Summarize one case without deleting failed records or numeric outliers.

    Each successful worker contributes its single aggregate step-time measurement. Failed
    workers remain in the raw replicate and failure counts but have no numeric value.
    """
    records = tuple(raw_replicates)
    if not records:
        msg = "cannot summarize an empty replicate collection"
        raise ValueError(msg)
    if minimum_replicates <= 0:
        msg = "minimum_replicates must be positive"
        raise ValueError(msg)
    if not math.isfinite(max_cv_percent) or max_cv_percent <= 0.0:
        msg = "max_cv_percent must be positive and finite"
        raise ValueError(msg)
    identities = {(record.case_identity, record.case_name) for record in records}
    if len(identities) != 1:
        msg = "replicate collection must contain exactly one case"
        raise ValueError(msg)
    case_identity, case_name = identities.pop()
    responses = tuple(
        response for record in records if (response := _successful_response(record)) is not None
    )
    step_times = tuple(_finite_number(response, "step_time_ms") for response in responses)
    success_count = len(step_times)
    if success_count < minimum_replicates:
        stability: Stability = "insufficient_samples"
    else:
        mean_step_time = statistics.fmean(step_times)
        population_stddev = statistics.pstdev(step_times)
        coefficient_of_variation = population_stddev / mean_step_time * 100.0
        stability = "unstable" if coefficient_of_variation > max_cv_percent else "stable"
    if not responses:
        return BenchmarkV2Summary(
            case_identity=case_identity,
            case_name=case_name,
            replicate_count=len(records),
            success_count=0,
            failure_count=len(records),
            median_step_time_ms=None,
            min_step_time_ms=None,
            max_step_time_ms=None,
            population_stddev_step_time_ms=None,
            median_absolute_deviation_step_time_ms=None,
            coefficient_of_variation_percent=None,
            median_tokens_per_second=None,
            median_final_rss_mib=None,
            peak_rss_scope=PEAK_RSS_SCOPE,
            max_peak_rss_mib=None,
            stability=stability,
        )
    median_step_time = statistics.median(step_times)
    median_absolute_deviation = statistics.median(
        abs(value - median_step_time) for value in step_times
    )
    population_stddev = statistics.pstdev(step_times)
    coefficient_of_variation = population_stddev / statistics.fmean(step_times) * 100.0
    return BenchmarkV2Summary(
        case_identity=case_identity,
        case_name=case_name,
        replicate_count=len(records),
        success_count=success_count,
        failure_count=len(records) - success_count,
        median_step_time_ms=median_step_time,
        min_step_time_ms=min(step_times),
        max_step_time_ms=max(step_times),
        population_stddev_step_time_ms=population_stddev,
        median_absolute_deviation_step_time_ms=median_absolute_deviation,
        coefficient_of_variation_percent=coefficient_of_variation,
        median_tokens_per_second=statistics.median(
            _finite_number(response, "tokens_per_second") for response in responses
        ),
        median_final_rss_mib=statistics.median(
            _finite_number(response, "final_rss_mib") for response in responses
        ),
        peak_rss_scope=PEAK_RSS_SCOPE,
        max_peak_rss_mib=max(_finite_number(response, "peak_rss_mib") for response in responses),
        stability=stability,
    )
