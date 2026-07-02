"""Measure CPU training throughput across a reproducible configuration matrix."""

import gc
import sys
from itertools import product
from statistics import median, pstdev
from time import perf_counter
from typing import TYPE_CHECKING

from minigpt.benchmark_report import write_benchmark_artifacts
from minigpt.benchmark_types import (
    BenchmarkAggregationError,
    BenchmarkArtifacts,
    BenchmarkCase,
    BenchmarkMeasurement,
    BenchmarkSummary,
)
from minigpt.benchmark_workload import TrainingStepWorkload
from minigpt.metrics import cpu_memory_mb

if TYPE_CHECKING:
    from minigpt.benchmark_config import BenchmarkConfig

EMPTY_MEASUREMENTS_REASON = "measurements must not be empty"
MIXED_CASES_REASON = "measurements must belong to one benchmark case"

__all__ = (
    "BenchmarkArtifacts",
    "BenchmarkCase",
    "BenchmarkMeasurement",
    "BenchmarkSummary",
    "expand_cases",
    "run_benchmark",
    "summarize_measurements",
)


def expand_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    """Expand the configured Cartesian product in deterministic order."""
    cases: list[BenchmarkCase] = []
    for model, thread_count, block_size, batch_size in product(
        config.model_sizes,
        config.thread_counts,
        config.block_sizes,
        config.batch_sizes,
    ):
        cases.append(
            BenchmarkCase(
                model_size=model.name,
                n_layer=model.n_layer,
                n_head=model.n_head,
                n_embd=model.n_embd,
                thread_count=thread_count,
                block_size=block_size,
                batch_size=batch_size,
            )
        )
    return cases


def summarize_measurements(
    measurements: list[BenchmarkMeasurement],
) -> BenchmarkSummary:
    """Compute robust central tendency and repeat variability."""
    if not measurements:
        raise BenchmarkAggregationError(EMPTY_MEASUREMENTS_REASON)
    case = measurements[0].case
    if any(measurement.case != case for measurement in measurements):
        raise BenchmarkAggregationError(MIXED_CASES_REASON)
    step_times = [measurement.step_time_ms for measurement in measurements]
    throughputs = [measurement.tokens_per_sec for measurement in measurements]
    memory_values = [measurement.cpu_memory_mb for measurement in measurements]
    median_step = median(step_times)
    mean_step = sum(step_times) / len(step_times)
    stddev_step = pstdev(step_times)
    mad_step = median([abs(value - median_step) for value in step_times])
    cv_percent = 0.0 if mean_step == 0.0 else stddev_step / mean_step * 100
    return BenchmarkSummary(
        case=case,
        parameter_count=measurements[0].parameter_count,
        repeat_count=len(measurements),
        median_step_time_ms=median_step,
        min_step_time_ms=min(step_times),
        max_step_time_ms=max(step_times),
        step_time_stddev_ms=stddev_step,
        step_time_mad_ms=mad_step,
        step_time_cv_percent=cv_percent,
        median_tokens_per_sec=median(throughputs),
        median_cpu_memory_mb=median(memory_values),
    )


def _measure_case(
    config: BenchmarkConfig,
    case: BenchmarkCase,
) -> list[BenchmarkMeasurement]:
    workload = TrainingStepWorkload(
        case,
        seed=config.seed,
        vocab_size=config.vocab_size,
    )
    for _ in range(config.warmup_steps):
        workload.step()

    measurements: list[BenchmarkMeasurement] = []
    for repeat_index in range(config.repeats):
        _ = gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            started = perf_counter()
            for _ in range(config.measurement_steps):
                workload.step()
            elapsed_seconds = perf_counter() - started
        finally:
            if gc_was_enabled:
                gc.enable()
        step_time_ms = elapsed_seconds * 1_000 / config.measurement_steps
        tokens_per_sec = workload.tokens_per_step / (step_time_ms / 1_000)
        measurements.append(
            BenchmarkMeasurement(
                case=case,
                repeat_index=repeat_index,
                step_time_ms=step_time_ms,
                tokens_per_sec=tokens_per_sec,
                cpu_memory_mb=cpu_memory_mb(),
                parameter_count=workload.parameter_count,
            )
        )
    return measurements


def run_benchmark(config: BenchmarkConfig) -> BenchmarkArtifacts:
    """Execute every CPU case and write raw and summarized artifacts."""
    all_measurements: list[BenchmarkMeasurement] = []
    summaries: list[BenchmarkSummary] = []
    cases = expand_cases(config)
    for index, case in enumerate(cases, start=1):
        _ = sys.stdout.write(f"[{index}/{len(cases)}] benchmark {case.label}\n")
        measurements = _measure_case(config, case)
        all_measurements.extend(measurements)
        summaries.append(summarize_measurements(measurements))
    return write_benchmark_artifacts(config, all_measurements, summaries)
