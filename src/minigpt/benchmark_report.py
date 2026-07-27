"""Persist raw CPU benchmark data and a reader-facing Markdown summary."""

from __future__ import annotations

import csv
import platform
import sys
from typing import TYPE_CHECKING

import psutil
import torch

from minigpt.benchmark_types import (
    BenchmarkArtifacts,
    BenchmarkMeasurement,
    BenchmarkSummary,
)

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.benchmark_config import BenchmarkConfig


def _case_fields(measurement: BenchmarkMeasurement) -> list[str | int | float]:
    case = measurement.case
    return [
        case.label,
        case.model_size,
        case.thread_count,
        case.block_size,
        case.batch_size,
        case.n_layer,
        case.n_head,
        case.n_embd,
        measurement.parameter_count,
        measurement.repeat_index,
        measurement.step_time_ms,
        measurement.tokens_per_sec,
        measurement.cpu_memory_mb,
    ]


def _write_raw_csv(path: Path, measurements: list[BenchmarkMeasurement]) -> None:
    headers = [
        "case",
        "model_size",
        "thread_count",
        "block_size",
        "batch_size",
        "n_layer",
        "n_head",
        "n_embd",
        "parameter_count",
        "repeat_index",
        "step_time_ms",
        "tokens_per_sec",
        "cpu_memory_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for measurement in measurements:
            writer.writerow(_case_fields(measurement))


def _write_summary_csv(path: Path, summaries: list[BenchmarkSummary]) -> None:
    headers = [
        "case",
        "model_size",
        "thread_count",
        "block_size",
        "batch_size",
        "parameter_count",
        "repeat_count",
        "median_step_time_ms",
        "min_step_time_ms",
        "max_step_time_ms",
        "step_time_stddev_ms",
        "step_time_mad_ms",
        "step_time_cv_percent",
        "median_tokens_per_sec",
        "median_cpu_memory_mb",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for summary in summaries:
            case = summary.case
            writer.writerow(
                [
                    case.label,
                    case.model_size,
                    case.thread_count,
                    case.block_size,
                    case.batch_size,
                    summary.parameter_count,
                    summary.repeat_count,
                    summary.median_step_time_ms,
                    summary.min_step_time_ms,
                    summary.max_step_time_ms,
                    summary.step_time_stddev_ms,
                    summary.step_time_mad_ms,
                    summary.step_time_cv_percent,
                    summary.median_tokens_per_sec,
                    summary.median_cpu_memory_mb,
                ]
            )


def _write_markdown(
    path: Path,
    config: BenchmarkConfig,
    summaries: list[BenchmarkSummary],
) -> None:
    best = max(summaries, key=lambda item: item.median_tokens_per_sec)
    lines = [
        "# CPU Training Benchmark",
        "",
        "## Methodology",
        "",
        f"- Warmup steps per case: {config.warmup_steps}",
        f"- Timed steps per repeat: {config.measurement_steps}",
        f"- Repeats per case: {config.repeats}",
        "- Timed region: data preparation + forward/backward + AdamW step.",
        "- Excluded: model construction, warmup, garbage collection, logging, and file I/O.",
        "- Summary: median, population standard deviation, MAD, and coefficient of variation.",
        "",
        "## Environment",
        "",
        f"- Python: {sys.version.split()[0]}",
        f"- PyTorch: {torch.__version__}",
        f"- Platform: {platform.platform()}",
        f"- Physical cores: {psutil.cpu_count(logical=False)}",
        f"- Logical cores: {psutil.cpu_count(logical=True)}",
        "",
        "## Results",
        "",
        "| Case | Params | Median ms | Tokens/s | CV % | RSS MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {summary.case.label} | {summary.parameter_count} | "
            f"{summary.median_step_time_ms:.3f} | "
            f"{summary.median_tokens_per_sec:.1f} | "
            f"{summary.step_time_cv_percent:.2f} | "
            f"{summary.median_cpu_memory_mb:.1f} |"
        )
        for summary in summaries
    )
    lines.extend(
        [
            "",
            "## Observed Best Throughput",
            "",
            f"`{best.case.label}` reached median {best.median_tokens_per_sec:.1f} tokens/s.",
            "",
            "Raw repeat-level data is retained in `benchmark_raw.csv`.",
        ]
    )
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_artifacts(
    config: BenchmarkConfig,
    measurements: list[BenchmarkMeasurement],
    summaries: list[BenchmarkSummary],
) -> BenchmarkArtifacts:
    """Write raw CSV, summary CSV, and a reproducible Markdown report."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = config.output_dir / "benchmark_raw.csv"
    summary_csv = config.output_dir / "benchmark_summary.csv"
    report_markdown = config.output_dir / "benchmark_report.md"
    _write_raw_csv(raw_csv, measurements)
    _write_summary_csv(summary_csv, summaries)
    _write_markdown(report_markdown, config, summaries)
    return BenchmarkArtifacts(raw_csv, summary_csv, report_markdown)
