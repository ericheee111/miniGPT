"""Orchestrate fresh-process Stage 11A serving executor comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
import yaml

from minigpt.serving_benchmark_config import (
    JsonValue,
    ServingBenchmarkConfig,
    load_serving_benchmark_config,
    resolved_config_document,
    resolved_config_sha256,
)
from minigpt.serving_simulator import SimulatorExecutor

if TYPE_CHECKING:
    from pathlib import Path

BenchmarkDocument: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ServingBenchmarkRun:
    """Return one immutable benchmark output location and strict verdict."""

    output_dir: Path
    summary_path: Path
    strict_verdict: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: BenchmarkDocument) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _environment(config: ServingBenchmarkConfig, source_commit: str) -> BenchmarkDocument:
    return cast(
        "BenchmarkDocument",
        {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_num_threads": config.torch_num_threads,
            "torch_num_interop_threads": config.torch_num_interop_threads,
            "source_commit": source_commit,
            "resolved_config_sha256": resolved_config_sha256(config),
            "timer": "time.perf_counter canonical fresh-process wall clock",
            "profiler_timings_are_canonical": False,
        },
    )


def _median_number(iterations: list[BenchmarkDocument], key: str) -> float:
    values: list[float] = []
    for iteration in iterations:
        value = iteration[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"worker iteration field {key} must be numeric"
            raise TypeError(msg)
        number = float(value)
        if not math.isfinite(number):
            msg = f"worker iteration field {key} must be finite"
            raise ValueError(msg)
        values.append(number)
    return statistics.median(values)


def _worker_record(  # noqa: PLR0913
    *,
    config_path: Path,
    config: ServingBenchmarkConfig,
    scenario: str,
    executor: SimulatorExecutor,
    replicate: int,
    execution_index: int,
    worker_dir: Path,
) -> BenchmarkDocument:
    output = worker_dir / f"{execution_index:04d}-{scenario}-{executor.value}.json"
    command = [
        sys.executable,
        "-m",
        "minigpt.serving_benchmark_worker",
        "--config",
        str(config_path.resolve()),
        "--scenario",
        scenario,
        "--executor",
        executor.value,
        "--output",
        str(output.resolve()),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.worker_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "execution_index": execution_index,
            "replicate": replicate,
            "scenario": scenario,
            "executor": executor.value,
            "status": "error",
            "error": f"worker timeout after {error.timeout} seconds",
            "worker_wall_seconds": time.perf_counter() - started,
        }
    if completed.returncode != 0 or not output.is_file():
        return {
            "execution_index": execution_index,
            "replicate": replicate,
            "scenario": scenario,
            "executor": executor.value,
            "status": "error",
            "error": f"worker exit {completed.returncode}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "worker_wall_seconds": time.perf_counter() - started,
        }
    raw = cast("object", json.loads(output.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        msg = "worker output must be a JSON object"
        raise TypeError(msg)
    response = cast("BenchmarkDocument", raw)
    iterations_value = response.get("iterations")
    if not isinstance(iterations_value, list) or not iterations_value:
        msg = "worker output must contain non-empty iterations"
        raise ValueError(msg)
    iterations = cast("list[BenchmarkDocument]", iterations_value)
    correctness_hashes = {cast("str", iteration["correctness_sha256"]) for iteration in iterations}
    return cast(
        "BenchmarkDocument",
        {
            "execution_index": execution_index,
            "replicate": replicate,
            "scenario": scenario,
            "executor": executor.value,
            "status": "ok",
            "worker_wall_seconds": time.perf_counter() - started,
            "elapsed_seconds": _median_number(iterations, "elapsed_seconds"),
            "request_throughput_per_second": _median_number(
                iterations, "request_throughput_per_second"
            ),
            "token_throughput_per_second": _median_number(
                iterations, "token_throughput_per_second"
            ),
            "median_ttft_seconds": _median_number(iterations, "median_ttft_seconds"),
            "median_tpot_seconds": _median_number(iterations, "median_tpot_seconds"),
            "median_e2e_seconds": _median_number(iterations, "median_e2e_seconds"),
            "median_queue_time_seconds": _median_number(iterations, "median_queue_time_seconds"),
            "median_prefill_latency_seconds": _median_number(
                iterations, "median_prefill_latency_seconds"
            ),
            "average_decode_batch_size": _median_number(iterations, "average_decode_batch_size"),
            "max_decode_batch_size": _median_number(iterations, "max_decode_batch_size"),
            "padded_cache_tokens": _median_number(iterations, "padded_cache_tokens"),
            "useful_cache_tokens": _median_number(iterations, "useful_cache_tokens"),
            "padding_waste_ratio": _median_number(iterations, "padding_waste_ratio"),
            "executor_time_seconds": _median_number(iterations, "executor_time_seconds"),
            "model_execution_time_seconds": _median_number(
                iterations, "model_execution_time_seconds"
            ),
            "batch_assembly_scatter_time_seconds": _median_number(
                iterations, "batch_assembly_scatter_time_seconds"
            ),
            "average_prefill_batch_size": _median_number(iterations, "average_prefill_batch_size"),
            "max_prefill_batch_size": _median_number(iterations, "max_prefill_batch_size"),
            "padded_prompt_tokens": _median_number(iterations, "padded_prompt_tokens"),
            "useful_prompt_tokens": _median_number(iterations, "useful_prompt_tokens"),
            "prompt_padding_waste_ratio": _median_number(iterations, "prompt_padding_waste_ratio"),
            "prefill_executor_time_seconds": _median_number(
                iterations, "prefill_executor_time_seconds"
            ),
            "prefill_model_execution_time_seconds": _median_number(
                iterations, "prefill_model_execution_time_seconds"
            ),
            "prefill_batch_assembly_scatter_time_seconds": _median_number(
                iterations, "prefill_batch_assembly_scatter_time_seconds"
            ),
            "worker_peak_rss_bytes": _median_number(iterations, "worker_peak_rss_bytes"),
            "correctness_sha256": (
                next(iter(correctness_hashes))
                if len(correctness_hashes) == 1
                else "iteration-mismatch"
            ),
            "iterations": iterations,
        },
    )


def _numeric(record: BenchmarkDocument, key: str) -> float:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"raw replicate {key} must be numeric"
        raise TypeError(msg)
    return float(value)


def _executor_summary(
    records: list[BenchmarkDocument],
    *,
    minimum_replicates: int,
    max_cv_percent: float,
) -> BenchmarkDocument:
    successful = [record for record in records if record["status"] == "ok"]
    elapsed = [_numeric(record, "elapsed_seconds") for record in successful]
    if elapsed:
        median_elapsed = statistics.median(elapsed)
        mad = statistics.median(abs(value - median_elapsed) for value in elapsed)
        cv = statistics.pstdev(elapsed) / statistics.fmean(elapsed) * 100.0
    else:
        median_elapsed = None
        mad = None
        cv = None
    stable = len(successful) >= minimum_replicates and cv is not None and cv <= max_cv_percent
    metric_keys = (
        "request_throughput_per_second",
        "token_throughput_per_second",
        "median_ttft_seconds",
        "median_tpot_seconds",
        "median_e2e_seconds",
        "median_queue_time_seconds",
        "median_prefill_latency_seconds",
        "average_decode_batch_size",
        "max_decode_batch_size",
        "padded_cache_tokens",
        "useful_cache_tokens",
        "padding_waste_ratio",
        "executor_time_seconds",
        "model_execution_time_seconds",
        "batch_assembly_scatter_time_seconds",
        "average_prefill_batch_size",
        "max_prefill_batch_size",
        "padded_prompt_tokens",
        "useful_prompt_tokens",
        "prompt_padding_waste_ratio",
        "prefill_executor_time_seconds",
        "prefill_model_execution_time_seconds",
        "prefill_batch_assembly_scatter_time_seconds",
        "worker_peak_rss_bytes",
    )
    medians: BenchmarkDocument = {
        f"median_{key}": (
            statistics.median(_numeric(record, key) for record in successful)
            if successful and all(key in record for record in successful)
            else None
        )
        for key in metric_keys
    }
    correctness_hashes = {cast("str", record["correctness_sha256"]) for record in successful}
    return {
        "replicate_count": len(records),
        "success_count": len(successful),
        "failure_count": len(records) - len(successful),
        "median_elapsed_seconds": median_elapsed,
        "median_absolute_deviation_seconds": mad,
        "coefficient_of_variation_percent": cv,
        "stability": "stable" if stable else "unstable_or_insufficient",
        "correctness_sha256": (
            next(iter(correctness_hashes)) if len(correctness_hashes) == 1 else None
        ),
        **medians,
    }


def _benchmark_executors(config: ServingBenchmarkConfig) -> tuple[SimulatorExecutor, ...]:
    if config.prefill is None:
        return (SimulatorExecutor.REFERENCE, SimulatorExecutor.CONTINUOUS_DECODE)
    return (
        SimulatorExecutor.REFERENCE,
        SimulatorExecutor.CONTINUOUS_DECODE,
        SimulatorExecutor.CONTINUOUS,
    )


def summarize_serving_records(
    records: list[BenchmarkDocument],
    config: ServingBenchmarkConfig,
) -> BenchmarkDocument:
    """Apply strict stability and correctness comparison without filtering outliers."""
    scenarios: list[JsonValue] = []
    overall_pass = True
    executors = _benchmark_executors(config)
    baseline_name = (
        SimulatorExecutor.REFERENCE
        if config.prefill is None
        else SimulatorExecutor.CONTINUOUS_DECODE
    )
    candidate_name = (
        SimulatorExecutor.CONTINUOUS_DECODE
        if config.prefill is None
        else SimulatorExecutor.CONTINUOUS
    )
    for scenario in config.scenarios:
        by_executor: dict[SimulatorExecutor, BenchmarkDocument] = {}
        for executor in executors:
            selected = [
                record
                for record in records
                if record["scenario"] == scenario.name and record["executor"] == executor.value
            ]
            by_executor[executor] = _executor_summary(
                selected,
                minimum_replicates=config.minimum_replicates,
                max_cv_percent=config.max_cv_percent,
            )
        baseline = by_executor[baseline_name]
        candidate = by_executor[candidate_name]
        correctness_hashes = {
            cast("str", summary["correctness_sha256"])
            for summary in by_executor.values()
            if summary["correctness_sha256"] is not None
        }
        correctness_matches = len(correctness_hashes) == 1 and all(
            summary["correctness_sha256"] is not None for summary in by_executor.values()
        )
        stable = baseline["stability"] == "stable" and candidate["stability"] == "stable"
        strict_verdict = "pass" if stable and correctness_matches else "not_comparable"
        overall_pass = overall_pass and strict_verdict == "pass"
        baseline_elapsed = baseline["median_elapsed_seconds"]
        candidate_elapsed = candidate["median_elapsed_seconds"]
        speedup: float | None = None
        conclusion = "not_comparable"
        if (
            strict_verdict == "pass"
            and isinstance(baseline_elapsed, (int, float))
            and isinstance(candidate_elapsed, (int, float))
            and float(candidate_elapsed) > 0.0
        ):
            speedup = float(baseline_elapsed) / float(candidate_elapsed)
            conclusion = "improved" if speedup > 1.0 else "no_improvement"
        scenario_document: BenchmarkDocument = {
            "scenario": scenario.name,
            "strict_verdict": strict_verdict,
            "correctness_matches": correctness_matches,
            "comparison_baseline": baseline_name.value,
            "comparison_candidate": candidate_name.value,
            "speedup_baseline_over_candidate": speedup,
            "speedup_reference_over_continuous": (speedup if config.prefill is None else None),
            "speedup_continuous_decode_over_continuous": (
                speedup if config.prefill is not None else None
            ),
            "performance_conclusion": conclusion,
        }
        for executor, executor_summary in by_executor.items():
            scenario_document[executor.value] = executor_summary
        scenarios.append(scenario_document)
    return {
        "schema_version": 1,
        "strict_verdict": "pass" if overall_pass else "not_comparable",
        "claim_policy": (
            "performance improvement may be claimed only for a scenario with strict_verdict=pass "
            "and performance_conclusion=improved"
        ),
        "scenarios": scenarios,
    }


def _summary_markdown(summary: BenchmarkDocument) -> str:
    scenarios = cast("list[BenchmarkDocument]", summary["scenarios"])
    header = "| Scenario | Verdict | Conclusion | Baseline | Candidate | Speedup | Candidate CV | Prefill batch | Prompt waste |"  # noqa: E501
    lines = [
        "# Fresh-process serving benchmark",
        "",
        f"Overall strict verdict: `{summary['strict_verdict']}`.",
        "",
        "Profiler timings are excluded from this canonical comparison.",
        "",
        header,
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        candidate_name = cast("str", scenario["comparison_candidate"])
        candidate = cast("BenchmarkDocument", scenario[candidate_name])
        speedup = scenario["speedup_baseline_over_candidate"]
        speedup_text = "" if speedup is None else f"{float(cast('float', speedup)):.3f}x"
        row = (
            "| "
            + " | ".join(
                str(value)
                for value in (
                    scenario["scenario"],
                    scenario["strict_verdict"],
                    scenario["performance_conclusion"],
                    scenario["comparison_baseline"],
                    candidate_name,
                    speedup_text,
                    candidate["coefficient_of_variation_percent"],
                    candidate["median_average_prefill_batch_size"],
                    candidate["median_prompt_padding_waste_ratio"],
                )
            )
            + " |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def run_serving_benchmark(
    config_path: Path,
    *,
    source_commit: str,
    output_root: Path | None = None,
) -> ServingBenchmarkRun:
    """Run the complete alternating fresh-process matrix and bind all artifacts."""
    config = load_serving_benchmark_config(config_path)
    root = config.output_root if output_root is None else output_root
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{source_commit[:12]}"
    output_dir = root / run_id
    worker_dir = output_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=False)
    resolved_path = output_dir / "resolved_config.yaml"
    _ = resolved_path.write_text(
        yaml.safe_dump(resolved_config_document(config), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    _write_json(output_dir / "environment.json", _environment(config, source_commit))
    records: list[BenchmarkDocument] = []
    execution_index = 0
    for replicate in range(config.replicates):
        configured_executors = _benchmark_executors(config)
        executor_order = (
            configured_executors if replicate % 2 == 0 else tuple(reversed(configured_executors))
        )
        for scenario in config.scenarios:
            for executor in executor_order:
                records.append(
                    _worker_record(
                        config_path=config_path,
                        config=config,
                        scenario=scenario.name,
                        executor=executor,
                        replicate=replicate,
                        execution_index=execution_index,
                        worker_dir=worker_dir,
                    )
                )
                execution_index += 1
    raw_path = output_dir / "raw_replicates.jsonl"
    _ = raw_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(
        output_dir / "execution_order.json",
        {
            "alternating_executor_order": True,
            "records": [
                {
                    "execution_index": record["execution_index"],
                    "replicate": record["replicate"],
                    "scenario": record["scenario"],
                    "executor": record["executor"],
                }
                for record in records
            ],
        },
    )
    summary = summarize_serving_records(records, config)
    summary.update(
        {
            "source_commit": source_commit,
            "resolved_config_sha256": resolved_config_sha256(config),
        }
    )
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _ = (output_dir / "summary.md").write_text(
        _summary_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    for worker_path in worker_dir.iterdir():
        worker_path.unlink()
    worker_dir.rmdir()
    manifest_path = output_dir / "artifact_manifest.json"
    artifacts = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_commit": source_commit,
            "artifacts": cast("list[JsonValue]", artifacts),
        },
    )
    return ServingBenchmarkRun(
        output_dir=output_dir,
        summary_path=summary_path,
        strict_verdict=cast("str", summary["strict_verdict"]),
    )
