import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import cast

from minigpt import benchmark, benchmark_config, profiling

PROJECT_ROOT = Path(__file__).parents[1]


def write_benchmark_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "benchmark.yaml"
    _ = config_path.write_text(
        f"""
seed: 123
vocab_size: 32
thread_counts: [1, 2]
block_sizes: [4, 8]
batch_sizes: [2]
model_sizes:
  tiny:
    n_layer: 1
    n_head: 1
    n_embd: 8
  small:
    n_layer: 2
    n_head: 2
    n_embd: 16
warmup_steps: 1
measurement_steps: 1
repeats: 2
output_dir: "{(tmp_path / "reports").as_posix()}"
profile:
  enabled: false
  thread_count: 1
  block_size: 4
  batch_size: 2
  model_size: tiny
  warmup_steps: 1
  active_steps: 2
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_benchmark_config_expands_cartesian_product(tmp_path: Path) -> None:
    # Given: two thread counts, two contexts, one batch size, and two models.
    config = benchmark_config.load_benchmark_config(write_benchmark_config(tmp_path))

    # When: the benchmark matrix is expanded.
    cases = benchmark.expand_cases(config)

    # Then: all eight unique configurations are present in stable order.
    assert len(cases) == 8
    assert cases[0].label == "tiny-t1-b2-s4"
    assert cases[-1].label == "small-t2-b2-s8"
    assert len({case.label for case in cases}) == 8


def test_summary_uses_median_and_reports_variability() -> None:
    # Given: three raw timings with one slower repeat.
    case = benchmark.BenchmarkCase(
        model_size="tiny",
        n_layer=1,
        n_head=1,
        n_embd=8,
        thread_count=1,
        block_size=4,
        batch_size=2,
    )
    measurements = [
        benchmark.BenchmarkMeasurement(case, 0, 10.0, 800.0, 100.0, 1_000),
        benchmark.BenchmarkMeasurement(case, 1, 11.0, 727.0, 101.0, 1_000),
        benchmark.BenchmarkMeasurement(case, 2, 12.0, 667.0, 102.0, 1_000),
    ]

    # When: raw repeats are summarized.
    summary = benchmark.summarize_measurements(measurements)

    # Then: median, population deviation, MAD, and CV remain independently visible.
    assert summary.median_step_time_ms == 11.0
    assert math.isclose(summary.step_time_stddev_ms, 0.81649658, rel_tol=1e-7)
    assert summary.step_time_mad_ms == 1.0
    assert math.isclose(summary.step_time_cv_percent, 7.422696, rel_tol=1e-7)
    assert summary.median_tokens_per_sec == 727.0


def test_run_benchmark_writes_raw_summary_and_markdown(tmp_path: Path) -> None:
    # Given: one tiny benchmark case with two measured repeats.
    parsed = benchmark_config.load_benchmark_config(write_benchmark_config(tmp_path))
    one_case_config = parsed.with_matrix(
        thread_counts=(1,),
        block_sizes=(4,),
        batch_sizes=(2,),
        model_size_names=("tiny",),
    )

    # When: the CPU benchmark executes.
    artifacts = benchmark.run_benchmark(one_case_config)

    # Then: raw repeats, one summary row, and methodology are durable.
    with artifacts.raw_csv.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    with artifacts.summary_csv.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    report = artifacts.report_markdown.read_text(encoding="utf-8")
    assert len(raw_rows) == 2
    assert len(summary_rows) == 1
    assert "# CPU Training Benchmark" in report
    assert "warmup" in report.lower()
    assert "median" in report.lower()


def test_run_profile_writes_trace_top_operators_and_report(tmp_path: Path) -> None:
    # Given: one tiny representative profiler configuration.
    config = benchmark_config.load_benchmark_config(write_benchmark_config(tmp_path))

    # When: CPU operator profiling executes.
    artifacts = profiling.run_profile(config)

    # Then: Chrome trace, scoped operators, and interpretation guidance are durable.
    trace = cast("object", json.loads(artifacts.trace_json.read_text(encoding="utf-8")))
    with artifacts.top_operators_csv.open(newline="", encoding="utf-8") as stream:
        operators = list(csv.DictReader(stream))
    operator_names = {row["operator"] for row in operators}
    operator_counts = {row["operator"]: int(row["count"]) for row in operators}
    report = artifacts.report_markdown.read_text(encoding="utf-8")
    assert isinstance(trace, dict)
    assert "traceEvents" in trace
    assert {"data_preparation", "forward_backward", "optimizer_step"} <= operator_names
    assert operator_counts["forward_backward"] == 2
    assert "Attention" in report
    assert "Chrome trace" in report


def test_benchmark_cli_runs_configured_matrix(tmp_path: Path) -> None:
    # Given: a tiny YAML benchmark matrix.
    config_path = write_benchmark_config(tmp_path)

    # When: the documented benchmark CLI runs.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "benchmark.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the process succeeds and prints its report path.
    assert completed.returncode == 0, completed.stderr
    assert "benchmark_report.md" in completed.stdout
    assert (tmp_path / "reports" / "benchmark_raw.csv").is_file()


def test_profile_cli_runs_configured_case(tmp_path: Path) -> None:
    # Given: a tiny YAML profiler case.
    config_path = write_benchmark_config(tmp_path)

    # When: the documented profiling CLI runs.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "profile_model.py", "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the process succeeds and exports a Chrome trace.
    assert completed.returncode == 0, completed.stderr
    assert "trace.json" in completed.stdout
    assert (tmp_path / "reports" / "profile" / "trace.json").is_file()
