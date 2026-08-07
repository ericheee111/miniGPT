from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from minigpt.serving_benchmark import (
    BenchmarkDocument,
    run_serving_benchmark,
    summarize_serving_records,
)
from minigpt.serving_benchmark_config import (
    InvalidServingBenchmarkConfigError,
    load_serving_benchmark_config,
    resolved_config_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path


def benchmark_document(output_root: Path, *, replicates: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_name": "stage11a-test",
        "benchmark_seed": 1101,
        "output_root": output_root.as_posix(),
        "worker_timeout_seconds": 60.0,
        "warmup_iterations": 0,
        "measurement_iterations": 1,
        "replicates": replicates,
        "minimum_replicates": replicates,
        "max_cv_percent": 10.0,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "vocab_size": 17,
        "model": {
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
            "bias": False,
        },
        "scenarios": [
            {
                "name": "burst-2",
                "arrival_ticks": [0, 1],
                "prompt_lengths": [2, 3],
                "generated_lengths": [3, 3],
                "cancellation_ticks": [None, None],
            }
        ],
    }


def write_config(path: Path, document: dict[str, object]) -> None:
    _ = path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def test_serving_benchmark_config_is_strict_and_hash_stable(tmp_path: Path) -> None:
    # Given: two byte-different YAML files with identical resolved semantics.
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    document = benchmark_document(tmp_path / "reports")
    write_config(first_path, document)
    _ = second_path.write_text(
        yaml.safe_dump(document, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    # When: both strict configs are loaded and hashed canonically.
    first = load_serving_benchmark_config(first_path)
    second = load_serving_benchmark_config(second_path)

    # Then: formatting does not change identity and mixed request lengths are preserved.
    assert first == second
    assert resolved_config_sha256(first) == resolved_config_sha256(second)
    assert first.scenarios[0].prompt_lengths == (2, 3)


def test_fresh_process_benchmark_writes_raw_order_statistics_and_hashes(tmp_path: Path) -> None:
    # Given: a one-replicate smoke matrix small enough for a unit-test subprocess.
    config_path = tmp_path / "benchmark.yaml"
    write_config(config_path, benchmark_document(tmp_path / "reports"))

    # When: reference and continuous workers run in separate fresh processes.
    result = run_serving_benchmark(config_path, source_commit="0123456789abcdef")

    # Then: strict comparison, raw replicates, order, environment, and artifact hashes exist.
    assert result.strict_verdict == "pass"
    summary = cast(
        "dict[str, object]",
        json.loads(result.summary_path.read_text(encoding="utf-8")),
    )
    scenarios = cast("list[dict[str, object]]", summary["scenarios"])
    assert scenarios[0]["correctness_matches"] is True
    assert scenarios[0]["strict_verdict"] == "pass"
    raw_lines = (
        (result.output_dir / "raw_replicates.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(raw_lines) == 2
    for raw_line in raw_lines:
        record = cast("dict[str, object]", json.loads(raw_line))
        for iteration in cast("list[dict[str, object]]", record["iterations"]):
            assert cast("float", iteration["median_ttft_seconds"]) >= 0.0
            assert cast("float", iteration["median_tpot_seconds"]) >= 0.0
            assert cast("float", iteration["median_e2e_seconds"]) >= 0.0
    manifest = cast(
        "dict[str, object]",
        json.loads((result.output_dir / "artifact_manifest.json").read_text(encoding="utf-8")),
    )
    for raw_entry in cast("list[dict[str, object]]", manifest["artifacts"]):
        artifact = result.output_dir / cast("str", raw_entry["path"])
        assert artifact.stat().st_size == raw_entry["bytes"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == raw_entry["sha256"]


def test_stage11b_benchmark_compares_decode_batching_with_full_continuous(
    tmp_path: Path,
) -> None:
    # Given: an explicit prefill policy and equal-length simultaneous prompts.
    document = benchmark_document(tmp_path / "reports")
    document["experiment_name"] = "stage11b-test"
    document["prefill"] = {
        "max_batch_size": 8,
        "max_batch_tokens": 64,
        "max_padding_ratio": 0.25,
    }
    scenarios = cast("list[dict[str, object]]", document["scenarios"])
    scenarios[0]["arrival_ticks"] = [0, 0]
    scenarios[0]["prompt_lengths"] = [3, 3]
    config_path = tmp_path / "benchmark-stage11b.yaml"
    write_config(config_path, document)

    # When: the fresh-process matrix runs all three serving executors.
    result = run_serving_benchmark(config_path, source_commit="fedcba9876543210")
    summary = cast(
        "dict[str, object]",
        json.loads(result.summary_path.read_text(encoding="utf-8")),
    )
    scenario = cast("list[dict[str, object]]", summary["scenarios"])[0]
    raw = [
        cast("dict[str, object]", json.loads(line))
        for line in (result.output_dir / "raw_replicates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    # Then: strict comparison isolates continuous_decode versus continuous and records prefill data.
    assert result.strict_verdict == "pass"
    assert {cast("str", record["executor"]) for record in raw} == {
        "reference",
        "continuous_decode",
        "continuous",
    }
    assert scenario["comparison_baseline"] == "continuous_decode"
    assert scenario["comparison_candidate"] == "continuous"
    assert scenario["correctness_matches"] is True
    continuous = cast("dict[str, object]", scenario["continuous"])
    assert cast("float", continuous["median_average_prefill_batch_size"]) == 2.0
    assert cast("float", continuous["median_prompt_padding_waste_ratio"]) == 0.0
    for record in raw:
        for iteration in cast("list[dict[str, object]]", record["iterations"]):
            assert cast("float", iteration["median_queue_time_seconds"]) >= 0.0
            assert cast("float", iteration["median_prefill_latency_seconds"]) >= 0.0
            assert cast("int", iteration["worker_peak_rss_bytes"]) > 0


def test_strict_comparison_reports_not_comparable_for_unstable_replicates(tmp_path: Path) -> None:
    # Given: matching correctness but deliberately unstable elapsed times for both executors.
    config_path = tmp_path / "benchmark.yaml"
    write_config(config_path, benchmark_document(tmp_path / "reports", replicates=2))
    config = load_serving_benchmark_config(config_path)
    records: list[BenchmarkDocument] = []
    for executor in ("reference", "continuous_decode"):
        for replicate, elapsed in enumerate((1.0, 2.0)):
            records.append(
                {
                    "status": "ok",
                    "scenario": "burst-2",
                    "executor": executor,
                    "replicate": replicate,
                    "elapsed_seconds": elapsed,
                    "request_throughput_per_second": 2.0 / elapsed,
                    "token_throughput_per_second": 6.0 / elapsed,
                    "median_ttft_seconds": 0.1,
                    "median_tpot_seconds": 0.1,
                    "median_e2e_seconds": 0.3,
                    "average_decode_batch_size": 2.0,
                    "max_decode_batch_size": 2.0,
                    "padded_cache_tokens": 8.0,
                    "useful_cache_tokens": 7.0,
                    "padding_waste_ratio": 0.125,
                    "executor_time_seconds": elapsed,
                    "model_execution_time_seconds": elapsed * 0.8,
                    "batch_assembly_scatter_time_seconds": elapsed * 0.2,
                    "correctness_sha256": "same",
                }
            )

    # When: the strict policy checks CV without filtering either raw replicate.
    summary = summarize_serving_records(records, config)

    # Then: no performance conclusion is permitted from unstable evidence.
    assert summary["strict_verdict"] == "not_comparable"
    scenario = cast("list[dict[str, object]]", summary["scenarios"])[0]
    assert scenario["performance_conclusion"] == "not_comparable"
    assert scenario["speedup_reference_over_continuous"] is None


def test_serving_benchmark_rejects_misaligned_scenario_vectors(tmp_path: Path) -> None:
    # Given: prompt lengths that do not align with the request arrival vector.
    document = benchmark_document(tmp_path / "reports")
    scenarios = cast("list[dict[str, object]]", document["scenarios"])
    scenarios[0]["prompt_lengths"] = [2]
    config_path = tmp_path / "invalid.yaml"
    write_config(config_path, document)

    # When/Then: strict loading rejects ambiguous request construction.
    with pytest.raises(InvalidServingBenchmarkConfigError, match="field lengths must match"):
        _ = load_serving_benchmark_config(config_path)
