from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.serving_simulator import (
    InvalidSimulatorConfigError,
    SimulatorExecutor,
    load_simulator_config,
    run_executor_equivalence,
    run_simulation,
)


def config_document(output_dir: str = "reports/serving-test") -> dict[str, object]:
    return {
        "schema_version": 1,
        "scenario_name": "test-scenario",
        "model_seed": 123,
        "tick_seconds": 1.0,
        "executor_clock_step_seconds": 0.01,
        "max_ticks": 20,
        "output_dir": output_dir,
        "vocab_size": 17,
        "model": {
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
            "bias": False,
        },
        "scheduler": {"max_active_requests": 2, "max_cached_tokens": 8},
        "requests": [
            {
                "request_id": "explicit",
                "arrival_time": 0.0,
                "prompt_tokens": [1, 2],
                "prompt_length": None,
                "max_new_tokens": 3,
                "temperature": 1.0,
                "top_k": None,
                "seed": 101,
                "cancellation_time": None,
            },
            {
                "request_id": "derived",
                "arrival_time": 1.0,
                "prompt_tokens": None,
                "prompt_length": 3,
                "max_new_tokens": 2,
                "temperature": 0.8,
                "top_k": 5,
                "seed": 202,
                "cancellation_time": None,
            },
        ],
    }


def write_config(path: Path, document: dict[str, object]) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_json_and_yaml_inputs_resolve_to_the_same_strict_config(tmp_path: Path) -> None:
    # Given: equivalent JSON and YAML-compatible documents.
    json_path = tmp_path / "workload.json"
    yaml_path = tmp_path / "workload.yaml"
    write_config(json_path, config_document())
    write_config(yaml_path, config_document())

    # When: both sources are loaded through the strict simulator parser.
    json_config = load_simulator_config(json_path)
    yaml_config = load_simulator_config(yaml_path)

    # Then: format choice does not change the resolved workload.
    assert json_config == yaml_config
    assert json_config.requests[0].prompt_tokens == (1, 2)
    assert json_config.requests[1].prompt_tokens == (15, 16, 0)


def test_simulation_outputs_are_byte_reproducible(tmp_path: Path) -> None:
    # Given: one deterministic workload loaded once.
    config_path = tmp_path / "workload.yaml"
    write_config(config_path, config_document())
    config = load_simulator_config(config_path)

    # When: it is run into two independent directories.
    first = run_simulation(config, output_dir=tmp_path / "first")
    second = run_simulation(config, output_dir=tmp_path / "second")

    # Then: every required artifact and all generated tokens match byte-for-byte.
    expected_names = {"events.jsonl", "requests.csv", "summary.json", "timeline.md"}
    assert {path.name for path in first.output_paths} == expected_names
    for name in expected_names:
        assert (first.output_dir / name).read_bytes() == (second.output_dir / name).read_bytes()
    assert first.generated_tokens == second.generated_tokens


def test_summary_and_request_rows_publish_control_plane_metrics(tmp_path: Path) -> None:
    # Given: a deterministic two-request simulation.
    config_path = tmp_path / "workload.json"
    write_config(config_path, config_document())

    # When: required artifacts are emitted.
    result = run_simulation(load_simulator_config(config_path), output_dir=tmp_path / "run")
    summary = cast(
        "dict[str, object]",
        json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8")),
    )
    request_csv = (result.output_dir / "requests.csv").read_text(encoding="utf-8")

    # Then: aggregate capacity/throughput and request latency definitions are present.
    assert summary["claim"] == (
        "logical serving correctness; wall-clock performance reported separately"
    )
    assert summary["executor"] == "reference"
    assert summary["completed_requests"] == 2
    assert summary["failed_requests"] == 0
    assert summary["generated_tokens"] == 5
    assert summary["peak_active_requests"] == 2
    assert "queue_time_seconds" in request_csv
    assert "time_to_first_token_seconds" in request_csv
    assert "time_per_output_token_seconds" in request_csv
    assert "end_to_end_latency_seconds" in request_csv
    assert "decode_batch_sizes" in summary
    assert "padding_waste_ratio" in summary
    assert "model_execution_time_seconds" in summary


def test_continuous_simulator_matches_reference_logical_contracts(tmp_path: Path) -> None:
    # Given: one mixed-length workload with independent RNG and a scheduled cancellation.
    document = config_document()
    document["executor"] = "continuous_decode"
    requests = cast("list[dict[str, object]]", document["requests"])
    requests[1]["cancellation_time"] = 2.0
    config_path = tmp_path / "workload.json"
    write_config(config_path, document)
    config = load_simulator_config(config_path)
    assert config.executor is SimulatorExecutor.CONTINUOUS_DECODE

    # When: the automatic comparison runs both executors from identical model/request seeds.
    comparison = run_executor_equivalence(config, output_dir=tmp_path / "comparison")

    # Then: every required logical contract is checked and equivalent.
    assert comparison.equivalent
    assert comparison.checked_contracts == (
        "generated_tokens",
        "request_terminal_states_and_cancellation",
        "fifo_admission_order",
        "logical_event_semantics",
        "request_metrics",
    )
    assert comparison.reference.generated_tokens == comparison.continuous.generated_tokens
    assert comparison.reference.request_statuses == comparison.continuous.request_statuses


def test_invalid_executor_name_is_rejected(tmp_path: Path) -> None:
    # Given: a strict simulator document naming an unsupported executor.
    document = config_document()
    document["executor"] = "paged_attention"
    config_path = tmp_path / "invalid-executor.json"
    write_config(config_path, document)

    # When/Then: loading rejects the out-of-scope executor explicitly.
    with pytest.raises(InvalidSimulatorConfigError, match="reference, continuous_decode"):
        _ = load_simulator_config(config_path)


def test_invalid_prompt_source_is_rejected(tmp_path: Path) -> None:
    # Given: a request specifying both explicit tokens and a derived prompt length.
    document = config_document()
    requests = cast("list[dict[str, object]]", document["requests"])
    requests[0]["prompt_length"] = 2
    config_path = tmp_path / "invalid.yaml"
    write_config(config_path, document)

    # When/Then: strict loading rejects ambiguous prompt construction.
    with pytest.raises(InvalidSimulatorConfigError, match="exactly one"):
        _ = load_simulator_config(config_path)


@pytest.mark.parametrize(
    "config_name",
    [
        "serving_single_request.yaml",
        "serving_burst_arrivals.yaml",
        "serving_cache_pressure.yaml",
    ],
)
def test_committed_scenarios_complete_with_expected_terminal_accounting(
    config_name: str,
    tmp_path: Path,
) -> None:
    # Given: one of the three fixed Stage 10 workload scenarios.
    config = load_simulator_config(Path("configs") / config_name)

    # When: the reference simulator runs the scenario.
    result = run_simulation(config, output_dir=tmp_path / config.scenario_name)
    terminal_count = (
        result.metrics.completed_requests
        + result.metrics.cancelled_requests
        + result.metrics.failed_requests
    )

    # Then: every submitted request becomes terminal without a runaway loop.
    assert terminal_count == result.metrics.total_requests
    assert result.metrics.active_requests == 0
    assert result.metrics.waiting_requests == 0
