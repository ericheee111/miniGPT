from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving_simulator import (
    InvalidSimulatorConfigError,
    SimulatorExecutor,
    load_simulator_config,
    run_cache_aware_prefill_equivalence,
    run_cache_backend_equivalence,
    run_executor_equivalence,
    run_paged_attention_equivalence,
    run_prefix_cache_equivalence,
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
    assert json_config.apc_prefill_strategy.value == "sequential"


def test_chunked_scheduler_config_runs_through_simulator(tmp_path: Path) -> None:
    # Given: direct paged simulation opts into Stage 16 scheduling fields.
    config_path = tmp_path / "chunked.json"
    document = config_document()
    document.update(
        {
            "executor": "paged_attention",
            "kv_cache_backend": "paged",
            "kv_cache": {"block_tokens": 2, "num_blocks": 12},
            "max_ticks": 50,
            "scheduler": {
                "max_active_requests": 2,
                "max_cached_tokens": 16,
                "max_scheduled_tokens": 4,
                "prefill_chunk_tokens": 2,
            },
        }
    )
    requests = cast("list[dict[str, object]]", document["requests"])
    requests[0]["prompt_tokens"] = [1, 2, 3, 4]
    requests[0]["max_new_tokens"] = 2
    write_config(config_path, document)

    # When: strict parsing and the deterministic simulator execute the workload.
    config = load_simulator_config(config_path)
    result = run_simulation(config, output_dir=tmp_path / "chunked")

    # Then: the optional fields survive parsing and chunk events are observable.
    assert config.scheduler.max_scheduled_tokens == 4
    assert config.scheduler.prefill_chunk_tokens == 2
    assert any(event.event_type.value == "PREFILL_CHUNK_STARTED" for event in result.events)
    summary = cast(
        "dict[str, object]",
        json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8")),
    )
    prefill_key = "prefill_" + "tokens_computed"
    prefix_key = "prefix_hit_" + "tokens"
    assert summary[prefill_key] == getattr(result.metrics, prefill_key)
    assert summary[prefix_key] == getattr(result.metrics, prefix_key)


@pytest.mark.parametrize(
    ("budget", "chunk_size", "match"),
    [
        (4, 3, "align"),
        (2, 2, "too small"),
    ],
)
def test_chunked_scheduler_rejects_invalid_alignment_or_budget(
    tmp_path: Path,
    budget: int,
    chunk_size: int,
    match: str,
) -> None:
    # Given: Stage 16 is requested with an invalid block alignment or token budget.
    config_path = tmp_path / f"invalid-chunk-{budget}-{chunk_size}.json"
    document = config_document()
    document.update(
        {
            "executor": "paged_attention",
            "kv_cache_backend": "paged",
            "kv_cache": {"block_tokens": 2, "num_blocks": 12},
            "scheduler": {
                "max_active_requests": 2,
                "max_cached_tokens": 16,
                "max_scheduled_tokens": budget,
                "prefill_chunk_tokens": chunk_size,
            },
        }
    )
    write_config(config_path, document)

    # When/Then: strict simulator parsing rejects the ambiguous schedule.
    with pytest.raises(InvalidSimulatorConfigError, match=match):
        _ = load_simulator_config(config_path)


def test_prefix_cache_requires_direct_paged_executor(tmp_path: Path) -> None:
    # Given: APC is enabled without the direct paged executor/storage pair.
    config_path = tmp_path / "invalid-prefix-cache.json"
    document = config_document()
    document["prefix_cache"] = {"enabled": True}
    write_config(config_path, document)

    # When/Then: strict configuration rejects the ambiguous cache mode.
    with pytest.raises(InvalidSimulatorConfigError, match="requires paged_attention"):
        _ = load_simulator_config(config_path)


def test_prefix_cache_simulation_preserves_logical_results_and_rng(tmp_path: Path) -> None:
    # Given: an exact repeated prompt on direct paged decode with APC enabled.
    config_path = tmp_path / "prefix-cache.json"
    document = config_document()
    document.update(
        {
            "executor": "paged_attention",
            "kv_cache_backend": "paged",
            "kv_cache": {"block_tokens": 2, "num_blocks": 12},
            "prefix_cache": {"enabled": True},
            "max_ticks": 50,
            "scheduler": {"max_active_requests": 2, "max_cached_tokens": 16},
        }
    )
    requests = cast("list[dict[str, object]]", document["requests"])
    requests[0]["prompt_tokens"] = [1, 2, 3, 4]
    requests[0]["max_new_tokens"] = 2
    requests[1]["prompt_tokens"] = [1, 2, 3, 4]
    requests[1]["prompt_length"] = None
    requests[1]["arrival_time"] = 10.0
    requests[1]["max_new_tokens"] = 2
    write_config(config_path, document)
    config = load_simulator_config(config_path)

    # When: the same workload runs with direct paged decode and direct paged plus APC.
    comparison = run_prefix_cache_equivalence(config, output_dir=tmp_path / "comparison")

    # Then: logical output/RNG/state/events match and the repeated prompt skips four tokens.
    assert comparison.equivalent
    assert "rng_state" in comparison.checked_contracts
    assert comparison.direct.generated_tokens == comparison.automatic_prefix_cache.generated_tokens
    assert (
        comparison.direct.generator_state_hashes
        == comparison.automatic_prefix_cache.generator_state_hashes
    )
    metrics = comparison.automatic_prefix_cache.metrics
    assert metrics.prefix_hit_requests == 1
    assert metrics.avoided_prefill_tokens == 4


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
    assert summary["kv_cache_backend"] == "dense"
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
        "cache_accounting",
        "logical_event_semantics",
        "request_metrics",
    )
    assert (
        comparison.reference.generated_tokens
        == comparison.continuous_decode.generated_tokens
        == comparison.continuous.generated_tokens
    )
    assert (
        comparison.reference.request_statuses
        == comparison.continuous_decode.request_statuses
        == comparison.continuous.request_statuses
    )


def test_continuous_simulator_writes_prefill_batch_events_and_metrics(tmp_path: Path) -> None:
    # Given: an explicit Stage 11B prefill policy and two equal-length eligible prompts.
    document = config_document()
    document["executor"] = "continuous"
    document["prefill"] = {
        "max_batch_size": 8,
        "max_batch_tokens": 32,
        "max_padding_ratio": 0.25,
    }
    requests = cast("list[dict[str, object]]", document["requests"])
    requests[1]["arrival_time"] = 0.0
    requests[1]["prompt_length"] = 2
    config_path = tmp_path / "stage11b.json"
    write_config(config_path, document)

    # When: the full continuous simulator executes the workload.
    result = run_simulation(load_simulator_config(config_path), output_dir=tmp_path / "run")
    event_lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (result.output_dir / "prefill_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    # Then: batch boundaries and prompt-utilization metrics are explicit and deterministic.
    assert {path.name for path in result.output_paths} == {
        "events.jsonl",
        "prefill_events.jsonl",
        "prefill_observations.jsonl",
        "requests.csv",
        "summary.json",
        "timeline.md",
    }
    assert [line["event_type"] for line in event_lines[:2]] == [
        "PREFILL_BATCH_STARTED",
        "PREFILL_BATCH_FINISHED",
    ]
    assert event_lines[1]["request_ids"] == ["explicit", "derived"]
    assert event_lines[1]["batch_size"] == 2
    assert event_lines[1]["useful_prompt_tokens"] == 4
    assert event_lines[1]["padded_prompt_tokens"] == 4
    assert result.metrics.max_prefill_batch_size == 2
    assert result.metrics.prompt_padding_waste_ratio == 0.0


def test_cache_aware_prefill_simulator_reports_real_suffix_batch_structure(
    tmp_path: Path,
) -> None:
    # Given: one prefix primer followed by four same-length cache-hit suffix requests.
    document = config_document()
    document.update(
        {
            "executor": "paged_attention",
            "kv_cache_backend": "paged",
            "kv_cache": {"block_tokens": 2, "num_blocks": 24},
            "prefix_cache": {"enabled": True},
            "apc_prefill_strategy": "batched",
            "prefill": {
                "max_batch_size": 8,
                "max_batch_tokens": 64,
                "max_padding_ratio": 0.25,
            },
            "max_ticks": 50,
            "model": {
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "dropout": 0.0,
                "bias": False,
            },
            "scheduler": {"max_active_requests": 4, "max_cached_tokens": 32},
        }
    )
    template = cast("list[dict[str, object]]", document["requests"])[0]
    requests: list[dict[str, object]] = [
        {
            **template,
            "request_id": "prime",
            "prompt_tokens": [1, 2, 3, 4],
            "max_new_tokens": 1,
        }
    ]
    requests.extend(
        {
            **template,
            "request_id": f"suffix-{index}",
            "arrival_time": 10.0,
            "prompt_tokens": [1, 2, 3, 4, suffix],
            "max_new_tokens": 1,
            "seed": 200 + index,
        }
        for index, suffix in enumerate((5, 6, 7, 8))
    )
    document["requests"] = requests
    config_path = tmp_path / "stage15.json"
    write_config(config_path, document)

    # When: the Stage 15 simulator executes the configured batched APC mode.
    config = load_simulator_config(config_path)
    result = run_simulation(config, output_dir=tmp_path / "stage15")
    summary = cast(
        "dict[str, object]",
        json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8")),
    )
    observations = [
        cast("dict[str, object]", json.loads(line))
        for line in (result.output_dir / "prefill_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    # Then: direct observations prove a size-four suffix batch and two total model calls.
    assert summary["apc_prefill_strategy"] == "batched"
    assert summary["cache_aware_prefill_batches"] == 2
    assert summary["cache_aware_prefill_model_calls"] == 2
    assert summary["suffix_prefill_batch_sizes"] == [1, 4]
    assert summary["average_suffix_prefill_batch_size"] == 2.5
    assert summary["max_suffix_prefill_batch_size"] == 4
    assert summary["suffix_useful_tokens"] == 8
    assert summary["suffix_padded_tokens"] == 8
    assert summary["suffix_padding_waste_ratio"] == 0.0
    assert summary["batched_suffix_requests"] == 5
    assert summary["prefix_hit_tokens"] == 16
    assert summary["avoided_prefill_tokens"] == 16
    assert observations[-1]["execution_mode"] == "batched_apc_suffix"
    assert observations[-1]["batch_size"] == 4
    assert observations[-1]["model_calls"] == 1

    # Then: the explicit sequential reference has identical logic but five model calls vs two.
    comparison = run_cache_aware_prefill_equivalence(
        config,
        output_dir=tmp_path / "stage15-equivalence",
    )
    assert comparison.equivalent
    assert sum(item.model_calls for item in comparison.sequential.prefill_observations) == 5
    assert sum(item.model_calls for item in comparison.batched.prefill_observations) == 2


def test_paged_backend_matches_dense_logical_contracts(tmp_path: Path) -> None:
    # Given: one fixed workload with a small, explicitly configured page pool.
    document = config_document()
    document["executor"] = "continuous"
    document["kv_cache_backend"] = "paged"
    document["kv_cache"] = {"block_tokens": 2, "num_blocks": 4}
    config_path = tmp_path / "paged.json"
    write_config(config_path, document)
    config = load_simulator_config(config_path)
    assert config.kv_cache_backend is KVCacheBackend.PAGED

    # When: dense and paged storage execute identical model and request seeds.
    comparison = run_cache_backend_equivalence(config, output_dir=tmp_path / "comparison")

    # Then: logical serving is identical and every physical block returns to the pool.
    assert comparison.equivalent
    assert comparison.checked_contracts == (
        "generated_tokens",
        "request_terminal_states_and_cancellation",
        "fifo_admission_order",
        "logical_event_semantics",
        "request_metrics",
        "logical_cache_accounting",
    )
    assert comparison.dense.generated_tokens == comparison.paged.generated_tokens
    assert comparison.dense.request_statuses == comparison.paged.request_statuses
    assert comparison.paged.metrics.total_blocks == 4
    assert comparison.paged.metrics.free_blocks == 4
    assert comparison.paged.metrics.allocated_blocks == 0
    assert comparison.paged.metrics.reserved_blocks == 0
    assert comparison.paged.metrics.peak_allocated_blocks > 0


def test_paged_backend_requires_explicit_pool_config(tmp_path: Path) -> None:
    # Given: paged storage is requested without a physical pool definition.
    document = config_document()
    document["kv_cache_backend"] = "paged"
    config_path = tmp_path / "missing-pool.json"
    write_config(config_path, document)

    # When/Then: strict loading rejects the incomplete storage contract.
    with pytest.raises(InvalidSimulatorConfigError, match="requires kv_cache"):
        _ = load_simulator_config(config_path)


def test_invalid_executor_name_is_rejected(tmp_path: Path) -> None:
    # Given: a strict simulator document naming an unsupported executor.
    document = config_document()
    document["executor"] = "unsupported"
    config_path = tmp_path / "invalid-executor.json"
    write_config(config_path, document)

    # When/Then: loading rejects the out-of-scope executor explicitly.
    with pytest.raises(
        InvalidSimulatorConfigError,
        match="reference, continuous_decode, continuous, paged_attention",
    ):
        _ = load_simulator_config(config_path)


def test_direct_paged_attention_matches_dense_and_materialized_simulation(tmp_path: Path) -> None:
    # Given: a block-size overflow workload configured for direct paged attention.
    config = load_simulator_config(Path("configs") / "serving_paged_overflow.yaml")

    # When: all three storage/decode strategies run from identical weights and request seeds.
    comparison = run_paged_attention_equivalence(config, output_dir=tmp_path / "comparison")

    # Then: logical contracts match and direct traversal removes dense cache padding.
    assert comparison.equivalent
    assert comparison.dense.generated_tokens == comparison.direct.generated_tokens
    assert comparison.materialized.events == comparison.direct.events
    assert "direct_decode_has_no_cache_padding" in comparison.checked_contracts
    assert comparison.direct.metrics.padding_waste_ratio == 0.0


def test_committed_direct_paged_attention_scenario_finishes_without_leaks(tmp_path: Path) -> None:
    # Given: the fixed Stage 13B mixed-length and cancellation workload.
    config = load_simulator_config(Path("configs") / "serving_paged_attention.yaml")
    assert config.executor is SimulatorExecutor.PAGED_ATTENTION

    # When: block-aware decode runs through the simulator entrypoint.
    result = run_simulation(config, output_dir=tmp_path / "direct")

    # Then: all requests are terminal and every physical block is released.
    terminal = (
        result.metrics.completed_requests
        + result.metrics.cancelled_requests
        + result.metrics.failed_requests
    )
    assert terminal == result.metrics.total_requests
    assert result.metrics.cancelled_requests == 1
    assert result.metrics.allocated_blocks == 0
    assert result.metrics.reserved_blocks == 0
    assert result.metrics.free_blocks == result.metrics.total_blocks


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


@pytest.mark.parametrize(
    ("config_name", "minimum_failed"),
    [
        ("serving_paged_normal_burst.yaml", 0),
        ("serving_paged_tiny_pool.yaml", 1),
        ("serving_paged_reuse.yaml", 0),
        ("serving_paged_cancellation_churn.yaml", 0),
        ("serving_paged_failure_rollback.yaml", 1),
        ("serving_paged_overflow.yaml", 0),
        ("serving_paged_fragmentation.yaml", 0),
    ],
)
def test_committed_paged_scenarios_release_all_physical_blocks(
    config_name: str,
    minimum_failed: int,
    tmp_path: Path,
) -> None:
    # Given: one fixed Stage 13A capacity, churn, overflow, or fragmentation scenario.
    config = load_simulator_config(Path("configs") / config_name)

    # When: the paged simulator reaches a terminal state for every request.
    result = run_simulation(config, output_dir=tmp_path / config.scenario_name)

    # Then: expected deterministic failures are isolated and the pool is fully released.
    terminal_count = (
        result.metrics.completed_requests
        + result.metrics.cancelled_requests
        + result.metrics.failed_requests
    )
    assert terminal_count == result.metrics.total_requests
    assert result.metrics.failed_requests >= minimum_failed
    assert result.metrics.allocated_blocks == 0
    assert result.metrics.reserved_blocks == 0
    assert result.metrics.free_blocks == result.metrics.total_blocks


@pytest.mark.parametrize(
    "config_name",
    [
        "serving_paged_normal_burst.yaml",
        "serving_paged_reuse.yaml",
        "serving_paged_cancellation_churn.yaml",
        "serving_paged_overflow.yaml",
        "serving_paged_fragmentation.yaml",
    ],
)
def test_committed_paged_scenarios_match_dense_correctness_contracts(
    config_name: str,
    tmp_path: Path,
) -> None:
    # Given: a fixed workload whose maximum reservations fit both storage backends.
    config = load_simulator_config(Path("configs") / config_name)

    # When: identical model and request seeds run through both backends.
    comparison = run_cache_backend_equivalence(config, output_dir=tmp_path / config.scenario_name)

    # Then: all logical correctness contracts remain identical.
    assert comparison.equivalent
