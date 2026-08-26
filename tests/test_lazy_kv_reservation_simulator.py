from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from minigpt.paged_kv_cache import PagedKVCacheConfig
from minigpt.serving import EngineEventType, SchedulerConfig
from minigpt.serving_simulator import (
    InvalidSimulatorConfigError,
    load_simulator_config,
    run_simulation,
)


def test_stage18_canonical_simulator_exercises_growth_pressure(tmp_path: Path) -> None:
    config = load_simulator_config(Path("configs/serving_lazy_kv_reservation.yaml"))
    output = tmp_path / "stage18-simulator"

    result = run_simulation(replace(config, output_dir=output), output_dir=output)

    assert config.scheduler.lazy_kv_reservation is True
    assert config.scheduler.kv_overcommit_ratio == 2.0
    assert result.metrics.completed_requests == 2
    assert result.metrics.reservation_growths > 0
    assert result.metrics.reservation_growth_tokens > 0
    assert result.metrics.reservation_growth_blocked > 0
    assert result.metrics.growth_pressure_preemptions > 0
    assert result.metrics.preemptions > 0
    assert result.metrics.resumes > 0
    assert any(
        event.event_type is EngineEventType.RESERVATION_GROWTH_BLOCKED for event in result.events
    )
    assert any(event.event_type is EngineEventType.RESERVATION_GROWN for event in result.events)
    raw_summary = cast(
        "object",
        json.loads((output / "summary.json").read_text(encoding="utf-8")),
    )
    assert isinstance(raw_summary, dict)
    summary = cast("dict[str, object]", raw_summary)
    assert summary["lazy_kv_reservation"] is True
    assert summary["kv_overcommit_ratio"] == 2.0
    assert summary["benchmark_strict_verdict"] == "descriptive_only"
    assert summary["wall_clock_performance_improvement"] is False
    requests = (output / "requests.csv").read_text(encoding="utf-8")
    assert "reservation_growth_count" in requests
    assert "reservation_growth_blocked_count" in requests


def test_stage18_simulator_matches_roomy_full_reservation_reference(tmp_path: Path) -> None:
    config = load_simulator_config(Path("configs/serving_lazy_kv_reservation.yaml"))
    pressured = run_simulation(
        replace(config, output_dir=tmp_path / "pressured"),
        output_dir=tmp_path / "pressured",
    )
    roomy_scheduler = SchedulerConfig(
        max_active_requests=2,
        max_cached_tokens=16,
        max_scheduled_tokens=8,
        prefill_chunk_tokens=2,
    )
    roomy = run_simulation(
        replace(
            config,
            scenario_name="lazy-kv-roomy-reference",
            paged_kv_cache=PagedKVCacheConfig(block_tokens=2, num_blocks=8),
            scheduler=roomy_scheduler,
            output_dir=tmp_path / "roomy",
        ),
        output_dir=tmp_path / "roomy",
    )

    assert pressured.generated_tokens == roomy.generated_tokens
    assert pressured.generator_state_hashes == roomy.generator_state_hashes
    assert pressured.request_statuses == roomy.request_statuses
    assert pressured.metrics.growth_pressure_preemptions > 0
    assert roomy.metrics.growth_pressure_preemptions == 0


def test_simulator_rejects_lazy_reservation_without_direct_paged_path(tmp_path: Path) -> None:
    source = Path("configs/serving_lazy_kv_reservation.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.yaml"
    _ = invalid.write_text(
        source.replace("executor: paged_attention", "executor: reference"),
        encoding="utf-8",
    )

    with pytest.raises(InvalidSimulatorConfigError, match="requires paged_attention"):
        _ = load_simulator_config(invalid)


def test_lazy_simulator_wraps_scheduler_dependency_errors(tmp_path: Path) -> None:
    config_path = Path("configs/serving_lazy_kv_reservation.yaml")
    document = cast(
        "dict[str, object]",
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    scheduler = cast("dict[str, object]", document["scheduler"])
    scheduler["kv_preemption"] = False
    invalid = tmp_path / "lazy-without-preemption.yaml"
    _ = invalid.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(InvalidSimulatorConfigError, match="requires kv_preemption"):
        _ = load_simulator_config(invalid)
