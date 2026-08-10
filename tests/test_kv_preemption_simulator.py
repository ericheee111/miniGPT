from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from minigpt.serving import EngineEventType, PrefillExecutionMode
from minigpt.serving_simulator import load_simulator_config, run_simulation


def test_stage17_canonical_simulator_exercises_pressure_and_recompute(tmp_path: Path) -> None:
    config = load_simulator_config(Path("configs/serving_kv_preemption.yaml"))
    output = tmp_path / "stage17-simulator"

    result = run_simulation(replace(config, output_dir=output), output_dir=output)

    assert config.scheduler.kv_preemption is True
    assert result.metrics.preemptions > 0
    assert result.metrics.resumes > 0
    assert result.metrics.recompute_tokens > 0
    assert any(event.event_type is EngineEventType.PREEMPTED for event in result.events)
    assert any(event.event_type is EngineEventType.RESUMED for event in result.events)
    recompute = [
        item
        for item in result.prefill_observations
        if item.execution_mode is PrefillExecutionMode.PREEMPTION_RECOMPUTE
    ]
    assert recompute
    assert sum(item.useful_prompt_tokens for item in recompute) == result.metrics.recompute_tokens
