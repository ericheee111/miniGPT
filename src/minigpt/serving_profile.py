"""Profile one Stage 11A serving workload outside canonical benchmark timing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from minigpt.model import GPT
from minigpt.serving_benchmark_config import (
    JsonValue,
    load_serving_benchmark_config,
)
from minigpt.serving_benchmark_worker import run_once
from minigpt.serving_simulator import SimulatorExecutor  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


def profile_serving_workload(
    config_path: Path,
    *,
    scenario_name: str,
    executor_name: SimulatorExecutor,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write descriptive operator evidence that is never benchmark throughput."""
    config = load_serving_benchmark_config(config_path)
    scenario = next(item for item in config.scenarios if item.name == scenario_name)
    torch.set_num_threads(config.torch_num_threads)
    torch.set_num_interop_threads(config.torch_num_interop_threads)
    _ = torch.default_generator.manual_seed(config.benchmark_seed)
    model = GPT(config.model)
    _ = model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"
    summary_path = output_dir / "summary.json"
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as profiler:
        metrics = run_once(config, scenario, executor_name, model)
    profiler.export_chrome_trace(str(trace_path))
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "scenario": scenario_name,
        "executor": executor_name.value,
        "descriptive_only": True,
        "canonical_timer": False,
        "warning": "profiler timings include instrumentation overhead and are not benchmark data",
        "workload_metrics": metrics,
    }
    _ = summary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary_path, trace_path
