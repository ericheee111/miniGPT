"""Run one fresh-process Stage 11A serving benchmark replicate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias, cast

import torch

from minigpt.model import GPT
from minigpt.serving import (
    ContinuousDecodeExecutor,
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    ReferenceExecutor,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.serving_benchmark_config import (
    JsonValue,
    ServingBenchmarkConfig,
    ServingBenchmarkScenario,
    load_serving_benchmark_config,
)
from minigpt.serving_simulator import SimulatorExecutor

WorkerDocument: TypeAlias = dict[str, JsonValue]


def _build_model(config: ServingBenchmarkConfig) -> GPT:
    _ = torch.default_generator.manual_seed(config.benchmark_seed)
    model = GPT(config.model)
    _ = model.eval()
    return model


def _requests(
    config: ServingBenchmarkConfig,
    scenario: ServingBenchmarkScenario,
) -> tuple[GenerationRequest, ...]:
    requests: list[GenerationRequest] = []
    for index, (arrival_tick, prompt_length, generated_length) in enumerate(
        zip(
            scenario.arrival_ticks,
            scenario.prompt_lengths,
            scenario.generated_lengths,
            strict=True,
        )
    ):
        prompt = tuple(
            (config.benchmark_seed + index * 13 + position) % config.vocab_size
            for position in range(prompt_length)
        )
        requests.append(
            GenerationRequest(
                request_id=f"request-{index:02d}",
                prompt_tokens=prompt,
                max_new_tokens=generated_length,
                temperature=1.0,
                top_k=1,
                seed=config.benchmark_seed + 1000 + index,
                arrival_time=float(arrival_tick),
            )
        )
    return tuple(requests)


def _finite_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _correctness_document(
    engine: ServingEngine, requests: tuple[GenerationRequest, ...]
) -> WorkerDocument:
    return {
        "generated_tokens": {
            request.request_id: list(engine.request_state(request.request_id).generated_tokens)
            for request in requests
        },
        "terminal_statuses": {
            request.request_id: engine.request_state(request.request_id).status.value
            for request in requests
        },
        "admission_order": [
            event.request_id
            for event in engine.events
            if event.event_type is EngineEventType.ADMITTED
        ],
        "cancelled_requests": [
            request.request_id
            for request in requests
            if engine.request_state(request.request_id).status is RequestStatus.CANCELLED
        ],
    }


def _correctness_sha256(document: WorkerDocument) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_once(
    config: ServingBenchmarkConfig,
    scenario: ServingBenchmarkScenario,
    executor_name: SimulatorExecutor,
    model: GPT,
) -> WorkerDocument:
    """Execute one complete workload and return canonical wall-clock metrics."""
    executor = (
        ContinuousDecodeExecutor(model)
        if executor_name is SimulatorExecutor.CONTINUOUS_DECODE
        else ReferenceExecutor(model)
    )
    requests = _requests(config, scenario)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=len(requests),
                max_cached_tokens=len(requests) * config.model.block_size,
            ),
            block_size=config.model.block_size,
        ),
        executor=executor,
    )
    pending = list(enumerate(requests))
    submitted: set[str] = set()
    cancelled: set[str] = set()
    start = time.perf_counter()
    max_ticks = max(scenario.arrival_ticks) + max(scenario.generated_lengths) + 4
    for tick in range(max_ticks):
        for index, request in tuple(pending):
            if scenario.arrival_ticks[index] <= tick:
                engine.submit(request)
                pending.remove((index, request))
                submitted.add(request.request_id)
        for index, cancellation_tick in enumerate(scenario.cancellation_ticks):
            request_id = requests[index].request_id
            if (
                cancellation_tick is not None
                and cancellation_tick <= tick
                and request_id not in cancelled
                and request_id in submitted
            ):
                state = engine.request_state(request_id)
                if state.status not in {
                    RequestStatus.FINISHED,
                    RequestStatus.CANCELLED,
                    RequestStatus.FAILED,
                }:
                    engine.cancel(request_id, at=time.perf_counter() - start)
                cancelled.add(request_id)
        if not engine.is_idle:
            engine.tick(now=time.perf_counter() - start)
        if not pending and engine.is_idle:
            break
    elapsed = time.perf_counter() - start
    if pending or not engine.is_idle:
        msg = f"benchmark scenario {scenario.name} did not become idle"
        raise RuntimeError(msg)

    request_metrics = tuple(engine.request_metrics(request.request_id) for request in requests)
    ttft = [
        metric.time_to_first_token_seconds
        for metric in request_metrics
        if metric.time_to_first_token_seconds is not None
    ]
    tpot = [
        metric.time_per_output_token_seconds
        for metric in request_metrics
        if metric.time_per_output_token_seconds is not None
    ]
    e2e = [
        metric.end_to_end_latency_seconds
        for metric in request_metrics
        if metric.end_to_end_latency_seconds is not None
    ]
    metrics = engine.metrics()
    correctness = _correctness_document(engine, requests)
    completed_or_cancelled = metrics.completed_requests + metrics.cancelled_requests
    return {
        "elapsed_seconds": elapsed,
        "request_throughput_per_second": completed_or_cancelled / elapsed,
        "token_throughput_per_second": metrics.generated_tokens / elapsed,
        "median_ttft_seconds": _finite_median(ttft),
        "median_tpot_seconds": _finite_median(tpot),
        "median_e2e_seconds": _finite_median(e2e),
        "average_decode_batch_size": metrics.average_decode_batch_size,
        "max_decode_batch_size": metrics.max_decode_batch_size,
        "padded_cache_tokens": metrics.padded_cache_tokens,
        "useful_cache_tokens": metrics.useful_cache_tokens,
        "padding_waste_ratio": metrics.padding_waste_ratio,
        "executor_time_seconds": metrics.executor_time_seconds,
        "model_execution_time_seconds": metrics.model_execution_time_seconds,
        "batch_assembly_scatter_time_seconds": metrics.batch_assembly_scatter_time_seconds,
        "generated_tokens": metrics.generated_tokens,
        "completed_requests": metrics.completed_requests,
        "cancelled_requests": metrics.cancelled_requests,
        "failed_requests": metrics.failed_requests,
        "correctness": correctness,
        "correctness_sha256": _correctness_sha256(correctness),
    }


def run_worker(
    config: ServingBenchmarkConfig,
    scenario: ServingBenchmarkScenario,
    executor_name: SimulatorExecutor,
) -> WorkerDocument:
    """Warm up and return every unfiltered canonical measurement iteration."""
    torch.set_num_threads(config.torch_num_threads)
    torch.set_num_interop_threads(config.torch_num_interop_threads)
    model = _build_model(config)
    for _ in range(config.warmup_iterations):
        _ = run_once(config, scenario, executor_name, model)
    iterations = [
        run_once(config, scenario, executor_name, model)
        for _ in range(config.measurement_iterations)
    ]
    return cast(
        "WorkerDocument",
        {
            "schema_version": 1,
            "scenario": scenario.name,
            "executor": executor_name.value,
            "model": cast("dict[str, JsonValue]", asdict(config.model)),
            "iterations": iterations,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Stage 11A serving benchmark worker.")
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--scenario", required=True)
    _ = parser.add_argument(
        "--executor", choices=[executor.value for executor in SimulatorExecutor]
    )
    _ = parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one worker payload and write its JSON response."""
    arguments = _parser().parse_args(argv)
    config = load_serving_benchmark_config(cast("Path", arguments.config))
    scenario_name = cast("str", arguments.scenario)
    scenario = next(scenario for scenario in config.scenarios if scenario.name == scenario_name)
    executor_name = SimulatorExecutor(cast("str", arguments.executor))
    document = run_worker(config, scenario, executor_name)
    output = cast("Path", arguments.output)
    _ = output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
