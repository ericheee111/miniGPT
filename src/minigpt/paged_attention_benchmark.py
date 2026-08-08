"""Run a descriptive CPU comparison of materialized and block-aware paged decode."""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import torch

from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend, PagedKVCacheConfig, PagedKVCachePool
from minigpt.serving import (
    ContinuousExecutor,
    EngineConfig,
    GenerationRequest,
    PagedAttentionExecutor,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from minigpt.data import JsonValue

Strategy: TypeAlias = Literal["materialized", "direct"]
_STRATEGIES: tuple[Strategy, ...] = ("materialized", "direct")
_CAVEAT = "single-machine CPU timings; stability and environment identity are not a release claim"


@dataclass(frozen=True, slots=True)
class PagedAttentionBenchmarkConfig:
    """Configure a small repeatable CPU serving comparison."""

    warmups: int = 1
    repeats: int = 5
    cache_access_iterations: int = 100

    def __post_init__(self) -> None:
        """Reject non-positive canonical sample counts."""
        if isinstance(self.warmups, bool) or self.warmups < 0:
            reason = "warmups must be a non-negative integer"
            raise ValueError(reason)
        for name, value in (
            ("repeats", self.repeats),
            ("cache_access_iterations", self.cache_access_iterations),
        ):
            if isinstance(value, bool) or value <= 0:
                reason = f"{name} must be a positive integer"
                raise ValueError(reason)


def _model() -> GPT:
    original_state = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260808)
        model = GPT(
            GPTConfig(
                vocab_size=31,
                block_size=16,
                n_layer=2,
                n_head=2,
                n_embd=32,
                dropout=0.0,
                bias=False,
            )
        )
    finally:
        torch.set_rng_state(original_state)
    return model.eval()


def _requests() -> tuple[GenerationRequest, ...]:
    return tuple(
        GenerationRequest(
            request_id=f"bench-{index}",
            prompt_tokens=tuple(range(1, prompt_length + 1)),
            max_new_tokens=4,
            seed=900 + index,
        )
        for index, prompt_length in enumerate((2, 5, 8, 11), start=1)
    )


def _engine(model: GPT, strategy: Strategy) -> ServingEngine:
    paged_config = PagedKVCacheConfig(block_tokens=4, num_blocks=16)
    pool = PagedKVCachePool.from_model(paged_config, model)
    executor = (
        ContinuousExecutor(model)
        if strategy == "materialized"
        else PagedAttentionExecutor(model, pool)
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=64),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged_config,
        ),
        executor=executor,
        paged_cache_pool=pool,
    )


def _run_once(model: GPT, strategy: Strategy) -> dict[str, JsonValue]:
    engine = _engine(model, strategy)
    requests = _requests()
    for request in requests:
        engine.submit(request)
    started = time.perf_counter()
    for _ in range(64):
        if engine.is_idle:
            break
        engine.tick()
    elapsed = time.perf_counter() - started
    if not engine.is_idle:
        reason = "paged attention benchmark engine did not become idle"
        raise RuntimeError(reason)
    metrics = engine.metrics()
    engine.verify_cache_invariants()
    return cast(
        "dict[str, JsonValue]",
        {
            "strategy": strategy,
            "e2e_seconds": elapsed,
            "executor_seconds": metrics.executor_time_seconds,
            "model_seconds": metrics.model_execution_time_seconds,
            "assembly_scatter_seconds": metrics.batch_assembly_scatter_time_seconds,
            "generated_tokens": {
                request.request_id: engine.request_state(request.request_id).generated_tokens
                for request in requests
            },
            "decode_batch_sizes": metrics.decode_batch_sizes,
            "padding_waste_ratio": metrics.padding_waste_ratio,
            "all_resources_released": (
                metrics.allocated_blocks == 0
                and metrics.reserved_blocks == 0
                and metrics.free_blocks == metrics.total_blocks
            ),
        },
    )


def _duration(call: Callable[[], object], iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        _ = call()
    return time.perf_counter() - started


def _cache_access_benchmark(
    model: GPT,
    *,
    iterations: int,
) -> dict[str, JsonValue]:
    config = PagedKVCacheConfig(block_tokens=4, num_blocks=16)
    pool = PagedKVCachePool.from_model(config, model)
    request_ids: list[str] = []
    for request in _requests():
        request_ids.append(request.request_id)
        pool.reserve(request.request_id, 4)
        tokens = torch.tensor([request.prompt_tokens], dtype=torch.long)
        _, cache = model.prefill(tokens)
        pool.write_prefill(request.request_id, cache)
    materialize_seconds = _duration(
        lambda: tuple(pool.materialize(request_id) for request_id in request_ids),
        iterations,
    )
    view_seconds = _duration(
        lambda: tuple(pool.request_view(request_id) for request_id in request_ids),
        iterations,
    )
    pool.release_all()
    pool.verify_invariants()
    return {
        "iterations": iterations,
        "requests_per_iteration": len(request_ids),
        "materialize_seconds": materialize_seconds,
        "request_view_seconds": view_seconds,
    }


def _summary(values: Sequence[float]) -> dict[str, JsonValue]:
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": list(values),
        "median": statistics.median(values),
        "mean": mean,
        "cv": deviation / mean if mean else math.inf,
    }


def run_paged_attention_benchmark(
    *,
    config: PagedAttentionBenchmarkConfig | None = None,
) -> dict[str, JsonValue]:
    """Return descriptive timings and reject any logical output divergence."""
    resolved = config or PagedAttentionBenchmarkConfig()
    model = _model()
    for _ in range(resolved.warmups):
        for strategy in _STRATEGIES:
            _ = _run_once(model, strategy)
    measurements: dict[Strategy, list[dict[str, JsonValue]]] = {
        "materialized": [],
        "direct": [],
    }
    for repeat in range(resolved.repeats):
        order = _STRATEGIES if repeat % 2 == 0 else tuple(reversed(_STRATEGIES))
        for strategy in order:
            measurements[strategy].append(_run_once(model, strategy))
    expected_tokens = measurements["materialized"][0]["generated_tokens"]
    if any(
        measurement["generated_tokens"] != expected_tokens
        for strategy in _STRATEGIES
        for measurement in measurements[strategy]
    ):
        reason = "paged attention benchmark strategies generated different tokens"
        raise RuntimeError(reason)
    strategies: dict[str, JsonValue] = {}
    for strategy in _STRATEGIES:
        rows = measurements[strategy]
        strategies[strategy] = {
            key: _summary([cast("float", row[key]) for row in rows])
            for key in (
                "e2e_seconds",
                "executor_seconds",
                "model_seconds",
                "assembly_scatter_seconds",
            )
        }
        cast("dict[str, JsonValue]", strategies[strategy]).update(
            {
                "decode_batch_sizes": rows[0]["decode_batch_sizes"],
                "padding_waste_ratio": rows[0]["padding_waste_ratio"],
                "all_resources_released": all(
                    row["all_resources_released"] is True for row in rows
                ),
            }
        )
    return {
        "schema_version": 1,
        "benchmark": "paged_attention_cpu_descriptive",
        "config": asdict(resolved),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "model": asdict(model.config),
        "workload": {
            "request_count": len(_requests()),
            "prompt_lengths": [len(request.prompt_tokens) for request in _requests()],
            "max_new_tokens": 4,
            "block_tokens": 4,
        },
        "generated_tokens": expected_tokens,
        "strategies": strategies,
        "cache_access": _cache_access_benchmark(
            model,
            iterations=resolved.cache_access_iterations,
        ),
        "verdict": "descriptive_only",
        "speedup_claim": False,
        "caveat": _CAVEAT,
    }


def write_paged_attention_benchmark(
    output_path: Path,
    *,
    config: PagedAttentionBenchmarkConfig | None = None,
) -> Path:
    """Write one LF-stable benchmark document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = run_paged_attention_benchmark(config=config)
    _ = output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path
