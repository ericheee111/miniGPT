"""Run fresh-process Stage 14 Automatic Prefix Caching CPU comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import psutil
import torch

from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    PagedAttentionExecutor,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from minigpt.data import JsonValue

Strategy: TypeAlias = Literal["paged_direct", "paged_direct_apc"]
_STRATEGIES: tuple[Strategy, ...] = ("paged_direct", "paged_direct_apc")
_PREFIX_EVENTS = frozenset(
    {
        EngineEventType.PREFIX_LOOKUP,
        EngineEventType.PREFIX_HIT,
        EngineEventType.PREFIX_PROMOTE,
        EngineEventType.PREFIX_EVICT,
    }
)
_CV_LIMIT = 0.10
_MIN_REPEATS = 3


@dataclass(frozen=True, slots=True)
class PrefixCacheBenchmarkConfig:
    """Configure fresh-process samples and strict timing stability."""

    warmups: int = 0
    repeats: int = 3
    cv_limit: float = _CV_LIMIT

    def __post_init__(self) -> None:
        """Reject insufficient or non-finite benchmark policy."""
        if isinstance(self.warmups, bool) or self.warmups < 0:
            reason = "warmups must be a non-negative integer"
            raise ValueError(reason)
        if isinstance(self.repeats, bool) or self.repeats < _MIN_REPEATS:
            reason = "repeats must be an integer of at least three"
            raise ValueError(reason)
        if not math.isfinite(self.cv_limit) or not 0.0 < self.cv_limit <= 1.0:
            reason = "cv_limit must be finite and in (0, 1]"
            raise ValueError(reason)


@dataclass(frozen=True, slots=True)
class _Workload:
    name: str
    phases: tuple[tuple[tuple[int, ...], ...], ...]
    num_blocks: int = 16


def _workloads() -> tuple[_Workload, ...]:
    common = (1, 2, 3, 4, 5, 6)
    return (
        _Workload("exact_repeated_prompt", (((1, 2, 3, 4),), ((1, 2, 3, 4),))),
        _Workload(
            "common_prefix_different_suffix",
            (((*common, 7),), ((*common, 8),), ((*common, 9),)),
        ),
        _Workload(
            "concurrent_same_prefix",
            (((1, 2, 3, 4),), ((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4))),
        ),
        _Workload("partial_block_prefix", (((1, 2, 3, 4),), ((1, 2, 3, 4, 5),))),
        _Workload(
            "low_reuse_random_prompts",
            tuple(((index, index + 1, index + 2, index + 3),) for index in range(1, 9, 2)),
        ),
        _Workload(
            "eviction_pressure",
            tuple(((index, index + 1, index + 2, index + 3),) for index in range(1, 13, 2)),
            num_blocks=6,
        ),
    )


def _model() -> GPT:
    original_state = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260809)
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


def _namespace(model: GPT, block_tokens: int) -> PrefixCacheNamespace:
    config_identity = hashlib.sha256(
        json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage14-benchmark-seed-20260809",
        model_config_identity=config_identity,
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=block_tokens,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _generator_hash(engine: ServingEngine, request_id: str) -> str:
    state = engine.request_state(request_id).generator.get_state()
    values = cast("list[int]", state.tolist())  # pyright: ignore[reportUnknownMemberType]
    return hashlib.sha256(bytes(values)).hexdigest()


def _peak_rss_bytes() -> int:
    memory = psutil.Process().memory_info()
    return int(getattr(memory, "peak_wset", memory.rss))


def run_prefix_cache_worker(workload_name: str, strategy: Strategy) -> dict[str, JsonValue]:
    """Run one isolated strategy/workload sample for the subprocess protocol."""
    workload = next(item for item in _workloads() if item.name == workload_name)
    torch.set_num_threads(1)
    model = _model()
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=workload.num_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model, paged.block_tokens)
        if strategy == "paged_direct_apc"
        else None,
    )
    origin = time.perf_counter()
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=4, max_cached_tokens=64),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(model, pool),
        paged_cache_pool=pool,
        clock=lambda: time.perf_counter() - origin,
    )
    request_ids: list[str] = []
    started = time.perf_counter()
    for phase_index, phase in enumerate(workload.phases):
        for request_index, prompt in enumerate(phase):
            request_id = f"{phase_index}-{request_index}"
            request_ids.append(request_id)
            engine.submit(
                GenerationRequest(
                    request_id=request_id,
                    prompt_tokens=prompt,
                    max_new_tokens=2,
                    seed=900 + phase_index * 10 + request_index,
                )
            )
        for _ in range(32):
            if engine.is_idle:
                break
            engine.tick()
        if not engine.is_idle:
            reason = f"prefix cache benchmark phase did not become idle: {workload.name}"
            raise RuntimeError(reason)
    elapsed = time.perf_counter() - started
    metrics = engine.metrics()
    ttft_values: list[float] = []
    for request_id in request_ids:
        value = engine.request_metrics(request_id).time_to_first_token_seconds
        if value is None:
            reason = "completed benchmark request omitted TTFT"
            raise RuntimeError(reason)
        ttft_values.append(value)
    ttft = statistics.median(ttft_values)
    logical_events = [
        [
            event.event_type.value,
            event.request_id,
            event.status.value,
            event.token_id,
            event.used_fallback,
        ]
        for event in engine.events
        if event.event_type not in _PREFIX_EVENTS
    ]
    result = cast(
        "dict[str, JsonValue]",
        {
            "workload": workload.name,
            "strategy": strategy,
            "e2e_seconds": elapsed,
            "ttft_seconds": ttft,
            "request_throughput_per_second": len(request_ids) / elapsed,
            "token_throughput_per_second": metrics.generated_tokens / elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "generated_tokens": {
                request_id: engine.request_state(request_id).generated_tokens
                for request_id in request_ids
            },
            "generator_state_hashes": {
                request_id: _generator_hash(engine, request_id) for request_id in request_ids
            },
            "terminal_states": {
                request_id: engine.request_state(request_id).status.value
                for request_id in request_ids
            },
            "logical_events": logical_events,
            "prefix_lookup_requests": metrics.prefix_lookup_requests,
            "prefix_hit_requests": metrics.prefix_hit_requests,
            "prefix_hit_tokens": metrics.prefix_hit_tokens,
            "prefix_miss_tokens": metrics.prefix_miss_tokens,
            "prefill_tokens_computed": metrics.prefill_tokens_computed,
            "avoided_prefill_tokens": metrics.avoided_prefill_tokens,
            "reused_blocks": metrics.prefix_hit_blocks,
            "evictions": metrics.prefix_cache_evictions,
        },
    )
    engine.release_all_cache_resources()
    pool.verify_invariants()
    released = pool.metrics()
    result["all_resources_released"] = (
        released.free_blocks == released.total_blocks
        and released.reserved_blocks == 0
        and released.active_shared_references == 0
    )
    return result


def _fresh_process(workload: str, strategy: Strategy) -> dict[str, JsonValue]:
    environment = os.environ.copy()
    environment.update({"PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "minigpt.prefix_cache_benchmark",
            "--worker",
            "--workload",
            workload,
            "--strategy",
            strategy,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    return cast("dict[str, JsonValue]", json.loads(completed.stdout))


def _summary(values: Sequence[float]) -> dict[str, JsonValue]:
    median = statistics.median(values)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "samples": list(values),
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "cv": deviation / mean if mean else math.inf,
    }


def _measurement_summary(rows: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    timings: dict[str, JsonValue] = {
        key: _summary([float(cast("int | float", row[key])) for row in rows])
        for key in (
            "e2e_seconds",
            "ttft_seconds",
            "request_throughput_per_second",
            "token_throughput_per_second",
            "peak_rss_bytes",
        )
    }
    first = rows[0]
    for key in (
        "prefix_lookup_requests",
        "prefix_hit_requests",
        "prefix_hit_tokens",
        "prefix_miss_tokens",
        "prefill_tokens_computed",
        "avoided_prefill_tokens",
        "reused_blocks",
        "evictions",
        "all_resources_released",
    ):
        timings[key] = first[key]
    prompt_tokens = cast("int", first["prefix_hit_tokens"]) + cast(
        "int", first["prefix_miss_tokens"]
    )
    lookup_requests = cast("int", first["prefix_lookup_requests"])
    hit_requests = cast("int", first["prefix_hit_requests"])
    timings["prefix_hit_request_ratio"] = hit_requests / lookup_requests if lookup_requests else 0.0
    timings["prefix_hit_token_ratio"] = (
        cast("int", first["prefix_hit_tokens"]) / prompt_tokens if prompt_tokens else 0.0
    )
    return timings


def _strict_verdict(
    direct: dict[str, JsonValue],
    apc: dict[str, JsonValue],
    *,
    cv_limit: float,
) -> str:
    direct_e2e = cast("dict[str, JsonValue]", direct["e2e_seconds"])
    apc_e2e = cast("dict[str, JsonValue]", apc["e2e_seconds"])
    if cast("float", direct_e2e["cv"]) > cv_limit or cast("float", apc_e2e["cv"]) > cv_limit:
        return "not_comparable"
    return (
        "pass" if cast("float", apc_e2e["median"]) < cast("float", direct_e2e["median"]) else "fail"
    )


def run_prefix_cache_benchmark(
    *,
    config: PrefixCacheBenchmarkConfig | None = None,
) -> dict[str, JsonValue]:
    """Run all fresh-process workloads and apply a conservative strict verdict."""
    resolved = config or PrefixCacheBenchmarkConfig()
    workload_documents: dict[str, JsonValue] = {}
    correctness = True
    verdicts: list[str] = []
    for workload in _workloads():
        for _ in range(resolved.warmups):
            for strategy in _STRATEGIES:
                _ = _fresh_process(workload.name, strategy)
        measurements = {
            strategy: [_fresh_process(workload.name, strategy) for _ in range(resolved.repeats)]
            for strategy in _STRATEGIES
        }
        expected = measurements["paged_direct"][0]
        contracts = (
            "generated_tokens",
            "generator_state_hashes",
            "terminal_states",
            "logical_events",
        )
        workload_correct = all(
            row[key] == expected[key]
            for strategy in _STRATEGIES
            for row in measurements[strategy]
            for key in contracts
        )
        correctness = correctness and workload_correct
        summaries: dict[str, JsonValue] = {
            strategy: _measurement_summary(measurements[strategy]) for strategy in _STRATEGIES
        }
        verdict = _strict_verdict(
            cast("dict[str, JsonValue]", summaries["paged_direct"]),
            cast("dict[str, JsonValue]", summaries["paged_direct_apc"]),
            cv_limit=resolved.cv_limit,
        )
        verdicts.append(verdict)
        workload_documents[workload.name] = cast(
            "dict[str, JsonValue]",
            {
                "correctness_equivalent": workload_correct,
                "strategies": summaries,
                "strict_verdict": verdict,
            },
        )
    overall = (
        "fail"
        if not correctness or "fail" in verdicts
        else "not_comparable"
        if "not_comparable" in verdicts
        else "pass"
    )
    return {
        "schema_version": 1,
        "benchmark": "automatic_prefix_caching_cpu_fresh_process",
        "comparison": "paged direct vs paged direct + APC",
        "config": asdict(resolved),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads_per_worker": 1,
        },
        "workloads": workload_documents,
        "correctness_equivalent": correctness,
        "strict_verdict": overall,
        "wall_clock_performance_improvement": overall == "pass",
        "claim_policy": "wall-clock improvement is claimed only when strict_verdict is pass",
        "implementation": "Python/PyTorch reference implementation; no fused PagedAttention kernel",
    }


def write_prefix_cache_benchmark(
    output_path: Path,
    *,
    config: PrefixCacheBenchmarkConfig | None = None,
) -> Path:
    """Write one LF-stable raw benchmark document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = run_prefix_cache_benchmark(config=config)
    _ = output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--worker", action="store_true")
    _ = parser.add_argument("--workload", choices=tuple(item.name for item in _workloads()))
    _ = parser.add_argument("--strategy", choices=_STRATEGIES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the private fresh-process worker protocol."""
    arguments = _parser().parse_args(argv)
    if not cast("bool", arguments.worker):
        reason = "prefix_cache_benchmark module is worker-only; use benchmark_prefix_cache.py"
        raise ValueError(reason)
    workload = cast("str | None", arguments.workload)
    strategy = cast("Strategy | None", arguments.strategy)
    if workload is None or strategy is None:
        reason = "worker mode requires --workload and --strategy"
        raise ValueError(reason)
    print(json.dumps(run_prefix_cache_worker(workload, strategy), sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
