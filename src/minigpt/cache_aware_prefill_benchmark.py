"""Run fresh-process Stage 15 sequential-versus-batched APC CPU comparisons."""

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
    APCPrefillStrategy,
    EngineConfig,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from minigpt.data import JsonValue

Strategy: TypeAlias = Literal["apc_sequential", "apc_batched"]
_STRATEGIES: tuple[Strategy, ...] = ("apc_sequential", "apc_batched")
_MIN_REPEATS = 3
_CV_LIMIT = 0.10


@dataclass(frozen=True, slots=True)
class CacheAwarePrefillBenchmarkConfig:
    """Configure fresh-process samples and strict timing stability."""

    warmups: int = 1
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
    max_padding_ratio: float = 1.0


def _workloads() -> tuple[_Workload, ...]:
    shared = (1, 2, 3, 4, 5, 6, 7, 8)
    mixed_four = (9, 10, 11, 12)
    mixed_eight = (13, 14, 15, 16, 17, 18, 19, 20)
    mixed_twelve = (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    return (
        _Workload(
            "repeated_prefix_short_suffix",
            ((shared,), tuple((*shared, token) for token in (9, 10, 11, 12))),
        ),
        _Workload(
            "repeated_prefix_variable_suffix",
            ((shared,), ((*shared, 9), (*shared, 10, 11), (*shared, 12, 13, 14, 15))),
        ),
        _Workload(
            "mixed_prefix_lengths",
            (
                (mixed_four, mixed_eight, mixed_twelve),
                (
                    (*mixed_four, 33, 34, 35, 36),
                    (*mixed_eight, 33, 34, 35, 36),
                    (*mixed_twelve, 33, 34, 35, 36),
                ),
            ),
        ),
        _Workload(
            "exact_hits_mixed_with_suffix_hits",
            ((shared,), (shared, (*shared, 9), (*shared, 10), (*shared, 11))),
        ),
        _Workload(
            "concurrent_same_prefix",
            (tuple((41, 42, 43, 44, 45, 46, 47, 48) for _ in range(4)),),
        ),
        _Workload(
            "low_reuse_random_prompts",
            (tuple(tuple((row * 7 + column) % 61 for column in range(8)) for row in range(1, 7)),),
        ),
        _Workload(
            "padding_pressure",
            (
                (shared,),
                (
                    (*shared, 9),
                    (*shared, 10, 11),
                    (*shared, 12, 13, 14, 15),
                    (*shared, 16, 17, 18, 19, 20, 21, 22, 23),
                ),
            ),
            max_padding_ratio=0.25,
        ),
    )


def _model() -> GPT:
    original_state = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260809)
        model = GPT(
            GPTConfig(
                vocab_size=64,
                block_size=32,
                n_layer=1,
                n_head=2,
                n_embd=16,
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
        model_checkpoint_identity="stage15-benchmark-seed-20260809",
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


def run_cache_aware_prefill_worker(
    workload_name: str,
    strategy: Strategy,
) -> dict[str, JsonValue]:
    """Run one isolated strategy/workload sample for the subprocess protocol."""
    workload = next(item for item in _workloads() if item.name == workload_name)
    torch.set_num_threads(1)
    model = _model()
    paged = PagedKVCacheConfig(block_tokens=4, num_blocks=128)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model, paged.block_tokens),
    )
    origin = time.perf_counter()
    executor = PagedAttentionExecutor(
        model,
        pool,
        prefill_config=PrefillBatchConfig(
            max_batch_size=8,
            max_batch_tokens=128,
            max_padding_ratio=workload.max_padding_ratio,
        ),
        prefix_prefill_strategy=(
            APCPrefillStrategy.SEQUENTIAL
            if strategy == "apc_sequential"
            else APCPrefillStrategy.BATCHED
        ),
    )
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=8, max_cached_tokens=256),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=executor,
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
                    seed=1500 + phase_index * 100 + request_index,
                    arrival_time=time.perf_counter() - origin,
                )
            )
        for _ in range(128):
            if engine.is_idle:
                break
            engine.tick()
        if not engine.is_idle:
            reason = f"cache-aware prefill workload did not become idle: {workload.name}"
            raise RuntimeError(reason)
    elapsed = time.perf_counter() - started
    metrics = engine.metrics()
    ttft_values = [
        engine.request_metrics(request_id).time_to_first_token_seconds for request_id in request_ids
    ]
    if any(value is None for value in ttft_values):
        reason = "completed benchmark request omitted TTFT"
        raise RuntimeError(reason)
    ttft = statistics.median(cast("list[float]", ttft_values))
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
            "logical_events": [
                [
                    event.event_type.value,
                    event.request_id,
                    event.status.value,
                    event.token_id,
                    event.detail,
                    event.used_fallback,
                ]
                for event in engine.events
            ],
            "cache_aware_prefill_batches": metrics.cache_aware_prefill_batches,
            "cache_aware_prefill_model_calls": metrics.cache_aware_prefill_model_calls,
            "suffix_prefill_batch_sizes": list(metrics.suffix_prefill_batch_sizes),
            "average_suffix_prefill_batch_size": metrics.average_suffix_prefill_batch_size,
            "max_suffix_prefill_batch_size": metrics.max_suffix_prefill_batch_size,
            "suffix_useful_tokens": metrics.suffix_useful_tokens,
            "suffix_padded_tokens": metrics.suffix_padded_tokens,
            "suffix_padding_waste_ratio": metrics.suffix_padding_waste_ratio,
            "exact_cache_hit_requests": metrics.exact_cache_hit_requests,
            "batched_suffix_requests": metrics.batched_suffix_requests,
            "prefix_hit_tokens": metrics.prefix_hit_tokens,
            "prefill_tokens_computed": metrics.prefill_tokens_computed,
            "avoided_prefill_tokens": metrics.avoided_prefill_tokens,
        },
    )
    engine.release_all_cache_resources()
    pool.verify_invariants()
    released = pool.metrics()
    result["all_resources_released"] = (
        released.free_blocks == released.total_blocks
        and released.reserved_blocks == 0
        and released.active_shared_references == 0
        and released.private_blocks == 0
    )
    return result


def _fresh_process(workload: str, strategy: Strategy) -> dict[str, JsonValue]:
    environment = os.environ.copy()
    environment.update({"PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "minigpt.cache_aware_prefill_benchmark",
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
    summary: dict[str, JsonValue] = {
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
        "cache_aware_prefill_batches",
        "cache_aware_prefill_model_calls",
        "suffix_prefill_batch_sizes",
        "average_suffix_prefill_batch_size",
        "max_suffix_prefill_batch_size",
        "suffix_useful_tokens",
        "suffix_padded_tokens",
        "suffix_padding_waste_ratio",
        "exact_cache_hit_requests",
        "batched_suffix_requests",
        "prefix_hit_tokens",
        "prefill_tokens_computed",
        "avoided_prefill_tokens",
        "all_resources_released",
    ):
        summary[key] = first[key]
    return summary


def _strict_verdict(
    sequential: dict[str, JsonValue],
    batched: dict[str, JsonValue],
    *,
    cv_limit: float,
) -> str:
    sequential_e2e = cast("dict[str, JsonValue]", sequential["e2e_seconds"])
    batched_e2e = cast("dict[str, JsonValue]", batched["e2e_seconds"])
    if (
        cast("float", sequential_e2e["cv"]) > cv_limit
        or cast("float", batched_e2e["cv"]) > cv_limit
    ):
        return "not_comparable"
    return (
        "pass"
        if cast("float", batched_e2e["median"]) < cast("float", sequential_e2e["median"])
        else "fail"
    )


def run_cache_aware_prefill_benchmark(
    *,
    config: CacheAwarePrefillBenchmarkConfig | None = None,
) -> dict[str, JsonValue]:
    """Run every fresh-process workload and apply a conservative strict verdict."""
    resolved = config or CacheAwarePrefillBenchmarkConfig()
    workload_documents: dict[str, JsonValue] = {}
    correctness = True
    structural_reduction = True
    verdicts: list[str] = []
    for workload in _workloads():
        for _ in range(resolved.warmups):
            for strategy in _STRATEGIES:
                _ = _fresh_process(workload.name, strategy)
        measurements = {
            strategy: [_fresh_process(workload.name, strategy) for _ in range(resolved.repeats)]
            for strategy in _STRATEGIES
        }
        expected = measurements["apc_sequential"][0]
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
        sequential_calls = cast(
            "int", measurements["apc_sequential"][0]["cache_aware_prefill_model_calls"]
        )
        batched_calls = cast(
            "int", measurements["apc_batched"][0]["cache_aware_prefill_model_calls"]
        )
        workload_reduction = batched_calls <= sequential_calls and (
            batched_calls < sequential_calls or sequential_calls <= 1
        )
        correctness = correctness and workload_correct
        structural_reduction = structural_reduction and workload_reduction
        summaries: dict[str, JsonValue] = {
            strategy: _measurement_summary(measurements[strategy]) for strategy in _STRATEGIES
        }
        verdict = _strict_verdict(
            cast("dict[str, JsonValue]", summaries["apc_sequential"]),
            cast("dict[str, JsonValue]", summaries["apc_batched"]),
            cv_limit=resolved.cv_limit,
        )
        verdicts.append(verdict)
        workload_documents[workload.name] = cast(
            "dict[str, JsonValue]",
            {
                "correctness_equivalent": workload_correct,
                "model_call_reduction": workload_reduction,
                "strategies": summaries,
                "strict_verdict": verdict,
            },
        )
    overall = (
        "fail"
        if not correctness or not structural_reduction or "fail" in verdicts
        else "not_comparable"
        if "not_comparable" in verdicts
        else "pass"
    )
    return {
        "schema_version": 1,
        "benchmark": "cache_aware_batched_paged_prefill_cpu_fresh_process",
        "comparison": "Stage 14 sequential APC vs Stage 15 batched APC",
        "config": asdict(resolved),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_num_threads_per_worker": 1,
        },
        "workloads": workload_documents,
        "correctness_equivalent": correctness,
        "model_call_reduction": structural_reduction,
        "strict_verdict": overall,
        "wall_clock_performance_improvement": overall == "pass",
        "claim_policy": "wall-clock improvement is claimed only when strict_verdict is pass",
        "implementation": "Python/PyTorch reference implementation; no fused kernel",
    }


def write_cache_aware_prefill_benchmark(
    output_path: Path,
    *,
    config: CacheAwarePrefillBenchmarkConfig | None = None,
) -> Path:
    """Write one LF-stable raw benchmark document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = run_cache_aware_prefill_benchmark(config=config)
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
        reason = (
            "cache_aware_prefill_benchmark is worker-only; use benchmark_cache_aware_prefill.py"
        )
        raise ValueError(reason)
    workload = cast("str | None", arguments.workload)
    strategy = cast("Strategy | None", arguments.strategy)
    if workload is None or strategy is None:
        reason = "worker mode requires --workload and --strategy"
        raise ValueError(reason)
    print(json.dumps(run_cache_aware_prefill_worker(workload, strategy), sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
