"""Load and execute deterministic offline serving-control-plane workloads."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Never, TypeAlias, cast

import torch
import yaml
from typing_extensions import override

from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    APCPrefillStrategy,
    ContinuousDecodeExecutor,
    ContinuousExecutor,
    EngineConfig,
    EngineEvent,
    EngineEventType,
    EngineMetrics,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    PrefillBatchEvent,
    PrefillBatchObservation,
    PrefillExecutionMode,
    ReferenceExecutor,
    RequestMetrics,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig, InvalidModelConfigError

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
ConfigValue: TypeAlias = (
    str | int | float | bool | list["ConfigValue"] | dict[str, "ConfigValue"] | None
)
ConfigMapping: TypeAlias = dict[str, ConfigValue]

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "executor",
        "kv_cache_backend",
        "kv_cache",
        "prefix_cache",
        "apc_prefill_strategy",
        "scenario_name",
        "model_seed",
        "tick_seconds",
        "executor_clock_step_seconds",
        "max_ticks",
        "output_dir",
        "vocab_size",
        "model",
        "prefill",
        "scheduler",
        "requests",
    }
)
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS - {
    "executor",
    "prefill",
    "kv_cache_backend",
    "kv_cache",
    "prefix_cache",
    "apc_prefill_strategy",
}
_MODEL_KEYS = frozenset({"block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"})
_SCHEDULER_KEYS = frozenset(
    {
        "max_active_requests",
        "max_cached_tokens",
        "max_scheduled_tokens",
        "prefill_chunk_tokens",
    }
)
_SCHEDULER_REQUIRED_KEYS = frozenset({"max_active_requests", "max_cached_tokens"})
_PREFILL_KEYS = frozenset({"max_batch_size", "max_batch_tokens", "max_padding_ratio"})
_KV_CACHE_KEYS = frozenset({"block_tokens", "num_blocks"})
_PREFIX_CACHE_KEYS = frozenset({"enabled"})
_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "arrival_time",
        "prompt_tokens",
        "prompt_length",
        "max_new_tokens",
        "temperature",
        "top_k",
        "seed",
        "cancellation_time",
    }
)


@dataclass(frozen=True, slots=True)
class InvalidSimulatorConfigError(ValueError):
    """Report malformed simulator input with its source path."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the source and failed constraint."""
        return f"invalid serving simulator config {self.source}: {self.reason}"


def _invalid(source: Path, reason: str) -> Never:
    raise InvalidSimulatorConfigError(source, reason)


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """Define one deterministic model, scheduler, clock, and arrival workload."""

    schema_version: int
    executor: SimulatorExecutor
    kv_cache_backend: KVCacheBackend
    paged_kv_cache: PagedKVCacheConfig | None
    prefix_cache_enabled: bool
    apc_prefill_strategy: APCPrefillStrategy
    scenario_name: str
    model_seed: int
    tick_seconds: float
    executor_clock_step_seconds: float
    max_ticks: int
    output_dir: Path
    model: GPTConfig
    scheduler: SchedulerConfig
    prefill: PrefillBatchConfig | None
    requests: tuple[GenerationRequest, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Return stable artifacts and terminal metrics for one workload."""

    output_dir: Path
    output_paths: tuple[Path, ...]
    metrics: EngineMetrics
    generated_tokens: dict[str, tuple[int, ...]]
    request_statuses: dict[str, RequestStatus]
    request_metrics: dict[str, RequestMetrics]
    generator_state_hashes: dict[str, str]
    admission_order: tuple[str, ...]
    events: tuple[EngineEvent, ...]
    prefill_events: tuple[PrefillBatchEvent, ...]
    prefill_observations: tuple[PrefillBatchObservation, ...]


class SimulatorExecutor(StrEnum):
    """Select per-request, decode-batched, or fully continuous execution."""

    REFERENCE = "reference"
    CONTINUOUS_DECODE = "continuous_decode"
    CONTINUOUS = "continuous"
    PAGED_ATTENTION = "paged_attention"


@dataclass(frozen=True, slots=True)
class ExecutorEquivalenceResult:
    """Return all three simulations after complete logical-contract validation."""

    reference: SimulationResult
    continuous_decode: SimulationResult
    continuous: SimulationResult
    equivalent: bool
    checked_contracts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheBackendEquivalenceResult:
    """Return dense/paged simulations after logical-contract validation."""

    dense: SimulationResult
    paged: SimulationResult
    equivalent: bool
    checked_contracts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PagedAttentionEquivalenceResult:
    """Return dense, materialized-paged, and direct-paged correctness runs."""

    dense: SimulationResult
    materialized: SimulationResult
    direct: SimulationResult
    equivalent: bool
    checked_contracts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrefixCacheEquivalenceResult:
    """Return direct-paged/APC runs after logical-contract validation."""

    direct: SimulationResult
    automatic_prefix_cache: SimulationResult
    equivalent: bool
    checked_contracts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheAwarePrefillEquivalenceResult:
    """Return sequential/batched APC runs after logical-contract validation."""

    sequential: SimulationResult
    batched: SimulationResult
    equivalent: bool
    checked_contracts: tuple[str, ...]


@dataclass(slots=True)
class StepClock:
    """Advance by a fixed amount on every clock observation."""

    step_seconds: float
    current_seconds: float = 0.0

    def __call__(self) -> float:
        """Return current logical time and advance one deterministic step."""
        value = self.current_seconds
        self.current_seconds += self.step_seconds
        return value


def _mapping(value: object, source: Path, context: str) -> ConfigMapping:
    if not isinstance(value, dict):
        _invalid(source, f"{context} must be a mapping")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        _invalid(source, f"{context} keys must be strings")
    return cast("ConfigMapping", raw)


def _exact_keys(
    document: ConfigMapping,
    expected: frozenset[str],
    source: Path,
    context: str,
) -> None:
    missing = expected - set(document)
    unexpected = set(document) - expected
    if missing:
        _invalid(source, f"{context} missing key {min(missing)!r}")
    if unexpected:
        _invalid(source, f"{context} has unexpected key {min(unexpected)!r}")


def _top_level_keys(document: ConfigMapping, source: Path) -> None:
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(document)
    unexpected = set(document) - _TOP_LEVEL_KEYS
    if missing:
        _invalid(source, f"document missing key {min(missing)!r}")
    if unexpected:
        _invalid(source, f"document has unexpected key {min(unexpected)!r}")


def _executor(document: ConfigMapping, source: Path) -> SimulatorExecutor:
    raw = document.get("executor", SimulatorExecutor.REFERENCE.value)
    if not isinstance(raw, str):
        _invalid(source, "executor must be a string")
    try:
        return SimulatorExecutor(raw)
    except ValueError:
        choices = ", ".join(executor.value for executor in SimulatorExecutor)
        _invalid(source, f"executor must be one of: {choices}")


def _cache_backend(document: ConfigMapping, source: Path) -> KVCacheBackend:
    raw = document.get("kv_cache_backend", KVCacheBackend.DENSE.value)
    if not isinstance(raw, str):
        _invalid(source, "kv_cache_backend must be a string")
    try:
        return KVCacheBackend(raw)
    except ValueError:
        choices = ", ".join(backend.value for backend in KVCacheBackend)
        _invalid(source, f"kv_cache_backend must be one of: {choices}")


def _integer(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> int:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(source, f"{key} must be an integer")
    if positive and value <= 0:
        _invalid(source, f"{key} must be positive")
    if non_negative and value < 0:
        _invalid(source, f"{key} must be non-negative")
    return value


def _number(
    document: ConfigMapping,
    key: str,
    source: Path,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(source, f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        _invalid(source, f"{key} must be finite")
    if positive and number <= 0.0:
        _invalid(source, f"{key} must be positive")
    if non_negative and number < 0.0:
        _invalid(source, f"{key} must be non-negative")
    return number


def _string(document: ConfigMapping, key: str, source: Path) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        _invalid(source, f"{key} must be a non-empty string")
    return value


def _model(document: ConfigMapping, source: Path, *, vocab_size: int) -> GPTConfig:
    raw = _mapping(document["model"], source, "model")
    _exact_keys(raw, _MODEL_KEYS, source, "model")
    bias = raw["bias"]
    if not isinstance(bias, bool):
        _invalid(source, "model.bias must be a boolean")
    try:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=_integer(raw, "block_size", source, positive=True),
            n_layer=_integer(raw, "n_layer", source, positive=True),
            n_head=_integer(raw, "n_head", source, positive=True),
            n_embd=_integer(raw, "n_embd", source, positive=True),
            dropout=_number(raw, "dropout", source, non_negative=True),
            bias=bias,
        )
    except InvalidModelConfigError as error:
        _invalid(source, str(error))


def _scheduler(document: ConfigMapping, source: Path) -> SchedulerConfig:
    raw = _mapping(document["scheduler"], source, "scheduler")
    missing = _SCHEDULER_REQUIRED_KEYS - set(raw)
    unexpected = set(raw) - _SCHEDULER_KEYS
    if missing:
        _invalid(source, f"scheduler missing key {min(missing)!r}")
    if unexpected:
        _invalid(source, f"scheduler has unexpected key {min(unexpected)!r}")
    max_scheduled = (
        _integer(raw, "max_scheduled_tokens", source, positive=True)
        if "max_scheduled_tokens" in raw
        else None
    )
    chunk_size = (
        _integer(raw, "prefill_chunk_tokens", source, positive=True)
        if "prefill_chunk_tokens" in raw
        else None
    )
    return SchedulerConfig(
        max_active_requests=_integer(raw, "max_active_requests", source, positive=True),
        max_cached_tokens=_integer(raw, "max_cached_tokens", source, positive=True),
        max_scheduled_tokens=max_scheduled,
        prefill_chunk_tokens=chunk_size,
    )


def _prefill(document: ConfigMapping, source: Path) -> PrefillBatchConfig | None:
    if "prefill" not in document:
        return None
    raw = _mapping(document["prefill"], source, "prefill")
    _exact_keys(raw, _PREFILL_KEYS, source, "prefill")
    return PrefillBatchConfig(
        max_batch_size=_integer(raw, "max_batch_size", source, positive=True),
        max_batch_tokens=_integer(raw, "max_batch_tokens", source, positive=True),
        max_padding_ratio=_number(raw, "max_padding_ratio", source, non_negative=True),
    )


def _paged_kv_cache(
    document: ConfigMapping,
    source: Path,
    *,
    backend: KVCacheBackend,
) -> PagedKVCacheConfig | None:
    if "kv_cache" not in document:
        if backend is KVCacheBackend.PAGED:
            _invalid(source, "paged kv_cache_backend requires kv_cache")
        return None
    raw = _mapping(document["kv_cache"], source, "kv_cache")
    _exact_keys(raw, _KV_CACHE_KEYS, source, "kv_cache")
    return PagedKVCacheConfig(
        block_tokens=_integer(raw, "block_tokens", source, positive=True),
        num_blocks=_integer(raw, "num_blocks", source, positive=True),
    )


def _prefix_cache_enabled(
    document: ConfigMapping,
    source: Path,
    *,
    backend: KVCacheBackend,
    executor: SimulatorExecutor,
) -> bool:
    if "prefix_cache" not in document:
        return False
    raw = _mapping(document["prefix_cache"], source, "prefix_cache")
    _exact_keys(raw, _PREFIX_CACHE_KEYS, source, "prefix_cache")
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        _invalid(source, "prefix_cache.enabled must be a boolean")
    if enabled and (
        backend is not KVCacheBackend.PAGED or executor is not SimulatorExecutor.PAGED_ATTENTION
    ):
        _invalid(source, "enabled prefix_cache requires paged_attention with paged KV cache")
    return enabled


def _apc_prefill_strategy(document: ConfigMapping, source: Path) -> APCPrefillStrategy:
    raw = document.get("apc_prefill_strategy", APCPrefillStrategy.SEQUENTIAL.value)
    if not isinstance(raw, str):
        _invalid(source, "apc_prefill_strategy must be a string")
    try:
        return APCPrefillStrategy(raw)
    except ValueError:
        choices = ", ".join(strategy.value for strategy in APCPrefillStrategy)
        _invalid(source, f"apc_prefill_strategy must be one of: {choices}")


def _optional_integer(document: ConfigMapping, key: str, source: Path) -> int | None:
    value = document[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _invalid(source, f"{key} must be null or a positive integer")
    return value


def _optional_number(document: ConfigMapping, key: str, source: Path) -> float | None:
    value = document[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(source, f"{key} must be null or a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        _invalid(source, f"{key} must be null or finite and non-negative")
    return number


def _prompt_tokens(
    document: ConfigMapping,
    source: Path,
    *,
    seed: int,
    vocab_size: int,
) -> tuple[int, ...]:
    raw_tokens = document["prompt_tokens"]
    raw_length = document["prompt_length"]
    if (raw_tokens is None) == (raw_length is None):
        _invalid(source, "each request must specify exactly one of prompt_tokens or prompt_length")
    if raw_tokens is not None:
        if not isinstance(raw_tokens, list) or not raw_tokens:
            _invalid(source, "prompt_tokens must be null or a non-empty list")
        values = cast("list[object]", raw_tokens)
        if any(type(token) is not int or token < 0 or token >= vocab_size for token in values):
            _invalid(source, "prompt_tokens must contain in-vocabulary integers")
        return cast("tuple[int, ...]", tuple(values))
    if isinstance(raw_length, bool) or not isinstance(raw_length, int) or raw_length <= 0:
        _invalid(source, "prompt_length must be null or a positive integer")
    return tuple((seed + index) % vocab_size for index in range(raw_length))


def _requests(
    document: ConfigMapping,
    source: Path,
    *,
    vocab_size: int,
) -> tuple[GenerationRequest, ...]:
    raw_requests = document["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        _invalid(source, "requests must be a non-empty list")
    requests: list[GenerationRequest] = []
    for index, item in enumerate(cast("list[object]", raw_requests)):
        context = f"requests[{index}]"
        raw = _mapping(item, source, context)
        _exact_keys(raw, _REQUEST_KEYS, source, context)
        seed = _integer(raw, "seed", source, non_negative=True)
        request = GenerationRequest(
            request_id=_string(raw, "request_id", source),
            prompt_tokens=_prompt_tokens(raw, source, seed=seed, vocab_size=vocab_size),
            max_new_tokens=_integer(raw, "max_new_tokens", source, non_negative=True),
            temperature=_number(raw, "temperature", source, positive=True),
            top_k=_optional_integer(raw, "top_k", source),
            seed=seed,
            arrival_time=_number(raw, "arrival_time", source, non_negative=True),
            cancellation_time=_optional_number(raw, "cancellation_time", source),
        )
        requests.append(request)
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        _invalid(source, "request IDs must be unique")
    return tuple(requests)


def load_simulator_config(source: Path) -> SimulatorConfig:
    """Load one strict JSON/YAML workload with resolved prompt tokens."""
    try:
        document = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), source, "document")
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        _invalid(source, str(error))
    _top_level_keys(document, source)
    schema_version = _integer(document, "schema_version", source, positive=True)
    if schema_version != 1:
        _invalid(source, "schema_version must equal 1")
    vocab_size = _integer(document, "vocab_size", source, positive=True)
    backend = _cache_backend(document, source)
    executor = _executor(document, source)
    if executor is SimulatorExecutor.PAGED_ATTENTION and backend is not KVCacheBackend.PAGED:
        _invalid(source, "paged_attention executor requires paged kv_cache_backend")
    prefix_cache_enabled = _prefix_cache_enabled(
        document,
        source,
        backend=backend,
        executor=executor,
    )
    paged_kv_cache = _paged_kv_cache(document, source, backend=backend)
    model = _model(document, source, vocab_size=vocab_size)
    scheduler = _scheduler(document, source)
    if scheduler.prefill_chunk_tokens is not None:
        if (
            executor is not SimulatorExecutor.PAGED_ATTENTION
            or backend is not KVCacheBackend.PAGED
            or paged_kv_cache is None
        ):
            _invalid(source, "chunked prefill requires paged_attention with paged KV cache")
        if scheduler.prefill_chunk_tokens % paged_kv_cache.block_tokens:
            _invalid(source, "prefill_chunk_tokens must align to kv_cache.block_tokens")
        minimum_budget = max(
            model.block_size, scheduler.max_active_requests - 1 + paged_kv_cache.block_tokens
        )
        if (
            scheduler.max_scheduled_tokens is None
            or scheduler.max_scheduled_tokens < minimum_budget
        ):
            _invalid(source, "max_scheduled_tokens is too small for chunked prefill")
    return SimulatorConfig(
        schema_version=schema_version,
        executor=executor,
        kv_cache_backend=backend,
        paged_kv_cache=paged_kv_cache,
        prefix_cache_enabled=prefix_cache_enabled,
        apc_prefill_strategy=_apc_prefill_strategy(document, source),
        scenario_name=_string(document, "scenario_name", source),
        model_seed=_integer(document, "model_seed", source, non_negative=True),
        tick_seconds=_number(document, "tick_seconds", source, positive=True),
        executor_clock_step_seconds=_number(
            document,
            "executor_clock_step_seconds",
            source,
            positive=True,
        ),
        max_ticks=_integer(document, "max_ticks", source, positive=True),
        output_dir=Path(_string(document, "output_dir", source)),
        model=model,
        scheduler=scheduler,
        prefill=_prefill(document, source),
        requests=_requests(document, source, vocab_size=vocab_size),
    )


def _build_model(config: SimulatorConfig) -> GPT:
    original_state = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(config.model_seed)
        model = GPT(config.model)
    finally:
        torch.set_rng_state(original_state)
    _ = model.eval()
    return model


def _identity_hash(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _generator_state_hash(engine: ServingEngine, request_id: str) -> str:
    state = engine.request_state(request_id).generator.get_state()
    values = cast("list[int]", state.tolist())  # pyright: ignore[reportUnknownMemberType]
    return hashlib.sha256(bytes(values)).hexdigest()


def _prefix_cache_namespace(config: SimulatorConfig, model: GPT) -> PrefixCacheNamespace | None:
    if not config.prefix_cache_enabled:
        return None
    paged = config.paged_kv_cache
    if paged is None:
        _invalid(Path("<runtime>"), "enabled prefix cache requires paged KV cache config")
    model_document = asdict(config.model)
    return PrefixCacheNamespace(
        model_checkpoint_identity=_identity_hash(
            {"model_seed": config.model_seed, "model_config": model_document}
        ),
        model_config_identity=_identity_hash(model_document),
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=paged.block_tokens,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _event_document(event: EngineEvent) -> dict[str, JsonValue]:
    return cast(
        "dict[str, JsonValue]",
        {
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "request_id": event.request_id,
            "status": event.status.value,
            "token_id": event.token_id,
            "detail": event.detail,
            "used_fallback": event.used_fallback,
            "active_requests": event.active_requests,
            "waiting_requests": event.waiting_requests,
            "cached_tokens": event.cached_tokens,
            "reserved_cache_tokens": event.reserved_cache_tokens,
        },
    )


def _request_document(engine: ServingEngine, request_id: str) -> dict[str, JsonValue]:
    state = engine.request_state(request_id)
    metrics = engine.request_metrics(request_id)
    return cast(
        "dict[str, JsonValue]",
        {
            "request_id": request_id,
            "status": state.status.value,
            "prompt_tokens": list(state.request.prompt_tokens),
            "generated_tokens": state.generated_tokens,
            "arrival_time": state.request.arrival_time,
            "admission_time": state.admission_time,
            "prefill_start_time": state.prefill_start_time,
            "first_token_time": state.first_token_time,
            "finish_time": state.finish_time,
            "queue_time_seconds": metrics.queue_time_seconds,
            "prefill_latency_seconds": metrics.prefill_latency_seconds,
            "time_to_first_token_seconds": metrics.time_to_first_token_seconds,
            "decode_latencies_seconds": list(metrics.decode_latencies_seconds),
            "time_per_output_token_seconds": metrics.time_per_output_token_seconds,
            "end_to_end_latency_seconds": metrics.end_to_end_latency_seconds,
            "failure_reason": metrics.failure_reason,
            "prefix_hit_blocks": metrics.prefix_hit_blocks,
            "prefix_hit_tokens": metrics.prefix_hit_tokens,
            "prefix_miss_tokens": metrics.prefix_miss_tokens,
            "prefill_tokens_computed": metrics.prefill_tokens_computed,
        },
    )


def _cache_aware_prefill_summary(
    observations: tuple[PrefillBatchObservation, ...],
) -> dict[str, JsonValue]:
    suffix_modes = {
        PrefillExecutionMode.SEQUENTIAL_APC_SUFFIX,
        PrefillExecutionMode.BATCHED_APC_SUFFIX,
    }
    apc = tuple(
        item
        for item in observations
        if item.execution_mode in {*suffix_modes, PrefillExecutionMode.EXACT_CACHE_HIT}
    )
    suffix = tuple(item for item in apc if item.execution_mode in suffix_modes)
    batched = tuple(
        item for item in suffix if item.execution_mode is PrefillExecutionMode.BATCHED_APC_SUFFIX
    )
    batch_sizes = [item.batch_size for item in suffix]
    useful_tokens = sum(item.useful_prompt_tokens for item in suffix)
    padded_tokens = sum(item.padded_prompt_tokens for item in suffix)
    return cast(
        "dict[str, JsonValue]",
        {
            "cache_aware_prefill_batches": len(batched),
            "cache_aware_prefill_model_calls": sum(item.model_calls for item in apc),
            "suffix_prefill_batch_sizes": batch_sizes,
            "average_suffix_prefill_batch_size": (
                sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
            ),
            "max_suffix_prefill_batch_size": max(batch_sizes, default=0),
            "suffix_useful_tokens": useful_tokens,
            "suffix_padded_tokens": padded_tokens,
            "suffix_padding_waste_ratio": (
                (padded_tokens - useful_tokens) / padded_tokens if padded_tokens else 0.0
            ),
            "exact_cache_hit_requests": sum(
                item.batch_size
                for item in apc
                if item.execution_mode is PrefillExecutionMode.EXACT_CACHE_HIT
            ),
            "batched_suffix_requests": sum(item.batch_size for item in batched),
        },
    )


def _summary_document(
    config: SimulatorConfig,
    metrics: EngineMetrics,
    observations: tuple[PrefillBatchObservation, ...],
) -> dict[str, JsonValue]:
    document = cast("dict[str, JsonValue]", asdict(metrics))
    document.update(
        {
            "schema_version": 1,
            "scenario_name": config.scenario_name,
            "claim": "logical serving correctness; wall-clock performance reported separately",
            "executor": config.executor.value,
            "kv_cache_backend": config.kv_cache_backend.value,
            "prefix_cache_enabled": config.prefix_cache_enabled,
            "apc_prefill_strategy": config.apc_prefill_strategy.value,
            "scheduling_level": (
                "iteration-level with tensor-level prefill and block-aware decode batching"
                if config.executor is SimulatorExecutor.PAGED_ATTENTION
                else (
                    "iteration-level with tensor-level prefill and decode batching"
                    if config.executor is SimulatorExecutor.CONTINUOUS
                    else (
                        "iteration-level with tensor-level decode batching"
                        if config.executor is SimulatorExecutor.CONTINUOUS_DECODE
                        else "iteration-level with per-request model execution"
                    )
                )
            ),
            "max_active_requests": config.scheduler.max_active_requests,
            "max_cached_tokens": config.scheduler.max_cached_tokens,
        }
    )
    document.update(_cache_aware_prefill_summary(observations))
    return document


def _write_json(path: Path, document: dict[str, JsonValue]) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_events(path: Path, events: tuple[EngineEvent, ...]) -> None:
    lines = [json.dumps(_event_document(event), sort_keys=True) for event in events]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_prefill_events(path: Path, events: tuple[PrefillBatchEvent, ...]) -> None:
    lines = [
        json.dumps(
            {
                "sequence": event.sequence,
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "request_ids": list(event.request_ids),
                "batch_size": event.batch_size,
                "useful_prompt_tokens": event.useful_prompt_tokens,
                "padded_prompt_tokens": event.padded_prompt_tokens,
                "padding_waste_ratio": event.padding_waste_ratio,
                "batch_failed": event.batch_failed,
            },
            sort_keys=True,
        )
        for event in events
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_prefill_observations(
    path: Path,
    observations: tuple[PrefillBatchObservation, ...],
) -> None:
    lines = [
        json.dumps(
            {
                **asdict(observation),
                "request_ids": list(observation.request_ids),
                "execution_mode": observation.execution_mode.value,
                "padding_waste_ratio": observation.padding_waste_ratio,
            },
            sort_keys=True,
        )
        for observation in observations
    ]
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_requests(
    path: Path,
    engine: ServingEngine,
    requests: tuple[GenerationRequest, ...],
) -> None:
    fieldnames: list[str] = [
        "request_id",
        "status",
        "prompt_tokens",
        "generated_tokens",
        "arrival_time",
        "admission_time",
        "prefill_start_time",
        "first_token_time",
        "finish_time",
        "queue_time_seconds",
        "prefill_latency_seconds",
        "time_to_first_token_seconds",
        "decode_latencies_seconds",
        "time_per_output_token_seconds",
        "end_to_end_latency_seconds",
        "failure_reason",
        "prefix_hit_blocks",
        "prefix_hit_tokens",
        "prefix_miss_tokens",
        "prefill_tokens_computed",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        _ = cast("object", writer.writeheader())
        for request in requests:
            row = _request_document(engine, request.request_id)
            for key in ("prompt_tokens", "generated_tokens", "decode_latencies_seconds"):
                row[key] = json.dumps(row[key], separators=(",", ":"))
            _ = cast("object", writer.writerow(row))


def _write_timeline(
    path: Path,
    events: tuple[EngineEvent, ...],
    scenario_name: str,
    executor: SimulatorExecutor,
) -> None:
    execution_note = (
        "Eligible prompts and decode rows are tensor-batched with dense padding."
        if executor is SimulatorExecutor.CONTINUOUS
        else (
            "Prefill per request; eligible decode rows use tensor-level dense batching."
            if executor is SimulatorExecutor.CONTINUOUS_DECODE
            else "Model prefill and decode calls both remain per request."
        )
    )
    lines = [
        f"# {scenario_name} timeline",
        "",
        "This is deterministic logical serving evidence, not canonical wall-clock benchmark data.",
        execution_note,
        "",
        "| Seq | Time | Event | Request | Status | Token | Active | Waiting | Cache | Reserved |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for event in events:
        token = "" if event.token_id is None else str(event.token_id)
        lines.append(
            "".join(
                (
                    f"| {event.sequence} | {event.timestamp:.6f} | {event.event_type.value} | ",
                    f"{event.request_id} | {event.status.value} | {token} | ",
                    f"{event.active_requests} | {event.waiting_requests} | ",
                    f"{event.cached_tokens} | {event.reserved_cache_tokens} |",
                )
            )
        )
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_simulation(  # noqa: C901, PLR0915
    config: SimulatorConfig,
    *,
    output_dir: Path | None = None,
) -> SimulationResult:
    """Run arrivals on logical time and write four deterministic artifacts."""
    destination = config.output_dir if output_dir is None else output_dir
    destination.mkdir(parents=True, exist_ok=True)
    model = _build_model(config)
    executor_clock = StepClock(config.executor_clock_step_seconds)
    telemetry_clock = StepClock(config.executor_clock_step_seconds / 10.0)
    paged_pool = (
        PagedKVCachePool.from_model(
            config.paged_kv_cache,
            model,
            prefix_cache_namespace=_prefix_cache_namespace(config, model),
        )
        if config.kv_cache_backend is KVCacheBackend.PAGED and config.paged_kv_cache is not None
        else None
    )
    if config.executor is SimulatorExecutor.PAGED_ATTENTION:
        if paged_pool is None:
            _invalid(Path("<runtime>"), "paged_attention executor requires paged cache pool")
        executor = PagedAttentionExecutor(
            model,
            paged_pool,
            prefill_config=config.prefill,
            prefix_prefill_strategy=config.apc_prefill_strategy,
            clock=executor_clock,
            telemetry_clock=telemetry_clock,
        )
    elif config.executor is SimulatorExecutor.CONTINUOUS:
        executor = ContinuousExecutor(
            model,
            prefill_config=config.prefill,
            clock=executor_clock,
            telemetry_clock=telemetry_clock,
        )
    elif config.executor is SimulatorExecutor.CONTINUOUS_DECODE:
        executor = ContinuousDecodeExecutor(
            model,
            clock=executor_clock,
            telemetry_clock=telemetry_clock,
        )
    else:
        executor = ReferenceExecutor(
            model,
            clock=executor_clock,
            telemetry_clock=telemetry_clock,
        )
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=config.scheduler,
            block_size=config.model.block_size,
            kv_cache_backend=config.kv_cache_backend,
            paged_kv_cache=(
                config.paged_kv_cache if config.kv_cache_backend is KVCacheBackend.PAGED else None
            ),
        ),
        executor=executor,
        paged_cache_pool=paged_pool,
        clock=lambda: 0.0,
    )
    pending = deque(
        sorted(enumerate(config.requests), key=lambda pair: (pair[1].arrival_time, pair[0]))
    )
    for tick_index in range(config.max_ticks):
        now = tick_index * config.tick_seconds
        while pending and pending[0][1].arrival_time <= now:
            _, request = pending.popleft()
            engine.submit(request)
        if not engine.is_idle:
            engine.tick(now=now)
        if not pending and engine.is_idle:
            break
    else:
        _invalid(Path("<runtime>"), f"simulation exceeded max_ticks={config.max_ticks}")

    events_path = destination / "events.jsonl"
    requests_path = destination / "requests.csv"
    summary_path = destination / "summary.json"
    prefill_observations = engine.prefill_observations
    timeline_path = destination / "timeline.md"
    _write_events(events_path, engine.events)
    _write_requests(requests_path, engine, config.requests)
    metrics = engine.metrics()
    _write_json(summary_path, _summary_document(config, metrics, prefill_observations))
    _write_timeline(timeline_path, engine.events, config.scenario_name, config.executor)
    output_paths = [events_path, requests_path, summary_path, timeline_path]
    if config.prefill is not None:
        prefill_events_path = destination / "prefill_events.jsonl"
        _write_prefill_events(prefill_events_path, engine.prefill_events)
        output_paths.append(prefill_events_path)
    if config.prefill is not None or config.prefix_cache_enabled:
        prefill_observations_path = destination / "prefill_observations.jsonl"
        _write_prefill_observations(prefill_observations_path, prefill_observations)
        output_paths.append(prefill_observations_path)
    generated = {
        request.request_id: tuple(engine.request_state(request.request_id).generated_tokens)
        for request in config.requests
    }
    request_statuses = {
        request.request_id: engine.request_state(request.request_id).status
        for request in config.requests
    }
    request_metrics = {
        request.request_id: engine.request_metrics(request.request_id)
        for request in config.requests
    }
    generator_state_hashes = {
        request.request_id: _generator_state_hash(engine, request.request_id)
        for request in config.requests
    }
    admission_order = tuple(
        event.request_id for event in engine.events if event.event_type.value == "admitted"
    )
    return SimulationResult(
        output_dir=destination,
        output_paths=tuple(output_paths),
        metrics=metrics,
        generated_tokens=generated,
        request_statuses=request_statuses,
        request_metrics=request_metrics,
        generator_state_hashes=generator_state_hashes,
        admission_order=admission_order,
        events=engine.events,
        prefill_events=engine.prefill_events,
        prefill_observations=prefill_observations,
    )


def run_executor_equivalence(
    config: SimulatorConfig,
    *,
    output_dir: Path,
) -> ExecutorEquivalenceResult:
    """Run all executors and reject any request-level logical contract divergence."""
    reference = run_simulation(
        replace(config, executor=SimulatorExecutor.REFERENCE),
        output_dir=output_dir / SimulatorExecutor.REFERENCE.value,
    )
    continuous_decode = run_simulation(
        replace(config, executor=SimulatorExecutor.CONTINUOUS_DECODE),
        output_dir=output_dir / SimulatorExecutor.CONTINUOUS_DECODE.value,
    )
    continuous = run_simulation(
        replace(config, executor=SimulatorExecutor.CONTINUOUS),
        output_dir=output_dir / SimulatorExecutor.CONTINUOUS.value,
    )
    comparisons: dict[str, bool] = {
        "generated_tokens": (
            reference.generated_tokens
            == continuous_decode.generated_tokens
            == continuous.generated_tokens
        ),
        "request_terminal_states_and_cancellation": (
            reference.request_statuses
            == continuous_decode.request_statuses
            == continuous.request_statuses
        ),
        "fifo_admission_order": (
            reference.admission_order
            == continuous_decode.admission_order
            == continuous.admission_order
        ),
        "cache_accounting": (
            reference.metrics.cached_tokens
            == continuous_decode.metrics.cached_tokens
            == continuous.metrics.cached_tokens
            and reference.metrics.reserved_cache_tokens
            == continuous_decode.metrics.reserved_cache_tokens
            == continuous.metrics.reserved_cache_tokens
            and reference.metrics.peak_cached_tokens
            == continuous_decode.metrics.peak_cached_tokens
            == continuous.metrics.peak_cached_tokens
            and reference.metrics.peak_reserved_cache_tokens
            == continuous_decode.metrics.peak_reserved_cache_tokens
            == continuous.metrics.peak_reserved_cache_tokens
        ),
        "logical_event_semantics": (
            reference.events == continuous_decode.events == continuous.events
        ),
        "request_metrics": (
            reference.request_metrics
            == continuous_decode.request_metrics
            == continuous.request_metrics
        ),
    }
    failed = tuple(name for name, matches in comparisons.items() if not matches)
    if failed:
        _invalid(Path("<equivalence>"), f"executor contracts differ: {', '.join(failed)}")
    return ExecutorEquivalenceResult(
        reference=reference,
        continuous_decode=continuous_decode,
        continuous=continuous,
        equivalent=True,
        checked_contracts=tuple(comparisons),
    )


def run_cache_backend_equivalence(
    config: SimulatorConfig,
    *,
    output_dir: Path,
) -> CacheBackendEquivalenceResult:
    """Run dense/paged storage and reject any logical serving divergence."""
    if config.paged_kv_cache is None:
        _invalid(Path("<equivalence>"), "cache backend equivalence requires kv_cache config")
    dense = run_simulation(
        replace(config, kv_cache_backend=KVCacheBackend.DENSE),
        output_dir=output_dir / KVCacheBackend.DENSE.value,
    )
    paged = run_simulation(
        replace(config, kv_cache_backend=KVCacheBackend.PAGED),
        output_dir=output_dir / KVCacheBackend.PAGED.value,
    )
    comparisons = {
        "generated_tokens": dense.generated_tokens == paged.generated_tokens,
        "request_terminal_states_and_cancellation": (
            dense.request_statuses == paged.request_statuses
        ),
        "fifo_admission_order": dense.admission_order == paged.admission_order,
        "logical_event_semantics": dense.events == paged.events,
        "request_metrics": dense.request_metrics == paged.request_metrics,
        "logical_cache_accounting": (
            dense.metrics.cached_tokens == paged.metrics.cached_tokens
            and dense.metrics.reserved_cache_tokens == paged.metrics.reserved_cache_tokens
            and dense.metrics.peak_cached_tokens == paged.metrics.peak_cached_tokens
            and dense.metrics.peak_reserved_cache_tokens == paged.metrics.peak_reserved_cache_tokens
        ),
    }
    failed = tuple(name for name, matches in comparisons.items() if not matches)
    if failed:
        _invalid(Path("<equivalence>"), f"cache backend contracts differ: {', '.join(failed)}")
    return CacheBackendEquivalenceResult(
        dense=dense,
        paged=paged,
        equivalent=True,
        checked_contracts=tuple(comparisons),
    )


def run_paged_attention_equivalence(
    config: SimulatorConfig,
    *,
    output_dir: Path,
) -> PagedAttentionEquivalenceResult:
    """Compare dense, materialized-paged, and direct block-aware decode contracts."""
    if config.paged_kv_cache is None:
        _invalid(Path("<equivalence>"), "paged attention equivalence requires kv_cache config")
    dense = run_simulation(
        replace(
            config,
            executor=SimulatorExecutor.CONTINUOUS,
            kv_cache_backend=KVCacheBackend.DENSE,
        ),
        output_dir=output_dir / "dense",
    )
    materialized = run_simulation(
        replace(
            config,
            executor=SimulatorExecutor.CONTINUOUS,
            kv_cache_backend=KVCacheBackend.PAGED,
        ),
        output_dir=output_dir / "materialized",
    )
    direct = run_simulation(
        replace(
            config,
            executor=SimulatorExecutor.PAGED_ATTENTION,
            kv_cache_backend=KVCacheBackend.PAGED,
        ),
        output_dir=output_dir / "direct",
    )
    comparisons = {
        "generated_tokens": (
            dense.generated_tokens == materialized.generated_tokens == direct.generated_tokens
        ),
        "request_terminal_states_and_cancellation": (
            dense.request_statuses == materialized.request_statuses == direct.request_statuses
        ),
        "fifo_admission_order": (
            dense.admission_order == materialized.admission_order == direct.admission_order
        ),
        "logical_event_semantics": dense.events == materialized.events == direct.events,
        "request_metrics": (
            dense.request_metrics == materialized.request_metrics == direct.request_metrics
        ),
        "logical_cache_accounting": (
            dense.metrics.peak_cached_tokens
            == materialized.metrics.peak_cached_tokens
            == direct.metrics.peak_cached_tokens
            and dense.metrics.peak_reserved_cache_tokens
            == materialized.metrics.peak_reserved_cache_tokens
            == direct.metrics.peak_reserved_cache_tokens
        ),
        "direct_decode_has_no_cache_padding": direct.metrics.padding_waste_ratio == 0.0,
    }
    failed = tuple(name for name, matches in comparisons.items() if not matches)
    if failed:
        _invalid(Path("<equivalence>"), f"paged attention contracts differ: {', '.join(failed)}")
    return PagedAttentionEquivalenceResult(
        dense=dense,
        materialized=materialized,
        direct=direct,
        equivalent=True,
        checked_contracts=tuple(comparisons),
    )


_PREFIX_EVENT_TYPES = frozenset(
    {
        EngineEventType.PREFIX_LOOKUP,
        EngineEventType.PREFIX_HIT,
        EngineEventType.PREFIX_PROMOTE,
        EngineEventType.PREFIX_EVICT,
    }
)


def _logical_request_events(events: tuple[EngineEvent, ...]) -> tuple[EngineEvent, ...]:
    filtered = (event for event in events if event.event_type not in _PREFIX_EVENT_TYPES)
    return tuple(replace(event, sequence=index) for index, event in enumerate(filtered))


def run_prefix_cache_equivalence(
    config: SimulatorConfig,
    *,
    output_dir: Path,
) -> PrefixCacheEquivalenceResult:
    """Compare direct paged decode with and without Automatic Prefix Caching."""
    if config.paged_kv_cache is None:
        _invalid(Path("<equivalence>"), "prefix cache equivalence requires kv_cache config")
    common = replace(
        config,
        executor=SimulatorExecutor.PAGED_ATTENTION,
        kv_cache_backend=KVCacheBackend.PAGED,
    )
    direct = run_simulation(
        replace(common, prefix_cache_enabled=False),
        output_dir=output_dir / "paged_direct",
    )
    automatic_prefix_cache = run_simulation(
        replace(common, prefix_cache_enabled=True),
        output_dir=output_dir / "paged_direct_apc",
    )
    comparisons = {
        "generated_tokens": direct.generated_tokens == automatic_prefix_cache.generated_tokens,
        "rng_state": (
            direct.generator_state_hashes == automatic_prefix_cache.generator_state_hashes
        ),
        "request_terminal_states_and_cancellation": (
            direct.request_statuses == automatic_prefix_cache.request_statuses
        ),
        "fifo_admission_order": direct.admission_order == automatic_prefix_cache.admission_order,
        "logical_request_events": (
            _logical_request_events(direct.events)
            == _logical_request_events(automatic_prefix_cache.events)
        ),
        "terminal_logical_cache_accounting": (
            direct.metrics.cached_tokens == automatic_prefix_cache.metrics.cached_tokens == 0
            and direct.metrics.reserved_cache_tokens
            == automatic_prefix_cache.metrics.reserved_cache_tokens
            == 0
        ),
    }
    failed = tuple(name for name, matches in comparisons.items() if not matches)
    if failed:
        _invalid(Path("<equivalence>"), f"prefix cache contracts differ: {', '.join(failed)}")
    return PrefixCacheEquivalenceResult(
        direct=direct,
        automatic_prefix_cache=automatic_prefix_cache,
        equivalent=True,
        checked_contracts=tuple(comparisons),
    )


def _stage15_event_semantics(events: tuple[EngineEvent, ...]) -> tuple[EngineEvent, ...]:
    return tuple(
        replace(event, sequence=index, timestamp=0.0) for index, event in enumerate(events)
    )


def run_cache_aware_prefill_equivalence(
    config: SimulatorConfig,
    *,
    output_dir: Path,
) -> CacheAwarePrefillEquivalenceResult:
    """Compare Stage 14 sequential APC with Stage 15 batched APC."""
    if config.paged_kv_cache is None:
        _invalid(Path("<equivalence>"), "cache-aware prefill requires paged KV cache config")
    common = replace(
        config,
        executor=SimulatorExecutor.PAGED_ATTENTION,
        kv_cache_backend=KVCacheBackend.PAGED,
        prefix_cache_enabled=True,
    )
    sequential = run_simulation(
        replace(common, apc_prefill_strategy=APCPrefillStrategy.SEQUENTIAL),
        output_dir=output_dir / "apc_sequential",
    )
    batched = run_simulation(
        replace(common, apc_prefill_strategy=APCPrefillStrategy.BATCHED),
        output_dir=output_dir / "apc_batched",
    )
    comparisons: dict[str, bool] = {
        "generated_tokens": sequential.generated_tokens == batched.generated_tokens,
        "rng_state": sequential.generator_state_hashes == batched.generator_state_hashes,
        "request_terminal_states_and_cancellation": (
            sequential.request_statuses == batched.request_statuses
        ),
        "fifo_admission_order": sequential.admission_order == batched.admission_order,
        "prefix_identity_and_logical_events": (
            _stage15_event_semantics(sequential.events) == _stage15_event_semantics(batched.events)
        ),
        "terminal_cache_ownership": (
            sequential.metrics.active_shared_references
            == batched.metrics.active_shared_references
            == 0
            and sequential.metrics.reserved_blocks == batched.metrics.reserved_blocks == 0
            and sequential.metrics.prefix_cache_blocks == batched.metrics.prefix_cache_blocks
            and sequential.metrics.allocated_blocks == batched.metrics.allocated_blocks
        ),
    }
    failed = tuple(name for name, matches in comparisons.items() if not matches)
    if failed:
        _invalid(
            Path("<equivalence>"), f"cache-aware prefill contracts differ: {', '.join(failed)}"
        )
    return CacheAwarePrefillEquivalenceResult(
        sequential=sequential,
        batched=batched,
        equivalent=True,
        checked_contracts=tuple(comparisons),
    )
