"""Deterministic multi-request serving control plane with a reference executor."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Never, Protocol, Self, cast, final

import torch
from torch import Tensor
from torch.nn import functional
from typing_extensions import override

from minigpt.layers import KVCache, LayerKVCache, PagedKVCacheView
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCacheMetrics,
    PagedKVCachePool,
)

if TYPE_CHECKING:
    from minigpt.model import GPT

Clock = Callable[[], float]
_LOGICAL_TIME_DECIMALS = 12
_EMPTY_PAGED_METRICS = PagedKVCacheMetrics(
    total_blocks=0,
    free_blocks=0,
    allocated_blocks=0,
    reserved_blocks=0,
    peak_allocated_blocks=0,
    peak_reserved_blocks=0,
    used_token_slots=0,
    allocated_token_slots=0,
    internal_fragmentation_tokens=0,
    internal_fragmentation_ratio=0.0,
    allocation_count=0,
    free_count=0,
    block_reuse_count=0,
)


class RequestStatus(StrEnum):
    """Describe one request's serving lifecycle state."""

    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    PREEMPTED = "preempted"
    RECOMPUTING = "recomputing"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class EngineEventType(StrEnum):
    """Identify deterministic control-plane transitions and outputs."""

    SUBMITTED = "submitted"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    ADMITTED = "admitted"
    PREFILL_STARTED = "prefill_started"
    PREFILL_CHUNK_STARTED = "PREFILL_CHUNK_STARTED"
    PREFILL_CHUNK_FINISHED = "PREFILL_CHUNK_FINISHED"
    PREEMPTED = "PREEMPTED"
    RECOMPUTE_STARTED = "RECOMPUTE_STARTED"
    RESUMED = "RESUMED"
    TOKEN = "token"  # noqa: S105
    FINISHED = "finished"
    FAILED = "failed"
    PREFIX_LOOKUP = "PREFIX_LOOKUP"
    PREFIX_HIT = "PREFIX_HIT"
    PREFIX_PROMOTE = "PREFIX_PROMOTE"
    PREFIX_EVICT = "PREFIX_EVICT"


@dataclass(frozen=True, slots=True)
class InvalidServingConfigError(ValueError):
    """Report an invalid request, scheduler, or engine setting."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed serving constraint."""
        return f"invalid serving configuration: {self.reason}"


@dataclass(frozen=True, slots=True)
class DuplicateRequestError(ValueError):
    """Report a request ID that already belongs to the engine."""

    request_id: str

    @override
    def __str__(self) -> str:
        """Render the duplicate request ID."""
        return f"request ID {self.request_id!r} already exists"


@dataclass(frozen=True, slots=True)
class UnknownRequestError(KeyError):
    """Report a request ID unknown to the engine."""

    request_id: str

    @override
    def __str__(self) -> str:
        """Render the missing request ID."""
        return f"unknown request ID {self.request_id!r}"


@dataclass(frozen=True, slots=True)
class EngineRunawayError(RuntimeError):
    """Report an engine that did not become idle within its explicit bound."""

    max_ticks: int

    @override
    def __str__(self) -> str:
        """Render the exceeded tick limit."""
        return f"serving engine did not become idle within {self.max_ticks} ticks"


def _invalid(reason: str) -> Never:
    raise InvalidServingConfigError(reason)


def _finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        _invalid(f"{name} must be finite and non-negative")


def _logical_elapsed(clock: Clock, start: float) -> float:
    """Normalize injected logical-clock subtraction across executor call order."""
    return round(max(0.0, clock() - start), _LOGICAL_TIME_DECIMALS)


def _require_paged_cache_length(actual: int, expected: int) -> None:
    if actual != expected:
        reason = f"paged cache length {actual} does not equal executor occupancy {expected}"
        raise RuntimeError(reason)


def _require_prefix_prefill_logits(logits: Tensor | None) -> Tensor:
    if logits is None:
        reason = "prefix-cache prefill omitted suffix boundary logits"
        raise RuntimeError(reason)
    return logits


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Define one independently sampled autoregressive request."""

    request_id: str
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    seed: int = 0
    arrival_time: float = 0.0
    cancellation_time: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed requests before they enter scheduler state."""
        if not self.request_id:
            _invalid("request_id must be non-empty")
        if not self.prompt_tokens:
            _invalid("prompt_tokens must be non-empty")
        if any(type(token) is not int or token < 0 for token in self.prompt_tokens):
            _invalid("prompt_tokens must contain non-negative integers")
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens < 0:
            _invalid("max_new_tokens must be a non-negative integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            _invalid("temperature must be finite and positive")
        if self.top_k is not None and (isinstance(self.top_k, bool) or self.top_k <= 0):
            _invalid("top_k must be null or a positive integer")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 2**63:
            _invalid("seed must be an integer in [0, 2**63)")
        _finite_non_negative(self.arrival_time, "arrival_time")
        if self.cancellation_time is not None:
            _finite_non_negative(self.cancellation_time, "cancellation_time")
            if self.cancellation_time < self.arrival_time:
                _invalid("cancellation_time must not precede arrival_time")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Configure strict FIFO admission and KV-cache reservations."""

    max_active_requests: int
    max_cached_tokens: int
    max_scheduled_tokens: int | None = None
    prefill_chunk_tokens: int | None = None
    kv_preemption: bool = False

    def __post_init__(self) -> None:
        """Require usable positive serving capacity."""
        if isinstance(self.max_active_requests, bool) or self.max_active_requests <= 0:
            _invalid("max_active_requests must be a positive integer")
        if isinstance(self.max_cached_tokens, bool) or self.max_cached_tokens <= 0:
            _invalid("max_cached_tokens must be a positive integer")
        stage16_values = (self.max_scheduled_tokens, self.prefill_chunk_tokens)
        if any(value is None for value in stage16_values) and any(
            value is not None for value in stage16_values
        ):
            _invalid("max_scheduled_tokens and prefill_chunk_tokens must be configured together")
        for name, value in (
            ("max_scheduled_tokens", self.max_scheduled_tokens),
            ("prefill_chunk_tokens", self.prefill_chunk_tokens),
        ):
            if value is not None and (isinstance(value, bool) or value <= 0):
                _invalid(f"{name} must be a positive integer")
        raw_preemption = cast("object", self.kv_preemption)
        if not isinstance(raw_preemption, bool):
            _invalid("kv_preemption must be a boolean")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Bind a scheduler policy to the executor's learned-position capacity."""

    scheduler: SchedulerConfig
    block_size: int
    kv_cache_backend: KVCacheBackend = KVCacheBackend.DENSE
    paged_kv_cache: PagedKVCacheConfig | None = None

    def __post_init__(self) -> None:
        """Reject an unusable learned-position window."""
        if isinstance(self.block_size, bool) or self.block_size <= 0:
            _invalid("block_size must be a positive integer")
        if self.kv_cache_backend is KVCacheBackend.DENSE and self.paged_kv_cache is not None:
            _invalid("dense backend must not define paged_kv_cache")
        if self.kv_cache_backend is KVCacheBackend.PAGED and self.paged_kv_cache is None:
            _invalid("paged backend requires paged_kv_cache")


@dataclass(slots=True)
class RequestState:
    """Hold mutable scheduler, cache, RNG, output, and timing state."""

    request: GenerationRequest
    generator: torch.Generator = field(repr=False)
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    kv_cache: KVCache | None = field(default=None, repr=False)
    reserved_cache_tokens: int = 0
    reserved_cache_blocks: int = 0
    cached_tokens: int = 0
    admission_time: float | None = None
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    prefill_latency_seconds: float | None = None
    decode_latencies_seconds: list[float] = field(default_factory=list)
    token_timestamps: list[float] = field(default_factory=list)
    failure_reason: str | None = None
    preemption_count: int = 0
    resume_count: int = 0
    recompute_tokens: int = 0
    last_recompute_time: float | None = None
    last_decode_tick: float | None = None
    prefix_hit_blocks: int = 0
    prefix_hit_tokens: int = 0
    prefix_miss_tokens: int = 0
    prefill_tokens_computed: int = 0
    prefill_logits_chunks: list[Tensor] = field(default_factory=list, repr=False)

    @property
    def all_tokens(self) -> tuple[int, ...]:
        """Return the immutable prompt followed by generated tokens."""
        return self.request.prompt_tokens + tuple(self.generated_tokens)


@dataclass(frozen=True, slots=True)
class CacheRecomputeResult:
    """Return one cache-only recomputation outcome without sampling."""

    request_id: str
    cache: KVCache | None
    cache_length: int
    latency_seconds: float
    error: str | None

    @classmethod
    def failure(cls, request_id: str, error: str, latency_seconds: float) -> Self:
        """Build a failed cache-only recomputation result."""
        return cls(request_id, None, 0, latency_seconds, error)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Return one isolated per-request executor outcome."""

    request_id: str
    token_id: int | None
    cache: KVCache | None
    cache_tokens: int
    latency_seconds: float
    error: str | None
    used_fallback: bool
    cache_is_delta: bool
    prefill_prefix_tokens: int
    prefill_logits: Tensor | None = field(repr=False)
    prefill_complete: bool = True

    @classmethod
    def success(  # noqa: PLR0913
        cls,
        *,
        request_id: str,
        token_id: int,
        cache: KVCache,
        cache_tokens: int,
        latency_seconds: float,
        used_fallback: bool,
        cache_is_delta: bool = False,
        prefill_prefix_tokens: int = 0,
        prefill_logits: Tensor | None = None,
    ) -> Self:
        """Build a successful token result."""
        return cls(
            request_id=request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=cache_tokens,
            latency_seconds=latency_seconds,
            error=None,
            used_fallback=used_fallback,
            cache_is_delta=cache_is_delta,
            prefill_prefix_tokens=prefill_prefix_tokens,
            prefill_logits=prefill_logits,
        )

    @classmethod
    def failure(cls, request_id: str, error: str, latency_seconds: float) -> Self:
        """Build an isolated executor failure result."""
        return cls(
            request_id=request_id,
            token_id=None,
            cache=None,
            cache_tokens=0,
            latency_seconds=latency_seconds,
            error=error,
            used_fallback=False,
            cache_is_delta=False,
            prefill_prefix_tokens=0,
            prefill_logits=None,
        )


@dataclass(frozen=True, slots=True)
class DecodeBatchObservation:
    """Separate decode batch utilization and wall-clock phase timings."""

    request_ids: tuple[str, ...]
    batch_size: int
    padded_cache_tokens: int
    useful_cache_tokens: int
    assembly_seconds: float
    model_seconds: float
    scatter_seconds: float
    executor_seconds: float
    batch_failed: bool


class APCPrefillStrategy(StrEnum):
    """Select the Stage 14 reference or Stage 15 batched APC prefill path."""

    SEQUENTIAL = "sequential"
    BATCHED = "batched"


class PrefillExecutionMode(StrEnum):
    """Identify the model-work path represented by a prefill observation."""

    FULL_DENSE = "full_dense"
    SEQUENTIAL_APC_SUFFIX = "sequential_apc_suffix"
    BATCHED_APC_SUFFIX = "batched_apc_suffix"
    EXACT_CACHE_HIT = "exact_cache_hit"
    OVERFLOW_DENSE_REBUILD = "overflow_dense_rebuild"
    CHUNKED_PAGED_PREFILL = "chunked_paged_prefill"
    PREEMPTION_RECOMPUTE = "preemption_recompute"


@dataclass(frozen=True, slots=True)
class PrefillBatchConfig:
    """Bound deterministic FIFO prompt grouping and dense padding."""

    max_batch_size: int = 8
    max_batch_tokens: int = 1024
    max_padding_ratio: float = 0.25

    def __post_init__(self) -> None:
        """Reject unusable batch and padding limits."""
        if isinstance(self.max_batch_size, bool) or self.max_batch_size <= 0:
            _invalid("prefill max_batch_size must be a positive integer")
        if isinstance(self.max_batch_tokens, bool) or self.max_batch_tokens <= 0:
            _invalid("prefill max_batch_tokens must be a positive integer")
        if not math.isfinite(self.max_padding_ratio) or not 0.0 <= self.max_padding_ratio <= 1.0:
            _invalid("prefill max_padding_ratio must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class PrefillBatchObservation:
    """Separate prefill utilization and wall-clock phase timings."""

    request_ids: tuple[str, ...]
    batch_size: int
    padded_prompt_tokens: int
    useful_prompt_tokens: int
    assembly_seconds: float
    model_seconds: float
    scatter_seconds: float
    executor_seconds: float
    started_at: float
    finished_at: float
    batch_failed: bool
    execution_mode: PrefillExecutionMode = PrefillExecutionMode.FULL_DENSE
    model_calls: int = 1
    prefix_hit_tokens: int = 0
    avoided_prefill_tokens: int = 0

    @property
    def padding_waste_ratio(self) -> float:
        """Return the fraction of dense prompt slots occupied only by padding."""
        if self.padded_prompt_tokens == 0:
            return 0.0
        return (self.padded_prompt_tokens - self.useful_prompt_tokens) / self.padded_prompt_tokens


class PrefillBatchEventType(StrEnum):
    """Identify executor-level prefill batch evidence without changing request events."""

    STARTED = "PREFILL_BATCH_STARTED"
    FINISHED = "PREFILL_BATCH_FINISHED"


@dataclass(frozen=True, slots=True)
class PrefillBatchEvent:
    """Record one prefill batch boundary and its utilization evidence."""

    sequence: int
    timestamp: float
    event_type: PrefillBatchEventType
    request_ids: tuple[str, ...]
    batch_size: int
    useful_prompt_tokens: int
    padded_prompt_tokens: int
    padding_waste_ratio: float
    batch_failed: bool


def _prefill_events(
    observation: PrefillBatchObservation,
    *,
    starting_sequence: int,
) -> tuple[PrefillBatchEvent, PrefillBatchEvent]:
    return (
        PrefillBatchEvent(
            sequence=starting_sequence,
            timestamp=observation.started_at,
            event_type=PrefillBatchEventType.STARTED,
            request_ids=observation.request_ids,
            batch_size=observation.batch_size,
            useful_prompt_tokens=observation.useful_prompt_tokens,
            padded_prompt_tokens=observation.padded_prompt_tokens,
            padding_waste_ratio=observation.padding_waste_ratio,
            batch_failed=observation.batch_failed,
        ),
        PrefillBatchEvent(
            sequence=starting_sequence + 1,
            timestamp=observation.finished_at,
            event_type=PrefillBatchEventType.FINISHED,
            request_ids=observation.request_ids,
            batch_size=observation.batch_size,
            useful_prompt_tokens=observation.useful_prompt_tokens,
            padded_prompt_tokens=observation.padded_prompt_tokens,
            padding_waste_ratio=observation.padding_waste_ratio,
            batch_failed=observation.batch_failed,
        ),
    )


class ServingExecutor(Protocol):
    """Advance request states without defining scheduler policy."""

    block_size: int

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return append-only model-batch telemetry."""
        ...

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return append-only prompt-batch telemetry."""
        ...

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return executor-level prompt batch boundary evidence."""
        ...

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Evaluate prompts separately and return at most one token per request."""
        ...

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Evaluate one new token separately for each active request."""
        ...


@final
class ReferenceExecutor:
    """Call Stage 9 model interfaces once per request, without tensor batching."""

    def __init__(
        self,
        model: GPT,
        *,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind a CPU model and injectable operation clock."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self._decode_observations: list[DecodeBatchObservation] = []
        self._prefill_observations: list[PrefillBatchObservation] = []
        self._prefill_events: list[PrefillBatchEvent] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return per-request reference decode calls as size-one batches."""
        return tuple(self._decode_observations)

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return one observation per reference prompt call."""
        return tuple(self._prefill_observations)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return size-one reference prompt batch boundaries."""
        return tuple(self._prefill_events)

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Prefill each prompt independently and sample its first token."""
        return tuple(self._prefill_one(state, used_fallback=False) for state in requests)

    def prefill_fallback(self, state: RequestState) -> ExecutionResult:
        """Rebuild one full learned-position window for overflow decode."""
        return self._prefill_one(state, used_fallback=True)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Decode once or rebuild a full learned-position window per request."""
        results: list[ExecutionResult] = []
        for state in active_requests:
            if state.cached_tokens >= self.block_size:
                results.append(self._prefill_one(state, used_fallback=True))
            else:
                results.append(self._decode_one(state))
        return tuple(results)

    def _prefill_one(self, state: RequestState, *, used_fallback: bool) -> ExecutionResult:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        context = state.all_tokens[-self.block_size :]
        try:
            assembly_start = self._telemetry_clock()
            token_ids = torch.tensor(
                (context,),
                dtype=torch.long,
                device=self._model.token_embedding.weight.device,
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, cache = self._model.prefill(token_ids)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            token_id = self._sample(state, logits[:, -1, :])
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = self._elapsed(logical_start)
            self._record_prefill(
                state,
                context_length=len(context),
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
                used_fallback=used_fallback,
            )
            return ExecutionResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                elapsed,
            )
        elapsed = self._elapsed(logical_start)
        self._record_prefill(
            state,
            context_length=len(context),
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
            used_fallback=used_fallback,
        )
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=len(context),
            latency_seconds=elapsed,
            used_fallback=used_fallback,
        )

    def _record_prefill(  # noqa: PLR0913
        self,
        state: RequestState,
        *,
        context_length: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
        used_fallback: bool,
    ) -> None:
        observation = PrefillBatchObservation(
            request_ids=(state.request.request_id,),
            batch_size=1,
            padded_prompt_tokens=context_length,
            useful_prompt_tokens=context_length,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
            started_at=logical_start,
            finished_at=logical_start + elapsed,
            batch_failed=batch_failed,
            execution_mode=(
                PrefillExecutionMode.OVERFLOW_DENSE_REBUILD
                if used_fallback
                else PrefillExecutionMode.FULL_DENSE
            ),
        )
        self._prefill_observations.append(observation)
        self._prefill_events.extend(
            _prefill_events(observation, starting_sequence=len(self._prefill_events))
        )

    def _decode_one(self, state: RequestState) -> ExecutionResult:
        start = self._clock()
        telemetry_start = self._telemetry_clock()
        if state.kv_cache is None or not state.generated_tokens:
            return ExecutionResult.failure(
                state.request.request_id,
                "decode requires a populated cache and generated token",
                self._elapsed(start),
            )
        try:
            token_ids = torch.tensor(
                ((state.generated_tokens[-1],),),
                dtype=torch.long,
                device=self._model.token_embedding.weight.device,
            )
            logits, cache = self._model.decode(token_ids, state.kv_cache)
            token_id = self._sample(state, logits[:, -1, :])
        except Exception as error:  # noqa: BLE001
            telemetry_elapsed = max(0.0, self._telemetry_clock() - telemetry_start)
            self._record_decode(state, telemetry_elapsed, batch_failed=True)
            return ExecutionResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                self._elapsed(start),
            )
        telemetry_elapsed = max(0.0, self._telemetry_clock() - telemetry_start)
        self._record_decode(state, telemetry_elapsed, batch_failed=False)
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=cache,
            cache_tokens=cache[0].length,
            latency_seconds=self._elapsed(start),
            used_fallback=False,
        )

    def _record_decode(
        self,
        state: RequestState,
        elapsed_seconds: float,
        *,
        batch_failed: bool,
    ) -> None:
        self._decode_observations.append(
            DecodeBatchObservation(
                request_ids=(state.request.request_id,),
                batch_size=1,
                padded_cache_tokens=state.cached_tokens,
                useful_cache_tokens=state.cached_tokens,
                assembly_seconds=0.0,
                model_seconds=elapsed_seconds,
                scatter_seconds=0.0,
                executor_seconds=elapsed_seconds,
                batch_failed=batch_failed,
            )
        )

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())

    def _elapsed(self, start: float) -> float:
        return _logical_elapsed(self._clock, start)


@final
class ContinuousDecodeExecutor:
    """Keep per-request prefill while batching valid single-token decode rows."""

    def __init__(
        self,
        model: GPT,
        *,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind one model plus separate logical-result and telemetry clocks."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self._reference = ReferenceExecutor(
            model,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._decode_observations: list[DecodeBatchObservation] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return one observation for every actual continuous decode model call."""
        return tuple(self._decode_observations)

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Expose the delegated per-request prefill observations."""
        return self._reference.prefill_observations

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Expose the delegated per-request prefill batch boundaries."""
        return self._reference.prefill_events

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Preserve Stage 10 per-request prompt execution."""
        return self._reference.prefill(requests)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Isolate invalid/overflow requests and batch the remaining decode rows."""
        results: dict[str, ExecutionResult] = {}
        batch: list[RequestState] = []
        for state in active_requests:
            error = self._validate_decode_state(state)
            if error is not None:
                results[state.request.request_id] = ExecutionResult.failure(
                    state.request.request_id,
                    error,
                    0.0,
                )
            elif state.cached_tokens >= self.block_size:
                results[state.request.request_id] = self._reference.prefill_fallback(state)
            else:
                batch.append(state)
        if batch:
            for result in self.decode_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in active_requests)

    def decode_batch(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Assemble, execute, sample, and scatter one variable-length decode batch."""
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        try:
            assembly_start = self._telemetry_clock()
            token_ids, dense_cache, cache_lengths = self._assemble(requests)
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, next_dense_cache = self._model.decode_batch(
                token_ids,
                dense_cache,
                cache_lengths,
            )
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            compact_caches = self._scatter(next_dense_cache, cache_lengths)
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            executor_seconds = max(0.0, self._telemetry_clock() - executor_start)
            useful = sum(state.cached_tokens for state in requests)
            padded = len(requests) * max((state.cached_tokens for state in requests), default=0)
            self._decode_observations.append(
                DecodeBatchObservation(
                    request_ids=request_ids,
                    batch_size=len(requests),
                    padded_cache_tokens=padded,
                    useful_cache_tokens=useful,
                    assembly_seconds=0.0,
                    model_seconds=executor_seconds,
                    scatter_seconds=0.0,
                    executor_seconds=executor_seconds,
                    batch_failed=True,
                )
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )

        elapsed = _logical_elapsed(self._clock, logical_start)
        results: list[ExecutionResult] = []
        for row, (state, cache) in enumerate(zip(requests, compact_caches, strict=True)):
            try:
                token_id = self._sample(state, logits[row : row + 1, -1, :])
            except Exception as error:  # noqa: BLE001
                results.append(
                    ExecutionResult.failure(
                        state.request.request_id,
                        f"{type(error).__name__}: {error}",
                        elapsed,
                    )
                )
                continue
            results.append(
                ExecutionResult.success(
                    request_id=state.request.request_id,
                    token_id=token_id,
                    cache=cache,
                    cache_tokens=state.cached_tokens + 1,
                    latency_seconds=elapsed,
                    used_fallback=False,
                )
            )
        executor_seconds = max(0.0, self._telemetry_clock() - executor_start)
        useful = int(cache_lengths.sum().item())
        padded = len(requests) * int(cache_lengths.max().item())
        self._decode_observations.append(
            DecodeBatchObservation(
                request_ids=request_ids,
                batch_size=len(requests),
                padded_cache_tokens=padded,
                useful_cache_tokens=useful,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=executor_seconds,
                batch_failed=False,
            )
        )
        return tuple(results)

    def _validate_decode_state(  # noqa: C901, PLR0911
        self,
        state: RequestState,
    ) -> str | None:
        if state.kv_cache is None or not state.generated_tokens:
            return "decode requires a populated cache and generated token"
        cache = cast("tuple[object, ...]", state.kv_cache)
        if len(cache) != self._model.config.n_layer:
            return "invalid KV cache: layer count does not match model"
        expected_device = self._model.token_embedding.weight.device
        expected_dtype = self._model.token_embedding.weight.dtype
        head_size = self._model.config.n_embd // self._model.config.n_head
        for layer_index, layer_cache in enumerate(cache):
            if not isinstance(layer_cache, LayerKVCache):
                return f"invalid KV cache: layer {layer_index} is not LayerKVCache"
            if layer_cache.key.shape != layer_cache.value.shape:
                return f"invalid KV cache: layer {layer_index} key/value shapes differ"
            expected_shape = (
                1,
                self._model.config.n_head,
                state.cached_tokens,
                head_size,
            )
            if tuple(layer_cache.key.shape) != expected_shape:
                return (
                    f"invalid KV cache: layer {layer_index} shape "
                    f"{tuple(layer_cache.key.shape)} must equal {expected_shape}"
                )
            for tensor in (layer_cache.key, layer_cache.value):
                if tensor.dtype != expected_dtype:
                    return f"invalid KV cache: layer {layer_index} dtype must equal model dtype"
                if tensor.device != expected_device:
                    return f"invalid KV cache: layer {layer_index} device must equal model device"
                if tensor.requires_grad:
                    return f"invalid KV cache: layer {layer_index} cache must be detached"
        if state.cached_tokens <= 0 or state.cached_tokens > self.block_size:
            return "invalid KV cache: cached token length is outside model capacity"
        return None

    def _assemble(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[Tensor, KVCache, Tensor]:
        batch_size = len(requests)
        padded_length = max(state.cached_tokens for state in requests)
        device = self._model.token_embedding.weight.device
        dtype = self._model.token_embedding.weight.dtype
        head_size = self._model.config.n_embd // self._model.config.n_head
        cache_lengths = torch.tensor(
            [state.cached_tokens for state in requests],
            dtype=torch.long,
            device=device,
        )
        token_ids = torch.tensor(
            [[state.generated_tokens[-1]] for state in requests],
            dtype=torch.long,
            device=device,
        )
        dense_layers: list[LayerKVCache] = []
        for layer_index in range(self._model.config.n_layer):
            key = torch.zeros(
                batch_size,
                self._model.config.n_head,
                padded_length,
                head_size,
                dtype=dtype,
                device=device,
            )
            value = torch.zeros_like(key)
            for row, state in enumerate(requests):
                cache = cast("KVCache", state.kv_cache)
                layer_cache = cache[layer_index]
                key[row, :, : state.cached_tokens, :] = layer_cache.key[0]
                value[row, :, : state.cached_tokens, :] = layer_cache.value[0]
            dense_layers.append(LayerKVCache(key=key.detach(), value=value.detach()))
        return token_ids, tuple(dense_layers), cache_lengths

    @staticmethod
    def _scatter(next_dense_cache: KVCache, cache_lengths: Tensor) -> tuple[KVCache, ...]:
        compact: list[KVCache] = []
        for row in range(cache_lengths.shape[0]):
            length = int(cache_lengths[row].item())
            layers: list[LayerKVCache] = []
            for dense_layer in next_dense_cache:
                key = torch.cat(
                    (
                        dense_layer.key[row : row + 1, :, :length, :],
                        dense_layer.key[row : row + 1, :, -1:, :],
                    ),
                    dim=2,
                ).clone()
                value = torch.cat(
                    (
                        dense_layer.value[row : row + 1, :, :length, :],
                        dense_layer.value[row : row + 1, :, -1:, :],
                    ),
                    dim=2,
                ).clone()
                layers.append(LayerKVCache(key=key.detach(), value=value.detach()))
            compact.append(tuple(layers))
        return tuple(compact)

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())


@final
class ContinuousExecutor:
    """Length-bucket prompt prefill and reuse Stage 11A continuous decode."""

    def __init__(
        self,
        model: GPT,
        *,
        prefill_config: PrefillBatchConfig | None = None,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind deterministic prompt grouping and the existing decode executor."""
        self._model = model
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self.prefill_config = prefill_config or PrefillBatchConfig()
        self._decode = ContinuousDecodeExecutor(
            model,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._prefill_observations: list[PrefillBatchObservation] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Expose the unchanged Stage 11A decode telemetry."""
        return self._decode.decode_observations

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return initial batched prefill plus Stage 9 overflow re-prefill calls."""
        return (*self._prefill_observations, *self._decode.prefill_observations)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return resequenced boundaries for initial and overflow prefill calls."""
        events: list[PrefillBatchEvent] = []
        for observation in self.prefill_observations:
            events.extend(_prefill_events(observation, starting_sequence=len(events)))
        return tuple(events)

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Validate independently, group valid FIFO prefixes, and prefill each batch."""
        results: dict[str, ExecutionResult] = {}
        valid: list[RequestState] = []
        for state in requests:
            error = self._validate_prefill_state(state)
            if error is None:
                valid.append(state)
            else:
                results[state.request.request_id] = ExecutionResult.failure(
                    state.request.request_id,
                    error,
                    0.0,
                )
        for batch in self._batches(valid):
            for result in self.prefill_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in requests)

    def prefill_batch(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Assemble, execute, sample, and scatter one padded prompt batch."""
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        lengths = tuple(len(state.request.prompt_tokens) for state in requests)
        useful_tokens = sum(lengths)
        padded_tokens = len(requests) * max(lengths, default=0)
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        try:
            assembly_start = self._telemetry_clock()
            token_ids, prompt_lengths = self._assemble_prefill(requests)
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, dense_cache = self._model.prefill_batch(token_ids, prompt_lengths)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            compact_caches = self._scatter_prefill(dense_cache, prompt_lengths)
            results: list[ExecutionResult] = []
            for row, (state, cache) in enumerate(zip(requests, compact_caches, strict=True)):
                try:
                    token_id = self._sample(state, logits[row : row + 1, -1, :])
                except Exception as error:  # noqa: BLE001
                    results.append(
                        ExecutionResult.failure(
                            state.request.request_id,
                            f"{type(error).__name__}: {error}",
                            0.0,
                        )
                    )
                    continue
                results.append(
                    ExecutionResult.success(
                        request_id=state.request.request_id,
                        token_id=token_id,
                        cache=cache,
                        cache_tokens=len(state.request.prompt_tokens),
                        latency_seconds=0.0,
                        used_fallback=False,
                    )
                )
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._record_prefill_batch(
                request_ids=request_ids,
                useful_tokens=useful_tokens,
                padded_tokens=padded_tokens,
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )

        elapsed = _logical_elapsed(self._clock, logical_start)
        results = [replace(result, latency_seconds=elapsed) for result in results]
        self._record_prefill_batch(
            request_ids=request_ids,
            useful_tokens=useful_tokens,
            padded_tokens=padded_tokens,
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
        )
        return tuple(results)

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Reuse Stage 11A validation, overflow fallback, batching, and sampling."""
        return self._decode.decode(active_requests)

    def _validate_prefill_state(self, state: RequestState) -> str | None:
        tokens = state.request.prompt_tokens
        if not tokens:
            return "prefill prompt must be non-empty"
        if len(tokens) > self.block_size:
            return f"prefill prompt length {len(tokens)} exceeds block_size {self.block_size}"
        if any(type(token) is not int for token in tokens):
            return "prefill prompt tokens must be integers"
        if any(token < 0 or token >= self._model.config.vocab_size for token in tokens):
            return "prefill prompt token is outside model vocabulary"
        return None

    def _batches(self, requests: Sequence[RequestState]) -> tuple[tuple[RequestState, ...], ...]:
        batches: list[tuple[RequestState, ...]] = []
        current: list[RequestState] = []
        for state in requests:
            if not current:
                current.append(state)
                continue
            candidate = [*current, state]
            lengths = [len(item.request.prompt_tokens) for item in candidate]
            padded = len(candidate) * max(lengths)
            useful = sum(lengths)
            waste_ratio = (padded - useful) / padded
            config = self.prefill_config
            fits = (
                len(candidate) <= config.max_batch_size
                and padded <= config.max_batch_tokens
                and waste_ratio <= config.max_padding_ratio
            )
            if fits:
                current.append(state)
            else:
                batches.append(tuple(current))
                current = [state]
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _assemble_prefill(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[Tensor, Tensor]:
        device = self._model.token_embedding.weight.device
        lengths = [len(state.request.prompt_tokens) for state in requests]
        token_ids = torch.zeros(
            (len(requests), max(lengths)),
            dtype=torch.long,
            device=device,
        )
        for row, state in enumerate(requests):
            row_tokens = torch.tensor(
                state.request.prompt_tokens,
                dtype=torch.long,
                device=device,
            )
            token_ids[row, : row_tokens.shape[0]] = row_tokens
        return token_ids, torch.tensor(lengths, dtype=torch.long, device=device)

    @staticmethod
    def _scatter_prefill(
        dense_cache: KVCache,
        prompt_lengths: Tensor,
    ) -> tuple[KVCache, ...]:
        compact: list[KVCache] = []
        for row, length_tensor in enumerate(prompt_lengths):
            length = int(length_tensor.item())
            layers = tuple(
                LayerKVCache(
                    key=layer.key[row : row + 1, :, :length, :].clone().detach(),
                    value=layer.value[row : row + 1, :, :length, :].clone().detach(),
                )
                for layer in dense_cache
            )
            compact.append(layers)
        return tuple(compact)

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())

    def _record_prefill_batch(  # noqa: PLR0913
        self,
        *,
        request_ids: tuple[str, ...],
        useful_tokens: int,
        padded_tokens: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
    ) -> None:
        observation = PrefillBatchObservation(
            request_ids=request_ids,
            batch_size=len(request_ids),
            padded_prompt_tokens=padded_tokens,
            useful_prompt_tokens=useful_tokens,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
            started_at=logical_start,
            finished_at=logical_start + elapsed,
            batch_failed=batch_failed,
        )
        self._prefill_observations.append(observation)


@final
class PagedAttentionExecutor:
    """Batch decode directly over Stage 13A physical block views."""

    def __init__(  # noqa: PLR0913
        self,
        model: GPT,
        paged_cache_pool: PagedKVCachePool,
        *,
        prefill_config: PrefillBatchConfig | None = None,
        prefix_prefill_strategy: APCPrefillStrategy = APCPrefillStrategy.SEQUENTIAL,
        clock: Clock = time.perf_counter,
        telemetry_clock: Clock = time.perf_counter,
    ) -> None:
        """Bind one model, its physical pool, dense prefill, and overflow fallback."""
        if (
            paged_cache_pool.n_layer != model.config.n_layer
            or paged_cache_pool.n_head != model.config.n_head
            or paged_cache_pool.head_size != model.config.n_embd // model.config.n_head
        ):
            _invalid("paged attention pool layout must match model")
        self._model = model
        self.paged_cache_pool = paged_cache_pool
        self._clock = clock
        self._telemetry_clock = telemetry_clock
        self.prefill_config = prefill_config or PrefillBatchConfig()
        self.prefix_prefill_strategy = prefix_prefill_strategy
        self._prefill = ContinuousExecutor(
            model,
            prefill_config=self.prefill_config,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._fallback = ReferenceExecutor(
            model,
            clock=clock,
            telemetry_clock=telemetry_clock,
        )
        self._decode_observations: list[DecodeBatchObservation] = []
        self._prefix_prefill_observations: list[PrefillBatchObservation] = []
        self.block_size = model.config.block_size

    @property
    def decode_observations(self) -> tuple[DecodeBatchObservation, ...]:
        """Return direct block-aware decode batch telemetry."""
        return tuple(self._decode_observations)

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return initial batched prefill plus dense overflow fallback telemetry."""
        return (
            *self._prefill.prefill_observations,
            *self._prefix_prefill_observations,
            *self._fallback.prefill_observations,
        )

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return resequenced initial and overflow prefill boundaries."""
        events: list[PrefillBatchEvent] = []
        for observation in self.prefill_observations:
            events.extend(_prefill_events(observation, starting_sequence=len(events)))
        return tuple(events)

    def prefill(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Reuse Stage 11B dense batched prefill before scattering into blocks."""
        if self.paged_cache_pool.prefix_cache_enabled:
            if self.prefix_prefill_strategy is APCPrefillStrategy.BATCHED:
                return self._prefill_prefix_cache_batched(requests)
            return tuple(self._prefill_prefix_cache_one(state) for state in requests)
        return self._prefill.prefill(requests)

    def prefill_chunks(
        self,
        requests: Sequence[RequestState],
        chunk_lengths: Sequence[int],
    ) -> tuple[ExecutionResult, ...]:
        """Evaluate one bounded prompt segment per request against paged history."""
        if len(requests) != len(chunk_lengths):
            _invalid("chunk length count must equal request count")
        if not requests:
            return ()
        starts: list[int] = []
        ends: list[int] = []
        for state, chunk_length in zip(requests, chunk_lengths, strict=True):
            if isinstance(chunk_length, bool) or chunk_length <= 0:
                _invalid("prefill chunk length must be a positive integer")
            start = self.paged_cache_pool.request_cache(state.request.request_id).cache_length
            end = start + chunk_length
            prompt_length = len(state.request.prompt_tokens)
            if end > prompt_length:
                _invalid("prefill chunk exceeds the remaining prompt")
            if end < prompt_length and end % self.paged_cache_pool.config.block_tokens != 0:
                _invalid("intermediate prefill chunks must end on a block boundary")
            starts.append(start)
            ends.append(end)
        return self._run_prefill_chunk_batch(requests, chunk_lengths, starts, ends)

    def _run_prefill_chunk_batch(
        self,
        requests: Sequence[RequestState],
        chunk_lengths: Sequence[int],
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> tuple[ExecutionResult, ...]:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        useful_tokens = sum(chunk_lengths)
        padded_tokens = len(requests) * max(chunk_lengths)
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        model_calls = 0

        def record(*, failed: bool) -> None:
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._prefix_prefill_observations.append(
                PrefillBatchObservation(
                    request_ids=request_ids,
                    batch_size=len(requests),
                    padded_prompt_tokens=padded_tokens,
                    useful_prompt_tokens=useful_tokens,
                    assembly_seconds=assembly_seconds,
                    model_seconds=model_seconds,
                    scatter_seconds=scatter_seconds,
                    executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
                    started_at=logical_start,
                    finished_at=logical_start + elapsed,
                    batch_failed=failed,
                    execution_mode=PrefillExecutionMode.CHUNKED_PAGED_PREFILL,
                    model_calls=model_calls,
                    prefix_hit_tokens=0,
                    avoided_prefill_tokens=0,
                )
            )

        try:
            assembly_start = self._telemetry_clock()
            token_ids, lengths, views, past_lengths = self._assemble_chunk_prefill(
                requests,
                chunk_lengths,
                starts,
                ends,
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            model_calls = 1
            logits, padded_delta = self._model.prefill_paged_batch(
                token_ids,
                lengths,
                views,
                past_lengths,
            )
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            results = self._scatter_chunk_prefill(
                requests,
                tuple(zip(chunk_lengths, starts, ends, strict=True)),
                logits,
                padded_delta,
            )
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            record(failed=True)
            elapsed = _logical_elapsed(self._clock, logical_start)
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )

        record(failed=False)
        elapsed = _logical_elapsed(self._clock, logical_start)
        return tuple(replace(result, latency_seconds=elapsed) for result in results)

    def _assemble_chunk_prefill(
        self,
        requests: Sequence[RequestState],
        chunk_lengths: Sequence[int],
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> tuple[Tensor, Tensor, tuple[PagedKVCacheView | None, ...], Tensor]:
        device = self._model.token_embedding.weight.device
        token_ids = torch.zeros(
            (len(requests), max(chunk_lengths)),
            dtype=torch.long,
            device=device,
        )
        views: list[PagedKVCacheView | None] = []
        for row, (state, chunk_length, start, end) in enumerate(
            zip(requests, chunk_lengths, starts, ends, strict=True)
        ):
            segment = state.request.prompt_tokens[start:end]
            if len(segment) != chunk_length:
                _invalid("prefill chunk length does not match prompt slice")
            token_ids[row, :chunk_length] = torch.tensor(
                segment,
                dtype=torch.long,
                device=device,
            )
            views.append(
                self.paged_cache_pool.request_view(state.request.request_id) if start else None
            )
        return (
            token_ids,
            torch.tensor(chunk_lengths, dtype=torch.long, device=device),
            tuple(views),
            torch.tensor(starts, dtype=torch.long, device=device),
        )

    def _scatter_chunk_prefill(
        self,
        requests: Sequence[RequestState],
        chunks: Sequence[tuple[int, int, int]],
        logits: Tensor,
        padded_delta: KVCache,
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for row, (state, chunk_length, start, end) in enumerate(
            ((row_state, *chunk) for row_state, chunk in zip(requests, chunks, strict=True))
        ):
            valid_logits = logits[row, :chunk_length].clone().detach()
            final_chunk = end == len(state.request.prompt_tokens)
            try:
                token_id = (
                    self._sample(state, logits[row : row + 1, chunk_length - 1, :])
                    if final_chunk
                    else None
                )
            except Exception as error:  # noqa: BLE001
                results.append(
                    ExecutionResult.failure(
                        state.request.request_id,
                        f"{type(error).__name__}: {error}",
                        0.0,
                    )
                )
                continue
            delta = tuple(
                LayerKVCache(
                    key=layer.key[row : row + 1, :, :chunk_length, :].clone().detach(),
                    value=layer.value[row : row + 1, :, :chunk_length, :].clone().detach(),
                )
                for layer in padded_delta
            )
            results.append(
                ExecutionResult(
                    request_id=state.request.request_id,
                    token_id=token_id,
                    cache=delta,
                    cache_tokens=end,
                    latency_seconds=0.0,
                    error=None,
                    used_fallback=False,
                    cache_is_delta=True,
                    prefill_prefix_tokens=start,
                    prefill_logits=valid_logits,
                    prefill_complete=final_chunk,
                )
            )
        return tuple(results)

    def _prefill_prefix_cache_batched(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[ExecutionResult, ...]:
        results: dict[str, ExecutionResult] = {}
        suffix_requests: list[RequestState] = []
        for state in requests:
            prompt_length = len(state.request.prompt_tokens)
            invalid_prefix = not 0 <= state.prefix_hit_tokens <= prompt_length
            invalid_prompt = self._validate_prefix_prefill_state(state) is not None
            if invalid_prefix or invalid_prompt or state.prefix_hit_tokens == prompt_length:
                result = self._prefill_prefix_cache_one(state)
                results[result.request_id] = result
            else:
                suffix_requests.append(state)
        for batch in self._prefix_batches(suffix_requests):
            for result in self._prefill_prefix_cache_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in requests)

    def _prefix_batches(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[tuple[RequestState, ...], ...]:
        batches: list[tuple[RequestState, ...]] = []
        current: list[RequestState] = []
        for state in requests:
            if not current:
                current.append(state)
                continue
            candidate = [*current, state]
            lengths = [
                len(item.request.prompt_tokens) - item.prefix_hit_tokens for item in candidate
            ]
            padded = len(candidate) * max(lengths)
            useful = sum(lengths)
            waste_ratio = (padded - useful) / padded
            config = self.prefill_config
            if (
                len(candidate) <= config.max_batch_size
                and padded <= config.max_batch_tokens
                and waste_ratio <= config.max_padding_ratio
            ):
                current.append(state)
            else:
                batches.append(tuple(current))
                current = [state]
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _validate_prefix_prefill_state(self, state: RequestState) -> str | None:
        tokens = state.request.prompt_tokens
        if not tokens:
            return "prefill prompt must be non-empty"
        if len(tokens) > self.block_size:
            return f"prefill prompt length {len(tokens)} exceeds block_size {self.block_size}"
        if any(type(token) is not int for token in tokens):
            return "prefill prompt tokens must be integers"
        if any(token < 0 or token >= self._model.config.vocab_size for token in tokens):
            return "prefill prompt token is outside model vocabulary"
        return None

    def _prefill_prefix_cache_batch(
        self,
        requests: Sequence[RequestState],
    ) -> tuple[ExecutionResult, ...]:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        computed_lengths = tuple(
            len(state.request.prompt_tokens) - state.prefix_hit_tokens for state in requests
        )
        prefix_tokens = sum(state.prefix_hit_tokens for state in requests)
        useful_tokens = sum(computed_lengths)
        padded_tokens = len(requests) * max(computed_lengths, default=0)
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        model_calls = 0
        try:
            assembly_start = self._telemetry_clock()
            token_ids, new_lengths, cache_views, past_lengths = self._assemble_prefix_prefill(
                requests, computed_lengths
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            model_calls = 1
            logits, padded_delta = self._model.prefill_paged_batch(
                token_ids,
                new_lengths,
                cache_views,
                past_lengths,
            )
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            results = self._scatter_prefix_prefill(
                requests,
                computed_lengths,
                logits,
                padded_delta,
            )
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._record_prefix_prefill_batch(
                request_ids=request_ids,
                useful_tokens=useful_tokens,
                padded_tokens=padded_tokens,
                prefix_hit_tokens=prefix_tokens,
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
                model_calls=model_calls,
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )
        elapsed = _logical_elapsed(self._clock, logical_start)
        self._record_prefix_prefill_batch(
            request_ids=request_ids,
            useful_tokens=useful_tokens,
            padded_tokens=padded_tokens,
            prefix_hit_tokens=prefix_tokens,
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
            model_calls=model_calls,
        )
        return tuple(replace(result, latency_seconds=elapsed) for result in results)

    def _assemble_prefix_prefill(
        self,
        requests: Sequence[RequestState],
        computed_lengths: Sequence[int],
    ) -> tuple[Tensor, Tensor, tuple[PagedKVCacheView | None, ...], Tensor]:
        device = self._model.token_embedding.weight.device
        token_ids = torch.zeros(
            (len(requests), max(computed_lengths)),
            dtype=torch.long,
            device=device,
        )
        views: list[PagedKVCacheView | None] = []
        past_lengths: list[int] = []
        for row, state in enumerate(requests):
            prefix_tokens = state.prefix_hit_tokens
            suffix = state.request.prompt_tokens[prefix_tokens:]
            token_ids[row, : len(suffix)] = torch.tensor(
                suffix,
                dtype=torch.long,
                device=device,
            )
            views.append(
                self.paged_cache_pool.request_view(state.request.request_id)
                if prefix_tokens
                else None
            )
            past_lengths.append(prefix_tokens)
        return (
            token_ids,
            torch.tensor(computed_lengths, dtype=torch.long, device=device),
            tuple(views),
            torch.tensor(past_lengths, dtype=torch.long, device=device),
        )

    def _scatter_prefix_prefill(
        self,
        requests: Sequence[RequestState],
        computed_lengths: Sequence[int],
        logits: Tensor,
        padded_delta: KVCache,
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for row, (state, computed_tokens) in enumerate(
            zip(requests, computed_lengths, strict=True)
        ):
            try:
                token_id = self._sample(state, logits[row : row + 1, computed_tokens - 1, :])
            except Exception as error:  # noqa: BLE001
                results.append(
                    ExecutionResult.failure(
                        state.request.request_id,
                        f"{type(error).__name__}: {error}",
                        0.0,
                    )
                )
                continue
            delta = tuple(
                LayerKVCache(
                    key=layer.key[row : row + 1, :, :computed_tokens, :].clone().detach(),
                    value=layer.value[row : row + 1, :, :computed_tokens, :].clone().detach(),
                )
                for layer in padded_delta
            )
            results.append(
                ExecutionResult.success(
                    request_id=state.request.request_id,
                    token_id=token_id,
                    cache=delta,
                    cache_tokens=len(state.request.prompt_tokens),
                    latency_seconds=0.0,
                    used_fallback=False,
                    prefill_prefix_tokens=state.prefix_hit_tokens,
                    prefill_logits=logits[row, :computed_tokens].clone().detach(),
                )
            )
        return tuple(results)

    def _prefill_prefix_cache_one(self, state: RequestState) -> ExecutionResult:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        prompt = state.request.prompt_tokens
        prefix_tokens = state.prefix_hit_tokens
        computed_tokens = len(prompt) - prefix_tokens
        try:
            if not 0 <= prefix_tokens <= len(prompt):
                _invalid("prefix hit tokens must be within the prompt")
            if any(token >= self._model.config.vocab_size for token in prompt):
                _invalid("prefill prompt token is outside model vocabulary")
            assembly_start = self._telemetry_clock()
            device = self._model.token_embedding.weight.device
            suffix = prompt[prefix_tokens:]
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            if not suffix:
                boundary = self.paged_cache_pool.prefix_boundary_logits_for_request(
                    state.request.request_id
                )
                logits = boundary.view(1, 1, -1)
                cache_delta: KVCache = ()
            else:
                token_ids = torch.tensor((suffix,), dtype=torch.long, device=device)
                if prefix_tokens:
                    logits, cache_delta = self._model.prefill_with_paged_prefix(
                        token_ids,
                        self.paged_cache_pool.request_view(state.request.request_id),
                        prefix_length=prefix_tokens,
                    )
                else:
                    logits, cache_delta = self._model.prefill_with_all_logits(token_ids)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            token_id = self._sample(state, logits[:, -1, :])
            suffix_logits = (
                logits[0].clone().detach() if suffix else logits.new_empty((0, logits.shape[-1]))
            )
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._record_prefix_prefill(
                state,
                computed_tokens=computed_tokens,
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                batch_failed=True,
            )
            return ExecutionResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                elapsed,
            )
        elapsed = _logical_elapsed(self._clock, logical_start)
        self._record_prefix_prefill(
            state,
            computed_tokens=computed_tokens,
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            batch_failed=False,
        )
        return ExecutionResult.success(
            request_id=state.request.request_id,
            token_id=token_id,
            cache=cache_delta,
            cache_tokens=len(prompt),
            latency_seconds=elapsed,
            used_fallback=False,
            prefill_prefix_tokens=prefix_tokens,
            prefill_logits=suffix_logits,
        )

    def _record_prefix_prefill(  # noqa: PLR0913
        self,
        state: RequestState,
        *,
        computed_tokens: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
    ) -> None:
        self._prefix_prefill_observations.append(
            PrefillBatchObservation(
                request_ids=(state.request.request_id,),
                batch_size=1,
                padded_prompt_tokens=computed_tokens,
                useful_prompt_tokens=computed_tokens,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
                started_at=logical_start,
                finished_at=logical_start + elapsed,
                batch_failed=batch_failed,
                execution_mode=(
                    PrefillExecutionMode.EXACT_CACHE_HIT
                    if computed_tokens == 0
                    else PrefillExecutionMode.SEQUENTIAL_APC_SUFFIX
                ),
                model_calls=0 if computed_tokens == 0 else 1,
                prefix_hit_tokens=state.prefix_hit_tokens,
                avoided_prefill_tokens=state.prefix_hit_tokens,
            )
        )

    def _record_prefix_prefill_batch(  # noqa: PLR0913
        self,
        *,
        request_ids: tuple[str, ...],
        useful_tokens: int,
        padded_tokens: int,
        prefix_hit_tokens: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        batch_failed: bool,
        model_calls: int,
    ) -> None:
        self._prefix_prefill_observations.append(
            PrefillBatchObservation(
                request_ids=request_ids,
                batch_size=len(request_ids),
                padded_prompt_tokens=padded_tokens,
                useful_prompt_tokens=useful_tokens,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
                started_at=logical_start,
                finished_at=logical_start + elapsed,
                batch_failed=batch_failed,
                execution_mode=PrefillExecutionMode.BATCHED_APC_SUFFIX,
                model_calls=model_calls,
                prefix_hit_tokens=prefix_hit_tokens,
                avoided_prefill_tokens=prefix_hit_tokens,
            )
        )

    def recompute_model_work(self, state: RequestState) -> int:
        """Return dense cache-only history work for one preempted request."""
        history = state.all_tokens[:-1][-self.block_size :]
        return len(history)

    def recompute_cache(self, state: RequestState) -> CacheRecomputeResult:
        """Rebuild resident history without sampling or advancing request RNG."""
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        history = state.all_tokens[:-1][-self.block_size :]
        if not history or not state.generated_tokens:
            return CacheRecomputeResult.failure(
                state.request.request_id,
                "recompute requires generated output and non-empty history",
                _logical_elapsed(self._clock, logical_start),
            )
        assembly_seconds = 0.0
        model_seconds = 0.0
        model_calls = 0
        try:
            assembly_start = self._telemetry_clock()
            inputs = torch.tensor(
                (history,),
                dtype=torch.long,
                device=self._model.token_embedding.weight.device,
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            model_calls = 1
            _logits, cache = self._model.prefill(inputs)
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            self._record_recompute(
                state,
                history_length=len(history),
                logical_start=logical_start,
                elapsed=elapsed,
                executor_start=executor_start,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                model_calls=model_calls,
                failed=True,
            )
            return CacheRecomputeResult.failure(
                state.request.request_id,
                f"{type(error).__name__}: {error}",
                elapsed,
            )
        elapsed = _logical_elapsed(self._clock, logical_start)
        self._record_recompute(
            state,
            history_length=len(history),
            logical_start=logical_start,
            elapsed=elapsed,
            executor_start=executor_start,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            model_calls=model_calls,
            failed=False,
        )
        return CacheRecomputeResult(
            request_id=state.request.request_id,
            cache=cache,
            cache_length=len(history),
            latency_seconds=elapsed,
            error=None,
        )

    def _record_recompute(  # noqa: PLR0913
        self,
        state: RequestState,
        *,
        history_length: int,
        logical_start: float,
        elapsed: float,
        executor_start: float,
        assembly_seconds: float,
        model_seconds: float,
        model_calls: int,
        failed: bool,
    ) -> None:
        self._prefix_prefill_observations.append(
            PrefillBatchObservation(
                request_ids=(state.request.request_id,),
                batch_size=1,
                padded_prompt_tokens=history_length,
                useful_prompt_tokens=history_length,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=0.0,
                executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
                started_at=logical_start,
                finished_at=logical_start + elapsed,
                batch_failed=failed,
                execution_mode=PrefillExecutionMode.PREEMPTION_RECOMPUTE,
                model_calls=model_calls,
            )
        )

    def decode(self, active_requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        """Isolate invalid/overflow rows and directly batch all other block views."""
        results: dict[str, ExecutionResult] = {}
        batch: list[RequestState] = []
        for state in active_requests:
            error = self._validate_decode_state(state)
            if error is not None:
                results[state.request.request_id] = ExecutionResult.failure(
                    state.request.request_id,
                    error,
                    0.0,
                )
            elif state.cached_tokens >= self.block_size:
                results[state.request.request_id] = self._fallback.prefill_fallback(state)
            else:
                batch.append(state)
        if batch:
            for result in self._decode_batch(batch):
                results[result.request_id] = result
        return tuple(results[state.request.request_id] for state in active_requests)

    def decode_model_work(self, state: RequestState) -> int:
        """Return actual model work charged for one Stage 16 decode operation."""
        error = self._validate_decode_state(state)
        if error is not None:
            return 0
        if state.cached_tokens >= self.block_size:
            return len(state.all_tokens[-self.block_size :])
        return 1

    def _decode_batch(self, requests: Sequence[RequestState]) -> tuple[ExecutionResult, ...]:
        logical_start = self._clock()
        executor_start = self._telemetry_clock()
        request_ids = tuple(state.request.request_id for state in requests)
        useful_tokens = sum(state.cached_tokens for state in requests)
        assembly_seconds = 0.0
        model_seconds = 0.0
        scatter_seconds = 0.0
        try:
            assembly_start = self._telemetry_clock()
            device = self._model.token_embedding.weight.device
            token_ids = torch.tensor(
                [[state.generated_tokens[-1]] for state in requests],
                dtype=torch.long,
                device=device,
            )
            cache_lengths = torch.tensor(
                [state.cached_tokens for state in requests],
                dtype=torch.long,
                device=device,
            )
            cache_views = tuple(
                self.paged_cache_pool.request_view(state.request.request_id) for state in requests
            )
            assembly_seconds = max(0.0, self._telemetry_clock() - assembly_start)
            model_start = self._telemetry_clock()
            logits, cache_delta = self._model.decode_paged_batch(
                token_ids,
                cache_views,
                cache_lengths,
            )
            model_seconds = max(0.0, self._telemetry_clock() - model_start)
            scatter_start = self._telemetry_clock()
            results = self._scatter_results(requests, logits, cache_delta)
            scatter_seconds = max(0.0, self._telemetry_clock() - scatter_start)
        except Exception as error:  # noqa: BLE001
            elapsed = _logical_elapsed(self._clock, logical_start)
            executor_seconds = max(0.0, self._telemetry_clock() - executor_start)
            self._record_decode(
                request_ids=request_ids,
                useful_tokens=useful_tokens,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=executor_seconds,
                batch_failed=True,
            )
            message = f"{type(error).__name__}: {error}"
            return tuple(
                ExecutionResult.failure(state.request.request_id, message, elapsed)
                for state in requests
            )
        elapsed = _logical_elapsed(self._clock, logical_start)
        results = tuple(replace(result, latency_seconds=elapsed) for result in results)
        self._record_decode(
            request_ids=request_ids,
            useful_tokens=useful_tokens,
            assembly_seconds=assembly_seconds,
            model_seconds=model_seconds,
            scatter_seconds=scatter_seconds,
            executor_seconds=max(0.0, self._telemetry_clock() - executor_start),
            batch_failed=False,
        )
        return results

    def _scatter_results(
        self,
        requests: Sequence[RequestState],
        logits: Tensor,
        cache_delta: KVCache,
    ) -> tuple[ExecutionResult, ...]:
        results: list[ExecutionResult] = []
        for row, state in enumerate(requests):
            try:
                token_id = self._sample(state, logits[row : row + 1, -1, :])
            except Exception as error:  # noqa: BLE001
                results.append(
                    ExecutionResult.failure(
                        state.request.request_id,
                        f"{type(error).__name__}: {error}",
                        0.0,
                    )
                )
                continue
            delta = tuple(
                LayerKVCache(
                    key=layer.key[row : row + 1].detach(),
                    value=layer.value[row : row + 1].detach(),
                )
                for layer in cache_delta
            )
            results.append(
                ExecutionResult.success(
                    request_id=state.request.request_id,
                    token_id=token_id,
                    cache=delta,
                    cache_tokens=state.cached_tokens + 1,
                    latency_seconds=0.0,
                    used_fallback=False,
                    cache_is_delta=True,
                )
            )
        return tuple(results)

    def _validate_decode_state(self, state: RequestState) -> str | None:
        if not state.generated_tokens or state.cached_tokens <= 0:
            return "paged decode requires a populated cache and generated token"
        request_id = state.request.request_id
        if not self.paged_cache_pool.has_request(request_id):
            return "paged decode requires an owned request block table"
        table = self.paged_cache_pool.request_cache(request_id)
        if table.cache_length != state.cached_tokens:
            return "paged request block table length differs from engine cache occupancy"
        if state.cached_tokens > self.block_size:
            return "paged cache token length exceeds model capacity"
        return None

    def _sample(self, state: RequestState, logits: Tensor) -> int:
        scaled = logits / state.request.temperature
        if state.request.top_k is not None:
            retained_count = min(state.request.top_k, self._model.config.vocab_size)
            retained = torch.topk(scaled, retained_count, dim=-1).values
            cutoff = retained[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < cutoff, -torch.inf)
        probabilities = functional.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=state.generator)
        return int(sampled.item())

    def _record_decode(  # noqa: PLR0913
        self,
        *,
        request_ids: tuple[str, ...],
        useful_tokens: int,
        assembly_seconds: float,
        model_seconds: float,
        scatter_seconds: float,
        executor_seconds: float,
        batch_failed: bool,
    ) -> None:
        self._decode_observations.append(
            DecodeBatchObservation(
                request_ids=request_ids,
                batch_size=len(request_ids),
                padded_cache_tokens=useful_tokens,
                useful_cache_tokens=useful_tokens,
                assembly_seconds=assembly_seconds,
                model_seconds=model_seconds,
                scatter_seconds=scatter_seconds,
                executor_seconds=executor_seconds,
                batch_failed=batch_failed,
            )
        )


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """Record one ordered transition with a scheduler snapshot."""

    sequence: int
    timestamp: float
    event_type: EngineEventType
    request_id: str
    status: RequestStatus
    token_id: int | None
    detail: str | None
    used_fallback: bool
    active_requests: int
    waiting_requests: int
    cached_tokens: int
    reserved_cache_tokens: int


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Summarize one request's latency and output accounting."""

    request_id: str
    status: RequestStatus
    queue_time_seconds: float | None
    prefill_latency_seconds: float | None
    time_to_first_token_seconds: float | None
    decode_latencies_seconds: tuple[float, ...]
    time_per_output_token_seconds: float | None
    end_to_end_latency_seconds: float | None
    generated_tokens: int
    failure_reason: str | None
    prefix_hit_blocks: int
    prefix_hit_tokens: int
    prefix_miss_tokens: int
    prefill_tokens_computed: int
    preemption_count: int
    resume_count: int
    recompute_tokens: int


@dataclass(frozen=True, slots=True)
class EngineMetrics:
    """Summarize lifecycle counts, capacity, occupancy, and throughput."""

    total_requests: int
    completed_requests: int
    cancelled_requests: int
    failed_requests: int
    active_requests: int
    waiting_requests: int
    cached_tokens: int
    reserved_cache_tokens: int
    kv_cache_backend: KVCacheBackend
    total_blocks: int
    free_blocks: int
    allocated_blocks: int
    reserved_blocks: int
    peak_allocated_blocks: int
    peak_reserved_blocks: int
    used_token_slots: int
    allocated_token_slots: int
    internal_fragmentation_tokens: int
    internal_fragmentation_ratio: float
    allocation_count: int
    free_count: int
    block_reuse_count: int
    peak_active_requests: int
    peak_waiting_requests: int
    peak_cached_tokens: int
    peak_reserved_cache_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    request_throughput_per_second: float
    token_throughput_per_second: float
    decode_batch_sizes: tuple[int, ...]
    average_decode_batch_size: float
    max_decode_batch_size: int
    padded_cache_tokens: int
    useful_cache_tokens: int
    padding_waste_ratio: float
    executor_time_seconds: float
    model_execution_time_seconds: float
    batch_assembly_scatter_time_seconds: float
    prefill_batch_sizes: tuple[int, ...]
    average_prefill_batch_size: float
    max_prefill_batch_size: int
    padded_prompt_tokens: int
    useful_prompt_tokens: int
    prompt_padding_waste_ratio: float
    prefill_executor_time_seconds: float
    prefill_model_execution_time_seconds: float
    prefill_batch_assembly_scatter_time_seconds: float
    cache_aware_prefill_batches: int
    cache_aware_prefill_model_calls: int
    suffix_prefill_batch_sizes: tuple[int, ...]
    average_suffix_prefill_batch_size: float
    max_suffix_prefill_batch_size: int
    suffix_useful_tokens: int
    suffix_padded_tokens: int
    suffix_padding_waste_ratio: float
    exact_cache_hit_requests: int
    batched_suffix_requests: int
    chunked_prefill_batches: int
    chunked_prefill_chunks: int
    chunked_prefill_useful_tokens: int
    prefix_cache_enabled: bool
    prefix_cache_blocks: int
    evictable_blocks: int
    active_shared_blocks: int
    active_shared_references: int
    prefix_cache_evictions: int
    prefix_lookup_requests: int
    prefix_hit_requests: int
    prefix_hit_blocks: int
    prefix_hit_tokens: int
    prefix_miss_tokens: int
    prefill_tokens_computed: int
    avoided_prefill_tokens: int
    prefix_hit_request_ratio: float
    prefix_hit_token_ratio: float
    preemptions: int
    resumes: int
    recompute_tokens: int


def _validate_chunked_engine(
    config: EngineConfig,
    executor: ServingExecutor,
    paged_cache_pool: PagedKVCachePool | None,
    *,
    direct_paged_decode: bool,
) -> None:
    chunk_size = cast("int | None", getattr(config.scheduler, "prefill_chunk_" + "tokens"))
    if config.scheduler.kv_preemption and chunk_size is None:
        _invalid("kv_preemption requires Stage 16 token-budget scheduling")
    if chunk_size is None:
        return
    if not direct_paged_decode or paged_cache_pool is None:
        _invalid("chunked scheduling requires the direct paged executor")
    block_size = paged_cache_pool.config.block_tokens
    if chunk_size % block_size:
        _invalid("prefill chunk size must align to paged blocks")
    if chunk_size > config.block_size:
        _invalid("prefill chunk size must not exceed model block size")
    budget = cast("int | None", getattr(config.scheduler, "max_scheduled_" + "tokens"))
    minimum_budget = max(config.block_size, config.scheduler.max_active_requests - 1 + block_size)
    if budget is None or budget < minimum_budget:
        _invalid("scheduled work budget is too small")
    if not isinstance(executor, PagedAttentionExecutor):
        _invalid("chunked scheduling requires the direct paged executor")
    batch_limit = cast("int", getattr(executor.prefill_config, "max_batch_" + "tokens"))
    if batch_limit < block_size:
        _invalid("prefill batch limit must fit one paged block")


@final
class ServingEngine:
    """Advance independent requests through a deterministic FIFO engine loop."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        executor: ServingExecutor,
        paged_cache_pool: PagedKVCachePool | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        """Bind deterministic scheduler state to an executor."""
        if executor.block_size != config.block_size:
            reason = "executor block_size must equal EngineConfig.block_size"
            _invalid(reason)
        if config.kv_cache_backend is KVCacheBackend.DENSE and paged_cache_pool is not None:
            _invalid("dense backend must not receive a paged cache pool")
        if config.kv_cache_backend is KVCacheBackend.PAGED:
            if paged_cache_pool is None:
                _invalid("paged backend requires a paged cache pool")
            if paged_cache_pool.config != config.paged_kv_cache:
                _invalid("paged cache pool config must equal EngineConfig.paged_kv_cache")
        direct_paged_decode = isinstance(executor, PagedAttentionExecutor)
        if direct_paged_decode and executor.paged_cache_pool is not paged_cache_pool:
            _invalid("paged attention executor and engine must share one cache pool")
        _validate_chunked_engine(
            config,
            executor,
            paged_cache_pool,
            direct_paged_decode=direct_paged_decode,
        )
        self.config = config
        self._executor = executor
        self._paged_cache_pool = paged_cache_pool
        self._direct_paged_decode = direct_paged_decode
        self._clock = clock
        self._states: dict[str, RequestState] = {}
        self._waiting: deque[str] = deque()
        self._active: list[str] = []
        self._pending_cancellations: set[str] = set()
        self._events: list[EngineEvent] = []
        self._last_tick_time: float | None = None
        self._peak_active = 0
        self._peak_waiting = 0
        self._peak_cached = 0
        self._peak_reserved = 0
        self._chunk_schedule_cursor: str | None = None
        self._chunk_resume_decode: str | None = None

    @property
    def events(self) -> tuple[EngineEvent, ...]:
        """Return an immutable snapshot of the append-only event stream."""
        return tuple(self._events)

    @property
    def prefill_events(self) -> tuple[PrefillBatchEvent, ...]:
        """Return executor-level prompt batch evidence separately from request events."""
        return self._executor.prefill_events

    @property
    def prefill_observations(self) -> tuple[PrefillBatchObservation, ...]:
        """Return executor observations with explicit prefill execution modes."""
        return self._executor.prefill_observations

    @property
    def paged_cache_pool(self) -> PagedKVCachePool | None:
        """Expose the optional storage backend for owner-thread diagnostics."""
        return self._paged_cache_pool

    @property
    def is_idle(self) -> bool:
        """Return whether no waiting or active request remains."""
        return not self._waiting and not self._active

    def verify_cache_invariants(self) -> None:
        """Verify storage ownership when the paged backend is active."""
        if self._paged_cache_pool is not None:
            self._paged_cache_pool.verify_invariants()

    def release_all_cache_resources(self) -> None:
        """Drop every reservation after a catastrophic owner-thread failure."""
        if self._paged_cache_pool is not None:
            self._paged_cache_pool.release_all()
        for state in self._states.values():
            state.kv_cache = None
            state.prefill_logits_chunks.clear()
            state.cached_tokens = 0
            state.reserved_cache_tokens = 0
            state.reserved_cache_blocks = 0
        self._active.clear()
        self._waiting.clear()
        self._pending_cancellations.clear()
        self._chunk_schedule_cursor = None
        self._chunk_resume_decode = None

    def submit(self, request: GenerationRequest) -> None:
        """Append a validated request to the FIFO waiting queue."""
        if request.request_id in self._states:
            raise DuplicateRequestError(request.request_id)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        state = RequestState(request=request, generator=generator)
        self._states[request.request_id] = state
        self._waiting.append(request.request_id)
        self._emit(
            event_type=EngineEventType.SUBMITTED,
            state=state,
            timestamp=request.arrival_time,
        )

    def cancel(self, request_id: str, *, at: float | None = None) -> None:
        """Queue an idempotent cancellation for the next tick's first phase."""
        state = self.request_state(request_id)
        if state.status in self._terminal_statuses():
            return
        self._pending_cancellations.add(request_id)
        timestamp = self._clock() if at is None else at
        _finite_non_negative(timestamp, "cancellation timestamp")
        self._emit(
            event_type=EngineEventType.CANCELLATION_REQUESTED,
            state=state,
            timestamp=timestamp,
        )

    def request_state(self, request_id: str) -> RequestState:
        """Return mutable state for inspection or executor input."""
        try:
            return self._states[request_id]
        except KeyError:
            raise UnknownRequestError(request_id) from None

    def tick(self, *, now: float | None = None) -> None:
        """Apply cancellation, admission, then at-most-once prefill/decode work."""
        tick_time = self._clock() if now is None else now
        _finite_non_negative(tick_time, "tick time")
        if self._last_tick_time is not None and tick_time < self._last_tick_time:
            _invalid("tick time must be monotonic")
        self._last_tick_time = tick_time
        if self.config.scheduler.prefill_chunk_tokens is not None:
            self._tick_chunked(tick_time)
            self._maybe_preempt_for_pressure(tick_time)
            self._update_peaks()
            return
        prefill_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status is RequestStatus.PREFILLING
        )
        decode_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status is RequestStatus.DECODING
        )

        self._apply_cancellations(tick_time)
        self._admit(tick_time)
        self._execute(prefill_ids, tick_time=tick_time, prefill=True)
        self._execute(decode_ids, tick_time=tick_time, prefill=False)
        self._update_peaks()

    def _tick_chunked(self, tick_time: float) -> None:
        scheduler = self.config.scheduler
        chunk_size = scheduler.prefill_chunk_tokens
        budget = scheduler.max_scheduled_tokens
        pool = self._paged_cache_pool
        executor = self._executor
        if (
            chunk_size is None
            or budget is None
            or pool is None
            or not isinstance(executor, PagedAttentionExecutor)
        ):
            _invalid("chunked prefill configuration is incomplete")
        block_size = pool.config.block_tokens
        if chunk_size % block_size:
            _invalid("prefill chunk size must align to paged blocks")
        minimum_budget = scheduler.max_active_requests - 1 + block_size
        if budget < minimum_budget:
            _invalid("scheduled work budget is too small")

        prefill_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status is RequestStatus.PREFILLING
        )
        decode_ids = tuple(
            request_id
            for request_id in self._active
            if self._states[request_id].status
            in {RequestStatus.DECODING, RequestStatus.RECOMPUTING}
        )
        self._apply_cancellations(tick_time)
        self._admit(tick_time)

        live_prefill = tuple(
            request_id
            for request_id in prefill_ids
            if self._states[request_id].status is RequestStatus.PREFILLING
        )
        live_decode = tuple(
            request_id
            for request_id in decode_ids
            if self._states[request_id].status
            in {RequestStatus.DECODING, RequestStatus.RECOMPUTING}
        )
        if self._chunk_schedule_cursor in live_prefill:
            self._run_chunked_prefill_first(
                live_prefill,
                live_decode,
                budget=budget,
                tick_time=tick_time,
            )
            return
        selected_decode, decode_work, deferred_decode = self._select_chunked_decode(
            live_decode, budget=budget
        )
        if selected_decode:
            self._execute_chunked_decode_fifo(selected_decode, tick_time=tick_time)
        remaining_budget = budget - decode_work
        if deferred_decode is not None:
            if live_prefill:
                self._chunk_resume_decode = deferred_decode
                self._chunk_schedule_cursor = live_prefill[0]
            else:
                self._chunk_schedule_cursor = deferred_decode
            return

        exact_ids = tuple(
            request_id
            for request_id in live_prefill
            if pool.request_cache(request_id).cache_length
            == len(self._states[request_id].all_tokens)
            - len(self._states[request_id].generated_tokens)
        )
        self._execute(exact_ids, tick_time=tick_time, prefill=True)
        exact_set = set(exact_ids)

        selected_ids, selected_lengths = self._select_chunked_prefill(
            live_prefill,
            excluded=exact_set,
            budget=remaining_budget,
        )
        if selected_ids:
            self._execute_prefill_chunks(
                selected_ids,
                selected_lengths,
                tick_time=tick_time,
            )
        selected_set = set(selected_ids)
        self._chunk_schedule_cursor = next(
            (
                request_id
                for request_id in live_prefill
                if request_id not in exact_set and request_id not in selected_set
            ),
            None,
        )

    def _execute_chunked_decode_fifo(
        self,
        request_ids: tuple[str, ...],
        *,
        tick_time: float,
    ) -> None:
        executor = self._executor
        if not isinstance(executor, PagedAttentionExecutor):
            _invalid("chunked decode execution requires the direct paged executor")
        normal_batch: list[str] = []
        for request_id in request_ids:
            state = self._states[request_id]
            if state.status is RequestStatus.RECOMPUTING:
                if normal_batch:
                    self._execute(tuple(normal_batch), tick_time=tick_time, prefill=False)
                    normal_batch.clear()
                self._execute_recompute(state, tick_time=tick_time)
                continue
            work = executor.decode_model_work(state)
            if work == 1:
                normal_batch.append(request_id)
                continue
            if normal_batch:
                self._execute(tuple(normal_batch), tick_time=tick_time, prefill=False)
                normal_batch.clear()
            self._execute((request_id,), tick_time=tick_time, prefill=False)
        if normal_batch:
            self._execute(tuple(normal_batch), tick_time=tick_time, prefill=False)

    def _execute_recompute(self, state: RequestState, *, tick_time: float) -> None:
        executor = self._executor
        pool = self._paged_cache_pool
        if not isinstance(executor, PagedAttentionExecutor) or pool is None:
            _invalid("preemption recompute requires the direct paged executor")
        if state.status is not RequestStatus.RECOMPUTING:
            _invalid("recompute requires a RECOMPUTING request")
        self._emit(
            event_type=EngineEventType.RECOMPUTE_STARTED,
            state=state,
            timestamp=tick_time,
        )
        result = executor.recompute_cache(state)
        if result.error is not None or result.cache is None:
            self._fail_state(
                state,
                tick_time + result.latency_seconds,
                result.error or "recompute failed",
            )
            return
        try:
            pool.write_prefill(state.request.request_id, result.cache)
        except Exception as error:  # noqa: BLE001
            self._fail_state(
                state,
                tick_time + result.latency_seconds,
                f"{type(error).__name__}: {error}",
            )
            return
        state.kv_cache = ()
        state.cached_tokens = result.cache_length
        state.recompute_tokens += result.cache_length
        state.resume_count += 1
        state.last_recompute_time = tick_time
        state.status = RequestStatus.DECODING
        self._emit(
            event_type=EngineEventType.RESUMED,
            state=state,
            timestamp=tick_time + result.latency_seconds,
            detail=f"recompute_tokens={result.cache_length}",
        )

    def _run_chunked_prefill_first(
        self,
        live_prefill: tuple[str, ...],
        live_decode: tuple[str, ...],
        *,
        budget: int,
        tick_time: float,
    ) -> None:
        _, pending_prefill = self._run_chunked_prefill_phase(
            live_prefill,
            budget=budget,
            tick_time=tick_time,
        )
        resume_decode = self._chunk_resume_decode
        if resume_decode in live_decode:
            self._chunk_schedule_cursor = resume_decode
        elif live_decode:
            self._chunk_schedule_cursor = live_decode[0]
        else:
            self._chunk_schedule_cursor = pending_prefill
        self._chunk_resume_decode = None

    def _rotate_chunked_ids(
        self,
        request_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        cursor = self._chunk_schedule_cursor
        if cursor is None or cursor not in request_ids:
            return request_ids
        index = request_ids.index(cursor)
        return (*request_ids[index:], *request_ids[:index])

    def _select_chunked_decode(
        self, request_ids: tuple[str, ...], *, budget: int
    ) -> tuple[tuple[str, ...], int, str | None]:
        executor = self._executor
        if not isinstance(executor, PagedAttentionExecutor):
            _invalid("chunked decode selection requires the direct paged executor")
        ordered = self._rotate_chunked_ids(request_ids)
        selected: list[str] = []
        used = 0
        deferred: str | None = None
        for request_id in ordered:
            state = self._states[request_id]
            work = (
                executor.recompute_model_work(state)
                if state.status is RequestStatus.RECOMPUTING
                else executor.decode_model_work(state)
            )
            if work > budget - used:
                deferred = request_id
                break
            selected.append(request_id)
            used += work
        return tuple(selected), used, deferred

    def _run_chunked_prefill_phase(
        self,
        request_ids: tuple[str, ...],
        *,
        budget: int,
        tick_time: float,
    ) -> tuple[int, str | None]:
        pool = self._paged_cache_pool
        if pool is None:
            _invalid("chunked prefill phase requires a paged cache pool")
        ordered = self._rotate_chunked_ids(request_ids)
        exact_ids = tuple(
            request_id
            for request_id in ordered
            if pool.request_cache(request_id).cache_length
            == len(self._states[request_id].request.prompt_tokens)
        )
        if exact_ids:
            self._execute(exact_ids, tick_time=tick_time, prefill=True)
        exact_set = set(exact_ids)
        candidates = tuple(request_id for request_id in ordered if request_id not in exact_set)
        selected_ids, selected_lengths = self._select_chunked_prefill(
            candidates,
            excluded=set(),
            budget=budget,
        )
        if selected_ids:
            self._execute_prefill_chunks(
                selected_ids,
                selected_lengths,
                tick_time=tick_time,
            )
        selected_set = set(selected_ids)
        pending = next(
            (request_id for request_id in candidates if request_id not in selected_set), None
        )
        return sum(selected_lengths), pending

    def _select_chunked_prefill(
        self,
        request_ids: tuple[str, ...],
        *,
        excluded: set[str],
        budget: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        pool = self._paged_cache_pool
        executor = self._executor
        chunk_size = self.config.scheduler.prefill_chunk_tokens
        if pool is None or chunk_size is None or not isinstance(executor, PagedAttentionExecutor):
            _invalid("chunked prefill selection requires the direct paged executor")
        block_size = pool.config.block_tokens
        selected_ids: list[str] = []
        selected_lengths: list[int] = []
        remaining_budget = budget
        for request_id in request_ids:
            if request_id in excluded or remaining_budget <= 0:
                continue
            state = self._states[request_id]
            start = pool.request_cache(request_id).cache_length
            prompt_length = len(state.all_tokens) - len(state.generated_tokens)
            remaining_prompt = prompt_length - start
            if remaining_prompt <= 0:
                continue
            length = min(
                chunk_size,
                remaining_budget,
                remaining_prompt,
                executor.prefill_config.max_batch_tokens,
            )
            if length < remaining_prompt:
                length -= length % block_size
            if length <= 0:
                break
            candidate_lengths = [*selected_lengths, length]
            padded = len(candidate_lengths) * max(candidate_lengths)
            useful = sum(candidate_lengths)
            padding_ratio = (padded - useful) / padded
            prefill_config = executor.prefill_config
            if (
                len(candidate_lengths) > prefill_config.max_batch_size
                or padded > prefill_config.max_batch_tokens
                or padding_ratio > prefill_config.max_padding_ratio
            ):
                break
            selected_ids.append(request_id)
            selected_lengths.append(length)
            remaining_budget -= length
        return tuple(selected_ids), tuple(selected_lengths)

    def run_until_idle(
        self,
        *,
        start_time: float,
        tick_seconds: float,
        max_ticks: int,
    ) -> None:
        """Advance a finite workload on a deterministic logical clock."""
        _finite_non_negative(start_time, "start_time")
        if not math.isfinite(tick_seconds) or tick_seconds <= 0.0:
            _invalid("tick_seconds must be finite and positive")
        if isinstance(max_ticks, bool) or max_ticks <= 0:
            _invalid("max_ticks must be a positive integer")
        for tick_index in range(max_ticks):
            if self.is_idle:
                return
            self.tick(now=start_time + tick_index * tick_seconds)
        if not self.is_idle:
            raise EngineRunawayError(max_ticks)

    def request_metrics(self, request_id: str) -> RequestMetrics:
        """Calculate stable latency definitions for one request."""
        state = self.request_state(request_id)
        arrival = state.request.arrival_time
        queue_time = (
            None if state.admission_time is None else max(0.0, state.admission_time - arrival)
        )
        ttft = None if state.first_token_time is None else state.first_token_time - arrival
        e2e = None if state.finish_time is None else state.finish_time - arrival
        decode_latencies = tuple(state.decode_latencies_seconds)
        tpot = None
        if state.first_token_time is not None:
            tpot = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
        return RequestMetrics(
            request_id=request_id,
            status=state.status,
            queue_time_seconds=queue_time,
            prefill_latency_seconds=state.prefill_latency_seconds,
            time_to_first_token_seconds=ttft,
            decode_latencies_seconds=decode_latencies,
            time_per_output_token_seconds=tpot,
            end_to_end_latency_seconds=e2e,
            generated_tokens=len(state.generated_tokens),
            failure_reason=state.failure_reason,
            prefix_hit_blocks=state.prefix_hit_blocks,
            prefix_hit_tokens=state.prefix_hit_tokens,
            prefix_miss_tokens=state.prefix_miss_tokens,
            prefill_tokens_computed=state.prefill_tokens_computed,
            preemption_count=state.preemption_count,
            resume_count=state.resume_count,
            recompute_tokens=state.recompute_tokens,
        )

    def metrics(self) -> EngineMetrics:
        """Calculate engine totals and descriptive workload throughput."""
        states = tuple(self._states.values())
        completed = sum(state.status is RequestStatus.FINISHED for state in states)
        cancelled = sum(state.status is RequestStatus.CANCELLED for state in states)
        failed = sum(state.status is RequestStatus.FAILED for state in states)
        generated = sum(len(state.generated_tokens) for state in states)
        finishes = [state.finish_time for state in states if state.finish_time is not None]
        if states and finishes:
            elapsed = max(0.0, max(finishes) - min(state.request.arrival_time for state in states))
        else:
            elapsed = 0.0
        request_throughput = completed / elapsed if elapsed > 0.0 else 0.0
        token_throughput = generated / elapsed if elapsed > 0.0 else 0.0
        observations = self._executor.decode_observations
        batch_sizes = tuple(observation.batch_size for observation in observations)
        padded_cache_tokens = sum(observation.padded_cache_tokens for observation in observations)
        useful_cache_tokens = sum(observation.useful_cache_tokens for observation in observations)
        padding_waste = padded_cache_tokens - useful_cache_tokens
        padding_waste_ratio = (
            padding_waste / padded_cache_tokens if padded_cache_tokens > 0 else 0.0
        )
        prefill_observations = self._executor.prefill_observations
        prefill_batch_sizes = tuple(observation.batch_size for observation in prefill_observations)
        padded_prompt_tokens = sum(
            observation.padded_prompt_tokens for observation in prefill_observations
        )
        useful_prompt_tokens = sum(
            observation.useful_prompt_tokens for observation in prefill_observations
        )
        prompt_padding_waste = padded_prompt_tokens - useful_prompt_tokens
        prompt_padding_waste_ratio = (
            prompt_padding_waste / padded_prompt_tokens if padded_prompt_tokens > 0 else 0.0
        )
        apc_observations = tuple(
            observation
            for observation in prefill_observations
            if observation.execution_mode
            in {
                PrefillExecutionMode.SEQUENTIAL_APC_SUFFIX,
                PrefillExecutionMode.BATCHED_APC_SUFFIX,
                PrefillExecutionMode.EXACT_CACHE_HIT,
            }
        )
        suffix_observations = tuple(
            observation
            for observation in apc_observations
            if observation.execution_mode
            in {
                PrefillExecutionMode.SEQUENTIAL_APC_SUFFIX,
                PrefillExecutionMode.BATCHED_APC_SUFFIX,
            }
        )
        batched_suffix_observations = tuple(
            observation
            for observation in suffix_observations
            if observation.execution_mode is PrefillExecutionMode.BATCHED_APC_SUFFIX
        )
        chunked_prefill_observations = tuple(
            observation
            for observation in prefill_observations
            if observation.execution_mode is PrefillExecutionMode.CHUNKED_PAGED_PREFILL
        )
        suffix_batch_sizes = tuple(observation.batch_size for observation in suffix_observations)
        suffix_useful_tokens = sum(
            observation.useful_prompt_tokens for observation in suffix_observations
        )
        suffix_padded_tokens = sum(
            observation.padded_prompt_tokens for observation in suffix_observations
        )
        cache_metrics = (
            self._paged_cache_pool.metrics()
            if self._paged_cache_pool is not None
            else _EMPTY_PAGED_METRICS
        )
        prefill_tokens_computed = sum(state.prefill_tokens_computed for state in states)
        prompt_tokens = sum(len(state.request.prompt_tokens) for state in states)
        avoided_prefill_tokens = max(0, prompt_tokens - prefill_tokens_computed)
        lookup_requests = cache_metrics.prefix_lookup_requests
        hit_request_ratio = (
            cache_metrics.prefix_hit_requests / lookup_requests if lookup_requests else 0.0
        )
        lookup_tokens = cache_metrics.prefix_hit_tokens + cache_metrics.prefix_miss_tokens
        hit_token_ratio = cache_metrics.prefix_hit_tokens / lookup_tokens if lookup_tokens else 0.0
        return EngineMetrics(
            total_requests=len(states),
            completed_requests=completed,
            cancelled_requests=cancelled,
            failed_requests=failed,
            active_requests=len(self._active),
            waiting_requests=len(self._waiting),
            cached_tokens=self._cached_tokens(),
            reserved_cache_tokens=self._reserved_cache_tokens(),
            kv_cache_backend=self.config.kv_cache_backend,
            total_blocks=cache_metrics.total_blocks,
            free_blocks=cache_metrics.free_blocks,
            allocated_blocks=cache_metrics.allocated_blocks,
            reserved_blocks=cache_metrics.reserved_blocks,
            peak_allocated_blocks=cache_metrics.peak_allocated_blocks,
            peak_reserved_blocks=cache_metrics.peak_reserved_blocks,
            used_token_slots=cache_metrics.used_token_slots,
            allocated_token_slots=cache_metrics.allocated_token_slots,
            internal_fragmentation_tokens=cache_metrics.internal_fragmentation_tokens,
            internal_fragmentation_ratio=cache_metrics.internal_fragmentation_ratio,
            allocation_count=cache_metrics.allocation_count,
            free_count=cache_metrics.free_count,
            block_reuse_count=cache_metrics.block_reuse_count,
            peak_active_requests=self._peak_active,
            peak_waiting_requests=self._peak_waiting,
            peak_cached_tokens=self._peak_cached,
            peak_reserved_cache_tokens=self._peak_reserved,
            generated_tokens=generated,
            elapsed_seconds=elapsed,
            request_throughput_per_second=request_throughput,
            token_throughput_per_second=token_throughput,
            decode_batch_sizes=batch_sizes,
            average_decode_batch_size=(sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0),
            max_decode_batch_size=max(batch_sizes, default=0),
            padded_cache_tokens=padded_cache_tokens,
            useful_cache_tokens=useful_cache_tokens,
            padding_waste_ratio=padding_waste_ratio,
            executor_time_seconds=sum(observation.executor_seconds for observation in observations),
            model_execution_time_seconds=sum(
                observation.model_seconds for observation in observations
            ),
            batch_assembly_scatter_time_seconds=sum(
                observation.assembly_seconds + observation.scatter_seconds
                for observation in observations
            ),
            prefill_batch_sizes=prefill_batch_sizes,
            average_prefill_batch_size=(
                sum(prefill_batch_sizes) / len(prefill_batch_sizes) if prefill_batch_sizes else 0.0
            ),
            max_prefill_batch_size=max(prefill_batch_sizes, default=0),
            padded_prompt_tokens=padded_prompt_tokens,
            useful_prompt_tokens=useful_prompt_tokens,
            prompt_padding_waste_ratio=prompt_padding_waste_ratio,
            prefill_executor_time_seconds=sum(
                observation.executor_seconds for observation in prefill_observations
            ),
            prefill_model_execution_time_seconds=sum(
                observation.model_seconds for observation in prefill_observations
            ),
            prefill_batch_assembly_scatter_time_seconds=sum(
                observation.assembly_seconds + observation.scatter_seconds
                for observation in prefill_observations
            ),
            cache_aware_prefill_batches=len(batched_suffix_observations),
            cache_aware_prefill_model_calls=sum(
                observation.model_calls for observation in apc_observations
            ),
            suffix_prefill_batch_sizes=suffix_batch_sizes,
            average_suffix_prefill_batch_size=(
                sum(suffix_batch_sizes) / len(suffix_batch_sizes) if suffix_batch_sizes else 0.0
            ),
            max_suffix_prefill_batch_size=max(suffix_batch_sizes, default=0),
            suffix_useful_tokens=suffix_useful_tokens,
            suffix_padded_tokens=suffix_padded_tokens,
            suffix_padding_waste_ratio=(
                (suffix_padded_tokens - suffix_useful_tokens) / suffix_padded_tokens
                if suffix_padded_tokens
                else 0.0
            ),
            exact_cache_hit_requests=sum(
                observation.batch_size
                for observation in apc_observations
                if observation.execution_mode is PrefillExecutionMode.EXACT_CACHE_HIT
            ),
            batched_suffix_requests=sum(
                observation.batch_size for observation in batched_suffix_observations
            ),
            chunked_prefill_batches=len(chunked_prefill_observations),
            chunked_prefill_chunks=sum(
                observation.batch_size for observation in chunked_prefill_observations
            ),
            chunked_prefill_useful_tokens=sum(
                observation.useful_prompt_tokens for observation in chunked_prefill_observations
            ),
            prefix_cache_enabled=(
                self._paged_cache_pool.prefix_cache_enabled
                if self._paged_cache_pool is not None
                else False
            ),
            prefix_cache_blocks=cache_metrics.prefix_cache_blocks,
            evictable_blocks=cache_metrics.evictable_blocks,
            active_shared_blocks=cache_metrics.active_shared_blocks,
            active_shared_references=cache_metrics.active_shared_references,
            prefix_cache_evictions=cache_metrics.prefix_cache_evictions,
            prefix_lookup_requests=cache_metrics.prefix_lookup_requests,
            prefix_hit_requests=cache_metrics.prefix_hit_requests,
            prefix_hit_blocks=cache_metrics.prefix_hit_blocks,
            prefix_hit_tokens=cache_metrics.prefix_hit_tokens,
            prefix_miss_tokens=cache_metrics.prefix_miss_tokens,
            prefill_tokens_computed=prefill_tokens_computed,
            avoided_prefill_tokens=avoided_prefill_tokens,
            prefix_hit_request_ratio=hit_request_ratio,
            prefix_hit_token_ratio=hit_token_ratio,
            preemptions=sum(state.preemption_count for state in states),
            resumes=sum(state.resume_count for state in states),
            recompute_tokens=sum(state.recompute_tokens for state in states),
        )

    def _apply_cancellations(self, tick_time: float) -> None:
        for state in tuple(self._states.values()):
            scheduled = (
                state.request.cancellation_time is not None
                and state.request.cancellation_time <= tick_time
            )
            if state.request.request_id not in self._pending_cancellations and not scheduled:
                continue
            self._pending_cancellations.discard(state.request.request_id)
            if state.status in self._terminal_statuses():
                continue
            self._remove_live_state(state)
            state.status = RequestStatus.CANCELLED
            state.finish_time = tick_time
            self._emit(
                event_type=EngineEventType.CANCELLED,
                state=state,
                timestamp=tick_time,
            )

    def _intrinsic_admission_failure(self, state: RequestState) -> str | None:
        """Return why one FIFO head can never fit, independent of resident requests."""
        required = self._reservation_for(state.request)
        scheduler = self.config.scheduler
        if required > scheduler.max_cached_tokens:
            return (
                f"required cache reservation {required} exceeds budget "
                f"{scheduler.max_cached_tokens}"
            )
        pool = self._paged_cache_pool
        if pool is None:
            return None
        required_blocks = pool.required_blocks(required)
        if required_blocks > pool.config.num_blocks:
            return (
                f"required cache reservation {required_blocks} blocks exceeds pool "
                f"{pool.config.num_blocks}"
            )
        return None

    def _admit(self, tick_time: float) -> None:  # noqa: C901, PLR0912, PLR0915
        scheduler = self.config.scheduler
        while self._waiting:
            state = self._states[self._waiting[0]]
            failure_reason = self._intrinsic_admission_failure(state)
            if failure_reason is not None:
                _ = self._waiting.popleft()
                state.failure_reason = failure_reason
                state.status = RequestStatus.FAILED
                state.finish_time = tick_time
                self._emit(
                    event_type=EngineEventType.FAILED,
                    state=state,
                    timestamp=tick_time,
                    detail=failure_reason,
                )
                continue
            if len(self._active) >= scheduler.max_active_requests:
                break
            resuming = state.status is RequestStatus.PREEMPTED
            required = self._reservation_for(state.request)
            required_blocks = 0
            if self._paged_cache_pool is not None:
                required_blocks = self._paged_cache_pool.required_blocks(required)
            if self._reserved_cache_tokens() + required > scheduler.max_cached_tokens:
                break
            if self._paged_cache_pool is not None:
                pool = self._paged_cache_pool
                fits = (
                    pool.can_reserve_with_prefix(
                        reserved_blocks=required_blocks,
                        prompt_tokens=state.request.prompt_tokens,
                    )
                    if pool.prefix_cache_enabled and not resuming
                    else pool.can_reserve(required_blocks)
                )
                if not fits:
                    break
            _ = self._waiting.popleft()
            if not resuming:
                state.admission_time = tick_time
            state.reserved_cache_tokens = required
            if state.request.max_new_tokens == 0:
                state.status = RequestStatus.FINISHED
                state.finish_time = tick_time
                state.reserved_cache_tokens = 0
                self._emit(
                    event_type=EngineEventType.ADMITTED,
                    state=state,
                    timestamp=tick_time,
                )
                self._emit(
                    event_type=EngineEventType.FINISHED,
                    state=state,
                    timestamp=tick_time,
                )
                continue
            if self._paged_cache_pool is not None:
                pool = self._paged_cache_pool
                if pool.prefix_cache_enabled and not resuming:
                    lookup = pool.reserve_with_prefix(
                        state.request.request_id,
                        reserved_blocks=required_blocks,
                        prompt_tokens=state.request.prompt_tokens,
                    )
                    state.prefix_hit_blocks = lookup.prefix_hit_blocks
                    state.prefix_hit_tokens = lookup.prefix_hit_tokens
                    state.prefix_miss_tokens = lookup.prefix_miss_tokens
                else:
                    pool.reserve(state.request.request_id, required_blocks)
                state.reserved_cache_blocks = required_blocks
            if resuming:
                state.status = RequestStatus.RECOMPUTING
                self._active.append(state.request.request_id)
                continue
            state.status = RequestStatus.PREFILLING
            self._active.append(state.request.request_id)
            if self._paged_cache_pool is not None and self._paged_cache_pool.prefix_cache_enabled:
                self._emit(
                    event_type=EngineEventType.PREFIX_LOOKUP,
                    state=state,
                    timestamp=tick_time,
                    detail=(
                        f"hit_blocks={state.prefix_hit_blocks};"
                        f"hit_tokens={state.prefix_hit_tokens};"
                        f"miss_tokens={state.prefix_miss_tokens}"
                    ),
                )
                if state.prefix_hit_blocks:
                    self._emit(
                        event_type=EngineEventType.PREFIX_HIT,
                        state=state,
                        timestamp=tick_time,
                        detail=f"shared_blocks={state.prefix_hit_blocks}",
                    )
                self._emit_prefix_evictions(state, tick_time)
            self._emit(
                event_type=EngineEventType.ADMITTED,
                state=state,
                timestamp=tick_time,
            )

    def _waiting_head_needs_kv(self) -> bool:
        if not self._waiting:
            return False
        state = self._states[self._waiting[0]]
        if self._intrinsic_admission_failure(state) is not None:
            return False
        required = self._reservation_for(state.request)
        scheduler = self.config.scheduler
        if self._reserved_cache_tokens() + required > scheduler.max_cached_tokens:
            return True
        pool = self._paged_cache_pool
        if pool is None:
            return False
        required_blocks = pool.required_blocks(required)
        if state.status is RequestStatus.PREEMPTED or not pool.prefix_cache_enabled:
            return not pool.can_reserve(required_blocks)
        return not pool.can_reserve_with_prefix(
            reserved_blocks=required_blocks,
            prompt_tokens=state.request.prompt_tokens,
        )

    def _maybe_preempt_for_pressure(self, tick_time: float) -> None:
        if not self.config.scheduler.kv_preemption or not self._waiting_head_needs_kv():
            return
        for request_id in tuple(self._active):
            state = self._states[request_id]
            if state.status is not RequestStatus.DECODING:
                continue
            if state.last_decode_tick != tick_time or state.last_recompute_time == tick_time:
                continue
            self._preempt_state(state, tick_time)
            return

    def _preempt_state(self, state: RequestState, tick_time: float) -> None:
        request_id = state.request.request_id
        pool = self._paged_cache_pool
        if pool is None or not pool.has_request(request_id):
            _invalid("KV preemption requires a resident paged request")
        if state.status is not RequestStatus.DECODING:
            _invalid("only DECODING requests may be preempted")
        pool.release(request_id)
        self._active.remove(request_id)
        state.kv_cache = None
        state.cached_tokens = 0
        state.reserved_cache_tokens = 0
        state.reserved_cache_blocks = 0
        state.prefill_logits_chunks.clear()
        state.preemption_count += 1
        state.status = RequestStatus.PREEMPTED
        self._waiting.append(request_id)
        if self._chunk_schedule_cursor == request_id:
            self._chunk_schedule_cursor = None
        if self._chunk_resume_decode == request_id:
            self._chunk_resume_decode = None
        preempt_time = state.token_timestamps[-1] if state.token_timestamps else tick_time
        self._emit(
            event_type=EngineEventType.PREEMPTED,
            state=state,
            timestamp=max(tick_time, preempt_time),
        )
        pool.verify_invariants()

    def _execute_prefill_chunks(
        self,
        request_ids: tuple[str, ...],
        chunk_lengths: tuple[int, ...],
        *,
        tick_time: float,
    ) -> None:
        if len(request_ids) != len(chunk_lengths):
            _invalid("chunk request and length counts must match")
        executor = self._executor
        pool = self._paged_cache_pool
        if not isinstance(executor, PagedAttentionExecutor) or pool is None:
            _invalid("chunk execution requires the direct paged executor")
        states = tuple(
            self._states[request_id]
            for request_id in request_ids
            if self._states[request_id].status is RequestStatus.PREFILLING
        )
        if len(states) != len(request_ids):
            _invalid("chunk schedule contains a non-prefilling request")
        for state, chunk_length in zip(states, chunk_lengths, strict=True):
            if state.prefill_start_time is None:
                state.prefill_start_time = tick_time
                self._emit(
                    event_type=EngineEventType.PREFILL_STARTED,
                    state=state,
                    timestamp=tick_time,
                )
            start = pool.request_cache(state.request.request_id).cache_length
            self._emit(
                event_type=EngineEventType.PREFILL_CHUNK_STARTED,
                state=state,
                timestamp=tick_time,
                detail=f"start={start};end={start + chunk_length}",
            )
        try:
            results = executor.prefill_chunks(states, chunk_lengths)
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}"
            results = tuple(
                ExecutionResult.failure(state.request.request_id, message, 0.0) for state in states
            )
        result_by_id = {result.request_id: result for result in results}
        duplicate_count = len(results) - len(result_by_id)
        for state in states:
            result = result_by_id.get(state.request.request_id)
            if result is None or duplicate_count:
                self._fail_state(state, tick_time, "executor returned malformed request results")
                continue
            result = self._commit_paged_result(
                state,
                result,
                prefill=True,
                tick_time=tick_time,
            )
            if result.error is None and result.prefill_complete:
                self._emit(
                    event_type=EngineEventType.PREFILL_CHUNK_FINISHED,
                    state=state,
                    timestamp=tick_time + result.latency_seconds,
                    detail=(f"start={state.cached_tokens};end={result.cache_tokens};final=true"),
                )
            self._apply_result(state, result, tick_time=tick_time, prefill=True)

    def _execute(  # noqa: C901
        self,
        request_ids: tuple[str, ...],
        *,
        tick_time: float,
        prefill: bool,
    ) -> None:
        expected_status = RequestStatus.PREFILLING if prefill else RequestStatus.DECODING
        states = tuple(
            self._states[request_id]
            for request_id in request_ids
            if self._states[request_id].status is expected_status
        )
        if not states:
            return
        if prefill:
            for state in states:
                state.prefill_start_time = tick_time
                self._emit(
                    event_type=EngineEventType.PREFILL_STARTED,
                    state=state,
                    timestamp=tick_time,
                )
        try:
            if self._paged_cache_pool is not None and not prefill and not self._direct_paged_decode:
                for state in states:
                    state.kv_cache = self._paged_cache_pool.materialize(state.request.request_id)
            results = self._executor.prefill(states) if prefill else self._executor.decode(states)
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}"
            results = tuple(
                ExecutionResult.failure(state.request.request_id, message, 0.0) for state in states
            )
        finally:
            if self._paged_cache_pool is not None:
                for state in states:
                    state.kv_cache = None
        result_by_id = {result.request_id: result for result in results}
        duplicate_count = len(results) - len(result_by_id)
        for state in states:
            result = result_by_id.get(state.request.request_id)
            if result is None or duplicate_count:
                self._fail_state(state, tick_time, "executor returned malformed request results")
                continue
            if self._paged_cache_pool is not None:
                result = self._commit_paged_result(
                    state,
                    result,
                    prefill=prefill,
                    tick_time=tick_time,
                )
            self._apply_result(state, result, tick_time=tick_time, prefill=prefill)

    def _write_paged_result_without_prefix_promotion(
        self,
        state: RequestState,
        result: ExecutionResult,
        cache: KVCache,
        *,
        prefill: bool,
    ) -> None:
        pool = self._paged_cache_pool
        if pool is None:
            _invalid("paged result write requires a cache pool")
        if prefill:
            pool.write_prefill(state.request.request_id, cache)
        elif result.used_fallback:
            pool.rebuild(state.request.request_id, cache)
        elif result.cache_is_delta:
            pool.append_delta(state.request.request_id, cache)
        else:
            pool.append(state.request.request_id, cache)

    def _commit_paged_result(
        self,
        state: RequestState,
        result: ExecutionResult,
        *,
        prefill: bool,
        tick_time: float,
    ) -> ExecutionResult:
        pool = self._paged_cache_pool
        if pool is None or result.error is not None or result.cache is None:
            return result
        cache = result.cache
        return_result = result
        try:
            if (
                prefill
                and result.cache_is_delta
                and (not result.prefill_complete or not pool.prefix_cache_enabled)
            ):
                pool.write_prefill_suffix(state.request.request_id, cache)
                table = pool.request_cache(state.request.request_id)
                _require_paged_cache_length(table.cache_length, result.cache_tokens)
                self._emit_prefix_evictions(state, tick_time + result.latency_seconds)
                return replace(result, cache=())
            if (
                prefill
                and result.cache_is_delta
                and result.prefill_complete
                and pool.prefix_cache_enabled
            ):
                current_logits = _require_prefix_prefill_logits(result.prefill_logits)
                chunks = (*state.prefill_logits_chunks, current_logits)
                semantic_prefix = state.prefix_hit_blocks * pool.config.block_tokens
                result = replace(
                    result,
                    prefill_logits=torch.cat(chunks, dim=0),
                    **{"prefill_prefix_" + "tokens": semantic_prefix},
                )
            if prefill and pool.prefix_cache_enabled:
                prefill_logits = _require_prefix_prefill_logits(result.prefill_logits)
                pool.write_prefill_suffix(state.request.request_id, cache)
                promotion = pool.promote_prompt_blocks(
                    state.request.request_id,
                    prompt_tokens=state.request.prompt_tokens,
                    prefix_hit_tokens=result.prefill_prefix_tokens,
                    suffix_logits=prefill_logits,
                )
                if promotion.promoted_blocks or promotion.duplicate_private_blocks_released:
                    self._emit(
                        event_type=EngineEventType.PREFIX_PROMOTE,
                        state=state,
                        timestamp=tick_time + result.latency_seconds,
                        detail=(
                            f"promoted={promotion.promoted_blocks};"
                            f"duplicates={promotion.duplicate_private_blocks_released}"
                        ),
                    )
                for prefix_hash in promotion.evicted_prefix_hashes:
                    self._emit(
                        event_type=EngineEventType.PREFIX_EVICT,
                        state=state,
                        timestamp=tick_time + result.latency_seconds,
                        detail=f"prefix_hash={prefix_hash}",
                    )
            else:
                self._write_paged_result_without_prefix_promotion(
                    state, result, cache, prefill=prefill
                )
            result = return_result
            table = pool.request_cache(state.request.request_id)
            _require_paged_cache_length(table.cache_length, result.cache_tokens)
            self._emit_prefix_evictions(state, tick_time + result.latency_seconds)
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}"
            return ExecutionResult.failure(
                state.request.request_id,
                message,
                result.latency_seconds,
            )
        return replace(result, cache=())

    def _emit_prefix_evictions(self, state: RequestState, timestamp: float) -> None:
        pool = self._paged_cache_pool
        if pool is None or not pool.prefix_cache_enabled:
            return
        for prefix_hash in pool.take_recent_evictions():
            self._emit(
                event_type=EngineEventType.PREFIX_EVICT,
                state=state,
                timestamp=timestamp,
                detail=f"prefix_hash={prefix_hash}",
            )

    def _apply_intermediate_prefill_result(
        self,
        state: RequestState,
        result: ExecutionResult,
        *,
        tick_time: float,
    ) -> None:
        invalid_output = "intermediate prefill chunk returned invalid output"
        if result.token_id is not None or result.prefill_logits is None:
            self._fail_state(state, tick_time, invalid_output)
            return
        computed_tokens = result.cache_tokens - result.prefill_prefix_tokens
        if computed_tokens <= 0:
            self._fail_state(state, tick_time, "intermediate prefill chunk made no progress")
            return
        state.kv_cache = result.cache
        state.cached_tokens = result.cache_tokens
        state.prefill_latency_seconds = (
            state.prefill_latency_seconds or 0.0
        ) + result.latency_seconds
        state.prefill_tokens_computed += computed_tokens
        pool = self._paged_cache_pool
        if pool is not None and pool.prefix_cache_enabled:
            state.prefill_logits_chunks.append(result.prefill_logits)
        state.status = RequestStatus.PREFILLING
        self._emit(
            event_type=EngineEventType.PREFILL_CHUNK_FINISHED,
            state=state,
            timestamp=tick_time + result.latency_seconds,
            detail=f"start={result.prefill_prefix_tokens};end={result.cache_tokens};final=false",
        )

    def _apply_result(
        self,
        state: RequestState,
        result: ExecutionResult,
        *,
        tick_time: float,
        prefill: bool,
    ) -> None:
        if result.error is not None:
            self._fail_state(state, tick_time + result.latency_seconds, result.error)
            return
        missing_token = result.token_id is None and (not prefill or result.prefill_complete)
        if missing_token or result.cache is None:
            self._fail_state(state, tick_time, "executor success omitted token or cache")
            return
        if result.latency_seconds < 0.0 or not math.isfinite(result.latency_seconds):
            self._fail_state(state, tick_time, "executor returned invalid latency")
            return
        if result.cache_tokens < 0 or result.cache_tokens > state.reserved_cache_tokens:
            reason = (
                f"executor cache occupancy {result.cache_tokens} exceeds reservation "
                f"{state.reserved_cache_tokens}"
            )
            self._fail_state(state, tick_time + result.latency_seconds, reason)
            return
        if prefill and not result.prefill_complete:
            self._apply_intermediate_prefill_result(state, result, tick_time=tick_time)
            return
        token_id = result.token_id
        if token_id is None:
            self._fail_state(state, tick_time, "executor success omitted token")
            return
        token_time = tick_time + result.latency_seconds
        state.generated_tokens.append(token_id)
        state.kv_cache = result.cache
        state.cached_tokens = result.cache_tokens
        state.token_timestamps.append(token_time)
        if prefill:
            state.prefill_latency_seconds = (
                state.prefill_latency_seconds or 0.0
            ) + result.latency_seconds
            state.first_token_time = token_time
            state.prefill_tokens_computed += result.cache_tokens - result.prefill_prefix_tokens
            state.prefill_logits_chunks.clear()
        else:
            state.decode_latencies_seconds.append(result.latency_seconds)
            state.last_decode_tick = tick_time
        self._emit(
            event_type=EngineEventType.TOKEN,
            state=state,
            timestamp=token_time,
            token_id=result.token_id,
            used_fallback=result.used_fallback,
        )
        if len(state.generated_tokens) >= state.request.max_new_tokens:
            self._finish_state(state, token_time)
        else:
            state.status = RequestStatus.DECODING

    def _finish_state(self, state: RequestState, timestamp: float) -> None:
        self._remove_live_state(state)
        state.status = RequestStatus.FINISHED
        state.finish_time = timestamp
        self._emit(
            event_type=EngineEventType.FINISHED,
            state=state,
            timestamp=timestamp,
        )

    def _fail_state(self, state: RequestState, timestamp: float, reason: str) -> None:
        self._remove_live_state(state)
        state.status = RequestStatus.FAILED
        state.failure_reason = reason
        state.finish_time = timestamp
        self._emit(
            event_type=EngineEventType.FAILED,
            state=state,
            timestamp=timestamp,
            detail=reason,
        )

    def _remove_live_state(self, state: RequestState) -> None:
        request_id = state.request.request_id
        if request_id in self._waiting:
            self._waiting.remove(request_id)
        if request_id in self._active:
            self._active.remove(request_id)
        if self._paged_cache_pool is not None and self._paged_cache_pool.has_request(request_id):
            self._paged_cache_pool.release(request_id)
        state.kv_cache = None
        state.prefill_logits_chunks.clear()
        state.cached_tokens = 0
        state.reserved_cache_tokens = 0
        state.reserved_cache_blocks = 0

    def _reservation_for(self, request: GenerationRequest) -> int:
        if request.max_new_tokens == 0:
            return 0
        prompt_length = min(len(request.prompt_tokens), self.config.block_size)
        return min(
            self.config.block_size,
            prompt_length + max(request.max_new_tokens - 1, 0),
        )

    def _emit(  # noqa: PLR0913
        self,
        *,
        event_type: EngineEventType,
        state: RequestState,
        timestamp: float,
        token_id: int | None = None,
        detail: str | None = None,
        used_fallback: bool = False,
    ) -> None:
        self._update_peaks()
        self._events.append(
            EngineEvent(
                sequence=len(self._events),
                timestamp=timestamp,
                event_type=event_type,
                request_id=state.request.request_id,
                status=state.status,
                token_id=token_id,
                detail=detail,
                used_fallback=used_fallback,
                active_requests=len(self._active),
                waiting_requests=len(self._waiting),
                cached_tokens=self._cached_tokens(),
                reserved_cache_tokens=self._reserved_cache_tokens(),
            )
        )

    def _update_peaks(self) -> None:
        self._peak_active = max(self._peak_active, len(self._active))
        self._peak_waiting = max(self._peak_waiting, len(self._waiting))
        self._peak_cached = max(self._peak_cached, self._cached_tokens())
        self._peak_reserved = max(self._peak_reserved, self._reserved_cache_tokens())

    def _cached_tokens(self) -> int:
        return sum(self._states[request_id].cached_tokens for request_id in self._active)

    def _reserved_cache_tokens(self) -> int:
        return sum(self._states[request_id].reserved_cache_tokens for request_id in self._active)

    @staticmethod
    def _terminal_statuses() -> frozenset[RequestStatus]:
        return frozenset({RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED})
